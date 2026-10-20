"""Onboarding quiz (SPEC §7): pick the most informative cell, have the fast model write a
question for it at an adaptive difficulty, grade the answer, update the map."""
from __future__ import annotations

import json
import math
import random
from datetime import datetime

from . import db, knowledge, mastery, taxonomy
from .config import MODES, Settings
from .llm import obj

# Normative and expressive knowing are hard to test with a question, so they are asked less.
MODE_WEIGHT = {"descriptive": 1.0, "explanatory": 1.0, "practical": 0.8, "normative": 0.6, "expressive": 0.4}


def descendants(children: dict, node_id: int) -> int:
    kids = children.get(node_id, [])
    return len(kids) + sum(descendants(children, k) for k in kids)


def select_cell(conn, settings: Settings, llm, embedder, now: datetime, rng: random.Random | None = None
                ) -> tuple[int, str]:
    """The (node, mode) with the highest Beta variance, weighted by node breadth.

    Starts at the second level of the tree; a node whose answers are strong opens its children
    (creating them if needed)."""
    rng = rng or random.Random()
    view = knowledge.MapView(conn, settings, now)
    asked: dict[tuple[int, str], int] = {}
    for r in conn.execute("SELECT node_id, mode FROM quiz_items WHERE score IS NOT NULL"):
        asked[(r["node_id"], r["mode"])] = asked.get((r["node_id"], r["mode"]), 0) + 1
    asked_node = {}
    for (n, _), c in asked.items():
        asked_node[n] = asked_node.get(n, 0) + c

    pool = [n["id"] for n in db.all_nodes(conn) if n["depth"] == 1]
    for nid in list(pool):
        mean_, ev = view.node(nid)
        if asked_node.get(nid) and ev >= 1 and mean_ > 0.7:     # strong: go one level deeper
            kids = view.children.get(nid) or taxonomy.refine(conn, settings, llm, embedder, nid, None, now,
                                                             source="quiz")[2]
            pool += kids
    view = knowledge.MapView(conn, settings, now)              # children may have been added

    best, best_score = None, -1.0
    for nid in pool:
        breadth = 1 + math.log1p(descendants(view.children, nid))
        for mode in MODES:
            cell = view.own(nid, mode)
            score = (cell.variance * breadth * MODE_WEIGHT[mode]
                     * 0.5 ** asked.get((nid, mode), 0) * 0.7 ** asked_node.get(nid, 0)
                     * (1 + 0.1 * rng.random()))
            if score > best_score:
                best, best_score = (nid, mode), score
    return best


def difficulty(conn, settings: Settings, node_id: int) -> float:
    """The staircase level of the node's root domain, from every answer so far (SPEC §7)."""
    answers = [(db.path(conn, r["node_id"])[0]["id"], r["score"])
               for r in conn.execute("SELECT node_id, score FROM quiz_items WHERE score IS NOT NULL ORDER BY ts, id")]
    overall, levels = mastery.staircase(answers, settings.quiz_start_difficulty, settings.quiz_step_up,
                                        settings.quiz_step_down)
    return round(levels.get(db.path(conn, node_id)[0]["id"], overall), 2)


ITEM_SYSTEM = """You write one quiz question that measures whether someone knows a topic, in a
given mode of knowing, at a given difficulty. Difficulty is absolute, not relative to anyone:
- 0.0–0.2: general knowledge most adults have ("Who painted the Mona Lisa?")
- 0.3–0.4: what a curious, well-read person knows; a good secondary-school level
- 0.5–0.6: an introductory university course or a good popular-science book
- 0.7–0.8: someone who studied the subject or follows it seriously
- 0.9–1.0: a specialist in the field
Ask about one central idea, fact or skill at exactly that level. Avoid trick questions and
obscure details (exact dates, numbers, minor names) unless the difficulty is high. Keep the
question short. Wrong options must be plausible but clearly wrong to someone at that level."""

ITEM_SCHEMA = obj({
    "question": {"type": "string"},
    "options": {"type": "array", "items": {"type": "string"}},
    "correct_index": {"type": "integer"},
    "answer": {"type": "string"},
})

GRADE_SYSTEM = """You grade a short free-text answer to a quiz question. Score from 0 (wrong or
empty) to 1 (fully correct and precise); partial credit for partly right answers. Be fair to
terse answers: precision matters, length does not. Give a one-line rationale."""

GRADE_SCHEMA = obj({"score": {"type": "number"}, "rationale": {"type": "string"}})


def generate_item(conn, settings: Settings, llm, node_id: int, mode: str, difficulty: float,
                  multiple_choice: bool, rng: random.Random | None = None) -> int:
    topic = " › ".join(n["name"] for n in db.path(conn, node_id))
    node = db.get_node(conn, node_id)
    kind = ("multiple choice with exactly 4 options, one correct; set correct_index (0-3) and put the correct "
            "option's text in answer" if multiple_choice else
            "a short free-text question answerable in a sentence; leave options empty, correct_index -1, and put a "
            "model answer in answer")
    prompt = (f"Topic: {topic} — {node['description']}\nMode: {mode}\nDifficulty: {difficulty}\n"
              f"Format: {kind}.")
    item = llm.json("quiz_item", ITEM_SYSTEM, prompt, ITEM_SCHEMA, settings.fast_model)
    options = item["options"][:4] if multiple_choice and len(item.get("options", [])) >= 2 else None
    if options and 0 <= item["correct_index"] < len(options):
        # Models tend to put the right option first, so shuffle and keep track of it.
        correct = options[item["correct_index"]]
        (rng or random.Random()).shuffle(options)
        answer = str(options.index(correct))
    else:
        options, answer = None, item["answer"]
    cur = conn.execute(
        "INSERT INTO quiz_items(node_id, mode, difficulty, question, options, answer, ts) VALUES(?,?,?,?,?,?,?)",
        (node_id, mode, difficulty, item["question"], json.dumps(options) if options else None, answer,
         knowledge.iso(knowledge.utcnow())))
    conn.commit()
    return cur.lastrowid


def start_round(conn) -> None:
    """A new quiz (first run or Recalibrate): progress counts from here."""
    discard_pending(conn)
    last = conn.execute("SELECT max(id) FROM quiz_items").fetchone()[0] or 0
    db.set_meta(conn, "quiz_round_start", str(last))
    conn.commit()


def round_answered(conn) -> int:
    start = int(db.get_meta(conn, "quiz_round_start") or 0)
    return conn.execute("SELECT count(*) FROM quiz_items WHERE score IS NOT NULL AND id > ?", (start,)).fetchone()[0]


def item_payload(conn, item_id: int, target: int) -> dict:
    r = conn.execute("SELECT * FROM quiz_items WHERE id=?", (item_id,)).fetchone()
    answered = round_answered(conn)
    return {"item_id": r["id"], "question": r["question"],
            "options": json.loads(r["options"]) if r["options"] else None,
            "topic": [n["name"] for n in db.path(conn, r["node_id"])], "mode": r["mode"],
            "difficulty": r["difficulty"], "progress": {"answered": answered, "target": target}}


def discard_pending(conn) -> None:
    """Drop unanswered items, e.g. at start-up, so one written by the offline fake (or an older
    prompt) isn't served again. They have no interactions, so the map is unaffected."""
    conn.execute("DELETE FROM quiz_items WHERE score IS NULL")
    conn.commit()


def next_item(conn, settings: Settings, llm, embedder, now: datetime) -> dict:
    pending = conn.execute("SELECT id FROM quiz_items WHERE score IS NULL ORDER BY id DESC LIMIT 1").fetchone()
    if pending:
        return item_payload(conn, pending["id"], settings.quiz_target)
    node_id, mode = select_cell(conn, settings, llm, embedder, now)
    item_id = generate_item(conn, settings, llm, node_id, mode, difficulty(conn, settings, node_id),
                            multiple_choice=round_answered(conn) % 3 != 2)
    return item_payload(conn, item_id, settings.quiz_target)


def grade(conn, settings: Settings, llm, item_id: int, answer: str) -> tuple[float, str, str]:
    """-> (score 0–1, rationale, the correct answer as text)."""
    r = conn.execute("SELECT * FROM quiz_items WHERE id=?", (item_id,)).fetchone()
    if r["options"]:
        options = json.loads(r["options"])
        correct = options[int(r["answer"])]
        ok = answer.strip() == r["answer"] or answer.strip() == correct
        return (1.0 if ok else 0.0), ("Correct." if ok else f"The answer is: {correct}"), correct
    if not answer.strip():
        return 0.0, "No answer.", r["answer"]
    prompt = f"Question: {r['question']}\nModel answer: {r['answer']}\nTheir answer: {answer}"
    out = llm.json("grade", GRADE_SYSTEM, prompt, GRADE_SCHEMA, settings.fast_model)
    return min(1.0, max(0.0, float(out["score"]))), out["rationale"], r["answer"]


def answer(conn, settings: Settings, llm, item_id: int, user_answer: str, now: datetime,
           dont_know: bool = False) -> dict:
    """Grade an answer and update the map. "I don't know" scores 0 like a wrong answer, without
    asking the model."""
    r = conn.execute("SELECT * FROM quiz_items WHERE id=?", (item_id,)).fetchone()
    if r is None:
        raise KeyError(item_id)
    if r["score"] is not None:
        raise ValueError("already answered")
    if dont_know:
        user_answer = ""
        correct = json.loads(r["options"])[int(r["answer"])] if r["options"] else r["answer"]
        score, rationale = 0.0, f"The answer is: {correct}"
    else:
        score, rationale, correct = grade(conn, settings, llm, item_id, user_answer)
    conn.execute("UPDATE quiz_items SET user_answer=?, score=?, ts=? WHERE id=?",
                 (user_answer, score, knowledge.iso(now), item_id))
    ev = {"kind": "quiz", "node_id": r["node_id"], "mode": r["mode"], "feedback": str(score),
          "diagnosis": json.dumps({"difficulty": r["difficulty"], "rationale": rationale, "dont_know": dont_know}),
          "ref_id": item_id, "query": r["question"], "ts": knowledge.iso(now)}
    deltas = knowledge.apply_event(conn, settings, ev, now)
    db.log_interaction(conn, deltas=deltas, **ev)
    conn.commit()
    return {"score": score, "rationale": rationale, "correct": correct, "deltas": deltas}
