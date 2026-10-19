"""Onboarding quiz (SPEC §7): pick the most informative cell, have the fast model write a
question for it, grade the answer, update the map."""
from __future__ import annotations

import json
import math
import random
from datetime import datetime

from . import db, knowledge, taxonomy
from .config import MODES, Settings
from .llm import obj

# Normative and expressive knowing are hard to test with a question, so they are asked less.
MODE_WEIGHT = {"descriptive": 1.0, "explanatory": 1.0, "practical": 0.8, "normative": 0.6, "expressive": 0.4}


def descendants(children: dict, node_id: int) -> int:
    kids = children.get(node_id, [])
    return len(kids) + sum(descendants(children, k) for k in kids)


def select_cell(conn, settings: Settings, llm, embedder, now: datetime, rng: random.Random | None = None
                ) -> tuple[int, str, float]:
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
    nid, mode = best
    # Pitch the item a little above the current estimate (the parent's, while this cell has no evidence).
    mean_, ev = view.mode(nid, mode)
    if ev < 0.5:
        parent = db.get_node(conn, nid)["parent_id"]
        mean_ = view.mode(parent, mode)[0] if parent else mean_
    return nid, mode, round(min(0.9, max(0.2, mean_ + 0.1)), 2)


ITEM_SYSTEM = """You write one quiz question that measures how well a person knows a topic, in a
given mode of knowing. The person is a theoretical physicist turned quant researcher with
broad interests. Difficulty runs from 0 (any educated adult) to 1 (expert in the field).
Write the question so that a correct answer genuinely shows that level of knowledge."""

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
                  multiple_choice: bool) -> int:
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
    answer = str(item["correct_index"]) if options else item["answer"]
    cur = conn.execute(
        "INSERT INTO quiz_items(node_id, mode, difficulty, question, options, answer, ts) VALUES(?,?,?,?,?,?,?)",
        (node_id, mode, difficulty, item["question"], json.dumps(options) if options else None, answer,
         knowledge.iso(knowledge.utcnow())))
    conn.commit()
    return cur.lastrowid


def item_payload(conn, item_id: int, target: int) -> dict:
    r = conn.execute("SELECT * FROM quiz_items WHERE id=?", (item_id,)).fetchone()
    answered = conn.execute("SELECT count(*) FROM quiz_items WHERE score IS NOT NULL").fetchone()[0]
    return {"item_id": r["id"], "question": r["question"],
            "options": json.loads(r["options"]) if r["options"] else None,
            "topic": [n["name"] for n in db.path(conn, r["node_id"])], "mode": r["mode"],
            "difficulty": r["difficulty"], "progress": {"answered": answered, "target": target}}


def next_item(conn, settings: Settings, llm, embedder, now: datetime) -> dict:
    pending = conn.execute("SELECT id FROM quiz_items WHERE score IS NULL ORDER BY id DESC LIMIT 1").fetchone()
    if pending:
        return item_payload(conn, pending["id"], settings.quiz_target)
    answered = conn.execute("SELECT count(*) FROM quiz_items WHERE score IS NOT NULL").fetchone()[0]
    node_id, mode, difficulty = select_cell(conn, settings, llm, embedder, now)
    item_id = generate_item(conn, settings, llm, node_id, mode, difficulty, multiple_choice=answered % 3 != 2)
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


def answer(conn, settings: Settings, llm, item_id: int, user_answer: str, now: datetime) -> dict:
    r = conn.execute("SELECT * FROM quiz_items WHERE id=?", (item_id,)).fetchone()
    if r is None:
        raise KeyError(item_id)
    if r["score"] is not None:
        raise ValueError("already answered")
    score, rationale, correct = grade(conn, settings, llm, item_id, user_answer)
    conn.execute("UPDATE quiz_items SET user_answer=?, score=?, ts=? WHERE id=?",
                 (user_answer, score, knowledge.iso(now), item_id))
    ev = {"kind": "quiz", "node_id": r["node_id"], "mode": r["mode"], "feedback": str(score),
          "diagnosis": json.dumps({"difficulty": r["difficulty"], "rationale": rationale}),
          "ref_id": item_id, "query": r["question"], "ts": knowledge.iso(now)}
    deltas = knowledge.apply_event(conn, settings, ev, now)
    db.log_interaction(conn, deltas=deltas, **ev)
    conn.commit()
    return {"score": score, "rationale": rationale, "correct": correct, "deltas": deltas}


def rate(conn, settings: Settings, ratings: dict[str, int], now: datetime) -> list[dict]:
    """Self-ratings (1–5) per root domain, as weak priors on the root cells."""
    roots = {n["name"]: n["id"] for n in db.children(conn, None)}
    by_id = {str(roots[name]): int(v) for name, v in ratings.items() if name in roots}
    ev = {"kind": "rating", "node_id": None, "mode": None, "diagnosis": json.dumps({"ratings": by_id}),
          "ts": knowledge.iso(now)}
    deltas = knowledge.apply_event(conn, settings, ev, now)
    db.log_interaction(conn, deltas=deltas, **ev)
    conn.commit()
    return deltas

