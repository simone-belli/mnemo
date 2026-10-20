"""The feedback loop (SPEC §6): a 1–5 difficulty rating, and repeat requests.

1 is "too easy, I knew it", 3 is "spot on", 5 is "lost me". A 5 comes with an optional note
and runs a diagnosis of what was missing."""
from __future__ import annotations

import json
from datetime import datetime

from . import db, knowledge
from .config import Settings
from .llm import obj

DIFFICULTIES = range(1, 6)
LOST_ME = 5

DIAGNOSE_SYSTEM = """A person was reading a short recap or explanation and said "lost me". Work out
why, from the text, her question and her optional note:
- missed_word: she just missed or misheard something; nothing is missing from what she knows.
- unfamiliar_term: one term she doesn't know; the rest was fine. Give the term.
- missing_prerequisite: the recap relies on a concept or result she hasn't got; give it as a
  short topic name that could be looked up on its own (e.g. "renormalisation group", "linear perspective").
- genuine_gap: the whole subtopic is beyond her current level.
Prefer the most specific explanation that fits. Give a one-sentence explanation."""

DIAGNOSE_SCHEMA = obj({
    "kind": {"type": "string", "enum": ["missed_word", "unfamiliar_term", "missing_prerequisite", "genuine_gap"]},
    "term": {"type": "string"},
    "explanation": {"type": "string"},
})


def diagnose(llm, settings: Settings, recall_row, note: str | None) -> dict:
    prompt = (f"Her question: {recall_row['query']}\n\nThe recap:\n{recall_row['response_md']}\n\n"
              f"Her note: {note or '(none)'}")
    return llm.json("diagnose", DIAGNOSE_SYSTEM, prompt, DIAGNOSE_SCHEMA, settings.fast_model)


def signal_for(difficulty: int) -> str:
    """How a rating is stored in `interactions.feedback`."""
    return f"difficulty:{difficulty}"


def handle(conn, settings: Settings, llm, tagger, interaction_id: int, difficulty: int | None, signal: str | None,
           note: str | None, now: datetime) -> dict:
    """Log a rating (`difficulty` 1–5) or a repeat request (`signal="repeat"`) on a recall or explanation."""
    rec = db.get_interaction(conn, interaction_id)
    if rec is None or rec["kind"] not in ("recall", "explain"):
        raise KeyError(interaction_id)
    if difficulty is not None:
        if difficulty not in DIFFICULTIES:
            raise ValueError(f"difficulty must be 1 to 5, not {difficulty}")
        signal = signal_for(difficulty)
    elif signal != "repeat":
        raise ValueError(f"unknown signal {signal!r}")

    diagnosis = None
    if difficulty == LOST_ME:
        diagnosis = diagnose(llm, settings, rec, note)
        if diagnosis["kind"] in ("unfamiliar_term", "missing_prerequisite") and diagnosis.get("term"):
            # Tag the term or prerequisite onto the map (creating its node if needed).
            t = tagger.tag(conn, diagnosis["term"], now, session_id=f"diagnosis-{interaction_id}")
            diagnosis |= {"target_node_id": t.node_id, "target_mode": t.mode, "target_path": t.path}
        if note:
            diagnosis["note"] = note

    ev = {"kind": "feedback", "query": rec["query"], "node_id": rec["node_id"], "mode": rec["mode"],
          "feedback": signal, "diagnosis": json.dumps(diagnosis) if diagnosis else None,
          "ref_id": interaction_id, "ts": knowledge.iso(now)}
    deltas = knowledge.apply_event(conn, settings, ev, now)
    db.log_interaction(conn, deltas=deltas, **ev)
    conn.commit()
    return {"signal": signal, "difficulty": difficulty, "deltas": deltas, "diagnosis": diagnosis,
            "depth_preference": db.get_depth_pref(conn, knowledge.branch_of(conn, rec["node_id"]))}
