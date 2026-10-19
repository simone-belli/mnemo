"""The feedback loop (SPEC §6): Deeper, Simpler, Knew that, Lost me, and repeat requests."""
from __future__ import annotations

import json
from datetime import datetime

from . import db, knowledge
from .config import Settings
from .llm import obj

SIGNALS = ("knew_that", "deeper", "simpler", "lost_me", "repeat")

DIAGNOSE_SYSTEM = """A person was reading a short recap and said "lost me". Work out why, from the
recap, her question and her optional note:
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


def handle(conn, settings: Settings, llm, tagger, interaction_id: int, signal: str, note: str | None,
           now: datetime) -> dict:
    rec = db.get_interaction(conn, interaction_id)
    if rec is None or rec["kind"] != "recall":
        raise KeyError(interaction_id)
    if signal not in SIGNALS:
        raise ValueError(f"unknown signal {signal!r}")

    diagnosis = None
    if signal == "lost_me":
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
    return {"signal": signal, "deltas": deltas, "diagnosis": diagnosis,
            "depth_preference": db.get_depth_pref(conn, knowledge.branch_of(conn, rec["node_id"]))}
