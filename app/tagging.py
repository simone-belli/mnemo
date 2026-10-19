"""Query -> (node, mode) distribution (SPEC §4).

1. embed the query; 2. descend the tree while one child is clearly best; 3. keep the
distribution at the stopping level; 4. a fast LLM judge confirms the node, picks the mode and
says whether the query is narrower than every candidate (then the tree is refined);
5. within a session, the distribution is smoothed with the previous turn's.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

from . import db, taxonomy
from .config import MODES, Settings
from .llm import obj


@dataclass
class TagResult:
    node_id: int
    mode: str
    confidence: float
    dist: list[dict]                  # [{node_id, name, mode, p}], best first
    path: list[str]
    query_vec: np.ndarray
    created: list[int] = field(default_factory=list)
    debug: dict = field(default_factory=dict)


def softmax(x: np.ndarray, temperature: float) -> np.ndarray:
    z = (x - x.max()) / max(temperature, 1e-6)
    e = np.exp(z)
    return e / e.sum()


JUDGE_SYSTEM = """You tag questions onto a personal map of knowledge.
Pick the candidate node that is the best home for the query (the most specific candidate that
still contains the whole query), and the mode of knowing the query asks for:
- descriptive: what is or was (facts, events, who/what/when)
- explanatory: why or how it works (mechanisms, causes, theory)
- normative: what is good or right (ethics, judgement, taste, evaluation)
- practical: how to do it (methods, techniques, recipes, skills)
- expressive: what it is like (experience, feeling, art as lived)
Set more_specific to true only when the query is clearly narrower than the chosen node and
deserves its own subtopic (e.g. a named work, person, experiment or technique), and give that
subtopic a short name in specific_topic."""

JUDGE_SCHEMA = obj({
    "node_id": {"type": "integer"},
    "confidence": {"type": "number"},
    "mode": {"type": "string", "enum": list(MODES)},
    "mode_weights": obj({m: {"type": "number"} for m in MODES}),
    "more_specific": {"type": "boolean"},
    "specific_topic": {"type": "string"},
})


class Tagger:
    def __init__(self, settings: Settings, llm, embedder):
        self.s, self.llm, self.embedder = settings, llm, embedder
        self.sessions: dict[str, dict] = {}       # session id -> {"dist": {(node, mode): p}, "turns": deque}

    # -------------------------------------------------------------- 2–3. descent

    def descend(self, conn, qvec: np.ndarray) -> tuple[int, list[tuple[int, float]], list[dict]]:
        """Walk down from the roots. Returns (stopping node, distribution at that level, trace)."""
        current, cos_current = None, -np.inf
        level: list[tuple[int, float]] = []
        trace = []
        while True:
            kids = [k for k in db.children(conn, current) if k["embedding"] is not None]
            if not kids:
                break
            sims = np.array([float(np.dot(db.from_blob(k["embedding"]), qvec)) for k in kids])
            p = softmax(sims, self.s.tag_temperature)
            best = int(p.argmax())
            step = {"at": current, "best": kids[best]["name"], "p": round(float(p[best]), 3),
                    "cos": round(float(sims[best]), 3), "cos_here": None if current is None else round(cos_current, 3)}
            go = current is None or (p[best] > self.s.tag_threshold and sims[best] >= cos_current)
            step["descend"] = bool(go)
            trace.append(step)
            if not go:
                break
            current, cos_current = kids[best]["id"], float(sims[best])
            level = [(k["id"], float(pk)) for k, pk in zip(kids, p)]
        return current, level, trace

    def candidates(self, conn, qvec: np.ndarray, stop: int, level: list[tuple[int, float]], k: int = 5) -> list:
        ids = {n["id"] for n in db.path(conn, stop)} | {i for i, _ in level} | {c["id"] for c in db.children(conn, stop)}
        rows = [db.get_node(conn, i) for i in ids]
        rows = [r for r in rows if r["embedding"] is not None]
        rows.sort(key=lambda r: -float(np.dot(db.from_blob(r["embedding"]), qvec)))
        top = rows[:k]
        if stop not in [r["id"] for r in top]:
            top = [db.get_node(conn, stop)] + top[:k - 1]
        return top

    # -------------------------------------------------------------- 4. judge

    def judge(self, conn, query: str, cands: list, turns: list[str]) -> dict:
        lines = [f"- id {c['id']}: {' › '.join(n['name'] for n in db.path(conn, c['id']))} — {c['description']}"
                 for c in cands]
        context = ("Recent questions in this session (oldest first):\n" + "\n".join(f"- {t}" for t in turns) + "\n\n") if turns else ""
        prompt = (f"{context}Query: {query!r}\n\nCandidate nodes:\n" + "\n".join(lines) +
                  "\n\nAnswer with the chosen node_id (one of the ids above), your confidence 0–1, the mode, "
                  "weights for all five modes summing to 1, and more_specific/specific_topic.")
        out = self.llm.json("judge", JUDGE_SYSTEM, prompt, JUDGE_SCHEMA, self.s.fast_model)
        if out.get("node_id") not in {c["id"] for c in cands}:
            out["node_id"] = cands[0]["id"]
            out["confidence"] = min(out.get("confidence", 0.5), 0.5)
        return out

    # -------------------------------------------------------------- all together

    def tag(self, conn, query: str, now: datetime, session_id: str | None = None,
            forced_node: int | None = None, allow_refine: bool = True) -> TagResult:
        qvec = self.embedder.embed([query])[0]
        session = self.sessions.setdefault(session_id or "-", {"dist": {}, "turns": deque(maxlen=4)})

        if forced_node is not None:
            stop, level, trace = forced_node, [(forced_node, 1.0)], []
            cands = [db.get_node(conn, forced_node)]
        else:
            stop, level, trace = self.descend(conn, qvec)
            cands = self.candidates(conn, qvec, stop, level)
        verdict = self.judge(conn, query, cands, list(session["turns"]))
        node_id, confidence = verdict["node_id"], float(verdict.get("confidence", 0.5))

        before = {n["id"] for n in db.all_nodes(conn)}
        refinements = []
        if allow_refine and forced_node is None and verdict.get("more_specific") and confidence >= self.s.tag_threshold:
            for _ in range(self.s.max_refine_steps):
                child, narrower, ids = taxonomy.refine(conn, self.s, self.llm, self.embedder, node_id, query, now)
                refinements.append({"from": node_id, "to": child, "narrower": narrower})
                if child == node_id:
                    break
                node_id = child
                if not narrower:
                    break

        created = [n["id"] for n in db.all_nodes(conn) if n["id"] not in before]

        # The distribution: the judge's pick carries its confidence, the descent level the rest.
        node_p: dict[int, float] = {}
        for nid, p in level:
            node_p[nid] = node_p.get(nid, 0.0) + (1 - confidence) * p
        node_p[node_id] = node_p.get(node_id, 0.0) + confidence
        mw = verdict.get("mode_weights") or {}
        total = sum(max(0.0, mw.get(m, 0.0)) for m in MODES) or 1.0
        mode_p = {m: max(0.0, mw.get(m, 0.0)) / total for m in MODES} if mw else {verdict["mode"]: 1.0}
        raw = {(n, m): pn * pm for n, pn in node_p.items() for m, pm in mode_p.items() if pn * pm > 0.01}

        # 5. Smoothing within a session. The judge's pick stays primary (the EMA only steadies the
        # spread), so a change of subject is never held back by the previous turn.
        a = self.s.tag_ema
        prev = session["dist"]
        smooth = {key: a * raw.get(key, 0.0) + (1 - a) * prev.get(key, 0.0) for key in set(raw) | set(prev)}
        smooth = {k: v for k, v in smooth.items() if v > 0.01}
        session["dist"] = smooth
        session["turns"].append(query)

        names = {n: db.get_node(conn, n)["name"] for n, _ in smooth}
        dist = sorted(({"node_id": n, "name": names[n], "mode": m, "p": round(p, 3)} for (n, m), p in smooth.items()),
                      key=lambda d: -d["p"])[:10]
        return TagResult(
            node_id=node_id, mode=verdict["mode"], confidence=confidence, dist=dist,
            path=[n["name"] for n in db.path(conn, node_id)], query_vec=qvec, created=created,
            debug={"descent": trace, "candidates": [c["name"] for c in cands], "judge": verdict,
                   "refinements": refinements, "raw": len(raw)},
        )
