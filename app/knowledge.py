"""The knowledge map as stored in the database: reading effective mastery, applying events,
building the profile sent to the recap model, finding the frontier, and replaying history.

The maths is in `mastery.py`; this module only moves rows in and out of it.
Every change to the map comes from an event with a timestamp, and `deltas_for()` turns an
event into mastery changes. Live requests and `replay()` both go through it, so the map
can always be rebuilt from `interactions` (with today's weights).
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import numpy as np

from . import db, mastery
from .config import MODES, Settings, Weights
from .mastery import Cell


def utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def iso(dt: datetime) -> str:
    return dt.isoformat(timespec="seconds")


# ---------------------------------------------------------------- reading

def load_cells(conn) -> dict[tuple[int, str], Cell]:
    return {k: Cell(a, b, db.parse_ts(ts)) for k, (a, b, ts) in db.get_cells(conn).items()}


def children_map(conn, before: str | None = None) -> dict[int | None, list[int]]:
    out: dict[int | None, list[int]] = {}
    for n in db.all_nodes(conn):
        if before is None or n["created_at"] <= before:
            out.setdefault(n["parent_id"], []).append(n["id"])
    return out


class MapView:
    """Effective mastery for the whole tree at one moment (decayed and aggregated upwards)."""

    def __init__(self, conn, settings: Settings, now: datetime):
        self.conn, self.now, self.h = conn, now, settings.half_life_days
        self.cells = load_cells(conn)
        self.children = children_map(conn)
        self.agg = mastery.aggregate_tree(self.children, self.cells, MODES, now, self.h)

    def own(self, node_id: int, mode: str) -> Cell:
        return mastery.decay(self.cells.get((node_id, mode), Cell()), self.now, self.h)

    def mode(self, node_id: int, mode: str) -> tuple[float, float]:
        return self.agg.get(node_id, {}).get(mode, (0.5, 0.0))

    def node(self, node_id: int) -> tuple[float, float]:
        return mastery.node_level(self.agg.get(node_id, {m: (0.5, 0.0) for m in MODES}))

    def summary(self, node_id: int) -> dict:
        m, e = self.node(node_id)
        return {"mean": round(m, 3), "evidence": round(e, 2)}


# ---------------------------------------------------------------- writing

def apply_delta(conn, settings: Settings, node_id: int, mode: str, d_alpha: float, d_beta: float,
                now: datetime) -> dict:
    a, b, ts = db.get_cell(conn, node_id, mode) or (1.0, 1.0, None)
    before = Cell(a, b, db.parse_ts(ts))
    after = mastery.update(before, d_alpha, d_beta, now, settings.half_life_days)
    db.put_cell(conn, node_id, mode, after.alpha, after.beta, iso(now))
    return {"node_id": node_id, "mode": mode, "d_alpha": d_alpha, "d_beta": d_beta,
            "before": round(mastery.decay(before, now, settings.half_life_days).mean, 3), "after": round(after.mean, 3)}


def touch(conn, settings: Settings, node_id: int, mode: str, now: datetime) -> None:
    apply_delta(conn, settings, node_id, mode, 0.0, 0.0, now)


def init_child_cells(conn, settings: Settings, node_id: int, now: datetime) -> None:
    """Correlation prior: a new child starts at its parent's effective mean, mode by mode."""
    node = db.get_node(conn, node_id)
    if node["parent_id"] is None:
        return
    view = MapView(conn, settings, now)
    for mode in MODES:
        c = mastery.child_prior(view.mode(node["parent_id"], mode)[0])
        db.put_cell(conn, node_id, mode, c.alpha, c.beta, None)


def branch_of(conn, node_id: int) -> int:
    """The field a node belongs to (its depth-1 ancestor; a root is its own branch)."""
    p = db.path(conn, node_id)
    return p[1]["id"] if len(p) > 1 else p[0]["id"]


def nudge_depth(conn, node_id: int, delta: float) -> float:
    branch = branch_of(conn, node_id)
    value = min(1.0, max(-1.0, db.get_depth_pref(conn, branch) + delta))
    db.put_depth_pref(conn, branch, value)
    return value


# ---------------------------------------------------------------- events -> deltas

def deltas_for(conn, ev: dict, w: Weights) -> tuple[list[tuple[int, str, float, float]], float]:
    """Mastery changes (node, mode, d_alpha, d_beta) and a depth-preference nudge for one event.

    `ev` is an interactions row as a dict. Recalls change nothing but `last_seen`; asking for an
    explanation is a mild "less known" signal."""
    kind, node, mode = ev["kind"], ev["node_id"], ev["mode"]
    diag = json.loads(ev["diagnosis"]) if ev.get("diagnosis") else {}
    if kind == "recall":
        return [(node, mode, 0.0, 0.0)], 0.0
    if kind == "explain":
        return [(node, mode, 0.0, w.explain)], 0.0
    if kind == "feedback":
        signal = ev["feedback"]
        if signal == "knew_that":
            return [(node, mode, w.knew_that, 0.0)], 0.0
        if signal == "deeper":
            return [(node, mode, w.deeper, 0.0)], w.depth_nudge
        if signal == "simpler":
            return [(node, mode, 0.0, w.simpler)], -w.depth_nudge
        if signal == "repeat":
            return [(node, mode, 0.0, w.repeat)], 0.0
        if signal == "lost_me":
            k, target = diag.get("kind"), diag.get("target_node_id")
            if k == "unfamiliar_term" and target:
                return [(target, diag.get("target_mode") or mode, 0.0, w.lost_term)], 0.0
            if k == "missing_prerequisite" and target:
                return [(target, diag.get("target_mode") or mode, 0.0, w.lost_prereq)], 0.0
            if k == "genuine_gap":
                return [(node, mode, 0.0, w.lost_gap)], 0.0
            return [], 0.0                       # a missed word: nothing to learn about the map
    if kind == "quiz":
        score = float(ev["feedback"])
        da, dbeta = mastery.quiz_delta(score, diag.get("difficulty", 0.5))
        out = [(node, mode, da, dbeta)]
        if score >= 0.7:                         # strong answer: siblings are probably decent too
            parent = db.get_node(conn, node)["parent_id"]
            for s in db.children(conn, parent):
                if s["id"] != node and s["created_at"] <= ev["ts"]:
                    out.append((s["id"], mode, w.sibling_spread, 0.0))
        return out, 0.0
    if kind == "rating":
        out = []
        for root_id, rating in diag.get("ratings", {}).items():
            da, dbeta = mastery.rating_prior(int(rating), w.rating_prior)
            out += [(int(root_id), m, da, dbeta) for m in MODES]
        return out, 0.0
    return [], 0.0


def apply_event(conn, settings: Settings, ev: dict, now: datetime) -> list[dict]:
    """Apply an event's deltas (and depth nudge) and return what changed, for the debug drawer."""
    changes, nudge = deltas_for(conn, ev, settings.weights)
    applied = [apply_delta(conn, settings, n, m, da, dbeta, now) for n, m, da, dbeta in changes]
    if nudge and ev.get("node_id"):
        nudge_depth(conn, ev["node_id"], nudge)
    return applied


def replay(conn, settings: Settings) -> int:
    """Rebuild mastery and preferences from scratch by replaying every node creation and interaction."""
    conn.execute("DELETE FROM mastery")
    conn.execute("DELETE FROM preferences")
    events = [(n["created_at"], 0, n["id"], dict(n)) for n in db.all_nodes(conn) if n["parent_id"] is not None]
    events += [(r["ts"], 1, r["id"], dict(r)) for r in conn.execute("SELECT * FROM interactions")]
    events.sort(key=lambda e: e[:3])
    for ts, is_interaction, _, row in events:
        now = db.parse_ts(ts)
        if is_interaction:
            apply_event(conn, settings, row, now)
        elif row["source"] != "seed":
            init_child_cells(conn, settings, row["id"], now)
    conn.commit()
    return len(events)


# ---------------------------------------------------------------- profile (SPEC §5.2)

def recent_gaps(conn, node_id: int, days: int = 60, limit: int = 5) -> list[dict]:
    """Recent 'lost me' diagnoses on this branch (the node, its ancestors and its descendants)."""
    ancestors = {n["id"] for n in db.path(conn, node_id)}
    since = iso(utcnow() - timedelta(days=days))
    out = []
    for r in conn.execute("SELECT * FROM interactions WHERE kind='feedback' AND feedback='lost_me' "
                          "AND diagnosis IS NOT NULL AND ts >= ? ORDER BY id DESC", (since,)):
        if r["node_id"] is None:
            continue
        on_branch = r["node_id"] in ancestors or node_id in {n["id"] for n in db.path(conn, r["node_id"])}
        d = json.loads(r["diagnosis"])
        if on_branch and d.get("kind") != "missed_word":
            out.append({"kind": d.get("kind"), "term": d.get("term"), "explanation": d.get("explanation"),
                        "when": r["ts"][:10]})
        if len(out) >= limit:
            break
    return out


def profile(conn, settings: Settings, node_id: int, mode: str, now: datetime) -> dict:
    view = MapView(conn, settings, now)
    node = db.get_node(conn, node_id)
    path = db.path(conn, node_id)

    def per_mode(nid: int) -> dict:
        out = {}
        for m in MODES:
            mean_, ev = view.mode(nid, m)
            out[m] = {"mean": round(mean_, 2), "sd": round(mastery.beta_variance_from(mean_, ev) ** 0.5, 2)}
        return out

    def brief(n) -> dict:
        return {"name": n["name"], **view.summary(n["id"])}

    parent = path[-2] if len(path) > 1 else None
    siblings = [s for s in db.children(conn, node["parent_id"]) if s["id"] != node_id] if parent else []
    return {
        "topic": " › ".join(n["name"] for n in path),
        "mode": mode,
        "node": {"name": node["name"], "description": node["description"],
                 "mastery_by_mode": per_mode(node_id), "overall": view.summary(node_id)},
        "parent": brief(parent) if parent else None,
        "siblings": [brief(s) for s in siblings][:12],
        "children": [brief(c) for c in db.children(conn, node_id)][:12],
        "recent_gaps": recent_gaps(conn, node_id),
        "depth_preference": round(db.get_depth_pref(conn, branch_of(conn, node_id)), 2),
    }


# ---------------------------------------------------------------- frontier (SPEC §3.4)

def frontier(conn, settings: Settings, query_vec: np.ndarray | None, primary_id: int, now: datetime,
             k: int = 3) -> list[dict]:
    """Nodes just past what I know, near the query. Ranked by similarity × (parent − own)."""
    view = MapView(conn, settings, now)
    nodes = db.all_nodes(conn)
    excluded = {n["id"] for n in db.path(conn, primary_id)}
    by_parent = view.children
    candidates = []
    for n in nodes:
        if n["id"] in excluded or n["parent_id"] is None or n["embedding"] is None:
            continue
        own_mean, own_ev = view.node(n["id"])
        parent_mean = view.node(n["parent_id"])[0]
        sib_means = [view.node(s)[0] for s in by_parent.get(n["parent_id"], []) if s != n["id"]]
        if not mastery.is_frontier(own_mean, own_ev, parent_mean, sib_means,
                                   settings.frontier_low, settings.frontier_high):
            continue
        sim = float(np.dot(db.from_blob(n["embedding"]), query_vec)) if query_vec is not None else 1.0
        known = max([parent_mean, *sib_means])
        candidates.append({"node_id": n["id"], "name": n["name"], "similarity": max(0.0, sim),
                           "parent_mean": known, "own_mean": own_mean})
    return mastery.rank_frontier(candidates, k)
