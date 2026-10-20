"""The knowledge model's maths (SPEC §3.3–3.4), as pure functions.

Each (node, mode) cell is a Beta(alpha, beta) belief about how well I know that cell.
Nothing here touches the database; `knowledge.py` feeds it rows.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, Mapping

PRIOR = (1.0, 1.0)


@dataclass(frozen=True)
class Cell:
    alpha: float = 1.0
    beta: float = 1.0
    last_seen: datetime | None = None

    @property
    def mean(self) -> float:
        return mean(self.alpha, self.beta)

    @property
    def variance(self) -> float:
        return variance(self.alpha, self.beta)

    @property
    def evidence(self) -> float:
        return evidence(self.alpha, self.beta)


def mean(a: float, b: float) -> float:
    return a / (a + b)


def variance(a: float, b: float) -> float:
    s = a + b
    return a * b / (s * s * (s + 1))


def evidence(a: float, b: float) -> float:
    """Pseudo-observations beyond the Beta(1, 1) prior; never negative."""
    return max(0.0, a + b - 2.0)


# ---------------------------------------------------------------- decay

def decay(cell: Cell, now: datetime, half_life_days: float) -> Cell:
    """Shrink the evidence back towards the prior: x_eff = 1 + (x - 1) * 2^(-dt/h)."""
    if cell.last_seen is None or half_life_days <= 0:
        return cell
    dt = max(0.0, (now - cell.last_seen).total_seconds() / 86400)
    k = 2.0 ** (-dt / half_life_days)
    return Cell(1 + (cell.alpha - 1) * k, 1 + (cell.beta - 1) * k, cell.last_seen)


def update(cell: Cell, d_alpha: float, d_beta: float, now: datetime, half_life_days: float) -> Cell:
    """Apply an observation: decay the old evidence to `now`, add the new, stamp `last_seen`."""
    c = decay(cell, now, half_life_days)
    return Cell(max(1e-3, c.alpha + d_alpha), max(1e-3, c.beta + d_beta), now)


def touch(cell: Cell, now: datetime, half_life_days: float) -> Cell:
    """Asked about it, nothing learned: only `last_seen` moves (evidence decayed up to now first)."""
    return update(cell, 0.0, 0.0, now, half_life_days)


# ---------------------------------------------------------------- priors

def child_prior(parent_mean: float, strength: float = 2.0) -> Cell:
    """A new child starts at its parent's effective mean, as a weak prior of `strength` pseudo-counts."""
    m = min(0.95, max(0.05, parent_mean))
    return Cell(strength * m, strength * (1 - m), None)


def rating_prior(rating: int, strength: float = 2.0) -> tuple[float, float]:
    """A 1–5 self-rating as (d_alpha, d_beta) adding up to `strength`."""
    m = (min(5, max(1, rating)) - 1) / 4
    return strength * m, strength * (1 - m)


# ---------------------------------------------------------------- aggregation

def combine(items: Iterable[tuple[float, float]], fallback: float) -> tuple[float, float]:
    """Evidence-weighted mean of (mean, evidence) pairs -> (mean, total evidence).

    With no evidence at all, the mean is `fallback`."""
    items = list(items)
    total = sum(w for _, w in items)
    if total <= 1e-9:
        return fallback, 0.0
    return sum(m * w for m, w in items) / total, total


def node_level(per_mode: Mapping[str, tuple[float, float]]) -> tuple[float, float]:
    """A node's mastery across the five modes: evidence-weighted, plain average if there is none."""
    vals = list(per_mode.values())
    fallback = sum(m for m, _ in vals) / len(vals) if vals else 0.5
    return combine(vals, fallback)


def aggregate_tree(children: Mapping[int | None, list[int]], cells: Mapping[tuple[int, str], Cell],
                   modes: Iterable[str], now: datetime, half_life_days: float
                   ) -> dict[int, dict[str, tuple[float, float]]]:
    """Effective (mean, evidence) for every node and mode, bottom-up.

    A node's effective value is the evidence-weighted mean of its own (decayed) cell and its
    children's effective values. `children[None]` lists the roots."""
    modes = list(modes)
    out: dict[int, dict[str, tuple[float, float]]] = {}

    def visit(node: int) -> dict[str, tuple[float, float]]:
        kids = [visit(k) for k in children.get(node, [])]
        res = {}
        for mode in modes:
            own = decay(cells.get((node, mode), Cell()), now, half_life_days)
            items = [(own.mean, own.evidence)] + [k[mode] for k in kids]
            res[mode] = combine(items, own.mean)
        out[node] = res
        return res

    for root in children.get(None, []):
        visit(root)
    return out


# ---------------------------------------------------------------- frontier

def beta_variance_from(mean_: float, evidence_: float) -> float:
    """Variance of the Beta with this mean and evidence (alpha + beta = evidence + 2)."""
    s = evidence_ + 2
    return variance(mean_ * s, (1 - mean_) * s)


def is_frontier(own_mean: float, own_evidence: float, parent_mean: float | None,
                sibling_means: Iterable[float], low: float = 0.4, high: float = 0.65,
                uncertain_evidence: float = 1.0) -> bool:
    """Low (< low) or very uncertain here, while the parent or a sibling is well known (> high)."""
    weak_here = own_mean < low or own_evidence < uncertain_evidence
    known_around = (parent_mean is not None and parent_mean > high) or any(m > high for m in sibling_means)
    return weak_here and known_around


def frontier_score(similarity: float, parent_mean: float, own_mean: float) -> float:
    return similarity * (parent_mean - own_mean)


def rank_frontier(candidates: Iterable[dict], k: int = 3) -> list[dict]:
    """Candidates carry `similarity`, `parent_mean`, `own_mean`; best `k` with a positive score."""
    scored = []
    for c in candidates:
        s = frontier_score(c["similarity"], c["parent_mean"], c["own_mean"])
        if s > 0:
            scored.append({**c, "score": s})
    return sorted(scored, key=lambda c: -c["score"])[:k]


# ---------------------------------------------------------------- levels (SPEC §5)

LEVELS = ("first contact", "introductory", "intermediate", "advanced", "expert")


def level(mean: float) -> int:
    """The level a recall is pitched at, from the cell's mastery: bands of 0.2."""
    return min(len(LEVELS) - 1, max(0, int(mean * 5 + 1e-9)))


def explain_level(recall_level: int) -> int:
    """An explanation is pitched one level simpler than a recall."""
    return max(0, recall_level - 1)


# ---------------------------------------------------------------- feedback

def staircase(answers: Iterable[tuple[int, float]], start: float, up: float, down: float,
              lo: float = 0.05, hi: float = 0.95) -> tuple[float, dict[int, float]]:
    """Quiz difficulty from the (domain, score) answers so far, in order -> (overall, per domain).

    A right answer raises the level by `up`, a wrong one lowers it by `down`, a partial score
    moves it in between. A domain starts at the overall level the first time it comes up, so
    doing well in one domain makes the next one start higher. With up < down the levels settle
    where about down / (up + down) of the answers are right (60% for 0.1 and 0.15)."""
    overall, levels = start, {}
    for domain, score in answers:
        step = up * score - down * (1 - score)
        levels[domain] = min(hi, max(lo, levels.get(domain, overall) + step))
        overall = min(hi, max(lo, overall + step))
    return overall, levels


def quiz_delta(score: float, difficulty: float) -> tuple[float, float]:
    """alpha += score, beta += 1 - score, scaled by difficulty: a hard item answered well counts
    for more, and so does an easy one answered badly."""
    d = min(1.0, max(0.0, difficulty))
    return score * (0.5 + d), (1 - score) * (1.5 - d)
