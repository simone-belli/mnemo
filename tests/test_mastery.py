"""Unit tests for the knowledge model's maths (SPEC §3.3–3.4, §12)."""
from datetime import datetime, timedelta, timezone

import pytest

from app import mastery as M
from app.mastery import Cell

T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
H = 120.0


def test_prior_mean_variance_evidence():
    c = Cell()
    assert c.mean == 0.5
    assert c.variance == pytest.approx(1 / 12)
    assert c.evidence == 0


def test_update_rules_knew_that_and_simpler():
    c = M.update(Cell(), 1.0, 0.0, T0, H)          # knew that
    assert (c.alpha, c.beta) == (2.0, 1.0)
    assert c.mean == pytest.approx(2 / 3)
    assert c.last_seen == T0
    c = M.update(c, 0.0, 0.5, T0, H)               # simpler, same moment: no decay
    assert (c.alpha, c.beta) == (2.0, 1.5)


def test_more_evidence_less_variance():
    a = M.update(Cell(), 2, 2, T0, H)
    assert a.mean == 0.5 and a.variance < Cell().variance


def test_decay_halves_evidence_after_one_half_life():
    c = Cell(5.0, 3.0, T0)
    d = M.decay(c, T0 + timedelta(days=H), H)
    assert d.alpha == pytest.approx(1 + 4 / 2)
    assert d.beta == pytest.approx(1 + 2 / 2)
    assert d.evidence == pytest.approx(c.evidence / 2)


def test_decay_tends_to_prior_and_never_past_it():
    c = Cell(9.0, 1.0, T0)
    far = M.decay(c, T0 + timedelta(days=50 * H), H)
    assert far.mean == pytest.approx(0.5, abs=1e-6)
    assert M.decay(c, T0, H) == c                   # no time, no decay
    assert M.decay(Cell(3, 3, None), T0, H) == Cell(3, 3, None)   # never seen: nothing to decay


def test_update_decays_before_adding():
    c = Cell(5.0, 1.0, T0)
    later = T0 + timedelta(days=H)
    u = M.update(c, 1.0, 0.0, later, H)
    assert u.alpha == pytest.approx(1 + 4 / 2 + 1)
    assert u.last_seen == later


def test_touch_only_moves_last_seen_and_decay():
    c = M.touch(Cell(3, 1, T0), T0, H)
    assert (c.alpha, c.beta, c.last_seen) == (3, 1, T0)


def test_combine_is_evidence_weighted():
    assert M.combine([(0.9, 3.0), (0.3, 1.0)], 0.5) == pytest.approx((0.75, 4.0))
    assert M.combine([(0.9, 0.0), (0.1, 0.0)], 0.42) == (0.42, 0.0)


def test_node_level_across_modes():
    per_mode = {"descriptive": (0.8, 2.0), "explanatory": (0.2, 2.0), "normative": (0.5, 0.0)}
    assert M.node_level(per_mode)[0] == pytest.approx(0.5)
    assert M.node_level({"a": (0.6, 0), "b": (0.8, 0)})[0] == pytest.approx(0.7)   # no evidence: plain mean


def test_aggregate_tree_upwards():
    #   1 ── 2
    #     └─ 3 ── 4
    children = {None: [1], 1: [2, 3], 3: [4]}
    cells = {(2, "d"): Cell(4, 1, T0), (4, "d"): Cell(1, 3, T0), (1, "d"): Cell(2, 2, T0)}
    agg = M.aggregate_tree(children, cells, ["d"], T0, H)
    assert agg[4]["d"] == pytest.approx((0.25, 2.0))
    assert agg[3]["d"] == pytest.approx((0.25, 2.0))           # no own evidence: just the child
    assert agg[2]["d"] == pytest.approx((0.8, 3.0))
    # node 1: own (0.5, 2), child 2 (0.8, 3), child 3 (0.25, 2)
    assert agg[1]["d"][0] == pytest.approx((0.5 * 2 + 0.8 * 3 + 0.25 * 2) / 7)
    assert agg[1]["d"][1] == pytest.approx(7.0)


def test_aggregate_tree_applies_decay():
    children = {None: [1]}
    cells = {(1, "d"): Cell(5, 1, T0)}
    later = M.aggregate_tree(children, cells, ["d"], T0 + timedelta(days=H), H)[1]["d"]
    assert later == pytest.approx((3 / 4, 2.0))


def test_child_prior_is_weak_and_at_parent_mean():
    c = M.child_prior(0.8)
    assert c.alpha + c.beta == pytest.approx(2.0)
    assert c.mean == pytest.approx(0.8)
    assert M.child_prior(1.0).mean == pytest.approx(0.95)       # clamped: never certain


def test_rating_prior():
    assert M.rating_prior(5) == (2.0, 0.0)
    assert M.rating_prior(1) == (0.0, 2.0)
    assert M.rating_prior(3) == (1.0, 1.0)


def test_is_frontier():
    assert M.is_frontier(0.3, 3, parent_mean=0.8, sibling_means=[])           # low here, known above
    assert M.is_frontier(0.5, 0.0, parent_mean=0.5, sibling_means=[0.7])      # uncertain, sibling known
    assert not M.is_frontier(0.3, 3, parent_mean=0.5, sibling_means=[0.5])    # nothing known around
    assert not M.is_frontier(0.7, 3, parent_mean=0.9, sibling_means=[])       # already known


def test_frontier_ranking():
    cands = [
        {"name": "near, big gap", "similarity": 0.9, "parent_mean": 0.9, "own_mean": 0.2},
        {"name": "far, big gap", "similarity": 0.2, "parent_mean": 0.9, "own_mean": 0.2},
        {"name": "near, small gap", "similarity": 0.9, "parent_mean": 0.7, "own_mean": 0.6},
        {"name": "known", "similarity": 0.9, "parent_mean": 0.7, "own_mean": 0.8},
    ]
    ranked = M.rank_frontier(cands, k=3)
    assert [c["name"] for c in ranked] == ["near, big gap", "far, big gap", "near, small gap"]
    assert ranked[0]["score"] == pytest.approx(0.9 * 0.7)


def test_quiz_delta_scaled_by_difficulty():
    hard_right = M.quiz_delta(1.0, 0.9)
    easy_right = M.quiz_delta(1.0, 0.1)
    assert hard_right[0] > easy_right[0] and hard_right[1] == easy_right[1] == 0
    easy_wrong = M.quiz_delta(0.0, 0.1)
    hard_wrong = M.quiz_delta(0.0, 0.9)
    assert easy_wrong[1] > hard_wrong[1]
    a, b = M.quiz_delta(0.5, 0.5)
    assert a == b == 0.5


def test_levels_and_explain_one_simpler():
    assert [M.LEVELS[M.level(m)] for m in (0.0, 0.19, 0.2, 0.5, 0.6, 0.79, 0.8, 1.0)] == [
        "first contact", "first contact", "introductory", "intermediate", "advanced", "advanced", "expert", "expert"]
    assert M.explain_level(M.level(0.5)) == M.level(0.5) - 1
    assert M.explain_level(0) == 0                                    # nothing simpler than first contact


def test_staircase_follows_answers_and_carries_over_to_new_domains():
    step = dict(start=0.25, up=0.1, down=0.15)
    assert M.staircase([], **step) == (0.25, {})
    overall, levels = M.staircase([(1, 1.0), (1, 1.0), (1, 1.0)], **step)
    assert levels[1] == pytest.approx(0.55) and overall == pytest.approx(0.55)
    overall, levels = M.staircase([(1, 1.0), (1, 1.0), (1, 1.0), (2, 0.0)], **step)
    assert levels[2] == pytest.approx(0.40) and levels[1] == pytest.approx(0.55)   # 2 starts where 1 got to
    _, levels = M.staircase([(1, 0.5)], **step)
    assert levels[1] == pytest.approx(0.225)                         # partial credit: a small step down
    _, levels = M.staircase([(1, 0.0)] * 10, **step)
    assert levels[1] == 0.05                                          # never below the floor
