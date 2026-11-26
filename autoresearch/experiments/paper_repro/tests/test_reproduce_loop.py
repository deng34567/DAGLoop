"""Reproduce-until-close (meeting 2026-08-03): retry a paper reproduction until the measured metric
approaches the paper's reported value, bounded. Pure -- attempt_fn is injected; no runner/network."""
from __future__ import annotations

import pytest

from autoresearch.experiments.paper_repro.reproduce_loop import (
    is_close,
    reproduce_until_close,
    reproduction_gap,
)


def test_gap_relative():
    assert reproduction_gap(0.90, 1.00, False) == pytest.approx(0.10)
    assert reproduction_gap(99.0, 99.0, False) == 0.0
    assert reproduction_gap(None, 1.0, False) is None
    assert reproduction_gap(1.0, 0.0, False) is None     # target 0 -> undefined


def test_is_close_within_tol():
    assert is_close(0.985, 1.00, False, tol=0.05) is True     # within 5%
    assert is_close(0.80, 1.00, False, tol=0.05) is False     # 20% off


def test_is_close_when_better_than_paper():
    # higher-is-better: beating the paper's number counts as reproduced
    assert is_close(0.995, 0.99, False, tol=0.001) is True
    # lower-is-better: a LOWER (better) loss than the paper also counts
    assert is_close(1.20, 1.40, True, tol=0.001) is True


def test_stops_as_soon_as_close():
    seq = iter([0.80, 0.985, 0.99])   # 2nd attempt is close -> stop before the 3rd
    def attempt(_fb):
        v = next(seq)
        return ({"v": v}, v)
    res = reproduce_until_close(attempt, 1.00, "accuracy", False, tol=0.05, max_attempts=3)
    assert res.close is True and res.attempts == 2
    assert res.best_measured == 0.985


def test_bounded_and_keeps_closest_when_never_close():
    seq = iter([0.50, 0.70, 0.85])    # never within 5% of 1.00, but 0.85 is closest
    def attempt(_fb):
        v = next(seq)
        return ({"v": v}, v)
    res = reproduce_until_close(attempt, 1.00, "accuracy", False, tol=0.05, max_attempts=3)
    assert res.attempts == 3 and res.close is False
    assert res.best_measured == 0.85          # kept the closest attempt


def test_feedback_flows_into_retry():
    seen = []
    seq = iter([0.50, 0.99])
    def attempt(fb):
        seen.append(fb)
        v = next(seq)
        return ({"v": v}, v)
    reproduce_until_close(attempt, 1.00, "accuracy", False, tol=0.05, max_attempts=3)
    assert seen[0] == ""                       # first attempt: no feedback
    assert "论文" in seen[1] and "0.5" in seen[1]   # retry told the gap vs the paper value


def test_unknown_target_runs_once():
    calls = {"n": 0}
    def attempt(_fb):
        calls["n"] += 1
        return ({"v": 0.9}, 0.9)
    res = reproduce_until_close(attempt, None, "accuracy", False, tol=0.05, max_attempts=5)
    assert calls["n"] == 1 and res.attempts == 1     # nothing to chase -> single run


def test_none_measured_does_not_win():
    seq = iter([None, 0.9])
    def attempt(_fb):
        v = next(seq)
        return ({"v": v}, v)
    res = reproduce_until_close(attempt, 1.00, "accuracy", False, tol=0.05, max_attempts=2)
    assert res.best_measured == 0.9            # the None attempt never becomes best
