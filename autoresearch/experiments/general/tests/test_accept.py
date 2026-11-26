"""Task 5: minimal P1 acceptance -- reproduce + isolation evidence -> verdict.
PURE (no container/LLM). ``isolated`` comes from Task 4 evidence, not hardcoded.
"""
from __future__ import annotations

from autoresearch.experiments.general import accept


def _res(baseline, method, valid=True, seed=0):
    return {"baseline_metric": baseline, "method_metric": method,
            "valid": valid, "seed": seed}


def test_non_numeric_metric_does_not_crash_decide():
    # A dict metric (the real-world bug) must NOT crash decide() with dict*float;
    # it is treated as "not improved" -> fail, with no exception.
    primary = _res({"average_bins": 10}, {"average_bins": 8}, seed=0)
    reproduce = _res({"average_bins": 11}, {"average_bins": 9}, seed=1)
    out = accept.decide(primary, reproduce, isolated=True)   # must not raise
    assert out["verdict"] == "fail" and out["accepted"] is False


def test_pass_when_better_reproduced_and_isolated():
    # lower is better: method 8 bins < baseline 10 bins, reproduced on a new seed
    primary = _res(10, 8, seed=0)
    reproduce = _res(11, 9, seed=1)
    out = accept.decide(primary, reproduce, isolated=True)
    assert out["verdict"] == "pass"
    assert out["accepted"] is True
    assert out["reasons"]


def test_fail_on_illegal_solution():
    primary = _res(10, None, valid=False)
    reproduce = _res(10, None, valid=False)
    out = accept.decide(primary, reproduce, isolated=True)
    assert out["verdict"] == "fail"
    assert out["accepted"] is False


def test_fail_when_not_better_than_baseline():
    # method uses MORE bins than baseline (lower is better) -> no improvement
    primary = _res(10, 12)
    reproduce = _res(10, 12)
    out = accept.decide(primary, reproduce, isolated=True)
    assert out["verdict"] == "fail"


def test_inconclusive_when_not_reproduced():
    primary = _res(10, 8)          # improves on seed 0
    reproduce = _res(10, 11)       # but worse on the fresh seed
    out = accept.decide(primary, reproduce, isolated=True)
    assert out["verdict"] == "inconclusive"


def test_inconclusive_when_isolation_broken():
    primary = _res(10, 8)
    reproduce = _res(11, 9)
    out = accept.decide(primary, reproduce, isolated=False)  # harness bypassed the isolator
    assert out["verdict"] == "inconclusive"
    assert out["accepted"] is False


def test_higher_is_better_direction():
    # accuracy-style metric: higher method beats baseline
    primary = _res(0.70, 0.85)
    reproduce = _res(0.71, 0.84)
    out = accept.decide(primary, reproduce, isolated=True, lower_is_better=False)
    assert out["verdict"] == "pass"


# --- B5: multiple reproduce seeds, majority must hold --------------------------

def test_majority_reproduce_passes():
    from autoresearch.experiments.general.accept import decide
    primary = _res(10, 8)  # improved (lower better)
    reproduces = [_res(10, 8), _res(10, 8), _res(10, 12)]  # 2/3 improved -> majority
    assert decide(primary, reproduces, isolated=True)["verdict"] == "pass"


def test_minority_reproduce_inconclusive():
    from autoresearch.experiments.general.accept import decide
    primary = _res(10, 8)
    reproduces = [_res(10, 8), _res(10, 12), _res(10, 12)]  # only 1/3 improved
    assert decide(primary, reproduces, isolated=True)["verdict"] == "inconclusive"


def test_single_reproduce_backward_compatible():
    from autoresearch.experiments.general.accept import decide
    assert decide(_res(10, 8), _res(10, 8), isolated=True)["verdict"] == "pass"       # dict
    assert decide(_res(10, 8), [_res(10, 12)], isolated=True)["verdict"] == "inconclusive"
