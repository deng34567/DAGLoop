"""Tests for the deterministic MC acceptance verdict (pure function -- no
container / LLM / network). Covers the five verdict paths."""
from __future__ import annotations

from autoresearch.experiments import mc_acceptance


def _exp(**over):
    base = {
        "metric_value": 4.0,
        "reproduce_metric_value": 3.6,
        "anti_cheat": {"over_budget": False, "tamper_detected": False,
                       "hardcoded_suspected": False},
    }
    base.update(over)
    return base


def test_pass_high_ratio_reproduced_no_flags():
    v = mc_acceptance.decide(_exp())
    assert v["verdict"] == "pass"
    assert v["accepted"] is True
    assert v["reasons"]  # a human-readable reason string is present


def test_fail_over_budget():
    v = mc_acceptance.decide(_exp(anti_cheat={"over_budget": True,
                                              "tamper_detected": False,
                                              "hardcoded_suspected": False}))
    assert v["verdict"] == "fail"
    assert v["accepted"] is False
    assert any("over_budget" in r for r in v["reasons"])


def test_fail_below_improvement_margin():
    v = mc_acceptance.decide(_exp(metric_value=1.05, reproduce_metric_value=1.04))
    assert v["verdict"] == "fail"
    assert v["accepted"] is False


def test_inconclusive_not_reproducible():
    # ratio clears the margin, but the reproduce ratio collapses
    v = mc_acceptance.decide(_exp(metric_value=4.0, reproduce_metric_value=1.2))
    assert v["verdict"] == "inconclusive"
    assert v["accepted"] is False


def test_inconclusive_hardcoded_suspected():
    v = mc_acceptance.decide(_exp(anti_cheat={"over_budget": False,
                                              "tamper_detected": False,
                                              "hardcoded_suspected": True}))
    assert v["verdict"] == "inconclusive"
    assert v["accepted"] is False
