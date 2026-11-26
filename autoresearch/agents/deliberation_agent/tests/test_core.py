"""DeliberationAgent core: bounded N-angle cross-reference -> schema-valid report."""
from __future__ import annotations

import json

import pytest
import yaml
from jsonschema import Draft202012Validator

from autoresearch.agents.deliberation_agent import core

_VALIDATOR = Draft202012Validator(
    yaml.safe_load(
        (core.Path(core.__file__).resolve().parent / "data-contracts"
         / "deliberation_report.schema.yaml").read_text(encoding="utf-8")))

_UNIVERSAL = {
    "approach": "A", "supporting": "S", "expected_baseline": "B=1.0", "revised": "r",
    "chosen_direction": "D", "chosen_baseline_hint": "B=1.0",
    "convergence": ["c1"], "divergence": ["d1"], "synthesis": "deep synth",
}

_SPEC = {"research_goal": "does X improve Y on Z?"}
_LIT = {"summary": "prior work suggests Y varies with X"}


class _Resp:
    def __init__(self, content):
        self.content = content


class _FakeLLM:
    """Returns a universal JSON for every call, unless a message contains ``raise_if`` (then
    raises, simulating a dead angle)."""
    def __init__(self, universal=None, raise_if=None):
        self._universal = universal if universal is not None else _UNIVERSAL
        self._raise_if = raise_if
        self.calls = []

    def chat(self, *, messages, **_):
        self.calls.append(messages)
        blob = " ".join(m.get("content", "") for m in messages)
        if self._raise_if and self._raise_if in blob:
            raise RuntimeError("simulated LLM failure")
        return _Resp(json.dumps(self._universal))


def test_deliberate_n3_produces_valid_report_with_2n_plus_1_calls():
    llm = _FakeLLM()
    r = core.deliberate(_SPEC, _LIT, llm, n_angles=3)
    rep = r.deliberation_report
    assert _VALIDATOR.is_valid(rep), sorted(e.message for e in _VALIDATOR.iter_errors(rep))
    assert len(rep["angles"]) == 3
    assert {a["angle"] for a in rep["angles"]} == set(core.DEFAULT_ANGLES)
    assert rep["synthesis"] == "deep synth"
    assert rep["convergence"] == ["c1"] and rep["divergence"] == ["d1"]
    assert rep["chosen_direction"] == "D" and rep["chosen_baseline_hint"] == "B=1.0"
    assert len(llm.calls) == 2 * 3 + 1          # 3 independent + 3 cross-ref + 1 synthesis


def test_deliberate_n2_is_bounded():
    llm = _FakeLLM()
    r = core.deliberate(_SPEC, _LIT, llm, n_angles=2)
    assert len(r.deliberation_report["angles"]) == 2
    assert len(llm.calls) == 2 * 2 + 1


def test_deliberate_drops_a_dead_angle_but_still_produces_report():
    # the empirical-baseline-first angle's LLM calls raise -> that angle is dropped, the
    # step still yields a valid report from the surviving angles (fail-soft).
    llm = _FakeLLM(raise_if="empirical-baseline-first")
    r = core.deliberate(_SPEC, _LIT, llm, n_angles=3)
    rep = r.deliberation_report
    assert _VALIDATOR.is_valid(rep)
    angles = {a["angle"] for a in rep["angles"]}
    assert "empirical-baseline-first" not in angles
    assert len(angles) == 2


def test_deliberate_all_angles_dead_raises():
    llm = _FakeLLM(raise_if="lens")   # every angle/synthesis system prompt contains "lens"/"MOSS"
    # use a substring present in every call: the shared context "Research goal"
    llm = _FakeLLM(raise_if="Research goal")
    with pytest.raises(core.DeliberationLLMError):
        core.deliberate(_SPEC, _LIT, llm, n_angles=3)


def test_minimal_report_is_schema_valid():
    assert _VALIDATOR.is_valid(core.minimal_report("x"))
    assert core.minimal_report()["angles"] == []


def test_norm_component_defaults_type_direct_gain_and_empty_indirect_check():
    # backward-compat: an old contract with no type defaults to the canonical direct-gain role.
    c = core._norm_component({"name": "opt", "purpose": "p", "mechanism": "m",
                              "expected_effect": "down 10%", "ablation_test": "swap back"})
    assert c["type"] == "direct_gain"
    assert c["functional_check"] == ""
    assert c["expected_effect"] == "down 10%"
    assert c["component_id"] == "opt"
    assert c["depends_on"] == []


def test_norm_component_legacy_functional_type_normalizes_to_indirect_gain():
    c = core._norm_component({"component_id": "split", "name": "decouple", "type": "FUNCTIONAL",
                              "depends_on": ["encoder"],
                              "functional_check": "|corr(C,S)|<0.1"})
    assert c["type"] == "indirect_gain"
    assert c["functional_check"] == "|corr(C,S)|<0.1"
    assert c["component_id"] == "split"
    assert c["depends_on"] == ["encoder"]


def test_norm_component_bad_type_falls_back_to_direct_gain():
    assert core._norm_component({"name": "x", "type": "weird"})["type"] == "direct_gain"


def test_norm_method_normalizes_each_component():
    m = core._norm_method({"approach": "a", "components": [
        {"name": "c1", "type": "functional", "functional_check": "t<0.5"},
        {"name": "c2"}]})
    assert [c["type"] for c in m["components"]] == ["indirect_gain", "direct_gain"]
    assert all("functional_check" in c and "ablation_test" in c for c in m["components"])
