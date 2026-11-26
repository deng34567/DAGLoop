"""Shared FakeLLM for DeliberationAgent tests + smoke wiring.

The deliberation core makes a VARIABLE number of calls (2N+1 for N angles), so the default
FakeLLM returns a single UNIVERSAL JSON object (valid for the angle, cross-reference, and
synthesis phases alike) for EVERY call -- robust regardless of N. Pass an explicit
``responses`` list to script an exact call sequence instead.
"""
from __future__ import annotations

import json

_UNIVERSAL = {
    "approach": "pursue the standardization ablation with a held-out split",
    "supporting": "prior work suggests scale-sensitive methods benefit from standardization",
    "expected_baseline": "raw-feature k-NN (~0.97 accuracy)",
    "revised": "no change after cross-reference",
    "chosen_direction": "compare standardized vs raw features under a fixed classifier",
    "chosen_baseline_hint": "raw-feature k-NN baseline",
    "convergence": ["a held-out test split is required"],
    "divergence": ["whether standardization helps depends on the metric's scale-sensitivity"],
    "synthesis": "the angles converge on a standardization ablation against a raw-feature baseline",
}

_METHOD = {
    "approach": "standardize the measured inputs before the fixed classifier",
    "changes_outline": ["add standardization to the measured candidate"],
    "why_it_should_win": "better-conditioned inputs should improve the frozen target metric",
    "assumptions": ["the candidate exposes the measured input transformation"],
    "components": [
        {
            "component_id": "standardize-inputs",
            "name": "standardize inputs",
            "depends_on": [],
            "purpose": "condition the measured inputs before the classifier",
            "inputs": ["baseline input features"],
            "outputs": ["standardized input features"],
            "acceptance_requires": [],
            "type": "direct_gain",
            "mechanism": "reduce scale imbalance seen by the fixed classifier",
            "expected_effect": "increase the frozen target metric by about 0.01",
            "ablation_test": "compare the official metric with and without standardization",
            "functional_check": "",
            "implementation_check": "the captured diff and runtime path both apply standardization",
            "optimization_space": "refine the standardization statistics without changing the evaluator",
            "downstream_contract": "provide standardized features to the fixed classifier",
        }
    ],
    "critique_of_other": [],
}


class _Resp:
    def __init__(self, content: str):
        self.content = content


class FakeLLM:
    """Returns queued ``responses`` in order, or the universal JSON for every call."""

    def __init__(self, responses=None):
        self._responses = list(responses) if responses else None
        self.calls: list = []

    def chat(self, *, messages, **_):
        self.calls.append(messages)
        if self._responses:
            return _Resp(self._responses.pop(0))
        system = str((messages or [{}])[0].get("content") or "")
        if "METHOD PROPOSER" in system:
            return _Resp(json.dumps(_METHOD))
        if "METHOD CRITIC" in system:
            return _Resp(json.dumps({"problems": []}))
        if "You judge whether" in system:
            return _Resp(json.dumps({"converged": True, "reason": "the methods agree"}))
        return _Resp(json.dumps(_UNIVERSAL))


__all__ = ["FakeLLM"]
