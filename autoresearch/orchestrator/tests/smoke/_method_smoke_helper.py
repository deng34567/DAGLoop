"""Shared helper for smoke tests: wire the phase-3.1 MethodAgent with a canned method."""
from __future__ import annotations

import json

from autoresearch.agents.method_agent.service import build_method_service


_VALID_METHOD = {
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
            "ablation_test": "signed_metric_improvement > 0",
            "functional_check": "",
            "implementation_check": "the captured diff and runtime path both apply standardization",
            "optimization_space": "refine the statistics without changing the evaluator",
            "downstream_contract": "provide standardized features to the fixed classifier",
        }
    ],
    "critique_of_other": [],
}


class MethodFakeLLM:
    """Canned MethodAgent LLM: returns a valid method JSON each call."""
    def __init__(self, response: str | None = None):
        self._resp = response or json.dumps(_VALID_METHOD)

    def chat(self, *, messages, response_format=None, max_tokens=None):
        system = str((messages or [{}])[0].get("content") or "")
        if "METHOD CRITIC" in system:
            content = json.dumps({"problems": []})
        elif "You judge whether" in system:
            content = json.dumps({"converged": True, "reason": "the methods agree"})
        else:
            content = self._resp
        return type("R", (), {"content": content})()


def wire_method_agent(bus, store, bus_client_view):
    """Subscribe a MethodAgent (canned method) on ``bus`` -- phase 3.1 GeneratingMethod consumer."""
    build_method_service(client=bus_client_view(bus, "MethodAgent"), spec_store=store,
                         llm_client=MethodFakeLLM())
