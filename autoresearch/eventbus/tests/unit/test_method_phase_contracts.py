"""EventBus contracts for the method and deliberation phase.

These events are used by the production orchestrator and handlers.  Keeping
them in the registry matters because the real EventBus rejects unknown event
types even though the in-process test/web bus does not.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from autoresearch.eventbus.broker.registry import Registry


EVENTBUS_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def registry() -> Registry:
    return Registry(EVENTBUS_ROOT / "eventbus-registry.yaml")


CASES = {
    "method.generation_requested": (
        "ResearchOrchestrator",
        {
            "problem_model_ref": "spec://problem_model/s/v1",
            "literature_brief_ref": "spec://literature_brief/s/v1",
            "critique_report_ref": None,
            "human_idea_seed_ref": None,
            "novelty_audit_report_ref": None,
            "research_memory_ref": "spec://research_memory/s/v1",
            "method_not_working": False,
            "component_ablation_failed": False,
            "ablation_failed_component": "",
        },
    ),
    "method.generated": (
        "MethodAgent",
        {"method_ref": "spec://method/s/v1"},
    ),
    "method.deliberation_requested": (
        "ResearchOrchestrator",
        {
            "problem_model_ref": "spec://problem_model/s/v1",
            "literature_brief_ref": "spec://literature_brief/s/v1",
            "human_idea_seed_ref": None,
            "method_ref": "spec://method/s/v1",
        },
    ),
    "method.deliberated": (
        "MethodAgent",
        {"method_ref": "spec://method/s/v1"},
    ),
    "method.revise_requested": (
        "ResearchOrchestrator",
        {
            "problem_model_ref": "spec://problem_model/s/v1",
            "literature_brief_ref": "spec://literature_brief/s/v1",
            "method_ref": "spec://method/s/v1",
            "experiment_feedback": "the prior run did not improve",
            "research_memory_ref": "spec://research_memory/s/v1",
        },
    ),
    "method.confirmed": (
        "Human",
        {"inject": False},
    ),
    "deliberation.requested": (
        "ResearchOrchestrator",
        {
            "research_spec_ref": "spec://research_spec/s/v1",
            "literature_brief_ref": "spec://literature_brief/s/v1",
        },
    ),
    "deliberation.ready": (
        "DeliberationAgent",
        {"deliberation_report_ref": "spec://deliberation_report/s/final"},
    ),
}


@pytest.mark.parametrize("event_type", CASES)
def test_method_phase_event_is_registered_and_accepts_production_payload(
    registry: Registry, event_type: str
) -> None:
    emitter, payload = CASES[event_type]
    entry = registry.get(event_type)

    assert entry.emitters == {emitter}
    assert list(registry.get_payload_validator(event_type, 1).iter_errors(payload)) == []
