"""Unit tests for the AutoResearch EventBus registry + example payloads.

Covers the M0 slice of the event catalog: problem_modeling.requested and
agent.failed. Later per-agent tasks will extend the registry with more events;
these tests only assert on the 2 events defined so far.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from autoresearch.eventbus.broker.registry import Registry

EVENTBUS_ROOT = Path(__file__).resolve().parents[2]
REGISTRY_PATH = EVENTBUS_ROOT / "eventbus-registry.yaml"
EXAMPLES_DIR = EVENTBUS_ROOT / "examples"

M0_EVENTS = ["problem_modeling.requested", "agent.failed"]


@pytest.fixture
def registry() -> Registry:
    return Registry(REGISTRY_PATH)


def test_registry_loads_and_validators_compile(registry: Registry):
    assert registry.get_payload_validator("problem_modeling.requested", 1)
    assert registry.get_payload_validator("agent.failed", 1)


@pytest.mark.parametrize("event_type", M0_EVENTS)
def test_examples_validate_against_schema(registry: Registry, event_type: str):
    validator = registry.get_payload_validator(event_type, 1)

    happy_payload = json.loads(
        (EXAMPLES_DIR / f"{event_type}.happy.json").read_text()
    )
    errors = list(validator.iter_errors(happy_payload))
    assert errors == [], (
        f"happy example for {event_type} should validate, got: "
        f"{[e.message for e in errors]}"
    )

    invalid_payload = json.loads(
        (EXAMPLES_DIR / f"{event_type}.invalid.json").read_text()
    )
    errors = list(validator.iter_errors(invalid_payload))
    assert errors, f"invalid example for {event_type} should NOT validate"


def test_problem_modeling_requested_registry_entry(registry: Registry):
    entry = registry.get("problem_modeling.requested")
    assert entry.direction == "outbound"
    assert entry.emitters == {"ResearchOrchestrator"}
    assert 1 in entry.schema_versions


def test_agent_failed_registry_entry(registry: Registry):
    entry = registry.get("agent.failed")
    assert entry.direction == "cross-cutting"
    assert "system" in entry.emitters
    assert "MOSS" in entry.emitters
    assert "CoderAgent" in entry.emitters
    # Orchestrator is the consumer of agent.failed, not an emitter.
    assert "ResearchOrchestrator" not in entry.emitters
