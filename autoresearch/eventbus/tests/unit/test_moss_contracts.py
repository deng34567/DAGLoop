"""Unit tests for the MOSS contract slice (MOSS Step 1, Task 1).

Covers: moss.intervention_requested, moss.intervention_completed (v1); the
MossDecision data-contract schema; the SpecStore whitelist extension for
moss_intervention; and confirmation that MOSS is already a legal
agent.failed / envelope emitter (no changes needed there).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from jsonschema import Draft202012Validator

from autoresearch.eventbus.broker.registry import Registry
from autoresearch.spec_store import EmitterNotAllowed, SpecStore

EVENTBUS_ROOT = Path(__file__).resolve().parents[2]
REGISTRY_PATH = EVENTBUS_ROOT / "eventbus-registry.yaml"
EXAMPLES_DIR = EVENTBUS_ROOT / "examples"

AUTORESEARCH_ROOT = EVENTBUS_ROOT.parent
MOSS_AGENT_ROOT = AUTORESEARCH_ROOT / "agents" / "moss_agent"
MOSS_DECISION_SCHEMA_PATH = (
    MOSS_AGENT_ROOT / "data-contracts" / "moss_decision.schema.yaml"
)

MOSS_EVENTS = [
    "moss.intervention_requested",
    "moss.intervention_completed",
]


@pytest.fixture
def registry() -> Registry:
    return Registry(REGISTRY_PATH)


# ---------------------------------------------------------------------------
# New events: moss.intervention_requested, moss.intervention_completed
# ---------------------------------------------------------------------------


def test_registry_loads_moss_validators(registry: Registry):
    for event_type in MOSS_EVENTS:
        assert registry.get_payload_validator(event_type, 1)


@pytest.mark.parametrize("event_type", MOSS_EVENTS)
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


def test_moss_intervention_requested_registry_entry(registry: Registry):
    entry = registry.get("moss.intervention_requested")
    assert entry.direction == "outbound"
    assert entry.emitters == {"ResearchOrchestrator"}
    assert 1 in entry.schema_versions


def test_moss_intervention_completed_registry_entry(registry: Registry):
    entry = registry.get("moss.intervention_completed")
    assert entry.direction == "outbound"
    assert entry.emitters == {"MOSS"}
    assert 1 in entry.schema_versions


def test_moss_intervention_requested_counter_snapshot_null_validates(
    registry: Registry,
):
    validator = registry.get_payload_validator("moss.intervention_requested", 1)
    payload = {"reason": "loop detected", "counter_snapshot": None}
    errors = list(validator.iter_errors(payload))
    assert errors == [], (
        f"moss.intervention_requested with counter_snapshot=null should "
        f"validate, got: {[e.message for e in errors]}"
    )


def test_moss_intervention_requested_without_counter_snapshot_validates(
    registry: Registry,
):
    validator = registry.get_payload_validator("moss.intervention_requested", 1)
    payload = {"reason": "loop detected"}
    errors = list(validator.iter_errors(payload))
    assert errors == [], (
        f"moss.intervention_requested without counter_snapshot should still "
        f"validate (optional), got: {[e.message for e in errors]}"
    )


def test_moss_intervention_requested_accepts_diagnostic_artifact_refs(registry: Registry):
    """The orchestrator's real MOSS payload carries evidence for root-cause diagnosis."""
    validator = registry.get_payload_validator("moss.intervention_requested", 1)
    payload = {
        "reason": "component_ablation_failed",
        "counter_snapshot": {
            "component_revision_round": 1,
            "ablation_failed_component": "optimizer",
        },
        "research_memory_ref": "spec://research_memory/sess-1/v1",
        "method_ref": "spec://method/sess-1/v1",
        "run_record_ref": "spec://run_record/sess-1/v1",
    }
    errors = list(validator.iter_errors(payload))
    assert errors == [], [e.message for e in errors]


# ---------------------------------------------------------------------------
# MossDecision data-contract schema
# ---------------------------------------------------------------------------


def test_moss_decision_fixture_validates():
    schema = yaml.safe_load(MOSS_DECISION_SCHEMA_PATH.read_text())
    validator = Draft202012Validator(schema)

    fixture = {
        "reason": "loop detected",
        "action": "inject_prompt_patch",
        "prompt_patch": {"agent": "CoderAgent", "content": "Try a different approach."},
        "rationale": "Coder has repeated the same failing patch 3 times.",
        "degraded": False,
    }
    errors = list(validator.iter_errors(fixture))
    assert errors == [], (
        f"MossDecision fixture should validate, got: {[e.message for e in errors]}"
    )


def test_moss_decision_with_null_prompt_patch_validates():
    schema = yaml.safe_load(MOSS_DECISION_SCHEMA_PATH.read_text())
    validator = Draft202012Validator(schema)

    fixture = {
        "reason": "loop detected",
        "action": "escalate",
        "prompt_patch": None,
        "degraded": False,
    }
    errors = list(validator.iter_errors(fixture))
    assert errors == [], (
        f"MossDecision with prompt_patch=null should validate, got: "
        f"{[e.message for e in errors]}"
    )


def test_moss_decision_with_prompt_patch_object_validates():
    schema = yaml.safe_load(MOSS_DECISION_SCHEMA_PATH.read_text())
    validator = Draft202012Validator(schema)

    fixture = {
        "reason": "loop detected",
        "action": "inject_prompt_patch",
        "prompt_patch": {"agent": "RunnerAgent", "content": "Reduce batch size."},
        "degraded": True,
    }
    errors = list(validator.iter_errors(fixture))
    assert errors == [], (
        f"MossDecision with prompt_patch object should validate, got: "
        f"{[e.message for e in errors]}"
    )


def test_moss_decision_prompt_patch_missing_content_fails():
    schema = yaml.safe_load(MOSS_DECISION_SCHEMA_PATH.read_text())
    validator = Draft202012Validator(schema)

    fixture = {
        "reason": "loop detected",
        "action": "inject_prompt_patch",
        "prompt_patch": {"agent": "RunnerAgent"},
        "degraded": True,
    }
    errors = list(validator.iter_errors(fixture))
    assert errors, "MossDecision with prompt_patch missing `content` should NOT validate"


def test_moss_decision_missing_degraded_fails():
    schema = yaml.safe_load(MOSS_DECISION_SCHEMA_PATH.read_text())
    validator = Draft202012Validator(schema)

    fixture = {
        "reason": "loop detected",
        "action": "escalate",
    }
    errors = list(validator.iter_errors(fixture))
    assert errors, "MossDecision missing `degraded` should NOT validate"


def test_moss_decision_extra_field_fails():
    schema = yaml.safe_load(MOSS_DECISION_SCHEMA_PATH.read_text())
    validator = Draft202012Validator(schema)

    fixture = {
        "reason": "loop detected",
        "action": "escalate",
        "degraded": False,
        "unexpected_field": "nope",
    }
    errors = list(validator.iter_errors(fixture))
    assert errors, "MossDecision with an extra property should NOT validate"


# ---------------------------------------------------------------------------
# SpecStore: moss_intervention kind
# ---------------------------------------------------------------------------


def test_store_accepts_moss_intervention_for_moss(tmp_path):
    store = SpecStore(tmp_path / "store")
    try:
        store.put(
            kind="moss_intervention",
            session_id="s1",
            key="intervention-1",
            obj={
                "reason": "loop detected",
                "action": "escalate",
                "degraded": False,
            },
            emitter="MOSS",
        )
    finally:
        store.close()


def test_store_rejects_wrong_emitter_for_moss_intervention(tmp_path):
    store = SpecStore(tmp_path / "store")
    try:
        with pytest.raises(EmitterNotAllowed):
            store.put(
                kind="moss_intervention",
                session_id="s1",
                key="intervention-1",
                obj={},
                emitter="ResearchOrchestrator",
            )
    finally:
        store.close()


# ---------------------------------------------------------------------------
# MOSS already a legal emitter (agent.failed / envelope) -- confirm, no changes.
# ---------------------------------------------------------------------------


def test_agent_failed_already_includes_moss(registry: Registry):
    entry = registry.get("agent.failed")
    assert "MOSS" in entry.emitters


def test_envelope_schema_already_allows_moss_emitter(registry: Registry):
    validator = registry.envelope_validator
    emitter_enum = validator.schema["properties"]["emitter"]["enum"]
    assert "MOSS" in emitter_enum
