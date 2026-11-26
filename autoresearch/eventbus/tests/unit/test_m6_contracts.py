"""Unit tests for the CriticAgent contract slice (M6 Task 1).

Covers: direction.critiqued, hypothesis.critique_requested,
hypothesis.critiqued (v1), confirms direction.critique_requested (from the
CritiquingDirection slice) is untouched, the CritiqueReport data-contract
schema, and the SpecStore whitelist extension for critique_report.
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
CRITIC_AGENT_ROOT = AUTORESEARCH_ROOT / "agents" / "critic_agent"
CRITIQUE_REPORT_SCHEMA_PATH = (
    CRITIC_AGENT_ROOT / "data-contracts" / "critique_report.schema.yaml"
)

M6_EVENTS = [
    "direction.critiqued",
    "hypothesis.critique_requested",
    "hypothesis.critiqued",
]


@pytest.fixture
def registry() -> Registry:
    return Registry(REGISTRY_PATH)


def test_registry_loads_m6_validators(registry: Registry):
    for event_type in M6_EVENTS:
        assert registry.get_payload_validator(event_type, 1)


@pytest.mark.parametrize("event_type", M6_EVENTS)
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


def test_direction_critiqued_registry_entry(registry: Registry):
    entry = registry.get("direction.critiqued")
    assert entry.direction == "outbound"
    assert entry.emitters == {"CriticAgent"}
    assert 1 in entry.schema_versions


def test_hypothesis_critique_requested_registry_entry(registry: Registry):
    entry = registry.get("hypothesis.critique_requested")
    assert entry.direction == "outbound"
    assert entry.emitters == {"ResearchOrchestrator"}
    assert 1 in entry.schema_versions


def test_hypothesis_critiqued_registry_entry(registry: Registry):
    entry = registry.get("hypothesis.critiqued")
    assert entry.direction == "outbound"
    assert entry.emitters == {"CriticAgent"}
    assert 1 in entry.schema_versions


def test_direction_critiqued_invalid_verdict_enum_fails(registry: Registry):
    """The invalid example specifically exercises a bad `verdict` enum value."""
    validator = registry.get_payload_validator("direction.critiqued", 1)
    invalid_payload = json.loads(
        (EXAMPLES_DIR / "direction.critiqued.invalid.json").read_text()
    )
    assert invalid_payload.get("verdict") not in (
        "pass",
        "fail",
    ), "fixture must exercise an invalid enum value for verdict"
    errors = list(validator.iter_errors(invalid_payload))
    assert errors, "invalid verdict enum value must fail validation"


def test_hypothesis_critiqued_invalid_verdict_enum_fails(registry: Registry):
    """The invalid example specifically exercises a bad `verdict` enum value."""
    validator = registry.get_payload_validator("hypothesis.critiqued", 1)
    invalid_payload = json.loads(
        (EXAMPLES_DIR / "hypothesis.critiqued.invalid.json").read_text()
    )
    assert invalid_payload.get("verdict") not in (
        "some_pass",
        "all_rejected",
    ), "fixture must exercise an invalid enum value for verdict"
    errors = list(validator.iter_errors(invalid_payload))
    assert errors, "invalid verdict enum value must fail validation"


def test_direction_critique_requested_untouched(registry: Registry):
    """The CritiquingDirection slice's direction.critique_requested must still load, unmodified."""
    entry = registry.get("direction.critique_requested")
    assert entry.direction == "outbound"
    assert entry.emitters == {"ResearchOrchestrator"}
    assert 1 in entry.schema_versions
    assert registry.get_payload_validator("direction.critique_requested", 1)


def test_critique_report_fixture_validates_against_schema():
    schema = yaml.safe_load(CRITIQUE_REPORT_SCHEMA_PATH.read_text())
    validator = Draft202012Validator(schema)

    direction_fixture = {
        "target": "direction",
        "verdict": "pass",
        "issues": [],
        "passing_hypothesis_ids": [],
    }
    errors = list(validator.iter_errors(direction_fixture))
    assert errors == [], f"CritiqueReport fixture should validate, got: {[e.message for e in errors]}"

    # Minimal: only required fields.
    minimal = {
        "target": "hypotheses",
        "verdict": "some_pass",
        "issues": ["hypothesis H2 lacks a falsifiable prediction"],
    }
    errors = list(validator.iter_errors(minimal))
    assert errors == [], f"minimal CritiqueReport should validate, got: {[e.message for e in errors]}"


def test_critique_report_invalid_fixture_fails_schema():
    schema = yaml.safe_load(CRITIQUE_REPORT_SCHEMA_PATH.read_text())
    validator = Draft202012Validator(schema)

    # Bad target enum value.
    bad_target = {
        "target": "not_a_real_value",
        "verdict": "pass",
        "issues": [],
    }
    errors = list(validator.iter_errors(bad_target))
    assert errors, "CritiqueReport with a bad target enum should NOT validate"

    # Bad verdict enum value.
    bad_verdict = {
        "target": "direction",
        "verdict": "not_a_real_value",
        "issues": [],
    }
    errors = list(validator.iter_errors(bad_verdict))
    assert errors, "CritiqueReport with a bad verdict enum should NOT validate"

    # Missing required fields.
    missing_required = {
        "passing_hypothesis_ids": [],
    }
    errors = list(validator.iter_errors(missing_required))
    assert errors, "CritiqueReport missing required fields should NOT validate"

    # Extra unknown key.
    extra_key = {
        "target": "direction",
        "verdict": "pass",
        "issues": [],
        "unexpected_field": "nope",
    }
    errors = list(validator.iter_errors(extra_key))
    assert errors, "CritiqueReport with an extra property should NOT validate"


def test_store_accepts_critique_report_for_critic_agent(tmp_path):
    store = SpecStore(tmp_path / "store")
    try:
        store.put(
            kind="critique_report",
            session_id="s1",
            key="final",
            obj={
                "target": "direction",
                "verdict": "pass",
                "issues": [],
                "passing_hypothesis_ids": [],
            },
            emitter="CriticAgent",
        )
    finally:
        store.close()


def test_store_rejects_wrong_emitter_for_critique_report(tmp_path):
    store = SpecStore(tmp_path / "store")
    try:
        with pytest.raises(EmitterNotAllowed):
            store.put(
                kind="critique_report",
                session_id="s1",
                key="final",
                obj={},
                emitter="HypothesisAgent",
            )
    finally:
        store.close()
