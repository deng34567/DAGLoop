"""Unit tests for the HypothesisAgent contract slice (M5 Task 1).

Covers: hypothesis.generation_requested (v1), hypothesis.generated (v1), the
Hypothesis/HypothesisBank data-contract schema, and the SpecStore whitelist
extension for hypothesis_bank.
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
HYPOTHESIS_AGENT_ROOT = AUTORESEARCH_ROOT / "agents" / "hypothesis_agent"
HYPOTHESIS_BANK_SCHEMA_PATH = (
    HYPOTHESIS_AGENT_ROOT / "data-contracts" / "hypothesis_bank.schema.yaml"
)

M5_EVENTS = [
    "hypothesis.generation_requested",
    "hypothesis.generated",
]


@pytest.fixture
def registry() -> Registry:
    return Registry(REGISTRY_PATH)


def test_registry_loads_m5_validators(registry: Registry):
    for event_type in M5_EVENTS:
        assert registry.get_payload_validator(event_type, 1)


@pytest.mark.parametrize("event_type", M5_EVENTS)
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


def test_hypothesis_generation_requested_registry_entry(registry: Registry):
    entry = registry.get("hypothesis.generation_requested")
    assert entry.direction == "outbound"
    assert entry.emitters == {"ResearchOrchestrator"}
    assert 1 in entry.schema_versions


def test_hypothesis_generated_registry_entry(registry: Registry):
    entry = registry.get("hypothesis.generated")
    assert entry.direction == "outbound"
    assert entry.emitters == {"HypothesisAgent"}
    assert 1 in entry.schema_versions


def test_hypothesis_generated_invalid_count_fails(registry: Registry):
    """The invalid example specifically exercises a non-integer `count`."""
    validator = registry.get_payload_validator("hypothesis.generated", 1)
    invalid_payload = json.loads(
        (EXAMPLES_DIR / "hypothesis.generated.invalid.json").read_text()
    )
    assert not isinstance(
        invalid_payload.get("count"), int
    ), "fixture must exercise a non-integer count value"
    errors = list(validator.iter_errors(invalid_payload))
    assert errors, "invalid count value must fail validation"


def test_hypothesis_bank_fixture_validates_against_schema():
    schema = yaml.safe_load(HYPOTHESIS_BANK_SCHEMA_PATH.read_text())
    validator = Draft202012Validator(schema)

    fixture = {
        "hypotheses": [
            {
                "id": "h1",
                "statement": "Fusing the two kernels reduces memory traffic.",
                "rationale": "Avoids a round-trip to HBM between the two ops.",
                "kind": "kernel",
                "verifiable": True,
                "expected_effect": "10-15% latency reduction on the target op.",
            },
            {
                "id": "h2",
                "statement": "A cosine LR schedule improves convergence.",
                "kind": "training",
                "verifiable": True,
            },
        ]
    }
    errors = list(validator.iter_errors(fixture))
    assert errors == [], f"HypothesisBank fixture should validate, got: {[e.message for e in errors]}"

    # Minimal: empty hypotheses list is allowed at schema level.
    minimal = {"hypotheses": []}
    errors = list(validator.iter_errors(minimal))
    assert errors == [], f"minimal HypothesisBank should validate, got: {[e.message for e in errors]}"


def test_hypothesis_bank_invalid_kind_enum_fails_schema():
    schema = yaml.safe_load(HYPOTHESIS_BANK_SCHEMA_PATH.read_text())
    validator = Draft202012Validator(schema)

    bad_kind = {
        "hypotheses": [
            {
                "id": "h1",
                "statement": "Something.",
                "kind": "not_a_real_kind",
                "verifiable": True,
            }
        ]
    }
    errors = list(validator.iter_errors(bad_kind))
    assert errors, "HypothesisBank with a bad kind enum should NOT validate"


def test_hypothesis_missing_required_field_fails_schema():
    schema = yaml.safe_load(HYPOTHESIS_BANK_SCHEMA_PATH.read_text())
    validator = Draft202012Validator(schema)

    # Missing `statement`.
    missing_required = {
        "hypotheses": [
            {
                "id": "h1",
                "kind": "idea",
                "verifiable": True,
            }
        ]
    }
    errors = list(validator.iter_errors(missing_required))
    assert errors, "Hypothesis missing a required field should NOT validate"


def test_store_accepts_hypothesis_bank_for_hypothesis_agent(tmp_path):
    store = SpecStore(tmp_path / "store")
    try:
        store.put(
            kind="hypothesis_bank",
            session_id="s1",
            key="final",
            obj={
                "hypotheses": [
                    {
                        "id": "h1",
                        "statement": "x",
                        "kind": "idea",
                        "verifiable": True,
                    }
                ]
            },
            emitter="HypothesisAgent",
        )
    finally:
        store.close()


def test_store_rejects_wrong_emitter_for_hypothesis_bank(tmp_path):
    store = SpecStore(tmp_path / "store")
    try:
        with pytest.raises(EmitterNotAllowed):
            store.put(
                kind="hypothesis_bank",
                session_id="s1",
                key="final",
                obj={},
                emitter="DeepResearchAgent",
            )
    finally:
        store.close()
