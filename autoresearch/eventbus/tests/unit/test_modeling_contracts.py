"""Unit tests for the ModelingProblem contract slice (Task 1).

Covers: problem.modeled, human.idea_provided, direction.critique_requested
(v1), confirms problem_modeling.model_requested (from M4) is untouched, the
ProblemModel data-contract schema, and the SpecStore whitelist extension for
problem_model.
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
PROBLEM_MODELING_AGENT_ROOT = AUTORESEARCH_ROOT / "agents" / "problem_modeling_agent"
PROBLEM_MODEL_SCHEMA_PATH = (
    PROBLEM_MODELING_AGENT_ROOT / "data-contracts" / "problem_model.schema.yaml"
)

MODELING_EVENTS = [
    "problem.modeled",
    "human.idea_provided",
    "direction.critique_requested",
]


@pytest.fixture
def registry() -> Registry:
    return Registry(REGISTRY_PATH)


def test_registry_loads_modeling_validators(registry: Registry):
    for event_type in MODELING_EVENTS:
        assert registry.get_payload_validator(event_type, 1)


@pytest.mark.parametrize("event_type", MODELING_EVENTS)
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


def test_problem_modeled_registry_entry(registry: Registry):
    entry = registry.get("problem.modeled")
    assert entry.direction == "outbound"
    assert entry.emitters == {"ProblemModelingAgent"}
    assert 1 in entry.schema_versions


def test_human_idea_provided_registry_entry(registry: Registry):
    entry = registry.get("human.idea_provided")
    assert entry.direction == "inbound"
    assert entry.emitters == {"Human"}
    assert 1 in entry.schema_versions


def test_direction_critique_requested_registry_entry(registry: Registry):
    entry = registry.get("direction.critique_requested")
    assert entry.direction == "outbound"
    assert entry.emitters == {"ResearchOrchestrator"}
    assert 1 in entry.schema_versions


def test_human_idea_provided_invalid_inject_not_bool_fails(registry: Registry):
    """The invalid example specifically exercises a non-bool `inject`."""
    validator = registry.get_payload_validator("human.idea_provided", 1)
    invalid_payload = json.loads(
        (EXAMPLES_DIR / "human.idea_provided.invalid.json").read_text()
    )
    assert not isinstance(invalid_payload.get("inject"), bool), (
        "fixture must exercise a non-bool `inject` value"
    )
    errors = list(validator.iter_errors(invalid_payload))
    assert errors, "invalid `inject` (non-bool) must fail validation"


def test_model_requested_untouched(registry: Registry):
    """M4's problem_modeling.model_requested must still load, unmodified."""
    entry = registry.get("problem_modeling.model_requested")
    assert entry.direction == "outbound"
    assert entry.emitters == {"ResearchOrchestrator"}
    assert 1 in entry.schema_versions
    assert registry.get_payload_validator("problem_modeling.model_requested", 1)


def test_problem_model_fixture_validates_against_schema():
    schema = yaml.safe_load(PROBLEM_MODEL_SCHEMA_PATH.read_text())
    validator = Draft202012Validator(schema)

    fixture = {
        "key_problems": ["No principled way to select retrieval strategy per query type."],
        "opportunities": ["Learned router could beat fixed heuristics."],
        "risks": ["Overfitting the router to the benchmark distribution."],
        "verifiable_paths": ["Ablate router vs. fixed strategy on held-out query types."],
    }
    errors = list(validator.iter_errors(fixture))
    assert errors == [], f"ProblemModel fixture should validate, got: {[e.message for e in errors]}"

    # Minimal: only required fields, others empty arrays.
    minimal = {
        "key_problems": ["TBD"],
        "opportunities": [],
        "risks": [],
        "verifiable_paths": ["TBD"],
    }
    errors = list(validator.iter_errors(minimal))
    assert errors == [], f"minimal ProblemModel should validate, got: {[e.message for e in errors]}"


def test_problem_model_missing_required_fixture_fails_schema():
    schema = yaml.safe_load(PROBLEM_MODEL_SCHEMA_PATH.read_text())
    validator = Draft202012Validator(schema)

    # Missing required verifiable_paths.
    missing_required = {
        "key_problems": ["x"],
        "opportunities": [],
        "risks": [],
    }
    errors = list(validator.iter_errors(missing_required))
    assert errors, "ProblemModel missing required fields should NOT validate"

    # Extra unknown key.
    extra_key = {
        "key_problems": ["x"],
        "opportunities": [],
        "risks": [],
        "verifiable_paths": ["x"],
        "unexpected_field": "nope",
    }
    errors = list(validator.iter_errors(extra_key))
    assert errors, "ProblemModel with an extra property should NOT validate"


def test_store_accepts_problem_model_for_problem_modeling_agent(tmp_path):
    store = SpecStore(tmp_path / "store")
    try:
        store.put(
            kind="problem_model",
            session_id="s1",
            key="v1",
            obj={
                "key_problems": ["x"],
                "opportunities": [],
                "risks": [],
                "verifiable_paths": ["x"],
            },
            emitter="ProblemModelingAgent",
        )
    finally:
        store.close()


def test_store_rejects_wrong_emitter_for_problem_model(tmp_path):
    store = SpecStore(tmp_path / "store")
    try:
        with pytest.raises(EmitterNotAllowed):
            store.put(
                kind="problem_model",
                session_id="s1",
                key="v1",
                obj={},
                emitter="DeepResearchAgent",
            )
    finally:
        store.close()
