"""Unit tests for the ReinterpretingSpec contract slice (Task 1).

Covers: spec.reinterpreted, novelty.audit_requested (v1), confirms
problem_modeling.reinterpret_requested (from M2) is untouched, the
SpecInterpretation data-contract schema, and the SpecStore whitelist
extension for spec_interpretation.
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
SPEC_INTERPRETATION_SCHEMA_PATH = (
    PROBLEM_MODELING_AGENT_ROOT / "data-contracts" / "spec_interpretation.schema.yaml"
)

REINTERPRET_EVENTS = [
    "spec.reinterpreted",
    "novelty.audit_requested",
]


@pytest.fixture
def registry() -> Registry:
    return Registry(REGISTRY_PATH)


def test_registry_loads_reinterpret_validators(registry: Registry):
    for event_type in REINTERPRET_EVENTS:
        assert registry.get_payload_validator(event_type, 1)


@pytest.mark.parametrize("event_type", REINTERPRET_EVENTS)
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


def test_spec_reinterpreted_registry_entry(registry: Registry):
    entry = registry.get("spec.reinterpreted")
    assert entry.direction == "outbound"
    assert entry.emitters == {"ProblemModelingAgent"}
    assert 1 in entry.schema_versions


def test_novelty_audit_requested_registry_entry(registry: Registry):
    entry = registry.get("novelty.audit_requested")
    assert entry.direction == "outbound"
    assert entry.emitters == {"ResearchOrchestrator"}
    assert 1 in entry.schema_versions


def test_spec_reinterpreted_invalid_disposition_enum_fails(registry: Registry):
    """The invalid example specifically exercises a bad `disposition` enum value."""
    validator = registry.get_payload_validator("spec.reinterpreted", 1)
    invalid_payload = json.loads(
        (EXAMPLES_DIR / "spec.reinterpreted.invalid.json").read_text()
    )
    assert invalid_payload.get("disposition") not in (
        "in_scope",
        "scope_changed",
        "out_of_scope",
    ), "fixture must exercise an invalid enum value for disposition"
    errors = list(validator.iter_errors(invalid_payload))
    assert errors, "invalid disposition enum value must fail validation"


def test_reinterpret_requested_untouched(registry: Registry):
    """M2's problem_modeling.reinterpret_requested must still load, unmodified."""
    entry = registry.get("problem_modeling.reinterpret_requested")
    assert entry.direction == "outbound"
    assert entry.emitters == {"ResearchOrchestrator"}
    assert 1 in entry.schema_versions
    assert registry.get_payload_validator("problem_modeling.reinterpret_requested", 1)


def test_spec_interpretation_fixture_validates_against_schema():
    schema = yaml.safe_load(SPEC_INTERPRETATION_SCHEMA_PATH.read_text())
    validator = Draft202012Validator(schema)

    semantics = {
        "comparison_statement": (
            "Under the selected benchmark protocol, compare score while allowing method changes."
        ),
        "research_question": "Does the method improve the primary metric?",
        "primary_metric": {
            "name": "score", "direction": "maximize",
            "measurement": "Read score from the canonical benchmark evaluator.",
        },
        "comparison_mode": "repository_defined",
        "comparison_basis": "Use the selected benchmark.",
        "budget_interpretation": "Use its documented budget.",
        "conditions_to_hold_constant": ["dataset and evaluator"],
        "method_freedom": ["method implementation"],
        "data_source": "The selected benchmark's canonical data.",
        "evaluator_source": "The selected benchmark's canonical evaluator.",
        "observed_outcomes": [],
        "outcome_criteria": {
            "success": "The valid method score improves on baseline.",
            "failure": "Execution fails or the valid score does not improve.",
            "inconclusive": "No valid comparable score is produced.",
        },
        "evidence": [{
            "source_type": "research_spec", "source_id": "research_spec",
            "supports": "The research goal defines the comparison question.",
        }],
        "uncertainties": [],
    }
    fixture = {
        "naive_assumptions": ["More data always helps."],
        "real_open_problem": "How to generalize under distribution shift with limited labels.",
        "topic_position": "Sits at the intersection of domain adaptation and active learning.",
        "disposition": "scope_changed",
        "scope_reason": "Literature review revealed the naive framing was already solved; scope narrowed to the harder sub-problem.",
        "evaluation_semantics": semantics,
    }
    errors = list(validator.iter_errors(fixture))
    assert errors == [], f"SpecInterpretation fixture should validate, got: {[e.message for e in errors]}"

    # Minimal: only required fields, arrays empty.
    minimal = {
        "real_open_problem": "TBD",
        "disposition": "in_scope",
        "scope_reason": "No change needed.",
        "evaluation_semantics": semantics,
    }
    errors = list(validator.iter_errors(minimal))
    assert errors == [], f"minimal SpecInterpretation should validate, got: {[e.message for e in errors]}"


def test_spec_interpretation_invalid_fixture_fails_schema():
    schema = yaml.safe_load(SPEC_INTERPRETATION_SCHEMA_PATH.read_text())
    validator = Draft202012Validator(schema)

    # Bad disposition enum value.
    bad_disposition = {
        "real_open_problem": "TBD",
        "disposition": "not_a_real_value",
        "scope_reason": "TBD",
    }
    errors = list(validator.iter_errors(bad_disposition))
    assert errors, "SpecInterpretation with a bad disposition enum should NOT validate"

    # Missing required fields.
    missing_required = {
        "naive_assumptions": [],
        "topic_position": "TBD",
    }
    errors = list(validator.iter_errors(missing_required))
    assert errors, "SpecInterpretation missing required fields should NOT validate"

    # Extra unknown key.
    extra_key = {
        "real_open_problem": "TBD",
        "disposition": "in_scope",
        "scope_reason": "TBD",
        "unexpected_field": "nope",
    }
    errors = list(validator.iter_errors(extra_key))
    assert errors, "SpecInterpretation with an extra property should NOT validate"


def test_store_accepts_spec_interpretation_for_problem_modeling_agent(tmp_path):
    store = SpecStore(tmp_path / "store")
    try:
        store.put(
            kind="spec_interpretation",
            session_id="s1",
            key="final",
            obj={
                "naive_assumptions": [],
                "real_open_problem": "x",
                "topic_position": "x",
                "disposition": "in_scope",
                "scope_reason": "x",
            },
            emitter="ProblemModelingAgent",
        )
    finally:
        store.close()


def test_store_rejects_wrong_emitter_for_spec_interpretation(tmp_path):
    store = SpecStore(tmp_path / "store")
    try:
        with pytest.raises(EmitterNotAllowed):
            store.put(
                kind="spec_interpretation",
                session_id="s1",
                key="final",
                obj={},
                emitter="DeepResearchAgent",
            )
    finally:
        store.close()
