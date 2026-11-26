"""Unit tests for the MemoryAgent ConsolidatingMemory contract slice (M13 Task 1).

Covers: memory.consolidation_requested, memory.consolidated (v1); the
ResearchMemory data-contract schema; the SpecStore whitelist extension for
research_memory; MemoryAgent as a legal agent.failed emitter; and the M12
schema extensions (conflict verdict + optional research_memory_ref) that
must remain backward-compatible with the existing M12 examples/fixtures.
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
MEMORY_AGENT_ROOT = AUTORESEARCH_ROOT / "agents" / "memory_agent"
RESEARCH_MEMORY_SCHEMA_PATH = (
    MEMORY_AGENT_ROOT / "data-contracts" / "research_memory.schema.yaml"
)

EVALUATOR_AGENT_ROOT = AUTORESEARCH_ROOT / "agents" / "evaluator_agent"
EVALUATION_REPORT_SCHEMA_PATH = (
    EVALUATOR_AGENT_ROOT / "data-contracts" / "evaluation_report.schema.yaml"
)

M13_EVENTS = [
    "memory.consolidation_requested",
    "memory.consolidated",
]


@pytest.fixture
def registry() -> Registry:
    return Registry(REGISTRY_PATH)


# ---------------------------------------------------------------------------
# New events: memory.consolidation_requested, memory.consolidated
# ---------------------------------------------------------------------------


def test_registry_loads_m13_validators(registry: Registry):
    for event_type in M13_EVENTS:
        assert registry.get_payload_validator(event_type, 1)


@pytest.mark.parametrize("event_type", M13_EVENTS)
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


def test_memory_consolidation_requested_registry_entry(registry: Registry):
    entry = registry.get("memory.consolidation_requested")
    assert entry.direction == "outbound"
    assert entry.emitters == {"ResearchOrchestrator"}
    assert 1 in entry.schema_versions


def test_memory_consolidated_registry_entry(registry: Registry):
    entry = registry.get("memory.consolidated")
    assert entry.direction == "outbound"
    assert entry.emitters == {"MemoryAgent"}
    assert 1 in entry.schema_versions


def test_memory_consolidated_invalid_routing_verdict_enum_fails(registry: Registry):
    """The invalid example specifically exercises a bad `routing_verdict` enum value."""
    validator = registry.get_payload_validator("memory.consolidated", 1)
    invalid_payload = json.loads(
        (EXAMPLES_DIR / "memory.consolidated.invalid.json").read_text()
    )
    assert invalid_payload.get("routing_verdict") not in (
        "continue_explore",
        "report",
        "need_experiment",
    ), "fixture must exercise an invalid enum value for routing_verdict"
    errors = list(validator.iter_errors(invalid_payload))
    assert errors, "invalid routing_verdict enum value must fail validation"


@pytest.mark.parametrize("diagnosis", [
    "execution_failure", "setting_issue", "component_expectation_failed",
    "method_hypothesis_falsified", "baseline_drift", "valid_improvement",
    "candidate_contract_violation", "attempt_accounting_violation", "inconclusive",
])
def test_memory_consolidated_accepts_evidence_diagnosis(registry: Registry, diagnosis):
    """Catch trusted loop diagnoses being rejected at the EventBus boundary."""
    validator = registry.get_payload_validator("memory.consolidated", 1)
    payload = {
        "research_memory_ref": "research_memory://sess/v1",
        "routing_verdict": "continue_explore",
        "diagnosis": diagnosis,
        "diagnosis_evidence": ["measured evidence"],
    }
    assert list(validator.iter_errors(payload)) == []


def test_memory_consolidated_rejects_unknown_diagnosis(registry: Registry):
    validator = registry.get_payload_validator("memory.consolidated", 1)
    payload = {
        "research_memory_ref": "research_memory://sess/v1",
        "routing_verdict": "continue_explore",
        "diagnosis": "guess_again",
    }
    assert list(validator.iter_errors(payload))


def test_memory_consolidation_requested_optional_refs_are_nullable(registry: Registry):
    validator = registry.get_payload_validator("memory.consolidation_requested", 1)
    payload = {
        "evaluation_report_ref": "evaluation_report://sess-123/exp-1",
        "run_record_ref": "run_record://sess-123/exp-1",
        "research_memory_ref": None,
        "verification_report_ref": None,
    }
    errors = list(validator.iter_errors(payload))
    assert errors == [], (
        f"memory.consolidation_requested with null optional refs should validate, "
        f"got: {[e.message for e in errors]}"
    )


def test_memory_consolidation_requested_accepts_theory_without_run(registry: Registry):
    validator = registry.get_payload_validator("memory.consolidation_requested", 1)
    payload = {
        "evaluation_report_ref": None,
        "run_record_ref": None,
        "research_memory_ref": None,
        "verification_report_ref": None,
        "baseline_result_ref": None,
        "failure_context": None,
        "theory_method_ref": "spec://method/sess-theory/v1",
        "problem_model_ref": "spec://problem_model/sess-theory/v1",
        "spec_interpretation_ref": "spec://spec_interpretation/sess-theory/v1",
    }
    assert list(validator.iter_errors(payload)) == []


# ---------------------------------------------------------------------------
# ResearchMemory data-contract schema
# ---------------------------------------------------------------------------


def test_research_memory_fixture_validates_with_empty_arrays():
    schema = yaml.safe_load(RESEARCH_MEMORY_SCHEMA_PATH.read_text())
    validator = Draft202012Validator(schema)

    minimal = {
        "verified_findings": [],
        "negative_results": [],
        "consolidated_insights": [],
        "divergence_seeds": [],
    }
    errors = list(validator.iter_errors(minimal))
    assert errors == [], (
        f"minimal ResearchMemory (empty required arrays) should validate, got: "
        f"{[e.message for e in errors]}"
    )


def test_research_memory_fixture_validates_with_null_best_metric():
    schema = yaml.safe_load(RESEARCH_MEMORY_SCHEMA_PATH.read_text())
    validator = Draft202012Validator(schema)

    full_fixture = {
        "best_metric": None,
        "best_result_ref": None,
        "verified_findings": ["finding A"],
        "negative_results": ["negative A"],
        "excluded_hypotheses": ["hyp-2"],
        "consolidated_insights": ["insight A"],
        "divergence_seeds": ["seed A"],
        "next_steps": ["run experiment B"],
    }
    errors = list(validator.iter_errors(full_fixture))
    assert errors == [], (
        f"ResearchMemory with null best_metric/best_result_ref should validate, got: "
        f"{[e.message for e in errors]}"
    )


def test_research_memory_missing_required_array_fails():
    schema = yaml.safe_load(RESEARCH_MEMORY_SCHEMA_PATH.read_text())
    validator = Draft202012Validator(schema)

    missing_negative_results = {
        "verified_findings": [],
        "consolidated_insights": [],
        "divergence_seeds": [],
    }
    errors = list(validator.iter_errors(missing_negative_results))
    assert errors, "ResearchMemory missing `negative_results` should NOT validate"


def test_research_memory_extra_field_fails():
    schema = yaml.safe_load(RESEARCH_MEMORY_SCHEMA_PATH.read_text())
    validator = Draft202012Validator(schema)

    extra_key = {
        "verified_findings": [],
        "negative_results": [],
        "consolidated_insights": [],
        "divergence_seeds": [],
        "unexpected_field": "nope",
    }
    errors = list(validator.iter_errors(extra_key))
    assert errors, "ResearchMemory with an extra property should NOT validate"


# ---------------------------------------------------------------------------
# SpecStore: research_memory kind + MemoryAgent as agent.failed emitter
# ---------------------------------------------------------------------------


def test_store_accepts_research_memory_for_memory_agent(tmp_path):
    store = SpecStore(tmp_path / "store")
    try:
        store.put(
            kind="research_memory",
            session_id="s1",
            key="consolidation-1",
            obj={
                "verified_findings": [],
                "negative_results": [],
                "consolidated_insights": [],
                "divergence_seeds": [],
            },
            emitter="MemoryAgent",
        )
    finally:
        store.close()


def test_store_rejects_wrong_emitter_for_research_memory(tmp_path):
    store = SpecStore(tmp_path / "store")
    try:
        with pytest.raises(EmitterNotAllowed):
            store.put(
                kind="research_memory",
                session_id="s1",
                key="consolidation-1",
                obj={},
                emitter="ResearchOrchestrator",
            )
    finally:
        store.close()


def test_agent_failed_now_includes_memory_agent(registry: Registry):
    entry = registry.get("agent.failed")
    assert "MemoryAgent" in entry.emitters


def test_agent_failed_still_includes_evaluator_agent(registry: Registry):
    """MemoryAgent's addition must not remove existing emitters."""
    entry = registry.get("agent.failed")
    assert "EvaluatorAgent" in entry.emitters


# ---------------------------------------------------------------------------
# M12 extension: `conflict` added to experiment.evaluated / EvaluationReport
# verdict enums (backward-compatible additive change).
# ---------------------------------------------------------------------------


def test_experiment_evaluated_conflict_verdict_validates(registry: Registry):
    validator = registry.get_payload_validator("experiment.evaluated", 1)
    payload = {
        "verdict": "conflict",
        "evaluation_report_ref": "evaluation_report://sess-123/exp-1",
    }
    errors = list(validator.iter_errors(payload))
    assert errors == [], (
        f"experiment.evaluated with verdict=conflict should validate, got: "
        f"{[e.message for e in errors]}"
    )


def test_experiment_evaluated_happy_example_still_validates(registry: Registry):
    """Existing M12 example must still validate after the enum extension."""
    validator = registry.get_payload_validator("experiment.evaluated", 1)
    happy_payload = json.loads(
        (EXAMPLES_DIR / "experiment.evaluated.happy.json").read_text()
    )
    errors = list(validator.iter_errors(happy_payload))
    assert errors == [], (
        f"existing experiment.evaluated.happy.json must still validate, got: "
        f"{[e.message for e in errors]}"
    )


def test_evaluation_report_conflict_verdict_validates():
    schema = yaml.safe_load(EVALUATION_REPORT_SCHEMA_PATH.read_text())
    validator = Draft202012Validator(schema)

    fixture = {
        "verdict": "conflict",
        "is_improvement": False,
        "suspicious": False,
        "anti_cheat_flags": [],
    }
    errors = list(validator.iter_errors(fixture))
    assert errors == [], (
        f"EvaluationReport with verdict=conflict should validate, got: "
        f"{[e.message for e in errors]}"
    )


# ---------------------------------------------------------------------------
# M12 extension: optional nullable `research_memory_ref` added to
# experiment.evaluation_requested (backward-compatible additive change).
# ---------------------------------------------------------------------------


def test_experiment_evaluation_requested_with_research_memory_ref_validates(
    registry: Registry,
):
    validator = registry.get_payload_validator("experiment.evaluation_requested", 1)
    payload = {
        "run_record_ref": "run_record://sess-123/exp-1",
        "verifier_spec_ref": "verifier_spec://sess-123/final",
        "benchmark_spec_ref": "benchmark_spec://sess-123/final",
        "research_memory_ref": "spec://x",
    }
    errors = list(validator.iter_errors(payload))
    assert errors == [], (
        f"experiment.evaluation_requested with research_memory_ref should validate, "
        f"got: {[e.message for e in errors]}"
    )


def test_experiment_evaluation_requested_without_research_memory_ref_validates(
    registry: Registry,
):
    validator = registry.get_payload_validator("experiment.evaluation_requested", 1)
    payload = {
        "run_record_ref": "run_record://sess-123/exp-1",
        "verifier_spec_ref": "verifier_spec://sess-123/final",
        "benchmark_spec_ref": "benchmark_spec://sess-123/final",
    }
    errors = list(validator.iter_errors(payload))
    assert errors == [], (
        f"experiment.evaluation_requested without research_memory_ref should still "
        f"validate (optional), got: {[e.message for e in errors]}"
    )


def test_experiment_evaluation_requested_with_null_research_memory_ref_validates(
    registry: Registry,
):
    validator = registry.get_payload_validator("experiment.evaluation_requested", 1)
    payload = {
        "run_record_ref": "run_record://sess-123/exp-1",
        "verifier_spec_ref": "verifier_spec://sess-123/final",
        "benchmark_spec_ref": "benchmark_spec://sess-123/final",
        "research_memory_ref": None,
    }
    errors = list(validator.iter_errors(payload))
    assert errors == [], (
        f"experiment.evaluation_requested with research_memory_ref=null should "
        f"validate (nullable), got: {[e.message for e in errors]}"
    )


def test_experiment_evaluation_requested_with_baseline_result_ref_validates(
    registry: Registry,
):
    validator = registry.get_payload_validator("experiment.evaluation_requested", 1)
    payload = {
        "run_record_ref": "run_record://sess-123/exp-1",
        "verifier_spec_ref": "verifier_spec://sess-123/final",
        "benchmark_spec_ref": "benchmark_spec://sess-123/final",
        "baseline_result_ref": "baseline_result://sess-123/final",
    }
    errors = list(validator.iter_errors(payload))
    assert errors == [], [e.message for e in errors]


def test_experiment_evaluation_requested_happy_example_still_validates(
    registry: Registry,
):
    """Existing M12 example must still validate after the additive extension."""
    validator = registry.get_payload_validator("experiment.evaluation_requested", 1)
    happy_payload = json.loads(
        (EXAMPLES_DIR / "experiment.evaluation_requested.happy.json").read_text()
    )
    errors = list(validator.iter_errors(happy_payload))
    assert errors == [], (
        f"existing experiment.evaluation_requested.happy.json must still validate, "
        f"got: {[e.message for e in errors]}"
    )
