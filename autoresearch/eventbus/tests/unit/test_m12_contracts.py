"""Unit tests for the EvaluatorAgent EvaluatingExperiment + VerifyingBestResult
contract slice (M12 Task 1).

Covers: experiment.evaluation_requested, experiment.evaluated,
result.verification_requested, result.verified (v1); the EvaluationReport and
VerificationReport data-contract schemas; and the SpecStore whitelist
extension for evaluation_report / verification_report.
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
EVALUATOR_AGENT_ROOT = AUTORESEARCH_ROOT / "agents" / "evaluator_agent"
EVALUATION_REPORT_SCHEMA_PATH = (
    EVALUATOR_AGENT_ROOT / "data-contracts" / "evaluation_report.schema.yaml"
)
VERIFICATION_REPORT_SCHEMA_PATH = (
    EVALUATOR_AGENT_ROOT / "data-contracts" / "verification_report.schema.yaml"
)

M12_EVENTS = [
    "experiment.evaluation_requested",
    "experiment.evaluated",
    "result.verification_requested",
    "result.verified",
]


@pytest.fixture
def registry() -> Registry:
    return Registry(REGISTRY_PATH)


def test_registry_loads_m12_validators(registry: Registry):
    for event_type in M12_EVENTS:
        assert registry.get_payload_validator(event_type, 1)


@pytest.mark.parametrize("event_type", M12_EVENTS)
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


def test_experiment_evaluation_requested_registry_entry(registry: Registry):
    entry = registry.get("experiment.evaluation_requested")
    assert entry.direction == "outbound"
    assert entry.emitters == {"ResearchOrchestrator"}
    assert 1 in entry.schema_versions


def test_experiment_evaluated_registry_entry(registry: Registry):
    entry = registry.get("experiment.evaluated")
    assert entry.direction == "outbound"
    assert entry.emitters == {"EvaluatorAgent"}
    assert 1 in entry.schema_versions


def test_result_verification_requested_registry_entry(registry: Registry):
    entry = registry.get("result.verification_requested")
    assert entry.direction == "outbound"
    assert entry.emitters == {"ResearchOrchestrator"}
    assert 1 in entry.schema_versions


def test_result_verified_registry_entry(registry: Registry):
    entry = registry.get("result.verified")
    assert entry.direction == "outbound"
    assert entry.emitters == {"EvaluatorAgent"}
    assert 1 in entry.schema_versions


def test_experiment_evaluated_invalid_verdict_enum_fails(registry: Registry):
    """The invalid example specifically exercises a bad `verdict` enum value."""
    validator = registry.get_payload_validator("experiment.evaluated", 1)
    invalid_payload = json.loads(
        (EXAMPLES_DIR / "experiment.evaluated.invalid.json").read_text()
    )
    assert invalid_payload.get("verdict") not in (
        "invalid",
        "valid_not_best",
        "new_best",
        "suspicious",
    ), "fixture must exercise an invalid enum value for verdict"
    errors = list(validator.iter_errors(invalid_payload))
    assert errors, "invalid verdict enum value must fail validation"


def test_experiment_evaluated_accepts_only_sha256_run_identity(registry: Registry):
    validator = registry.get_payload_validator("experiment.evaluated", 1)
    payload = {
        "verdict": "valid_not_best",
        "evaluation_report_ref": "evaluation_report://sess-123/v1",
        "run_record_digest": "sha256:" + "a" * 64,
    }
    assert list(validator.iter_errors(payload)) == []
    payload["run_record_digest"] = "same-mutable-ref"
    assert list(validator.iter_errors(payload))


def test_result_verified_invalid_verdict_enum_fails(registry: Registry):
    """The invalid example specifically exercises a bad `verdict` enum value."""
    validator = registry.get_payload_validator("result.verified", 1)
    invalid_payload = json.loads(
        (EXAMPLES_DIR / "result.verified.invalid.json").read_text()
    )
    assert invalid_payload.get("verdict") not in (
        "verified",
        "failed",
    ), "fixture must exercise an invalid enum value for verdict"
    errors = list(validator.iter_errors(invalid_payload))
    assert errors, "invalid verdict enum value must fail validation"


def test_evaluation_report_fixture_validates_against_schema():
    schema = yaml.safe_load(EVALUATION_REPORT_SCHEMA_PATH.read_text())
    validator = Draft202012Validator(schema)

    full_fixture = {
        "verdict": "new_best",
        "metric_value": 12.4,
        "baseline_metric": 10.1,
        "is_improvement": True,
        "suspicious": False,
        "anti_cheat_flags": [],
        "reasoning": "Metric improved beyond noise threshold.",
    }
    errors = list(validator.iter_errors(full_fixture))
    assert errors == [], f"EvaluationReport fixture should validate, got: {[e.message for e in errors]}"

    # Minimal: only required fields.
    minimal = {
        "verdict": "invalid",
        "is_improvement": False,
        "suspicious": False,
        "anti_cheat_flags": [],
    }
    errors = list(validator.iter_errors(minimal))
    assert errors == [], f"minimal EvaluationReport should validate, got: {[e.message for e in errors]}"


def test_evaluation_report_nullable_optionals_validate():
    """metric_value and baseline_metric must accept null (additionalProperties:false
    means the core emits them explicitly as null rather than omitting them)."""
    schema = yaml.safe_load(EVALUATION_REPORT_SCHEMA_PATH.read_text())
    validator = Draft202012Validator(schema)

    with_nulls = {
        "verdict": "invalid",
        "metric_value": None,
        "baseline_metric": None,
        "is_improvement": False,
        "suspicious": False,
        "anti_cheat_flags": [],
        "reasoning": None,
    }
    errors = list(validator.iter_errors(with_nulls))
    assert errors == [], (
        f"EvaluationReport with null optionals should validate, got: "
        f"{[e.message for e in errors]}"
    )


def test_evaluation_report_missing_required_fields_fails():
    schema = yaml.safe_load(EVALUATION_REPORT_SCHEMA_PATH.read_text())
    validator = Draft202012Validator(schema)

    missing_verdict = {
        "is_improvement": False,
        "suspicious": False,
        "anti_cheat_flags": [],
    }
    errors = list(validator.iter_errors(missing_verdict))
    assert errors, "EvaluationReport missing `verdict` should NOT validate"

    missing_is_improvement = {
        "verdict": "invalid",
        "suspicious": False,
        "anti_cheat_flags": [],
    }
    errors = list(validator.iter_errors(missing_is_improvement))
    assert errors, "EvaluationReport missing `is_improvement` should NOT validate"

    missing_suspicious = {
        "verdict": "invalid",
        "is_improvement": False,
        "anti_cheat_flags": [],
    }
    errors = list(validator.iter_errors(missing_suspicious))
    assert errors, "EvaluationReport missing `suspicious` should NOT validate"

    missing_anti_cheat_flags = {
        "verdict": "invalid",
        "is_improvement": False,
        "suspicious": False,
    }
    errors = list(validator.iter_errors(missing_anti_cheat_flags))
    assert errors, "EvaluationReport missing `anti_cheat_flags` should NOT validate"


def test_evaluation_report_extra_field_fails():
    schema = yaml.safe_load(EVALUATION_REPORT_SCHEMA_PATH.read_text())
    validator = Draft202012Validator(schema)

    extra_key = {
        "verdict": "invalid",
        "is_improvement": False,
        "suspicious": False,
        "anti_cheat_flags": [],
        "unexpected_field": "nope",
    }
    errors = list(validator.iter_errors(extra_key))
    assert errors, "EvaluationReport with an extra property should NOT validate"


def test_evaluation_report_invalid_verdict_enum_fails():
    schema = yaml.safe_load(EVALUATION_REPORT_SCHEMA_PATH.read_text())
    validator = Draft202012Validator(schema)

    bad_verdict = {
        "verdict": "definitely_the_best",
        "is_improvement": False,
        "suspicious": False,
        "anti_cheat_flags": [],
    }
    errors = list(validator.iter_errors(bad_verdict))
    assert errors, "EvaluationReport with an invalid verdict enum should NOT validate"


def test_verification_report_fixture_validates_against_schema():
    schema = yaml.safe_load(VERIFICATION_REPORT_SCHEMA_PATH.read_text())
    validator = Draft202012Validator(schema)

    full_fixture = {
        "verdict": "verified",
        "checks": ["re-ran experiment", "compared metric to baseline"],
        "anti_cheat_flags": [],
        "reasoning": "All checks passed under the frozen protocol.",
    }
    errors = list(validator.iter_errors(full_fixture))
    assert errors == [], f"VerificationReport fixture should validate, got: {[e.message for e in errors]}"

    minimal = {
        "verdict": "failed",
        "checks": [],
        "anti_cheat_flags": ["metric mismatch on re-run"],
    }
    errors = list(validator.iter_errors(minimal))
    assert errors == [], f"minimal VerificationReport should validate, got: {[e.message for e in errors]}"


def test_verification_report_nullable_reasoning_validates():
    schema = yaml.safe_load(VERIFICATION_REPORT_SCHEMA_PATH.read_text())
    validator = Draft202012Validator(schema)

    with_null = {
        "verdict": "verified",
        "checks": [],
        "anti_cheat_flags": [],
        "reasoning": None,
    }
    errors = list(validator.iter_errors(with_null))
    assert errors == [], (
        f"VerificationReport with null reasoning should validate, got: "
        f"{[e.message for e in errors]}"
    )


def test_verification_report_missing_required_fields_fails():
    schema = yaml.safe_load(VERIFICATION_REPORT_SCHEMA_PATH.read_text())
    validator = Draft202012Validator(schema)

    missing_verdict = {
        "checks": [],
        "anti_cheat_flags": [],
    }
    errors = list(validator.iter_errors(missing_verdict))
    assert errors, "VerificationReport missing `verdict` should NOT validate"

    missing_checks = {
        "verdict": "verified",
        "anti_cheat_flags": [],
    }
    errors = list(validator.iter_errors(missing_checks))
    assert errors, "VerificationReport missing `checks` should NOT validate"

    missing_anti_cheat_flags = {
        "verdict": "verified",
        "checks": [],
    }
    errors = list(validator.iter_errors(missing_anti_cheat_flags))
    assert errors, "VerificationReport missing `anti_cheat_flags` should NOT validate"


def test_verification_report_extra_field_fails():
    schema = yaml.safe_load(VERIFICATION_REPORT_SCHEMA_PATH.read_text())
    validator = Draft202012Validator(schema)

    extra_key = {
        "verdict": "verified",
        "checks": [],
        "anti_cheat_flags": [],
        "unexpected_field": "nope",
    }
    errors = list(validator.iter_errors(extra_key))
    assert errors, "VerificationReport with an extra property should NOT validate"


def test_verification_report_invalid_verdict_enum_fails():
    schema = yaml.safe_load(VERIFICATION_REPORT_SCHEMA_PATH.read_text())
    validator = Draft202012Validator(schema)

    bad_verdict = {
        "verdict": "definitely_verified",
        "checks": [],
        "anti_cheat_flags": [],
    }
    errors = list(validator.iter_errors(bad_verdict))
    assert errors, "VerificationReport with an invalid verdict enum should NOT validate"


def test_store_accepts_evaluation_report_for_evaluator_agent(tmp_path):
    store = SpecStore(tmp_path / "store")
    try:
        store.put(
            kind="evaluation_report",
            session_id="s1",
            key="exp-1",
            obj={
                "verdict": "new_best",
                "is_improvement": True,
                "suspicious": False,
                "anti_cheat_flags": [],
            },
            emitter="EvaluatorAgent",
        )
    finally:
        store.close()


def test_store_rejects_wrong_emitter_for_evaluation_report(tmp_path):
    store = SpecStore(tmp_path / "store")
    try:
        with pytest.raises(EmitterNotAllowed):
            store.put(
                kind="evaluation_report",
                session_id="s1",
                key="exp-1",
                obj={},
                emitter="ResearchOrchestrator",
            )
    finally:
        store.close()


def test_store_accepts_verification_report_for_evaluator_agent(tmp_path):
    store = SpecStore(tmp_path / "store")
    try:
        store.put(
            kind="verification_report",
            session_id="s1",
            key="exp-1",
            obj={
                "verdict": "verified",
                "checks": [],
                "anti_cheat_flags": [],
            },
            emitter="EvaluatorAgent",
        )
    finally:
        store.close()


def test_store_rejects_wrong_emitter_for_verification_report(tmp_path):
    store = SpecStore(tmp_path / "store")
    try:
        with pytest.raises(EmitterNotAllowed):
            store.put(
                kind="verification_report",
                session_id="s1",
                key="exp-1",
                obj={},
                emitter="ResearchOrchestrator",
            )
    finally:
        store.close()
