"""Unit tests for the RunnerAgent RunningExperiment contract slice (M11 Task 1).

Covers: experiment.run_requested, experiment.run_completed (v1), the
RunRecord data-contract schema, and the SpecStore whitelist extension
for run_record.
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
RUNNER_AGENT_ROOT = AUTORESEARCH_ROOT / "agents" / "runner_agent"
RUN_RECORD_SCHEMA_PATH = (
    RUNNER_AGENT_ROOT / "data-contracts" / "run_record.schema.yaml"
)

M11_EVENTS = [
    "experiment.run_requested",
    "experiment.run_completed",
]


@pytest.fixture
def registry() -> Registry:
    return Registry(REGISTRY_PATH)


def test_registry_loads_m11_validators(registry: Registry):
    for event_type in M11_EVENTS:
        assert registry.get_payload_validator(event_type, 1)


@pytest.mark.parametrize("event_type", M11_EVENTS)
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


def test_experiment_run_requested_registry_entry(registry: Registry):
    entry = registry.get("experiment.run_requested")
    assert entry.direction == "outbound"
    assert entry.emitters == {"ResearchOrchestrator"}
    assert 1 in entry.schema_versions


def test_experiment_run_completed_registry_entry(registry: Registry):
    entry = registry.get("experiment.run_completed")
    assert entry.direction == "outbound"
    assert entry.emitters == {"RunnerAgent"}
    assert 1 in entry.schema_versions


def test_experiment_run_completed_invalid_verdict_enum_fails(registry: Registry):
    """The invalid example specifically exercises a bad `verdict` enum value."""
    validator = registry.get_payload_validator("experiment.run_completed", 1)
    invalid_payload = json.loads(
        (EXAMPLES_DIR / "experiment.run_completed.invalid.json").read_text()
    )
    assert invalid_payload.get("verdict") not in (
        "succeeded",
        "failed",
    ), "fixture must exercise an invalid enum value for verdict"
    errors = list(validator.iter_errors(invalid_payload))
    assert errors, "invalid verdict enum value must fail validation"


def test_run_record_fixture_validates_against_schema():
    schema = yaml.safe_load(RUN_RECORD_SCHEMA_PATH.read_text())
    validator = Draft202012Validator(schema)

    full_fixture = {
        "experiment_id": "exp-1",
        "succeeded": True,
        "exit_code": 0,
        "metric_value": 12.4,
        "stdout_excerpt": "throughput: 12.4 tok/s\n",
        "stderr_excerpt": "",
        "environment": "python3.11, cuda 12.1, 1x A100",
        "seed": "42",
        "runtime_seconds": 123.4,
        "artifacts": ["artifact://sess-123/plot.png"],
    }
    errors = list(validator.iter_errors(full_fixture))
    assert errors == [], f"RunRecord fixture should validate, got: {[e.message for e in errors]}"

    # Minimal: only required fields.
    minimal = {
        "succeeded": False,
        "exit_code": 1,
        "stdout_excerpt": "",
        "environment": "python3.11",
    }
    errors = list(validator.iter_errors(minimal))
    assert errors == [], f"minimal RunRecord should validate, got: {[e.message for e in errors]}"


def test_run_record_nullable_optionals_validate():
    """Optional numeric/string fields must accept null (additionalProperties:false
    means the core emits them explicitly as null rather than omitting them)."""
    schema = yaml.safe_load(RUN_RECORD_SCHEMA_PATH.read_text())
    validator = Draft202012Validator(schema)

    with_nulls = {
        "experiment_id": "exp-1",
        "succeeded": True,
        "exit_code": 0,
        "metric_value": None,
        "stdout_excerpt": "ok",
        "stderr_excerpt": None,
        "environment": "python3.11",
        "seed": None,
        "runtime_seconds": None,
        "artifacts": [],
    }
    errors = list(validator.iter_errors(with_nulls))
    assert errors == [], (
        f"RunRecord with null optionals should validate, got: "
        f"{[e.message for e in errors]}"
    )


def test_run_record_missing_required_fields_fails():
    schema = yaml.safe_load(RUN_RECORD_SCHEMA_PATH.read_text())
    validator = Draft202012Validator(schema)

    missing_succeeded = {
        "exit_code": 0,
        "stdout_excerpt": "",
        "environment": "python3.11",
    }
    errors = list(validator.iter_errors(missing_succeeded))
    assert errors, "RunRecord missing `succeeded` should NOT validate"

    missing_exit_code = {
        "succeeded": True,
        "stdout_excerpt": "",
        "environment": "python3.11",
    }
    errors = list(validator.iter_errors(missing_exit_code))
    assert errors, "RunRecord missing `exit_code` should NOT validate"

    missing_stdout_excerpt = {
        "succeeded": True,
        "exit_code": 0,
        "environment": "python3.11",
    }
    errors = list(validator.iter_errors(missing_stdout_excerpt))
    assert errors, "RunRecord missing `stdout_excerpt` should NOT validate"

    missing_environment = {
        "succeeded": True,
        "exit_code": 0,
        "stdout_excerpt": "",
    }
    errors = list(validator.iter_errors(missing_environment))
    assert errors, "RunRecord missing `environment` should NOT validate"


def test_run_record_extra_field_fails():
    schema = yaml.safe_load(RUN_RECORD_SCHEMA_PATH.read_text())
    validator = Draft202012Validator(schema)

    extra_key = {
        "succeeded": True,
        "exit_code": 0,
        "stdout_excerpt": "",
        "environment": "python3.11",
        "unexpected_field": "nope",
    }
    errors = list(validator.iter_errors(extra_key))
    assert errors, "RunRecord with an extra property should NOT validate"


def test_store_accepts_run_record_for_runner_agent(tmp_path):
    store = SpecStore(tmp_path / "store")
    try:
        store.put(
            kind="run_record",
            session_id="s1",
            key="exp-1",
            obj={
                "succeeded": True,
                "exit_code": 0,
                "stdout_excerpt": "",
                "environment": "python3.11",
            },
            emitter="RunnerAgent",
        )
    finally:
        store.close()


def test_store_rejects_wrong_emitter_for_run_record(tmp_path):
    store = SpecStore(tmp_path / "store")
    try:
        with pytest.raises(EmitterNotAllowed):
            store.put(
                kind="run_record",
                session_id="s1",
                key="exp-1",
                obj={},
                emitter="ResearchOrchestrator",
            )
    finally:
        store.close()
