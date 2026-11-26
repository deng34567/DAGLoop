"""Unit tests for the RunnerAgent contract slice (M10 Task 1).

Covers: baseline.reproduction_requested, baseline.reproduced (v1), the
BaselineResult data-contract schema, and the SpecStore whitelist extension
for baseline_result.
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
BASELINE_RESULT_SCHEMA_PATH = (
    RUNNER_AGENT_ROOT / "data-contracts" / "baseline_result.schema.yaml"
)

M10_EVENTS = [
    "baseline.reproduction_requested",
    "baseline.reproduced",
]


@pytest.fixture
def registry() -> Registry:
    return Registry(REGISTRY_PATH)


def test_registry_loads_m10_validators(registry: Registry):
    for event_type in M10_EVENTS:
        assert registry.get_payload_validator(event_type, 1)


@pytest.mark.parametrize("event_type", M10_EVENTS)
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


def test_baseline_reproduction_requested_registry_entry(registry: Registry):
    entry = registry.get("baseline.reproduction_requested")
    assert entry.direction == "outbound"
    assert entry.emitters == {"ResearchOrchestrator"}
    assert 1 in entry.schema_versions


def test_baseline_reproduced_registry_entry(registry: Registry):
    entry = registry.get("baseline.reproduced")
    assert entry.direction == "outbound"
    assert entry.emitters == {"RunnerAgent"}
    assert 1 in entry.schema_versions


def test_baseline_reproduced_invalid_verdict_enum_fails(registry: Registry):
    """The invalid example specifically exercises a bad `verdict` enum value."""
    validator = registry.get_payload_validator("baseline.reproduced", 1)
    invalid_payload = json.loads(
        (EXAMPLES_DIR / "baseline.reproduced.invalid.json").read_text()
    )
    assert invalid_payload.get("verdict") not in (
        "reproduced",
        "failed",
    ), "fixture must exercise an invalid enum value for verdict"
    errors = list(validator.iter_errors(invalid_payload))
    assert errors, "invalid verdict enum value must fail validation"


def test_baseline_result_fixture_validates_against_schema():
    schema = yaml.safe_load(BASELINE_RESULT_SCHEMA_PATH.read_text())
    validator = Draft202012Validator(schema)

    full_fixture = {
        "benchmark_id": "bench-42",
        "reproduced": True,
        "metric_value": 12.4,
        "reported_metric": 12.5,
        "tolerance": 0.1,
        "environment": "python3.11, cuda 12.1, 1x A100",
        "command": "python bench.py --mode baseline",
        "seed": "42",
        "runtime_seconds": 123.4,
        "exit_code": 0,
        "log_excerpt": "throughput: 12.4 tok/s\n",
    }
    errors = list(validator.iter_errors(full_fixture))
    assert errors == [], f"BaselineResult fixture should validate, got: {[e.message for e in errors]}"

    # Minimal: only required fields.
    minimal = {
        "benchmark_id": "bench-1",
        "reproduced": False,
        "command": "python bench.py",
        "exit_code": 1,
        "environment": "python3.11",
    }
    errors = list(validator.iter_errors(minimal))
    assert errors == [], f"minimal BaselineResult should validate, got: {[e.message for e in errors]}"


def test_baseline_result_nullable_optionals_validate():
    """Optional numeric/string fields must accept null (additionalProperties:false
    means the core emits them explicitly as null rather than omitting them)."""
    schema = yaml.safe_load(BASELINE_RESULT_SCHEMA_PATH.read_text())
    validator = Draft202012Validator(schema)

    with_nulls = {
        "benchmark_id": "bench-1",
        "reproduced": True,
        "metric_value": None,
        "reported_metric": None,
        "tolerance": None,
        "environment": "python3.11",
        "command": "python bench.py",
        "seed": None,
        "runtime_seconds": None,
        "exit_code": 0,
        "log_excerpt": None,
    }
    errors = list(validator.iter_errors(with_nulls))
    assert errors == [], (
        f"BaselineResult with null optionals should validate, got: "
        f"{[e.message for e in errors]}"
    )


def test_baseline_result_missing_required_fields_fails():
    schema = yaml.safe_load(BASELINE_RESULT_SCHEMA_PATH.read_text())
    validator = Draft202012Validator(schema)

    missing_reproduced = {
        "benchmark_id": "bench-1",
        "command": "python bench.py",
        "exit_code": 0,
        "environment": "python3.11",
    }
    errors = list(validator.iter_errors(missing_reproduced))
    assert errors, "BaselineResult missing `reproduced` should NOT validate"

    missing_exit_code = {
        "benchmark_id": "bench-1",
        "reproduced": True,
        "command": "python bench.py",
        "environment": "python3.11",
    }
    errors = list(validator.iter_errors(missing_exit_code))
    assert errors, "BaselineResult missing `exit_code` should NOT validate"


def test_baseline_result_extra_field_fails():
    schema = yaml.safe_load(BASELINE_RESULT_SCHEMA_PATH.read_text())
    validator = Draft202012Validator(schema)

    extra_key = {
        "benchmark_id": "bench-1",
        "reproduced": True,
        "command": "python bench.py",
        "exit_code": 0,
        "environment": "python3.11",
        "unexpected_field": "nope",
    }
    errors = list(validator.iter_errors(extra_key))
    assert errors, "BaselineResult with an extra property should NOT validate"


def test_store_accepts_baseline_result_for_runner_agent(tmp_path):
    store = SpecStore(tmp_path / "store")
    try:
        store.put(
            kind="baseline_result",
            session_id="s1",
            key="bench-1",
            obj={
                "benchmark_id": "bench-1",
                "reproduced": True,
                "command": "python bench.py",
                "exit_code": 0,
                "environment": "python3.11",
            },
            emitter="RunnerAgent",
        )
    finally:
        store.close()


def test_store_rejects_wrong_emitter_for_baseline_result(tmp_path):
    store = SpecStore(tmp_path / "store")
    try:
        with pytest.raises(EmitterNotAllowed):
            store.put(
                kind="baseline_result",
                session_id="s1",
                key="bench-1",
                obj={},
                emitter="ResearchOrchestrator",
            )
    finally:
        store.close()
