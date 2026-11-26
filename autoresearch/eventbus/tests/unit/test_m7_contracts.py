"""Unit tests for the EvaluatorAgent contract slice (M7 Task 1).

Covers: benchmark.selection_requested, benchmark.spec_ready,
verifier.preparation_requested, verifier.spec_frozen (v1), the BenchmarkSpec
and VerifierSpec data-contract schemas, and the SpecStore whitelist extension
for benchmark_spec / verifier_spec.
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
BENCHMARK_SPEC_SCHEMA_PATH = (
    EVALUATOR_AGENT_ROOT / "data-contracts" / "benchmark_spec.schema.yaml"
)
VERIFIER_SPEC_SCHEMA_PATH = (
    EVALUATOR_AGENT_ROOT / "data-contracts" / "verifier_spec.schema.yaml"
)

M7_EVENTS = [
    "benchmark.selection_requested",
    "benchmark.spec_ready",
    "verifier.preparation_requested",
    "verifier.spec_frozen",
]


@pytest.fixture
def registry() -> Registry:
    return Registry(REGISTRY_PATH)


def test_registry_loads_m7_validators(registry: Registry):
    for event_type in M7_EVENTS:
        assert registry.get_payload_validator(event_type, 1)


@pytest.mark.parametrize("event_type", M7_EVENTS)
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


def test_benchmark_selection_requested_registry_entry(registry: Registry):
    entry = registry.get("benchmark.selection_requested")
    assert entry.direction == "outbound"
    assert entry.emitters == {"ResearchOrchestrator"}
    assert 1 in entry.schema_versions


def test_benchmark_spec_ready_registry_entry(registry: Registry):
    entry = registry.get("benchmark.spec_ready")
    assert entry.direction == "outbound"
    assert entry.emitters == {"EvaluatorAgent"}
    assert 1 in entry.schema_versions


def test_verifier_preparation_requested_registry_entry(registry: Registry):
    entry = registry.get("verifier.preparation_requested")
    assert entry.direction == "outbound"
    assert entry.emitters == {"ResearchOrchestrator"}
    assert 1 in entry.schema_versions


def test_verifier_spec_frozen_registry_entry(registry: Registry):
    entry = registry.get("verifier.spec_frozen")
    assert entry.direction == "outbound"
    assert entry.emitters == {"EvaluatorAgent"}
    assert 1 in entry.schema_versions


def test_verifier_spec_frozen_invalid_verdict_enum_fails(registry: Registry):
    """The invalid example specifically exercises a bad `verdict` enum value."""
    validator = registry.get_payload_validator("verifier.spec_frozen", 1)
    invalid_payload = json.loads(
        (EXAMPLES_DIR / "verifier.spec_frozen.invalid.json").read_text()
    )
    assert invalid_payload.get("verdict") not in (
        "ok",
        "incomplete",
        "invalid",
    ), "fixture must exercise an invalid enum value for verdict"
    errors = list(validator.iter_errors(invalid_payload))
    assert errors, "invalid verdict enum value must fail validation"


def test_benchmark_spec_fixture_validates_against_schema():
    schema = yaml.safe_load(BENCHMARK_SPEC_SCHEMA_PATH.read_text())
    validator = Draft202012Validator(schema)

    full_fixture = {
        "benchmark_id": "bench-42",
        "name": "Matmul kernel throughput",
        "baseline": {"command": "python bench.py --mode baseline", "reported_metric": 12.5},
        "metric": "tokens_per_second",
        "run_command": "python bench.py --mode candidate",
        "resource_requirements": "1x A100, 16GB RAM",
        "forbidden_files": ["bench.py", "harness/"],
    }
    errors = list(validator.iter_errors(full_fixture))
    assert errors == [], f"BenchmarkSpec fixture should validate, got: {[e.message for e in errors]}"

    # Minimal: only required fields (baseline requires only `command`).
    minimal = {
        "benchmark_id": "bench-1",
        "name": "minimal bench",
        "metric": "latency_ms",
        "run_command": "python run.py",
        "baseline": {"command": "python run.py --baseline"},
    }
    errors = list(validator.iter_errors(minimal))
    assert errors == [], f"minimal BenchmarkSpec should validate, got: {[e.message for e in errors]}"


def test_benchmark_spec_invalid_fixture_fails_schema():
    schema = yaml.safe_load(BENCHMARK_SPEC_SCHEMA_PATH.read_text())
    validator = Draft202012Validator(schema)

    # Missing required fields.
    missing_required = {
        "benchmark_id": "bench-1",
    }
    errors = list(validator.iter_errors(missing_required))
    assert errors, "BenchmarkSpec missing required fields should NOT validate"

    # baseline missing its required `command`.
    bad_baseline = {
        "benchmark_id": "bench-1",
        "name": "bad baseline",
        "metric": "latency_ms",
        "run_command": "python run.py",
        "baseline": {"reported_metric": 1.0},
    }
    errors = list(validator.iter_errors(bad_baseline))
    assert errors, "BenchmarkSpec with baseline missing `command` should NOT validate"

    # Extra unknown key.
    extra_key = {
        "benchmark_id": "bench-1",
        "name": "extra key",
        "metric": "latency_ms",
        "run_command": "python run.py",
        "baseline": {"command": "python run.py --baseline"},
        "unexpected_field": "nope",
    }
    errors = list(validator.iter_errors(extra_key))
    assert errors, "BenchmarkSpec with an extra property should NOT validate"


def test_verifier_spec_fixture_validates_against_schema():
    schema = yaml.safe_load(VERIFIER_SPEC_SCHEMA_PATH.read_text())
    validator = Draft202012Validator(schema)

    full_fixture = {
        "evaluator_hash": "sha256:abc123",
        "metric": "tokens_per_second",
        "data_split": {"in_loop": "split_a", "held_out": "split_b"},
        "run_budget": "10 runs, 1h wall-clock",
        "seed_policy": "fixed seed=42 across runs",
        "editable_region": ["src/kernel.py"],
        "forbidden_files": ["harness/eval.py"],
        "result_parse_schema": {"type": "object"},
    }
    errors = list(validator.iter_errors(full_fixture))
    assert errors == [], f"VerifierSpec fixture should validate, got: {[e.message for e in errors]}"

    # Minimal: only required fields.
    minimal = {
        "evaluator_hash": "sha256:abc123",
        "metric": "latency_ms",
        "data_split": {"in_loop": "split_a", "held_out": "split_b"},
        "editable_region": ["src/kernel.py"],
        "forbidden_files": [],
    }
    errors = list(validator.iter_errors(minimal))
    assert errors == [], f"minimal VerifierSpec should validate, got: {[e.message for e in errors]}"


def test_verifier_spec_missing_editable_region_fails():
    schema = yaml.safe_load(VERIFIER_SPEC_SCHEMA_PATH.read_text())
    validator = Draft202012Validator(schema)

    missing_editable_region = {
        "evaluator_hash": "sha256:abc123",
        "metric": "latency_ms",
        "data_split": {"in_loop": "split_a", "held_out": "split_b"},
        "forbidden_files": [],
    }
    errors = list(validator.iter_errors(missing_editable_region))
    assert errors, "VerifierSpec missing `editable_region` should NOT validate"


def test_verifier_spec_missing_data_split_fails():
    schema = yaml.safe_load(VERIFIER_SPEC_SCHEMA_PATH.read_text())
    validator = Draft202012Validator(schema)

    missing_data_split = {
        "evaluator_hash": "sha256:abc123",
        "metric": "latency_ms",
        "editable_region": ["src/kernel.py"],
        "forbidden_files": [],
    }
    errors = list(validator.iter_errors(missing_data_split))
    assert errors, "VerifierSpec missing `data_split` should NOT validate"

    # data_split present but missing `held_out`.
    incomplete_data_split = {
        "evaluator_hash": "sha256:abc123",
        "metric": "latency_ms",
        "data_split": {"in_loop": "split_a"},
        "editable_region": ["src/kernel.py"],
        "forbidden_files": [],
    }
    errors = list(validator.iter_errors(incomplete_data_split))
    assert errors, "VerifierSpec with data_split missing `held_out` should NOT validate"


def test_verifier_spec_invalid_fixture_fails_schema():
    schema = yaml.safe_load(VERIFIER_SPEC_SCHEMA_PATH.read_text())
    validator = Draft202012Validator(schema)

    extra_key = {
        "evaluator_hash": "sha256:abc123",
        "metric": "latency_ms",
        "data_split": {"in_loop": "split_a", "held_out": "split_b"},
        "editable_region": ["src/kernel.py"],
        "forbidden_files": [],
        "unexpected_field": "nope",
    }
    errors = list(validator.iter_errors(extra_key))
    assert errors, "VerifierSpec with an extra property should NOT validate"


def test_store_accepts_benchmark_spec_for_evaluator_agent(tmp_path):
    store = SpecStore(tmp_path / "store")
    try:
        store.put(
            kind="benchmark_spec",
            session_id="s1",
            key="final",
            obj={
                "benchmark_id": "bench-1",
                "name": "minimal bench",
                "metric": "latency_ms",
                "run_command": "python run.py",
                "baseline": {"command": "python run.py --baseline"},
            },
            emitter="EvaluatorAgent",
        )
    finally:
        store.close()


def test_store_rejects_wrong_emitter_for_benchmark_spec(tmp_path):
    store = SpecStore(tmp_path / "store")
    try:
        with pytest.raises(EmitterNotAllowed):
            store.put(
                kind="benchmark_spec",
                session_id="s1",
                key="final",
                obj={},
                emitter="ResearchOrchestrator",
            )
    finally:
        store.close()


def test_store_accepts_verifier_spec_for_evaluator_agent(tmp_path):
    store = SpecStore(tmp_path / "store")
    try:
        store.put(
            kind="verifier_spec",
            session_id="s1",
            key="final",
            obj={
                "evaluator_hash": "sha256:abc123",
                "metric": "latency_ms",
                "data_split": {"in_loop": "split_a", "held_out": "split_b"},
                "editable_region": ["src/kernel.py"],
                "forbidden_files": [],
            },
            emitter="EvaluatorAgent",
        )
    finally:
        store.close()


def test_store_rejects_wrong_emitter_for_verifier_spec(tmp_path):
    store = SpecStore(tmp_path / "store")
    try:
        with pytest.raises(EmitterNotAllowed):
            store.put(
                kind="verifier_spec",
                session_id="s1",
                key="final",
                obj={},
                emitter="ResearchOrchestrator",
            )
    finally:
        store.close()
