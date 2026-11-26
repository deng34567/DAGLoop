"""Unit tests for the ExperimentPlannerAgent contract slice (M8 Task 1).

Covers: experiment.planning_requested, experiment.plan_ready,
experiment.review_requested, experiment.plan_reviewed (v1), the
ExperimentPlan data-contract schema, the SpecStore whitelist extension for
experiment_plan, and the critique_report `target` enum extension to include
`experiment_plan`.
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
EXPERIMENT_PLANNER_AGENT_ROOT = AUTORESEARCH_ROOT / "agents" / "experiment_planner_agent"
EXPERIMENT_PLAN_SCHEMA_PATH = (
    EXPERIMENT_PLANNER_AGENT_ROOT / "data-contracts" / "experiment_plan.schema.yaml"
)
CRITIQUE_REPORT_SCHEMA_PATH = (
    AUTORESEARCH_ROOT / "agents" / "critic_agent" / "data-contracts" / "critique_report.schema.yaml"
)

M8_EVENTS = [
    "experiment.planning_requested",
    "experiment.plan_ready",
    "experiment.review_requested",
    "experiment.plan_reviewed",
]


@pytest.fixture
def registry() -> Registry:
    return Registry(REGISTRY_PATH)


def test_registry_loads_m8_validators(registry: Registry):
    for event_type in M8_EVENTS:
        assert registry.get_payload_validator(event_type, 1)


@pytest.mark.parametrize("event_type", M8_EVENTS)
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


def test_experiment_planning_requested_registry_entry(registry: Registry):
    entry = registry.get("experiment.planning_requested")
    assert entry.direction == "outbound"
    assert entry.emitters == {"ResearchOrchestrator"}
    assert 1 in entry.schema_versions


def test_experiment_plan_ready_registry_entry(registry: Registry):
    entry = registry.get("experiment.plan_ready")
    assert entry.direction == "outbound"
    assert entry.emitters == {"ExperimentPlannerAgent"}
    assert 1 in entry.schema_versions


def test_experiment_review_requested_registry_entry(registry: Registry):
    entry = registry.get("experiment.review_requested")
    assert entry.direction == "outbound"
    assert entry.emitters == {"ResearchOrchestrator"}
    assert 1 in entry.schema_versions


def test_experiment_plan_reviewed_registry_entry(registry: Registry):
    entry = registry.get("experiment.plan_reviewed")
    assert entry.direction == "outbound"
    assert entry.emitters == {"CriticAgent", "Human"}
    assert 1 in entry.schema_versions


def test_experiment_plan_reviewed_invalid_verdict_enum_fails(registry: Registry):
    """The invalid example specifically exercises a bad `verdict` enum value."""
    validator = registry.get_payload_validator("experiment.plan_reviewed", 1)
    invalid_payload = json.loads(
        (EXAMPLES_DIR / "experiment.plan_reviewed.invalid.json").read_text()
    )
    assert invalid_payload.get("verdict") not in ("pass", "fail"), (
        "fixture must exercise an invalid enum value for verdict"
    )
    errors = list(validator.iter_errors(invalid_payload))
    assert errors, "invalid verdict enum value must fail validation"


def test_experiment_plan_fixture_validates_against_schema():
    schema = yaml.safe_load(EXPERIMENT_PLAN_SCHEMA_PATH.read_text())
    validator = Draft202012Validator(schema)

    full_fixture = {
        "hypothesis_id": "hyp-1",
        "change_summary": "Fuse the two matmul kernels into one to reduce launch overhead.",
        "changes": [
            {"file": "src/kernel.py", "description": "Fuse matmul_a and matmul_b into matmul_fused."},
        ],
        "ablations": ["no_fusion_baseline"],
        "expected_outcome": "10% throughput improvement on the benchmark.",
        "resource_budget": "1x A100, 2h wall-clock",
        "success_criteria": "tokens_per_second improves by >= 10% vs baseline",
    }
    errors = list(validator.iter_errors(full_fixture))
    assert errors == [], f"ExperimentPlan fixture should validate, got: {[e.message for e in errors]}"

    minimal = {
        "hypothesis_id": "hyp-1",
        "change_summary": "Minimal change.",
        "changes": [{"file": "src/kernel.py", "description": "Tweak loop order."}],
        "expected_outcome": "Small speedup.",
    }
    errors = list(validator.iter_errors(minimal))
    assert errors == [], f"minimal ExperimentPlan should validate, got: {[e.message for e in errors]}"


def test_experiment_plan_missing_required_fails():
    schema = yaml.safe_load(EXPERIMENT_PLAN_SCHEMA_PATH.read_text())
    validator = Draft202012Validator(schema)

    missing_required = {"hypothesis_id": "hyp-1"}
    errors = list(validator.iter_errors(missing_required))
    assert errors, "ExperimentPlan missing required fields should NOT validate"


def test_experiment_plan_changes_item_extra_field_fails():
    """`changes[]` items are additionalProperties:false -- a stray `code_diff`
    field is the key declarative, no-code guard at schema level."""
    schema = yaml.safe_load(EXPERIMENT_PLAN_SCHEMA_PATH.read_text())
    validator = Draft202012Validator(schema)

    with_code_diff = {
        "hypothesis_id": "hyp-1",
        "change_summary": "Minimal change.",
        "changes": [
            {
                "file": "src/kernel.py",
                "description": "Tweak loop order.",
                "code_diff": "--- a/src/kernel.py\n+++ b/src/kernel.py\n",
            }
        ],
        "expected_outcome": "Small speedup.",
    }
    errors = list(validator.iter_errors(with_code_diff))
    assert errors, "ExperimentPlan changes[] item with an extra `code_diff` field should NOT validate"


def test_experiment_plan_invalid_fixture_fails_schema():
    schema = yaml.safe_load(EXPERIMENT_PLAN_SCHEMA_PATH.read_text())
    validator = Draft202012Validator(schema)

    extra_key = {
        "hypothesis_id": "hyp-1",
        "change_summary": "Minimal change.",
        "changes": [{"file": "src/kernel.py", "description": "Tweak loop order."}],
        "expected_outcome": "Small speedup.",
        "unexpected_field": "nope",
    }
    errors = list(validator.iter_errors(extra_key))
    assert errors, "ExperimentPlan with an extra top-level property should NOT validate"


def test_critique_report_target_experiment_plan_validates():
    schema = yaml.safe_load(CRITIQUE_REPORT_SCHEMA_PATH.read_text())
    validator = Draft202012Validator(schema)

    report = {
        "target": "experiment_plan",
        "verdict": "pass",
        "issues": [],
    }
    errors = list(validator.iter_errors(report))
    assert errors == [], (
        f"CritiqueReport with target=experiment_plan should validate, got: "
        f"{[e.message for e in errors]}"
    )


def test_store_accepts_experiment_plan_for_experiment_planner_agent(tmp_path):
    store = SpecStore(tmp_path / "store")
    try:
        store.put(
            kind="experiment_plan",
            session_id="s1",
            key="final",
            obj={
                "hypothesis_id": "hyp-1",
                "change_summary": "Minimal change.",
                "changes": [{"file": "src/kernel.py", "description": "Tweak loop order."}],
                "expected_outcome": "Small speedup.",
            },
            emitter="ExperimentPlannerAgent",
        )
    finally:
        store.close()


def test_store_rejects_wrong_emitter_for_experiment_plan(tmp_path):
    store = SpecStore(tmp_path / "store")
    try:
        with pytest.raises(EmitterNotAllowed):
            store.put(
                kind="experiment_plan",
                session_id="s1",
                key="final",
                obj={},
                emitter="ResearchOrchestrator",
            )
    finally:
        store.close()
