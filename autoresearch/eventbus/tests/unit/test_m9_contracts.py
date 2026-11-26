"""Unit tests for the CoderAgent contract slice (M9 Task 1).

Covers: code.implementation_requested, code.implemented (v1), the
ExperimentPatch data-contract schema, and the SpecStore whitelist extension
for experiment_patch.
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
CODER_AGENT_ROOT = AUTORESEARCH_ROOT / "agents" / "coder_agent"
EXPERIMENT_PATCH_SCHEMA_PATH = (
    CODER_AGENT_ROOT / "data-contracts" / "experiment_patch.schema.yaml"
)

M9_EVENTS = [
    "code.implementation_requested",
    "code.implemented",
]


@pytest.fixture
def registry() -> Registry:
    return Registry(REGISTRY_PATH)


def test_registry_loads_m9_validators(registry: Registry):
    for event_type in M9_EVENTS:
        assert registry.get_payload_validator(event_type, 1)


@pytest.mark.parametrize("event_type", M9_EVENTS)
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


def test_code_implementation_requested_registry_entry(registry: Registry):
    entry = registry.get("code.implementation_requested")
    assert entry.direction == "outbound"
    assert entry.emitters == {"ResearchOrchestrator"}
    assert 1 in entry.schema_versions


def test_code_implemented_registry_entry(registry: Registry):
    entry = registry.get("code.implemented")
    assert entry.direction == "outbound"
    assert entry.emitters == {"CoderAgent"}
    assert 1 in entry.schema_versions


def test_agent_failed_still_includes_coder_agent(registry: Registry):
    """CoderAgent was already in agent.failed's emitter list -- M9 must not
    remove it."""
    entry = registry.get("agent.failed")
    assert "CoderAgent" in entry.emitters


def test_experiment_patch_fixture_validates_against_schema():
    schema = yaml.safe_load(EXPERIMENT_PATCH_SCHEMA_PATH.read_text())
    validator = Draft202012Validator(schema)

    full_fixture = {
        "hypothesis_id": "hyp-1",
        "summary": "Fuse the two matmul kernels into one to reduce launch overhead.",
        "changes": [
            {
                "file": "src/kernel.py",
                "patch": "--- a/src/kernel.py\n+++ b/src/kernel.py\n@@ -1,2 +1,2 @@\n-old\n+new\n",
            },
        ],
        "touched_files": ["src/kernel.py"],
    }
    errors = list(validator.iter_errors(full_fixture))
    assert errors == [], f"ExperimentPatch fixture should validate, got: {[e.message for e in errors]}"

    minimal = {
        "summary": "Minimal patch.",
        "changes": [{"file": "src/kernel.py", "patch": "diff --git a/src/kernel.py b/src/kernel.py\n"}],
        "touched_files": ["src/kernel.py"],
    }
    errors = list(validator.iter_errors(minimal))
    assert errors == [], f"minimal ExperimentPatch should validate, got: {[e.message for e in errors]}"


def test_experiment_patch_missing_required_fails():
    schema = yaml.safe_load(EXPERIMENT_PATCH_SCHEMA_PATH.read_text())
    validator = Draft202012Validator(schema)

    missing_required = {"hypothesis_id": "hyp-1"}
    errors = list(validator.iter_errors(missing_required))
    assert errors, "ExperimentPatch missing required fields should NOT validate"


def test_experiment_patch_changes_item_missing_file_fails():
    schema = yaml.safe_load(EXPERIMENT_PATCH_SCHEMA_PATH.read_text())
    validator = Draft202012Validator(schema)

    missing_file = {
        "summary": "Minimal patch.",
        "changes": [{"patch": "diff --git a/src/kernel.py b/src/kernel.py\n"}],
        "touched_files": ["src/kernel.py"],
    }
    errors = list(validator.iter_errors(missing_file))
    assert errors, "ExperimentPatch changes[] item missing `file` should NOT validate"


def test_experiment_patch_missing_touched_files_fails():
    schema = yaml.safe_load(EXPERIMENT_PATCH_SCHEMA_PATH.read_text())
    validator = Draft202012Validator(schema)

    missing_touched_files = {
        "summary": "Minimal patch.",
        "changes": [{"file": "src/kernel.py", "patch": "diff --git a/src/kernel.py b/src/kernel.py\n"}],
    }
    errors = list(validator.iter_errors(missing_touched_files))
    assert errors, "ExperimentPatch missing touched_files should NOT validate"


def test_experiment_patch_invalid_fixture_fails_schema():
    """additionalProperties:false at the top level -- a stray field should fail."""
    schema = yaml.safe_load(EXPERIMENT_PATCH_SCHEMA_PATH.read_text())
    validator = Draft202012Validator(schema)

    extra_key = {
        "summary": "Minimal patch.",
        "changes": [{"file": "src/kernel.py", "patch": "diff --git a/src/kernel.py b/src/kernel.py\n"}],
        "touched_files": ["src/kernel.py"],
        "unexpected_field": "nope",
    }
    errors = list(validator.iter_errors(extra_key))
    assert errors, "ExperimentPatch with an extra top-level property should NOT validate"


def test_store_accepts_experiment_patch_for_coder_agent(tmp_path):
    store = SpecStore(tmp_path / "store")
    try:
        store.put(
            kind="experiment_patch",
            session_id="s1",
            key="final",
            obj={
                "summary": "Minimal patch.",
                "changes": [{"file": "src/kernel.py", "patch": "diff --git a/src/kernel.py b/src/kernel.py\n"}],
                "touched_files": ["src/kernel.py"],
            },
            emitter="CoderAgent",
        )
    finally:
        store.close()


def test_store_rejects_wrong_emitter_for_experiment_patch(tmp_path):
    store = SpecStore(tmp_path / "store")
    try:
        with pytest.raises(EmitterNotAllowed):
            store.put(
                kind="experiment_patch",
                session_id="s1",
                key="final",
                obj={},
                emitter="ResearchOrchestrator",
            )
    finally:
        store.close()
