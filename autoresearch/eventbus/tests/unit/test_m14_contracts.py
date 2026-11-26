"""Unit tests for the ReportAgent contract slice (M14 Task 1).

Covers: report.requested, report.ready (v1); the Report data-contract schema
(the "honest report" structure -- summary + 8 required classification arrays
+ experiment_ledger); the SpecStore whitelist extension for report; and
ReportAgent as an already-legal agent.failed emitter / envelope.schema.json
emitter (confirmed, not modified by this slice).
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
REPORT_AGENT_ROOT = AUTORESEARCH_ROOT / "agents" / "report_agent"
REPORT_SCHEMA_PATH = REPORT_AGENT_ROOT / "data-contracts" / "report.schema.yaml"

M14_EVENTS = [
    "report.requested",
    "report.ready",
]

REQUIRED_CLASSIFICATION_ARRAYS = [
    "verified_findings",
    "negative_results",
    "unverified_ideas",
    "experiment_not_run",
    "replication_failed",
    "external_data_incomplete",
    "analysis_caveats",
    "known_risks",
]


@pytest.fixture
def registry() -> Registry:
    return Registry(REGISTRY_PATH)


def _minimal_report() -> dict:
    report = {name: [] for name in REQUIRED_CLASSIFICATION_ARRAYS}
    report["summary"] = "A minimal honest report."
    report["experiment_ledger"] = []
    return report


# ---------------------------------------------------------------------------
# New events: report.requested, report.ready
# ---------------------------------------------------------------------------


def test_registry_loads_m14_validators(registry: Registry):
    for event_type in M14_EVENTS:
        assert registry.get_payload_validator(event_type, 1)


@pytest.mark.parametrize("event_type", M14_EVENTS)
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


def test_report_requested_registry_entry(registry: Registry):
    entry = registry.get("report.requested")
    assert entry.direction == "outbound"
    assert entry.emitters == {"ResearchOrchestrator"}
    assert 1 in entry.schema_versions


def test_report_ready_registry_entry(registry: Registry):
    entry = registry.get("report.ready")
    assert entry.direction == "outbound"
    assert entry.emitters == {"ReportAgent"}
    assert 1 in entry.schema_versions


def test_report_requested_missing_optional_spec_ref_validates(registry: Registry):
    validator = registry.get_payload_validator("report.requested", 1)
    payload = {"research_memory_ref": "research_memory://sess-123/consolidation-1"}
    errors = list(validator.iter_errors(payload))
    assert errors == [], (
        f"report.requested without research_spec_ref should validate (optional), "
        f"got: {[e.message for e in errors]}"
    )


def test_report_requested_null_optional_spec_ref_validates(registry: Registry):
    validator = registry.get_payload_validator("report.requested", 1)
    payload = {
        "research_memory_ref": "research_memory://sess-123/consolidation-1",
        "research_spec_ref": None,
    }
    errors = list(validator.iter_errors(payload))
    assert errors == [], (
        f"report.requested with research_spec_ref=null should validate (nullable), "
        f"got: {[e.message for e in errors]}"
    )


def test_report_ready_invalid_missing_report_ref(registry: Registry):
    validator = registry.get_payload_validator("report.ready", 1)
    invalid_payload = json.loads(
        (EXAMPLES_DIR / "report.ready.invalid.json").read_text()
    )
    assert "report_ref" not in invalid_payload
    errors = list(validator.iter_errors(invalid_payload))
    assert errors, "report.ready missing report_ref should NOT validate"


# ---------------------------------------------------------------------------
# Report data-contract schema
# ---------------------------------------------------------------------------


def test_report_fixture_validates_with_empty_arrays():
    schema = yaml.safe_load(REPORT_SCHEMA_PATH.read_text())
    validator = Draft202012Validator(schema)

    errors = list(validator.iter_errors(_minimal_report()))
    assert errors == [], (
        f"minimal Report (empty required arrays) should validate, got: "
        f"{[e.message for e in errors]}"
    )


def test_report_full_fixture_validates():
    schema = yaml.safe_load(REPORT_SCHEMA_PATH.read_text())
    validator = Draft202012Validator(schema)

    full_fixture = _minimal_report()
    full_fixture.update(
        {
            "verified_findings": ["finding A"],
            "negative_results": ["negative A"],
            "unverified_ideas": ["idea A"],
            "experiment_not_run": ["exp not run A"],
            "replication_failed": ["replication A"],
            "external_data_incomplete": ["external A"],
            "analysis_caveats": ["caveat A"],
            "known_risks": ["risk A"],
            "experiment_ledger": [
                {
                    "experiment_id": "exp-1",
                    "metric_value": None,
                    "verdict": "pass",
                    "notes": "n/a",
                }
            ],
            "report_failed": False,
        }
    )
    errors = list(validator.iter_errors(full_fixture))
    assert errors == [], (
        f"full Report fixture should validate, got: {[e.message for e in errors]}"
    )


@pytest.mark.parametrize("missing_field", REQUIRED_CLASSIFICATION_ARRAYS)
def test_report_missing_required_array_fails(missing_field: str):
    schema = yaml.safe_load(REPORT_SCHEMA_PATH.read_text())
    validator = Draft202012Validator(schema)

    report = _minimal_report()
    del report[missing_field]
    errors = list(validator.iter_errors(report))
    assert errors, f"Report missing `{missing_field}` should NOT validate"


def test_report_missing_summary_fails():
    schema = yaml.safe_load(REPORT_SCHEMA_PATH.read_text())
    validator = Draft202012Validator(schema)

    report = _minimal_report()
    del report["summary"]
    errors = list(validator.iter_errors(report))
    assert errors, "Report missing `summary` should NOT validate"


def test_report_missing_experiment_ledger_fails():
    schema = yaml.safe_load(REPORT_SCHEMA_PATH.read_text())
    validator = Draft202012Validator(schema)

    report = _minimal_report()
    del report["experiment_ledger"]
    errors = list(validator.iter_errors(report))
    assert errors, "Report missing `experiment_ledger` should NOT validate"


def test_report_extra_field_fails():
    schema = yaml.safe_load(REPORT_SCHEMA_PATH.read_text())
    validator = Draft202012Validator(schema)

    report = _minimal_report()
    report["unexpected_field"] = "nope"
    errors = list(validator.iter_errors(report))
    assert errors, "Report with an extra property should NOT validate"


def test_report_ledger_item_with_null_metric_value_validates():
    schema = yaml.safe_load(REPORT_SCHEMA_PATH.read_text())
    validator = Draft202012Validator(schema)

    report = _minimal_report()
    report["experiment_ledger"] = [{"metric_value": None}]
    errors = list(validator.iter_errors(report))
    assert errors == [], (
        f"ledger item with metric_value=null should validate, got: "
        f"{[e.message for e in errors]}"
    )


def test_report_ledger_item_extra_field_fails():
    schema = yaml.safe_load(REPORT_SCHEMA_PATH.read_text())
    validator = Draft202012Validator(schema)

    report = _minimal_report()
    report["experiment_ledger"] = [{"unexpected_ledger_field": "nope"}]
    errors = list(validator.iter_errors(report))
    assert errors, "ledger item with an extra property should NOT validate"


# ---------------------------------------------------------------------------
# SpecStore: report kind + ReportAgent as emitter
# ---------------------------------------------------------------------------


def test_store_accepts_report_for_report_agent(tmp_path):
    store = SpecStore(tmp_path / "store")
    try:
        store.put(
            kind="report",
            session_id="s1",
            key="final",
            obj=_minimal_report(),
            emitter="ReportAgent",
        )
    finally:
        store.close()


def test_store_rejects_wrong_emitter_for_report(tmp_path):
    store = SpecStore(tmp_path / "store")
    try:
        with pytest.raises(EmitterNotAllowed):
            store.put(
                kind="report",
                session_id="s1",
                key="final",
                obj={},
                emitter="ResearchOrchestrator",
            )
    finally:
        store.close()


# ---------------------------------------------------------------------------
# ReportAgent is ALREADY a legal agent.failed emitter / envelope emitter --
# confirm, don't remove/re-add.
# ---------------------------------------------------------------------------


def test_agent_failed_already_includes_report_agent(registry: Registry):
    entry = registry.get("agent.failed")
    assert "ReportAgent" in entry.emitters


def test_envelope_schema_already_includes_report_agent():
    envelope_schema_path = EVENTBUS_ROOT / "schemas" / "envelope.schema.json"
    envelope_schema = json.loads(envelope_schema_path.read_text())
    emitter_enum = envelope_schema["properties"]["emitter"]["enum"]
    assert "ReportAgent" in emitter_enum
