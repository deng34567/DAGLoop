"""Unit tests for the DataAnalysisAgent + LabInTheLoop contract slice (Task 1).

Covers: data_analysis.requested, data_analysis.ready (outbound);
user.decision_provided, external.data_uploaded (inbound, emitter 'Human');
the ExternalDataAnalysisReport data-contract schema; the SpecStore whitelist
extension for external_data_analysis_report; and the addition of
DataAnalysisAgent as a NEW legal agent.failed / envelope emitter. Also covers
the new SessionState field external_data_analysis_report_ref.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from jsonschema import Draft202012Validator

from autoresearch.eventbus.broker.registry import Registry
from autoresearch.orchestrator.session_state import SessionState
from autoresearch.spec_store import EmitterNotAllowed, SpecStore

EVENTBUS_ROOT = Path(__file__).resolve().parents[2]
REGISTRY_PATH = EVENTBUS_ROOT / "eventbus-registry.yaml"
EXAMPLES_DIR = EVENTBUS_ROOT / "examples"

AUTORESEARCH_ROOT = EVENTBUS_ROOT.parent
DATA_ANALYSIS_AGENT_ROOT = AUTORESEARCH_ROOT / "agents" / "data_analysis_agent"
REPORT_SCHEMA_PATH = (
    DATA_ANALYSIS_AGENT_ROOT / "data-contracts" / "external_data_analysis_report.schema.yaml"
)

DATA_ANALYSIS_EVENTS = [
    "data_analysis.requested",
    "data_analysis.ready",
    "user.decision_provided",
    "external.data_uploaded",
]


@pytest.fixture
def registry() -> Registry:
    return Registry(REGISTRY_PATH)


# ---------------------------------------------------------------------------
# New events
# ---------------------------------------------------------------------------


def test_registry_loads_data_analysis_validators(registry: Registry):
    for event_type in DATA_ANALYSIS_EVENTS:
        assert registry.get_payload_validator(event_type, 1)


@pytest.mark.parametrize("event_type", DATA_ANALYSIS_EVENTS)
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


def test_data_analysis_requested_registry_entry(registry: Registry):
    entry = registry.get("data_analysis.requested")
    assert entry.direction == "outbound"
    assert entry.emitters == {"ResearchOrchestrator"}
    assert 1 in entry.schema_versions


def test_data_analysis_ready_registry_entry(registry: Registry):
    entry = registry.get("data_analysis.ready")
    assert entry.direction == "outbound"
    assert entry.emitters == {"DataAnalysisAgent"}
    assert 1 in entry.schema_versions


def test_user_decision_provided_registry_entry(registry: Registry):
    entry = registry.get("user.decision_provided")
    assert entry.direction == "inbound"
    assert entry.emitters == {"Human"}
    assert 1 in entry.schema_versions


def test_external_data_uploaded_registry_entry(registry: Registry):
    entry = registry.get("external.data_uploaded")
    assert entry.direction == "inbound"
    assert entry.emitters == {"Human"}
    assert 1 in entry.schema_versions


def test_data_analysis_ready_bad_consensus_status_fails(registry: Registry):
    validator = registry.get_payload_validator("data_analysis.ready", 1)
    payload = {
        "external_data_analysis_report_ref": "ref://external_data_analysis_report/s1/v1",
        "consensus_status": "unknown",
    }
    errors = list(validator.iter_errors(payload))
    assert errors, "data_analysis.ready with bad consensus_status should NOT validate"


# ---------------------------------------------------------------------------
# ExternalDataAnalysisReport data-contract schema
# ---------------------------------------------------------------------------


def test_external_data_analysis_report_fixture_validates():
    schema = yaml.safe_load(REPORT_SCHEMA_PATH.read_text())
    validator = Draft202012Validator(schema)

    fixture = {
        "qc_summary": "All trajectories passed QC.",
        "trajectories": [
            {"trajectory_id": "t1", "conclusion": "Effect confirmed.", "notes": "n=30"},
            {"trajectory_id": "t2", "conclusion": "Effect confirmed.", "notes": None},
        ],
        "consensus_conclusion": "Effect is real and reproducible.",
        "consensus_status": "converged",
        "divergence_points": [],
        "robustness": "high",
        "need_human_review": False,
    }
    errors = list(validator.iter_errors(fixture))
    assert errors == [], (
        f"ExternalDataAnalysisReport fixture should validate, got: "
        f"{[e.message for e in errors]}"
    )


def test_external_data_analysis_report_missing_required_fails():
    schema = yaml.safe_load(REPORT_SCHEMA_PATH.read_text())
    validator = Draft202012Validator(schema)

    fixture = {
        "qc_summary": "All trajectories passed QC.",
        "trajectories": [],
    }
    errors = list(validator.iter_errors(fixture))
    assert errors, "ExternalDataAnalysisReport missing required fields should NOT validate"


def test_external_data_analysis_report_bad_consensus_status_enum_fails():
    schema = yaml.safe_load(REPORT_SCHEMA_PATH.read_text())
    validator = Draft202012Validator(schema)

    fixture = {
        "qc_summary": "All trajectories passed QC.",
        "trajectories": [],
        "consensus_conclusion": "n/a",
        "consensus_status": "unknown",
        "divergence_points": [],
        "robustness": "high",
        "need_human_review": False,
    }
    errors = list(validator.iter_errors(fixture))
    assert errors, "ExternalDataAnalysisReport with bad consensus_status enum should NOT validate"


def test_external_data_analysis_report_bad_robustness_enum_fails():
    schema = yaml.safe_load(REPORT_SCHEMA_PATH.read_text())
    validator = Draft202012Validator(schema)

    fixture = {
        "qc_summary": "All trajectories passed QC.",
        "trajectories": [],
        "consensus_conclusion": "n/a",
        "consensus_status": "converged",
        "divergence_points": [],
        "robustness": "extreme",
        "need_human_review": False,
    }
    errors = list(validator.iter_errors(fixture))
    assert errors, "ExternalDataAnalysisReport with bad robustness enum should NOT validate"


def test_external_data_analysis_report_extra_field_fails():
    schema = yaml.safe_load(REPORT_SCHEMA_PATH.read_text())
    validator = Draft202012Validator(schema)

    fixture = {
        "qc_summary": "All trajectories passed QC.",
        "trajectories": [],
        "consensus_conclusion": "n/a",
        "consensus_status": "converged",
        "divergence_points": [],
        "robustness": "high",
        "need_human_review": False,
        "unexpected_field": "nope",
    }
    errors = list(validator.iter_errors(fixture))
    assert errors, "ExternalDataAnalysisReport with an extra property should NOT validate"


# ---------------------------------------------------------------------------
# SpecStore: external_data_analysis_report kind
# ---------------------------------------------------------------------------


def test_store_accepts_external_data_analysis_report_for_data_analysis_agent(tmp_path):
    store = SpecStore(tmp_path / "store")
    try:
        store.put(
            kind="external_data_analysis_report",
            session_id="s1",
            key="report-1",
            obj={
                "qc_summary": "All trajectories passed QC.",
                "trajectories": [],
                "consensus_conclusion": "n/a",
                "consensus_status": "converged",
                "divergence_points": [],
                "robustness": "high",
                "need_human_review": False,
            },
            emitter="DataAnalysisAgent",
        )
    finally:
        store.close()


def test_store_rejects_wrong_emitter_for_external_data_analysis_report(tmp_path):
    store = SpecStore(tmp_path / "store")
    try:
        with pytest.raises(EmitterNotAllowed):
            store.put(
                kind="external_data_analysis_report",
                session_id="s1",
                key="report-1",
                obj={},
                emitter="ResearchOrchestrator",
            )
    finally:
        store.close()


# ---------------------------------------------------------------------------
# DataAnalysisAgent as a NEW legal agent.failed / envelope emitter.
# ---------------------------------------------------------------------------


def test_agent_failed_includes_data_analysis_agent(registry: Registry):
    entry = registry.get("agent.failed")
    assert "DataAnalysisAgent" in entry.emitters


def test_envelope_schema_allows_data_analysis_agent_emitter(registry: Registry):
    validator = registry.envelope_validator
    emitter_enum = validator.schema["properties"]["emitter"]["enum"]
    assert "DataAnalysisAgent" in emitter_enum


def test_agent_failed_payload_schema_allows_data_analysis_agent():
    schema_path = EVENTBUS_ROOT / "schemas" / "agent.failed.v1.schema.json"
    schema = json.loads(schema_path.read_text())
    agent_name_enum = schema["properties"]["agent_name"]["enum"]
    assert "DataAnalysisAgent" in agent_name_enum


# ---------------------------------------------------------------------------
# The 2 inbound human events emitted by 'Human' validate at the envelope level.
# ---------------------------------------------------------------------------


def test_envelope_schema_allows_human_emitter(registry: Registry):
    validator = registry.envelope_validator
    emitter_enum = validator.schema["properties"]["emitter"]["enum"]
    assert "Human" in emitter_enum


# ---------------------------------------------------------------------------
# SessionState: external_data_analysis_report_ref
# ---------------------------------------------------------------------------


def test_session_state_has_external_data_analysis_report_ref_defaulting_to_none():
    state = SessionState(session_id="s1")
    assert state.external_data_analysis_report_ref is None
    # The plural list field must still exist (do NOT re-add it).
    assert state.external_data_analysis_report_refs == []
