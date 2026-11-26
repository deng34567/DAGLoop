"""Unit tests for the EvidenceAuditAgent contract slice (Task 1).

Covers: evidence.audit_requested, evidence.audited,
evidence.citation_audit_requested, evidence.citation_audited (v1); the
EvidenceAuditReport and CitationAuditReport data-contract schemas; the
SpecStore whitelist extension for evidence_audit_report / citation_audit_report;
and the addition of EvidenceAuditAgent as a NEW legal agent.failed / envelope
emitter. Also covers the two new SessionState refs.
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
EVIDENCE_AUDIT_AGENT_ROOT = AUTORESEARCH_ROOT / "agents" / "evidence_audit_agent"
EVIDENCE_AUDIT_REPORT_SCHEMA_PATH = (
    EVIDENCE_AUDIT_AGENT_ROOT / "data-contracts" / "evidence_audit_report.schema.yaml"
)
CITATION_AUDIT_REPORT_SCHEMA_PATH = (
    EVIDENCE_AUDIT_AGENT_ROOT / "data-contracts" / "citation_audit_report.schema.yaml"
)

EVIDENCE_EVENTS = [
    "evidence.audit_requested",
    "evidence.audited",
    "evidence.citation_audit_requested",
    "evidence.citation_audited",
]


@pytest.fixture
def registry() -> Registry:
    return Registry(REGISTRY_PATH)


# ---------------------------------------------------------------------------
# New events
# ---------------------------------------------------------------------------


def test_registry_loads_evidence_audit_validators(registry: Registry):
    for event_type in EVIDENCE_EVENTS:
        assert registry.get_payload_validator(event_type, 1)


@pytest.mark.parametrize("event_type", EVIDENCE_EVENTS)
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


def test_evidence_audit_requested_registry_entry(registry: Registry):
    entry = registry.get("evidence.audit_requested")
    assert entry.direction == "outbound"
    assert entry.emitters == {"ResearchOrchestrator"}
    assert 1 in entry.schema_versions


def test_evidence_audited_registry_entry(registry: Registry):
    entry = registry.get("evidence.audited")
    assert entry.direction == "outbound"
    assert entry.emitters == {"EvidenceAuditAgent"}
    assert 1 in entry.schema_versions


def test_evidence_citation_audit_requested_registry_entry(registry: Registry):
    entry = registry.get("evidence.citation_audit_requested")
    assert entry.direction == "outbound"
    assert entry.emitters == {"ResearchOrchestrator"}
    assert 1 in entry.schema_versions


def test_evidence_citation_audited_registry_entry(registry: Registry):
    entry = registry.get("evidence.citation_audited")
    assert entry.direction == "outbound"
    assert entry.emitters == {"EvidenceAuditAgent"}
    assert 1 in entry.schema_versions


# ---------------------------------------------------------------------------
# EvidenceAuditReport data-contract schema
# ---------------------------------------------------------------------------


def test_evidence_audit_report_fixture_validates():
    schema = yaml.safe_load(EVIDENCE_AUDIT_REPORT_SCHEMA_PATH.read_text())
    validator = Draft202012Validator(schema)

    fixture = {
        "claim_citation_matrix": [
            {
                "claim": "Method X improves accuracy by 5%.",
                "citations": ["Smith2023", "Doe2022"],
                "support": "strong",
            }
        ],
        "missing_papers": ["Lee2021"],
        "expired_citations": ["OldPaper2015"],
        "evidence_strength": "moderate",
        "notes": "Overall solid coverage.",
    }
    errors = list(validator.iter_errors(fixture))
    assert errors == [], (
        f"EvidenceAuditReport fixture should validate, got: {[e.message for e in errors]}"
    )


def test_evidence_audit_report_with_null_notes_validates():
    schema = yaml.safe_load(EVIDENCE_AUDIT_REPORT_SCHEMA_PATH.read_text())
    validator = Draft202012Validator(schema)

    fixture = {
        "claim_citation_matrix": [],
        "missing_papers": [],
        "expired_citations": [],
        "evidence_strength": "weak",
        "notes": None,
    }
    errors = list(validator.iter_errors(fixture))
    assert errors == [], (
        f"EvidenceAuditReport with notes=null should validate, got: "
        f"{[e.message for e in errors]}"
    )


def test_evidence_audit_report_missing_required_fails():
    schema = yaml.safe_load(EVIDENCE_AUDIT_REPORT_SCHEMA_PATH.read_text())
    validator = Draft202012Validator(schema)

    fixture = {
        "missing_papers": [],
        "expired_citations": [],
        "evidence_strength": "strong",
    }
    errors = list(validator.iter_errors(fixture))
    assert errors, "EvidenceAuditReport missing `claim_citation_matrix` should NOT validate"


def test_evidence_audit_report_bad_support_enum_fails():
    schema = yaml.safe_load(EVIDENCE_AUDIT_REPORT_SCHEMA_PATH.read_text())
    validator = Draft202012Validator(schema)

    fixture = {
        "claim_citation_matrix": [
            {"claim": "X", "citations": [], "support": "somewhat"}
        ],
        "missing_papers": [],
        "expired_citations": [],
        "evidence_strength": "strong",
    }
    errors = list(validator.iter_errors(fixture))
    assert errors, "EvidenceAuditReport with bad support enum should NOT validate"


def test_evidence_audit_report_bad_evidence_strength_enum_fails():
    schema = yaml.safe_load(EVIDENCE_AUDIT_REPORT_SCHEMA_PATH.read_text())
    validator = Draft202012Validator(schema)

    fixture = {
        "claim_citation_matrix": [],
        "missing_papers": [],
        "expired_citations": [],
        "evidence_strength": "super_strong",
    }
    errors = list(validator.iter_errors(fixture))
    assert errors, "EvidenceAuditReport with bad evidence_strength enum should NOT validate"


def test_evidence_audit_report_extra_field_fails():
    schema = yaml.safe_load(EVIDENCE_AUDIT_REPORT_SCHEMA_PATH.read_text())
    validator = Draft202012Validator(schema)

    fixture = {
        "claim_citation_matrix": [],
        "missing_papers": [],
        "expired_citations": [],
        "evidence_strength": "strong",
        "unexpected_field": "nope",
    }
    errors = list(validator.iter_errors(fixture))
    assert errors, "EvidenceAuditReport with an extra property should NOT validate"


# ---------------------------------------------------------------------------
# CitationAuditReport data-contract schema
# ---------------------------------------------------------------------------


def test_citation_audit_report_fixture_validates():
    schema = yaml.safe_load(CITATION_AUDIT_REPORT_SCHEMA_PATH.read_text())
    validator = Draft202012Validator(schema)

    fixture = {
        "unsupported_claims": [
            {"claim": "Method X is the fastest.", "why": "No benchmark cited."}
        ],
        "checked_count": 12,
        "notes": "One claim lacks supporting evidence.",
    }
    errors = list(validator.iter_errors(fixture))
    assert errors == [], (
        f"CitationAuditReport fixture should validate, got: {[e.message for e in errors]}"
    )


def test_citation_audit_report_with_null_notes_validates():
    schema = yaml.safe_load(CITATION_AUDIT_REPORT_SCHEMA_PATH.read_text())
    validator = Draft202012Validator(schema)

    fixture = {
        "unsupported_claims": [],
        "checked_count": 0,
        "notes": None,
    }
    errors = list(validator.iter_errors(fixture))
    assert errors == [], (
        f"CitationAuditReport with notes=null should validate, got: "
        f"{[e.message for e in errors]}"
    )


def test_citation_audit_report_missing_checked_count_fails():
    schema = yaml.safe_load(CITATION_AUDIT_REPORT_SCHEMA_PATH.read_text())
    validator = Draft202012Validator(schema)

    fixture = {
        "unsupported_claims": [],
    }
    errors = list(validator.iter_errors(fixture))
    assert errors, "CitationAuditReport missing `checked_count` should NOT validate"


def test_citation_audit_report_extra_field_fails():
    schema = yaml.safe_load(CITATION_AUDIT_REPORT_SCHEMA_PATH.read_text())
    validator = Draft202012Validator(schema)

    fixture = {
        "unsupported_claims": [],
        "checked_count": 3,
        "unexpected_field": "nope",
    }
    errors = list(validator.iter_errors(fixture))
    assert errors, "CitationAuditReport with an extra property should NOT validate"


# ---------------------------------------------------------------------------
# SpecStore: evidence_audit_report, citation_audit_report kinds
# ---------------------------------------------------------------------------


def test_store_accepts_evidence_audit_report_for_evidence_audit_agent(tmp_path):
    store = SpecStore(tmp_path / "store")
    try:
        store.put(
            kind="evidence_audit_report",
            session_id="s1",
            key="evidence-audit-1",
            obj={
                "claim_citation_matrix": [],
                "missing_papers": [],
                "expired_citations": [],
                "evidence_strength": "strong",
            },
            emitter="EvidenceAuditAgent",
        )
    finally:
        store.close()


def test_store_rejects_wrong_emitter_for_evidence_audit_report(tmp_path):
    store = SpecStore(tmp_path / "store")
    try:
        with pytest.raises(EmitterNotAllowed):
            store.put(
                kind="evidence_audit_report",
                session_id="s1",
                key="evidence-audit-1",
                obj={},
                emitter="ResearchOrchestrator",
            )
    finally:
        store.close()


def test_store_accepts_citation_audit_report_for_evidence_audit_agent(tmp_path):
    store = SpecStore(tmp_path / "store")
    try:
        store.put(
            kind="citation_audit_report",
            session_id="s1",
            key="citation-audit-1",
            obj={
                "unsupported_claims": [],
                "checked_count": 0,
            },
            emitter="EvidenceAuditAgent",
        )
    finally:
        store.close()


def test_store_rejects_wrong_emitter_for_citation_audit_report(tmp_path):
    store = SpecStore(tmp_path / "store")
    try:
        with pytest.raises(EmitterNotAllowed):
            store.put(
                kind="citation_audit_report",
                session_id="s1",
                key="citation-audit-1",
                obj={},
                emitter="ResearchOrchestrator",
            )
    finally:
        store.close()


# ---------------------------------------------------------------------------
# EvidenceAuditAgent as a NEW legal agent.failed / envelope emitter.
# ---------------------------------------------------------------------------


def test_agent_failed_includes_evidence_audit_agent(registry: Registry):
    entry = registry.get("agent.failed")
    assert "EvidenceAuditAgent" in entry.emitters


def test_envelope_schema_allows_evidence_audit_agent_emitter(registry: Registry):
    validator = registry.envelope_validator
    emitter_enum = validator.schema["properties"]["emitter"]["enum"]
    assert "EvidenceAuditAgent" in emitter_enum


def test_agent_failed_payload_schema_allows_evidence_audit_agent():
    schema_path = EVENTBUS_ROOT / "schemas" / "agent.failed.v1.schema.json"
    schema = json.loads(schema_path.read_text())
    agent_name_enum = schema["properties"]["agent_name"]["enum"]
    assert "EvidenceAuditAgent" in agent_name_enum


# ---------------------------------------------------------------------------
# SessionState: evidence_audit_report_ref, citation_audit_report_ref
# ---------------------------------------------------------------------------


def test_session_state_has_evidence_audit_refs_defaulting_to_none():
    state = SessionState(session_id="s1")
    assert state.evidence_audit_report_ref is None
    assert state.citation_audit_report_ref is None
