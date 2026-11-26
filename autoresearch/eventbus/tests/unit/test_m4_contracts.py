"""Unit tests for the NoveltyAuditAgent contract slice (M4 Task 1).

Covers: novelty.audit_ready, problem_modeling.model_requested (v1), confirms
novelty.audit_requested (from the ReinterpretingSpec slice) is untouched, the
NoveltyAuditReport data-contract schema, and the SpecStore whitelist
extension for novelty_audit_report.
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
NOVELTY_AUDIT_AGENT_ROOT = AUTORESEARCH_ROOT / "agents" / "novelty_audit_agent"
NOVELTY_AUDIT_REPORT_SCHEMA_PATH = (
    NOVELTY_AUDIT_AGENT_ROOT / "data-contracts" / "novelty_audit_report.schema.yaml"
)

M4_EVENTS = [
    "novelty.audit_ready",
    "problem_modeling.model_requested",
]


@pytest.fixture
def registry() -> Registry:
    return Registry(REGISTRY_PATH)


def test_registry_loads_m4_validators(registry: Registry):
    for event_type in M4_EVENTS:
        assert registry.get_payload_validator(event_type, 1)


@pytest.mark.parametrize("event_type", M4_EVENTS)
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


def test_novelty_audit_ready_registry_entry(registry: Registry):
    entry = registry.get("novelty.audit_ready")
    assert entry.direction == "outbound"
    assert entry.emitters == {"NoveltyAuditAgent"}
    assert 1 in entry.schema_versions


def test_problem_modeling_model_requested_registry_entry(registry: Registry):
    entry = registry.get("problem_modeling.model_requested")
    assert entry.direction == "outbound"
    assert entry.emitters == {"ResearchOrchestrator"}
    assert 1 in entry.schema_versions


def test_novelty_audit_ready_invalid_verdict_enum_fails(registry: Registry):
    """The invalid example specifically exercises a bad `verdict` enum value."""
    validator = registry.get_payload_validator("novelty.audit_ready", 1)
    invalid_payload = json.loads(
        (EXAMPLES_DIR / "novelty.audit_ready.invalid.json").read_text()
    )
    assert invalid_payload.get("verdict") not in (
        "pass",
        "evidence_insufficient",
    ), "fixture must exercise an invalid enum value for verdict"
    errors = list(validator.iter_errors(invalid_payload))
    assert errors, "invalid verdict enum value must fail validation"


def test_novelty_audit_requested_untouched(registry: Registry):
    """The ReinterpretingSpec slice's novelty.audit_requested must still load, unmodified."""
    entry = registry.get("novelty.audit_requested")
    assert entry.direction == "outbound"
    assert entry.emitters == {"ResearchOrchestrator"}
    assert 1 in entry.schema_versions
    assert registry.get_payload_validator("novelty.audit_requested", 1)


@pytest.mark.parametrize(
    ("event_type", "payload"),
    [
        ("evidence.audit_requested", {
            "research_spec_ref": "spec://research/s/final",
            "literature_brief_ref": "spec://literature/s/target",
            "spec_interpretation_ref": "spec://interpret/s/final",
            "method_ref": "spec://method/s/v1",
            "targeted_search_id": "target-1",
        }),
        ("evidence.audited", {
            "evidence_audit_report_ref": "spec://evidence/s/target",
            "targeted_search_id": "target-1",
        }),
        ("novelty.audit_requested", {
            "research_spec_ref": "spec://research/s/final",
            "literature_brief_ref": "spec://literature/s/target",
            "spec_interpretation_ref": "spec://interpret/s/final",
            "method_ref": "spec://method/s/v1",
            "targeted_search_id": "target-1",
        }),
        ("novelty.audit_ready", {
            "verdict": "evidence_insufficient",
            "novelty_audit_report_ref": "spec://novelty/s/target",
            "targeted_search_id": "target-1",
        }),
    ],
)
def test_targeted_search_identity_is_valid_across_audit_events(
    registry: Registry, event_type: str, payload: dict,
):
    validator = registry.get_payload_validator(event_type, 1)
    assert list(validator.iter_errors(payload)) == []


def test_novelty_audit_report_fixture_validates_against_schema():
    schema = yaml.safe_load(NOVELTY_AUDIT_REPORT_SCHEMA_PATH.read_text())
    validator = Draft202012Validator(schema)

    fixture = {
        "verdict": "pass",
        "citation_sufficiency": "Sufficient: 12 relevant prior works were surveyed.",
        "stale_citations": ["Smith et al. 2015"],
        "unsupported_claims": [],
        "missing_work": [],
        "novelty_status": "novel",
        "notes": "No overlapping prior work found in the surveyed literature.",
    }
    errors = list(validator.iter_errors(fixture))
    assert errors == [], f"NoveltyAuditReport fixture should validate, got: {[e.message for e in errors]}"

    # Minimal: only required fields, arrays empty.
    minimal = {
        "verdict": "evidence_insufficient",
        "citation_sufficiency": "Insufficient: literature brief lacked recent surveys.",
        "novelty_status": "uncertain",
    }
    errors = list(validator.iter_errors(minimal))
    assert errors == [], f"minimal NoveltyAuditReport should validate, got: {[e.message for e in errors]}"


def test_novelty_audit_report_invalid_fixture_fails_schema():
    schema = yaml.safe_load(NOVELTY_AUDIT_REPORT_SCHEMA_PATH.read_text())
    validator = Draft202012Validator(schema)

    # Bad verdict enum value.
    bad_verdict = {
        "verdict": "not_a_real_value",
        "citation_sufficiency": "TBD",
        "novelty_status": "novel",
    }
    errors = list(validator.iter_errors(bad_verdict))
    assert errors, "NoveltyAuditReport with a bad verdict enum should NOT validate"

    # Bad novelty_status enum value.
    bad_novelty_status = {
        "verdict": "pass",
        "citation_sufficiency": "TBD",
        "novelty_status": "not_a_real_value",
    }
    errors = list(validator.iter_errors(bad_novelty_status))
    assert errors, "NoveltyAuditReport with a bad novelty_status enum should NOT validate"

    # Missing required fields.
    missing_required = {
        "stale_citations": [],
        "unsupported_claims": [],
    }
    errors = list(validator.iter_errors(missing_required))
    assert errors, "NoveltyAuditReport missing required fields should NOT validate"

    # Extra unknown key.
    extra_key = {
        "verdict": "pass",
        "citation_sufficiency": "TBD",
        "novelty_status": "novel",
        "unexpected_field": "nope",
    }
    errors = list(validator.iter_errors(extra_key))
    assert errors, "NoveltyAuditReport with an extra property should NOT validate"


def test_store_accepts_novelty_audit_report_for_novelty_audit_agent(tmp_path):
    store = SpecStore(tmp_path / "store")
    try:
        store.put(
            kind="novelty_audit_report",
            session_id="s1",
            key="final",
            obj={
                "verdict": "pass",
                "citation_sufficiency": "x",
                "stale_citations": [],
                "unsupported_claims": [],
                "missing_work": [],
                "novelty_status": "novel",
            },
            emitter="NoveltyAuditAgent",
        )
    finally:
        store.close()


def test_store_rejects_wrong_emitter_for_novelty_audit_report(tmp_path):
    store = SpecStore(tmp_path / "store")
    try:
        with pytest.raises(EmitterNotAllowed):
            store.put(
                kind="novelty_audit_report",
                session_id="s1",
                key="final",
                obj={},
                emitter="DeepResearchAgent",
            )
    finally:
        store.close()
