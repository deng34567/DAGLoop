"""Unit tests for the M3 KnowledgeIndexAgent contract slice.

Covers: knowledge.ingest_requested, knowledge.retrieve_requested,
knowledge.retrieve_result (all v1), confirms literature.brief_ready (from M2)
is untouched, the IndexEntry data-contract schema, and the SpecStore
whitelist extension for knowledge_index.
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
KNOWLEDGE_INDEX_AGENT_ROOT = AUTORESEARCH_ROOT / "agents" / "knowledge_index_agent"
INDEX_ENTRY_SCHEMA_PATH = (
    KNOWLEDGE_INDEX_AGENT_ROOT / "data-contracts" / "index_entry.schema.yaml"
)

M3_EVENTS = [
    "knowledge.ingest_requested",
    "knowledge.retrieve_requested",
    "knowledge.retrieve_result",
]


@pytest.fixture
def registry() -> Registry:
    return Registry(REGISTRY_PATH)


def test_registry_loads_m3_validators(registry: Registry):
    for event_type in M3_EVENTS:
        assert registry.get_payload_validator(event_type, 1)


@pytest.mark.parametrize("event_type", M3_EVENTS)
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


def test_ingest_requested_registry_entry(registry: Registry):
    entry = registry.get("knowledge.ingest_requested")
    assert entry.direction == "inbound"
    assert entry.emitters == {
        "ProblemModelingAgent",
        "DeepResearchAgent",
        "NoveltyAuditAgent",
        "ResearchOrchestrator",
    }
    assert 1 in entry.schema_versions


def test_retrieve_requested_registry_entry(registry: Registry):
    entry = registry.get("knowledge.retrieve_requested")
    assert entry.direction == "inbound"
    assert entry.emitters == {
        "ProblemModelingAgent",
        "DeepResearchAgent",
        "NoveltyAuditAgent",
        "CriticAgent",
        "EvaluatorAgent",
        "ReportAgent",
        "ResearchOrchestrator",
    }
    assert 1 in entry.schema_versions


def test_retrieve_result_registry_entry(registry: Registry):
    entry = registry.get("knowledge.retrieve_result")
    assert entry.direction == "outbound"
    assert entry.emitters == {"KnowledgeIndexAgent"}
    assert 1 in entry.schema_versions


def test_literature_brief_ready_untouched(registry: Registry):
    """M2's literature.brief_ready must still load, unmodified by M3."""
    entry = registry.get("literature.brief_ready")
    assert entry.direction == "outbound"
    assert entry.emitters == {"DeepResearchAgent"}
    assert 1 in entry.schema_versions
    assert registry.get_payload_validator("literature.brief_ready", 1)


def test_index_entry_fixture_validates_against_schema():
    schema = yaml.safe_load(INDEX_ENTRY_SCHEMA_PATH.read_text())
    validator = Draft202012Validator(schema)

    fixture = {
        "path_or_ref": "literature_brief://sess-1/final",
        "type": "literature_brief",
        "summary": "Retrieval-augmented generation literature landscape.",
        "tags": ["rag", "literature"],
        "source": "DeepResearchAgent",
        "produced_by_state": "DeepResearching",
    }
    errors = list(validator.iter_errors(fixture))
    assert errors == [], f"IndexEntry fixture should validate, got: {[e.message for e in errors]}"

    # tags is the one field that may be omitted (defaults to empty array).
    minimal = {
        "path_or_ref": "evidence://sess-1/e1",
        "type": "evidence",
        "summary": "Docker build failed.",
        "source": "AcceptanceAgent",
        "produced_by_state": "Acceptance",
    }
    errors = list(validator.iter_errors(minimal))
    assert errors == [], f"minimal IndexEntry should validate, got: {[e.message for e in errors]}"


def test_index_entry_invalid_fixture_fails_schema():
    schema = yaml.safe_load(INDEX_ENTRY_SCHEMA_PATH.read_text())
    validator = Draft202012Validator(schema)

    # Missing required "summary".
    missing_required = {
        "path_or_ref": "evidence://sess-1/e1",
        "type": "evidence",
        "source": "AcceptanceAgent",
        "produced_by_state": "Acceptance",
    }
    errors = list(validator.iter_errors(missing_required))
    assert errors, "IndexEntry missing 'summary' should NOT validate"

    # Extra unknown key.
    extra_key = {
        "path_or_ref": "evidence://sess-1/e1",
        "type": "evidence",
        "summary": "Docker build failed.",
        "source": "AcceptanceAgent",
        "produced_by_state": "Acceptance",
        "unexpected_field": "nope",
    }
    errors = list(validator.iter_errors(extra_key))
    assert errors, "IndexEntry with an extra property should NOT validate"


def test_store_accepts_knowledge_index_for_knowledge_index_agent(tmp_path):
    store = SpecStore(tmp_path / "store")
    try:
        store.put(
            kind="knowledge_index",
            session_id="s1",
            key="entry-1",
            obj={
                "path_or_ref": "evidence://sess-1/e1",
                "type": "evidence",
                "summary": "Docker build failed.",
                "tags": [],
                "source": "AcceptanceAgent",
                "produced_by_state": "Acceptance",
            },
            emitter="KnowledgeIndexAgent",
        )
    finally:
        store.close()


def test_store_rejects_wrong_emitter_for_knowledge_index(tmp_path):
    store = SpecStore(tmp_path / "store")
    try:
        with pytest.raises(EmitterNotAllowed):
            store.put(
                kind="knowledge_index",
                session_id="s1",
                key="entry-1",
                obj={},
                emitter="DeepResearchAgent",
            )
    finally:
        store.close()
