"""Unit tests for the M2 DeepResearchAgent contract slice.

Covers: literature.brief_ready, problem_modeling.reinterpret_requested (v1),
confirms deep_research.requested (from M1) is untouched, the LiteratureBrief +
FieldTrajectoryBrief data-contract schemas, and the SpecStore whitelist
extension for literature_brief / field_trajectory_brief.
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
DEEP_RESEARCH_AGENT_ROOT = AUTORESEARCH_ROOT / "agents" / "deep_research_agent"
LITERATURE_BRIEF_SCHEMA_PATH = (
    DEEP_RESEARCH_AGENT_ROOT / "data-contracts" / "literature_brief.schema.yaml"
)
FIELD_TRAJECTORY_BRIEF_SCHEMA_PATH = (
    DEEP_RESEARCH_AGENT_ROOT / "data-contracts" / "field_trajectory_brief.schema.yaml"
)

M2_EVENTS = [
    "literature.brief_ready",
    "problem_modeling.reinterpret_requested",
]


@pytest.fixture
def registry() -> Registry:
    return Registry(REGISTRY_PATH)


def test_registry_loads_m2_validators(registry: Registry):
    for event_type in M2_EVENTS:
        assert registry.get_payload_validator(event_type, 1)


@pytest.mark.parametrize("event_type", M2_EVENTS)
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


def test_literature_brief_ready_registry_entry(registry: Registry):
    entry = registry.get("literature.brief_ready")
    assert entry.direction == "outbound"
    assert entry.emitters == {"DeepResearchAgent"}
    assert 1 in entry.schema_versions


def test_reinterpret_requested_registry_entry(registry: Registry):
    entry = registry.get("problem_modeling.reinterpret_requested")
    assert entry.direction == "outbound"
    assert entry.emitters == {"ResearchOrchestrator"}
    assert 1 in entry.schema_versions


def test_deep_research_requested_untouched(registry: Registry):
    """M1's deep_research.requested must still load, unmodified by M2."""
    entry = registry.get("deep_research.requested")
    assert entry.direction == "outbound"
    assert entry.emitters == {"ResearchOrchestrator"}
    assert 1 in entry.schema_versions
    assert registry.get_payload_validator("deep_research.requested", 1)


def test_deep_research_requested_accepts_store_refs_for_targeted_method_search(
    registry: Registry,
):
    validator = registry.get_payload_validator("deep_research.requested", 1)
    payload = {
        "research_spec_ref": "spec://research_spec/s/final",
        "method_ref": "spec://method/s/v1",
        "novelty_audit_report_ref": "spec://novelty_audit_report/s/v1",
        "targeted_search_id": "target-1",
    }
    assert list(validator.iter_errors(payload)) == []

    ready = {
        "literature_brief_ref": "spec://literature_brief/s/target-1",
        "field_trajectory_brief_ref": "spec://trajectory/s/target-1",
        "method_ref": "spec://method/s/v1",
        "novelty_audit_report_ref": "spec://novelty_audit_report/s/v1",
        "targeted_search_id": "target-1",
    }
    ready_validator = registry.get_payload_validator("literature.brief_ready", 1)
    assert list(ready_validator.iter_errors(ready)) == []


def test_literature_brief_fixture_validates_against_schema():
    schema = yaml.safe_load(LITERATURE_BRIEF_SCHEMA_PATH.read_text())
    validator = Draft202012Validator(schema)

    fixture = {
        "summary": "Retrieval-augmented fine-tuning has been explored across several benchmarks.",
        "related_work": [
            {"title": "RAG for QA", "id_or_url": "arxiv:2005.11401", "year": 2020},
        ],
        "key_papers": [
            {
                "title": "Retrieval-Augmented Generation",
                "id_or_url": "arxiv:2005.11401",
                "year": 2020,
                "note": "Seminal RAG paper.",
            }
        ],
        "existing_benchmarks": [
            {"name": "Natural Questions", "id_or_url": "https://ai.google.com/research/NaturalQuestions"},
        ],
    }
    errors = list(validator.iter_errors(fixture))
    assert errors == [], f"LiteratureBrief fixture should validate, got: {[e.message for e in errors]}"

    # Minimal: only required fields, other arrays empty.
    minimal = {
        "summary": "TBD",
        "related_work": [],
        "key_papers": [
            {"title": "Some Paper", "id_or_url": "arxiv:0000.0000"},
        ],
        "existing_benchmarks": [],
    }
    errors = list(validator.iter_errors(minimal))
    assert errors == [], f"minimal LiteratureBrief should validate, got: {[e.message for e in errors]}"


def test_literature_brief_invalid_fixture_fails_schema():
    schema = yaml.safe_load(LITERATURE_BRIEF_SCHEMA_PATH.read_text())
    validator = Draft202012Validator(schema)

    # Missing required "summary".
    missing_required = {
        "related_work": [],
        "key_papers": [{"title": "Some Paper", "id_or_url": "arxiv:0000.0000"}],
        "existing_benchmarks": [],
    }
    errors = list(validator.iter_errors(missing_required))
    assert errors, "LiteratureBrief missing 'summary' should NOT validate"

    # Extra unknown key.
    extra_key = {
        "summary": "TBD",
        "related_work": [],
        "key_papers": [{"title": "Some Paper", "id_or_url": "arxiv:0000.0000"}],
        "existing_benchmarks": [],
        "unexpected_field": "nope",
    }
    errors = list(validator.iter_errors(extra_key))
    assert errors, "LiteratureBrief with an extra property should NOT validate"


def test_field_trajectory_brief_fixture_validates_against_schema():
    schema = yaml.safe_load(FIELD_TRAJECTORY_BRIEF_SCHEMA_PATH.read_text())
    validator = Draft202012Validator(schema)

    fixture = {
        "origin": "Introduced as an extension of open-domain QA retrieval.",
        "current_state": "Widely adopted in production LLM pipelines.",
        "main_method_families": ["dense retrieval", "hybrid retrieval", "re-ranking"],
        "benchmark_evolution": "Moved from Natural Questions to long-tail scientific QA sets.",
        "open_problems": ["long-tail factual accuracy", "retrieval latency at scale"],
    }
    errors = list(validator.iter_errors(fixture))
    assert errors == [], f"FieldTrajectoryBrief fixture should validate, got: {[e.message for e in errors]}"


def test_field_trajectory_brief_invalid_fixture_fails_schema():
    schema = yaml.safe_load(FIELD_TRAJECTORY_BRIEF_SCHEMA_PATH.read_text())
    validator = Draft202012Validator(schema)

    # Missing required fields.
    missing_required = {
        "main_method_families": [],
        "open_problems": [],
    }
    errors = list(validator.iter_errors(missing_required))
    assert errors, "FieldTrajectoryBrief missing required fields should NOT validate"

    # Extra unknown key.
    extra_key = {
        "origin": "TBD",
        "current_state": "TBD",
        "main_method_families": [],
        "benchmark_evolution": "TBD",
        "open_problems": [],
        "unexpected_field": "nope",
    }
    errors = list(validator.iter_errors(extra_key))
    assert errors, "FieldTrajectoryBrief with an extra property should NOT validate"


def test_store_accepts_deep_research_briefs_for_deep_research_agent(tmp_path):
    store = SpecStore(tmp_path / "store")
    try:
        store.put(
            kind="literature_brief",
            session_id="s1",
            key="final",
            obj={"summary": "x", "related_work": [], "key_papers": [], "existing_benchmarks": []},
            emitter="DeepResearchAgent",
        )
        store.put(
            kind="field_trajectory_brief",
            session_id="s1",
            key="final",
            obj={
                "origin": "x",
                "current_state": "x",
                "main_method_families": [],
                "benchmark_evolution": "x",
                "open_problems": [],
            },
            emitter="DeepResearchAgent",
        )
    finally:
        store.close()


def test_store_rejects_wrong_emitter_for_deep_research_briefs(tmp_path):
    store = SpecStore(tmp_path / "store")
    try:
        with pytest.raises(EmitterNotAllowed):
            store.put(
                kind="literature_brief",
                session_id="s1",
                key="final",
                obj={},
                emitter="ProblemModelingAgent",
            )
        with pytest.raises(EmitterNotAllowed):
            store.put(
                kind="field_trajectory_brief",
                session_id="s1",
                key="final",
                obj={},
                emitter="CoderAgent",
            )
    finally:
        store.close()
