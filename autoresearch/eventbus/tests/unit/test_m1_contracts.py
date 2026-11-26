"""Unit tests for the M1 ProblemModelingAgent clarification contract slice.

Covers: problem_modeling.clarification_question, human.clarification_answer,
research_spec.ready (v1), the 'Human' envelope emitter, the ResearchSpec
data-contract schema, and the SpecStore whitelist extension for
research_spec / research_spec_draft.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
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
RESEARCH_SPEC_SCHEMA_PATH = (
    AUTORESEARCH_ROOT
    / "agents"
    / "problem_modeling_agent"
    / "data-contracts"
    / "research_spec.schema.yaml"
)

M1_EVENTS = [
    "problem_modeling.clarification_question",
    "human.clarification_answer",
    "research_spec.ready",
]


@pytest.fixture
def registry() -> Registry:
    return Registry(REGISTRY_PATH)


def test_registry_loads_m1_validators(registry: Registry):
    for event_type in M1_EVENTS:
        assert registry.get_payload_validator(event_type, 1)


@pytest.mark.parametrize("event_type", M1_EVENTS)
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


def test_clarification_question_registry_entry(registry: Registry):
    entry = registry.get("problem_modeling.clarification_question")
    assert entry.direction == "outbound"
    assert entry.emitters == {"ProblemModelingAgent"}
    assert 1 in entry.schema_versions


def test_human_clarification_answer_registry_entry(registry: Registry):
    entry = registry.get("human.clarification_answer")
    assert entry.direction == "inbound"
    assert entry.emitters == {"Human"}
    assert 1 in entry.schema_versions


def test_research_spec_ready_registry_entry(registry: Registry):
    entry = registry.get("research_spec.ready")
    assert entry.direction == "outbound"
    assert entry.emitters == {"ProblemModelingAgent"}
    assert 1 in entry.schema_versions


def test_human_is_legal_envelope_emitter(registry: Registry):
    envelope = {
        "event_id": str(uuid.uuid4()),
        "session_id": str(uuid.uuid4()),
        "event_type": "human.clarification_answer",
        "emitter": "Human",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "schema_version": 1,
        "cost_delta": "0",
        "payload": {"question_id": "q-001", "answer_text": "benchmark report"},
    }
    errors = list(registry.envelope_validator.iter_errors(envelope))
    assert errors == [], f"envelope with emitter=Human should validate, got: {errors}"


def test_research_spec_fixture_validates_against_schema():
    schema = yaml.safe_load(RESEARCH_SPEC_SCHEMA_PATH.read_text())
    validator = Draft202012Validator(schema)

    fixture = {
        "topic": "Retrieval-augmented fine-tuning for long-tail scientific QA",
        "research_goal": (
            "Determine whether retrieval-augmented fine-tuning improves factual "
            "accuracy on long-tail scientific QA within a two-week compute budget."
        ),
        "task_mode": "benchmark",
        "budget": "two weeks of A100 compute",
        "deliverable_form": "benchmark report",
        "editable_scope": ["retrieval index", "fine-tuning recipe"],
        "forbidden_items": ["training on the held-out test set"],
        "open_questions": [],
    }
    errors = list(validator.iter_errors(fixture))
    assert errors == [], f"ResearchSpec fixture should validate, got: {[e.message for e in errors]}"

    # Minimal draft (most fields null/empty) must also validate.
    draft = {
        "topic": "TBD",
        "research_goal": "TBD",
        "task_mode": None,
        "budget": None,
        "deliverable_form": None,
        "editable_scope": [],
        "forbidden_items": [],
        "open_questions": ["What is the target deliverable form?"],
    }
    errors = list(validator.iter_errors(draft))
    assert errors == [], f"ResearchSpec draft should validate, got: {[e.message for e in errors]}"


def test_store_accepts_research_spec_for_problem_modeling_agent(tmp_path):
    store = SpecStore(tmp_path / "store")
    try:
        store.put_blob(
            kind="research_spec",
            session_id="s1",
            key="final",
            data=b"{}",
            emitter="ProblemModelingAgent",
            ext="json",
        )
        store.put_blob(
            kind="research_spec_draft",
            session_id="s1",
            key="draft-1",
            data=b"{}",
            emitter="ProblemModelingAgent",
            ext="json",
        )
    finally:
        store.close()


def test_store_rejects_wrong_emitter_for_research_spec(tmp_path):
    store = SpecStore(tmp_path / "store")
    try:
        with pytest.raises(EmitterNotAllowed):
            store.put_blob(
                kind="research_spec",
                session_id="s1",
                key="final",
                data=b"{}",
                emitter="CoderAgent",
                ext="json",
            )
        with pytest.raises(EmitterNotAllowed):
            store.put_blob(
                kind="research_spec_draft",
                session_id="s1",
                key="draft-1",
                data=b"{}",
                emitter="Human",
                ext="json",
            )
    finally:
        store.close()
