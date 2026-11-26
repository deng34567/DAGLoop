"""
Tests for ProblemModelingAgent's EventBus handler.
ProblemModelingAgent EventBus handler 的测试。

FakeBusClient (captures publishes; mirrors the real EventBusClient's
publish/subscribe signature — same shape as the M0 FakeBusClient in
autoresearch/orchestrator/tests/unit/test_engine_min.py) + FakeLLM (canned
JSON queue) + a real tmp SpecStore. The FakeBusClient does NOT auto-deliver
events to subscribed handlers — each test calls the handler's event methods
directly with a constructed envelope (mirrors vibe/agents/doc_agent's tests
calling handler.handle(...) directly).
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from typing import Any, Optional

import pytest
import yaml
from jsonschema import Draft202012Validator

from autoresearch.agents.problem_modeling_agent import core
from autoresearch.agents.problem_modeling_agent.handler import ProblemModelingAgentHandler
from autoresearch.agents.problem_modeling_agent.tests.conftest import FakeLLM
from autoresearch.spec_store import SpecStore
from autoresearch.spec_store import refs as spec_refs

SCHEMA_PATH = (
    __import__("pathlib").Path(__file__).resolve().parents[1]
    / "data-contracts"
    / "research_spec.schema.yaml"
)


def _spec_validator() -> Draft202012Validator:
    schema = yaml.safe_load(SCHEMA_PATH.read_text(encoding="utf-8"))
    return Draft202012Validator(schema)


VALID_SPEC = {
    "topic": "LLM coding agent evaluation",
    "research_goal": "Benchmark autonomous coding agents on real-world repos.",
    "task_mode": "benchmark",
    "budget": "two weeks",
    "deliverable_form": "benchmark report",
    "editable_scope": ["benchmark harness"],
    "forbidden_items": ["production deploy"],
    "open_questions": [],
}


def _ask_json(question_id: str = "q-001") -> str:
    return json.dumps(
        {
            "action": "ask",
            "question": {
                "question_id": question_id,
                "question_text": "What is the primary deliverable form you expect?",
                "options": ["paper draft", "code repo", "benchmark report"],
                "allow_free_text": True,
            },
        }
    )


def _finalize_json(spec: dict) -> str:
    return json.dumps({"action": "finalize", "spec": spec})


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


@dataclass
class _Published:
    event_type: str
    session_id: str
    payload: dict
    causation_id: Optional[str]


class FakeBusClient:
    """Captures publishes; mirrors the real EventBusClient.publish/subscribe
    signature. Does NOT auto-deliver events to subscribed handlers."""

    def __init__(self) -> None:
        self.emitter_name = "ProblemModelingAgent"
        self.published: list[_Published] = []
        self.subscriptions: list[tuple[str, Any]] = []

    def subscribe(self, pattern: str, handler: Any) -> None:
        self.subscriptions.append((pattern, handler))

    async def publish(
        self,
        *,
        event_type: str,
        session_id: str,
        payload: dict,
        cost_delta: str = "0",
        causation_id: Optional[str] = None,
        auto_commit: bool = True,
    ) -> str:
        self.published.append(
            _Published(
                event_type=event_type,
                session_id=session_id,
                payload=payload,
                causation_id=causation_id,
            )
        )
        return f"evt-{len(self.published)}"


@dataclass
class _Envelope:
    """The subset of the real broker Envelope the handler reads."""

    event_id: str
    session_id: str
    event_type: str
    emitter: str
    payload: dict


def _make_envelope(event_type: str, session_id: str, payload: dict, emitter: str) -> _Envelope:
    return _Envelope(
        event_id=str(uuid.uuid4()),
        session_id=session_id,
        event_type=event_type,
        emitter=emitter,
        payload=payload,
    )


@pytest.fixture()
def store(tmp_path):
    s = SpecStore(tmp_path / "store")
    yield s
    s.close()


# ---------------------------------------------------------------------------
# (a) problem_modeling.requested -> clarification_question + draft persisted
# ---------------------------------------------------------------------------


async def test_requested_publishes_question_and_persists_draft(store):
    llm = FakeLLM([_ask_json()])
    handler = ProblemModelingAgentHandler(spec_store=store, llm_client=llm)
    client = FakeBusClient()
    handler.register(client)

    session_id = str(uuid.uuid4())
    env = _make_envelope(
        "problem_modeling.requested",
        session_id,
        {"research_goal_text": "I want to benchmark coding agents."},
        "ResearchOrchestrator",
    )

    await handler.handle_requested(client, env)

    questions = [
        p for p in client.published if p.event_type == "problem_modeling.clarification_question"
    ]
    assert len(questions) == 1
    q = questions[0]
    assert q.session_id == session_id
    assert q.causation_id == env.event_id
    assert q.payload["question_id"] == "q-001"
    assert q.payload["allow_free_text"] is True

    draft_ref = spec_refs.encode("research_spec_draft", session_id, "v1")
    draft = store.get(draft_ref)
    assert draft["research_goal_text"] == "I want to benchmark coding agents."
    assert draft["answered"] == []
    assert draft["draft_spec"]["topic"] is None
    assert draft["pending_question"] == q.payload


async def test_restart_request_preserves_original_goal_and_working_state(store):
    llm = FakeLLM([_ask_json(), _ask_json()])
    handler = ProblemModelingAgentHandler(spec_store=store, llm_client=llm)
    client = FakeBusClient()
    session_id = str(uuid.uuid4())

    await handler.handle_requested(client, _make_envelope(
        "problem_modeling.requested",
        session_id,
        {"research_goal_text": "Improve the specified repository benchmark."},
        "ResearchOrchestrator",
    ))
    draft_ref = spec_refs.encode("research_spec_draft", session_id, "v1")
    before = store.get(draft_ref)

    await handler.handle_requested(client, _make_envelope(
        "problem_modeling.requested",
        session_id,
        {"research_goal_text": "(restarted)"},
        "ResearchOrchestrator",
    ))

    after = store.get(draft_ref)
    assert after == before
    assert after["research_goal_text"] == "Improve the specified repository benchmark."


# ---------------------------------------------------------------------------
# (b) answer sequence -> ask once, then finalize -> research_spec.ready
# ---------------------------------------------------------------------------


async def test_answer_sequence_finalizes_to_valid_research_spec(store):
    llm = FakeLLM([_ask_json(), _finalize_json(VALID_SPEC)])
    handler = ProblemModelingAgentHandler(spec_store=store, llm_client=llm)
    client = FakeBusClient()
    handler.register(client)

    session_id = str(uuid.uuid4())
    req_env = _make_envelope(
        "problem_modeling.requested",
        session_id,
        {"research_goal_text": "I want to benchmark coding agents."},
        "ResearchOrchestrator",
    )
    await handler.handle_requested(client, req_env)

    answer_env = _make_envelope(
        "human.clarification_answer",
        session_id,
        {"question_id": "q-001", "answer_text": "benchmark report"},
        "Human",
    )
    await handler.handle_answer(client, answer_env)

    ready = [p for p in client.published if p.event_type == "research_spec.ready"]
    assert len(ready) == 1
    assert ready[0].causation_id == answer_env.event_id
    ref = ready[0].payload["research_spec_ref"]
    spec = store.get(ref)
    assert spec == VALID_SPEC

    validator = _spec_validator()
    errors = list(validator.iter_errors(spec))
    assert errors == [], [e.message for e in errors]

    # working state was updated with the answer before finalizing
    draft_ref = spec_refs.encode("research_spec_draft", session_id, "v1")
    draft = store.get(draft_ref)
    assert draft["answered"] == [{"question_id": "q-001", "answer_text": "benchmark report"}]


# ---------------------------------------------------------------------------
# (c) provenance: answer payload carries ONLY {question_id, answer_text} —
# draft_spec / research_goal_text must come from the STORE, not the payload.
# ---------------------------------------------------------------------------


async def test_answer_reads_working_state_from_store_not_payload(store):
    llm = FakeLLM([_ask_json(), _finalize_json(VALID_SPEC)])
    handler = ProblemModelingAgentHandler(spec_store=store, llm_client=llm)
    client = FakeBusClient()
    handler.register(client)

    session_id = str(uuid.uuid4())
    req_env = _make_envelope(
        "problem_modeling.requested",
        session_id,
        {"research_goal_text": "I want to benchmark coding agents."},
        "ResearchOrchestrator",
    )
    await handler.handle_requested(client, req_env)

    # The answer payload has ONLY question_id/answer_text — no goal text, no
    # draft — exactly matching the human.clarification_answer schema.
    answer_env = _make_envelope(
        "human.clarification_answer",
        session_id,
        {"question_id": "q-001", "answer_text": "benchmark report"},
        "Human",
    )
    await handler.handle_answer(client, answer_env)

    # clarify_step's second call must have been built from the research goal
    # text that was only ever recorded in the store on the FIRST turn — never
    # present in the answer event's payload.
    second_call_messages = llm.calls[1]
    user_message = next(m["content"] for m in second_call_messages if m["role"] == "user")
    assert "I want to benchmark coding agents." in user_message

    ready = [p for p in client.published if p.event_type == "research_spec.ready"]
    assert len(ready) == 1


# ---------------------------------------------------------------------------
# (d) core error (malformed JSON) -> agent.failed, no crash
# ---------------------------------------------------------------------------


async def test_core_error_publishes_agent_failed(store):
    llm = FakeLLM(["not json at all"] * (core.MAX_RETRIES + 1))
    handler = ProblemModelingAgentHandler(spec_store=store, llm_client=llm)
    client = FakeBusClient()
    handler.register(client)

    session_id = str(uuid.uuid4())
    env = _make_envelope(
        "problem_modeling.requested",
        session_id,
        {"research_goal_text": "I want to benchmark coding agents."},
        "ResearchOrchestrator",
    )

    await handler.handle_requested(client, env)

    failed = [p for p in client.published if p.event_type == "agent.failed"]
    assert len(failed) == 1
    assert failed[0].session_id == session_id
    assert failed[0].causation_id == env.event_id
    assert failed[0].payload["agent_name"] == "ProblemModelingAgent"
    assert failed[0].payload["reason_classification"] == "permanent"
    assert failed[0].payload["underlying_reason"]

    assert not [
        p for p in client.published if p.event_type == "problem_modeling.clarification_question"
    ]
    assert not [p for p in client.published if p.event_type == "research_spec.ready"]
