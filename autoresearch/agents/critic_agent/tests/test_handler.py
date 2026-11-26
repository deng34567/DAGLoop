"""
Tests for CriticAgent's EventBus handler.
CriticAgent EventBus handler 的测试。

FakeBusClient (captures publishes; mirrors the real EventBusClient's
publish/subscribe signature — same shape as the M4 FakeBusClient in
novelty_audit_agent/tests/test_handler.py) + FakeLLM (from this package's
tests/conftest.py) + a real tmp SpecStore. The FakeBusClient does NOT
auto-deliver events to subscribed handlers — each test calls the handler's
event method directly with a constructed Envelope.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import pytest
import yaml
from jsonschema import Draft202012Validator

from autoresearch.agents.critic_agent import core
from autoresearch.agents.critic_agent.handler import CriticAgentHandler
from autoresearch.agents.critic_agent.tests.conftest import FakeLLM
from autoresearch.spec_store import SpecStore

_CONTRACTS_DIR = Path(__file__).resolve().parents[1] / "data-contracts"


def _validator(name: str) -> Draft202012Validator:
    schema = yaml.safe_load((_CONTRACTS_DIR / name).read_text(encoding="utf-8"))
    return Draft202012Validator(schema)


CRITIQUE_REPORT_VALIDATOR = _validator("critique_report.schema.yaml")

VALID_RESEARCH_SPEC = {
    "topic": "Retrieval-augmented generation for long-tail scientific QA",
    "research_goal": "Propose a new RAG method for long-tail scientific question answering.",
}

VALID_PROBLEM_MODEL = {
    "key_problems": ["Long-tail entities are poorly retrieved by dense retrievers."],
    "opportunities": ["Hybrid sparse+dense retrieval may close the gap."],
    "risks": ["Benchmark may already be saturated."],
    "verifiable_paths": ["Ablate retriever type on a long-tail QA benchmark."],
}

VALID_HYPOTHESIS_BANK = {
    "hypotheses": [
        {
            "id": "h1",
            "statement": "Hybrid sparse+dense retrieval improves long-tail QA accuracy.",
            "kind": "idea",
            "verifiable": True,
        },
        {
            "id": "h2",
            "statement": "Larger context windows alone fix long-tail QA.",
            "kind": "idea",
            "verifiable": True,
        },
    ]
}

VALID_DIRECTION_REPORT_PASS = {
    "target": "direction",
    "verdict": "pass",
    "issues": [],
    "passing_hypothesis_ids": [],
}

VALID_DIRECTION_REPORT_FAIL = {
    "target": "direction",
    "verdict": "fail",
    "issues": ["Scope creep beyond the frozen ResearchSpec."],
    "passing_hypothesis_ids": [],
}

VALID_HYPOTHESES_REPORT_SOME_PASS = {
    "target": "hypotheses",
    "verdict": "some_pass",
    "issues": ["h2 is too narrow / trivially confounded."],
    "passing_hypothesis_ids": ["h1"],
}

VALID_HYPOTHESES_REPORT_ALL_REJECTED = {
    "target": "hypotheses",
    "verdict": "all_rejected",
    "issues": ["h1 is not verifiable.", "h2 is not verifiable."],
    "passing_hypothesis_ids": [],
}


def _report_json(report: dict) -> str:
    return json.dumps(report)


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
        self.emitter_name = "CriticAgent"
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


def _seed_direction_inputs(store: SpecStore, session_id: str) -> dict[str, str]:
    """Seed the store with the ProblemModel + ResearchSpec CriticAgent reads
    for CritiquingDirection, as upstream agents would; return their refs."""
    problem_model_ref = store.put(
        kind="problem_model",
        session_id=session_id,
        key="final",
        obj=VALID_PROBLEM_MODEL,
        emitter="ProblemModelingAgent",
    )
    research_spec_ref = store.put(
        kind="research_spec",
        session_id=session_id,
        key="final",
        obj=VALID_RESEARCH_SPEC,
        emitter="ProblemModelingAgent",
    )
    return {
        "problem_model_ref": problem_model_ref,
        "research_spec_ref": research_spec_ref,
    }


def _seed_hypotheses_inputs(store: SpecStore, session_id: str) -> dict[str, str]:
    """Seed the store with the HypothesisBank + ResearchSpec CriticAgent
    reads for CritiquingHypotheses, as upstream agents would; return their
    refs."""
    hypothesis_bank_ref = store.put(
        kind="hypothesis_bank",
        session_id=session_id,
        key="final",
        obj=VALID_HYPOTHESIS_BANK,
        emitter="HypothesisAgent",
    )
    research_spec_ref = store.put(
        kind="research_spec",
        session_id=session_id,
        key="final",
        obj=VALID_RESEARCH_SPEC,
        emitter="ProblemModelingAgent",
    )
    return {
        "hypothesis_bank_ref": hypothesis_bank_ref,
        "research_spec_ref": research_spec_ref,
    }


# ---------------------------------------------------------------------------
# (a) direction.critique_requested -> direction.critiqued, ref resolves to a
# schema-valid CritiqueReport (target=direction), inputs read from the STORE
# (provenance), never from the event payload (which carries only refs).
# ---------------------------------------------------------------------------


async def test_direction_requested_publishes_critiqued_with_resolvable_ref(store):
    llm = FakeLLM([_report_json(VALID_DIRECTION_REPORT_PASS)])
    handler = CriticAgentHandler(spec_store=store, llm_client=llm)
    client = FakeBusClient()
    handler.register(client)

    session_id = str(uuid.uuid4())
    refs = _seed_direction_inputs(store, session_id)

    env = _make_envelope(
        "direction.critique_requested", session_id, refs, "ResearchOrchestrator"
    )

    await handler.handle_direction_critique(client, env)

    critiqued = [p for p in client.published if p.event_type == "direction.critiqued"]
    assert len(critiqued) == 1
    payload = critiqued[0].payload
    assert set(payload.keys()) == {"verdict", "critique_report_ref"}
    assert payload["verdict"] == "pass"

    report = store.get(payload["critique_report_ref"])
    errors = list(CRITIQUE_REPORT_VALIDATOR.iter_errors(report))
    assert errors == [], [e.message for e in errors]
    assert report["target"] == "direction"
    assert report["verdict"] == "pass"


async def test_direction_requested_reads_inputs_from_store_not_payload(store):
    """The event payload carries ONLY the two refs (per the
    direction.critique_requested schema) — the handler must resolve the full
    objects via the store, never expect them inline in the payload."""
    llm = FakeLLM([_report_json(VALID_DIRECTION_REPORT_PASS)])
    handler = CriticAgentHandler(spec_store=store, llm_client=llm)
    client = FakeBusClient()
    handler.register(client)

    session_id = str(uuid.uuid4())
    refs = _seed_direction_inputs(store, session_id)

    env = _make_envelope(
        "direction.critique_requested", session_id, refs, "ResearchOrchestrator"
    )
    assert set(env.payload.keys()) == {"problem_model_ref", "research_spec_ref"}

    await handler.handle_direction_critique(client, env)

    critiqued = [p for p in client.published if p.event_type == "direction.critiqued"]
    assert len(critiqued) == 1

    [messages] = llm.calls
    user_message = next(m["content"] for m in messages if m["role"] == "user")
    assert VALID_RESEARCH_SPEC["topic"] in user_message
    assert VALID_PROBLEM_MODEL["key_problems"][0] in user_message


@pytest.mark.parametrize(
    "canned_report",
    [VALID_DIRECTION_REPORT_PASS, VALID_DIRECTION_REPORT_FAIL],
    ids=["pass", "fail"],
)
async def test_direction_both_verdicts_forwarded(store, canned_report):
    llm = FakeLLM([_report_json(canned_report)])
    handler = CriticAgentHandler(spec_store=store, llm_client=llm)
    client = FakeBusClient()
    handler.register(client)

    session_id = str(uuid.uuid4())
    refs = _seed_direction_inputs(store, session_id)

    env = _make_envelope(
        "direction.critique_requested", session_id, refs, "ResearchOrchestrator"
    )

    await handler.handle_direction_critique(client, env)

    critiqued = [p for p in client.published if p.event_type == "direction.critiqued"]
    assert len(critiqued) == 1
    assert critiqued[0].payload["verdict"] == canned_report["verdict"]


async def test_direction_causation_id_set(store):
    llm = FakeLLM([_report_json(VALID_DIRECTION_REPORT_PASS)])
    handler = CriticAgentHandler(spec_store=store, llm_client=llm)
    client = FakeBusClient()
    handler.register(client)

    session_id = str(uuid.uuid4())
    refs = _seed_direction_inputs(store, session_id)

    env = _make_envelope(
        "direction.critique_requested", session_id, refs, "ResearchOrchestrator"
    )

    await handler.handle_direction_critique(client, env)

    critiqued = [p for p in client.published if p.event_type == "direction.critiqued"]
    assert len(critiqued) == 1
    assert critiqued[0].causation_id == env.event_id


async def test_direction_core_error_publishes_agent_failed(store):
    llm = FakeLLM(["not json at all"] * (core.MAX_RETRIES + 1))
    handler = CriticAgentHandler(spec_store=store, llm_client=llm)
    client = FakeBusClient()
    handler.register(client)

    session_id = str(uuid.uuid4())
    refs = _seed_direction_inputs(store, session_id)

    env = _make_envelope(
        "direction.critique_requested", session_id, refs, "ResearchOrchestrator"
    )

    await handler.handle_direction_critique(client, env)

    failed = [p for p in client.published if p.event_type == "agent.failed"]
    assert len(failed) == 1
    assert failed[0].session_id == session_id
    assert failed[0].causation_id == env.event_id
    assert failed[0].payload["agent_name"] == "CriticAgent"
    assert failed[0].payload["reason_classification"] == "permanent"
    assert failed[0].payload["underlying_reason"]

    assert not [p for p in client.published if p.event_type == "direction.critiqued"]


# ---------------------------------------------------------------------------
# (b) hypothesis.critique_requested -> hypothesis.critiqued, ref resolves to
# a schema-valid CritiqueReport (target=hypotheses), inputs read from the
# STORE (provenance), never from the event payload.
# ---------------------------------------------------------------------------


async def test_hypotheses_requested_publishes_critiqued_with_resolvable_ref(store):
    llm = FakeLLM([_report_json(VALID_HYPOTHESES_REPORT_SOME_PASS)])
    handler = CriticAgentHandler(spec_store=store, llm_client=llm)
    client = FakeBusClient()
    handler.register(client)

    session_id = str(uuid.uuid4())
    refs = _seed_hypotheses_inputs(store, session_id)

    env = _make_envelope(
        "hypothesis.critique_requested", session_id, refs, "ResearchOrchestrator"
    )

    await handler.handle_hypothesis_critique(client, env)

    critiqued = [p for p in client.published if p.event_type == "hypothesis.critiqued"]
    assert len(critiqued) == 1
    payload = critiqued[0].payload
    assert set(payload.keys()) == {"verdict", "critique_report_ref", "passing_hypothesis_ids"}
    assert payload["verdict"] == "some_pass"
    assert payload["passing_hypothesis_ids"] == ["h1"]

    report = store.get(payload["critique_report_ref"])
    errors = list(CRITIQUE_REPORT_VALIDATOR.iter_errors(report))
    assert errors == [], [e.message for e in errors]
    assert report["target"] == "hypotheses"
    assert report["verdict"] == "some_pass"


async def test_hypotheses_requested_reads_inputs_from_store_not_payload(store):
    llm = FakeLLM([_report_json(VALID_HYPOTHESES_REPORT_SOME_PASS)])
    handler = CriticAgentHandler(spec_store=store, llm_client=llm)
    client = FakeBusClient()
    handler.register(client)

    session_id = str(uuid.uuid4())
    refs = _seed_hypotheses_inputs(store, session_id)

    env = _make_envelope(
        "hypothesis.critique_requested", session_id, refs, "ResearchOrchestrator"
    )
    assert set(env.payload.keys()) == {"hypothesis_bank_ref", "research_spec_ref"}

    await handler.handle_hypothesis_critique(client, env)

    critiqued = [p for p in client.published if p.event_type == "hypothesis.critiqued"]
    assert len(critiqued) == 1

    [messages] = llm.calls
    user_message = next(m["content"] for m in messages if m["role"] == "user")
    assert VALID_RESEARCH_SPEC["topic"] in user_message
    assert VALID_HYPOTHESIS_BANK["hypotheses"][0]["statement"] in user_message


@pytest.mark.parametrize(
    "canned_report",
    [VALID_HYPOTHESES_REPORT_SOME_PASS, VALID_HYPOTHESES_REPORT_ALL_REJECTED],
    ids=["some_pass", "all_rejected"],
)
async def test_hypotheses_both_verdicts_forwarded(store, canned_report):
    llm = FakeLLM([_report_json(canned_report)])
    handler = CriticAgentHandler(spec_store=store, llm_client=llm)
    client = FakeBusClient()
    handler.register(client)

    session_id = str(uuid.uuid4())
    refs = _seed_hypotheses_inputs(store, session_id)

    env = _make_envelope(
        "hypothesis.critique_requested", session_id, refs, "ResearchOrchestrator"
    )

    await handler.handle_hypothesis_critique(client, env)

    critiqued = [p for p in client.published if p.event_type == "hypothesis.critiqued"]
    assert len(critiqued) == 1
    assert critiqued[0].payload["verdict"] == canned_report["verdict"]
    assert critiqued[0].payload["passing_hypothesis_ids"] == canned_report["passing_hypothesis_ids"]


async def test_hypotheses_causation_id_set(store):
    llm = FakeLLM([_report_json(VALID_HYPOTHESES_REPORT_SOME_PASS)])
    handler = CriticAgentHandler(spec_store=store, llm_client=llm)
    client = FakeBusClient()
    handler.register(client)

    session_id = str(uuid.uuid4())
    refs = _seed_hypotheses_inputs(store, session_id)

    env = _make_envelope(
        "hypothesis.critique_requested", session_id, refs, "ResearchOrchestrator"
    )

    await handler.handle_hypothesis_critique(client, env)

    critiqued = [p for p in client.published if p.event_type == "hypothesis.critiqued"]
    assert len(critiqued) == 1
    assert critiqued[0].causation_id == env.event_id


async def test_hypotheses_core_error_publishes_agent_failed(store):
    llm = FakeLLM(["not json at all"] * (core.MAX_RETRIES + 1))
    handler = CriticAgentHandler(spec_store=store, llm_client=llm)
    client = FakeBusClient()
    handler.register(client)

    session_id = str(uuid.uuid4())
    refs = _seed_hypotheses_inputs(store, session_id)

    env = _make_envelope(
        "hypothesis.critique_requested", session_id, refs, "ResearchOrchestrator"
    )

    await handler.handle_hypothesis_critique(client, env)

    failed = [p for p in client.published if p.event_type == "agent.failed"]
    assert len(failed) == 1
    assert failed[0].session_id == session_id
    assert failed[0].causation_id == env.event_id
    assert failed[0].payload["agent_name"] == "CriticAgent"
    assert failed[0].payload["reason_classification"] == "permanent"
    assert failed[0].payload["underlying_reason"]

    assert not [p for p in client.published if p.event_type == "hypothesis.critiqued"]


# ---------------------------------------------------------------------------
# register() wires BOTH subscriptions.
# ---------------------------------------------------------------------------


def test_register_wires_both_subscriptions(store):
    llm = FakeLLM([])
    handler = CriticAgentHandler(spec_store=store, llm_client=llm)
    client = FakeBusClient()
    handler.register(client)

    patterns = [pattern for pattern, _ in client.subscriptions]
    assert "direction.critique_requested" in patterns
    assert "hypothesis.critique_requested" in patterns
