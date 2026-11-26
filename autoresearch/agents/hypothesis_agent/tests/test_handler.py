"""
Tests for HypothesisAgent's EventBus handler.
HypothesisAgent EventBus handler 的测试。

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

from autoresearch.agents.hypothesis_agent import core
from autoresearch.agents.hypothesis_agent.handler import HypothesisAgentHandler
from autoresearch.agents.hypothesis_agent.tests.conftest import FakeLLM
from autoresearch.spec_store import SpecStore

_CONTRACTS_DIR = Path(__file__).resolve().parents[1] / "data-contracts"


def _validator(name: str) -> Draft202012Validator:
    schema = yaml.safe_load((_CONTRACTS_DIR / name).read_text(encoding="utf-8"))
    return Draft202012Validator(schema)


HYPOTHESIS_BANK_VALIDATOR = _validator("hypothesis_bank.schema.yaml")

VALID_RESEARCH_SPEC = {
    "topic": "Retrieval-augmented generation for long-tail scientific QA",
    "research_goal": "Propose a new RAG method for long-tail scientific question answering.",
}

VALID_LITERATURE_BRIEF = {
    "summary": "The field combines dense retrieval with generative models to ground answers.",
    "related_work": [],
    "key_papers": [
        {
            "title": "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
            "id_or_url": "arxiv:2005.11401",
            "year": 2020,
        },
    ],
    "existing_benchmarks": [],
}


def _hypothesis(hid: str, kind: str = "idea") -> dict:
    return {
        "id": hid,
        "statement": f"Statement for {hid}: doing X improves Y by Z%.",
        "rationale": f"Rationale for {hid}, grounded in the literature brief.",
        "kind": kind,
        "verifiable": True,
        "expected_effect": "Metric improves by an estimated 5-10%.",
    }


VALID_HYPOTHESES = [
    _hypothesis("h1", "idea"),
    _hypothesis("h2", "training"),
    _hypothesis("h3", "kernel"),
]


def _bank_json(hypotheses: list[dict]) -> str:
    return json.dumps({"hypotheses": hypotheses})


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
        self.emitter_name = "HypothesisAgent"
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


def _seed_inputs(store: SpecStore, session_id: str) -> dict[str, str]:
    """Seed the store with the two frozen inputs HypothesisAgent reads, as
    upstream agents would; return their refs."""
    research_spec_ref = store.put(
        kind="research_spec",
        session_id=session_id,
        key="final",
        obj=VALID_RESEARCH_SPEC,
        emitter="ProblemModelingAgent",
    )
    literature_brief_ref = store.put(
        kind="literature_brief",
        session_id=session_id,
        key="final",
        obj=VALID_LITERATURE_BRIEF,
        emitter="DeepResearchAgent",
    )
    return {
        "research_spec_ref": research_spec_ref,
        "literature_brief_ref": literature_brief_ref,
    }


# ---------------------------------------------------------------------------
# (a) + (b): hypothesis.generation_requested -> hypothesis.generated, ref
# resolves to a schema-valid HypothesisBank with a matching count; both
# inputs are read from the STORE (provenance), never from the event payload
# (which carries only refs).
# ---------------------------------------------------------------------------


async def test_requested_publishes_generated_with_resolvable_ref(store):
    llm = FakeLLM([_bank_json(VALID_HYPOTHESES)])
    handler = HypothesisAgentHandler(spec_store=store, llm_client=llm)
    client = FakeBusClient()
    handler.register(client)

    session_id = str(uuid.uuid4())
    refs = _seed_inputs(store, session_id)

    env = _make_envelope(
        "hypothesis.generation_requested", session_id, refs, "ResearchOrchestrator"
    )

    await handler.handle_requested(client, env)

    generated = [p for p in client.published if p.event_type == "hypothesis.generated"]
    assert len(generated) == 1
    payload = generated[0].payload
    assert set(payload.keys()) == {"hypothesis_bank_ref", "count"}
    assert payload["count"] == len(VALID_HYPOTHESES)

    bank = store.get(payload["hypothesis_bank_ref"])
    errors = list(HYPOTHESIS_BANK_VALIDATOR.iter_errors(bank))
    assert errors == [], [e.message for e in errors]
    assert bank == {"hypotheses": VALID_HYPOTHESES}


async def test_requested_reads_both_inputs_from_store_not_payload(store):
    """The event payload carries ONLY the two refs (per the
    hypothesis.generation_requested schema) — the handler must resolve the
    full objects via the store, never expect them inline in the payload."""
    llm = FakeLLM([_bank_json(VALID_HYPOTHESES)])
    handler = HypothesisAgentHandler(spec_store=store, llm_client=llm)
    client = FakeBusClient()
    handler.register(client)

    session_id = str(uuid.uuid4())
    refs = _seed_inputs(store, session_id)

    env = _make_envelope(
        "hypothesis.generation_requested", session_id, refs, "ResearchOrchestrator"
    )
    assert set(env.payload.keys()) == {"research_spec_ref", "literature_brief_ref"}

    await handler.handle_requested(client, env)

    generated = [p for p in client.published if p.event_type == "hypothesis.generated"]
    assert len(generated) == 1

    # The user message given to the LLM must reflect content that only
    # exists in the STORE (not in the event payload).
    [messages] = llm.calls
    user_message = next(m["content"] for m in messages if m["role"] == "user")
    assert VALID_RESEARCH_SPEC["topic"] in user_message
    assert VALID_LITERATURE_BRIEF["summary"] in user_message


# ---------------------------------------------------------------------------
# (c) core error (malformed LLM JSON) -> agent.failed, no crash, no
# hypothesis.generated.
# ---------------------------------------------------------------------------


async def test_core_error_publishes_agent_failed(store):
    llm = FakeLLM(["not json at all"] * (core.MAX_RETRIES + 1))
    handler = HypothesisAgentHandler(spec_store=store, llm_client=llm)
    client = FakeBusClient()
    handler.register(client)

    session_id = str(uuid.uuid4())
    refs = _seed_inputs(store, session_id)

    env = _make_envelope(
        "hypothesis.generation_requested", session_id, refs, "ResearchOrchestrator"
    )

    await handler.handle_requested(client, env)

    failed = [p for p in client.published if p.event_type == "agent.failed"]
    assert len(failed) == 1
    assert failed[0].session_id == session_id
    assert failed[0].causation_id == env.event_id
    assert failed[0].payload["agent_name"] == "HypothesisAgent"
    assert failed[0].payload["reason_classification"] == "permanent"
    assert failed[0].payload["underlying_reason"]

    assert not [p for p in client.published if p.event_type == "hypothesis.generated"]


# ---------------------------------------------------------------------------
# (d) causation_id is set on published events
# ---------------------------------------------------------------------------


async def test_causation_id_set_on_generated(store):
    llm = FakeLLM([_bank_json(VALID_HYPOTHESES)])
    handler = HypothesisAgentHandler(spec_store=store, llm_client=llm)
    client = FakeBusClient()
    handler.register(client)

    session_id = str(uuid.uuid4())
    refs = _seed_inputs(store, session_id)

    env = _make_envelope(
        "hypothesis.generation_requested", session_id, refs, "ResearchOrchestrator"
    )

    await handler.handle_requested(client, env)

    generated = [p for p in client.published if p.event_type == "hypothesis.generated"]
    assert len(generated) == 1
    assert generated[0].causation_id == env.event_id


# ---------------------------------------------------------------------------
# (e) MOSS prompt_patch: config_root/agent_config/HypothesisAgent/prompt_patch.md
# gets read by self.config.prompt_patch() and injected into the system message.
# ---------------------------------------------------------------------------


async def test_prompt_patch_from_config_root_injected_into_system_message(store, tmp_path):
    patch_dir = tmp_path / "agent_config" / "HypothesisAgent"
    patch_dir.mkdir(parents=True)
    (patch_dir / "prompt_patch.md").write_text("STEER: prefer X", encoding="utf-8")

    llm = FakeLLM([_bank_json(VALID_HYPOTHESES)])
    handler = HypothesisAgentHandler(spec_store=store, llm_client=llm, config_root=tmp_path)
    client = FakeBusClient()
    handler.register(client)

    session_id = str(uuid.uuid4())
    refs = _seed_inputs(store, session_id)

    env = _make_envelope(
        "hypothesis.generation_requested", session_id, refs, "ResearchOrchestrator"
    )

    await handler.handle_requested(client, env)

    [messages] = llm.calls
    system_message = next(m["content"] for m in messages if m["role"] == "system")
    assert "STEER: prefer X" in system_message
    assert "[MOSS-injected guidance]" in system_message
