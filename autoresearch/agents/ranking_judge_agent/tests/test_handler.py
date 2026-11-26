"""
Tests for RankingJudgeAgent's EventBus handler.
RankingJudgeAgent EventBus handler 的测试。

FakeBusClient (captures publishes; mirrors the real EventBusClient's
publish/subscribe signature -- same shape as evidence_audit_agent/tests/
test_handler.py) + FakeLLM (from this package's tests/conftest.py) + a real
tmp SpecStore. The FakeBusClient does NOT auto-deliver events to subscribed
handlers -- each test calls the handler's event method directly with a
constructed Envelope.

ONE subscription is covered: ranking.requested -> ranking.ready.
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

from autoresearch.agents.ranking_judge_agent.handler import RankingJudgeAgentHandler
from autoresearch.agents.ranking_judge_agent.tests.conftest import FakeLLM
from autoresearch.spec_store import SpecStore

_CONTRACTS_DIR = Path(__file__).resolve().parents[1] / "data-contracts"


def _validator(name: str) -> Draft202012Validator:
    schema = yaml.safe_load((_CONTRACTS_DIR / name).read_text(encoding="utf-8"))
    return Draft202012Validator(schema)


RANKING_REPORT_VALIDATOR = _validator("ranking_report.schema.yaml")

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

VALID_HYPOTHESIS_BANK = {
    "hypotheses": [
        {
            "id": "H1",
            "statement": "Dense retrieval with a reranker improves long-tail QA accuracy.",
            "kind": "idea",
            "verifiable": True,
        },
        {
            "id": "H2",
            "statement": "Sparse retrieval alone is sufficient for long-tail QA.",
            "kind": "idea",
            "verifiable": True,
        },
        {
            "id": "H3",
            "statement": "Hybrid retrieval outperforms both dense-only and sparse-only.",
            "kind": "idea",
            "verifiable": True,
        },
    ]
}

VALID_RANKING_LLM_RESPONSE = {
    "pairwise_results": [
        {"a": "H1", "b": "H2", "winner": "H1"},
        {"a": "H1", "b": "H3", "winner": "H3"},
        {"a": "H2", "b": "H3", "winner": "H3"},
    ],
    "diversity_status": "diverse",
}


def _json(obj: dict) -> str:
    return json.dumps(obj)


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
        self.emitter_name = "RankingJudgeAgent"
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


def _seed_bank(store: SpecStore, session_id: str) -> str:
    return store.put(
        kind="hypothesis_bank",
        session_id=session_id,
        key="final",
        obj=VALID_HYPOTHESIS_BANK,
        emitter="HypothesisAgent",
    )


# ---------------------------------------------------------------------------
# ranking.requested -> ranking.ready (all candidates -- no candidate_ids)
# ---------------------------------------------------------------------------


async def test_ranking_requested_all_candidates_publishes_ready_with_resolvable_ref(store):
    llm = FakeLLM([_json(VALID_RANKING_LLM_RESPONSE)])
    handler = RankingJudgeAgentHandler(spec_store=store, llm_client=llm)
    client = FakeBusClient()
    handler.register(client)

    session_id = str(uuid.uuid4())
    bank_ref = _seed_bank(store, session_id)

    env = _make_envelope(
        "ranking.requested",
        session_id,
        {"candidate_kind": "hypothesis", "source_ref": bank_ref},
        "ResearchOrchestrator",
    )

    await handler.handle_ranking_requested(client, env)

    ready = [p for p in client.published if p.event_type == "ranking.ready"]
    assert len(ready) == 1
    payload = ready[0].payload
    assert set(payload.keys()) == {"ranking_report_ref", "candidate_kind"}
    assert payload["candidate_kind"] == "hypothesis"

    report = store.get(payload["ranking_report_ref"])
    errors = list(RANKING_REPORT_VALIDATOR.iter_errors(report))
    assert errors == [], [e.message for e in errors]
    assert set(report["ranked_shortlist"]) == {"H1", "H2", "H3"}


async def test_ranking_requested_all_candidates_reads_bank_from_store_not_payload(store):
    llm = FakeLLM([_json(VALID_RANKING_LLM_RESPONSE)])
    handler = RankingJudgeAgentHandler(spec_store=store, llm_client=llm)
    client = FakeBusClient()
    handler.register(client)

    session_id = str(uuid.uuid4())
    bank_ref = _seed_bank(store, session_id)

    env = _make_envelope(
        "ranking.requested",
        session_id,
        {"candidate_kind": "hypothesis", "source_ref": bank_ref},
        "ResearchOrchestrator",
    )

    await handler.handle_ranking_requested(client, env)

    [messages] = llm.calls
    user_message = next(m["content"] for m in messages if m["role"] == "user")
    assert "H1" in user_message
    assert "H2" in user_message
    assert "H3" in user_message
    assert VALID_HYPOTHESIS_BANK["hypotheses"][0]["statement"] in user_message


async def test_ranking_requested_causation_id_set(store):
    llm = FakeLLM([_json(VALID_RANKING_LLM_RESPONSE)])
    handler = RankingJudgeAgentHandler(spec_store=store, llm_client=llm)
    client = FakeBusClient()
    handler.register(client)

    session_id = str(uuid.uuid4())
    bank_ref = _seed_bank(store, session_id)

    env = _make_envelope(
        "ranking.requested",
        session_id,
        {"candidate_kind": "hypothesis", "source_ref": bank_ref},
        "ResearchOrchestrator",
    )

    await handler.handle_ranking_requested(client, env)

    ready = [p for p in client.published if p.event_type == "ranking.ready"]
    assert len(ready) == 1
    assert ready[0].causation_id == env.event_id


# ---------------------------------------------------------------------------
# ranking.requested -> ranking.ready (candidate_ids subset)
# ---------------------------------------------------------------------------


async def test_ranking_requested_candidate_ids_subset_covers_only_subset(store):
    subset_response = {
        "pairwise_results": [{"a": "H1", "b": "H3", "winner": "H3"}],
        "diversity_status": "diverse",
    }
    llm = FakeLLM([_json(subset_response)])
    handler = RankingJudgeAgentHandler(spec_store=store, llm_client=llm)
    client = FakeBusClient()
    handler.register(client)

    session_id = str(uuid.uuid4())
    bank_ref = _seed_bank(store, session_id)

    env = _make_envelope(
        "ranking.requested",
        session_id,
        {
            "candidate_kind": "hypothesis",
            "source_ref": bank_ref,
            "candidate_ids": ["H1", "H3"],
        },
        "ResearchOrchestrator",
    )

    await handler.handle_ranking_requested(client, env)

    ready = [p for p in client.published if p.event_type == "ranking.ready"]
    assert len(ready) == 1

    report = store.get(ready[0].payload["ranking_report_ref"])
    errors = list(RANKING_REPORT_VALIDATOR.iter_errors(report))
    assert errors == [], [e.message for e in errors]
    assert set(report["ranked_shortlist"]) == {"H1", "H3"}

    [messages] = llm.calls
    user_message = next(m["content"] for m in messages if m["role"] == "user")
    assert "H2" not in user_message


# ---------------------------------------------------------------------------
# Failure path
# ---------------------------------------------------------------------------


async def test_ranking_requested_infra_error_publishes_agent_failed(store):
    class RaisingLLM:
        def chat(self, **kwargs):
            raise RuntimeError("connection reset")

    handler = RankingJudgeAgentHandler(spec_store=store, llm_client=RaisingLLM())
    client = FakeBusClient()
    handler.register(client)

    session_id = str(uuid.uuid4())
    bank_ref = _seed_bank(store, session_id)

    env = _make_envelope(
        "ranking.requested",
        session_id,
        {"candidate_kind": "hypothesis", "source_ref": bank_ref},
        "ResearchOrchestrator",
    )

    await handler.handle_ranking_requested(client, env)

    failed = [p for p in client.published if p.event_type == "agent.failed"]
    assert len(failed) == 1
    assert failed[0].session_id == session_id
    assert failed[0].causation_id == env.event_id
    assert failed[0].payload["agent_name"] == "RankingJudgeAgent"
    assert failed[0].payload["reason_classification"] == "transient"
    assert failed[0].payload["underlying_reason"]

    assert not [p for p in client.published if p.event_type == "ranking.ready"]


async def test_ranking_requested_prompt_patch_from_config_root_injected(store, tmp_path):
    patch_dir = tmp_path / "agent_config" / "RankingJudgeAgent"
    patch_dir.mkdir(parents=True)
    (patch_dir / "prompt_patch.md").write_text("STEER: prefer diverse candidates", encoding="utf-8")

    llm = FakeLLM([_json(VALID_RANKING_LLM_RESPONSE)])
    handler = RankingJudgeAgentHandler(spec_store=store, llm_client=llm, config_root=tmp_path)
    client = FakeBusClient()
    handler.register(client)

    session_id = str(uuid.uuid4())
    bank_ref = _seed_bank(store, session_id)

    env = _make_envelope(
        "ranking.requested",
        session_id,
        {"candidate_kind": "hypothesis", "source_ref": bank_ref},
        "ResearchOrchestrator",
    )

    await handler.handle_ranking_requested(client, env)

    [messages] = llm.calls
    system_message = next(m["content"] for m in messages if m["role"] == "system")
    assert "STEER: prefer diverse candidates" in system_message
    assert "[MOSS-injected guidance]" in system_message
