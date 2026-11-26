"""
Tests for MemoryAgent's EventBus handler.
MemoryAgent EventBus handler 的测试。

FakeBusClient (captures publishes; mirrors the real EventBusClient's
publish/subscribe signature -- same shape as experiment_planner_agent's
FakeBusClient) + FakeLLM (from this package's tests/conftest.py) + a real tmp
SpecStore. The FakeBusClient does NOT auto-deliver events to subscribed
handlers -- each test calls the handler's event method directly with a
constructed Envelope.
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

from autoresearch.agents.memory_agent import core
from autoresearch.agents.memory_agent.handler import MemoryAgentHandler
from autoresearch.agents.memory_agent.tests.conftest import FakeLLM
from autoresearch.eventbus.broker.envelope import EnvelopeValidator
from autoresearch.spec_store import SpecStore

_CONTRACTS_DIR = Path(__file__).resolve().parents[1] / "data-contracts"
_EVENTBUS_SCHEMAS_DIR = Path(__file__).resolve().parents[3] / "eventbus" / "schemas"


def _validator(name: str) -> Draft202012Validator:
    schema = yaml.safe_load((_CONTRACTS_DIR / name).read_text(encoding="utf-8"))
    return Draft202012Validator(schema)


RESEARCH_MEMORY_VALIDATOR = _validator("research_memory.schema.yaml")

EVALUATION_REPORT = {
    "verdict": "valid_not_best",
    "metric_value": 0.75,
    "baseline_metric": 0.80,
    "is_improvement": False,
    "suspicious": False,
    "anti_cheat_flags": [],
    "reasoning": "Metric did not improve over baseline.",
}

EVALUATION_REPORT_NEW_BEST = {
    "verdict": "new_best",
    "metric_value": 0.91,
    "baseline_metric": 0.80,
    "is_improvement": True,
    "suspicious": False,
    "anti_cheat_flags": [],
    "reasoning": "Metric improved over baseline.",
}

RUN_RECORD = {
    "experiment_id": "exp-1",
    "succeeded": True,
    "exit_code": 0,
    "metric_value": 0.91,
    "stdout_excerpt": "training complete",
    "stderr_excerpt": None,
    "environment": "python 3.11, cuda 12.1",
    "seed": "42",
    "runtime_seconds": 120.5,
    "artifacts": [],
}

PRIOR_RESEARCH_MEMORY = {
    "best_metric": 0.80,
    "best_result_ref": "research_store://run_record/prior-best",
    "verified_findings": ["Baseline established at 0.80."],
    "negative_results": [],
    "excluded_hypotheses": [],
    "consolidated_insights": ["Larger batch size helps."],
    "divergence_seeds": [],
    "next_steps": [],
}

VERIFICATION_REPORT_VERIFIED = {
    "verdict": "verified",
    "checks": ["re-ran on held-out split"],
    "anti_cheat_flags": [],
    "reasoning": "Reproduced.",
}


def _llm_research_memory(**overrides: object) -> dict:
    memory = {
        "verified_findings": ["New finding: metric improved with technique X."],
        "negative_results": [],
        "excluded_hypotheses": [],
        "consolidated_insights": ["Technique X is promising."],
        "divergence_seeds": ["Try technique Y next."],
        "next_steps": ["Run technique Y."],
    }
    memory.update(overrides)
    return memory


def _json(obj: dict) -> str:
    return json.dumps(obj)


def _consolidate_response(routing_verdict: str = "continue_explore") -> str:
    return _json(
        {
            "research_memory": _llm_research_memory(),
            "routing_verdict": routing_verdict,
        }
    )


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
        self.emitter_name = "MemoryAgent"
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


def _seed_consolidation_inputs(
    store: SpecStore,
    session_id: str,
    *,
    evaluation_report: dict = EVALUATION_REPORT,
    run_record: dict = RUN_RECORD,
    prior_research_memory: dict | None = None,
    verification_report: dict | None = None,
) -> dict:
    """Seed the store with EvaluationReport + RunRecord (as upstream agents
    would), plus optional prior ResearchMemory / VerificationReport; return
    the memory.consolidation_requested payload (refs)."""
    evaluation_report_ref = store.put(
        kind="evaluation_report",
        session_id=session_id,
        key="v1",
        obj=evaluation_report,
        emitter="EvaluatorAgent",
    )
    run_record_ref = store.put(
        kind="run_record",
        session_id=session_id,
        key="v1",
        obj=run_record,
        emitter="RunnerAgent",
    )
    payload: dict[str, Any] = {
        "evaluation_report_ref": evaluation_report_ref,
        "run_record_ref": run_record_ref,
    }
    if prior_research_memory is not None:
        payload["research_memory_ref"] = store.put(
            kind="research_memory",
            session_id=session_id,
            key="v0",
            obj=prior_research_memory,
            emitter="MemoryAgent",
        )
    if verification_report is not None:
        payload["verification_report_ref"] = store.put(
            kind="verification_report",
            session_id=session_id,
            key="v1",
            obj=verification_report,
            emitter="EvaluatorAgent",
        )
    return payload


# ---------------------------------------------------------------------------
# (a) memory.consolidation_requested -> memory.consolidated, ref resolves to
# a schema-valid ResearchMemory.
# ---------------------------------------------------------------------------


async def test_consolidation_requested_publishes_consolidated_with_resolvable_ref(store):
    llm = FakeLLM([_consolidate_response("continue_explore")])
    handler = MemoryAgentHandler(spec_store=store, llm_client=llm)
    client = FakeBusClient()
    handler.register(client)

    session_id = str(uuid.uuid4())
    payload = _seed_consolidation_inputs(store, session_id)

    env = _make_envelope(
        "memory.consolidation_requested", session_id, payload, "ResearchOrchestrator"
    )

    await handler.handle_consolidation_requested(client, env)

    consolidated = [p for p in client.published if p.event_type == "memory.consolidated"]
    assert len(consolidated) == 1
    result_payload = consolidated[0].payload
    assert set(result_payload.keys()) == {"research_memory_ref", "routing_verdict"}
    assert result_payload["routing_verdict"] == "continue_explore"

    memory = store.get(result_payload["research_memory_ref"])
    errors = list(RESEARCH_MEMORY_VALIDATOR.iter_errors(memory))
    assert errors == [], [e.message for e in errors]


async def test_consolidation_requested_with_optional_refs_present(store):
    llm = FakeLLM([_consolidate_response("report")])
    handler = MemoryAgentHandler(spec_store=store, llm_client=llm)
    client = FakeBusClient()
    handler.register(client)

    session_id = str(uuid.uuid4())
    payload = _seed_consolidation_inputs(
        store,
        session_id,
        evaluation_report=EVALUATION_REPORT_NEW_BEST,
        prior_research_memory=PRIOR_RESEARCH_MEMORY,
        verification_report=VERIFICATION_REPORT_VERIFIED,
    )
    assert set(payload.keys()) == {
        "evaluation_report_ref",
        "run_record_ref",
        "research_memory_ref",
        "verification_report_ref",
    }

    env = _make_envelope(
        "memory.consolidation_requested", session_id, payload, "ResearchOrchestrator"
    )

    await handler.handle_consolidation_requested(client, env)

    consolidated = [p for p in client.published if p.event_type == "memory.consolidated"]
    assert len(consolidated) == 1
    memory = store.get(consolidated[0].payload["research_memory_ref"])
    # new_best + verified -> best_metric computed from run_record, per core's guard.
    assert memory["best_metric"] == RUN_RECORD["metric_value"]


# ---------------------------------------------------------------------------
# (b) provenance: prior ResearchMemory / VerificationReport read from the
# store when their refs are given in the payload, and NOT read (None passed
# to core.consolidate) when absent.
# ---------------------------------------------------------------------------


async def test_consolidation_requested_reads_prior_memory_and_verification_when_given(
    store, monkeypatch
):
    captured: dict[str, Any] = {}
    real_consolidate = core.consolidate

    def _spy_consolidate(
        evaluation_report, run_record, prior_research_memory, verification_report, llm_client,
        *, prompt_patch=None,
    ):
        captured["evaluation_report"] = evaluation_report
        captured["run_record"] = run_record
        captured["prior_research_memory"] = prior_research_memory
        captured["verification_report"] = verification_report
        return real_consolidate(
            evaluation_report, run_record, prior_research_memory, verification_report, llm_client,
            prompt_patch=prompt_patch,
        )

    monkeypatch.setattr(core, "consolidate", _spy_consolidate)

    llm = FakeLLM([_consolidate_response()])
    handler = MemoryAgentHandler(spec_store=store, llm_client=llm)
    client = FakeBusClient()
    handler.register(client)

    session_id = str(uuid.uuid4())
    payload = _seed_consolidation_inputs(
        store,
        session_id,
        prior_research_memory=PRIOR_RESEARCH_MEMORY,
        verification_report=VERIFICATION_REPORT_VERIFIED,
    )

    env = _make_envelope(
        "memory.consolidation_requested", session_id, payload, "ResearchOrchestrator"
    )

    await handler.handle_consolidation_requested(client, env)

    assert captured["evaluation_report"] == EVALUATION_REPORT
    assert captured["run_record"] == RUN_RECORD
    assert captured["prior_research_memory"] == PRIOR_RESEARCH_MEMORY
    assert captured["verification_report"] == VERIFICATION_REPORT_VERIFIED


async def test_consolidation_requested_passes_none_when_optional_refs_absent(store, monkeypatch):
    captured: dict[str, Any] = {}
    real_consolidate = core.consolidate

    def _spy_consolidate(
        evaluation_report, run_record, prior_research_memory, verification_report, llm_client,
        *, prompt_patch=None,
    ):
        captured["prior_research_memory"] = prior_research_memory
        captured["verification_report"] = verification_report
        return real_consolidate(
            evaluation_report, run_record, prior_research_memory, verification_report, llm_client,
            prompt_patch=prompt_patch,
        )

    monkeypatch.setattr(core, "consolidate", _spy_consolidate)

    llm = FakeLLM([_consolidate_response()])
    handler = MemoryAgentHandler(spec_store=store, llm_client=llm)
    client = FakeBusClient()
    handler.register(client)

    session_id = str(uuid.uuid4())
    payload = _seed_consolidation_inputs(store, session_id)
    assert "research_memory_ref" not in payload
    assert "verification_report_ref" not in payload

    env = _make_envelope(
        "memory.consolidation_requested", session_id, payload, "ResearchOrchestrator"
    )

    await handler.handle_consolidation_requested(client, env)

    assert captured["prior_research_memory"] is None
    assert captured["verification_report"] is None


async def test_consolidation_requested_treats_null_refs_as_absent(store, monkeypatch):
    """payload may carry explicit null refs (per the schema's nullable optional
    fields) -- these must also be treated as absent."""
    captured: dict[str, Any] = {}
    real_consolidate = core.consolidate

    def _spy_consolidate(
        evaluation_report, run_record, prior_research_memory, verification_report, llm_client,
        *, prompt_patch=None,
    ):
        captured["prior_research_memory"] = prior_research_memory
        captured["verification_report"] = verification_report
        return real_consolidate(
            evaluation_report, run_record, prior_research_memory, verification_report, llm_client,
            prompt_patch=prompt_patch,
        )

    monkeypatch.setattr(core, "consolidate", _spy_consolidate)

    llm = FakeLLM([_consolidate_response()])
    handler = MemoryAgentHandler(spec_store=store, llm_client=llm)
    client = FakeBusClient()
    handler.register(client)

    session_id = str(uuid.uuid4())
    payload = _seed_consolidation_inputs(store, session_id)
    payload["research_memory_ref"] = None
    payload["verification_report_ref"] = None

    env = _make_envelope(
        "memory.consolidation_requested", session_id, payload, "ResearchOrchestrator"
    )

    await handler.handle_consolidation_requested(client, env)

    assert captured["prior_research_memory"] is None
    assert captured["verification_report"] is None


# ---------------------------------------------------------------------------
# (c) core failure (malformed LLM response) -> agent.failed; never a crash.
# ---------------------------------------------------------------------------


async def test_consolidation_requested_core_error_publishes_agent_failed(store):
    llm = FakeLLM(["not json at all"] * (core.MAX_RETRIES + 1))
    handler = MemoryAgentHandler(spec_store=store, llm_client=llm)
    client = FakeBusClient()
    handler.register(client)

    session_id = str(uuid.uuid4())
    payload = _seed_consolidation_inputs(store, session_id)

    env = _make_envelope(
        "memory.consolidation_requested", session_id, payload, "ResearchOrchestrator"
    )

    await handler.handle_consolidation_requested(client, env)

    failed = [p for p in client.published if p.event_type == "agent.failed"]
    assert len(failed) == 1
    assert failed[0].session_id == session_id
    assert failed[0].causation_id == env.event_id
    assert failed[0].payload["agent_name"] == "MemoryAgent"
    assert failed[0].payload["reason_classification"] == "permanent"
    assert failed[0].payload["underlying_reason"]

    assert not [p for p in client.published if p.event_type == "memory.consolidated"]


async def test_consolidation_requested_unexpected_error_publishes_agent_failed(store):
    """A non-core exception (e.g. a broken ref) must also become agent.failed,
    never a crash."""
    llm = FakeLLM([_consolidate_response()])
    handler = MemoryAgentHandler(spec_store=store, llm_client=llm)
    client = FakeBusClient()
    handler.register(client)

    session_id = str(uuid.uuid4())
    payload = {
        "evaluation_report_ref": "evaluation_report://does-not-exist/v1",
        "run_record_ref": "run_record://does-not-exist/v1",
    }

    env = _make_envelope(
        "memory.consolidation_requested", session_id, payload, "ResearchOrchestrator"
    )

    await handler.handle_consolidation_requested(client, env)

    failed = [p for p in client.published if p.event_type == "agent.failed"]
    assert len(failed) == 1
    assert failed[0].payload["agent_name"] == "MemoryAgent"
    assert failed[0].payload["reason_classification"] in ("transient", "permanent")
    assert failed[0].payload["underlying_reason"]


# ---------------------------------------------------------------------------
# (d) causation_id set.
# ---------------------------------------------------------------------------


async def test_consolidation_requested_causation_id_set(store):
    llm = FakeLLM([_consolidate_response()])
    handler = MemoryAgentHandler(spec_store=store, llm_client=llm)
    client = FakeBusClient()
    handler.register(client)

    session_id = str(uuid.uuid4())
    payload = _seed_consolidation_inputs(store, session_id)

    env = _make_envelope(
        "memory.consolidation_requested", session_id, payload, "ResearchOrchestrator"
    )

    await handler.handle_consolidation_requested(client, env)

    consolidated = [p for p in client.published if p.event_type == "memory.consolidated"]
    assert len(consolidated) == 1
    assert consolidated[0].causation_id == env.event_id


# ---------------------------------------------------------------------------
# (e) enum prerequisite: a published envelope with emitter:"MemoryAgent"
# validates against envelope.schema.json.
# ---------------------------------------------------------------------------


def test_memory_agent_emitter_validates_against_envelope_schema():
    schema_path = _EVENTBUS_SCHEMAS_DIR / "envelope.schema.json"
    envelope_dict = {
        "event_id": str(uuid.uuid4()),
        "session_id": str(uuid.uuid4()),
        "event_type": "memory.consolidated",
        "emitter": "MemoryAgent",
        "timestamp": "2026-07-08T00:00:00Z",
        "schema_version": 1,
        "cost_delta": "0",
        "payload": {
            "research_memory_ref": "research_memory://sess-123/v1",
            "routing_verdict": "continue_explore",
        },
    }
    errors = EnvelopeValidator.validate(envelope_dict, schema_path)
    assert errors == [], errors


def test_memory_agent_agent_failed_validates_against_schema():
    _CONTRACTS = json.loads(
        (_EVENTBUS_SCHEMAS_DIR / "agent.failed.v1.schema.json").read_text()
    )
    validator = Draft202012Validator(_CONTRACTS)
    payload = {
        "agent_name": "MemoryAgent",
        "reason_classification": "permanent",
        "underlying_reason": "LLM did not produce a valid response after retries",
    }
    errors = list(validator.iter_errors(payload))
    assert errors == [], [e.message for e in errors]


# ---------------------------------------------------------------------------
# register() wires the subscription.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# MOSS prompt_patch: config_root/agent_config/MemoryAgent/prompt_patch.md gets
# read by self.config.prompt_patch() and injected into the system message.
# ---------------------------------------------------------------------------


async def test_consolidation_requested_prompt_patch_from_config_root_injected_into_system_message(
    store, tmp_path
):
    patch_dir = tmp_path / "agent_config" / "MemoryAgent"
    patch_dir.mkdir(parents=True)
    (patch_dir / "prompt_patch.md").write_text("STEER: prefer X", encoding="utf-8")

    llm = FakeLLM([_consolidate_response("continue_explore")])
    handler = MemoryAgentHandler(spec_store=store, llm_client=llm, config_root=tmp_path)
    client = FakeBusClient()
    handler.register(client)

    session_id = str(uuid.uuid4())
    payload = _seed_consolidation_inputs(store, session_id)

    env = _make_envelope(
        "memory.consolidation_requested", session_id, payload, "ResearchOrchestrator"
    )

    await handler.handle_consolidation_requested(client, env)

    [messages] = llm.calls
    system_message = next(m["content"] for m in messages if m["role"] == "system")
    assert "STEER: prefer X" in system_message
    assert "[MOSS-injected guidance]" in system_message


def test_register_wires_subscription(store):
    llm = FakeLLM([])
    handler = MemoryAgentHandler(spec_store=store, llm_client=llm)
    client = FakeBusClient()
    handler.register(client)

    patterns = [pattern for pattern, _ in client.subscriptions]
    assert "memory.consolidation_requested" in patterns
