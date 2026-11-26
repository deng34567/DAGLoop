"""
Tests for EvaluatorAgent's EventBus handler.
EvaluatorAgent EventBus handler 的测试。

FakeBusClient (captures publishes; mirrors the real EventBusClient's
publish/subscribe signature -- same shape as critic_agent's FakeBusClient) +
FakeLLM (from this package's tests/conftest.py) + a real tmp SpecStore. The
FakeBusClient does NOT auto-deliver events to subscribed handlers -- each
test calls the handler's event method directly with a constructed Envelope.
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

from autoresearch.agents.evaluator_agent import core
from autoresearch.agents.evaluator_agent.handler import EvaluatorAgentHandler
from autoresearch.agents.evaluator_agent.tests.conftest import FakeLLM
from autoresearch.spec_store import SpecStore

_CONTRACTS_DIR = Path(__file__).resolve().parents[1] / "data-contracts"


def _validator(name: str) -> Draft202012Validator:
    schema = yaml.safe_load((_CONTRACTS_DIR / name).read_text(encoding="utf-8"))
    return Draft202012Validator(schema)


BENCHMARK_SPEC_VALIDATOR = _validator("benchmark_spec.schema.yaml")
VERIFIER_SPEC_VALIDATOR = _validator("verifier_spec.schema.yaml")

VALID_RESEARCH_SPEC = {
    "topic": "Retrieval-augmented generation for long-tail scientific QA",
    "research_goal": "Propose a new RAG method for long-tail scientific question answering.",
}

VALID_LITERATURE_BRIEF = {
    "summary": "Dense retrievers underperform on long-tail entities.",
    "key_papers": [
        {"title": "Hybrid Retrieval for Long-Tail QA", "id_or_url": "arxiv:1234.5678"},
    ],
    "existing_benchmarks": [
        {"name": "LongTailQA", "id_or_url": "https://example.org/longtailqa"},
    ],
}

VALID_BENCHMARK_SPEC = {
    "benchmark_id": "longtailqa-v1",
    "name": "LongTailQA",
    "metric": "exact_match",
    "run_command": "python run_benchmark.py --model candidate",
    "baseline": {
        "command": "python run_benchmark.py --model baseline",
        "reported_metric": 0.42,
    },
    "resource_requirements": "1x A100 GPU",
    "forbidden_files": ["benchmark/data/held_out.jsonl"],
}

EVALUATION_SEMANTICS = {
    "comparison_statement": (
        "Under the canonical LongTailQA protocol, compare exact_match while allowing retrieval "
        "and generation method changes."
    ),
    "research_question": "Does the method improve exact match under the canonical protocol?",
    "primary_metric": {"name": "exact_match", "direction": "maximize"},
    "comparison_mode": "repository_defined",
    "comparison_basis": "LongTailQA defines the canonical comparison.",
    "budget_interpretation": "Use the benchmark's documented evaluation budget.",
    "conditions_to_hold_constant": ["dataset and evaluator"],
    "method_freedom": ["retrieval and generation method"],
    "evidence": [{
        "source_type": "existing_benchmark",
        "source_id": "https://example.org/longtailqa",
        "supports": "The retrieved benchmark defines exact match evaluation.",
    }],
    "uncertainties": [],
}


def _valid_verifier_spec(**overrides: object) -> dict:
    spec = {
        "evaluator_hash": "sha256:abc123",
        "metric": "exact_match",
        "data_split": {"in_loop": "dev", "held_out": "test"},
        "run_budget": "10 runs, 1h wall-clock",
        "seed_policy": "fixed seed=42 across all runs",
        "editable_region": ["src/model.py", "src/train.py"],
        "forbidden_files": ["benchmark/data/held_out.jsonl"],
        "result_parse_schema": {"type": "object"},
    }
    spec.update(overrides)
    return spec


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
        self.emitter_name = "EvaluatorAgent"
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


def _seed_selection_inputs(store: SpecStore, session_id: str) -> dict[str, str]:
    """Seed the store with the ResearchSpec + LiteratureBrief EvaluatorAgent
    reads for SelectingBenchmark, as upstream agents would; return their
    refs."""
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
    spec_interpretation_ref = store.put(
        kind="spec_interpretation",
        session_id=session_id,
        key="v1",
        obj={
            "disposition": "in_scope",
            "real_open_problem": "Long-tail QA evaluation.",
            "scope_reason": "A canonical benchmark exists.",
            "evaluation_semantics": EVALUATION_SEMANTICS,
        },
        emitter="ProblemModelingAgent",
    )
    return {
        "research_spec_ref": research_spec_ref,
        "literature_brief_ref": literature_brief_ref,
        "spec_interpretation_ref": spec_interpretation_ref,
    }


def _seed_preparation_inputs(store: SpecStore, session_id: str) -> dict[str, str]:
    """Seed the store with the BenchmarkSpec + ResearchSpec EvaluatorAgent
    reads for PreparingVerifier, as an earlier handler run + upstream agent
    would; return their refs."""
    benchmark_spec_ref = store.put(
        kind="benchmark_spec",
        session_id=session_id,
        key="v1",
        obj=VALID_BENCHMARK_SPEC,
        emitter="EvaluatorAgent",
    )
    research_spec_ref = store.put(
        kind="research_spec",
        session_id=session_id,
        key="final",
        obj=VALID_RESEARCH_SPEC,
        emitter="ProblemModelingAgent",
    )
    return {
        "benchmark_spec_ref": benchmark_spec_ref,
        "research_spec_ref": research_spec_ref,
    }


# ---------------------------------------------------------------------------
# (a) benchmark.selection_requested -> benchmark.spec_ready, ref resolves to
# a schema-valid BenchmarkSpec, inputs read from the STORE (provenance),
# never from the event payload (which carries only refs).
# ---------------------------------------------------------------------------


async def test_selection_requested_publishes_spec_ready_with_resolvable_ref(store):
    llm = FakeLLM([_json(VALID_BENCHMARK_SPEC)])
    handler = EvaluatorAgentHandler(spec_store=store, llm_client=llm)
    client = FakeBusClient()
    handler.register(client)

    session_id = str(uuid.uuid4())
    refs = _seed_selection_inputs(store, session_id)

    env = _make_envelope(
        "benchmark.selection_requested", session_id, refs, "ResearchOrchestrator"
    )

    await handler.handle_benchmark_selection(client, env)

    ready = [p for p in client.published if p.event_type == "benchmark.spec_ready"]
    assert len(ready) == 1
    payload = ready[0].payload
    assert set(payload.keys()) == {"benchmark_spec_ref", "benchmark_id"}
    assert payload["benchmark_id"] == VALID_BENCHMARK_SPEC["benchmark_id"]

    spec = store.get(payload["benchmark_spec_ref"])
    errors = list(BENCHMARK_SPEC_VALIDATOR.iter_errors(spec))
    assert errors == [], [e.message for e in errors]
    assert spec["benchmark_id"] == VALID_BENCHMARK_SPEC["benchmark_id"]
    assert spec["evaluation_semantics"] == EVALUATION_SEMANTICS


async def test_selection_requested_reads_inputs_from_store_not_payload(store):
    """The event payload carries ONLY the two refs (per the
    benchmark.selection_requested schema) -- the handler must resolve the
    full objects via the store, never expect them inline in the payload."""
    llm = FakeLLM([_json(VALID_BENCHMARK_SPEC)])
    handler = EvaluatorAgentHandler(spec_store=store, llm_client=llm)
    client = FakeBusClient()
    handler.register(client)

    session_id = str(uuid.uuid4())
    refs = _seed_selection_inputs(store, session_id)

    env = _make_envelope(
        "benchmark.selection_requested", session_id, refs, "ResearchOrchestrator"
    )
    assert set(env.payload.keys()) == {
        "research_spec_ref", "literature_brief_ref", "spec_interpretation_ref",
    }

    await handler.handle_benchmark_selection(client, env)

    ready = [p for p in client.published if p.event_type == "benchmark.spec_ready"]
    assert len(ready) == 1

    [messages] = llm.calls
    user_message = next(m["content"] for m in messages if m["role"] == "user")
    assert VALID_RESEARCH_SPEC["topic"] in user_message
    assert VALID_LITERATURE_BRIEF["summary"] in user_message
    assert EVALUATION_SEMANTICS["research_question"] in user_message


async def test_selection_causation_id_set(store):
    llm = FakeLLM([_json(VALID_BENCHMARK_SPEC)])
    handler = EvaluatorAgentHandler(spec_store=store, llm_client=llm)
    client = FakeBusClient()
    handler.register(client)

    session_id = str(uuid.uuid4())
    refs = _seed_selection_inputs(store, session_id)

    env = _make_envelope(
        "benchmark.selection_requested", session_id, refs, "ResearchOrchestrator"
    )

    await handler.handle_benchmark_selection(client, env)

    ready = [p for p in client.published if p.event_type == "benchmark.spec_ready"]
    assert len(ready) == 1
    assert ready[0].causation_id == env.event_id


async def test_selection_core_error_publishes_agent_failed(store):
    llm = FakeLLM(["not json at all"] * (core.MAX_RETRIES + 1))
    handler = EvaluatorAgentHandler(spec_store=store, llm_client=llm)
    client = FakeBusClient()
    handler.register(client)

    session_id = str(uuid.uuid4())
    refs = _seed_selection_inputs(store, session_id)

    env = _make_envelope(
        "benchmark.selection_requested", session_id, refs, "ResearchOrchestrator"
    )

    await handler.handle_benchmark_selection(client, env)

    failed = [p for p in client.published if p.event_type == "agent.failed"]
    assert len(failed) == 1
    assert failed[0].session_id == session_id
    assert failed[0].causation_id == env.event_id
    assert failed[0].payload["agent_name"] == "EvaluatorAgent"
    assert failed[0].payload["reason_classification"] == "permanent"
    assert failed[0].payload["underlying_reason"]

    assert not [p for p in client.published if p.event_type == "benchmark.spec_ready"]


# ---------------------------------------------------------------------------
# (b) verifier.preparation_requested -> verifier.spec_frozen, ref resolves to
# a schema-valid VerifierSpec, inputs read from the STORE (provenance), never
# from the event payload.
# ---------------------------------------------------------------------------


async def test_preparation_requested_publishes_spec_frozen_with_resolvable_ref(store):
    good_response = {"verifier_spec": _valid_verifier_spec(), "verdict": "ok"}
    llm = FakeLLM([_json(good_response)])
    handler = EvaluatorAgentHandler(spec_store=store, llm_client=llm)
    client = FakeBusClient()
    handler.register(client)

    session_id = str(uuid.uuid4())
    refs = _seed_preparation_inputs(store, session_id)

    env = _make_envelope(
        "verifier.preparation_requested", session_id, refs, "ResearchOrchestrator"
    )

    await handler.handle_verifier_preparation(client, env)

    frozen = [p for p in client.published if p.event_type == "verifier.spec_frozen"]
    assert len(frozen) == 1
    payload = frozen[0].payload
    assert set(payload.keys()) == {"verifier_spec_ref", "verdict"}
    assert payload["verdict"] == "ok"

    spec = store.get(payload["verifier_spec_ref"])
    errors = list(VERIFIER_SPEC_VALIDATOR.iter_errors(spec))
    assert errors == [], [e.message for e in errors]


async def test_preparation_requested_reads_inputs_from_store_not_payload(store):
    good_response = {"verifier_spec": _valid_verifier_spec(), "verdict": "ok"}
    llm = FakeLLM([_json(good_response)])
    handler = EvaluatorAgentHandler(spec_store=store, llm_client=llm)
    client = FakeBusClient()
    handler.register(client)

    session_id = str(uuid.uuid4())
    refs = _seed_preparation_inputs(store, session_id)

    env = _make_envelope(
        "verifier.preparation_requested", session_id, refs, "ResearchOrchestrator"
    )
    assert set(env.payload.keys()) == {"benchmark_spec_ref", "research_spec_ref"}

    await handler.handle_verifier_preparation(client, env)

    frozen = [p for p in client.published if p.event_type == "verifier.spec_frozen"]
    assert len(frozen) == 1

    [messages] = llm.calls
    user_message = next(m["content"] for m in messages if m["role"] == "user")
    assert VALID_RESEARCH_SPEC["topic"] in user_message
    assert VALID_BENCHMARK_SPEC["benchmark_id"] in user_message


@pytest.mark.parametrize("verdict", ["ok", "incomplete", "invalid"])
async def test_preparation_verdicts_forwarded(store, verdict):
    response = {"verifier_spec": _valid_verifier_spec(), "verdict": verdict}
    llm = FakeLLM([_json(response)])
    handler = EvaluatorAgentHandler(spec_store=store, llm_client=llm)
    client = FakeBusClient()
    handler.register(client)

    session_id = str(uuid.uuid4())
    refs = _seed_preparation_inputs(store, session_id)

    env = _make_envelope(
        "verifier.preparation_requested", session_id, refs, "ResearchOrchestrator"
    )

    await handler.handle_verifier_preparation(client, env)

    frozen = [p for p in client.published if p.event_type == "verifier.spec_frozen"]
    assert len(frozen) == 1
    assert frozen[0].payload["verdict"] == verdict


async def test_preparation_causation_id_set(store):
    good_response = {"verifier_spec": _valid_verifier_spec(), "verdict": "ok"}
    llm = FakeLLM([_json(good_response)])
    handler = EvaluatorAgentHandler(spec_store=store, llm_client=llm)
    client = FakeBusClient()
    handler.register(client)

    session_id = str(uuid.uuid4())
    refs = _seed_preparation_inputs(store, session_id)

    env = _make_envelope(
        "verifier.preparation_requested", session_id, refs, "ResearchOrchestrator"
    )

    await handler.handle_verifier_preparation(client, env)

    frozen = [p for p in client.published if p.event_type == "verifier.spec_frozen"]
    assert len(frozen) == 1
    assert frozen[0].causation_id == env.event_id


async def test_preparation_core_error_publishes_agent_failed(store):
    llm = FakeLLM(["not json at all"] * (core.MAX_RETRIES + 1))
    handler = EvaluatorAgentHandler(spec_store=store, llm_client=llm)
    client = FakeBusClient()
    handler.register(client)

    session_id = str(uuid.uuid4())
    refs = _seed_preparation_inputs(store, session_id)

    env = _make_envelope(
        "verifier.preparation_requested", session_id, refs, "ResearchOrchestrator"
    )

    await handler.handle_verifier_preparation(client, env)

    failed = [p for p in client.published if p.event_type == "agent.failed"]
    assert len(failed) == 1
    assert failed[0].session_id == session_id
    assert failed[0].causation_id == env.event_id
    assert failed[0].payload["agent_name"] == "EvaluatorAgent"
    assert failed[0].payload["reason_classification"] == "permanent"
    assert failed[0].payload["underlying_reason"]

    assert not [p for p in client.published if p.event_type == "verifier.spec_frozen"]


# ---------------------------------------------------------------------------
# register() wires BOTH subscriptions.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# MOSS prompt_patch: config_root/agent_config/EvaluatorAgent/prompt_patch.md
# gets read by self.config.prompt_patch() and injected into the system
# message (checked here on the SelectingBenchmark flow).
# ---------------------------------------------------------------------------


async def test_selection_prompt_patch_from_config_root_injected_into_system_message(
    store, tmp_path
):
    patch_dir = tmp_path / "agent_config" / "EvaluatorAgent"
    patch_dir.mkdir(parents=True)
    (patch_dir / "prompt_patch.md").write_text("STEER: prefer X", encoding="utf-8")

    llm = FakeLLM([_json(VALID_BENCHMARK_SPEC)])
    handler = EvaluatorAgentHandler(spec_store=store, llm_client=llm, config_root=tmp_path)
    client = FakeBusClient()
    handler.register(client)

    session_id = str(uuid.uuid4())
    refs = _seed_selection_inputs(store, session_id)

    env = _make_envelope(
        "benchmark.selection_requested", session_id, refs, "ResearchOrchestrator"
    )

    await handler.handle_benchmark_selection(client, env)

    [messages] = llm.calls
    system_message = next(m["content"] for m in messages if m["role"] == "system")
    assert "STEER: prefer X" in system_message
    assert "[MOSS-injected guidance]" in system_message


def test_register_wires_both_subscriptions(store):
    llm = FakeLLM([])
    handler = EvaluatorAgentHandler(spec_store=store, llm_client=llm)
    client = FakeBusClient()
    handler.register(client)

    patterns = [pattern for pattern, _ in client.subscriptions]
    assert "benchmark.selection_requested" in patterns
    assert "verifier.preparation_requested" in patterns
