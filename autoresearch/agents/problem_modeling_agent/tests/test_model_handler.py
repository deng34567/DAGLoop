"""
Tests for ProblemModelingAgent's ModelingProblem EventBus handler.
ProblemModelingAgent ModelingProblem EventBus handler 的测试。

FakeBusClient (imported from test_handler.py — captures publishes, does NOT
auto-deliver) + FakeLLM (canned JSON queue) + a real tmp SpecStore. Each test
calls handler.handle_model_requested(...) directly with a constructed
envelope carrying only the three refs (mirrors the
problem_modeling.model_requested schema).
"""

from __future__ import annotations

import json
import uuid

import pytest
import yaml
from jsonschema import Draft202012Validator

from autoresearch.agents.problem_modeling_agent.handler import ProblemModelingAgentHandler
from autoresearch.agents.problem_modeling_agent.model_core import MAX_RETRIES
from autoresearch.agents.problem_modeling_agent.tests.conftest import FakeLLM
from autoresearch.agents.problem_modeling_agent.tests.test_handler import (
    FakeBusClient,
    _make_envelope,
)
from autoresearch.spec_store import SpecStore

_CONTRACTS_DIR = (
    __import__("pathlib").Path(__file__).resolve().parents[1] / "data-contracts"
)


def _validator(schema_name: str) -> Draft202012Validator:
    schema = yaml.safe_load((_CONTRACTS_DIR / schema_name).read_text(encoding="utf-8"))
    return Draft202012Validator(schema)


PROBLEM_MODEL_VALIDATOR = _validator("problem_model.schema.yaml")

RESEARCH_SPEC = {
    "topic": "LLM coding agent evaluation",
    "research_goal": "Benchmark autonomous coding agents on real-world repos.",
    "task_mode": "benchmark",
    "budget": "two weeks",
    "deliverable_form": "benchmark report",
    "editable_scope": ["benchmark harness"],
    "forbidden_items": ["production deploy"],
    "open_questions": [],
}

LITERATURE_BRIEF = {
    "summary": "Several recent benchmarks evaluate agents on real repos.",
    "key_papers": [
        {"title": "SWE-bench", "id_or_url": "arxiv:2310.06770", "year": 2023},
    ],
}

NOVELTY_AUDIT_REPORT = {
    "verdict": "pass",
    "citation_sufficiency": "The literature brief adequately covers the space.",
    "novelty_status": "novel",
}

PROBLEM_MODEL = {
    "key_problems": ["Reproducible evaluation of agentic coding at scale."],
    "opportunities": ["A shared harness across agent frameworks."],
    "risks": ["Benchmark contamination from training data."],
    "verifiable_paths": ["Run held-out repos through the harness and score pass/fail."],
}


def _llm_response(problem_model: dict | None = None) -> str:
    return json.dumps(problem_model if problem_model is not None else PROBLEM_MODEL)


@pytest.fixture()
def store(tmp_path):
    s = SpecStore(tmp_path / "store")
    yield s
    s.close()


def _seed_inputs(store, session_id: str) -> dict[str, str]:
    """Seed the store with the 3 inputs a real NoveltyAuditing -> ModelingProblem
    transition would have produced, return their refs.
    像真实 NoveltyAuditing -> ModelingProblem 迁移一样，把 3 份输入种入 store，
    返回各自的 ref。"""
    research_spec_ref = store.put(
        kind="research_spec",
        session_id=session_id,
        key="final",
        obj=RESEARCH_SPEC,
        emitter="ProblemModelingAgent",
    )
    literature_brief_ref = store.put(
        kind="literature_brief",
        session_id=session_id,
        key="final",
        obj=LITERATURE_BRIEF,
        emitter="DeepResearchAgent",
    )
    novelty_audit_report_ref = store.put(
        kind="novelty_audit_report",
        session_id=session_id,
        key="final",
        obj=NOVELTY_AUDIT_REPORT,
        emitter="NoveltyAuditAgent",
    )
    return {
        "research_spec_ref": research_spec_ref,
        "literature_brief_ref": literature_brief_ref,
        "novelty_audit_report_ref": novelty_audit_report_ref,
    }


# ---------------------------------------------------------------------------
# (a) happy path -> problem.modeled with a ref resolving to a schema-valid
# ProblemModel
# ---------------------------------------------------------------------------


async def test_model_requested_publishes_problem_modeled(store):
    llm = FakeLLM([_llm_response()])
    handler = ProblemModelingAgentHandler(spec_store=store, llm_client=llm)
    client = FakeBusClient()
    handler.register(client)

    session_id = str(uuid.uuid4())
    refs = _seed_inputs(store, session_id)
    env = _make_envelope(
        "problem_modeling.model_requested",
        session_id,
        refs,
        "ResearchOrchestrator",
    )

    await handler.handle_model_requested(client, env)

    published = [p for p in client.published if p.event_type == "problem.modeled"]
    assert len(published) == 1
    evt = published[0]
    assert evt.session_id == session_id
    assert evt.causation_id == env.event_id

    problem_model = store.get(evt.payload["problem_model_ref"])
    errors = list(PROBLEM_MODEL_VALIDATOR.iter_errors(problem_model))
    assert errors == [], [e.message for e in errors]
    assert problem_model == PROBLEM_MODEL

    assert not [p for p in client.published if p.event_type == "agent.failed"]


# ---------------------------------------------------------------------------
# (b) provenance: payload carries ONLY refs; the 3 objects fed to the LLM
# must come from the STORE (their content shows up in the LLM call), not the
# payload (which never contains the objects themselves).
# ---------------------------------------------------------------------------


async def test_model_requested_reads_all_three_inputs_from_store_not_payload(store):
    llm = FakeLLM([_llm_response()])
    handler = ProblemModelingAgentHandler(spec_store=store, llm_client=llm)
    client = FakeBusClient()
    handler.register(client)

    session_id = str(uuid.uuid4())
    refs = _seed_inputs(store, session_id)
    env = _make_envelope(
        "problem_modeling.model_requested",
        session_id,
        refs,
        "ResearchOrchestrator",
    )
    # payload is exactly the 3 refs — no embedded objects, matching the
    # problem_modeling.model_requested schema (additionalProperties: false).
    assert set(env.payload.keys()) == {
        "research_spec_ref",
        "literature_brief_ref",
        "novelty_audit_report_ref",
    }

    await handler.handle_model_requested(client, env)

    assert len(llm.calls) == 1
    user_message = next(m["content"] for m in llm.calls[0] if m["role"] == "user")
    assert RESEARCH_SPEC["topic"] in user_message
    assert LITERATURE_BRIEF["summary"] in user_message
    assert NOVELTY_AUDIT_REPORT["citation_sufficiency"] in user_message


# ---------------------------------------------------------------------------
# (c) core error (malformed JSON) -> agent.failed, no crash, no problem.modeled
# ---------------------------------------------------------------------------


async def test_model_core_error_publishes_agent_failed(store):
    llm = FakeLLM(["not json at all"] * (MAX_RETRIES + 1))
    handler = ProblemModelingAgentHandler(spec_store=store, llm_client=llm)
    client = FakeBusClient()
    handler.register(client)

    session_id = str(uuid.uuid4())
    refs = _seed_inputs(store, session_id)
    env = _make_envelope(
        "problem_modeling.model_requested",
        session_id,
        refs,
        "ResearchOrchestrator",
    )

    await handler.handle_model_requested(client, env)

    failed = [p for p in client.published if p.event_type == "agent.failed"]
    assert len(failed) == 1
    assert failed[0].session_id == session_id
    assert failed[0].causation_id == env.event_id
    assert failed[0].payload["agent_name"] == "ProblemModelingAgent"
    assert failed[0].payload["reason_classification"] == "permanent"
    assert failed[0].payload["underlying_reason"]

    assert not [p for p in client.published if p.event_type == "problem.modeled"]


# ---------------------------------------------------------------------------
# (d) causation_id already asserted inline above (test a and c); a dedicated
# check on an unexpected (non-ModelError) exception too, for completeness.
# ---------------------------------------------------------------------------


async def test_model_unexpected_error_publishes_agent_failed_transient(store):
    class ExplodingStore:
        def get(self, ref):
            raise RuntimeError("store exploded")

        def put(self, **kwargs):
            raise AssertionError("should not be called")

    handler = ProblemModelingAgentHandler(spec_store=ExplodingStore(), llm_client=FakeLLM([]))
    client = FakeBusClient()
    handler.register(client)

    session_id = str(uuid.uuid4())
    env = _make_envelope(
        "problem_modeling.model_requested",
        session_id,
        {
            "research_spec_ref": "research_spec://sess/final",
            "literature_brief_ref": "literature_brief://sess/final",
            "novelty_audit_report_ref": "novelty_audit_report://sess/final",
        },
        "ResearchOrchestrator",
    )

    await handler.handle_model_requested(client, env)

    failed = [p for p in client.published if p.event_type == "agent.failed"]
    assert len(failed) == 1
    assert failed[0].causation_id == env.event_id
    assert failed[0].payload["reason_classification"] == "transient"


# ---------------------------------------------------------------------------
# (e) REGRESSION: registering the handler still wires all four subscriptions,
# and the model_requested subscription doesn't interfere with the others.
# ---------------------------------------------------------------------------


def test_register_wires_all_four_subscriptions(store):
    handler = ProblemModelingAgentHandler(spec_store=store, llm_client=FakeLLM([]))
    client = FakeBusClient()
    handler.register(client)

    subscribed_types = {pattern for pattern, _fn in client.subscriptions}
    assert subscribed_types == {
        "problem_modeling.requested",
        "human.clarification_answer",
        "problem_modeling.reinterpret_requested",
        "problem_modeling.model_requested",
    }


# ---------------------------------------------------------------------------
# (f) MOSS prompt_patch: config_root/agent_config/ProblemModelingAgent/
# prompt_patch.md gets read by self.config.prompt_patch() and injected into
# the system message.
# ---------------------------------------------------------------------------


async def test_model_requested_prompt_patch_from_config_root_injected(store, tmp_path):
    patch_dir = tmp_path / "agent_config" / "ProblemModelingAgent"
    patch_dir.mkdir(parents=True)
    (patch_dir / "prompt_patch.md").write_text("STEER: prefer X", encoding="utf-8")

    llm = FakeLLM([_llm_response()])
    handler = ProblemModelingAgentHandler(spec_store=store, llm_client=llm, config_root=tmp_path)
    client = FakeBusClient()
    handler.register(client)

    session_id = str(uuid.uuid4())
    refs = _seed_inputs(store, session_id)
    env = _make_envelope(
        "problem_modeling.model_requested",
        session_id,
        refs,
        "ResearchOrchestrator",
    )

    await handler.handle_model_requested(client, env)

    [messages] = llm.calls
    system_message = next(m["content"] for m in messages if m["role"] == "system")
    assert "STEER: prefer X" in system_message
    assert "[MOSS-injected guidance]" in system_message
