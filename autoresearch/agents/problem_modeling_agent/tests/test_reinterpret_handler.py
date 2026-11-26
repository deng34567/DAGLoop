"""
Tests for ProblemModelingAgent's ReinterpretingSpec EventBus handler.
ProblemModelingAgent ReinterpretingSpec EventBus handler 的测试。

FakeBusClient (imported from test_handler.py — captures publishes, does NOT
auto-deliver) + FakeLLM (canned JSON queue) + a real tmp SpecStore. Each test
calls handler.handle_reinterpret_requested(...) directly with a constructed
envelope carrying only the three refs (mirrors the
problem_modeling.reinterpret_requested schema).
"""

from __future__ import annotations

import json
import uuid

import pytest
import yaml
from jsonschema import Draft202012Validator

from autoresearch.agents.problem_modeling_agent.handler import ProblemModelingAgentHandler
from autoresearch.agents.problem_modeling_agent.reinterpret_core import MAX_RETRIES
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


SPEC_INTERPRETATION_VALIDATOR = _validator("spec_interpretation.schema.yaml")
RESEARCH_SPEC_VALIDATOR = _validator("research_spec.schema.yaml")

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

REVISED_RESEARCH_SPEC = {**RESEARCH_SPEC, "research_goal": "Benchmark agents with a revised scope."}

FIELD_TRAJECTORY_BRIEF = {
    "origin": "Grew out of program synthesis + code LLMs.",
    "current_state": "Agentic coding benchmarks are proliferating.",
    "main_method_families": ["retrieval-augmented agents", "tool-use agents"],
    "open_problems": ["long-horizon planning", "flaky test handling"],
}

LITERATURE_BRIEF = {
    "summary": "Several recent benchmarks evaluate agents on real repos.",
    "key_papers": [
        {"title": "SWE-bench", "id_or_url": "arxiv:2310.06770", "year": 2023},
    ],
}


EVALUATION_SEMANTICS = {
    "comparison_statement": (
        "Under the same repository protocol, compare the canonical resolved score between the "
        "baseline and method while allowing method implementation changes."
    ),
    "research_question": "Does the method improve the canonical score under the stated protocol?",
    "primary_metric": {
        "name": "resolved", "direction": "maximize",
        "measurement": "Read the canonical resolved score from the evaluator.",
    },
    "comparison_mode": "repository_defined",
    "comparison_basis": "The retrieved paper defines the accepted comparison.",
    "budget_interpretation": "Use the repository-documented evaluation budget.",
    "conditions_to_hold_constant": ["dataset and evaluator"],
    "method_freedom": ["method implementation"],
    "data_source": "The benchmark's canonical held-out repository set.",
    "evaluator_source": "The benchmark's canonical evaluator.",
    "observed_outcomes": ["runtime"],
    "evaluation_notes": [
        "Resolve task-specific evaluation precautions from the selected repository."
    ],
    "outcome_criteria": {
        "success": "The valid method score improves on the valid baseline score.",
        "failure": "Execution fails or the valid method score does not improve.",
        "inconclusive": "No valid comparable score is produced.",
    },
    "evidence": [{
        "source_type": "key_paper",
        "source_id": "arxiv:2310.06770",
        "supports": "The paper defines a canonical evaluator.",
    }],
    "uncertainties": [],
}


def _spec_interpretation(disposition: str) -> dict:
    return {
        "naive_assumptions": ["Assumed a clean-room benchmark was novel."],
        "real_open_problem": "Reproducible evaluation of agentic coding at scale.",
        "topic_position": "Adjacent to several existing agentic benchmarks.",
        "disposition": disposition,
        "scope_reason": "There is a clear, automatable verifier (test pass/fail).",
        "evaluation_semantics": EVALUATION_SEMANTICS,
    }


def _llm_response(disposition: str, revised_spec: dict | None = None) -> str:
    return json.dumps(
        {
            "spec_interpretation": _spec_interpretation(disposition),
            "revised_research_spec": revised_spec if revised_spec is not None else REVISED_RESEARCH_SPEC,
        }
    )


@pytest.fixture()
def store(tmp_path):
    s = SpecStore(tmp_path / "store")
    yield s
    s.close()


def _seed_inputs(store, session_id: str) -> dict[str, str]:
    """Seed the store with the 3 inputs a real DeepResearching -> ReinterpretingSpec
    transition would have produced, return their refs.
    像真实 DeepResearching -> ReinterpretingSpec 迁移一样，把 3 份输入种入
    store，返回各自的 ref。"""
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
    field_trajectory_brief_ref = store.put(
        kind="field_trajectory_brief",
        session_id=session_id,
        key="final",
        obj=FIELD_TRAJECTORY_BRIEF,
        emitter="DeepResearchAgent",
    )
    return {
        "research_spec_ref": research_spec_ref,
        "literature_brief_ref": literature_brief_ref,
        "field_trajectory_brief_ref": field_trajectory_brief_ref,
    }


# ---------------------------------------------------------------------------
# (a) happy path -> spec.reinterpreted with disposition + 2 schema-valid refs
# ---------------------------------------------------------------------------


async def test_reinterpret_requested_publishes_spec_reinterpreted(store):
    llm = FakeLLM([_llm_response("scope_changed")])
    handler = ProblemModelingAgentHandler(spec_store=store, llm_client=llm)
    client = FakeBusClient()
    handler.register(client)

    session_id = str(uuid.uuid4())
    refs = _seed_inputs(store, session_id)
    env = _make_envelope(
        "problem_modeling.reinterpret_requested",
        session_id,
        refs,
        "ResearchOrchestrator",
    )

    await handler.handle_reinterpret_requested(client, env)

    published = [p for p in client.published if p.event_type == "spec.reinterpreted"]
    assert len(published) == 1
    evt = published[0]
    assert evt.session_id == session_id
    assert evt.causation_id == env.event_id
    assert evt.payload["disposition"] == "scope_changed"

    spec_interpretation = store.get(evt.payload["spec_interpretation_ref"])
    si_errors = list(SPEC_INTERPRETATION_VALIDATOR.iter_errors(spec_interpretation))
    assert si_errors == [], [e.message for e in si_errors]
    assert spec_interpretation["evaluation_semantics"] == EVALUATION_SEMANTICS

    revised_spec = store.get(evt.payload["revised_research_spec_ref"])
    rs_errors = list(RESEARCH_SPEC_VALIDATOR.iter_errors(revised_spec))
    assert rs_errors == [], [e.message for e in rs_errors]
    assert revised_spec == REVISED_RESEARCH_SPEC

    assert not [p for p in client.published if p.event_type == "agent.failed"]


async def test_semantic_guardian_rejects_drift_and_retries_restatement(store):
    """The independent audit repairs a CPU/default rewrite of an explicit GPU comparison."""
    bad = json.loads(_llm_response("scope_changed"))
    bad["spec_interpretation"]["evaluation_semantics"] = {
        **EVALUATION_SEMANTICS,
        "comparison_statement": (
            "On one CPU core, train a newly invented tiny model for 300 seconds and compare loss."
        ),
        "primary_metric": {
            "name": "loss", "direction": "minimize",
            "measurement": "Read final loss from the proposed evaluator.",
        },
    }
    good = json.loads(_llm_response("in_scope", revised_spec=RESEARCH_SPEC))
    good["spec_interpretation"]["evaluation_semantics"] = {
        **EVALUATION_SEMANTICS,
        "comparison_statement": (
            "On one GPU of the same class, give baseline and method 300 seconds and compare "
            "final val_bpb under the repository's canonical data, tokenizer, and evaluator."
        ),
        "primary_metric": {
            "name": "val_bpb", "direction": "minimize",
            "measurement": "Read final val_bpb from the canonical evaluator.",
        },
    }
    drift_audit = json.dumps({
        "verdict": "comparison_changed",
        "issues": ["CPU and loss replaced the explicit GPU and val_bpb comparison"],
        "remediation": "restore one GPU, 300 seconds, and final val_bpb",
        "reasoning": "The restatement changed explicit user intent.",
    })
    pass_audit = json.dumps({
        "verdict": "pass", "issues": [], "remediation": None,
        "reasoning": "The explicit comparison is preserved.",
    })
    llm = FakeLLM([json.dumps(bad), drift_audit, json.dumps(good), pass_audit])
    handler = ProblemModelingAgentHandler(
        spec_store=store, llm_client=llm, semantic_guardian_llm=llm
    )
    client = FakeBusClient()

    session_id = str(uuid.uuid4())
    refs = _seed_inputs(store, session_id)
    store.put(
        kind="research_spec_draft", session_id=session_id, key="v1",
        obj={
            "draft_spec": {}, "answered": [],
            "research_goal_text": (
                "On one GPU, give baseline and method 300 seconds and compare final val_bpb."
            ),
        },
        emitter="ProblemModelingAgent",
    )
    env = _make_envelope(
        "problem_modeling.reinterpret_requested", session_id, refs, "ResearchOrchestrator"
    )

    await handler.handle_reinterpret_requested(client, env)

    events = [p for p in client.published if p.event_type == "spec.reinterpreted"]
    assert len(events) == 1
    accepted = store.get(events[0].payload["spec_interpretation_ref"])
    assert accepted["evaluation_semantics"]["primary_metric"]["name"] == "val_bpb"
    assert "one GPU" in accepted["evaluation_semantics"]["comparison_statement"]
    assert "SWE-bench" in llm.calls[1][1]["content"]
    assert "Agentic coding benchmarks are proliferating" in llm.calls[1][1]["content"]
    assert "restore one GPU" in llm.calls[2][0]["content"]
    assert not [p for p in client.published if p.event_type == "agent.failed"]


# ---------------------------------------------------------------------------
# (a2) revised spec is ALWAYS persisted, even when disposition is in_scope
# (disposition is only for routing, per Task 2's note).
# ---------------------------------------------------------------------------


async def test_revised_spec_persisted_even_when_in_scope(store):
    llm = FakeLLM([_llm_response("in_scope", revised_spec=RESEARCH_SPEC)])
    handler = ProblemModelingAgentHandler(spec_store=store, llm_client=llm)
    client = FakeBusClient()
    handler.register(client)

    session_id = str(uuid.uuid4())
    refs = _seed_inputs(store, session_id)
    env = _make_envelope(
        "problem_modeling.reinterpret_requested",
        session_id,
        refs,
        "ResearchOrchestrator",
    )

    await handler.handle_reinterpret_requested(client, env)

    evt = [p for p in client.published if p.event_type == "spec.reinterpreted"][0]
    assert evt.payload["disposition"] == "in_scope"
    # revised_research_spec_ref must resolve regardless of disposition
    assert store.get(evt.payload["revised_research_spec_ref"]) == RESEARCH_SPEC


# ---------------------------------------------------------------------------
# (b) provenance: payload carries ONLY refs; the 3 objects fed to the LLM
# must come from the STORE (their content shows up in the LLM call), not the
# payload (which never contains the objects themselves).
# ---------------------------------------------------------------------------


async def test_reinterpret_reads_all_three_inputs_from_store_not_payload(store):
    llm = FakeLLM([_llm_response("in_scope")])
    handler = ProblemModelingAgentHandler(spec_store=store, llm_client=llm)
    client = FakeBusClient()
    handler.register(client)

    session_id = str(uuid.uuid4())
    refs = _seed_inputs(store, session_id)
    original_goal = "Improve the supplied benchmark without changing its scientific meaning."
    handler._save_working_state(session_id, {
        "draft_spec": RESEARCH_SPEC,
        "answered": [],
        "research_goal_text": original_goal,
    })
    env = _make_envelope(
        "problem_modeling.reinterpret_requested",
        session_id,
        refs,
        "ResearchOrchestrator",
    )
    # payload is exactly the 3 refs — no embedded objects, matching the
    # problem_modeling.reinterpret_requested schema (additionalProperties: false).
    assert set(env.payload.keys()) == {
        "research_spec_ref",
        "literature_brief_ref",
        "field_trajectory_brief_ref",
    }

    await handler.handle_reinterpret_requested(client, env)

    assert len(llm.calls) == 1
    user_message = next(m["content"] for m in llm.calls[0] if m["role"] == "user")
    assert RESEARCH_SPEC["topic"] in user_message
    assert original_goal in user_message
    assert LITERATURE_BRIEF["summary"] in user_message
    assert FIELD_TRAJECTORY_BRIEF["origin"] in user_message


# ---------------------------------------------------------------------------
# (c) core error (malformed JSON) -> agent.failed, no crash, no spec.reinterpreted
# ---------------------------------------------------------------------------


async def test_reinterpret_core_error_publishes_agent_failed(store):
    llm = FakeLLM(["not json at all"] * (MAX_RETRIES + 1))
    handler = ProblemModelingAgentHandler(spec_store=store, llm_client=llm)
    client = FakeBusClient()
    handler.register(client)

    session_id = str(uuid.uuid4())
    refs = _seed_inputs(store, session_id)
    env = _make_envelope(
        "problem_modeling.reinterpret_requested",
        session_id,
        refs,
        "ResearchOrchestrator",
    )

    await handler.handle_reinterpret_requested(client, env)

    failed = [p for p in client.published if p.event_type == "agent.failed"]
    assert len(failed) == 1
    assert failed[0].session_id == session_id
    assert failed[0].causation_id == env.event_id
    assert failed[0].payload["agent_name"] == "ProblemModelingAgent"
    assert failed[0].payload["reason_classification"] == "permanent"
    assert failed[0].payload["underlying_reason"]

    assert not [p for p in client.published if p.event_type == "spec.reinterpreted"]
