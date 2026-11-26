"""
Functional tests for ProblemModelingAgent: drive the REAL
``build_problem_modeling_service`` on a shared ``DeliveringBus`` (real
subscribe/dispatch routing, not a hand-called handler method), with only the
LLM mocked (``FakeLLM``) and a real tmp ``SpecStore``.

Three flows, one happy-path test each:
  1. ClarifyingResearchGoal round-trip: problem_modeling.requested ->
     clarification_question -> human.clarification_answer -> research_spec.ready.
  2. ReinterpretingSpec: problem_modeling.reinterpret_requested -> spec.reinterpreted.
  3. ModelingProblem: problem_modeling.model_requested -> problem.modeled.

针对 ProblemModelingAgent 的功能测试：在共享 ``DeliveringBus``（真实
subscribe/dispatch 路由，而非直接调用 handler 方法）上跑真实的
``build_problem_modeling_service``，只 mock LLM（``FakeLLM``），SpecStore 为
真实的临时实例。
"""

from __future__ import annotations

import json
from pathlib import Path

import aiosqlite
import pytest
import yaml
from jsonschema import Draft202012Validator

from autoresearch.agents.problem_modeling_agent.service import build_problem_modeling_service
from autoresearch.agents.problem_modeling_agent.tests.conftest import FakeLLM
from autoresearch.spec_store import SpecStore
from autoresearch.testing.delivering_bus import DeliveringBus

_CONTRACTS_DIR = Path(__file__).resolve().parents[1] / "data-contracts"


def _validator(name: str) -> Draft202012Validator:
    schema = yaml.safe_load((_CONTRACTS_DIR / name).read_text(encoding="utf-8"))
    return Draft202012Validator(schema)


RESEARCH_SPEC_VALIDATOR = _validator("research_spec.schema.yaml")
SPEC_INTERPRETATION_VALIDATOR = _validator("spec_interpretation.schema.yaml")
PROBLEM_MODEL_VALIDATOR = _validator("problem_model.schema.yaml")


@pytest.fixture
async def bus(tmp_path):
    conn = await aiosqlite.connect(str(tmp_path / "functional.db"))
    try:
        yield DeliveringBus(conn)
    finally:
        await conn.close()


@pytest.fixture
def store(tmp_path):
    s = SpecStore(tmp_path / "store")
    yield s
    s.close()


# ---------------------------------------------------------------------------
# Canned LLM shapes (mirrors tests/test_handler.py / test_reinterpret_handler.py
# / test_model_handler.py -- do not invent new shapes).
# ---------------------------------------------------------------------------

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


REINTERPRET_RESEARCH_SPEC = VALID_SPEC
REVISED_RESEARCH_SPEC = {**VALID_SPEC, "research_goal": "Benchmark agents with a revised scope."}

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


def _spec_interpretation(disposition: str) -> dict:
    return {
        "naive_assumptions": ["Assumed a clean-room benchmark was novel."],
        "real_open_problem": "Reproducible evaluation of agentic coding at scale.",
        "topic_position": "Adjacent to several existing agentic benchmarks.",
        "disposition": disposition,
        "scope_reason": "There is a clear, automatable verifier (test pass/fail).",
        "evaluation_semantics": {
            "comparison_statement": (
                "Under the same held-out repositories and evaluator, compare test pass rate "
                "while allowing agent implementation changes."
            ),
            "research_question": "Does the method improve the primary benchmark metric?",
            "primary_metric": {
                "name": "test_pass_rate", "direction": "maximize",
                "measurement": "Read held-out test pass rate from the canonical evaluator.",
            },
            "comparison_mode": "repository_defined",
            "comparison_basis": "Use the selected repository benchmark.",
            "budget_interpretation": "Use the benchmark's documented budget.",
            "conditions_to_hold_constant": ["dataset and evaluator"],
            "method_freedom": ["method implementation"],
            "data_source": "The benchmark's canonical held-out repositories.",
            "evaluator_source": "The benchmark's canonical test evaluator.",
            "observed_outcomes": ["runtime"],
            "evaluation_notes": [
                "Resolve task-specific precautions from authoritative benchmark files."
            ],
            "outcome_criteria": {
                "success": "The valid method pass rate exceeds the valid baseline.",
                "failure": "Execution fails or pass rate does not improve.",
                "inconclusive": "No valid comparable pass rate is produced.",
            },
            "evidence": [{
                "source_type": "research_spec", "source_id": "research_spec",
                "supports": "The research goal asks for benchmark improvement.",
            }],
            "uncertainties": ["The exact metric is selected downstream."],
        },
    }


def _reinterpret_json(disposition: str, revised_spec: dict | None = None) -> str:
    return json.dumps(
        {
            "spec_interpretation": _spec_interpretation(disposition),
            "revised_research_spec": revised_spec if revised_spec is not None else REVISED_RESEARCH_SPEC,
        }
    )


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


def _problem_model_json() -> str:
    return json.dumps(PROBLEM_MODEL)


# ---------------------------------------------------------------------------
# (1) ClarifyingResearchGoal round-trip
# ---------------------------------------------------------------------------


async def test_clarify_round_trip_produces_research_spec(bus, store):
    llm = FakeLLM([_ask_json(), _finalize_json(VALID_SPEC)])
    build_problem_modeling_service(
        client=bus.view("ProblemModelingAgent"),
        spec_store=store,
        llm_client=llm,
    )

    session_id = "sess-clarify-1"
    await bus.view("ResearchOrchestrator").publish(
        event_type="problem_modeling.requested",
        session_id=session_id,
        payload={"research_goal_text": "I want to benchmark coding agents."},
    )

    questions = [
        e for e in bus.published if e.event_type == "problem_modeling.clarification_question"
    ]
    assert len(questions) == 1
    question_id = questions[0].payload["question_id"]

    await bus.view("Human").publish(
        event_type="human.clarification_answer",
        session_id=session_id,
        payload={"question_id": question_id, "answer_text": "benchmark report"},
    )

    ready = [e for e in bus.published if e.event_type == "research_spec.ready"]
    assert len(ready) == 1
    spec = store.get(ready[0].payload["research_spec_ref"])
    errors = list(RESEARCH_SPEC_VALIDATOR.iter_errors(spec))
    assert errors == [], [e.message for e in errors]
    assert spec == VALID_SPEC

    assert not [e for e in bus.published if e.event_type == "agent.failed"]


# ---------------------------------------------------------------------------
# (2) ReinterpretingSpec
# ---------------------------------------------------------------------------


async def test_reinterpret_requested_produces_spec_interpretation(bus, store):
    llm = FakeLLM([_reinterpret_json("scope_changed")])
    build_problem_modeling_service(
        client=bus.view("ProblemModelingAgent"),
        spec_store=store,
        llm_client=llm,
    )

    session_id = "sess-reinterpret-1"
    research_spec_ref = store.put(
        kind="research_spec",
        session_id=session_id,
        key="final",
        obj=REINTERPRET_RESEARCH_SPEC,
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

    await bus.view("ResearchOrchestrator").publish(
        event_type="problem_modeling.reinterpret_requested",
        session_id=session_id,
        payload={
            "research_spec_ref": research_spec_ref,
            "literature_brief_ref": literature_brief_ref,
            "field_trajectory_brief_ref": field_trajectory_brief_ref,
        },
    )

    published = [e for e in bus.published if e.event_type == "spec.reinterpreted"]
    assert len(published) == 1
    evt = published[0]
    assert evt.payload["disposition"] == "scope_changed"

    spec_interpretation = store.get(evt.payload["spec_interpretation_ref"])
    si_errors = list(SPEC_INTERPRETATION_VALIDATOR.iter_errors(spec_interpretation))
    assert si_errors == [], [e.message for e in si_errors]

    revised_spec = store.get(evt.payload["revised_research_spec_ref"])
    rs_errors = list(RESEARCH_SPEC_VALIDATOR.iter_errors(revised_spec))
    assert rs_errors == [], [e.message for e in rs_errors]

    assert not [e for e in bus.published if e.event_type == "agent.failed"]


# ---------------------------------------------------------------------------
# (3) ModelingProblem
# ---------------------------------------------------------------------------


async def test_model_requested_produces_problem_model(bus, store):
    llm = FakeLLM([_problem_model_json()])
    build_problem_modeling_service(
        client=bus.view("ProblemModelingAgent"),
        spec_store=store,
        llm_client=llm,
    )

    session_id = "sess-model-1"
    research_spec_ref = store.put(
        kind="research_spec",
        session_id=session_id,
        key="final",
        obj=VALID_SPEC,
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

    await bus.view("ResearchOrchestrator").publish(
        event_type="problem_modeling.model_requested",
        session_id=session_id,
        payload={
            "research_spec_ref": research_spec_ref,
            "literature_brief_ref": literature_brief_ref,
            "novelty_audit_report_ref": novelty_audit_report_ref,
        },
    )

    published = [e for e in bus.published if e.event_type == "problem.modeled"]
    assert len(published) == 1
    problem_model = store.get(published[0].payload["problem_model_ref"])
    errors = list(PROBLEM_MODEL_VALIDATOR.iter_errors(problem_model))
    assert errors == [], [e.message for e in errors]
    assert problem_model == PROBLEM_MODEL

    assert not [e for e in bus.published if e.event_type == "agent.failed"]
