"""
Functional tests for CriticAgent: drive the REAL ``build_critic_service`` on
a shared ``DeliveringBus`` (real subscribe/dispatch routing, not a
hand-called handler method), with only the LLM mocked (``FakeLLM``) and a
real tmp ``SpecStore``.

Three flows, one happy-path test each:
  1. CritiquingDirection: direction.critique_requested -> direction.critiqued.
  2. CritiquingHypotheses: hypothesis.critique_requested -> hypothesis.critiqued.
  3. ReviewingExperimentPlan: experiment.review_requested -> experiment.plan_reviewed.

针对 CriticAgent 的功能测试：在共享 ``DeliveringBus``（真实 subscribe/
dispatch 路由，而非直接调用 handler 方法）上跑真实的 ``build_critic_service``，
只 mock LLM（``FakeLLM``），SpecStore 为真实的临时实例。
"""

from __future__ import annotations

import json
from pathlib import Path

import aiosqlite
import pytest
import yaml
from jsonschema import Draft202012Validator

from autoresearch.agents.critic_agent.service import build_critic_service
from autoresearch.agents.critic_agent.tests.conftest import FakeLLM
from autoresearch.spec_store import SpecStore
from autoresearch.testing.delivering_bus import DeliveringBus

_CONTRACTS_DIR = Path(__file__).resolve().parents[1] / "data-contracts"


def _validator(name: str) -> Draft202012Validator:
    schema = yaml.safe_load((_CONTRACTS_DIR / name).read_text(encoding="utf-8"))
    return Draft202012Validator(schema)


CRITIQUE_REPORT_VALIDATOR = _validator("critique_report.schema.yaml")


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
# Canned shapes (mirrors tests/test_handler.py and
# tests/test_experiment_plan_review.py -- do not invent new shapes).
# ---------------------------------------------------------------------------

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

VALID_HYPOTHESES_REPORT_SOME_PASS = {
    "target": "hypotheses",
    "verdict": "some_pass",
    "issues": ["h2 is too narrow / trivially confounded."],
    "passing_hypothesis_ids": ["h1"],
}

# editable_region / forbidden_files disjoint; the plan's changes[].file is a
# subset of editable_region, satisfying the ReviewingExperimentPlan gate.
VERIFIER_SPEC = {
    "evaluator_hash": "sha256:deadbeef",
    "metric": "accuracy",
    "data_split": {"in_loop": "dev", "held_out": "test"},
    "run_budget": "10 runs, 1h wall-clock",
    "seed_policy": "fixed seed 42 across runs",
    "editable_region": ["src/model.py", "src/retriever.py"],
    "forbidden_files": ["eval/harness.py"],
    "result_parse_schema": {"type": "object"},
}

EXPERIMENT_PLAN_IN_SCOPE = {
    "hypothesis_id": "h1",
    "change_summary": "Add hybrid sparse+dense retrieval.",
    "changes": [
        {"file": "src/retriever.py", "description": "Add a sparse+dense hybrid retriever."},
    ],
    "expected_outcome": "Higher long-tail QA accuracy vs. the dense-only baseline.",
}

VALID_EXPERIMENT_PLAN_REPORT_PASS = {
    "target": "experiment_plan",
    "verdict": "pass",
    "issues": [],
    "passing_hypothesis_ids": [],
}


def _json(obj: dict) -> str:
    return json.dumps(obj)


# ---------------------------------------------------------------------------
# (1) direction.critique_requested -> direction.critiqued
# ---------------------------------------------------------------------------


async def test_direction_critique_requested_produces_critiqued(bus, store):
    llm = FakeLLM([_json(VALID_DIRECTION_REPORT_PASS)])
    build_critic_service(
        client=bus.view("CriticAgent"),
        spec_store=store,
        llm_client=llm,
    )

    session_id = "sess-direction-1"
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

    await bus.view("ResearchOrchestrator").publish(
        event_type="direction.critique_requested",
        session_id=session_id,
        payload={
            "problem_model_ref": problem_model_ref,
            "research_spec_ref": research_spec_ref,
        },
    )

    critiqued = [e for e in bus.published if e.event_type == "direction.critiqued"]
    assert len(critiqued) == 1
    assert critiqued[0].payload["verdict"] == "pass"

    report = store.get(critiqued[0].payload["critique_report_ref"])
    errors = list(CRITIQUE_REPORT_VALIDATOR.iter_errors(report))
    assert errors == [], [e.message for e in errors]
    assert report["target"] == "direction"
    assert report["verdict"] == "pass"

    assert not [e for e in bus.published if e.event_type == "agent.failed"]


# ---------------------------------------------------------------------------
# (2) hypothesis.critique_requested -> hypothesis.critiqued
# ---------------------------------------------------------------------------


async def test_hypothesis_critique_requested_produces_critiqued(bus, store):
    llm = FakeLLM([_json(VALID_HYPOTHESES_REPORT_SOME_PASS)])
    build_critic_service(
        client=bus.view("CriticAgent"),
        spec_store=store,
        llm_client=llm,
    )

    session_id = "sess-hypotheses-1"
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

    await bus.view("ResearchOrchestrator").publish(
        event_type="hypothesis.critique_requested",
        session_id=session_id,
        payload={
            "hypothesis_bank_ref": hypothesis_bank_ref,
            "research_spec_ref": research_spec_ref,
        },
    )

    critiqued = [e for e in bus.published if e.event_type == "hypothesis.critiqued"]
    assert len(critiqued) == 1
    assert critiqued[0].payload["verdict"] == "some_pass"
    assert set(critiqued[0].payload["passing_hypothesis_ids"]) <= {"h1", "h2"}

    report = store.get(critiqued[0].payload["critique_report_ref"])
    errors = list(CRITIQUE_REPORT_VALIDATOR.iter_errors(report))
    assert errors == [], [e.message for e in errors]
    assert report["target"] == "hypotheses"

    assert not [e for e in bus.published if e.event_type == "agent.failed"]


# ---------------------------------------------------------------------------
# (3) experiment.review_requested -> experiment.plan_reviewed
# ---------------------------------------------------------------------------


async def test_experiment_review_requested_produces_plan_reviewed(bus, store):
    llm = FakeLLM([_json(VALID_EXPERIMENT_PLAN_REPORT_PASS)])
    build_critic_service(
        client=bus.view("CriticAgent"),
        spec_store=store,
        llm_client=llm,
    )

    session_id = "sess-plan-review-1"
    experiment_plan_ref = store.put(
        kind="experiment_plan",
        session_id=session_id,
        key="v1",
        obj=EXPERIMENT_PLAN_IN_SCOPE,
        emitter="ExperimentPlannerAgent",
    )
    verifier_spec_ref = store.put(
        kind="verifier_spec",
        session_id=session_id,
        key="v1",
        obj=VERIFIER_SPEC,
        emitter="EvaluatorAgent",
    )

    await bus.view("ResearchOrchestrator").publish(
        event_type="experiment.review_requested",
        session_id=session_id,
        payload={
            "experiment_plan_ref": experiment_plan_ref,
            "verifier_spec_ref": verifier_spec_ref,
        },
    )

    reviewed = [e for e in bus.published if e.event_type == "experiment.plan_reviewed"]
    assert len(reviewed) == 1
    assert reviewed[0].payload["verdict"] == "pass"

    report = store.get(reviewed[0].payload["review_report_ref"])
    errors = list(CRITIQUE_REPORT_VALIDATOR.iter_errors(report))
    assert errors == [], [e.message for e in errors]
    assert report["target"] == "experiment_plan"
    assert report["verdict"] == "pass"

    assert not [e for e in bus.published if e.event_type == "agent.failed"]
