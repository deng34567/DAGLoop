"""
Functional tests for ExperimentPlannerAgent: drive the REAL
``build_experiment_planner_service`` on a shared ``DeliveringBus`` (real
subscribe/dispatch routing, not a hand-called handler method), with only the
LLM mocked (``FakeLLM``) and a real tmp ``SpecStore``.

One flow, one happy-path test:
  PlanningExperiment: experiment.planning_requested -> experiment.plan_ready.

针对 ExperimentPlannerAgent 的功能测试：在共享 ``DeliveringBus``（真实
subscribe/dispatch 路由，而非直接调用 handler 方法）上跑真实的
``build_experiment_planner_service``，只 mock LLM（``FakeLLM``），SpecStore
为真实的临时实例。
"""

from __future__ import annotations

import json
from pathlib import Path

import aiosqlite
import pytest
import yaml
from jsonschema import Draft202012Validator

from autoresearch.agents.experiment_planner_agent.service import (
    build_experiment_planner_service,
)
from autoresearch.agents.experiment_planner_agent.tests.conftest import FakeLLM
from autoresearch.spec_store import SpecStore
from autoresearch.testing.delivering_bus import DeliveringBus

_CONTRACTS_DIR = Path(__file__).resolve().parents[1] / "data-contracts"


def _validator(name: str) -> Draft202012Validator:
    schema = yaml.safe_load((_CONTRACTS_DIR / name).read_text(encoding="utf-8"))
    return Draft202012Validator(schema)


EXPERIMENT_PLAN_VALIDATOR = _validator("experiment_plan.schema.yaml")


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
# Canned shapes (mirrors tests/test_handler.py -- do not invent new shapes).
# ---------------------------------------------------------------------------

VALID_RESEARCH_SPEC = {
    "topic": "Retrieval-augmented generation for long-tail scientific QA",
    "research_goal": "Propose a new RAG method for long-tail scientific question answering.",
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

VALID_VERIFIER_SPEC = {
    "evaluator_hash": "sha256:abc123",
    "metric": "exact_match",
    "data_split": {"in_loop": "dev", "held_out": "test"},
    "run_budget": "10 runs, 1h wall-clock",
    "seed_policy": "fixed seed=42 across all runs",
    "editable_region": ["src/model.py", "src/train.py"],
    "forbidden_files": ["benchmark/data/held_out.jsonl"],
    "result_parse_schema": {"type": "object"},
}

HYP_1 = {
    "id": "h1",
    "statement": "Adding a learned gating mechanism to the retriever improves long-tail recall.",
    "rationale": "Dense retrievers underperform on long-tail entities.",
    "kind": "training",
    "verifiable": True,
    "expected_effect": "exact_match improves on the long-tail subset.",
}

VALID_HYPOTHESIS_BANK = {"hypotheses": [HYP_1]}

# changes[].file is a subset of VALID_VERIFIER_SPEC["editable_region"].
VALID_PLAN = {
    "hypothesis_id": "h1",
    "change_summary": "Add a learned gating layer before retrieval fusion.",
    "changes": [
        {"file": "src/model.py", "description": "Add a gating module class."},
        {"file": "src/train.py", "description": "Wire the gating module into the training loop."},
    ],
    "ablations": ["no_gating_baseline"],
    "expected_outcome": "Improved exact_match on the long-tail subset.",
    "resource_budget": "1x A100 GPU, 4h wall-clock",
    "success_criteria": "exact_match improves by >=2 points over baseline on the held-out split.",
}


def _json(obj: dict) -> str:
    return json.dumps(obj)


# ---------------------------------------------------------------------------
# experiment.planning_requested -> experiment.plan_ready
# ---------------------------------------------------------------------------


async def test_planning_requested_produces_plan_ready(bus, store):
    llm = FakeLLM([_json(VALID_PLAN)])
    build_experiment_planner_service(
        client=bus.view("ExperimentPlannerAgent"),
        spec_store=store,
        llm_client=llm,
    )

    session_id = "sess-planning-1"
    hypothesis_bank_ref = store.put(
        kind="hypothesis_bank",
        session_id=session_id,
        key="final",
        obj=VALID_HYPOTHESIS_BANK,
        emitter="HypothesisAgent",
    )
    verifier_spec_ref = store.put(
        kind="verifier_spec",
        session_id=session_id,
        key="v1",
        obj=VALID_VERIFIER_SPEC,
        emitter="EvaluatorAgent",
    )
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

    await bus.view("ResearchOrchestrator").publish(
        event_type="experiment.planning_requested",
        session_id=session_id,
        payload={
            "hypothesis_bank_ref": hypothesis_bank_ref,
            "selected_hypothesis_id": "h1",
            "verifier_spec_ref": verifier_spec_ref,
            "benchmark_spec_ref": benchmark_spec_ref,
            "research_spec_ref": research_spec_ref,
        },
    )

    ready = [e for e in bus.published if e.event_type == "experiment.plan_ready"]
    assert len(ready) == 1

    plan = store.get(ready[0].payload["experiment_plan_ref"])
    errors = list(EXPERIMENT_PLAN_VALIDATOR.iter_errors(plan))
    assert errors == [], [e.message for e in errors]
    assert plan["hypothesis_id"] == "h1"
    assert {c["file"] for c in plan["changes"]} <= set(VALID_VERIFIER_SPEC["editable_region"])

    assert not [e for e in bus.published if e.event_type == "agent.failed"]
