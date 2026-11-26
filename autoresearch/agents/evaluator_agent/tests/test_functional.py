"""
Functional tests for EvaluatorAgent: drive the REAL
``build_evaluator_service`` on a shared ``DeliveringBus`` (real
subscribe/dispatch routing, not a hand-called handler method), with only the
LLM mocked (``FakeLLM``) and a real tmp ``SpecStore``.

Four flows, one happy-path test each:
  1. SelectingBenchmark: benchmark.selection_requested -> benchmark.spec_ready.
  2. PreparingVerifier: verifier.preparation_requested -> verifier.spec_frozen.
  3. EvaluatingExperiment: experiment.evaluation_requested -> experiment.evaluated.
  4. VerifyingBestResult: result.verification_requested -> result.verified.

针对 EvaluatorAgent 的功能测试：在共享 ``DeliveringBus``（真实
subscribe/dispatch 路由，而非直接调用 handler 方法）上跑真实的
``build_evaluator_service``，只 mock LLM（``FakeLLM``），SpecStore 为
真实的临时实例。
"""

from __future__ import annotations

import json
from pathlib import Path

import aiosqlite
import pytest
import yaml
from jsonschema import Draft202012Validator

from autoresearch.agents.evaluator_agent.service import build_evaluator_service
from autoresearch.agents.evaluator_agent.tests.conftest import FakeLLM
from autoresearch.spec_store import SpecStore
from autoresearch.testing.delivering_bus import DeliveringBus

_CONTRACTS_DIR = Path(__file__).resolve().parents[1] / "data-contracts"


def _validator(name: str) -> Draft202012Validator:
    schema = yaml.safe_load((_CONTRACTS_DIR / name).read_text(encoding="utf-8"))
    return Draft202012Validator(schema)


BENCHMARK_SPEC_VALIDATOR = _validator("benchmark_spec.schema.yaml")
VERIFIER_SPEC_VALIDATOR = _validator("verifier_spec.schema.yaml")
EVALUATION_REPORT_VALIDATOR = _validator("evaluation_report.schema.yaml")
VERIFICATION_REPORT_VALIDATOR = _validator("verification_report.schema.yaml")


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

VALID_LITERATURE_BRIEF = {
    "summary": "Dense retrievers underperform on long-tail entities.",
    "key_papers": [
        {"title": "Hybrid Retrieval for Long-Tail QA", "id_or_url": "arxiv:1234.5678"},
    ],
    "existing_benchmarks": [
        {"name": "LongTailQA", "id_or_url": "https://example.org/longtailqa"},
    ],
}

EVALUATION_SEMANTICS = {
    "comparison_statement": (
        "Under the canonical LongTailQA protocol, compare exact_match while allowing retrieval "
        "and generation method changes."
    ),
    "research_question": "Does the method improve exact match on LongTailQA?",
    "primary_metric": {"name": "exact_match", "direction": "maximize"},
    "comparison_mode": "repository_defined",
    "comparison_basis": "Use the repository benchmark.",
    "budget_interpretation": "Use its documented budget.",
    "conditions_to_hold_constant": ["dataset and evaluator"],
    "method_freedom": ["method implementation"],
    "evidence": [{
        "source_type": "existing_benchmark",
        "source_id": "https://example.org/longtailqa",
        "supports": "The benchmark defines the evaluator.",
    }],
    "uncertainties": [],
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


def _valid_verifier_spec(**overrides: object) -> dict:
    # editable_region / forbidden_files are disjoint, and no editable_region
    # path contains a frozen-core token (eval/evaluat/verif/harness/grade/
    # metric) -- satisfies both core.prepare_verifier guards.
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


def _valid_run_record(**overrides: object) -> dict:
    record = {
        "experiment_id": "exp-1",
        "succeeded": True,
        "exit_code": 0,
        "metric_value": 0.51,
        "stdout_excerpt": "epoch 10: exact_match=0.51",
        "stderr_excerpt": None,
        "environment": "python3.11, cuda12.1, 1x A100",
        "seed": "42",
        "runtime_seconds": 120.0,
        "artifacts": [],
    }
    record.update(overrides)
    return record


# RunRecord.metric_value (0.51) > VALID_BENCHMARK_SPEC.baseline.reported_metric
# (0.42) -- the deterministic is_improvement guard in core.evaluate_experiment
# computes is_improvement=True, so the canned LLM verdict must be "new_best".
VALID_RUN_RECORD = _valid_run_record(metric_value=0.51)

VALID_EVALUATION_REPORT_NEW_BEST = {
    "verdict": "new_best",
    "metric_value": 0.51,
    "baseline_metric": 0.42,
    "is_improvement": True,
    "suspicious": False,
    "anti_cheat_flags": [],
    "reasoning": None,
}


def _json(obj: dict) -> str:
    return json.dumps(obj)


# ---------------------------------------------------------------------------
# (1) benchmark.selection_requested -> benchmark.spec_ready
# ---------------------------------------------------------------------------


async def test_selection_requested_produces_spec_ready(bus, store):
    llm = FakeLLM([_json(VALID_BENCHMARK_SPEC)])
    build_evaluator_service(
        client=bus.view("EvaluatorAgent"),
        spec_store=store,
        llm_client=llm,
    )

    session_id = "sess-benchmark-1"
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
        key="final",
        obj={"evaluation_semantics": EVALUATION_SEMANTICS},
        emitter="ProblemModelingAgent",
    )

    await bus.view("ResearchOrchestrator").publish(
        event_type="benchmark.selection_requested",
        session_id=session_id,
        payload={
            "research_spec_ref": research_spec_ref,
            "literature_brief_ref": literature_brief_ref,
            "spec_interpretation_ref": spec_interpretation_ref,
        },
    )

    ready = [e for e in bus.published if e.event_type == "benchmark.spec_ready"]
    assert len(ready) == 1
    assert ready[0].payload["benchmark_id"] == VALID_BENCHMARK_SPEC["benchmark_id"]

    spec = store.get(ready[0].payload["benchmark_spec_ref"])
    errors = list(BENCHMARK_SPEC_VALIDATOR.iter_errors(spec))
    assert errors == [], [e.message for e in errors]
    assert {k: v for k, v in spec.items() if k != "evaluation_semantics"} == VALID_BENCHMARK_SPEC
    assert spec["evaluation_semantics"] == EVALUATION_SEMANTICS

    assert not [e for e in bus.published if e.event_type == "agent.failed"]


# ---------------------------------------------------------------------------
# (2) verifier.preparation_requested -> verifier.spec_frozen
# ---------------------------------------------------------------------------


async def test_preparation_requested_produces_spec_frozen(bus, store):
    good_response = {"verifier_spec": _valid_verifier_spec(), "verdict": "ok"}
    llm = FakeLLM([_json(good_response)])
    build_evaluator_service(
        client=bus.view("EvaluatorAgent"),
        spec_store=store,
        llm_client=llm,
    )

    session_id = "sess-verifier-1"
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
        event_type="verifier.preparation_requested",
        session_id=session_id,
        payload={
            "benchmark_spec_ref": benchmark_spec_ref,
            "research_spec_ref": research_spec_ref,
        },
    )

    frozen = [e for e in bus.published if e.event_type == "verifier.spec_frozen"]
    assert len(frozen) == 1
    assert frozen[0].payload["verdict"] == "ok"

    spec = store.get(frozen[0].payload["verifier_spec_ref"])
    errors = list(VERIFIER_SPEC_VALIDATOR.iter_errors(spec))
    assert errors == [], [e.message for e in errors]
    assert spec == _valid_verifier_spec()

    assert not [e for e in bus.published if e.event_type == "agent.failed"]


# ---------------------------------------------------------------------------
# (3) experiment.evaluation_requested -> experiment.evaluated
# ---------------------------------------------------------------------------


async def test_evaluation_requested_produces_evaluated(bus, store):
    llm_response = {
        "verdict": "new_best",
        "suspicious": False,
        "anti_cheat_flags": [],
        "reasoning": "0.51 beats the baseline 0.42.",
    }
    llm = FakeLLM([_json(llm_response)])
    build_evaluator_service(
        client=bus.view("EvaluatorAgent"),
        spec_store=store,
        llm_client=llm,
    )

    session_id = "sess-evaluation-1"
    run_record_ref = store.put(
        kind="run_record",
        session_id=session_id,
        key="v1",
        obj=VALID_RUN_RECORD,
        emitter="RunnerAgent",
    )
    verifier_spec_ref = store.put(
        kind="verifier_spec",
        session_id=session_id,
        key="v1",
        obj=_valid_verifier_spec(),
        emitter="EvaluatorAgent",
    )
    benchmark_spec_ref = store.put(
        kind="benchmark_spec",
        session_id=session_id,
        key="v1",
        obj=VALID_BENCHMARK_SPEC,
        emitter="EvaluatorAgent",
    )

    await bus.view("ResearchOrchestrator").publish(
        event_type="experiment.evaluation_requested",
        session_id=session_id,
        payload={
            "run_record_ref": run_record_ref,
            "verifier_spec_ref": verifier_spec_ref,
            "benchmark_spec_ref": benchmark_spec_ref,
        },
    )

    evaluated = [e for e in bus.published if e.event_type == "experiment.evaluated"]
    assert len(evaluated) == 1
    assert evaluated[0].payload["verdict"] == "new_best"

    report = store.get(evaluated[0].payload["evaluation_report_ref"])
    errors = list(EVALUATION_REPORT_VALIDATOR.iter_errors(report))
    assert errors == [], [e.message for e in errors]
    assert report["verdict"] == "new_best"
    assert report["is_improvement"] is True

    assert not [e for e in bus.published if e.event_type == "agent.failed"]


# ---------------------------------------------------------------------------
# (4) result.verification_requested -> result.verified
# ---------------------------------------------------------------------------


async def test_verification_requested_produces_verified(bus, store):
    llm_response = {
        "verdict": "verified",
        "checks": ["cross-seed: fixed seed=42", "held-out: metric on held-out split"],
        "anti_cheat_flags": [],
        "reasoning": "Result reproduces under the frozen protocol.",
    }
    llm = FakeLLM([_json(llm_response)])
    build_evaluator_service(
        client=bus.view("EvaluatorAgent"),
        spec_store=store,
        llm_client=llm,
    )

    session_id = "sess-verification-1"
    evaluation_report_ref = store.put(
        kind="evaluation_report",
        session_id=session_id,
        key="v1",
        obj=VALID_EVALUATION_REPORT_NEW_BEST,
        emitter="EvaluatorAgent",
    )
    run_record_ref = store.put(
        kind="run_record",
        session_id=session_id,
        key="v1",
        obj=VALID_RUN_RECORD,
        emitter="RunnerAgent",
    )
    verifier_spec_ref = store.put(
        kind="verifier_spec",
        session_id=session_id,
        key="v1",
        obj=_valid_verifier_spec(),
        emitter="EvaluatorAgent",
    )

    await bus.view("ResearchOrchestrator").publish(
        event_type="result.verification_requested",
        session_id=session_id,
        payload={
            "evaluation_report_ref": evaluation_report_ref,
            "run_record_ref": run_record_ref,
            "verifier_spec_ref": verifier_spec_ref,
        },
    )

    verified = [e for e in bus.published if e.event_type == "result.verified"]
    assert len(verified) == 1
    assert verified[0].payload["verdict"] == "verified"

    report = store.get(verified[0].payload["verification_report_ref"])
    errors = list(VERIFICATION_REPORT_VALIDATOR.iter_errors(report))
    assert errors == [], [e.message for e in errors]
    assert report["verdict"] == "verified"

    assert not [e for e in bus.published if e.event_type == "agent.failed"]
