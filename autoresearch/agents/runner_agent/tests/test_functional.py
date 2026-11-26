"""
Functional tests for RunnerAgent: drive the REAL ``build_runner_service`` on
a shared ``DeliveringBus`` (real subscribe/dispatch routing, not a
hand-called handler method), with a fake execution backend
(``FakeRunnerBackend``) and a real tmp ``SpecStore``. RunnerAgent has no LLM.

Two flows, two happy-path tests:
  ReproducingBaseline: baseline.reproduction_requested -> baseline.reproduced.
  RunningExperiment (M11): experiment.run_requested ->
    experiment.run_completed.

针对 RunnerAgent 的功能测试：在共享 ``DeliveringBus``（真实 subscribe/
dispatch 路由，而非直接调用 handler 方法）上跑真实的
``build_runner_service``，用假执行后端（``FakeRunnerBackend``），SpecStore
为真实的临时实例。RunnerAgent 没有 LLM。
"""

from __future__ import annotations

from pathlib import Path

import aiosqlite
import pytest
import yaml
from jsonschema import Draft202012Validator

from autoresearch.agents._reuse.runner_backend import FakeRunnerBackend
from autoresearch.agents.runner_agent.service import build_runner_service
from autoresearch.spec_store import SpecStore
from autoresearch.testing.delivering_bus import DeliveringBus

_CONTRACTS_DIR = Path(__file__).resolve().parents[1] / "data-contracts"
BASELINE_RESULT_VALIDATOR = Draft202012Validator(
    yaml.safe_load((_CONTRACTS_DIR / "baseline_result.schema.yaml").read_text(encoding="utf-8"))
)
RUN_RECORD_VALIDATOR = Draft202012Validator(
    yaml.safe_load((_CONTRACTS_DIR / "run_record.schema.yaml").read_text(encoding="utf-8"))
)


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

VALID_BENCHMARK_SPEC = {
    "benchmark_id": "longtailqa-v1",
    "name": "LongTailQA",
    "metric": "exact_match",
    "run_command": "python run_benchmark.py --model candidate",
    "baseline": {
        "command": "python run_benchmark.py --model baseline",
        "reported_metric": 0.85,
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

VALID_EXPERIMENT_PATCH = {
    "hypothesis_id": "h1",
    "summary": "Add dropout regularization to the model.",
    "changes": [
        {"file": "src/model.py", "patch": "--- a/src/model.py\n+++ b/src/model.py\n@@\n+dropout=0.1\n"},
    ],
    "touched_files": ["src/model.py"],
}


# ---------------------------------------------------------------------------
# baseline.reproduction_requested -> baseline.reproduced
# ---------------------------------------------------------------------------


async def test_reproduction_requested_produces_reproduced(bus, store):
    backend = FakeRunnerBackend(metric_value=0.85, exit_code=0)
    build_runner_service(
        client=bus.view("RunnerAgent"),
        spec_store=store,
        backend=backend,
    )

    session_id = "sess-baseline-1"
    benchmark_spec_ref = store.put(
        kind="benchmark_spec",
        session_id=session_id,
        key="v1",
        obj=VALID_BENCHMARK_SPEC,
        emitter="EvaluatorAgent",
    )
    verifier_spec_ref = store.put(
        kind="verifier_spec",
        session_id=session_id,
        key="v1",
        obj=VALID_VERIFIER_SPEC,
        emitter="EvaluatorAgent",
    )

    await bus.view("ResearchOrchestrator").publish(
        event_type="baseline.reproduction_requested",
        session_id=session_id,
        payload={
            "benchmark_spec_ref": benchmark_spec_ref,
            "verifier_spec_ref": verifier_spec_ref,
        },
    )

    reproduced = [e for e in bus.published if e.event_type == "baseline.reproduced"]
    assert len(reproduced) == 1
    assert reproduced[0].payload["verdict"] == "reproduced"
    assert reproduced[0].payload["benchmark_id"] == VALID_BENCHMARK_SPEC["benchmark_id"]

    result = store.get(reproduced[0].payload["baseline_result_ref"])
    errors = list(BASELINE_RESULT_VALIDATOR.iter_errors(result))
    assert errors == [], [e.message for e in errors]
    assert result["reproduced"] is True

    assert not [e for e in bus.published if e.event_type == "agent.failed"]


# ---------------------------------------------------------------------------
# experiment.run_requested -> experiment.run_completed (M11)
# ---------------------------------------------------------------------------


async def test_run_requested_produces_run_completed(bus, store):
    backend = FakeRunnerBackend(metric_value=0.91, exit_code=0, stdout="ok")
    build_runner_service(
        client=bus.view("RunnerAgent"),
        spec_store=store,
        backend=backend,
    )

    session_id = "sess-run-1"
    experiment_patch_ref = store.put(
        kind="experiment_patch",
        session_id=session_id,
        key="v1",
        obj=VALID_EXPERIMENT_PATCH,
        emitter="CoderAgent",
    )
    verifier_spec_ref = store.put(
        kind="verifier_spec",
        session_id=session_id,
        key="v1",
        obj=VALID_VERIFIER_SPEC,
        emitter="EvaluatorAgent",
    )

    await bus.view("ResearchOrchestrator").publish(
        event_type="experiment.run_requested",
        session_id=session_id,
        payload={
            "experiment_patch_ref": experiment_patch_ref,
            "verifier_spec_ref": verifier_spec_ref,
        },
    )

    completed = [e for e in bus.published if e.event_type == "experiment.run_completed"]
    assert len(completed) == 1
    assert completed[0].payload["verdict"] == "succeeded"

    record = store.get(completed[0].payload["run_record_ref"])
    errors = list(RUN_RECORD_VALIDATOR.iter_errors(record))
    assert errors == [], [e.message for e in errors]
    assert record["succeeded"] is True

    assert not [e for e in bus.published if e.event_type == "agent.failed"]


# ---------------------------------------------------------------------------
# build_runner_service forwards the OPTIONAL P1 gen-experiment backend/llm to
# the handler (Step 0: threads p1_backend/p1_llm; default None -> inert).
# ---------------------------------------------------------------------------


async def test_build_runner_service_threads_p1_backend_and_llm(bus, store):
    p1_backend = object()
    p1_llm = object()
    handler = build_runner_service(
        client=bus.view("RunnerAgent"),
        spec_store=store,
        backend=FakeRunnerBackend(),
        p1_backend=p1_backend,
        p1_llm=p1_llm,
    )
    assert handler.p1_backend is p1_backend
    assert handler.p1_llm is p1_llm


async def test_build_runner_service_p1_defaults_none(bus, store):
    # Byte-identical default: no p1 injection -> both None (gen branch inert).
    handler = build_runner_service(
        client=bus.view("RunnerAgent"),
        spec_store=store,
        backend=FakeRunnerBackend(),
    )
    assert handler.p1_backend is None
    assert handler.p1_llm is None
