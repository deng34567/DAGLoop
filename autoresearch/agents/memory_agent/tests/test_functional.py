"""
Functional test for MemoryAgent: drive the REAL ``build_memory_service`` on a
shared ``DeliveringBus`` (real subscribe/dispatch routing, not a hand-called
handler method), with only the LLM mocked (``FakeLLM``) and a real tmp
``SpecStore``.

One flow, two happy-path tests: ConsolidatingMemory:
``memory.consolidation_requested`` -> ``memory.consolidated``, with and
without the optional VerificationReport ref.

针对 MemoryAgent 的功能测试：在共享 ``DeliveringBus``（真实 subscribe/dispatch
路由，而非直接调用 handler 方法）上跑真实的 ``build_memory_service``，只 mock
LLM（``FakeLLM``），SpecStore 为真实的临时实例。
"""

from __future__ import annotations

import json
from pathlib import Path

import aiosqlite
import pytest
import yaml
from jsonschema import Draft202012Validator

from autoresearch.agents.memory_agent.service import build_memory_service
from autoresearch.agents.memory_agent.tests.conftest import FakeLLM
from autoresearch.spec_store import SpecStore
from autoresearch.testing.delivering_bus import DeliveringBus

_CONTRACTS_DIR = Path(__file__).resolve().parents[1] / "data-contracts"


def _validator(name: str) -> Draft202012Validator:
    schema = yaml.safe_load((_CONTRACTS_DIR / name).read_text(encoding="utf-8"))
    return Draft202012Validator(schema)


RESEARCH_MEMORY_VALIDATOR = _validator("research_memory.schema.yaml")


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

VERIFICATION_REPORT_VERIFIED = {
    "verdict": "verified",
    "checks": ["re-ran on held-out split"],
    "anti_cheat_flags": [],
    "reasoning": "Reproduced.",
}


def _llm_research_memory() -> dict:
    return {
        "verified_findings": ["New finding: metric improved with technique X."],
        "negative_results": [],
        "excluded_hypotheses": [],
        "consolidated_insights": ["Technique X is promising."],
        "divergence_seeds": ["Try technique Y next."],
        "next_steps": ["Run technique Y."],
    }


def _consolidate_response(routing_verdict: str = "continue_explore") -> str:
    return json.dumps(
        {
            "research_memory": _llm_research_memory(),
            "routing_verdict": routing_verdict,
        }
    )


# ---------------------------------------------------------------------------
# memory.consolidation_requested -> memory.consolidated
# ---------------------------------------------------------------------------


async def test_consolidation_requested_produces_consolidated(bus, store):
    llm = FakeLLM([_consolidate_response("continue_explore")])
    build_memory_service(
        client=bus.view("MemoryAgent"),
        spec_store=store,
        llm_client=llm,
    )

    session_id = "sess-memory-1"
    evaluation_report_ref = store.put(
        kind="evaluation_report",
        session_id=session_id,
        key="v1",
        obj=EVALUATION_REPORT_NEW_BEST,
        emitter="EvaluatorAgent",
    )
    run_record_ref = store.put(
        kind="run_record",
        session_id=session_id,
        key="v1",
        obj=RUN_RECORD,
        emitter="RunnerAgent",
    )

    await bus.view("ResearchOrchestrator").publish(
        event_type="memory.consolidation_requested",
        session_id=session_id,
        payload={
            "evaluation_report_ref": evaluation_report_ref,
            "run_record_ref": run_record_ref,
        },
    )

    consolidated = [e for e in bus.published if e.event_type == "memory.consolidated"]
    assert len(consolidated) == 1
    assert consolidated[0].payload["routing_verdict"] == "continue_explore"

    memory = store.get(consolidated[0].payload["research_memory_ref"])
    errors = list(RESEARCH_MEMORY_VALIDATOR.iter_errors(memory))
    assert errors == [], [e.message for e in errors]

    assert not [e for e in bus.published if e.event_type == "agent.failed"]


async def test_consolidation_requested_with_verification_report_computes_best_metric(bus, store):
    llm = FakeLLM([_consolidate_response("continue_explore")])
    build_memory_service(
        client=bus.view("MemoryAgent"),
        spec_store=store,
        llm_client=llm,
    )

    session_id = "sess-memory-2"
    evaluation_report_ref = store.put(
        kind="evaluation_report",
        session_id=session_id,
        key="v1",
        obj=EVALUATION_REPORT_NEW_BEST,
        emitter="EvaluatorAgent",
    )
    run_record_ref = store.put(
        kind="run_record",
        session_id=session_id,
        key="v1",
        obj=RUN_RECORD,
        emitter="RunnerAgent",
    )
    verification_report_ref = store.put(
        kind="verification_report",
        session_id=session_id,
        key="v1",
        obj=VERIFICATION_REPORT_VERIFIED,
        emitter="EvaluatorAgent",
    )

    await bus.view("ResearchOrchestrator").publish(
        event_type="memory.consolidation_requested",
        session_id=session_id,
        payload={
            "evaluation_report_ref": evaluation_report_ref,
            "run_record_ref": run_record_ref,
            "verification_report_ref": verification_report_ref,
        },
    )

    consolidated = [e for e in bus.published if e.event_type == "memory.consolidated"]
    assert len(consolidated) == 1

    memory = store.get(consolidated[0].payload["research_memory_ref"])
    errors = list(RESEARCH_MEMORY_VALIDATOR.iter_errors(memory))
    assert errors == [], [e.message for e in errors]
    # new_best + verified -> best_metric computed from run_record, per core's
    # deterministic best-tracking guard (never trusted from the LLM).
    assert memory["best_metric"] == RUN_RECORD["metric_value"]

    assert not [e for e in bus.published if e.event_type == "agent.failed"]
