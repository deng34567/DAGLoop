"""
Functional test for ReportAgent: drive the REAL ``build_report_service`` on a
shared ``DeliveringBus`` (real subscribe/dispatch routing, not a hand-called
handler method), with only the LLM mocked (``FakeLLM``) and a real tmp
``SpecStore``.

One flow, two happy-path tests: Reporting: ``report.requested`` ->
``report.ready``, with and without the optional ResearchSpec ref.

针对 ReportAgent 的功能测试：在共享 ``DeliveringBus``（真实 subscribe/dispatch
路由，而非直接调用 handler 方法）上跑真实的 ``build_report_service``，只 mock
LLM（``FakeLLM``），SpecStore 为真实的临时实例。
"""

from __future__ import annotations

import json
from pathlib import Path

import aiosqlite
import pytest
import yaml
from jsonschema import Draft202012Validator

from autoresearch.agents.report_agent.service import build_report_service
from autoresearch.agents.report_agent.tests.conftest import FakeLLM
from autoresearch.spec_store import SpecStore
from autoresearch.testing.delivering_bus import DeliveringBus

_CONTRACTS_DIR = Path(__file__).resolve().parents[1] / "data-contracts"


def _validator(name: str) -> Draft202012Validator:
    schema = yaml.safe_load((_CONTRACTS_DIR / name).read_text(encoding="utf-8"))
    return Draft202012Validator(schema)


REPORT_VALIDATOR = _validator("report.schema.yaml")


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

RESEARCH_MEMORY = {
    "best_metric": 0.91,
    "best_result_ref": "research_store://run_record/best",
    "verified_findings": [
        "Technique X improves metric over baseline.",
        "Batch size 64 is stable.",
    ],
    "negative_results": ["Technique Z degrades the metric."],
    "excluded_hypotheses": ["Hypothesis about learning rate warmup."],
    "consolidated_insights": ["Technique X is promising."],
    "divergence_seeds": ["Try technique Y next."],
    "next_steps": ["Run technique Y."],
}

RESEARCH_SPEC = {
    "research_question": "Does technique X improve the metric?",
}


def _report(**overrides: object) -> dict:
    report = {
        "summary": "Technique X improves the metric; technique Z does not.",
        "verified_findings": ["Technique X improves metric over baseline."],
        "negative_results": ["Technique Z degrades the metric."],
        "unverified_ideas": ["Technique W might help."],
        "experiment_not_run": ["Ablation on dataset size."],
        "replication_failed": [],
        "external_data_incomplete": [],
        "analysis_caveats": ["Only one seed was tried."],
        "known_risks": ["Overfitting to validation split."],
        "experiment_ledger": [
            {
                "experiment_id": "exp-1",
                "metric_value": 0.91,
                "verdict": "new_best",
                "notes": "Technique X applied.",
            }
        ],
    }
    report.update(overrides)
    return report


def _json(obj: dict) -> str:
    return json.dumps(obj)


# ---------------------------------------------------------------------------
# report.requested -> report.ready
# ---------------------------------------------------------------------------


async def test_report_requested_produces_ready(bus, store):
    llm = FakeLLM([_json(_report())])
    build_report_service(
        client=bus.view("ReportAgent"),
        spec_store=store,
        llm_client=llm,
    )

    session_id = "sess-report-1"
    research_memory_ref = store.put(
        kind="research_memory",
        session_id=session_id,
        key="v1",
        obj=RESEARCH_MEMORY,
        emitter="MemoryAgent",
    )
    research_spec_ref = store.put(
        kind="research_spec",
        session_id=session_id,
        key="v1",
        obj=RESEARCH_SPEC,
        emitter="ProblemModelingAgent",
    )

    await bus.view("ResearchOrchestrator").publish(
        event_type="report.requested",
        session_id=session_id,
        payload={
            "research_memory_ref": research_memory_ref,
            "research_spec_ref": research_spec_ref,
        },
    )

    ready = [e for e in bus.published if e.event_type == "report.ready"]
    assert len(ready) == 1

    report = store.get(ready[0].payload["report_ref"])
    errors = list(REPORT_VALIDATOR.iter_errors(report))
    assert errors == [], [e.message for e in errors]
    assert report["verified_findings"] == ["Technique X improves metric over baseline."]

    assert not [e for e in bus.published if e.event_type == "agent.failed"]


async def test_report_requested_without_research_spec_ref_produces_ready(bus, store):
    llm = FakeLLM([_json(_report())])
    build_report_service(
        client=bus.view("ReportAgent"),
        spec_store=store,
        llm_client=llm,
    )

    session_id = "sess-report-2"
    research_memory_ref = store.put(
        kind="research_memory",
        session_id=session_id,
        key="v1",
        obj=RESEARCH_MEMORY,
        emitter="MemoryAgent",
    )

    await bus.view("ResearchOrchestrator").publish(
        event_type="report.requested",
        session_id=session_id,
        payload={"research_memory_ref": research_memory_ref},
    )

    ready = [e for e in bus.published if e.event_type == "report.ready"]
    assert len(ready) == 1

    report = store.get(ready[0].payload["report_ref"])
    errors = list(REPORT_VALIDATOR.iter_errors(report))
    assert errors == [], [e.message for e in errors]

    assert not [e for e in bus.published if e.event_type == "agent.failed"]
