"""
Functional test for DataAnalysisAgent: drive the REAL
``build_data_analysis_service`` on a shared ``DeliveringBus`` (real
subscribe/dispatch routing, not a hand-called handler method), with only the
LLM mocked (``FakeLLM``) and a real tmp ``SpecStore``.

One flow, happy-path: data_analysis.requested -> data_analysis.ready.

针对 DataAnalysisAgent 的功能测试：在共享 ``DeliveringBus``（真实
subscribe/dispatch 路由，而非直接调用 handler 方法）上跑真实的
``build_data_analysis_service``，只 mock LLM（``FakeLLM``），SpecStore 为真实
的临时实例。
"""

from __future__ import annotations

import json
from pathlib import Path

import aiosqlite
import pytest
import yaml
from jsonschema import Draft202012Validator

from autoresearch.agents.data_analysis_agent.service import build_data_analysis_service
from autoresearch.agents.data_analysis_agent.tests.conftest import FakeLLM
from autoresearch.spec_store import SpecStore
from autoresearch.testing.delivering_bus import DeliveringBus

_CONTRACTS_DIR = Path(__file__).resolve().parents[1] / "data-contracts"


def _validator(name: str) -> Draft202012Validator:
    schema = yaml.safe_load((_CONTRACTS_DIR / name).read_text(encoding="utf-8"))
    return Draft202012Validator(schema)


EXTERNAL_DATA_ANALYSIS_REPORT_VALIDATOR = _validator(
    "external_data_analysis_report.schema.yaml"
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


EXTERNAL_DATA_UPLOAD = {
    "description": "CSV of 30 trial runs, effect measured on primary endpoint.",
}

EXTERNAL_EXPERIMENT_PACKAGE = {
    "protocol_summary": "Randomized controlled trial protocol, n=30, primary endpoint X.",
}

VALID_ANALYSIS_LLM_RESPONSE = {
    "qc_summary": "All trajectories passed QC.",
    "trajectories": [
        {"trajectory_id": "t1", "conclusion": "Effect confirmed.", "notes": None},
        {"trajectory_id": "t2", "conclusion": "Effect confirmed.", "notes": None},
        {"trajectory_id": "t3", "conclusion": "Effect confirmed.", "notes": None},
        {"trajectory_id": "t4", "conclusion": "Effect confirmed.", "notes": None},
        {"trajectory_id": "t5", "conclusion": "No effect detected.", "notes": None},
    ],
    "consensus_conclusion": "Effect is real and reproducible.",
    "divergence_points": ["t5 found no effect"],
    "robustness": "high",
    "need_human_review": False,
}


def _json(obj: dict) -> str:
    return json.dumps(obj)


async def test_data_analysis_requested_produces_ready(bus, store):
    llm = FakeLLM([_json(VALID_ANALYSIS_LLM_RESPONSE)])
    build_data_analysis_service(
        client=bus.view("DataAnalysisAgent"),
        spec_store=store,
        llm_client=llm,
    )

    session_id = "sess-data-analysis-1"
    data_ref = store.put(
        kind="audit",
        session_id=session_id,
        key="external-data-upload",
        obj=EXTERNAL_DATA_UPLOAD,
        emitter="Human",
    )
    package_ref = store.put(
        kind="experiment_plan",
        session_id=session_id,
        key="external-experiment-package",
        obj=EXTERNAL_EXPERIMENT_PACKAGE,
        emitter="ExperimentPlannerAgent",
    )

    await bus.view("ResearchOrchestrator").publish(
        event_type="data_analysis.requested",
        session_id=session_id,
        payload={
            "external_data_upload_ref": data_ref,
            "external_experiment_package_ref": package_ref,
            "strictness": 0,
        },
    )

    ready = [e for e in bus.published if e.event_type == "data_analysis.ready"]
    assert len(ready) == 1
    assert ready[0].payload["consensus_status"] == "converged"

    report = store.get(ready[0].payload["external_data_analysis_report_ref"])
    errors = list(EXTERNAL_DATA_ANALYSIS_REPORT_VALIDATOR.iter_errors(report))
    assert errors == [], [e.message for e in errors]
    assert report["consensus_status"] == "converged"

    assert not [e for e in bus.published if e.event_type == "agent.failed"]
