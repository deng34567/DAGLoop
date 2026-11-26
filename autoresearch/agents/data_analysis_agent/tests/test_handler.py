"""
Tests for DataAnalysisAgent's EventBus handler.
DataAnalysisAgent EventBus handler 的测试。

FakeBusClient (captures publishes; mirrors the real EventBusClient's
publish/subscribe signature -- same shape as ranking_judge_agent/tests/
test_handler.py) + FakeLLM (from this package's tests/conftest.py) + a real
tmp SpecStore. The FakeBusClient does NOT auto-deliver events to subscribed
handlers -- each test calls the handler's event method directly with a
constructed Envelope.

ONE subscription is covered: data_analysis.requested -> data_analysis.ready.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import pytest
import yaml
from jsonschema import Draft202012Validator

from autoresearch.agents.data_analysis_agent.handler import DataAnalysisAgentHandler
from autoresearch.agents.data_analysis_agent.tests.conftest import FakeLLM
from autoresearch.spec_store import SpecStore

_CONTRACTS_DIR = Path(__file__).resolve().parents[1] / "data-contracts"


def _validator(name: str) -> Draft202012Validator:
    schema = yaml.safe_load((_CONTRACTS_DIR / name).read_text(encoding="utf-8"))
    return Draft202012Validator(schema)


EXTERNAL_DATA_ANALYSIS_REPORT_VALIDATOR = _validator(
    "external_data_analysis_report.schema.yaml"
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

EXTERNAL_DATA_UPLOAD = {
    "description": "CSV of 30 trial runs, effect measured on primary endpoint.",
}

EXTERNAL_EXPERIMENT_PACKAGE = {
    "protocol_summary": "Randomized controlled trial protocol, n=30, primary endpoint X.",
}

# 4/5 trajectories converge on the same (normalized) conclusion -> top_share
# 0.8 >= BASE_THRESHOLD (0.6) at strictness=0 -> "converged".
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


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


@dataclass
class _Published:
    event_type: str
    session_id: str
    payload: dict
    causation_id: Optional[str]


class FakeBusClient:
    """Captures publishes; mirrors the real EventBusClient.publish/subscribe
    signature. Does NOT auto-deliver events to subscribed handlers."""

    def __init__(self) -> None:
        self.emitter_name = "DataAnalysisAgent"
        self.published: list[_Published] = []
        self.subscriptions: list[tuple[str, Any]] = []

    def subscribe(self, pattern: str, handler: Any) -> None:
        self.subscriptions.append((pattern, handler))

    async def publish(
        self,
        *,
        event_type: str,
        session_id: str,
        payload: dict,
        cost_delta: str = "0",
        causation_id: Optional[str] = None,
        auto_commit: bool = True,
    ) -> str:
        self.published.append(
            _Published(
                event_type=event_type,
                session_id=session_id,
                payload=payload,
                causation_id=causation_id,
            )
        )
        return f"evt-{len(self.published)}"


@dataclass
class _Envelope:
    """The subset of the real broker Envelope the handler reads."""

    event_id: str
    session_id: str
    event_type: str
    emitter: str
    payload: dict


def _make_envelope(event_type: str, session_id: str, payload: dict, emitter: str) -> _Envelope:
    return _Envelope(
        event_id=str(uuid.uuid4()),
        session_id=session_id,
        event_type=event_type,
        emitter=emitter,
        payload=payload,
    )


@pytest.fixture()
def store(tmp_path):
    s = SpecStore(tmp_path / "store")
    yield s
    s.close()


def _seed_refs(store: SpecStore, session_id: str) -> tuple[str, str]:
    """Seed an external_data_upload artifact + an external_experiment_package
    (a plan) artifact; return (external_data_upload_ref,
    external_experiment_package_ref). The handler only resolves these by ref
    via ``spec_store.get`` -- neither kind's exact producer is built yet, so
    seeding uses already-whitelisted kinds/emitters."""
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
    return data_ref, package_ref


# ---------------------------------------------------------------------------
# data_analysis.requested -> data_analysis.ready
# ---------------------------------------------------------------------------


async def test_data_analysis_requested_publishes_ready_with_resolvable_report(store):
    llm = FakeLLM([_json(VALID_ANALYSIS_LLM_RESPONSE)])
    handler = DataAnalysisAgentHandler(spec_store=store, llm_client=llm)
    client = FakeBusClient()
    handler.register(client)

    session_id = str(uuid.uuid4())
    data_ref, package_ref = _seed_refs(store, session_id)

    env = _make_envelope(
        "data_analysis.requested",
        session_id,
        {
            "external_data_upload_ref": data_ref,
            "external_experiment_package_ref": package_ref,
            "strictness": 0,
        },
        "ResearchOrchestrator",
    )

    await handler.handle_data_analysis_requested(client, env)

    ready = [p for p in client.published if p.event_type == "data_analysis.ready"]
    assert len(ready) == 1
    payload = ready[0].payload
    assert set(payload.keys()) == {"external_data_analysis_report_ref", "consensus_status"}

    report = store.get(payload["external_data_analysis_report_ref"])
    errors = list(EXTERNAL_DATA_ANALYSIS_REPORT_VALIDATOR.iter_errors(report))
    assert errors == [], [e.message for e in errors]

    assert payload["consensus_status"] == report["consensus_status"]
    assert payload["consensus_status"] == "converged"

    assert not [p for p in client.published if p.event_type == "agent.failed"]


async def test_data_analysis_requested_reads_refs_from_store_not_payload(store):
    llm = FakeLLM([_json(VALID_ANALYSIS_LLM_RESPONSE)])
    handler = DataAnalysisAgentHandler(spec_store=store, llm_client=llm)
    client = FakeBusClient()
    handler.register(client)

    session_id = str(uuid.uuid4())
    data_ref, package_ref = _seed_refs(store, session_id)

    env = _make_envelope(
        "data_analysis.requested",
        session_id,
        {
            "external_data_upload_ref": data_ref,
            "external_experiment_package_ref": package_ref,
            "strictness": 0,
        },
        "ResearchOrchestrator",
    )

    await handler.handle_data_analysis_requested(client, env)

    [messages] = llm.calls
    user_message = next(m["content"] for m in messages if m["role"] == "user")
    assert EXTERNAL_DATA_UPLOAD["description"] in user_message
    assert EXTERNAL_EXPERIMENT_PACKAGE["protocol_summary"] in user_message


async def test_data_analysis_requested_causation_id_set(store):
    llm = FakeLLM([_json(VALID_ANALYSIS_LLM_RESPONSE)])
    handler = DataAnalysisAgentHandler(spec_store=store, llm_client=llm)
    client = FakeBusClient()
    handler.register(client)

    session_id = str(uuid.uuid4())
    data_ref, package_ref = _seed_refs(store, session_id)

    env = _make_envelope(
        "data_analysis.requested",
        session_id,
        {
            "external_data_upload_ref": data_ref,
            "external_experiment_package_ref": package_ref,
            "strictness": 0,
        },
        "ResearchOrchestrator",
    )

    await handler.handle_data_analysis_requested(client, env)

    ready = [p for p in client.published if p.event_type == "data_analysis.ready"]
    assert len(ready) == 1
    assert ready[0].causation_id == env.event_id


# ---------------------------------------------------------------------------
# Failure path
# ---------------------------------------------------------------------------


async def test_data_analysis_requested_infra_error_publishes_agent_failed(store):
    class RaisingLLM:
        def chat(self, **kwargs):
            raise RuntimeError("connection reset")

    handler = DataAnalysisAgentHandler(spec_store=store, llm_client=RaisingLLM())
    client = FakeBusClient()
    handler.register(client)

    session_id = str(uuid.uuid4())
    data_ref, package_ref = _seed_refs(store, session_id)

    env = _make_envelope(
        "data_analysis.requested",
        session_id,
        {
            "external_data_upload_ref": data_ref,
            "external_experiment_package_ref": package_ref,
            "strictness": 0,
        },
        "ResearchOrchestrator",
    )

    await handler.handle_data_analysis_requested(client, env)

    failed = [p for p in client.published if p.event_type == "agent.failed"]
    assert len(failed) == 1
    assert failed[0].session_id == session_id
    assert failed[0].causation_id == env.event_id
    assert failed[0].payload["agent_name"] == "DataAnalysisAgent"
    assert failed[0].payload["reason_classification"] == "transient"
    assert failed[0].payload["underlying_reason"]

    assert not [p for p in client.published if p.event_type == "data_analysis.ready"]


async def test_data_analysis_requested_prompt_patch_from_config_root_injected(store, tmp_path):
    patch_dir = tmp_path / "agent_config" / "DataAnalysisAgent"
    patch_dir.mkdir(parents=True)
    (patch_dir / "prompt_patch.md").write_text(
        "STEER: flag low-n samples for human review", encoding="utf-8"
    )

    llm = FakeLLM([_json(VALID_ANALYSIS_LLM_RESPONSE)])
    handler = DataAnalysisAgentHandler(spec_store=store, llm_client=llm, config_root=tmp_path)
    client = FakeBusClient()
    handler.register(client)

    session_id = str(uuid.uuid4())
    data_ref, package_ref = _seed_refs(store, session_id)

    env = _make_envelope(
        "data_analysis.requested",
        session_id,
        {
            "external_data_upload_ref": data_ref,
            "external_experiment_package_ref": package_ref,
            "strictness": 0,
        },
        "ResearchOrchestrator",
    )

    await handler.handle_data_analysis_requested(client, env)

    [messages] = llm.calls
    system_message = next(m["content"] for m in messages if m["role"] == "system")
    assert "STEER: flag low-n samples for human review" in system_message
    assert "[MOSS-injected guidance]" in system_message
