"""
Tests for RunnerAgent's EventBus handler -- RunningExperiment (M11).
RunnerAgent EventBus handler 的测试 —— RunningExperiment（M11）。

FakeBusClient (captures publishes; does NOT auto-deliver -- each test calls
the handler's event method directly with a constructed Envelope) +
FakeRunnerBackend (from autoresearch.agents._reuse.runner_backend) + a real
tmp SpecStore. Mirrors autoresearch/agents/runner_agent/tests/test_handler.py
(M10 reproduction flow) for the new experiment.run_requested ->
experiment.run_completed flow.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import pytest
import yaml
from jsonschema import Draft202012Validator

from autoresearch.agents._reuse.runner_backend import FakeRunnerBackend
from autoresearch.agents.runner_agent.handler import RunnerAgentHandler
from autoresearch.spec_store import SpecStore

_CONTRACTS_DIR = Path(__file__).resolve().parents[1] / "data-contracts"
RUN_RECORD_VALIDATOR = Draft202012Validator(
    yaml.safe_load((_CONTRACTS_DIR / "run_record.schema.yaml").read_text(encoding="utf-8"))
)

VALID_EXPERIMENT_PATCH = {
    "hypothesis_id": "hyp-1",
    "summary": "Add dropout regularization to the model.",
    "changes": [
        {"file": "src/model.py", "patch": "--- a/src/model.py\n+++ b/src/model.py\n@@\n+dropout=0.1\n"},
    ],
    "touched_files": ["src/model.py"],
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


# ---------------------------------------------------------------------------
# Test doubles (mirrors test_handler.py)
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
        self.emitter_name = "RunnerAgent"
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


class _RaisingBackend:
    def run_experiment(self, experiment_patch: dict, verifier_spec: dict) -> dict:
        raise RuntimeError("backend blew up")


@pytest.fixture()
def store(tmp_path):
    s = SpecStore(tmp_path / "store")
    yield s
    s.close()


def _seed_run_inputs(store: SpecStore, session_id: str) -> dict[str, str]:
    """Seed the store with the ExperimentPatch (CoderAgent) + VerifierSpec
    (EvaluatorAgent) RunnerAgent reads for RunningExperiment; return their
    refs."""
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
    return {
        "experiment_patch_ref": experiment_patch_ref,
        "verifier_spec_ref": verifier_spec_ref,
    }


# ---------------------------------------------------------------------------
# (a) exit_code 0 -> experiment.run_completed {verdict: "succeeded"}, ref
# resolves to a schema-valid RunRecord with succeeded==True.
# ---------------------------------------------------------------------------


async def test_run_requested_success_publishes_succeeded_verdict(store):
    backend = FakeRunnerBackend(exit_code=0)
    handler = RunnerAgentHandler(spec_store=store, backend=backend)
    client = FakeBusClient()
    handler.register(client)

    session_id = str(uuid.uuid4())
    refs = _seed_run_inputs(store, session_id)

    env = _make_envelope("experiment.run_requested", session_id, refs, "ResearchOrchestrator")

    await handler.handle_run_requested(client, env)

    completed = [p for p in client.published if p.event_type == "experiment.run_completed"]
    assert len(completed) == 1
    payload = completed[0].payload
    assert payload["verdict"] == "succeeded"

    record = store.get(payload["run_record_ref"])
    errors = list(RUN_RECORD_VALIDATOR.iter_errors(record))
    assert errors == [], [e.message for e in errors]
    assert record["succeeded"] is True


# ---------------------------------------------------------------------------
# (b) exit_code 1 -> experiment.run_completed {verdict: "failed"}, RunRecord
# succeeded==False.
# ---------------------------------------------------------------------------


async def test_run_requested_nonzero_exit_publishes_failed_verdict(store):
    backend = FakeRunnerBackend(exit_code=1)
    handler = RunnerAgentHandler(spec_store=store, backend=backend)
    client = FakeBusClient()
    handler.register(client)

    session_id = str(uuid.uuid4())
    refs = _seed_run_inputs(store, session_id)

    env = _make_envelope("experiment.run_requested", session_id, refs, "ResearchOrchestrator")

    await handler.handle_run_requested(client, env)

    completed = [p for p in client.published if p.event_type == "experiment.run_completed"]
    assert len(completed) == 1
    assert completed[0].payload["verdict"] == "failed"

    record = store.get(completed[0].payload["run_record_ref"])
    errors = list(RUN_RECORD_VALIDATOR.iter_errors(record))
    assert errors == [], [e.message for e in errors]
    assert record["succeeded"] is False


# ---------------------------------------------------------------------------
# (c) provenance -- both artifacts are read from the store, never from the
# event payload (which carries only refs).
# ---------------------------------------------------------------------------


async def test_run_requested_reads_inputs_from_store_not_payload(store):
    backend = FakeRunnerBackend(exit_code=0)
    handler = RunnerAgentHandler(spec_store=store, backend=backend)
    client = FakeBusClient()
    handler.register(client)

    session_id = str(uuid.uuid4())
    refs = _seed_run_inputs(store, session_id)

    env = _make_envelope("experiment.run_requested", session_id, refs, "ResearchOrchestrator")
    assert set(env.payload.keys()) == {"experiment_patch_ref", "verifier_spec_ref"}

    await handler.handle_run_requested(client, env)

    completed = [p for p in client.published if p.event_type == "experiment.run_completed"]
    assert len(completed) == 1


# ---------------------------------------------------------------------------
# (d) backend raises -> agent.failed, transient classification, never
# crashes.
# ---------------------------------------------------------------------------


async def test_run_requested_backend_raises_publishes_agent_failed(store):
    handler = RunnerAgentHandler(spec_store=store, backend=_RaisingBackend())
    client = FakeBusClient()
    handler.register(client)

    session_id = str(uuid.uuid4())
    refs = _seed_run_inputs(store, session_id)

    env = _make_envelope("experiment.run_requested", session_id, refs, "ResearchOrchestrator")

    await handler.handle_run_requested(client, env)

    failed = [p for p in client.published if p.event_type == "agent.failed"]
    assert len(failed) == 1
    assert failed[0].session_id == session_id
    assert failed[0].causation_id == env.event_id
    assert failed[0].payload["agent_name"] == "RunnerAgent"
    assert failed[0].payload["reason_classification"] == "transient"
    assert failed[0].payload["underlying_reason"]

    assert not [p for p in client.published if p.event_type == "experiment.run_completed"]


# ---------------------------------------------------------------------------
# (e) causation_id set on the success path.
# ---------------------------------------------------------------------------


async def test_run_requested_causation_id_set(store):
    backend = FakeRunnerBackend(exit_code=0)
    handler = RunnerAgentHandler(spec_store=store, backend=backend)
    client = FakeBusClient()
    handler.register(client)

    session_id = str(uuid.uuid4())
    refs = _seed_run_inputs(store, session_id)

    env = _make_envelope("experiment.run_requested", session_id, refs, "ResearchOrchestrator")

    await handler.handle_run_requested(client, env)

    completed = [p for p in client.published if p.event_type == "experiment.run_completed"]
    assert len(completed) == 1
    assert completed[0].causation_id == env.event_id


# ---------------------------------------------------------------------------
# (f) register() wires BOTH subscriptions (M10 reproduction + M11 run).
# ---------------------------------------------------------------------------


def test_register_wires_both_subscriptions(store):
    handler = RunnerAgentHandler(spec_store=store, backend=FakeRunnerBackend())
    client = FakeBusClient()
    handler.register(client)

    patterns = [pattern for pattern, _ in client.subscriptions]
    assert "baseline.reproduction_requested" in patterns
    assert "experiment.run_requested" in patterns
