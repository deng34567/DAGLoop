"""Task 7: env+task_class-gated MC branch in RunningExperiment (default OFF).

The default RunnerAgent path must stay byte-identical -- so the branch fires
only when BOTH the env flag is set AND task_class matches, AND an mc_runner is
injected. All mocked; no container.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any, Optional

import pytest

from autoresearch.agents._reuse.runner_backend import FakeRunnerBackend
from autoresearch.agents.runner_agent.handler import RunnerAgentHandler
from autoresearch.experiments import mc_wiring
from autoresearch.spec_store import SpecStore


# ------------------------- pure branch selection -----------------------------

def test_mc_branch_selected_requires_env_and_task_class():
    assert mc_wiring.mc_branch_selected(
        "mc_variance_reduction", env={"AUTORESEARCH_MC_EXPERIMENT": "1"}) is True
    # env off -> default path
    assert mc_wiring.mc_branch_selected("mc_variance_reduction", env={}) is False
    # wrong task_class -> default path
    assert mc_wiring.mc_branch_selected(
        "other", env={"AUTORESEARCH_MC_EXPERIMENT": "1"}) is False
    # no task_class -> default path
    assert mc_wiring.mc_branch_selected(
        None, env={"AUTORESEARCH_MC_EXPERIMENT": "1"}) is False


def test_build_run_record_shape():
    rec = mc_wiring.build_run_record(
        {"rmse_baseline": 0.009},
        {"metric_value": 4.6, "rmse_proposed": 0.002, "over_budget": False,
         "hardcoded_suspected": False, "agent_status": "FINISHED"},
    )
    assert rec["succeeded"] is True and rec["exit_code"] == 0
    assert rec["metric_value"] == 4.6
    # over-budget/zero-ratio -> failed record
    bad = mc_wiring.build_run_record({}, {"metric_value": 0.0, "over_budget": True})
    assert bad["succeeded"] is False and bad["exit_code"] == 1


# --------------------- handler MC branch (mocked, no container) --------------

@dataclass
class _Published:
    event_type: str
    session_id: str
    payload: dict
    causation_id: Optional[str]


class FakeBusClient:
    def __init__(self):
        self.emitter_name = "RunnerAgent"
        self.published: list[_Published] = []

    def subscribe(self, pattern, handler):
        pass

    async def publish(self, *, event_type, session_id, payload, cost_delta="0",
                      causation_id=None, auto_commit=True):
        self.published.append(_Published(event_type, session_id, payload, causation_id))
        return f"evt-{len(self.published)}"


@dataclass
class _Envelope:
    event_id: str
    session_id: str
    event_type: str
    emitter: str
    payload: dict


class FakeMCRunner:
    """Records that it ran; returns canned baseline/experiment results."""

    def __init__(self):
        self.calls: list[str] = []

    def run_baseline(self, handle=None):
        self.calls.append("baseline")
        return {"metric_value": 1.0, "rmse_baseline": 0.009, "rmse_proposed": 0.009}

    def run_experiment(self, research_task, handle=None):
        self.calls.append("experiment")
        return {
            "metric_value": 4.6, "rmse_baseline": 0.009, "rmse_proposed": 0.002,
            "over_budget": False, "hardcoded_suspected": False,
            "agent_status": "FINISHED", "code": "def proposed_estimate(): ...",
        }


@pytest.fixture()
def store(tmp_path):
    s = SpecStore(tmp_path / "store")
    yield s
    s.close()


_VERIFIER = {
    "evaluator_hash": "sha256:abc", "metric": "rmse_ratio",
    "data_split": {"in_loop": "dev", "held_out": "test"},
    "run_budget": "8 seeds", "seed_policy": "fixed",
    "editable_region": ["method.py"], "forbidden_files": ["run.py"],
    "result_parse_schema": {"type": "object"},
}


def _seed(store, session_id, patch):
    pref = store.put(kind="experiment_patch", session_id=session_id, key="v1",
                     obj=patch, emitter="CoderAgent")
    vref = store.put(kind="verifier_spec", session_id=session_id, key="v1",
                     obj=_VERIFIER, emitter="EvaluatorAgent")
    return {"experiment_patch_ref": pref, "verifier_spec_ref": vref}


def _env(session_id, payload):
    return _Envelope(str(uuid.uuid4()), session_id, "experiment.run_requested",
                     "ResearchOrchestrator", payload)


async def test_handler_routes_to_mc_runner_when_env_and_task_class(store, monkeypatch):
    monkeypatch.setenv("AUTORESEARCH_MC_EXPERIMENT", "1")
    mc = FakeMCRunner()
    # default backend must NOT be used on the MC branch
    default_backend = FakeRunnerBackend(exit_code=0, metric_value=0.123)
    handler = RunnerAgentHandler(spec_store=store, backend=default_backend, mc_runner=mc)
    client = FakeBusClient()

    sid = str(uuid.uuid4())
    patch = {"hypothesis_id": "h1", "task_class": "mc_variance_reduction",
             "change_summary": "reduce MC variance", "changes": []}
    await handler.handle_run_requested(client, _env(sid, _seed(store, sid, patch)))

    completed = [p for p in client.published if p.event_type == "experiment.run_completed"]
    assert len(completed) == 1
    rec = store.get(completed[0].payload["run_record_ref"])
    assert rec["metric_value"] == 4.6  # from the MC runner, not the default backend
    assert mc.calls == ["baseline", "experiment"]


async def test_handler_uses_default_path_when_env_off(store, monkeypatch):
    monkeypatch.delenv("AUTORESEARCH_MC_EXPERIMENT", raising=False)
    mc = FakeMCRunner()
    default_backend = FakeRunnerBackend(exit_code=0, metric_value=0.123)
    handler = RunnerAgentHandler(spec_store=store, backend=default_backend, mc_runner=mc)
    client = FakeBusClient()

    sid = str(uuid.uuid4())
    # even WITH task_class, env off -> default RunnerBackend path
    patch = {"hypothesis_id": "h1", "task_class": "mc_variance_reduction",
             "summary": "x", "changes": [], "touched_files": []}
    await handler.handle_run_requested(client, _env(sid, _seed(store, sid, patch)))

    completed = [p for p in client.published if p.event_type == "experiment.run_completed"]
    assert len(completed) == 1
    rec = store.get(completed[0].payload["run_record_ref"])
    assert rec["metric_value"] == 0.123  # default backend
    assert mc.calls == []  # MC runner never touched
