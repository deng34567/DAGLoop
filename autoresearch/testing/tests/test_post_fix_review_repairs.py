"""Real handler/SDK regressions for the five post-fix review counterexamples."""
from __future__ import annotations

import asyncio
from contextlib import closing
import importlib
import json
import runpy
from pathlib import Path
from queue import Queue
import sqlite3
import subprocess
from types import SimpleNamespace

import aiosqlite
import pytest

from autoresearch.agents._reuse.openhands_backend import OpenHandsBackend
from autoresearch.agents._reuse.runner_backend import FakeRunnerBackend
from autoresearch.agents.runner_agent.handler import RunnerAgentHandler, _map_repo_result
from autoresearch.experiments.repo_improve.checkpoint_artifacts import snapshot_checkpoint
from autoresearch.experiments.repo_improve.component_plan import (
    build_component_plan, evaluate_component_contract, validate_component_acceptance,
)
from autoresearch.experiments.repo_improve.runtime_evidence import bind_component_measurements, save_output
from autoresearch.experiments.repo_improve.tests.test_component_plan import _component
from autoresearch.spec_store import SpecStore
from autoresearch.spec_store.identity import artifact_digest
from autoresearch.testing.delivering_bus import DeliveringBus


@pytest.fixture
def checkpoint(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "solve.py").write_text("def solve(x): return x\n")
    def git(*args):
        subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)
    git("init", "-q")
    git("add", "solve.py")
    git("-c", "user.name=test", "-c", "user.email=test@example.invalid", "commit", "-qm", "baseline")
    (repo / "solve.py").write_text("def solve(x): return x + 1\n")
    return snapshot_checkpoint(str(repo), "selected")


@pytest.mark.parametrize("quantity", ["separation_score", "performance_improvement"])
@pytest.mark.parametrize("pointer", ["/runtime_seconds", "/measurements/check", "/" ])
def test_acceptance_cannot_substitute_an_unrelated_runtime_scalar(tmp_path, quantity, pointer):
    component = _component("check", "indirect_gain")
    component["functional_check"] = f"{quantity} >= 0.9"
    plan = build_component_plan({"components": [component]})
    validate_component_acceptance(plan)
    record = dict(attempt_id="attempt-candidate", launched=True, finished=True, rc=0)
    payload = {"runtime_seconds": 11, quantity: 0.1, "measurements": {"check": 11}, "": 11}
    record.update(save_output(tmp_path, record["attempt_id"], json.dumps(payload), ""))
    bound = bind_component_measurements([
        {"component": "check", "actual": 11, "type": "indirect_gain",
         "evidence": {"source": "runtime_attempt", "ref": record["attempt_id"], "json_pointer": pointer}}
    ], record, {record["attempt_id"]: record})
    result = evaluate_component_contract(plan.node("check"), parent_metric=10,
        candidate_metric=11, lower_is_better=True, ablation_report=bound)
    assert result["passed"] is False
    assert "quantity" in result["reason"]


@pytest.mark.parametrize("actual,passed", [(0.1, False), (0.95, True)])
def test_declared_quantity_can_be_bound_to_its_nested_output(tmp_path, actual, passed):
    component = _component("check", "indirect_gain")
    component["functional_check"] = "separation_score >= 0.9"
    node = build_component_plan({"components": [component]}).node("check")
    record = dict(attempt_id="attempt-candidate", launched=True, finished=True, rc=0)
    record.update(save_output(tmp_path, record["attempt_id"], json.dumps({"checks": {"separation_score": actual}}), ""))
    bound = bind_component_measurements([
        {"component": "check", "actual": actual,
         "evidence": {"source": "runtime_attempt", "ref": record["attempt_id"], "json_pointer": "/checks/separation_score"}}
    ], record, {record["attempt_id"]: record})
    assert evaluate_component_contract(node, parent_metric=None, candidate_metric=None,
        lower_is_better=True, ablation_report=bound)["passed"] is passed


def test_reconciled_candidate_keeps_the_original_control_and_can_be_revalidated(tmp_path, monkeypatch, checkpoint):
    from autoresearch.experiments.repo_improve.agentic_improve import _apply_prior_worktree, _capture_component_diff
    control = tmp_path / "control"
    candidate = tmp_path / "candidate"
    for directory in (control, candidate):
        subprocess.run(["git", "clone", "-q", checkpoint, str(directory)], check=True)
    _apply_prior_worktree(str(candidate), checkpoint)
    # The inherited candidate is unchanged, but its scientific contrast is nonempty.
    assert _capture_component_diff(str(candidate), [checkpoint]).strip() == ""
    contrast = _capture_component_diff(str(candidate), [str(control)])
    assert "+ 1" in contrast
    baseline_metric = 10 - 2 * runpy.run_path(str(control / "solve.py"))["solve"](0)
    candidate_metric = 10 - 2 * runpy.run_path(str(candidate / "solve.py"))["solve"](0)
    assert (baseline_metric, candidate_metric) == (10, 8)
    with closing(SpecStore(tmp_path / "store")) as store:
        selected = {"checkpoint": checkpoint, "off_method_observation": {"valid": True},
                    "baseline_metric": 10, "method_metric": 8}
        selected_ref = store.put(kind="gen_experiment_result", session_id="s1",
            key="snapshot-" + artifact_digest(selected).split(":", 1)[1], obj=selected, emitter="RunnerAgent")
        method = {"approach": "adopt the measured mechanism", "components": [_component("gain")],
                  "reconciled_result_ref": selected_ref}
        store.put(kind="method", session_id="s1", key="current", obj=method, emitter="MethodAgent")
        handler = RunnerAgentHandler(spec_store=store, backend=FakeRunnerBackend())
        monkeypatch.setenv("AUTORESEARCH_WEB_WORKDIR", str(tmp_path / "work"))
        handler._component_directive("s1")
        assert handler._prior_worktree("s1") == checkpoint
        assert handler._prior_worktree("s1", dependencies_only=True) == []
        seen = []
        def review(method, diff, llm, **kwargs):
            seen.append((diff, kwargs))
            assert not kwargs.get("target_delta_empty")
            return {"verdict": "faithful", "component_reviews": []}
        monkeypatch.setattr("autoresearch.agents.method_implementation_agent.core.review_implementation", review)
        result = handler._review_method_implementation("s1", {
            "candidate": {"valid": True, "metric": candidate_metric},
            "baseline": {"valid": True, "metric": baseline_metric},
            "verdict": {"improved": True}, "diff": "+def solve(x): return x + 1",
            "parent_relative_diff": contrast, "checkpoint": checkpoint, "lower_is_better": True,
            "candidate_admissibility": {"verdict": "pass"},
        }, "method")
        assert seen and "+ 1" in seen[0][0]
        _, mapped = _map_repo_result("goal", result)
        finalized = handler._finalize_component_attempt("s1", mapped)
        assert handler._active_component_context["s1"]["mode"] == "revalidation"
        assert "component_execution" not in finalized
        assert finalized["checkpoint"] == checkpoint


@pytest.mark.parametrize("mutation", ["source", "stdout", "context"])
def test_review_recovery_rejects_changed_measured_identity(tmp_path, monkeypatch, checkpoint, mutation):
    with closing(SpecStore(tmp_path / "store")) as store:
        baseline = {"source_kind": "live_repo", "metric_value": 10}
        store.put(kind="baseline_result", session_id="s1", key="v1", obj=baseline, emitter="RunnerAgent")
        store.put(kind="method", session_id="s1", key="current",
                  obj={"components": [_component("gain")]}, emitter="MethodAgent")
        handler = RunnerAgentHandler(spec_store=store, backend=FakeRunnerBackend())
        monkeypatch.setenv("AUTORESEARCH_WEB_WORKDIR", str(tmp_path / "work"))
        handler._component_directive("s1")
        receipt = dict(attempt_id="attempt-candidate", launched=True, finished=True, rc=0)
        receipt.update(save_output(tmp_path, receipt["attempt_id"], '{"method":8}', ""))
        monkeypatch.setattr("autoresearch.agents.method_implementation_agent.core.review_implementation",
            lambda *a, **k: {"verdict": "insufficient_evidence", "review_error": "transport unavailable"})
        original = {"baseline": {"metric": 10, "valid": True}, "candidate": {"metric": 8, "valid": True},
                    "verdict": {"improved": True}, "diff": "+def solve(x): return x + 1",
                    "checkpoint": checkpoint, "review_receipts": [receipt], "lower_is_better": True}
        handler._review_method_implementation("s1", original, "method")
        if mutation == "source":
            (Path(checkpoint) / "solve.py").write_text("def solve(x): return -999\n")
        elif mutation == "stdout":
            Path(receipt["stdout_path"]).write_text('{"method":0}')
        else:
            handler._active_component_context["s1"]["ledger"].root_checkpoint = "changed-parent"
            monkeypatch.setattr(handler, "_component_directive", lambda _: "")
        def unexpected_review(*a, **k):
            pytest.fail("changed source/output/context must be rejected before review")
        monkeypatch.setattr("autoresearch.agents.method_implementation_agent.core.review_implementation", unexpected_review)
        with pytest.raises(Exception, match="identity|context"):
            handler._resume_implementation_review("s1", "goal", baseline)
        pending = store.get("spec://pending_implementation_review/s1/v1")
        assert pending["status"] == "pending"
        assert store.get(pending["input_ref"])["result"]["candidate"]["metric"] == 8


@pytest.mark.parametrize("outer_timeout", [None, 5])
def test_sdk_internal_typeerror_does_not_issue_a_second_run_post(monkeypatch, outer_timeout):
    from openhands.sdk.conversation import RemoteConversation
    module = importlib.import_module(RemoteConversation.__module__)
    remote = object.__new__(RemoteConversation)
    remote._terminal_status_queue = Queue()
    remote._client = object()
    remote._conversation_action_base_path = "/api/conversations"
    remote._id = "offline-test"
    posts = []
    def request(client, method, path, **kwargs):
        posts.append(path)
        return SimpleNamespace(status_code=200)
    def wait(*a):
        raise TypeError("internal wait failure after launch")
    monkeypatch.setattr(module, "_send_request", request)
    monkeypatch.setattr(remote, "_wait_for_run_completion", wait)
    with pytest.raises(TypeError, match="internal wait failure"):
        OpenHandsBackend._run_convo(remote, outer_timeout)
    assert len(posts) == 1


def test_old_sdk_without_timeout_is_called_once():
    class OldConversation:
        calls = 0
        def run(self):
            self.calls += 1
    convo = OldConversation()
    OpenHandsBackend._run_convo(convo, None)
    assert convo.calls == 1


def test_review_identity_failure_pauses_without_experiment_or_attempt(monkeypatch, tmp_path):
    from autoresearch.agents.runner_agent.core import RunnerParseError
    async def exercise():
        with closing(SpecStore(tmp_path / "store")) as store:
            baseline_ref = store.put(kind="baseline_result", session_id="s1", key="v1",
                obj={"source_kind": "live_repo", "metric_value": 10, "repo_url": "offline://test"}, emitter="RunnerAgent")
            handler = RunnerAgentHandler(spec_store=store, backend=FakeRunnerBackend())
            monkeypatch.setattr(handler, "_agentic_repo_branch", lambda: True)
            monkeypatch.setattr(handler, "_original_question", lambda _: "goal")
            def recover(*a):
                raise RunnerParseError("pending review source/output identity cannot be verified")
            def unexpected(*a, **k):
                pytest.fail("invalid saved review evidence must not rerun work or spend an attempt")
            monkeypatch.setattr(handler, "_resume_implementation_review", recover)
            monkeypatch.setattr(handler, "_run_repo_phase", unexpected)
            monkeypatch.setattr(handler, "_record_component_failure", unexpected)
            failures = []
            async def fail(*args):
                failures.append(args)
            monkeypatch.setattr(handler, "_fail", fail)
            await handler.handle_run_requested(object(), SimpleNamespace(
                session_id="s1", event_id="request-1", payload={"baseline_result_ref": baseline_ref}))
            assert len(failures) == 1
            assert failures[0][4] == "implementation_review_unavailable"
    asyncio.run(exercise())


def test_completion_enqueue_failure_replays_only_handoff(tmp_path, monkeypatch):
    async def exercise():
        raw = await aiosqlite.connect(tmp_path / "bus.db")
        class FailResultInsert:
            fail = True
            async def execute(self, sql, params=()):
                if self.fail and "INSERT OR IGNORE INTO web_pending_delivery" in sql and json.loads(params[2])["event_type"] == "experiment.run_completed":
                    self.fail = False
                    raise sqlite3.OperationalError("database is locked")
                return await raw.execute(sql, params)
            async def commit(self):
                await raw.commit()
            async def rollback(self):
                await raw.rollback()
        bus = DeliveringBus(FailResultInsert())
        with closing(SpecStore(tmp_path / "store")) as store:
            baseline_ref = store.put(kind="baseline_result", session_id="s1", key="v1",
                obj={"source_kind": "live_repo", "metric_value": 10, "repo_url": "offline://test"}, emitter="RunnerAgent")
            method_ref = store.put(kind="method", session_id="s1", key="current", obj={"components": []}, emitter="MethodAgent")
            handler = RunnerAgentHandler(spec_store=store, backend=FakeRunnerBackend())
            calls = []
            def execute(*a, **k):
                calls.append("execute")
                return _map_repo_result("goal", {"baseline": {"metric": 10, "valid": True},
                    "candidate": {"metric": 8, "valid": True}, "verdict": {"improved": True},
                    "diff": "+implementation", "checkpoint": "/offline/checkpoint", "lower_is_better": True,
                    "candidate_admissibility": {"verdict": "pass"}, "method_implementation_review": {"verdict": "faithful"}})
            monkeypatch.setattr(handler, "_agentic_repo_branch", lambda: True)
            monkeypatch.setattr(handler, "_gen_branch", lambda: True)
            monkeypatch.setattr(handler, "_original_question", lambda _: "goal")
            monkeypatch.setattr(handler, "_run_repo_phase", execute)
            monkeypatch.setattr(handler, "_record_experience_card", lambda *a: None)
            async def producer(env):
                await handler.handle_run_requested(bus.view("RunnerAgent"), env)
            async def handoff(env):
                calls.append("handoff")
                assert store.get(env.payload["run_record_ref"])["metric_value"] == 8
            bus.subscribe("experiment.run_requested", producer)
            bus.subscribe("experiment.run_completed", handoff)
            request = dict(event_type="experiment.run_requested", session_id="s1",
                           payload={"baseline_result_ref": baseline_ref, "method_ref": method_ref}, emitter="engine")
            with pytest.raises(sqlite3.OperationalError):
                await bus.publish(**request)
            assert calls == ["execute"]
            # Restart the handler too: the handoff receipt must be durable.
            handler = RunnerAgentHandler(spec_store=store, backend=FakeRunnerBackend())
            monkeypatch.setattr(handler, "_run_repo_phase", execute)
            await bus.replay_pending("s1")
            assert calls == ["execute", "handoff"]
            # A distinct request is a new experiment, even with the same method.
            monkeypatch.setattr(handler, "_agentic_repo_branch", lambda: True)
            monkeypatch.setattr(handler, "_gen_branch", lambda: True)
            monkeypatch.setattr(handler, "_original_question", lambda _: "goal")
            monkeypatch.setattr(handler, "_record_experience_card", lambda *a: None)
            await bus.publish(**request)
            assert calls == ["execute", "handoff", "execute", "handoff"]
        await raw.close()
    asyncio.run(exercise())
