"""Cross-step recovery through real adapters and durable baseline event delivery."""
from __future__ import annotations

import asyncio
from contextlib import closing
import importlib.util
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
from types import SimpleNamespace

import aiosqlite
import pytest

from autoresearch.agents._reuse.runner_backend import FakeRunnerBackend
from autoresearch.agents.runner_agent import core
from autoresearch.agents.runner_agent.handler import RunnerAgentHandler, _parse_agentic_result
from autoresearch.experiments.repo_improve.checkpoint_artifacts import snapshot_checkpoint
from autoresearch.experiments.repo_improve.runtime_evidence import save_output
from autoresearch.experiments.repo_improve.tests.test_component_plan import _component
from autoresearch.spec_store import SpecStore
from autoresearch.testing.delivering_bus import DeliveringBus


@pytest.mark.parametrize("adapter", ["local", "remote_entry"])
def test_normal_review_recovery_preserves_host_receipts_through_adapter(
    tmp_path, monkeypatch, capsys, adapter,
):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "solve.py").write_text("def solve(x): return x\n")
    def git(*args):
        subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)
    git("init", "-q")
    git("add", "solve.py")
    git("-c", "user.name=test", "-c", "user.email=test@example.invalid", "commit", "-qm", "baseline")
    (repo / "solve.py").write_text("def solve(x): return x + 1\n")
    checkpoint = snapshot_checkpoint(str(repo), "review")
    receipt = dict(attempt_id="attempt-candidate", launched=True, finished=True, rc=0)
    receipt.update(save_output(tmp_path, receipt["attempt_id"], '{"method":8}', ""))
    raw = {
        "baseline": SimpleNamespace(metric_value=10, valid=True, error=None),
        "candidate": SimpleNamespace(metric_value=8, valid=True, error=None),
        "verdict": {"improved": True}, "diff": "+def solve(x): return x + 1",
        "checkpoint": checkpoint, "review_receipts": [receipt], "lower_is_better": True,
        "candidate_admissibility": {"verdict": "pass"},
    }
    executions, reviews = [], []
    def execute(*args, **kwargs):
        executions.append("measure")
        return raw
    def review(*args, **kwargs):
        reviews.append("review")
        if len(reviews) == 1:
            return {"verdict": "insufficient_evidence", "review_error": "transport unavailable"}
        return {"verdict": "faithful", "component_reviews": []}
    monkeypatch.setenv("AUTORESEARCH_WEB_WORKDIR", str(tmp_path / "work"))
    monkeypatch.setenv("AUTORESEARCH_AGENTIC_IMPROVE", "1")
    monkeypatch.setattr("autoresearch.experiments.repo_improve.agentic_improve.run_agentic_improve", execute)
    monkeypatch.setattr("autoresearch.agents.method_implementation_agent.core.review_implementation", review)
    with closing(SpecStore(tmp_path / "store")) as store:
        baseline = {"source_kind": "live_repo", "metric_value": 10}
        store.put(kind="baseline_result", session_id="s1", key="v1", obj=baseline, emitter="RunnerAgent")
        store.put(kind="method", session_id="s1", key="current",
                  obj={"components": [_component("gain")]}, emitter="MethodAgent")
        handler = RunnerAgentHandler(spec_store=store, backend=FakeRunnerBackend())
        handler._component_directive("s1")
        if adapter == "local":
            handler._run_repo_experiment_local("s1", "goal", repo_url="offline://review",
                execution_phase="method", frozen_baseline=baseline)
        else:
            # Execute the real remote entry through RESULT_JSON, with no Docker/GPU/network.
            entry = Path(__file__).resolve().parents[3] / "scripts/dgx/agentic_run.py"
            spec = importlib.util.spec_from_file_location("review_remote_entry", entry)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            with monkeypatch.context() as remote:
                remote.setattr(sys, "argv", [str(entry), "goal"])
                remote.setattr(module, "_session_workroot", lambda _: str(tmp_path / "remote"))
                remote.setattr(module, "_select_repo_urls", lambda *args: ("offline://review", []))
                remote.setattr(module, "_acquire_session_lock", lambda *args: None)
                remote.setattr(module, "_acquire_gpu_locks", lambda *args, **kwargs: [])
                remote.setattr(module, "_remove_orphan_agent_containers", lambda *args: None)
                remote.setattr(module, "_resolve_uncertain_attempt_without_owner", lambda *args: None)
                remote.setattr(module.os, "_exit", lambda code: None)
                module.main()
            serialized = _parse_agentic_result(capsys.readouterr().out, 0)
            assert serialized["review_receipts"] == [receipt]
            handler._review_method_implementation("s1", serialized, "method")
        pending = store.get("spec://pending_implementation_review/s1/v1")
        captured = store.get(pending["input_ref"])
        assert captured["result"]["review_receipts"] == [receipt]
        # Survive a Runner restart. Recover the original 8-point measurement by review only.
        handler = RunnerAgentHandler(spec_store=store, backend=FakeRunnerBackend())
        _, recovered = handler._resume_implementation_review("s1", "goal", baseline)
        assert recovered["method_metric"] == 8
        assert recovered.get("implementation_review_pending") is not True
        assert executions == ["measure"] and reviews == ["review", "review"]


@pytest.mark.parametrize("branch", ["repo", "failed_repo", "core", "gen", "discovery"])
def test_baseline_completion_replay_only_hands_off_original_snapshot(tmp_path, monkeypatch, branch):
    async def exercise():
        raw = await aiosqlite.connect(tmp_path / "bus.db")
        class FailCompletedInsert:
            fail = True
            async def execute(self, sql, params=()):
                if (self.fail and sql.lstrip().startswith("INSERT") and "web_pending_delivery" in sql
                        and json.loads(params[2])["event_type"] == "baseline.reproduced"):
                    self.fail = False
                    raise sqlite3.OperationalError("database is locked")
                return await raw.execute(sql, params)
            async def commit(self):
                await raw.commit()
            async def rollback(self):
                await raw.rollback()
        bus = DeliveringBus(FailCompletedInsert())
        try:
            with closing(SpecStore(tmp_path / "store")) as store:
                benchmark_ref = store.put(kind="benchmark_spec", session_id="s1", key="v1",
                    obj={"benchmark_id": "offline-baseline", "baseline": {"reported_metric": 10}},
                    emitter="EvaluatorAgent")
                backend = FakeRunnerBackend(metric_value=10)
                measurements, handed_off, requests = [], [], []
                def execute(*args, **kwargs):
                    assert kwargs["execution_phase"] == "baseline"
                    measurements.append("measure")
                    return ({"succeeded": True, "exit_code": 0, "stdout_excerpt": '{"baseline":10}'},
                            {"baseline_metric": 10, "baseline_valid": branch != "failed_repo",
                             "repo_commit": "offline-source", "metric_name": "score", "config": "fixed"})
                def make_handler():
                    handler = RunnerAgentHandler(spec_store=store, backend=backend)
                    monkeypatch.setattr(handler, "_agentic_repo_branch", lambda: branch in {"repo", "failed_repo", "discovery"})
                    monkeypatch.setattr(handler, "_gen_branch", lambda: branch == "gen")
                    monkeypatch.setattr(handler, "_resolve_repo_url", lambda *args: None if branch == "discovery" else "offline://review")
                    monkeypatch.setattr(handler, "_original_question", lambda _: "goal")
                    monkeypatch.setattr(handler, "_run_repo_phase", execute)
                    return handler
                handler = make_handler()
                async def producer(env):
                    requests.append(env)
                    await handler.handle_reproduction_requested(bus.view("RunnerAgent"), env)
                async def handoff(env):
                    handed_off.append(env.payload)
                bus.subscribe("baseline.reproduction_requested", producer)
                bus.subscribe("baseline.reproduced", handoff)
                request = dict(event_type="baseline.reproduction_requested", session_id="s1",
                    payload={"benchmark_spec_ref": benchmark_ref}, emitter="ResearchOrchestrator")
                with pytest.raises(sqlite3.OperationalError, match="database is locked"):
                    await bus.publish(**request)
                saved = store.get("spec://baseline_result/s1/v1")
                calls_before = (len(measurements), backend.calls)
                # Overwriting the mutable latest slot must not replace the original handoff.
                store.put(kind="baseline_result", session_id="s1", key="v1",
                          obj={**saved, "metric_value": 999}, emitter="RunnerAgent")
                handler = make_handler()
                assert await bus.replay_pending("s1")
                assert (len(measurements), backend.calls) == calls_before
                assert len(handed_off) == 1
                assert store.get(handed_off[0]["baseline_result_ref"]) == saved
                assert "/snapshot-" in handed_off[0]["baseline_result_ref"]
                expected = "failed" if branch in {"failed_repo", "discovery"} else "reproduced"
                assert handed_off[0]["verdict"] == expected
                # A new event with identical payload remains a new baseline request.
                await bus.publish(**request)
                assert len(handed_off) == 2
                if branch in {"repo", "failed_repo"}:
                    assert len(measurements) == calls_before[0] + 1
                elif branch == "core":
                    assert backend.calls == calls_before[1] + 1
                # Restart may toggle the recovery hint without changing the experiment.
                recovery = SimpleNamespace(session_id="s1", event_id=requests[0].event_id,
                    payload={**request["payload"], "recover_completed_result": True})
                calls_before_recovery = (len(measurements), backend.calls)
                await handler.handle_reproduction_requested(bus.view("RunnerAgent"), recovery)
                assert (len(measurements), backend.calls) == calls_before_recovery
                assert store.get(handed_off[-1]["baseline_result_ref"]) == saved
                # Changed experimental conditions cannot borrow the old request's receipt.
                changed = SimpleNamespace(session_id="s1", event_id=requests[0].event_id,
                    payload={**request["payload"], "confirmed_boundary": {"accelerator_count": 2}})
                with pytest.raises(core.RunnerParseError, match="identity mismatch"):
                    await handler.handle_reproduction_requested(bus.view("RunnerAgent"), changed)
        finally:
            await raw.close()
    asyncio.run(exercise())
