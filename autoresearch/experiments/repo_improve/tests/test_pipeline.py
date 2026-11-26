"""End-to-end repo-improve orchestration with fake clone/LLM/runner (no GPU, no network)."""
from __future__ import annotations

import pytest
import re
import dataclasses
import json
import os
from pathlib import Path

from autoresearch.experiments.repo_improve.pipeline import run_repo_improve
from autoresearch.experiments.repo_improve.runner import RunResult
from autoresearch.experiments.repo_improve import runner as _runner


class _FakeLLM:
    def __init__(self, responses):
        self._responses = list(responses)

    def chat(self, *, messages, response_format=None, max_tokens=None):
        content = self._responses.pop(0) if self._responses else ""
        return type("R", (), {"content": content})()


def _fake_clone(repo_url, dest, **kw):
    Path(dest).mkdir(parents=True, exist_ok=True)
    Path(dest, "train.py").write_text("learning_rate = 0.001\n# model + train loop\n")
    Path(dest, "program.md").write_text("Improve val_bpb within the 5 min budget.\n")
    Path(dest, "prepare.py").write_text("# trusted data + evaluate_bpb (protected)\n")
    return dest


_GOOD_PATCH = json.dumps({"rationale": "tune LR schedule", "edits": [
    {"file": "train.py", "find": "learning_rate = 0.001", "replace": "learning_rate = 0.0015"}]})


def _runner_returning(baseline, candidate):
    def fake(benchmark, work_dir, prepare=True, seed=None, **kw):
        val = candidate if "candidate" in work_dir else baseline
        return RunResult(benchmark.metric, val, True)
    return fake


def _runner_per_seed(values):
    """values keyed by ("baseline"|"candidate", seed) -> metric."""
    def fake(benchmark, work_dir, prepare=True, seed=None, **kw):
        which = "candidate" if "candidate" in work_dir else "baseline"
        return RunResult(benchmark.metric, values[(which, seed)], True)
    return fake


def test_baseline_transient_failure_retries_once(tmp_path):
    # the UNCHANGED baseline flakes on the first call (transient) then succeeds -> the round must
    # retry and proceed to a real verdict, not die (this is what killed round-2/3 in the loop).
    calls = {"baseline": 0}

    def fake(benchmark, work_dir, prepare=True, seed=None, **kw):
        which = "candidate" if "candidate" in work_dir else "baseline"
        if which == "candidate":
            return RunResult(benchmark.metric, 0.90, True)
        calls["baseline"] += 1
        if calls["baseline"] == 1:
            return RunResult(benchmark.metric, None, False, error="flaky exit 1")
        return RunResult(benchmark.metric, 0.94, True)

    events = []
    out = run_repo_improve(
        _FakeLLM([_IS_REPO, _INFER, _INFER, _GOOD_PATCH]), "improve the tiny-lm training",
        workroot=str(tmp_path), emit=lambda e, p: events.append(e),
        clone_fn=_fake_clone, run_variant_fn=fake)
    assert calls["baseline"] >= 2                       # retried after the transient failure
    assert "baseline_retry" in events
    assert out["baseline"].metric_value == 0.94         # succeeded on retry
    assert out["verdict"]["verdict"] == "pass"          # round reached a real verdict, didn't die


def test_baseline_hard_failure_still_fails_after_retry(tmp_path):
    # if the baseline fails BOTH times it's not transient -> honest fail (one retry, not infinite).
    calls = {"baseline": 0}

    def fake(benchmark, work_dir, prepare=True, seed=None, **kw):
        if "candidate" in work_dir:
            return RunResult(benchmark.metric, 0.90, True)
        calls["baseline"] += 1
        return RunResult(benchmark.metric, None, False, error="broken repo")

    out = run_repo_improve(
        _FakeLLM([_IS_REPO, _INFER, _INFER, _GOOD_PATCH]), "improve the tiny-lm training",
        workroot=str(tmp_path), clone_fn=_fake_clone, run_variant_fn=fake)
    assert calls["baseline"] == 2                        # exactly one retry, then give up
    assert out["verdict"]["verdict"] == "fail"


def test_candidate_workspace_is_prepared(tmp_path):
    # each variant workspace is isolated, so the candidate MUST run prepare too -- else its
    # training dies on the data the baseline prepared in a different dir (the nanoGPT
    # 'train.bin not found' live failure). Record the prepare flag per workspace.
    seen = []

    def spy(benchmark, work_dir, prepare=True, seed=None, **kw):
        # classify by the workspace's OWN dir name (tmp_path may itself contain "candidate")
        base = os.path.basename(work_dir.rstrip("/"))
        which = "candidate" if base.startswith("candidate") else "baseline"
        if seed == 0:
            seen.append((which, prepare))
        return RunResult(benchmark.metric, 0.90 if which == "candidate" else 0.94, True)

    run_repo_improve(_FakeLLM([_IS_REPO, _INFER, _INFER, _GOOD_PATCH]), "improve the tiny-lm training",
                     workroot=str(tmp_path), clone_fn=_fake_clone, run_variant_fn=spy)
    assert ("baseline", True) in seen        # baseline prepares its workspace
    assert ("candidate", True) in seen       # candidate prepares ITS OWN workspace (the fix)


def test_full_flow_improvement_passes(tmp_path):
    # dynamic discovery (empty registry): LLM names the repo -> clone -> infer boundary -> patch
    events = []
    out = run_repo_improve(
        _FakeLLM([_IS_REPO, _INFER, _INFER, _GOOD_PATCH]), "improve the tiny-lm training",
        workroot=str(tmp_path), emit=lambda e, p: events.append(e),
        clone_fn=_fake_clone, run_variant_fn=_runner_returning(0.94, 0.90))
    assert out["verdict"]["verdict"] == "pass" and out["verdict"]["improved"] is True
    assert out["baseline"].metric_value == 0.94 and out["candidate"].metric_value == 0.90
    assert events == [
        "benchmark_selected", "cloned", "boundary_inferred", "baseline_done", "patch_ready",
        "candidate_done", "reproduce_done", "verdict", "report"]
    assert "train.py" in out["report"]                 # the changed file is named in the report
    assert "prepare.py" in out["report"]               # the protected (restored) file is disclosed
    # no seed_env inferred here -> reproduce is honestly labelled a same-seed rerun, not a seed swap
    assert "同种子重跑" in out["report"] and "run0" in out["report"]


def test_improvement_that_does_not_reproduce_fails(tmp_path):
    # wins on seed0 but not seed1 -> not robust -> honest fail
    out = run_repo_improve(
        _FakeLLM([_IS_REPO, _INFER, _INFER, _GOOD_PATCH]), "improve the tiny-lm training",
        workroot=str(tmp_path), clone_fn=_fake_clone,
        run_variant_fn=_runner_per_seed({
            ("baseline", 0): 0.94, ("candidate", 0): 0.90,     # seed0: improves
            ("baseline", 1): 0.94, ("candidate", 1): 0.95}))   # seed1: worse
    assert out["verdict"]["improved"] is False and "未复现" in out["verdict"]["reason"]
    # P6: the ACTUAL reproduce-run numbers are recorded in the reason (auditable), not just a
    # "did not replicate" label -- here seed1's 0.94->0.95 (candidate got worse) must be present.
    assert "0.94->0.95" in out["verdict"]["reason"]


def test_reproduce_can_be_disabled(tmp_path):
    events = []
    out = run_repo_improve(
        _FakeLLM([_IS_REPO, _INFER, _INFER, _GOOD_PATCH]), "improve the tiny-lm training",
        workroot=str(tmp_path), emit=lambda e, p: events.append(e), reproduce=False,
        clone_fn=_fake_clone, run_variant_fn=_runner_returning(0.94, 0.90))
    assert out["verdict"]["improved"] is True
    assert "reproduce_done" not in events


def test_full_flow_honest_negative(tmp_path):
    # candidate is WORSE -> honest "not improved", not faked
    out = run_repo_improve(
        _FakeLLM([_IS_REPO, _INFER, _INFER, _GOOD_PATCH]), "提升 tiny-lm 的训练",
        workroot=str(tmp_path), clone_fn=_fake_clone,
        run_variant_fn=_runner_returning(0.94, 0.97))
    assert out["verdict"]["verdict"] == "fail" and out["verdict"]["improved"] is False
    assert "未改进" in out["verdict"]["reason"]


def test_unmatched_request_returns_error(tmp_path):
    out = run_repo_improve(_FakeLLM([]), "book me a flight to Tokyo",
                           workroot=str(tmp_path), clone_fn=_fake_clone,
                           run_variant_fn=_runner_returning(0.9, 0.8))
    assert "error" in out and out.get("verdict") is None


_IS_REPO = json.dumps({"is_repo": True,
                       "repo_url": "https://github.com/some-owner/tiny-lm",
                       "name": "tiny-lm"})
_INFER = json.dumps({"editable_files": ["train.py"], "protected_files": ["prepare.py"],
                     "metric": "val_bpb", "lower_is_better": True,
                     "prepare_cmd": "python prepare.py", "run_cmd": "python train.py",
                     "budget_s": 300})


def test_baseline_phase_stops_before_proposal(monkeypatch, tmp_path):
    """Catch baseline-only execution leaking into method proposal or candidate execution."""
    from autoresearch.experiments.repo_improve import pipeline as P

    calls = []

    def run_pristine(benchmark, work_dir, **kwargs):
        calls.append(os.path.basename(work_dir))
        return RunResult(benchmark.metric, 0.94, True)

    def proposal_must_not_run(*args, **kwargs):
        raise AssertionError("baseline phase called proposal")

    monkeypatch.setattr(P, "propose_patch", proposal_must_not_run)
    out = P.run_repo_improve(
        _FakeLLM([_IS_REPO, _INFER]), "measure the tiny-lm baseline",
        workroot=str(tmp_path), clone_fn=_fake_clone, run_variant_fn=run_pristine,
        execution_phase="baseline",
    )

    assert calls == ["baseline_ws"]
    assert out["baseline"].metric_value == 0.94
    assert out["candidate"] is None
    assert out["patch"] is None
    assert out["verdict"] == {
        "verdict": "pass", "improved": False,
        "reason": "baseline measured: val_bpb=0.94",
    }


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_baseline_phase_rejects_nonfinite_pristine_metric(monkeypatch, tmp_path, value):
    """A runner's valid flag cannot make a non-finite baseline pass the hard gate."""
    from autoresearch.experiments.repo_improve import pipeline as P

    monkeypatch.setattr(
        P, "propose_patch",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("invalid baseline reached proposal")),
    )
    out = P.run_repo_improve(
        _FakeLLM([_IS_REPO, _INFER]), "measure the tiny-lm baseline",
        workroot=str(tmp_path), clone_fn=_fake_clone,
        run_variant_fn=lambda benchmark, work_dir, **kwargs: RunResult(
            benchmark.metric, value, True),
        execution_phase="baseline",
    )

    assert out["baseline"].valid is False
    assert out["candidate"] is None
    assert out["verdict"]["verdict"] == "fail"
    assert out["verdict"]["improved"] is False
    assert "finite baseline" in out["verdict"]["reason"]


def test_method_phase_uses_frozen_comparator_without_pristine_run(monkeypatch, tmp_path):
    """Catch method-only execution rerunning or replacing its frozen comparator."""
    from autoresearch.experiments.repo_improve import pipeline as P

    calls = []

    def run_candidate(benchmark, work_dir, **kwargs):
        workspace = os.path.basename(work_dir)
        calls.append(workspace)
        if workspace == "baseline_ws":
            raise AssertionError("method phase reran pristine baseline")
        return RunResult(benchmark.metric, 0.90, True)

    monkeypatch.setattr(P, "recall_setup", lambda *args, **kwargs: {
        "prepare_cmd": "python prepare.py", "run_cmd": "python train.py",
        "metric": "stale_recalled_metric", "lower_is_better": False,
        "verified_in": "host", "recorded_at": "earlier",
    })
    monkeypatch.setattr(
        P, "open_agent_sandbox",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("method phase opened a sandbox to rerun baseline")),
    )
    out = P.run_repo_improve(
        _FakeLLM([_IS_REPO, _INFER, _INFER, _GOOD_PATCH]), "improve the tiny-lm training",
        workroot=str(tmp_path), clone_fn=_fake_clone, run_variant_fn=run_candidate,
        setup_memory=object(), force_sandbox=True,
        execution_phase="method", frozen_baseline={
            "metric_value": 0.94,
            "metric_name": "frozen_val_bpb",
            "lower_is_better": True,
        },
    )

    assert calls == ["candidate_ws"]
    assert out["baseline"].metric_value == 0.94
    assert out["baseline"].metric_name == "frozen_val_bpb"
    assert out["candidate"].metric_value == 0.90
    assert out["metric"] == "frozen_val_bpb"
    assert out["verdict"]["improved"] is True


@pytest.mark.parametrize("value", [None, float("nan"), float("inf"), float("-inf")])
def test_method_phase_rejects_nonfinite_frozen_metric_before_clone(tmp_path, value):
    """Method-only execution fails closed before external work without a real comparator."""
    clone_called = False

    def clone_must_not_run(*args, **kwargs):
        nonlocal clone_called
        clone_called = True
        raise AssertionError("invalid frozen baseline reached clone")

    with pytest.raises(ValueError, match="finite metric_value"):
        run_repo_improve(
            _FakeLLM([]), "improve the tiny-lm training", workroot=str(tmp_path),
            clone_fn=clone_must_not_run, run_variant_fn=_runner_returning(0.94, 0.90),
            execution_phase="method", frozen_baseline={"metric_value": value},
        )

    assert clone_called is False


def test_dynamic_repo_full_flow(tmp_path):
    # a request for an UNREGISTERED repo: LLM names the URL (discover) -> clone -> LLM infers
    # the trusted-eval boundary (infer_benchmark) -> patch -> honest verdict. No static registry.
    events = []
    out = run_repo_improve(
        _FakeLLM([_IS_REPO, _INFER, _INFER, _GOOD_PATCH]),
        "improve the some-owner/tiny-classifier accuracy on its benchmark",
        workroot=str(tmp_path), emit=lambda e, p: events.append(e),
        clone_fn=_fake_clone, run_variant_fn=_runner_returning(0.94, 0.90))
    assert out["verdict"]["improved"] is True
    assert "boundary_inferred" in events          # dynamic path ran
    assert events.index("boundary_inferred") < events.index("baseline_done")
    assert out["benchmark"] == "tiny-lm"


def test_uninferable_boundary_degrades_to_permissive(tmp_path):
    # meeting 2026-07-25 item 4: strict inference names no editable file, but the repo HAS code ->
    # DEGRADE to a permissive boundary (the repo's own code is editable) and PROCEED, rather than
    # refuse the task (the old fail-closed behavior that killed SOL/nanoGPT under glm).
    events = []
    no_editable = json.dumps({"editable_files": [], "protected_files": ["prepare.py"],
                              "metric": "m", "lower_is_better": True,
                              "prepare_cmd": "true", "run_cmd": "python train.py",
                              "budget_s": 300})
    run_repo_improve(
        _FakeLLM([_IS_REPO, no_editable]),
        "improve the some-owner/tiny-classifier accuracy on its benchmark",
        workroot=str(tmp_path), emit=lambda e, p: events.append(e),
        clone_fn=_fake_clone, run_variant_fn=_runner_returning(0.94, 0.90))
    assert "boundary_permissive_fallback" in events   # degraded instead of refusing
    assert "baseline_done" in events                  # the experiment actually ran


def _fake_clone_no_code(repo_url, dest, **kw):
    Path(dest).mkdir(parents=True, exist_ok=True)
    Path(dest, "README.md").write_text("just docs, no code here\n")
    return dest


def test_no_code_file_still_errors(tmp_path):
    # a repo with NO editable code file at all still fails-closed (there is genuinely nothing to
    # improve) -- the permissive fallback only degrades to code files that EXIST.
    events = []
    no_editable = json.dumps({"editable_files": [], "protected_files": [],
                              "metric": "m", "lower_is_better": True,
                              "prepare_cmd": "true", "run_cmd": "true", "budget_s": 300})
    out = run_repo_improve(
        _FakeLLM([_IS_REPO, no_editable]),
        "improve the some-owner/tiny-classifier accuracy on its benchmark",
        workroot=str(tmp_path), emit=lambda e, p: events.append(e),
        clone_fn=_fake_clone_no_code, run_variant_fn=_runner_returning(0.94, 0.90))
    assert "error" in out and out.get("verdict") is None
    assert "baseline_done" not in events          # never ran an experiment


def test_candidate_only_gets_editable_file(tmp_path):
    # anti-cheat: the patch must only reach train.py; prepare.py stays pristine in the
    # candidate workspace even though the fake clone wrote it.
    run_repo_improve(_FakeLLM([_IS_REPO, _INFER, _INFER, _GOOD_PATCH]), "improve the tiny-lm training",
                     workroot=str(tmp_path), clone_fn=_fake_clone,
                     run_variant_fn=_runner_returning(0.94, 0.90))
    cand_prepare = Path(tmp_path, "candidate_ws", "prepare.py").read_text()
    assert cand_prepare == "# trusted data + evaluate_bpb (protected)\n"  # untouched
    cand_train = Path(tmp_path, "candidate_ws", "train.py").read_text()
    assert "0.0015" in cand_train  # the patch DID reach the editable file


# ---------------------------------------------------------------------------
# 层3 接线:宿主侧自愈耗尽后自动升级到沙箱;沙箱执行器接管后续所有运行。
# ---------------------------------------------------------------------------



class _FakeSandboxBackend:
    """假沙箱:build_sandbox_propose 用。run_agent_task 模拟容器内 agent —— 把本轮 edit 写到
    task 里点名的编辑文件绝对路径(即 <workroot>/agent_roundN/<rel>,已由 prepare 建好),
    build_sandbox_propose 随后从宿主读回。execute_command 处理 cat。"""
    def __init__(self, edits_seq=None):
        self.edits_seq = list(edits_seq or [{"train.py": "improved"}])
        self.agent_calls = 0

    def run_agent_task(self, handle, task, *, llm=None, timeout=None):
        self.agent_calls += 1
        # the agent works in /workspace/repo; record this round's edits for export_workspace to
        # surface (build_sandbox_propose exports that dir back to the host and diffs vs pristine).
        self._last_edits = self.edits_seq.pop(0) if self.edits_seq else {"train.py": "improved"}
        return type("R", (), {"status": "FINISHED"})()

    def export_workspace(self, handle, dest, *, src="/workspace/repo"):
        os.makedirs(dest, exist_ok=True)
        for rel, content in getattr(self, "_last_edits", {}).items():
            p = os.path.join(dest, rel)
            os.makedirs(os.path.dirname(p) or dest, exist_ok=True)
            with open(p, "w", encoding="utf-8") as fh:
                fh.write(content)
        return dest

    def execute_command(self, handle, cmd, *, cwd="/w", timeout=None):
        class _R: exit_code, stdout, stderr, timed_out = 0, "", "", False
        return _R()

    def commit_sandbox(self, handle, image_tag):
        self.committed = image_tag
        return True


def test_pipeline_escalates_to_sandbox_when_host_baseline_keeps_failing(monkeypatch, tmp_path):
    from autoresearch.experiments.repo_improve import pipeline as P

    closed = {"n": 0}
    sandbox_runs = []

    def _fake_escalate(**kw):
        healed = dataclasses.replace(kw["benchmark"], prepare_cmd="uv sync",
                                     run_cmd="uv run python train.py", metric="bpb")

        def _sandbox_run(bench, ws, *, prepare=True, seed=None, cuda_device=None, **_):
            sandbox_runs.append((ws, seed))
            return _runner.RunResult("bpb", 2.0 if "baseline" in ws else 1.0, True)

        def _close():
            closed["n"] += 1
        return healed, _sandbox_run, _close, _FakeSandboxBackend(), object()

    monkeypatch.setattr(P, "escalate_to_sandbox", _fake_escalate)
    monkeypatch.setattr(P, "_correct_run", lambda *a, **k: None)   # 层2 直接放弃

    def _always_fail_on_host(bench, ws, **kw):
        return _runner.RunResult(bench.metric, None, False, error="host cannot run this")

    out = P.run_repo_improve(
        _FakeLLM([_IS_REPO, _INFER, _INFER, json.dumps({"train.py": "faster training code"})]), "improve nanochat", workroot=str(tmp_path),
        run_variant_fn=_always_fail_on_host, clone_fn=_fake_clone,
        reproduce=False, sandbox_escalation=True,
    )

    # 升级后拿到了真实基线,不再是"基线未能跑通"
    assert out["baseline"].valid is True and out["baseline"].metric_value == 2.0
    # 候选由通用优化循环在沙箱里跑(opt_round* 工作区),且改进(1.0<2.0);沙箱只拆一次
    assert any("opt_round" in ws for ws, _ in sandbox_runs)
    assert out["candidate"].valid is True and out["candidate"].metric_value == 1.0
    assert closed["n"] == 1, "沙箱必须被拆掉,且只拆一次"


_SOL_JSONL = (
    '{"workload": "w0", "evaluation": {"status": "passed", "performance": {"latency_ms": 0.10}}}\n'
    '{"workload": "w1", "evaluation": {"status": "passed", "performance": {"latency_ms": 0.20}}}\n'
)


def test_escalated_baseline_heals_wrong_metric_name_by_reparsing(monkeypatch, tmp_path):
    # THE live SOL bug + fix: the sandbox discovery agent names the metric `kernel_latency_us`, but
    # the trusted runner can only read the trace's `evaluation.performance.latency_ms` -> baseline runs
    # yet parses to None. The escalate path (previously missing a self-heal) must now re-infer the
    # metric name from the ACTUAL output and RE-PARSE the retained output -- WITHOUT re-running the
    # (expensive) sandbox baseline.
    from autoresearch.experiments.repo_improve import pipeline as P

    baseline_runs = {"n": 0}

    def _fake_escalate(**kw):
        # escalation returns a benchmark with the WRONG metric name (what the agent inferred).
        healed = dataclasses.replace(kw["benchmark"], prepare_cmd="true",
                                     run_cmd="run_kernel", metric="kernel_latency_us")

        def _sandbox_run(bench, ws, *, prepare=True, seed=None, cuda_device=None, **_):
            # classify by the workspace's OWN dir name (tmp_path itself contains "baseline")
            if os.path.basename(ws.rstrip("/")).startswith("baseline"):
                baseline_runs["n"] += 1
                # runs fine, prints the real traces, but the WRONG metric name -> unparseable -> invalid,
                # with the real output retained so a name-fix can re-parse it.
                return _runner.RunResult(bench.metric, None, False,
                                         error="could not find metric", output=_SOL_JSONL)
            return _runner.RunResult(bench.metric, 0.12, True)   # candidate (opt_round)

        return healed, _sandbox_run, (lambda: None), _FakeSandboxBackend(), object()

    # the self-heal corrects ONLY the metric name to the dotted JSON path (commands unchanged).
    def _heal(_pristine, benchmark, _err, _out, _req, _llm):
        return dataclasses.replace(benchmark, metric="evaluation.performance.latency_ms")

    monkeypatch.setattr(P, "escalate_to_sandbox", _fake_escalate)
    monkeypatch.setattr(P, "_correct_run", _heal)

    def _always_fail_on_host(bench, ws, **kw):
        return _runner.RunResult(bench.metric, None, False, error="host cannot run this")

    out = P.run_repo_improve(
        _FakeLLM([_IS_REPO, _INFER, _INFER, json.dumps({"train.py": "faster"})]),
        "improve the SOL rmsnorm kernel", workroot=str(tmp_path),
        run_variant_fn=_always_fail_on_host, clone_fn=_fake_clone,
        reproduce=False, sandbox_escalation=True,
    )

    # baseline is now VALID at the re-parsed value (0.10+0.20)/2 = 0.15, via the corrected dotted path
    assert out["baseline"].valid is True
    assert out["baseline"].metric_value == pytest.approx(0.15)
    assert out["benchmark"] and out["baseline"].metric_name == "evaluation.performance.latency_ms"
    # and the baseline sandbox run happened ONCE -- the fix re-parsed, it did NOT re-run the baseline.
    assert baseline_runs["n"] == 1


def test_sandbox_is_closed_even_when_the_run_fails_later(monkeypatch, tmp_path):
    from autoresearch.experiments.repo_improve import pipeline as P

    closed = {"n": 0}

    def _fake_escalate(**kw):
        healed = dataclasses.replace(kw["benchmark"], metric="bpb")

        def _sandbox_run(bench, ws, **kwargs):
            # 基线过,候选炸 -> 后面走失败分支,沙箱仍必须拆
            if "baseline" in ws:
                return _runner.RunResult("bpb", 2.0, True)
            raise RuntimeError("boom in candidate")

        return healed, _sandbox_run, (lambda: closed.__setitem__("n", closed["n"] + 1)), _FakeSandboxBackend(), object()

    monkeypatch.setattr(P, "escalate_to_sandbox", _fake_escalate)
    monkeypatch.setattr(P, "_correct_run", lambda *a, **k: None)

    try:
        P.run_repo_improve(
            _FakeLLM([_IS_REPO, _INFER, _INFER, _GOOD_PATCH]), "improve x", workroot=str(tmp_path),
            run_variant_fn=lambda b, w, **k: _runner.RunResult(b.metric, None, False, error="nope"),
            clone_fn=_fake_clone, reproduce=False, sandbox_escalation=True,
        )
    except RuntimeError:
        pass
    assert closed["n"] == 1, "异常路径也必须拆沙箱"


def test_force_sandbox_routes_host_repo_through_agent(monkeypatch, tmp_path):
    # item 6 mandate: even a repo whose baseline runs fine on the HOST (minGPT/nanoGPT) must improve
    # through the in-container agent. force_sandbox opens an agent sandbox, re-runs the baseline in it,
    # and the improve step goes through build_sandbox_propose (a handle now exists).
    from autoresearch.experiments.repo_improve import pipeline as P

    opened = {"n": 0}
    agent_edits = _FakeSandboxBackend(edits_seq=[{"train.py": "agent-edited"}])

    def _fake_open(**kw):
        opened["n"] += 1

        def _sandbox_run(bench, ws, *, prepare=True, seed=None, cuda_device=None, **_):
            base = os.path.basename(ws.rstrip("/"))
            return _runner.RunResult(bench.metric, 0.94 if base.startswith("baseline") else 0.90, True)

        return kw["benchmark"], _sandbox_run, (lambda: None), agent_edits, object()

    monkeypatch.setattr(P, "open_agent_sandbox", _fake_open)

    # host baseline runs fine (0.94) -- no escalation would ever trigger; force_sandbox must still fire.
    out = P.run_repo_improve(
        _FakeLLM([_IS_REPO, _INFER, _INFER, _GOOD_PATCH]), "improve the tiny-lm training",
        workroot=str(tmp_path), clone_fn=_fake_clone,
        run_variant_fn=lambda b, w, **k: _runner.RunResult(b.metric, 0.94, True),
        reproduce=False, force_sandbox=True,
    )
    assert opened["n"] == 1                         # the agent sandbox WAS opened for a host repo
    assert agent_edits.agent_calls >= 1             # improve went through the in-container agent
    assert out["baseline"].metric_value == 0.94 and out["candidate"].metric_value == 0.90


def test_force_sandbox_degrades_to_host_when_sandbox_cannot_reproduce(monkeypatch, tmp_path):
    # honest degradation: if the forced sandbox can't reproduce the baseline, keep the VALID host
    # result + host route rather than failing a good baseline just to force the agent path.
    from autoresearch.experiments.repo_improve import pipeline as P

    def _fake_open(**kw):
        def _sandbox_run(bench, ws, *, prepare=True, seed=None, cuda_device=None, **_):
            return _runner.RunResult(bench.metric, None, False, error="sandbox missing CUDA")
        return kw["benchmark"], _sandbox_run, (lambda: None), _FakeSandboxBackend(), object()

    monkeypatch.setattr(P, "open_agent_sandbox", _fake_open)
    monkeypatch.setattr(P, "_correct_run", lambda *a, **k: None)   # no metric-name fix available

    out = P.run_repo_improve(
        _FakeLLM([_IS_REPO, _INFER, _INFER, _GOOD_PATCH]), "improve the tiny-lm training",
        workroot=str(tmp_path), clone_fn=_fake_clone,
        run_variant_fn=_runner_returning(0.94, 0.90),   # host runs both fine
        reproduce=False, force_sandbox=True,
    )
    # the host baseline (0.94) survived; the run did NOT fail just because the sandbox couldn't repro
    assert out["baseline"].valid is True and out["baseline"].metric_value == 0.94
    assert out["candidate"].metric_value == 0.90


def test_escalation_is_opt_out(monkeypatch, tmp_path):
    from autoresearch.experiments.repo_improve import pipeline as P

    called = {"n": 0}
    monkeypatch.setattr(P, "escalate_to_sandbox",
                        lambda **kw: called.__setitem__("n", called["n"] + 1) or None)
    monkeypatch.setattr(P, "_correct_run", lambda *a, **k: None)

    out = P.run_repo_improve(
        _FakeLLM([_IS_REPO, _INFER, _INFER, _GOOD_PATCH]), "improve x", workroot=str(tmp_path),
        run_variant_fn=lambda b, w, **k: _runner.RunResult(b.metric, None, False, error="nope"),
        clone_fn=_fake_clone, reproduce=False, sandbox_escalation=False,
    )
    assert called["n"] == 0
    assert out["verdict"]["verdict"] == "fail"      # 诚实失败,不偷偷升级


# ---------------------------------------------------------------------------
# 记忆横切层:层3 成功后固化;下次同仓库+同上下文先复用(命中就不必再开沙箱)。
# ---------------------------------------------------------------------------


def test_successful_escalation_is_remembered(monkeypatch, tmp_path):
    from autoresearch.experiments.repo_improve import pipeline as P
    from autoresearch.experiments.repo_improve.setup_memory import SetupMemory, recall_setup

    mem = SetupMemory(str(tmp_path / "mem.json"))

    def _fake_escalate(**kw):
        healed = dataclasses.replace(kw["benchmark"], prepare_cmd="uv sync",
                                     run_cmd="uv run python train.py", metric="bpb")
        return healed, (lambda b, ws, **k: _runner.RunResult("bpb", 2.0, True)), (lambda: None), _FakeSandboxBackend(), object()

    monkeypatch.setattr(P, "escalate_to_sandbox", _fake_escalate)
    monkeypatch.setattr(P, "_correct_run", lambda *a, **k: None)

    P.run_repo_improve(
        _FakeLLM([_IS_REPO, _INFER, _INFER, json.dumps({"train.py": "faster training code"})]), "improve nanochat",
        workroot=str(tmp_path), clone_fn=_fake_clone, reproduce=False,
        run_variant_fn=lambda b, w, **k: _runner.RunResult(b.metric, None, False, error="nope"),
        sandbox_escalation=True, setup_memory=mem, cuda_device="3",
    )

    hit = recall_setup(mem, repo_url="https://github.com/some-owner/tiny-lm", budget_s=300, gpu="3")
    assert hit is not None and hit["run_cmd"] == "uv run python train.py"
    assert hit["evidence"]["metric_value"] == 2.0      # 带证据入库


def test_remembered_setup_is_rerun_in_its_image_not_on_the_host(monkeypatch, tmp_path):
    """记忆里的 setup 是在沙箱里验证过的(.venv 等宿主没有),命中后必须直接回到镜像里跑
    已知命令 —— 既不在宿主试(必挂),也不再开发现 agent 沙箱(那才是要省的 20 分钟)。"""
    from autoresearch.experiments.repo_improve import pipeline as P
    from autoresearch.experiments.repo_improve.setup_memory import SetupMemory, remember_setup

    mem = SetupMemory(str(tmp_path / "mem.json"))
    remember_setup(mem, repo_url="https://github.com/some-owner/tiny-lm",
                   prepare_cmd="uv sync", run_cmd="uv run python train.py",
                   metric="bpb", lower_is_better=True, metric_value=2.0,
                   budget_s=300, gpu="3", verified_in="sandbox",
                   image="autoresearch-setup-deadbeef")

    escalated = {"n": 0}
    monkeypatch.setattr(P, "escalate_to_sandbox",
                        lambda **kw: escalated.__setitem__("n", escalated["n"] + 1) or None)

    known_image = {"cmd": None, "image": None}

    def _fake_known_image(*, workroot, benchmark, cuda_device, emit, server_image, env=None):
        known_image["cmd"] = benchmark.run_cmd
        known_image["image"] = server_image

        def _sandbox_run(bench, ws, *, prepare=True, seed=None, cuda_device=None, **_):
            return _runner.RunResult(bench.metric, 2.0 if "baseline" in ws else 1.5, True)
        return benchmark, _sandbox_run, (lambda: None), _FakeSandboxBackend(), object()

    monkeypatch.setattr(P, "run_in_known_image", _fake_known_image)

    host_runs = []

    def _runner_fn(bench, ws, **kw):        # the HOST runner -- must NOT see the sandbox setup
        host_runs.append(bench.run_cmd)
        return _runner.RunResult(bench.metric, None, False, error="host cannot run this")

    out = P.run_repo_improve(
        _FakeLLM([_IS_REPO, _INFER, _INFER, json.dumps({"train.py": "faster training code"})]), "improve nanochat",
        workroot=str(tmp_path), clone_fn=_fake_clone, reproduce=False,
        run_variant_fn=_runner_fn, sandbox_escalation=True, setup_memory=mem, cuda_device="3",
    )

    assert known_image["cmd"] == "uv run python train.py", "记忆里的命令应在镜像里直接采用"
    assert known_image["image"] == "autoresearch-setup-deadbeef", "应在冻结的那个镜像里跑"
    assert escalated["n"] == 0, "命中记忆就不该再开发现 agent 沙箱"
    assert "uv run python train.py" not in host_runs, "沙箱验证过的 setup 不该在宿主试"
    assert out["baseline"].valid is True


def test_stale_recall_falls_back_to_full_rediscovery(monkeypatch, tmp_path):
    """记忆是 HINT 不是真理:仓库变了、命令跑不通时,关掉这个镜像沙箱、回落到完整发现。"""
    from autoresearch.experiments.repo_improve import pipeline as P
    from autoresearch.experiments.repo_improve.setup_memory import SetupMemory, remember_setup

    mem = SetupMemory(str(tmp_path / "mem.json"))
    remember_setup(mem, repo_url="https://github.com/some-owner/tiny-lm",
                   prepare_cmd="uv sync", run_cmd="stale command",
                   metric="bpb", lower_is_better=True, metric_value=2.0,
                   budget_s=300, gpu="3", verified_in="sandbox",
                   image="autoresearch-setup-stale00")

    closed = {"n": 0}

    def _fake_known_image(*, workroot, benchmark, cuda_device, emit, server_image, env=None):
        def _sandbox_run(bench, ws, *, prepare=True, seed=None, cuda_device=None, **_):
            return _runner.RunResult(bench.metric, None, False, error="repo moved")  # 记忆已失效
        return benchmark, _sandbox_run, (lambda: closed.__setitem__("n", closed["n"] + 1)), \
            _FakeSandboxBackend(), object()

    monkeypatch.setattr(P, "run_in_known_image", _fake_known_image)

    escalated = {"n": 0}

    def _fake_escalate(**kw):
        escalated["n"] += 1
        healed = dataclasses.replace(kw["benchmark"], prepare_cmd="uv sync",
                                     run_cmd="fresh command", metric="bpb")

        def _sandbox_run(bench, ws, *, prepare=True, seed=None, cuda_device=None, **_):
            return _runner.RunResult("bpb", 2.0 if "baseline" in ws else 1.0, True)
        return healed, _sandbox_run, (lambda: None), _FakeSandboxBackend(), object()

    monkeypatch.setattr(P, "escalate_to_sandbox", _fake_escalate)

    out = P.run_repo_improve(
        _FakeLLM([_IS_REPO, _INFER, _INFER, json.dumps({"train.py": "faster"})]), "improve nanochat",
        workroot=str(tmp_path), clone_fn=_fake_clone, reproduce=False,
        run_variant_fn=(lambda b, ws, **k: _runner.RunResult(b.metric, None, False, error="host")),
        sandbox_escalation=True, setup_memory=mem, cuda_device="3",
    )

    assert closed["n"] == 1, "失效的镜像沙箱应被关掉"
    assert escalated["n"] == 1, "记忆失效应回落到完整发现"
    assert out["baseline"].valid is True


def test_discovery_freezes_the_container_into_a_recallable_image(monkeypatch, tmp_path):
    """发现成功后必须 docker commit 冻结容器成镜像,并把镜像名连同 setup 一起存进记忆 ——
    下次 recall 才能在同一个环境里重放(光靠两条命令重建不出 agent 装的容器级依赖)。"""
    from autoresearch.experiments.repo_improve import pipeline as P
    from autoresearch.experiments.repo_improve.setup_memory import (
        SetupMemory, recall_setup, setup_image_tag)

    mem = SetupMemory(str(tmp_path / "mem.json"))
    backend = _FakeSandboxBackend()

    def _fake_escalate(**kw):
        healed = dataclasses.replace(kw["benchmark"], prepare_cmd="uv sync",
                                     run_cmd="uv run python train.py", metric="bpb")

        def _sandbox_run(bench, ws, *, prepare=True, seed=None, cuda_device=None, **_):
            return _runner.RunResult("bpb", 2.0 if "baseline" in ws else 1.0, True)
        return healed, _sandbox_run, (lambda: None), backend, object()

    monkeypatch.setattr(P, "escalate_to_sandbox", _fake_escalate)

    out = P.run_repo_improve(
        _FakeLLM([_IS_REPO, _INFER, _INFER, json.dumps({"train.py": "faster"})]), "improve tiny-lm",
        workroot=str(tmp_path), clone_fn=_fake_clone, reproduce=False,
        run_variant_fn=(lambda b, ws, **k: _runner.RunResult(b.metric, None, False, error="host")),
        sandbox_escalation=True, setup_memory=mem, cuda_device="3",
    )

    assert getattr(backend, "committed", None), "发现成功后应 docker commit 冻结容器"
    hit = recall_setup(mem, repo_url="https://github.com/some-owner/tiny-lm", budget_s=300, gpu="3")
    assert hit and hit["image"] == backend.committed, "记忆里应存下冻结的镜像名"
    assert hit["image"] == setup_image_tag("https://github.com/some-owner/tiny-lm", 300, "3"), \
        "镜像名应是 (repo,budget,gpu) 的确定性 tag"
    assert out["baseline"].valid is True


def test_boundary_inference_is_retried_before_giving_up(tmp_path):
    """真机抓到:同一个仓库第一次推断成功、重跑却返回 None(LLM 抖动),
    整轮直接判死。边界推断是一次性的 LLM 调用,失败重试一次远比放弃便宜。"""
    from autoresearch.experiments.repo_improve import pipeline as P

    calls = {"n": 0}
    real_infer = P.infer_benchmark

    def _flaky_infer(*a, **kw):
        calls["n"] += 1
        if calls["n"] == 1:
            return None            # 第一次抖动
        return real_infer(*a, **kw)

    P_infer = P.infer_benchmark
    P.infer_benchmark = _flaky_infer
    try:
        out = P.run_repo_improve(
            _FakeLLM([_IS_REPO, _INFER, _INFER, _INFER, _INFER, _GOOD_PATCH]),
            "improve the tiny-lm training", workroot=str(tmp_path),
            clone_fn=_fake_clone, run_variant_fn=_runner_returning(0.94, 0.90),
            reproduce=False, sandbox_escalation=False)
    finally:
        P.infer_benchmark = P_infer

    assert calls["n"] >= 2, "推断失败必须重试"
    assert out.get("error") is None, f"重试后不该再判死: {out.get('error')}"


# ---------------------------------------------------------------------------
# SOL 接入:检测到 SOL 仓库 -> 用适配器给的确定边界(不走 LLM 推断)+ 专用镜像沙箱。
# ---------------------------------------------------------------------------

_SOL_IS_REPO = json.dumps({"is_repo": True,
                           "repo_url": "https://github.com/NVIDIA/SOL-ExecBench",
                           "name": "SOL-ExecBench"})


def _fake_clone_sol(repo_url, dest, **kw):
    import pathlib
    prob = pathlib.Path(dest) / "tests/sol_execbench/samples/rmsnorm"
    prob.mkdir(parents=True, exist_ok=True)
    (prob / "definition.json").write_text('{"name":"rmsnorm_h4096"}')
    (prob / "workload.jsonl").write_text('{"uuid":"x"}\n')
    (prob / "solution_cuda.json").write_text(json.dumps(
        {"name": "s", "sources": [{"path": "kernel.cu", "content": "__global__ void k(){}"}]}))
    (pathlib.Path(dest) / "src/sol_execbench").mkdir(parents=True, exist_ok=True)
    (pathlib.Path(dest) / "src/sol_execbench/sol_score.py").write_text("# scorer")
    return dest


def test_repo_needing_its_own_image_goes_through_general_infer_no_special_case(monkeypatch, tmp_path):
    """SOL-类仓库(实现嵌在 JSON、harness 打分、要 CUDA 镜像)现在走的是和 train.py 仓库
    完全相同的通用 infer_benchmark 路径 —— 通用推断出 editable=solution JSON + image=build:...,
    pipeline 通用地把 image 解析成镜像并路由进去。零 SOL 特判。"""
    from autoresearch.experiments.repo_improve import pipeline as P
    from autoresearch.experiments.repo_improve.benchmark import Benchmark

    # 通用 infer 对 SOL-类仓库返回:editable=JSON 实现、protected=harness、点路径指标、需要自带镜像。
    infer_called = {"n": 0}
    def _fake_infer(pristine, request, stub, llm, overrides=None):
        infer_called["n"] += 1
        return dataclasses.replace(
            stub,
            editable_files=("samples/rmsnorm/solution_cuda.json",),
            protected_files=("src/harness.py", "samples/rmsnorm/definition.json"),
            prepare_cmd="mkdir -p /tmp/trace && uv sync",
            run_cmd="FLASHINFER_TRACE_DIR=/tmp/trace uv run bench --json",
            metric="evaluation.performance.latency_ms", lower_is_better=True,
            image="build:docker/Dockerfile")
    monkeypatch.setattr(P, "infer_benchmark", _fake_infer)

    # 通用镜像解析(不真 build docker):把 build:... 解析成一个已构建的 tag。
    resolved = {}
    def _fake_resolve(spec, pristine, emit):
        resolved["spec"] = spec
        return "autoresearch-repo-deadbeef"
    monkeypatch.setattr(P, "_resolve_repo_image", _fake_resolve)

    captured = {}
    def _fake_known_image(**kw):
        captured.update(kw)
        b = kw["benchmark"]
        return (b, (lambda bm, ws, **k: _runner.RunResult(
            bm.metric, 0.005 if "baseline" in ws else 0.004, True)), lambda: None,
            _FakeSandboxBackend([{"samples/rmsnorm/solution_cuda.json": "faster"}]), object())
    monkeypatch.setattr(P, "run_in_known_image", _fake_known_image)

    _edit = json.dumps({"samples/rmsnorm/solution_cuda.json": '{"sources": [{"content": "faster"}]}'})
    out = P.run_repo_improve(
        _FakeLLM([_SOL_IS_REPO, _edit]),
        "improve sol-execbench rmsnorm kernel latency",
        workroot=str(tmp_path), clone_fn=_fake_clone_sol, reproduce=False,
        run_variant_fn=lambda b, w, **k: _runner.RunResult(b.metric, None, False, error="host cannot"),
        cuda_device="3")

    # 走了通用 infer(不是特判);image 经通用解析路由进 run_in_known_image。
    assert infer_called["n"] >= 1
    assert resolved["spec"] == "build:docker/Dockerfile"
    assert captured.get("server_image") == "autoresearch-repo-deadbeef"
    b = captured["benchmark"]
    assert b.metric == "evaluation.performance.latency_ms"
    assert any("solution_cuda.json" in f for f in b.editable_files)
    assert out["baseline"].metric_value == 0.005 and out["candidate"].metric_value == 0.004
