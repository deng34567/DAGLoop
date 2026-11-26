"""Trusted runner: metric parsing, fail-closed contract, GPU pinning (fake executor)."""
from __future__ import annotations

import pytest

import dataclasses

from autoresearch.experiments.repo_improve.benchmark import EXAMPLE_BENCHMARK
from autoresearch.experiments.repo_improve.runner import (
    RunResult, _parse_metric, reparse_metric, run_variant)


# One SOL-style run: the eval driver prints ONE compact JSON trace per workload (json.dumps), with
# the number nested at evaluation.performance.latency_ms -- exactly the live SOL output shape.
_SOL_JSONL = (
    "building kernel...\ncompiled ok\n"
    '{"workload": "w0", "evaluation": {"status": "passed", "performance": {"latency_ms": 0.10}}}\n'
    '{"workload": "w1", "evaluation": {"status": "passed", "performance": {"latency_ms": 0.20}}}\n'
    "done\n"
)


class _CP:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode, self.stdout, self.stderr = returncode, stdout, stderr


def test_parse_metric_takes_last_occurrence():
    assert _parse_metric("val_bpb", "val_bpb: 1.20\n...\nval_bpb = 0.9109") == 0.9109
    assert _parse_metric("val_bpb", "no metric here") is None
    assert _parse_metric("val_bpb", "final val_bpb 0.87 done") == 0.87


def test_parse_metric_separators_are_interchangeable():
    # the inferred name may be underscored while the repo prints it with a space or slash --
    # this is the nanoGPT live failure (`val_loss` inferred, `val loss` printed).
    line = "step 250: train loss 1.9, val loss 1.4677"
    assert _parse_metric("val_loss", line) == 1.4677   # underscore name matches spaced print
    assert _parse_metric("val loss", line) == 1.4677
    assert _parse_metric("val/loss", "val/loss 0.9109") == 0.9109
    assert _parse_metric("test accuracy", "test_accuracy = 0.94") == 0.94
    # a token that genuinely isn't there still returns None (no false positives)
    assert _parse_metric("train_bpb", "step 3: val loss 1.4") is None


def test_run_variant_parses_metric_from_output():
    def executor(cmd, **kw):
        if "prepare" in cmd:
            return _CP(0)
        return _CP(0, stdout="step 100 ...\nval_bpb: 0.9312\n")
    res = run_variant(EXAMPLE_BENCHMARK, "/tmp/ws", executor=executor)
    assert res.valid and res.metric_value == 0.9312


def test_run_variant_fails_closed_on_run_error():
    def executor(cmd, **kw):
        return _CP(0) if "prepare" in cmd else _CP(1, stderr="CUDA error")
    res = run_variant(EXAMPLE_BENCHMARK, "/tmp/ws", executor=executor)
    assert res.valid is False and res.metric_value is None and "exit 1" in res.error


def test_run_variant_fails_closed_when_metric_missing():
    def executor(cmd, **kw):
        return _CP(0, stdout="trained ok but printed no metric")
    res = run_variant(EXAMPLE_BENCHMARK, "/tmp/ws", executor=executor, prepare=False)
    assert res.valid is False and "val_bpb" in res.error


class _CappedCP(_CP):
    """A cap-killed run: nonzero (SIGKILL) returncode but carries partial output + timed_out flag."""
    def __init__(self, stdout="", stderr=""):
        super().__init__(returncode=-9, stdout=stdout, stderr=stderr)
        self.timed_out = True


def test_capped_run_with_metric_is_valid_at_cap():
    # a script that trains UNBOUNDED (minGPT chargpt.py) is hard-killed at the shared wall-clock cap;
    # the FAIR result is the metric it had reached at the cap -- parse it from the partial output.
    def executor(cmd, **kw):
        return _CappedCP(stdout="step 900: val_bpb 1.05\nstep 1000: val_bpb 0.981\n")  # killed here
    res = run_variant(EXAMPLE_BENCHMARK, "/tmp/ws", executor=executor, prepare=False)
    assert res.valid is True and res.metric_value == 0.981   # last metric before the cap-kill


def test_capped_run_without_metric_is_invalid():
    # a cap-kill with NO parseable metric in the partial output is an honest failure, not a win.
    def executor(cmd, **kw):
        return _CappedCP(stdout="loading data...\ninitializing model...\n")   # killed before any metric
    res = run_variant(EXAMPLE_BENCHMARK, "/tmp/ws", executor=executor, prepare=False)
    assert res.valid is False and "cap" in res.error.lower()


def test_genuine_nonzero_exit_still_fails_even_though_capped_path_exists():
    # a real crash (not a cap-kill: no timed_out flag) is still an honest failure with its exit code.
    def executor(cmd, **kw):
        return _CP(1, stdout="step 10: val_bpb 0.9\n", stderr="RuntimeError")  # has a metric BUT crashed
    res = run_variant(EXAMPLE_BENCHMARK, "/tmp/ws", executor=executor, prepare=False)
    assert res.valid is False and "exit 1" in res.error


def test_dotted_metric_path_parses_sol_style_jsonl():
    # the CORRECT name for SOL is the dotted JSON path; _parse_metric averages it over the traces.
    assert _parse_metric("evaluation.performance.latency_ms", _SOL_JSONL) == pytest.approx(0.15)  # (0.10+0.20)/2
    # the HALLUCINATED name the discovery agent produced matches nothing -> None (the live bug).
    assert _parse_metric("kernel_latency_us", _SOL_JSONL) is None


def test_run_variant_retains_full_output_when_metric_unparseable():
    # on a metric-parse miss, run_variant keeps a LARGE slice of the real output (not just 800 chars)
    # so the self-heal can see the actual metric-bearing traces and a name-fix can re-parse them.
    bench = dataclasses.replace(EXAMPLE_BENCHMARK, metric="kernel_latency_us")

    def executor(cmd, **kw):
        return _CP(0, stdout=_SOL_JSONL)
    res = run_variant(bench, "/tmp/ws", executor=executor, prepare=False)
    assert res.valid is False and res.metric_value is None
    assert "latency_ms" in res.output           # the real trace was retained
    assert '"evaluation"' in res.output


def test_reparse_metric_fixes_name_without_rerunning():
    # THE self-heal fast path: a run that failed only because the metric NAME was wrong is fixed by
    # re-parsing its retained output with the corrected dotted path -- no re-run needed.
    failed = RunResult("kernel_latency_us", None, False, error="not found", output=_SOL_JSONL)
    assert reparse_metric(failed, "evaluation.performance.latency_ms") == pytest.approx(0.15)
    # a name that STILL isn't in the output -> None (caller must actually re-run)
    assert reparse_metric(failed, "throughput_gbps") is None
    # no retained output -> nothing to re-parse
    assert reparse_metric(RunResult("m", None, False), "evaluation.performance.latency_ms") is None


def test_run_variant_pins_cuda_device_and_seed():
    seen = {}
    def executor(cmd, **kw):
        seen["env"] = kw.get("env")
        return _CP(0, stdout="val_bpb 0.9\n")
    bench = dataclasses.replace(EXAMPLE_BENCHMARK, seed_env="RI_SEED")
    run_variant(bench, "/tmp/ws", executor=executor, prepare=False, cuda_device="7", seed=1)
    assert seen["env"]["CUDA_VISIBLE_DEVICES"] == "7"
    assert seen["env"]["RI_SEED"] == "1"


def test_trusted_eval_output_is_authoritative(tmp_path):
    # train.py prints a (potentially cheated) number; the trusted re-eval prints the real
    # one -- run_variant must return the TRUSTED value, not train.py's.
    bench = dataclasses.replace(
        EXAMPLE_BENCHMARK,
        trusted_eval_files=(("trusted_eval.py", "# trusted eval, agent never sees this\n"),),
        trusted_eval_cmd="python trusted_eval.py")
    def executor(cmd, **kw):
        if "trusted_eval" in cmd:
            return _CP(0, stdout="val_bpb: 0.9300\n")     # authoritative
        return _CP(0, stdout="val_bpb: 0.5000\n")         # train.py's self-report (ignored)
    res = run_variant(bench, str(tmp_path), executor=executor, prepare=False)
    assert res.valid and res.metric_value == 0.93
    # the trusted eval file was written into the workspace
    assert (tmp_path / "trusted_eval.py").exists()


def test_capped_run_hard_kills_process_tree_on_timeout():
    # fair-budget/anti-cheat: on the wall-clock cap the WHOLE process group is killed, so a training
    # grandchild can't outlive its budget (shell=True + run(timeout) would orphan it). The call now
    # RETURNS a timed_out CompletedProcess (carrying partial output) instead of raising, so the
    # at-cap metric can be read -- but the grandchild is still dead.
    import os, subprocess, time
    from autoresearch.experiments.repo_improve.runner import _capped_run
    r = _capped_run("python -c 'import time; print(\"started\", flush=True); time.sleep(30)'",
                    cwd="/tmp", shell=True, env=dict(os.environ),
                    capture_output=True, text=True, timeout=1)
    assert getattr(r, "timed_out", False) is True   # flagged as a cap-kill
    assert r.returncode != 0                          # SIGKILL, not a clean exit
    assert "started" in (r.stdout or "")              # partial output preserved for metric parsing
    time.sleep(0.5)
    pg = subprocess.run("pgrep -f 'time.sleep(30)' || true", shell=True, capture_output=True, text=True)
    assert "sleep(30)" not in pg.stdout   # no orphaned grandchild survives the budget


# ---------------------------------------------------------------------------
# JSONL trace 指标(SOL-ExecBench 这类):输出不是 "metric: 数字",而是每个 workload
# 一行 JSON。指标名写成点号路径即可;正确性未过的一律不计分。
# ---------------------------------------------------------------------------

import json as _json


def _sol_line(latency, status="PASSED"):
    return _json.dumps({
        "definition": "rmsnorm_h4096",
        "evaluation": {"status": status,
                       "correctness": {"has_nan": False},
                       "performance": {"latency_ms": latency, "speedup_factor": 0.0}},
    })


def test_dotted_path_metric_is_read_from_jsonl_and_averaged():
    out = "\n".join([_sol_line(0.004704), _sol_line(0.005664), _sol_line(0.010192)])
    v = _parse_metric("evaluation.performance.latency_ms", out)
    assert v is not None and abs(v - (0.004704 + 0.005664 + 0.010192) / 3) < 1e-9


def test_a_failed_workload_invalidates_the_whole_metric():
    # 反作弊要害:算得快但正确性没过 -> 不给分数,而不是把它平均进去
    out = "\n".join([_sol_line(0.001), _sol_line(0.002, status="FAILED")])
    assert _parse_metric("evaluation.performance.latency_ms", out) is None


def test_plain_text_metric_still_wins_and_json_is_only_a_fallback():
    # 老路径不受影响:文本里能抓到就用文本(JSONL 只是兜底)
    assert _parse_metric("val_bpb", "val_bpb: 0.91\n") == 0.91
    # 非 JSON 的噪声行不影响解析
    out = "loading...\n" + _sol_line(0.5) + "\nDone.\n"
    assert _parse_metric("evaluation.performance.latency_ms", out) == 0.5


def test_missing_path_or_no_json_returns_none():
    assert _parse_metric("evaluation.performance.latency_ms", "no json here") is None
    assert _parse_metric("a.b.c", _sol_line(0.1)) is None


def test_idempotent_prepare_guards_venv_creation():
    # P4: a prepare that creates a venv must be safe to re-run (else the baseline dies on retry with
    # "a virtual environment already exists"). The venv dir gets an rm -rf prepended; other prepares
    # (and already-guarded ones) are untouched.
    from autoresearch.experiments.repo_improve.runner import _idempotent_prepare
    assert _idempotent_prepare("uv venv .venv && uv pip install -e .") == \
        "rm -rf .venv; uv venv .venv && uv pip install -e ."
    assert _idempotent_prepare("python -m venv env && pip install -r reqs.txt").startswith("rm -rf env;")
    assert _idempotent_prepare("pip install -e .") == "pip install -e ."          # no venv -> untouched
    assert _idempotent_prepare("rm -rf .venv; uv venv .venv") == "rm -rf .venv; uv venv .venv"  # already guarded
    assert _idempotent_prepare("true") == "true"
    assert _idempotent_prepare(None) is None
