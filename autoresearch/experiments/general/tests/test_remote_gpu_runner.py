"""remote_gpu_runner: transport is fully mocked (no ssh/scp/network). Verifies the
command it builds (pinned CUDA card + trusted orchestrator), result parsing, and the
fail-closed contract on a non-zero remote exit.
"""
from __future__ import annotations

import json

from autoresearch.experiments.general.remote_gpu_runner import run_experiment_remote


class _R:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode, self.stdout, self.stderr = returncode, stdout, stderr


def _fake(calls, *, orch_rc=0, result=None):
    """A subprocess.run stand-in that answers each ssh/scp by inspecting the command."""
    result = result or {"baseline_metric": 1.0, "method_metric": 0.5,
                        "valid": True, "seed": 0}

    def run(argv, **kw):
        calls.append(argv)
        if argv[0] == "scp":
            return _R(0)
        cmd = argv[-1]                       # ssh ... <host> <cmd>
        if cmd == "echo $HOME":
            return _R(0, stdout="/home/u\n")
        if cmd.startswith("rm -rf") or cmd.startswith("rm -rf") or "mkdir" in cmd:
            return _R(0)
        if "_orchestrator.py" in cmd:
            return _R(orch_rc, stdout="", stderr="boom" if orch_rc else "")
        if "result.json" in cmd:            # the cat
            return _R(0, stdout=json.dumps(result))
        return _R(0)
    return run


def test_remote_runner_returns_trusted_result_and_pins_card():
    calls = []
    out = run_experiment_remote(
        "def make_instances(s): ...", "def solve(i): return []", seed=3,
        host="dgx-x", python_bin="/opt/py", cuda_device="7",
        io_contract={"x": 1}, runner=_fake(calls),
    )
    assert out["valid"] is True and out["isolated"] is True
    assert out["baseline_metric"] == 1.0 and out["method_metric"] == 0.5
    # the run pinned the card and drove the TRUSTED orchestrator (not the AI harness)
    joined = " ".join(c[-1] for c in calls if c[0] == "ssh")
    assert "CUDA_VISIBLE_DEVICES=7" in joined
    assert "_orchestrator.py 3" in joined
    assert "/opt/py" in joined


def test_remote_runner_fails_closed_on_nonzero_orchestrator_exit():
    out = run_experiment_remote(
        "h", "m", seed=1, host="h", cuda_device="0",
        runner=_fake([], orch_rc=1),
    )
    assert out["valid"] is False and out["isolated"] is False
    assert out["baseline_metric"] is None and "boom" in (out["error"] or "")


def test_remote_runner_fails_closed_on_unreadable_result():
    def run(argv, **kw):
        cmd = argv[-1] if argv[0] == "ssh" else ""
        if cmd == "echo $HOME":
            return _R(0, stdout="/home/u\n")
        if "result.json" in cmd:
            return _R(0, stdout="not json")
        return _R(0)
    out = run_experiment_remote("h", "m", seed=0, host="h", runner=run)
    assert out["valid"] is False and out["isolated"] is False


# --- connection multiplexing + retry (fix for the ProxyJump SSH-burst refusal) --------
import types  # noqa: E402

from autoresearch.experiments.general import remote_gpu_runner as _rgr  # noqa: E402


def _no_sleep(monkeypatch):
    monkeypatch.setattr(_rgr, "time", types.SimpleNamespace(sleep=lambda s: None))


def test_remote_runner_retries_on_connection_refusal(monkeypatch):
    _no_sleep(monkeypatch)
    calls = []
    state = {"mkdir_fails": 2}  # first 2 mkdir handshakes refused, 3rd succeeds

    def run(argv, **kw):
        calls.append(argv)
        if argv[0] == "scp":
            return _R(0)
        cmd = argv[-1]
        if "mkdir" in cmd:
            if state["mkdir_fails"] > 0:
                state["mkdir_fails"] -= 1
                return _R(255, stderr="kex_exchange_identification: Connection closed by remote host")
            return _R(0)
        if "_orchestrator.py" in cmd:
            return _R(0)
        if "result.json" in cmd:
            return _R(0, stdout=json.dumps({"baseline_metric": 1.0, "method_metric": 0.5,
                                            "valid": True, "seed": 0}))
        return _R(0)

    out = run_experiment_remote("h", "m", seed=0, host="h", runner=run)
    assert out["valid"] is True and out["isolated"] is True
    mkdir_calls = [c for c in calls if c[0] == "ssh" and "mkdir" in c[-1]]
    assert len(mkdir_calls) == 3      # 2 refusals retried, then success


def test_remote_runner_does_not_retry_a_clean_orchestrator_failure(monkeypatch):
    _no_sleep(monkeypatch)
    calls = []

    def run(argv, **kw):
        calls.append(argv)
        if argv[0] == "scp":
            return _R(0)
        cmd = argv[-1]
        if "_orchestrator.py" in cmd:
            return _R(1, stderr="AssertionError: method produced no solution")  # real failure
        return _R(0)

    out = run_experiment_remote("h", "m", seed=0, host="h", runner=run)
    assert out["valid"] is False
    orch = [c for c in calls if c[0] == "ssh" and "_orchestrator.py" in c[-1]]
    assert len(orch) == 1             # a real non-connection failure is NOT retried


def test_remote_runner_multiplexes_over_one_master_and_closes_it(monkeypatch):
    _no_sleep(monkeypatch)
    calls = []
    run_experiment_remote("h", "m", seed=0, host="h", runner=_fake(calls))
    ops = [c for c in calls if c[0] in ("ssh", "scp")]
    paths = set()
    for c in ops:
        assert "ControlMaster=auto" in c
        cp = [a for a in c if a.startswith("ControlPath=")]
        assert cp, f"op missing ControlPath: {c}"
        paths.add(cp[0])
    assert len(paths) == 1            # every op reused the SAME master socket
    # the master was explicitly closed at the end
    assert any(c[0] == "ssh" and "-O" in c and "exit" in c for c in calls)
