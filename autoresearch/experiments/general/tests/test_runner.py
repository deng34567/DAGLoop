"""Task 4: the Docker runner -- assemble (string -> tempfile -> upload) + upload
the FIXED trusted orchestrator + isolator + drive the run via ``_orchestrator.py``.
FULLY MOCKED workspace. ``isolated`` is STRUCTURAL: it is True iff the trusted
orchestrator ran to completion (exit 0) -- NOT a spoofable textual check.
"""
from __future__ import annotations

import json
from pathlib import Path

from autoresearch.agents._reuse import openhands_backend as ob
from autoresearch.experiments.general import runner as gen_runner

_HARNESS = (
    "def make_instances(seed): return [{'n': 1}]\n"
    "def run_baseline(instances): return 10\n"
    "def validate(instance, solution): return True\n"
    "def score(instances, solutions): return 8\n"
)
_METHOD = "def solve(instance):\n    return []\n"


class _Workspace:
    """Records uploads (with the LOCAL file's content, read at call time) and
    answers execute_command per a script."""

    def __init__(self, *, result_json, run_exit_code=0):
        self._result_json = result_json
        self._run_exit_code = run_exit_code
        self.uploads = []          # (remote, local_content)
        self.commands = []

    def file_upload(self, local, remote):
        try:
            content = Path(local).read_text()
        except Exception:
            content = None
        self.uploads.append((str(remote), content))

    def execute_command(self, cmd, cwd=None, timeout=None):
        self.commands.append(cmd)
        exit_code, stdout = 0, ""
        if "_orchestrator.py" in cmd:
            exit_code = self._run_exit_code
        elif "cat" in cmd and "result.json" in cmd:
            stdout = json.dumps(self._result_json)
        return type("R", (), {"exit_code": exit_code, "stdout": stdout, "stderr": "err",
                              "timeout_occurred": False})()


def _remotes(ws):
    return {r for r, _ in ws.uploads}


def test_run_experiment_uploads_strings_plus_trusted_orchestrator_and_isolator():
    ws = _Workspace(result_json={"baseline_metric": 10, "method_metric": 8,
                                 "valid": True, "seed": 0})
    handle = ob.SandboxHandle(workspace=ws)

    out = gen_runner.run_experiment(ob.OpenHandsBackend(), _HARNESS, _METHOD,
                                    seed=0, handle=handle)

    # harness/method strings were written to LOCAL temp files then uploaded (content round-trips)
    up = dict(ws.uploads)
    assert up["/workspace/harness.py"] == _HARNESS
    assert up["/workspace/method.py"] == _METHOD
    # the FIXED trusted orchestrator + isolator modules were uploaded from the project
    for name in ("_orchestrator.py", "isolated_method.py", "_method_driver.py"):
        assert f"/workspace/{name}" in _remotes(ws)
    # stale result.json cleared BEFORE running; the TRUSTED ORCHESTRATOR (not the
    # harness) is what runs, with the seed as argv
    assert any("rm -f" in c and "result.json" in c for c in ws.commands)
    assert any("_orchestrator.py 0" in c for c in ws.commands)
    assert not any("python harness.py" in c for c in ws.commands)  # harness never run directly
    # result surfaced + isolation STRUCTURAL True (trusted path ran to completion)
    assert out["baseline_metric"] == 10 and out["method_metric"] == 8
    assert out["valid"] is True
    assert out["isolated"] is True


def test_run_experiment_nonzero_exit_is_invalid_not_isolated_and_ignores_stale_result():
    ws = _Workspace(result_json={"baseline_metric": 1, "method_metric": 0,
                                 "valid": True, "seed": 0},
                    run_exit_code=1)
    handle = ob.SandboxHandle(workspace=ws)

    out = gen_runner.run_experiment(ob.OpenHandsBackend(), _HARNESS, _METHOD,
                                    seed=0, handle=handle)

    assert out["valid"] is False
    assert "error" in out
    # the planted result.json (valid=True) is NEVER read on a non-zero exit
    assert not any("cat" in c and "result.json" in c for c in ws.commands)
    assert out.get("method_metric") is None
    # the trusted path did NOT complete -> isolation cannot be claimed
    assert out["isolated"] is False


def test_run_experiment_installs_declared_dependencies():
    ws = _Workspace(result_json={"baseline_metric": 10, "method_metric": 8,
                                 "valid": True, "seed": 0})
    handle = ob.SandboxHandle(workspace=ws)

    gen_runner.run_experiment(ob.OpenHandsBackend(), _HARNESS, _METHOD,
                              seed=0, handle=handle, dependencies=["numpy"])

    assert any(c == "pip install -q numpy" for c in ws.commands)


def test_run_experiment_no_dependencies_issues_no_pip_install():
    ws = _Workspace(result_json={"baseline_metric": 10, "method_metric": 8,
                                 "valid": True, "seed": 0})
    handle = ob.SandboxHandle(workspace=ws)

    gen_runner.run_experiment(ob.OpenHandsBackend(), _HARNESS, _METHOD,
                              seed=0, handle=handle)

    assert not any("pip install" in c for c in ws.commands)


def test_run_experiment_sanitizes_injecting_dependency():
    ws = _Workspace(result_json={"baseline_metric": 10, "method_metric": 8,
                                 "valid": True, "seed": 0})
    handle = ob.SandboxHandle(workspace=ws)

    gen_runner.run_experiment(ob.OpenHandsBackend(), _HARNESS, _METHOD, seed=0,
                              handle=handle, dependencies=["numpy", "$(rm -rf /)"])

    assert any(c == "pip install -q numpy" for c in ws.commands)
    assert not any("rm -rf" in c for c in ws.commands)


def test_is_isolated_textual_heuristic_is_gone():
    # the spoofable textual check must not exist as an isolation guarantee anymore
    assert not hasattr(gen_runner, "_is_isolated")
