"""Tests for the b-layer ``MonteCarloExperimentRunner`` -- FULLY MOCKED.

The default suite NEVER starts a real container or hits an LLM/network: every
test injects a mock workspace via the M9 ``SandboxHandle`` seam and mocks
``file_upload`` / ``execute_command`` / ``run_agent_task``. Real connectivity is
opt-in only (the Task 10 smoke script).
"""
from __future__ import annotations

import json

from autoresearch.agents._reuse import openhands_backend as ob
from autoresearch.experiments import mc_runner


class FakeWorkspace:
    """Records file_upload calls and answers execute_command deterministically.

    ``result_json`` is what ``cat result.json`` returns; other commands exit 0.
    """

    def __init__(self, result_json: dict):
        self._result_json = result_json
        self.uploads: list[tuple[str, str]] = []
        self.commands: list[str] = []

    def file_upload(self, local, remote):
        self.uploads.append((str(local), str(remote)))

    def execute_command(self, cmd, cwd=None, timeout=None):
        self.commands.append(cmd)
        if "cat result.json" in cmd:
            stdout = json.dumps(self._result_json)
        else:
            stdout = ""
        return type(
            "R", (), {"exit_code": 0, "stdout": stdout, "stderr": "", "timeout_occurred": False}
        )()


def _handle(result_json: dict) -> ob.SandboxHandle:
    return ob.SandboxHandle(workspace=FakeWorkspace(result_json))


def test_run_baseline_uploads_trusted_files_and_returns_metric():
    result = {
        "metric_value": 1.0,
        "rmse_baseline": 0.01,
        "rmse_proposed": 0.01,
        "over_budget": False,
        "hardcoded_suspected": False,
    }
    h = _handle(result)
    runner = mc_runner.MonteCarloExperimentRunner(backend=ob.OpenHandsBackend())

    out = runner.run_baseline(handle=h)

    assert out["metric_value"] == 1.0
    ws = h.workspace
    remotes = {r for _, r in ws.uploads}
    # all trusted files + the baseline method.py were uploaded
    for name in mc_runner._TRUSTED + ("method.py",):
        assert f"/workspace/{name}" in remotes
    # scoring actually ran the trusted harness
    assert any("python run.py" in c for c in ws.commands)
    # deps installed for the in-container quadrature evaluator
    assert any("pip install" in c and "scipy" in c for c in ws.commands)


# ------------------------------- Task 5 --------------------------------------

class RecordingWorkspace:
    """Shares one ordered ``events`` log with the backend so call order across
    ``run_agent_task`` / ``file_upload`` / ``execute_command`` is observable."""

    def __init__(self, events, result_json, agent_code="def proposed_estimate(): ..."):
        self.events = events
        self._result_json = result_json
        self._agent_code = agent_code

    def file_upload(self, local, remote):
        self.events.append(("upload", str(remote)))

    def execute_command(self, cmd, cwd=None, timeout=None):
        self.events.append(("cmd", cmd))
        if "cat result.json" in cmd:
            stdout = json.dumps(self._result_json)
        elif "cat method.py" in cmd or "method.py" in cmd and "cat" in cmd:
            stdout = self._agent_code
        else:
            stdout = ""
        return type(
            "R", (), {"exit_code": 0, "stdout": stdout, "stderr": "", "timeout_occurred": False}
        )()


class RecordingBackend(ob.OpenHandsBackend):
    def __init__(self, events, agent_status="FINISHED"):
        super().__init__()
        self.events = events
        self.agent_status = agent_status
        self.last_task = None
        self.last_timeout = None

    def run_agent_task(self, handle, task, *, llm=None, cli_mode=True, timeout=None):
        self.events.append(("agent", task))
        self.last_task = task
        self.last_timeout = timeout
        return ob.RunResult(status=self.agent_status)


def test_run_experiment_agent_writes_method_then_trusted_reupload_then_score():
    events: list = []
    result = {
        "metric_value": 4.6,
        "rmse_baseline": 0.009,
        "rmse_proposed": 0.002,
        "over_budget": False,
        "hardcoded_suspected": False,
        "bias": 0.0001,
    }
    ws = RecordingWorkspace(events, result, agent_code="# agent's variance reduction")
    handle = ob.SandboxHandle(workspace=ws)
    backend = RecordingBackend(events)
    runner = mc_runner.MonteCarloExperimentRunner(backend=backend)

    out = runner.run_experiment("reduce the MC variance", handle=handle)

    # result carries the trusted metric + agent status + the agent's code
    assert out["metric_value"] == 4.6
    assert out["agent_status"] == "FINISHED"
    assert out["code"] == "# agent's variance reduction"

    # the contract prompt mentions the editable file and the budgeted seam
    assert "method.py" in backend.last_task
    assert "sample_f" in backend.last_task

    # ---- CALL ORDER: agent -> re-upload trusted -> score ----
    kinds = events
    pos_agent = next(i for i, e in enumerate(kinds) if e[0] == "agent")
    pos_score = next(
        i for i, e in enumerate(kinds)
        if e[0] == "cmd" and "python run.py" in e[1] and i > pos_agent
    )
    # every trusted file is re-uploaded strictly between the agent run and scoring
    for name in mc_runner._TRUSTED:
        reupload = [
            i for i, e in enumerate(kinds)
            if e[0] == "upload" and e[1] == f"/workspace/{name}" and pos_agent < i < pos_score
        ]
        assert reupload, f"{name} not re-uploaded before scoring"
    # method.py is NOT re-uploaded after the agent (agent's method is preserved)
    assert not any(
        e[0] == "upload" and e[1] == "/workspace/method.py" and i > pos_agent
        for i, e in enumerate(kinds)
    )


def test_run_experiment_withholds_integrand_then_reuploads_before_each_score():
    """M11 core: integrand.py is WITHHELD while the agent authors method.py, and
    RE-UPLOADED before EACH scoring run (run.py deletes it after import). The
    reproduce run passes fresh, disjoint seeds INLINE. Both metrics returned."""
    events: list = []
    result = {
        "metric_value": 3.5, "rmse_baseline": 0.009, "rmse_proposed": 0.0026,
        "over_budget": False, "hardcoded_suspected": False, "bias": 0.0,
    }
    ws = RecordingWorkspace(events, result, agent_code="# agent method")
    handle = ob.SandboxHandle(workspace=ws)
    backend = RecordingBackend(events)
    runner = mc_runner.MonteCarloExperimentRunner(backend=backend)

    out = runner.run_experiment("reduce variance", handle=handle)

    pos_agent = next(i for i, e in enumerate(events) if e[0] == "agent")

    # (1) WITHHELD: integrand.py NOT uploaded before the agent authors; the other
    #     authoring files ARE (so the agent has run.py/baseline to work against).
    pre_agent = [e[1] for i, e in enumerate(events) if e[0] == "upload" and i < pos_agent]
    assert "/workspace/integrand.py" not in pre_agent
    assert "/workspace/method.py" in pre_agent
    assert "/workspace/run.py" in pre_agent
    assert "/workspace/baseline.py" in pre_agent

    # (2) integrand.py is re-uploaded immediately BEFORE each `python run.py`
    score_pos = [i for i, e in enumerate(events) if e[0] == "cmd" and "python run.py" in e[1]]
    assert len(score_pos) == 2  # primary + reproduce
    prev = pos_agent
    for sp in score_pos:
        integ = [
            i for i, e in enumerate(events)
            if e[0] == "upload" and e[1] == "/workspace/integrand.py" and prev < i < sp
        ]
        assert integ, "integrand.py not re-uploaded before this score"
        prev = sp

    # (3) exactly ONE run carries an INLINE seed override, disjoint from default 0..7
    inline = [e[1] for _i, e in enumerate(events)
              if e[0] == "cmd" and "python run.py" in e[1] and "AUTORESEARCH_MC_SEEDS=" in e[1]]
    assert len(inline) == 1
    seed_str = inline[0].split("AUTORESEARCH_MC_SEEDS=", 1)[1].split()[0]
    repro_seeds = {int(s) for s in seed_str.split(",")}
    assert repro_seeds == set(mc_runner.REPRODUCE_SEEDS)
    assert repro_seeds.isdisjoint(set(range(8)))  # disjoint from DEFAULT_PROBLEM seeds

    # (4) both metrics surfaced
    assert out["metric_value"] == 3.5
    assert out["reproduce_metric_value"] == 3.5
    assert out["reproduce_seeds"] == list(mc_runner.REPRODUCE_SEEDS)


def test_run_experiment_passes_generous_agent_timeout_backstop(monkeypatch):
    result = {"metric_value": 1.0, "over_budget": False, "hardcoded_suspected": False}
    backend = RecordingBackend([])
    runner = mc_runner.MonteCarloExperimentRunner(backend=backend)

    # default: generous 2h backstop
    monkeypatch.delenv("AUTORESEARCH_MC_AGENT_TIMEOUT", raising=False)
    runner.run_experiment("t", handle=ob.SandboxHandle(workspace=RecordingWorkspace([], result)))
    assert backend.last_timeout == 7200.0

    # env-configurable
    monkeypatch.setenv("AUTORESEARCH_MC_AGENT_TIMEOUT", "600")
    runner.run_experiment("t", handle=ob.SandboxHandle(workspace=RecordingWorkspace([], result)))
    assert backend.last_timeout == 600.0


# ------------------------------- Task 6 --------------------------------------

def _trusted_local(name: str) -> str:
    """The real host-side content of a trusted file (what the agent SHOULD leave)."""
    runner = mc_runner.MonteCarloExperimentRunner(backend=ob.OpenHandsBackend())
    return runner._local_path(name).read_text(encoding="utf-8")


class TamperWorkspace:
    """``cat`` returns the real trusted content for each trusted file, except
    ``overrides`` (simulating an agent that edited a trusted file)."""

    def __init__(self, result_json, *, overrides=None, agent_code="# agent code",
                 scoring_exit_code=0):
        self._result_json = result_json
        self._overrides = overrides or {}
        self._agent_code = agent_code
        self._scoring_exit_code = scoring_exit_code

    def file_upload(self, local, remote):
        pass

    def execute_command(self, cmd, cwd=None, timeout=None):
        stdout = ""
        exit_code = 0
        if "python run.py" in cmd:
            exit_code = self._scoring_exit_code
        elif "cat result.json" in cmd:
            stdout = json.dumps(self._result_json)
        elif cmd.startswith("cat "):
            name = cmd.split("cat ", 1)[1].strip()
            if name == "method.py":
                stdout = self._agent_code
            elif name in self._overrides:
                stdout = self._overrides[name]
            else:
                stdout = _trusted_local(name)
        return type(
            "R", (), {"exit_code": exit_code, "stdout": stdout, "stderr": "",
                      "timeout_occurred": False}
        )()


def test_over_budget_result_zeroes_metric_and_flags_anti_cheat():
    result = {
        "metric_value": 0.0,
        "rmse_baseline": 0.009,
        "rmse_proposed": 0.001,
        "over_budget": True,
        "hardcoded_suspected": False,
        "bias": 0.02,
    }
    handle = ob.SandboxHandle(workspace=TamperWorkspace(result))
    events: list = []
    runner = mc_runner.MonteCarloExperimentRunner(backend=RecordingBackend(events))

    out = runner.run_experiment("task", handle=handle)

    assert out["metric_value"] == 0.0
    assert out["anti_cheat"]["over_budget"] is True
    assert out["anti_cheat"]["bias"] == 0.02
    # no trusted file tampered -> tamper_detected False
    assert out["anti_cheat"]["tamper_detected"] is False


def test_agent_tampering_trusted_file_is_flagged_but_score_stays_trusted():
    # trusted re-upload guarantees this result; over_budget False -> metric kept
    result = {
        "metric_value": 3.2,
        "rmse_baseline": 0.009,
        "rmse_proposed": 0.003,
        "over_budget": False,
        "hardcoded_suspected": False,
        "bias": 0.0,
    }
    ws = TamperWorkspace(result, overrides={"run.py": "print('METRIC=999999')  # cheat"})
    handle = ob.SandboxHandle(workspace=ws)
    events: list = []
    runner = mc_runner.MonteCarloExperimentRunner(backend=RecordingBackend(events))

    out = runner.run_experiment("task", handle=handle)

    assert out["anti_cheat"]["tamper_detected"] is True
    # score still comes from the re-uploaded trusted run.py, not the agent's cheat
    assert out["metric_value"] == 3.2


def test_planted_result_json_with_scoring_crash_scores_zero():
    """Finding 2: agent plants result.json=999 and makes scoring exit non-zero.
    The runner must return 0, never the stale/planted value."""
    planted = {"metric_value": 999.0, "over_budget": False, "hardcoded_suspected": False}
    ws = TamperWorkspace(planted, scoring_exit_code=1)  # `python run.py` fails
    handle = ob.SandboxHandle(workspace=ws)
    runner = mc_runner.MonteCarloExperimentRunner(backend=RecordingBackend([]))

    out = runner.run_experiment("task", handle=handle)

    assert out["metric_value"] == 0.0  # planted 999 is NOT read
