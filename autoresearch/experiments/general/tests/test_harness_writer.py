"""Task 2: the harness-writer agent (role A, the AI "ruler"). FULLY MOCKED:
mock ``run_agent_task`` (FINISHED) + a mock workspace whose ``cat harness.py``
returns a preset source. No container / LLM / network.
"""
from __future__ import annotations

from autoresearch.agents._reuse import openhands_backend as ob
from autoresearch.experiments.general import harness_writer

_DESIGN = {
    "question": "beat First-Fit?",
    "task": "1D bin packing, capacity 1.0",
    "instance_spec": "200 sizes from default_rng(seed)",
    "baseline": "First-Fit",
    "method_interface": "solve(instance) -> list[int]",
    "metric": "bins used, lower better; valid = all placed, no overflow",
    "success_criterion": "fewer bins than First-Fit",
}

_HARNESS_SRC = "# harness written by the agent\ndef make_instances(seed): ...\n"


class _Workspace:
    def __init__(self, harness_src):
        self._harness_src = harness_src
        self.uploads = []
        self.commands = []

    def file_upload(self, local, remote):
        self.uploads.append((str(local), str(remote)))

    def execute_command(self, cmd, cwd=None, timeout=None):
        self.commands.append(cmd)
        stdout = self._harness_src if cmd.strip() == "cat harness.py" else ""
        return type("R", (), {"exit_code": 0, "stdout": stdout, "stderr": "",
                              "timeout_occurred": False})()


class _Backend(ob.OpenHandsBackend):
    def __init__(self):
        super().__init__()
        self.last_task = None
        self.last_timeout = None

    def run_agent_task(self, handle, task, *, llm=None, cli_mode=True, timeout=None):
        self.last_task = task
        self.last_timeout = timeout
        return ob.RunResult(status="FINISHED")


def test_write_harness_returns_source_and_uploads_trusted_orchestrator():
    ws = _Workspace(_HARNESS_SRC)
    handle = ob.SandboxHandle(workspace=ws)
    backend = _Backend()

    src = harness_writer.write_harness(backend, handle, _DESIGN)

    assert src == _HARNESS_SRC
    # the FIXED trusted orchestrator + self-test wrapper + isolator are pre-placed
    # so the agent can self-test (via _selftest.py, which does NOT delete harness.py)
    remotes = {r for _, r in ws.uploads}
    for name in ("_orchestrator.py", "_selftest.py", "isolated_method.py", "_method_driver.py"):
        assert f"/workspace/{name}" in remotes


def test_harness_prompt_is_a_pure_library_that_never_touches_the_method():
    backend = _Backend()
    ws = _Workspace(_HARNESS_SRC)
    harness_writer.write_harness(backend, ob.SandboxHandle(workspace=ws), _DESIGN)

    task = backend.last_task
    # the four pure-library functions
    for token in ("make_instances", "run_baseline", "validate", "score"):
        assert token in task, f"prompt missing {token!r}"
    # the design flows in
    assert "First-Fit" in task
    # the harness is a PURE LIBRARY: no __main__, and it must never run/import the method
    lowered = task.lower()
    assert "library" in lowered
    assert "__main__" in task  # states: do NOT write a __main__
    assert "import method" in lowered
    assert "must not" in lowered or "do not" in lowered or "never" in lowered
    # the method-running / result.json plumbing moved to the trusted orchestrator,
    # so the harness prompt must NOT instruct the harness to do those itself
    assert "run_method_isolated" not in task
    assert "sys.argv" not in task
    # self-test uses the NON-deleting wrapper _selftest.py, not _orchestrator.py
    assert "_selftest.py" in task
    assert "_orchestrator.py" not in task


def test_write_harness_installs_declared_dependencies_before_agent():
    ws = _Workspace(_HARNESS_SRC)
    backend = _Backend()
    design = dict(_DESIGN, dependencies=["numpy", "scipy"])

    harness_writer.write_harness(backend, ob.SandboxHandle(workspace=ws), design)

    assert "pip install -q numpy scipy" in ws.commands


def test_write_harness_no_deps_issues_no_pip_install():
    ws = _Workspace(_HARNESS_SRC)
    backend = _Backend()
    design = dict(_DESIGN, dependencies=[])

    harness_writer.write_harness(backend, ob.SandboxHandle(workspace=ws), design)

    assert not any("pip install" in c for c in ws.commands)


def test_write_harness_sanitizes_injecting_dependency():
    ws = _Workspace(_HARNESS_SRC)
    backend = _Backend()
    design = dict(_DESIGN, dependencies=["numpy", "numpy; rm -rf /"])

    harness_writer.write_harness(backend, ob.SandboxHandle(workspace=ws), design)

    # the injecting entry is dropped; only the safe package survives, un-injected
    assert "pip install -q numpy" in ws.commands
    assert not any("rm -rf" in c for c in ws.commands)


def test_harness_prompt_defaults_to_lower_is_better():
    """No lower_is_better flag (or True) -> the ruler is told LOWER is better,
    matching accept.decide's default direction."""
    backend = _Backend()
    ws = _Workspace(_HARNESS_SRC)
    harness_writer.write_harness(backend, ob.SandboxHandle(workspace=ws), _DESIGN)
    task = backend.last_task
    assert "LOWER is better" in task
    assert "HIGHER is better" not in task


def test_harness_prompt_directs_higher_is_better_when_flagged():
    """lower_is_better=False (e.g. knapsack total value) -> the ruler MUST return the
    RAW metric where HIGHER is better and MUST NOT negate it / turn it into a gap, so
    the harness's metric direction agrees with design.lower_is_better and decide()."""
    backend = _Backend()
    ws = _Workspace(_HARNESS_SRC)
    design = dict(_DESIGN, lower_is_better=False)
    harness_writer.write_harness(backend, ob.SandboxHandle(workspace=ws), design)
    task = backend.last_task
    assert "HIGHER is better" in task
    # it must NOT still tell the ruler to minimize
    assert "LOWER is better" not in task
    # and it must forbid the sign-flip / gap conversion that caused the inverted verdict
    lowered = task.lower()
    assert "negate" in lowered or "sign" in lowered or "gap" in lowered


def test_harness_prompt_mentions_pip_self_serve():
    backend = _Backend()
    ws = _Workspace(_HARNESS_SRC)
    harness_writer.write_harness(backend, ob.SandboxHandle(workspace=ws), _DESIGN)
    assert "pip install" in backend.last_task.lower()


def test_write_harness_timeout_backstop(monkeypatch):
    backend = _Backend()
    ws = _Workspace(_HARNESS_SRC)
    monkeypatch.delenv("AUTORESEARCH_GEN_AGENT_TIMEOUT", raising=False)
    harness_writer.write_harness(backend, ob.SandboxHandle(workspace=ws), _DESIGN)
    assert backend.last_timeout == 7200.0

    monkeypatch.setenv("AUTORESEARCH_GEN_AGENT_TIMEOUT", "600")
    harness_writer.write_harness(backend, ob.SandboxHandle(workspace=ws), _DESIGN)
    assert backend.last_timeout == 600.0


def test_harness_prompt_gpu_hint_only_when_gpu():
    backend = _Backend(); ws = _Workspace(_HARNESS_SRC)
    harness_writer.write_harness(backend, ob.SandboxHandle(workspace=ws), dict(_DESIGN, gpu=True))
    assert "GPU EXPERIMENT" in backend.last_task and "torch" in backend.last_task
    backend2 = _Backend()
    harness_writer.write_harness(backend2, ob.SandboxHandle(workspace=_Workspace(_HARNESS_SRC)), _DESIGN)
    assert "GPU EXPERIMENT" not in backend2.last_task


def test_harness_prompt_heldout_hint_only_for_ml_metric():
    backend = _Backend()
    harness_writer.write_harness(backend, ob.SandboxHandle(workspace=_Workspace(_HARNESS_SRC)),
                                 dict(_DESIGN, metric="classification accuracy (higher is better)"))
    assert "HELD-OUT VALIDATION" in backend.last_task
    # combinatorial metric -> no held-out hint (structural split not applicable)
    backend2 = _Backend()
    harness_writer.write_harness(backend2, ob.SandboxHandle(workspace=_Workspace(_HARNESS_SRC)),
                                 dict(_DESIGN, metric="number of bins used"))
    assert "HELD-OUT VALIDATION" not in backend2.last_task
