"""Task 3: the method-writer agent (role B), isolated from the ruler. FULLY
MOCKED. The prompt gives ONLY the interface + task + baseline description -- NEVER
the harness source or the scoring internals.
"""
from __future__ import annotations

from autoresearch.agents._reuse import openhands_backend as ob
from autoresearch.experiments.general import method_writer

_DESIGN = {
    "question": "beat First-Fit?",
    "task": "1D bin packing, capacity 1.0",
    "instance_spec": "200 sizes from default_rng(seed)",
    "baseline": "First-Fit",
    "method_interface": "solve(instance) -> list[int]",
    "metric": "bins used, lower better; valid = all placed, no overflow",
    "success_criterion": "fewer bins than First-Fit",
}

_METHOD_SRC = "def solve(instance):\n    return []\n"


class _Workspace:
    def __init__(self, method_src):
        self._method_src = method_src
        self.commands = []

    def file_upload(self, local, remote):
        pass

    def execute_command(self, cmd, cwd=None, timeout=None):
        self.commands.append(cmd)
        stdout = self._method_src if cmd.strip() == "cat method.py" else ""
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


def test_write_method_returns_source():
    ws = _Workspace(_METHOD_SRC)
    backend = _Backend()
    src = method_writer.write_method(backend, ob.SandboxHandle(workspace=ws), _DESIGN)
    assert src == _METHOD_SRC


def test_method_prompt_has_interface_but_not_the_ruler_internals():
    backend = _Backend()
    ws = _Workspace(_METHOD_SRC)
    method_writer.write_method(backend, ob.SandboxHandle(workspace=ws), _DESIGN)
    task = backend.last_task

    # it gets the interface + task + baseline description
    assert "solve(instance)" in task
    assert "1D bin packing" in task
    assert "First-Fit" in task

    # it must NOT see the harness / the ruler's scoring internals (write-time isolation)
    for leaked in ("make_instances", "run_method_isolated", "result.json", "def score"):
        assert leaked not in task, f"prompt leaks ruler internal {leaked!r}"


def test_write_method_installs_declared_dependencies_before_agent():
    ws = _Workspace(_METHOD_SRC)
    backend = _Backend()
    design = dict(_DESIGN, dependencies=["numpy"])

    method_writer.write_method(backend, ob.SandboxHandle(workspace=ws), design)

    assert "pip install -q numpy" in ws.commands


def test_write_method_no_deps_issues_no_pip_install():
    ws = _Workspace(_METHOD_SRC)
    backend = _Backend()
    design = dict(_DESIGN, dependencies=[])

    method_writer.write_method(backend, ob.SandboxHandle(workspace=ws), design)

    assert not any("pip install" in c for c in ws.commands)


def test_write_method_sanitizes_injecting_dependency():
    ws = _Workspace(_METHOD_SRC)
    backend = _Backend()
    design = dict(_DESIGN, dependencies=["$(evil)", "numpy"])

    method_writer.write_method(backend, ob.SandboxHandle(workspace=ws), design)

    assert "pip install -q numpy" in ws.commands
    assert not any("evil" in c for c in ws.commands)


def test_method_prompt_mentions_pip_self_serve():
    backend = _Backend()
    ws = _Workspace(_METHOD_SRC)
    method_writer.write_method(backend, ob.SandboxHandle(workspace=ws), _DESIGN)
    assert "pip install" in backend.last_task.lower()


def test_write_method_timeout_backstop(monkeypatch):
    backend = _Backend()
    ws = _Workspace(_METHOD_SRC)
    monkeypatch.delenv("AUTORESEARCH_GEN_AGENT_TIMEOUT", raising=False)
    method_writer.write_method(backend, ob.SandboxHandle(workspace=ws), _DESIGN)
    assert backend.last_timeout == 7200.0

    monkeypatch.setenv("AUTORESEARCH_GEN_AGENT_TIMEOUT", "600")
    method_writer.write_method(backend, ob.SandboxHandle(workspace=ws), _DESIGN)
    assert backend.last_timeout == 600.0


# --- D11: enforced shape self-test ---------------------------------------------

class _CountBackend(_Backend):
    def __init__(self):
        super().__init__()
        self.n_tasks = 0
        self.tasks = []

    def run_agent_task(self, handle, task, *, llm=None, cli_mode=True, timeout=None):
        self.n_tasks += 1
        self.tasks.append(task)
        return ob.RunResult(status="FINISHED")


class _ShapeWS(_Workspace):
    def __init__(self, method_src, shape_out):
        super().__init__(method_src)
        self._shape_out = shape_out

    def execute_command(self, cmd, cwd=None, timeout=None):
        self.commands.append(cmd)
        if "_shapecheck" in cmd:
            out = self._shape_out
        elif cmd.strip() == "cat method.py":
            out = self._method_src
        else:
            out = ""
        return type("R", (), {"exit_code": 0, "stdout": out, "stderr": "", "timeout_occurred": False})()


_IOC_DESIGN = dict(_DESIGN, io_contract={"instance_example": {"n": 1}, "solution_example": [0, 1]})


def test_d11_corrective_turn_on_shape_fail():
    ws = _ShapeWS(_METHOD_SRC, "SHAPE_FAIL: solution: expected int, got float")
    backend = _CountBackend()
    method_writer.write_method(backend, ob.SandboxHandle(workspace=ws), _IOC_DESIGN)
    assert backend.n_tasks == 2                     # initial write + ONE corrective turn
    assert "WRONG SHAPE" in backend.tasks[1]


def test_d11_no_corrective_turn_on_shape_ok():
    ws = _ShapeWS(_METHOD_SRC, "SHAPE_OK")
    backend = _CountBackend()
    method_writer.write_method(backend, ob.SandboxHandle(workspace=ws), _IOC_DESIGN)
    assert backend.n_tasks == 1                     # no corrective turn when shape is right


def test_d11_skipped_without_io_contract():
    ws = _ShapeWS(_METHOD_SRC, "SHAPE_FAIL: x")
    backend = _CountBackend()
    method_writer.write_method(backend, ob.SandboxHandle(workspace=ws), _DESIGN)  # no io_contract
    assert backend.n_tasks == 1
    assert not any("_shapecheck" in c for c in ws.commands)


# --- scalar-output shape confusion: the example VALUES are illustrative, not targets --
# (a method-writer read the example's approximation=0.935891 as a REQUIRED output and
#  tried to hardcode it instead of COMPUTING the integral -> the shape example must say
#  it pins STRUCTURE/TYPES only.)
_SCALAR_IOC = {"instance_example": {"a": 0.0, "b": 1.0},
               "solution_example": {"method": "simpsons", "approximation": 0.935891}}


def test_method_prompt_marks_example_values_as_illustrative():
    backend = _Backend()
    ws = _Workspace(_METHOD_SRC)
    method_writer.write_method(backend, ob.SandboxHandle(workspace=ws),
                               dict(_DESIGN, io_contract=_SCALAR_IOC))
    task = backend.last_task.lower()
    assert "illustrative" in task or "compute your own" in task or "not hardcode" in task


def test_d11_shape_fix_prompt_is_about_structure_not_the_example_value():
    ws = _ShapeWS(_METHOD_SRC, "SHAPE_FAIL: solution: expected dict, got float")
    backend = _CountBackend()
    method_writer.write_method(backend, ob.SandboxHandle(workspace=ws),
                               dict(_DESIGN, io_contract=_SCALAR_IOC))
    fix = backend.tasks[1].lower()
    assert "structure" in fix or "types" in fix
    assert "illustrative" in fix or "compute your own" in fix or "not hardcode" in fix


def test_method_prompt_gpu_block_only_when_gpu():
    backend = _Backend()
    method_writer.write_method(backend, ob.SandboxHandle(workspace=_Workspace(_METHOD_SRC)), dict(_DESIGN, gpu=True))
    assert "GPU EXPERIMENT" in backend.last_task
    backend2 = _Backend()
    method_writer.write_method(backend2, ob.SandboxHandle(workspace=_Workspace(_METHOD_SRC)), _DESIGN)
    assert "GPU EXPERIMENT" not in backend2.last_task
