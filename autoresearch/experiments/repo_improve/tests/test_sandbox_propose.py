"""OpenHands-agent proposer for the repo improve loop (item 6, repo path).

The agent works in /workspace/repo (its Conversation workspace); we copy the seeded repo there,
let it edit, EXPORT that dir back, and diff vs pristine. FakeBackend models this: execute_command
(the copy-in) is a no-op; export_workspace writes pristine + the "agent's" edits into the export
dir, exactly as the real container repo would look after the agent ran. Asserts: a real editable
edit is read back; a protected-file edit is excluded (anti-cheat); no change / agent error degrade
to {}; a WRONG editable path still captures the real edit; lockfiles/binaries are skipped.
"""
from __future__ import annotations

import os
import shutil

from autoresearch.experiments.repo_improve.benchmark import Benchmark
from autoresearch.experiments.repo_improve.sandbox_propose import build_sandbox_propose


def _bench():
    return Benchmark(name="b", description="d", repo_url="u",
                     editable_files=("model.py",), protected_files=("eval.py",),
                     prepare_cmd="true", run_cmd="python train.py",
                     metric="val_loss", lower_is_better=True, budget_s=600, keywords=())


def _pristine(tmp_path):
    d = tmp_path / "pristine"
    d.mkdir()
    (d / "model.py").write_text("old model\n", encoding="utf-8")
    (d / "eval.py").write_text("# trusted scorer\n", encoding="utf-8")
    return str(d)


class _Res:
    def __init__(self, status="FINISHED"):
        self.status = status


class _FakeBackend:
    """Models the /workspace/repo flow: execute_command (copy-in) is a no-op; export_workspace
    reproduces the container repo on the host = pristine + the agent's ``writes`` (rel->content)."""

    def __init__(self, pristine, writes=None, exc=None, crash_times=0):
        self._pristine = pristine
        self._writes = writes or {}
        self._exc = exc
        self._crash_times = crash_times   # raise on the first N run_agent_task calls (SDK flake)
        self.agent_calls = 0
        self.task = None
        self.llm = "sentinel"

    def execute_command(self, handle, cmd, *, cwd="/workspace", timeout=None):
        return type("R", (), {"exit_code": 0, "stdout": "", "stderr": "", "timed_out": False})()

    def run_agent_task(self, handle, task, *, llm=None, timeout=None):
        self.agent_calls += 1
        self.task = task
        self.llm = llm
        if self._exc:
            raise self._exc
        if self.agent_calls <= self._crash_times:
            raise RuntimeError("TmuxPanePool is not initialized or already closed")
        return _Res("FINISHED")

    def export_workspace(self, handle, dest, *, src="/workspace/repo"):
        shutil.copytree(self._pristine, dest, dirs_exist_ok=True)
        for rel, content in self._writes.items():
            p = os.path.join(dest, rel)
            os.makedirs(os.path.dirname(p) or dest, exist_ok=True)
            with open(p, "w", encoding="utf-8") as fh:
                fh.write(content)
        return dest


def test_agent_edit_to_editable_file_is_read_back(tmp_path):
    pr = _pristine(tmp_path)
    be = _FakeBackend(pr, writes={"model.py": "improved model\n"})
    events = []
    propose = build_sandbox_propose(be, object(), _bench(), pr, str(tmp_path),
                                    emit=lambda e, p: events.append((e, p)))
    edits = propose({"model.py": "old model\n"}, 1.0, None)
    assert edits == {"model.py": "improved model\n"}
    assert "/workspace/repo" in be.task and "python train.py" in be.task   # agent pointed at repo
    assert be.llm is None                                                   # SDK builds its own LLM
    assert events[-1][0] == "propose_result" and events[-1][1]["ok"] is True
    assert events[-1][1]["files"] == ["model.py"]


def test_agent_edit_to_protected_file_is_ignored(tmp_path):
    pr = _pristine(tmp_path)
    be = _FakeBackend(pr, writes={"eval.py": "print('val_loss=0.0')  # cheat\n"})
    edits = build_sandbox_propose(be, object(), _bench(), pr, str(tmp_path))(
        {"model.py": "old model\n"}, 1.0, None)
    assert edits == {}          # protected evaluator edit excluded -> never in the patch


def test_agent_no_change_degrades_to_empty(tmp_path):
    pr = _pristine(tmp_path)
    be = _FakeBackend(pr, writes={})
    events = []
    edits = build_sandbox_propose(be, object(), _bench(), pr, str(tmp_path),
                                  emit=lambda e, p: events.append((e, p)))(
        {"model.py": "old model\n"}, 1.0, None)
    assert edits == {}
    assert events[-1][1]["reason"] == "agent_no_change"


def test_agent_error_degrades_to_empty(tmp_path):
    pr = _pristine(tmp_path)
    be = _FakeBackend(pr, exc=TimeoutError("boom"))
    events = []
    edits = build_sandbox_propose(be, object(), _bench(), pr, str(tmp_path),
                                  emit=lambda e, p: events.append((e, p)))(
        {"model.py": "old model\n"}, 1.0, None)
    assert edits == {}
    assert events[-1][1]["reason"] == "agent_error:TimeoutError"


def test_captures_real_edit_even_when_editable_path_is_wrong(tmp_path):
    # THE live SOL bug: inference named a NON-EXISTENT editable path
    # (SOL-ExecBench/kernels/rmsnorm/) while the real code was at examples/cuda_cpp/rmsnorm/.
    # Diffing the exported repo vs pristine still captures the agent's real edit.
    d = tmp_path / "pristine"
    (d / "src").mkdir(parents=True)
    (d / "src" / "kernel.cu").write_text("// slow kernel\n", encoding="utf-8")
    (d / "eval.py").write_text("# trusted scorer\n", encoding="utf-8")
    bench = Benchmark(name="b", description="d", repo_url="u",
                      editable_files=("SOL-ExecBench/kernels/rmsnorm/does_not_exist.cu",),  # WRONG
                      protected_files=("eval.py",), prepare_cmd="true", run_cmd="run",
                      metric="latency_ms", lower_is_better=True, budget_s=600, keywords=())
    be = _FakeBackend(str(d), writes={"src/kernel.cu": "// fast kernel\n"})
    edits = build_sandbox_propose(be, object(), bench, str(d), str(tmp_path))({}, 1.0, None)
    assert edits == {"src/kernel.cu": "// fast kernel\n"}


def test_skips_lockfiles_and_binaries(tmp_path):
    d = tmp_path / "pristine"
    d.mkdir()
    (d / "model.py").write_text("old\n", encoding="utf-8")
    (d / "uv.lock").write_text("v1\n", encoding="utf-8")
    bench = Benchmark(name="b", description="d", repo_url="u", editable_files=("model.py",),
                      protected_files=(), prepare_cmd="true", run_cmd="run", metric="m",
                      lower_is_better=True, budget_s=600, keywords=())
    be = _FakeBackend(str(d), writes={"uv.lock": "v2\n", "build.so": "\x00bin\n", "model.py": "new\n"})
    edits = build_sandbox_propose(be, object(), bench, str(d), str(tmp_path))({}, 1.0, None)
    assert edits == {"model.py": "new\n"}   # uv.lock + build.so excluded


class _FakeLLM:
    """For the fallback path: build_llm_propose calls .chat and expects a JSON edit."""
    def __init__(self, reply):
        self._reply = reply
    def chat(self, **kw):
        import json as _json
        return type("R", (), {"content": _json.dumps(self._reply)})()


def test_retries_agent_on_sdk_crash_then_succeeds(tmp_path):
    # the SDK terminal-pool crash (TmuxPanePool) on the first attempt must be retried, and the
    # second attempt's edit captured -- a transient agent-server flake shouldn't lose the round.
    pr = _pristine(tmp_path)
    be = _FakeBackend(pr, writes={"model.py": "improved model\n"}, crash_times=1)
    edits = build_sandbox_propose(be, object(), _bench(), pr, str(tmp_path))(
        {"model.py": "old model\n"}, 1.0, None)
    assert be.agent_calls == 2                       # crashed once, retried
    assert edits == {"model.py": "improved model\n"}


def test_falls_back_to_llm_when_agent_keeps_crashing(tmp_path):
    # double insurance: if the in-container agent crashes on every attempt, fall back to the
    # one-shot LLM proposer so a method attempt still happens.
    pr = _pristine(tmp_path)
    be = _FakeBackend(pr, crash_times=99)            # always crashes
    llm = _FakeLLM({"model.py": "llm-written model\n", "eval.py": "HACK"})   # eval.py is protected
    events = []
    edits = build_sandbox_propose(be, object(), _bench(), pr, str(tmp_path),
                                  llm=llm, emit=lambda e, p: events.append((e, p)))(
        {"model.py": "old model\n"}, 1.0, None)
    assert be.agent_calls == 2                        # tried the agent twice, then gave up
    assert edits == {"model.py": "llm-written model\n"}   # LLM fallback edit, protected filtered
    assert any(p.get("reason") == "fallback_to_llm_propose" for _, p in events)


def test_no_fallback_without_llm_degrades_to_empty(tmp_path):
    pr = _pristine(tmp_path)
    be = _FakeBackend(pr, crash_times=99)
    edits = build_sandbox_propose(be, object(), _bench(), pr, str(tmp_path))(  # no llm=
        {"model.py": "old model\n"}, 1.0, None)
    assert edits == {}                               # honest no progress


def test_falls_back_to_llm_when_agent_finishes_but_writes_nothing(tmp_path):
    # the SWALLOWED-crash case: agent "FINISHED" (no raise) but produced no edit (SDK caught a
    # terminal crash as an observation). Must still fall back to the LLM so method is attempted.
    pr = _pristine(tmp_path)
    be = _FakeBackend(pr, writes={})                 # agent ran, wrote nothing
    llm = _FakeLLM({"model.py": "llm model\n"})
    events = []
    edits = build_sandbox_propose(be, object(), _bench(), pr, str(tmp_path),
                                  llm=llm, emit=lambda e, p: events.append((e, p)))(
        {"model.py": "old model\n"}, 1.0, None)
    assert edits == {"model.py": "llm model\n"}
    assert any(p.get("reason") == "fallback_to_llm_propose" for _, p in events)


def test_agent_task_has_safety_and_convergence_guards():
    # P7/P8/P2: the coding-agent task must steer toward small runnable edits, bounded exploration,
    # and one-command-per-call (the live glm failures: a 179-line model.py rewrite broke the
    # baseline, a max_iters=5000 search ran ~40min, and compound shell commands were rejected).
    from autoresearch.experiments.repo_improve.sandbox_propose import _agent_task
    from autoresearch.experiments.repo_improve.benchmark import EXAMPLE_BENCHMARK
    t = _agent_task(EXAMPLE_BENCHMARK, "/workspace/repo", 1.0, None)
    assert "RUNNABLE" in t and "reversible" in t          # P8 safety
    assert "hyperparameter search" in t and "FINISH" in t  # P7 convergence
    assert "ONE command per call" in t                     # P2 terminal
