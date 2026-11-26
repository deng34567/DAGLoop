"""GEN sandbox-write (item 6): open sandbox -> agent writes code -> export -> experiment_patch.

Uses a FakeBackend (no docker): its export writes canned files INCLUDING a protected evaluator
file, so the tests assert the anti-cheat filter drops it and partial/failed runs degrade honestly.
"""
from __future__ import annotations

import os
from pathlib import Path

from autoresearch.agents._reuse.openhands_backend import InfraError
from autoresearch.agents._reuse.sandbox_write import write_experiment_in_sandbox


class _Res:
    def __init__(self, status: str):
        self.status = status
        self.cost = "0"


class _FakeBackend:
    """Records the agent task; export drops canned files (incl. a protected eval.py) into dest."""

    def __init__(self, *, status: str = "FINISHED", files: dict | None = None,
                 fail: str | None = None):
        self._status = status
        self._files = files if files is not None else {
            "model.py": "class Net: ...\n",
            "train.py": "print('val_loss=0.5')\n",
            "eval.py": "# trusted scorer -- must NOT reach the patch\n",
        }
        self._fail = fail
        self.task = None
        self.closed = False
        self.opened = False

    def open_sandbox(self, *, cuda_device=None, mounts=None, env=None):
        self.opened = True
        if self._fail == "open":
            raise InfraError("no docker")
        return object()

    def run_agent_task(self, handle, task, *, llm=None, timeout=None):
        self.task = task
        self.llm = llm
        return _Res(self._status)

    def export_workspace(self, handle, dest, *, src="/workspace"):
        d = Path(dest)
        d.mkdir(parents=True, exist_ok=True)
        for rel, content in self._files.items():
            p = d / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding="utf-8")
        return d

    def close_sandbox(self, handle):
        self.closed = True


_PLAN = {"change_summary": "train a tiny net", "changes": [
    {"file": "model.py", "description": "define the net"},
    {"file": "train.py", "description": "train + print val_loss"}]}
_VSPEC = {"metric": "val_loss", "editable_region": ["model.py", "train.py"],
          "protected_files": ["eval.py"]}


def test_writes_patch_and_filters_protected_evaluator(tmp_path):
    be = _FakeBackend()
    patch = write_experiment_in_sandbox(
        plan=_PLAN, verifier_spec=_VSPEC, workroot=str(tmp_path), backend=be)
    assert patch is not None
    files = {c["file"] for c in patch["changes"]}
    assert files == {"model.py", "train.py"}      # eval.py (protected) dropped -> anti-cheat
    assert "eval.py" not in patch["touched_files"]
    assert patch["changes"][0]["patch"]           # carries the file CONTENT
    assert be.closed                              # sandbox always torn down


def test_task_prompt_names_metric_and_hints_location():
    be = _FakeBackend()
    write_experiment_in_sandbox(plan=_PLAN, verifier_spec=_VSPEC, workroot=".", backend=be)
    assert "val_loss" in be.task                  # the harness metric is stated
    assert "model.py" in be.task                  # editable_region surfaced as a hint
    assert be.llm is None                          # never hands our wrapper to the SDK agent


def test_partial_write_on_timeout_is_still_exported(tmp_path):
    # a non-FINISHED status must NOT discard files the agent already wrote
    be = _FakeBackend(status="TIMEOUT", files={"train.py": "print('val_loss=1.0')\n"})
    patch = write_experiment_in_sandbox(
        plan=_PLAN, verifier_spec=_VSPEC, workroot=str(tmp_path), backend=be)
    assert patch is not None and patch["touched_files"] == ["train.py"]


def test_infra_failure_degrades_to_none(tmp_path):
    be = _FakeBackend(fail="open")
    patch = write_experiment_in_sandbox(
        plan=_PLAN, verifier_spec=_VSPEC, workroot=str(tmp_path), backend=be)
    assert patch is None                          # honest degrade, never a fake result


def test_empty_write_returns_none(tmp_path):
    be = _FakeBackend(files={})                    # agent wrote nothing usable
    patch = write_experiment_in_sandbox(
        plan=_PLAN, verifier_spec=_VSPEC, workroot=str(tmp_path), backend=be)
    assert patch is None
