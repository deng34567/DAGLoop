"""clone_repo: reuse a same-repo cache, but RE-CLONE when a DIFFERENT repo is cached.

Regression for the dynamic-discovery bug: a fixed workroot on the GPU host kept a prior run's
``pristine/`` (e.g. nanoGPT), and the idempotent "non-empty dest -> reuse" made a later
dynamic run (e.g. CIFAR) silently reuse the WRONG repo. clone_repo now checks ``origin``.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from autoresearch.experiments.repo_improve.clone import clone_repo


def _runner(origin=None, clone_ok=True):
    """Fake subprocess runner: answers `git remote get-url origin` with ``origin`` (None -> no
    remote), and `git clone` by materializing a minimal dest."""
    calls = []

    def run(cmd, **kw):
        calls.append(cmd)
        if "remote" in cmd:
            if origin is None:
                return SimpleNamespace(returncode=1, stdout="", stderr="no origin")
            return SimpleNamespace(returncode=0, stdout=origin + "\n", stderr="")
        if "clone" in cmd:
            dest = Path(cmd[-1])
            if clone_ok:
                dest.mkdir(parents=True, exist_ok=True)
                (dest / "train.py").write_text("# freshly cloned\n")
                return SimpleNamespace(returncode=0, stdout="", stderr="")
            return SimpleNamespace(returncode=1, stdout="", stderr="clone failed")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    run.calls = calls
    return run


def _cloned(calls):
    return any("clone" in c for c in calls)


def test_clone_into_empty_dest(tmp_path):
    r = _runner()
    clone_repo("https://github.com/o/repoA", str(tmp_path / "pristine"), runner=r)
    assert _cloned(r.calls)


def test_reuse_when_same_repo(tmp_path):
    dest = tmp_path / "pristine"; dest.mkdir(); (dest / "f").write_text("x")
    r = _runner(origin="https://github.com/o/repoA")
    clone_repo("https://github.com/o/repoA.git", str(dest), runner=r)   # .git suffix normalized
    assert not _cloned(r.calls)                       # SAME repo -> reused, no re-clone


def test_reclone_when_different_repo(tmp_path):
    # a stale clone of repoA is cached; requesting repoB must WIPE + re-clone.
    dest = tmp_path / "pristine"; dest.mkdir(); (dest / "stale").write_text("old repoA")
    r = _runner(origin="https://github.com/o/repoA")
    clone_repo("https://github.com/o/repoB", str(dest), runner=r)
    assert _cloned(r.calls)                            # DIFFERENT repo -> re-cloned
    assert not (dest / "stale").exists()               # old repo wiped
    assert (dest / "train.py").exists()                # fresh clone materialized


def test_reclone_when_no_git_remote(tmp_path):
    dest = tmp_path / "pristine"; dest.mkdir(); (dest / "f").write_text("x")
    r = _runner(origin=None)                           # unknown origin -> not same -> re-clone
    clone_repo("https://github.com/o/repoB", str(dest), runner=r)
    assert _cloned(r.calls)


@pytest.mark.skip(reason="mechanical protected-file restore DISABLED per meeting 2026-07-25 item 4 "
                         "(trust the process; catch cheating by diff/report review). Un-skip this "
                         "test together with un-commenting the restore block in clone.py.")
def test_prepare_variant_workspace_restores_protected_directory(tmp_path):
    # a protected entry can be a whole DIRECTORY (package/data folder), not just a file --
    # restoration must copytree it, not crash with IsADirectoryError (SOL-ExecBench: src/sol_execbench).
    from autoresearch.experiments.repo_improve.benchmark import Benchmark
    from autoresearch.experiments.repo_improve.clone import prepare_variant_workspace
    pristine = tmp_path / "pristine"; (pristine / "src" / "pkg").mkdir(parents=True)
    (pristine / "src" / "pkg" / "eval.py").write_text("TRUSTED = 1\n")
    (pristine / "train.py").write_text("x = 1\n")
    bench = Benchmark(name="b", description="", repo_url="u", editable_files=("train.py",),
                      protected_files=("src/pkg",), prepare_cmd="true", run_cmd="python train.py",
                      metric="m", lower_is_better=True, budget_s=300)
    ws = tmp_path / "ws"
    # candidate tries to tamper the protected package dir + edits train.py
    prepare_variant_workspace(str(pristine), str(ws), bench,
                              {"train.py": "x = 2\n", "src/pkg/eval.py": "TRUSTED = 999  # cheat\n"})
    assert (ws / "train.py").read_text() == "x = 2\n"                  # editable edit applied
    assert (ws / "src" / "pkg" / "eval.py").read_text() == "TRUSTED = 1\n"  # protected dir restored
