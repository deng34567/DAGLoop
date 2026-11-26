"""Isolation tests for LocalSubprocessRunnerBackend on the toy_pairwise
benchmark. These REALLY run subprocesses (our own toy code -- safe, fast, no
network). They prove the real "code -> real metric" loop + the sandbox guards.

NO LLM, NO Redis, NO network, single serial process.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

from autoresearch.agents._reuse.runner_backend import LocalSubprocessRunnerBackend

_BENCH = Path(__file__).resolve().parents[3] / "benchmarks" / "toy_pairwise"

# verifier-first: only solution.py is editable; run.py (the trusted harness) is forbidden.
VERIFIER = {"editable_region": ["solution.py"], "forbidden_files": ["run.py"]}


def _backend(timeout=60.0):
    return LocalSubprocessRunnerBackend(_BENCH, timeout=timeout, run_cmd=(sys.executable, "run.py"))


# The O(N) optimized solution (correct + much faster than the naive baseline).
OPTIMIZED = {"changes": [{"file": "solution.py", "patch":
    "def solve(data):\n"
    "    s = sum(data)\n"
    "    sq = sum(x * x for x in data)\n"
    "    return (s * s - sq) // 2\n"}]}

WRONG = {"changes": [{"file": "solution.py", "patch": "def solve(data):\n    return 42\n"}]}
HANG = {"changes": [{"file": "solution.py", "patch": "def solve(data):\n    while True:\n        pass\n"}]}
FORBIDDEN = {"changes": [{"file": "run.py", "patch": "print('hacked')\n"}]}
ESCAPE = {"changes": [{"file": "../evil.py", "patch": "print('escape')\n"}]}


def test_run_baseline_real_metric():
    r = _backend().run_baseline({}, VERIFIER)
    assert r["exit_code"] == 0
    assert r["metric_value"] > 0.0          # correct baseline -> nonzero score
    assert r["runtime_seconds"] > 0.0
    assert "run.py" in r["command"]


def test_optimized_patch_beats_baseline():
    b = _backend()
    base = b.run_baseline({}, VERIFIER)
    exp = b.run_experiment(OPTIMIZED, VERIFIER)
    assert exp["exit_code"] == 0
    assert exp["metric_value"] > 0.0
    # the O(N) solution is dramatically faster than the O(N^2) baseline -> higher metric
    assert exp["metric_value"] > base["metric_value"]


def test_wrong_result_scores_zero():
    # a fast-but-wrong solution fails the correctness gate in run.py -> metric 0
    r = _backend().run_experiment(WRONG, VERIFIER)
    assert r["exit_code"] == 0            # the process ran fine...
    assert r["metric_value"] == 0.0       # ...but the answer was wrong -> 0


def test_forbidden_file_rejected():
    r = _backend().run_experiment(FORBIDDEN, VERIFIER)
    assert r["exit_code"] == 126
    assert r["metric_value"] == 0.0
    assert "run.py" in (r["stdout"] + str(r["stderr"]))


def test_path_escape_rejected():
    r = _backend().run_experiment(ESCAPE, VERIFIER)
    assert r["exit_code"] == 126
    assert r["metric_value"] == 0.0


def test_timeout_hang_patch():
    r = _backend(timeout=3.0).run_experiment(HANG, VERIFIER)
    assert r["metric_value"] == 0.0
    assert r["exit_code"] == -1           # timed out
    assert "TIMEOUT" in str(r["stderr"])
