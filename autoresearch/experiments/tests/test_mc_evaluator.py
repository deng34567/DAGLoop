"""Tests for the a-layer trusted MC evaluator core (pure Python, real
scipy/numpy + LOCAL subprocesses -- fast, no docker/network/LLM).

Covers the trusted integrand + problem metadata (Task 1), the pure metric math
(``_compute_metrics``), the PROCESS-ISOLATED budgeted scoring (Finding 1: the
``import integrand`` self-integration cheat must now score 0), truth-reliability
(Finding 4), and the trusted ``run.py`` harness (Task 3). Benchmark files are
*flat* modules, so we load / copy them by file path.
"""
from __future__ import annotations

import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

MC_DIR = Path(__file__).resolve().parents[2] / "benchmarks" / "mc_variance_reduction"
DRIVER = MC_DIR / "_method_driver.py"
BASELINE = MC_DIR / "baseline.py"

# Small NON-constant problem for fast isolated tests (a constant integrand would
# make baseline RMSE 0 -> 0/0). Isolation spawns a subprocess per (method, seed).
_PROBLEM = {"dim": 2, "n_samples": 200, "seeds": [0, 1, 2, 3]}


def _integrand(x):
    return math.exp(math.sin(2.0 * math.pi * sum(x))) / (1.0 + sum(xi * xi for xi in x))


def _load_flat(name: str):
    spec = importlib.util.spec_from_file_location(name, MC_DIR / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write_method(tmp_path, src: str) -> Path:
    p = tmp_path / "cand.py"
    p.write_text(src, encoding="utf-8")
    return p


def _evaluate(method_path):
    from autoresearch.experiments import mc_evaluator

    return mc_evaluator.evaluate(
        _PROBLEM, _integrand, str(method_path),
        baseline_path=str(BASELINE), driver_path=str(DRIVER),
    )


# ------------------------------- Task 1 --------------------------------------

def test_integrand_is_callable_smooth_and_non_constant():
    integrand = _load_flat("integrand").integrand
    a = integrand([0.1, 0.2, 0.3, 0.4])
    b = integrand([0.9, 0.8, 0.7, 0.6])
    assert isinstance(a, float) and math.isfinite(a)
    assert a != b


def test_problem_metadata_has_fields_and_no_integrand():
    problem = _load_flat("problem").DEFAULT_PROBLEM
    assert problem["dim"] >= 1
    assert problem["n_samples"] > 0
    assert len(problem["seeds"]) >= 2
    for v in problem.values():
        assert not callable(v)
    assert "integrand" not in problem


# ----------------------- pure metric math (_compute_metrics) -----------------

def test__compute_metrics_ratio_and_cap():
    from autoresearch.experiments import mc_evaluator as ev

    # proposed errors 10x smaller -> ratio ~10
    r = ev._compute_metrics(
        [0.1, -0.1, 0.1, -0.1], [0.01, -0.01, 0.01, -0.01],
        over_budget=False, prop_fail=False, prop_calls_max=200, n_samples=200,
    )
    assert r["ratio"] == pytest.approx(10.0)
    # near-zero proposed RMSE -> capped at MAX_RATIO (no blow-up)
    r2 = ev._compute_metrics(
        [0.1, -0.1], [1e-18, -1e-18],
        over_budget=False, prop_fail=False, prop_calls_max=200, n_samples=200,
    )
    assert r2["ratio"] == ev.MAX_RATIO


def test__compute_metrics_over_budget_or_fail_zeroes():
    from autoresearch.experiments import mc_evaluator as ev

    over = ev._compute_metrics(
        [0.1, -0.1], [0.001, -0.001],
        over_budget=True, prop_fail=False, prop_calls_max=200, n_samples=200,
    )
    fail = ev._compute_metrics(
        [0.1, -0.1], [0.001, -0.001],
        over_budget=False, prop_fail=True, prop_calls_max=200, n_samples=200,
    )
    assert over["ratio"] == 0.0 and fail["ratio"] == 0.0


def test__compute_metrics_flat_lowcalls_is_flagged():
    from autoresearch.experiments import mc_evaluator as ev

    # flat cross-seed spread AND far fewer than N calls -> hardcode suspected
    r = ev._compute_metrics(
        [0.1, -0.1, 0.05, -0.07], [0.5001, 0.5001, 0.5001, 0.5001],
        over_budget=False, prop_fail=False, prop_calls_max=2, n_samples=200,
    )
    assert r["hardcoded_suspected"] is True


def test__compute_metrics_high_ratio_full_budget_not_flagged():
    # Finding 3: a genuinely excellent (>100x) reducer that used the FULL budget
    # is flat but must NOT be flagged as hardcoded.
    from autoresearch.experiments import mc_evaluator as ev

    r = ev._compute_metrics(
        [0.1, -0.1, 0.05, -0.07], [1e-4, -1e-4, 0.5e-4, -0.7e-4],
        over_budget=False, prop_fail=False, prop_calls_max=200, n_samples=200,
    )
    assert r["ratio"] == ev.MAX_RATIO  # capped, huge reduction
    assert r["hardcoded_suspected"] is False


# ----------------------- isolation / anti-gaming -----------------------------

_IMPORT_CHEAT = """\
import integrand as _ig            # trusted f as a sibling -- must be UNREACHABLE
def proposed_estimate(problem, sample_f, n_samples, rng):
    d = problem["dim"]; g = 20
    xs = [(i + 0.5) / g for i in range(g)]
    tot = 0.0
    for a in xs:
        for b in xs:
            tot += _ig.integrand([a, b])   # dense uncounted self-integration
    return tot / (g ** d)
"""

_OPEN_CHEAT = """\
def proposed_estimate(problem, sample_f, n_samples, rng):
    src = open("integrand.py").read()   # try to read the trusted f source
    return 0.5815
"""

_OVER_BUDGET = """\
def proposed_estimate(problem, sample_f, n_samples, rng):
    d = problem["dim"]
    return sum(sample_f([rng.random() for _ in range(d)])
               for _ in range(2 * n_samples)) / (2 * n_samples)
"""

_ANTITHETIC = """\
def proposed_estimate(problem, sample_f, n_samples, rng):
    d = problem["dim"]; tot = 0.0
    for _ in range(n_samples // 2):
        x = [rng.random() for _ in range(d)]
        tot += sample_f(x) + sample_f([1.0 - xi for xi in x])
    return tot / n_samples
"""

_RAISES = """\
def proposed_estimate(problem, sample_f, n_samples, rng):
    raise ValueError("boom")
"""


def test_import_integrand_cheat_is_defeated(tmp_path):
    r = _evaluate(_write_method(tmp_path, _IMPORT_CHEAT))
    # isolation makes `import integrand` fail -> method failed -> metric 0
    assert r["metric_value"] == 0
    assert (r["over_budget"] is True) or (r["n_seeds"] == 0)


def test_open_integrand_file_cheat_is_defeated(tmp_path):
    r = _evaluate(_write_method(tmp_path, _OPEN_CHEAT))
    assert r["metric_value"] == 0  # integrand.py not in the isolated dir


def test_over_budget_cheat_scores_zero(tmp_path):
    r = _evaluate(_write_method(tmp_path, _OVER_BUDGET))
    assert r["over_budget"] is True
    assert r["metric_value"] == 0


def test_honest_method_still_works(tmp_path):
    r = _evaluate(_write_method(tmp_path, _ANTITHETIC))
    from autoresearch.experiments import mc_evaluator as ev

    assert 1.0 < r["metric_value"] <= ev.MAX_RATIO
    assert r["over_budget"] is False
    assert r["hardcoded_suspected"] is False


def test_proposed_exception_scores_zero(tmp_path):
    r = _evaluate(_write_method(tmp_path, _RAISES))
    assert r["metric_value"] == 0


def test_baseline_vs_baseline_ratio_is_one():
    from autoresearch.experiments import mc_evaluator

    r = mc_evaluator.evaluate(
        _PROBLEM, _integrand, str(BASELINE),
        baseline_path=str(BASELINE), driver_path=str(DRIVER),
    )
    # identical file + same seeds -> identical errors -> ratio exactly 1
    assert r["metric_value"] == pytest.approx(1.0)


def test_records_truth_error_and_reliability_flag(tmp_path):
    # Finding 4: nquad abs error captured and flagged reliable when << MC RMSE
    r = _evaluate(_write_method(tmp_path, _ANTITHETIC))
    assert r["i_true_abs_error"] >= 0
    assert r["i_true_abs_error"] < r["rmse_baseline"]
    assert r["truth_reliable"] is True


# ------------------------------- Task 3 --------------------------------------

def test_trusted_run_py_scores_method_against_baseline(tmp_path):
    """Assemble the flat container layout on the host and run ``run.py``."""
    import shutil

    from autoresearch.experiments import mc_evaluator

    for name in ("run.py", "integrand.py", "problem.py", "baseline.py",
                 "method.py", "_method_driver.py"):
        shutil.copy2(MC_DIR / name, tmp_path / name)
    shutil.copy2(mc_evaluator.__file__, tmp_path / "mc_evaluator.py")

    proc = subprocess.run(
        [sys.executable, "run.py"],
        cwd=str(tmp_path), capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    assert "METRIC=" in proc.stdout

    result = json.loads((tmp_path / "result.json").read_text(encoding="utf-8"))
    assert math.isfinite(result["metric_value"])
    assert result["metric_value"] > 0  # baseline copy vs baseline -> ratio 1
    assert result["over_budget"] is False
    assert result["hardcoded_suspected"] is False


# --------------------------- Task 1 (M11): run.py hardening ------------------
# run.py deletes the integrand SOURCE after importing it (so an isolated method
# can't read /workspace/integrand.py by absolute path to self-integrate) and
# honours an INLINE ``AUTORESEARCH_MC_SEEDS`` override (fewer seeds -> fast/local
# tests, and disjoint fresh seeds for the runner's reproduce score).

def _assemble_and_run(tmp_path, *, method_src=None, seeds="0,1"):
    """Copy the flat container layout into tmp_path (optionally overriding
    method.py) and run the trusted run.py, with an inline seed override for
    speed. Returns the completed subprocess."""
    import os
    import shutil

    from autoresearch.experiments import mc_evaluator

    for name in ("run.py", "integrand.py", "problem.py", "baseline.py",
                 "method.py", "_method_driver.py"):
        shutil.copy2(MC_DIR / name, tmp_path / name)
    shutil.copy2(mc_evaluator.__file__, tmp_path / "mc_evaluator.py")
    if method_src is not None:
        (tmp_path / "method.py").write_text(method_src, encoding="utf-8")

    env = {**os.environ}
    if seeds is not None:
        env["AUTORESEARCH_MC_SEEDS"] = seeds
    return subprocess.run(
        [sys.executable, "run.py"],
        cwd=str(tmp_path), capture_output=True, text=True, timeout=120, env=env,
    )


def test_run_py_removes_integrand_source_after_import(tmp_path):
    proc = _assemble_and_run(tmp_path, seeds="0,1")
    assert proc.returncode == 0, proc.stderr
    # once run.py has the integrand in memory, its SOURCE is gone
    assert not (tmp_path / "integrand.py").exists()
    result = json.loads((tmp_path / "result.json").read_text(encoding="utf-8"))
    assert math.isfinite(result["metric_value"])
    assert result["metric_value"] > 0  # baseline copy vs baseline -> ratio ~1


def test_absolute_path_read_cheat_is_defeated(tmp_path):
    # a method that ignores sample_f and instead reads the trusted integrand
    # source by ABSOLUTE path to self-integrate -- defeated because run.py has
    # already deleted integrand.py -> the open() raises -> the method fails -> 0.
    cheat = (
        "def proposed_estimate(problem, sample_f, n_samples, rng):\n"
        f"    with open({str(tmp_path / 'integrand.py')!r}) as fh:\n"
        "        ns = {}\n"
        "        exec(fh.read(), ns)\n"
        "    f = ns['integrand']\n"
        "    import itertools\n"
        "    d = problem['dim']\n"
        "    vals = [f(list(c)) for c in itertools.product([0.25, 0.75], repeat=d)]\n"
        "    return sum(vals) / len(vals)\n"
    )
    proc = _assemble_and_run(tmp_path, method_src=cheat, seeds="0,1")
    assert proc.returncode == 0, proc.stderr
    result = json.loads((tmp_path / "result.json").read_text(encoding="utf-8"))
    assert result["metric_value"] == 0.0  # absolute-path read cheat scores 0


def test_seed_override_env_changes_seeds(tmp_path):
    proc = _assemble_and_run(tmp_path, seeds="10,11,12")
    assert proc.returncode == 0, proc.stderr
    result = json.loads((tmp_path / "result.json").read_text(encoding="utf-8"))
    assert result["n_seeds"] == 3  # inline override -> exactly the 3 given seeds
