"""a-layer: the TRUSTED Monte-Carlo variance-reduction evaluator (pure Python).

This is the "ruler". It is shared by the trusted ``run.py`` harness (uploaded
flat into the container) and the host-side unit tests, so it depends ONLY on
scipy/numpy/stdlib -- never on the ``autoresearch`` package. No Docker, no LLM,
no network.

Anti-gaming design (see the plan's threat model + code-review findings):

* **Process isolation (Finding 1).** A method is scored in an ISOLATED
  subprocess (``_method_driver.py``) whose temp dir / PYTHONPATH / cwd do NOT
  contain the integrand. The method reaches ``f`` ONLY through a piped
  ``sample_f`` served by THIS trusted parent, which holds the integrand,
  evaluates it, and COUNTS every call. So ``import integrand`` /
  ``open('integrand.py')`` to self-integrate uncounted simply fails (the
  integrand is unreachable), and calling ``sample_f`` more than ``n_samples``
  times is ``over_budget`` -> ``metric_value = 0``.
* **Independent truth (Finding 4).** The evaluator computes the true integral
  ITSELF with ``scipy.integrate.nquad`` (unlimited budget, off-line) and records
  its absolute error estimate; if that error is not << the baseline MC RMSE the
  truth is flagged unreliable.
* **Combined hardcode flag (Finding 3).** ``hardcoded_suspected`` fires only
  when the cross-seed spread collapses (< 1% of the baseline spread) AND the
  method used far fewer than ``n_samples`` calls -- so a genuinely excellent
  (>100x variance reduction, full-budget) method is NOT falsely flagged. It is a
  SIGNAL; it never gates/zeros the metric.
* **Ratio cap.** ``metric_value`` is capped at ``MAX_RATIO`` so a near-zero
  proposed RMSE (or a 0/0) cannot mint an unbounded score.
"""
from __future__ import annotations

import math
from pathlib import Path

MAX_RATIO = 1e3  # cap: blocks rmse_p->0 blow-ups and closed-form "infinite score"

# runtime cap (kill runaways) = 2x budget; over_budget is judged strictly at > N.
_RUNTIME_CALL_CAP = 2


def _find_driver() -> str:
    """Locate ``_method_driver.py`` (flat next to this module in the container,
    or the benchmark dir on the host)."""
    candidates = [
        Path.cwd() / "_method_driver.py",
        Path(__file__).resolve().parent / "_method_driver.py",
        Path(__file__).resolve().parents[1]
        / "benchmarks" / "mc_variance_reduction" / "_method_driver.py",
    ]
    for c in candidates:
        if c.is_file():
            return str(c)
    raise FileNotFoundError("_method_driver.py not found on any known path")


def _quadrature_truth(problem, integrand):
    """Independent trusted truth via unlimited-budget quadrature (d<=4 fine).
    Returns (value, abs_error_estimate)."""
    from scipy.integrate import nquad

    d = problem["dim"]
    val, err = nquad(lambda *x: integrand(list(x)), [[0.0, 1.0]] * d)
    return float(val), float(err)


def run_method_isolated(method_path, problem, integrand, seed, *,
                        driver_path=None, timeout=30.0):
    """Score one method file in an ISOLATED subprocess; the parent serves and
    COUNTS every ``sample_f`` call.

    Returns ``(estimate|None, n_calls, over_budget, failed)``. ``over_budget`` is
    True when the method makes strictly more than ``n_samples`` calls (a runaway
    is killed at 2x). ``failed`` is True on exception / timeout / no estimate.
    """
    import json
    import os
    import select
    import shutil
    import subprocess
    import sys
    import tempfile
    import time

    driver_path = driver_path or _find_driver()
    N = problem["n_samples"]
    tmp = Path(tempfile.mkdtemp(prefix="mc-iso-"))
    try:
        shutil.copy2(method_path, tmp / "method.py")
        shutil.copy2(driver_path, tmp / "_method_driver.py")
        cfg = json.dumps({"problem": dict(problem), "seed": int(seed)})
        # inherit env so python+numpy import, but EMPTY PYTHONPATH so nothing
        # extra (e.g. /workspace) is importable -> integrand is unreachable.
        env = {**os.environ, "PYTHONPATH": ""}
        proc = subprocess.Popen(
            [sys.executable, "_method_driver.py", cfg],
            cwd=str(tmp),
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,  # avoid stderr-buffer-fill deadlock
            text=True,
        )
        n_calls = 0
        over_budget = False
        failed = False
        estimate = None
        deadline = time.monotonic() + timeout
        try:
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    proc.kill()
                    failed = True
                    break
                rlist, _, _ = select.select([proc.stdout], [], [], remaining)
                if not rlist:
                    proc.kill()
                    failed = True
                    break
                line = proc.stdout.readline()
                if line == "":
                    break  # child ended
                try:
                    msg = json.loads(line)
                except Exception:
                    continue
                if "x" in msg:
                    n_calls += 1
                    if n_calls > _RUNTIME_CALL_CAP * N:
                        proc.kill()
                        over_budget = True
                        break
                    y = integrand(list(msg["x"]))
                    proc.stdin.write(json.dumps({"y": float(y)}) + "\n")
                    proc.stdin.flush()
                elif "estimate" in msg:
                    estimate = float(msg["estimate"])
                    break
                elif "error" in msg:
                    failed = True
                    break
        finally:
            try:
                proc.stdin.close()
            except Exception:
                pass
            try:
                proc.wait(timeout=5)
            except Exception:
                proc.kill()
        if n_calls > N:  # strict budget judgement
            over_budget = True
        if estimate is None and not over_budget:
            failed = True
        return estimate, n_calls, over_budget, failed
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _rmse(errs):
    if not errs:
        return float("inf")
    return math.sqrt(sum(e * e for e in errs) / len(errs))


def _mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def _spread(xs):
    """Population standard deviation of the cross-seed errors."""
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))


def _compute_metrics(base_errs, prop_errs, *, over_budget, prop_fail,
                     prop_calls_max, n_samples) -> dict:
    """PURE metric math (unit-testable with arrays). Returns ratio + RMSEs +
    bias + the combined ``hardcoded_suspected`` signal."""
    rmse_b = _rmse(base_errs)
    rmse_p = _rmse(prop_errs) if prop_errs else float("inf")

    if over_budget or prop_fail:
        ratio = 0.0
    elif rmse_p > 0:
        ratio = min(rmse_b / rmse_p, MAX_RATIO)
    else:
        ratio = MAX_RATIO

    bias = abs(_mean(prop_errs)) if prop_errs else 0.0
    # Finding 3: flat cross-seed spread ALONE is not enough (a >100x reducer is
    # legitimately flat). Require it ALSO used far fewer than the budget.
    flat = _spread(prop_errs) < 0.01 * (_spread(base_errs) or 1.0)
    few_calls = prop_calls_max < 0.5 * n_samples
    hardcoded_suspected = bool(flat and few_calls)

    return {
        "ratio": ratio,
        "rmse_baseline": rmse_b,
        "rmse_proposed": rmse_p,
        "bias": bias,
        "hardcoded_suspected": hardcoded_suspected,
    }


def evaluate(problem, integrand, method_path, *, baseline_path,
             driver_path=None, timeout=30.0) -> dict:
    """Score the method FILE at ``method_path`` against the trusted baseline FILE
    at ``baseline_path`` honestly (each run in process isolation).

    Returns a dict with ``metric_value`` (= min(rmse_baseline/rmse_proposed,
    MAX_RATIO), or 0 on budget/failure), the trusted truth + its abs-error +
    reliability flag, both RMSEs, and the anti-gaming signals.
    """
    i_true, i_true_err = _quadrature_truth(problem, integrand)
    N = problem["n_samples"]

    def _run_all(path):
        ests, calls = [], []
        ob = fail = False
        for s in problem["seeds"]:
            e, n, o, f = run_method_isolated(
                path, problem, integrand, s, driver_path=driver_path, timeout=timeout
            )
            ests.append(e)
            calls.append(n)
            ob = ob or o
            fail = fail or f or (e is None)
        return ests, calls, ob, fail

    base_ests, _base_calls, base_ob, base_fail = _run_all(baseline_path)
    prop_ests, prop_calls, prop_ob, prop_fail = _run_all(method_path)

    base_errs = [e - i_true for e in base_ests if e is not None]
    prop_errs = [e - i_true for e in prop_ests if e is not None]
    prop_calls_max = max(prop_calls) if prop_calls else 0

    m = _compute_metrics(
        base_errs, prop_errs,
        over_budget=prop_ob, prop_fail=prop_fail,
        prop_calls_max=prop_calls_max, n_samples=N,
    )

    rmse_b = m["rmse_baseline"]
    truth_reliable = bool(
        math.isfinite(rmse_b) and rmse_b > 0 and i_true_err < 0.1 * rmse_b
    )

    return {
        "metric_value": m["ratio"],
        "i_true": i_true,
        "i_true_abs_error": i_true_err,
        "truth_reliable": truth_reliable,
        "rmse_baseline": rmse_b,
        "rmse_proposed": m["rmse_proposed"],
        "n_seeds": len(prop_errs),
        "over_budget": prop_ob,
        "baseline_failed": base_fail or base_ob,
        "bias": m["bias"],
        "hardcoded_suspected": m["hardcoded_suspected"],
    }
