"""TRUSTED harness for the mc_variance_reduction benchmark (NOT editable).

Runs inside the container as a FLAT module set. It does NOT import ``method.py``
itself -- the untrusted method is run in PROCESS ISOLATION by the evaluator (see
``mc_evaluator.run_method_isolated`` + ``_method_driver.py``), so it can neither
reach the trusted integrand (bypassing the sample counter) nor crash this
harness at import time.

The editable ``method.py`` is scored against the TRUSTED fixed ``baseline.py``.
Emits ``METRIC=<float>`` on stdout AND a ``result.json`` the runner reads.
"""
import json
import os
from glob import glob
from pathlib import Path

from integrand import integrand  # trusted f (served to methods only via sample_f)
from problem import DEFAULT_PROBLEM

import mc_evaluator  # flat-uploaded trusted evaluator

# The integrand is now held IN MEMORY (quadrature truth + sample_f use this
# object). Delete its source immediately so an isolated method can no longer
# read /workspace/integrand.py by ABSOLUTE PATH to self-integrate uncounted.
Path("integrand.py").unlink(missing_ok=True)
for _pyc in glob("__pycache__/integrand*.pyc"):
    Path(_pyc).unlink()


def _problem():
    """DEFAULT_PROBLEM, optionally with an INLINE seed override so the runner
    can re-score the same method on FRESH seeds (reproduce). ``execute_command``
    has no env kwarg, so the runner passes ``AUTORESEARCH_MC_SEEDS=...`` inline."""
    raw = os.environ.get("AUTORESEARCH_MC_SEEDS")
    if raw:
        seeds = [int(s) for s in raw.split(",") if s.strip()]
        return {**DEFAULT_PROBLEM, "seeds": seeds}
    return DEFAULT_PROBLEM


def main():
    r = mc_evaluator.evaluate(
        _problem(),
        integrand,
        "method.py",
        baseline_path="baseline.py",
    )
    print(f"METRIC={r['metric_value']}")
    with open("result.json", "w", encoding="utf-8") as f:
        json.dump(r, f)


if __name__ == "__main__":
    main()
