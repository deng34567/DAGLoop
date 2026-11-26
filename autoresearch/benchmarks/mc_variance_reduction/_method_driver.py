"""TRUSTED driver that runs an untrusted ``method.py`` in PROCESS ISOLATION.

This is spawned by ``mc_evaluator.run_method_isolated`` in a temp dir that
contains ONLY a copy of ``method.py`` + this file -- NO integrand, NO problem
module, and with ``PYTHONPATH=""`` and cwd = that temp dir. So a method that
tries ``import integrand`` / ``open('integrand.py')`` to bypass the sample
counter fails: the integrand is unreachable here. The ONLY way to touch f is the
piped ``sample_f`` served by the trusted parent (which counts every call).

Protocol (newline-delimited JSON over stdio):
- config comes in as argv[1] = ``{"problem": {...}, "seed": int}``
- each ``sample_f(x)`` writes ``{"x": [...]}`` to stdout, reads ``{"y": v}`` from stdin
- final result: ``{"estimate": v}`` (or ``{"error": "..."}``) on stdout
"""
import json
import sys


def main():
    cfg = json.loads(sys.argv[1])
    problem = cfg["problem"]
    seed = cfg["seed"]
    n_samples = problem["n_samples"]

    import numpy as np

    rng = np.random.default_rng(seed)

    def sample_f(x):
        sys.stdout.write(json.dumps({"x": [float(v) for v in x]}) + "\n")
        sys.stdout.flush()
        line = sys.stdin.readline()
        return json.loads(line)["y"]

    try:
        import method  # the copied, untrusted method (imports resolved HERE, isolated)

        est = method.proposed_estimate(problem, sample_f, n_samples, rng)
        out = {"estimate": float(est)}
    except Exception as e:  # import-time or runtime failure -> failed, not a score
        out = {"error": repr(e)}
    sys.stdout.write(json.dumps(out) + "\n")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
