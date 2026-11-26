"""Harness for the toy_pairwise benchmark (NOT in the editable_region -- holds
the trusted reference, so an optimized solution.py cannot cheat correctness).

Runs solution.solve on a fixed deterministic input, checks it against the
reference, times it, and emits BOTH ``METRIC=<float>`` on stdout AND a
``result.json`` -- the RunnerBackend reads result.json deterministically.

METRIC = 1000/runtime  (higher = faster) when correct, else 0.0.
"""
import json
import time

from solution import solve

# fixed deterministic input (a cheap LCG, no imports, reproducible)
DATA = [((i * 2654435761) % 10007) - 5000 for i in range(3000)]


def _reference(data):
    s = sum(data)
    sq = sum(x * x for x in data)
    return (s * s - sq) // 2


def main():
    ref = _reference(DATA)
    t0 = time.perf_counter()
    out = solve(DATA)
    dt = time.perf_counter() - t0
    correct = (out == ref)
    metric = (1000.0 / dt) if (correct and dt > 0) else 0.0
    print(f"METRIC={metric}")
    print(f"correct={correct} runtime_seconds={dt:.5f} result={out} reference={ref}")
    with open("result.json", "w", encoding="utf-8") as f:
        json.dump({"metric_value": metric, "correct": correct, "runtime_seconds": dt}, f)


if __name__ == "__main__":
    main()
