"""FIXED, TRUSTED orchestrator -- the run-time anti-cheat spine (NOT AI-written).

Run as ``python _orchestrator.py <seed>`` in /workspace. It, and not the
AI-written harness, DRIVES the run, so runtime isolation holds BY CONSTRUCTION:

1. ``import harness`` -- load the AI-written ruler (a PURE LIBRARY:
   make_instances / run_baseline / validate / score) into memory.
2. IMMEDIATELY delete ``harness.py`` (relative to cwd) + any ``__pycache__/
   harness*.pyc`` -- so NOTHING spawned afterwards (the isolated method
   subprocess) can read the ruler by absolute path or decompile its bytecode. The
   already-imported module stays live in memory. (Mirrors M11's integrand
   deletion: import-then-delete.)
3. ``instances = harness.make_instances(seed)``; ``baseline = harness.run_baseline``.
4. ``solutions = run_method_isolated(instances, "method.py")`` -- the trusted
   isolator runs the untrusted method in a separate subprocess.
5. validate each solution with ``harness.validate``; score the valid ones with
   ``harness.score`` (any invalid/missing -> valid=False, method_metric=None).
6. write ``result.json {baseline_metric, method_metric, valid, seed}``.

Because the AI harness is a library that NEVER runs or imports the method, and the
method runs only inside the isolator AFTER harness.py is gone, the method cannot
read the ruler. RESIDUAL (P2): a maximally-adversarial (colluding) harness could
stash secret ground truth in ANY /workspace file -- including re-creating
harness.py after deletion (make_instances/run_baseline run before the method), or
just leaking its answer key straight through the ``instances`` channel; a full
filesystem sandbox (bwrap/unshare/separate container) closes that. Stdlib only.
"""
import json
import os
import sys
from pathlib import Path


def main() -> None:
    seed = int(sys.argv[1])

    # Read the AUTHORING self-test flag BEFORE ``import harness`` so an AI-written
    # harness cannot spoof it (it is decided before any harness code runs). During
    # scoring the runner NEVER sets this, so the deletion (anti-cheat) still fires.
    selftest = os.environ.get("AUTORESEARCH_ORCH_SELFTEST") == "1"

    import harness  # load the AI-written ruler (pure library) into memory

    # Remove the ruler from disk BEFORE any untrusted code runs (paths relative to
    # cwd so this is testable off-container). The in-memory module survives. In the
    # AUTHORING self-test path we SKIP this so the agent can re-run its self-test
    # freely (harness.py must not vanish after one run); scoring always deletes.
    if not selftest:
        Path("harness.py").unlink(missing_ok=True)
        pycache = Path("__pycache__")
        if pycache.is_dir():
            for pyc in pycache.glob("harness*.pyc"):
                try:
                    pyc.unlink()
                except OSError:
                    pass

    from isolated_method import run_method_isolated

    def _scalar(x, what):
        # The metric MUST be a single number (lower=better). A dict/list is a
        # harness contract violation -> raise a CLEAR error here (which the runner
        # reports as valid=False) instead of letting it crash the verdict layer
        # downstream with a cryptic ``dict * float`` TypeError.
        if isinstance(x, bool) or not isinstance(x, (int, float)):
            raise ValueError(
                f"harness.{what}() must return a single number (lower is better); "
                f"got {type(x).__name__}: {x!r}"
            )
        return float(x)

    # Load the LEAK-SAFE shape contract (the design's concrete example solution) so
    # a format mismatch is caught + explained separately from the harness's
    # feasibility check. Absent contract -> shape check no-ops (falls back to validate).
    example_solution = example_inst = None
    try:
        ioc = json.loads(Path("io_contract.json").read_text())
        example_solution = (ioc or {}).get("solution_example")
        example_inst = (ioc or {}).get("instance_example")
    except Exception:
        pass
    try:
        from contract import check_shape as _check_shape, strip_to_contract as _strip_fn
    except Exception:
        _check_shape = _strip_fn = None

    # anti-cheat b: the METHOD sees instances stripped to the DECLARED fields, so a
    # colluding harness can't smuggle an answer key via an extra field. The harness's
    # own validate/score keep the FULL instances (their legitimate computation).
    strip_on = os.environ.get("AUTORESEARCH_GEN_STRIP_INSTANCES", "1") == "1"

    def _strip(inst):
        if strip_on and _strip_fn is not None and example_inst is not None:
            return _strip_fn(inst, example_inst)
        return inst

    def _shape_check(sol):
        if _check_shape is None or example_solution is None:
            return True, ""  # no contract available -> skip; harness.validate still runs
        try:
            return _check_shape(sol, example_solution)
        except Exception:
            return True, ""  # never let a contract bug reject a solution

    def _fmt_method_error(merr):
        # Turn the isolator's leak-safe per-instance method error into a short
        # human string (the method's own exception; never any ruler internals).
        if not merr:
            return "method returned None (no solution and no error reported)"
        return (f"method {merr.get('phase', '?')} error: "
                f"{merr.get('type', '?')}: {merr.get('msg', '')}").strip()

    # anti-cheat c: snapshot the workspace BEFORE the harness's make_instances /
    # run_baseline run, so any file they create (a re-created harness.py or a stashed
    # answer key) can be removed BEFORE the method could read it. Env-disablable.
    clean_ws = os.environ.get("AUTORESEARCH_GEN_CLEAN_WORKSPACE", "1") == "1" and not selftest

    def _top_files():
        try:
            return {n for n in os.listdir(".") if os.path.isfile(n)}
        except OSError:
            return set()

    pre_files = _top_files() if clean_ws else None

    instances = harness.make_instances(seed)
    baseline_metric = _scalar(harness.run_baseline(instances), "run_baseline")

    stashed_removed = []
    if pre_files is not None:
        for name in _top_files() - pre_files:
            try:
                Path(name).unlink()
                stashed_removed.append(name)
            except OSError:
                pass
        pycache = Path("__pycache__")
        if pycache.is_dir():  # a re-created harness's bytecode, too
            for pyc in pycache.glob("harness*.pyc"):
                try:
                    pyc.unlink()
                except OSError:
                    pass

    # the method only ever sees the stripped instances (anti-cheat b)
    method_instances = [_strip(inst) for inst in instances]
    solutions, method_diag = run_method_isolated(
        method_instances, "method.py", return_diagnostics=True
    )
    method_errors = method_diag.get("errors") or []

    valid_solutions = []
    all_valid = bool(instances)
    n_valid = 0
    first_failure = None
    for i, (inst, sol) in enumerate(zip(instances, solutions)):
        kind = detail = None
        if sol is None:
            # the method crashed / never produced a solution -- its OWN error
            # (leak-safe: no ruler internals) came back over the isolator's pipe.
            merr = method_errors[i] if i < len(method_errors) else None
            kind, detail, ok = "method_error", _fmt_method_error(merr), False
        else:
            shape_ok, shape_reason = _shape_check(sol)
            if not shape_ok:
                # wrong SHAPE -> leak-safe reason (from the public example, not the
                # ruler); this is the reason the P2 repair loop feeds back.
                kind, detail, ok = "shape_mismatch", shape_reason, False
            else:
                try:
                    ok = bool(harness.validate(inst, sol))
                    if not ok:
                        # well-formed but the ruler rejected it -> generic (leak-safe)
                        # detail; the harness's specific reason is NOT surfaced.
                        kind, detail = "invalid_solution", "solution is well-formed but the ruler rejected it (feasibility/constraint)"
                except Exception as exc:  # noqa: BLE001
                    kind, detail, ok = "validate_error", f"{type(exc).__name__}: {exc}", False
        if ok:
            n_valid += 1
        else:
            all_valid = False
            if first_failure is None:
                first_failure = {"index": i, "kind": kind, "detail": (detail or "")[:500]}
        valid_solutions.append(sol if ok else None)

    if all_valid:
        method_metric = _scalar(harness.score(instances, valid_solutions), "score")
    else:
        # invalid/missing solution -> fail closed; the verdict layer rejects it.
        method_metric = None

    diagnostics = {
        "n_instances": len(instances),
        "n_valid": n_valid,
        "validity_rate": (n_valid / len(instances)) if instances else 0.0,
        "first_failure": first_failure,
        "method_stderr_tail": (method_diag.get("stderr_tail") or "")[-800:],
        "timed_out": bool(method_diag.get("timed_out")),
        # anti-cheat c: files the harness created during make_instances/run_baseline
        # that were removed before the method ran (a stash / re-created ruler).
        "stashed_files_removed": stashed_removed,
    }

    with open("result.json", "w") as f:
        json.dump(
            {
                "baseline_metric": baseline_metric,
                "method_metric": method_metric,
                "valid": bool(all_valid),
                "seed": seed,
                "diagnostics": diagnostics,
            },
            f,
        )


if __name__ == "__main__":
    main()
