"""Task 6: the general honest-experiment pipeline -- one user question, end to end.

``run_pipeline(backend, llm, question, *, emit=None, pause=None) -> dict``:
1. design = design_experiment(llm, question)                     -> emit design_ready
2. harness = write_harness(backend, handle_A, design)            -> emit harness_ready
   method  = write_method(backend, handle_B, design)  (SEPARATE sandbox = write-time
             isolation from the ruler)                            -> emit method_ready
3. primary   = run_experiment(backend, harness, method, seed=0)
   reproduce = run_experiment(backend, harness, method, seed=1)   -> emit run_done
4. isolated = primary.isolated AND reproduce.isolated  (STRUCTURAL: True iff the
   trusted orchestrator drove both runs to completion; not a textual heuristic)
   verdict  = decide(primary, reproduce, isolated=isolated)
5. report   = build_report(...)  (honest wording)                 -> emit verdict

Key human-confirmation gate (Q3): with ``AUTORESEARCH_GEN_MANUAL=1`` and a
``pause`` callback, the run pauses after design_ready / harness_ready so a human
can inspect and continue.

Default test suite never touches Docker/LLM: tests mock the design/writers/runner
seams (module attributes) + a fake backend.
"""
from __future__ import annotations

import os
import random

from autoresearch.experiments.general.accept import decide
from autoresearch.experiments.general.classify import classify
from autoresearch.experiments.general.design import design_experiment
from autoresearch.experiments.general.harness_writer import write_harness
from autoresearch.experiments.general.method_writer import write_method
from autoresearch.experiments.general.remote_gpu_runner import run_experiment_remote
from autoresearch.experiments.general.report import build_report
from autoresearch.experiments.general.runner import run_experiment

_MANUAL_ENV = "AUTORESEARCH_GEN_MANUAL"
# GPU experiments run the scored step on a remote GPU node (see remote_gpu_runner);
# the design sets design["gpu"], or this env forces it.
_GPU_ENV = "AUTORESEARCH_GEN_GPU"
# Bounded format-repair rounds (P2): the self-repair literature saturates ~2 rounds.
_MAX_REPAIR = int(os.environ.get("AUTORESEARCH_GEN_MAX_REPAIR", "2"))
# Upper bound for the random reproduce seed -- SMALL so seed-range-capped harnesses
# still accept it (a 6-digit seed broke a real run's reproduce). Env-tunable.
_REPRODUCE_SEED_MAX = int(os.environ.get("AUTORESEARCH_GEN_REPRODUCE_SEED_MAX", "99"))
# B5: reproduce on this many fresh seeds; the improvement must hold on a MAJORITY.
_N_REPRODUCE = int(os.environ.get("AUTORESEARCH_GEN_N_REPRODUCE", "3"))
# D9: retry a RUN-LEVEL failure (isolated False = infra/container flake) this many times.
_RUN_RETRIES = int(os.environ.get("AUTORESEARCH_GEN_RUN_RETRIES", "1"))
# A2: if the method's result is essentially IDENTICAL to the baseline (a trivial
# variant, |m-b| within this relative epsilon), do ONE rewrite asking for a
# SUBSTANTIVELY DIFFERENT algorithm. Leak-safe: the feedback is qualitative
# ("you were ~ baseline"), never a score, and it drives DIFFERENCE not victory (so
# it can't be used to game the metric). Env-disablable.
_QUALITY_REPAIR = os.environ.get("AUTORESEARCH_GEN_QUALITY_REPAIR", "1") == "1"
_QUALITY_EPS = float(os.environ.get("AUTORESEARCH_GEN_QUALITY_EPS", "0.001"))
_QUALITY_FEEDBACK = (
    "Your method's result was essentially IDENTICAL to the baseline -- you likely "
    "produced a trivial variant, not a genuinely different algorithm. Propose a "
    "SUBSTANTIVELY DIFFERENT, stronger approach (a real algorithmic change for this "
    "task, not a rename or a no-op tweak). Keep the SAME solution shape."
)


def _trivial_variant(r: dict) -> bool:
    """True iff the method is valid but its metric is ~identical to the baseline's
    (a trivial variant that didn't really do anything different)."""
    pb, pm = r.get("baseline_metric"), r.get("method_metric")
    if not r.get("valid") or pb is None or pm is None:
        return False
    if isinstance(pb, bool) or isinstance(pm, bool) \
            or not isinstance(pb, (int, float)) or not isinstance(pm, (int, float)):
        return False
    return abs(pm - pb) <= _QUALITY_EPS * max(abs(pb), 1e-12)


def run_pipeline(backend, llm, question, *, emit=None, pause=None,
                 seed_primary=0, seed_reproduce=None, design=None) -> dict:
    def _emit(event, payload):
        if emit is not None:
            emit(event, payload)

    # B5 + anti-cheat d: reproduce on K FRESH SMALL RANDOM seeds (majority must
    # hold), so a method tuned to specific seeds can't fake reproducibility and a
    # single flaky/lucky seed can't decide the verdict. Seeds are SMALL (default
    # 2..99) because AI harnesses often cap the seed range. An explicit
    # seed_reproduce (tests) forces exactly that one seed.
    if seed_reproduce is not None:
        reproduce_seeds = [seed_reproduce]
    else:
        reproduce_seeds = [random.randint(2, _REPRODUCE_SEED_MAX) for _ in range(max(1, _N_REPRODUCE))]

    manual = os.environ.get(_MANUAL_ENV) == "1"

    def _maybe_pause(event):
        if manual and pause is not None:
            pause(event)

    # 1) design -- use a pre-generated, HUMAN-REVIEWED design if one is supplied
    #    (the ReviewingExperimentPlan gate), else generate it here.
    if design is None:
        design = design_experiment(llm, question)
    _emit("design_ready", {"design": design})
    _maybe_pause("design_ready")

    # 2) harness (ruler) -- container A
    handle_a = backend.open_sandbox()
    try:
        harness = write_harness(backend, handle_a, design)
    finally:
        backend.close_sandbox(handle_a)
    _emit("harness_ready", {"harness": harness})
    _maybe_pause("harness_ready")

    deps = design.get("dependencies") or []
    ioc = design.get("io_contract")

    # GPU experiments (e.g. training) run the SCORED step on a remote GPU node via the
    # same trusted orchestrator, instead of the local Docker sandbox. The design step
    # sets design["gpu"] for training-style questions; env forces it for testing.
    gpu = bool(design.get("gpu")) or os.environ.get(_GPU_ENV) == "1"

    def _exec(m, seed):
        if gpu:
            return run_experiment_remote(harness, m, seed, dependencies=deps, io_contract=ioc)
        return run_experiment(backend, harness, m, seed, dependencies=deps, io_contract=ioc)

    def _write_method_fresh(feedback=None):
        # each attempt gets a FRESH sandbox (write-time isolation from the ruler).
        h = backend.open_sandbox()
        try:
            return write_method(backend, h, design, feedback=feedback)
        finally:
            backend.close_sandbox(h)

    def _run(m, seed):
        # D9: a RUN-LEVEL CRASH -- the orchestrator did not complete, so even the
        # harness's baseline never computed (isolated False AND baseline_metric None):
        # a container / pip / seed-range flake, NOT a method/validate failure -- is
        # retried once, so a transient infra hiccup doesn't tank a clean result.
        def _crashed(x):
            return (not x.get("isolated")) and x.get("baseline_metric") is None
        r = _exec(m, seed)
        tries = 0
        while _crashed(r) and tries < _RUN_RETRIES:
            tries += 1
            _emit("run_retry", {"seed": seed, "attempt": tries, "error": r.get("error")})
            r = _exec(m, seed)
        return r

    def _run_primary(m):
        return _run(m, seed_primary)

    # 2b) method + BOUNDED FORMAT-REPAIR LOOP (P2): if the run is invalid for a
    #     method-side, LEAK-SAFE reason (own error / wrong shape / infeasible /
    #     timeout, per classify()), re-write solve() with that feedback and re-run.
    #     A harness bug / broken isolation is NOT repairable -> we stop (no death loop).
    method = _write_method_fresh()
    _emit("method_ready", {"method": method})
    primary = _run_primary(method)
    repairs = 0
    while not primary.get("valid") and repairs < _MAX_REPAIR:
        decision = classify(primary)
        if not decision.get("repairable"):
            break
        repairs += 1
        _emit("method_repair", {"round": repairs, "kind": decision.get("kind"),
                                "feedback": decision.get("feedback")})
        method = _write_method_fresh(feedback=decision.get("feedback"))
        _emit("method_ready", {"method": method})
        primary = _run_primary(method)

    # A2: method-quality nudge -- a VALID method whose result is ~identical to the
    # baseline is a trivial variant; one rewrite asks for a substantively different
    # algorithm (leak-safe, difference not victory).
    if _QUALITY_REPAIR and _trivial_variant(primary):
        _emit("method_quality_repair", {"reason": "method result ≈ baseline (trivial variant)"})
        method = _write_method_fresh(feedback=_QUALITY_FEEDBACK)
        _emit("method_ready", {"method": method})
        primary = _run_primary(method)

    # 3) reproduce the FINAL method on K fresh seeds (majority must hold, B5;
    #    each with a run-level retry, D9)
    reproduces = [_run(method, s) for s in reproduce_seeds]
    reproduce = reproduces[0]  # representative, for the report + backward-compat field
    _emit("run_done", {"primary": primary, "reproduce": reproduce,
                       "reproduces": reproduces, "repairs": repairs})

    # 4) verdict from STRUCTURAL isolation (True iff the trusted orchestrator drove
    #    the runs to completion; set by the runner, not a textual heuristic)
    isolated = bool(primary.get("isolated")) and any(bool(r.get("isolated")) for r in reproduces)
    # respect the metric DIRECTION (a higher-is-better metric like knapsack value
    # must not be judged with the fewer-is-better rule).
    lower_is_better = bool(design.get("lower_is_better", True))
    verdict = decide(primary, reproduces, isolated=isolated, lower_is_better=lower_is_better)

    # 5) honest report
    report = build_report(design, primary, reproduce, verdict)
    _emit("verdict", {"verdict": verdict, "report": report})

    return {
        "design": design,
        "harness": harness,
        "method": method,
        "primary": primary,
        "reproduce": reproduce,
        "reproduces": reproduces,
        "isolated": isolated,
        "verdict": verdict,
        "report": report,
        "repairs": repairs,
    }


__all__ = ["run_pipeline"]
