"""Task 6: the honest-experiment pipeline + report. FULLY MOCKED -- design /
writers / runner seams replaced; a fake backend (no Docker). Asserts the pipeline
wires the pieces, emits events in order, derives isolation from evidence, and
builds an honest report.
"""
from __future__ import annotations

import pytest

from autoresearch.experiments.general import pipeline as gen_pipeline
from autoresearch.experiments.general import report as gen_report

_DESIGN = {
    "question": "Can a heuristic beat First-Fit on synthetic bin packing?",
    "task": "1D bin packing, capacity 1.0",
    "instance_spec": "200 sizes from default_rng(seed)",
    "baseline": "First-Fit",
    "method_interface": "solve(instance) -> list[int]",
    "metric": "bins used, lower better",
    "success_criterion": "fewer bins than First-Fit",
}


class _FakeBackend:
    def __init__(self):
        self.opened = 0
        self.closed = 0

    def open_sandbox(self, *, workspace_factory=None):
        self.opened += 1
        return object()

    def close_sandbox(self, handle):
        self.closed += 1


def _install_mocks(monkeypatch, *, primary, reproduce):
    monkeypatch.setattr(gen_pipeline, "_N_REPRODUCE", 1)  # these tests mock 1 reproduce
    monkeypatch.setattr(gen_pipeline, "design_experiment", lambda llm, q: dict(_DESIGN, question=q))
    monkeypatch.setattr(gen_pipeline, "write_harness", lambda b, h, d: "# harness src")
    monkeypatch.setattr(gen_pipeline, "write_method", lambda b, h, d, feedback=None: "def solve(i): return []")
    runs = iter([primary, reproduce])
    monkeypatch.setattr(gen_pipeline, "run_experiment",
                        lambda b, hc, mc, seed, dependencies=(), io_contract=None: next(runs))


def test_pipeline_runs_end_to_end_and_emits_events_in_order(monkeypatch):
    primary = {"baseline_metric": 10, "method_metric": 8, "valid": True, "seed": 0, "isolated": True}
    reproduce = {"baseline_metric": 11, "method_metric": 9, "valid": True, "seed": 1, "isolated": True}
    _install_mocks(monkeypatch, primary=primary, reproduce=reproduce)

    events = []
    backend = _FakeBackend()
    out = gen_pipeline.run_pipeline(
        backend, llm=object(), question="beat First-Fit?",
        emit=lambda e, p: events.append((e, p)),
    )

    # two SEPARATE sandboxes opened (harness A + method B), both closed
    assert backend.opened == 2 and backend.closed == 2

    # events in order
    assert [e for e, _ in events] == [
        "design_ready", "harness_ready", "method_ready", "run_done", "verdict",
    ]

    # verdict = pass (better + reproduced + isolated), isolation is evidence-derived
    assert out["isolated"] is True
    assert out["verdict"]["verdict"] == "pass"
    assert out["design"]["question"] == "beat First-Fit?"
    assert out["harness"] == "# harness src"
    assert "method" in out


def test_pipeline_forwards_declared_dependencies_to_runner(monkeypatch):
    primary = {"baseline_metric": 10, "method_metric": 8, "valid": True, "seed": 0, "isolated": True}
    reproduce = {"baseline_metric": 11, "method_metric": 9, "valid": True, "seed": 1, "isolated": True}
    monkeypatch.setattr(gen_pipeline, "_N_REPRODUCE", 1)
    monkeypatch.setattr(gen_pipeline, "design_experiment",
                        lambda llm, q: dict(_DESIGN, question=q, dependencies=["numpy"]))
    monkeypatch.setattr(gen_pipeline, "write_harness", lambda b, h, d: "# harness src")
    monkeypatch.setattr(gen_pipeline, "write_method", lambda b, h, d, feedback=None: "def solve(i): return []")
    runs = iter([primary, reproduce])
    seen = []
    monkeypatch.setattr(gen_pipeline, "run_experiment",
                        lambda b, hc, mc, seed, dependencies=(), io_contract=None: (seen.append(dependencies), next(runs))[1])

    gen_pipeline.run_pipeline(_FakeBackend(), llm=object(), question="q")

    assert seen == [["numpy"], ["numpy"]]  # forwarded to BOTH primary + reproduce


def test_pipeline_isolation_evidence_gates_verdict(monkeypatch):
    # both runs report better numbers, but the harness bypassed the isolator
    primary = {"baseline_metric": 10, "method_metric": 8, "valid": True, "seed": 0, "isolated": False}
    reproduce = {"baseline_metric": 11, "method_metric": 9, "valid": True, "seed": 1, "isolated": False}
    _install_mocks(monkeypatch, primary=primary, reproduce=reproduce)

    out = gen_pipeline.run_pipeline(_FakeBackend(), llm=object(), question="q")
    assert out["isolated"] is False
    assert out["verdict"]["verdict"] == "inconclusive"


def test_report_has_real_numbers_isolation_and_honesty_note():
    primary = {"baseline_metric": 10, "method_metric": 8, "valid": True, "seed": 0, "isolated": True}
    reproduce = {"baseline_metric": 11, "method_metric": 9, "valid": True, "seed": 1, "isolated": True}
    verdict = {"verdict": "pass", "accepted": True, "reasons": ["ok"]}

    text = gen_report.build_report(_DESIGN, primary, reproduce, verdict)

    assert "First-Fit" in text          # design
    assert "10" in text and "8" in text  # real numbers
    assert "隔离" in text                # isolation evidence
    assert "诚实" in text                # honesty note
    assert "P2" in text                  # adversarial review deferred
    assert "pass" in text or "通过" in text


# --- P2: bounded format-repair loop -------------------------------------------

def _invalid(kind="shape_mismatch", detail="expected int, got float", timed_out=False):
    return {"valid": False, "isolated": True,
            "diagnostics": {"first_failure": {"kind": kind, "detail": detail},
                            "timed_out": timed_out}}


def _valid(seed):
    return {"baseline_metric": 10, "method_metric": 8, "valid": True, "seed": seed, "isolated": True}


def _mock_writers(monkeypatch, feedbacks):
    monkeypatch.setattr(gen_pipeline, "_N_REPRODUCE", 1)  # repair tests mock 1 reproduce
    monkeypatch.setattr(gen_pipeline, "design_experiment", lambda llm, q: dict(_DESIGN, question=q))
    monkeypatch.setattr(gen_pipeline, "write_harness", lambda b, h, d: "# harness")

    def _wm(b, h, d, feedback=None):
        feedbacks.append(feedback)
        return "def solve(i): return []"

    monkeypatch.setattr(gen_pipeline, "write_method", _wm)


def test_repair_loop_fixes_invalid_then_passes(monkeypatch):
    feedbacks = []
    _mock_writers(monkeypatch, feedbacks)
    runs = iter([_invalid(), _valid(0), _valid(1)])  # primary#0 invalid, primary#1 valid, reproduce
    monkeypatch.setattr(gen_pipeline, "run_experiment",
                        lambda b, hc, mc, seed, dependencies=(), io_contract=None: next(runs))
    events = []
    out = gen_pipeline.run_pipeline(_FakeBackend(), llm=object(), question="q",
                                    emit=lambda e, p: events.append((e, p)))
    assert out["repairs"] == 1
    assert out["verdict"]["verdict"] == "pass"
    assert feedbacks[0] is None                       # first attempt: no feedback
    assert "expected int, got float" in (feedbacks[1] or "")   # repair got the shape reason
    assert ("method_repair", {"round": 1, "kind": "shape_mismatch",
                              "feedback": feedbacks[1]}) in events


def test_repair_loop_stops_at_cap(monkeypatch):
    feedbacks = []
    _mock_writers(monkeypatch, feedbacks)
    # always invalid -> repair should stop at _MAX_REPAIR (2) then reproduce
    runs = iter([_invalid(), _invalid(), _invalid(), _invalid()])
    monkeypatch.setattr(gen_pipeline, "run_experiment",
                        lambda b, hc, mc, seed, dependencies=(), io_contract=None: next(runs))
    out = gen_pipeline.run_pipeline(_FakeBackend(), llm=object(), question="q")
    assert out["repairs"] == gen_pipeline._MAX_REPAIR   # bounded, never infinite


def test_repair_loop_skips_nonrepairable(monkeypatch):
    feedbacks = []
    _mock_writers(monkeypatch, feedbacks)
    # a harness/validate bug is NOT the method's fault -> no repair attempts
    runs = iter([_invalid(kind="validate_error", detail="TypeError"), _valid(1)])
    monkeypatch.setattr(gen_pipeline, "run_experiment",
                        lambda b, hc, mc, seed, dependencies=(), io_contract=None: next(runs))
    out = gen_pipeline.run_pipeline(_FakeBackend(), llm=object(), question="q")
    assert out["repairs"] == 0                         # did NOT loop on a harness bug
    assert len(feedbacks) == 1                         # only the initial write_method


# --- anti-cheat d: fresh random reproduction seed ------------------------------

def test_reproduce_uses_a_fresh_random_seed(monkeypatch):
    feedbacks = []
    _mock_writers(monkeypatch, feedbacks)
    seeds = []
    monkeypatch.setattr(
        gen_pipeline, "run_experiment",
        lambda b, hc, mc, seed, dependencies=(), io_contract=None: (seeds.append(seed), _valid(seed))[1],
    )
    gen_pipeline.run_pipeline(_FakeBackend(), llm=object(), question="q")
    assert seeds[0] == 0                 # primary = reference seed 0
    assert seeds[1] not in (0, 1)         # reproduce = a fresh random seed, not the old fixed 1
    # SMALL so seed-range-capped harnesses accept it (a 6-digit seed broke a real run)
    assert 2 <= seeds[1] <= gen_pipeline._REPRODUCE_SEED_MAX


def test_pipeline_respects_higher_is_better(monkeypatch):
    # knapsack-style: method value HIGHER than baseline -> PASS (not inverted to fail)
    monkeypatch.setattr(gen_pipeline, "design_experiment",
                        lambda llm, q: dict(_DESIGN, question=q, lower_is_better=False))
    monkeypatch.setattr(gen_pipeline, "write_harness", lambda b, h, d: "# h")
    monkeypatch.setattr(gen_pipeline, "write_method", lambda b, h, d, feedback=None: "def solve(i): return []")
    primary = {"baseline_metric": 1088.0, "method_metric": 1609.0, "valid": True, "seed": 0, "isolated": True}
    reproduce = {"baseline_metric": 1100.0, "method_metric": 1600.0, "valid": True, "seed": 7, "isolated": True}
    runs = iter([primary, reproduce])
    monkeypatch.setattr(gen_pipeline, "run_experiment",
                        lambda b, hc, mc, seed, dependencies=(), io_contract=None: next(runs))
    out = gen_pipeline.run_pipeline(_FakeBackend(), llm=object(), question="knapsack", seed_reproduce=7)
    assert out["verdict"]["verdict"] == "pass"   # 1609 > 1088 on higher-is-better = a real win


def test_pipeline_runs_k_reproduce_seeds(monkeypatch):
    monkeypatch.setattr(gen_pipeline, "_N_REPRODUCE", 3)
    monkeypatch.setattr(gen_pipeline, "design_experiment", lambda llm, q: dict(_DESIGN, question=q))
    monkeypatch.setattr(gen_pipeline, "write_harness", lambda b, h, d: "# h")
    monkeypatch.setattr(gen_pipeline, "write_method", lambda b, h, d, feedback=None: "def solve(i): return []")
    seeds = []
    def _run(b, hc, mc, seed, dependencies=(), io_contract=None):
        seeds.append(seed)
        # primary + 2/3 reproduces improve -> pass on majority
        return {"baseline_metric": 10, "method_metric": (8 if seed != seeds.count(seed) * -1 else 12),
                "valid": True, "seed": seed, "isolated": True}
    # simpler: primary(0) improved; reproduces improved except make one worse
    calls = {"n": 0}
    def _run2(b, hc, mc, seed, dependencies=(), io_contract=None):
        calls["n"] += 1
        seeds.append(seed)
        m = 12 if calls["n"] == 3 else 8   # 2nd reproduce (3rd call) fails; still 2/3 majority
        return {"baseline_metric": 10, "method_metric": m, "valid": True, "seed": seed, "isolated": True}
    monkeypatch.setattr(gen_pipeline, "run_experiment", _run2)
    out = gen_pipeline.run_pipeline(_FakeBackend(), llm=object(), question="q")
    assert len(out["reproduces"]) == 3          # ran K=3 reproduce seeds
    assert seeds[0] == 0                          # primary first
    assert out["verdict"]["verdict"] == "pass"    # 2/3 reproduced -> pass


def test_run_level_failure_retried_once(monkeypatch):
    monkeypatch.setattr(gen_pipeline, "_N_REPRODUCE", 1)
    monkeypatch.setattr(gen_pipeline, "design_experiment", lambda llm, q: dict(_DESIGN, question=q))
    monkeypatch.setattr(gen_pipeline, "write_harness", lambda b, h, d: "# h")
    monkeypatch.setattr(gen_pipeline, "write_method", lambda b, h, d, feedback=None: "def solve(i): return []")
    calls = {"n": 0}
    def _run(b, hc, mc, seed, dependencies=(), io_contract=None):
        calls["n"] += 1
        if calls["n"] == 1:  # primary's first run: a RUN-LEVEL flake (isolated False)
            return {"valid": False, "isolated": False, "baseline_metric": None,
                    "method_metric": None, "error": "container flake"}
        return {"baseline_metric": 10, "method_metric": 8, "valid": True, "seed": seed, "isolated": True}
    monkeypatch.setattr(gen_pipeline, "run_experiment", _run)
    events = []
    out = gen_pipeline.run_pipeline(_FakeBackend(), llm=object(), question="q",
                                    emit=lambda e, p: events.append((e, p)))
    assert calls["n"] == 3                              # primary(fail)->retry(ok) + reproduce(ok)
    assert out["verdict"]["verdict"] == "pass"          # the retry salvaged a clean result
    assert any(e == "run_retry" for e, _ in events)


def test_method_failure_not_retried_as_run(monkeypatch):
    # a METHOD failure (isolated True, valid False) is NOT a run-level retry -> the
    # format-repair loop handles it instead.
    monkeypatch.setattr(gen_pipeline, "_N_REPRODUCE", 1)
    monkeypatch.setattr(gen_pipeline, "design_experiment", lambda llm, q: dict(_DESIGN, question=q))
    monkeypatch.setattr(gen_pipeline, "write_harness", lambda b, h, d: "# h")
    monkeypatch.setattr(gen_pipeline, "write_method", lambda b, h, d, feedback=None: "def solve(i): return []")
    calls = {"n": 0}
    def _run(b, hc, mc, seed, dependencies=(), io_contract=None):
        calls["n"] += 1
        return {"valid": False, "isolated": True, "baseline_metric": 10, "method_metric": None,
                "diagnostics": {"first_failure": {"kind": "shape_mismatch", "detail": "x"}, "timed_out": False}}
    monkeypatch.setattr(gen_pipeline, "run_experiment", _run)
    events = []
    gen_pipeline.run_pipeline(_FakeBackend(), llm=object(), question="q",
                              emit=lambda e, p: events.append((e, p)))
    assert not any(e == "run_retry" for e, _ in events)   # no run-level retry
    assert any(e == "method_repair" for e, _ in events)    # repair loop fired instead


# --- A2: method-quality nudge (trivial-variant rewrite) ------------------------

def test_trivial_variant_detection():
    tv = gen_pipeline._trivial_variant
    assert tv({"valid": True, "baseline_metric": 10.0, "method_metric": 10.0}) is True
    assert tv({"valid": True, "baseline_metric": 10.0, "method_metric": 8.0}) is False
    assert tv({"valid": False, "baseline_metric": 10.0, "method_metric": None}) is False


def test_quality_repair_on_trivial_variant(monkeypatch):
    feedbacks = []
    _mock_writers(monkeypatch, feedbacks)  # sets _N_REPRODUCE=1
    runs = iter([
        {"baseline_metric": 10, "method_metric": 10, "valid": True, "seed": 0, "isolated": True},  # trivial variant
        {"baseline_metric": 10, "method_metric": 8, "valid": True, "seed": 0, "isolated": True},    # after quality rewrite
        {"baseline_metric": 10, "method_metric": 8, "valid": True, "seed": 1, "isolated": True},     # reproduce
    ])
    monkeypatch.setattr(gen_pipeline, "run_experiment",
                        lambda b, hc, mc, seed, dependencies=(), io_contract=None: next(runs))
    events = []
    out = gen_pipeline.run_pipeline(_FakeBackend(), llm=object(), question="q",
                                    emit=lambda e, p: events.append((e, p)))
    assert any(e == "method_quality_repair" for e, _ in events)
    assert gen_pipeline._QUALITY_FEEDBACK in feedbacks   # the leak-safe difference-feedback was fed
    assert out["verdict"]["verdict"] == "pass"


def test_no_quality_repair_when_method_differs(monkeypatch):
    feedbacks = []
    _mock_writers(monkeypatch, feedbacks)
    runs = iter([_valid(0), _valid(1)])   # method 8 vs baseline 10 -> clearly different
    monkeypatch.setattr(gen_pipeline, "run_experiment",
                        lambda b, hc, mc, seed, dependencies=(), io_contract=None: next(runs))
    events = []
    gen_pipeline.run_pipeline(_FakeBackend(), llm=object(), question="q",
                              emit=lambda e, p: events.append((e, p)))
    assert not any(e == "method_quality_repair" for e, _ in events)


def test_pipeline_uses_prereviewed_design_and_skips_generation(monkeypatch):
    """Design-review gate foundation: a human-REVIEWED design passed in is used as-is;
    the pipeline must NOT re-generate (and thus re-diverge from) the approved design."""
    primary = {"baseline_metric": 10, "method_metric": 8, "valid": True, "seed": 0, "isolated": True}
    reproduce = {"baseline_metric": 11, "method_metric": 9, "valid": True, "seed": 1, "isolated": True}
    _install_mocks(monkeypatch, primary=primary, reproduce=reproduce)
    def _boom(llm, q):
        raise AssertionError("design_experiment must NOT be called when a design is supplied")
    monkeypatch.setattr(gen_pipeline, "design_experiment", _boom)
    reviewed = dict(_DESIGN, question="q", task="HUMAN-REVIEWED-TASK")
    out = gen_pipeline.run_pipeline(_FakeBackend(), object(), "q", design=reviewed, seed_reproduce=1)
    assert out["design"]["task"] == "HUMAN-REVIEWED-TASK"


def test_pipeline_routes_gpu_experiment_to_remote_runner(monkeypatch):
    """A design with gpu=True runs the SCORED step on the remote GPU node (trusted
    orchestrator on the GPU card), NOT the local Docker runner."""
    monkeypatch.setattr(gen_pipeline, "_N_REPRODUCE", 1)
    monkeypatch.setattr(gen_pipeline, "design_experiment",
                        lambda llm, q: dict(_DESIGN, question=q, gpu=True))
    monkeypatch.setattr(gen_pipeline, "write_harness", lambda b, h, d: "# harness")
    monkeypatch.setattr(gen_pipeline, "write_method",
                        lambda b, h, d, feedback=None: "def solve(i): return []")
    monkeypatch.setattr(gen_pipeline, "run_experiment",
                        lambda *a, **k: (_ for _ in ()).throw(
                            AssertionError("local runner used for a GPU experiment")))
    calls = []
    def fake_remote(harness, method, seed, dependencies=(), io_contract=None):
        calls.append(seed)
        return {"baseline_metric": 1.0, "method_metric": 0.5, "valid": True,
                "seed": seed, "isolated": True}
    monkeypatch.setattr(gen_pipeline, "run_experiment_remote", fake_remote)

    out = gen_pipeline.run_pipeline(_FakeBackend(), object(), "does warmup help?",
                                    seed_reproduce=1)
    assert calls, "remote GPU runner was not called"
    assert out["verdict"] is not None
