"""Unit tests for M11 Task 4: orchestrator wiring for RunningExperiment ->
EvaluatingExperiment/ImplementingExperiment/MossIntervening + MAP counter F
run-failure loop.

Covers:
  - the RunningExperiment entry action (experiment.run_requested).
  - ``_on_RunningExperiment_experiment_run_completed`` (MAP counter F,
    implementation_fix_round[experiment_id], threshold 3):
      - verdict "succeeded" -> preserve F until Memory classifies the exact run;
        EvaluatingExperiment (no entry action -> pauses; M12);
        run_record_ref appended to run_record_refs.
      - verdict "failed"    -> bump F[experiment_id]; below threshold ->
        ImplementingExperiment (re-emits code.implementation_requested, i.e.
        CoderAgent re-generates); at/above threshold -> MossIntervening with
        current_moss_reason == "implementation_thrashing".

Also locks the M9 correction: ``_on_ImplementingExperiment_code_implemented``'s
COMPLIANT branch no longer resets F -- F now accumulates across the
implement -> run -> evaluate -> memory cycle and is reset by a non-failure
Memory diagnosis. See test_engine_coder_advance.py's updated
``test_code_implemented_compliant_advances_to_running_experiment``.

Reuses FakeBusClient from test_engine_min.py, mirroring
test_engine_runner_advance.py's patterns (does NOT auto-deliver; drives the
engine directly via ``eng.tick(event)``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from autoresearch.orchestrator.engine import OrchestratorEngine
from autoresearch.spec_store import SpecStore
from autoresearch.orchestrator.session_state import (
    SessionState,
    load_session_state,
    save_session_state,
)
from autoresearch.orchestrator.tests.unit.test_engine_min import FakeBusClient, fake_bus  # noqa: F401

SM_ROOT = Path(__file__).resolve().parents[2] / "state-machine"


@dataclass
class _Evt:
    event_type: str
    session_id: str
    emitter: str
    payload: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _seed_running_experiment_session(
    eng: OrchestratorEngine, sid: str, *, experiment_id: str = "exp1"
) -> str:
    """Persist a session in RunningExperiment with its upstream refs set."""
    experiment_plan_ref = f"experiment_plan://{sid}/final"
    experiment_patch_ref = f"experiment_patch://{sid}/final"
    verifier_spec_ref = f"verifier_spec://{sid}/final"
    state = SessionState(
        session_id=sid,
        user_id="u1",
        current_state="RunningExperiment",
        current_experiment_id=experiment_id,
        method_ref=f"method://{sid}/current",
        experiment_plan_refs=[experiment_plan_ref],
        experiment_patch_refs=[experiment_patch_ref],
        verifier_spec_ref=verifier_spec_ref,
    )
    await save_session_state(eng.client.db, state, auto_commit=True)
    return verifier_spec_ref


async def _set_map_counter(fake_bus, sid: str, name: str, key: str, value: int) -> None:
    """Directly set a map counter key on a persisted session (test setup only)."""
    state = await load_session_state(fake_bus.db, sid)
    assert state is not None
    getattr(state, name)[key] = value
    await save_session_state(fake_bus.db, state, auto_commit=True)


# ---------------------------------------------------------------------------
# (a): RunningExperiment entry action
# ---------------------------------------------------------------------------


async def test_running_experiment_entry_action_emits_run_requested(fake_bus):
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = "sess-a"
    verifier_spec_ref = await _seed_running_experiment_session(eng, sid)
    state = await load_session_state(fake_bus.db, sid)

    await eng._execute_entry_action(state)

    emitted = [p for p in fake_bus.published if p.event_type == "experiment.run_requested"]
    assert len(emitted) == 1
    assert emitted[0].payload == {
        "experiment_patch_ref": f"experiment_patch://{sid}/final",
        "verifier_spec_ref": verifier_spec_ref,
        "benchmark_spec_ref": None,
        "baseline_result_ref": None,
        "method_ref": f"method://{sid}/current",
        "confirmed_repo_url": None,   # no repo confirmed -> current (self-write/explicit) behavior
        "confirmed_boundary": {},     # no user boundary override -> pure inference
        "spec_interpretation_ref": None,
        "recover_completed_result": False,
    }


# ---------------------------------------------------------------------------
# (b): run_completed "succeeded" -> EvaluatingExperiment, F preserved for Memory
# ---------------------------------------------------------------------------


async def test_run_completed_succeeded_defers_counter_reset_to_memory(fake_bus):
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = "sess-b"
    await _seed_running_experiment_session(eng, sid)
    await _set_map_counter(fake_bus, sid, "implementation_fix_round", "exp1", 2)

    run_record_ref = f"run_record://{sid}/exp1"
    await eng.tick(
        _Evt(
            "experiment.run_completed",
            sid,
            "RunnerAgent",
            payload={"verdict": "succeeded", "run_record_ref": run_record_ref},
        )
    )

    st = await load_session_state(fake_bus.db, sid)
    assert st is not None
    assert st.current_state == "EvaluatingExperiment"
    assert st.implementation_fix_round == {"exp1": 2}
    assert st.run_record_refs == [run_record_ref]


async def test_failed_component_attempt_is_verified_before_debug_routing(fake_bus, tmp_path):
    store = SpecStore(tmp_path / "store")
    try:
        eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT, spec_store=store)
        await eng.start()
        sid = "sess-component-command-failure"
        await _seed_running_experiment_session(eng, sid)
        store.put(kind="gen_experiment_result", session_id=sid, key="current", obj={
            "verdict": "invalid", "method_metric": None,
            "component_execution": {
                "mode": "component", "component_id": "feature",
                "execution_succeeded": False, "attempts": 1,
            },
        }, emitter="RunnerAgent")

        await eng.tick(_Evt(
            "experiment.run_completed", sid, "RunnerAgent",
            payload={"verdict": "failed", "run_record_ref": f"run_record://{sid}/attempt1"},
        ))

        state = await load_session_state(fake_bus.db, sid)
        assert state.current_state == "EvaluatingExperiment"
        assert state.implementation_fix_round == {}
        assert state.run_record_refs == [f"run_record://{sid}/attempt1"]
    finally:
        store.close()


async def test_failed_component_ablation_is_evaluated_before_any_moss_route(fake_bus, tmp_path):
    """Sandbox component labels cannot bypass EvaluatingExperiment and memory consolidation."""
    store = SpecStore(tmp_path / "store")
    try:
        eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT, spec_store=store)
        await eng.start()
        sid = "sess-ablation"
        await _seed_running_experiment_session(eng, sid)
        run_ref = store.put(
            kind="run_record", session_id=sid, key="v1",
            obj={"succeeded": True}, emitter="RunnerAgent")
        store.put(
            kind="gen_experiment_result", session_id=sid, key="current",
            obj={
                "operator": "check-expectation",
                "ablation_report": [
                    {"component": "optimizer", "type": "gain", "actual": 0.0,
                     "expected": {"operator": "<=", "value": -0.02},
                     "evidence": {"source": "run_record", "ref": "artifact://opt"},
                     "verdict": "met"},
                    {"component": "scheduler", "type": "functional", "actual": 1.0,
                     "expected": {"operator": ">=", "value": 1.0},
                     "evidence": {"source": "run_record", "ref": "artifact://scheduler"},
                     "verdict": "met"},
                ],
            }, emitter="RunnerAgent")

        await eng.tick(_Evt(
            "experiment.run_completed", sid, "RunnerAgent",
            payload={"verdict": "succeeded", "run_record_ref": run_ref},
        ))

        st = await load_session_state(fake_bus.db, sid)
        assert st.current_state == "EvaluatingExperiment"
        assert st.current_moss_reason is None
        assert st.component_ablation_failed is False
        assert not [p for p in fake_bus.published if p.event_type == "moss.intervention_requested"]
    finally:
        store.close()


async def test_component_ablation_cap_is_not_used_before_evaluation(fake_bus, tmp_path):
    store = SpecStore(tmp_path / "store")
    try:
        eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT, spec_store=store)
        await eng.start()
        sid = "sess-ablation-cap"
        await _seed_running_experiment_session(eng, sid)
        state = await load_session_state(fake_bus.db, sid)
        state.component_revision_round = 3
        state.research_memory_ref = f"research_memory://{sid}/v1"
        await save_session_state(fake_bus.db, state, auto_commit=True)
        run_ref = store.put(kind="run_record", session_id=sid, key="v1",
                            obj={"succeeded": True}, emitter="RunnerAgent")
        store.put(kind="gen_experiment_result", session_id=sid, key="current",
                  obj={"operator": "check-expectation", "ablation_report": [
                      {"component": "optimizer", "type": "gain", "actual": "no change",
                       "expected": "-0.02 bpb", "verdict": "unmet"}]},
                  emitter="RunnerAgent")

        await eng.tick(_Evt("experiment.run_completed", sid, "RunnerAgent",
                            payload={"verdict": "succeeded", "run_record_ref": run_ref}))

        state = await load_session_state(fake_bus.db, sid)
        assert state.current_state == "EvaluatingExperiment"
        assert state.component_revision_round == 3
        assert not [p for p in fake_bus.published if p.event_type == "moss.intervention_requested"]
    finally:
        store.close()


async def test_revising_method_receives_failed_component_name(fake_bus):
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = "sess-component-feedback"
    state = SessionState(
        session_id=sid, user_id="u1", current_state="RevisingMethod",
        problem_model_ref=f"problem_model://{sid}/v1",
        literature_brief_ref=f"literature_brief://{sid}/v1",
        method_ref=f"method://{sid}/v1",
        component_ablation_failed=True,
        ablation_failed_component="optimizer",
    )

    await eng._execute_entry_action(state)

    request = [p for p in fake_bus.published if p.event_type == "method.revise_requested"][-1]
    assert request.payload["experiment_feedback"] == "component ablation failed: optimizer"


async def test_gain_components_met_nonwin_is_evaluated_before_method_routing(
        fake_bus, tmp_path):
    store = SpecStore(tmp_path / "store")
    try:
        eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT, spec_store=store)
        await eng.start()
        sid = "sess-method-contradiction"
        await _seed_running_experiment_session(eng, sid)
        run_ref = store.put(kind="run_record", session_id=sid, key="v1",
                            obj={"succeeded": True}, emitter="RunnerAgent")
        store.put(kind="gen_experiment_result", session_id=sid, key="current", obj={
            "operator": "check-expectation", "verdict": "valid_not_best",
            "ablation_report": [{
                "component": "optimizer", "type": "gain", "actual": "local proxy improved",
                "expected": "lower final val_bpb", "verdict": "met",
            }],
        }, emitter="RunnerAgent")

        await eng.tick(_Evt("experiment.run_completed", sid, "RunnerAgent",
                            payload={"verdict": "succeeded", "run_record_ref": run_ref}))

        state = await load_session_state(fake_bus.db, sid)
        assert state.current_state == "EvaluatingExperiment"
        assert state.current_moss_reason is None
        assert state.method_not_working is False
        assert state.method_swap_round == 0
    finally:
        store.close()


async def test_method_swap_cap_is_not_used_before_evaluation(fake_bus, tmp_path):
    store = SpecStore(tmp_path / "store")
    try:
        eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT, spec_store=store)
        await eng.start()
        sid = "sess-method-cap"
        await _seed_running_experiment_session(eng, sid)
        state = await load_session_state(fake_bus.db, sid)
        state.method_swap_round = 3
        state.research_memory_ref = f"research_memory://{sid}/v1"
        await save_session_state(fake_bus.db, state, auto_commit=True)
        run_ref = store.put(kind="run_record", session_id=sid, key="v1",
                            obj={"succeeded": True}, emitter="RunnerAgent")
        store.put(kind="gen_experiment_result", session_id=sid, key="current", obj={
            "operator": "check-expectation", "verdict": "valid_not_best",
            "ablation_report": [{
                "component": "optimizer", "type": "gain", "actual": "proxy improved",
                "expected": "lower val_bpb", "verdict": "met",
            }],
        }, emitter="RunnerAgent")

        await eng.tick(_Evt("experiment.run_completed", sid, "RunnerAgent",
                            payload={"verdict": "succeeded", "run_record_ref": run_ref}))

        state = await load_session_state(fake_bus.db, sid)
        assert state.current_state == "EvaluatingExperiment"
        assert state.method_swap_round == 3
    finally:
        store.close()


async def test_functional_component_met_without_gain_continues_to_evaluator(fake_bus, tmp_path):
    store = SpecStore(tmp_path / "store")
    try:
        eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT, spec_store=store)
        await eng.start()
        sid = "sess-functional-component"
        await _seed_running_experiment_session(eng, sid)
        run_ref = store.put(kind="run_record", session_id=sid, key="v1",
                            obj={"succeeded": True}, emitter="RunnerAgent")
        store.put(kind="gen_experiment_result", session_id=sid, key="current", obj={
            "operator": "check-expectation", "verdict": "valid_not_best",
            "ablation_report": [{
                "component": "decoupler", "type": "functional", "actual": "corr=0.04",
                "expected": "corr<0.1", "verdict": "met",
            }],
        }, emitter="RunnerAgent")

        await eng.tick(_Evt("experiment.run_completed", sid, "RunnerAgent",
                            payload={"verdict": "succeeded", "run_record_ref": run_ref}))

        state = await load_session_state(fake_bus.db, sid)
        assert state.current_state == "EvaluatingExperiment"
    finally:
        store.close()


async def test_low_confidence_gain_does_not_falsify_method(fake_bus, tmp_path):
    store = SpecStore(tmp_path / "store")
    try:
        eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT, spec_store=store)
        await eng.start()
        sid = "sess-low-confidence-gain"
        await _seed_running_experiment_session(eng, sid)
        run_ref = store.put(kind="run_record", session_id=sid, key="v1",
                            obj={"succeeded": True}, emitter="RunnerAgent")
        store.put(kind="gen_experiment_result", session_id=sid, key="current", obj={
            "operator": "check-expectation", "verdict": "valid_not_best",
            "ablation_report": [{
                "component": "optimizer", "type": "gain", "actual": "noisy proxy",
                "expected": "lower final val_bpb", "verdict": "low_confidence",
            }],
        }, emitter="RunnerAgent")

        await eng.tick(_Evt("experiment.run_completed", sid, "RunnerAgent",
                            payload={"verdict": "succeeded", "run_record_ref": run_ref}))

        state = await load_session_state(fake_bus.db, sid)
        assert state.current_state == "EvaluatingExperiment"
        assert state.method_not_working is False
    finally:
        store.close()


async def test_baseline_drift_is_evaluated_and_consolidated_before_rerouting(fake_bus, tmp_path):
    store = SpecStore(tmp_path / "store")
    try:
        eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT, spec_store=store)
        await eng.start()
        sid = "sess-baseline-drift"
        await _seed_running_experiment_session(eng, sid)
        state = await load_session_state(fake_bus.db, sid)
        state.baseline_result_ref = f"baseline_result://{sid}/v1"
        await save_session_state(fake_bus.db, state, auto_commit=True)
        run_ref = store.put(kind="run_record", session_id=sid, key="v1",
                            obj={"succeeded": True}, emitter="RunnerAgent")
        store.put(kind="gen_experiment_result", session_id=sid, key="current", obj={
            "operator": "improve", "verdict": "invalid", "accepted": False,
            "diagnosis": "baseline_drift", "reasons": ["repository commit changed"],
        }, emitter="RunnerAgent")

        await eng.tick(_Evt("experiment.run_completed", sid, "RunnerAgent",
                            payload={"verdict": "succeeded", "run_record_ref": run_ref}))

        state = await load_session_state(fake_bus.db, sid)
        assert state.current_state == "EvaluatingExperiment"
        assert state.baseline_result_ref == f"baseline_result://{sid}/v1"
        requests = [p for p in fake_bus.published
                    if p.event_type == "baseline.reproduction_requested"]
        assert requests == []
    finally:
        store.close()


async def test_trusted_execution_failure_does_not_reset_persisted_debug_counter(
        fake_bus, tmp_path):
    """A fresh sandbox process may succeed while its trusted phase diagnosis is still failure."""
    store = SpecStore(tmp_path / "store")
    try:
        eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT, spec_store=store)
        await eng.start()
        sid = "sess-trusted-execution-failure"
        await _seed_running_experiment_session(eng, sid)
        await _set_map_counter(fake_bus, sid, "implementation_fix_round", "exp1", 2)
        run_ref = store.put(
            kind="run_record", session_id=sid, key="v1",
            obj={"succeeded": True}, emitter="RunnerAgent")
        store.put(
            kind="gen_experiment_result", session_id=sid, key="current",
            obj={
                "verdict": "invalid", "accepted": False,
                "diagnosis": "execution_failure",
                "diagnosis_evidence": ["candidate process exited before a metric"],
            }, emitter="RunnerAgent")

        await eng.tick(_Evt(
            "experiment.run_completed", sid, "RunnerAgent",
            payload={"verdict": "succeeded", "run_record_ref": run_ref},
        ))

        state = await load_session_state(fake_bus.db, sid)
        assert state.current_state == "EvaluatingExperiment"
        assert state.implementation_fix_round == {"exp1": 2}
    finally:
        store.close()


async def test_stale_best_result_cannot_reset_repeated_execution_failure_cap(
        fake_bus, tmp_path):
    """The best-so-far artifact may describe an older valid run, not this failed run."""
    store = SpecStore(tmp_path / "store")
    try:
        eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT, spec_store=store)
        await eng.start()
        sid = "sess-stale-best-execution-failure"
        await _seed_running_experiment_session(eng, sid)
        await _set_map_counter(fake_bus, sid, "implementation_fix_round", "exp1", 2)
        run_ref = store.put(
            kind="run_record", session_id=sid, key="v1",
            obj={"succeeded": True}, emitter="RunnerAgent")
        # The current key is deliberately the retained best result from an older round.
        store.put(
            kind="gen_experiment_result", session_id=sid, key="current",
            obj={"verdict": "valid_not_best", "accepted": False},
            emitter="RunnerAgent")

        await eng.tick(_Evt(
            "experiment.run_completed", sid, "RunnerAgent",
            payload={"verdict": "succeeded", "run_record_ref": run_ref},
        ))
        state = await load_session_state(fake_bus.db, sid)
        state.current_state = "ConsolidatingMemory"
        await save_session_state(fake_bus.db, state, auto_commit=True)

        await eng.tick(_Evt(
            "memory.consolidated", sid, "MemoryConsolidatorAgent",
            payload={
                "routing_verdict": "continue_explore",
                "research_memory_ref": f"research_memory://{sid}/v1",
                "diagnosis": "execution_failure",
                "diagnosis_evidence": ["third sandbox wrote no parseable result"],
            },
        ))

        state = await load_session_state(fake_bus.db, sid)
        assert state.implementation_fix_round == {"exp1": 3}
        assert state.current_state == "MossIntervening"
        assert state.current_moss_reason == "implementation_thrashing"
    finally:
        store.close()


# ---------------------------------------------------------------------------
# (c): run_completed "failed", F absent -> bump to 1, retry ImplementingExperiment
# ---------------------------------------------------------------------------


async def test_run_completed_failed_missing_key_bumps_to_one_and_retries(fake_bus):
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = "sess-c"
    await _seed_running_experiment_session(eng, sid)

    n_before = len(fake_bus.published)
    await eng.tick(
        _Evt(
            "experiment.run_completed",
            sid,
            "RunnerAgent",
            payload={"verdict": "failed", "run_record_ref": f"run_record://{sid}/attempt1"},
        )
    )

    st = await load_session_state(fake_bus.db, sid)
    assert st is not None
    assert st.implementation_fix_round == {"exp1": 1}
    assert st.current_state == "ImplementingExperiment"
    assert st.current_moss_reason is None
    assert st.run_record_refs == [f"run_record://{sid}/attempt1"]

    new_events = fake_bus.published[n_before:]
    assert any(p.event_type == "code.implementation_requested" for p in new_events)


# ---------------------------------------------------------------------------
# (d): run_completed "failed" at threshold -> MossIntervening
# ---------------------------------------------------------------------------


async def test_run_completed_failed_at_threshold_trips_moss(fake_bus):
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = "sess-d"
    await _seed_running_experiment_session(eng, sid)
    await _set_map_counter(fake_bus, sid, "implementation_fix_round", "exp1", 2)

    n_before = len(fake_bus.published)
    await eng.tick(
        _Evt(
            "experiment.run_completed",
            sid,
            "RunnerAgent",
            payload={"verdict": "failed", "run_record_ref": f"run_record://{sid}/attempt3"},
        )
    )

    st = await load_session_state(fake_bus.db, sid)
    assert st is not None
    assert st.implementation_fix_round == {"exp1": 3}
    assert st.current_state == "MossIntervening"
    assert st.current_moss_reason == "implementation_thrashing"

    new_events = fake_bus.published[n_before:]
    assert not any(p.event_type == "code.implementation_requested" for p in new_events)


# ---------------------------------------------------------------------------
# unknown verdict: defensive no-op
# ---------------------------------------------------------------------------


async def test_run_completed_unknown_verdict_is_defensive_noop(fake_bus):
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = "sess-e"
    await _seed_running_experiment_session(eng, sid)

    n_before = len(fake_bus.published)
    await eng.tick(
        _Evt(
            "experiment.run_completed",
            sid,
            "RunnerAgent",
            payload={"verdict": "something_weird", "run_record_ref": f"run_record://{sid}/x"},
        )
    )

    st = await load_session_state(fake_bus.db, sid)
    assert st is not None
    assert st.current_state == "RunningExperiment"
    assert st.implementation_fix_round == {}
    assert fake_bus.published[n_before:] == []
