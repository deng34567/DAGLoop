"""Unit tests for M12 Task 4: orchestrator wiring for EvaluatingExperiment ->
{VerifyingBestResult | ConsolidatingMemory | MossIntervening} and
VerifyingBestResult -> {ConsolidatingMemory | RunningExperiment |
MossIntervening} + MAP counter G (result_verification_fail_round).

Covers:
  - the EvaluatingExperiment entry action (experiment.evaluation_requested).
  - the VerifyingBestResult entry action (result.verification_requested).
  - ``_on_EvaluatingExperiment_experiment_evaluated``:
      - verdict "invalid" / "valid_not_best" -> ConsolidatingMemory (no entry
        action -> pauses; M13); evaluation_report_ref appended.
      - verdict "new_best" -> VerifyingBestResult (entry action emits
        result.verification_requested).
      - verdict "suspicious" -> MossIntervening with current_moss_reason ==
        "result_suspicious".
  - ``_on_VerifyingBestResult_result_verified`` (MAP counter G,
    result_verification_fail_round[result_id], threshold 2, reason
    "result_suspicious"):
      - verdict "verified" -> reset G[result_id] (delete key);
        ConsolidatingMemory (no entry action -> pauses; M13).
      - verdict "failed", G absent -> bump G[result_id] to 1; below threshold
        -> RunningExperiment (re-emits experiment.run_requested, i.e.
        stricter rerun).
      - verdict "failed", G already at 1 -> bump to 2, at/above threshold ->
        MossIntervening with current_moss_reason == "result_suspicious".

Also regression-checks (h): M11 RunningExperiment advance (delegated to
test_engine_running_advance.py, run alongside this file) and M8/M7
(SelectingBenchmark/PreparingVerifier, test_engine_evaluator_advance.py)
still green.

Reuses FakeBusClient from test_engine_min.py, mirroring
test_engine_running_advance.py's patterns (does NOT auto-deliver; drives the
engine directly via ``eng.tick(event)``).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

from autoresearch.orchestrator.engine import OrchestratorEngine
from autoresearch.orchestrator.session_state import (
    SessionState,
    load_session_state,
    save_session_state,
)
from autoresearch.orchestrator.tests.unit.test_engine_min import FakeBusClient, fake_bus  # noqa: F401
from autoresearch.spec_store import SpecStore

SM_ROOT = Path(__file__).resolve().parents[2] / "state-machine"


@dataclass
class _Evt:
    event_type: str
    session_id: str
    emitter: str
    payload: dict = field(default_factory=dict)


def _artifact_digest(obj: dict) -> str:
    canonical = json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return "sha256:" + hashlib.sha256(canonical.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _seed_evaluating_experiment_session(
    eng: OrchestratorEngine, sid: str, *, experiment_id: str = "exp1"
) -> None:
    """Persist a session in EvaluatingExperiment with its upstream refs set."""
    state = SessionState(
        session_id=sid,
        user_id="u1",
        current_state="EvaluatingExperiment",
        current_experiment_id=experiment_id,
        run_record_refs=[f"run_record://{sid}/final"],
        verifier_spec_ref=f"verifier_spec://{sid}/final",
        benchmark_spec_ref=f"benchmark_spec://{sid}/final",
        baseline_result_ref=f"baseline_result://{sid}/final",
    )
    await save_session_state(eng.client.db, state, auto_commit=True)


async def _seed_verifying_best_result_session(
    eng: OrchestratorEngine, sid: str, *, experiment_id: str = "exp1"
) -> None:
    """Persist a session in VerifyingBestResult with its upstream refs set."""
    state = SessionState(
        session_id=sid,
        user_id="u1",
        current_state="VerifyingBestResult",
        current_experiment_id=experiment_id,
        experiment_patch_refs=[f"experiment_patch://{sid}/final"],
        run_record_refs=[f"run_record://{sid}/final"],
        evaluation_report_refs=[f"evaluation_report://{sid}/final"],
        verifier_spec_ref=f"verifier_spec://{sid}/final",
    )
    await save_session_state(eng.client.db, state, auto_commit=True)


async def _set_map_counter(fake_bus, sid: str, name: str, key: str, value: int) -> None:
    """Directly set a map counter key on a persisted session (test setup only)."""
    state = await load_session_state(fake_bus.db, sid)
    assert state is not None
    getattr(state, name)[key] = value
    await save_session_state(fake_bus.db, state, auto_commit=True)


# ---------------------------------------------------------------------------
# (a): entry actions
# ---------------------------------------------------------------------------


async def test_evaluating_experiment_entry_action_emits_evaluation_requested(fake_bus):
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = "sess-a1"
    await _seed_evaluating_experiment_session(eng, sid)
    state = await load_session_state(fake_bus.db, sid)

    await eng._execute_entry_action(state)

    emitted = [p for p in fake_bus.published if p.event_type == "experiment.evaluation_requested"]
    assert len(emitted) == 1
    assert emitted[0].payload == {
        "run_record_ref": f"run_record://{sid}/final",
        "verifier_spec_ref": f"verifier_spec://{sid}/final",
        "benchmark_spec_ref": f"benchmark_spec://{sid}/final",
        "baseline_result_ref": f"baseline_result://{sid}/final",
        "research_memory_ref": None,
        "spec_interpretation_ref": None,
    }


async def test_verifying_best_result_entry_action_emits_verification_requested(fake_bus):
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = "sess-a2"
    await _seed_verifying_best_result_session(eng, sid)
    state = await load_session_state(fake_bus.db, sid)

    await eng._execute_entry_action(state)

    emitted = [p for p in fake_bus.published if p.event_type == "result.verification_requested"]
    assert len(emitted) == 1
    assert emitted[0].payload == {
        "evaluation_report_ref": f"evaluation_report://{sid}/final",
        "run_record_ref": f"run_record://{sid}/final",
        "verifier_spec_ref": f"verifier_spec://{sid}/final",
        "spec_interpretation_ref": None,
    }


# ---------------------------------------------------------------------------
# (b): experiment.evaluated "new_best" -> VerifyingBestResult
# ---------------------------------------------------------------------------


async def test_experiment_evaluated_new_best_advances_to_verifying_best_result(fake_bus):
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = "sess-b"
    await _seed_evaluating_experiment_session(eng, sid)

    report_ref = f"evaluation_report://{sid}/attempt1"
    await eng.tick(
        _Evt(
            "experiment.evaluated",
            sid,
            "EvaluatorAgent",
            payload={"verdict": "new_best", "evaluation_report_ref": report_ref},
        )
    )

    st = await load_session_state(fake_bus.db, sid)
    assert st is not None
    assert st.current_state == "VerifyingBestResult"
    assert st.evaluation_report_refs == [report_ref]


# ---------------------------------------------------------------------------
# (c): every ordinary result is verified before memory consolidation
# ---------------------------------------------------------------------------


async def test_experiment_evaluated_invalid_advances_to_verification(fake_bus):
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = "sess-c1"
    await _seed_evaluating_experiment_session(eng, sid)

    report_ref = f"evaluation_report://{sid}/attempt1"
    await eng.tick(
        _Evt(
            "experiment.evaluated",
            sid,
            "EvaluatorAgent",
            payload={"verdict": "invalid", "evaluation_report_ref": report_ref},
        )
    )

    st = await load_session_state(fake_bus.db, sid)
    assert st is not None
    assert st.current_state == "VerifyingBestResult"
    assert st.evaluation_report_refs == [report_ref]


async def test_current_evaluation_binds_to_current_run_after_unevaluated_failure(fake_bus):
    """An earlier failed run must not make the current real evaluation look stale."""
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = "sess-current-evaluation-binding"
    historical_failed_run = f"run_record://{sid}/historical-failed"
    current_run = f"run_record://{sid}/current"
    state = SessionState(
        session_id=sid, user_id="u1", current_state="EvaluatingExperiment",
        current_experiment_id="exp-current",
        run_record_refs=[historical_failed_run, current_run],
        verifier_spec_ref=f"verifier_spec://{sid}/final",
        benchmark_spec_ref=f"benchmark_spec://{sid}/final",
    )
    await save_session_state(fake_bus.db, state, auto_commit=True)
    report_ref = f"evaluation_report://{sid}/current"

    await eng.tick(_Evt(
        "experiment.evaluated", sid, "EvaluatorAgent",
        payload={"verdict": "invalid", "evaluation_report_ref": report_ref},
    ))

    request = [p for p in fake_bus.published
               if p.event_type == "result.verification_requested"][-1]
    assert request.payload["run_record_ref"] == current_run
    assert request.payload["evaluation_report_ref"] == report_ref


async def test_overwritten_v1_run_accepts_evaluation_with_current_content_identity(
        fake_bus, tmp_path):
    """The content gate must accept a report created from the current bytes at the reused ref."""
    store = SpecStore(tmp_path / "store")
    try:
        eng = OrchestratorEngine(
            client=fake_bus, skill_root=SM_ROOT, spec_store=store)
        await eng.start()
        sid = "sess-current-v1-evaluation"
        same_run_ref = store.put(
            kind="run_record", session_id=sid, key="v1",
            obj={
                "succeeded": False, "exit_code": 1, "stdout_excerpt": "old failure",
                "stderr_excerpt": "old", "environment": "old", "artifacts": [],
            }, emitter="RunnerAgent")
        current_run = {
            "succeeded": True, "exit_code": 0, "metric_value": 0.8,
            "stdout_excerpt": "current candidate", "stderr_excerpt": None,
            "environment": "current", "artifacts": [],
        }
        assert store.put(
            kind="run_record", session_id=sid, key="v1",
            obj=current_run, emitter="RunnerAgent") == same_run_ref
        current_digest = _artifact_digest(current_run)
        report_ref = store.put(
            kind="evaluation_report", session_id=sid, key="v1",
            obj={
                "verdict": "valid_not_best", "is_improvement": False,
                "suspicious": False, "anti_cheat_flags": [],
                "run_record_digest": current_digest,
            }, emitter="EvaluatorAgent")
        state = SessionState(
            session_id=sid, user_id="u1", current_state="EvaluatingExperiment",
            current_experiment_id="exp-current",
            run_record_refs=[same_run_ref, same_run_ref],
            verifier_spec_ref=f"verifier_spec://{sid}/v1",
            benchmark_spec_ref=f"benchmark_spec://{sid}/v1",
        )
        await save_session_state(fake_bus.db, state, auto_commit=True)

        await eng.tick(_Evt(
            "experiment.evaluated", sid, "EvaluatorAgent",
            payload={
                "verdict": "valid_not_best", "evaluation_report_ref": report_ref,
                "run_record_digest": current_digest,
            },
        ))

        request = [p for p in fake_bus.published
                   if p.event_type == "result.verification_requested"][-1]
        assert request.payload["run_record_ref"] == same_run_ref
        assert request.payload["evaluation_report_ref"] == report_ref
    finally:
        store.close()


async def test_experiment_evaluated_valid_not_best_advances_to_verification(fake_bus):
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = "sess-c2"
    await _seed_evaluating_experiment_session(eng, sid)

    report_ref = f"evaluation_report://{sid}/attempt1"
    await eng.tick(
        _Evt(
            "experiment.evaluated",
            sid,
            "EvaluatorAgent",
            payload={"verdict": "valid_not_best", "evaluation_report_ref": report_ref},
        )
    )

    st = await load_session_state(fake_bus.db, sid)
    assert st is not None
    assert st.current_state == "VerifyingBestResult"
    assert st.evaluation_report_refs == [report_ref]


# ---------------------------------------------------------------------------
# (d): experiment.evaluated "suspicious" -> MossIntervening
# ---------------------------------------------------------------------------


async def test_experiment_evaluated_suspicious_trips_moss(fake_bus):
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = "sess-d"
    await _seed_evaluating_experiment_session(eng, sid)

    report_ref = f"evaluation_report://{sid}/attempt1"
    await eng.tick(
        _Evt(
            "experiment.evaluated",
            sid,
            "EvaluatorAgent",
            payload={"verdict": "suspicious", "evaluation_report_ref": report_ref},
        )
    )

    st = await load_session_state(fake_bus.db, sid)
    assert st is not None
    assert st.current_state == "MossIntervening"
    assert st.current_moss_reason == "result_suspicious"
    assert st.evaluation_report_refs == [report_ref]


async def test_correctable_comparison_change_returns_to_method_with_a_bounded_retry(
        fake_bus, tmp_path):
    """A trusted semantic correction is actionable, not merely discarded as suspicious."""
    store = SpecStore(tmp_path / "comparison-store")
    try:
        eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT, spec_store=store)
        await eng.start()
        sid = "sess-comparison-correction"
        await _seed_evaluating_experiment_session(eng, sid)
        state = await load_session_state(fake_bus.db, sid)
        state.experiment_plan_refs = [f"experiment_plan://{sid}/v1"]
        await save_session_state(fake_bus.db, state, auto_commit=True)
        report_ref = store.put(
            kind="evaluation_report", session_id=sid, key="v1",
            obj={
                "verdict": "suspicious", "suspicious": True,
                "anti_cheat_flags": [
                    "comparison_semantics_changed:method used a different model scale"],
                "reasoning": (
                    "Required correction: rerun the method with the frozen model scale."),
            }, emitter="EvaluatorAgent")

        await eng.tick(_Evt(
            "experiment.evaluated", sid, "EvaluatorAgent",
            payload={"verdict": "suspicious", "evaluation_report_ref": report_ref},
        ))

        st = await load_session_state(fake_bus.db, sid)
        assert st.current_state == "ImplementingExperiment"
        assert st.result_verification_fail_round == {"exp1": 1}
        assert st.current_moss_reason is None
        assert [p for p in fake_bus.published
                if p.event_type == "code.implementation_requested"]
    finally:
        store.close()


async def test_structured_preflight_component_failure_consolidates_before_redesign(
        fake_bus, tmp_path):
    """An impossible confirmed component is method evidence, not a run bug to retry."""
    store = SpecStore(tmp_path / "preflight-component-store")
    try:
        eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT, spec_store=store)
        await eng.start()
        sid = "sess-preflight-component"
        await _seed_evaluating_experiment_session(eng, sid)
        store.put(
            kind="gen_experiment_result", session_id=sid, key="current",
            obj={
                "verdict": "invalid", "accepted": False,
                "baseline_metric": 1.0, "method_metric": None,
                "ablation_report": [{
                    "component": "undefined optimizer", "type": "gain", "actual": 0,
                    "expected": {"operator": "==", "value": 1},
                    "evidence": {
                        "source": "tracked_repository_source", "ref": "train.py:1-20"},
                    "verdict": "unmet",
                }],
            }, emitter="RunnerAgent")
        report_ref = store.put(
            kind="evaluation_report", session_id=sid, key="v1",
            obj={
                "verdict": "suspicious", "suspicious": True,
                "anti_cheat_flags": [
                    "comparison_semantics_changed:no method metric was produced"],
                "reasoning": "Required correction: choose an implementable method.",
            }, emitter="EvaluatorAgent")

        await eng.tick(_Evt(
            "experiment.evaluated", sid, "EvaluatorAgent",
            payload={"verdict": "suspicious", "evaluation_report_ref": report_ref},
        ))

        st = await load_session_state(fake_bus.db, sid)
        assert st.current_state == "ConsolidatingMemory"
        assert st.result_verification_fail_round == {}
        requests = [p for p in fake_bus.published
                    if p.event_type == "memory.consolidation_requested"]
        assert requests
    finally:
        store.close()


async def test_repeated_comparison_change_reaches_existing_suspicious_moss_cap(
        fake_bus, tmp_path):
    store = SpecStore(tmp_path / "comparison-cap-store")
    try:
        eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT, spec_store=store)
        await eng.start()
        sid = "sess-comparison-cap"
        await _seed_evaluating_experiment_session(eng, sid)
        state = await load_session_state(fake_bus.db, sid)
        state.experiment_plan_refs = [f"experiment_plan://{sid}/v1"]
        state.result_verification_fail_round = {"exp1": 1}
        await save_session_state(fake_bus.db, state, auto_commit=True)
        report_ref = store.put(
            kind="evaluation_report", session_id=sid, key="v1",
            obj={
                "verdict": "suspicious", "suspicious": True,
                "anti_cheat_flags": [
                    "comparison_semantics_changed:method changed the evaluator"],
                "reasoning": "Required correction: restore the frozen evaluator.",
            }, emitter="EvaluatorAgent")

        await eng.tick(_Evt(
            "experiment.evaluated", sid, "EvaluatorAgent",
            payload={"verdict": "suspicious", "evaluation_report_ref": report_ref},
        ))

        st = await load_session_state(fake_bus.db, sid)
        assert st.current_state == "MossIntervening"
        assert st.current_moss_reason == "result_suspicious"
        assert st.result_verification_fail_round == {"exp1": 2}
    finally:
        store.close()


# ---------------------------------------------------------------------------
# (M13 Task 5): experiment.evaluated "conflict" -> {VerifyingBestResult |
# MossIntervening}, routed by SCALAR counter I (inconsistency_bounce_count,
# threshold 2, reason "result_inconsistency").
# ---------------------------------------------------------------------------


async def test_experiment_evaluated_conflict_below_threshold_advances_to_verifying_best_result(fake_bus):
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = "sess-conflict1"
    await _seed_evaluating_experiment_session(eng, sid)

    report_ref = f"evaluation_report://{sid}/attempt1"
    await eng.tick(
        _Evt(
            "experiment.evaluated",
            sid,
            "EvaluatorAgent",
            payload={"verdict": "conflict", "evaluation_report_ref": report_ref},
        )
    )

    st = await load_session_state(fake_bus.db, sid)
    assert st is not None
    assert st.inconsistency_bounce_count == 1
    assert st.current_state == "VerifyingBestResult"
    assert st.current_moss_reason is None
    assert st.evaluation_report_refs == [report_ref]


async def test_experiment_evaluated_conflict_at_threshold_trips_moss(fake_bus):
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = "sess-conflict2"
    await _seed_evaluating_experiment_session(eng, sid)
    state = await load_session_state(fake_bus.db, sid)
    assert state is not None
    state.inconsistency_bounce_count = 1
    await save_session_state(fake_bus.db, state, auto_commit=True)

    report_ref = f"evaluation_report://{sid}/attempt2"
    await eng.tick(
        _Evt(
            "experiment.evaluated",
            sid,
            "EvaluatorAgent",
            payload={"verdict": "conflict", "evaluation_report_ref": report_ref},
        )
    )

    st = await load_session_state(fake_bus.db, sid)
    assert st is not None
    assert st.inconsistency_bounce_count == 2
    assert st.current_state == "MossIntervening"
    assert st.current_moss_reason == "result_inconsistency"
    assert st.evaluation_report_refs == [report_ref]


# ---------------------------------------------------------------------------
# experiment.evaluated unknown verdict: defensive no-op
# ---------------------------------------------------------------------------


async def test_experiment_evaluated_unknown_verdict_is_defensive_noop(fake_bus):
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = "sess-unk1"
    await _seed_evaluating_experiment_session(eng, sid)

    n_before = len(fake_bus.published)
    await eng.tick(
        _Evt(
            "experiment.evaluated",
            sid,
            "EvaluatorAgent",
            payload={"verdict": "something_weird", "evaluation_report_ref": "x"},
        )
    )

    st = await load_session_state(fake_bus.db, sid)
    assert st is not None
    assert st.current_state == "EvaluatingExperiment"
    assert fake_bus.published[n_before:] == []


# ---------------------------------------------------------------------------
# (e): result.verified "verified" -> ConsolidatingMemory, G reset (absent)
# ---------------------------------------------------------------------------


async def test_result_verified_verified_advances_to_consolidating_memory_and_resets_g(fake_bus):
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = "sess-e"
    await _seed_verifying_best_result_session(eng, sid)
    await _set_map_counter(fake_bus, sid, "result_verification_fail_round", "exp1", 1)

    report_ref = f"verification_report://{sid}/attempt1"
    await eng.tick(
        _Evt(
            "result.verified",
            sid,
            "VerifierAgent",
            payload={"verdict": "verified", "verification_report_ref": report_ref},
        )
    )

    st = await load_session_state(fake_bus.db, sid)
    assert st is not None
    assert st.current_state == "ConsolidatingMemory"
    assert "exp1" not in st.result_verification_fail_round
    assert st.verification_report_refs == [report_ref]


# ---------------------------------------------------------------------------
# (f): result.verified "failed", G absent -> bump to 1, retry RunningExperiment
# ---------------------------------------------------------------------------


async def test_result_verified_failed_missing_key_bumps_to_one_and_reruns(fake_bus):
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = "sess-f"
    await _seed_verifying_best_result_session(eng, sid)

    n_before = len(fake_bus.published)
    report_ref = f"verification_report://{sid}/attempt1"
    await eng.tick(
        _Evt(
            "result.verified",
            sid,
            "VerifierAgent",
            payload={"verdict": "failed", "verification_report_ref": report_ref},
        )
    )

    st = await load_session_state(fake_bus.db, sid)
    assert st is not None
    assert st.result_verification_fail_round == {"exp1": 1}
    assert st.current_state == "RunningExperiment"
    assert st.current_moss_reason is None
    assert st.verification_report_refs == [report_ref]

    new_events = fake_bus.published[n_before:]
    assert any(p.event_type == "experiment.run_requested" for p in new_events)


async def test_failed_component_integration_consolidates_for_composition_retry(
        fake_bus, tmp_path):
    store = SpecStore(tmp_path / "integration-verification-store")
    try:
        eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT, spec_store=store)
        await eng.start()
        sid = "sess-integration-verification"
        await _seed_verifying_best_result_session(eng, sid)
        store.put(
            kind="gen_experiment_result", session_id=sid, key="current",
            obj={
                "verdict": "invalid", "accepted": False,
                "component_execution": {
                    "mode": "integration", "status": "execution_failed",
                    "all_components_accepted": True,
                },
            }, emitter="RunnerAgent")

        report_ref = f"verification_report://{sid}/attempt1"
        await eng.tick(_Evt(
            "result.verified", sid, "VerifierAgent",
            payload={"verdict": "failed", "verification_report_ref": report_ref},
        ))

        st = await load_session_state(fake_bus.db, sid)
        assert st.current_state == "ConsolidatingMemory"
        assert st.result_verification_fail_round == {}
        assert st.current_moss_reason is None
        assert st.verification_report_refs == [report_ref]
    finally:
        store.close()


# ---------------------------------------------------------------------------
# (g): result.verified "failed" at threshold -> MossIntervening
# ---------------------------------------------------------------------------


async def test_result_verified_failed_at_threshold_trips_moss(fake_bus):
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = "sess-g"
    await _seed_verifying_best_result_session(eng, sid)
    await _set_map_counter(fake_bus, sid, "result_verification_fail_round", "exp1", 1)

    n_before = len(fake_bus.published)
    report_ref = f"verification_report://{sid}/attempt2"
    await eng.tick(
        _Evt(
            "result.verified",
            sid,
            "VerifierAgent",
            payload={"verdict": "failed", "verification_report_ref": report_ref},
        )
    )

    st = await load_session_state(fake_bus.db, sid)
    assert st is not None
    assert st.result_verification_fail_round == {"exp1": 2}
    assert st.current_state == "MossIntervening"
    assert st.current_moss_reason == "result_suspicious"

    new_events = fake_bus.published[n_before:]
    assert not any(p.event_type == "experiment.run_requested" for p in new_events)


# ---------------------------------------------------------------------------
# result.verified unknown verdict: defensive no-op
# ---------------------------------------------------------------------------


async def test_result_verified_unknown_verdict_is_defensive_noop(fake_bus):
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = "sess-unk2"
    await _seed_verifying_best_result_session(eng, sid)

    n_before = len(fake_bus.published)
    await eng.tick(
        _Evt(
            "result.verified",
            sid,
            "VerifierAgent",
            payload={"verdict": "something_weird", "verification_report_ref": "x"},
        )
    )

    st = await load_session_state(fake_bus.db, sid)
    assert st is not None
    assert st.current_state == "VerifyingBestResult"
    assert st.result_verification_fail_round == {}
    assert fake_bus.published[n_before:] == []
