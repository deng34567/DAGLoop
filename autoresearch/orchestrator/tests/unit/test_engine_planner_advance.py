"""Unit tests for Task 5: orchestrator wiring for PlanningExperiment ->
ReviewingExperimentPlan + MAP counter E.

Covers:
  - the PlanningExperiment entry action (experiment.planning_requested).
  - ``_on_PlanningExperiment_experiment_plan_ready`` -> ReviewingExperimentPlan,
    appending the ref to ``experiment_plan_refs``, whose entry action emits
    experiment.review_requested.
  - ``_on_ReviewingExperimentPlan_experiment_plan_reviewed`` (counter E,
    experiment_plan_review_round[hypothesis_id], threshold 3, reason
    "experiment_design_thrashing"):
      - verdict "pass" -> reset E[hypothesis_id] (delete key); ImplementingExperiment
        (no entry action -> pauses).
      - verdict "fail" -> bump E[hypothesis_id]; below threshold -> PlanningExperiment
        (re-emits experiment.planning_requested); at/above threshold ->
        MossIntervening with current_moss_reason == "experiment_design_thrashing".

Reuses FakeBusClient/fake_bus from test_engine_min.py and ``_Evt`` /
``_set_map_counter`` from test_engine_evaluator_advance.py (does NOT
auto-deliver; drives the engine directly via ``eng.tick(event)``), mirroring
the existing advance-test files' patterns.
"""

from __future__ import annotations

from pathlib import Path

from autoresearch.orchestrator.engine import OrchestratorEngine
from autoresearch.orchestrator.session_state import (
    SessionState,
    load_session_state,
    save_session_state,
)
from autoresearch.orchestrator.tests.unit.test_engine_evaluator_advance import (
    _Evt,
    _set_map_counter,
)
from autoresearch.orchestrator.tests.unit.test_engine_min import FakeBusClient, fake_bus  # noqa: F401

SM_ROOT = Path(__file__).resolve().parents[2] / "state-machine"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _seed_planning_experiment_session(
    eng: OrchestratorEngine, sid: str, *, hypothesis_id: str = "h1"
) -> SessionState:
    """Persist a session in PlanningExperiment with its upstream refs set."""
    state = SessionState(
        session_id=sid,
        user_id="u1",
        current_state="PlanningExperiment",
        hypothesis_bank_ref=f"hypothesis_bank://{sid}/final",
        current_hypothesis_id=hypothesis_id,
        verifier_spec_ref=f"verifier_spec://{sid}/final",
        benchmark_spec_ref=f"benchmark_spec://{sid}/bench-1",
        research_spec_ref=f"research_spec://{sid}/revised",
    )
    await save_session_state(eng.client.db, state, auto_commit=True)
    return state


async def _seed_reviewing_experiment_plan_session(
    eng: OrchestratorEngine, sid: str, *, hypothesis_id: str = "h1"
) -> str:
    """Persist a session in ReviewingExperimentPlan with its upstream refs set."""
    experiment_plan_ref = f"experiment_plan://{sid}/draft1"
    state = SessionState(
        session_id=sid,
        user_id="u1",
        current_state="ReviewingExperimentPlan",
        current_hypothesis_id=hypothesis_id,
        verifier_spec_ref=f"verifier_spec://{sid}/final",
        experiment_plan_refs=[experiment_plan_ref],
    )
    await save_session_state(eng.client.db, state, auto_commit=True)
    return experiment_plan_ref


# ---------------------------------------------------------------------------
# (a): PlanningExperiment entry action
# ---------------------------------------------------------------------------


async def test_planning_experiment_entry_action_emits_planning_requested(fake_bus):
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = "sess-a"
    state = await _seed_planning_experiment_session(eng, sid)

    await eng._execute_entry_action(state)

    emitted = [p for p in fake_bus.published if p.event_type == "experiment.planning_requested"]
    assert len(emitted) == 1
    assert emitted[0].payload == {
        # 甲: method_ref added (the confirmed method to implement on the benchmark mainline).
        "method_ref": state.method_ref,
        "hypothesis_bank_ref": state.hypothesis_bank_ref,
        "selected_hypothesis_id": "h1",
        "verifier_spec_ref": state.verifier_spec_ref,
        "benchmark_spec_ref": state.benchmark_spec_ref,
        "research_spec_ref": state.research_spec_ref,
        "baseline_result_ref": state.baseline_result_ref,
        "spec_interpretation_ref": state.spec_interpretation_ref,
        "research_memory_ref": state.research_memory_ref,
        "plan_review_ref": None,
    }


# ---------------------------------------------------------------------------
# (b): PlanningExperiment -> experiment.plan_ready -> ReviewingExperimentPlan
# ---------------------------------------------------------------------------


async def test_experiment_plan_ready_advances_to_reviewing_experiment_plan(fake_bus):
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = "sess-b"
    state = await _seed_planning_experiment_session(eng, sid)
    state.research_memory_ref = f"spec://research_memory/{sid}/bound"
    await save_session_state(fake_bus.db, state, auto_commit=True)

    experiment_plan_ref = f"experiment_plan://{sid}/draft1"
    n_before = len(fake_bus.published)
    await eng.tick(
        _Evt(
            "experiment.plan_ready",
            sid,
            "ExperimentPlannerAgent",
            payload={"experiment_plan_ref": experiment_plan_ref},
        )
    )

    st = await load_session_state(fake_bus.db, sid)
    assert st is not None
    assert st.current_state == "ReviewingExperimentPlan"
    assert st.experiment_plan_refs == [experiment_plan_ref]

    new_events = fake_bus.published[n_before:]
    emitted = [p for p in new_events if p.event_type == "experiment.review_requested"]
    assert len(emitted) == 1
    assert emitted[0].payload == {
        "experiment_plan_ref": experiment_plan_ref,
        "verifier_spec_ref": st.verifier_spec_ref,
        "benchmark_spec_ref": st.benchmark_spec_ref,
        "baseline_result_ref": st.baseline_result_ref,
        "method_ref": st.method_ref,
        "research_memory_ref": st.research_memory_ref,
    }


# ---------------------------------------------------------------------------
# (c)-(e): ReviewingExperimentPlan -> experiment.plan_reviewed (counter E)
# ---------------------------------------------------------------------------


async def test_experiment_plan_reviewed_pass_resets_counter_and_advances(fake_bus):
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = "sess-c"
    await _seed_reviewing_experiment_plan_session(eng, sid)
    await _set_map_counter(fake_bus, sid, "experiment_plan_review_round", "h1", 2)

    review_report_ref = f"review_report://{sid}/final"
    commits_before = fake_bus.db.commit_count
    await eng.tick(
        _Evt(
            "experiment.plan_reviewed",
            sid,
            "ExperimentReviewAgent",
            payload={"review_report_ref": review_report_ref, "verdict": "pass"},
        )
    )

    st = await load_session_state(fake_bus.db, sid)
    assert st is not None
    assert st.current_state == "ImplementingExperiment"
    assert "h1" not in st.experiment_plan_review_round
    assert st.critique_report_refs == [review_report_ref]
    assert fake_bus.db.commit_count - commits_before == 1


async def test_experiment_plan_reviewed_fail_missing_key_bumps_to_one_and_replans(fake_bus):
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = "sess-d"
    await _seed_reviewing_experiment_plan_session(eng, sid)

    n_before = len(fake_bus.published)
    await eng.tick(
        _Evt(
            "experiment.plan_reviewed",
            sid,
            "ExperimentReviewAgent",
            payload={"review_report_ref": f"review_report://{sid}/draft", "verdict": "fail"},
        )
    )

    st = await load_session_state(fake_bus.db, sid)
    assert st is not None
    assert st.experiment_plan_review_round == {"h1": 1}
    assert st.current_state == "PlanningExperiment"
    assert st.current_moss_reason is None

    new_events = fake_bus.published[n_before:]
    assert any(p.event_type == "experiment.planning_requested" for p in new_events)


async def test_benchmark_method_review_counter_uses_method_ref_not_null(fake_bus):
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = "sess-method-review"
    method_ref = f"spec://method/{sid}/v1"
    state = SessionState(
        session_id=sid,
        user_id="u1",
        current_state="ReviewingExperimentPlan",
        current_hypothesis_id=None,
        method_ref=method_ref,
        experiment_plan_refs=[f"spec://experiment_plan/{sid}/v1"],
    )
    await save_session_state(eng.client.db, state, auto_commit=True)

    await eng.tick(
        _Evt(
            "experiment.plan_reviewed",
            sid,
            "CriticAgent",
            payload={"review_report_ref": f"spec://critique_report/{sid}/v1",
                     "verdict": "fail"},
        )
    )

    st = await load_session_state(fake_bus.db, sid)
    assert st is not None
    assert st.experiment_plan_review_round == {method_ref: 1}
    assert "null" not in st.experiment_plan_review_round


async def test_experiment_plan_reviewed_fail_at_threshold_trips_moss(fake_bus):
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = "sess-e"
    await _seed_reviewing_experiment_plan_session(eng, sid)
    await _set_map_counter(fake_bus, sid, "experiment_plan_review_round", "h1", 2)

    n_before = len(fake_bus.published)
    await eng.tick(
        _Evt(
            "experiment.plan_reviewed",
            sid,
            "ExperimentReviewAgent",
            payload={"review_report_ref": f"review_report://{sid}/draft3", "verdict": "fail"},
        )
    )

    st = await load_session_state(fake_bus.db, sid)
    assert st is not None
    assert st.experiment_plan_review_round == {"h1": 3}
    assert st.current_state == "MossIntervening"
    assert st.current_moss_reason == "experiment_design_thrashing"

    new_events = fake_bus.published[n_before:]
    assert not any(p.event_type == "experiment.planning_requested" for p in new_events)
