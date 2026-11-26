"""Unit tests for DataAnalysisAgent + LabInTheLoop milestone Task 4: the
lab-track ENTRY + mode-branches.

Covers:
  - ``_on_AwaitingUserDecision_user_decision_provided`` (dispatch for
    ``user.decision_provided``), now routed via the user-decision
    reason-matrix (see test_user_decision_matrix.py for the matrix/guard
    itself and test_engine_user_decision_advance.py for the general
    matrix-routing coverage); this file only keeps the out_of_scope-specific
    cases:
      - reason "out_of_scope", decision "proceed_out_of_scope" ->
        state.mode = "lab_in_the_loop"; transition into RoutingResearchMode (its
        matrix-mandated next_state; a no-entry-action pause).
      - an illegal decision for the recorded reason -> defensive no-op
        (logged), stays paused at AwaitingUserDecision, mode unchanged.
  - ``_route_research_mode`` ADD-ed lab branch:
      - mode "lab_in_the_loop" -> GeneratingHypotheses (skips
        benchmark/verifier).
      - mode "benchmark" (default) -> SelectingBenchmark, UNCHANGED.
  - ``_on_ReviewingExperimentPlan_experiment_plan_reviewed`` pass branch,
    mode-routed:
      - mode "lab_in_the_loop" -> PreparingExternalExperiment, immediately
        auto-advanced (DataAnalysisAgent + LabInTheLoop milestone Task 5's
        ``_advance_preparing_external_experiment``) to
        AwaitingExternalExperimentResult -- see
        test_engine_external_experiment_advance.py for the auto-advance
        itself; this file only asserts the pre-Task-5 transition still fires.
      - mode "benchmark" (default) -> ImplementingExperiment, UNCHANGED.
      - fail branch (counter E) -> MossIntervening/PlanningExperiment
        routing UNCHANGED regardless of mode.

Every mode-branch DEFAULTS to benchmark -- the existing mainline e2e (M0-M11)
must stay byte-for-byte unchanged; that invariant is exercised here by the
explicit mode="benchmark" cases alongside the new lab_in_the_loop ones.

Reuses FakeBusClient/fake_bus from test_engine_min.py (does NOT auto-deliver;
drives the engine directly via ``eng.tick(event)`` / helper calls), mirroring
test_engine_planner_advance.py's and test_engine_critic_advance.py's patterns.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from autoresearch.orchestrator.engine import OrchestratorEngine
from autoresearch.orchestrator.session_state import (
    SessionState,
    load_session_state,
    save_session_state,
)
from autoresearch.orchestrator.tests.unit.test_engine_evaluator_advance import _set_map_counter
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


async def _seed_awaiting_user_decision_session(eng: OrchestratorEngine, sid: str) -> SessionState:
    """Persist a session in AwaitingUserDecision, refs set as they would be by
    the time out_of_scope is decided at ReinterpretingSpec (see
    ``_on_ReinterpretingSpec_spec_reinterpreted``): research_spec_ref,
    literature_brief_ref, spec_interpretation_ref all present, plus the
    ``awaiting_user_decision_reason`` that entry point now sets.
    """
    state = SessionState(
        session_id=sid,
        user_id="u1",
        current_state="AwaitingUserDecision",
        awaiting_user_decision_reason="out_of_scope",
        research_spec_ref=f"research_spec://{sid}/revised",
        literature_brief_ref=f"literature_brief://{sid}/final",
        spec_interpretation_ref=f"spec_interpretation://{sid}/final",
    )
    await save_session_state(eng.client.db, state, auto_commit=True)
    return state


async def _seed_selecting_track_session(eng: OrchestratorEngine, sid: str, *, mode: str) -> SessionState:
    """Persist a session in RoutingResearchMode with refs for either branch's
    entry action (GeneratingHypotheses needs research_spec_ref +
    literature_brief_ref; SelectingBenchmark needs the same two)."""
    state = SessionState(
        session_id=sid,
        user_id="u1",
        current_state="RoutingResearchMode",
        mode=mode,
        research_spec_ref=f"research_spec://{sid}/revised",
        literature_brief_ref=f"literature_brief://{sid}/final",
    )
    await save_session_state(eng.client.db, state, auto_commit=True)
    return state


async def _seed_reviewing_experiment_plan_session(
    eng: OrchestratorEngine, sid: str, *, mode: str, hypothesis_id: str = "h1"
) -> str:
    """Persist a session in ReviewingExperimentPlan with its upstream refs
    set (mirrors test_engine_planner_advance.py's helper, plus ``mode``)."""
    experiment_plan_ref = f"experiment_plan://{sid}/draft1"
    state = SessionState(
        session_id=sid,
        user_id="u1",
        current_state="ReviewingExperimentPlan",
        mode=mode,
        current_hypothesis_id=hypothesis_id,
        verifier_spec_ref=f"verifier_spec://{sid}/final",
        experiment_plan_refs=[experiment_plan_ref],
    )
    await save_session_state(eng.client.db, state, auto_commit=True)
    return experiment_plan_ref


# ---------------------------------------------------------------------------
# (a)-(b): AwaitingUserDecision -> user.decision_provided
# ---------------------------------------------------------------------------


async def test_proceed_out_of_scope_sets_lab_mode_and_advances_to_selecting_track(fake_bus):
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = "sess-a"
    await _seed_awaiting_user_decision_session(eng, sid)

    n_before = len(fake_bus.published)
    await eng.tick(
        _Evt(
            "user.decision_provided",
            sid,
            "Human",
            payload={"decision": "proceed_out_of_scope"},
        )
    )

    st = await load_session_state(fake_bus.db, sid)
    assert st is not None
    assert st.mode == "lab_in_the_loop"
    # user-decision reason-matrix: out_of_scope.proceed_out_of_scope -> RoutingResearchMode
    # (a no-entry-action pause; RoutingResearchMode only branches via
    # ``_route_research_mode``, which nothing here calls).
    assert st.current_state == "RoutingResearchMode"
    assert st.awaiting_user_decision_reason is None

    new_events = fake_bus.published[n_before:]
    assert new_events == []


async def test_unknown_decision_is_defensive_noop(fake_bus):
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = "sess-b"
    await _seed_awaiting_user_decision_session(eng, sid)

    n_before = len(fake_bus.published)
    await eng.tick(
        _Evt(
            "user.decision_provided",
            sid,
            "Human",
            payload={"decision": "abort"},
        )
    )

    st = await load_session_state(fake_bus.db, sid)
    assert st is not None
    assert st.current_state == "AwaitingUserDecision"
    assert st.awaiting_user_decision_reason == "out_of_scope"
    assert st.mode == "benchmark"
    assert fake_bus.published[n_before:] == []


# ---------------------------------------------------------------------------
# (c): _route_research_mode mode branches
# ---------------------------------------------------------------------------


async def test_route_research_mode_lab_mode_advances_to_generating_hypotheses(fake_bus):
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = "sess-c"
    state = await _seed_selecting_track_session(eng, sid, mode="lab_in_the_loop")

    await eng._route_research_mode(state)

    st = await load_session_state(fake_bus.db, sid)
    assert st is not None
    assert st.current_state == "GeneratingHypotheses"


async def test_route_research_mode_benchmark_mode_goes_to_deliberating(fake_bus):
    # benchmark mode now inserts the bounded deep-deliberation step BEFORE SelectingBenchmark:
    # RoutingResearchMode -> Deliberating (entry action emits deliberation.requested).
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = "sess-d"
    state = await _seed_selecting_track_session(eng, sid, mode="benchmark")

    n_before = len(fake_bus.published)
    await eng._route_research_mode(state)

    st = await load_session_state(fake_bus.db, sid)
    assert st is not None
    # Meeting 2026-07-29: Deliberating (深思) retired -> benchmark mode routes straight to
    # SelectingBenchmark (no deep-deliberation hop before the experiment).
    assert st.current_state == "SelectingBenchmark"
    new = fake_bus.published[n_before:]
    assert any(p.event_type == "benchmark.selection_requested" for p in new)


# ---------------------------------------------------------------------------
# (d): ReviewingExperimentPlan pass/fail branches, mode-routed
# ---------------------------------------------------------------------------


async def test_experiment_plan_reviewed_pass_lab_mode_advances_to_preparing_external(fake_bus):
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = "sess-e"
    await _seed_reviewing_experiment_plan_session(eng, sid, mode="lab_in_the_loop")

    await eng.tick(
        _Evt(
            "experiment.plan_reviewed",
            sid,
            "ExperimentReviewAgent",
            payload={"review_report_ref": f"review_report://{sid}/final", "verdict": "pass"},
        )
    )

    st = await load_session_state(fake_bus.db, sid)
    assert st is not None
    # PreparingExternalExperiment is immediately auto-advanced (Task 5's
    # _advance_preparing_external_experiment) to AwaitingExternalExperimentResult;
    # see test_engine_external_experiment_advance.py for that auto-advance itself.
    assert st.current_state == "AwaitingExternalExperimentResult"
    assert "h1" not in st.experiment_plan_review_round


async def test_experiment_plan_reviewed_pass_benchmark_mode_unchanged(fake_bus):
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = "sess-f"
    await _seed_reviewing_experiment_plan_session(eng, sid, mode="benchmark")

    await eng.tick(
        _Evt(
            "experiment.plan_reviewed",
            sid,
            "ExperimentReviewAgent",
            payload={"review_report_ref": f"review_report://{sid}/final", "verdict": "pass"},
        )
    )

    st = await load_session_state(fake_bus.db, sid)
    assert st is not None
    assert st.current_state == "ImplementingExperiment"


async def test_experiment_plan_reviewed_fail_routing_unchanged_regardless_of_mode(fake_bus):
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = "sess-g"
    await _seed_reviewing_experiment_plan_session(eng, sid, mode="lab_in_the_loop")
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
