"""Unit tests for Task 4: CriticAgent verdict routing + counters A/C, plus
RankingJudgeAgent milestone Task 4's rank -> rank-informed critique co-audit
wired at CritiquingHypotheses.

Covers:
  - the CritiquingHypotheses entry action, which now (RankingJudgeAgent
    milestone Task 4) emits ``ranking.requested`` with {candidate_kind:
    "hypothesis", source_ref: hypothesis_bank_ref} -- NOT
    hypothesis.critique_requested directly; candidate_ids is omitted since
    the entry lambda has no store access (RankingJudgeAgent enumerates ALL
    bank hypotheses). (CritiquingDirection's direction.critique_requested
    entry action was wired in a prior milestone and is NOT touched here.)
  - ``_on_CritiquingHypotheses_ranking_ready``: IN-STATE (not a
    ``_transition``) -- stores ``hypothesis_ranking_report_ref`` and
    publishes ``hypothesis.critique_requested`` carrying
    {hypothesis_bank_ref, research_spec_ref, ranking_report_ref}, atomically
    (save -> publish -> one commit), mirroring
    ``_on_AuditingLiteratureNovelty_evidence_audited``. current_state stays
    CritiquingHypotheses.
  - ``_on_CritiquingDirection_direction_critiqued`` (counter A,
    direction_critique_round, threshold 3, reason "direction_thrashing"):
      - verdict "pass" -> reset A to 0; RoutingResearchMode (no entry action ->
        pauses).
      - verdict "fail" -> bump A; below threshold -> ModelingProblem
        (re-emits problem_modeling.model_requested); at/above threshold ->
        MossIntervening with current_moss_reason == "direction_thrashing".
      - unknown verdict -> defensive no-op (logged), no crash.
  - ``_on_CritiquingHypotheses_hypothesis_critiqued`` (counter C,
    hypothesis_reject_round, threshold 3, reason "hypothesis_stagnation"):
      - verdict "some_pass" -> reset C to 0; store the WHOLE
        ``passing_hypothesis_ids`` list on SessionState; transition to
        ManagingHypothesisPortfolio (no entry action -> pauses); does NOT set
        current_hypothesis_id and does NOT auto-advance (Task 5 replaces
        ``_advance_managing_hypothesis_portfolio`` with a ranking-driven
        entry action).
      - verdict "all_rejected" -> bump C; below threshold ->
        GeneratingHypotheses (re-emits hypothesis.generation_requested);
        at/above threshold -> MossIntervening with current_moss_reason ==
        "hypothesis_stagnation".
      - unknown verdict -> defensive no-op (logged), no crash.

Both handlers append ``event.payload["critique_report_ref"]`` to
``state.critique_report_refs`` regardless of verdict.

Reuses FakeBusClient from test_engine_min.py (does NOT auto-deliver; the test
drives the engine directly via ``eng.tick(event)`` / ``eng._execute_entry_action``),
mirroring test_engine_novelty_advance.py's and test_engine_hypothesis_advance.py's
patterns.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pytest

from autoresearch.orchestrator.engine import OrchestratorEngine
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


async def _advance_to_critiquing_direction(eng: OrchestratorEngine, sid: str) -> tuple[str, str]:
    """Drive a fresh session (ClarifyingResearchGoal) into CritiquingDirection.

    Returns (problem_model_ref, research_spec_ref) -- both read off
    SessionState by CritiquingDirection's entry action (already wired; not
    under test here).
    """
    research_spec_ref = f"spec://research_spec/{sid}/final"
    await eng.tick(
        _Evt(
            "research_spec.ready",
            sid,
            "ProblemModelingAgent",
            payload={"research_spec_ref": research_spec_ref},
        )
    )
    await eng.tick(
        _Evt(
            "literature.brief_ready",
            sid,
            "DeepResearchAgent",
            payload={
                "literature_brief_ref": f"spec://literature_brief/{sid}/final",
                "field_trajectory_brief_ref": f"spec://field_trajectory_brief/{sid}/final",
            },
        )
    )
    revised_research_spec_ref = f"research_spec://{sid}/revised"
    await eng.tick(
        _Evt(
            "spec.reinterpreted",
            sid,
            "ProblemModelingAgent",
            payload={
                "disposition": "in_scope",
                "spec_interpretation_ref": f"spec_interpretation://{sid}/final",
                "revised_research_spec_ref": revised_research_spec_ref,
            },
        )
    )
    await eng.tick(
        _Evt(
            "novelty.audit_ready",
            sid,
            "NoveltyAuditAgent",
            payload={
                "verdict": "pass",
                "novelty_audit_report_ref": f"novelty_audit_report://{sid}/final",
            },
        )
    )
    problem_model_ref = f"problem_model://{sid}/final"
    await eng.tick(
        _Evt(
            "problem.modeled",
            sid,
            "ProblemModelingAgent",
            payload={"problem_model_ref": problem_model_ref},
        )
    )
    await eng.tick(
        _Evt(
            "human.idea_provided",
            sid,
            "Human",
            payload={"inject": False},
        )
    )
    return problem_model_ref, revised_research_spec_ref


async def _seed_critiquing_hypotheses_session(eng: OrchestratorEngine, sid: str) -> tuple[str, str]:
    """Persist a session in CritiquingHypotheses with the 2 upstream refs set.

    Isolated construction (mirrors test_engine_hypothesis_advance.py's
    ``_seed_generating_hypotheses_session``): CritiquingHypotheses is reachable
    from GeneratingHypotheses, which is itself unreachable from the mainline
    at M5/M6, so this builds the SessionState directly.
    """
    research_spec_ref = f"research_spec://{sid}/revised"
    hypothesis_bank_ref = f"hypothesis_bank://{sid}/final"
    state = SessionState(
        session_id=sid,
        user_id="u1",
        current_state="CritiquingHypotheses",
        research_spec_ref=research_spec_ref,
        hypothesis_bank_ref=hypothesis_bank_ref,
    )
    await save_session_state(eng.client.db, state, auto_commit=True)
    return research_spec_ref, hypothesis_bank_ref


async def _set_counter(fake_bus, sid: str, name: str, value: int) -> None:
    """Directly set a scalar counter on a persisted session (test setup only)."""
    state = await load_session_state(fake_bus.db, sid)
    assert state is not None
    setattr(state, name, value)
    await save_session_state(fake_bus.db, state, auto_commit=True)


# ---------------------------------------------------------------------------
# (a)-(c): CritiquingDirection -> direction.critiqued (counter A)
#
# meeting 2026-07-25 item 7 (全删): CritiquingDirection is RETIRED -- the mainline no longer routes
# through it (the machine doesn't re-critique the human-confirmed direction). These handler tests
# exercise an unreachable state; they are SKIPPED and will be deleted together with the CriticAgent
# direction-critique subsystem in the C-phase (甲) cleanup.
# ---------------------------------------------------------------------------
_RETIRED_ITEM7 = pytest.mark.skip(
    reason="CritiquingDirection retired per meeting item 7; removed with CriticAgent in 甲 cleanup")


@_RETIRED_ITEM7
async def test_direction_critiqued_pass_advances_to_selecting_track(fake_bus):
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = await eng.start_session("goal", user_id="u1")
    await _advance_to_critiquing_direction(eng, sid)

    critique_report_ref = f"critique_report://{sid}/direction/final"
    n_before = len(fake_bus.published)
    commits_before = fake_bus.db.commit_count
    await eng.tick(
        _Evt(
            "direction.critiqued",
            sid,
            "CriticAgent",
            payload={"verdict": "pass", "critique_report_ref": critique_report_ref},
        )
    )

    st = await load_session_state(fake_bus.db, sid)
    assert st is not None
    # phase 3.1: a passed direction now goes to GeneratingMethod (derive the explicit method) FIRST,
    # before the track/benchmark is selected. GeneratingMethod emits method.generation_requested.
    assert st.current_state == "GeneratingMethod"
    assert st.direction_critique_round == 0
    assert st.critique_report_refs == [critique_report_ref]

    assert fake_bus.db.commit_count - commits_before == 1
    new_events = fake_bus.published[n_before:]
    assert sum(1 for p in new_events if p.event_type == "method.generation_requested") == 1


@_RETIRED_ITEM7
async def test_direction_critiqued_fail_first_round_returns_to_modeling_problem(fake_bus):
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = await eng.start_session("goal", user_id="u1")
    await _advance_to_critiquing_direction(eng, sid)

    n_before = len(fake_bus.published)
    await eng.tick(
        _Evt(
            "direction.critiqued",
            sid,
            "CriticAgent",
            payload={
                "verdict": "fail",
                "critique_report_ref": f"critique_report://{sid}/direction/round1",
            },
        )
    )

    st = await load_session_state(fake_bus.db, sid)
    assert st is not None
    assert st.direction_critique_round == 1
    assert st.current_state == "ModelingProblem"
    assert st.current_moss_reason is None

    new_events = fake_bus.published[n_before:]
    assert any(p.event_type == "problem_modeling.model_requested" for p in new_events)
    assert not any(p.event_type for p in new_events if "moss" in p.event_type.lower())


@_RETIRED_ITEM7
async def test_direction_critiqued_fail_at_threshold_trips_moss(fake_bus):
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = await eng.start_session("goal", user_id="u1")
    await _advance_to_critiquing_direction(eng, sid)
    await _set_counter(fake_bus, sid, "direction_critique_round", 2)

    n_before = len(fake_bus.published)
    await eng.tick(
        _Evt(
            "direction.critiqued",
            sid,
            "CriticAgent",
            payload={
                "verdict": "fail",
                "critique_report_ref": f"critique_report://{sid}/direction/round3",
            },
        )
    )

    st = await load_session_state(fake_bus.db, sid)
    assert st is not None
    assert st.direction_critique_round == 3
    assert st.current_state == "MossIntervening"
    assert st.current_moss_reason == "direction_thrashing"

    new_events = fake_bus.published[n_before:]
    assert not any(p.event_type == "problem_modeling.model_requested" for p in new_events)


@_RETIRED_ITEM7
async def test_direction_critiqued_unknown_verdict_is_defensive_noop(fake_bus):
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = await eng.start_session("goal", user_id="u1")
    await _advance_to_critiquing_direction(eng, sid)

    n_before = len(fake_bus.published)
    await eng.tick(
        _Evt(
            "direction.critiqued",
            sid,
            "CriticAgent",
            payload={
                "verdict": "something_weird",
                "critique_report_ref": f"critique_report://{sid}/direction/final",
            },
        )
    )

    st = await load_session_state(fake_bus.db, sid)
    assert st is not None
    assert st.current_state == "CritiquingDirection"
    assert st.direction_critique_round == 0
    assert fake_bus.published[n_before:] == []


# ---------------------------------------------------------------------------
# (d): CritiquingHypotheses entry action
# ---------------------------------------------------------------------------


async def test_critiquing_hypotheses_entry_action_emits_ranking_requested(fake_bus):
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = "sess-1"
    _research_spec_ref, hypothesis_bank_ref = await _seed_critiquing_hypotheses_session(eng, sid)
    state = await load_session_state(fake_bus.db, sid)
    assert state is not None

    n_before = len(fake_bus.published)
    await eng._execute_entry_action(state)

    new_events = fake_bus.published[n_before:]
    emitted = [p for p in new_events if p.event_type == "ranking.requested"]
    assert len(emitted) == 1
    assert emitted[0].session_id == sid
    assert emitted[0].payload == {
        "candidate_kind": "hypothesis",
        "source_ref": hypothesis_bank_ref,
    }
    # No hypothesis.critique_requested yet -- that only fires after
    # ranking.ready (_on_CritiquingHypotheses_ranking_ready, below).
    assert not [p for p in new_events if p.event_type == "hypothesis.critique_requested"]


# ---------------------------------------------------------------------------
# (d'): CritiquingHypotheses -> ranking.ready (IN-STATE co-audit)
# ---------------------------------------------------------------------------


async def test_ranking_ready_stores_ref_and_publishes_critique_requested_in_state(fake_bus):
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = "sess-ranking-1"
    research_spec_ref, hypothesis_bank_ref = await _seed_critiquing_hypotheses_session(eng, sid)

    n_before = len(fake_bus.published)
    commits_before = fake_bus.db.commit_count
    await eng.tick(
        _Evt(
            "ranking.ready",
            sid,
            "RankingJudgeAgent",
            payload={"ranking_report_ref": "r1", "candidate_kind": "hypothesis"},
        )
    )

    st = await load_session_state(fake_bus.db, sid)
    assert st is not None
    assert st.hypothesis_ranking_report_ref == "r1"
    # IN-STATE: current_state does NOT change.
    assert st.current_state == "CritiquingHypotheses"

    # Atomic: exactly one commit (save + publish bundled), mirroring
    # _on_AuditingLiteratureNovelty_evidence_audited.
    assert fake_bus.db.commit_count - commits_before == 1

    new_events = fake_bus.published[n_before:]
    emitted = [p for p in new_events if p.event_type == "hypothesis.critique_requested"]
    assert len(emitted) == 1
    assert emitted[0].session_id == sid
    assert emitted[0].payload == {
        "hypothesis_bank_ref": hypothesis_bank_ref,
        "research_spec_ref": research_spec_ref,
        "ranking_report_ref": "r1",
    }


# ---------------------------------------------------------------------------
# (e)-(g): CritiquingHypotheses -> hypothesis.critiqued (counter C)
# ---------------------------------------------------------------------------


async def test_hypothesis_critiqued_some_pass_advances_to_managing_portfolio(fake_bus):
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = "sess-2"
    await _seed_critiquing_hypotheses_session(eng, sid)

    critique_report_ref = f"critique_report://{sid}/hypothesis/final"
    commits_before = fake_bus.db.commit_count
    await eng.tick(
        _Evt(
            "hypothesis.critiqued",
            sid,
            "CriticAgent",
            payload={"verdict": "some_pass", "critique_report_ref": critique_report_ref},
        )
    )

    st = await load_session_state(fake_bus.db, sid)
    assert st is not None
    assert st.current_state == "ManagingHypothesisPortfolio"
    assert st.hypothesis_reject_round == 0
    assert st.critique_report_refs == [critique_report_ref]

    # ManagingHypothesisPortfolio has no entry action -> pauses -> one commit.
    assert fake_bus.db.commit_count - commits_before == 1


async def test_hypothesis_critiqued_all_rejected_first_round_returns_to_generating(fake_bus):
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = "sess-3"
    await _seed_critiquing_hypotheses_session(eng, sid)

    n_before = len(fake_bus.published)
    await eng.tick(
        _Evt(
            "hypothesis.critiqued",
            sid,
            "CriticAgent",
            payload={
                "verdict": "all_rejected",
                "critique_report_ref": f"critique_report://{sid}/hypothesis/round1",
            },
        )
    )

    st = await load_session_state(fake_bus.db, sid)
    assert st is not None
    assert st.hypothesis_reject_round == 1
    assert st.current_state == "GeneratingHypotheses"
    assert st.current_moss_reason is None

    new_events = fake_bus.published[n_before:]
    assert any(p.event_type == "hypothesis.generation_requested" for p in new_events)


async def test_hypothesis_critiqued_all_rejected_at_threshold_trips_moss(fake_bus):
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = "sess-4"
    await _seed_critiquing_hypotheses_session(eng, sid)
    await _set_counter(fake_bus, sid, "hypothesis_reject_round", 2)

    n_before = len(fake_bus.published)
    await eng.tick(
        _Evt(
            "hypothesis.critiqued",
            sid,
            "CriticAgent",
            payload={
                "verdict": "all_rejected",
                "critique_report_ref": f"critique_report://{sid}/hypothesis/round3",
            },
        )
    )

    st = await load_session_state(fake_bus.db, sid)
    assert st is not None
    assert st.hypothesis_reject_round == 3
    assert st.current_state == "MossIntervening"
    assert st.current_moss_reason == "hypothesis_stagnation"

    new_events = fake_bus.published[n_before:]
    assert not any(p.event_type == "hypothesis.generation_requested" for p in new_events)


async def test_hypothesis_critiqued_unknown_verdict_is_defensive_noop(fake_bus):
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = "sess-5"
    await _seed_critiquing_hypotheses_session(eng, sid)

    n_before = len(fake_bus.published)
    await eng.tick(
        _Evt(
            "hypothesis.critiqued",
            sid,
            "CriticAgent",
            payload={
                "verdict": "something_weird",
                "critique_report_ref": f"critique_report://{sid}/hypothesis/final",
            },
        )
    )

    st = await load_session_state(fake_bus.db, sid)
    assert st is not None
    assert st.current_state == "CritiquingHypotheses"
    assert st.hypothesis_reject_round == 0
    assert fake_bus.published[n_before:] == []


async def _pass_method_novelty_audit(eng, sid):
    # The deliberated method first gets its own targeted literature pass. Drive that result, then
    # the evidence -> novelty co-audit, to a pass so the flow reaches ConfirmingMethod.
    state = await load_session_state(eng.client.db, sid)
    targeted_id = state.novelty_targeted_search_id
    literature_payload = {
        "literature_brief_ref": f"lb://{sid}/method",
        "field_trajectory_brief_ref": f"ft://{sid}/method",
        "method_ref": state.method_ref,
        "targeted_search_id": targeted_id,
    }
    if state.novelty_audit_report_ref:
        literature_payload["novelty_audit_report_ref"] = state.novelty_audit_report_ref
    await eng.tick(_Evt("literature.brief_ready", sid, "DeepResearchAgent",
                        payload=literature_payload))
    await eng.tick(_Evt("evidence.audited", sid, "EvidenceAuditAgent",
                        payload={"evidence_audit_report_ref": f"e://{sid}/1",
                                 "targeted_search_id": targeted_id}))
    await eng.tick(_Evt("novelty.audit_ready", sid, "NoveltyAuditAgent",
                        payload={"verdict": "pass", "novelty_audit_report_ref": f"n://{sid}/1",
                                 "targeted_search_id": targeted_id}))


async def _freeze_first_benchmark_baseline(eng, sid):
    await eng.tick(_Evt(
        "benchmark.spec_ready", sid, "BenchmarkSelectorAgent",
        payload={"benchmark_spec_ref": f"benchmark_spec://{sid}/v1",
                 "benchmark_id": "benchmark-1"},
    ))
    await eng.tick(_Evt(
        "baseline.reproduced", sid, "ExperimentRunnerAgent",
        payload={"verdict": "reproduced",
                 "baseline_result_ref": f"baseline_result://{sid}/v1",
                 "benchmark_id": "benchmark-1"},
    ))


async def test_method_step_generate_confirm_then_selecting_track(fake_bus):
    # A benchmark freezes its baseline before Method generation and deliberation.
    # Then: GeneratingMethod -> (method.generated) -> DeliberatingMethod
    # (bounded proposer<->critic debate) -> (method.deliberated) -> DeepResearching (searches the
    # resulting method) -> AuditingLiteratureNovelty -> ConfirmingMethod -> (confirmed, no inject) ->
    # ComponentPlan audit -> PlanningExperiment.
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = await eng.start_session("goal", user_id="u1")
    await _advance_to_critiquing_direction(eng, sid)
    await eng.tick(_Evt("direction.critiqued", sid, "CriticAgent",
                        payload={"verdict": "pass", "critique_report_ref": f"cr://{sid}/1"}))
    st = await load_session_state(fake_bus.db, sid)
    assert st.current_state == "SelectingBenchmark"
    await _freeze_first_benchmark_baseline(eng, sid)
    st = await load_session_state(fake_bus.db, sid)
    assert st.current_state == "GeneratingMethod"
    assert st.baseline_result_ref == f"baseline_result://{sid}/v1"
    await eng.tick(_Evt("method.generated", sid, "MethodAgent",
                        payload={"method_ref": f"method://{sid}/v1"}))
    st = await load_session_state(fake_bus.db, sid)
    # meeting ①: the draft method first goes through the bounded proposer<->critic debate.
    assert st.current_state == "DeliberatingMethod" and st.method_ref == f"method://{sid}/v1"
    await eng.tick(_Evt("method.deliberated", sid, "MethodAgent",
                        payload={"method_ref": f"method://{sid}/v1"}))
    st = await load_session_state(fake_bus.db, sid)
    assert st.current_state == "DeepResearching" and st.method_ref == f"method://{sid}/v1"
    searches = [p for p in fake_bus.published if p.event_type == "deep_research.requested"]
    assert searches[-1].payload["method_ref"] == f"method://{sid}/v1"
    assert searches[-1].payload["targeted_search_id"] == st.novelty_targeted_search_id
    await _pass_method_novelty_audit(eng, sid)
    st = await load_session_state(fake_bus.db, sid)
    assert st.current_state == "ConfirmingMethod"
    await eng.tick(_Evt("method.confirmed", sid, "Human", payload={"inject": False}))
    st = await load_session_state(fake_bus.db, sid)
    assert st.current_state == "DeliberatingMethod"
    assert st.component_plan_audit_pending is True
    assert st.baseline_result_ref == f"baseline_result://{sid}/v1"


async def test_replacement_method_confirm_reuses_existing_repo_baseline(fake_bus):
    """A method swap must not select the same benchmark and reproduce its baseline again."""
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = "replacement-method-reuses-baseline"
    state = SessionState(
        session_id=sid,
        user_id="u1",
        current_state="ConfirmingMethod",
        mode="benchmark",
        method_ref=f"method://{sid}/v2",
        method_swap_round=1,
        research_spec_ref=f"research_spec://{sid}/v1",
        benchmark_spec_ref=f"benchmark_spec://{sid}/v1",
        baseline_result_ref=f"baseline_result://{sid}/v1",
        current_benchmark_id="repo-benchmark",
        confirmed_repo_url="https://example.invalid/owner/repo",
    )
    await save_session_state(fake_bus.db, state, auto_commit=True)

    n_before = len(fake_bus.published)
    await eng.tick(_Evt("method.confirmed", sid, "Human", payload={"inject": False}))

    st = await load_session_state(fake_bus.db, sid)
    assert st is not None
    assert st.current_state == "DeliberatingMethod"
    assert st.baseline_result_ref == f"baseline_result://{sid}/v1"
    assert st.current_experiment_id is None
    assert st.component_plan_audit_pending is True
    emitted = [item for item in fake_bus.published[n_before:]]
    assert [item.event_type for item in emitted] == ["method.deliberation_requested"]
    assert emitted[0].payload["component_plan_audit"] is True


async def test_replacement_method_selection_reentry_still_reuses_same_baseline(fake_bus):
    """Resume recovery: a swap already in selection must not relaunch the same baseline."""
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = "replacement-method-selection-reentry"
    original_spec = f"benchmark_spec://{sid}/v1"
    state = SessionState(
        session_id=sid,
        user_id="u1",
        current_state="SelectingBenchmark",
        mode="benchmark",
        method_ref=f"method://{sid}/v2",
        method_swap_round=1,
        research_spec_ref=f"research_spec://{sid}/v1",
        benchmark_spec_ref=original_spec,
        baseline_result_ref=f"baseline_result://{sid}/v1",
        current_benchmark_id="repo-benchmark",
    )
    await save_session_state(fake_bus.db, state, auto_commit=True)

    n_before = len(fake_bus.published)
    await eng.tick(_Evt(
        "benchmark.spec_ready", sid, "BenchmarkSelectorAgent",
        payload={
            "benchmark_spec_ref": f"benchmark_spec://{sid}/duplicate",
            "benchmark_id": "repo-benchmark",
        },
    ))

    st = await load_session_state(fake_bus.db, sid)
    assert st is not None
    assert st.current_state == "DeliberatingMethod"
    assert st.benchmark_spec_ref == original_spec
    assert st.baseline_result_ref == f"baseline_result://{sid}/v1"
    assert st.current_experiment_id is None
    assert st.component_plan_audit_pending is True
    emitted = [item for item in fake_bus.published[n_before:]]
    assert [item.event_type for item in emitted] == ["method.deliberation_requested"]
    assert emitted[0].payload["component_plan_audit"] is True


async def test_method_confirm_inject_re_generates(fake_bus):
    # phase 3.1/3.4: injecting a change at ConfirmingMethod re-derives the method (back to
    # GeneratingMethod), not straight forward. (Reaches ConfirmingMethod via the 3.3 novelty audit.)
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = await eng.start_session("goal", user_id="u1")
    await _advance_to_critiquing_direction(eng, sid)
    await eng.tick(_Evt("direction.critiqued", sid, "CriticAgent",
                        payload={"verdict": "pass", "critique_report_ref": f"cr://{sid}/1"}))
    await _freeze_first_benchmark_baseline(eng, sid)
    await eng.tick(_Evt("method.generated", sid, "MethodAgent", payload={"method_ref": f"m://{sid}/v1"}))
    await eng.tick(_Evt("method.deliberated", sid, "MethodAgent", payload={"method_ref": f"m://{sid}/v1"}))
    await _pass_method_novelty_audit(eng, sid)
    st = await load_session_state(fake_bus.db, sid)
    assert st.current_state == "ConfirmingMethod"
    n = len(fake_bus.published)
    await eng.tick(_Evt("method.confirmed", sid, "Human",
                        payload={"inject": True, "human_idea_seed_ref": f"seed://{sid}/v1"}))
    st = await load_session_state(fake_bus.db, sid)
    assert st.current_state == "GeneratingMethod"     # re-derive, not proceed
    assert st.human_idea_seed_ref == f"seed://{sid}/v1"
    assert [p for p in fake_bus.published[n:] if p.event_type == "method.generation_requested"]
