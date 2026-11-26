"""Unit tests for M13 Task 4: orchestrator ConsolidatingMemory routing.

Covers:
  - the ConsolidatingMemory entry action (memory.consolidation_requested),
    including the None-guards on research_memory_ref (unset) and
    verification_report_ref (empty verification_report_refs list).
  - ``_on_ConsolidatingMemory_memory_consolidated``:
      - routing_verdict "continue_explore" -> GeneratingHypotheses (its
        EXISTING M5 entry action re-emits hypothesis.generation_requested,
        CLOSING the exploration loop); research_memory_ref stored.
      - routing_verdict "report" -> Reporting (no entry action -> pauses;
        M14).
      - routing_verdict "need_experiment" -> PlanningExperiment (its
        EXISTING M8 entry action emits experiment.planning_requested).
      - unknown routing_verdict -> defensive no-op (logged).

Also regression-checks (e): M12 evaluation advance, M11 running advance, and
M8 planner advance still green (run alongside this file).

Reuses FakeBusClient from test_engine_min.py, mirroring
test_engine_evaluation_advance.py's patterns (does NOT auto-deliver; drives
the engine directly via ``eng.tick(event)``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pytest

from autoresearch.orchestrator.engine import (
    OrchestratorEngine, _COMPONENT_REVISION_CAP, _METHOD_REVISION_CAP, _METHOD_SWAP_CAP)
from autoresearch.orchestrator.session_state import (
    SessionState,
    load_session_state,
    save_session_state,
)
from autoresearch.orchestrator.tests.unit.test_engine_min import FakeBusClient, fake_bus  # noqa: F401

SM_ROOT = Path(__file__).resolve().parents[2] / "state-machine"


@pytest.mark.parametrize('action', ['retry_failed_state', 'report_now'])
def test_memory_retry_preserves_experiment_evidence_without_new_failure(fake_bus, action):
    engine = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    state = SessionState(
        session_id='retry-memory', mode='benchmark', previous_state='MossIntervening',
        method_ref='method-current', run_record_refs=['run-measured'],
        evaluation_report_refs=['evaluation-measured'],
        last_evaluated_run_record_ref='run-measured',
        last_agent_failure={'failed_state': 'ConsolidatingMemory'},
        moss_interventions=[{'reason': 'agent_failure', 'action': action,
                             'moss_intervention_ref': 'moss-memory-retry'}],
    )

    payload = engine._memory_consolidation_payload(state)

    assert payload['run_record_ref'] == 'run-measured'
    assert payload['evaluation_report_ref'] == 'evaluation-measured'
    if action == 'retry_failed_state':
        assert payload['failure_context'] is None
    else:
        assert payload['failure_context']['reason'] == 'agent_failure'
        assert payload['failure_context']['action'] == 'report_now'


@pytest.mark.parametrize('component_round,method_round,requested', [
    (0, 0, False), (_COMPONENT_REVISION_CAP, 0, True),
    (0, _METHOD_REVISION_CAP - 1, True), (0, _METHOD_REVISION_CAP, True)])
def test_revision_budget_requests_qualified_closeout(fake_bus, component_round, method_round, requested):
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    state = SessionState(session_id='closeout', user_id='u', mode='benchmark',
        method_ref='method-current', baseline_result_ref='baseline-frozen',
        component_revision_round=component_round, method_revision_round=method_round)
    payload = eng._memory_consolidation_payload(state)
    assert bool(payload.get('qualified_component_closeout_requested')) is requested
    if requested:
        assert payload['method_ref'] == state.method_ref



@pytest.mark.parametrize('enabled', [False, True])
@pytest.mark.parametrize('rounds', [0, 1, 2, 3])
def test_whole_feedback_budget_requests_existing_component_closeout(fake_bus, monkeypatch, enabled, rounds):
    monkeypatch.setenv('AUTORESEARCH_METHOD_FEEDBACK', '1' if enabled else '0')
    monkeypatch.setenv('AUTORESEARCH_METHOD_FEEDBACK_ROUNDS', '2')
    engine = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    state = SessionState(session_id='whole-feedback-closeout', mode='benchmark',
        method_ref='method-current', baseline_result_ref='baseline-frozen',
        method_feedback_round=rounds)
    payload = engine._memory_consolidation_payload(state)
    assert bool(payload.get('qualified_component_closeout_requested')) is (enabled and rounds >= 2)
    assert state.method_feedback_round == rounds
    if enabled and rounds >= 2:
        assert payload['method_ref'] == state.method_ref


def test_last_verified_integration_retry_requests_component_closeout(fake_bus, monkeypatch):
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    sid = "closeout-after-integration-retries"
    state = SessionState(
        session_id=sid, user_id="u", mode="benchmark",
        method_ref="method-current", baseline_result_ref="baseline-frozen",
        current_experiment_id=sid,
    )
    threshold, _ = eng._counter_threshold("implementation_fix_round")
    state.implementation_fix_round[sid] = threshold - 1
    monkeypatch.setattr(eng, "_current_gen_result", lambda _: {
        "method_ref": state.method_ref,
        "verdict": "valid_not_best",
        "execution_succeeded": True,
        "component_execution": {
            "mode": "integration",
            "all_components_accepted": True,
        },
    })

    payload = eng._memory_consolidation_payload(state)

    assert payload["qualified_component_closeout_requested"] is True
    assert payload["method_ref"] == state.method_ref


@dataclass
class _Evt:
    event_type: str
    session_id: str
    emitter: str
    payload: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _seed_consolidating_memory_session(
    eng: OrchestratorEngine,
    sid: str,
    *,
    research_memory_ref: str | None = None,
    verification_report_refs: list[str] | None = None,
) -> None:
    """Persist a session in ConsolidatingMemory with its upstream refs set."""
    state = SessionState(
        session_id=sid,
        user_id="u1",
        current_state="ConsolidatingMemory",
        evaluation_report_refs=[f"evaluation_report://{sid}/final"],
        run_record_refs=[f"run_record://{sid}/final"],
        research_memory_ref=research_memory_ref,
        verification_report_refs=verification_report_refs or [],
        research_spec_ref=f"research_spec://{sid}/final",
        literature_brief_ref=f"literature_brief://{sid}/final",
        hypothesis_bank_ref=f"hypothesis_bank://{sid}/final",
        current_hypothesis_id="hyp1",
        verifier_spec_ref=f"verifier_spec://{sid}/final",
        benchmark_spec_ref=f"benchmark_spec://{sid}/final",
        experiment_plan_refs=[f"experiment_plan://{sid}/final"],
    )
    await save_session_state(eng.client.db, state, auto_commit=True)


# ---------------------------------------------------------------------------
# (a): entry action
# ---------------------------------------------------------------------------


async def test_consolidating_memory_entry_action_emits_consolidation_requested_no_refs(fake_bus):
    """research_memory_ref unset and verification_report_refs empty -> both None."""
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = "sess-a1"
    await _seed_consolidating_memory_session(eng, sid)
    state = await load_session_state(fake_bus.db, sid)

    await eng._execute_entry_action(state)

    emitted = [p for p in fake_bus.published if p.event_type == "memory.consolidation_requested"]
    assert len(emitted) == 1
    assert emitted[0].payload == {
        "evaluation_report_ref": f"evaluation_report://{sid}/final",
        "run_record_ref": f"run_record://{sid}/final",
        "research_memory_ref": None,
            "verification_report_ref": None,
            "baseline_result_ref": None,
            "failure_context": None,
    }


async def test_consolidating_memory_entry_action_emits_consolidation_requested_with_refs(fake_bus):
    """research_memory_ref set and verification_report_refs non-empty -> both populated."""
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = "sess-a2"
    await _seed_consolidating_memory_session(
        eng,
        sid,
        research_memory_ref=f"research_memory://{sid}/v1",
        verification_report_refs=[f"verification_report://{sid}/final"],
    )
    state = await load_session_state(fake_bus.db, sid)

    await eng._execute_entry_action(state)

    emitted = [p for p in fake_bus.published if p.event_type == "memory.consolidation_requested"]
    assert len(emitted) == 1
    assert emitted[0].payload == {
        "evaluation_report_ref": f"evaluation_report://{sid}/final",
        "run_record_ref": f"run_record://{sid}/final",
        "research_memory_ref": f"research_memory://{sid}/v1",
            "verification_report_ref": f"verification_report://{sid}/final",
            "baseline_result_ref": None,
            "failure_context": None,
    }


# ---------------------------------------------------------------------------
# (b): memory.consolidated "continue_explore" -> GeneratingHypotheses (LOOP CLOSES)
# ---------------------------------------------------------------------------


async def test_memory_consolidated_continue_explore_closes_exploration_loop(fake_bus):
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = "sess-b"
    await _seed_consolidating_memory_session(eng, sid)

    n_before = len(fake_bus.published)
    mem_ref = f"research_memory://{sid}/v2"
    await eng.tick(
        _Evt(
            "memory.consolidated",
            sid,
            "MemoryConsolidatorAgent",
            payload={"routing_verdict": "continue_explore", "research_memory_ref": mem_ref},
        )
    )

    st = await load_session_state(fake_bus.db, sid)
    assert st is not None
    # meeting 甲: continue_explore now re-plans another implementation round of the confirmed
    # method (PlanningExperiment), not a new hypothesis portfolio.
    assert st.current_state == "PlanningExperiment"
    assert st.research_memory_ref == mem_ref

    new_events = fake_bus.published[n_before:]
    # The exploration loop CLOSES via PlanningExperiment's entry action (experiment.planning_requested).
    assert any(p.event_type == "experiment.planning_requested" for p in new_events)


# ---------------------------------------------------------------------------
# (c): memory.consolidated "report" -> Reporting
# ---------------------------------------------------------------------------


async def test_memory_consolidated_report_advances_to_reporting_and_pauses(fake_bus):
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = "sess-c"
    await _seed_consolidating_memory_session(eng, sid)

    n_before = len(fake_bus.published)
    mem_ref = f"research_memory://{sid}/v2"
    await eng.tick(
        _Evt(
            "memory.consolidated",
            sid,
            "MemoryConsolidatorAgent",
            payload={"routing_verdict": "report", "research_memory_ref": mem_ref},
        )
    )

    st = await load_session_state(fake_bus.db, sid)
    assert st is not None
    assert st.current_state == "Reporting"
    assert st.research_memory_ref == mem_ref

    # Reporting's entry action (M14; EvidenceAuditAgent milestone Task 5
    # swapped this to the Mode C citation-audit gate) fires
    # evidence.citation_audit_requested on entry.
    new_events = fake_bus.published[n_before:]
    assert any(p.event_type == "evidence.citation_audit_requested" for p in new_events)


async def test_component_accepted_replans_without_consuming_outer_round(fake_bus):
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = "sess-component-accepted"
    await _seed_consolidating_memory_session(eng, sid)

    await eng.tick(_Evt(
        "memory.consolidated", sid, "MemoryConsolidatorAgent",
        payload={
            "routing_verdict": "need_experiment",
            "research_memory_ref": f"research_memory://{sid}/v2",
            "diagnosis": "component_accepted",
            "diagnosis_evidence": ["feature-split passed"],
        },
    ))

    state = await load_session_state(fake_bus.db, sid)
    assert state.current_state == "PlanningExperiment"
    assert state.exploration_loop_round == 0
    assert state.setting_rework_round == 0


async def test_component_progress_replans_same_active_node_without_outer_round(fake_bus):
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = "sess-component-progress"
    await _seed_consolidating_memory_session(eng, sid)

    await eng.tick(_Evt(
        "memory.consolidated", sid, "MemoryConsolidatorAgent",
        payload={
            "routing_verdict": "need_experiment",
            "research_memory_ref": f"research_memory://{sid}/v2",
            "diagnosis": "component_progress",
            "diagnosis_evidence": ["component passed; improve from its checkpoint"],
        },
    ))

    state = await load_session_state(fake_bus.db, sid)
    assert state.current_state == "PlanningExperiment"
    assert state.exploration_loop_round == 0


async def test_component_combination_failure_replans_without_method_swap(fake_bus):
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = "sess-component-combination"
    await _seed_consolidating_memory_session(eng, sid)
    state = await load_session_state(fake_bus.db, sid)
    original_method_ref = state.method_ref

    await eng.tick(_Evt(
        "memory.consolidated", sid, "MemoryConsolidatorAgent",
        payload={
            "routing_verdict": "need_experiment",
            "research_memory_ref": f"research_memory://{sid}/v2",
            "diagnosis": "component_combination_retry",
            "diagnosis_evidence": ["individual components pass; integration does not"],
        },
    ))

    state = await load_session_state(fake_bus.db, sid)
    assert state.current_state == "PlanningExperiment"
    assert state.method_ref == original_method_ref
    assert state.method_not_working is False


async def test_component_combination_budget_exhaustion_reports_without_method_swap(fake_bus):
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = "sess-component-combination-exhausted"
    await _seed_consolidating_memory_session(eng, sid)
    state = await load_session_state(fake_bus.db, sid)
    original_method_ref = state.method_ref
    threshold, _ = eng._counter_threshold("implementation_fix_round")
    state.implementation_fix_round[sid] = threshold - 1
    await save_session_state(fake_bus.db, state, auto_commit=True)

    await eng.tick(_Evt(
        "memory.consolidated", sid, "MemoryConsolidatorAgent",
        payload={
            "routing_verdict": "need_experiment",
            "research_memory_ref": f"research_memory://{sid}/v2",
            "diagnosis": "component_combination_retry",
            "diagnosis_evidence": ["integration still does not meet expectation"],
        },
    ))

    state = await load_session_state(fake_bus.db, sid)
    assert state.current_state == "Reporting"
    assert state.method_ref == original_method_ref
    assert state.method_not_working is False


async def test_component_retry_returns_to_implementation(fake_bus):
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = "sess-component-retry"
    await _seed_consolidating_memory_session(eng, sid)

    await eng.tick(_Evt(
        "memory.consolidated", sid, "MemoryConsolidatorAgent",
        payload={
            "routing_verdict": "need_experiment",
            "research_memory_ref": f"research_memory://{sid}/v2",
            "diagnosis": "component_contract_retry",
            "diagnosis_evidence": ["feature-split crashed"],
        },
    ))

    state = await load_session_state(fake_bus.db, sid)
    assert state.current_state == "ImplementingExperiment"
    assert state.implementation_fix_round


async def test_component_retry_without_plan_restores_planning_prerequisite(fake_bus):
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = "sess-component-retry-no-plan"
    await _seed_consolidating_memory_session(eng, sid)
    state = await load_session_state(fake_bus.db, sid)
    state.experiment_plan_refs = []
    await save_session_state(fake_bus.db, state, auto_commit=True)

    await eng.tick(_Evt(
        "memory.consolidated", sid, "MemoryConsolidatorAgent",
        payload={
            "routing_verdict": "need_experiment",
            "research_memory_ref": f"research_memory://{sid}/v2",
            "diagnosis": "component_contract_retry",
            "diagnosis_evidence": ["current component needs repair"],
        },
    ))

    state = await load_session_state(fake_bus.db, sid)
    assert state.current_state == "PlanningExperiment"


async def test_non_failure_memory_classification_resets_implementation_counter(fake_bus):
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = "sess-memory-resets-implementation-counter"
    await _seed_consolidating_memory_session(eng, sid)
    state = await load_session_state(fake_bus.db, sid)
    state.current_experiment_id = "exp1"
    state.implementation_fix_round = {"exp1": 2}
    await save_session_state(fake_bus.db, state, auto_commit=True)

    await eng.tick(_Evt(
        "memory.consolidated", sid, "MemoryConsolidatorAgent",
        payload={
            "routing_verdict": "report",
            "research_memory_ref": f"research_memory://{sid}/v2",
            "diagnosis": "honest_negative",
        },
    ))

    state = await load_session_state(fake_bus.db, sid)
    assert state.current_state == "Reporting"
    assert state.implementation_fix_round == {}


# ---------------------------------------------------------------------------
# force-keep-searching (opt-in): a WIN ("report") keeps searching instead of reporting
# ---------------------------------------------------------------------------


async def test_force_keep_searching_turns_win_into_another_round(fake_bus, monkeypatch):
    """AUTORESEARCH_FORCE_KEEP_SEARCHING=1: 'report' (win) below the loop cap -> keep searching
    (PlanningExperiment) so the operator-typed search gets exercised, instead of Reporting."""
    monkeypatch.setenv("AUTORESEARCH_FORCE_KEEP_SEARCHING", "1")
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = "sess-cforce"
    await _seed_consolidating_memory_session(eng, sid)
    mem_ref = f"research_memory://{sid}/v2"
    await eng.tick(
        _Evt("memory.consolidated", sid, "MemoryConsolidatorAgent",
             payload={"routing_verdict": "report", "research_memory_ref": mem_ref}))
    st = await load_session_state(fake_bus.db, sid)
    assert st.current_state == "PlanningExperiment"     # forced another round, not Reporting
    assert st.exploration_loop_round == 1               # the loop-back was counted


async def test_force_keep_searching_off_by_default_win_reports(fake_bus, monkeypatch):
    """Default (flag unset): a win still reports as before -- no behavior change."""
    monkeypatch.delenv("AUTORESEARCH_FORCE_KEEP_SEARCHING", raising=False)
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = "sess-cnoforce"
    await _seed_consolidating_memory_session(eng, sid)
    await eng.tick(
        _Evt("memory.consolidated", sid, "MemoryConsolidatorAgent",
             payload={"routing_verdict": "report",
                      "research_memory_ref": f"research_memory://{sid}/v2"}))
    st = await load_session_state(fake_bus.db, sid)
    assert st.current_state == "Reporting"              # unchanged default


# ---------------------------------------------------------------------------
# (d): memory.consolidated "need_experiment" -> PlanningExperiment
# ---------------------------------------------------------------------------


async def test_memory_consolidated_need_experiment_advances_to_planning_experiment(fake_bus):
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = "sess-d"
    await _seed_consolidating_memory_session(eng, sid)

    n_before = len(fake_bus.published)
    mem_ref = f"research_memory://{sid}/v2"
    await eng.tick(
        _Evt(
            "memory.consolidated",
            sid,
            "MemoryConsolidatorAgent",
            payload={"routing_verdict": "need_experiment", "research_memory_ref": mem_ref},
        )
    )

    st = await load_session_state(fake_bus.db, sid)
    assert st is not None
    assert st.current_state == "PlanningExperiment"
    assert st.research_memory_ref == mem_ref

    new_events = fake_bus.published[n_before:]
    assert any(p.event_type == "experiment.planning_requested" for p in new_events)


# ---------------------------------------------------------------------------
# memory.consolidated unknown routing_verdict: defensive no-op
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# bounded idea<->experiment loop: round counter + MAX cap forces Reporting
# ---------------------------------------------------------------------------


async def test_exploration_loop_increments_round_below_cap(fake_bus, monkeypatch):
    monkeypatch.setenv("AUTORESEARCH_LOOP_MAX_ROUNDS", "3")
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = "sess-loop1"
    await _seed_consolidating_memory_session(eng, sid)  # exploration_loop_round starts at 0

    await eng.tick(_Evt("memory.consolidated", sid, "MemoryConsolidatorAgent",
                        payload={"routing_verdict": "continue_explore",
                                 "research_memory_ref": f"research_memory://{sid}/v2"}))

    st = await load_session_state(fake_bus.db, sid)
    assert st.current_state == "PlanningExperiment"      # 甲: below cap -> re-plan (no hypotheses)
    assert st.exploration_loop_round == 1               # counted this round


async def test_exploration_loop_caps_at_max_rounds_and_forces_reporting(fake_bus, monkeypatch):
    monkeypatch.setenv("AUTORESEARCH_LOOP_MAX_ROUNDS", "3")
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = "sess-loop-cap"
    await _seed_consolidating_memory_session(eng, sid)
    # already spent 2 rounds; the next continue_explore hits the cap (3) -> forced Reporting.
    st0 = await load_session_state(fake_bus.db, sid)
    st0.exploration_loop_round = 2
    await save_session_state(fake_bus.db, st0, auto_commit=True)

    await eng.tick(_Evt("memory.consolidated", sid, "MemoryConsolidatorAgent",
                        payload={"routing_verdict": "continue_explore",
                                 "research_memory_ref": f"research_memory://{sid}/v3"}))

    st = await load_session_state(fake_bus.db, sid)
    assert st.current_state == "Reporting"              # cap reached -> forced report, no loop
    assert st.exploration_loop_round == 3


async def test_inconclusive_cap_escalates_to_method_reconsideration(fake_bus, monkeypatch):
    """Catch repeated fair non-wins reporting before the outer method loop is reached."""
    monkeypatch.setenv("AUTORESEARCH_LOOP_MAX_ROUNDS", "3")
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = "sess-inconclusive-cap"
    await _seed_consolidating_memory_session(eng, sid)
    state = await load_session_state(fake_bus.db, sid)
    state.exploration_loop_round = 2
    await save_session_state(fake_bus.db, state, auto_commit=True)

    await eng.tick(_Evt(
        "memory.consolidated", sid, "MemoryConsolidatorAgent",
        payload={
            "routing_verdict": "continue_explore",
            "research_memory_ref": f"research_memory://{sid}/v3",
            "diagnosis": "inconclusive",
            "diagnosis_evidence": ["three fair runs did not improve"],
        },
    ))

    state = await load_session_state(fake_bus.db, sid)
    assert state.current_state == "MossIntervening"
    assert state.current_moss_reason == "method_not_working"
    assert state.method_not_working is True
    assert state.setting_rework_round == 0


@pytest.mark.parametrize("diagnosis", [
    "execution_failure", "candidate_contract_violation", "attempt_accounting_violation",
])
async def test_execution_diagnoses_return_to_implementation_without_setting_rework(
        fake_bus, diagnosis):
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = f"sess-{diagnosis}"
    await _seed_consolidating_memory_session(eng, sid)
    seeded = await load_session_state(fake_bus.db, sid)
    seeded.current_experiment_id = "exp-debug"
    await save_session_state(fake_bus.db, seeded, auto_commit=True)

    await eng.tick(_Evt(
        "memory.consolidated", sid, "MemoryConsolidatorAgent",
        payload={
            "routing_verdict": "continue_explore", "research_memory_ref": f"rm://{sid}/v",
            "diagnosis": diagnosis, "diagnosis_evidence": ["candidate did not run fairly"],
        },
    ))

    state = await load_session_state(fake_bus.db, sid)
    assert state.current_state == "ImplementingExperiment"
    assert state.setting_rework_round == 0
    assert state.implementation_fix_round == {"exp-debug": 1}


@pytest.mark.parametrize("mode", ["component", "integration"])
@pytest.mark.parametrize("diagnosis", ["component_contract_retry", "component_execution_blocked"])
async def test_confirmed_illegal_component_attempt_enters_local_plan_audit(
        fake_bus, monkeypatch, mode, diagnosis):
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = f"illegal-{mode}"
    await _seed_consolidating_memory_session(eng, sid)
    state = await load_session_state(fake_bus.db, sid)
    state.mode = "benchmark"
    state.method_ref = f"method://{sid}/confirmed"
    state.baseline_result_ref = f"baseline_result://{sid}/frozen"
    await save_session_state(fake_bus.db, state, auto_commit=True)
    monkeypatch.setattr(eng, "_current_gen_result", lambda _sid: {
        "diagnosis": "candidate_contract_violation",
        "method_ref": f"method://{sid}/confirmed",
        "run_record_snapshot_ref": f"run_record://{sid}/final",
        "component_execution": {
            "mode": mode, "component_id": "projection" if mode == "component" else None,
            "status": "exhausted" if diagnosis == "component_execution_blocked" else "contract_failed",
            "diff_ref": "checkpoint://candidate#git-diff",
        },
        "method_implementation_review": {
            "verdict": "faithful",
            "task_contract_review": {
                "verdict": "violation", "violations": ["edits outside allowed scope"],
            },
        },
        "diff": "+unauthorized_file()",
    })

    await eng.tick(_Evt("memory.consolidated", sid, "MemoryConsolidatorAgent", payload={
        "routing_verdict": "need_experiment",
        "research_memory_ref": f"research_memory://{sid}/current",
        "diagnosis": diagnosis,
        "diagnosis_evidence": ["candidate violated the repository contract"],
    }))

    state = await load_session_state(fake_bus.db, sid)
    assert state.current_state == "DeliberatingMethod"
    assert state.component_plan_audit_pending is True
    assert state.ablation_failed_component == (
        "projection" if mode == "component" else "__integration__")
    assert state.method_ref == f"method://{sid}/confirmed"
    assert state.baseline_result_ref == f"baseline_result://{sid}/frozen"
    event = next(e for e in fake_bus.published if e.event_type == "method.deliberation_requested")
    assert event.payload["component_plan_allow_unchanged"] is True
    assert "edits outside allowed scope" in event.payload["component_plan_revision_feedback"]
    assert "+unauthorized_file()" in event.payload["component_plan_revision_feedback"]
    assert "Do not use this candidate's score" in event.payload["component_plan_revision_feedback"]


async def test_unproven_contract_compliance_stays_on_implementation_retry(fake_bus, monkeypatch):
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = "unproven-compliance"
    await _seed_consolidating_memory_session(eng, sid)
    state = await load_session_state(fake_bus.db, sid)
    state.mode = "benchmark"
    state.method_ref = f"method://{sid}/confirmed"
    state.baseline_result_ref = f"baseline_result://{sid}/frozen"
    await save_session_state(fake_bus.db, state, auto_commit=True)
    monkeypatch.setattr(eng, "_current_gen_result", lambda _sid: {
        "diagnosis": "candidate_contract_violation",
        "method_ref": f"method://{sid}/confirmed",
        "run_record_snapshot_ref": f"run_record://{sid}/final",
        "component_execution": {"mode": "component", "component_id": "projection"},
        "method_implementation_review": {
            "task_contract_review": {"verdict": "insufficient_evidence"},
        },
    })
    await eng.tick(_Evt("memory.consolidated", sid, "MemoryConsolidatorAgent", payload={
        "routing_verdict": "need_experiment",
        "research_memory_ref": f"research_memory://{sid}/current",
        "diagnosis": "component_contract_retry",
    }))
    state = await load_session_state(fake_bus.db, sid)
    assert state.current_state == "ImplementingExperiment"
    assert state.component_plan_audit_pending is False


async def test_novelty_exhaustion_reports_without_entering_experiment_states(
        fake_bus, monkeypatch):
    monkeypatch.setenv("AUTORESEARCH_FORCE_KEEP_SEARCHING", "1")
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = "sess-novelty-exhausted"
    state = SessionState(
        session_id=sid,
        user_id="u1",
        current_state="ConsolidatingMemory",
        research_spec_ref=f"research_spec://{sid}/final",
    )
    await save_session_state(fake_bus.db, state, auto_commit=True)

    await eng.tick(_Evt(
        "memory.consolidated", sid, "MemoryConsolidatorAgent",
        payload={
            "routing_verdict": "report",
            "research_memory_ref": f"research_memory://{sid}/v1",
            "diagnosis": "literature_novelty_exhausted",
            "diagnosis_evidence": ["two targeted searches remained inconclusive"],
        },
    ))

    state = await load_session_state(fake_bus.db, sid)
    assert state.current_state == "Reporting"
    assert not state.experiment_plan_refs
    assert any(item.event_type == "evidence.citation_audit_requested"
               for item in fake_bus.published)

    await eng.tick(_Evt(
        "evidence.citation_audited", sid, "EvidenceAuditAgent",
        payload={"citation_audit_report_ref": f"citation_audit://{sid}/v1"},
    ))
    await eng.tick(_Evt(
        "report.ready", sid, "ReportingAgent",
        payload={"report_ref": f"report://{sid}/final"},
    ))
    state = await load_session_state(fake_bus.db, sid)
    assert state.current_state == "Done"
    assert state.final_report_ref == f"report://{sid}/final"


async def test_baseline_reproduction_failure_reports_without_an_experiment_plan(
        fake_bus, monkeypatch):
    monkeypatch.setenv("AUTORESEARCH_FORCE_KEEP_SEARCHING", "1")
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = "sess-baseline-exhausted"
    state = SessionState(
        session_id=sid,
        user_id="u1",
        current_state="ConsolidatingMemory",
        research_spec_ref=f"research_spec://{sid}/final",
    )
    await save_session_state(fake_bus.db, state, auto_commit=True)

    await eng.tick(_Evt(
        "memory.consolidated", sid, "MemoryConsolidatorAgent",
        payload={
            "routing_verdict": "report",
            "research_memory_ref": f"research_memory://{sid}/v1",
            "diagnosis": "baseline_reproduction_failed",
            "diagnosis_evidence": ["six clean attempts hit the same dependency failure"],
        },
    ))

    state = await load_session_state(fake_bus.db, sid)
    assert state.current_state == "Reporting"
    assert not state.experiment_plan_refs
    assert any(item.event_type == "evidence.citation_audit_requested"
               for item in fake_bus.published)


@pytest.mark.parametrize("diagnosis", [
    "execution_failure", "candidate_contract_violation", "attempt_accounting_violation",
])
async def test_repeated_execution_diagnosis_hits_persisted_debug_cap_and_moss(
        fake_bus, diagnosis):
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = f"sess-{diagnosis}-cap"
    await _seed_consolidating_memory_session(eng, sid)
    state = await load_session_state(fake_bus.db, sid)
    state.current_experiment_id = "exp-debug"
    state.implementation_fix_round = {"exp-debug": 2}
    await save_session_state(fake_bus.db, state, auto_commit=True)

    await eng.tick(_Evt(
        "memory.consolidated", sid, "MemoryConsolidatorAgent",
        payload={
            "routing_verdict": "continue_explore",
            "research_memory_ref": f"rm://{sid}/v",
            "diagnosis": diagnosis,
            "diagnosis_evidence": ["third fresh sandbox failed the same trusted gate"],
        },
    ))

    state = await load_session_state(fake_bus.db, sid)
    assert state.implementation_fix_round == {"exp-debug": 3}
    assert state.current_state == "MossIntervening"
    assert state.current_moss_reason == "implementation_thrashing"


@pytest.mark.parametrize("case", ["unstarted_timeout", "classified_timeout", "no_plan", "started_timeout",
                                  "execution_succeeded", "implementation_error", "old_run",
                                  "infrastructure_exhausted"])
@pytest.mark.parametrize("provider_code", [
    "llm_provider_timeout", "llm_provider_content_filter",
    "clone failed: git clone produced an empty/incomplete repo; Disk quota exceeded",
    "clone failed: fatal: unable to access repository; connection timed out",
])
async def test_preexecution_provider_failure_keeps_separate_infrastructure_budget(
    fake_bus, monkeypatch, case, provider_code,
):
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = "sess-transport-budget-" + case
    await _seed_consolidating_memory_session(eng, sid)
    state = await load_session_state(fake_bus.db, sid)
    state.mode = "benchmark"
    state.method_ref = "spec://method/" + sid + "/snapshot-current"
    state.current_experiment_id = "exp-debug"
    state.implementation_fix_round = {"exp-debug": 2}
    if case == "no_plan":
        state.experiment_plan_refs = []
    await save_session_state(fake_bus.db, state, auto_commit=True)
    result = {
        "method_ref": state.method_ref, "run_record_snapshot_ref": state.run_record_refs[-1],
        "diagnosis": "execution_failure", "method_metric": None,
        "error": (provider_code if provider_code.startswith("clone failed:")
                  else "Conversation run failed; " + provider_code),
        "component_execution_started": False,
        "component_execution": {"mode": "component", "status": "execution_failed",
                                "execution_succeeded": False,
                                "attempts": 1, "infrastructure_attempts": 2},
    }
    if case == "started_timeout":
        result["component_execution_started"] = True
        result["component_attempt_id"] = "attempt-measured"
    elif case == "execution_succeeded":
        result["component_execution"]["execution_succeeded"] = True
    elif case == "implementation_error":
        result["error"] = "compiler rejected the candidate source"
    elif case == "old_run":
        result["run_record_snapshot_ref"] = "run_record://old/final"
    elif case == "infrastructure_exhausted":
        result["component_execution"].update(status="infrastructure_exhausted",
                                            infrastructure_attempts=8)
    monkeypatch.setattr(eng, "_current_gen_result", lambda _: result)

    diagnosis = ("component_execution_blocked" if case == "infrastructure_exhausted"
                 else "execution_failure")
    if case == "classified_timeout":
        from autoresearch.agents.memory_agent.handler import _classify_component_evidence
        diagnosis, _ = _classify_component_evidence(result)
        assert diagnosis == "execution_failure"
    await eng.tick(_Evt("memory.consolidated", sid, "MemoryConsolidatorAgent", payload={
        "routing_verdict": "continue_explore", "research_memory_ref": f"rm://{sid}/v",
        "diagnosis": diagnosis, "diagnosis_evidence": [result["error"]],
    }))
    current = await load_session_state(fake_bus.db, sid)
    if case in {"unstarted_timeout", "classified_timeout", "no_plan"}:
        assert current.current_state == ("PlanningExperiment" if case == "no_plan"
                                         else "ImplementingExperiment")
        assert current.implementation_fix_round == {"exp-debug": 2}
        assert current.method_ref == state.method_ref
        assert current.run_record_refs == state.run_record_refs
        assert not current.current_moss_reason
    elif case == "infrastructure_exhausted":
        assert current.current_state == "Reporting"
        assert not current.implementation_fix_round
        assert current.run_record_refs == state.run_record_refs
    else:
        assert current.current_state == "MossIntervening"
        assert current.implementation_fix_round == {"exp-debug": 3}
        assert current.current_moss_reason == "implementation_thrashing"


async def test_component_failure_routes_to_targeted_moss(fake_bus):
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = "sess-component-failure"
    await _seed_consolidating_memory_session(eng, sid)

    await eng.tick(_Evt(
        "memory.consolidated", sid, "MemoryConsolidatorAgent",
        payload={
            "routing_verdict": "continue_explore", "research_memory_ref": f"rm://{sid}/v",
            "diagnosis": "component_expectation_failed",
            "diagnosis_evidence": ["router: separation score below contract"],
        },
    ))

    state = await load_session_state(fake_bus.db, sid)
    assert state.current_state == "MossIntervening"
    assert state.research_memory_ref == f"rm://{sid}/v"
    assert state.current_moss_reason == "component_ablation_failed"
    assert state.ablation_failed_component == "router: separation score below contract"


@pytest.mark.parametrize(
    ("swaps", "best_result_ref", "expected"),
    [(0, None, "Reporting"),
     (_METHOD_SWAP_CAP, None, "Reporting"),
     (0, "result://accepted", "Reporting")],
)
async def test_exhausted_component_revisions_preserve_method_swap_for_unfinished_benchmark(
        fake_bus, swaps, best_result_ref, expected):
    from types import SimpleNamespace

    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = "sess-component-cap"
    await _seed_consolidating_memory_session(eng, sid)
    state = await load_session_state(fake_bus.db, sid)
    state.mode = "benchmark"
    state.method_ref = f"method://{sid}/current"
    state.baseline_result_ref = f"baseline_result://{sid}/frozen"
    state.component_revision_round = _COMPONENT_REVISION_CAP
    state.method_swap_round = swaps
    await save_session_state(fake_bus.db, state, auto_commit=True)
    eng.spec_store = SimpleNamespace(get=lambda _ref: {"best_result_ref": best_result_ref})

    await eng.tick(_Evt(
        "memory.consolidated", sid, "MemoryConsolidatorAgent",
        payload={
            "routing_verdict": "report",
            "research_memory_ref": f"research_memory://{sid}/failed",
            "diagnosis": "component_expectation_failed",
            "diagnosis_evidence": ["scorer accuracy below the task gate"],
        },
    ))

    state = await load_session_state(fake_bus.db, sid)
    assert state.current_state == expected
    assert state.research_memory_ref == f"research_memory://{sid}/failed"
    assert state.ablation_failed_component == "scorer accuracy below the task gate"
    assert state.method_swap_round == swaps + int(expected == "MossIntervening")
    if expected == "MossIntervening":
        assert state.method_not_working is True
        assert state.current_moss_reason == "method_not_working"
        request = next(e for e in fake_bus.published
                       if e.event_type == "moss.intervention_requested")
        assert request.payload["research_memory_ref"] == state.research_memory_ref
        assert request.payload["spec_interpretation_ref"] == state.spec_interpretation_ref


async def test_method_falsified_routes_directly_to_moss(fake_bus):
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = "sess-method-falsified"
    await _seed_consolidating_memory_session(eng, sid)

    await eng.tick(_Evt(
        "memory.consolidated", sid, "MemoryConsolidatorAgent",
        payload={
            "routing_verdict": "continue_explore", "research_memory_ref": f"rm://{sid}/v",
            "diagnosis": "method_hypothesis_falsified",
            "diagnosis_evidence": ["all gain components passed but metric did not improve"],
        },
    ))

    state = await load_session_state(fake_bus.db, sid)
    assert state.current_state == "MossIntervening"
    assert state.current_moss_reason == "method_not_working"


async def test_true_baseline_drift_alone_reproduces_baseline(fake_bus):
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = "sess-baseline-drift"
    await _seed_consolidating_memory_session(eng, sid)
    state = await load_session_state(fake_bus.db, sid)
    state.baseline_result_ref = f"baseline_result://{sid}/v1"
    await save_session_state(fake_bus.db, state, auto_commit=True)

    await eng.tick(_Evt(
        "memory.consolidated", sid, "MemoryConsolidatorAgent",
        payload={
            "routing_verdict": "continue_explore", "research_memory_ref": f"rm://{sid}/v",
            "diagnosis": "baseline_drift", "diagnosis_evidence": ["repository commit changed"],
        },
    ))

    state = await load_session_state(fake_bus.db, sid)
    assert state.current_state == "ReproducingBaseline"
    assert state.baseline_result_ref is None


async def test_memory_consolidated_unknown_verdict_is_defensive_noop(fake_bus):
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = "sess-unk"
    await _seed_consolidating_memory_session(eng, sid)

    n_before = len(fake_bus.published)
    await eng.tick(
        _Evt(
            "memory.consolidated",
            sid,
            "MemoryConsolidatorAgent",
            payload={"routing_verdict": "something_weird", "research_memory_ref": "x"},
        )
    )

    st = await load_session_state(fake_bus.db, sid)
    assert st is not None
    assert st.current_state == "ConsolidatingMemory"
    assert fake_bus.published[n_before:] == []


# ---------------------------------------------------------------------------
# phase 3.2: two-layer failure loop (inner setting rework -> outer method revision)
# ---------------------------------------------------------------------------

async def _need_experiment(eng, fake_bus, sid):
    await eng.tick(_Evt("memory.consolidated", sid, "MemoryConsolidatorAgent",
                        payload={"routing_verdict": "need_experiment",
                                 "research_memory_ref": f"rm://{sid}/v"}))
    return await load_session_state(fake_bus.db, sid)


async def test_need_experiment_inner_setting_loop_then_outer_method_revision(fake_bus, monkeypatch):
    # inner (setting) loop re-plans up to cap 3; once exhausted, the OUTER loop revises the METHOD.
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = "sess-2layer"
    await _seed_consolidating_memory_session(eng, sid)
    # pre-set the inner counter to its cap so the NEXT need_experiment escalates to the method layer
    st = await load_session_state(fake_bus.db, sid)
    st.setting_rework_round = 3
    await save_session_state(fake_bus.db, st, auto_commit=True)
    monkeypatch.setattr(eng, "_current_gen_result", lambda _sid: {
        "verdict": "valid_not_best", "method_metric": 1.0,
        "measurement_evidence": {"valid": True},
    })
    st = await _need_experiment(eng, fake_bus, sid)
    assert st.current_state == "RevisingMethod"      # inner exhausted -> revise the method
    assert st.setting_rework_round == 0              # inner reset for the new method
    assert st.method_revision_round == 1
    assert st.method_feedback_round == 1


async def test_need_experiment_intermediate_revision_does_not_consume_complete_budget(
        fake_bus, monkeypatch):
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = "sess-intermediate-2layer"
    await _seed_consolidating_memory_session(eng, sid)
    st = await load_session_state(fake_bus.db, sid)
    st.setting_rework_round = 3
    st.method_revision_round = _METHOD_REVISION_CAP
    await save_session_state(fake_bus.db, st, auto_commit=True)
    monkeypatch.setattr(eng, "_current_gen_result", lambda _sid: {
        "verdict": "invalid", "method_metric": None,
    })

    st = await _need_experiment(eng, fake_bus, sid)

    assert st.current_state == "RevisingMethod"
    assert st.setting_rework_round == 0
    assert st.method_revision_round == _METHOD_REVISION_CAP
    assert st.method_feedback_round == 0


async def test_need_experiment_both_layers_exhausted_swaps_method(fake_bus, monkeypatch):
    # meeting ②: settings tuned (inner cap) AND method revised (outer cap), still no win -> the method
    # is FALSIFIED -> MOSS intervention to SWAP for a fundamentally different method (bounded, not
    # endless). Not a straight Reporting anymore.
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = "sess-exhausted"
    await _seed_consolidating_memory_session(eng, sid)
    st = await load_session_state(fake_bus.db, sid)
    st.setting_rework_round = 3
    st.method_revision_round = _METHOD_REVISION_CAP - 1  # final slot is closeout-only
    await save_session_state(fake_bus.db, st, auto_commit=True)
    monkeypatch.setattr(eng, "_current_gen_result", lambda _sid: {
        "verdict": "valid_not_best", "method_metric": 1.0,
        "measurement_evidence": {"valid": True},
    })
    st = await _need_experiment(eng, fake_bus, sid)
    assert st.current_state == "MossIntervening"
    assert st.current_moss_reason == "method_not_working"
    assert st.method_swap_round == 1
    assert st.method_revision_round == 0        # reset so the NEW method gets its own tuning budget
    assert st.method_not_working is True


async def test_need_experiment_not_truncated_by_idea_loop_cap(fake_bus, monkeypatch):
    # meeting ②: the FAILURE loop (need_experiment) is bounded by its OWN setting/method/swap caps,
    # NOT the idea-loop cap -- so a not-working method can still escalate (and eventually SWAP) even
    # when the idea budget is spent. (Before ②, a spent idea cap force-reported here, so the
    # method-revision/swap layers never fired under a low LOOP_MAX_ROUNDS.)
    monkeypatch.setenv("AUTORESEARCH_LOOP_MAX_ROUNDS", "2")
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = "sess-idea-spent"
    await _seed_consolidating_memory_session(eng, sid)
    st = await load_session_state(fake_bus.db, sid)
    st.exploration_loop_round = 5      # idea budget already blown past its cap
    await save_session_state(fake_bus.db, st, auto_commit=True)
    st = await _need_experiment(eng, fake_bus, sid)
    assert st.current_state == "PlanningExperiment"   # failure loop proceeds, NOT force-reported
    assert st.setting_rework_round == 1
    assert st.exploration_loop_round == 5             # failure loop does NOT count against the idea cap


async def test_need_experiment_method_swap_exhausted_reports(fake_bus, monkeypatch):
    # meeting ②: even swapping methods was exhausted (multiple distinct methods, all tuned+revised,
    # none won) -> honest Reporting. Guaranteed termination, no endless method churn.
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = "sess-swap-exhausted"
    await _seed_consolidating_memory_session(eng, sid)
    st = await load_session_state(fake_bus.db, sid)
    st.setting_rework_round = 3
    st.method_revision_round = _METHOD_REVISION_CAP
    st.method_swap_round = _METHOD_SWAP_CAP       # swap cap reached
    await save_session_state(fake_bus.db, st, auto_commit=True)
    monkeypatch.setattr(eng, "_current_gen_result", lambda _sid: {
        "verdict": "valid_not_best", "method_metric": 1.0,
        "measurement_evidence": {"valid": True},
    })
    st = await _need_experiment(eng, fake_bus, sid)
    assert st.current_state == "Reporting"


@pytest.mark.parametrize("swaps,expected", [(0, "GeneratingMethod"),
                                               (_METHOD_SWAP_CAP, "Reporting")])
async def test_discarded_benchmark_hypothesis_keeps_negative_memory_and_swaps_method(
        fake_bus, swaps, expected):
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = f"discarded-benchmark-{swaps}"
    await _seed_consolidating_memory_session(eng, sid)
    state = await load_session_state(fake_bus.db, sid)
    state.mode = "benchmark"
    state.method_ref = f"method://{sid}/current"
    state.baseline_result_ref = f"baseline_result://{sid}/frozen"
    state.method_swap_round = swaps
    state.moss_interventions = [{"reason": "implementation_thrashing",
                                 "action": "discard_hypothesis",
                                 "method_ref": state.method_ref}]
    await save_session_state(fake_bus.db, state, auto_commit=True)

    await eng.tick(_Evt("memory.consolidated", sid, "MemoryConsolidatorAgent", payload={
        "routing_verdict": "report", "diagnosis": "component_execution_blocked",
        "research_memory_ref": f"research_memory://{sid}/negative",
    }))
    state = await load_session_state(fake_bus.db, sid)
    assert state.current_state == expected
    assert state.research_memory_ref == f"research_memory://{sid}/negative"
    assert state.method_swap_round == min(swaps + 1, _METHOD_SWAP_CAP)
    assert state.method_not_working is (swaps < _METHOD_SWAP_CAP)


@pytest.mark.parametrize("revision_round,has_finding,expected", [
    (0, True, "RevisingMethod"),
    (_METHOD_REVISION_CAP, True, "RevisingMethod"),
    (0, False, "Reporting"),
])
async def test_blocked_benchmark_with_measured_component_gain_revises_before_reporting(
        fake_bus, revision_round, has_finding, expected):
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = f"blocked-with-evidence-{revision_round}-{has_finding}"
    await _seed_consolidating_memory_session(eng, sid)
    state = await load_session_state(fake_bus.db, sid)
    state.mode = "benchmark"
    state.method_ref = f"method://{sid}/current"
    state.baseline_result_ref = f"baseline_result://{sid}/frozen"
    state.method_revision_round = revision_round
    await save_session_state(fake_bus.db, state, auto_commit=True)

    class _Store:
        def get(self, ref):
            return {"best_result_ref": None,
                    "diagnosis": "component_execution_blocked",
                    "diagnosis_evidence": ["component execution exhausted"],
                    "verified_findings": ["component A measured a gain"] if has_finding else []}

    eng.spec_store = _Store()
    memory_ref = f"research_memory://{sid}/v1"
    await eng.tick(_Evt("memory.consolidated", sid, "MemoryConsolidatorAgent", payload={
        "routing_verdict": "report", "diagnosis": "component_execution_blocked",
        "research_memory_ref": memory_ref,
    }))
    state = await load_session_state(fake_bus.db, sid)
    assert state.current_state == expected
    assert state.method_revision_round == revision_round
    if expected == "RevisingMethod":
        assert "component A measured a gain" in eng._method_revision_feedback(state)


async def test_blocked_benchmark_with_only_real_failed_attempts_still_revises(fake_bus,
                                                                                monkeypatch):
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = "blocked-with-failed-patch"
    await _seed_consolidating_memory_session(eng, sid)
    state = await load_session_state(fake_bus.db, sid)
    state.mode = "benchmark"
    state.method_ref = f"method://{sid}/current"
    state.baseline_result_ref = f"baseline_result://{sid}/frozen"
    await save_session_state(fake_bus.db, state, auto_commit=True)

    class _Store:
        def get(self, ref):
            return {"best_result_ref": None, "diagnosis": "component_execution_blocked",
                    "diagnosis_evidence": ["check was not executable"],
                    "verified_findings": []}

    eng.spec_store = _Store()
    monkeypatch.setattr(eng, "_current_gen_result", lambda _sid: {
        "method_ref": state.method_ref,
        "component_execution": {
            "mode": "component", "component_id": "build", "status": "exhausted",
            "attempts": 6, "infrastructure_attempts": 0,
            "attempt_diff_ref": "checkpoint://failed#git-diff",
            "attempt_contract_result": {"reason": "check was not executable"},
        },
    })
    await eng.tick(_Evt("memory.consolidated", sid, "MemoryConsolidatorAgent", payload={
        "routing_verdict": "report", "diagnosis": "component_execution_blocked",
        "research_memory_ref": f"research_memory://{sid}/v1",
    }))
    state = await load_session_state(fake_bus.db, sid)
    assert state.current_state == "RevisingMethod"
    assert "checkpoint://failed#git-diff" in eng._method_revision_feedback(state)


def test_replacement_method_debate_keeps_prior_moss_diagnosis(fake_bus):
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    state = SessionState(session_id="s", user_id="u1", current_state="DeliberatingMethod",
                         method_ref="method://s/new", method_swap_round=1,
                         moss_interventions=[{
                             "method_ref": "method://s/failed",
                             "moss_intervention_ref": "moss_intervention://s/failed",
                         }])
    _, make_payload = eng._entry_actions["DeliberatingMethod"]
    assert make_payload(state, None)["moss_intervention_ref"] == "moss_intervention://s/failed"
    state.method_ref = "method://s/failed"
    assert make_payload(state, None)["moss_intervention_ref"] is None


def test_method_debate_carries_active_revision_review(fake_bus):
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    state = SessionState(session_id="s", user_id="u1", current_state="DeliberatingMethod",
                         method_ref="method://s/new", method_feedback_in_progress=True,
                         method_feedback_ref="method_improvement_review://s/bound")
    _, make_payload = eng._entry_actions["DeliberatingMethod"]
    assert make_payload(state, None)["method_feedback_ref"] == "method_improvement_review://s/bound"
    state.method_feedback_in_progress = False
    assert make_payload(state, None)["method_feedback_ref"] is None


async def test_revising_method_generated_reenters_method_deliberation(fake_bus):
    # A revised draft must rebuild component contracts before experiment planning.  Skipping the
    # method-debate layer leaves ``components``/ablations empty and breaks the next expectation loop.
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = "sess-revise"
    st = SessionState(session_id=sid, user_id="u1", current_state="RevisingMethod",
                      method_ref=f"method://{sid}/v1",
                      last_agent_failure={"underlying_reason_code":
                                          "confirmed_method_contract_conflict"},
                      component_revision_round=10,
                      literature_audit_round=2,
                      novelty_targeted_search_round=2,
                      novelty_targeted_search_pending=True,
                      novelty_targeted_search_id="old-method-search")
    await save_session_state(eng.client.db, st, auto_commit=True)
    await eng.tick(_Evt("method.generated", sid, "MethodAgent",
                        payload={"method_ref": f"method://{sid}/v2"}))
    st = await load_session_state(fake_bus.db, sid)
    assert st.current_state == "DeliberatingMethod" and st.method_ref == f"method://{sid}/v2"
    assert st.last_agent_failure is None
    assert st.current_experiment_id == f"method://{sid}/v2"
    assert st.component_revision_round == 0
    assert st.literature_audit_round == 0
    assert st.novelty_targeted_search_round == 0
    assert st.novelty_targeted_search_pending is False
    assert st.novelty_targeted_search_id is None
    requested = [
        event for event in fake_bus.published
        if event.event_type == "method.deliberation_requested"
    ]
    assert len(requested) == 1
    assert requested[0].payload["method_ref"] == f"method://{sid}/v2"


async def test_off_method_gain_reconciles_method_description_without_swapping(fake_bus):
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = "sess-method-reconcile"
    await _seed_consolidating_memory_session(eng, sid)
    state = await load_session_state(fake_bus.db, sid)
    state.method_ref = f"method://{sid}/v1"
    state.problem_model_ref = f"problem_model://{sid}/v1"
    await save_session_state(fake_bus.db, state, auto_commit=True)

    await eng.tick(_Evt(
        "memory.consolidated", sid, "MemoryConsolidatorAgent",
        payload={
            "routing_verdict": "continue_explore",
            "research_memory_ref": f"research_memory://{sid}/v1",
            "diagnosis": "method_description_reconciliation",
            "diagnosis_evidence": ["off-method clipping explains the gain"],
        },
    ))

    state = await load_session_state(fake_bus.db, sid)
    assert state.current_state == "RevisingMethod"
    assert state.method_revision_round == 0
    assert state.method_swap_round == 0
    emitted = [event for event in fake_bus.published
               if event.event_type == "method.revise_requested"]
    assert len(emitted) == 1
    assert emitted[0].payload["research_memory_ref"] == f"research_memory://{sid}/v1"


async def test_legal_integration_observation_enters_formal_revision(fake_bus, monkeypatch):
    from autoresearch.agents.memory_agent.handler import _classify_component_evidence

    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = "integration-reconcile"
    await _seed_consolidating_memory_session(eng, sid)
    state = await load_session_state(fake_bus.db, sid)
    state.mode = "benchmark"
    state.method_ref = f"method://{sid}/confirmed"
    state.baseline_result_ref = f"baseline_result://{sid}/frozen"
    state.problem_model_ref = f"problem_model://{sid}/v1"
    await save_session_state(fake_bus.db, state, auto_commit=True)
    gen = {
        "verdict": "valid_not_best", "accepted": False, "method_metric": 0.9,
        "method_implementation_review": {
            "verdict": "drifted", "task_contract_review": {"verdict": "compliant"}},
        "off_method_observation": {"valid": True},
        "component_execution": {"mode": "integration", "component_id": "__integration__",
                                "status": "valid_not_best", "all_components_accepted": True},
    }
    monkeypatch.setattr(eng, "_current_gen_result", lambda _: gen)
    diagnosis, evidence = _classify_component_evidence(gen)
    await eng.tick(_Evt("memory.consolidated", sid, "MemoryConsolidatorAgent", payload={
        "routing_verdict": "continue_explore",
        "research_memory_ref": f"research_memory://{sid}/current",
        "diagnosis": diagnosis, "diagnosis_evidence": evidence,
    }))
    state = await load_session_state(fake_bus.db, sid)
    assert state.current_state == "RevisingMethod"
    assert state.method_ref == f"method://{sid}/confirmed"
    assert state.baseline_result_ref == f"baseline_result://{sid}/frozen"
    assert state.method_revision_round == 0
    assert state.method_swap_round == 0
    assert state.component_revision_round == 0
    assert len([e for e in fake_bus.published if e.event_type == "method.revise_requested"]) == 1


@pytest.mark.parametrize(
    ("trigger", "initial_revision", "initial_feedback", "expected_revision", "expected_feedback"),
    [
        ("implementation_failure", _METHOD_REVISION_CAP, 0, _METHOD_REVISION_CAP, 0),
        ("qualified_component_closeout", _METHOD_REVISION_CAP, 0, _METHOD_REVISION_CAP, 0),
        ("component_method_reconciliation", _METHOD_REVISION_CAP, 0,
         _METHOD_REVISION_CAP, 0),
        ("complete_verified_no_gain", 0, 0, 1, 1),
    ],
)
async def test_only_complete_metric_method_revision_consumes_whole_method_budget(
        fake_bus, monkeypatch, trigger, initial_revision, initial_feedback,
        expected_revision, expected_feedback):
    from types import SimpleNamespace

    monkeypatch.setenv("AUTORESEARCH_METHOD_FEEDBACK", "1")
    monkeypatch.setenv("AUTORESEARCH_METHOD_FEEDBACK_ROUNDS", "3")
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = f"method-budget-{trigger}"
    await _seed_consolidating_memory_session(eng, sid)
    method_ref = f"spec://method/{sid}/current"
    baseline_ref = f"spec://baseline_result/{sid}/frozen"
    memory_ref = f"spec://research_memory/{sid}/current"
    review_ref = f"spec://method_improvement_review/{sid}/current"
    state = await load_session_state(fake_bus.db, sid)
    state.mode = "benchmark"
    state.method_ref = method_ref
    state.baseline_result_ref = baseline_ref
    state.method_revision_round = initial_revision
    state.method_feedback_round = initial_feedback
    await save_session_state(fake_bus.db, state, auto_commit=True)

    objects = {
        baseline_ref: {"metric_value": 10.0},
        memory_ref: {
            "best_result_ref": None,
            "best_metric": None,
            "method_feedback_refs": [review_ref],
        },
        review_ref: {
            "status": "completed",
            "trigger": trigger,
            "method_ref": method_ref,
            "baseline_ref": baseline_ref,
            "incumbent_ref": baseline_ref,
            "incumbent_metric": 10.0,
            "qualified_closeout_components": (
                [{"component_id": "component-a"}]
                if trigger == "qualified_component_closeout" else []
            ),
            "decision": {"route": "revise_method"},
        },
    }
    eng.spec_store = SimpleNamespace(get=objects.__getitem__)

    await eng.tick(_Evt("memory.consolidated", sid, "MemoryConsolidatorAgent", payload={
        "routing_verdict": "need_experiment",
        "research_memory_ref": memory_ref,
        "diagnosis": "method_improvement_proposed",
        "diagnosis_evidence": [review_ref],
    }))

    state = await load_session_state(fake_bus.db, sid)
    assert state.current_state == "RevisingMethod"
    assert state.method_revision_round == expected_revision
    assert state.method_feedback_round == expected_feedback
    assert state.method_feedback_ref == review_ref


async def test_reconciliation_feedback_uses_current_review_not_stale_moss(fake_bus, monkeypatch):
    from types import SimpleNamespace

    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    state = SessionState(
        session_id="reconcile", user_id="u1", current_state="RevisingMethod",
        research_memory_ref="memory://current",
        moss_interventions=[{"moss_intervention_ref": "moss://old"}],
    )
    objects = {
        "memory://current": {
            "diagnosis": "method_description_reconciliation",
            "diagnosis_evidence": ["legal measured observation retained"],
        },
        "moss://old": {"action": "rethink_method", "prompt_patch": "discard the old algorithm"},
    }
    eng.spec_store = SimpleNamespace(get=objects.__getitem__)
    monkeypatch.setattr(eng, "_current_gen_result", lambda _sid: {
        "method_implementation_review": {
            "verdict": "drifted",
            "target_component_id": "consumer",
            "issues": ["producer implementation duplicated"],
            "remediation": "consume the accepted producer output",
            "boundary_reviews": [{"component_id": "producer", "status": "dependency_modified"}],
        },
        "component_execution": {"diff_ref": "checkpoint://candidate#git-diff"},
        "run_record_snapshot_ref": "run_record://reconcile/current",
        "diff": "diff --git a/producer.py b/producer.py",
        "changes_summary": "candidate changed producer",
    })
    objects["run_record://reconcile/current"] = {
        "stdout_excerpt": "official metric improved",
        "stderr_excerpt": "training warning",
    }
    feedback = eng._method_revision_feedback(state)
    assert '"verdict": "drifted"' in feedback
    assert "consumer" in feedback
    assert "producer implementation duplicated" in feedback
    assert "consume the accepted producer output" in feedback
    assert "checkpoint://candidate#git-diff" in feedback
    assert "official metric improved" in feedback
    assert "diff --git a/producer.py" in feedback
    assert "discard the old algorithm" not in feedback
    assert "Frozen MOSS revision guidance" not in feedback


async def test_host_revalidation_decision_hides_conflicting_advisory_actions(fake_bus):
    from types import SimpleNamespace

    review_ref = "review://host-revalidation"
    review = {
        "status": "completed",
        "trigger": "component_method_reconciliation",
        "result_ref": "result://exact-candidate",
        "decision_source": "host_revalidation_closeout_policy",
        "decision": {
            "route": "revise_method",
            "retain": ["exact candidate checkpoint"],
            "modify": ["bind Method to exact candidate"],
            "add": [],
            "remove": [],
        },
        "advisory_review_failure": {
            "route": "revise_method",
            "remove": ["remove measured SIMD source"],
            "reuse": ["/checkpoint/slower-incumbent"],
        },
        "off_method_evidence": {"candidate": {
            "result_ref": "result://exact-candidate",
            "checkpoint": "/checkpoint/exact-candidate",
            "diff_ref": "/checkpoint/exact-candidate#git-diff",
            "diff": "diff --git a/solve.c b/solve.c\n+measured SIMD source",
        }},
    }
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    eng.spec_store = SimpleNamespace(get=lambda ref: review if ref == review_ref else {})
    state = SessionState(
        session_id="host-revalidation", current_state="RevisingMethod",
        method_feedback_in_progress=True, method_feedback_ref=review_ref,
    )

    feedback = eng._method_revision_feedback(state)

    assert "host_decision_precedence" in feedback
    assert "superseded_for_method_revision" in feedback
    assert "describe every implemented mechanism" in feedback
    assert "/checkpoint/exact-candidate#git-diff" in feedback
    assert "remove measured SIMD source" not in feedback
    assert "/checkpoint/slower-incumbent" not in feedback


async def test_legacy_qualified_closeout_feedback_uses_selected_not_latest_checkpoint(fake_bus):
    from types import SimpleNamespace

    review_ref = "review://qualified-closeout"
    review = {
        "status": "completed",
        "trigger": "qualified_component_closeout",
        "result_ref": "result://latest-unqualified",
        "decision_source": "host_closeout_policy",
        "decision": {
            "route": "revise_method",
            "retain": ["The latest checkpoint /checkpoint/latest-unqualified"],
            "reuse": ["/checkpoint/qualified-best"],
            "add": [],
        },
        "advisory_review_failure": {"route": "unavailable"},
        "qualified_closeout_components": [{
            "checkpoint": "/checkpoint/qualified-best",
            "diff_ref": "/checkpoint/qualified-best#git-diff",
            "run_ref": "run://qualified-best",
        }],
    }
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    eng.spec_store = SimpleNamespace(get=lambda ref: review if ref == review_ref else {})
    state = SessionState(
        session_id="qualified-closeout", current_state="RevisingMethod",
        method_feedback_in_progress=True, method_feedback_ref=review_ref,
    )

    feedback = eng._method_revision_feedback(state)

    assert "/checkpoint/qualified-best" in feedback
    assert "/checkpoint/qualified-best#git-diff" in feedback
    assert "/checkpoint/latest-unqualified" not in feedback


@pytest.mark.parametrize("plan_method,target,verdict,includes_rejection", [
    ("method://current", "experiment_plan", "fail", True),
    ("method://older", "experiment_plan", "fail", False),
    ("method://current", "experiment_plan", "pass", False),
    ("method://current", "method", "fail", False),
])
async def test_reconciliation_feedback_keeps_current_plan_rejection(
        fake_bus, monkeypatch, plan_method, target, verdict, includes_rejection):
    from types import SimpleNamespace

    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    state = SessionState(
        session_id="reconcile-plan", user_id="u1", current_state="DeliberatingMethod",
        method_ref="method://current", component_plan_audit_pending=True,
        research_memory_ref="memory://current",
        experiment_plan_refs=["plan://current"], critique_report_refs=["critique://current"],
    )
    objects = {
        "memory://current": {
            "diagnosis": "method_description_reconciliation",
            "diagnosis_evidence": ["retain the earlier measured feature observation"],
        },
        "plan://current": {"hypothesis_id": plan_method},
        "critique://current": {
            "target": target, "verdict": verdict,
            "issues": ["current plan does not specify how to measure its control loss"],
        },
    }
    eng.spec_store = SimpleNamespace(get=objects.__getitem__)
    monkeypatch.setattr(eng, "_current_gen_result", lambda _sid: {
        "method_ref": "method://older",
        "method_implementation_review": {"verdict": "drifted", "issues": ["old feature mismatch"]},
        "diff": "diff --git a/feature.py b/feature.py",
    })
    feedback = eng._method_revision_feedback(state)
    assert "retain the earlier measured feature observation" in feedback
    assert "old feature mismatch" in feedback
    assert "diff --git a/feature.py" in feedback
    assert ("current plan does not specify how to measure its control loss" in feedback) is includes_rejection
    if includes_rejection:
        assert "plan://current" in feedback and "critique://current" in feedback


@pytest.mark.parametrize("matching_component", [True, False])
@pytest.mark.parametrize("prior_audit_failure", [True, False])
async def test_component_revision_receives_measured_contract_failure(
        fake_bus, monkeypatch, matching_component, prior_audit_failure):
    from types import SimpleNamespace

    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    state = SessionState(
        session_id="contract-repair", user_id="u1", current_state="DeliberatingMethod",
        component_plan_audit_pending=True, component_ablation_failed=True,
        ablation_failed_component="projection",
        moss_interventions=[{"moss_intervention_ref": "moss://fallback"}],
    )
    if prior_audit_failure:
        state.last_agent_failure = {
            "underlying_reason_code": "component_plan_audit_failed",
            "underlying_reason": "unchanged exhausted plan",
        }
    eng.spec_store = SimpleNamespace(get=lambda _ref: {
        "action": "revise_component", "rationale": "conservative fallback",
    })
    monkeypatch.setattr(eng, "_current_gen_result", lambda _sid: {
        "component_execution": {
            "mode": "component",
            "component_id": "projection" if matching_component else "other",
            "status": "contract_failed", "execution_succeeded": True,
            "contract_result": {
                "passed": False, "actual": 0.37,
                "expected": {"operator": "<=", "value": 0.01},
                "evidence": {"ref": "attempt://projection/probe"},
                "reason": "orthogonality check failed",
            },
        },
        "method_implementation_review": {"verdict": "faithful", "issues": []},
    })
    feedback = eng._method_revision_feedback(state)
    if prior_audit_failure:
        assert "unchanged exhausted plan" in feedback
    assert "conservative fallback" in feedback
    if matching_component:
        assert '"actual": 0.37' in feedback
        assert '"value": 0.01' in feedback
        assert "attempt://projection/probe" in feedback
        assert '"verdict": "faithful"' in feedback
    else:
        assert "attempt://projection/probe" not in feedback


async def test_component_revision_does_not_replay_stale_method_feedback(
        fake_bus, monkeypatch):
    """A later component-only repair must use its current MOSS diagnosis."""
    from types import SimpleNamespace

    old_review = {
        "status": "completed",
        "trigger": "mir_task_contract_violation",
        "decision": {
            "route": "revise_method",
            "validation_plan": "rerun the old baseline and whole method",
        },
    }
    current_moss = {
        "action": "revise_component",
        "rationale": "replace SHA distinctness with inode/mtime freshness evidence",
    }
    objects = {"review://old": old_review, "moss://current": current_moss}
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    eng.spec_store = SimpleNamespace(get=objects.__getitem__)
    state = SessionState(
        session_id="stale-method-feedback", user_id="u1",
        current_state="DeliberatingMethod", method_ref="method://current",
        baseline_result_ref="baseline://frozen",
        method_feedback_in_progress=True, method_feedback_ref="review://old",
        component_plan_audit_pending=True, component_ablation_failed=True,
        ablation_failed_component="cap128-conv700",
        moss_interventions=[{"moss_intervention_ref": "moss://current"}],
    )
    monkeypatch.setattr(eng, "_current_gen_result", lambda _sid: {
        "component_execution": {
            "mode": "component", "component_id": "cap128-conv700",
            "status": "contract_failed", "execution_succeeded": True,
        },
        "method_implementation_review": {
            "verdict": "drifted",
            "task_contract_review": {"verdict": "compliant"},
        },
    })

    payload = eng._entry_actions["DeliberatingMethod"][1](state, None)

    assert payload["method_feedback_ref"] is None
    assert "inode/mtime freshness evidence" in payload["component_plan_revision_feedback"]
    assert "rerun the old baseline and whole method" not in payload[
        "component_plan_revision_feedback"]


async def test_component_revision_keeps_memory_reason_despite_component_id_spelling(
        fake_bus, monkeypatch):
    from types import SimpleNamespace

    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    state = SessionState(
        session_id="contract-repair", user_id="u1", current_state="DeliberatingMethod",
        component_plan_audit_pending=True, component_ablation_failed=True,
        ablation_failed_component="PROJECTION_LAYER", research_memory_ref="memory://current",
    )
    eng.spec_store = SimpleNamespace(get=lambda _ref: {
        "diagnosis": "component_expectation_failed",
        "diagnosis_evidence": [
            "PROJECTION_LAYER", "mechanism requires two outputs but its declared interface has three",
        ],
    })
    monkeypatch.setattr(eng, "_current_gen_result", lambda _sid: {
        "component_execution": {
            "mode": "component", "component_id": "projection-layer",
            "contract_result": {"actual": 0.37},
        },
    })
    feedback = eng._method_revision_feedback(state)
    assert "mechanism requires two outputs but its declared interface has three" in feedback
    # Preserve the diagnosis without inventing an ID match for measurements.
    assert '"actual": 0.37' not in feedback


@pytest.mark.parametrize(
    ("exhausted", "method_exhausted"),
    [(False, False), (True, False), (True, True)],
)
async def test_component_reconciliation_audits_plan_before_revising_method(
        fake_bus, monkeypatch, exhausted, method_exhausted):
    from autoresearch.orchestrator.engine import _COMPONENT_REVISION_CAP

    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = "component-reconcile"
    await _seed_consolidating_memory_session(eng, sid)
    state = await load_session_state(fake_bus.db, sid)
    state.mode = "benchmark"
    state.method_ref = f"method://{sid}/confirmed"
    state.baseline_result_ref = f"baseline_result://{sid}/frozen"
    state.problem_model_ref = f"problem_model://{sid}/v1"
    state.component_revision_round = _COMPONENT_REVISION_CAP if exhausted else 0
    state.method_revision_round = _METHOD_REVISION_CAP if method_exhausted else 0
    await save_session_state(fake_bus.db, state, auto_commit=True)
    monkeypatch.setattr(eng, "_current_gen_result", lambda _sid: {
        "component_execution": {
            "mode": "component", "component_id": "projection", "status": "drifted",
            "execution_succeeded": True,
        },
    })

    await eng.tick(_Evt("memory.consolidated", sid, "MemoryConsolidatorAgent", payload={
        "routing_verdict": "continue_explore",
        "research_memory_ref": f"research_memory://{sid}/current",
        "diagnosis": "method_description_reconciliation",
        "diagnosis_evidence": ["legal observed projection differs from its component contract"],
    }))
    state = await load_session_state(fake_bus.db, sid)
    assert state.method_ref == f"method://{sid}/confirmed"
    assert state.baseline_result_ref == f"baseline_result://{sid}/frozen"
    assert state.research_memory_ref == f"research_memory://{sid}/current"
    expected_method_round = _METHOD_REVISION_CAP if method_exhausted else 0
    assert state.method_revision_round == expected_method_round
    assert state.method_swap_round == 0
    if exhausted:
        assert state.current_state == "RevisingMethod"
        assert state.component_plan_audit_pending is False
        assert len([e for e in fake_bus.published
                    if e.event_type == "method.revise_requested"]) == 1
        return
    assert not [e for e in fake_bus.published if e.event_type == "method.revise_requested"]
    assert state.current_state == "DeliberatingMethod"
    assert state.component_revision_round == 1
    assert state.component_plan_audit_pending is True
    event = next(e for e in fake_bus.published if e.event_type == "method.deliberation_requested")
    assert event.payload["component_plan_audit"] is True
    assert "projection" in event.payload["component_plan_revision_feedback"]

    await eng.tick(_Evt("method.deliberated", sid, "DeliberationAgent", payload={
        "method_ref": f"method://{sid}/confirmed",
    }))
    state = await load_session_state(fake_bus.db, sid)
    assert state.current_state == "PlanningExperiment"
    assert state.component_plan_audit_pending is False
    assert state.novelty_targeted_search_pending is False


async def test_fundamentally_new_method_resets_per_method_budgets_not_swap_budget(fake_bus):
    """Three exhausted old-method rounds must not make the new method's first non-win terminal."""
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = "sess-new-method-fresh-budget"
    old_experiment = f"method://{sid}/old"
    state = SessionState(
        session_id=sid, user_id="u1", current_state="GeneratingMethod",
        method_ref=old_experiment, current_experiment_id=old_experiment,
        method_not_working=True, method_swap_round=2,
        exploration_loop_round=3, setting_rework_round=3, method_revision_round=3,
        component_revision_round=3, agent_failure_round=3,
        implementation_fix_round={old_experiment: 3},
    )
    await save_session_state(fake_bus.db, state, auto_commit=True)
    new_method = f"method://{sid}/new"

    await eng.tick(_Evt(
        "method.generated", sid, "MethodAgent", payload={"method_ref": new_method}))

    state = await load_session_state(fake_bus.db, sid)
    assert state.method_swap_round == 2
    assert state.current_experiment_id == new_method
    assert state.exploration_loop_round == 0
    assert state.setting_rework_round == 0
    assert state.method_revision_round == 0
    assert state.component_revision_round == 0
    assert state.agent_failure_round == 0
    assert old_experiment not in state.implementation_fix_round

    # Drive the new method's first fair non-win at the trusted consolidation boundary.
    state.current_state = "ConsolidatingMemory"
    state.experiment_plan_refs = [f"experiment_plan://{sid}/new"]
    state.run_record_refs = [f"run_record://{sid}/new"]
    state.evaluation_report_refs = [f"evaluation_report://{sid}/new"]
    await save_session_state(fake_bus.db, state, auto_commit=True)
    await eng.tick(_Evt(
        "memory.consolidated", sid, "MemoryConsolidatorAgent",
        payload={
            "routing_verdict": "continue_explore",
            "research_memory_ref": f"research_memory://{sid}/new",
            "diagnosis": "inconclusive",
            "diagnosis_evidence": ["first fair new-method run did not improve"],
        },
    ))

    state = await load_session_state(fake_bus.db, sid)
    assert state.current_state == "PlanningExperiment"
    assert state.exploration_loop_round == 1
    assert state.method_swap_round == 2
