"""Unit tests for AuditingLiteratureNovelty verdict routing + counter K (post-3.3).

3.3 moved the literature-novelty audit to AFTER GeneratingMethod, so it now audits the METHOD's
novelty and routes:
  - verdict "pass"                 -> ConfirmingMethod (human gate confirms the vetted method),
                                       K reset to 0.
  - verdict "evidence_insufficient" -> bounded targeted search for the SAME method.
  - targeted-search budget exhausted -> MossIntervening, current_moss_reason ==
                                        "literature_novelty_audit_failed".

An idea_theory session is seeded DIRECTLY into AuditingLiteratureNovelty (reaching it via the real path now
requires driving through method generation); the routing handler under test only reads the verdict
+ K off the event/state.

Reuses FakeBusClient from test_engine_min.py (does NOT auto-deliver; the test drives the engine
directly via ``eng.tick(event)``).
"""

# Imported pytest fixtures intentionally share their parameter names below.
# ruff: noqa: F811

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

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


async def _seed_auditing_literature_novelty(
    eng: OrchestratorEngine,
    sid: str,
    *,
    literature_audit_round: int = 0,
    novelty_targeted_search_round: int = 0,
) -> tuple[str, str]:
    """Persist a session paused in AuditingLiteratureNovelty with the upstream refs the pass ->
    ConfirmingMethod / fail -> GeneratingMethod branches need (problem_model_ref + method_ref are
    what GeneratingMethod's entry action reads). Returns (research_spec_ref, literature_brief_ref)."""
    research_spec_ref = f"research_spec://{sid}/revised"
    literature_brief_ref = f"spec://literature_brief/{sid}/final"
    state = SessionState(
        session_id=sid,
        user_id="u1",
        mode="idea_theory",
        current_state="AuditingLiteratureNovelty",
        research_spec_ref=research_spec_ref,
        literature_brief_ref=literature_brief_ref,
        spec_interpretation_ref=f"spec_interpretation://{sid}/final",
        problem_model_ref=f"problem_model://{sid}/final",
        method_ref=f"method://{sid}/v1",
        literature_audit_round=literature_audit_round,
        novelty_targeted_search_round=novelty_targeted_search_round,
    )
    await save_session_state(eng.client.db, state, auto_commit=True)
    return research_spec_ref, literature_brief_ref


async def test_pass_verdict_advances_to_confirming_method(fake_bus):
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = await eng.start_session("goal", user_id="u1")
    await _seed_auditing_literature_novelty(eng, sid)

    novelty_audit_report_ref = f"novelty_audit_report://{sid}/final"
    n_before = len(fake_bus.published)
    await eng.tick(
        _Evt(
            "novelty.audit_ready",
            sid,
            "NoveltyAuditAgent",
            payload={
                "verdict": "pass",
                "novelty_audit_report_ref": novelty_audit_report_ref,
            },
        )
    )

    st = await load_session_state(fake_bus.db, sid)
    assert st is not None
    assert st.current_state == "ConfirmingMethod"   # human gate, no entry action -> pauses
    assert st.literature_audit_round == 0
    assert "AuditingLiteratureNovelty" not in st.agent_retry_count
    assert st.novelty_audit_report_ref == novelty_audit_report_ref
    # ConfirmingMethod has no entry action -> nothing emitted (the session pauses for the human).
    assert fake_bus.published[n_before:] == []


async def test_pass_verdict_transition_is_atomic_single_commit(fake_bus):
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = await eng.start_session("goal", user_id="u1")
    await _seed_auditing_literature_novelty(eng, sid)

    commits_before = fake_bus.db.commit_count
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

    assert fake_bus.db.commit_count - commits_before == 1


async def test_evidence_insufficient_starts_targeted_search_for_same_method(fake_bus):
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = await eng.start_session("goal", user_id="u1")
    await _seed_auditing_literature_novelty(eng, sid)

    n_before = len(fake_bus.published)
    await eng.tick(
        _Evt(
            "novelty.audit_ready",
            sid,
            "NoveltyAuditAgent",
            payload={
                "verdict": "evidence_insufficient",
                "novelty_audit_report_ref": f"novelty_audit_report://{sid}/final",
            },
        )
    )

    st = await load_session_state(fake_bus.db, sid)
    assert st is not None
    assert st.literature_audit_round == 0
    assert st.novelty_targeted_search_round == 1
    assert st.current_state == "DeepResearching"
    assert st.current_moss_reason is None

    new_events = fake_bus.published[n_before:]
    searches = [p for p in new_events if p.event_type == "deep_research.requested"]
    assert len(searches) == 1
    assert searches[0].payload["method_ref"] == f"method://{sid}/v1"
    assert searches[0].payload["novelty_audit_report_ref"] == (
        f"novelty_audit_report://{sid}/final")
    assert searches[0].payload["targeted_search_id"] == st.novelty_targeted_search_id
    assert not any(p.event_type == "method.generation_requested" for p in new_events)


async def test_not_feasible_first_round_returns_to_generating_method(fake_bus):
    # meeting 2026-07-25 item 9: the method audit now judges feasibility too. A novel-but-infeasible
    # method (fails relevance / logic / implementability) routes back to re-derive the method, sharing
    # the same K counter / loop as evidence_insufficient.
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = await eng.start_session("goal", user_id="u1")
    await _seed_auditing_literature_novelty(eng, sid)

    n_before = len(fake_bus.published)
    await eng.tick(
        _Evt(
            "novelty.audit_ready",
            sid,
            "NoveltyAuditAgent",
            payload={
                "verdict": "not_feasible",
                "novelty_audit_report_ref": f"novelty_audit_report://{sid}/final",
            },
        )
    )

    st = await load_session_state(fake_bus.db, sid)
    assert st is not None
    assert st.literature_audit_round == 1
    assert st.current_state == "GeneratingMethod"
    assert st.current_moss_reason is None
    method_reqs = [p for p in fake_bus.published[n_before:]
                   if p.event_type == "method.generation_requested"]
    assert len(method_reqs) == 1


async def test_not_feasible_at_threshold_trips_moss(fake_bus):
    # meeting item 9: repeated not_feasible verdicts are bounded by the same K threshold -> MOSS.
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = await eng.start_session("goal", user_id="u1")
    await _seed_auditing_literature_novelty(eng, sid, literature_audit_round=1)

    await eng.tick(
        _Evt(
            "novelty.audit_ready",
            sid,
            "NoveltyAuditAgent",
            payload={
                "verdict": "not_feasible",
                "novelty_audit_report_ref": f"novelty_audit_report://{sid}/final",
            },
        )
    )

    st = await load_session_state(fake_bus.db, sid)
    assert st is not None
    assert st.literature_audit_round == 2
    assert st.current_state == "MossIntervening"


async def test_evidence_insufficient_after_targeted_search_cap_trips_moss(fake_bus):
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = await eng.start_session("goal", user_id="u1")
    await _seed_auditing_literature_novelty(
        eng, sid, novelty_targeted_search_round=2)

    n_before = len(fake_bus.published)
    await eng.tick(
        _Evt(
            "novelty.audit_ready",
            sid,
            "NoveltyAuditAgent",
            payload={
                "verdict": "evidence_insufficient",
                "novelty_audit_report_ref": f"novelty_audit_report://{sid}/round2",
            },
        )
    )

    st = await load_session_state(fake_bus.db, sid)
    assert st is not None
    assert st.literature_audit_round == 0
    assert st.novelty_targeted_search_round == 2
    assert st.current_state == "MossIntervening"
    assert st.current_moss_reason == "literature_novelty_audit_failed"

    new_events = fake_bus.published[n_before:]
    assert not any(p.event_type == "method.generation_requested" for p in new_events)


async def test_targeted_literature_search_reaudits_the_same_method(fake_bus):
    """MOSS literature research must close the evidence gap without moving the method target."""
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = await eng.start_session("goal", user_id="u1")
    await _seed_auditing_literature_novelty(eng, sid)
    audit_ref = f"novelty_audit_report://{sid}/round2"

    await eng.tick(
        _Evt(
            "novelty.audit_ready",
            sid,
            "NoveltyAuditAgent",
            payload={"verdict": "evidence_insufficient",
                     "novelty_audit_report_ref": audit_ref},
        )
    )
    deep = [p for p in fake_bus.published if p.event_type == "deep_research.requested"][-1]
    targeted_id = deep.payload["targeted_search_id"]
    assert deep.payload == {
        "research_spec_ref": f"research_spec://{sid}/revised",
        "method_ref": f"method://{sid}/v1",
        "novelty_audit_report_ref": audit_ref,
        "targeted_search_id": targeted_id,
    }

    # A delayed result from an older search cannot consume the current one-shot request.
    await eng.tick(
        _Evt(
            "literature.brief_ready", sid, "DeepResearchAgent",
            payload={
                "literature_brief_ref": f"spec://literature_brief/{sid}/stale",
                "field_trajectory_brief_ref": f"spec://field_trajectory/{sid}/stale",
                "method_ref": f"method://{sid}/v1",
                "novelty_audit_report_ref": audit_ref,
                "targeted_search_id": "stale-search",
            },
        )
    )
    state = await load_session_state(fake_bus.db, sid)
    assert state.current_state == "DeepResearching"
    assert state.novelty_targeted_search_id == targeted_id

    await eng.tick(
        _Evt(
            "literature.brief_ready",
            sid,
            "DeepResearchAgent",
            payload={
                "literature_brief_ref": f"spec://literature_brief/{sid}/targeted",
                "field_trajectory_brief_ref": f"spec://field_trajectory/{sid}/targeted",
                "method_ref": f"method://{sid}/v1",
                "novelty_audit_report_ref": audit_ref,
                "targeted_search_id": targeted_id,
            },
        )
    )

    state = await load_session_state(fake_bus.db, sid)
    assert state.current_state == "AuditingLiteratureNovelty"
    assert state.method_ref == f"method://{sid}/v1"
    assert state.novelty_targeted_search_id == targeted_id
    audit = [p for p in fake_bus.published if p.event_type == "evidence.audit_requested"][-1]
    assert audit.payload["method_ref"] == f"method://{sid}/v1"
    assert audit.payload["targeted_search_id"] == targeted_id

    # The evidence and novelty stages retain the same request identity.
    await eng.tick(
        _Evt(
            "evidence.audited", sid, "EvidenceAuditAgent",
            payload={"evidence_audit_report_ref": f"spec://evidence/{sid}/targeted",
                     "targeted_search_id": targeted_id},
        )
    )
    novelty = [p for p in fake_bus.published if p.event_type == "novelty.audit_requested"][-1]
    assert novelty.payload["targeted_search_id"] == targeted_id

    await eng.tick(
        _Evt(
            "novelty.audit_ready", sid, "NoveltyAuditAgent",
            payload={"verdict": "evidence_insufficient",
                     "novelty_audit_report_ref": f"spec://novelty/{sid}/targeted",
                     "targeted_search_id": targeted_id},
        )
    )
    state = await load_session_state(fake_bus.db, sid)
    assert state.current_state == "DeepResearching"
    assert state.awaiting_user_decision_reason is None
    assert state.novelty_targeted_search_round == 2
    assert state.novelty_targeted_search_id is not None
    next_search = [
        p for p in fake_bus.published if p.event_type == "deep_research.requested"][-1]
    assert next_search.payload["method_ref"] == f"method://{sid}/v1"
    assert next_search.payload["novelty_audit_report_ref"] == (
        f"spec://novelty/{sid}/targeted")


async def test_targeted_literature_search_cap_replaces_the_exhausted_method(fake_bus):
    """Search exhaustion replaces the method instead of reporting before experiments."""
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = await eng.start_session("goal", user_id="u1")
    state = SessionState(
        session_id=sid,
        user_id="u1",
        current_state="MossIntervening",
        current_moss_reason="literature_novelty_audit_failed",
        research_spec_ref=f"research_spec://{sid}/revised",
        method_ref=f"method://{sid}/v3",
        novelty_audit_report_ref=f"novelty_audit_report://{sid}/round6",
        novelty_targeted_search_round=2,
    )
    await save_session_state(fake_bus.db, state, auto_commit=True)

    await eng.tick(
        _Evt(
            "moss.intervention_completed",
            sid,
            "MOSS",
            payload={
                "reason": "literature_novelty_audit_failed",
                "action": "targeted_search",
                "moss_intervention_ref": f"spec://moss/{sid}/cap",
                "degraded": False,
            },
        )
    )

    state = await load_session_state(fake_bus.db, sid)
    assert state.current_state == "GeneratingMethod"
    assert state.method_not_working is True
    assert state.method_swap_round == 1
    assert state.novelty_targeted_search_id is None
    assert state.novelty_targeted_search_round == 2
    assert state.moss_interventions[-1]["action"] == "targeted_search"
    method_request = [
        item for item in fake_bus.published
        if item.event_type == "method.generation_requested"][-1]
    assert method_request.payload["novelty_audit_report_ref"] == (
        f"novelty_audit_report://{sid}/round6")


async def test_targeted_literature_search_reports_after_method_swap_cap(fake_bus):
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = await eng.start_session("goal", user_id="u1")
    state = SessionState(
        session_id=sid,
        user_id="u1",
        current_state="MossIntervening",
        current_moss_reason="literature_novelty_audit_failed",
        research_spec_ref=f"research_spec://{sid}/revised",
        method_ref=f"method://{sid}/v3",
        novelty_audit_report_ref=f"novelty_audit_report://{sid}/round6",
        novelty_targeted_search_round=2,
        method_swap_round=3,
    )
    await save_session_state(fake_bus.db, state, auto_commit=True)

    await eng.tick(
        _Evt(
            "moss.intervention_completed",
            sid,
            "MOSS",
            payload={
                "reason": "literature_novelty_audit_failed",
                "action": "targeted_search",
                "moss_intervention_ref": f"spec://moss/{sid}/cap",
                "degraded": False,
            },
        )
    )

    state = await load_session_state(fake_bus.db, sid)
    assert state.current_state == "ConsolidatingMemory"


async def test_moss_accept_caveat_continues_without_user_pause(fake_bus):
    """MOSS's accepted novelty caveat is already the autonomous decision."""
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = await eng.start_session("goal", user_id="u1")
    state = SessionState(
        session_id=sid,
        user_id="u1",
        current_state="MossIntervening",
        current_moss_reason="literature_novelty_audit_failed",
        research_spec_ref=f"research_spec://{sid}/revised",
        method_ref=f"method://{sid}/v1",
        novelty_audit_report_ref=f"novelty_audit_report://{sid}/round1",
    )
    await save_session_state(fake_bus.db, state, auto_commit=True)

    await eng.tick(
        _Evt(
            "moss.intervention_completed",
            sid,
            "MOSS",
            payload={
                "reason": "literature_novelty_audit_failed",
                "action": "accept_caveat",
                "moss_intervention_ref": f"spec://moss/{sid}/caveat",
                "degraded": False,
            },
        )
    )

    state = await load_session_state(fake_bus.db, sid)
    assert state.current_state == "ConfirmingMethod"
    assert state.awaiting_user_decision_reason is None
    assert state.decision_caveats == ["报告标注 novelty caveat"]
    assert state.moss_interventions[-1]["action"] == "accept_caveat"


async def test_exhausted_search_replaces_method_on_degraded_human_papers_action(fake_bus):
    """A degraded MOSS fallback must not reopen an autonomous human checkpoint."""
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = await eng.start_session("goal", user_id="u1")
    state = SessionState(
        session_id=sid,
        user_id="u1",
        current_state="MossIntervening",
        current_moss_reason="literature_novelty_audit_failed",
        research_spec_ref=f"research_spec://{sid}/revised",
        method_ref=f"method://{sid}/v2",
        novelty_audit_report_ref=f"novelty_audit_report://{sid}/round6",
        novelty_targeted_search_round=2,
        method_swap_round=1,
    )
    await save_session_state(fake_bus.db, state, auto_commit=True)

    await eng.tick(
        _Evt(
            "moss.intervention_completed",
            sid,
            "MOSS",
            payload={
                "reason": "literature_novelty_audit_failed",
                "action": "request_human_papers",
                "moss_intervention_ref": f"spec://moss/{sid}/degraded",
                "degraded": True,
            },
        )
    )

    state = await load_session_state(fake_bus.db, sid)
    assert state.current_state == "GeneratingMethod"
    assert state.method_not_working is True
    assert state.method_swap_round == 2


async def test_resume_migrates_pending_only_targeted_search(fake_bus):
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = await eng.start_session("goal", user_id="u1")
    state = SessionState(
        session_id=sid,
        user_id="u1",
        current_state="DeepResearching",
        research_spec_ref=f"research_spec://{sid}/revised",
        method_ref=f"method://{sid}/v1",
        novelty_audit_report_ref=f"novelty_audit_report://{sid}/round2",
        novelty_targeted_search_pending=True,
    )
    await save_session_state(fake_bus.db, state, auto_commit=True)

    await eng.resume_to_current_state(sid)

    resumed = await load_session_state(fake_bus.db, sid)
    assert resumed.novelty_targeted_search_id
    request = [
        item for item in fake_bus.published
        if item.event_type == "deep_research.requested"][-1]
    assert request.payload["targeted_search_id"] == resumed.novelty_targeted_search_id
    assert request.payload["method_ref"] == state.method_ref


async def test_resume_pending_target_without_frozen_refs_fails_closed(fake_bus):
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = await eng.start_session("goal", user_id="u1")
    state = SessionState(
        session_id=sid,
        user_id="u1",
        current_state="DeepResearching",
        research_spec_ref=f"research_spec://{sid}/revised",
        novelty_targeted_search_pending=True,
    )
    await save_session_state(fake_bus.db, state, auto_commit=True)

    result = await eng.resume_to_current_state(sid)

    resumed = await load_session_state(fake_bus.db, sid)
    assert result["resumed"] is False
    assert resumed.current_state == "AwaitingUserDecision"
    assert resumed.awaiting_user_decision_reason == "literature_uncertain"


async def test_unknown_verdict_is_defensive_noop(fake_bus):
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = await eng.start_session("goal", user_id="u1")
    await _seed_auditing_literature_novelty(eng, sid)

    n_before = len(fake_bus.published)
    await eng.tick(
        _Evt(
            "novelty.audit_ready",
            sid,
            "NoveltyAuditAgent",
            payload={
                "verdict": "something_weird",
                "novelty_audit_report_ref": f"novelty_audit_report://{sid}/final",
            },
        )
    )

    st = await load_session_state(fake_bus.db, sid)
    assert st is not None
    assert st.current_state == "AuditingLiteratureNovelty"
    assert st.literature_audit_round == 0
    assert fake_bus.published[n_before:] == []
