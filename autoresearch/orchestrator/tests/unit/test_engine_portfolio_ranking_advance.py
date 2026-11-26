"""Unit tests for RankingJudgeAgent milestone Task 5: ManagingHypothesisPortfolio
becomes ranking-driven.

Covers:
  - the NEW ManagingHypothesisPortfolio entry action, which emits
    ``ranking.requested {candidate_kind: "hypothesis", candidate_ids:
    state.passing_hypothesis_ids, source_ref: state.hypothesis_bank_ref}`` --
    unlike CritiquingHypotheses' entry action, ``candidate_ids`` IS supplied
    here so RankingJudgeAgent ranks just the passing subset.
  - ``_on_ManagingHypothesisPortfolio_ranking_ready``: resolves the
    RankingReport from the store and routes on ``diversity_status``:
      - "diverse"   -> select the TOP-ranked hypothesis (``ranked_shortlist[0]``,
        NOT the first passing id); reset counter L
        (search_diversity_collapse_round) to 0; transition to
        PlanningExperiment (its EXISTING M8 entry action fires).
      - "collapsed" -> bump counter L; below threshold (2) ->
        GeneratingHypotheses (broaden); at/above threshold ->
        current_moss_reason == "search_diversity_collapse" and transition to
        MossIntervening. THIS is the previously-unreachable counter L /
        MOSS reason -- activated here.
      - unknown diversity_status -> defensive no-op (logged), no crash.

``_advance_managing_hypothesis_portfolio`` is DELETED by this task (replaced
by the above); its direct-call coverage in test_engine_portfolio_advance.py
is removed alongside it.

Reuses FakeBusClient/fake_bus from test_engine_min.py (does NOT auto-deliver;
drives the engine directly via ``eng.tick(event)`` / ``eng._execute_entry_action``),
and a REAL tmp SpecStore (mirroring test_engine_coder_advance.py) since the
ranking_ready handler must resolve the RankingReport independently of the
event payload.
"""

from __future__ import annotations

from pathlib import Path

from autoresearch.orchestrator.engine import OrchestratorEngine
from autoresearch.orchestrator.session_state import (
    SessionState,
    load_session_state,
    save_session_state,
)
from autoresearch.orchestrator.tests.unit.test_engine_critic_advance import _Evt
from autoresearch.orchestrator.tests.unit.test_engine_min import FakeBusClient, fake_bus  # noqa: F401
from autoresearch.spec_store.store import SpecStore

SM_ROOT = Path(__file__).resolve().parents[2] / "state-machine"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _seed_managing_portfolio_session(
    eng: OrchestratorEngine,
    sid: str,
    *,
    passing_hypothesis_ids: list[str] = ["h1", "h2"],
    search_diversity_collapse_round: int = 0,
) -> tuple[str, str]:
    """Persist a session in ManagingHypothesisPortfolio with the upstream refs
    PlanningExperiment's entry action needs, plus ``passing_hypothesis_ids``.
    Returns (hypothesis_bank_ref, research_spec_ref)."""
    hypothesis_bank_ref = f"hypothesis_bank://{sid}/final"
    research_spec_ref = f"research_spec://{sid}/revised"
    state = SessionState(
        session_id=sid,
        user_id="u1",
        current_state="ManagingHypothesisPortfolio",
        research_spec_ref=research_spec_ref,
        hypothesis_bank_ref=hypothesis_bank_ref,
        verifier_spec_ref=f"verifier_spec://{sid}/final",
        benchmark_spec_ref=f"benchmark_spec://{sid}/final",
        passing_hypothesis_ids=list(passing_hypothesis_ids),
        search_diversity_collapse_round=search_diversity_collapse_round,
    )
    await save_session_state(eng.client.db, state, auto_commit=True)
    return hypothesis_bank_ref, research_spec_ref


def _seed_ranking_report(store: SpecStore, sid: str, key: str, *, ranked_shortlist: list[str], diversity_status: str) -> str:
    return store.put(
        kind="ranking_report",
        session_id=sid,
        key=key,
        obj={
            "candidate_kind": "hypothesis",
            "pairwise_results": [],
            "btl_scores": [],
            "ranked_shortlist": ranked_shortlist,
            "diversity_status": diversity_status,
        },
        emitter="RankingJudgeAgent",
    )


# ---------------------------------------------------------------------------
# (a): ManagingHypothesisPortfolio entry action
# ---------------------------------------------------------------------------


async def test_managing_portfolio_entry_action_emits_ranking_requested_with_candidate_ids(
    fake_bus, tmp_path
):
    store = SpecStore(tmp_path / "store")
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT, spec_store=store)
    await eng.start()
    sid = "sess-a"
    hypothesis_bank_ref, _research_spec_ref = await _seed_managing_portfolio_session(eng, sid)
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
        "candidate_ids": ["h1", "h2"],
        "source_ref": hypothesis_bank_ref,
    }


# ---------------------------------------------------------------------------
# (b): ranking.ready, diverse -> top-ranked hypothesis + PlanningExperiment + L reset
# ---------------------------------------------------------------------------


async def test_ranking_ready_diverse_selects_top_ranked_and_advances(fake_bus, tmp_path):
    store = SpecStore(tmp_path / "store")
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT, spec_store=store)
    await eng.start()
    sid = "sess-b"
    await _seed_managing_portfolio_session(eng, sid, search_diversity_collapse_round=1)
    ranking_report_ref = _seed_ranking_report(
        store, sid, "final", ranked_shortlist=["h2", "h1"], diversity_status="diverse"
    )

    n_before = len(fake_bus.published)
    await eng.tick(
        _Evt(
            "ranking.ready",
            sid,
            "RankingJudgeAgent",
            payload={"ranking_report_ref": ranking_report_ref, "candidate_kind": "hypothesis"},
        )
    )

    st = await load_session_state(fake_bus.db, sid)
    assert st is not None
    assert st.hypothesis_ranking_report_ref == ranking_report_ref
    # TOP-ranked (shortlist[0] == "h2"), NOT the first passing id ("h1").
    assert st.current_hypothesis_id == "h2"
    assert st.current_experiment_id == "h2"
    assert st.search_diversity_collapse_round == 0
    assert st.current_state == "PlanningExperiment"

    new_events = fake_bus.published[n_before:]
    emitted = [p for p in new_events if p.event_type == "experiment.planning_requested"]
    assert len(emitted) == 1
    assert emitted[0].payload["selected_hypothesis_id"] == "h2"


# ---------------------------------------------------------------------------
# (c): ranking.ready, collapsed, below threshold -> bump L, broaden (GeneratingHypotheses)
# ---------------------------------------------------------------------------


async def test_ranking_ready_collapsed_below_threshold_broadens(fake_bus, tmp_path):
    store = SpecStore(tmp_path / "store")
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT, spec_store=store)
    await eng.start()
    sid = "sess-c"
    await _seed_managing_portfolio_session(eng, sid, search_diversity_collapse_round=0)
    ranking_report_ref = _seed_ranking_report(
        store, sid, "final", ranked_shortlist=["h1", "h2"], diversity_status="collapsed"
    )

    n_before = len(fake_bus.published)
    await eng.tick(
        _Evt(
            "ranking.ready",
            sid,
            "RankingJudgeAgent",
            payload={"ranking_report_ref": ranking_report_ref, "candidate_kind": "hypothesis"},
        )
    )

    st = await load_session_state(fake_bus.db, sid)
    assert st is not None
    assert st.search_diversity_collapse_round == 1
    assert st.current_state == "GeneratingHypotheses"
    assert st.current_moss_reason is None

    new_events = fake_bus.published[n_before:]
    assert any(p.event_type == "hypothesis.generation_requested" for p in new_events)


# ---------------------------------------------------------------------------
# (d): ranking.ready, collapsed, at threshold -> counter L trips MOSS
# (search_diversity_collapse) -- the previously-unreachable path.
# ---------------------------------------------------------------------------


async def test_ranking_ready_collapsed_at_threshold_trips_moss(fake_bus, tmp_path):
    store = SpecStore(tmp_path / "store")
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT, spec_store=store)
    await eng.start()
    sid = "sess-d"
    await _seed_managing_portfolio_session(eng, sid, search_diversity_collapse_round=1)
    ranking_report_ref = _seed_ranking_report(
        store, sid, "final", ranked_shortlist=["h1", "h2"], diversity_status="collapsed"
    )

    n_before = len(fake_bus.published)
    await eng.tick(
        _Evt(
            "ranking.ready",
            sid,
            "RankingJudgeAgent",
            payload={"ranking_report_ref": ranking_report_ref, "candidate_kind": "hypothesis"},
        )
    )

    st = await load_session_state(fake_bus.db, sid)
    assert st is not None
    assert st.search_diversity_collapse_round == 2
    assert st.current_state == "MossIntervening"
    assert st.current_moss_reason == "search_diversity_collapse"

    new_events = fake_bus.published[n_before:]
    assert any(p.event_type == "moss.intervention_requested" for p in new_events)


# ---------------------------------------------------------------------------
# (e): unknown diversity_status -> defensive no-op, no crash.
# ---------------------------------------------------------------------------


async def test_ranking_ready_unknown_diversity_status_is_noop(fake_bus, tmp_path):
    store = SpecStore(tmp_path / "store")
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT, spec_store=store)
    await eng.start()
    sid = "sess-e"
    await _seed_managing_portfolio_session(eng, sid)
    ranking_report_ref = _seed_ranking_report(
        store, sid, "final", ranked_shortlist=["h1", "h2"], diversity_status="unknown_status"
    )

    n_before = len(fake_bus.published)
    await eng.tick(
        _Evt(
            "ranking.ready",
            sid,
            "RankingJudgeAgent",
            payload={"ranking_report_ref": ranking_report_ref, "candidate_kind": "hypothesis"},
        )
    )

    st = await load_session_state(fake_bus.db, sid)
    assert st is not None
    assert st.current_state == "ManagingHypothesisPortfolio"
    assert st.current_hypothesis_id is None

    new_events = fake_bus.published[n_before:]
    assert not any(
        p.event_type in ("experiment.planning_requested", "hypothesis.generation_requested",
                          "moss.intervention_requested")
        for p in new_events
    )
