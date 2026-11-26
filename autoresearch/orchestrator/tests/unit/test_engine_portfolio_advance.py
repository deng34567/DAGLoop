"""Unit tests for the ``some_pass`` branch of
``_on_CritiquingHypotheses_hypothesis_critiqued``, UPDATED by RankingJudgeAgent
milestone Task 4 (stores ``passing_hypothesis_ids`` instead of auto-advancing)
and Task 5 (ManagingHypothesisPortfolio is now ranking-driven -- see
test_engine_portfolio_ranking_advance.py for its entry action + ranking_ready
handler).

Covers (post RankingJudgeAgent Task 4/5):
  - the ``some_pass`` branch stores the WHOLE ``passing_hypothesis_ids`` list
    off the event payload onto ``state.passing_hypothesis_ids``, resets
    counter C, and transitions to ManagingHypothesisPortfolio -- whose entry
    action (``ranking.requested``, Task 5) fires immediately as part of that
    same ``_transition`` -- WITHOUT itself setting ``current_hypothesis_id``
    or ``current_experiment_id`` (selecting the actual hypothesis to advance
    is ``_on_ManagingHypothesisPortfolio_ranking_ready``'s job, once
    ``ranking.ready`` comes back).
      - non-empty -> passing_hypothesis_ids stored; ManagingHypothesisPortfolio's
        entry action emits ``ranking.requested``, but no
        ``experiment.planning_requested`` yet (that requires a subsequent
        ``ranking.ready``, not exercised here).
      - empty (contract violation, defensive) -> short-circuits exactly as
        before: does NOT set passing_hypothesis_ids, does NOT auto-advance;
        session transitions to ManagingHypothesisPortfolio (whose entry
        action still fires, with empty candidate_ids).
  - ``_advance_managing_hypothesis_portfolio`` (the old direct-select helper)
    was DELETED by Task 5, replaced by the ranking-driven entry action +
    ``_on_ManagingHypothesisPortfolio_ranking_ready``; no longer exercised
    here.

Reuses FakeBusClient/fake_bus from test_engine_min.py and
``_seed_critiquing_hypotheses_session`` / ``_Evt`` from
test_engine_critic_advance.py (does NOT auto-deliver; drives the engine
directly via ``eng.tick(event)``), mirroring the existing advance-test files'
patterns.
"""

from __future__ import annotations

from pathlib import Path

from autoresearch.orchestrator.engine import OrchestratorEngine
from autoresearch.orchestrator.session_state import load_session_state, save_session_state
from autoresearch.orchestrator.tests.unit.test_engine_critic_advance import (
    _Evt,
    _seed_critiquing_hypotheses_session,
)
from autoresearch.orchestrator.tests.unit.test_engine_min import FakeBusClient, fake_bus  # noqa: F401

SM_ROOT = Path(__file__).resolve().parents[2] / "state-machine"


async def test_some_pass_stores_passing_ids_and_does_not_auto_advance(fake_bus):
    """RankingJudgeAgent milestone Task 4: some_pass now stores the WHOLE
    passing_hypothesis_ids list and stops at ManagingHypothesisPortfolio --
    the ranking-driven selection + auto-advance is Task 5's job."""
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = "sess-portfolio-a"
    await _seed_critiquing_hypotheses_session(eng, sid)

    critique_report_ref = f"critique_report://{sid}/hypothesis/final"
    n_before = len(fake_bus.published)
    commits_before = fake_bus.db.commit_count
    await eng.tick(
        _Evt(
            "hypothesis.critiqued",
            sid,
            "CriticAgent",
            payload={
                "verdict": "some_pass",
                "critique_report_ref": critique_report_ref,
                "passing_hypothesis_ids": ["h1", "h2"],
            },
        )
    )

    st = await load_session_state(fake_bus.db, sid)
    assert st is not None
    assert st.passing_hypothesis_ids == ["h1", "h2"]
    assert st.current_hypothesis_id is None
    assert st.current_experiment_id is None
    assert st.hypothesis_reject_round == 0
    assert st.critique_report_refs == [critique_report_ref]
    # Stops at ManagingHypothesisPortfolio -- picking the actual hypothesis to
    # advance is _on_ManagingHypothesisPortfolio_ranking_ready's job (Task 5),
    # triggered by a subsequent ranking.ready, not exercised here.
    assert st.current_state == "ManagingHypothesisPortfolio"

    # ManagingHypothesisPortfolio's entry action (ranking.requested, Task 5)
    # is bundled into the same _transition commit -> still exactly one commit.
    assert fake_bus.db.commit_count - commits_before == 1

    new_events = fake_bus.published[n_before:]
    assert not any(p.event_type == "experiment.planning_requested" for p in new_events)


async def test_some_pass_empty_passing_ids_short_circuits(fake_bus):
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = "sess-portfolio-b"
    await _seed_critiquing_hypotheses_session(eng, sid)
    # Pre-set a prior selection to confirm the defensive path does NOT clobber it.
    state = await load_session_state(fake_bus.db, sid)
    assert state is not None
    state.current_hypothesis_id = "prev-h"
    await save_session_state(fake_bus.db, state, auto_commit=True)

    n_before = len(fake_bus.published)
    commits_before = fake_bus.db.commit_count
    await eng.tick(
        _Evt(
            "hypothesis.critiqued",
            sid,
            "CriticAgent",
            payload={
                "verdict": "some_pass",
                "critique_report_ref": f"critique_report://{sid}/hypothesis/final",
                "passing_hypothesis_ids": [],
            },
        )
    )

    st = await load_session_state(fake_bus.db, sid)
    assert st is not None
    assert st.current_state == "ManagingHypothesisPortfolio"
    assert st.current_hypothesis_id == "prev-h"
    assert st.hypothesis_reject_round == 0

    # ManagingHypothesisPortfolio's entry action (ranking.requested, Task 5)
    # is bundled into the same _transition commit -> still exactly one commit.
    assert fake_bus.db.commit_count - commits_before == 1
    new_events = fake_bus.published[n_before:]
    assert not any(p.event_type == "experiment.planning_requested" for p in new_events)
