"""RankingJudgeAgent milestone Task 7: two dedicated integration smokes,
single session, no Redis, no real LLM, no real execution -- proving the two
headline behaviors this milestone adds on top of the (already-covered)
``ranking.requested -> ranking.ready`` functional round-trip
(``autoresearch/agents/ranking_judge_agent/tests/test_functional.py``):

1. ``test_hypothesis_pipeline_selects_top_ranked_into_planning_experiment``:
   the FULL hypothesis pipeline, from CritiquingHypotheses through
   RankingJudgeAgent's rank -> rank-informed critique co-audit (Task 4) and
   ManagingHypothesisPortfolio's ranking-driven auto-advance (Task 5),
   proving the hypothesis actually selected into PlanningExperiment is the
   TOP-BTL-ranked one -- not merely the first id CriticAgent happened to pass
   -- by rigging the portfolio-stage pairwise judgment so the SECOND
   candidate beats the first.

2. ``test_diversity_collapse_trips_counter_l_into_moss``: counter L
   (``search_diversity_collapse_round``, threshold 2) was UNREACHABLE before
   this milestone (nothing ever produced a ``diversity_status="collapsed"``
   RankingReport); this smoke wires RankingJudgeAgent with a canned
   "collapsed" pairwise response and drives ManagingHypothesisPortfolio's
   entry action twice (two independent isolated sessions, per-session state
   pre-seeded with the counter value the prior round would have left) to
   show BOTH halves of the counter's behavior: first collapse bumps L to 1
   and routes back to GeneratingHypotheses (below threshold); second
   collapse bumps L to 2 (== threshold) and trips MossIntervening with
   ``state.current_moss_reason == "search_diversity_collapse"``.

No concurrency, no Redis, no real LLM call, no real execution backend
anywhere in this file (every LLM is a canned FakeLLM queue).
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import aiosqlite
import pytest
import yaml
from jsonschema import Draft202012Validator

from autoresearch.agents.critic_agent.service import build_critic_service
from autoresearch.agents.critic_agent.tests.conftest import FakeLLM as CriticFakeLLM
from autoresearch.agents.ranking_judge_agent.service import build_ranking_judge_service
from autoresearch.agents.ranking_judge_agent.tests.conftest import FakeLLM as RankingJudgeFakeLLM
from autoresearch.orchestrator.engine import OrchestratorEngine
from autoresearch.orchestrator.session_state import SessionState, load_session_state, save_session_state
from autoresearch.spec_store import SpecStore
from autoresearch.testing.delivering_bus import DeliveringBus

SM_ROOT = Path(__file__).resolve().parents[2] / "state-machine"

RANKING_JUDGE_CONTRACTS_DIR = (
    Path(__file__).resolve().parents[3] / "agents" / "ranking_judge_agent" / "data-contracts"
)
RANKING_REPORT_SCHEMA_PATH = RANKING_JUDGE_CONTRACTS_DIR / "ranking_report.schema.yaml"


def _validator(path: Path) -> Draft202012Validator:
    schema = yaml.safe_load(path.read_text(encoding="utf-8"))
    return Draft202012Validator(schema)


def _dumps(obj: dict) -> str:
    return json.dumps(obj)


@pytest.fixture
async def bus(tmp_path):
    conn = await aiosqlite.connect(str(tmp_path / "ranking_judge_smoke.db"))
    try:
        yield DeliveringBus(conn)
    finally:
        await conn.close()


@pytest.fixture
def store(tmp_path):
    s = SpecStore(tmp_path / "store")
    yield s
    s.close()


# ===========================================================================
# (1) Hypothesis pipeline: CritiquingHypotheses -> RankingJudgeAgent (rank ->
# rank-informed critique co-audit) -> CriticAgent (some_pass) ->
# ManagingHypothesisPortfolio -> RankingJudgeAgent (ranking-driven
# auto-advance) -> PlanningExperiment, selecting the TOP-ranked hypothesis.
# ===========================================================================

VALID_RESEARCH_SPEC = {
    "topic": "Retrieval-augmented generation for long-tail scientific QA",
    "research_goal": "Propose a new RAG method for long-tail scientific question answering.",
}


def _hypothesis(hid: str) -> dict:
    return {
        "id": hid,
        "statement": f"Statement for {hid}: doing X improves Y by Z%.",
        "rationale": f"Rationale for {hid}, grounded in the literature brief.",
        "kind": "idea",
        "verifiable": True,
        "expected_effect": "Metric improves by an estimated 5-10%.",
    }


# Three-hypothesis bank; CriticAgent's canned some_pass verdict below passes
# TWO of them ("h1"/"h2"), so ManagingHypothesisPortfolio's ranking-driven
# selection (over a REAL 2-candidate pairwise judgment, not the 0/1-candidate
# trivial short-circuit) is what actually decides which one advances.
VALID_HYPOTHESES = [_hypothesis("h1"), _hypothesis("h2"), _hypothesis("h3")]

VALID_HYPOTHESES_REPORT_SOME_PASS = {
    "target": "hypotheses",
    "verdict": "some_pass",
    "issues": ["h3 is too narrow / trivially confounded."],
    "passing_hypothesis_ids": ["h1", "h2"],
}

# RankingJudgeAgent's FIRST canned response: CritiquingHypotheses' entry
# action ranks ALL THREE bank hypotheses (rank -> rank-informed critique
# co-audit, RankingJudgeAgent milestone Task 4). Its own ranked_shortlist
# isn't what gets selected -- only informs CriticAgent's critique -- so any
# schema-valid pairwise judgment works here.
RANKING_LLM_RESPONSE_ALL_THREE = {
    "pairwise_results": [
        {"a": "h1", "b": "h2", "winner": "h1", "rationale": "h1 edges h2."},
        {"a": "h1", "b": "h3", "winner": "h1", "rationale": "h1 beats h3."},
        {"a": "h2", "b": "h3", "winner": "h2", "rationale": "h2 beats h3."},
    ],
    "diversity_status": "diverse",
}

# RankingJudgeAgent's SECOND canned response: ManagingHypothesisPortfolio's
# entry action ranks ONLY the passing subset ("h1"/"h2", candidate_ids
# filter) -- deliberately rigged so "h2" (NOT "h1", the first-passing /
# first-in-bank-order id) wins the only pair, proving the selection is
# genuinely TOP-BTL-ranked, not a first-passing-id shortcut.
RANKING_LLM_RESPONSE_PORTFOLIO = {
    "pairwise_results": [
        {"a": "h1", "b": "h2", "winner": "h2", "rationale": "h2 is more falsifiable and higher-impact."},
    ],
    "diversity_status": "diverse",
}


async def test_hypothesis_pipeline_selects_top_ranked_into_planning_experiment(bus, store):
    # --- wire OrchestratorEngine (WITH spec_store -- ManagingHypothesisPortfolio's
    # ranking-driven auto-advance reads the RankingReport independently of the
    # event payload) + RankingJudgeAgent + CriticAgent -- all on the SAME bus,
    # each with its own emitter identity.
    eng = OrchestratorEngine(client=bus.view("ResearchOrchestrator"), skill_root=SM_ROOT, spec_store=store)
    await eng.start()

    ranking_llm = RankingJudgeFakeLLM(
        [
            _dumps(RANKING_LLM_RESPONSE_ALL_THREE),
            _dumps(RANKING_LLM_RESPONSE_PORTFOLIO),
        ]
    )
    build_ranking_judge_service(
        client=bus.view("RankingJudgeAgent"),
        spec_store=store,
        llm_client=ranking_llm,
    )

    critic_llm = CriticFakeLLM([_dumps(VALID_HYPOTHESES_REPORT_SOME_PASS)])
    build_critic_service(
        client=bus.view("CriticAgent"),
        spec_store=store,
        llm_client=critic_llm,
    )

    # --- Isolated construction: seed the HypothesisBank + ResearchSpec
    # directly into the store, as HypothesisAgent / ProblemModelingAgent would
    # have upstream, then construct + save a SessionState already sitting in
    # CritiquingHypotheses.
    sid = str(uuid.uuid4())
    hypothesis_bank_ref = store.put(
        kind="hypothesis_bank",
        session_id=sid,
        key="final",
        obj={"hypotheses": VALID_HYPOTHESES},
        emitter="HypothesisAgent",
    )
    research_spec_ref = store.put(
        kind="research_spec",
        session_id=sid,
        key="final",
        obj=VALID_RESEARCH_SPEC,
        emitter="ProblemModelingAgent",
    )
    state = SessionState(
        session_id=sid,
        user_id="u1",
        current_state="CritiquingHypotheses",
        research_spec_ref=research_spec_ref,
        hypothesis_bank_ref=hypothesis_bank_ref,
    )
    await save_session_state(bus.db, state, auto_commit=True)

    # --- Drive it: resume_to_current_state re-emits CritiquingHypotheses'
    # entry action (ranking.requested, all 3 bank hypotheses) -> (delivered)
    # -> RankingJudgeAgent ranks all three (1st queued response), persists a
    # RankingReport, publishes ranking.ready -> (delivered) -> orchestrator's
    # _on_CritiquingHypotheses_ranking_ready stores hypothesis_ranking_report_ref
    # and publishes (IN-STATE) hypothesis.critique_requested -> (delivered) ->
    # CriticAgent resolves the HypothesisBank + ResearchSpec (+ RankingReport),
    # runs its canned some_pass critique (passing h1/h2), persists the
    # CritiqueReport, publishes hypothesis.critiqued(some_pass) -> (delivered)
    # -> orchestrator resets counter C, stores passing_hypothesis_ids=["h1",
    # "h2"], and transitions CritiquingHypotheses -> ManagingHypothesisPortfolio,
    # whose entry action (ranking.requested, candidate_ids=["h1","h2"]) ->
    # (delivered) -> RankingJudgeAgent ranks the passing subset (2nd queued
    # response, h2 beats h1), persists a second RankingReport, publishes
    # ranking.ready -> (delivered) -> orchestrator's
    # _on_ManagingHypothesisPortfolio_ranking_ready (RankingJudgeAgent
    # milestone Task 5) selects the TOP-ranked hypothesis ("h2"), resets
    # counter L, and transitions ManagingHypothesisPortfolio ->
    # PlanningExperiment, whose entry action (experiment.planning_requested)
    # has no consumer wired in this smoke, so the loop pauses there. Called
    # exactly once.
    result = await eng.resume_to_current_state(sid)
    assert result["resumed"] is True

    # --- Assertions -------------------------------------------------------
    final_state = await load_session_state(bus.db, sid)
    assert final_state is not None
    assert final_state.current_state == "PlanningExperiment"

    # THE HEADLINE PROOF: the selected hypothesis is the TOP of the
    # portfolio-stage ranking's shortlist -- not "h1" (the first-passing /
    # first-in-bank-order id).
    assert final_state.hypothesis_ranking_report_ref is not None
    portfolio_ranking_report = store.get(final_state.hypothesis_ranking_report_ref)
    portfolio_ranking_errors = list(
        _validator(RANKING_REPORT_SCHEMA_PATH).iter_errors(portfolio_ranking_report)
    )
    assert portfolio_ranking_errors == [], [e.message for e in portfolio_ranking_errors]
    assert portfolio_ranking_report["ranked_shortlist"][0] == "h2"
    assert final_state.current_hypothesis_id == portfolio_ranking_report["ranked_shortlist"][0]
    assert final_state.current_hypothesis_id == "h2"
    assert final_state.current_experiment_id == final_state.current_hypothesis_id
    assert final_state.search_diversity_collapse_round == 0

    assert final_state.passing_hypothesis_ids == ["h1", "h2"]
    assert final_state.hypothesis_bank_ref is not None

    # Exactly TWO ranking.ready events (CritiquingHypotheses's full-bank
    # ranking, then ManagingHypothesisPortfolio's passing-subset ranking).
    ranking_ready_events = [e for e in bus.published if e.event_type == "ranking.ready"]
    assert len(ranking_ready_events) == 2

    critiqued_events = [e for e in bus.published if e.event_type == "hypothesis.critiqued"]
    assert len(critiqued_events) == 1
    assert critiqued_events[0].payload["verdict"] == "some_pass"

    # No agent.failed anywhere along the way.
    assert not [e for e in bus.published if e.event_type == "agent.failed"]


# ===========================================================================
# (2) Counter L (search_diversity_collapse_round, threshold 2): previously
# UNREACHABLE -- proves BOTH the below-threshold bump (-> GeneratingHypotheses)
# and the at-threshold trip (-> MossIntervening, current_moss_reason ==
# "search_diversity_collapse").
# ===========================================================================

VALID_HYPOTHESES_L = [_hypothesis("h1"), _hypothesis("h2")]

# RankingJudgeAgent's canned "collapsed" pairwise response -- same shape used
# for BOTH phases below (a fresh FakeLLM queue per phase).
COLLAPSED_RANKING_LLM_RESPONSE = {
    "pairwise_results": [
        {"a": "h1", "b": "h2", "winner": "h1", "rationale": "Near-duplicate branches; h1 marginally ahead."},
    ],
    "diversity_status": "collapsed",
}


def _seed_bank_and_state(store, *, current_l: int) -> tuple[str, SessionState]:
    sid = str(uuid.uuid4())
    hypothesis_bank_ref = store.put(
        kind="hypothesis_bank",
        session_id=sid,
        key="final",
        obj={"hypotheses": VALID_HYPOTHESES_L},
        emitter="HypothesisAgent",
    )
    state = SessionState(
        session_id=sid,
        user_id="u1",
        current_state="ManagingHypothesisPortfolio",
        hypothesis_bank_ref=hypothesis_bank_ref,
        passing_hypothesis_ids=["h1", "h2"],
        search_diversity_collapse_round=current_l,
    )
    return sid, state


async def test_diversity_collapse_trips_counter_l_into_moss(bus, store):
    # --- wire OrchestratorEngine (WITH spec_store) + RankingJudgeAgent -- all
    # on the SAME bus, each with its own emitter identity. No CriticAgent
    # needed: both phases drive ManagingHypothesisPortfolio directly
    # (isolated construction), never CritiquingHypotheses.
    eng = OrchestratorEngine(client=bus.view("ResearchOrchestrator"), skill_root=SM_ROOT, spec_store=store)
    await eng.start()

    ranking_llm = RankingJudgeFakeLLM(
        [
            _dumps(COLLAPSED_RANKING_LLM_RESPONSE),  # phase A: L 0 -> 1
            _dumps(COLLAPSED_RANKING_LLM_RESPONSE),  # phase B: L 1 -> 2 (== threshold)
        ]
    )
    build_ranking_judge_service(
        client=bus.view("RankingJudgeAgent"),
        spec_store=store,
        llm_client=ranking_llm,
    )

    # --- Phase A: below threshold. Isolated construction: a session already
    # sitting in ManagingHypothesisPortfolio with search_diversity_collapse_round
    # == 0 (as if this is its FIRST portfolio pass).
    sid_a, state_a = _seed_bank_and_state(store, current_l=0)
    await save_session_state(bus.db, state_a, auto_commit=True)

    # --- Drive it: resume_to_current_state re-emits ManagingHypothesisPortfolio's
    # entry action (ranking.requested, candidate_ids=["h1","h2"]) ->
    # (delivered) -> RankingJudgeAgent runs a REAL 2-candidate pairwise
    # judgment (1st queued response, diversity_status="collapsed"), persists
    # a RankingReport, publishes ranking.ready -> (delivered) ->
    # orchestrator's _on_ManagingHypothesisPortfolio_ranking_ready bumps
    # counter L (search_diversity_collapse_round) from 0 to 1 -- BELOW the
    # threshold (2) -- and transitions ManagingHypothesisPortfolio ->
    # GeneratingHypotheses (broaden the pool), whose entry action
    # (hypothesis.generation_requested) has no consumer wired in this smoke,
    # so the loop pauses there.
    result_a = await eng.resume_to_current_state(sid_a)
    assert result_a["resumed"] is True

    state_a_final = await load_session_state(bus.db, sid_a)
    assert state_a_final is not None
    assert state_a_final.current_state == "GeneratingHypotheses"
    assert state_a_final.search_diversity_collapse_round == 1
    assert state_a_final.current_moss_reason is None

    # --- Phase B: at threshold. Isolated construction: a SEPARATE session
    # already sitting in ManagingHypothesisPortfolio with
    # search_diversity_collapse_round == 1 -- as if it already went through
    # ONE prior collapse round (phase A's outcome, replayed via direct
    # SessionState construction per the milestone's isolation-drive
    # allowance rather than re-driving the full GeneratingHypotheses ->
    # CritiquingHypotheses -> ManagingHypothesisPortfolio loop).
    sid_b, state_b = _seed_bank_and_state(store, current_l=1)
    await save_session_state(bus.db, state_b, auto_commit=True)

    # --- Drive it: resume_to_current_state re-emits ManagingHypothesisPortfolio's
    # entry action again -> (delivered) -> RankingJudgeAgent runs the SECOND
    # queued "collapsed" pairwise judgment, persists a RankingReport, publishes
    # ranking.ready -> (delivered) -> orchestrator bumps counter L from 1 to 2
    # -- AT the threshold (2) -- sets state.current_moss_reason to the
    # counter's own with_reason ("search_diversity_collapse", counters.yaml
    # counter L) and transitions ManagingHypothesisPortfolio ->
    # MossIntervening, whose entry action (moss.intervention_requested) has
    # no consumer wired in this smoke, so the loop pauses there -- THE
    # HEADLINE PROOF: counter L, previously unreachable, now trips MOSS.
    result_b = await eng.resume_to_current_state(sid_b)
    assert result_b["resumed"] is True

    state_b_final = await load_session_state(bus.db, sid_b)
    assert state_b_final is not None
    assert state_b_final.current_state == "MossIntervening"
    assert state_b_final.search_diversity_collapse_round == 2
    assert state_b_final.current_moss_reason == "search_diversity_collapse"

    # --- Both RankingReports are schema-valid and honestly report "collapsed".
    ranking_ready_events = [e for e in bus.published if e.event_type == "ranking.ready"]
    assert len(ranking_ready_events) == 2
    for event in ranking_ready_events:
        report = store.get(event.payload["ranking_report_ref"])
        errors = list(_validator(RANKING_REPORT_SCHEMA_PATH).iter_errors(report))
        assert errors == [], [e.message for e in errors]
        assert report["diversity_status"] == "collapsed"

    # No agent.failed anywhere along the way.
    assert not [e for e in bus.published if e.event_type == "agent.failed"]
