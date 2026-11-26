"""M6 Task 5 integration smokes: single session, no Redis, no real LLM.

TWO smokes, both driving genuine subscribe/dispatch routing through a shared
in-memory ``DeliveringBus`` (copied from ``test_mainline_to_critique_smoke.py``
to keep this file self-contained):

1. ``test_mainline_reaches_selecting_track_single_session`` -- extends
   ``test_mainline_to_critique_smoke.py`` by one more hop: wires CriticAgent
   alongside the same five mainline components (OrchestratorEngine,
   ProblemModelingAgent, DeepResearchAgent, EvidenceAuditAgent,
   NoveltyAuditAgent) so the
   ``direction.critique_requested`` emitted at ``CritiquingDirection`` is
   actually consumed. CriticAgent's canned (pass) verdict routes
   CritiquingDirection -> RoutingResearchMode (no entry action), which
   auto-advances (M7 Task 4) to SelectingBenchmark; its entry action
   (benchmark.selection_requested) has no consumer wired in this smoke, so
   the loop pauses there.

2. ``test_hypothesis_critique_loop_reaches_managing_portfolio_single_session``
   -- extends ``test_m5_hypothesis_smoke.py``'s isolated-construction pattern
   by three more hops: wires RankingJudgeAgent + CriticAgent alongside
   OrchestratorEngine + HypothesisAgent so the ``ranking.requested`` emitted
   at ``CritiquingHypotheses`` (RankingJudgeAgent milestone Task 4) is
   actually consumed, RankingJudgeAgent's ``ranking.ready`` triggers the
   IN-STATE co-audit publish of ``hypothesis.critique_requested`` (carrying
   ``ranking_report_ref``), and CriticAgent's canned (some_pass) verdict,
   grounded in the SAME hypothesis ids the HypothesisAgent's FakeLLM
   generates ("h1"/"h2"), routes CritiquingHypotheses ->
   ManagingHypothesisPortfolio, whose entry action re-ranks the single
   passing hypothesis ("h1", the 0/1-candidate trivial short-circuit, no LLM
   call) and auto-advances (RankingJudgeAgent milestone Task 5) into
   PlanningExperiment.

No concurrency, no Redis, no real LLM call and no real search backend
anywhere in this file (every LLM is a canned FakeLLM queue; FakeBackend
canned papers; human answers/idea-checkpoint response are simulated, not a
real UI/human).
"""

from __future__ import annotations

import fnmatch
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import aiosqlite
import pytest
import yaml
from jsonschema import Draft202012Validator

from autoresearch.agents._reuse.deep_research_backend import FakeBackend
from autoresearch.agents.critic_agent.service import build_critic_service
from autoresearch.agents.critic_agent.tests.conftest import FakeLLM as CriticFakeLLM
from autoresearch.agents.deep_research_agent.service import build_deep_research_service
from autoresearch.agents.deliberation_agent.service import build_deliberation_service
from autoresearch.agents.deliberation_agent.tests.conftest import FakeLLM as DeliberationFakeLLM
from autoresearch.agents.deep_research_agent.tests.conftest import FakeLLM as DeepResearchFakeLLM
from autoresearch.agents.evidence_audit_agent.service import build_evidence_audit_service
from autoresearch.agents.evidence_audit_agent.tests.conftest import FakeLLM as EvidenceAuditFakeLLM
from autoresearch.agents.hypothesis_agent.service import build_hypothesis_service
from autoresearch.agents.hypothesis_agent.tests.conftest import FakeLLM as HypothesisFakeLLM
from autoresearch.agents.novelty_audit_agent.service import build_novelty_audit_service
from autoresearch.agents.novelty_audit_agent.tests.conftest import FakeLLM as NoveltyAuditFakeLLM
from autoresearch.agents.problem_modeling_agent.service import build_problem_modeling_service
from autoresearch.agents.problem_modeling_agent.tests.conftest import FakeLLM as ProblemModelingFakeLLM
from autoresearch.agents.ranking_judge_agent.service import build_ranking_judge_service
from autoresearch.agents.ranking_judge_agent.tests.conftest import FakeLLM as RankingJudgeFakeLLM
from autoresearch.eventbus.broker.envelope import Envelope
from autoresearch.orchestrator.engine import OrchestratorEngine
from autoresearch.orchestrator.session_state import SessionState, load_session_state, save_session_state
from autoresearch.spec_store import SpecStore

SM_ROOT = Path(__file__).resolve().parents[2] / "state-machine"
PROBLEM_MODELING_CONTRACTS_DIR = (
    Path(__file__).resolve().parents[3] / "agents" / "problem_modeling_agent" / "data-contracts"
)
PROBLEM_MODEL_SCHEMA_PATH = PROBLEM_MODELING_CONTRACTS_DIR / "problem_model.schema.yaml"
CRITIC_CONTRACTS_DIR = Path(__file__).resolve().parents[3] / "agents" / "critic_agent" / "data-contracts"
CRITIQUE_REPORT_SCHEMA_PATH = CRITIC_CONTRACTS_DIR / "critique_report.schema.yaml"
HYPOTHESIS_CONTRACTS_DIR = (
    Path(__file__).resolve().parents[3] / "agents" / "hypothesis_agent" / "data-contracts"
)
HYPOTHESIS_BANK_SCHEMA_PATH = HYPOTHESIS_CONTRACTS_DIR / "hypothesis_bank.schema.yaml"


def _validator(path: Path) -> Draft202012Validator:
    schema = yaml.safe_load(path.read_text(encoding="utf-8"))
    return Draft202012Validator(schema)


# ---------------------------------------------------------------------------
# DeliveringBus: minimal in-memory broker that actually routes.
# Copied from test_mainline_to_critique_smoke.py to keep this file
# self-contained.
# ---------------------------------------------------------------------------


class DeliveringBus:
    """Shared broker state: subscriptions, the real sqlite handle, and every
    published ``Envelope`` -- delivered synchronously to matching subscribers
    (mirrors ``Dispatcher.dispatch``'s sequential fnmatch routing)."""

    def __init__(self, db: aiosqlite.Connection) -> None:
        self.db = db
        self.subscriptions: list[tuple[str, Any]] = []
        self.published: list[Envelope] = []

    def subscribe(self, pattern: str, handler: Any) -> None:
        self.subscriptions.append((pattern, handler))

    async def publish(
        self,
        *,
        event_type: str,
        session_id: str,
        payload: dict,
        emitter: str,
        cost_delta: str = "0",
        causation_id: Optional[str] = None,
        auto_commit: bool = True,
    ) -> str:
        env = Envelope(
            event_id=str(uuid.uuid4()),
            session_id=session_id,
            event_type=event_type,
            emitter=emitter,
            timestamp=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            schema_version=1,
            cost_delta=cost_delta,
            payload=payload,
            causation_id=causation_id,
        )
        self.published.append(env)
        if auto_commit:
            await self.db.commit()
        for pattern, handler in list(self.subscriptions):
            if fnmatch.fnmatchcase(event_type, pattern):
                await handler(env)
        return env.event_id


class BusClientView:
    """One emitter's handle onto a shared ``DeliveringBus`` -- mirrors how
    each real process holds its own ``EventBusClient`` (its own
    ``emitter_name``) while every client shares one broker/outbox."""

    def __init__(self, bus: DeliveringBus, emitter_name: str) -> None:
        self._bus = bus
        self.emitter_name = emitter_name

    @property
    def db(self) -> aiosqlite.Connection:
        return self._bus.db

    def subscribe(self, pattern: str, handler: Any) -> None:
        self._bus.subscribe(pattern, handler)

    async def publish(
        self,
        *,
        event_type: str,
        session_id: str,
        payload: dict,
        cost_delta: str = "0",
        causation_id: Optional[str] = None,
        auto_commit: bool = True,
    ) -> str:
        return await self._bus.publish(
            event_type=event_type,
            session_id=session_id,
            payload=payload,
            emitter=self.emitter_name,
            cost_delta=cost_delta,
            causation_id=causation_id,
            auto_commit=auto_commit,
        )


@pytest.fixture
async def bus(tmp_path):
    conn = await aiosqlite.connect(str(tmp_path / "m6_critic_smoke.db"))
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
# Smoke 1: mainline start_session -> ... -> CritiquingDirection ->
# CriticAgent (pass) -> RoutingResearchMode -> (auto-advance) -> SelectingBenchmark.
# ===========================================================================

FINAL_SPEC = {
    "topic": "modded-nanogpt speedrun optimization",
    "research_goal": (
        "Reduce modded-nanogpt training wall-clock time to a new speedrun "
        "record while preserving the target validation loss."
    ),
    "task_mode": "experiment_validation",
    "budget": "two weeks",
    "deliverable_form": "benchmark report",
    "editable_scope": ["training script", "optimizer config"],
    "forbidden_items": ["changing the model architecture"],
    "open_questions": [],
}

# Grounded in the FakeBackend's default seminal + frontier papers (see
# autoresearch/agents/_reuse/deep_research_backend.py) so core's
# anti-hallucination check (key_papers subset of retrieved papers) passes.
SEMINAL_PAPER = {
    "title": "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
    "id_or_url": "arxiv:2005.11401",
    "year": 2020,
    "note": "Seminal RAG paper.",
}
FRONTIER_PAPER = {
    "title": "Long-Tail Scientific QA via Retrieval-Augmented Fine-Tuning",
    "id_or_url": "arxiv:2401.00000",
    "year": 2024,
    "note": "Recent frontier work on long-tail scientific QA.",
}

VALID_NOVELTY_AUDIT_REPORT_PASS = {
    "verdict": "pass",
    "citation_sufficiency": "Citations are sufficient and current for the proposed direction.",
    "stale_citations": [],
    "unsupported_claims": [],
    "missing_work": [],
    "novelty_status": "novel",
    "method_signature": {
        "primary_core": {
            "core_id": "core-1",
            "intervention": "Change the optimizer update.",
            "causal_mechanism": "Improve optimization efficiency.",
            "pipeline_role": "training update",
            "expected_effect": "Reach the target metric more efficiently.",
        },
        "supporting_components": [],
        "composition_claim": {"material": False, "mechanism": ""},
    },
    "prior_work_comparisons": [
        {
            "prior_work_id": "arxiv:2005.11401",
            "primary_core_relation": "distinct",
            "difference_scope": "material_core",
            "composition_relation": "not_applicable",
            "supporting_overlap": [],
            "evidence": ["The cited work studies retrieval augmentation, not optimizer updates."],
        }
    ],
    "retrieval_coverage": {
        "covered_core_ids": ["core-1"],
        "compared_prior_work_ids": ["arxiv:2005.11401"],
        "unresolved_candidates": [],
    },
    "harness_decision": {
        "novelty_status": "novel",
        "verdict": "pass",
        "decisive_prior_work_ids": [],
        "rationale": "All grounded closest-work comparisons are distinct at the primary causal mechanism.",
    },
    "notes": "No critical gaps found.",
}

VALID_EVIDENCE_AUDIT_REPORT = {
    "claim_citation_matrix": [
        {
            "claim": "Reducing modded-nanogpt wall-clock time is achievable via optimizer config changes.",
            "citations": ["arxiv:2005.11401"],
            "support": "strong",
        }
    ],
    "missing_papers": [],
    "expired_citations": [],
    "evidence_strength": "strong",
    "notes": "Citations sufficiently support the ResearchSpec's claims.",
}

VALID_PROBLEM_MODEL = {
    "key_problems": ["Reducing modded-nanogpt wall-clock time without regressing validation loss."],
    "opportunities": ["Reuse RAG-adjacent retrieval tricks are out of scope; focus on optimizer config."],
    "risks": ["Overfitting the speedrun to the benchmark's specific hardware."],
    "verifiable_paths": ["Re-run the benchmark suite and compare wall-clock + validation loss to baseline."],
}

VALID_DIRECTION_REPORT_PASS = {
    "target": "direction",
    "verdict": "pass",
    "issues": [],
    "passing_hypothesis_ids": [],
}


def _ask_json() -> str:
    return json.dumps(
        {
            "action": "ask",
            "question": {
                "question_id": "q-1",
                "question_text": "What is the primary deliverable form you expect?",
                "options": ["paper draft", "code repo", "benchmark report"],
                "allow_free_text": True,
            },
        }
    )


def _finalize_json(spec: dict) -> str:
    return json.dumps({"action": "finalize", "spec": spec})


def _reinterpret_json(revised_spec: dict) -> str:
    return json.dumps(
        {
            "spec_interpretation": {
                "naive_assumptions": [],
                "real_open_problem": (
                    "Speedrunning modded-nanogpt remains open given the RAG "
                    "literature surfaced is orthogonal to the training setup."
                ),
                "topic_position": "Unaffected by the retrieved RAG literature.",
                "disposition": "in_scope",
                "scope_reason": "The original ResearchSpec still holds; no scope change needed.",
                "evaluation_semantics": {
                    "comparison_statement": "Under the same canonical benchmark data, evaluator and resources, compare exact_match while allowing method implementation changes.",
                    "research_question": "Does the method improve exact_match under the canonical benchmark?",
                    "primary_metric": {"name": "exact_match", "direction": "maximize", "measurement": "Read exact_match from the canonical evaluator."},
                    "comparison_mode": "repository_defined",
                    "comparison_basis": "Use the selected repository benchmark.",
                    "budget_interpretation": "Use the benchmark's documented budget.",
                    "conditions_to_hold_constant": ["benchmark data, evaluator and resources"],
                    "method_freedom": ["method implementation"],
                    "data_source": "Canonical benchmark data.",
                    "evaluator_source": "Canonical benchmark evaluator.",
                    "observed_outcomes": ["runtime"],
                    "evaluation_notes": [],
                    "outcome_criteria": {"success": "Valid method exact_match improves.", "failure": "Execution fails or exact_match does not improve.", "inconclusive": "No valid comparable exact_match is produced."},
                    "evidence": [{"source_type": "research_spec", "source_id": "research_spec", "supports": "The research goal asks for benchmark improvement."}],
                    "uncertainties": [],
                },
            },
            "revised_research_spec": revised_spec,
        }
    )


def _synthesis_json() -> str:
    return json.dumps(
        {
            "literature_brief": {
                "summary": "RAG combines dense retrieval with generation.",
                "related_work": [],
                "key_papers": [SEMINAL_PAPER, FRONTIER_PAPER],
                "existing_benchmarks": [],
            },
            "field_trajectory_brief": {
                "origin": "RAG originated from combining retrieval with seq2seq generation.",
                "current_state": "Frontier work extends RAG to long-tail scientific QA.",
                "main_method_families": ["dense retrieval", "hybrid retrieval"],
                "open_problems": ["long-tail factual accuracy"],
            },
        }
    )


def _audit_report_json() -> str:
    return json.dumps(VALID_NOVELTY_AUDIT_REPORT_PASS)


def _evidence_audit_report_json() -> str:
    return json.dumps(VALID_EVIDENCE_AUDIT_REPORT)


def _problem_model_json() -> str:
    return json.dumps(VALID_PROBLEM_MODEL)


def _direction_report_json() -> str:
    return json.dumps(VALID_DIRECTION_REPORT_PASS)


class _MethodFakeLLM:
    """Canned MethodAgent LLM: returns queued method JSON (phase 3.1)."""
    def __init__(self, responses):
        self._r = list(responses)

    def chat(self, *, messages, response_format=None, max_tokens=None):
        return type("R", (), {"content": self._r.pop(0) if self._r else ""})()


async def test_mainline_reaches_selecting_track_single_session(bus, store):
    # --- wire OrchestratorEngine + ProblemModelingAgent + DeepResearchAgent +
    # EvidenceAuditAgent + NoveltyAuditAgent + CriticAgent -- all on the SAME
    # bus, each with its own emitter identity.
    eng = OrchestratorEngine(client=BusClientView(bus, "ResearchOrchestrator"), skill_root=SM_ROOT)
    await eng.start()

    # ONE FakeLLM handles all four ProblemModelingAgent LLM calls in order:
    # clarify-ask, clarify-finalize, reinterpret-in_scope, model_step.
    problem_modeling_llm = ProblemModelingFakeLLM(
        [_ask_json(), _finalize_json(FINAL_SPEC), _reinterpret_json(FINAL_SPEC), _problem_model_json()]
    )
    build_problem_modeling_service(
        client=BusClientView(bus, "ProblemModelingAgent"),
        spec_store=store,
        llm_client=problem_modeling_llm,
    )

    backend = FakeBackend()
    deep_research_llm = DeepResearchFakeLLM([
        json.dumps({"query": "literature search keywords"}),
        _synthesis_json(),
        json.dumps({"query": "closest prior method keywords", "method_primary_core":
                    VALID_NOVELTY_AUDIT_REPORT_PASS["method_signature"]["primary_core"]}),
        _synthesis_json(),
    ])
    build_deliberation_service(
        client=BusClientView(bus, "DeliberationAgent"),
        spec_store=store,
        llm_client=DeliberationFakeLLM(),
    )
    build_deep_research_service(
        client=BusClientView(bus, "DeepResearchAgent"),
        spec_store=store,
        backend=backend,
        llm_client=deep_research_llm,
    )

    evidence_audit_llm = EvidenceAuditFakeLLM([_evidence_audit_report_json()])
    build_evidence_audit_service(
        client=BusClientView(bus, "EvidenceAuditAgent"),
        spec_store=store,
        llm_client=evidence_audit_llm,
    )

    novelty_audit_llm = NoveltyAuditFakeLLM([_audit_report_json()])
    build_novelty_audit_service(
        client=BusClientView(bus, "NoveltyAuditAgent"),
        spec_store=store,
        llm_client=novelty_audit_llm,
    )

    # ONE FakeLLM handles CriticAgent's single direction-critique call:
    # canned pass verdict.
    critic_llm = CriticFakeLLM([_direction_report_json()])
    build_critic_service(
        client=BusClientView(bus, "CriticAgent"),
        spec_store=store,
        llm_client=critic_llm,
    )

    # phase 3.1: MethodAgent derives the explicit method after the direction passes.
    from autoresearch.agents.method_agent.service import build_method_service
    build_method_service(
        client=BusClientView(bus, "MethodAgent"), spec_store=store,
        llm_client=_MethodFakeLLM([json.dumps({"approach": "a", "changes_outline": ["c"],
                                               "why_it_should_win": "w", "assumptions": []})]),
    )

    # --- 1. start_session -> problem_modeling.requested -> (delivered) -----
    # -> agent asks its first clarification question, all synchronously
    # inside this one call (genuine subscribe/dispatch routing, not manual).
    sid = await eng.start_session("optimize modded-nanogpt speedrun", user_id="u1")

    questions = [
        e for e in bus.published if e.event_type == "problem_modeling.clarification_question"
    ]
    assert len(questions) == 1
    question_id = questions[0].payload["question_id"]

    state = await load_session_state(bus.db, sid)
    assert state is not None
    assert state.current_state == "ClarifyingResearchGoal"

    # --- 2. simulate the human answering -> (delivered) -> agent finalizes -
    # -> research_spec.ready -> ... -> ModelingProblem -> problem.modeled ->
    # (delivered) -> orchestrator transitions ModelingProblem ->
    # HumanIdeaCheckpoint (setting human_checkpoint_return_to =
    # "CritiquingDirection"). HumanIdeaCheckpoint has no entry action, so the
    # loop pauses there awaiting the human. See test_m4_novelty_audit_smoke.py
    # for the detailed play-by-play of this cascade.
    human = BusClientView(bus, "Human")
    await human.publish(
        event_type="human.clarification_answer",
        session_id=sid,
        payload={"question_id": question_id, "answer_text": "benchmark report"},
    )

    checkpoint_state = await load_session_state(bus.db, sid)
    assert checkpoint_state is not None
    assert checkpoint_state.current_state == "HumanIdeaCheckpoint"
    assert checkpoint_state.human_checkpoint_return_to == "SelectingBenchmark"
    assert checkpoint_state.problem_model_ref is not None

    # --- 3. simulate the human at HumanIdeaCheckpoint -> publish -----------
    # human.idea_provided {inject: false} -> (delivered) -> orchestrator's
    # _on_HumanIdeaCheckpoint_human_idea_provided routes to
    # state.human_checkpoint_return_to ("CritiquingDirection"), transitions
    # HumanIdeaCheckpoint -> CritiquingDirection, and emits its entry action
    # direction.critique_requested -> (delivered) -> CriticAgent resolves the
    # ProblemModel + ResearchSpec from the store, runs its (canned pass)
    # critique_direction, persists the CritiqueReport, and publishes
    # direction.critiqued(pass) -> (delivered) -> orchestrator resets counter
    # A to 0, transitions CritiquingDirection -> RoutingResearchMode (no entry
    # action), and immediately auto-advances (M7 Task 4:
    # ``_route_research_mode``) RoutingResearchMode -> SelectingBenchmark
    # (mode defaults to "benchmark"), whose entry action
    # (benchmark.selection_requested) has no consumer wired in this smoke, so
    # the loop pauses there.
    await human.publish(
        event_type="human.idea_provided",
        session_id=sid,
        payload={"inject": False},
    )

    # No benchmark selector is wired here; Method must wait for a frozen baseline.
    method_state = await load_session_state(bus.db, sid)
    assert method_state.current_state == "SelectingBenchmark"
    assert method_state.method_ref is None

    # --- Assertions -----------------------------------------------------
    final_state = await load_session_state(bus.db, sid)
    assert final_state is not None
    assert final_state.current_state == "SelectingBenchmark"
    assert final_state.direction_critique_round == 0

    # problem_model_ref still resolves to a schema-valid ProblemModel.
    problem_model = store.get(final_state.problem_model_ref)
    assert problem_model == VALID_PROBLEM_MODEL
    problem_model_errors = list(_validator(PROBLEM_MODEL_SCHEMA_PATH).iter_errors(problem_model))
    assert problem_model_errors == [], [e.message for e in problem_model_errors]

    # meeting item 7: the machine no longer emits a direction critique after the human confirmed.
    assert [e for e in bus.published if e.event_type == "direction.critiqued"] == []

    # No agent.failed anywhere along the way.
    assert not [e for e in bus.published if e.event_type == "agent.failed"]


# ===========================================================================
# Smoke 2: isolated GeneratingHypotheses -> HypothesisAgent ->
# CritiquingHypotheses -> CriticAgent (some_pass) ->
# ManagingHypothesisPortfolio.
# ===========================================================================

VALID_RESEARCH_SPEC = {
    "topic": "Retrieval-augmented generation for long-tail scientific QA",
    "research_goal": "Propose a new RAG method for long-tail scientific question answering.",
}

VALID_LITERATURE_BRIEF = {
    "summary": "The field combines dense retrieval with generative models to ground answers.",
    "related_work": [],
    "key_papers": [
        {
            "title": "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
            "id_or_url": "arxiv:2005.11401",
            "year": 2020,
        },
    ],
    "existing_benchmarks": [],
}


def _hypothesis(hid: str, kind: str) -> dict:
    return {
        "id": hid,
        "statement": f"Statement for {hid}: doing X improves Y by Z%.",
        "rationale": f"Rationale for {hid}, grounded in the literature brief.",
        "kind": kind,
        "verifiable": True,
        "expected_effect": "Metric improves by an estimated 5-10%.",
    }


# Fixed ids ("h1"/"h2") shared between HypothesisAgent's canned bank and
# CriticAgent's canned passing_hypothesis_ids so critique_hypotheses's
# grounding check (passing ids subset of bank ids) passes.
VALID_HYPOTHESES = [
    _hypothesis("h1", "idea"),
    _hypothesis("h2", "training"),
]

VALID_HYPOTHESES_REPORT_SOME_PASS = {
    "target": "hypotheses",
    "verdict": "some_pass",
    "issues": ["h2 is too narrow / trivially confounded."],
    "passing_hypothesis_ids": ["h1"],
}

# RankingJudgeAgent's canned pairwise-judgment response for the two-hypothesis
# bank above (h1 beats h2); rank_candidates computes btl_scores/ranked_shortlist
# deterministically from this.
VALID_RANKING_PAIRWISE_RESPONSE = {
    "pairwise_results": [
        {"a": "h1", "b": "h2", "winner": "h1", "rationale": "h1 is more falsifiable."},
    ],
    "diversity_status": "diverse",
}


def _bank_json(hypotheses: list[dict]) -> str:
    return json.dumps({"hypotheses": hypotheses})


def _hypotheses_report_json() -> str:
    return json.dumps(VALID_HYPOTHESES_REPORT_SOME_PASS)


def _ranking_pairwise_json() -> str:
    return json.dumps(VALID_RANKING_PAIRWISE_RESPONSE)


async def test_hypothesis_critique_loop_reaches_managing_portfolio_single_session(bus, store):
    # --- wire OrchestratorEngine (WITH spec_store -- ManagingHypothesisPortfolio's
    # entry action / ranking-driven auto-advance, RankingJudgeAgent milestone
    # Task 5, reads the RankingReport independently of the event payload) +
    # HypothesisAgent + CriticAgent -- all on the SAME bus, each with its own
    # emitter identity.
    eng = OrchestratorEngine(
        client=BusClientView(bus, "ResearchOrchestrator"), skill_root=SM_ROOT, spec_store=store
    )
    await eng.start()

    hypothesis_llm = HypothesisFakeLLM([_bank_json(VALID_HYPOTHESES)])
    build_hypothesis_service(
        client=BusClientView(bus, "HypothesisAgent"),
        spec_store=store,
        llm_client=hypothesis_llm,
    )

    # RankingJudgeAgent: consumes CritiquingHypotheses' entry action
    # (ranking.requested), needed to even reach CriticAgent now that the
    # rank -> rank-informed critique co-audit is wired (RankingJudgeAgent
    # milestone Task 4).
    ranking_llm = RankingJudgeFakeLLM([_ranking_pairwise_json()])
    build_ranking_judge_service(
        client=BusClientView(bus, "RankingJudgeAgent"),
        spec_store=store,
        llm_client=ranking_llm,
    )

    # ONE FakeLLM handles CriticAgent's single hypotheses-critique call:
    # canned some_pass verdict whose passing_hypothesis_ids ("h1") is a
    # subset of the HypothesisAgent-generated bank's ids ("h1"/"h2").
    critic_llm = CriticFakeLLM([_hypotheses_report_json()])
    build_critic_service(
        client=BusClientView(bus, "CriticAgent"),
        spec_store=store,
        llm_client=critic_llm,
    )

    # --- Isolated construction: GeneratingHypotheses is NOT reachable from
    # start_session yet (intermediate states unbuilt), so seed its two
    # upstream inputs directly into the store, as ProblemModelingAgent /
    # DeepResearchAgent would have, then construct + save a SessionState
    # directly in GeneratingHypotheses with both refs set.
    sid = str(uuid.uuid4())
    research_spec_ref = store.put(
        kind="research_spec",
        session_id=sid,
        key="final",
        obj=VALID_RESEARCH_SPEC,
        emitter="ProblemModelingAgent",
    )
    literature_brief_ref = store.put(
        kind="literature_brief",
        session_id=sid,
        key="final",
        obj=VALID_LITERATURE_BRIEF,
        emitter="DeepResearchAgent",
    )
    state = SessionState(
        session_id=sid,
        user_id="u1",
        current_state="GeneratingHypotheses",
        research_spec_ref=research_spec_ref,
        literature_brief_ref=literature_brief_ref,
    )
    await save_session_state(bus.db, state, auto_commit=True)

    # --- Drive it: resume_to_current_state re-emits the GeneratingHypotheses
    # entry action (hypothesis.generation_requested) -> (delivered) ->
    # HypothesisAgent resolves both inputs from the store, runs its (canned)
    # generate_step, persists a HypothesisBank, and publishes
    # hypothesis.generated -> (delivered) -> orchestrator's tick ->
    # _on_GeneratingHypotheses_hypothesis_generated transitions
    # GeneratingHypotheses -> CritiquingHypotheses, whose entry action
    # (ranking.requested) -> (delivered) -> RankingJudgeAgent resolves the
    # HypothesisBank from the store, runs its (canned) rank_candidates,
    # persists the RankingReport, and publishes ranking.ready -> (delivered)
    # -> orchestrator's _on_CritiquingHypotheses_ranking_ready stores
    # hypothesis_ranking_report_ref and publishes (IN-STATE)
    # hypothesis.critique_requested (now carrying ranking_report_ref) ->
    # (delivered) -> CriticAgent resolves the HypothesisBank + ResearchSpec
    # (+ RankingReport) from the store, runs its (canned some_pass)
    # critique_hypotheses, persists the CritiqueReport, and publishes
    # hypothesis.critiqued(some_pass) -> (delivered) -> orchestrator resets
    # counter C to 0, stores passing_hypothesis_ids, and transitions
    # CritiquingHypotheses -> ManagingHypothesisPortfolio, whose entry action
    # (ranking.requested, candidate_ids=["h1"]) -> (delivered) ->
    # RankingJudgeAgent takes the 0/1-candidate trivial short-circuit (no LLM
    # call -- the queued FakeLLM response is untouched by this second call),
    # persists a second RankingReport, and publishes ranking.ready ->
    # (delivered) -> orchestrator's _on_ManagingHypothesisPortfolio_ranking_ready
    # (RankingJudgeAgent milestone Task 5) selects the top-ranked hypothesis
    # ("h1"), resets counter L to 0, and transitions ManagingHypothesisPortfolio
    # -> PlanningExperiment, whose entry action (experiment.planning_requested)
    # has no consumer wired in this smoke, so the loop pauses there. Called
    # exactly once.
    result = await eng.resume_to_current_state(sid)
    assert result["resumed"] is True

    # --- Assertions -------------------------------------------------------
    final_state = await load_session_state(bus.db, sid)
    assert final_state is not None
    assert final_state.current_state == "PlanningExperiment"
    assert final_state.passing_hypothesis_ids == ["h1"]
    assert final_state.current_hypothesis_id == "h1"
    assert final_state.current_experiment_id == "h1"
    assert final_state.hypothesis_reject_round == 0
    assert final_state.search_diversity_collapse_round == 0
    assert final_state.hypothesis_bank_ref is not None
    assert final_state.hypothesis_ranking_report_ref is not None

    # hypothesis_bank_ref resolves in the Spec Store to a schema-valid
    # HypothesisBank.
    bank = store.get(final_state.hypothesis_bank_ref)
    assert bank == {"hypotheses": VALID_HYPOTHESES}
    bank_errors = list(_validator(HYPOTHESIS_BANK_SCHEMA_PATH).iter_errors(bank))
    assert bank_errors == [], [e.message for e in bank_errors]
    bank_ids = {h["id"] for h in bank["hypotheses"]}

    # Exactly one hypothesis.critiqued(some_pass); its passing_hypothesis_ids
    # is a non-empty subset of the bank's ids (grounded).
    critiqued_events = [e for e in bus.published if e.event_type == "hypothesis.critiqued"]
    assert len(critiqued_events) == 1
    assert critiqued_events[0].session_id == sid
    assert critiqued_events[0].payload["verdict"] == "some_pass"
    passing_ids = critiqued_events[0].payload["passing_hypothesis_ids"]
    assert passing_ids
    assert set(passing_ids) <= bank_ids

    # The ref appended to SessionState.critique_report_refs resolves to a
    # schema-valid CritiqueReport (target=hypotheses).
    critique_report_ref = critiqued_events[0].payload["critique_report_ref"]
    assert critique_report_ref in final_state.critique_report_refs

    critique_report = store.get(critique_report_ref)
    assert critique_report["target"] == "hypotheses"
    assert critique_report["verdict"] == "some_pass"
    critique_report_errors = list(_validator(CRITIQUE_REPORT_SCHEMA_PATH).iter_errors(critique_report))
    assert critique_report_errors == [], [e.message for e in critique_report_errors]

    # No agent.failed anywhere along the way.
    assert not [e for e in bus.published if e.event_type == "agent.failed"]
