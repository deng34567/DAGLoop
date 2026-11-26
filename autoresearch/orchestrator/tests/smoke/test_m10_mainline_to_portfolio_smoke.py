"""M10 Task 5 integration smoke: THE mainline-connection smoke, single
session, no Redis, no real LLM, no real execution.

Extends ``test_m7_verifier_smoke.py``'s
``test_mainline_reaches_reproducing_baseline_single_session`` by wiring
RunnerAgent (``build_runner_service(..., backend=FakeRunnerBackend(...))``),
HypothesisAgent, and RankingJudgeAgent alongside the same seven mainline
components (OrchestratorEngine, ProblemModelingAgent, DeepResearchAgent,
EvidenceAuditAgent, NoveltyAuditAgent, CriticAgent, EvaluatorAgent) plus a
second, IN-ORDER
CriticAgent response, so the ``baseline.reproduction_requested`` event
emitted at ReproducingBaseline is actually consumed -- RunnerAgent reproduces
the baseline, connecting ReproducingBaseline -> GeneratingHypotheses ->
CritiquingHypotheses (rank -> rank-informed critique co-audit, RankingJudgeAgent
milestone Task 4) -> ManagingHypothesisPortfolio (ranking-driven auto-advance,
RankingJudgeAgent milestone Task 5) -> PlanningExperiment, where the loop
pauses (no ExperimentPlannerAgent wired in this smoke).

THE HEADLINE: this is the first single-session drive of the FULL connected
mainline, from ``start_session`` all the way to ``PlanningExperiment``
-- proving the mainline (spec -> literature -> novelty -> problem model ->
direction critique -> benchmark -> verifier -> baseline) and the M5/M6
hypothesis loop (generate -> critique -> portfolio -> top-ranked selection)
are now ONE connected flow, not two disjoint islands.

The ``DeliveringBus`` broker is copied from ``test_m7_verifier_smoke.py``
(itself copied from ``test_m6_critic_smoke.py``) to keep this file
self-contained.

No concurrency, no Redis, no real LLM call, no real search backend and no
real execution backend anywhere in this file (every LLM is a canned FakeLLM
queue; FakeBackend canned papers; FakeRunnerBackend canned metric; human
answers/idea-checkpoint response are simulated, not a real UI/human).
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
from autoresearch.agents._reuse.runner_backend import FakeRunnerBackend
from autoresearch.agents.critic_agent.service import build_critic_service
from autoresearch.orchestrator.tests.smoke._method_smoke_helper import wire_method_agent
from autoresearch.agents.critic_agent.tests.conftest import FakeLLM as CriticFakeLLM
from autoresearch.agents.deep_research_agent.service import build_deep_research_service
from autoresearch.agents.deliberation_agent.service import build_deliberation_service
from autoresearch.agents.deliberation_agent.tests.conftest import FakeLLM as DeliberationFakeLLM
from autoresearch.agents.deep_research_agent.tests.conftest import FakeLLM as DeepResearchFakeLLM
from autoresearch.agents.evaluator_agent.service import build_evaluator_service
from autoresearch.agents.evaluator_agent.tests.conftest import FakeLLM as EvaluatorFakeLLM
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
from autoresearch.agents.runner_agent.service import build_runner_service
from autoresearch.eventbus.broker.envelope import Envelope
from autoresearch.orchestrator.engine import OrchestratorEngine
from autoresearch.orchestrator.session_state import load_session_state
from autoresearch.spec_store import SpecStore

SM_ROOT = Path(__file__).resolve().parents[2] / "state-machine"
EVALUATOR_CONTRACTS_DIR = (
    Path(__file__).resolve().parents[3] / "agents" / "evaluator_agent" / "data-contracts"
)
BENCHMARK_SPEC_SCHEMA_PATH = EVALUATOR_CONTRACTS_DIR / "benchmark_spec.schema.yaml"
VERIFIER_SPEC_SCHEMA_PATH = EVALUATOR_CONTRACTS_DIR / "verifier_spec.schema.yaml"
RUNNER_CONTRACTS_DIR = Path(__file__).resolve().parents[3] / "agents" / "runner_agent" / "data-contracts"
BASELINE_RESULT_SCHEMA_PATH = RUNNER_CONTRACTS_DIR / "baseline_result.schema.yaml"
HYPOTHESIS_CONTRACTS_DIR = (
    Path(__file__).resolve().parents[3] / "agents" / "hypothesis_agent" / "data-contracts"
)
HYPOTHESIS_BANK_SCHEMA_PATH = HYPOTHESIS_CONTRACTS_DIR / "hypothesis_bank.schema.yaml"


def _validator(path: Path) -> Draft202012Validator:
    schema = yaml.safe_load(path.read_text(encoding="utf-8"))
    return Draft202012Validator(schema)


# ---------------------------------------------------------------------------
# DeliveringBus: minimal in-memory broker that actually routes.
# Copied from test_m7_verifier_smoke.py to keep this file self-contained.
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
        event_id: Optional[str] = None,
    ) -> str:
        env = Envelope(
            event_id=event_id or str(uuid.uuid4()),
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
        event_id: Optional[str] = None,
    ) -> str:
        return await self._bus.publish(
            event_type=event_type,
            session_id=session_id,
            payload=payload,
            emitter=self.emitter_name,
            cost_delta=cost_delta,
            causation_id=causation_id,
            auto_commit=auto_commit,
            event_id=event_id,
        )


@pytest.fixture
async def bus(tmp_path):
    conn = await aiosqlite.connect(str(tmp_path / "m10_mainline_smoke.db"))
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
# Smoke: start_session -> ... -> ReproducingBaseline -> RunnerAgent
# (reproduced) -> GeneratingHypotheses -> HypothesisAgent -> hypothesis.
# generated -> CritiquingHypotheses -> RankingJudgeAgent (ranking.ready) ->
# CriticAgent (some_pass) -> hypothesis.critiqued ->
# ManagingHypothesisPortfolio (pauses; RankingJudgeAgent milestone Task 5
# wires the ranking-driven auto-advance into PlanningExperiment).
# ===========================================================================

FINAL_SPEC = {
    "topic": "modded-nanogpt speedrun optimization",
    "research_goal": (
        "Reduce modded-nanogpt training wall-clock time to a new speedrun "
        "record while preserving the target validation loss."
    ),
    "task_mode": "benchmark",
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

# EvaluatorAgent's first canned response: a schema-valid BenchmarkSpec whose
# forbidden_files is DISJOINT from the VerifierSpec's editable_region below,
# and whose baseline.reported_metric (0.85) matches FakeRunnerBackend's canned
# metric_value (also 0.85) within RunnerAgent's default tolerance (0.02).
VALID_BENCHMARK_SPEC = {
    "benchmark_id": "modded-nanogpt-speedrun-v1",
    "name": "modded-nanogpt speedrun",
    "metric": "wall_clock_seconds",
    "run_command": "python run_benchmark.py --model candidate",
    "baseline": {
        "command": "python run_benchmark.py --model baseline",
        "reported_metric": 0.85,
    },
    "resource_requirements": "8x A100 GPU",
    "forbidden_files": ["eval/scorer.py", "data/test/"],
}

# EvaluatorAgent's second canned response: a schema-valid VerifierSpec (ok)
# whose editable_region contains no eval/evaluat/verif/harness/grade/metric
# tokens and is disjoint from forbidden_files -- passes both core guards.
VALID_VERIFIER_SPEC = {
    "evaluator_hash": "sha256:def456",
    "metric": "wall_clock_seconds",
    "data_split": {"in_loop": "dev", "held_out": "test"},
    "run_budget": "10 runs, 1h wall-clock",
    "seed_policy": "fixed seed=42 across all runs",
    "editable_region": ["src/model.py", "src/train.py"],
    "forbidden_files": ["eval/scorer.py", "data/test/"],
    "result_parse_schema": {"type": "object"},
}

# HypothesisAgent's canned response: a schema-valid HypothesisBank with TWO
# hypotheses (ids "h1"/"h2"); CriticAgent's canned hypotheses-critique below
# passes only "h1" -- ids are a SUBSET of the bank's, satisfying the
# anti-hallucination grounding guard in critic_agent/core.py.
VALID_HYPOTHESIS_BANK = {
    "hypotheses": [
        {
            "id": "h1",
            "statement": "Fusing the optimizer's momentum buffers reduces wall-clock time.",
            "rationale": "Momentum buffer fusion cuts kernel launch overhead on this hardware.",
            "kind": "code_optimization",
            "verifiable": True,
            "expected_effect": "wall_clock_seconds improves by an estimated 5-10%.",
        },
        {
            "id": "h2",
            "statement": "Switching to a larger batch size alone reduces wall-clock time.",
            "rationale": "Larger batches amortize fixed per-step overhead.",
            "kind": "training",
            "verifiable": True,
            "expected_effect": "wall_clock_seconds improves by an estimated 2-5%.",
        },
    ]
}

# CriticAgent's second canned response: a schema-valid CritiqueReport
# (target=hypotheses, verdict=some_pass) citing only "h1", a real id in the
# bank above.
VALID_HYPOTHESES_REPORT_SOME_PASS = {
    "target": "hypotheses",
    "verdict": "some_pass",
    "issues": ["h2 is too narrow / trivially confounded with batch-size-only changes."],
    "passing_hypothesis_ids": ["h1"],
}

# RankingJudgeAgent's canned pairwise-judgment response for the two-hypothesis
# bank above (h1 beats h2); needed to even reach CriticAgent now that
# CritiquingHypotheses' entry action is ranking.requested (RankingJudgeAgent
# milestone Task 4).
VALID_RANKING_PAIRWISE_RESPONSE = {
    "pairwise_results": [
        {"a": "h1", "b": "h2", "winner": "h1", "rationale": "h1 is more falsifiable."},
    ],
    "diversity_status": "diverse",
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


def _hypotheses_report_json() -> str:
    return json.dumps(VALID_HYPOTHESES_REPORT_SOME_PASS)


def _benchmark_spec_json() -> str:
    return json.dumps(VALID_BENCHMARK_SPEC)


def _verifier_spec_json() -> str:
    return json.dumps({"verifier_spec": VALID_VERIFIER_SPEC, "verdict": "ok"})


def _hypothesis_bank_json() -> str:
    return json.dumps(VALID_HYPOTHESIS_BANK)


def _ranking_pairwise_json() -> str:
    return json.dumps(VALID_RANKING_PAIRWISE_RESPONSE)


async def test_mainline_reaches_managing_hypothesis_portfolio_single_session(bus, store):
    # --- wire OrchestratorEngine (WITH spec_store -- ManagingHypothesisPortfolio's
    # entry action / ranking-driven auto-advance, RankingJudgeAgent milestone
    # Task 5, reads the RankingReport independently of the event payload) +
    # ProblemModelingAgent + DeepResearchAgent + EvidenceAuditAgent +
    # NoveltyAuditAgent + CriticAgent + EvaluatorAgent + RunnerAgent +
    # HypothesisAgent -- all on the SAME bus, each with its own emitter
    # identity.
    eng = OrchestratorEngine(
        client=BusClientView(bus, "ResearchOrchestrator"), skill_root=SM_ROOT, spec_store=store
    )
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

    # ONE FakeLLM handles CriticAgent's TWO calls IN ORDER: direction critique
    # (CritiquingDirection, hit first) then hypotheses critique
    # (CritiquingHypotheses, hit later, after the baseline is reproduced).
    critic_llm = CriticFakeLLM([_direction_report_json(), _hypotheses_report_json()])
    build_critic_service(
        client=BusClientView(bus, "CriticAgent"),
        spec_store=store,
        llm_client=critic_llm,
    )
    # phase 3.1: wire the MethodAgent (GeneratingMethod consumer, canned method).
    wire_method_agent(bus, store, BusClientView)

    # RankingJudgeAgent: consumes CritiquingHypotheses' entry action
    # (ranking.requested), needed to even reach CriticAgent's hypotheses
    # critique now that the rank -> rank-informed critique co-audit is wired
    # (RankingJudgeAgent milestone Task 4).
    ranking_llm = RankingJudgeFakeLLM([_ranking_pairwise_json()])
    build_ranking_judge_service(
        client=BusClientView(bus, "RankingJudgeAgent"),
        spec_store=store,
        llm_client=ranking_llm,
    )

    # ONE FakeLLM handles EvaluatorAgent's TWO calls in order:
    # select_benchmark, then prepare_verifier (verdict "ok").
    evaluator_llm = EvaluatorFakeLLM([_benchmark_spec_json(), _verifier_spec_json()])
    build_evaluator_service(
        client=BusClientView(bus, "EvaluatorAgent"),
        spec_store=store,
        llm_client=evaluator_llm,
    )

    # RunnerAgent: FakeRunnerBackend's canned metric_value (0.85) matches
    # VALID_BENCHMARK_SPEC's baseline.reported_metric (0.85) within the
    # default tolerance (0.02) -- reproduced == True. NEVER a real backend.
    build_runner_service(
        client=BusClientView(bus, "RunnerAgent"),
        spec_store=store,
        backend=FakeRunnerBackend(metric_value=0.85, exit_code=0),
    )

    # HypothesisAgent: canned HypothesisBank with ids h1/h2, "h1" of which is
    # the id CriticAgent's hypotheses-critique passes.
    hypothesis_llm = HypothesisFakeLLM([_hypothesis_bank_json()])
    build_hypothesis_service(
        client=BusClientView(bus, "HypothesisAgent"),
        spec_store=store,
        llm_client=hypothesis_llm,
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
    # ProblemModel + ResearchSpec from the store, runs its (canned pass, 1st
    # queued response) critique_direction, persists the CritiqueReport, and
    # publishes direction.critiqued(pass) -> (delivered) -> orchestrator
    # resets counter A to 0, transitions CritiquingDirection ->
    # RoutingResearchMode (no entry action), and immediately auto-advances (M7
    # Task 4) RoutingResearchMode -> SelectingBenchmark (mode defaults to
    # "benchmark"), whose entry action (benchmark.selection_requested) ->
    # (delivered) -> EvaluatorAgent resolves the ResearchSpec + LiteratureBrief
    # from the store, runs its (canned) select_benchmark, persists the
    # BenchmarkSpec, and publishes benchmark.spec_ready -> (delivered) ->
    # orchestrator stores benchmark_spec_ref + current_benchmark_id and
    # transitions SelectingBenchmark -> PreparingVerifier, whose entry action
    # (verifier.preparation_requested) -> (delivered) -> EvaluatorAgent
    # resolves the BenchmarkSpec + ResearchSpec from the store, runs its
    # (canned, verdict "ok") prepare_verifier, persists the VerifierSpec, and
    # publishes verifier.spec_frozen(ok) -> (delivered) -> orchestrator
    # stores verifier_spec_ref and transitions PreparingVerifier ->
    # ReproducingBaseline, whose entry action (baseline.reproduction_requested,
    # M10 Task 1) -> (delivered) -> RunnerAgent resolves the BenchmarkSpec +
    # VerifierSpec from the store, runs FakeRunnerBackend (canned
    # metric_value=0.85, matching baseline.reported_metric within tolerance),
    # persists the BaselineResult (reproduced=True), and publishes
    # baseline.reproduced(reproduced) -> (delivered) -> orchestrator's
    # CONNECTOR handler resets counter B[benchmark_id] (delete-key) and
    # transitions ReproducingBaseline -> GeneratingHypotheses, whose EXISTING
    # (M5) entry action (hypothesis.generation_requested) -> (delivered) ->
    # HypothesisAgent resolves the ResearchSpec + LiteratureBrief from the
    # store, runs its (canned) generate, persists the HypothesisBank (ids
    # h1/h2), and publishes hypothesis.generated -> (delivered) ->
    # orchestrator stores hypothesis_bank_ref and transitions
    # GeneratingHypotheses -> CritiquingHypotheses, whose entry action
    # (ranking.requested, RankingJudgeAgent milestone Task 4) -> (delivered)
    # -> RankingJudgeAgent resolves the HypothesisBank from the store, runs
    # its (canned) rank_candidates, persists the RankingReport, and publishes
    # ranking.ready -> (delivered) -> orchestrator's
    # _on_CritiquingHypotheses_ranking_ready stores hypothesis_ranking_
    # report_ref and publishes (IN-STATE) hypothesis.critique_requested (now
    # carrying ranking_report_ref) -> (delivered) -> CriticAgent resolves the
    # HypothesisBank + ResearchSpec (+ RankingReport) from the store, runs its
    # (canned, 2nd queued response, verdict "some_pass", passing_hypothesis_
    # ids=["h1"]) critique_hypotheses, persists the CritiqueReport, and
    # publishes hypothesis.critiqued(some_pass) -> (delivered) -> orchestrator
    # resets counter C to 0, stores passing_hypothesis_ids, and transitions
    # CritiquingHypotheses -> ManagingHypothesisPortfolio, whose entry action
    # (ranking.requested, candidate_ids=["h1"]) -> (delivered) ->
    # RankingJudgeAgent takes the 0/1-candidate trivial short-circuit (no LLM
    # call -- the queued FakeLLM response is untouched by this second call),
    # persists a second RankingReport, and publishes ranking.ready ->
    # (delivered) -> orchestrator's _on_ManagingHypothesisPortfolio_ranking_ready
    # (RankingJudgeAgent milestone Task 5) selects the top-ranked hypothesis
    # ("h1"), resets counter L to 0, and transitions ManagingHypothesisPortfolio
    # -> PlanningExperiment, whose entry action (experiment.planning_requested)
    # has no consumer wired in this smoke, so the loop pauses there -- THE
    # MAINLINE IS NOW CONNECTED TO THE HYPOTHESIS LOOP, end to end, in one
    # session.
    await human.publish(
        event_type="human.idea_provided",
        session_id=sid,
        payload={"inject": False},
    )
    # phase 3.1: pass -> GeneratingMethod -> ConfirmingMethod (pauses); confirm the method to proceed.
    await human.publish(event_type="method.confirmed", session_id=sid, payload={"inject": False})

    # --- Assertions ---------------------------------------------------------
    final_state = await load_session_state(bus.db, sid)
    assert final_state is not None
    # meeting 2026-07-25 甲: the benchmark mainline now goes ReproducingBaseline -> PlanningExperiment
    # DIRECTLY (plan how to implement the confirmed method); the GeneratingHypotheses ->
    # CritiquingHypotheses -> ManagingHypothesisPortfolio outer loop is retired on the mainline
    # (it remains for lab_in_the_loop mode). So there are no confirmed hypotheses / ranking, and
    # current_experiment_id is anchored from the confirmed method rather than a hypothesis id.
    assert final_state.current_state == "PlanningExperiment"
    assert final_state.current_experiment_id is not None
    assert final_state.direction_critique_round == 0
    assert final_state.search_diversity_collapse_round == 0

    # benchmark_id key is absent from baseline_reproduce_round (MAP counter B
    # reset = delete-key on the "reproduced" branch).
    assert VALID_BENCHMARK_SPEC["benchmark_id"] not in final_state.baseline_reproduce_round

    # baseline_result_ref is set and resolves to a schema-valid BaselineResult
    # with reproduced == True.
    assert final_state.baseline_result_ref is not None
    baseline_result = store.get(final_state.baseline_result_ref)
    baseline_result_errors = list(_validator(BASELINE_RESULT_SCHEMA_PATH).iter_errors(baseline_result))
    assert baseline_result_errors == [], [e.message for e in baseline_result_errors]
    assert baseline_result["reproduced"] is True
    assert baseline_result["benchmark_id"] == VALID_BENCHMARK_SPEC["benchmark_id"]

    # 甲: no hypothesis bank / hypothesis events on the benchmark mainline (that outer loop is
    # retired). Exactly one baseline.reproduced(reproduced) still fired, and the mainline advanced
    # to PlanningExperiment (asserted above) instead of the hypothesis loop.
    reproduced_events = [e for e in bus.published if e.event_type == "baseline.reproduced"]
    assert len(reproduced_events) == 1
    assert reproduced_events[0].session_id == sid
    assert reproduced_events[0].payload["verdict"] == "reproduced"
    assert reproduced_events[0].payload["baseline_result_ref"] == final_state.baseline_result_ref
    assert [e for e in bus.published if e.event_type == "hypothesis.generated"] == []

    # No agent.failed anywhere along the way.
    assert not [e for e in bus.published if e.event_type == "agent.failed"]
