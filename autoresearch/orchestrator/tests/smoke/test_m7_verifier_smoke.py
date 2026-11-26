"""M7 Task 5 integration smoke: single session, no Redis, no real LLM.

Extends ``test_m6_critic_smoke.py``'s
``test_mainline_reaches_selecting_track_single_session`` by wiring
EvaluatorAgent alongside the same six mainline components
(OrchestratorEngine, ProblemModelingAgent, DeepResearchAgent,
EvidenceAuditAgent, NoveltyAuditAgent, CriticAgent) so the
``benchmark.selection_requested`` and
``verifier.preparation_requested`` events emitted at SelectingBenchmark and
PreparingVerifier are actually consumed. EvaluatorAgent's canned
BenchmarkSpec + (ok) VerifierSpec route SelectingBenchmark ->
PreparingVerifier -> ReproducingBaseline, whose lack of a consumer (M10
RunnerAgent) pauses the loop there -- the mainline now reaches
ReproducingBaseline with a frozen VerifierSpec on the books BEFORE any
candidate solution exists: the verifier-first invariant, exercised
end-to-end.

The ``DeliveringBus`` broker is copied from ``test_m6_critic_smoke.py`` (itself
copied from ``test_mainline_to_critique_smoke.py``) to keep this file
self-contained.

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
from autoresearch.agents.novelty_audit_agent.service import build_novelty_audit_service
from autoresearch.agents.novelty_audit_agent.tests.conftest import FakeLLM as NoveltyAuditFakeLLM
from autoresearch.agents.problem_modeling_agent.service import build_problem_modeling_service
from autoresearch.agents.problem_modeling_agent.tests.conftest import FakeLLM as ProblemModelingFakeLLM
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


def _validator(path: Path) -> Draft202012Validator:
    schema = yaml.safe_load(path.read_text(encoding="utf-8"))
    return Draft202012Validator(schema)


# ---------------------------------------------------------------------------
# DeliveringBus: minimal in-memory broker that actually routes.
# Copied from test_m6_critic_smoke.py to keep this file self-contained.
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
    conn = await aiosqlite.connect(str(tmp_path / "m7_verifier_smoke.db"))
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
# Smoke: mainline start_session -> ... -> CritiquingDirection -> CriticAgent
# (pass) -> RoutingResearchMode -> (auto-advance) -> SelectingBenchmark ->
# EvaluatorAgent (benchmark) -> benchmark.spec_ready -> PreparingVerifier ->
# EvaluatorAgent (verifier, ok) -> verifier.spec_frozen -> ReproducingBaseline.
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

# EvaluatorAgent's first canned response: a schema-valid BenchmarkSpec whose
# forbidden_files is DISJOINT from the VerifierSpec's editable_region below.
VALID_BENCHMARK_SPEC = {
    "benchmark_id": "modded-nanogpt-speedrun-v1",
    "name": "modded-nanogpt speedrun",
    "metric": "wall_clock_seconds",
    "run_command": "python run_benchmark.py --model candidate",
    "baseline": {
        "command": "python run_benchmark.py --model baseline",
        "reported_metric": 1800.0,
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


def _benchmark_spec_json() -> str:
    return json.dumps(VALID_BENCHMARK_SPEC)


def _verifier_spec_json() -> str:
    return json.dumps({"verifier_spec": VALID_VERIFIER_SPEC, "verdict": "ok"})


async def test_mainline_reaches_reproducing_baseline_single_session(bus, store):
    # --- wire OrchestratorEngine + ProblemModelingAgent + DeepResearchAgent +
    # EvidenceAuditAgent + NoveltyAuditAgent + CriticAgent + EvaluatorAgent --
    # all on the SAME bus, each with its own emitter identity.
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
    # phase 3.1: wire the MethodAgent (GeneratingMethod consumer, canned method).
    wire_method_agent(bus, store, BusClientView)

    # ONE FakeLLM handles EvaluatorAgent's TWO calls in order:
    # select_benchmark, then prepare_verifier (verdict "ok").
    evaluator_llm = EvaluatorFakeLLM([_benchmark_spec_json(), _verifier_spec_json()])
    build_evaluator_service(
        client=BusClientView(bus, "EvaluatorAgent"),
        spec_store=store,
        llm_client=evaluator_llm,
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
    # action), and immediately auto-advances (M7 Task 4) RoutingResearchMode ->
    # SelectingBenchmark (mode defaults to "benchmark"), whose entry action
    # (benchmark.selection_requested) -> (delivered) -> EvaluatorAgent
    # resolves the ResearchSpec + LiteratureBrief from the store, runs its
    # (canned) select_benchmark, persists the BenchmarkSpec, and publishes
    # benchmark.spec_ready -> (delivered) -> orchestrator stores
    # benchmark_spec_ref + current_benchmark_id and transitions
    # SelectingBenchmark -> PreparingVerifier, whose entry action
    # (verifier.preparation_requested) -> (delivered) -> EvaluatorAgent
    # resolves the BenchmarkSpec + ResearchSpec from the store, runs its
    # (canned, verdict "ok") prepare_verifier, persists the VerifierSpec, and
    # publishes verifier.spec_frozen(ok) -> (delivered) -> orchestrator
    # stores verifier_spec_ref and transitions PreparingVerifier ->
    # ReproducingBaseline, which has no entry action (and no consumer until
    # M10 RunnerAgent), so the loop pauses there -- the mainline has now
    # reached ReproducingBaseline with a frozen VerifierSpec on the books
    # BEFORE any candidate solution exists.
    await human.publish(
        event_type="human.idea_provided",
        session_id=sid,
        payload={"inject": False},
    )
    # No Runner is wired; the flow waits for baseline reproduction before Method generation.

    # --- Assertions ---------------------------------------------------------
    final_state = await load_session_state(bus.db, sid)
    assert final_state is not None
    assert final_state.current_state == "ReproducingBaseline"
    assert final_state.direction_critique_round == 0

    # benchmark_spec_ref + current_benchmark_id are set and resolve to a schema-valid BenchmarkSpec.
    # meeting #1: the pre-baseline verifier is GONE, so SelectingBenchmark goes STRAIGHT to
    # ReproducingBaseline and verifier_spec_ref is never frozen (stays None).
    assert final_state.benchmark_spec_ref is not None
    assert final_state.verifier_spec_ref is None
    assert final_state.current_benchmark_id == VALID_BENCHMARK_SPEC["benchmark_id"]

    benchmark_spec = store.get(final_state.benchmark_spec_ref)
    assert {
        key: value for key, value in benchmark_spec.items()
        if key != "evaluation_semantics"
    } == VALID_BENCHMARK_SPEC
    assert benchmark_spec["evaluation_semantics"]["primary_metric"] == {
        "name": "exact_match", "direction": "maximize",
        "measurement": "Read exact_match from the canonical evaluator."}
    benchmark_spec_errors = list(_validator(BENCHMARK_SPEC_SCHEMA_PATH).iter_errors(benchmark_spec))
    assert benchmark_spec_errors == [], [e.message for e in benchmark_spec_errors]

    # meeting #1: the pre-baseline verifier is gone, so there is no verifier-first invariant to
    # exercise and no verifier.spec_frozen event -- the flow goes straight to ReproducingBaseline.
    # Exactly one benchmark.spec_ready, and NO verifier.spec_frozen.
    spec_ready_events = [e for e in bus.published if e.event_type == "benchmark.spec_ready"]
    assert len(spec_ready_events) == 1
    assert spec_ready_events[0].session_id == sid
    assert spec_ready_events[0].payload["benchmark_spec_ref"] == final_state.benchmark_spec_ref
    assert spec_ready_events[0].payload["benchmark_id"] == VALID_BENCHMARK_SPEC["benchmark_id"]

    assert not [e for e in bus.published if e.event_type == "verifier.spec_frozen"]

    # No agent.failed anywhere along the way.
    assert not [e for e in bus.published if e.event_type == "agent.failed"]
