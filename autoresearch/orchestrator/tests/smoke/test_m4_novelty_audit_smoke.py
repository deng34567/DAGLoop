"""ModelingProblem end-to-end integration smoke (M4 Task 5): single session,
no Redis, no real LLM.

Extends the reinterpret smoke (``test_reinterpret_smoke.py``) by two more
hops: through ``AuditingLiteratureNovelty`` (now co-audited by
EvidenceAuditAgent's Mode A before NoveltyAuditAgent runs) to
``ModelingProblem``. Reuses the same ``DeliveringBus``/``BusClientView``
pattern (copied here so this file stays self-contained) to genuinely exercise
subscribe/dispatch routing between FIVE independently-built components --
``OrchestratorEngine``, the ``ProblemModelingAgent`` service, the
``DeepResearchAgent`` service, the ``EvidenceAuditAgent`` service, and the
``NoveltyAuditAgent`` service -- wired on one shared in-memory bus.

Drives, in one linear session:
  1. ``eng.start_session(...)`` -> ``problem_modeling.requested`` ->
     (delivered) -> ProblemModelingAgent seeds a draft + asks one
     clarification question.
  2. Simulated human answer (``human.clarification_answer``, emitter
     ``"Human"``) -> (delivered) -> ProblemModelingAgent finalizes the
     ResearchSpec -> ``research_spec.ready`` -> (delivered) -> orchestrator
     transitions ClarifyingResearchGoal -> DeepResearching -> emits
     ``deep_research.requested`` -> (delivered) -> DeepResearchAgent runs its
     two-pass FakeBackend search + FakeLLM synthesis, persists a
     LiteratureBrief + FieldTrajectoryBrief, and publishes
     ``literature.brief_ready`` -> (delivered) -> orchestrator transitions
     DeepResearching -> ReinterpretingSpec -> emits
     ``problem_modeling.reinterpret_requested`` -> (delivered) ->
     ProblemModelingAgent resolves the ResearchSpec + both briefs, runs its
     reinterpret_step (canned ``in_scope`` verdict), persists the
     SpecInterpretation + revised ResearchSpec, and publishes
     ``spec.reinterpreted`` -> (delivered) -> orchestrator routes on
     disposition=in_scope, transitions ReinterpretingSpec ->
     AuditingLiteratureNovelty, and emits its entry action
     ``evidence.audit_requested`` -> (delivered) -> EvidenceAuditAgent
     resolves the revised ResearchSpec + LiteratureBrief + SpecInterpretation
     from the store, runs its (canned) Mode A audit_literature_support,
     persists the EvidenceAuditReport, and publishes ``evidence.audited`` ->
     (delivered) -> orchestrator's
     ``_on_AuditingLiteratureNovelty_evidence_audited`` stores the
     EvidenceAuditReport ref (staying in AuditingLiteratureNovelty -- NOT a
     transition) and publishes ``novelty.audit_requested`` (now carrying
     ``evidence_audit_report_ref`` too) -> (delivered) -> NoveltyAuditAgent
     resolves the revised ResearchSpec + LiteratureBrief + SpecInterpretation
     + EvidenceAuditReport from the store, runs its (canned ``pass``)
     audit_step, persists the NoveltyAuditReport, and publishes
     ``novelty.audit_ready`` -> (delivered) -> orchestrator routes on
     verdict=pass (resetting counter K to 0), transitions
     AuditingLiteratureNovelty -> ModelingProblem, and emits its entry action
     ``problem_modeling.model_requested`` -> (delivered) ->
     ProblemModelingAgent resolves the ResearchSpec + LiteratureBrief +
     NoveltyAuditReport, runs its (canned) model_step, persists the
     ProblemModel, and publishes ``problem.modeled`` -> (delivered) ->
     orchestrator transitions ModelingProblem -> HumanIdeaCheckpoint (setting
     ``human_checkpoint_return_to = "CritiquingDirection"``).
     HumanIdeaCheckpoint has no entry action, so the loop pauses there
     awaiting the human (this smoke injects no ``human.idea_provided``).

No concurrency, no Redis, no real LLM call and no real search backend
anywhere in this file (four FakeLLMs canned queues; FakeBackend canned
papers; human answer is simulated, not a real UI/human; PaperQA2Backend is
never instantiated).
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
from autoresearch.agents.deep_research_agent.tests.conftest import FakeLLM as DeepResearchFakeLLM
from autoresearch.agents.evidence_audit_agent.service import build_evidence_audit_service
from autoresearch.agents.evidence_audit_agent.tests.conftest import FakeLLM as EvidenceAuditFakeLLM
from autoresearch.agents.novelty_audit_agent.service import build_novelty_audit_service
from autoresearch.agents.novelty_audit_agent.tests.conftest import FakeLLM as NoveltyAuditFakeLLM
from autoresearch.agents.problem_modeling_agent.service import build_problem_modeling_service
from autoresearch.agents.problem_modeling_agent.tests.conftest import FakeLLM as ProblemModelingFakeLLM
from autoresearch.eventbus.broker.envelope import Envelope
from autoresearch.orchestrator.engine import OrchestratorEngine
from autoresearch.orchestrator.session_state import load_session_state, save_session_state
from autoresearch.spec_store import SpecStore

SM_ROOT = Path(__file__).resolve().parents[2] / "state-machine"
PROBLEM_MODELING_CONTRACTS_DIR = (
    Path(__file__).resolve().parents[3] / "agents" / "problem_modeling_agent" / "data-contracts"
)
NOVELTY_AUDIT_CONTRACTS_DIR = (
    Path(__file__).resolve().parents[3] / "agents" / "novelty_audit_agent" / "data-contracts"
)
EVIDENCE_AUDIT_CONTRACTS_DIR = (
    Path(__file__).resolve().parents[3] / "agents" / "evidence_audit_agent" / "data-contracts"
)
RESEARCH_SPEC_SCHEMA_PATH = PROBLEM_MODELING_CONTRACTS_DIR / "research_spec.schema.yaml"
PROBLEM_MODEL_SCHEMA_PATH = PROBLEM_MODELING_CONTRACTS_DIR / "problem_model.schema.yaml"
NOVELTY_AUDIT_REPORT_SCHEMA_PATH = NOVELTY_AUDIT_CONTRACTS_DIR / "novelty_audit_report.schema.yaml"
EVIDENCE_AUDIT_REPORT_SCHEMA_PATH = EVIDENCE_AUDIT_CONTRACTS_DIR / "evidence_audit_report.schema.yaml"

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


# ---------------------------------------------------------------------------
# DeliveringBus: minimal in-memory broker that actually routes (Task 5).
# Copied from test_reinterpret_smoke.py to keep this file self-contained.
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
    conn = await aiosqlite.connect(str(tmp_path / "novelty_audit_smoke.db"))
    try:
        yield DeliveringBus(conn)
    finally:
        await conn.close()


@pytest.fixture
def store(tmp_path):
    s = SpecStore(tmp_path / "store")
    yield s
    s.close()


def _validator(path: Path) -> Draft202012Validator:
    schema = yaml.safe_load(path.read_text(encoding="utf-8"))
    return Draft202012Validator(schema)


# ---------------------------------------------------------------------------
# The smoke
# ---------------------------------------------------------------------------


async def test_novelty_audit_smoke_single_session(bus, store):
    # --- wire OrchestratorEngine + ProblemModelingAgent + DeepResearchAgent +
    # EvidenceAuditAgent + NoveltyAuditAgent -- all on the SAME bus, each with
    # its own emitter identity.
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

    # 3.3: the novelty audit now runs AFTER GeneratingMethod, so to exercise it we also wire the
    # CriticAgent (direction gate) + MethodAgent that precede it.
    critic_llm = CriticFakeLLM([json.dumps(
        {"target": "direction", "verdict": "pass", "issues": [], "passing_hypothesis_ids": []})])
    build_critic_service(
        client=BusClientView(bus, "CriticAgent"), spec_store=store, llm_client=critic_llm,
    )

    class _MethodFakeLLM:
        def __init__(self, responses):
            self._r = list(responses)

        def chat(self, *, messages, response_format=None, max_tokens=None):
            return type("R", (), {"content": self._r.pop(0) if self._r else ""})()

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
    # -> research_spec.ready -> (delivered) -> orchestrator transitions to
    # DeepResearching -> deep_research.requested -> (delivered) ->
    # DeepResearchAgent resolves the spec, runs its two-pass FakeBackend
    # search + FakeLLM synthesis, persists both briefs, and publishes
    # literature.brief_ready -> (delivered) -> orchestrator transitions to
    # ReinterpretingSpec -> emits problem_modeling.reinterpret_requested ->
    # (delivered) -> ProblemModelingAgent resolves the ResearchSpec + both
    # briefs, runs its (canned in_scope) reinterpret_step, persists the
    # SpecInterpretation + revised ResearchSpec, and publishes
    # spec.reinterpreted -> (delivered) -> orchestrator routes on
    # disposition=in_scope, transitions to AuditingLiteratureNovelty, and
    # emits evidence.audit_requested -> (delivered) -> EvidenceAuditAgent
    # resolves the revised spec + literature brief + spec interpretation from
    # the store, runs its (canned) Mode A audit_literature_support, persists
    # the EvidenceAuditReport, and publishes evidence.audited -> (delivered)
    # -> orchestrator's _on_AuditingLiteratureNovelty_evidence_audited stores
    # the ref (staying in AuditingLiteratureNovelty) and publishes
    # novelty.audit_requested (now carrying evidence_audit_report_ref too) ->
    # (delivered) -> NoveltyAuditAgent resolves the revised spec + literature
    # brief + spec interpretation + evidence audit report from the store,
    # runs its (canned pass) audit_step, persists the NoveltyAuditReport, and
    # publishes novelty.audit_ready -> (delivered) -> orchestrator routes on
    # verdict=pass, resets counter K, and transitions to ModelingProblem,
    # emitting problem_modeling.model_requested -> (delivered) ->
    # ProblemModelingAgent resolves the revised spec + literature brief +
    # novelty audit report from the store, runs its (canned) model_step,
    # persists the ProblemModel, and publishes problem.modeled (no
    # orchestrator consumer yet; loop pauses here). All synchronous, one call.
    human = BusClientView(bus, "Human")
    await human.publish(
        event_type="human.clarification_answer",
        session_id=sid,
        payload={"question_id": question_id, "answer_text": "benchmark report"},
    )

    # --- 3. the clarification cascade now rests at HumanIdeaCheckpoint (3.3: the audit no longer
    # runs before the method). Drive the idea checkpoint (no inject) -> GeneratingMethod ->
    # DeliberatingMethod -> a second, method-targeted DeepResearching pass -> the evidence+novelty
    # co-audit -> ConfirmingMethod (pauses for the human method gate).
    checkpoint_state = await load_session_state(bus.db, sid)
    assert checkpoint_state is not None
    assert checkpoint_state.current_state == "HumanIdeaCheckpoint"
    # the audit has NOT run yet at this point.
    assert not [e for e in bus.published if e.event_type == "evidence.audit_requested"]

    # This focused smoke exercises the non-benchmark Method/novelty path. The benchmark
    # path now reproduces a baseline before generating Method and is covered separately.
    checkpoint_state.mode = "idea_theory"
    checkpoint_state.human_checkpoint_return_to = "GeneratingMethod"
    await save_session_state(bus.db, checkpoint_state)

    await human.publish(event_type="human.idea_provided", session_id=sid, payload={"inject": False})

    # --- Assertions -----------------------------------------------------
    final_state = await load_session_state(bus.db, sid)
    assert final_state is not None
    assert final_state.current_state == "ConfirmingMethod"   # 3.3: reached via the method-novelty audit
    assert final_state.research_spec_ref is not None
    assert final_state.literature_brief_ref is not None
    assert final_state.field_trajectory_brief_ref is not None
    assert final_state.spec_interpretation_ref is not None
    assert final_state.method_ref is not None
    assert final_state.evidence_audit_report_ref is not None
    assert final_state.novelty_audit_report_ref is not None
    # Counter K (literature_audit_round) reset to 0 on a pass verdict.
    assert final_state.literature_audit_round == 0

    # Exactly one evidence.audit_requested (AuditingLiteratureNovelty's entry
    # action) and exactly one evidence.audited in response.
    evidence_requested_events = [e for e in bus.published if e.event_type == "evidence.audit_requested"]
    assert len(evidence_requested_events) == 1
    assert evidence_requested_events[0].session_id == sid
    targeted_search_id = evidence_requested_events[0].payload["targeted_search_id"]

    deep_research_events = [e for e in bus.published if e.event_type == "deep_research.requested"]
    assert len(deep_research_events) == 2
    assert deep_research_events[-1].payload["method_ref"] == final_state.method_ref
    assert deep_research_events[-1].payload["targeted_search_id"] == targeted_search_id

    evidence_audited_events = [e for e in bus.published if e.event_type == "evidence.audited"]
    assert len(evidence_audited_events) == 1
    assert evidence_audited_events[0].session_id == sid
    assert evidence_audited_events[0].payload["evidence_audit_report_ref"] == final_state.evidence_audit_report_ref

    # Exactly one novelty.audit_requested, published IN-STATE (still
    # AuditingLiteratureNovelty) by the orchestrator's evidence_audited
    # handler, carrying evidence_audit_report_ref alongside the original
    # three refs.
    novelty_requested_events = [e for e in bus.published if e.event_type == "novelty.audit_requested"]
    assert len(novelty_requested_events) == 1
    assert novelty_requested_events[0].session_id == sid
    assert novelty_requested_events[0].payload == {
        "research_spec_ref": final_state.research_spec_ref,
        "literature_brief_ref": final_state.literature_brief_ref,
        "spec_interpretation_ref": final_state.spec_interpretation_ref,
        "evidence_audit_report_ref": final_state.evidence_audit_report_ref,
        "method_ref": final_state.method_ref,   # 3.3: the audit judges the method's novelty
        "targeted_search_id": targeted_search_id,
    }

    # Exactly one novelty.audit_ready, verdict pass, carrying the report ref.
    audit_ready_events = [e for e in bus.published if e.event_type == "novelty.audit_ready"]
    assert len(audit_ready_events) == 1
    assert audit_ready_events[0].session_id == sid
    assert audit_ready_events[0].payload["verdict"] == "pass"
    assert audit_ready_events[0].payload["novelty_audit_report_ref"] == final_state.novelty_audit_report_ref

    # The method was generated (before the audit) and a pass routed the flow to the ConfirmingMethod
    # human gate (asserted above). Exactly one method.generated along the way.
    method_events = [e for e in bus.published if e.event_type == "method.generated"]
    assert len(method_events) == 1
    assert method_events[0].session_id == sid
    # The problem was modelled during the earlier cascade (ModelingProblem -> HumanIdeaCheckpoint).
    modeled_events = [e for e in bus.published if e.event_type == "problem.modeled"]
    assert len(modeled_events) == 1
    problem_model = store.get(modeled_events[0].payload["problem_model_ref"])
    assert problem_model == VALID_PROBLEM_MODEL
    problem_model_errors = list(_validator(PROBLEM_MODEL_SCHEMA_PATH).iter_errors(problem_model))
    assert problem_model_errors == [], [e.message for e in problem_model_errors]

    # No agent.failed anywhere along the way.
    assert not [e for e in bus.published if e.event_type == "agent.failed"]

    # novelty_audit_report_ref resolves in the Spec Store to a schema-valid
    # NoveltyAuditReport with verdict pass.
    report = store.get(final_state.novelty_audit_report_ref)
    assert report == VALID_NOVELTY_AUDIT_REPORT_PASS
    report_errors = list(_validator(NOVELTY_AUDIT_REPORT_SCHEMA_PATH).iter_errors(report))
    assert report_errors == [], [e.message for e in report_errors]

    # evidence_audit_report_ref resolves in the Spec Store to a schema-valid
    # EvidenceAuditReport.
    evidence_report = store.get(final_state.evidence_audit_report_ref)
    assert evidence_report == VALID_EVIDENCE_AUDIT_REPORT
    evidence_report_errors = list(_validator(EVIDENCE_AUDIT_REPORT_SCHEMA_PATH).iter_errors(evidence_report))
    assert evidence_report_errors == [], [e.message for e in evidence_report_errors]

    # Provenance + schema validity: research_spec_ref still points at the
    # REVISED spec, which resolves in the Spec Store and validates against
    # research_spec.schema.yaml.
    revised_spec = store.get(final_state.research_spec_ref)
    assert revised_spec == FINAL_SPEC
    spec_errors = list(_validator(RESEARCH_SPEC_SCHEMA_PATH).iter_errors(revised_spec))
    assert spec_errors == [], [e.message for e in spec_errors]
