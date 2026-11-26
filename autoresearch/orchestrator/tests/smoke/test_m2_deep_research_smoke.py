"""M2 end-to-end integration smoke (Task 5): single session, no Redis, no LLM.

Extends the M1 smoke (``test_m1_clarify_smoke.py``) by one more hop: through
``DeepResearchAgent`` to ``ReinterpretingSpec``. Reuses the same
``DeliveringBus``/``BusClientView`` pattern (copied here so this file stays
self-contained) to genuinely exercise subscribe/dispatch routing between
THREE independently-built components -- ``OrchestratorEngine``, the
``ProblemModelingAgent`` service, and the ``DeepResearchAgent`` service --
wired on one shared in-memory bus.

Drives, in one linear session:
  1. ``eng.start_session(...)`` -> ``problem_modeling.requested`` ->
     (delivered) -> ProblemModelingAgent seeds a draft + asks one
     clarification question.
  2. Simulated human answer (``human.clarification_answer``, emitter
     ``"Human"``) -> (delivered) -> ProblemModelingAgent finalizes the
     ResearchSpec -> ``research_spec.ready`` -> (delivered) -> orchestrator
     transitions ClarifyingResearchGoal -> DeepResearching -> emits
     ``deep_research.requested``.
  3. (delivered) -> DeepResearchAgent resolves the ResearchSpec from the
     Spec Store, runs its two-pass FakeBackend search + FakeLLM synthesis,
     persists a LiteratureBrief + FieldTrajectoryBrief, and publishes
     ``literature.brief_ready`` -> (delivered) -> orchestrator transitions
     DeepResearching -> ReinterpretingSpec -> emits
     ``problem_modeling.reinterpret_requested`` -> (delivered) ->
     ProblemModelingAgent resolves the ResearchSpec + both briefs, runs its
     reinterpret_step, persists the SpecInterpretation + revised
     ResearchSpec, and publishes ``spec.reinterpreted`` -> (delivered) ->
     orchestrator routes on disposition=in_scope and transitions
     ReinterpretingSpec -> AuditingLiteratureNovelty, emitting its entry
     action ``evidence.audit_requested`` (no consumer yet -- EvidenceAuditAgent
     milestone -- so the loop pauses there instead). This file only validates
     the M2 hop (through
     DeepResearching); the ReinterpretingSpec -> AuditingLiteratureNovelty
     hop is covered by ``test_reinterpret_smoke.py``.

No concurrency, no Redis, no real LLM call and no real search backend
anywhere in this file (two FakeLLM canned queues; FakeBackend canned
papers; human answer is simulated, not a real UI/human;
PaperQA2Backend is never instantiated).
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
from autoresearch.agents.deep_research_agent.service import build_deep_research_service
from autoresearch.agents.deep_research_agent.tests.conftest import FakeLLM as DeepResearchFakeLLM
from autoresearch.agents.problem_modeling_agent.service import build_problem_modeling_service
from autoresearch.agents.problem_modeling_agent.tests.conftest import FakeLLM as ProblemModelingFakeLLM
from autoresearch.eventbus.broker.envelope import Envelope
from autoresearch.orchestrator.engine import OrchestratorEngine
from autoresearch.orchestrator.session_state import load_session_state
from autoresearch.spec_store import SpecStore

SM_ROOT = Path(__file__).resolve().parents[2] / "state-machine"
RESEARCH_SPEC_SCHEMA_PATH = (
    Path(__file__).resolve().parents[3]
    / "agents"
    / "problem_modeling_agent"
    / "data-contracts"
    / "research_spec.schema.yaml"
)
DEEP_RESEARCH_CONTRACTS_DIR = (
    Path(__file__).resolve().parents[3]
    / "agents"
    / "deep_research_agent"
    / "data-contracts"
)

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


# ---------------------------------------------------------------------------
# DeliveringBus: minimal in-memory broker that actually routes (Task 5).
# Copied from test_m1_clarify_smoke.py to keep this file self-contained.
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
    conn = await aiosqlite.connect(str(tmp_path / "m2_smoke.db"))
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


async def test_m2_deep_research_smoke_single_session(bus, store):
    # --- wire OrchestratorEngine + ProblemModelingAgent + DeepResearchAgent -
    # all on the SAME bus, each with its own emitter identity.
    eng = OrchestratorEngine(client=BusClientView(bus, "ResearchOrchestrator"), skill_root=SM_ROOT)
    await eng.start()

    problem_modeling_llm = ProblemModelingFakeLLM(
        # 3.3: reinterpret(in_scope) now flows into ModelingProblem (novelty audit moved later), so
        # ProblemModelingAgent also runs model_step -> add its canned reply (rests at HumanIdeaCheckpoint).
        [_ask_json(), _finalize_json(FINAL_SPEC), _reinterpret_json(FINAL_SPEC),
         json.dumps({"key_problems": ["p"], "opportunities": ["o"], "risks": ["r"],
                     "verifiable_paths": ["v"]})]
    )
    build_problem_modeling_service(
        client=BusClientView(bus, "ProblemModelingAgent"),
        spec_store=store,
        llm_client=problem_modeling_llm,
    )

    backend = FakeBackend()
    deep_research_llm = DeepResearchFakeLLM([json.dumps({"query": "literature search keywords"}), _synthesis_json()])
    build_deep_research_service(
        client=BusClientView(bus, "DeepResearchAgent"),
        spec_store=store,
        backend=backend,
        llm_client=deep_research_llm,
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
    # (delivered) -> ProblemModelingAgent reinterprets (canned in_scope) and
    # publishes spec.reinterpreted -> (delivered) -> orchestrator advances to
    # AuditingLiteratureNovelty. All synchronous, one call.
    human = BusClientView(bus, "Human")
    await human.publish(
        event_type="human.clarification_answer",
        session_id=sid,
        payload={"question_id": question_id, "answer_text": "benchmark report"},
    )

    # --- Assertions -----------------------------------------------------
    final_state = await load_session_state(bus.db, sid)
    assert final_state is not None
    # 3.3: reinterpret(in_scope) -> ModelingProblem -> HumanIdeaCheckpoint (pauses; novelty audit
    # moved to after the method, so it does not run here).
    assert final_state.current_state == "HumanIdeaCheckpoint"
    assert final_state.research_spec_ref is not None
    assert final_state.literature_brief_ref is not None
    assert final_state.field_trajectory_brief_ref is not None
    assert final_state.spec_interpretation_ref is not None

    # The ORIGINAL (pre-reinterpret) ResearchSpec ref, captured off
    # research_spec.ready -- final_state.research_spec_ref has since been
    # overwritten to the REVISED ref by the ReinterpretingSpec -> Auditing
    # LiteratureNovelty transition (see assertions on reinterpreted_events
    # below).
    ready_events = [e for e in bus.published if e.event_type == "research_spec.ready"]
    assert len(ready_events) == 1
    original_research_spec_ref = ready_events[0].payload["research_spec_ref"]

    deep_requested_events = [e for e in bus.published if e.event_type == "deep_research.requested"]
    assert len(deep_requested_events) == 1
    assert deep_requested_events[0].session_id == sid
    assert deep_requested_events[0].payload == {"research_spec_ref": original_research_spec_ref}

    brief_ready_events = [e for e in bus.published if e.event_type == "literature.brief_ready"]
    assert len(brief_ready_events) == 1
    assert brief_ready_events[0].session_id == sid
    assert brief_ready_events[0].payload == {
        "literature_brief_ref": final_state.literature_brief_ref,
        "field_trajectory_brief_ref": final_state.field_trajectory_brief_ref,
    }

    reinterpret_events = [
        e for e in bus.published if e.event_type == "problem_modeling.reinterpret_requested"
    ]
    assert len(reinterpret_events) == 1
    assert reinterpret_events[0].session_id == sid
    assert reinterpret_events[0].payload == {
        "research_spec_ref": original_research_spec_ref,
        "literature_brief_ref": final_state.literature_brief_ref,
        "field_trajectory_brief_ref": final_state.field_trajectory_brief_ref,
    }

    # ProblemModelingAgent consumes problem_modeling.reinterpret_requested and
    # publishes spec.reinterpreted; the orchestrator now routes on its
    # disposition (in_scope here) and advances ReinterpretingSpec ->
    # AuditingLiteratureNovelty, updating research_spec_ref to the revised
    # ref (see docstring above).
    reinterpreted_events = [e for e in bus.published if e.event_type == "spec.reinterpreted"]
    assert len(reinterpreted_events) == 1
    assert reinterpreted_events[0].session_id == sid
    assert reinterpreted_events[0].payload["disposition"] == "in_scope"
    spec_interpretation = store.get(reinterpreted_events[0].payload["spec_interpretation_ref"])
    assert spec_interpretation["disposition"] == "in_scope"
    revised_spec = store.get(reinterpreted_events[0].payload["revised_research_spec_ref"])
    assert revised_spec == FINAL_SPEC
    assert final_state.research_spec_ref == reinterpreted_events[0].payload["revised_research_spec_ref"]
    assert final_state.spec_interpretation_ref == reinterpreted_events[0].payload["spec_interpretation_ref"]

    # 3.3: ModelingProblem's entry action fires (problem_modeling.model_requested); the
    # literature-novelty audit does NOT run here anymore (it moved to after GeneratingMethod).
    model_events = [e for e in bus.published if e.event_type == "problem_modeling.model_requested"]
    assert len(model_events) == 1
    assert model_events[0].session_id == sid
    assert not [e for e in bus.published if e.event_type == "evidence.audit_requested"]

    # No agent.failed anywhere along the way.
    assert not [e for e in bus.published if e.event_type == "agent.failed"]

    # Provenance: the ResearchSpec came from the Spec Store (via the ref
    # held on SessionState / carried by research_spec.ready), not the event
    # payload itself -- and it validates against research_spec.schema.yaml.
    spec = store.get(final_state.research_spec_ref)
    assert spec == FINAL_SPEC
    spec_errors = list(_validator(RESEARCH_SPEC_SCHEMA_PATH).iter_errors(spec))
    assert spec_errors == [], [e.message for e in spec_errors]

    # The two briefs resolve in the store to schema-valid, grounded objects.
    literature_brief = store.get(final_state.literature_brief_ref)
    literature_errors = list(
        _validator(DEEP_RESEARCH_CONTRACTS_DIR / "literature_brief.schema.yaml").iter_errors(
            literature_brief
        )
    )
    assert literature_errors == [], [e.message for e in literature_errors]

    field_trajectory_brief = store.get(final_state.field_trajectory_brief_ref)
    field_trajectory_errors = list(
        _validator(
            DEEP_RESEARCH_CONTRACTS_DIR / "field_trajectory_brief.schema.yaml"
        ).iter_errors(field_trajectory_brief)
    )
    assert field_trajectory_errors == [], [e.message for e in field_trajectory_errors]

    # Grounding (anti-hallucination): key_papers must be a subset of the
    # papers the FakeBackend actually returned.
    retrieved_ids = {p["id_or_url"] for p in backend._seminal + backend._frontier}
    key_paper_ids = {p["id_or_url"] for p in literature_brief["key_papers"]}
    assert key_paper_ids <= retrieved_ids
    assert key_paper_ids  # non-trivially grounded, not vacuously empty
    assert backend.seminal_calls == 1
    assert backend.frontier_calls == 1
