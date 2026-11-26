"""M1 end-to-end integration smoke (Task 5): single session, no Redis, no LLM.

Unlike the M0 smoke (``test_m0_smoke.py``), which drives ``OrchestratorEngine``
against a non-delivering ``FakeBusClient``, this smoke genuinely exercises
subscribe/dispatch routing between TWO independently-built components —
``OrchestratorEngine`` and the ``ProblemModelingAgent`` service
(``build_problem_modeling_service``) — wired on one shared in-memory
``DeliveringBus``.

``DeliveringBus`` mirrors the real ``EventBusClient`` + ``Dispatcher``
(``autoresearch/eventbus/broker/{client.py,dispatch.py}``) just enough to
route: ``publish(...)`` builds a real ``Envelope`` and synchronously
dispatches it (fnmatch, sequential, single-threaded) to every subscriber
whose pattern matches — no Redis, no outbox, no schema/whitelist validation.
Each caller gets a ``BusClientView`` (its own ``emitter_name``, mirroring how
every real process holds its own ``EventBusClient`` while sharing one
broker) so the engine's self-emit skip (``emitter == "ResearchOrchestrator"``)
and the agent's emitter tagging both work exactly as in production.

Drives, in one linear session:
  1. ``eng.start_session(...)`` -> ``problem_modeling.requested`` ->
     (delivered) -> agent seeds a draft + asks one clarification question.
  2. Simulated human answer (``human.clarification_answer``, emitter
     ``"Human"``) -> (delivered) -> agent finalizes the ResearchSpec ->
     ``research_spec.ready`` -> (delivered) -> orchestrator transitions
     ClarifyingResearchGoal -> DeepResearching -> emits
     ``deep_research.requested``.

No concurrency, no Redis, no real LLM call anywhere in this file (FakeLLM
canned queue; human answer is simulated, not a real UI/human).
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

from autoresearch.agents.problem_modeling_agent.service import build_problem_modeling_service
from autoresearch.agents.problem_modeling_agent.tests.conftest import FakeLLM
from autoresearch.eventbus.broker.envelope import Envelope
from autoresearch.orchestrator.engine import OrchestratorEngine
from autoresearch.orchestrator.session_state import load_session_state
from autoresearch.spec_store import SpecStore

SM_ROOT = Path(__file__).resolve().parents[2] / "state-machine"
SCHEMA_PATH = (
    Path(__file__).resolve().parents[3]
    / "agents"
    / "problem_modeling_agent"
    / "data-contracts"
    / "research_spec.schema.yaml"
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


# ---------------------------------------------------------------------------
# DeliveringBus: minimal in-memory broker that actually routes (Task 5).
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
    conn = await aiosqlite.connect(str(tmp_path / "m1_smoke.db"))
    try:
        yield DeliveringBus(conn)
    finally:
        await conn.close()


@pytest.fixture
def store(tmp_path):
    s = SpecStore(tmp_path / "store")
    yield s
    s.close()


def _spec_validator() -> Draft202012Validator:
    schema = yaml.safe_load(SCHEMA_PATH.read_text(encoding="utf-8"))
    return Draft202012Validator(schema)


# ---------------------------------------------------------------------------
# The smoke
# ---------------------------------------------------------------------------


async def test_m1_clarify_smoke_single_session(bus, store):
    # --- wire OrchestratorEngine + ProblemModelingAgent on the SAME bus -----
    eng = OrchestratorEngine(client=BusClientView(bus, "ResearchOrchestrator"), skill_root=SM_ROOT)
    await eng.start()

    llm = FakeLLM([_ask_json(), _finalize_json(FINAL_SPEC)])
    build_problem_modeling_service(
        client=BusClientView(bus, "ProblemModelingAgent"),
        spec_store=store,
        llm_client=llm,
    )

    # --- 1. start_session -> problem_modeling.requested -> (delivered) -----
    # -> agent asks its first clarification question, all synchronously
    # inside this one call (genuine subscribe/dispatch routing, not manual).
    sid = await eng.start_session("optimize modded-nanogpt speedrun", user_id="u1")

    requested = [e for e in bus.published if e.event_type == "problem_modeling.requested"]
    assert len(requested) == 1
    assert requested[0].session_id == sid

    questions = [
        e for e in bus.published if e.event_type == "problem_modeling.clarification_question"
    ]
    assert len(questions) == 1
    assert questions[0].session_id == sid
    question_id = questions[0].payload["question_id"]
    assert question_id == "q-1"

    state = await load_session_state(bus.db, sid)
    assert state is not None
    assert state.current_state == "ClarifyingResearchGoal"

    # --- 2. simulate the human answering -> (delivered) -> agent finalizes -
    # -> research_spec.ready -> (delivered) -> orchestrator transitions to
    # DeepResearching -> deep_research.requested.
    human = BusClientView(bus, "Human")
    await human.publish(
        event_type="human.clarification_answer",
        session_id=sid,
        payload={"question_id": question_id, "answer_text": "benchmark report"},
    )

    # --- Assertions -----------------------------------------------------
    final_state = await load_session_state(bus.db, sid)
    assert final_state is not None
    assert final_state.current_state == "DeepResearching"
    assert final_state.research_spec_ref is not None

    ready_events = [e for e in bus.published if e.event_type == "research_spec.ready"]
    assert len(ready_events) == 1
    assert ready_events[0].payload["research_spec_ref"] == final_state.research_spec_ref

    deep_events = [e for e in bus.published if e.event_type == "deep_research.requested"]
    assert len(deep_events) == 1
    assert deep_events[0].session_id == sid
    assert deep_events[0].payload == {"research_spec_ref": final_state.research_spec_ref}

    # Provenance: the ResearchSpec came from the Spec Store (via the ref
    # held on SessionState / carried by research_spec.ready), not the event
    # payload itself -- and it validates against research_spec.schema.yaml.
    spec = store.get(final_state.research_spec_ref)
    assert spec == FINAL_SPEC
    validator = _spec_validator()
    errors = list(validator.iter_errors(spec))
    assert errors == [], [e.message for e in errors]
    assert spec["topic"]
    assert spec["research_goal"]
