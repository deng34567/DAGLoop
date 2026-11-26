"""GeneratingHypotheses isolated integration smoke (M5 Task 5): single
session, no Redis, no real LLM.

``GeneratingHypotheses`` is NOT reachable from ``start_session`` yet
(intermediate states between ``ModelingProblem`` and ``GeneratingHypotheses``
are unbuilt) -- mirrors the isolation-construction pattern in
``test_engine_hypothesis_advance.py``. This smoke goes one step further than
that unit test by driving the isolated session through a REAL
``DeliveringBus`` (genuine subscribe/dispatch routing, not a manual handler
call), so the full cascade -- entry action -> HypothesisAgent -> orchestrator
tick -> transition -- is exercised for real across the two independently-built
components: ``OrchestratorEngine`` and the HypothesisAgent service.

Reuses the ``DeliveringBus``/``BusClientView`` pattern from
``test_m4_novelty_audit_smoke.py`` (copied here so this file stays
self-contained).

Drives, in one linear session:
  1. Seed a valid ``ResearchSpec`` (kind="research_spec", emitter=
     "ProblemModelingAgent") and a ``LiteratureBrief`` (kind=
     "literature_brief", emitter="DeepResearchAgent") directly into the Spec
     Store, as upstream agents would have.
  2. Construct + save a ``SessionState`` directly in ``GeneratingHypotheses``
     with ``research_spec_ref`` / ``literature_brief_ref`` set (isolated
     construction -- no prior states are driven).
  3. ``eng.resume_to_current_state(session_id)`` re-emits the
     ``GeneratingHypotheses`` entry action, ``hypothesis.generation_requested``
     -> (delivered) -> HypothesisAgent resolves both inputs from the store,
     runs its (canned) ``generate_step``, persists a HypothesisBank, and
     publishes ``hypothesis.generated`` -> (delivered) -> orchestrator's
     ``tick`` -> ``_on_GeneratingHypotheses_hypothesis_generated`` transitions
     GeneratingHypotheses -> CritiquingHypotheses (no entry action there yet --
     CriticAgent is M6 -- so the loop pauses).

No concurrency, no Redis, no real LLM call anywhere in this file (one FakeLLM
canned queue). ``resume_to_current_state`` is called exactly once -- a second
call would re-emit the entry action and break the "exactly one" event counts.
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

from autoresearch.agents.hypothesis_agent.service import build_hypothesis_service
from autoresearch.agents.hypothesis_agent.tests.conftest import FakeLLM
from autoresearch.eventbus.broker.envelope import Envelope
from autoresearch.orchestrator.engine import OrchestratorEngine
from autoresearch.orchestrator.session_state import SessionState, load_session_state, save_session_state
from autoresearch.spec_store import SpecStore

SM_ROOT = Path(__file__).resolve().parents[2] / "state-machine"
HYPOTHESIS_CONTRACTS_DIR = (
    Path(__file__).resolve().parents[3] / "agents" / "hypothesis_agent" / "data-contracts"
)
HYPOTHESIS_BANK_SCHEMA_PATH = HYPOTHESIS_CONTRACTS_DIR / "hypothesis_bank.schema.yaml"

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


VALID_HYPOTHESES = [
    _hypothesis("h1", "idea"),
    _hypothesis("h2", "training"),
]


def _bank_json(hypotheses: list[dict]) -> str:
    return json.dumps({"hypotheses": hypotheses})


# ---------------------------------------------------------------------------
# DeliveringBus: minimal in-memory broker that actually routes (mirrors M4).
# Copied from test_m4_novelty_audit_smoke.py to keep this file self-contained.
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
    conn = await aiosqlite.connect(str(tmp_path / "hypothesis_smoke.db"))
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


async def test_hypothesis_generation_smoke_single_session(bus, store):
    # --- wire OrchestratorEngine + HypothesisAgent -- both on the SAME bus,
    # each with its own emitter identity.
    eng = OrchestratorEngine(client=BusClientView(bus, "ResearchOrchestrator"), skill_root=SM_ROOT)
    await eng.start()

    hypothesis_llm = FakeLLM([_bank_json(VALID_HYPOTHESES)])
    build_hypothesis_service(
        client=BusClientView(bus, "HypothesisAgent"),
        spec_store=store,
        llm_client=hypothesis_llm,
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
    # generate_step, persists the HypothesisBank, and publishes
    # hypothesis.generated -> (delivered) -> orchestrator's tick ->
    # _on_GeneratingHypotheses_hypothesis_generated transitions
    # GeneratingHypotheses -> CritiquingHypotheses (no entry action there yet
    # -- CriticAgent is M6 -- so the loop pauses). Called exactly once.
    result = await eng.resume_to_current_state(sid)
    assert result["resumed"] is True

    # --- Assertions -------------------------------------------------------
    final_state = await load_session_state(bus.db, sid)
    assert final_state is not None
    assert final_state.current_state == "CritiquingHypotheses"
    assert final_state.hypothesis_bank_ref is not None

    # Exactly one hypothesis.generation_requested and one hypothesis.generated.
    requested_events = [e for e in bus.published if e.event_type == "hypothesis.generation_requested"]
    assert len(requested_events) == 1
    assert requested_events[0].session_id == sid
    assert requested_events[0].payload == {
        "research_spec_ref": research_spec_ref,
        "literature_brief_ref": literature_brief_ref,
        "research_memory_ref": None,   # idea<->experiment loop: unset on the first pass
    }

    generated_events = [e for e in bus.published if e.event_type == "hypothesis.generated"]
    assert len(generated_events) == 1
    assert generated_events[0].session_id == sid
    assert generated_events[0].payload["hypothesis_bank_ref"] == final_state.hypothesis_bank_ref
    assert generated_events[0].payload["count"] == len(VALID_HYPOTHESES)

    # No agent.failed anywhere along the way.
    assert not [e for e in bus.published if e.event_type == "agent.failed"]

    # hypothesis_bank_ref resolves in the Spec Store to a schema-valid
    # HypothesisBank whose hypotheses count matches hypothesis.generated's count.
    bank = store.get(final_state.hypothesis_bank_ref)
    assert bank == {"hypotheses": VALID_HYPOTHESES}
    assert len(bank["hypotheses"]) == generated_events[0].payload["count"]
    bank_errors = list(_validator(HYPOTHESIS_BANK_SCHEMA_PATH).iter_errors(bank))
    assert bank_errors == [], [e.message for e in bank_errors]
