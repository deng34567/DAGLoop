"""Unit tests for M1 Task 4: ClarifyingResearchGoal -> DeepResearching wiring.

Covers: the ``_on_ClarifyingResearchGoal_research_spec_ready`` handler advances
the state machine, persists ``research_spec_ref`` into SessionState, and emits
the DeepResearching entry action (``deep_research.requested``) atomically (one
commit). Also covers that the ClarifyingResearchGoal dialogue events
(``problem_modeling.clarification_question`` / ``human.clarification_answer``)
remain no-ops for orchestrator state (no ``_on_*`` handler wired for them).

Reuses FakeBusClient from test_engine_min.py (does NOT auto-deliver; the test
drives the engine directly via ``eng.tick(event)``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from autoresearch.orchestrator.engine import OrchestratorEngine
from autoresearch.orchestrator.session_state import load_session_state
from autoresearch.orchestrator.tests.unit.test_engine_min import FakeBusClient, fake_bus  # noqa: F401

SM_ROOT = Path(__file__).resolve().parents[2] / "state-machine"


@dataclass
class _Evt:
    event_type: str
    session_id: str
    emitter: str
    payload: dict = field(default_factory=dict)


async def test_research_spec_ready_advances_to_deep_researching_and_stores_ref(fake_bus):
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = await eng.start_session("goal", user_id="u1")

    ref = f"spec://research_spec/{sid}/final"
    await eng.tick(
        _Evt(
            "research_spec.ready",
            sid,
            "ProblemModelingAgent",
            payload={"research_spec_ref": ref},
        )
    )

    st = await load_session_state(fake_bus.db, sid)
    assert st is not None
    assert st.current_state == "DeepResearching"
    assert st.research_spec_ref == ref

    emitted = [p for p in fake_bus.published if p.event_type == "deep_research.requested"]
    assert len(emitted) == 1
    assert emitted[0].session_id == sid
    assert emitted[0].payload == {"research_spec_ref": ref}


async def test_research_spec_ready_transition_is_atomic_single_commit(fake_bus):
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = await eng.start_session("goal", user_id="u1")

    ref = f"spec://research_spec/{sid}/final"
    commits_before = fake_bus.db.commit_count
    await eng.tick(
        _Evt(
            "research_spec.ready",
            sid,
            "ProblemModelingAgent",
            payload={"research_spec_ref": ref},
        )
    )

    assert fake_bus.db.commit_count - commits_before == 1


@pytest.mark.parametrize(
    "event_type,emitter,payload",
    [
        (
            "problem_modeling.clarification_question",
            "ProblemModelingAgent",
            {"question_id": "q-1", "question_text": "What is the budget?"},
        ),
        (
            "human.clarification_answer",
            "Human",
            {"question_id": "q-1", "answer_text": "two weeks"},
        ),
    ],
)
async def test_clarification_dialogue_events_are_noop_for_orchestrator_state(
    fake_bus, event_type: str, emitter: str, payload: dict
):
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = await eng.start_session("goal", user_id="u1")

    await eng.tick(_Evt(event_type, sid, emitter, payload=payload))

    st = await load_session_state(fake_bus.db, sid)
    assert st.current_state == "ClarifyingResearchGoal"
