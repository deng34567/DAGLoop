"""Unit tests for M2 Task 4: DeepResearching -> ReinterpretingSpec wiring.

Covers: the ``_on_DeepResearching_literature_brief_ready`` handler advances
the state machine, persists ``literature_brief_ref`` + ``field_trajectory_brief_ref``
into SessionState, and emits the ReinterpretingSpec entry action
(``problem_modeling.reinterpret_requested``) atomically (one commit).
``problem_modeling.reinterpret_requested`` has no consumer in M2 (deferred);
the loop pauses at ReinterpretingSpec (same shape as M1 pausing at
DeepResearching).

Reuses FakeBusClient from test_engine_min.py (does NOT auto-deliver; the test
drives the engine directly via ``eng.tick(event)``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

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


async def _advance_to_deep_researching(eng: OrchestratorEngine, sid: str) -> str:
    """Drive a fresh session (ClarifyingResearchGoal) into DeepResearching.

    Returns the research_spec_ref stored along the way.
    """
    research_spec_ref = f"spec://research_spec/{sid}/final"
    await eng.tick(
        _Evt(
            "research_spec.ready",
            sid,
            "ProblemModelingAgent",
            payload={"research_spec_ref": research_spec_ref},
        )
    )
    return research_spec_ref


async def test_literature_brief_ready_advances_to_reinterpreting_spec_and_stores_refs(fake_bus):
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = await eng.start_session("goal", user_id="u1")
    research_spec_ref = await _advance_to_deep_researching(eng, sid)

    literature_brief_ref = f"spec://literature_brief/{sid}/final"
    field_trajectory_brief_ref = f"spec://field_trajectory_brief/{sid}/final"
    await eng.tick(
        _Evt(
            "literature.brief_ready",
            sid,
            "DeepResearchAgent",
            payload={
                "literature_brief_ref": literature_brief_ref,
                "field_trajectory_brief_ref": field_trajectory_brief_ref,
            },
        )
    )

    st = await load_session_state(fake_bus.db, sid)
    assert st is not None
    assert st.current_state == "ReinterpretingSpec"
    assert st.literature_brief_ref == literature_brief_ref
    assert st.field_trajectory_brief_ref == field_trajectory_brief_ref

    emitted = [p for p in fake_bus.published if p.event_type == "problem_modeling.reinterpret_requested"]
    assert len(emitted) == 1
    assert emitted[0].session_id == sid
    assert emitted[0].payload == {
        "research_spec_ref": research_spec_ref,
        "literature_brief_ref": literature_brief_ref,
        "field_trajectory_brief_ref": field_trajectory_brief_ref,
    }


async def test_literature_brief_ready_transition_is_atomic_single_commit(fake_bus):
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = await eng.start_session("goal", user_id="u1")
    await _advance_to_deep_researching(eng, sid)

    commits_before = fake_bus.db.commit_count
    await eng.tick(
        _Evt(
            "literature.brief_ready",
            sid,
            "DeepResearchAgent",
            payload={
                "literature_brief_ref": f"spec://literature_brief/{sid}/final",
                "field_trajectory_brief_ref": f"spec://field_trajectory_brief/{sid}/final",
            },
        )
    )

    assert fake_bus.db.commit_count - commits_before == 1


async def test_clarify_to_deep_researching_still_works(fake_bus):
    """Regression: the M1 handler/entry action are untouched by this wiring."""
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = await eng.start_session("goal", user_id="u1")

    research_spec_ref = await _advance_to_deep_researching(eng, sid)

    st = await load_session_state(fake_bus.db, sid)
    assert st.current_state == "DeepResearching"
    assert st.research_spec_ref == research_spec_ref

    emitted = [p for p in fake_bus.published if p.event_type == "deep_research.requested"]
    assert len(emitted) == 1
    assert emitted[0].payload == {"research_spec_ref": research_spec_ref}
