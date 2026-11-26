"""Unit tests for Task 4: GeneratingHypotheses entry action + advance.

Covers:
  - the GeneratingHypotheses entry action, which emits
    hypothesis.generation_requested with {research_spec_ref, literature_brief_ref}.
  - ``_on_GeneratingHypotheses_hypothesis_generated``: GeneratingHypotheses ->
    CritiquingHypotheses (state machine V1.10 §1), storing hypothesis_bank_ref.
    CritiquingHypotheses' entry action (``ranking.requested`` as of the
    RankingJudgeAgent milestone Task 4; was hypothesis.critique_requested
    directly at M6) is exercised in test_engine_critic_advance.py; here we
    only assert it fires alongside the transition, without checking its
    payload shape.

GeneratingHypotheses is NOT reachable from the current mainline (intermediate
states unbuilt) -- this wiring is exercised IN ISOLATION by constructing a
SessionState directly in GeneratingHypotheses, mirroring the dependency-order
note in the task description.

Reuses FakeBusClient from test_engine_min.py (does NOT auto-deliver; the test
drives the engine directly via ``eng.tick(event)`` / ``eng._execute_entry_action``),
mirroring test_engine_novelty_advance.py's pattern.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from autoresearch.orchestrator.engine import OrchestratorEngine
from autoresearch.orchestrator.session_state import (
    SessionState,
    load_session_state,
    save_session_state,
)
from autoresearch.orchestrator.tests.unit.test_engine_min import FakeBusClient, fake_bus  # noqa: F401

SM_ROOT = Path(__file__).resolve().parents[2] / "state-machine"


@dataclass
class _Evt:
    event_type: str
    session_id: str
    emitter: str
    payload: dict = field(default_factory=dict)


async def _seed_generating_hypotheses_session(eng: OrchestratorEngine, sid: str) -> tuple[str, str]:
    """Persist a session in GeneratingHypotheses with the 2 upstream refs set.

    Isolated construction (GeneratingHypotheses is unreachable from the
    mainline at M5): builds the SessionState directly rather than driving it
    through prior states.
    """
    research_spec_ref = f"research_spec://{sid}/revised"
    literature_brief_ref = f"spec://literature_brief/{sid}/final"
    state = SessionState(
        session_id=sid,
        user_id="u1",
        current_state="GeneratingHypotheses",
        research_spec_ref=research_spec_ref,
        literature_brief_ref=literature_brief_ref,
    )
    await save_session_state(eng.client.db, state, auto_commit=True)
    return research_spec_ref, literature_brief_ref


async def test_entry_action_emits_hypothesis_generation_requested(fake_bus):
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = "sess-1"
    research_spec_ref, literature_brief_ref = await _seed_generating_hypotheses_session(eng, sid)
    state = await load_session_state(fake_bus.db, sid)
    assert state is not None

    n_before = len(fake_bus.published)
    await eng._execute_entry_action(state)

    new_events = fake_bus.published[n_before:]
    emitted = [p for p in new_events if p.event_type == "hypothesis.generation_requested"]
    assert len(emitted) == 1
    assert emitted[0].session_id == sid
    assert emitted[0].payload == {
        "research_spec_ref": research_spec_ref,
        "literature_brief_ref": literature_brief_ref,
        "research_memory_ref": None,   # idea<->experiment loop: unset on the first pass
    }


async def test_hypothesis_generated_advances_to_critiquing_hypotheses(fake_bus):
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = "sess-2"
    await _seed_generating_hypotheses_session(eng, sid)

    hypothesis_bank_ref = f"hypothesis_bank://{sid}/final"
    await eng.tick(
        _Evt(
            "hypothesis.generated",
            sid,
            "HypothesisAgent",
            payload={"hypothesis_bank_ref": hypothesis_bank_ref},
        )
    )

    st = await load_session_state(fake_bus.db, sid)
    assert st is not None
    assert st.current_state == "CritiquingHypotheses"
    assert st.hypothesis_bank_ref == hypothesis_bank_ref

    # CritiquingHypotheses' entry action (RankingJudgeAgent milestone Task 4)
    # re-emits ranking.requested; no other events are emitted.
    new_events = [p for p in fake_bus.published if p.event_type not in
                  ("hypothesis.generation_requested", "ranking.requested")]
    assert new_events == []


async def test_hypothesis_generated_advance_is_atomic_single_commit(fake_bus):
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = "sess-3"
    await _seed_generating_hypotheses_session(eng, sid)

    commits_before = fake_bus.db.commit_count
    await eng.tick(
        _Evt(
            "hypothesis.generated",
            sid,
            "HypothesisAgent",
            payload={"hypothesis_bank_ref": f"hypothesis_bank://{sid}/final"},
        )
    )

    assert fake_bus.db.commit_count - commits_before == 1
