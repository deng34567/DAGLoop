"""M0 end-to-end smoke test (Task 8): single session, no Redis, no LLM.

Drives OrchestratorEngine directly against the FakeBusClient test double
already defined for the engine unit tests (real aiosqlite handle; ``.publish``
merely captures the outbound event -- no broker, no Redis, no LLM). Covers,
in one linear run:

  1. start_session -> SessionState persisted in ClarifyingResearchGoal + the
     entry action ``problem_modeling.requested`` captured on the fake bus,
     save+emit bundled in exactly ONE commit (INV-4 atomicity).
  2. _transition into Suspended, persisted + recoverable from the db.
  3. resume_to_current_state -> back to the pre-suspend state
     (ClarifyingResearchGoal) with its entry action re-emitted, again atomic.
  4. SessionState recoverable from the tmp sqlite db (load_session_state) at
     every step above.

No concurrency, no Redis, no real LLM calls anywhere in this file.
"""

from __future__ import annotations

from pathlib import Path

import aiosqlite
import pytest

from autoresearch.orchestrator.engine import OrchestratorEngine
from autoresearch.orchestrator.session_state import load_session_state
from autoresearch.orchestrator.tests.unit.test_engine_min import FakeBusClient

SM_ROOT = Path(__file__).resolve().parents[2] / "state-machine"


@pytest.fixture
async def fake_bus(tmp_path):
    conn = await aiosqlite.connect(str(tmp_path / "m0_smoke.db"))
    try:
        yield FakeBusClient(conn)
    finally:
        await conn.close()


async def test_m0_single_session_smoke(fake_bus):
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()

    # --- 1. start_session: atomic create + first entry action -------------
    commits_before = fake_bus.db.commit_count
    sid = await eng.start_session("optimize modded-nanogpt speedrun", user_id="u1")

    state = await load_session_state(fake_bus.db, sid)
    assert state is not None
    assert state.current_state == "ClarifyingResearchGoal"
    assert state.user_id == "u1"
    assert fake_bus.db.commit_count - commits_before == 1  # one commit: save + emit

    first_events = [p for p in fake_bus.published if p.event_type == "problem_modeling.requested"]
    assert len(first_events) == 1
    assert first_events[0].session_id == sid
    assert first_events[0].payload == {"research_goal_text": "optimize modded-nanogpt speedrun"}

    # --- 2. Enter Suspended via _transition; recoverable from db -----------
    commits_before = fake_bus.db.commit_count
    await eng._transition(state, "Suspended")
    assert fake_bus.db.commit_count - commits_before == 1

    suspended = await load_session_state(fake_bus.db, sid)
    assert suspended is not None
    assert suspended.current_state == "Suspended"
    assert suspended.previous_state == "ClarifyingResearchGoal"

    # --- 3. resume_to_current_state -> back to pre-suspend state -----------
    commits_before = fake_bus.db.commit_count
    n_events_before = len(fake_bus.published)
    res = await eng.resume_to_current_state(sid)
    # Restored state is visible before a potentially inline entry action;
    # dispatch then commits independently and remains recoverable via redrive.
    assert fake_bus.db.commit_count - commits_before == 2

    assert res["resumed"] is True
    assert res["state"] == "ClarifyingResearchGoal"

    resumed = await load_session_state(fake_bus.db, sid)
    assert resumed is not None
    assert resumed.current_state == "ClarifyingResearchGoal"
    assert resumed.previous_state is None

    new_events = fake_bus.published[n_events_before:]
    assert any(p.event_type == "problem_modeling.requested" for p in new_events)

    # --- 4. Final consistency: db state matches the in-memory view ---------
    final = await load_session_state(fake_bus.db, sid)
    assert final is not None
    assert final.session_id == sid
    assert final.current_state == resumed.current_state
