"""Unit tests for M14 Task 4: orchestrator Reporting routing -> terminal Done.

Covers:
  - the Reporting entry action (``evidence.citation_audit_requested`` --
    EvidenceAuditAgent milestone Task 5 -- was ``report.requested`` before
    that milestone), carrying research_memory_ref + research_spec_ref off
    SessionState.
  - ``_on_Reporting_report_ready``: stores event.payload["report_ref"] into
    SessionState.final_report_ref (the EXISTING dormant field) and atomically
    transitions into Done -- the terminal state, no special "complete_session"
    call, just ``_transition(state, "Done")``.
  - Done is terminal: ``_TERMINAL_STATES`` includes "Done"; ``tick``
    short-circuits sessions already in Done (no further handler dispatch).

Also regression-checks (d): run alongside test_engine_consolidation_advance.py
(M13) and test_engine_evaluation_advance.py (M12) -- both must stay green.

Reuses FakeBusClient from test_engine_min.py, mirroring
test_engine_consolidation_advance.py's patterns (does NOT auto-deliver; drives
the engine directly via ``eng.tick(event)``).
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _seed_reporting_session(
    eng: OrchestratorEngine,
    sid: str,
    *,
    research_memory_ref: str | None = None,
    research_spec_ref: str | None = None,
) -> None:
    """Persist a session in Reporting with its upstream refs set."""
    state = SessionState(
        session_id=sid,
        user_id="u1",
        current_state="Reporting",
        research_memory_ref=research_memory_ref or f"research_memory://{sid}/final",
        research_spec_ref=research_spec_ref or f"research_spec://{sid}/final",
    )
    await save_session_state(eng.client.db, state, auto_commit=True)


# ---------------------------------------------------------------------------
# (a): entry action
# ---------------------------------------------------------------------------


async def test_reporting_entry_action_emits_evidence_citation_audit_requested(fake_bus):
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = "sess-a"
    await _seed_reporting_session(eng, sid)
    state = await load_session_state(fake_bus.db, sid)

    await eng._execute_entry_action(state)

    emitted = [
        p for p in fake_bus.published if p.event_type == "evidence.citation_audit_requested"
    ]
    assert len(emitted) == 1
    assert emitted[0].payload == {
        "research_memory_ref": f"research_memory://{sid}/final",
        "research_spec_ref": f"research_spec://{sid}/final",
    }

    # The OLD entry event must NOT be emitted directly on entry anymore.
    assert not [p for p in fake_bus.published if p.event_type == "report.requested"]


# ---------------------------------------------------------------------------
# (b): report.ready -> Done, final_report_ref stored, atomic single commit
# ---------------------------------------------------------------------------


async def test_report_ready_advances_to_done_and_stores_final_report_ref(fake_bus):
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = "sess-b"
    await _seed_reporting_session(eng, sid)

    report_ref = f"final_report://{sid}/v1"
    commits_before = fake_bus.db.commit_count
    await eng.tick(
        _Evt(
            "report.ready",
            sid,
            "ReportingAgent",
            payload={"report_ref": report_ref},
        )
    )

    st = await load_session_state(fake_bus.db, sid)
    assert st is not None
    assert st.current_state == "Done"
    assert st.final_report_ref == report_ref

    # Done has no entry action -> the transition's only DB write is the
    # SessionState save, bundled into exactly ONE commit (INV-4).
    assert fake_bus.db.commit_count - commits_before == 1


# ---------------------------------------------------------------------------
# (c): Done is terminal
# ---------------------------------------------------------------------------


async def test_done_is_in_terminal_states():
    from autoresearch.orchestrator.engine import _TERMINAL_STATES, classify_resumability

    assert "Done" in _TERMINAL_STATES
    assert classify_resumability("Done") == "terminal"


async def test_tick_is_noop_once_session_is_done(fake_bus):
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = "sess-c"
    await _seed_reporting_session(eng, sid)

    report_ref = f"final_report://{sid}/v1"
    await eng.tick(
        _Evt("report.ready", sid, "ReportingAgent", payload={"report_ref": report_ref})
    )
    st = await load_session_state(fake_bus.db, sid)
    assert st.current_state == "Done"

    n_before = len(fake_bus.published)
    # A further event delivered to the now-Done session must be a no-op:
    # ``tick`` short-circuits terminal states before any dispatch.
    await eng.tick(
        _Evt("report.ready", sid, "ReportingAgent", payload={"report_ref": "some_other_ref"})
    )

    st_after = await load_session_state(fake_bus.db, sid)
    assert st_after.current_state == "Done"
    assert st_after.final_report_ref == report_ref  # unchanged
    assert fake_bus.published[n_before:] == []
