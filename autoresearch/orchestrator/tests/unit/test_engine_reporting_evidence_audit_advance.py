"""Unit tests for the EvidenceAuditAgent Mode C citation-regression audit
wired into ``Reporting`` (EvidenceAuditAgent milestone, Task 5).

Covers:
  (a) Reporting's entry action now emits
      ``evidence.citation_audit_requested`` (NOT ``report.requested``).
  (b) ``evidence.citation_audited{citation_audit_report_ref}`` ->
      ``state.citation_audit_report_ref`` is set, ``report.requested`` is
      published IN-STATE (carrying the citation_audit_report_ref too), and
      the session STAYS in Reporting (no ``_transition``).
  (c) the existing ``_on_Reporting_report_ready`` handler still works
      unchanged after (b): a ``report.ready{report_ref}`` routes to Done.

Mirrors test_engine_evidence_audit_advance.py's patterns exactly (same
FakeBusClient, does NOT auto-deliver; drives the engine directly via
``eng.tick(event)``).
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


async def _seed_reporting_session(
    eng: OrchestratorEngine,
    sid: str,
    *,
    research_memory_ref: str | None = None,
    research_spec_ref: str | None = None,
) -> None:
    """Persist a session in Reporting with its upstream refs set (mirrors
    test_engine_reporting_advance.py's helper)."""
    state = SessionState(
        session_id=sid,
        user_id="u1",
        current_state="Reporting",
        research_memory_ref=research_memory_ref or f"research_memory://{sid}/final",
        research_spec_ref=research_spec_ref or f"research_spec://{sid}/final",
    )
    await save_session_state(eng.client.db, state, auto_commit=True)


# ---------------------------------------------------------------------------
# (a) entry action emits evidence.citation_audit_requested
# ---------------------------------------------------------------------------


async def test_reporting_entry_emits_evidence_citation_audit_requested(fake_bus):
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
    assert emitted[0].session_id == sid
    assert emitted[0].payload == {
        "research_memory_ref": f"research_memory://{sid}/final",
        "research_spec_ref": f"research_spec://{sid}/final",
    }

    # The OLD entry event must NOT be emitted directly on entry anymore.
    assert not [p for p in fake_bus.published if p.event_type == "report.requested"]


# ---------------------------------------------------------------------------
# (b) evidence.citation_audited -> stores ref, publishes report.requested
# IN-STATE (no transition), atomically (single commit).
# ---------------------------------------------------------------------------


async def test_evidence_citation_audited_publishes_report_requested_in_state(fake_bus):
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = "sess-b"
    await _seed_reporting_session(eng, sid)

    n_before = len(fake_bus.published)
    commits_before = fake_bus.db.commit_count
    await eng.tick(
        _Evt(
            "evidence.citation_audited",
            sid,
            "EvidenceAuditAgent",
            payload={"citation_audit_report_ref": "c1"},
        )
    )

    st = await load_session_state(fake_bus.db, sid)
    assert st is not None
    assert st.citation_audit_report_ref == "c1"
    # STAYS in Reporting -- no _transition.
    assert st.current_state == "Reporting"

    new_events = fake_bus.published[n_before:]
    report_requested = [p for p in new_events if p.event_type == "report.requested"]
    assert len(report_requested) == 1
    assert report_requested[0].session_id == sid
    assert report_requested[0].payload == {
        "research_memory_ref": f"research_memory://{sid}/final",
        "research_spec_ref": f"research_spec://{sid}/final",
        "citation_audit_report_ref": "c1",
        "literature_brief_ref": None,
        "method_ref": None,
        "evidence_audit_report_ref": None,
        "novelty_audit_report_ref": None,
    }

    # Atomic: save + publish bundled in exactly one commit.
    assert fake_bus.db.commit_count - commits_before == 1


# ---------------------------------------------------------------------------
# (c) existing report.ready -> Done handler still works unchanged after (b).
# ---------------------------------------------------------------------------


async def test_report_ready_still_routes_to_done_after_citation_audit(fake_bus):
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = "sess-c"
    await _seed_reporting_session(eng, sid)

    await eng.tick(
        _Evt(
            "evidence.citation_audited",
            sid,
            "EvidenceAuditAgent",
            payload={"citation_audit_report_ref": "c1"},
        )
    )

    report_ref = f"final_report://{sid}/v1"
    await eng.tick(
        _Evt("report.ready", sid, "ReportingAgent", payload={"report_ref": report_ref})
    )

    st = await load_session_state(fake_bus.db, sid)
    assert st is not None
    assert st.current_state == "Done"
    assert st.final_report_ref == report_ref
