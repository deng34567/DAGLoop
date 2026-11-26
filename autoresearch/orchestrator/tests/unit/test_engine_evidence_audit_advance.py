"""Unit tests for the EvidenceAuditAgent Mode A co-audit wired into
``AuditingLiteratureNovelty`` (EvidenceAuditAgent milestone, Task 4).

Covers (post-3.3, the audit now runs after GeneratingMethod and judges the METHOD's novelty):
  (a) AuditingLiteratureNovelty's entry action emits ``evidence.audit_requested`` (NOT
      ``novelty.audit_requested``), carrying the ``method_ref``.
  (b) ``evidence.audited{evidence_audit_report_ref}`` -> ``state.evidence_audit_report_ref``
      is set, ``novelty.audit_requested`` is published IN-STATE (carrying evidence_audit_report_ref
      + method_ref), and the session STAYS in AuditingLiteratureNovelty (no ``_transition``).
  (c) the K-routing handler still works: a ``novelty.audit_ready{verdict:"pass"}`` routes to
      ConfirmingMethod (the human method gate).

The session is seeded DIRECTLY into AuditingLiteratureNovelty (reaching it via the real path now
requires driving through method generation).

Reuses FakeBusClient from test_engine_min.py (does NOT auto-deliver; the test
drives the engine directly via ``eng.tick(event)``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pytest

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


async def _advance_to_auditing_literature_novelty(
    eng: OrchestratorEngine, sid: str
) -> tuple[str, str, str, str]:
    """Seed a session paused in AuditingLiteratureNovelty (3.3: reached after GeneratingMethod) and
    run its entry action (as a real transition would), so ``evidence.audit_requested`` is emitted.
    Returns (research_spec_ref, literature_brief_ref, spec_interpretation_ref, method_ref)."""
    research_spec_ref = f"research_spec://{sid}/revised"
    literature_brief_ref = f"spec://literature_brief/{sid}/final"
    spec_interpretation_ref = f"spec_interpretation://{sid}/final"
    method_ref = f"method://{sid}/v1"
    state = SessionState(
        session_id=sid,
        user_id="u1",
        current_state="AuditingLiteratureNovelty",
        research_spec_ref=research_spec_ref,
        literature_brief_ref=literature_brief_ref,
        spec_interpretation_ref=spec_interpretation_ref,
        problem_model_ref=f"problem_model://{sid}/final",
        method_ref=method_ref,
    )
    await save_session_state(eng.client.db, state, auto_commit=True)
    await eng._execute_entry_action(state)
    return research_spec_ref, literature_brief_ref, spec_interpretation_ref, method_ref


# ---------------------------------------------------------------------------
# (a) entry action emits evidence.audit_requested
# ---------------------------------------------------------------------------


async def test_auditing_literature_novelty_entry_emits_evidence_audit_requested(fake_bus):
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = await eng.start_session("goal", user_id="u1")
    research_spec_ref, literature_brief_ref, spec_interpretation_ref, method_ref = (
        await _advance_to_auditing_literature_novelty(eng, sid)
    )

    st = await load_session_state(fake_bus.db, sid)
    assert st is not None
    assert st.current_state == "AuditingLiteratureNovelty"

    evidence_events = [p for p in fake_bus.published if p.event_type == "evidence.audit_requested"]
    assert len(evidence_events) == 1
    assert evidence_events[0].session_id == sid
    assert evidence_events[0].payload == {
        "research_spec_ref": research_spec_ref,
        "literature_brief_ref": literature_brief_ref,
        "spec_interpretation_ref": spec_interpretation_ref,
        "method_ref": method_ref,   # 3.3: the audit judges the method's novelty
    }

    # The OLD entry event must NOT be emitted directly on entry anymore.
    assert not [p for p in fake_bus.published if p.event_type == "novelty.audit_requested"]


# ---------------------------------------------------------------------------
# (b) evidence.audited -> stores ref, publishes novelty.audit_requested
# IN-STATE (no transition), atomically (single commit).
# ---------------------------------------------------------------------------


async def test_evidence_audited_publishes_novelty_audit_requested_in_state(fake_bus):
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = await eng.start_session("goal", user_id="u1")
    research_spec_ref, literature_brief_ref, spec_interpretation_ref, method_ref = (
        await _advance_to_auditing_literature_novelty(eng, sid)
    )

    n_before = len(fake_bus.published)
    commits_before = fake_bus.db.commit_count
    await eng.tick(
        _Evt(
            "evidence.audited",
            sid,
            "EvidenceAuditAgent",
            payload={"evidence_audit_report_ref": "e1"},
        )
    )

    st = await load_session_state(fake_bus.db, sid)
    assert st is not None
    assert st.evidence_audit_report_ref == "e1"
    # STAYS in AuditingLiteratureNovelty -- no _transition.
    assert st.current_state == "AuditingLiteratureNovelty"

    new_events = fake_bus.published[n_before:]
    novelty_requested = [p for p in new_events if p.event_type == "novelty.audit_requested"]
    assert len(novelty_requested) == 1
    assert novelty_requested[0].session_id == sid
    assert novelty_requested[0].payload == {
        "research_spec_ref": research_spec_ref,
        "literature_brief_ref": literature_brief_ref,
        "spec_interpretation_ref": spec_interpretation_ref,
        "evidence_audit_report_ref": "e1",
        "method_ref": method_ref,   # 3.3: the audit judges the method's novelty
    }

    # Atomic: save + publish bundled in exactly one commit.
    assert fake_bus.db.commit_count - commits_before == 1


# ---------------------------------------------------------------------------
# (c) the K-routing handler still works after (b): a
# novelty.audit_ready{verdict:"pass"} routes AuditingLiteratureNovelty ->
# ConfirmingMethod (3.3: the human method gate), resetting counter K.
# ---------------------------------------------------------------------------


async def test_novelty_audit_ready_pass_routes_to_confirming_method(fake_bus):
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = await eng.start_session("goal", user_id="u1")
    await _advance_to_auditing_literature_novelty(eng, sid)

    await eng.tick(
        _Evt(
            "evidence.audited",
            sid,
            "EvidenceAuditAgent",
            payload={"evidence_audit_report_ref": "e1"},
        )
    )

    n_before = len(fake_bus.published)
    await eng.tick(
        _Evt(
            "novelty.audit_ready",
            sid,
            "NoveltyAuditAgent",
            payload={"verdict": "pass", "novelty_audit_report_ref": "n1"},
        )
    )

    st = await load_session_state(fake_bus.db, sid)
    assert st is not None
    assert st.current_state == "ConfirmingMethod"   # human gate, no entry action -> pauses
    assert st.novelty_audit_report_ref == "n1"
    assert st.literature_audit_round == 0
    assert fake_bus.published[n_before:] == []


@pytest.mark.parametrize(
    ("mode", "verdict", "expected_state"),
    [
        ("benchmark", "not_novel", "ConfirmingMethod"),
        ("benchmark", "evidence_insufficient", "ConfirmingMethod"),
        ("benchmark", "not_feasible", "GeneratingMethod"),
        ("idea_theory", "not_novel", "GeneratingMethod"),
    ],
)
async def test_benchmark_novelty_is_not_an_effectiveness_gate(
    fake_bus, mode, verdict, expected_state
):
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = await eng.start_session("goal", user_id="u1")
    state = SessionState(
        session_id=sid,
        user_id="u1",
        mode=mode,
        current_state="AuditingLiteratureNovelty",
        method_ref=f"method://{sid}/v1",
    )
    await save_session_state(fake_bus.db, state, auto_commit=True)

    await eng.tick(_Evt(
        "novelty.audit_ready", sid, "NoveltyAuditAgent",
        payload={"verdict": verdict, "novelty_audit_report_ref": "novelty://result"},
    ))

    updated = await load_session_state(fake_bus.db, sid)
    assert updated.current_state == expected_state
    assert updated.novelty_audit_report_ref == "novelty://result"
    assert updated.literature_audit_round == (0 if expected_state == "ConfirmingMethod" else 1)
