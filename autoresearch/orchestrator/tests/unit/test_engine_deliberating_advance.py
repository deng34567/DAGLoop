"""Unit tests for the Deliberating state (bounded deep-deliberation step).

  - _route_research_mode (benchmark) enters Deliberating (covered in
    test_engine_lab_track_advance.py); here we cover the exit:
  - _on_Deliberating_deliberation_ready: store deliberation_report_ref and advance to
    SelectingBenchmark, whose entry action (benchmark.selection_requested) now carries the
    deliberation_report_ref. FAIL-OPEN: a minimal report still carries a ref, so the flow
    proceeds regardless.
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


async def _seed_deliberating(eng: OrchestratorEngine, sid: str) -> SessionState:
    state = SessionState(
        session_id=sid, user_id="u1", current_state="Deliberating", mode="benchmark",
        research_spec_ref=f"research_spec://{sid}/revised",
        literature_brief_ref=f"literature_brief://{sid}/final",
    )
    await save_session_state(eng.client.db, state, auto_commit=True)
    return state


async def test_deliberation_ready_stores_ref_and_advances_to_selecting_benchmark(fake_bus):
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = "sess-delib"
    await _seed_deliberating(eng, sid)

    report_ref = f"spec://deliberation_report/{sid}/final"
    n_before = len(fake_bus.published)
    await eng.tick(_Evt("deliberation.ready", sid, "DeliberationAgent",
                        payload={"deliberation_report_ref": report_ref}))

    st = await load_session_state(fake_bus.db, sid)
    assert st is not None
    assert st.current_state == "SelectingBenchmark"
    assert st.deliberation_report_ref == report_ref
    # the SelectingBenchmark entry action fired and carries the deliberation ref
    new = fake_bus.published[n_before:]
    sel = [p for p in new if p.event_type == "benchmark.selection_requested"]
    assert sel and sel[0].payload.get("deliberation_report_ref") == report_ref


async def test_deliberation_ready_without_ref_still_advances(fake_bus):
    # fail-open: even if no ref is present, the flow proceeds to SelectingBenchmark.
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = "sess-delib2"
    await _seed_deliberating(eng, sid)

    await eng.tick(_Evt("deliberation.ready", sid, "DeliberationAgent", payload={}))

    st = await load_session_state(fake_bus.db, sid)
    assert st is not None and st.current_state == "SelectingBenchmark"
    assert st.deliberation_report_ref is None
