"""Unit tests for DataAnalysisAgent + LabInTheLoop milestone Task 6:
AnalyzingExternalData -- request analysis, route on consensus, ACTIVATE
counter M (data_analysis_consensus_round[external_experiment_id],
threshold 2, with_reason data_analysis_consensus_failed).

Covers:
  - AnalyzingExternalData's entry action (``data_analysis.requested``):
    payload carries ``external_data_upload_ref`` (the latest upload),
    ``external_experiment_package_ref``, and ``strictness`` (the current M
    count for ``current_external_experiment_id`` -- 0 first time, higher on
    a diverge-retry).
  - ``_on_AnalyzingExternalData_data_analysis_ready`` (dispatch for
    ``data_analysis.ready``):
      - stores the report ref (singular + appended plural).
      - ``consensus_status: "converged"`` -> reset M[eid] (delete key);
        AwaitingUserDecision (clean no-entry-action pause -- NOT
        EvaluatingExperiment/ConsolidatingMemory/Reporting, whose entry
        actions read benchmark-only refs the lab path never sets).
      - ``consensus_status: "diverged"`` -> bump M[eid]; below threshold ->
        self-transition back to AnalyzingExternalData (re-emits
        data_analysis.requested with a stricter ``strictness``); at/above
        threshold -> set current_moss_reason to "data_analysis_consensus_failed"
        and transition to MossIntervening (no entry action -> pauses).
      - M is keyed per ``current_external_experiment_id`` (a different eid
        has its own independent count).

Reuses FakeBusClient/fake_bus from test_engine_min.py (does NOT
auto-deliver; drives the engine directly via ``eng.tick(event)``), mirroring
test_engine_external_experiment_advance.py's patterns.
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
    event_id: str | None = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _seed_analyzing_external_data_session(
    eng: OrchestratorEngine,
    sid: str,
    *,
    current_external_experiment_id: str = "ext1",
    data_analysis_consensus_round: dict[str, int] | None = None,
) -> SessionState:
    state = SessionState(
        session_id=sid,
        user_id="u1",
        current_state="AnalyzingExternalData",
        current_external_experiment_id=current_external_experiment_id,
        external_data_upload_refs=[f"upload://{sid}/1", f"upload://{sid}/2"],
        external_experiment_package_ref=f"package://{sid}/1",
        data_analysis_consensus_round=data_analysis_consensus_round or {},
    )
    await save_session_state(eng.client.db, state, auto_commit=True)
    return state


# ---------------------------------------------------------------------------
# (a)-(b): entry action
# ---------------------------------------------------------------------------


async def test_entry_action_emits_data_analysis_requested_strictness_zero(fake_bus):
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = "sess-a"
    state = await _seed_analyzing_external_data_session(eng, sid)

    await eng._execute_entry_action(state)

    assert len(fake_bus.published) == 1
    evt = fake_bus.published[0]
    assert evt.event_type == "data_analysis.requested"
    assert evt.payload["external_data_upload_ref"] == state.external_data_upload_refs[-1]
    assert evt.payload["external_experiment_package_ref"] == state.external_experiment_package_ref
    assert evt.payload["strictness"] == 0


async def test_entry_action_strictness_reflects_current_m_count(fake_bus):
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = "sess-b"
    state = await _seed_analyzing_external_data_session(
        eng, sid, current_external_experiment_id="ext1",
        data_analysis_consensus_round={"ext1": 1},
    )

    await eng._execute_entry_action(state)

    evt = fake_bus.published[0]
    assert evt.payload["strictness"] == 1


# ---------------------------------------------------------------------------
# (c): converged -> AwaitingUserDecision, report ref stored, M deleted
# ---------------------------------------------------------------------------


async def test_data_analysis_ready_converged_advances_to_awaiting_user_decision(fake_bus):
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = "sess-c"
    await _seed_analyzing_external_data_session(
        eng, sid, current_external_experiment_id="ext1",
        data_analysis_consensus_round={"ext1": 1},
    )

    await eng.tick(
        _Evt(
            "data_analysis.ready",
            sid,
            "DataAnalysisAgent",
            payload={"external_data_analysis_report_ref": "r1", "consensus_status": "converged"},
        )
    )

    st = await load_session_state(fake_bus.db, sid)
    assert st is not None
    assert st.external_data_analysis_report_ref == "r1"
    assert st.external_data_analysis_report_refs == ["r1"]
    assert "ext1" not in st.data_analysis_consensus_round
    assert st.current_state == "AwaitingUserDecision"


# ---------------------------------------------------------------------------
# (d)-(e): diverged -> bump M; below threshold retries, at threshold MOSS
# ---------------------------------------------------------------------------


async def test_data_analysis_ready_diverged_below_threshold_retries(fake_bus):
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = "sess-d"
    await _seed_analyzing_external_data_session(eng, sid, current_external_experiment_id="ext1")

    await eng.tick(
        _Evt(
            "data_analysis.ready",
            sid,
            "DataAnalysisAgent",
            payload={"external_data_analysis_report_ref": "r2", "consensus_status": "diverged"},
        )
    )

    st = await load_session_state(fake_bus.db, sid)
    assert st is not None
    assert st.data_analysis_consensus_round.get("ext1") == 1
    assert st.current_state == "AnalyzingExternalData"


async def test_data_analysis_ready_diverged_at_threshold_moss_intervenes(fake_bus):
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = "sess-e"
    await _seed_analyzing_external_data_session(
        eng, sid, current_external_experiment_id="ext1",
        data_analysis_consensus_round={"ext1": 1},
    )

    await eng.tick(
        _Evt(
            "data_analysis.ready",
            sid,
            "DataAnalysisAgent",
            payload={"external_data_analysis_report_ref": "r3", "consensus_status": "diverged"},
        )
    )

    st = await load_session_state(fake_bus.db, sid)
    assert st is not None
    assert st.data_analysis_consensus_round.get("ext1") == 2
    assert st.current_state == "MossIntervening"
    assert st.current_moss_reason == "data_analysis_consensus_failed"


# ---------------------------------------------------------------------------
# (f): M key is per current_external_experiment_id
# ---------------------------------------------------------------------------


async def test_data_analysis_ready_diverged_m_keyed_per_experiment_id(fake_bus):
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = "sess-f"
    await _seed_analyzing_external_data_session(
        eng, sid, current_external_experiment_id="ext2",
        data_analysis_consensus_round={"ext1": 1},
    )

    await eng.tick(
        _Evt(
            "data_analysis.ready",
            sid,
            "DataAnalysisAgent",
            payload={"external_data_analysis_report_ref": "r4", "consensus_status": "diverged"},
        )
    )

    st = await load_session_state(fake_bus.db, sid)
    assert st is not None
    assert st.data_analysis_consensus_round.get("ext1") == 1
    assert st.data_analysis_consensus_round.get("ext2") == 1
    assert st.current_state == "AnalyzingExternalData"
