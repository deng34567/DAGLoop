"""Unit tests for DataAnalysisAgent + LabInTheLoop milestone Task 5:
PreparingExternalExperiment (orchestrator-only auto-advance) ->
AwaitingExternalExperimentResult (human data upload).

Covers:
  - ``_advance_preparing_external_experiment``: sets
    ``state.external_experiment_package_ref`` from
    ``experiment_plan_refs[-1]`` (the approved plan, reused as the external
    protocol), appends it to ``external_experiment_package_refs``, and
    transitions (no entry action) into AwaitingExternalExperimentResult.
    Chained right after the Task 4 lab-branch ``_transition`` in
    ``_on_ReviewingExperimentPlan_experiment_plan_reviewed`` -- mirrors how
    ``_route_research_mode`` is chained after the some_pass/pass branch.
  - ``_on_AwaitingExternalExperimentResult_external_data_uploaded``
    (dispatch for ``external.data_uploaded``):
      - records the upload ref + current_external_experiment_id always.
      - ``complete: true`` -> AnalyzingExternalData (entry action wired in a
        later milestone task; here it just transitions and pauses).
      - ``complete: false`` -> AwaitingUserDecision (external_data_incomplete;
        v1 doesn't wire that disposition further).
  - regression: the benchmark ReviewingExperimentPlan pass branch is
    UNCHANGED (still lands on ImplementingExperiment, no auto-advance to any
    external state).

Reuses FakeBusClient/fake_bus from test_engine_min.py (does NOT auto-deliver;
drives the engine directly via ``eng.tick(event)`` / helper calls), mirroring
test_engine_lab_track_advance.py's patterns.
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


async def _seed_reviewing_experiment_plan_session(
    eng: OrchestratorEngine, sid: str, *, mode: str, hypothesis_id: str = "h1"
) -> str:
    """Persist a session in ReviewingExperimentPlan with its upstream refs
    set (mirrors test_engine_lab_track_advance.py's helper)."""
    experiment_plan_ref = f"experiment_plan://{sid}/draft1"
    state = SessionState(
        session_id=sid,
        user_id="u1",
        current_state="ReviewingExperimentPlan",
        mode=mode,
        current_hypothesis_id=hypothesis_id,
        verifier_spec_ref=f"verifier_spec://{sid}/final",
        experiment_plan_refs=[experiment_plan_ref],
    )
    await save_session_state(eng.client.db, state, auto_commit=True)
    return experiment_plan_ref


async def _seed_awaiting_external_experiment_result_session(
    eng: OrchestratorEngine, sid: str, *, package_ref: str = "p1"
) -> SessionState:
    state = SessionState(
        session_id=sid,
        user_id="u1",
        current_state="AwaitingExternalExperimentResult",
        external_experiment_package_ref=package_ref,
        external_experiment_package_refs=[package_ref],
    )
    await save_session_state(eng.client.db, state, auto_commit=True)
    return state


# ---------------------------------------------------------------------------
# (a): lab pass branch auto-advances PreparingExternalExperiment ->
# AwaitingExternalExperimentResult
# ---------------------------------------------------------------------------


async def test_experiment_plan_reviewed_pass_lab_mode_auto_advances_to_awaiting_external_result(fake_bus):
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = "sess-a"
    plan_ref = await _seed_reviewing_experiment_plan_session(eng, sid, mode="lab_in_the_loop")

    await eng.tick(
        _Evt(
            "experiment.plan_reviewed",
            sid,
            "ExperimentReviewAgent",
            payload={"review_report_ref": f"review_report://{sid}/final", "verdict": "pass"},
        )
    )

    st = await load_session_state(fake_bus.db, sid)
    assert st is not None
    assert st.external_experiment_package_ref == plan_ref
    assert plan_ref in st.external_experiment_package_refs
    assert st.current_state == "AwaitingExternalExperimentResult"


# ---------------------------------------------------------------------------
# (b)-(c): external.data_uploaded at AwaitingExternalExperimentResult
# ---------------------------------------------------------------------------


async def test_external_data_uploaded_complete_advances_to_analyzing_external_data(fake_bus):
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = "sess-b"
    await _seed_awaiting_external_experiment_result_session(eng, sid)

    await eng.tick(
        _Evt(
            "external.data_uploaded",
            sid,
            "Lab",
            payload={
                "external_data_upload_ref": "u1",
                "external_experiment_id": "ext1",
                "complete": True,
            },
        )
    )

    st = await load_session_state(fake_bus.db, sid)
    assert st is not None
    assert st.current_external_experiment_id == "ext1"
    assert "u1" in st.external_data_upload_refs
    assert st.current_state == "AnalyzingExternalData"


async def test_external_data_uploaded_incomplete_advances_to_awaiting_user_decision(fake_bus):
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = "sess-c"
    await _seed_awaiting_external_experiment_result_session(eng, sid)

    await eng.tick(
        _Evt(
            "external.data_uploaded",
            sid,
            "Lab",
            payload={
                "external_data_upload_ref": "u2",
                "external_experiment_id": "ext2",
                "complete": False,
            },
        )
    )

    st = await load_session_state(fake_bus.db, sid)
    assert st is not None
    assert st.current_external_experiment_id == "ext2"
    assert "u2" in st.external_data_upload_refs
    assert st.current_state == "AwaitingUserDecision"


# ---------------------------------------------------------------------------
# (d): regression -- benchmark ReviewingExperimentPlan pass branch unchanged
# ---------------------------------------------------------------------------


async def test_experiment_plan_reviewed_pass_benchmark_mode_unchanged(fake_bus):
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = "sess-d"
    await _seed_reviewing_experiment_plan_session(eng, sid, mode="benchmark")

    await eng.tick(
        _Evt(
            "experiment.plan_reviewed",
            sid,
            "ExperimentReviewAgent",
            payload={"review_report_ref": f"review_report://{sid}/final", "verdict": "pass"},
        )
    )

    st = await load_session_state(fake_bus.db, sid)
    assert st is not None
    assert st.current_state == "ImplementingExperiment"
    assert st.external_experiment_package_ref is None
    assert st.external_experiment_package_refs == []
