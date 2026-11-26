"""DataAnalysisAgent + LabInTheLoop milestone Task 7: two dedicated
integration smokes, single session, no Redis, no real LLM, no real execution
(no notebooks) -- proving the two headline behaviors this milestone adds on
top of the (already-covered) ``data_analysis.requested -> data_analysis.ready``
functional round-trip
(``autoresearch/agents/data_analysis_agent/tests/test_functional.py``):

1. ``test_lab_track_converges_to_awaiting_user_decision``: a human data upload
   at AwaitingExternalExperimentResult drives the session into
   AnalyzingExternalData, whose entry action requests analysis;
   DataAnalysisAgent returns N mostly-agreeing trajectories, the DETERMINISTIC
   ``_consensus_status`` computes "converged", and the orchestrator routes to
   a CLEAN AwaitingUserDecision pause (NOT EvaluatingExperiment, whose entry
   action reads benchmark-only refs the lab path never set and would crash).
   The external-path evaluate/report adaptation is DEFERRED.

2. ``test_consensus_divergence_trips_counter_m_into_moss``: counter M
   (``data_analysis_consensus_round``, map keyed by external_experiment_id,
   threshold 2) was UNREACHABLE before this milestone (nothing ever produced a
   ``consensus_status="diverged"`` analysis). With M[eid] pre-seeded at 1 (the
   value a prior diverge would have left), a second diverging analysis bumps M
   to 2 (== threshold) and trips MossIntervening with
   ``state.current_moss_reason == "data_analysis_consensus_failed"``.

The lab-track mode-branches (AwaitingUserDecision proceed_out_of_scope,
RoutingResearchMode -> GeneratingHypotheses, ReviewingExperimentPlan(pass) ->
PreparingExternalExperiment) are unit-tested in
``test_engine_lab_track_advance.py`` / ``test_engine_external_experiment_advance.py``.

LabInTheLoop is a RESERVED / out-of-scope mode; every mode-branch defaults to
benchmark, so the benchmark main loop is byte-for-byte unchanged. No
concurrency, no Redis, no real LLM call, no real execution backend anywhere.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import aiosqlite
import pytest
import yaml
from jsonschema import Draft202012Validator

from autoresearch.agents.data_analysis_agent.service import build_data_analysis_service
from autoresearch.agents.data_analysis_agent.tests.conftest import FakeLLM as DataAnalysisFakeLLM
from autoresearch.orchestrator.engine import OrchestratorEngine
from autoresearch.orchestrator.session_state import (
    SessionState,
    load_session_state,
    save_session_state,
)
from autoresearch.spec_store import SpecStore
from autoresearch.testing.delivering_bus import DeliveringBus

SM_ROOT = Path(__file__).resolve().parents[2] / "state-machine"

_CONTRACTS_DIR = (
    Path(__file__).resolve().parents[3]
    / "agents"
    / "data_analysis_agent"
    / "data-contracts"
)
_REPORT_VALIDATOR = Draft202012Validator(
    yaml.safe_load(
        (_CONTRACTS_DIR / "external_data_analysis_report.schema.yaml").read_text(
            encoding="utf-8"
        )
    )
)


def _dumps(obj: dict) -> str:
    return json.dumps(obj)


@pytest.fixture
async def bus(tmp_path):
    conn = await aiosqlite.connect(str(tmp_path / "lab_smoke.db"))
    try:
        yield DeliveringBus(conn)
    finally:
        await conn.close()


@pytest.fixture
def store(tmp_path):
    s = SpecStore(tmp_path / "store")
    yield s
    s.close()


EXTERNAL_DATA_UPLOAD = {"description": "CSV of 30 external trial runs, primary endpoint X."}
EXTERNAL_EXPERIMENT_PACKAGE = {"protocol_summary": "RCT protocol, n=30, primary endpoint X."}

# 4 of 5 trajectories agree -> top_share 0.8 >= 0.6 threshold -> "converged"
# (the aggregate verdict is computed deterministically; the LLM never emits
# consensus_status).
CONVERGING_ANALYSIS = {
    "qc_summary": "All trajectories passed QC.",
    "trajectories": [
        {"trajectory_id": "t1", "conclusion": "Effect confirmed.", "notes": None},
        {"trajectory_id": "t2", "conclusion": "Effect confirmed.", "notes": None},
        {"trajectory_id": "t3", "conclusion": "Effect confirmed.", "notes": None},
        {"trajectory_id": "t4", "conclusion": "Effect confirmed.", "notes": None},
        {"trajectory_id": "t5", "conclusion": "No effect detected.", "notes": None},
    ],
    "consensus_conclusion": "Effect is real and reproducible.",
    "divergence_points": ["t5 found no effect"],
    "robustness": "high",
    "need_human_review": False,
}

# Evenly split -> top_share 0.4 < threshold (0.6 at strictness 0, 0.7 at
# strictness 1) -> "diverged".
DIVERGING_ANALYSIS = {
    "qc_summary": "QC passed; conclusions do not agree.",
    "trajectories": [
        {"trajectory_id": "t1", "conclusion": "Effect confirmed.", "notes": None},
        {"trajectory_id": "t2", "conclusion": "Effect confirmed.", "notes": None},
        {"trajectory_id": "t3", "conclusion": "No effect detected.", "notes": None},
        {"trajectory_id": "t4", "conclusion": "No effect detected.", "notes": None},
        {"trajectory_id": "t5", "conclusion": "Inconclusive.", "notes": None},
    ],
    "consensus_conclusion": "No agreement across trajectories.",
    "divergence_points": ["effect vs no-effect split 2/2"],
    "robustness": "low",
    "need_human_review": True,
}


def _seed_refs(store, sid):
    upload_ref = store.put(
        kind="audit",
        session_id=sid,
        key="external-data-upload",
        obj=EXTERNAL_DATA_UPLOAD,
        emitter="Human",
    )
    package_ref = store.put(
        kind="experiment_plan",
        session_id=sid,
        key="external-experiment-package",
        obj=EXTERNAL_EXPERIMENT_PACKAGE,
        emitter="ExperimentPlannerAgent",
    )
    return upload_ref, package_ref


# ===========================================================================
# (1) human upload -> AnalyzingExternalData -> DataAnalysisAgent (converging)
#     -> converged -> AwaitingUserDecision (clean pause).
# ===========================================================================
async def test_lab_track_converges_to_awaiting_user_decision(bus, store):
    eng = OrchestratorEngine(
        client=bus.view("ResearchOrchestrator"), skill_root=SM_ROOT, spec_store=store
    )
    await eng.start()
    build_data_analysis_service(
        client=bus.view("DataAnalysisAgent"),
        spec_store=store,
        llm_client=DataAnalysisFakeLLM([_dumps(CONVERGING_ANALYSIS)]),
    )

    sid = str(uuid.uuid4())
    upload_ref, package_ref = _seed_refs(store, sid)
    # Session sits at AwaitingExternalExperimentResult (lab track, awaiting the
    # human data upload); PreparingExternalExperiment already set the package ref.
    state = SessionState(
        session_id=sid,
        user_id="u1",
        current_state="AwaitingExternalExperimentResult",
        mode="lab_in_the_loop",
        external_experiment_package_ref=package_ref,
    )
    await save_session_state(bus.db, state, auto_commit=True)

    # The human uploads complete data -> orchestrator (_on_AwaitingExternal...)
    # records the upload + external_experiment_id, transitions to
    # AnalyzingExternalData, whose entry action emits data_analysis.requested
    # (external_data_upload_ref = the just-uploaded ref, strictness 0) ->
    # DataAnalysisAgent produces N converging trajectories, deterministic
    # consensus == "converged" -> data_analysis.ready -> orchestrator resets
    # M[ext1] and routes to a clean AwaitingUserDecision pause.
    await bus.view("Human").publish(
        event_type="external.data_uploaded",
        session_id=sid,
        payload={
            "external_data_upload_ref": upload_ref,
            "external_experiment_id": "ext1",
            "complete": True,
        },
    )

    final_state = await load_session_state(bus.db, sid)
    assert final_state is not None
    assert final_state.current_state == "AwaitingUserDecision"
    assert final_state.current_external_experiment_id == "ext1"
    assert "ext1" not in final_state.data_analysis_consensus_round  # M reset on converge

    ready = [e for e in bus.published if e.event_type == "data_analysis.ready"]
    assert len(ready) == 1
    assert ready[0].payload["consensus_status"] == "converged"

    report = store.get(final_state.external_data_analysis_report_ref)
    assert list(_REPORT_VALIDATOR.iter_errors(report)) == []
    assert report["consensus_status"] == "converged"

    assert not [e for e in bus.published if e.event_type == "agent.failed"]


# ===========================================================================
# (2) AnalyzingExternalData with M[ext1]=1 -> DataAnalysisAgent (diverging)
#     -> diverged -> M[ext1]=2 (== threshold) -> MossIntervening.
#     Proves the previously-unreachable counter M + data_analysis_consensus_failed.
# ===========================================================================
async def test_consensus_divergence_trips_counter_m_into_moss(bus, store):
    eng = OrchestratorEngine(
        client=bus.view("ResearchOrchestrator"), skill_root=SM_ROOT, spec_store=store
    )
    await eng.start()
    build_data_analysis_service(
        client=bus.view("DataAnalysisAgent"),
        spec_store=store,
        llm_client=DataAnalysisFakeLLM([_dumps(DIVERGING_ANALYSIS)]),
    )

    sid = str(uuid.uuid4())
    upload_ref, package_ref = _seed_refs(store, sid)
    # Session already in AnalyzingExternalData, with M[ext1] pre-seeded at 1
    # (the value a prior diverge round would have left).
    state = SessionState(
        session_id=sid,
        user_id="u1",
        current_state="AnalyzingExternalData",
        mode="lab_in_the_loop",
        external_experiment_package_ref=package_ref,
        external_data_upload_refs=[upload_ref],
        current_external_experiment_id="ext1",
        data_analysis_consensus_round={"ext1": 1},
    )
    await save_session_state(bus.db, state, auto_commit=True)

    # resume re-emits AnalyzingExternalData's entry action (data_analysis.requested,
    # strictness = M[ext1] = 1) -> DataAnalysisAgent produces evenly-split
    # trajectories, deterministic consensus == "diverged" -> data_analysis.ready
    # -> orchestrator bumps M[ext1] 1->2 (== threshold 2) -> MossIntervening.
    result = await eng.resume_to_current_state(sid)
    assert result["resumed"] is True

    final_state = await load_session_state(bus.db, sid)
    assert final_state is not None
    assert final_state.current_state == "MossIntervening"
    assert final_state.current_moss_reason == "data_analysis_consensus_failed"
    assert final_state.data_analysis_consensus_round["ext1"] == 2

    ready = [e for e in bus.published if e.event_type == "data_analysis.ready"]
    assert len(ready) == 1
    assert ready[0].payload["consensus_status"] == "diverged"

    assert not [e for e in bus.published if e.event_type == "agent.failed"]
