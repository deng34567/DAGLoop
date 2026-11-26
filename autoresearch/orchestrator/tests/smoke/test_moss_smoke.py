"""MossIntervening -> MOSS -> route-back integration smoke (MOSS Step 1 Task
6): single session per scenario, mocked LLM, ``spec_store`` engine, no Redis,
no real execution, no concurrency -- PROVES A MossIntervening PAUSE RESOLVES
VIA A LEGAL MOSS ACTION AND ROUTES BACK.

Two independent scenarios, each its own session on its own ``DeliveringBus``
(the ``bus``/``store`` fixtures are function-scoped -- fresh sqlite + broker
state + tmp SpecStore per test -- so the two scenarios' FakeLLM queues never
interact):

Scenario 1 (route back + counter reset, direction_thrashing/rewrite_framing):
a session constructed directly in ``MossIntervening`` with
``current_moss_reason="direction_thrashing"`` and
``direction_critique_round=2`` (so the ``"A reset to 0"`` counter_update on
``rewrite_framing`` is observable). ``eng.resume_to_current_state`` re-emits
MossIntervening's EXISTING entry action (``moss.intervention_requested{reason:
"direction_thrashing"}``) -> (delivered) -> MOSS (FakeLLM canned action
``"rewrite_framing"``) -> ``moss.intervention_completed`` -> (delivered) ->
orchestrator's ``_on_MossIntervening_moss_intervention_completed`` looks up
(direction_thrashing, rewrite_framing) -> ``next_state=ModelingProblem``,
``counter_updates=["A reset to 0"]``, appends a MossInterventionRecord,
clears ``current_moss_reason``, transitions -> ModelingProblem's EXISTING
entry action fires ``problem_modeling.model_requested`` (no
ProblemModelingAgent wired in this smoke) -> PAUSES there.

Scenario 2 (discard + delete counter key, result_suspicious/discard_result):
a session constructed directly in ``MossIntervening`` with
``current_moss_reason="result_suspicious"``, ``current_result_id="r1"``,
``result_verification_fail_round={"r1": 2}`` (+ ``evaluation_report_refs`` /
``run_record_refs`` pre-populated -- ConsolidatingMemory's EXISTING entry
action reads ``[-1]`` off both, mirroring
``test_engine_moss_advance.py``'s ``_seed_moss_intervening_session``). MOSS's
FakeLLM canned action ``"discard_result"`` -> orchestrator looks up
(result_suspicious, discard_result) -> ``next_state=ConsolidatingMemory``,
``counter_updates=["G delete key current_result_id"]`` (deletes
``state.result_verification_fail_round["r1"]``) -> ConsolidatingMemory's
EXISTING entry action fires ``memory.consolidation_requested`` (no
MemoryAgent wired) -> PAUSES there.

No concurrency, no Redis, no real LLM/execution call anywhere in this file
(each scenario's MOSS FakeLLM has exactly ONE canned response). Each
scenario calls ``resume_to_current_state`` exactly once -- a second call
would re-emit the entry action and break the "exactly one" event-count
assertions.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import aiosqlite
import pytest
import yaml
from jsonschema import Draft202012Validator

from autoresearch.agents.moss_agent.service import build_moss_service
from autoresearch.agents.moss_agent.tests.conftest import FakeLLM
from autoresearch.orchestrator.engine import OrchestratorEngine
from autoresearch.orchestrator.session_state import SessionState, load_session_state, save_session_state
from autoresearch.spec_store import SpecStore
from autoresearch.testing.delivering_bus import DeliveringBus

SM_ROOT = Path(__file__).resolve().parents[2] / "state-machine"
MOSS_CONTRACTS_DIR = Path(__file__).resolve().parents[3] / "agents" / "moss_agent" / "data-contracts"
MOSS_DECISION_SCHEMA_PATH = MOSS_CONTRACTS_DIR / "moss_decision.schema.yaml"


def _validator(path: Path) -> Draft202012Validator:
    schema = yaml.safe_load(path.read_text(encoding="utf-8"))
    return Draft202012Validator(schema)


MOSS_DECISION_VALIDATOR = _validator(MOSS_DECISION_SCHEMA_PATH)


def _json(obj: dict) -> str:
    return json.dumps(obj)


@pytest.fixture
async def bus(tmp_path):
    conn = await aiosqlite.connect(str(tmp_path / "moss_smoke.db"))
    try:
        yield DeliveringBus(conn)
    finally:
        await conn.close()


@pytest.fixture
def store(tmp_path):
    s = SpecStore(tmp_path / "store")
    yield s
    s.close()


# ===========================================================================
# Scenario 1: MossIntervening(direction_thrashing) -> MOSS(rewrite_framing)
# -> ModelingProblem (route back + "A reset to 0" counter reset).
# ===========================================================================


async def test_moss_intervening_rewrite_framing_routes_back_to_modeling_problem(bus, store):
    # --- wire OrchestratorEngine (WITH spec_store) + MOSS on the SAME bus,
    # each with its own emitter identity.
    eng = OrchestratorEngine(
        client=bus.view("ResearchOrchestrator"), skill_root=SM_ROOT, spec_store=store
    )
    await eng.start()

    moss_llm = FakeLLM([_json({"action": "rewrite_framing", "rationale": "thrashed twice"})])
    build_moss_service(client=bus.view("MOSS"), spec_store=store, llm_client=moss_llm)

    # --- Isolated construction: a session directly in MossIntervening, with
    # direction_critique_round=2 so the "A reset to 0" counter_update is
    # observable.
    sid = str(uuid.uuid4())
    state = SessionState(
        session_id=sid,
        user_id="u1",
        current_state="MossIntervening",
        current_moss_reason="direction_thrashing",
        direction_critique_round=2,
    )
    await save_session_state(bus.db, state, auto_commit=True)

    # --- Drive it: resume_to_current_state re-emits the MossIntervening
    # entry action and cascades to ModelingProblem, where it pauses (no
    # ProblemModelingAgent wired). Called exactly once.
    result = await eng.resume_to_current_state(sid)
    assert result["resumed"] is True

    # --- Assertions ---------------------------------------------------------
    final_state = await load_session_state(bus.db, sid)
    assert final_state is not None

    assert final_state.current_state == "ModelingProblem"
    assert final_state.current_moss_reason is None
    assert final_state.direction_critique_round == 0

    assert len(final_state.moss_interventions) == 1
    record = final_state.moss_interventions[0]
    assert record["reason"] == "direction_thrashing"
    assert record["action"] == "rewrite_framing"
    assert record["degraded"] is False

    moss_decision = store.get(record["moss_intervention_ref"])
    errors = list(MOSS_DECISION_VALIDATOR.iter_errors(moss_decision))
    assert errors == [], [e.message for e in errors]
    assert moss_decision["action"] == "rewrite_framing"

    completed_events = [e for e in bus.published if e.event_type == "moss.intervention_completed"]
    assert len(completed_events) == 1
    assert completed_events[0].session_id == sid

    assert not [e for e in bus.published if e.event_type == "agent.failed"]


# ===========================================================================
# Scenario 2: MossIntervening(result_suspicious) -> MOSS(discard_result) ->
# ConsolidatingMemory (discard + "G delete key current_result_id").
# ===========================================================================


async def test_moss_intervening_discard_result_routes_back_to_consolidating_memory(bus, store):
    eng = OrchestratorEngine(
        client=bus.view("ResearchOrchestrator"), skill_root=SM_ROOT, spec_store=store
    )
    await eng.start()

    moss_llm = FakeLLM([_json({"action": "discard_result", "rationale": "suspicious metric"})])
    build_moss_service(client=bus.view("MOSS"), spec_store=store, llm_client=moss_llm)

    # --- Isolated construction: a session directly in MossIntervening.
    # evaluation_report_refs / run_record_refs are pre-populated because
    # ConsolidatingMemory's EXISTING entry action reads [-1] off both.
    sid = str(uuid.uuid4())
    state = SessionState(
        session_id=sid,
        user_id="u1",
        current_state="MossIntervening",
        current_moss_reason="result_suspicious",
        current_result_id="r1",
        result_verification_fail_round={"r1": 2},
        evaluation_report_refs=[f"evaluation_report://{sid}/final"],
        run_record_refs=[f"run_record://{sid}/final"],
    )
    await save_session_state(bus.db, state, auto_commit=True)

    result = await eng.resume_to_current_state(sid)
    assert result["resumed"] is True

    # --- Assertions ---------------------------------------------------------
    final_state = await load_session_state(bus.db, sid)
    assert final_state is not None

    assert final_state.current_state == "ConsolidatingMemory"
    assert "r1" not in final_state.result_verification_fail_round
    assert final_state.current_moss_reason is None

    assert len(final_state.moss_interventions) == 1
    record = final_state.moss_interventions[0]
    assert record["reason"] == "result_suspicious"
    assert record["action"] == "discard_result"
    assert record["degraded"] is False

    moss_decision = store.get(record["moss_intervention_ref"])
    errors = list(MOSS_DECISION_VALIDATOR.iter_errors(moss_decision))
    assert errors == [], [e.message for e in errors]

    completed_events = [e for e in bus.published if e.event_type == "moss.intervention_completed"]
    assert len(completed_events) == 1
    assert completed_events[0].session_id == sid

    assert not [e for e in bus.published if e.event_type == "agent.failed"]
