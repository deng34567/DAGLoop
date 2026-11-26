"""Unit tests for M9 Task 4: orchestrator wiring for ImplementingExperiment ->
RunningExperiment/ImplementingExperiment/MossIntervening + MAP counter F.

Covers:
  - the ImplementingExperiment entry action (code.implementation_requested).
  - ``_on_ImplementingExperiment_code_implemented`` (the INDEPENDENT protocol
    gate, counter F, implementation_fix_round[experiment_id], threshold 3):
      - reads the FROZEN VerifierSpec from the store (NOT the coder's word)
        and independently recomputes touched-files vs editable_region /
        forbidden_files (default-deny: anything not explicitly in
        editable_region is a violation).
      - no violation -> RunningExperiment (no entry action -> pauses; M10).
        F is NOT reset here (M11 correction): a compliant patch does not
        reset F -- see test_engine_running_advance.py for the actual F reset
        (on RunningExperiment's "succeeded" branch).
      - violation    -> bump F[experiment_id]; below threshold ->
        ImplementingExperiment (re-emits code.implementation_requested, i.e.
        retry); at/above threshold -> MossIntervening with
        current_moss_reason == "protocol_violation".

Reuses FakeBusClient/fake_bus from test_engine_min.py and ``_Evt`` /
``_set_map_counter`` from test_engine_evaluator_advance.py (does NOT
auto-deliver; drives the engine directly via ``eng.tick(event)``), mirroring
the existing advance-test files' patterns. Uses a REAL tmp SpecStore (no
fakes) since the gate's whole point is reading the frozen spec independently
of the CoderAgent's claim.
"""

from __future__ import annotations

from pathlib import Path

from autoresearch.orchestrator.engine import OrchestratorEngine
from autoresearch.orchestrator.session_state import (
    SessionState,
    load_session_state,
    save_session_state,
)
from autoresearch.orchestrator.tests.unit.test_engine_evaluator_advance import (
    _Evt,
    _set_map_counter,
)
from autoresearch.orchestrator.tests.unit.test_engine_min import FakeBusClient, fake_bus  # noqa: F401
from autoresearch.spec_store.store import SpecStore

SM_ROOT = Path(__file__).resolve().parents[2] / "state-machine"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _seed_implementing_experiment_session(
    eng: OrchestratorEngine,
    store: SpecStore,
    sid: str,
    *,
    experiment_id: str = "exp1",
    editable_region: list[str] = ["src/model.py"],
    forbidden_files: list[str] = ["eval/scorer.py"],
) -> str:
    """Persist a session in ImplementingExperiment with upstream refs set, and
    seed the store with the FROZEN VerifierSpec the gate must read. Returns
    the verifier_spec_ref."""
    verifier_spec_ref = store.put(
        kind="verifier_spec",
        session_id=sid,
        key="final",
        obj={"editable_region": editable_region, "forbidden_files": forbidden_files},
        emitter="EvaluatorAgent",
    )
    experiment_plan_ref = f"experiment_plan://{sid}/final"
    state = SessionState(
        session_id=sid,
        user_id="u1",
        current_state="ImplementingExperiment",
        current_experiment_id=experiment_id,
        verifier_spec_ref=verifier_spec_ref,
        experiment_plan_refs=[experiment_plan_ref],
    )
    await save_session_state(eng.client.db, state, auto_commit=True)
    return verifier_spec_ref


def _seed_patch(store: SpecStore, sid: str, key: str, touched_files: list[str]) -> str:
    return store.put(
        kind="experiment_patch",
        session_id=sid,
        key=key,
        obj={"touched_files": touched_files},
        emitter="CoderAgent",
    )


# ---------------------------------------------------------------------------
# (a): ImplementingExperiment entry action
# ---------------------------------------------------------------------------


async def test_implementing_experiment_entry_action_emits_implementation_requested(
    fake_bus, tmp_path
):
    store = SpecStore(tmp_path / "store")
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT, spec_store=store)
    await eng.start()
    sid = "sess-a"
    verifier_spec_ref = await _seed_implementing_experiment_session(eng, store, sid)
    state = await load_session_state(fake_bus.db, sid)

    await eng._execute_entry_action(state)

    emitted = [p for p in fake_bus.published if p.event_type == "code.implementation_requested"]
    assert len(emitted) == 1
    assert emitted[0].payload == {
        "experiment_plan_ref": f"experiment_plan://{sid}/final",
        "verifier_spec_ref": verifier_spec_ref,
    }


# ---------------------------------------------------------------------------
# (b): compliant patch -> RunningExperiment, F reset (deleted)
# ---------------------------------------------------------------------------


async def test_code_implemented_compliant_advances_to_running_experiment(fake_bus, tmp_path):
    store = SpecStore(tmp_path / "store")
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT, spec_store=store)
    await eng.start()
    sid = "sess-b"
    await _seed_implementing_experiment_session(eng, store, sid)
    await _set_map_counter(fake_bus, sid, "implementation_fix_round", "exp1", 1)
    patch_ref = _seed_patch(store, sid, "patch1", ["src/model.py"])

    await eng.tick(
        _Evt(
            "code.implemented",
            sid,
            "CoderAgent",
            payload={"experiment_patch_ref": patch_ref},
        )
    )

    st = await load_session_state(fake_bus.db, sid)
    assert st is not None
    assert st.current_state == "RunningExperiment"
    # M11 correction: a compliant patch does NOT reset F -- F accumulates
    # across the implement -> run -> fail -> re-implement cycle and is reset
    # only by RunningExperiment's "succeeded" branch (M11 Task 4).
    assert st.implementation_fix_round == {"exp1": 1}
    assert st.experiment_patch_refs == [patch_ref]


# ---------------------------------------------------------------------------
# (c): forbidden-file patch, F absent -> bump to 1, retry ImplementingExperiment
# ---------------------------------------------------------------------------


async def test_code_implemented_forbidden_touch_missing_key_bumps_to_one_and_retries(
    fake_bus, tmp_path
):
    store = SpecStore(tmp_path / "store")
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT, spec_store=store)
    await eng.start()
    sid = "sess-c"
    await _seed_implementing_experiment_session(eng, store, sid)
    patch_ref = _seed_patch(store, sid, "patch1", ["eval/scorer.py"])

    n_before = len(fake_bus.published)
    await eng.tick(
        _Evt(
            "code.implemented",
            sid,
            "CoderAgent",
            payload={"experiment_patch_ref": patch_ref},
        )
    )

    st = await load_session_state(fake_bus.db, sid)
    assert st is not None
    assert st.implementation_fix_round == {"exp1": 1}
    assert st.current_state == "ImplementingExperiment"
    assert st.current_moss_reason is None
    assert st.experiment_patch_refs == [patch_ref]

    new_events = fake_bus.published[n_before:]
    assert any(p.event_type == "code.implementation_requested" for p in new_events)


# ---------------------------------------------------------------------------
# (d): default-deny -- touching a file not in editable_region AND not in
# forbidden_files is still a violation.
# ---------------------------------------------------------------------------


async def test_code_implemented_default_deny_untouched_region_is_violation(fake_bus, tmp_path):
    store = SpecStore(tmp_path / "store")
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT, spec_store=store)
    await eng.start()
    sid = "sess-d"
    await _seed_implementing_experiment_session(eng, store, sid)
    patch_ref = _seed_patch(store, sid, "patch1", ["src/other.py"])

    n_before = len(fake_bus.published)
    await eng.tick(
        _Evt(
            "code.implemented",
            sid,
            "CoderAgent",
            payload={"experiment_patch_ref": patch_ref},
        )
    )

    st = await load_session_state(fake_bus.db, sid)
    assert st is not None
    assert st.implementation_fix_round == {"exp1": 1}
    assert st.current_state == "ImplementingExperiment"

    new_events = fake_bus.published[n_before:]
    assert any(p.event_type == "code.implementation_requested" for p in new_events)


# ---------------------------------------------------------------------------
# (e): violation at threshold -> MossIntervening, current_moss_reason ==
# "protocol_violation" (explicit -- NOT the counter's own with_reason).
# ---------------------------------------------------------------------------


async def test_code_implemented_violation_at_threshold_trips_moss(fake_bus, tmp_path):
    store = SpecStore(tmp_path / "store")
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT, spec_store=store)
    await eng.start()
    sid = "sess-e"
    await _seed_implementing_experiment_session(eng, store, sid)
    await _set_map_counter(fake_bus, sid, "implementation_fix_round", "exp1", 2)
    patch_ref = _seed_patch(store, sid, "patch1", ["eval/scorer.py"])

    n_before = len(fake_bus.published)
    await eng.tick(
        _Evt(
            "code.implemented",
            sid,
            "CoderAgent",
            payload={"experiment_patch_ref": patch_ref},
        )
    )

    st = await load_session_state(fake_bus.db, sid)
    assert st is not None
    assert st.implementation_fix_round == {"exp1": 3}
    assert st.current_state == "MossIntervening"
    assert st.current_moss_reason == "protocol_violation"

    new_events = fake_bus.published[n_before:]
    assert not any(p.event_type == "code.implementation_requested" for p in new_events)
