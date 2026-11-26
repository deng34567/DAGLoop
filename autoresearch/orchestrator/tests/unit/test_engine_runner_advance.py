"""Unit tests for Task 4: the ReproducingBaseline connector + MAP counter B.

Covers:
  - the ReproducingBaseline entry action (``baseline.reproduction_requested``),
    newly added here -- M7 left ReproducingBaseline as a pause with NO entry
    action; this test exercises it via the ``PreparingVerifier`` "ok" verdict,
    which transitions into ReproducingBaseline.
  - ``_on_ReproducingBaseline_baseline_reproduced`` (counter B,
    baseline_reproduce_round[benchmark_id], threshold 3, reason
    "baseline_reproduction_thrashing"):
      - verdict "reproduced" -> reset B[benchmark_id] (delete key);
        GeneratingHypotheses -- THE CONNECTOR: its EXISTING M5 entry action
        emits hypothesis.generation_requested, proving the hypothesis loop is
        wired up.
      - verdict "failed"     -> bump B[benchmark_id]; below threshold -> self-
        transition back to ReproducingBaseline (re-emits
        baseline.reproduction_requested, i.e. retry); at/above threshold ->
        MossIntervening with current_moss_reason ==
        "baseline_reproduction_thrashing".

Reuses FakeBusClient from test_engine_min.py, mirroring
test_engine_evaluator_advance.py's patterns (does NOT auto-deliver; drives the
engine directly via ``eng.tick(event)``).
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
from autoresearch.spec_store import SpecStore

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


async def _seed_reproducing_baseline_session(
    eng: OrchestratorEngine, sid: str, *, benchmark_id: str = "bench1",
    confirmed_repo_url: str | None = None,
) -> None:
    """Persist a session in ReproducingBaseline with its upstream refs set."""
    state = SessionState(
        session_id=sid,
        user_id="u1",
        current_state="ReproducingBaseline",
        research_spec_ref=f"research_spec://{sid}/revised",
        literature_brief_ref=f"literature_brief://{sid}/final",
        benchmark_spec_ref=f"benchmark_spec://{sid}/{benchmark_id}",
        verifier_spec_ref=f"verifier_spec://{sid}/final",
        method_ref=f"method://{sid}/confirmed",
        current_benchmark_id=benchmark_id,
        confirmed_repo_url=confirmed_repo_url,
    )
    await save_session_state(eng.client.db, state, auto_commit=True)


async def _set_map_counter(fake_bus, sid: str, name: str, key: str, value: int) -> None:
    """Directly set a map counter key on a persisted session (test setup only)."""
    state = await load_session_state(fake_bus.db, sid)
    assert state is not None
    getattr(state, name)[key] = value
    await save_session_state(fake_bus.db, state, auto_commit=True)


# ---------------------------------------------------------------------------
# (a): ReproducingBaseline entry action
# ---------------------------------------------------------------------------


async def test_reproducing_baseline_entry_action_emits_reproduction_requested(fake_bus):
    """PreparingVerifier's "ok" verdict transitions into ReproducingBaseline,
    whose entry action (newly wired here) must emit
    baseline.reproduction_requested with the benchmark + verifier spec refs.
    """
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = "sess-a"
    benchmark_spec_ref = f"benchmark_spec://{sid}/bench1"
    state = SessionState(
        session_id=sid,
        user_id="u1",
        current_state="PreparingVerifier",
        research_spec_ref=f"research_spec://{sid}/revised",
        benchmark_spec_ref=benchmark_spec_ref,
        current_benchmark_id="bench1",
    )
    await save_session_state(eng.client.db, state, auto_commit=True)

    verifier_spec_ref = f"verifier_spec://{sid}/final"
    n_before = len(fake_bus.published)
    await eng.tick(
        _Evt(
            "verifier.spec_frozen",
            sid,
            "VerifierAgent",
            payload={"verifier_spec_ref": verifier_spec_ref, "verdict": "ok"},
        )
    )

    st = await load_session_state(fake_bus.db, sid)
    assert st is not None
    assert st.current_state == "ReproducingBaseline"

    new_events = fake_bus.published[n_before:]
    emitted = [p for p in new_events if p.event_type == "baseline.reproduction_requested"]
    assert len(emitted) == 1
    assert emitted[0].payload == {
        "benchmark_spec_ref": benchmark_spec_ref,
        "verifier_spec_ref": verifier_spec_ref,
        "confirmed_repo_url": None,
        "confirmed_boundary": {},
        "recover_completed_result": False,
    }


# ---------------------------------------------------------------------------
# (b): baseline.reproduced "reproduced" -> component-plan deliberation -> PlanningExperiment
# meeting 2026-07-25 甲: the benchmark mainline no longer runs the GeneratingHypotheses ->
# portfolio outer loop; the confirmed method goes straight to planning HOW TO IMPLEMENT it.
# ---------------------------------------------------------------------------


async def test_baseline_reproduced_reproduced_audits_component_plan_before_planning(fake_bus):
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = "sess-b"
    await _seed_reproducing_baseline_session(eng, sid, benchmark_id="bench1")
    await _set_map_counter(fake_bus, sid, "baseline_reproduce_round", "bench1", 2)

    baseline_result_ref = f"baseline_result://{sid}/final"
    n_before = len(fake_bus.published)
    await eng.tick(
        _Evt(
            "baseline.reproduced",
            sid,
            "ExperimentRunnerAgent",
            payload={
                "verdict": "reproduced",
                "baseline_result_ref": baseline_result_ref,
                "benchmark_id": "bench1",
            },
        )
    )

    st = await load_session_state(fake_bus.db, sid)
    assert st is not None
    assert st.current_state == "DeliberatingMethod"
    assert st.component_plan_audit_pending is True
    assert "bench1" not in st.baseline_reproduce_round
    assert st.baseline_result_ref == baseline_result_ref
    # The experiment identity is installed only after the component-plan audit returns.
    assert st.current_experiment_id is None

    # The existing A1/A2/MOSS/Critic deliberation is reused, scoped to ComponentPlan audit.
    new_events = fake_bus.published[n_before:]
    emitted = [p for p in new_events if p.event_type == "method.deliberation_requested"]
    assert len(emitted) == 1
    assert emitted[0].payload["method_ref"] == f"method://{sid}/confirmed"
    assert emitted[0].payload["baseline_result_ref"] == baseline_result_ref
    assert emitted[0].payload["component_plan_audit"] is True

    st.last_agent_failure = {"underlying_reason_code":
                             "confirmed_method_contract_conflict"}
    await save_session_state(fake_bus.db, st, auto_commit=True)
    await eng.tick(_Evt(
        "method.deliberated", sid, "MethodAgent",
        payload={"method_ref": f"method://{sid}/component-plan-audited"},
    ))
    st = await load_session_state(fake_bus.db, sid)
    assert st.current_state == "PlanningExperiment"
    assert st.last_agent_failure is None
    assert st.component_plan_audit_pending is False
    assert st.current_experiment_id == f"method://{sid}/component-plan-audited"
    planned = [
        event for event in fake_bus.published[n_before:]
        if event.event_type == "experiment.planning_requested"
    ]
    assert len(planned) == 1


async def test_first_benchmark_baseline_precedes_method_generation(fake_bus):
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = "sess-baseline-first"
    await _seed_reproducing_baseline_session(eng, sid, benchmark_id="bench1")
    state = await load_session_state(fake_bus.db, sid)
    state.method_ref = None
    await save_session_state(fake_bus.db, state, auto_commit=True)

    await eng.tick(_Evt(
        "baseline.reproduced", sid, "ExperimentRunnerAgent",
        payload={"verdict": "reproduced", "baseline_result_ref": f"baseline_result://{sid}/v1",
                 "benchmark_id": "bench1"},
    ))

    updated = await load_session_state(fake_bus.db, sid)
    assert updated.current_state == "GeneratingMethod"
    assert updated.baseline_result_ref == f"baseline_result://{sid}/v1"
    assert updated.component_plan_audit_pending is False
    requested = [p for p in fake_bus.published if p.event_type == "method.generation_requested"]
    assert requested[-1].payload["baseline_result_ref"] == updated.baseline_result_ref


async def test_repo_baseline_ignores_stale_benchmark_event(fake_bus, tmp_path):
    """A delayed OLD benchmark result must not open the current NEW benchmark gate."""
    store = SpecStore(tmp_path / "store")
    try:
        eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT, spec_store=store)
        await eng.start()
        sid = "sess-stale-baseline"
        await _seed_reproducing_baseline_session(
            eng, sid, benchmark_id="NEW",
            confirmed_repo_url="https://github.com/example/repo",
        )

        await eng.tick(_Evt(
            "baseline.reproduced", sid, "ExperimentRunnerAgent",
            payload={
                "verdict": "reproduced",
                "baseline_result_ref": f"baseline_result://{sid}/old",
                "benchmark_id": "OLD",
            },
        ))

        state = await load_session_state(fake_bus.db, sid)
        assert state.current_state == "ReproducingBaseline"
        assert state.baseline_result_ref is None
        assert state.baseline_reproduce_round == {}
    finally:
        store.close()


async def test_repo_baseline_missing_current_artifact_counts_as_failed_attempt(fake_bus, tmp_path):
    """A current `reproduced` claim with no resolvable artifact is a retry, not success."""
    store = SpecStore(tmp_path / "store")
    try:
        eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT, spec_store=store)
        await eng.start()
        sid = "sess-null-baseline"
        await _seed_reproducing_baseline_session(
            eng, sid, benchmark_id="NEW",
            confirmed_repo_url="https://github.com/example/repo",
        )

        await eng.tick(_Evt(
            "baseline.reproduced", sid, "ExperimentRunnerAgent",
            payload={
                "verdict": "reproduced",
                "baseline_result_ref": f"baseline_result://{sid}/missing",
                "benchmark_id": "NEW",
            },
        ))

        state = await load_session_state(fake_bus.db, sid)
        assert state.current_state == "ReproducingBaseline"
        assert state.baseline_result_ref is None
        assert state.baseline_reproduce_round == {"NEW": 1}
    finally:
        store.close()


async def test_repo_real_baseline_advances_without_fairness_attestation(
    fake_bus, tmp_path,
):
    """A finite baseline from the first sandbox needs no fairness attestation."""
    store = SpecStore(tmp_path / "store")
    try:
        eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT, spec_store=store)
        await eng.start()
        sid = "sess-relaxed-baseline"
        await _seed_reproducing_baseline_session(
            eng, sid, benchmark_id="bench1",
            confirmed_repo_url="https://github.com/example/repo",
        )
        artifact_ref = store.put(
            kind="baseline_result", session_id=sid, key="live",
            obj={
                "benchmark_id": "bench1", "reproduced": True,
                "command": "repository baseline sandbox", "exit_code": 0,
                "reported_metric": None, "metric_value": 1.0,
                "source_kind": "live_repo",
                "repo_url": "https://github.com/example/repo",
                "environment": "baseline sandbox",
                "benchmark_contract": None, "contract_fingerprint": None,
                "host_attestation": None,
            }, emitter="RunnerAgent")

        await eng.tick(_Evt(
            "baseline.reproduced", sid, "ExperimentRunnerAgent",
            payload={
                "verdict": "reproduced", "baseline_result_ref": artifact_ref,
                "benchmark_id": "bench1",
            },
        ))

        state = await load_session_state(fake_bus.db, sid)
        assert state.current_state == "DeliberatingMethod"
        assert state.component_plan_audit_pending is True
        assert state.baseline_result_ref == artifact_ref
    finally:
        store.close()


async def test_resolved_repository_baseline_binds_session_before_method(
    fake_bus, tmp_path,
):
    """A repository found after an auto checkpoint becomes the session's frozen main repo."""
    store = SpecStore(tmp_path / "store")
    try:
        eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT, spec_store=store)
        await eng.start()
        sid = "sess-resolved-repo"
        await _seed_reproducing_baseline_session(eng, sid, benchmark_id="bench1")
        repo_url = "https://github.com/example/autoresearch"
        artifact_ref = store.put(
            kind="baseline_result", session_id=sid, key="live",
            obj={
                "benchmark_id": "bench1", "reproduced": True,
                "command": "repository baseline sandbox", "exit_code": 0,
                "reported_metric": None, "metric_value": 1.0,
                "source_kind": "live_repo", "repo_url": repo_url,
                "environment": "baseline sandbox",
            }, emitter="RunnerAgent")

        await eng.tick(_Evt(
            "baseline.reproduced", sid, "ExperimentRunnerAgent",
            payload={
                "verdict": "reproduced", "baseline_result_ref": artifact_ref,
                "benchmark_id": "bench1", "repo_url": repo_url,
            },
        ))

        state = await load_session_state(fake_bus.db, sid)
        assert state.current_state == "DeliberatingMethod"
        assert state.component_plan_audit_pending is True
        assert state.confirmed_repo_url == repo_url
    finally:
        store.close()


async def test_repo_baseline_rejects_cross_repo_relabel_attack(
        fake_bus, tmp_path, monkeypatch):
    """A live artifact for another repository cannot be relabeled with this benchmark id."""
    store = SpecStore(tmp_path / "store")
    try:
        eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT, spec_store=store)
        await eng.start()
        sid = "sess-cross-repo-baseline"
        confirmed_repo = "https://github.com/example/confirmed"
        await _seed_reproducing_baseline_session(
            eng, sid, benchmark_id="NEW", confirmed_repo_url=confirmed_repo)
        artifact_ref = store.put(
            kind="baseline_result", session_id=sid, key="cross-repo",
            obj={
                "benchmark_id": "NEW", "source_kind": "live_repo",
                "repo_url": "https://github.com/attacker/different",
            }, emitter="RunnerAgent")
        await eng.tick(_Evt(
            "baseline.reproduced", sid, "ExperimentRunnerAgent",
            payload={
                "verdict": "reproduced", "baseline_result_ref": artifact_ref,
                "benchmark_id": "NEW",
            },
        ))

        state = await load_session_state(fake_bus.db, sid)
        assert state.current_state == "ReproducingBaseline"
        assert state.baseline_result_ref is None
        assert state.baseline_reproduce_round == {"NEW": 1}
    finally:
        store.close()


# ---------------------------------------------------------------------------
# (c): baseline.reproduced "failed" (B absent) -> bump to 1, retry (self-transition)
# ---------------------------------------------------------------------------


async def test_baseline_reproduced_failed_missing_key_bumps_to_one_and_retries(fake_bus):
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = "sess-c"
    await _seed_reproducing_baseline_session(eng, sid, benchmark_id="bench1")

    n_before = len(fake_bus.published)
    await eng.tick(
        _Evt(
            "baseline.reproduced",
            sid,
            "ExperimentRunnerAgent",
            payload={
                "verdict": "failed",
                "baseline_result_ref": f"baseline_result://{sid}/attempt1",
                "benchmark_id": "bench1",
            },
        )
    )

    st = await load_session_state(fake_bus.db, sid)
    assert st is not None
    assert st.baseline_reproduce_round == {"bench1": 1}
    assert st.baseline_result_ref is None
    assert st.last_baseline_attempt_ref == f"baseline_result://{sid}/attempt1"
    assert st.current_state == "ReproducingBaseline"
    assert st.current_moss_reason is None

    new_events = fake_bus.published[n_before:]
    assert any(p.event_type == "baseline.reproduction_requested" for p in new_events)


# ---------------------------------------------------------------------------
# (d): baseline.reproduced failures remain bounded, with a relaxed threshold
# ---------------------------------------------------------------------------


async def test_baseline_reproduced_third_failure_still_retries(fake_bus):
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = "sess-d"
    await _seed_reproducing_baseline_session(eng, sid, benchmark_id="bench1")
    await _set_map_counter(fake_bus, sid, "baseline_reproduce_round", "bench1", 2)

    n_before = len(fake_bus.published)
    await eng.tick(
        _Evt(
            "baseline.reproduced",
            sid,
            "ExperimentRunnerAgent",
            payload={
                "verdict": "failed",
                "baseline_result_ref": f"baseline_result://{sid}/attempt3",
                "benchmark_id": "bench1",
            },
        )
    )

    st = await load_session_state(fake_bus.db, sid)
    assert st is not None
    assert st.baseline_reproduce_round == {"bench1": 3}
    assert st.current_state == "ReproducingBaseline"
    assert st.current_moss_reason is None

    new_events = fake_bus.published[n_before:]
    assert any(p.event_type == "baseline.reproduction_requested" for p in new_events)


@pytest.mark.parametrize("repo_url", [None, "https://github.com/example/unavailable"])
async def test_baseline_reproduced_sixth_failure_trips_moss(fake_bus, repo_url):
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = "sess-d-six"
    await _seed_reproducing_baseline_session(
        eng, sid, benchmark_id="bench1", confirmed_repo_url=repo_url)
    await _set_map_counter(fake_bus, sid, "baseline_reproduce_round", "bench1", 5)

    n_before = len(fake_bus.published)
    await eng.tick(
        _Evt(
            "baseline.reproduced",
            sid,
            "ExperimentRunnerAgent",
            payload={
                "verdict": "failed",
                "baseline_result_ref": f"baseline_result://{sid}/attempt6",
                "benchmark_id": "bench1",
            },
        )
    )

    st = await load_session_state(fake_bus.db, sid)
    assert st is not None
    assert st.baseline_reproduce_round == {"bench1": 6}
    assert st.current_state == "MossIntervening"
    assert st.current_moss_reason == "baseline_reproduction_thrashing"
    assert st.last_baseline_attempt_ref == f"baseline_result://{sid}/attempt6"
    assert st.baseline_result_ref is None

    new_events = fake_bus.published[n_before:]
    assert not any(p.event_type == "baseline.reproduction_requested" for p in new_events)
    intervention = next(p for p in new_events if p.event_type == "moss.intervention_requested")
    assert intervention.payload["baseline_attempt_ref"] == st.last_baseline_attempt_ref


# ---------------------------------------------------------------------------
# unknown verdict: defensive no-op
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("current_state", ["SelectingBenchmark", "DeliberatingMethod"])
async def test_failed_legacy_baseline_never_unlocks_component_plan(fake_bus, tmp_path, current_state):
    store = SpecStore(tmp_path / "store")
    try:
        eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT, spec_store=store)
        await eng.start()
        sid = "legacy-failed-baseline"
        bad_ref = store.put(kind="baseline_result", session_id=sid, key="failed",
            obj={"benchmark_id": "bench1", "reproduced": False, "metric_value": None},
            emitter="RunnerAgent")
        state = SessionState(
            session_id=sid, user_id="u1", current_state=current_state,
            mode="benchmark", method_ref=f"method://{sid}/v1",
            benchmark_spec_ref=f"benchmark_spec://{sid}/v1", current_benchmark_id="bench1",
            baseline_result_ref=bad_ref, component_plan_audit_pending=True,
        )
        await save_session_state(fake_bus.db, state, auto_commit=True)
        if current_state == "SelectingBenchmark":
            await eng.tick(_Evt("benchmark.spec_ready", sid, "EvaluatorAgent",
                payload={"benchmark_spec_ref": state.benchmark_spec_ref, "benchmark_id": "bench1"}))
        else:
            await eng.resume_to_current_state(sid)
        state = await load_session_state(fake_bus.db, sid)
        assert state.current_state == "ReproducingBaseline"
        assert fake_bus.published[-1].event_type == "baseline.reproduction_requested"
        assert state.method_ref == f"method://{sid}/v1"
        if current_state == "DeliberatingMethod":
            assert state.baseline_result_ref is None
            assert state.last_baseline_attempt_ref == bad_ref
    finally:
        store.close()


async def test_baseline_reproduced_unknown_verdict_is_defensive_noop(fake_bus):
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = "sess-e"
    await _seed_reproducing_baseline_session(eng, sid, benchmark_id="bench1")

    n_before = len(fake_bus.published)
    await eng.tick(
        _Evt(
            "baseline.reproduced",
            sid,
            "ExperimentRunnerAgent",
            payload={
                "verdict": "something_weird",
                "baseline_result_ref": f"baseline_result://{sid}/x",
                "benchmark_id": "bench1",
            },
        )
    )

    st = await load_session_state(fake_bus.db, sid)
    assert st is not None
    assert st.current_state == "ReproducingBaseline"
    assert st.baseline_reproduce_round == {}
    assert fake_bus.published[n_before:] == []
