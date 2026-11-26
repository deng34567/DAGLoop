"""Unit tests for Task 4: orchestrator wiring for RoutingResearchMode auto-advance
+ SelectingBenchmark/PreparingVerifier + MAP counter B.

Covers:
  - the CritiquingDirection "pass" branch now auto-advances RoutingResearchMode ->
    SelectingBenchmark (mode == "benchmark" is SessionState's default), so
    ``benchmark.selection_requested`` is emitted without any extra event.
  - the SelectingBenchmark entry action (benchmark.selection_requested).
  - ``_on_SelectingBenchmark_benchmark_spec_ready`` -> PreparingVerifier,
    storing benchmark_spec_ref + current_benchmark_id, whose entry action
    emits verifier.preparation_requested.
  - ``_on_PreparingVerifier_verifier_spec_frozen`` (counter B,
    baseline_reproduce_round[benchmark_id], configured threshold, reason
    "baseline_reproduction_thrashing"):
      - verdict "ok"         -> ReproducingBaseline (no entry action -> pauses).
      - verdict "incomplete" -> bump B[benchmark_id]; below threshold ->
        SelectingBenchmark (re-emits benchmark.selection_requested); at/above
        threshold -> MossIntervening with current_moss_reason ==
        "baseline_reproduction_thrashing".
      - verdict "invalid"    -> MossIntervening with current_moss_reason ==
        "verifier_invalid" (no counter bump).

Reuses FakeBusClient from test_engine_min.py and
``_advance_to_critiquing_direction`` from test_engine_critic_advance.py (does
NOT auto-deliver; drives the engine directly via ``eng.tick(event)``),
mirroring the existing advance-test files' patterns.
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
from autoresearch.orchestrator.tests.unit.test_engine_critic_advance import (
    _advance_to_critiquing_direction,
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


async def _seed_selecting_benchmark_session(eng: OrchestratorEngine, sid: str) -> str:
    """Persist a session in SelectingBenchmark with its upstream refs set."""
    research_spec_ref = f"research_spec://{sid}/revised"
    state = SessionState(
        session_id=sid,
        user_id="u1",
        current_state="SelectingBenchmark",
        research_spec_ref=research_spec_ref,
        literature_brief_ref=f"literature_brief://{sid}/final",
    )
    await save_session_state(eng.client.db, state, auto_commit=True)
    return research_spec_ref


async def _seed_preparing_verifier_session(
    eng: OrchestratorEngine, sid: str, *, benchmark_id: str = "bench-1"
) -> str:
    """Persist a session in PreparingVerifier with its upstream refs set."""
    benchmark_spec_ref = f"benchmark_spec://{sid}/{benchmark_id}"
    state = SessionState(
        session_id=sid,
        user_id="u1",
        current_state="PreparingVerifier",
        research_spec_ref=f"research_spec://{sid}/revised",
        benchmark_spec_ref=benchmark_spec_ref,
        current_benchmark_id=benchmark_id,
    )
    await save_session_state(eng.client.db, state, auto_commit=True)
    return benchmark_spec_ref


async def _set_map_counter(fake_bus, sid: str, name: str, key: str, value: int) -> None:
    """Directly set a map counter key on a persisted session (test setup only)."""
    state = await load_session_state(fake_bus.db, sid)
    assert state is not None
    getattr(state, name)[key] = value
    await save_session_state(fake_bus.db, state, auto_commit=True)


# ---------------------------------------------------------------------------
# (a): RoutingResearchMode auto-advance (direction.critiqued pass)
# ---------------------------------------------------------------------------


@pytest.mark.skip(reason="CritiquingDirection retired per meeting item 7; direction.critiqued no "
                         "longer routes (checkpoint goes straight to GeneratingMethod). Removed "
                         "with the CriticAgent direction subsystem in the C-phase (甲) cleanup.")
async def test_direction_critiqued_pass_auto_advances_to_selecting_benchmark(fake_bus):
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = await eng.start_session("goal", user_id="u1")
    await _advance_to_critiquing_direction(eng, sid)

    critique_report_ref = f"critique_report://{sid}/direction/final"
    n_before = len(fake_bus.published)
    commits_before = fake_bus.db.commit_count
    await eng.tick(
        _Evt(
            "direction.critiqued",
            sid,
            "CriticAgent",
            payload={"verdict": "pass", "critique_report_ref": critique_report_ref},
        )
    )

    st = await load_session_state(fake_bus.db, sid)
    assert st is not None
    # phase 3.1: a passed direction now goes to GeneratingMethod (derive the explicit method) before
    # the track/benchmark is selected.
    assert st.current_state == "GeneratingMethod"
    assert st.direction_critique_round == 0

    new_events = fake_bus.published[n_before:]
    emitted = [p for p in new_events if p.event_type == "method.generation_requested"]
    assert len(emitted) == 1

    assert fake_bus.db.commit_count - commits_before == 1


# ---------------------------------------------------------------------------
# (b): SelectingBenchmark -> benchmark.spec_ready -> ReproducingBaseline (meeting #1: the
# pre-baseline verifier is gone, so no PreparingVerifier hop and no verifier_spec is frozen)
# ---------------------------------------------------------------------------


async def test_benchmark_spec_ready_advances_to_reproducing_baseline(fake_bus):
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = "sess-b"
    research_spec_ref = await _seed_selecting_benchmark_session(eng, sid)

    benchmark_spec_ref = f"benchmark_spec://{sid}/bench-1"
    n_before = len(fake_bus.published)
    await eng.tick(
        _Evt(
            "benchmark.spec_ready",
            sid,
            "BenchmarkSelectorAgent",
            payload={"benchmark_spec_ref": benchmark_spec_ref, "benchmark_id": "bench-1"},
        )
    )

    st = await load_session_state(fake_bus.db, sid)
    assert st is not None
    assert st.current_state == "ReproducingBaseline"     # straight to baseline, no verifier hop
    assert st.benchmark_spec_ref == benchmark_spec_ref
    assert st.current_benchmark_id == "bench-1"
    assert st.verifier_spec_ref is None                  # no pre-baseline verifier is frozen

    new_events = fake_bus.published[n_before:]
    # the pre-baseline verifier step is gone -> no verifier.preparation_requested; ReproducingBaseline's
    # entry action asks the runner to reproduce the baseline instead.
    assert not [p for p in new_events if p.event_type == "verifier.preparation_requested"]
    baseline_reqs = [p for p in new_events if p.event_type == "baseline.reproduction_requested"]
    assert len(baseline_reqs) == 1
    assert baseline_reqs[0].payload == {
        "benchmark_spec_ref": benchmark_spec_ref,
        "verifier_spec_ref": None,
        "confirmed_repo_url": None,
        "confirmed_boundary": {},
        "recover_completed_result": False,
    }


# ---------------------------------------------------------------------------
# (c)-(f): PreparingVerifier -> verifier.spec_frozen (counter B)
# ---------------------------------------------------------------------------


async def test_verifier_spec_frozen_ok_advances_to_reproducing_baseline(fake_bus):
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = "sess-c"
    await _seed_preparing_verifier_session(eng, sid, benchmark_id="bench-1")

    verifier_spec_ref = f"verifier_spec://{sid}/final"
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
    assert st.verifier_spec_ref == verifier_spec_ref
    assert st.current_state == "ReproducingBaseline"
    assert st.baseline_reproduce_round == {}


async def test_verifier_spec_frozen_incomplete_missing_key_bumps_to_one_and_reselects(fake_bus):
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = "sess-d"
    await _seed_preparing_verifier_session(eng, sid, benchmark_id="bench-1")

    n_before = len(fake_bus.published)
    await eng.tick(
        _Evt(
            "verifier.spec_frozen",
            sid,
            "VerifierAgent",
            payload={"verifier_spec_ref": f"verifier_spec://{sid}/draft", "verdict": "incomplete"},
        )
    )

    st = await load_session_state(fake_bus.db, sid)
    assert st is not None
    assert st.baseline_reproduce_round == {"bench-1": 1}
    assert st.current_state == "SelectingBenchmark"
    assert st.current_moss_reason is None

    new_events = fake_bus.published[n_before:]
    assert any(p.event_type == "benchmark.selection_requested" for p in new_events)


async def test_verifier_spec_frozen_incomplete_at_threshold_trips_moss(fake_bus):
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = "sess-e"
    await _seed_preparing_verifier_session(eng, sid, benchmark_id="bench-1")
    threshold, _ = eng._counter_threshold("baseline_reproduce_round")
    await _set_map_counter(
        fake_bus, sid, "baseline_reproduce_round", "bench-1", threshold - 1)

    n_before = len(fake_bus.published)
    await eng.tick(
        _Evt(
            "verifier.spec_frozen",
            sid,
            "VerifierAgent",
            payload={"verifier_spec_ref": f"verifier_spec://{sid}/draft3", "verdict": "incomplete"},
        )
    )

    st = await load_session_state(fake_bus.db, sid)
    assert st is not None
    assert st.baseline_reproduce_round == {"bench-1": threshold}
    assert st.current_state == "MossIntervening"
    assert st.current_moss_reason == "baseline_reproduction_thrashing"

    new_events = fake_bus.published[n_before:]
    assert not any(p.event_type == "benchmark.selection_requested" for p in new_events)


async def test_verifier_spec_frozen_invalid_trips_moss(fake_bus):
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = "sess-f"
    await _seed_preparing_verifier_session(eng, sid, benchmark_id="bench-1")

    await eng.tick(
        _Evt(
            "verifier.spec_frozen",
            sid,
            "VerifierAgent",
            payload={"verifier_spec_ref": f"verifier_spec://{sid}/bad", "verdict": "invalid"},
        )
    )

    st = await load_session_state(fake_bus.db, sid)
    assert st is not None
    assert st.current_state == "MossIntervening"
    assert st.current_moss_reason == "verifier_invalid"
    # Invalid verdict does not touch counter B.
    assert st.baseline_reproduce_round == {}


async def test_verifier_spec_frozen_unknown_verdict_is_defensive_noop(fake_bus):
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    sid = "sess-g"
    await _seed_preparing_verifier_session(eng, sid, benchmark_id="bench-1")

    n_before = len(fake_bus.published)
    await eng.tick(
        _Evt(
            "verifier.spec_frozen",
            sid,
            "VerifierAgent",
            payload={"verifier_spec_ref": f"verifier_spec://{sid}/x", "verdict": "something_weird"},
        )
    )

    st = await load_session_state(fake_bus.db, sid)
    assert st is not None
    assert st.current_state == "PreparingVerifier"
    assert st.baseline_reproduce_round == {}
    assert fake_bus.published[n_before:] == []
