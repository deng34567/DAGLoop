"""Unit tests for Task 4's MAP counter helpers: ``_bump_map_counter`` /
``_reset_map_counter``.

Mirrors the scalar ``_bump_counter``/``_reset_counter`` helpers but operates
on a dict-valued SessionState field (V1.10 §4 Map counters B/D/E/F/G/J/M),
delegating to the module-level ``bump_map_counter``/``reset_map_counter`` in
``session_state.py``. Per counters.yaml's ``initial_per_key: 1``, a missing
key is created at 1 (not 0->1 via increment).
"""

from __future__ import annotations

from pathlib import Path

from autoresearch.orchestrator.engine import OrchestratorEngine
from autoresearch.orchestrator.session_state import SessionState
from autoresearch.orchestrator.tests.unit.test_engine_min import fake_bus  # noqa: F401

SM_ROOT = Path(__file__).resolve().parents[2] / "state-machine"


def _make_engine(fake_bus) -> OrchestratorEngine:
    return OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)


async def test_bump_map_counter_missing_key_sets_to_one(fake_bus):
    eng = _make_engine(fake_bus)
    state = SessionState(session_id="s1")

    eng._bump_map_counter(state, "baseline_reproduce_round", "bench-1")

    assert state.baseline_reproduce_round == {"bench-1": 1}


async def test_bump_map_counter_existing_key_increments(fake_bus):
    eng = _make_engine(fake_bus)
    state = SessionState(session_id="s1")

    eng._bump_map_counter(state, "baseline_reproduce_round", "bench-1")
    eng._bump_map_counter(state, "baseline_reproduce_round", "bench-1")

    assert state.baseline_reproduce_round == {"bench-1": 2}


async def test_reset_map_counter_deletes_key(fake_bus):
    eng = _make_engine(fake_bus)
    state = SessionState(session_id="s1")
    eng._bump_map_counter(state, "baseline_reproduce_round", "bench-1")

    eng._reset_map_counter(state, "baseline_reproduce_round", "bench-1")

    assert state.baseline_reproduce_round == {}


async def test_reset_map_counter_missing_key_is_noop(fake_bus):
    eng = _make_engine(fake_bus)
    state = SessionState(session_id="s1")

    eng._reset_map_counter(state, "baseline_reproduce_round", "bench-does-not-exist")

    assert state.baseline_reproduce_round == {}


async def test_bump_map_counter_two_keys_are_independent(fake_bus):
    eng = _make_engine(fake_bus)
    state = SessionState(session_id="s1")

    eng._bump_map_counter(state, "baseline_reproduce_round", "bench-1")
    eng._bump_map_counter(state, "baseline_reproduce_round", "bench-2")
    eng._bump_map_counter(state, "baseline_reproduce_round", "bench-1")

    assert state.baseline_reproduce_round == {"bench-1": 2, "bench-2": 1}
