"""Unit tests for Task 4: generic counter helpers (data-driven from counters.yaml).

Covers the reusable counter machinery (M4 Task 4; M5-M11 will reuse it):
  - ``_counter_threshold(name)`` looks up the loaded CounterSpec by ``.name``
    (``self.skill.counters`` is keyed by counter id, e.g. "K", not by name) and
    returns ``(threshold, with_reason)``.
  - ``_bump_counter(state, name)`` increments the scalar SessionState attribute.
  - ``_reset_counter(state, name)`` zeros the scalar SessionState attribute.

Uses the K counter (literature_audit_round) as the concrete example: per
counters.yaml, threshold=2, with_reason=literature_novelty_audit_failed.
"""

from __future__ import annotations

from pathlib import Path

from autoresearch.orchestrator.engine import OrchestratorEngine
from autoresearch.orchestrator.session_state import SessionState
from autoresearch.orchestrator.tests.unit.test_engine_min import FakeBusClient, fake_bus  # noqa: F401

SM_ROOT = Path(__file__).resolve().parents[2] / "state-machine"


def _make_state(sid: str = "s1") -> SessionState:
    return SessionState(session_id=sid)


async def test_bump_counter_increments_scalar_field(fake_bus):
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    state = _make_state()
    assert state.literature_audit_round == 0

    eng._bump_counter(state, "literature_audit_round")
    assert state.literature_audit_round == 1

    eng._bump_counter(state, "literature_audit_round")
    assert state.literature_audit_round == 2


async def test_reset_counter_zeros_scalar_field(fake_bus):
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    state = _make_state()
    state.literature_audit_round = 2

    eng._reset_counter(state, "literature_audit_round")
    assert state.literature_audit_round == 0


async def test_counter_threshold_reads_loaded_spec_for_K(fake_bus):
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)

    threshold, reason = eng._counter_threshold("literature_audit_round")

    assert threshold == 2
    assert reason == "literature_novelty_audit_failed"
