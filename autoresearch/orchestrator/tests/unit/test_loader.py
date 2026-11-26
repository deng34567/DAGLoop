"""Consistency tests for the AutoResearch V1.10 state-machine loader.

The loader running its fail-fast consistency checks without raising is the
acceptance signal: a green loader ⟺ the transcribed YAML is internally
self-consistent. Assertions are relational wherever possible (every referenced
state/reason exists) rather than hardcoded counts, beyond the headline figures
34 states / 14 MOSS reasons / counters A-M.
"""

from pathlib import Path

import pytest

from autoresearch.orchestrator.loader import (
    LoadedSkill,
    load_skill,
)

STATE_MACHINE_ROOT = (
    Path(__file__).resolve().parents[2] / "state-machine"
)

# The 14 MOSS reasons per V1.10 §6.
EXPECTED_MOSS_REASONS = {
    "agent_failure",
    "direction_thrashing",
    "literature_novelty_audit_failed",
    "baseline_reproduction_thrashing",
    "verifier_invalid",
    "hypothesis_stagnation",
    "search_diversity_collapse",
    "theory_thrashing",
    "experiment_design_thrashing",
    "implementation_thrashing",
    "result_suspicious",
    "data_analysis_consensus_failed",
    "result_inconsistency",
    "benchmark_too_narrow",
    "protocol_violation",
    "method_not_working",   # 会议② (2026-07-31)
    "component_ablation_failed",   # 会议 (2026-08-03) 乙
}

EXPECTED_COUNTERS = set("ABCDEFGHIJKLM")  # A .. M


@pytest.fixture(scope="module")
def skill() -> LoadedSkill:
    # Loading runs _run_consistency_checks; any inconsistency raises here.
    return load_skill(STATE_MACHINE_ROOT)


def test_load_returns_loaded_skill(skill):
    assert isinstance(skill, LoadedSkill)


def test_exactly_34_states(skill):
    # 34 core + Deliberating + phase-3.1/3.2 (GeneratingMethod, ConfirmingMethod, RevisingMethod)
    # + meeting① DeliberatingMethod = 39
    assert len(skill.states) == 39, sorted(skill.states)


def test_moss_matrix_covers_all_14_reasons(skill):
    assert set(skill.moss_action_matrix) == EXPECTED_MOSS_REASONS


def test_component_ablation_failed_routes_to_component_plan_audit(skill):
    # Component-plan failures can revise the plan; a falsified core mechanism can revise Method.
    actions = skill.moss_action_matrix["component_ablation_failed"]
    routing = {a: spec.get("next_state") for a, spec in actions.items()}
    assert routing == {
        "revise_component": "DeliberatingMethod",
        "revise_method": "RevisingMethod",
    }
    assert "report_now" not in routing
    assert "DeliberatingMethod" in skill.states


def test_counters_a_through_m_present(skill):
    assert set(skill.counters) == EXPECTED_COUNTERS


def test_every_transition_next_state_exists(skill):
    """Relational: no transition targets a state absent from states.yaml."""
    state_ids = set(skill.states)
    for state_id, rules in skill.transitions.items():
        for rule in rules:
            targets = set()
            if rule.next_state and not rule.next_state.startswith("{"):
                targets.add(rule.next_state)
            for nb in rule.next_branch or []:
                ns = nb.get("next_state")
                if ns and not str(ns).startswith("{"):
                    targets.add(str(ns))
            for tc in rule.threshold_check or []:
                ns = tc.get("next_state")
                if ns and not str(ns).startswith("{"):
                    targets.add(str(ns))
            missing = targets - state_ids
            assert not missing, f"{state_id} -> unknown {missing}"


def test_every_moss_action_next_state_exists(skill):
    state_ids = set(skill.states)
    for reason, actions in skill.moss_action_matrix.items():
        for action_name, spec in actions.items():
            ns = spec.get("next_state")
            if ns and not str(ns).startswith("{"):
                assert ns in state_ids, f"{reason}.{action_name} -> {ns}"


def test_every_user_decision_reason_has_abandon(skill):
    for reason, actions in skill.user_decision_matrix.items():
        assert "abandon" in actions, f"{reason} missing abandon"


def test_every_user_decision_next_state_exists(skill):
    state_ids = set(skill.states)
    for reason, actions in skill.user_decision_matrix.items():
        for action_name, spec in actions.items():
            ns = spec.get("next_state")
            if ns and not str(ns).startswith("{"):
                assert ns in state_ids, f"{reason}.{action_name} -> {ns}"
