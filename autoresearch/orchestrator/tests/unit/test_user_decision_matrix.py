"""Tests for the user-decision reason-matrix loader + UserDecisionGuard.

Loads the EXISTING committed SSOT
(autoresearch/orchestrator/state-machine/user-decisions/reason-matrix.yaml) --
no new matrix YAML is introduced here. Mirrors test_moss_matrix.py. The key
cross-check: every NON-TEMPLATE next_state (one that does not contain "{") must
be a real state in states.yaml.

Pure yaml/logic tests: no LLM, no execution, no Redis, no network.
"""

from pathlib import Path

import pytest
import yaml

from autoresearch.orchestrator.user_decision_matrix import (
    UserDecisionGuard,
    load_user_decision_matrix,
)

STATE_MACHINE_ROOT = Path(__file__).resolve().parents[2] / "state-machine"


@pytest.fixture(scope="module")
def matrix():
    return load_user_decision_matrix()


@pytest.fixture(scope="module")
def guard(matrix):
    return UserDecisionGuard(matrix)


def test_matrix_loads_all_reasons(matrix):
    data = yaml.safe_load(
        (STATE_MACHINE_ROOT / "user-decisions" / "reason-matrix.yaml").read_text()
    )
    expected_reasons = {entry["reason"] for entry in data["reasons"]}
    assert set(matrix) == expected_reasons
    assert len(matrix) == len(expected_reasons)


def test_every_reason_has_at_least_one_action(matrix):
    for reason, actions in matrix.items():
        assert len(actions) >= 1, reason


def test_every_reason_includes_abandon(matrix):
    for reason, actions in matrix.items():
        assert "abandon" in actions, reason


def test_is_legal_true_for_known_action(guard):
    assert guard.is_legal("literature_uncertain", "continue_with_caveat") is True


def test_is_legal_false_for_bogus_action(guard):
    assert guard.is_legal("literature_uncertain", "bogus") is False


def test_is_legal_false_for_unknown_reason(guard):
    assert guard.is_legal("no_such_reason", "abandon") is False


def test_legal_actions_empty_for_unknown_reason(guard):
    assert guard.legal_actions("no_such_reason") == []


def test_next_state_continue_with_caveat(guard):
    assert guard.next_state("literature_uncertain", "continue_with_caveat") == "ConfirmingMethod"


def test_next_state_abandon_is_always_failed(guard, matrix):
    for reason in matrix:
        assert guard.next_state(reason, "abandon") == "Failed"


def test_side_effects_continue_with_caveat(guard):
    assert guard.side_effects("literature_uncertain", "continue_with_caveat") == [
        "报告标注 novelty caveat"
    ]


def test_side_effects_empty_when_absent(guard):
    assert guard.side_effects("literature_uncertain", "human_literature_seed") == []


def test_experiment_design_thrashing_can_revise_method(guard):
    assert guard.is_legal("experiment_design_thrashing", "revise_method")
    assert guard.next_state(
        "experiment_design_thrashing", "revise_method"
    ) == "GeneratingMethod"


def test_experiment_design_thrashing_can_revise_component(guard):
    assert guard.is_legal("experiment_design_thrashing", "revise_component")
    assert guard.next_state("experiment_design_thrashing", "revise_component") == "DeliberatingMethod"


# --- Cross-check: every NON-TEMPLATE next_state is a real state id. ---


def test_every_non_template_next_state_is_a_real_state(matrix):
    states_yaml = yaml.safe_load(
        (STATE_MACHINE_ROOT / "states" / "states.yaml").read_text()
    )
    state_ids = {s["id"] for s in states_yaml["states"]}
    checked_any = False
    for reason, actions in matrix.items():
        for action, spec in actions.items():
            ns = spec["next_state"]
            if "{" in ns:
                continue
            checked_any = True
            assert ns in state_ids, f"{reason}.{action} -> unknown state {ns!r}"
    assert checked_any
