"""Tests for the MOSS action-matrix loader + MossActionGuard.

Loads the EXISTING committed SSOT
(autoresearch/orchestrator/state-machine/moss/moss-action-matrix.yaml) -- no
new matrix YAML is introduced here. Two cross-checks are the key value of
this module: every next_state referenced by the matrix must be a real state
in states.yaml, and every counter_updates string must parse against a real
counter id (A-M) and, for "delete key <field>", a real SessionState field.

Pure yaml/logic tests: no LLM, no execution, no Redis, no network.
"""

import re
from pathlib import Path

import pytest
import yaml

from autoresearch.orchestrator.moss_matrix import (
    MossActionGuard,
    load_moss_action_matrix,
)
from autoresearch.orchestrator.session_state import SessionState

STATE_MACHINE_ROOT = Path(__file__).resolve().parents[2] / "state-machine"

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

COUNTER_UPDATE_RE = re.compile(r"^([A-M]) (reset to 0|delete key ([a-z_]+))$")


@pytest.fixture(scope="module")
def matrix():
    return load_moss_action_matrix()


@pytest.fixture(scope="module")
def guard(matrix):
    return MossActionGuard(matrix)


def test_matrix_loads_all_reasons(matrix):
    assert len(matrix) == 17   # 14 core + two method/component reasons + generic agent recovery
    assert set(matrix) == EXPECTED_MOSS_REASONS


def test_every_reason_has_at_least_one_action(matrix):
    for reason, actions in matrix.items():
        assert len(actions) >= 1, reason


def test_is_legal_true_for_known_action(guard):
    assert guard.is_legal("direction_thrashing", "rewrite_framing") is True


def test_is_legal_false_for_bogus_action(guard):
    assert guard.is_legal("direction_thrashing", "bogus") is False


def test_is_legal_false_for_unknown_reason(guard):
    assert guard.is_legal("no_such_reason", "rewrite_framing") is False


def test_legal_actions_empty_for_unknown_reason(guard):
    assert guard.legal_actions("no_such_reason") == []


def test_method_not_working_must_rethink_while_swap_budget_remains(guard):
    """The engine reports directly once the swap cap is exhausted.

    Reaching MOSS with this reason therefore proves that another swap is
    still available; MOSS must not terminate the large method loop early.
    """
    assert guard.legal_actions("method_not_working") == ["rethink_method"]
    assert guard.next_state("method_not_working", "rethink_method") == "GeneratingMethod"


def test_component_failure_cannot_report_before_engine_cap(guard):
    """Reaching MOSS means the engine's component revision budget remains.

    The engine reports directly once that budget is exhausted, so exposing
    ``report_now`` here would let MOSS terminate the method loop early.
    """
    assert guard.legal_actions("component_ablation_failed") == [
        "revise_component", "revise_method",
    ]


def test_next_state_rewrite_framing(guard):
    assert guard.next_state("direction_thrashing", "rewrite_framing") == "ModelingProblem"


def test_with_reason_escalate_to_user(guard):
    assert (
        guard.with_reason("direction_thrashing", "escalate_to_user")
        == "direction_thrashing"
    )


def test_counter_updates_rewrite_framing(guard):
    assert guard.counter_updates("direction_thrashing", "rewrite_framing") == [
        "A reset to 0"
    ]


def test_with_reason_none_when_absent(guard):
    assert guard.with_reason("direction_thrashing", "rewrite_framing") is None


def test_fallback_is_always_legal_for_every_reason(guard, matrix):
    for reason in matrix:
        action = guard.fallback(reason)
        assert action in guard.legal_actions(reason), (reason, action)


def test_fallback_prefers_escalate_to_user(guard):
    assert guard.fallback("direction_thrashing") == "escalate_to_user"


def test_fallback_prefers_request_human_when_no_escalate(guard):
    # theory_thrashing has no escalate_to_user, no request_human_*/human_data_review,
    # no research_rework -> falls through to legal_actions[0].
    assert guard.fallback("result_inconsistency") == "research_rework"


# --- Cross-check 1: every next_state in the matrix is a real state id. ---


def test_every_next_state_is_a_real_state(matrix):
    states_yaml = yaml.safe_load(
        (STATE_MACHINE_ROOT / "states/states.yaml").read_text()
    )
    state_ids = {s["id"] for s in states_yaml["states"]}
    for reason, actions in matrix.items():
        for action, spec in actions.items():
            ns = spec["next_state"]
            if ns and not str(ns).startswith("{"):
                assert ns in state_ids, f"{reason}.{action} -> unknown state {ns!r}"


# --- Cross-check 2: every counter_updates string parses against real ---
# --- counter ids (A-M) and real SessionState fields.                 ---


def test_every_counter_update_string_parses(matrix):
    counters_yaml = yaml.safe_load(
        (STATE_MACHINE_ROOT / "counters/counters.yaml").read_text()
    )
    counter_ids = {c["id"] for c in counters_yaml["counters"]}
    sample_state = SessionState(session_id="test")

    found_any = False
    for reason, actions in matrix.items():
        for action, spec in actions.items():
            for update in spec["counter_updates"]:
                found_any = True
                m = COUNTER_UPDATE_RE.match(update)
                assert m, f"{reason}.{action}: unparseable counter_update {update!r}"
                letter = m.group(1)
                assert letter in counter_ids, (
                    f"{reason}.{action}: unknown counter id {letter!r}"
                )
                field_name = m.group(3)
                if field_name is not None:
                    assert hasattr(sample_state, field_name), (
                        f"{reason}.{action}: unknown SessionState field "
                        f"{field_name!r}"
                    )
    assert found_any
