"""
YAML loader: parses the AutoResearch orchestrator state-machine YAML files into
in-memory data structures used by the engine.

Ported from vibe/orchestrator/loader.py and adapted to AutoResearch V1.10:
  - MossActionMatrix and the UserDecision reason-matrix live in their own files
    (moss/, user-decisions/) rather than inside transitions.yaml.
  - No entry-actions / agent-failure-routing / eventbus-contract files (those
    are out of scope for this skill), so those loads are dropped.

Performs startup consistency checks; fails fast on any spec inconsistency.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml

logger = logging.getLogger(__name__)


@dataclass
class StateInfo:
    id: str
    phase: str
    role: Any
    state_type: str
    durable: bool
    note: Optional[str] = None
    parallel: bool = False


@dataclass
class TransitionRule:
    """One transition rule under a state.

    Fields mirror the structure in transitions.yaml entries.
    """
    on_event: str
    guard: Optional[str] = None
    counter_updates: list[Any] = field(default_factory=list)
    side_effects: list[Any] = field(default_factory=list)
    threshold_check: Optional[list[dict]] = None
    next_state: Optional[str] = None
    next_branch: Optional[list[dict]] = None
    else_branch: Optional[dict] = None
    emit: Optional[Any] = None
    dispatch_table: Optional[str] = None
    raw: dict = field(default_factory=dict)


@dataclass
class CounterSpec:
    id: str
    name: str
    type: str           # scalar | map
    scope: str
    threshold: int
    threshold_action: dict


@dataclass
class LoadedSkill:
    """All loaded YAML structures."""
    states: dict[str, StateInfo]
    transitions: dict[str, list[TransitionRule]]
    moss_action_matrix: dict[str, dict[str, dict]]    # reason -> action -> spec
    counters: dict[str, CounterSpec]
    user_decision_matrix: dict[str, dict[str, dict]]  # reason -> action -> spec


def load_skill(skill_root: Path) -> LoadedSkill:
    """Load all skill YAMLs and run consistency checks."""
    skill_root = Path(skill_root)

    # 1. states
    states_yaml = yaml.safe_load((skill_root / "states/states.yaml").read_text())
    states = {}
    for s in states_yaml["states"]:
        states[s["id"]] = StateInfo(
            id=s["id"],
            phase=s.get("phase", ""),
            role=s.get("role"),
            state_type=s["state_type"],
            durable=s.get("durable", True),
            note=s.get("note"),
            parallel=s.get("parallel", False),
        )

    # 2. transitions
    trans_yaml = yaml.safe_load((skill_root / "transitions/transitions.yaml").read_text())
    transitions: dict[str, list[TransitionRule]] = {}
    for entry in trans_yaml["state_transitions"]:
        state_id = entry["state"]
        rules = []
        for t in entry.get("transitions", []):
            rules.append(TransitionRule(
                on_event=t.get("on_event", ""),
                guard=t.get("guard"),
                counter_updates=t.get("counter_updates", []),
                side_effects=t.get("side_effects", []),
                threshold_check=t.get("threshold_check"),
                next_state=t.get("next_state"),
                next_branch=t.get("next_branch"),
                else_branch=t.get("else_branch"),
                emit=t.get("emit"),
                dispatch_table=t.get("dispatch_table"),
                raw=t,
            ))
        transitions[state_id] = rules

    # 3. counters
    counters_yaml = yaml.safe_load((skill_root / "counters/counters.yaml").read_text())
    counters = {}
    for c in counters_yaml["counters"]:
        counters[c["id"]] = CounterSpec(
            id=c["id"],
            name=c["name"],
            type=c["type"],
            scope=c.get("scope", ""),
            threshold=c.get("threshold", 0),
            threshold_action=c.get("threshold_action", {}),
        )

    # 4. MOSS action matrix (its own file in AutoResearch)
    moss_yaml = yaml.safe_load((skill_root / "moss/moss-action-matrix.yaml").read_text())
    moss_matrix: dict[str, dict[str, dict]] = {}
    for entry in moss_yaml.get("MossActionMatrix", []):
        reason = entry["reason"]
        by_action = {}
        for a in entry.get("actions", []):
            by_action[a["action"]] = a
        moss_matrix[reason] = by_action

    # 5. user-decisions reason-matrix
    udm_yaml = yaml.safe_load(
        (skill_root / "user-decisions/reason-matrix.yaml").read_text()
    )
    user_decision_matrix: dict[str, dict[str, dict]] = {}
    for entry in udm_yaml.get("reasons", []):
        reason = entry["reason"]
        by_action = {}
        for a in entry.get("actions", []):
            by_action[a["action"]] = a
        user_decision_matrix[reason] = by_action

    loaded = LoadedSkill(
        states=states,
        transitions=transitions,
        moss_action_matrix=moss_matrix,
        counters=counters,
        user_decision_matrix=user_decision_matrix,
    )

    _run_consistency_checks(loaded)
    return loaded


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
    "method_not_working",   # 会议② (2026-07-31): method falsified -> swap or report
    "component_ablation_failed",  # 会议 (2026-08-03) 乙: a declared component's ablation checkpoint
                                  # fails -> MOSS routes back to the method layer (revise / rethink)
}

# Counters A .. M per V1.10 §4.
EXPECTED_COUNTERS = set("ABCDEFGHIJKLM")


def _run_consistency_checks(s: LoadedSkill) -> None:
    """Fail-fast on spec inconsistencies."""
    state_ids = set(s.states.keys())

    def _resolves(ns: Any) -> bool:
        """A concrete (non-template) target must be a known state."""
        return not str(ns).startswith("{")

    # (1) transitions.yaml: every concrete next_state target exists.
    def _collect_next_states(rules: list[TransitionRule]) -> set[str]:
        result = set()
        for r in rules:
            if r.next_state and _resolves(r.next_state):
                result.add(r.next_state)
            for nb in r.next_branch or []:
                ns = nb.get("next_state")
                if ns and _resolves(ns):
                    result.add(str(ns))
            for tc in r.threshold_check or []:
                if "next_state" in tc and _resolves(tc["next_state"]):
                    result.add(str(tc["next_state"]))
        return result

    for state_id, rules in s.transitions.items():
        for ns in _collect_next_states(rules):
            if ns not in state_ids:
                raise ValueError(
                    f"transitions.yaml: state {state_id} references unknown next_state '{ns}'"
                )

    # (2) MossActionMatrix covers all 14 MOSS reasons, and every concrete
    #     action next_state exists.
    missing = EXPECTED_MOSS_REASONS - s.moss_action_matrix.keys()
    if missing:
        raise ValueError(f"MossActionMatrix missing reasons: {missing}")
    extra = s.moss_action_matrix.keys() - EXPECTED_MOSS_REASONS
    if extra:
        raise ValueError(f"MossActionMatrix has unexpected reasons: {extra}")
    for reason, actions in s.moss_action_matrix.items():
        for action_name, spec in actions.items():
            ns = spec.get("next_state")
            if ns and _resolves(ns) and ns not in state_ids:
                raise ValueError(
                    f"MossActionMatrix[{reason}].{action_name} -> unknown state '{ns}'"
                )

    # (3) UserDecisionMatrix: every reason has 'abandon', and every concrete
    #     action next_state exists.
    for reason, actions in s.user_decision_matrix.items():
        if "abandon" not in actions:
            raise ValueError(f"UserDecisionMatrix reason {reason} missing 'abandon' action")
        for action_name, spec in actions.items():
            ns = spec.get("next_state")
            if ns and _resolves(ns) and ns not in state_ids:
                raise ValueError(
                    f"UserDecisionMatrix[{reason}].{action_name} -> unknown state '{ns}'"
                )

    # (4) counters A .. M all present.
    missing_counters = EXPECTED_COUNTERS - s.counters.keys()
    if missing_counters:
        raise ValueError(f"counters.yaml missing counters: {missing_counters}")

    logger.info(
        "Skill loaded: %d states / %d transition rules / %d MOSS reasons / %d "
        "UserDecision reasons / %d counters",
        len(s.states),
        sum(len(rs) for rs in s.transitions.values()),
        len(s.moss_action_matrix),
        len(s.user_decision_matrix),
        len(s.counters),
    )
