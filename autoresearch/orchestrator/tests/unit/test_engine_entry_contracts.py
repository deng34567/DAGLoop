"""Cross-interface checks for orchestrator entry events.

The in-process bus used by Web and most smoke tests does not validate payloads.
These tests run the real EventBus payload validators against the exact payloads
built by the production engine so schema drift cannot silently break a deployed
broker.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from autoresearch.eventbus.broker.registry import Registry
from autoresearch.orchestrator.engine import OrchestratorEngine
from autoresearch.orchestrator.session_state import SessionState


ROOT = Path(__file__).resolve().parents[3]
SKILL_ROOT = ROOT / "orchestrator" / "state-machine"
EVENTBUS_ROOT = ROOT / "eventbus"


class _Client:
    pass


class _Store:
    def __init__(self, objects: dict[str, dict]):
        self.objects = objects

    def get(self, ref: str) -> dict:
        return self.objects[ref]


@pytest.mark.parametrize(
    ("state_name", "state_values"),
    [
        ("GeneratingHypotheses", {"research_memory_ref": None}),
        ("GeneratingHypotheses", {"research_memory_ref": "spec://research_memory/s/v1"}),
        ("ModelingProblem", {"human_idea_seed_ref": None}),
        ("ModelingProblem", {"human_idea_seed_ref": "spec://human_idea_seed/s/v1"}),
        (
            "GeneratingMethod",
            {
                "method_ref": "spec://method/s/prior",
                "method_not_working": True,
            },
        ),
        (
            "GeneratingMethod",
            {
                "method_ref": "spec://method/s/prior-component",
                "component_ablation_failed": True,
                "ablation_failed_component": "component-a",
            },
        ),
        (
            "DeliberatingMethod",
            {"method_ref": "spec://method/s/draft"},
        ),
        (
            "RevisingMethod",
            {
                "method_ref": "spec://method/s/prior",
                "current_moss_reason": "method_not_working",
            },
        ),
        # Benchmark mainline: plan the confirmed method; the retired verifier and
        # hypothesis-portfolio refs are deliberately absent.
        (
            "PlanningExperiment",
            {
                "method_ref": "spec://method/s/v1",
                "hypothesis_bank_ref": None,
                "current_hypothesis_id": None,
                "verifier_spec_ref": None,
            },
        ),
        # Lab/portfolio path: retain the older selected-hypothesis interface.
        (
            "PlanningExperiment",
            {
                "method_ref": None,
                "hypothesis_bank_ref": "spec://hypothesis_bank/s/v1",
                "current_hypothesis_id": "hyp-1",
                "verifier_spec_ref": "spec://verifier_spec/s/v1",
            },
        ),
        ("SelectingBenchmark", {"deliberation_report_ref": None}),
        (
            "SelectingBenchmark",
            {"deliberation_report_ref": "spec://deliberation_report/s/v1"},
        ),
        ("ReviewingExperimentPlan", {"verifier_spec_ref": None}),
        (
            "ConsolidatingMemory",
            {
                "method_ref": "spec://method/s/v1",
                "component_revision_round": 10,
                "method_feedback_ref": "spec://method_improvement_review/s/applied",
            },
        ),
        (
            "ReviewingExperimentPlan",
            {"verifier_spec_ref": "spec://verifier_spec/s/v1"},
        ),
        ("ImplementingExperiment", {"verifier_spec_ref": None}),
        (
            "ImplementingExperiment",
            {"verifier_spec_ref": "spec://verifier_spec/s/v1"},
        ),
        (
            "EvaluatingExperiment",
            {"verifier_spec_ref": None, "research_memory_ref": None},
        ),
        (
            "EvaluatingExperiment",
            {
                "verifier_spec_ref": "spec://verifier_spec/s/v1",
                "research_memory_ref": "spec://research_memory/s/v1",
            },
        ),
        (
            "RunningExperiment",
            {
                "experiment_patch_refs": ["spec://experiment_patch/s/v1"],
                "verifier_spec_ref": "spec://verifier_spec/s/v1",
                "baseline_result_ref": "spec://baseline_result/s/v1",
            },
        ),
    ],
)
def test_entry_payload_matches_registered_schema(
    state_name: str, state_values: dict[str, str | None]
) -> None:
    engine = OrchestratorEngine(client=_Client(), skill_root=SKILL_ROOT)
    registry = Registry(EVENTBUS_ROOT / "eventbus-registry.yaml")
    state = SessionState(session_id="00000000-0000-4000-8000-000000000001")
    state.research_spec_ref = "spec://research_spec/s/v1"
    state.literature_brief_ref = "spec://literature_brief/s/v1"
    state.spec_interpretation_ref = "spec://spec_interpretation/s/v1"
    state.novelty_audit_report_ref = "spec://novelty_audit_report/s/v1"
    state.problem_model_ref = "spec://problem_model/s/v1"
    state.research_memory_ref = "spec://research_memory/s/v1"
    state.benchmark_spec_ref = "spec://benchmark_spec/s/v1"
    state.baseline_result_ref = "spec://baseline_result/s/v1"
    state.experiment_plan_refs = ["spec://experiment_plan/s/v1"]
    state.run_record_refs = ["spec://run_record/s/v1"]
    for name, value in state_values.items():
        setattr(state, name, value)

    event_type, build_payload = engine._entry_actions[state_name]
    payload = build_payload(state, None)

    assert list(registry.get_payload_validator(event_type, 1).iter_errors(payload)) == []
    if state_name == "SelectingBenchmark":
        assert payload["spec_interpretation_ref"] == state.spec_interpretation_ref
    if state_name == "RunningExperiment":
        assert payload["benchmark_spec_ref"] == state.benchmark_spec_ref
    if state_name == "GeneratingMethod":
        assert payload["prior_method_ref"] == state.method_ref
        assert payload["research_memory_ref"] == state.research_memory_ref
    if state_name in {
        "GeneratingMethod", "DeliberatingMethod", "RevisingMethod",
        "PlanningExperiment", "ReviewingExperimentPlan",
    }:
        assert payload["baseline_result_ref"] == state.baseline_result_ref
    if state_name in {"GeneratingMethod", "DeliberatingMethod", "RevisingMethod"}:
        assert payload["spec_interpretation_ref"] == state.spec_interpretation_ref


def test_replanning_entry_carries_latest_plan_review() -> None:
    engine = OrchestratorEngine(client=_Client(), skill_root=SKILL_ROOT)
    state = SessionState(
        session_id="00000000-0000-4000-8000-000000000001",
        current_state="PlanningExperiment",
        previous_state="ReviewingExperimentPlan",
        method_ref="spec://method/s/v1",
        benchmark_spec_ref="spec://benchmark_spec/s/v1",
        research_spec_ref="spec://research_spec/s/v1",
        critique_report_refs=["spec://critique_report/s/plan-fail"],
    )

    _, build_payload = engine._entry_actions["PlanningExperiment"]
    payload = build_payload(state, None)

    assert payload["plan_review_ref"] == "spec://critique_report/s/plan-fail"


def test_moss_entry_carries_latest_plan_review_as_diagnostic() -> None:
    engine = OrchestratorEngine(client=_Client(), skill_root=SKILL_ROOT)
    registry = Registry(EVENTBUS_ROOT / "eventbus-registry.yaml")
    state = SessionState(
        session_id="00000000-0000-4000-8000-000000000001",
        current_state="MossIntervening",
        current_moss_reason="experiment_design_thrashing",
        critique_report_refs=["spec://critique_report/s/plan-fail"],
    )

    event_type, build_payload = engine._entry_actions["MossIntervening"]
    payload = build_payload(state, None)

    assert list(registry.get_payload_validator(event_type, 1).iter_errors(payload)) == []
    assert payload["critique_report_ref"] == "spec://critique_report/s/plan-fail"


@pytest.mark.parametrize("action,expected", [
    ("fix_environment", True), ("revalidate_baseline", True), ("change_plan", False),
])
def test_baseline_entry_carries_only_applicable_recovery(action, expected):
    engine = OrchestratorEngine(client=_Client(), skill_root=SKILL_ROOT)
    registry = Registry(EVENTBUS_ROOT / "eventbus-registry.yaml")
    state = SessionState(
        session_id="00000000-0000-4000-8000-000000000001",
        benchmark_spec_ref="spec://benchmark_spec/s/v1",
        moss_interventions=[{"action": action, "moss_intervention_ref": "spec://moss_intervention/s/fix"}],
    )
    event_type, build_payload = engine._entry_actions["ReproducingBaseline"]
    payload = build_payload(state, None)
    assert (payload.get("moss_intervention_ref") == "spec://moss_intervention/s/fix") is expected
    assert list(registry.get_payload_validator(event_type, 1).iter_errors(payload)) == []


def test_moss_entry_carries_failed_baseline_without_freezing_it() -> None:
    engine = OrchestratorEngine(client=_Client(), skill_root=SKILL_ROOT)
    registry = Registry(EVENTBUS_ROOT / "eventbus-registry.yaml")
    state = SessionState(
        session_id="00000000-0000-4000-8000-000000000001",
        current_state="MossIntervening",
        current_moss_reason="baseline_reproduction_thrashing",
        last_baseline_attempt_ref="spec://baseline_result/s/failed",
    )
    event_type, build_payload = engine._entry_actions["MossIntervening"]
    payload = build_payload(state, None)
    assert list(registry.get_payload_validator(event_type, 1).iter_errors(payload)) == []
    assert payload["baseline_attempt_ref"] == state.last_baseline_attempt_ref
    assert state.baseline_result_ref is None


@pytest.mark.parametrize("previous_state,reason,expected", [
    ("RunningExperiment", "component_ablation_failed", True),
    ("ConsolidatingMemory", "component_ablation_failed", False),
    ("RunningExperiment", "result_suspicious", False),
])
def test_moss_entry_scopes_current_agent_failure(previous_state, reason, expected) -> None:
    engine = OrchestratorEngine(client=_Client(), skill_root=SKILL_ROOT)
    registry = Registry(EVENTBUS_ROOT / "eventbus-registry.yaml")
    failure = {"agent_name": "RunnerAgent", "reason_classification": "permanent",
               "underlying_reason_code": "component_plan_invalid",
               "underlying_reason": "dependency checkpoint conflict for shared_driver.cpp"}
    state = SessionState(
        session_id="00000000-0000-4000-8000-000000000001",
        current_state="MossIntervening", previous_state=previous_state,
        current_moss_reason=reason, last_agent_failure=failure,
    )
    event_type, build_payload = engine._entry_actions["MossIntervening"]
    payload = build_payload(state, None)
    assert list(registry.get_payload_validator(event_type, 1).iter_errors(payload)) == []
    assert payload["agent_failure"] == (failure if expected else None)


def test_moss_runtime_failure_does_not_receive_stale_plan_review() -> None:
    engine = OrchestratorEngine(client=_Client(), skill_root=SKILL_ROOT)
    registry = Registry(EVENTBUS_ROOT / "eventbus-registry.yaml")
    state = SessionState(
        session_id="00000000-0000-4000-8000-000000000001",
        current_state="MossIntervening",
        current_moss_reason="result_suspicious",
        run_record_refs=["spec://run_record/s/dependency-fail"],
        critique_report_refs=["spec://critique_report/s/old-plan-pass"],
    )

    event_type, build_payload = engine._entry_actions["MossIntervening"]
    payload = build_payload(state, None)

    assert list(registry.get_payload_validator(event_type, 1).iter_errors(payload)) == []
    assert payload["run_record_ref"] == "spec://run_record/s/dependency-fail"
    assert payload["critique_report_ref"] is None


def test_component_revision_carries_latest_moss_guidance() -> None:
    intervention_ref = "spec://moss_intervention/s/component-v1"
    intervention = {
        "reason": "component_ablation_failed",
        "action": "revise_component",
        "prompt_patch": {
            "agent": "ExperimentPlannerAgent",
            "content": "Use a repository-verifiable component or give a complete mechanism.",
        },
        "rationale": "The named optimizer has no implementation or formula.",
        "degraded": False,
    }
    engine = OrchestratorEngine(
        client=_Client(), skill_root=SKILL_ROOT,
        spec_store=_Store({intervention_ref: intervention}),
    )
    state = SessionState(
        session_id="00000000-0000-4000-8000-000000000001",
        current_state="RevisingMethod",
        problem_model_ref="spec://problem_model/s/v1",
        literature_brief_ref="spec://literature_brief/s/v1",
        method_ref="spec://method/s/v1",
        baseline_result_ref="spec://baseline_result/s/v1",
        component_ablation_failed=True,
        ablation_failed_component="undefined optimizer",
        moss_interventions=[{
            "reason": "component_ablation_failed",
            "action": "revise_component",
            "moss_intervention_ref": intervention_ref,
            "degraded": False,
        }],
    )

    _, build_payload = engine._entry_actions["RevisingMethod"]
    feedback = build_payload(state, None)["experiment_feedback"]

    assert "undefined optimizer" in feedback
    assert "repository-verifiable component" in feedback
    assert "no implementation or formula" in feedback
