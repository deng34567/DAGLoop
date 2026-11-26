from __future__ import annotations

import json

import pytest

from autoresearch.agents.comparison_guardian_agent.core import (
    ComparisonGuardianParseError,
    _load_semantic_system_prompt,
    audit_comparison_integrity,
    audit_repository_comparison,
    audit_semantic_restatement,
)
from autoresearch.agents.evaluator_agent.tests.conftest import FakeLLM


SEMANTICS = {
    "comparison_statement": (
        "On one GPU of the same class, with equal model scale, data, tokenizer, canonical "
        "evaluator, and 300 seconds of pure training for baseline and method, compare final "
        "val_bpb; training steps, tokens, and throughput are observed outcomes."
    ),
    "research_question": "Does the method lower final val_bpb after 300 seconds of training?",
    "primary_metric": {"name": "val_bpb", "direction": "minimize"},
    "comparison_mode": "quality_at_fixed_time",
    "comparison_basis": "Compare model quality at the same pure-training time budget.",
    "budget_interpretation": "Each side receives 300 seconds of pure training.",
    "conditions_to_hold_constant": [
        "one GPU of the same class", "model scale", "dataset", "tokenizer",
        "canonical evaluator", "300-second pure-training window",
    ],
    "method_freedom": [
        "training algorithm and necessary hyperparameters",
        "completed steps, tokens, and throughput within the fixed time window",
    ],
    "evidence": [{
        "source_type": "research_spec", "source_id": "research_spec",
        "supports": "The task asks for quality after a fixed training duration.",
    }],
    "uncertainties": [],
}


def _benchmark() -> dict:
    return {"metric": "val_bpb", "evaluation_semantics": SEMANTICS}


def _record(report: str) -> dict:
    return {
        "succeeded": True, "metric_value": 0.91, "runtime_seconds": 330.0,
        "environment": "repo_improve|verdict=new_best|baseline=0.94|method=0.91",
        "stdout_excerpt": report,
    }


def test_comparison_guardian_uses_reasoning_safe_output_limit():
    class RecordingLLM(FakeLLM):
        def __init__(self, responses):
            super().__init__(responses)
            self.max_tokens = []

        def chat(self, *, messages, response_format=None, max_tokens=None,
                 stream=None, stream_max_s=None):
            self.max_tokens.append(max_tokens)
            return super().chat(
                messages=messages,
                response_format=response_format,
                max_tokens=max_tokens,
                stream=stream,
                stream_max_s=stream_max_s,
            )

    llm = RecordingLLM([json.dumps({
        "verdict": "pass", "issues": [], "remediation": None,
        "reasoning": "The restatement preserves the comparison.",
    })])

    audit_semantic_restatement("Optimize the official metric.", {}, {}, llm)

    assert llm.max_tokens == [262144]


def test_indirect_probe_audit_receives_actual_source_and_observation_contract():
    scope = {"mode": "component", "role": "indirect_gain", "acceptance_check": "fraction > 0",
             "validation_contract": {"implementation_check": "Measure actual routed events."}}
    probe = {"attempt_id": "attempt-probe", "command_sources_stable": True,
             "command_sources": [{"path": "/tmp/check.py", "source": "print({'fraction': 0.02})"}],
             "verified_output_fields": {"/fraction": 0.02}}
    run = {"candidate": {"metric": None, "valid": False},
           "component_attempt_id": "attempt-probe", "attempt_review_evidence": [probe],
           "ablation_report": [{"actual": 0.02, "_host_bound": True}]}
    llm = FakeLLM([json.dumps({"verdict": "insufficient_evidence",
        "issues": ["The script does not measure routed events."],
        "remediation": "Exercise the actual candidate.", "reasoning": "A configuration is not an observation."})])
    audit = audit_repository_comparison("Optimize quality.", None, run, llm,
                                       execution_phase="method", component_scope=scope)
    context = json.loads(llm.calls[0][1]["content"])
    assert context["component_scope"] == scope
    assert context["actual_result"]["attempt_review_evidence"] == [probe]
    assert context["actual_result"]["ablation_report"] == run["ablation_report"]
    assert "actual candidate calls" in context["indirect_measurement_check"]
    assert "Source captured after execution is diagnostic" in context["indirect_measurement_check"]
    system = llm.calls[0][0]["content"]
    assert "independent reimplementation" in system
    assert "not observed sampled events" in system
    assert "Zero failures with no actual checks" in system
    assert audit["verdict"] == "insufficient_evidence"


@pytest.mark.parametrize("phase,scope,component_review", [
    ("method", {"mode": "component", "role": "direct_gain"}, True),
    ("method", {"mode": "component", "role": "indirect_gain"}, True),
    ("method", {"mode": "integration"}, False),
    ("method", None, False),
    ("baseline", {"mode": "component"}, False),
])
def test_repository_audit_uses_host_stage_for_component_evidence(
        phase, scope, component_review):
    run = {"candidate": {"metric": 4.99, "valid": True},
           "component_scope": {"mode": "component"}}
    llm = FakeLLM([json.dumps({
        "verdict": "insufficient_evidence", "issues": ["Missing component validity evidence."],
        "remediation": "Provide the required component check.",
        "reasoning": "The component contract still requires an observed validity check.",
    })])
    audit = audit_repository_comparison(
        "Integrate all components and run the final official evaluator.", {}, run, llm,
        execution_phase=phase, component_scope=scope)
    context = json.loads(llm.calls[0][1]["content"])
    assert ("component_measurement_check" in context) is component_review
    if component_review:
        check = context["component_measurement_check"]
        assert "validation_contract" in check
        assert "final integration" in check
        assert "unless required by this component" in check
    assert audit["verdict"] == "insufficient_evidence"
    assert run["candidate"] == {"metric": 4.99, "valid": True}


def test_component_parent_comparison_keeps_frozen_baseline_as_context():
    scope = {"mode": "component", "role": "direct_gain",
             "validation_contract": {"parent": "dependency checkpoint",
                 "comparison": "Compare parent and candidate on the same local inputs."}}
    run = {"baseline": {"metric": 10.0, "valid": True},
           "candidate": {"metric": 8.0, "valid": True},
           "ablation_report": [{"parent_metric": 9.0, "actual": 8.0,
                                 "_host_bound": True}]}
    llm = FakeLLM([json.dumps({"verdict": "pass", "issues": [], "remediation": None,
        "reasoning": "The bound component comparison is 9.0 versus 8.0, not 10.0 versus 8.0."})])

    audit_repository_comparison(
        "Optimize the official metric.", None, run, llm,
        execution_phase="method", baseline_result={"metric_value": 10.0},
        component_scope=scope)

    context = json.loads(llm.calls[0][1]["content"])
    check = context["component_measurement_check"]
    assert "declared component parent" in check
    assert "contextual frozen baseline" in check
    assert "does not establish an official whole-method score" in check
    assert context["actual_result"]["baseline"] == run["baseline"]
    assert context["actual_result"]["ablation_report"] == run["ablation_report"]


def test_repository_audit_receives_attempt_reference_policy():
    trace = [{"attempt_id": "attempt-reference", "stdout_sha256": "a" * 64,
              "verified_output_fields": {
                  "/method": 1.25, "/notes/reference_raw/rate": 100,
                  "/notes/reference_policy": "stored reference from an earlier run",
                  "/notes/formula": "candidate_rate / reference_rate"}}]
    run = {"metric": "rate_ratio", "candidate": {"metric": 1.25, "valid": True},
           "repository_contract": {"sources": [{"path": "instruction.md",
                "content": "Measure reference and candidate together in each evaluation."}]},
           "attempt_review_evidence": trace}
    llm = FakeLLM([json.dumps({
        "verdict": "comparison_changed", "issues": ["An earlier reference was substituted."],
        "remediation": "Measure the reference in this evaluation.",
        "reasoning": "The native formula uses a stored reference rather than the required pair.",
    })])

    audit = audit_repository_comparison("Optimize the official ratio.", None, run, llm,
                                       execution_phase="method")

    context = json.loads(llm.calls[0][1]["content"])
    assert context["actual_result"]["attempt_review_evidence"] == trace
    assert "each evaluation" in str(context["actual_result"]["repository_contract"])
    assert audit["verdict"] == "comparison_changed"


def test_semantic_guardian_rejects_cross_unit_paired_metric_baselines():
    prompt = " ".join(_load_semantic_system_prompt().lower().split())
    assert "derived paired metric" in prompt
    assert "the same directly measured quantity and unit" in prompt
    assert "only after both runs" in prompt


def test_semantic_guardian_preserves_constrained_objective():
    prompt = " ".join(_load_semantic_system_prompt().split())
    assert "optimization objective from feasibility constraints" in prompt
    assert "A constraint measurement is not the primary objective" in prompt


@pytest.mark.parametrize("phase", ["baseline", "method"])
def test_repository_audit_receives_constrained_objective_without_metric_substitution(phase):
    task = "Minimize encoded bytes subject to reconstruction error <= 0.01."
    semantics = {"primary_metric": {"name": "reconstruction_error", "direction": "minimize"}}
    run = {"metric": "reconstruction_error", "baseline": {"metric": 0.02, "valid": True}}
    llm = FakeLLM([json.dumps({
        "verdict": "baseline_invalid", "issues": ["Error replaced the encoded-size objective."],
        "remediation": "Record bytes and the separate error constraint result.",
        "reasoning": "The scalar is real but measures feasibility, not the requested objective.",
    })])
    audit = audit_repository_comparison(task, semantics, run, llm, execution_phase=phase)
    system = llm.calls[0][0]["content"]
    context = json.loads(llm.calls[0][1]["content"])
    assert "optimization objective from feasibility constraints" in system
    assert "A constraint measurement is not the primary objective" in system
    assert "This does not waive objective identity" in system
    assert "binding to that existing measurement" in system
    assert context["original_task"] == task
    assert context["execution_phase"] == phase
    assert audit["verdict"] == "baseline_invalid"


@pytest.mark.parametrize("target_reached", [False, True])
def test_repository_audit_distinguishes_goal_conditioned_zero_from_valid_zero(target_reached):
    task = "Minimize energy required to reach the declared quality target."
    run = {
        "metric": "energy_to_target", "baseline": {"metric": 0, "valid": True},
        "config": {"target_reached_by_baseline": target_reached},
    }
    expected = "pass" if target_reached else "baseline_invalid"
    llm = FakeLLM([json.dumps({
        "verdict": expected,
        "issues": [] if target_reached else ["The target has not been reached."],
        "remediation": None if target_reached else "Establish energy to reach the target.",
        "reasoning": "Goal attainment determines whether this conditional metric is defined.",
    })])
    result = audit_repository_comparison(
        task, {"primary_metric": {"name": "energy_to_target", "direction": "minimize"}},
        run, llm, execution_phase="baseline",
    )
    system = llm.calls[0][0]["content"]
    context = json.loads(llm.calls[0][1]["content"])
    assert "goal-conditioned metric" in system
    assert "an unattained goal is not zero cost to attain it" in system
    assert "explicit finite failure penalty" in system
    assert "A legitimate zero result remains valid" in system
    assert "first state the original optimized quantity" in context["metric_definition_check"]
    assert "method-only" in context["metric_definition_check"]
    assert "return baseline_invalid" in context["metric_definition_check"]
    assert context["actual_result"]["config"]["target_reached_by_baseline"] is target_reached
    assert result["verdict"] == expected


def test_baseline_definition_check_separates_static_cost_from_eligibility():
    llm = FakeLLM([json.dumps({"verdict": "pass", "issues": [], "remediation": None,
                              "reasoning": "Measured bytes are defined; score remains zero."})])
    run = {"metric": "bytes", "baseline": {"metric": 1200, "valid": True},
           "repository_contract": {"sources": [{"path": "task.toml", "content":
               '[optimization]\nmetric="bytes"\ndirection="lower"'}]}}
    audit_repository_comparison("Optimize the repository-defined objective.",
        {"primary_metric": {"name": "repository-defined", "direction": "maximize"}},
        run, llm, execution_phase="baseline")
    context = json.loads(llm.calls[0][1]["content"])
    rule = context["metric_definition_check"]
    assert "static measured objective" in rule
    assert "not the agent's cost-to-target label" in rule
    assert "explicitly deferred generated" in rule
    assert "without claiming feasibility or weakening method gates" in rule
    assert "reproduction validity, NOT whether the starter qualifies" in rule
    assert "do not require a feasible starter" in rule
    assert context["actual_result"]["repository_contract"] == run["repository_contract"]


@pytest.mark.parametrize("phase", ["baseline", "method"])
def test_frozen_starter_eligibility_is_distinct_from_candidate_eligibility(phase):
    frozen = {
        "metric_name": "encoded_bytes", "metric_value": 1200, "reproduced": True,
        "measurement_evidence": {"value": 1200, "valid": True, "attempt_id": "starter"},
        "log_excerpt": "Official output: error=0.04; feasible=false; score=0.",
    }
    candidate = {"metric": 900, "valid": True, "error": 0.03}
    verdict = "comparison_changed" if phase == "method" else "pass"
    llm = FakeLLM([json.dumps({
        "verdict": verdict,
        "issues": ["Candidate eligibility is falsely asserted."] if phase == "method" else [],
        "remediation": "Bind and enforce candidate error <= 0.01." if phase == "method" else None,
        "reasoning": "Starter size and its failed quality gate are already measured.",
    })])
    audit = audit_repository_comparison(
        "Minimize encoded bytes subject to reconstruction error <= 0.01.",
        {"primary_metric": {"name": "encoded_bytes", "direction": "minimize"}},
        {"metric": "encoded_bytes", "baseline": {"metric": 1200, "valid": True},
         "candidate": candidate}, llm, execution_phase=phase, baseline_result=frozen,
    )
    system = llm.calls[0][0]["content"]
    context = json.loads(llm.calls[0][1]["content"])
    assert "also apply to a frozen baseline during method audits" in system
    assert "candidate eligibility defects do not invalidate that starting control" in system
    assert "bound logs or receipts" in system
    assert "missing a duplicate normalized field alone" in system
    assert "Do not pass a claimed final Method improvement" in system
    assert "require the eligibility outcome to gate acceptance" in system
    assert context["frozen_baseline"] == frozen
    assert context["actual_result"]["candidate"] == candidate
    assert audit["verdict"] == verdict


def test_semantic_guardian_preserves_multi_quantity_composite_metrics():
    prompt = " ".join(_load_semantic_system_prompt().split())
    assert "single raw quantity" in prompt
    assert "preserves the declared objective's ordering" in prompt
    assert "Do not replace a composite metric with one constituent" in prompt
    assert "same declared formula and reference" in prompt


@pytest.mark.parametrize("phase", ["baseline", "method"])
def test_repository_audit_receives_actual_metric_identity_and_preserves_rejection(phase):
    semantics = {"primary_metric": {"name": "composite_score", "direction": "maximize"}}
    run = {"metric": "throughput", "metric_unit": "items/s", "lower_is_better": False,
           "baseline": {"metric": 20.0, "valid": True},
           "candidate": {"metric": 22.0, "valid": True}}
    frozen = {"metric_name": "throughput", "metric_unit": "items/s", "metric_value": 20.0}
    llm = FakeLLM([json.dumps({
        "verdict": "baseline_invalid", "issues": ["One constituent replaced the composite."],
        "remediation": "Measure the declared composite using its common reference.",
        "reasoning": "The frozen scalar does not preserve the task objective.",
    })])

    audit = audit_repository_comparison(
        "Optimize a composite of quality and resource use.", semantics, run, llm,
        execution_phase=phase, baseline_result=frozen)

    context = json.loads(llm.calls[0][1]["content"])
    for key in ("metric", "metric_unit", "lower_is_better"):
        assert context["actual_result"][key] == run[key]
    assert context["frozen_baseline"] == frozen
    assert context["evaluation_semantics"] == semantics
    assert "Do not replace a composite metric with one constituent" in llm.calls[0][0]["content"]
    assert audit["verdict"] == "baseline_invalid"
    assert run["baseline"]["valid"] is True  # Audit does not mutate the input record.


@pytest.mark.parametrize("phase", ["baseline", "method"])
def test_repository_audit_checks_async_completion_boundary(phase):
    run = {"config": {"timer_start": "worker_0", "timer_stop": "worker_1",
                      "join_all_workers": "after_stop", "completion_dependencies": []}}
    llm = FakeLLM([json.dumps({
        "verdict": "baseline_invalid" if phase == "baseline" else "comparison_changed",
        "issues": ["The stop event does not cover completion of every worker."],
        "remediation": "Establish a common start and join all required work before stopping.",
        "reasoning": "Joining after recording stop does not extend that timestamp.",
    })])
    result = audit_repository_comparison(
        "Measure completion latency of the full parallel batch.", {}, run, llm,
        execution_phase=phase)
    system = llm.calls[0][0]["content"]
    assert "completion latency or batch makespan" in system
    assert "all required asynchronous work" in system
    assert "after recording the stop timestamp" in system
    assert "submission latency or a single operation's latency" in system
    assert json.loads(llm.calls[0][1]["content"])["actual_result"]["config"] == run["config"]
    assert result["verdict"] == ("baseline_invalid" if phase == "baseline" else "comparison_changed")


def test_guardian_preserves_free_notes_without_imposing_a_fixed_checklist():
    semantic_prompt = _load_semantic_system_prompt().lower()

    assert "evaluation_notes" in semantic_prompt
    assert "do not require any fixed category" in semantic_prompt


def test_indirect_component_audit_preserves_incidental_metric_observation():
    run = {"baseline": {"metric": 0.2, "valid": True},
           "candidate": {"metric": 0.23, "valid": True}}
    llm = FakeLLM([json.dumps({
        "verdict": "pass", "issues": [], "remediation": None,
        "reasoning": "The component establishes a property; the score is only an observation.",
    })])
    audit_repository_comparison(
        "Improve the declared task metric.", {}, run, llm,
        execution_phase="method",
        component_scope={"component_id": "probe", "role": "indirect_gain"})
    system = llm.calls[0][0]["content"]
    assert "improves incidentally" in system
    assert "Never replace a measured candidate number with the baseline" in system
    assert "separately from causal attribution" in system
    assert "controlled rerun" in system
    assert run["candidate"]["metric"] == 0.23


def test_semantic_guardian_checks_derived_totals_without_merging_distinct_objects():
    prompt = " ".join(_load_semantic_system_prompt().split())
    assert "Check derived quantities against the explicit requirements" in prompt
    assert "preserves one while producing another" in prompt
    assert "rather than freeze a guessed total" in prompt


def test_repository_audit_keeps_original_task_above_conflicting_generated_total():
    original = "Preserve three source buffers and produce three separate output buffers."
    generated = {"conditions_to_hold_constant": ["three buffers total"]}
    run = {"baseline": {"metric": 4.0, "valid": True},
           "config": {"source_buffers": 3, "output_buffers": 3, "total_buffers": 6}}
    llm = FakeLLM([json.dumps({
        "verdict": "pass", "issues": [], "remediation": None,
        "reasoning": "Six buffers match the original task; the generated total is erroneous.",
    })])

    result = audit_repository_comparison(
        original, generated, run, llm, execution_phase="baseline")

    assert result["verdict"] == "pass"
    system = llm.calls[0][0]["content"]
    assert "original task outranks generated evaluation_semantics" in system
    assert "do not attribute the latter to the user" in system
    context = json.loads(llm.calls[0][1]["content"])
    assert context["original_task"] == original
    assert context["evaluation_semantics"] == generated
    assert context["actual_result"]["config"]["total_buffers"] == 6


def test_repository_audit_receives_confirmed_local_hardware_boundary():
    boundary = {
        "selected_source": "confirmed_boundary",
        "benchmark_kind": "user_defined",
        "provenance": [{"source": "confirmed_boundary", "conflicts_with_official": True}],
        "controls": {"resources": {"accelerator_type": "H800"}},
    }
    llm = FakeLLM([json.dumps({
        "verdict": "pass", "issues": [], "remediation": None,
        "reasoning": "The local H800 comparison uses the confirmed boundary.",
    })])
    audit_repository_comparison(
        "Use the repository evaluator.", None,
        {"benchmark_contract": boundary, "baseline": {"metric": 0.53, "valid": True}},
        llm, execution_phase="baseline")
    system = llm.calls[0][0]["content"]
    context = json.loads(llm.calls[0][1]["content"])
    assert "host-selected benchmark_contract" in system
    assert "same-boundary local comparison" in system
    assert context["actual_result"]["benchmark_contract"] == boundary


def test_repository_audit_scopes_native_baseline_rule_to_baseline():
    original = (
        "Measure the native repository baseline, not a self-written weak baseline. "
        "Optimize the method algorithm using the same inputs, correctness and timing."
    )
    run = {
        "baseline": {"metric": 4.0, "valid": True},
        "candidate": {"metric": 2.0, "valid": True},
        "config": {"baseline_source": "native_repository",
                   "method_algorithm": "new_candidate_algorithm"},
    }
    llm = FakeLLM([json.dumps({
        "verdict": "pass", "issues": [], "remediation": None,
        "reasoning": "The baseline remains native; the method replacement is permitted.",
    })])
    result = audit_repository_comparison(
        original, None, run, llm, execution_phase="method")

    assert result["verdict"] == "pass"
    system = llm.calls[0][0]["content"]
    assert "does not forbid a new method algorithm" in system
    assert "method-side restriction before rejecting that replacement" in system
    assert "still enforce shared inputs, correctness, hardware, budgets and timing boundaries" in system
    context = json.loads(llm.calls[0][1]["content"])
    assert context["original_task"] == original
    assert context["actual_result"]["config"] == run["config"]


@pytest.mark.parametrize("phase", ["baseline", "method"])
def test_repository_audit_requires_evidence_for_side_specific_exemptions(phase):
    original = "Use only the standard library. Measure the selected repository baseline."
    llm = FakeLLM([json.dumps({
        "verdict": "baseline_invalid", "issues": ["The baseline uses an external dependency."],
        "remediation": "Select a task-matching baseline.",
        "reasoning": "The original task grants no baseline exemption.",
    })])
    audit_repository_comparison(original, None, {
        "baseline": {"metric": 4.0, "valid": True},
        "config": {"baseline_dependencies": ["external_library"]},
    }, llm, execution_phase=phase)
    system = llm.calls[0][0]["content"]
    assert "quote the task evidence for any side-specific exemption" in system
    assert "Public repository provenance does not itself exempt" in system
    assert json.loads(llm.calls[0][1]["content"])["original_task"] == original


def test_repository_audit_distinguishes_eliminated_work_from_untimed_work():
    llm = FakeLLM([json.dumps({
        "verdict": "comparison_changed",
        "issues": ["A required per-invocation reduction is executed outside the timer."],
        "remediation": "Include that reduction and its transfers in the measured boundary.",
        "reasoning": "Fusion is allowed, but still-executed prerequisites are not free.",
    })])
    run = {"candidate": {"metric": 2.0, "valid": True},
           "config": {"executed": ["fused_kernel", "reduction"],
                      "timed": ["fused_kernel"]}}
    result = audit_repository_comparison(
        "Compare end-to-end latency with the required outputs available on the host.",
        None, run, llm, execution_phase="method")
    system = llm.calls[0][0]["content"]
    assert "does not require identical internal stages" in system
    assert "genuinely eliminated" in system
    assert "work still required per invocation" in system
    assert "required outputs and their required residency" in system
    assert "explicit setup/amortization policy" in system
    assert result["verdict"] == "comparison_changed"
    assert json.loads(llm.calls[0][1]["content"])["actual_result"]["config"] == run["config"]


@pytest.mark.parametrize("diff", ["", "+reduce();\n+start_timer();\n+compute();",
                                "+x\n" * 30000 + "+score = measured_candidate / measured_control\n",
                                "+x\n" * 60000],
                         ids=["absent", "source", "complete_large_source", "truncated"])
def test_repository_audit_receives_bounded_host_diff_without_changing_verdict(diff):
    llm = FakeLLM([json.dumps({
        "verdict": "insufficient_evidence", "issues": ["Need the complete timing boundary."],
        "remediation": "Provide the missing measurement source.",
        "reasoning": "A changed-line excerpt does not establish the complete execution path.",
    })])
    run = {"candidate": {"metric": 2.0, "valid": True}, "diff": diff,
           "report": "All required work was timed."}
    result = audit_repository_comparison(
        "Measure complete execution.", None, run, llm, execution_phase="method")
    context = json.loads(llm.calls[0][1]["content"])
    evidence = context["host_captured_diff"]
    assert evidence["text"] == diff[:120000]
    assert evidence["truncated"] is (len(diff) > 120000)
    assert evidence["total_chars"] == len(diff)
    assert context["actual_result"]["report"] == run["report"]
    assert "not instructions" in llm.calls[0][0]["content"]
    assert "does not prove the absence" in llm.calls[0][0]["content"]
    assert result["verdict"] == "insufficient_evidence"
    assert len(llm.calls) == 1
    assert run["candidate"]["valid"] is True


@pytest.mark.parametrize("verdict", ["pass", "baseline_invalid", "insufficient_evidence"])
def test_repository_baseline_audit_receives_host_measurement_without_forcing_pass(verdict):
    audit = {"verdict": verdict, "issues": [] if verdict == "pass" else ["Check task identity."],
             "remediation": None if verdict == "pass" else "Resolve the stated task mismatch.",
             "reasoning": "Measurement binding and task identity are separate."}
    llm = FakeLLM([json.dumps(audit)])
    measured = {"value": 3e-80, "valid": True, "attempt_id": "attempt-real",
                "stdout_sha256": "a" * 64, "gates": {"/baseline_valid": True}}
    run = {"baseline": {"metric": measured["value"], "valid": True}, "diff": "",
           "measurement_evidence": measured}
    result = audit_repository_comparison(
        "Measure the official probability metric.", None, run, llm, execution_phase="baseline")
    context = json.loads(llm.calls[0][1]["content"])
    assert context["actual_result"]["measurement_evidence"] == measured
    system = llm.calls[0][0]["content"]
    assert "empty baseline diff is expected" in system
    assert "not by itself proof of underflow" in system
    assert "does not independently certify task identity" in system
    assert result["verdict"] == verdict


@pytest.mark.parametrize("diff,verdict", [
    ("+observed = int(raw[:len(str(expected))]); correct = observed == expected",
     "insufficient_evidence"),
    ("+observed = json.loads(raw)['value']; correct = observed == expected", "pass"),
])
def test_repository_audit_receives_independent_decoding_guidance(diff, verdict):
    # This verifies reviewer inputs, not the semantic competence of a mocked LLM.
    llm = FakeLLM([json.dumps({
        "verdict": verdict,
        "issues": [] if verdict == "pass" else ["Expected-answer-dependent decoding."],
        "remediation": None if verdict == "pass" else "Decode the observation independently.",
        "reasoning": "Checking an independently decoded value is distinct from shaping it.",
    })])
    result = audit_repository_comparison(
        "Optimize runtime while preserving correct outputs.", None,
        {"candidate": {"metric": 2.0, "valid": True}, "diff": diff},
        llm, execution_phase="method")
    system = llm.calls[0][0]["content"]
    assert "decode observed outputs independently" in system
    assert "expected answers to choose token boundaries" in system
    assert "Comparing an independently decoded value" in system
    assert json.loads(llm.calls[0][1]["content"])["host_captured_diff"]["text"] == diff
    assert len(llm.calls) == 1
    assert result["verdict"] == verdict


def test_guardian_prompt_rejects_unauthorized_availability_fallback():
    prompt = " ".join(_load_semantic_system_prompt().lower().split())
    assert '"if unavailable" fallback' in prompt
    assert "not permission to replace" in prompt
    assert "unless the user's instructions authorize substitution" in prompt
    assert "unresolved availability note that preserves the choice is acceptable" in prompt


def test_guardian_accepts_fixed_time_step_difference_as_observed_outcome():
    llm = FakeLLM([json.dumps({
        "verdict": "pass", "issues": [], "remediation": None,
        "reasoning": "Both used 300 seconds; different steps are an observed throughput result.",
    })])
    result = audit_comparison_integrity(
        _record("COMPARISON_AUDIT: baseline_steps=900 method_steps=1100; both train_s=300"),
        _benchmark(), llm,
    )

    assert result["verdict"] == "pass"
    prompt = llm.calls[0][1]["content"]
    assert SEMANTICS["comparison_statement"] in prompt
    assert "different steps" in llm.calls[0][0]["content"]


def test_guardian_prompt_preserves_frozen_aggregation_quantifier():
    llm = FakeLLM([json.dumps({
        "verdict": "pass", "issues": [], "remediation": None,
        "reasoning": "The stated mean meets the frozen aggregate threshold.",
    })])
    benchmark = _benchmark()
    benchmark["evaluation_semantics"] = {
        **SEMANTICS,
        "comparison_statement": (
            "Compare mean training time across four runs when mean validation loss is at "
            "most 3.28; individual runs are observations."
        ),
    }

    result = audit_comparison_integrity(
        _record(
            "COMPARISON_AUDIT: val_losses=[3.2793,3.2807,3.2805,3.2780]; "
            "mean_val_loss=3.279625"
        ),
        benchmark,
        llm,
    )

    assert result["verdict"] == "pass"
    system_prompt = llm.calls[0][0]["content"]
    assert "quantifiers and aggregation rule exactly" in system_prompt
    assert "does not silently become a requirement" in system_prompt
    assert '"first threshold crossing"' in system_prompt
    assert "Require an early-stop" in system_prompt
    assert "does not change the comparison" in system_prompt


def test_guardian_receives_frozen_baseline_as_authoritative_comparison_side():
    llm = FakeLLM([json.dumps({
        "verdict": "pass", "issues": [], "remediation": None,
        "reasoning": "The method preserved the accepted baseline conditions.",
    })])
    baseline = {
        "reproduced": True,
        "metric_value": 0.94,
        "metric_name": "val_bpb",
        "fixed_config": {
            "gpu": "H800", "data": "accepted canonical repository data",
            "training_seconds": 300,
        },
    }

    result = audit_comparison_integrity(
        _record("COMPARISON_AUDIT: method_train_s=300; same data and evaluator"),
        _benchmark(), llm, baseline_result=baseline,
    )

    assert result["verdict"] == "pass"
    prompt = llm.calls[0][1]["content"]
    assert "Frozen reproduced BaselineResult" in prompt
    assert "accepted canonical repository data" in prompt
    assert "Do not re-litigate the accepted baseline" in prompt
    assert "frozen BaselineResult wins" in prompt


def test_guardian_reports_changed_training_budget_with_remediation():
    llm = FakeLLM([json.dumps({
        "verdict": "comparison_changed",
        "issues": ["method used 600 seconds while baseline used 300 seconds"],
        "remediation": "rerun the method with 300 seconds of pure training",
        "reasoning": "The claimed gain changes the fixed-time comparison.",
    })])

    result = audit_comparison_integrity(
        _record("COMPARISON_AUDIT: baseline train_s=300; method train_s=600"),
        _benchmark(), llm,
    )

    assert result["verdict"] == "comparison_changed"
    assert result["issues"] == ["method used 600 seconds while baseline used 300 seconds"]
    assert "300 seconds" in result["remediation"]


def test_guardian_requires_issues_and_remediation_for_nonpass_verdict():
    llm = FakeLLM([json.dumps({
        "verdict": "comparison_changed", "issues": [], "remediation": None,
        "reasoning": "changed",
    })] * 3)

    with pytest.raises(ComparisonGuardianParseError):
        audit_comparison_integrity(_record("missing"), _benchmark(), llm)


def test_invalid_audit_retry_restates_schema_and_preserves_responses():
    from types import SimpleNamespace
    from autoresearch.agents.comparison_guardian_agent.core import _run_audit

    responses = [json.dumps({"status": "pass", "reason": "wrong schema"})] * 3
    messages = [{"role": "system", "content": "Audit supplied evidence."}]

    class LLM:
        def chat(self, **kwargs):
            if len(messages) > 1:
                correction = messages[-1]["content"]
                for field in ["verdict", "issues", "remediation", "reasoning"]:
                    assert field in correction
                assert "baseline_invalid" in correction
            return SimpleNamespace(content=responses[0])

    with pytest.raises(ComparisonGuardianParseError) as exc:
        _run_audit(messages, LLM(), label="repository comparison", allow_baseline_invalid=True)

    assert exc.value.responses == responses


def test_semantic_guardian_audits_restatement_against_original_user_goal():
    llm = FakeLLM([json.dumps({
        "verdict": "comparison_changed",
        "issues": ["the restatement changed one GPU into a CPU-only experiment"],
        "remediation": "restore the explicit single-GPU 300-second comparison",
        "reasoning": "The original user goal is authoritative over a generic default budget.",
    })])

    result = audit_semantic_restatement(
        original_user_goal=(
            "On one GPU, give baseline and method 300 seconds and compare final val_bpb."
        ),
        research_spec={"budget": "fast CPU-only", "research_goal": "compare val_bpb"},
        spec_interpretation={"evaluation_semantics": {
            **SEMANTICS,
            "comparison_statement": "On one CPU core, compare a tiny model after 300 seconds.",
        }},
        llm_client=llm,
    )

    assert result["verdict"] == "comparison_changed"
    prompt = llm.calls[0][1]["content"]
    assert "Original user goal (highest authority)" in prompt
    assert "On one GPU" in prompt
    assert "fast CPU-only" in prompt


def test_semantic_guardian_accepts_a_faithful_operational_restatement():
    llm = FakeLLM([json.dumps({
        "verdict": "pass", "issues": [], "remediation": None,
        "reasoning": "The metric, direction, hardware, budget, and constants were preserved.",
    })])

    result = audit_semantic_restatement(
        original_user_goal=(
            "On one GPU, give baseline and method 300 seconds and compare final val_bpb."
        ),
        research_spec={"budget": "300 seconds on one GPU", "research_goal": "compare val_bpb"},
        spec_interpretation={"evaluation_semantics": SEMANTICS},
        llm_client=llm,
        literature_brief={
            "summary": "The repository paper defines the canonical evaluator.",
            "key_papers": [{"title": "Repository benchmark paper", "id_or_url": "paper:1"}],
        },
        field_trajectory_brief={
            "current_state": "Repository-defined speed comparisons are standard.",
            "open_problems": ["reproducible time-to-target measurement"],
        },
    )

    assert result["verdict"] == "pass"
    prompt = llm.calls[0][1]["content"]
    assert "Frozen LiteratureBrief" in prompt
    assert "Repository benchmark paper" in prompt
    assert "Frozen FieldTrajectoryBrief" in prompt
    assert "time-to-target measurement" in prompt


def test_semantic_guardian_defers_unverified_repository_protocol_details():
    llm = FakeLLM([json.dumps({
        "verdict": "insufficient_evidence",
        "issues": ["the concrete repository protocol has not been verified"],
        "remediation": "defer repository-dependent details until benchmark selection",
        "reasoning": "A discovery summary does not prove exact repository settings.",
    })])

    audit_semantic_restatement(
        original_user_goal="Use the benchmark's authoritative protocol.",
        research_spec={"research_goal": "improve the benchmark"},
        spec_interpretation={"evaluation_semantics": SEMANTICS},
        llm_client=llm,
    )

    prompt = llm.calls[0][0]["content"].lower()
    assert "before benchmark selection" in prompt
    assert "discovery evidence, not verified proof" in prompt
    assert "repository-dependent detail unresolved" in prompt


def test_semantic_guardian_preserves_single_run_success_definition():
    llm = FakeLLM([json.dumps({
        "verdict": "comparison_changed",
        "issues": ["the restatement added mandatory repeated trials"],
        "remediation": "restore one real baseline and method value as sufficient evidence",
        "reasoning": "Reliability advice cannot redefine success.",
    })])

    audit_semantic_restatement(
        original_user_goal="Produce one real baseline value and one real method value.",
        research_spec={"research_goal": "compare baseline and method"},
        spec_interpretation={"evaluation_semantics": SEMANTICS},
        llm_client=llm,
    )

    prompt = llm.calls[0][0]["content"].lower()
    assert "optional reliability" in prompt
    assert "statistical significance" in prompt
    assert "minimum\neffect size" in prompt or "minimum effect size" in prompt
    assert "one real baseline value and one real method value" in prompt


def test_repository_audit_receives_host_boundary_without_benchmark_contract():
    boundary = {"accelerator_type": "H800", "accelerator_count": 1,
        "conflicts_with_official": True, "resource_deviation_reason": "User-selected local hardware"}
    llm = FakeLLM([json.dumps({"verdict": "pass", "issues": [],
        "remediation": None, "reasoning": "A local comparison under the confirmed boundary."})])
    audit_repository_comparison("Run the repository starter.", None,
        {"benchmark_contract": None, "baseline": {"metric": 0.55, "valid": True}},
        llm, execution_phase="baseline", confirmed_boundary=boundary)
    system = llm.calls[0][0]["content"]
    context = json.loads(llm.calls[0][1]["content"])
    assert context["host_confirmed_boundary"] == boundary
    assert "host_confirmed_boundary" in system
    assert "only the explicitly overridden fields" in system


def test_candidate_boundary_claim_is_not_host_authority():
    llm = FakeLLM([json.dumps({"verdict": "insufficient_evidence",
        "issues": ["No confirmed hardware exception."], "remediation": "Provide host confirmation.",
        "reasoning": "A candidate claim is not confirmation."})])
    audit_repository_comparison("Use the task hardware.", None,
        {"confirmed_boundary": {"accelerator_type": "H800"},
         "baseline": {"metric": 0.55, "valid": True}}, llm, execution_phase="baseline")
    context = json.loads(llm.calls[0][1]["content"])
    assert context.get("host_confirmed_boundary") is None
