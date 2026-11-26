"""
Tests for the MOSS decide core. / MOSS decide 核心的测试。

No EventBus / Store / Redis / network — only a FakeLLM (canned JSON queue)
plus a REAL MossActionGuard loaded from the SSOT action-matrix YAML.
``decide`` is PURE: the guard is INJECTED (never constructed inside core),
and the returned dict is validated against ``moss_decision.schema.yaml``.

The crown jewel here: MOSS NEVER hard-stalls. When the LLM can't produce a
legal, leak-free decision within the retry budget, ``decide`` DEGRADES to
``guard.fallback(reason)`` with ``degraded=True`` instead of raising — unless
``strict=True``, in which case it raises ``MossQualityExhausted``. Only an
infra-level LLM failure raises (``MossLLMError``), and that is never
internally retried.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from jsonschema import Draft202012Validator

from autoresearch.agents.moss_agent.core import (
    MAX_RETRIES,
    PATCHABLE_AGENTS,
    MossLLMError,
    MossQualityExhausted,
    _patch_content_leaks,
    decide,
)
from autoresearch.agents.moss_agent.tests.conftest import FakeLLM, RaisingLLM
from autoresearch.orchestrator.moss_matrix import MossActionGuard, load_moss_action_matrix

_CONTRACTS_DIR = Path(__file__).resolve().parents[1] / "data-contracts"


def _validator(name: str) -> Draft202012Validator:
    schema = yaml.safe_load((_CONTRACTS_DIR / name).read_text(encoding="utf-8"))
    return Draft202012Validator(schema)


MOSS_DECISION_VALIDATOR = _validator("moss_decision.schema.yaml")


@pytest.fixture
def guard() -> MossActionGuard:
    return MossActionGuard(load_moss_action_matrix())


@pytest.mark.parametrize("preparation_failure", [False, True])
def test_invalid_component_preparation_does_not_rebuild_core_method(guard, preparation_failure):
    diagnostic = {"scope": {"ref": None}, "record": {
        "underlying_reason_code": "component_plan_invalid" if preparation_failure else "ablation_failed",
        "underlying_reason": "dependency checkpoint conflict" if preparation_failure else "Core assumption refuted.",
    }}
    method = {"method_ref": "spec://method/current", "components": [{
        "component_id": "sampler", "mechanism": "Use the accepted normalization dependency.",
    }]}
    scope = {"method_ref": method["method_ref"], "diagnostic_ref": None}
    llm = FakeLLM([
        json.dumps({**scope, "action": "revise_method", "rationale": "Revise the Method."}),
        json.dumps({**scope, "action": "revise_component", "rationale": "Repair the dependency join.",
                    "plan_diagnosis": {"component_id": "sampler",
                                       "plan_quote": method["components"][0]["mechanism"]}}),
    ])
    decision = decide("component_ablation_failed", {}, guard, llm,
                      diagnostic=diagnostic, method_context=method)
    assert decision["action"] == ("revise_component" if preparation_failure else "revise_method")
    assert decision["degraded"] is False


def _json(obj: dict) -> str:
    return json.dumps(obj)


def _assert_schema_valid(decision: dict) -> None:
    errors = list(MOSS_DECISION_VALIDATOR.iter_errors(decision))
    assert errors == [], [e.message for e in errors]


# ---------------------------------------------------------------------------
# (a) LLM returns a legal action -> clean MossDecision, degraded=False,
# schema-valid.
# ---------------------------------------------------------------------------


def test_decide_legal_action_returns_clean_decision(guard):
    response = {"action": "rewrite_framing", "rationale": "direction has thrashed twice"}
    llm = FakeLLM([_json(response)])

    result = decide("direction_thrashing", {"A": 2}, guard, llm)

    assert result["reason"] == "direction_thrashing"
    assert result["action"] == "rewrite_framing"
    assert result["prompt_patch"] is None
    assert result["degraded"] is False
    _assert_schema_valid(result)
    assert len(llm.calls) == 1
    system = llm.calls[0][0]["content"]
    assert "Tool-call serialization belongs to the configured SDK" in system
    assert "do not invent syntax, require native function calls, or forbid XML universally" in system


def test_plan_review_without_components_cannot_request_component_revision(guard):
    llm = FakeLLM([_json({"action": "revise_component", "rationale": "Unknown contract"})]
                  * (MAX_RETRIES + 1))
    result = decide("experiment_design_thrashing", {}, guard, llm)
    assert result["degraded"] is True
    assert result["action"] != "revise_component"


@pytest.mark.parametrize("has_components", [False, True])
def test_plan_thrashing_preserves_frozen_component_method(guard, has_components):
    method = {"method_ref": "spec://method/current", "components": [{
        "component_id": "sampler", "mechanism": "Use the qualified normalization dependency.",
    }]} if has_components else None
    scope = {"method_ref": method["method_ref"]} if method else {}
    llm = FakeLLM([
        _json({**scope, "action": "back_to_hypothesis", "rationale": "The same configuration was repeated."}),
        _json({**scope, "action": "revise_component", "rationale": "Revise the exhausted local configuration.",
               "plan_diagnosis": {"component_id": "sampler",
                                  "plan_quote": "Use the qualified normalization dependency."}}),
    ])
    result = decide("experiment_design_thrashing", {}, guard, llm, method_context=method)
    assert result["action"] == ("revise_component" if has_components else "back_to_hypothesis")
    assert result["degraded"] is False


def test_method_failure_uses_existing_eight_views_in_single_moss_call(guard):
    llm = FakeLLM([_json({"action": "rethink_method", "rationale":
                          "Wrong: measured score did not improve under the frozen evaluator; "
                          "test a different bottleneck."})])
    result = decide("method_not_working", {}, guard, llm,
                    research_memory={"verified_findings": ["score did not improve"]})
    assert result["action"] == "rethink_method"
    assert len(llm.calls) == 1
    prompt = llm.calls[0][1]["content"]
    assert "eight diagnostic perspectives" in prompt
    assert "mechanism_understanding" in prompt
    assert "score did not improve" in prompt


def test_component_repair_does_not_inherit_method_failure_rubric(guard):
    llm = FakeLLM([_json({"action": "revise_component", "rationale": "Fix the dependency."})])
    decide("component_ablation_failed", {}, guard, llm)
    assert "eight diagnostic perspectives" not in llm.calls[0][1]["content"]


@pytest.mark.parametrize("mir,task_verdict", [
    ("insufficient_evidence", "compliant"),
    ("faithful", "insufficient_evidence"),
    ("faithful", "compliant"),
])
def test_unverified_review_cannot_discard_a_measured_method(guard, mir, task_verdict):
    scope = {"method_ref": "spec://method/example/current",
             "diagnostic_ref": "spec://diagnostic_report/example/current"}
    response = {**scope, "action": "discard_hypothesis", "rationale": "No useful gain.",
                "plan_diagnosis": {"component_id": "restore", "plan_quote": "Restore accepted code"}}
    repair = {**response, "action": "change_plan", "rationale": "Bind the missing restore check."}
    llm = FakeLLM([_json(response), _json(repair)])
    result = decide(
        "implementation_thrashing", {}, guard, llm,
        method_context={"method_ref": scope["method_ref"], "components": [
            {"component_id": "restore", "mechanism": "Restore accepted code"}]},
        diagnostic={"scope": {"ref": scope["diagnostic_ref"]}, "result": {
            "method_metric": 1.03,
            "component_execution": {"execution_succeeded": True,
                                    "status": "execution_failed"},
            "method_implementation_review": {"verdict": mir,
                "task_contract_review": {"verdict": task_verdict}}}},
    )
    unverified = "insufficient_evidence" in {mir, task_verdict}
    assert result["action"] == ("change_plan" if unverified else "discard_hypothesis")
    assert result["degraded"] is False
    assert len(llm.calls) == (2 if unverified else 1)
    _assert_schema_valid(result)


@pytest.mark.parametrize("reason", ["component_ablation_failed", "experiment_design_thrashing"])
@pytest.mark.parametrize("wrong_ref", [None, "method_ref", "diagnostic_ref"])
def test_scoped_decision_prompt_and_guard_require_the_same_fields(guard, wrong_ref, reason):
    method_ref = "spec://method/example/current"
    diagnostic_ref = "spec://run_record/example/current"
    response = {
        "action": "revise_component", "rationale": "repair the current component contract",
        "method_ref": method_ref, "diagnostic_ref": diagnostic_ref,
        "plan_diagnosis": {"component_id": "projection", "plan_quote": "Output two views"},
    }
    if wrong_ref:
        response[wrong_ref] = "spec://stale/example/old"
    llm = FakeLLM([_json(response)] * (MAX_RETRIES + 1))
    result = decide(
        reason, {}, guard, llm,
        method_context={"method_ref": method_ref, "components": [
            {"component_id": "projection", "mechanism": "Output two views"}]},
        diagnostic={"scope": {"ref": diagnostic_ref}},
    )
    assert result["degraded"] is (wrong_ref is not None)
    system = llm.calls[0][0]["content"]
    for field in ("method_ref", "diagnostic_ref", "plan_diagnosis"):
        assert field in system
    assert "exactly this shape" not in system
    closing = llm.calls[0][1]["content"].rsplit("Respond with a single JSON object:", 1)[1]
    example, _ = json.JSONDecoder().raw_decode(closing.lstrip())
    assert example["method_ref"] == method_ref
    assert example["diagnostic_ref"] == diagnostic_ref
    for field in ("method_ref", "diagnostic_ref", "plan_diagnosis"):
        assert field in closing


@pytest.mark.parametrize("reason", ["component_ablation_failed", "experiment_design_thrashing"])
@pytest.mark.parametrize("reported_id,quote,accepted", [
    ("projection-norm", "Output two views", True),
    ("projection_norm", "Output two views", True),
    ("another-component", "Output two views", False),
    ("projection-norm", "An obsolete check", False),
])
def test_moss_recognizes_canonical_component_id_without_waiving_quote(
        guard, reported_id, quote, accepted, reason):
    response = {
        "action": "revise_component", "rationale": "Repair the current check.",
        "method_ref": "method-current", "diagnostic_ref": "run-current",
        "plan_diagnosis": {"component_id": reported_id, "plan_quote": quote},
    }
    llm = FakeLLM([_json(response)] * (MAX_RETRIES + 1))
    result = decide(
        reason, {}, guard, llm,
        method_context={"method_ref": "method-current", "components": [
            {"component_id": "projection_norm", "mechanism": "Output two views"}]},
        diagnostic={"scope": {"ref": "run-current"}},
    )
    assert result["degraded"] is (not accepted)
    if accepted:
        assert len(llm.calls) == 1
        assert result["action"] == "revise_component"


@pytest.mark.parametrize("reported_id", ["projection", "old-component"])
def test_stale_contract_retry_supplies_current_quotes_and_keeps_scope(guard, reported_id):
    response = {
        "action": "revise_component", "rationale": "Repair the current check.",
        "method_ref": "method-current", "diagnostic_ref": "run-current",
        "plan_diagnosis": {"component_id": reported_id, "plan_quote": "Obsolete mechanism"},
    }
    corrected = {**response, "plan_diagnosis": {
        "component_id": "projection", "plan_quote": "Output two views"}}
    llm = FakeLLM([_json(response), _json(corrected)])
    result = decide(
        "experiment_design_thrashing", {}, guard, llm,
        method_context={"method_ref": "method-current", "components": [
            {"component_id": "projection", "mechanism": "Output two views",
             "implementation_check": "Inspect both outputs"}]},
        diagnostic={"scope": {"ref": "run-current"}},
    )
    feedback = llm.calls[1][-1]["content"]
    assert "Current contract fields: " in feedback
    contracts = json.loads(feedback.split("Current contract fields: ", 1)[1]
                           .split("\nRespond again", 1)[0])
    assert contracts == [{"component_id": "projection", "mechanism": "Output two views",
                          "implementation_check": "Inspect both outputs"}]
    assert result["degraded"] is False
    assert result["method_ref"] == "method-current"
    assert result["diagnostic_ref"] == "run-current"
    assert result["plan_diagnosis"] == corrected["plan_diagnosis"]
    assert len(llm.calls) == 2


@pytest.mark.parametrize("response,category", [
    ("not JSON: private response", "invalid_json"),
    ('["private response"]', "non_object_json"),
    ('{"action":"private response"}', "illegal_action"),
    ('{"action":"rewrite_framing","prompt_patch":"private response"}', "invalid_patch_shape"),
])
def test_degraded_decision_keeps_failure_category_without_raw_response(guard, response, category):
    llm = FakeLLM([response] * (MAX_RETRIES + 1))
    result = decide("direction_thrashing", {}, guard, llm)
    assert result["degraded"] is True
    assert category in result["rationale"]
    assert "private response" not in result["rationale"]
    assert result["action"] == guard.fallback("direction_thrashing")
    _assert_schema_valid(result)


# ---------------------------------------------------------------------------
# (b) LLM returns an ILLEGAL action every time -> after retries, fallback to
# guard.fallback(reason) (== "escalate_to_user" for direction_thrashing),
# degraded=True, NO raise (the never-hard-stall contract).
# ---------------------------------------------------------------------------


def test_component_failure_guidance_repairs_invalid_acceptance_not_measurement_meaning(guard):
    llm = FakeLLM([_json({"action": "revise_component", "rationale": "Repair the acceptance input."})])
    decide("component_ablation_failed", {}, guard, llm)
    system = llm.calls[0][0]["content"]
    assert "repair the ComponentPlan's acceptance input or predicate" in system
    assert "Do not reinterpret the measured quantity" in system
    assert "official task metric, evaluator, or correctness requirements" in system


def test_staged_direct_gain_guidance_checks_label_against_permitted_intervention(guard):
    # This checks prompt/context delivery and legal routing, not live LLM judgment.
    method = {"components": [
        {"component_id": "producer", "type": "direct_gain",
         "mechanism": "Produce an intermediate representation; integration belongs to consumer.",
         "ablation_test": "official metric improves over parent"},
        {"component_id": "consumer", "depends_on": ["producer"],
         "mechanism": "Integrate the representation into prediction."},
    ]}
    llm = FakeLLM([_json({"action": "change_plan", "rationale":
                        "Review the producer acceptance contract before another implementation."})])
    result = decide("implementation_thrashing", {}, guard, llm, method_context=method,
                    diagnostic={"error": "Intermediate probe passes; official metric is unchanged."})
    assert result["action"] == "change_plan"
    assert result["degraded"] is False
    system, user = (message["content"] for message in llm.calls[0][:2])
    assert "Do not assume a frozen direct_gain label is correct" in system
    assert "trace whether its permitted intervention can affect its acceptance metric" in system
    assert "cannot justify shrinking the patch and repeating the same gain test" in system
    assert "Do not silently relabel the component or accept a probe as final improvement" in system
    assert "integration belongs to consumer" in user
    assert "Intermediate probe passes" in user


def test_stale_attempt_receipt_does_not_block_component_plan_repair(guard):
    llm = FakeLLM([_json({"action": "change_plan", "rationale":
                         "Prior valid measurements show the component contract cannot pass."})])
    result = decide("implementation_thrashing", {}, guard, llm, diagnostic={"result": {
        "diagnosis": "attempt_accounting_violation",
        "component_execution": {"status": "execution_failed", "execution_succeeded": False,
                                "infrastructure_attempts": 2},
        "component_execution_started": False,
        "component_attempt_id": None,
        "component_failed_attempt_id": None,
    }})
    assert result["action"] == "change_plan"
    assert "Do not infer a current host launch failure" in llm.calls[0][1]["content"]


def test_baseline_diagnosis_separates_execution_from_quality_gate(guard):
    llm = FakeLLM([_json({"action": "fix_environment", "rationale": "Inspect execution evidence."})])
    decide("baseline_reproduction_thrashing", {}, guard, llm)
    system = llm.calls[0][0]["content"]
    assert "A null accepted metric or nonzero wrapper exit alone" in system
    assert "quality-gate rejection" in system
    assert "do not relabel an infeasible measurement as a qualified baseline" in system
    assert "fix_environment requires concrete setup or result-collection evidence" in system


def test_refuted_component_claim_is_revised_not_enforced_as_truth(guard):
    llm = FakeLLM([_json({"action": "revise_component",
                         "rationale": "Revise the disproven upstream claim."})])
    decide("component_ablation_failed", {}, guard, llm,
           diagnostic={"error": "An independently verified counterexample refutes the accepted upstream claim."})
    system = llm.calls[0][0]["content"]
    assert "Frozen component claims are hypotheses, not ground truth" in system
    assert "Do not require a downstream report to repeat a refuted conclusion" in system
    assert "revise the faulty upstream claim and affected acceptance checks" in system
    assert "Preserve the official task requirements and original measurements" in system
    assert "independently verified counterexample" in llm.calls[0][1]["content"]


def test_baseline_revalidation_is_a_legal_explicit_decision(guard):
    llm = FakeLLM([_json({"action": "revalidate_baseline", "prompt_patch": None,
                         "rationale": "The frozen runtime cannot execute the allocated device kernels."})])
    result = decide("implementation_thrashing", {}, guard, llm)
    assert result["action"] == "revalidate_baseline"
    assert result["degraded"] is False
    assert result["prompt_patch"] is None
    assert guard.fallback("implementation_thrashing") == "escalate_to_user"
    assert "Do not encode baseline reproduction as a method" in llm.calls[0][0]["content"]


def test_parent_receipt_failure_is_not_evidence_of_invalid_baseline(guard):
    llm = FakeLLM([_json({"action": "shrink_patch",
                         "rationale": "Repair the missing dependency-parent measurement."})])
    decide("implementation_thrashing", {}, guard, llm,
           diagnostic={"error": "parent/control has no registered successful attempt"})
    system = llm.calls[0][0]["content"]
    assert "Distinguish the frozen task baseline" in system
    assert "even when that component has no dependencies" in system
    assert "legal implementation-repair action" in system
    assert "not merely an absent control receipt" in system
    assert "parent/control has no registered successful attempt" in llm.calls[0][1]["content"]


def test_experiment_repair_does_not_require_predicting_a_win(guard):
    llm = FakeLLM([_json({"action": "back_to_hypothesis",
                         "rationale": "Preserve the frozen comparison; test the revised hypothesis."})])
    result = decide("experiment_design_thrashing", {}, guard, llm)
    assert result["degraded"] is False
    system = llm.calls[0][0]["content"]
    assert "Separate a success criterion from a forecast" in system
    assert "Do not require expected_outcome to predict a win" in system
    assert "does not authorize changing the frozen baseline" in system


def test_timeout_guidance_preserves_evaluation_population(guard):
    llm = FakeLLM([_json({"action": "shrink_patch", "rationale": "Resume incomplete evaluation."})])
    decide("implementation_thrashing", {}, guard, llm)
    system = llm.calls[0][0]["content"]
    assert "incremental output is recovery evidence, not permission" in system
    assert "sample IDs, denominator" in system
    assert "never silently drop slow or failed samples" in system
    assert "selected before" in system
    assert "same sample evaluated for both arms" in system


def test_slow_run_guidance_checks_execution_device_before_blame(guard):
    llm = FakeLLM([_json({"action": "shrink_patch", "rationale": "Probe the execution environment first."})])
    decide("implementation_thrashing", {}, guard, llm,
           diagnostic={"environment": "allocated accelerator", "stderr_excerpt": "no result"},
           research_memory={"negative_results": ["actual execution device: CPU"]})
    system = llm.calls[0][0]["content"]
    assert "allocated device with the actual execution device" in system
    assert "kernel/runtime errors" in system
    assert "Do not infer code overhead from elapsed time alone" in system
    assert "small device operation using the actual interpreter" in system
    assert "not permission to silently change devices" in system
    user = llm.calls[0][1]["content"]
    assert "actual execution device: CPU" in user
    assert "allocated accelerator" in user


def test_recovery_guidance_preserves_task_and_revalidates_changed_baseline(guard):
    llm = FakeLLM([_json({"action": "shrink_patch", "rationale": "Repair the runtime."})])
    decide("implementation_thrashing", {}, guard, llm)
    system = llm.calls[0][0]["content"]
    assert "failure does not establish that the method" in system
    assert "task-required training budgets, model" in system
    assert "shortened diagnostic run is only a probe" in system
    assert "request baseline revalidation through" in system
    assert "do not silently accept the old baseline or rewrite its value" in system


def test_decide_illegal_action_every_time_degrades_to_fallback_never_raises(guard):
    response = {"action": "do_whatever"}
    llm = FakeLLM([_json(response)] * (MAX_RETRIES + 1))

    result = decide("direction_thrashing", {"A": 2}, guard, llm)

    assert result["degraded"] is True
    assert result["action"] == guard.fallback("direction_thrashing")
    assert result["action"] == "escalate_to_user"
    assert result["prompt_patch"] is None
    _assert_schema_valid(result)
    assert len(llm.calls) == MAX_RETRIES + 1


# ---------------------------------------------------------------------------
# (c) legal action but a leaking prompt_patch -> retry; clean patch on retry
# is accepted; leaking every time -> fallback (degraded).
# ---------------------------------------------------------------------------


def test_decide_leaking_patch_then_clean_patch_on_retry_is_accepted(guard):
    leaking_response = {
        "action": "rewrite_framing",
        "prompt_patch": {
            "agent": "CoderAgent",
            "content": "这是第3轮触发 MOSS 的阈值，请注意 counter 快到了。",
        },
    }
    clean_response = {
        "action": "rewrite_framing",
        "prompt_patch": {"agent": "CoderAgent", "content": "prefer smaller, reversible changes"},
    }
    llm = FakeLLM([_json(leaking_response), _json(clean_response)])

    result = decide("direction_thrashing", {"A": 2}, guard, llm)

    assert result["degraded"] is False
    assert result["action"] == "rewrite_framing"
    assert result["prompt_patch"] == {
        "agent": "CoderAgent",
        "content": "prefer smaller, reversible changes",
    }
    _assert_schema_valid(result)
    assert len(llm.calls) == 2


def test_decide_leaking_patch_every_time_degrades_to_fallback(guard):
    leaking_response = {
        "action": "rewrite_framing",
        "prompt_patch": {
            "agent": "CoderAgent",
            "content": "这是第3轮触发 MOSS 的阈值，counter 快到了。",
        },
    }
    llm = FakeLLM([_json(leaking_response)] * (MAX_RETRIES + 1))

    result = decide("direction_thrashing", {"A": 2}, guard, llm)

    assert result["degraded"] is True
    assert result["action"] == guard.fallback("direction_thrashing")
    assert result["prompt_patch"] is None
    _assert_schema_valid(result)
    assert len(llm.calls) == MAX_RETRIES + 1


@pytest.mark.parametrize("guidance", [
    "Keep the official accuracy threshold unchanged; diagnose the candidate failure.",
    "保留任务准确率阈值，用真实评测定位失败原因，不降低验收门槛。",
    "Use the observed counterexample to repair the current component.",
    "Check the hardware performance counter before optimizing the kernel.",
])
def test_task_threshold_feedback_is_not_an_orchestration_leak(guard, guidance):
    assert not _patch_content_leaks(guidance)
    llm = FakeLLM([_json({"action": "rewrite_framing", "prompt_patch": {
        "agent": "CoderAgent", "content": guidance}})])
    decision = decide("direction_thrashing", {"A": 2}, guard, llm)
    assert decision["degraded"] is False
    assert decision["prompt_patch"]["content"] == guidance
    assert len(llm.calls) == 1


@pytest.mark.parametrize("guidance", [
    "The retry counter is 3.", "The session threshold was reached.",
    "Our orchestration failure threshold is 6.", "Change the revision limit.",
    "重试阈值已到，修改会话计数器。", "这次调度预算已经耗尽。",
    "MOSS intervened.", "这是第3轮。",
])
def test_orchestration_bookkeeping_is_still_rejected(guidance):
    assert _patch_content_leaks(guidance)


# ---------------------------------------------------------------------------
# (d) strict=True + illegal every time -> raises MossQualityExhausted.
# ---------------------------------------------------------------------------


def test_decide_strict_illegal_every_time_raises_quality_exhausted(guard):
    response = {"action": "do_whatever"}
    llm = FakeLLM([_json(response)] * (MAX_RETRIES + 1))

    with pytest.raises(MossQualityExhausted):
        decide("direction_thrashing", {"A": 2}, guard, llm, strict=True)

    assert len(llm.calls) == MAX_RETRIES + 1


# ---------------------------------------------------------------------------
# (e) llm_client.chat raises -> MossLLMError (transient, no internal retry).
# ---------------------------------------------------------------------------


def test_decide_llm_infra_failure_raises_moss_llm_error_no_retry(guard):
    llm = RaisingLLM(RuntimeError("connection reset"))

    with pytest.raises(MossLLMError):
        decide("direction_thrashing", {"A": 2}, guard, llm)

    assert len(llm.calls) == 1


# ---------------------------------------------------------------------------
# (f) legal decision WITH a clean prompt_patch -> returned with the patch,
# degraded=False, schema-valid.
# ---------------------------------------------------------------------------


def test_decide_legal_action_with_clean_prompt_patch(guard):
    response = {
        "action": "rewrite_framing",
        "prompt_patch": {"agent": "CoderAgent", "content": "prefer smaller changes"},
        "rationale": "keep changes reviewable",
    }
    llm = FakeLLM([_json(response)])

    result = decide("direction_thrashing", {"A": 2}, guard, llm)

    assert result["degraded"] is False
    assert result["prompt_patch"] == {"agent": "CoderAgent", "content": "prefer smaller changes"}
    _assert_schema_valid(result)
    assert len(llm.calls) == 1


# ---------------------------------------------------------------------------
# non-dict / malformed JSON also degrades rather than raising (never-hard-stall).
# ---------------------------------------------------------------------------


def test_decide_malformed_json_every_time_degrades_to_fallback(guard):
    llm = FakeLLM(["not json at all"] * (MAX_RETRIES + 1))

    result = decide("direction_thrashing", {"A": 2}, guard, llm)

    assert result["degraded"] is True
    assert result["action"] == guard.fallback("direction_thrashing")
    _assert_schema_valid(result)
    assert len(llm.calls) == MAX_RETRIES + 1


def test_decide_recovers_after_one_bad_response(guard):
    llm = FakeLLM(["not json at all", _json({"action": "rewrite_framing"})])

    result = decide("direction_thrashing", {"A": 2}, guard, llm)

    assert result["degraded"] is False
    assert result["action"] == "rewrite_framing"
    assert len(llm.calls) == 2


def test_decide_feeds_research_memory_into_the_prompt(guard):
    # 3.7: when a consolidated ResearchMemory is passed, its content appears in the user message
    # MOSS sees (diagnostic context) -- but it never affects the returned action's legality.
    response = {"action": "rewrite_framing", "rationale": "informed by prior findings"}
    llm = FakeLLM([_json(response)])
    memory = {"verified_findings": ["baseline val_bpb=1.017 reproduces"],
              "negative_findings": ["wider MLP did not help"]}

    result = decide("direction_thrashing", {"A": 2}, guard, llm, research_memory=memory)

    assert result["action"] == "rewrite_framing"  # still a legal decision
    user_msg = llm.calls[0][1]["content"]
    assert "ResearchMemory" in user_msg
    assert "wider MLP did not help" in user_msg  # the actual finding was handed to MOSS


def test_decide_surfaces_off_method_contradiction_before_long_memory(guard):
    llm = FakeLLM([_json({"action": "rewrite_framing"})])
    memory = {"negative_results": ["failed attempt " + "x" * 3000],
              "consolidated_insights": ["off-method metric=913 improved"],
              "reconciliation_candidates": [{"result_ref": "spec://result/off"}]}
    decide("direction_thrashing", {"A": 2}, guard, llm, research_memory=memory)
    user_msg = llm.calls[0][1]["content"]
    assert "off-method metric=913 improved" in user_msg
    assert user_msg.index("Earlier off-Method observations") < user_msg.index(
        "Consolidated ResearchMemory")
    assert "not accepted results" in user_msg


def test_decide_without_research_memory_omits_the_block(guard):
    # backward-compatible: no memory -> no ResearchMemory block (context is purely additive).
    llm = FakeLLM([_json({"action": "rewrite_framing"})])
    decide("direction_thrashing", {"A": 2}, guard, llm)
    assert "Consolidated ResearchMemory" not in llm.calls[0][1]["content"]


def test_decide_counter_snapshot_may_be_none(guard):
    llm = FakeLLM([_json({"action": "rewrite_framing"})])

    result = decide("direction_thrashing", None, guard, llm)

    assert result["degraded"] is False
    _assert_schema_valid(result)


# ---------------------------------------------------------------------------
# PATCHABLE_AGENTS whitelist: a prompt_patch targeting a non-patchable agent
# (e.g. "RunnerAgent" — not an LLM agent MOSS may patch) is rejected like any
# other quality failure -> retry -> degrade (never-hard-stall). A patch
# targeting a whitelisted agent with clean content is accepted.
# ---------------------------------------------------------------------------


def test_decide_prompt_patch_agent_not_whitelisted_degrades_to_fallback(guard):
    response = {
        "action": "rewrite_framing",
        "prompt_patch": {"agent": "RunnerAgent", "content": "prefer smaller changes"},
    }
    llm = FakeLLM([_json(response)] * (MAX_RETRIES + 1))

    result = decide("direction_thrashing", {"A": 2}, guard, llm)

    assert result["degraded"] is True
    assert result["action"] == guard.fallback("direction_thrashing")
    assert result["prompt_patch"] is None
    _assert_schema_valid(result)
    assert len(llm.calls) == MAX_RETRIES + 1


def test_decide_prompt_patch_agent_whitelisted_is_accepted(guard):
    assert "CoderAgent" in PATCHABLE_AGENTS
    response = {
        "action": "rewrite_framing",
        "prompt_patch": {"agent": "CoderAgent", "content": "prefer smaller, reversible changes"},
    }
    llm = FakeLLM([_json(response)])

    result = decide("direction_thrashing", {"A": 2}, guard, llm)

    assert result["degraded"] is False
    assert result["prompt_patch"] == {
        "agent": "CoderAgent",
        "content": "prefer smaller, reversible changes",
    }
    _assert_schema_valid(result)
    assert len(llm.calls) == 1


def test_patchable_agents_whitelist_contents():
    assert PATCHABLE_AGENTS == {
        "ProblemModelingAgent",
        "DeepResearchAgent",
        "NoveltyAuditAgent",
        "EvidenceAuditAgent",
        "HypothesisAgent",
        "CriticAgent",
        "EvaluatorAgent",
        "ExperimentPlannerAgent",
        "CoderAgent",
        "MemoryAgent",
        "ReportAgent",
        "RankingJudgeAgent",
        "DataAnalysisAgent",
    }


def test_decide_result_suspicious_blocks_rerun_loopers(guard):
    """E13 anti-livelock: the re-run/re-verify loopers (strict_rerun, add_holdout) are
    removed from result_suspicious's choices, so a LLM that keeps picking one degrades to
    the converging fallback (escalate_to_user) instead of churning the experiment."""
    llm = FakeLLM([_json({"action": "strict_rerun"})] * (MAX_RETRIES + 1))
    result = decide("result_suspicious", {"G": 2}, guard, llm)
    assert result["action"] == "escalate_to_user"   # guard.fallback(result_suspicious)
    assert result["degraded"] is True               # the looper was rejected every time
    _assert_schema_valid(result)


def test_decide_result_suspicious_still_allows_discard(guard):
    """The non-looping converging actions (discard_result / escalate_to_user) are
    untouched -- MOSS can still discard a suspicious result."""
    llm = FakeLLM([_json({"action": "discard_result"})])
    result = decide("result_suspicious", {"G": 2}, guard, llm)
    assert result["action"] == "discard_result"
    assert result["degraded"] is False
    _assert_schema_valid(result)


# --- #4 MOSS closes the loop: feed the responsible writer specific corrective guidance
#     (route back to the writer, NOT MOSS editing the artifact -- preserves isolation) ---

from autoresearch.agents.moss_agent.core import (  # noqa: E402
    _RESPONSIBLE_AGENT,
    _build_decide_user_message,
)


def test_responsible_agents_are_all_patchable():
    # every fix-target MOSS is told to feed must be an agent it can actually patch.
    assert set(_RESPONSIBLE_AGENT.values()) <= PATCHABLE_AGENTS


def test_build_message_names_responsible_writer_and_demands_a_patch():
    msg = _build_decide_user_message("result_suspicious", {"G": 2},
                                     ["discard_result", "escalate_to_user"], "CoderAgent")
    low = msg.lower()
    assert "CoderAgent" in msg                       # names the responsible writer
    assert "close the loop" in low and "fix" in low  # get in and fix, don't just report
    assert "must propose a prompt_patch" in low or "must" in low
    assert "never edit" in low or "you never edit" in low or "feed the writer" in low


def test_build_message_without_responsible_keeps_patch_optional():
    msg = _build_decide_user_message("cost_exceeded", None, ["report_now"], None)
    assert "optionally propose" in msg.lower()       # unchanged behaviour for non-fixable
    assert "close the loop" not in msg.lower()


def test_decide_result_suspicious_prompt_targets_the_coder(guard):
    # end-to-end through decide(): the user message MOSS sees for a suspicious result
    # names CoderAgent as the writer to feed corrective guidance to.
    llm = FakeLLM([_json({"action": "discard_result",
                          "prompt_patch": {"agent": "CoderAgent",
                                           "content": "return plain serialisable values"}})])
    result = decide("result_suspicious", {"G": 2}, guard, llm)
    user_msg = llm.calls[0][1]["content"]
    assert "CoderAgent" in user_msg
    assert result["prompt_patch"]["agent"] == "CoderAgent"
    _assert_schema_valid(result)



def test_recovery_advice_requires_state_restoration_probe_before_expensive_retry(guard):
    llm = FakeLLM([_json({"action": "shrink_patch", "rationale": "Check restore path first."})])
    decide("implementation_thrashing", {}, guard, llm)
    system = llm.calls[0][0]["content"]
    assert "unverified cause as a hypothesis" in system
    assert "actual installed version and active execution branch" in system
    assert "state restoration" in system
    assert "smallest probe" in system
    assert "nonzero update" in system
    assert "fallback evaluation of the unchanged baseline" in system


def test_baseline_setup_observation_does_not_replace_primary_metric(guard):
    llm = FakeLLM([_json({"action": "fix_environment", "rationale": "Run the actual reference solution."})])
    decide("baseline_reproduction_thrashing", {}, guard, llm)
    system = llm.calls[0][0]["content"]
    assert "setup success is not primary-metric reproduction" in system
    assert "do not promote a null primary metric" in system
    assert "reference solution under the declared task objective" in system


def test_baseline_repository_reselection_is_not_environment_repair(guard):
    llm = FakeLLM([_json({"action": "change_benchmark", "rationale": "Reselect a compatible repository."})])
    decide("baseline_reproduction_thrashing", {}, guard, llm)
    system = llm.calls[0][0]["content"]
    assert "fix_environment keeps the confirmed repository" in system
    assert "change_benchmark also covers selecting a different repository" in system
    assert "preserving the original task and evaluation semantics" in system
    assert "prompt_patch cannot update confirmed_repo_url" in system
