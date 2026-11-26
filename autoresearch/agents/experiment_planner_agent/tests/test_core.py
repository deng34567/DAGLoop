"""
Tests for ExperimentPlannerAgent's pure core. / ExperimentPlannerAgent 纯核心的测试。

No EventBus / Store / Redis / network — only a FakeLLM (canned JSON queue).
``plan_experiment`` is PURE: it takes input dicts and an llm_client, and
returns a schema-valid, DECLARATIVE ``ExperimentPlan`` dict.

``plan_experiment``'s parse-time guards (NOT trusting the prompt) are the
crown jewel here:
  1. editable_region scope: every ``changes[i]["file"]`` MUST be a member of
     ``verifier_spec["editable_region"]`` AND NOT in
     ``verifier_spec["forbidden_files"]``.
  2. no-code (declarative) guard: none of ``_FORBIDDEN_FIELDS`` (code, diff,
     patch, implementation, code_diff) may appear anywhere in the parsed
     object.
Both are enforced independently of whatever the LLM/prompt claims, mirroring
evaluator_agent's verifier-first guard pattern.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from jsonschema import Draft202012Validator

from autoresearch.agents.experiment_planner_agent.core import (
    MAX_RETRIES,
    PlannerParseError,
    _load_plan_experiment_system_prompt,
    plan_experiment,
)
from autoresearch.agents.experiment_planner_agent.tests.conftest import FakeLLM

_CONTRACTS_DIR = Path(__file__).resolve().parents[1] / "data-contracts"


def _validator(name: str) -> Draft202012Validator:
    schema = yaml.safe_load((_CONTRACTS_DIR / name).read_text(encoding="utf-8"))
    return Draft202012Validator(schema)


EXPERIMENT_PLAN_VALIDATOR = _validator("experiment_plan.schema.yaml")

RESEARCH_SPEC = {
    "topic": "Retrieval-augmented generation for long-tail scientific QA",
    "research_goal": "Propose a new RAG method for long-tail scientific question answering.",
}

BENCHMARK_SPEC = {
    "benchmark_id": "longtailqa-v1",
    "name": "LongTailQA",
    "metric": "exact_match",
    "run_command": "python run_benchmark.py --model candidate",
    "baseline": {
        "command": "python run_benchmark.py --model baseline",
        "reported_metric": 0.42,
    },
    "resource_requirements": "1x A100 GPU",
    "forbidden_files": ["benchmark/data/held_out.jsonl"],
}

VERIFIER_SPEC = {
    "evaluator_hash": "sha256:abc123",
    "metric": "exact_match",
    "data_split": {"in_loop": "dev", "held_out": "test"},
    "run_budget": "10 runs, 1h wall-clock",
    "seed_policy": "fixed seed=42 across all runs",
    "editable_region": ["src/model.py", "src/train.py"],
    "forbidden_files": ["benchmark/data/held_out.jsonl"],
    "result_parse_schema": {"type": "object"},
}

SELECTED_HYPOTHESIS = {
    "id": "hyp-1",
    "statement": "Adding a learned gating mechanism to the retriever improves long-tail recall.",
    "rationale": "Dense retrievers underperform on long-tail entities.",
    "kind": "training",
    "verifiable": True,
    "expected_effect": "exact_match improves on the long-tail subset.",
}


def _json(obj: dict) -> str:
    return json.dumps(obj)


def _valid_plan(**overrides: object) -> dict:
    plan = {
        "hypothesis_id": "hyp-1",
        "change_summary": "Add a learned gating layer before retrieval fusion.",
        "changes": [
            {"file": "src/model.py", "description": "Add a gating module class."},
            {"file": "src/train.py", "description": "Wire the gating module into the training loop."},
        ],
        "ablations": ["no_gating_baseline"],
        "expected_outcome": "Improved exact_match on the long-tail subset.",
        "resource_budget": "1x A100 GPU, 4h wall-clock",
        "success_criteria": "exact_match improves by >=2 points over baseline on the held-out split.",
    }
    plan.update(overrides)
    return plan


def _plan_experiment(llm):
    return plan_experiment(SELECTED_HYPOTHESIS, VERIFIER_SPEC, BENCHMARK_SPEC, RESEARCH_SPEC, llm)


# ---------------------------------------------------------------------------
# (a) valid -> schema-valid ExperimentPlan, all changes[].file subset of
# editable_region.
# ---------------------------------------------------------------------------


def test_plan_experiment_valid_returns_schema_valid_plan():
    llm = FakeLLM([_json(_valid_plan())])

    result = _plan_experiment(llm)

    errors = list(EXPERIMENT_PLAN_VALIDATOR.iter_errors(result))
    assert errors == [], [e.message for e in errors]
    assert result["hypothesis_id"] == "hyp-1"
    files = {c["file"] for c in result["changes"]}
    assert files <= set(VERIFIER_SPEC["editable_region"])
    assert len(llm.calls) == 1


def test_plan_experiment_recovers_after_one_bad_response():
    llm = FakeLLM(["not json at all", _json(_valid_plan())])

    result = _plan_experiment(llm)

    assert result["hypothesis_id"] == "hyp-1"
    assert len(llm.calls) == 2


def test_plan_experiment_empty_editable_region_is_explicitly_permissive():
    llm = FakeLLM([_json(_valid_plan(changes=[
        {"file": "train.py", "description": "Implement the selected training method."},
    ]))])

    result = plan_experiment(
        SELECTED_HYPOTHESIS, {}, BENCHMARK_SPEC, RESEARCH_SPEC, llm)

    assert result["changes"][0]["file"] == "train.py"
    [messages] = llm.calls
    prompt = "\n".join(str(message["content"]) for message in messages)
    assert "empty editable_region means no predeclared file boundary" in prompt
    assert "whole repository is provisionally editable" in prompt


def test_plan_experiment_prompt_includes_frozen_baseline_runtime_evidence():
    llm = FakeLLM([_json(_valid_plan())])
    baseline_result = {
        "metric_name": "pure_train_time_seconds",
        "metric_value": 196.682,
        "fixed_config": "Muon + AdamW hybrid; bf16; manual all-reduce",
    }

    plan_experiment(
        SELECTED_HYPOTHESIS,
        VERIFIER_SPEC,
        BENCHMARK_SPEC,
        RESEARCH_SPEC,
        llm,
        baseline_result=baseline_result,
    )

    [messages] = llm.calls
    prompt = "\n".join(str(message["content"]) for message in messages)
    assert "Frozen BaselineResult" in prompt
    assert "196.682" in prompt
    assert "Muon + AdamW hybrid" in prompt


def test_plan_experiment_replan_prompt_includes_previous_review_at_end():
    llm = FakeLLM([_json(_valid_plan())])
    review = {"verdict": "fail", "issues": ["Do not change evaluator cadence."]}

    plan_experiment(
        SELECTED_HYPOTHESIS,
        VERIFIER_SPEC,
        BENCHMARK_SPEC,
        RESEARCH_SPEC,
        llm,
        plan_review=review,
    )

    [messages] = llm.calls
    prompt = messages[-1]["content"]
    assert "Previous ExperimentPlan review" in prompt
    assert "Do not change evaluator cadence." in prompt
    assert prompt.rstrip().endswith(
        "do not repeat a rejected component or comparison shortcut unchanged."
    )


# ---------------------------------------------------------------------------
# (b) a change file NOT in editable_region -> typed error (scope guard).
# ---------------------------------------------------------------------------


def test_plan_experiment_change_file_outside_editable_region_raises_typed_error():
    bad_plan = _valid_plan(
        changes=[
            {"file": "src/model.py", "description": "Add a gating module class."},
            {"file": "src/unrelated_module.py", "description": "Sneak in an out-of-scope change."},
        ]
    )
    llm = FakeLLM([_json(bad_plan)] * (MAX_RETRIES + 1))

    with pytest.raises(PlannerParseError):
        _plan_experiment(llm)

    assert len(llm.calls) == MAX_RETRIES + 1


# ---------------------------------------------------------------------------
# (c) a change file that IS in forbidden_files -> typed error.
# ---------------------------------------------------------------------------


def test_plan_experiment_change_file_in_forbidden_files_raises_typed_error():
    # Deliberately overlapping editable_region/forbidden_files: plan_experiment
    # must not trust that verifier_spec is internally disjoint -- it must
    # still reject a changes[].file that is in forbidden_files even if it is
    # also (nominally) in editable_region.
    verifier_spec = {
        **VERIFIER_SPEC,
        "editable_region": ["src/model.py", "src/train.py", "src/secret.py"],
        "forbidden_files": ["benchmark/data/held_out.jsonl", "src/secret.py"],
    }
    bad_plan = _valid_plan(
        changes=[
            {"file": "src/model.py", "description": "Add a gating module class."},
            {"file": "src/secret.py", "description": "Touch a forbidden file."},
        ]
    )
    llm = FakeLLM([_json(bad_plan)] * (MAX_RETRIES + 1))

    with pytest.raises(PlannerParseError):
        plan_experiment(SELECTED_HYPOTHESIS, verifier_spec, BENCHMARK_SPEC, RESEARCH_SPEC, llm)

    assert len(llm.calls) == MAX_RETRIES + 1


# ---------------------------------------------------------------------------
# (d) a forbidden code field -> typed error (no-code guard).
# ---------------------------------------------------------------------------


def test_plan_experiment_top_level_code_field_raises_typed_error():
    bad_plan = _valid_plan(code="def gate(x): return x * 2")
    llm = FakeLLM([_json(bad_plan)] * (MAX_RETRIES + 1))

    with pytest.raises(PlannerParseError):
        _plan_experiment(llm)

    assert len(llm.calls) == MAX_RETRIES + 1


def test_plan_experiment_change_code_diff_field_raises_typed_error():
    bad_plan = _valid_plan(
        changes=[
            {
                "file": "src/model.py",
                "description": "Add a gating module class.",
                "code_diff": "+ class Gate(nn.Module): ...",
            },
        ]
    )
    llm = FakeLLM([_json(bad_plan)] * (MAX_RETRIES + 1))

    with pytest.raises(PlannerParseError):
        _plan_experiment(llm)

    assert len(llm.calls) == MAX_RETRIES + 1


# ---------------------------------------------------------------------------
# (e) malformed JSON -> retry -> typed error.
# ---------------------------------------------------------------------------


def test_plan_experiment_malformed_json_retries_then_raises_typed_error():
    llm = FakeLLM(["not json at all"] * (MAX_RETRIES + 1))

    with pytest.raises(PlannerParseError):
        _plan_experiment(llm)

    assert len(llm.calls) == MAX_RETRIES + 1


def test_plan_experiment_schema_invalid_plan_retries_then_raises_typed_error():
    bad_plan = _valid_plan()
    del bad_plan["change_summary"]
    llm = FakeLLM([_json(bad_plan)] * (MAX_RETRIES + 1))

    with pytest.raises(PlannerParseError):
        _plan_experiment(llm)

    assert len(llm.calls) == MAX_RETRIES + 1


def test_component_plan_rejects_stale_target_named_before_current_dependency():
    from autoresearch.agents.experiment_planner_agent.core import _component_target_error

    selected = {
        "kind": "confirmed_method_component",
        "current_component": {"component_id": "mlp-train-144", "name": "MLP training"},
    }
    stale = {
        "change_summary": (
            "Implement the complete-candidate-eval component: compose the accepted "
            "feat-extractor-36 and mlp-train-144 components, then run the final evaluator."
        )
    }
    assert _component_target_error(selected, stale) is not None
    stale["change_summary"] = stale["change_summary"].replace("component:", "component using")
    assert _component_target_error(selected, stale) is not None
    assert _component_target_error(selected, {
        "change_summary": (
            "Improve the mlp-train-144 component: use feat-extractor-36 output."
        )
    }) is None


# ---------------------------------------------------------------------------
# MOSS prompt_patch: no-patch unchanged / with-patch injected.
# ---------------------------------------------------------------------------


def test_plan_experiment_no_prompt_patch_leaves_system_message_unchanged():
    llm = FakeLLM([_json(_valid_plan())])

    _plan_experiment(llm)

    [messages] = llm.calls
    system_message = next(m["content"] for m in messages if m["role"] == "system")
    assert system_message == _load_plan_experiment_system_prompt()


def test_plan_experiment_prompt_patch_appended_to_system_message():
    llm = FakeLLM([_json(_valid_plan())])

    plan_experiment(
        SELECTED_HYPOTHESIS, VERIFIER_SPEC, BENCHMARK_SPEC, RESEARCH_SPEC, llm,
        prompt_patch="STEER: prefer X",
    )

    [messages] = llm.calls
    system_message = next(m["content"] for m in messages if m["role"] == "system")
    assert "STEER: prefer X" in system_message
    assert "[MOSS-injected guidance]" in system_message


def test_bound_method_revalidation_uses_a_fixed_plan_despite_old_exploration_guidance():
    class NoPlanningCall:
        def chat(self, **kwargs):
            pytest.fail("Revalidating an already selected source must not invent another treatment")
    selected = {"id": "method://retained", "kind": "confirmed_method_revalidation",
                "reconciled_result_ref": "result://retained"}
    plan = plan_experiment(selected, {}, {}, {}, NoPlanningCall(),
        prompt_patch="Always introduce a new component delta before evaluation.",
        plan_review={"issues": ["The component is already implemented; pick a new delta."]})
    assert plan["hypothesis_id"] == selected["id"]
    assert plan["changes"] == []
    assert selected["reconciled_result_ref"] in plan["change_summary"]
    assert "official evaluator" in plan["success_criteria"]
    assert "Method" in plan["success_criteria"]


@pytest.mark.parametrize("bound_feedback", [True, False])
def test_bound_revalidation_replans_required_repairs_or_a_rejected_plan(bound_feedback):
    selected = {
        "id": "method://retained", "kind": "confirmed_method_revalidation",
        "reconciled_result_ref": "result://retained",
        "changes_outline": ["Repair the existing input normalization in src/model.py."],
    }
    feedback = {"decision": {"route": "revise_method", "modify": selected["changes_outline"]}}
    review = {"verdict": "fail", "issues": [
        "The retained source still needs the frozen input normalization repair."]}
    if bound_feedback:
        selected["applied_method_feedback"] = feedback
    expected = _valid_plan(
        hypothesis_id=selected["id"],
        change_summary="Repair only the frozen input normalization, then revalidate.",
        changes=[{"file": "src/model.py", "description": selected["changes_outline"][0]}],
    )
    llm = FakeLLM([_json(expected)])

    plan = plan_experiment(
        selected, VERIFIER_SPEC, BENCHMARK_SPEC, RESEARCH_SPEC, llm,
        plan_review=None if bound_feedback else review,
    )

    assert plan == expected
    [messages] = llm.calls
    user_message = next(m["content"] for m in messages if m["role"] == "user")
    if bound_feedback:
        assert "applied_method_feedback" in user_message
        assert selected["changes_outline"][0] in user_message
    else:
        assert review["issues"][0] in user_message
