"""
Tests for ProblemModelingAgent's pure ReinterpretingSpec core (scope gate).
ProblemModelingAgent 纯 ReinterpretingSpec 核心（scope gate）的测试。

No EventBus / Store / Redis / network — only a FakeLLM (canned JSON queue).
不涉及 EventBus / Store / Redis / 网络 —— 只用 FakeLLM（预设 JSON 响应队列）。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from jsonschema import Draft202012Validator

from autoresearch.agents.problem_modeling_agent.reinterpret_core import (
    MAX_RETRIES,
    ReinterpretParseError,
    ReinterpretResult,
    _load_system_prompt,
    reinterpret_step,
)
from autoresearch.agents.problem_modeling_agent.tests.conftest import FakeLLM

_CONTRACTS_DIR = Path(__file__).resolve().parents[1] / "data-contracts"


def _validator(schema_name: str) -> Draft202012Validator:
    schema = yaml.safe_load((_CONTRACTS_DIR / schema_name).read_text(encoding="utf-8"))
    return Draft202012Validator(schema)


SPEC_INTERPRETATION_VALIDATOR = _validator("spec_interpretation.schema.yaml")
RESEARCH_SPEC_VALIDATOR = _validator("research_spec.schema.yaml")


def test_reinterpret_prompt_forbids_invented_scientific_failure_caps():
    prompt = _load_system_prompt().lower()
    assert "must not invent a numeric timeout, budget, or cap" in prompt
    assert "execution safety limit" in prompt
    assert "require repeated trials, statistical significance" in prompt
    assert "arbitrary minimum effect size" in prompt
    assert "optional recommendation" in prompt
    assert "time_to_target" in prompt
    assert "do not invent a fixed scientific duration" in prompt
    assert "watchdog" in prompt


def test_reinterpret_prompt_keeps_paired_derived_metrics_out_of_baseline_artifacts():
    prompt = " ".join(_load_system_prompt().lower().split())
    assert "derived paired metric" in prompt
    assert "the same directly measured quantity and unit" in prompt
    assert "only after both runs" in prompt


def test_reinterpret_keeps_constrained_objective_separate_from_feasibility():
    prompt = " ".join(_load_system_prompt().split())
    assert "optimization objective from feasibility constraints" in prompt
    assert "Do not promote a constraint measurement to primary_metric" in prompt
    assert "infeasible observation" in prompt


def test_reinterpret_prompt_defers_unverified_repository_protocol_details():
    prompt = _load_system_prompt().lower()
    assert "no repository has yet been selected, cloned, or inspected" in prompt
    assert "do not guess one from a" in prompt
    assert "benchmark name or a secondary summary" in prompt
    assert "benchmark selection will resolve it from the selected" in prompt
    assert "repository's own files" in prompt


def test_reinterpret_prompt_defines_optional_validation_study():
    prompt = _load_system_prompt().lower()
    assert "validation_study" in prompt
    assert "survey_validation" in prompt
    assert "literature search itself" in prompt
    assert "use null" in prompt


def test_reinterpret_prompt_preserves_explicit_choice_despite_availability_uncertainty():
    prompt = " ".join(_load_system_prompt().lower().split())
    assert "explicit user-selected research object or version" in prompt
    assert "adding a conditional fallback" in prompt
    assert "preserve the exact choice" in prompt
    assert "disclosure in a final report is not authorization" in prompt


def test_reinterpret_prompt_repairs_benchmark_mislabeled_as_validation():
    prompt = _load_system_prompt()

    assert "a runnable repository baseline" in prompt
    assert "comparison by the same fixed evaluator" in prompt
    assert "correct `task_mode` to `benchmark`" in prompt


def test_reinterpret_prompt_requests_free_task_specific_notes_without_a_fixed_taxonomy():
    prompt = " ".join(_load_system_prompt().lower().split())

    assert "evaluation_notes" in prompt
    assert "free, task-specific list" in prompt
    assert "do not work through a preset checklist" in prompt
    assert "use `[]`" in prompt

RESEARCH_SPEC = {
    "topic": "LLM coding agent evaluation",
    "research_goal": "Benchmark autonomous coding agents on real-world repos.",
    "task_mode": "benchmark",
    "budget": "two weeks",
    "deliverable_form": "benchmark report",
    "editable_scope": ["benchmark harness"],
    "forbidden_items": ["production deploy"],
    "open_questions": [],
}

FIELD_TRAJECTORY_BRIEF = {
    "origin": "Grew out of program synthesis + code LLMs.",
    "current_state": "Agentic coding benchmarks are proliferating.",
    "main_method_families": ["retrieval-augmented agents", "tool-use agents"],
    "open_problems": ["long-horizon planning", "flaky test handling"],
}

LITERATURE_BRIEF = {
    "summary": "Several recent benchmarks evaluate agents on real repos.",
    "key_papers": [
        {"title": "SWE-bench", "id_or_url": "arxiv:2310.06770", "year": 2023},
    ],
}


EVALUATION_SEMANTICS = {
    "comparison_statement": (
        "Under the same repository protocol, compare the canonical resolved score between the "
        "baseline and method while allowing method implementation changes."
    ),
    "research_question": "Does the method improve the canonical score under the stated protocol?",
    "primary_metric": {
        "name": "resolved",
        "direction": "maximize",
        "measurement": "Read the canonical resolved score from the benchmark evaluator.",
    },
    "comparison_mode": "repository_defined",
    "comparison_basis": "The retrieved paper defines the accepted comparison.",
    "budget_interpretation": "Use the repository-documented evaluation budget.",
    "conditions_to_hold_constant": ["dataset and evaluator"],
    "method_freedom": ["method implementation"],
    "data_source": "The benchmark's canonical held-out repository set.",
    "evaluator_source": "The benchmark's canonical evaluator.",
    "observed_outcomes": ["runtime", "attempt count"],
    "evaluation_notes": [
        "Resolve any repository-specific scoring precautions from authoritative files before execution."
    ],
    "outcome_criteria": {
        "success": "A valid method score is higher than the valid baseline score.",
        "failure": "Execution fails or the valid method score does not improve.",
        "inconclusive": "No valid comparable score is produced.",
    },
    "evidence": [{
        "source_type": "key_paper",
        "source_id": "arxiv:2310.06770",
        "supports": "The paper defines a canonical evaluator.",
    }],
    "uncertainties": [],
}


def test_evaluation_semantics_requires_complete_comparison_statement():
    semantics = dict(EVALUATION_SEMANTICS)
    semantics.pop("comparison_statement")
    obj = _spec_interpretation("in_scope")
    obj["evaluation_semantics"] = semantics

    errors = list(SPEC_INTERPRETATION_VALIDATOR.iter_errors(obj))

    assert any("comparison_statement" in error.message for error in errors)


@pytest.mark.parametrize("field", [
    "data_source", "evaluator_source", "observed_outcomes", "outcome_criteria",
])
def test_context_packet_requires_explicit_measurement_semantics(field):
    semantics = dict(EVALUATION_SEMANTICS)
    semantics.pop(field)
    obj = _spec_interpretation("in_scope")
    obj["evaluation_semantics"] = semantics

    errors = list(SPEC_INTERPRETATION_VALIDATOR.iter_errors(obj))

    assert any(field in error.message for error in errors)


def test_context_packet_requires_metric_measurement_procedure():
    semantics = dict(EVALUATION_SEMANTICS)
    semantics["primary_metric"] = {"name": "resolved", "direction": "maximize"}
    obj = _spec_interpretation("in_scope")
    obj["evaluation_semantics"] = semantics

    errors = list(SPEC_INTERPRETATION_VALIDATOR.iter_errors(obj))

    assert any("measurement" in error.message for error in errors)


def test_context_packet_accepts_runnable_survey_validation_study():
    obj = _spec_interpretation("in_scope")
    obj["validation_study"] = {
        "kind": "survey_validation",
        "question": "Does publication volume predict lower marginal evidence gain?",
        "sampling_frame": "A dated sample of public AI-paper metadata.",
        "data_sources": ["OpenAlex", "Crossref"],
        "variables": ["publication_year", "citation_count", "abstract_similarity"],
        "analysis": "Report descriptive statistics and a permutation test.",
        "falsification_condition": "No association or an association in the opposite direction.",
        "budget": "At most 500 records and 10 minutes of analysis.",
    }

    errors = list(SPEC_INTERPRETATION_VALIDATOR.iter_errors(obj))

    assert errors == [], [error.message for error in errors]


def test_validation_study_rejects_missing_sampling_frame():
    obj = _spec_interpretation("in_scope")
    obj["validation_study"] = {
        "kind": "survey_validation",
        "question": "Does publication volume predict lower marginal evidence gain?",
        "data_sources": ["OpenAlex"],
        "variables": ["publication_year"],
        "analysis": "Report a trend.",
        "falsification_condition": "No trend.",
        "budget": "100 records.",
    }

    errors = list(SPEC_INTERPRETATION_VALIDATOR.iter_errors(obj))

    messages = [error.message for error in errors]
    messages.extend(
        child.message for error in errors for child in error.context)
    assert any("sampling_frame" in message for message in messages)


def _spec_interpretation(disposition: str) -> dict:
    return {
        "naive_assumptions": ["Assumed a clean-room benchmark was novel."],
        "real_open_problem": "Reproducible evaluation of agentic coding at scale.",
        "topic_position": "Adjacent to several existing agentic benchmarks.",
        "disposition": disposition,
        "scope_reason": "There is a clear, automatable verifier (test pass/fail).",
        "evaluation_semantics": EVALUATION_SEMANTICS,
    }


def _response(disposition: str, revised_spec: dict | None = None) -> str:
    return json.dumps(
        {
            "spec_interpretation": _spec_interpretation(disposition),
            "revised_research_spec": revised_spec if revised_spec is not None else RESEARCH_SPEC,
        }
    )


# ---------------------------------------------------------------------------
# (a) valid path -> ReinterpretResult; both parts schema-valid; disposition readable
# ---------------------------------------------------------------------------


def test_valid_path_returns_schema_valid_result():
    llm = FakeLLM([_response("in_scope")])

    result = reinterpret_step(
        research_spec=RESEARCH_SPEC,
        field_trajectory_brief=FIELD_TRAJECTORY_BRIEF,
        literature_brief=LITERATURE_BRIEF,
        llm_client=llm,
    )

    assert isinstance(result, ReinterpretResult)
    assert result.disposition == "in_scope"
    si_errors = list(SPEC_INTERPRETATION_VALIDATOR.iter_errors(result.spec_interpretation))
    assert si_errors == [], [e.message for e in si_errors]
    rs_errors = list(RESEARCH_SPEC_VALIDATOR.iter_errors(result.revised_research_spec))
    assert rs_errors == [], [e.message for e in rs_errors]
    assert len(llm.calls) == 1


def test_original_user_task_is_supplied_verbatim_as_highest_authority():
    original = "Improve the repository task; preserve every explicit user constraint: α/β."
    llm = FakeLLM([_response("in_scope")])

    reinterpret_step(
        research_spec=RESEARCH_SPEC,
        field_trajectory_brief=FIELD_TRAJECTORY_BRIEF,
        literature_brief=LITERATURE_BRIEF,
        llm_client=llm,
        original_user_goal=original,
    )

    user_message = next(
        message["content"] for message in llm.calls[0] if message["role"] == "user")
    assert original in user_message
    assert user_message.index(original) < user_message.index("Frozen ResearchSpec")


def test_retry_keeps_bounded_base_context_instead_of_replaying_malformed_output():
    malformed = "MALFORMED_SENTINEL_" + ("x" * 2000)
    llm = FakeLLM([malformed, _response("in_scope")])

    result = reinterpret_step(
        research_spec=RESEARCH_SPEC,
        field_trajectory_brief=FIELD_TRAJECTORY_BRIEF,
        literature_brief=LITERATURE_BRIEF,
        llm_client=llm,
        original_user_goal="Keep the original task intact.",
    )

    assert result.disposition == "in_scope"
    assert all(
        malformed not in message["content"]
        for message in llm.calls[1]
        if isinstance(message.get("content"), str)
    )
    assert [message["role"] for message in llm.calls[1]] == ["system", "user", "user"]


def test_schema_retry_identifies_the_exact_misplaced_enum_field():
    invalid = _spec_interpretation("in_scope")
    invalid["evaluation_semantics"] = dict(EVALUATION_SEMANTICS)
    invalid["evaluation_semantics"]["comparison_mode"] = "open_ended_improvement"
    invalid["evaluation_semantics"]["primary_metric"] = dict(
        EVALUATION_SEMANTICS["primary_metric"])
    invalid["evaluation_semantics"]["primary_metric"]["direction"] = "repository_defined"
    bad_response = json.dumps({
        "spec_interpretation": invalid,
        "revised_research_spec": RESEARCH_SPEC,
    })
    llm = FakeLLM([bad_response, _response("in_scope")])

    result = reinterpret_step(
        research_spec=RESEARCH_SPEC,
        field_trajectory_brief=FIELD_TRAJECTORY_BRIEF,
        literature_brief=LITERATURE_BRIEF,
        llm_client=llm,
    )

    assert result.disposition == "in_scope"
    retry = llm.calls[1][-1]["content"]
    assert "$.evaluation_semantics.primary_metric.direction" in retry
    assert "repository_defined" in retry
    assert "primary_metric.direction" in retry
    assert "comparison_mode, not a direction" in retry


@pytest.mark.parametrize("invalid_direction", ["repository_defined", "improve"])
def test_repository_direction_stays_unresolved_until_live_baseline(invalid_direction):
    interpretation = _spec_interpretation("in_scope")
    interpretation["evaluation_semantics"] = dict(EVALUATION_SEMANTICS)
    interpretation["evaluation_semantics"]["primary_metric"] = dict(
        EVALUATION_SEMANTICS["primary_metric"])
    interpretation["evaluation_semantics"]["primary_metric"]["direction"] = invalid_direction
    llm = FakeLLM([json.dumps({
        "spec_interpretation": interpretation,
        "revised_research_spec": RESEARCH_SPEC,
    })])

    result = reinterpret_step(
        research_spec=RESEARCH_SPEC,
        field_trajectory_brief=FIELD_TRAJECTORY_BRIEF,
        literature_brief=LITERATURE_BRIEF,
        llm_client=llm,
    )

    semantics = result.spec_interpretation["evaluation_semantics"]
    assert semantics["primary_metric"]["direction"] == "unresolved"
    assert any("official repository evaluator" in item for item in semantics["uncertainties"])
    assert len(llm.calls) == 1
    from autoresearch.experiments.repo_improve.benchmark_contract import (
        reconcile_repository_evaluation_semantics,
    )
    resolved = reconcile_repository_evaluation_semantics(semantics, {
        "source_kind": "live_repo", "reproduced": True,
        "metric_name": "official_score", "lower_is_better": False,
    })
    assert resolved["primary_metric"]["direction"] == "maximize"


def test_unresolved_direction_is_not_valid_for_nonrepository_comparison():
    interpretation = _spec_interpretation("in_scope")
    interpretation["evaluation_semantics"] = dict(EVALUATION_SEMANTICS)
    interpretation["evaluation_semantics"]["comparison_mode"] = "open_ended_improvement"
    interpretation["evaluation_semantics"]["primary_metric"] = {
        **EVALUATION_SEMANTICS["primary_metric"], "direction": "unresolved"}
    errors = _validator("spec_interpretation.schema.yaml").iter_errors(interpretation)
    assert any("minimize" in error.message for error in errors)


def test_new_reinterpretation_requires_explicit_evaluation_notes():
    interpretation = _spec_interpretation("in_scope")
    interpretation["evaluation_semantics"] = dict(EVALUATION_SEMANTICS)
    interpretation["evaluation_semantics"].pop("evaluation_notes")
    response = json.dumps({
        "spec_interpretation": interpretation,
        "revised_research_spec": RESEARCH_SPEC,
    })

    with pytest.raises(ReinterpretParseError):
        reinterpret_step(
            research_spec=RESEARCH_SPEC,
            field_trajectory_brief=FIELD_TRAJECTORY_BRIEF,
            literature_brief=LITERATURE_BRIEF,
            llm_client=FakeLLM([response] * (MAX_RETRIES + 1)),
        )


# ---------------------------------------------------------------------------
# (b) each of the 3 dispositions round-trips
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("disposition", ["in_scope", "scope_changed", "out_of_scope"])
def test_each_disposition_round_trips(disposition):
    llm = FakeLLM([_response(disposition)])

    result = reinterpret_step(
        research_spec=RESEARCH_SPEC,
        field_trajectory_brief=FIELD_TRAJECTORY_BRIEF,
        literature_brief=LITERATURE_BRIEF,
        llm_client=llm,
    )

    assert result.disposition == disposition
    assert result.spec_interpretation["disposition"] == disposition


# ---------------------------------------------------------------------------
# (c) malformed JSON -> bounded retry -> typed error
# ---------------------------------------------------------------------------


def test_malformed_json_retries_then_raises_typed_error():
    llm = FakeLLM(["not json at all"] * (MAX_RETRIES + 1))

    with pytest.raises(ReinterpretParseError):
        reinterpret_step(
            research_spec=RESEARCH_SPEC,
            field_trajectory_brief=FIELD_TRAJECTORY_BRIEF,
            literature_brief=LITERATURE_BRIEF,
            llm_client=llm,
        )

    assert len(llm.calls) == MAX_RETRIES + 1


# ---------------------------------------------------------------------------
# (d) invalid disposition enum value -> parse failure, never returned
# ---------------------------------------------------------------------------


def test_invalid_disposition_enum_retries_then_raises_typed_error():
    llm = FakeLLM([_response("maybe_in_scope")] * (MAX_RETRIES + 1))

    with pytest.raises(ReinterpretParseError):
        reinterpret_step(
            research_spec=RESEARCH_SPEC,
            field_trajectory_brief=FIELD_TRAJECTORY_BRIEF,
            literature_brief=LITERATURE_BRIEF,
            llm_client=llm,
        )

    assert len(llm.calls) == MAX_RETRIES + 1


# ---------------------------------------------------------------------------
# (e) revised_research_spec fails research_spec.schema.yaml -> typed error
# ---------------------------------------------------------------------------


def test_invalid_revised_research_spec_retries_then_raises_typed_error():
    bad_spec = {k: v for k, v in RESEARCH_SPEC.items() if k != "research_goal"}
    llm = FakeLLM([_response("in_scope", revised_spec=bad_spec)] * (MAX_RETRIES + 1))

    with pytest.raises(ReinterpretParseError):
        reinterpret_step(
            research_spec=RESEARCH_SPEC,
            field_trajectory_brief=FIELD_TRAJECTORY_BRIEF,
            literature_brief=LITERATURE_BRIEF,
            llm_client=llm,
        )

    assert len(llm.calls) == MAX_RETRIES + 1


def test_invalid_spec_interpretation_missing_required_field_raises_typed_error():
    bad_interpretation = {
        k: v for k, v in _spec_interpretation("in_scope").items() if k != "scope_reason"
    }
    response = json.dumps(
        {
            "spec_interpretation": bad_interpretation,
            "revised_research_spec": RESEARCH_SPEC,
        }
    )
    llm = FakeLLM([response] * (MAX_RETRIES + 1))

    with pytest.raises(ReinterpretParseError):
        reinterpret_step(
            research_spec=RESEARCH_SPEC,
            field_trajectory_brief=FIELD_TRAJECTORY_BRIEF,
            literature_brief=LITERATURE_BRIEF,
            llm_client=llm,
        )

    assert len(llm.calls) == MAX_RETRIES + 1


def test_missing_evaluation_semantics_retries_then_raises_typed_error():
    bad_interpretation = {
        k: v for k, v in _spec_interpretation("in_scope").items()
        if k != "evaluation_semantics"
    }
    response = json.dumps({
        "spec_interpretation": bad_interpretation,
        "revised_research_spec": RESEARCH_SPEC,
    })
    llm = FakeLLM([response] * (MAX_RETRIES + 1))

    with pytest.raises(ReinterpretParseError):
        reinterpret_step(
            research_spec=RESEARCH_SPEC,
            field_trajectory_brief=FIELD_TRAJECTORY_BRIEF,
            literature_brief=LITERATURE_BRIEF,
            llm_client=llm,
        )

    assert len(llm.calls) == MAX_RETRIES + 1


@pytest.mark.parametrize("mode", [
    "quality_at_fixed_time",
    "quality_at_fixed_work",
    "time_to_target",
    "repository_defined",
    "open_ended_improvement",
])
def test_each_evaluation_comparison_mode_round_trips(mode):
    interpretation = _spec_interpretation("in_scope")
    interpretation["evaluation_semantics"] = {
        **EVALUATION_SEMANTICS, "comparison_mode": mode,
    }
    response = json.dumps({
        "spec_interpretation": interpretation,
        "revised_research_spec": RESEARCH_SPEC,
    })

    result = reinterpret_step(
        research_spec=RESEARCH_SPEC,
        field_trajectory_brief=FIELD_TRAJECTORY_BRIEF,
        literature_brief=LITERATURE_BRIEF,
        llm_client=FakeLLM([response]),
    )

    assert result.spec_interpretation["evaluation_semantics"]["comparison_mode"] == mode


def test_unresolved_metric_and_direction_are_retried_then_rejected():
    interpretation = _spec_interpretation("in_scope")
    interpretation["evaluation_semantics"] = {
        **EVALUATION_SEMANTICS,
        "primary_metric": {"name": None, "direction": None},
        "uncertainties": ["Resolve the canonical primary metric from the repository."],
        "evaluation_notes": ["Do not infer unresolved repository behavior from its name."],
    }
    response = json.dumps({
        "spec_interpretation": interpretation,
        "revised_research_spec": RESEARCH_SPEC,
    })

    with pytest.raises(ReinterpretParseError):
        reinterpret_step(
            research_spec=RESEARCH_SPEC,
            field_trajectory_brief=FIELD_TRAJECTORY_BRIEF,
            literature_brief=LITERATURE_BRIEF,
            llm_client=FakeLLM([response] * (MAX_RETRIES + 1)),
        )


def test_evaluation_evidence_must_reference_an_actual_input_source():
    interpretation = _spec_interpretation("in_scope")
    interpretation["evaluation_semantics"] = {
        **EVALUATION_SEMANTICS,
        "evidence": [{
            "source_type": "key_paper",
            "source_id": "invented-paper",
            "supports": "An unsupported claim.",
        }],
    }
    response = json.dumps({
        "spec_interpretation": interpretation,
        "revised_research_spec": RESEARCH_SPEC,
    })
    llm = FakeLLM([response] * (MAX_RETRIES + 1))

    with pytest.raises(ReinterpretParseError, match="valid reinterpretation"):
        reinterpret_step(
            research_spec=RESEARCH_SPEC,
            field_trajectory_brief=FIELD_TRAJECTORY_BRIEF,
            literature_brief=LITERATURE_BRIEF,
            llm_client=llm,
        )

    assert len(llm.calls) == MAX_RETRIES + 1
    assert "invented-paper" in llm.calls[-1][-1]["content"]


def test_malformed_json_eventually_succeeds_within_retry_budget():
    llm = FakeLLM(["not json at all", _response("scope_changed")])

    result = reinterpret_step(
        research_spec=RESEARCH_SPEC,
        field_trajectory_brief=FIELD_TRAJECTORY_BRIEF,
        literature_brief=LITERATURE_BRIEF,
        llm_client=llm,
    )

    assert result.disposition == "scope_changed"
    assert len(llm.calls) == 2


# ---------------------------------------------------------------------------
# MOSS prompt_patch: no-patch unchanged / with-patch injected.
# ---------------------------------------------------------------------------


def test_no_prompt_patch_leaves_system_message_unchanged():
    llm = FakeLLM([_response("in_scope")])

    reinterpret_step(
        research_spec=RESEARCH_SPEC,
        field_trajectory_brief=FIELD_TRAJECTORY_BRIEF,
        literature_brief=LITERATURE_BRIEF,
        llm_client=llm,
    )

    [messages] = llm.calls
    system_message = next(m["content"] for m in messages if m["role"] == "system")
    assert system_message == _load_system_prompt()


def test_prompt_patch_appended_to_system_message():
    llm = FakeLLM([_response("in_scope")])

    reinterpret_step(
        research_spec=RESEARCH_SPEC,
        field_trajectory_brief=FIELD_TRAJECTORY_BRIEF,
        literature_brief=LITERATURE_BRIEF,
        llm_client=llm,
        prompt_patch="STEER: prefer X",
    )

    [messages] = llm.calls
    system_message = next(m["content"] for m in messages if m["role"] == "system")
    assert "STEER: prefer X" in system_message
    assert "[MOSS-injected guidance]" in system_message
