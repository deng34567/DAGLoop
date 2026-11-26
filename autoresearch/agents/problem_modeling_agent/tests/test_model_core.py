"""
Tests for ProblemModelingAgent's pure ModelingProblem core.
ProblemModelingAgent 纯 ModelingProblem 核心的测试。

No EventBus / Store / Redis / network — only a FakeLLM (canned JSON queue).
不涉及 EventBus / Store / Redis / 网络 —— 只用 FakeLLM（预设 JSON 响应队列）。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from jsonschema import Draft202012Validator

from autoresearch.agents.problem_modeling_agent.model_core import (
    MAX_RETRIES,
    ModelLLMError,
    ModelParseError,
    _load_system_prompt,
    model_step,
)
from autoresearch.agents.problem_modeling_agent.tests.conftest import FakeLLM

_CONTRACTS_DIR = Path(__file__).resolve().parents[1] / "data-contracts"


def _validator(schema_name: str) -> Draft202012Validator:
    schema = yaml.safe_load((_CONTRACTS_DIR / schema_name).read_text(encoding="utf-8"))
    return Draft202012Validator(schema)


PROBLEM_MODEL_VALIDATOR = _validator("problem_model.schema.yaml")

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

LITERATURE_BRIEF = {
    "summary": "Several recent benchmarks evaluate agents on real repos.",
    "key_papers": [
        {"title": "SWE-bench", "id_or_url": "arxiv:2310.06770", "year": 2023},
    ],
}

NOVELTY_AUDIT_REPORT = {
    "verdict": "pass",
    "citation_sufficiency": "Literature adequately covers the space.",
    "stale_citations": [],
    "unsupported_claims": [],
    "missing_work": [],
    "novelty_status": "incremental",
    "notes": "Incremental but well-scoped.",
}


def _problem_model(**overrides) -> dict:
    base = {
        "key_problems": ["Agents struggle with long-horizon planning."],
        "opportunities": ["Combine retrieval with test-driven repair."],
        "risks": ["Benchmark leakage from pretraining corpora."],
        "verifiable_paths": ["Run held-out repo tasks with pass/fail tests."],
    }
    base.update(overrides)
    return base


def _response(**overrides) -> str:
    return json.dumps(_problem_model(**overrides))


# ---------------------------------------------------------------------------
# (a) valid path -> schema-valid ProblemModel
# ---------------------------------------------------------------------------


def test_valid_path_returns_schema_valid_problem_model():
    llm = FakeLLM([_response()])

    result = model_step(
        research_spec=RESEARCH_SPEC,
        literature_brief=LITERATURE_BRIEF,
        novelty_audit_report=NOVELTY_AUDIT_REPORT,
        llm_client=llm,
    )

    assert isinstance(result, dict)
    errors = list(PROBLEM_MODEL_VALIDATOR.iter_errors(result))
    assert errors == [], [e.message for e in errors]
    assert result["key_problems"] == ["Agents struggle with long-horizon planning."]
    assert len(llm.calls) == 1


# ---------------------------------------------------------------------------
# (b) malformed JSON -> bounded retry -> typed error
# ---------------------------------------------------------------------------


def test_malformed_json_retries_then_raises_typed_error():
    llm = FakeLLM(["not json at all"] * (MAX_RETRIES + 1))

    with pytest.raises(ModelParseError):
        model_step(
            research_spec=RESEARCH_SPEC,
            literature_brief=LITERATURE_BRIEF,
            novelty_audit_report=NOVELTY_AUDIT_REPORT,
            llm_client=llm,
        )

    assert len(llm.calls) == MAX_RETRIES + 1


def test_malformed_json_eventually_succeeds_within_retry_budget():
    llm = FakeLLM(["not json at all", _response()])

    result = model_step(
        research_spec=RESEARCH_SPEC,
        literature_brief=LITERATURE_BRIEF,
        novelty_audit_report=NOVELTY_AUDIT_REPORT,
        llm_client=llm,
    )

    assert result["verifiable_paths"] == ["Run held-out repo tasks with pass/fail tests."]
    assert len(llm.calls) == 2


# ---------------------------------------------------------------------------
# (c) missing required field -> parse failure/retry -> typed error
# ---------------------------------------------------------------------------


def test_missing_key_problems_retries_then_raises_typed_error():
    bad_model = {k: v for k, v in _problem_model().items() if k != "key_problems"}
    response = json.dumps(bad_model)
    llm = FakeLLM([response] * (MAX_RETRIES + 1))

    with pytest.raises(ModelParseError):
        model_step(
            research_spec=RESEARCH_SPEC,
            literature_brief=LITERATURE_BRIEF,
            novelty_audit_report=NOVELTY_AUDIT_REPORT,
            llm_client=llm,
        )

    assert len(llm.calls) == MAX_RETRIES + 1


def test_missing_verifiable_paths_retries_then_raises_typed_error():
    bad_model = {k: v for k, v in _problem_model().items() if k != "verifiable_paths"}
    response = json.dumps(bad_model)
    llm = FakeLLM([response] * (MAX_RETRIES + 1))

    with pytest.raises(ModelParseError):
        model_step(
            research_spec=RESEARCH_SPEC,
            literature_brief=LITERATURE_BRIEF,
            novelty_audit_report=NOVELTY_AUDIT_REPORT,
            llm_client=llm,
        )

    assert len(llm.calls) == MAX_RETRIES + 1


# ---------------------------------------------------------------------------
# LLM infra failure -> transient typed error, no internal retry
# ---------------------------------------------------------------------------


def test_llm_client_raising_wraps_in_typed_transient_error():
    class BoomLLM:
        def chat(self, *, messages, response_format=None, max_tokens=None):
            raise RuntimeError("upstream 500")

    with pytest.raises(ModelLLMError):
        model_step(
            research_spec=RESEARCH_SPEC,
            literature_brief=LITERATURE_BRIEF,
            novelty_audit_report=NOVELTY_AUDIT_REPORT,
            llm_client=BoomLLM(),
        )


# ---------------------------------------------------------------------------
# MOSS prompt_patch: no-patch unchanged / with-patch injected.
# ---------------------------------------------------------------------------


def test_no_prompt_patch_leaves_system_message_unchanged():
    llm = FakeLLM([_response()])

    model_step(
        research_spec=RESEARCH_SPEC,
        literature_brief=LITERATURE_BRIEF,
        novelty_audit_report=NOVELTY_AUDIT_REPORT,
        llm_client=llm,
    )

    [messages] = llm.calls
    system_message = next(m["content"] for m in messages if m["role"] == "system")
    assert system_message == _load_system_prompt()


def test_prompt_patch_appended_to_system_message():
    llm = FakeLLM([_response()])

    model_step(
        research_spec=RESEARCH_SPEC,
        literature_brief=LITERATURE_BRIEF,
        novelty_audit_report=NOVELTY_AUDIT_REPORT,
        llm_client=llm,
        prompt_patch="STEER: prefer X",
    )

    [messages] = llm.calls
    system_message = next(m["content"] for m in messages if m["role"] == "system")
    assert "STEER: prefer X" in system_message
    assert "[MOSS-injected guidance]" in system_message


def test_model_step_incorporates_injected_human_idea():
    # meeting 3.4: when the human injects a redirection at the checkpoint, the re-model must be
    # told to honor it (else the re-derived direction just reproduces the old one).
    llm = FakeLLM([_response()])
    model_step(
        research_spec=RESEARCH_SPEC, literature_brief=LITERATURE_BRIEF,
        novelty_audit_report=NOVELTY_AUDIT_REPORT, llm_client=llm,
        human_idea="Actually focus on a curriculum-learning angle, not the RL one.",
    )
    user_msg = next(m["content"] for m in llm.calls[0] if m["role"] == "user")
    assert "curriculum-learning angle" in user_msg
    assert "MUST honor" in user_msg          # framed as an override the model must follow


def test_model_prompt_does_not_invent_evaluation_protocol():
    prompt = _load_system_prompt()

    assert "Do not add new evaluation requirements" in prompt
    assert "seeds, repetitions, aggregation" in prompt
    assert "record it as unresolved" in prompt
