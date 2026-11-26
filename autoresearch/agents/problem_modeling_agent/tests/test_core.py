"""
Tests for ProblemModelingAgent's pure clarification core.
ProblemModelingAgent 纯澄清核心的测试。

No EventBus / Store / Redis / network — only a FakeLLM (canned JSON queue).
不涉及 EventBus / Store / Redis / 网络 —— 只用 FakeLLM（预设 JSON 响应队列）。
"""

from __future__ import annotations

import json

import pytest
import yaml
from jsonschema import Draft202012Validator

from autoresearch.agents.problem_modeling_agent.core import (
    MAX_RETRIES,
    ClarifyDecision,
    ClarifyParseError,
    _load_system_prompt,
    clarify_step,
)
from autoresearch.agents.problem_modeling_agent.tests.conftest import FakeLLM

SCHEMA_PATH = (
    __import__("pathlib").Path(__file__).resolve().parents[1]
    / "data-contracts"
    / "research_spec.schema.yaml"
)


def _spec_validator() -> Draft202012Validator:
    schema = yaml.safe_load(SCHEMA_PATH.read_text(encoding="utf-8"))
    return Draft202012Validator(schema)


VALID_SPEC = {
    "topic": "LLM coding agent evaluation",
    "research_goal": "Benchmark autonomous coding agents on real-world repos.",
    "task_mode": "benchmark",
    "budget": "two weeks",
    "deliverable_form": "benchmark report",
    "editable_scope": ["benchmark harness"],
    "forbidden_items": ["production deploy"],
    "open_questions": [],
}

EMPTY_DRAFT: dict = {
    "topic": None,
    "research_goal": None,
    "task_mode": None,
    "budget": None,
    "deliverable_form": None,
    "editable_scope": [],
    "forbidden_items": [],
    "open_questions": [],
}


def _ask_json(question_id: str = "q-001") -> str:
    return json.dumps(
        {
            "action": "ask",
            "question": {
                "question_id": question_id,
                "question_text": "What is the primary deliverable form you expect?",
                "options": ["paper draft", "code repo", "benchmark report"],
                "allow_free_text": True,
            },
        }
    )


def _finalize_json(spec: dict) -> str:
    return json.dumps({"action": "finalize", "spec": spec})


# ---------------------------------------------------------------------------
# (a) first call, no answers -> a well-formed question, done=False
# ---------------------------------------------------------------------------


def test_first_call_returns_question_when_no_answers():
    llm = FakeLLM([_ask_json()])

    decision = clarify_step(
        research_goal_text="I want to benchmark coding agents.",
        draft_spec=EMPTY_DRAFT,
        answered=[],
        llm_client=llm,
    )

    assert isinstance(decision, ClarifyDecision)
    assert decision.done is False
    assert decision.spec is None
    q = decision.question
    assert q is not None
    assert q["question_id"] == "q-001"
    assert isinstance(q["question_text"], str) and q["question_text"]
    assert q["allow_free_text"] is True
    assert q["options"] == ["paper draft", "code repo", "benchmark report"]
    assert len(llm.calls) == 1


# ---------------------------------------------------------------------------
# (b) after N answers, LLM finalizes with a schema-valid spec
# ---------------------------------------------------------------------------


def test_finalize_after_answers_returns_valid_spec():
    llm = FakeLLM([_finalize_json(VALID_SPEC)])
    answered = [
        {"question_id": "q-001", "answer_text": "benchmark report"},
        {"question_id": "q-002", "answer_text": "two weeks"},
    ]

    decision = clarify_step(
        research_goal_text="I want to benchmark coding agents.",
        draft_spec=EMPTY_DRAFT,
        answered=answered,
        llm_client=llm,
    )

    assert decision.done is True
    assert decision.question is None
    assert decision.spec == VALID_SPEC

    validator = _spec_validator()
    errors = list(validator.iter_errors(decision.spec))
    assert errors == [], [e.message for e in errors]


# ---------------------------------------------------------------------------
# (c) max-question guard forces finalize even if the LLM tries to ask again
# ---------------------------------------------------------------------------


def test_max_questions_guard_forces_finalize():
    llm = FakeLLM([_ask_json("q-999")])  # LLM misbehaves and tries to ask again
    draft_spec = {
        **EMPTY_DRAFT,
        "topic": VALID_SPEC["topic"],
        "research_goal": VALID_SPEC["research_goal"],
    }
    answered = [
        {"question_id": "q-001", "answer_text": "a1"},
        {"question_id": "q-002", "answer_text": "a2"},
    ]

    decision = clarify_step(
        research_goal_text="I want to benchmark coding agents.",
        draft_spec=draft_spec,
        answered=answered,
        llm_client=llm,
        max_questions=2,
    )

    assert decision.done is True
    assert decision.question is None
    assert decision.spec is not None
    assert decision.spec["topic"] == VALID_SPEC["topic"]
    assert decision.spec["research_goal"] == VALID_SPEC["research_goal"]

    validator = _spec_validator()
    errors = list(validator.iter_errors(decision.spec))
    assert errors == [], [e.message for e in errors]


# ---------------------------------------------------------------------------
# (d) malformed JSON -> bounded retry -> typed error, no crash
# ---------------------------------------------------------------------------


def test_malformed_json_retries_then_raises_typed_error():
    llm = FakeLLM(["not json at all"] * (MAX_RETRIES + 1))

    with pytest.raises(ClarifyParseError):
        clarify_step(
            research_goal_text="I want to benchmark coding agents.",
            draft_spec=EMPTY_DRAFT,
            answered=[],
            llm_client=llm,
        )

    assert len(llm.calls) == MAX_RETRIES + 1


# ---------------------------------------------------------------------------
# (e) finalize with an INVALID spec -> treated as parse failure, never done
# ---------------------------------------------------------------------------


def test_finalize_with_invalid_spec_retries_then_raises_typed_error():
    invalid_spec_extra_key = {**VALID_SPEC, "unexpected_field": "nope"}
    llm = FakeLLM([_finalize_json(invalid_spec_extra_key)] * (MAX_RETRIES + 1))

    with pytest.raises(ClarifyParseError):
        clarify_step(
            research_goal_text="I want to benchmark coding agents.",
            draft_spec=EMPTY_DRAFT,
            answered=[],
            llm_client=llm,
        )

    assert len(llm.calls) == MAX_RETRIES + 1


def test_finalize_with_missing_required_field_raises_typed_error():
    invalid_spec_missing_required = {k: v for k, v in VALID_SPEC.items() if k != "research_goal"}
    llm = FakeLLM([_finalize_json(invalid_spec_missing_required)] * (MAX_RETRIES + 1))

    with pytest.raises(ClarifyParseError):
        clarify_step(
            research_goal_text="I want to benchmark coding agents.",
            draft_spec=EMPTY_DRAFT,
            answered=[],
            llm_client=llm,
        )


# ---------------------------------------------------------------------------
# MOSS prompt_patch: no-patch unchanged / with-patch injected.
# ---------------------------------------------------------------------------


def test_no_prompt_patch_leaves_system_message_unchanged():
    llm = FakeLLM([_ask_json()])

    clarify_step(
        research_goal_text="I want to benchmark coding agents.",
        draft_spec=EMPTY_DRAFT,
        answered=[],
        llm_client=llm,
    )

    [messages] = llm.calls
    system_message = next(m["content"] for m in messages if m["role"] == "system")
    assert system_message == _load_system_prompt()


def test_prompt_patch_appended_to_system_message():
    llm = FakeLLM([_ask_json()])

    clarify_step(
        research_goal_text="I want to benchmark coding agents.",
        draft_spec=EMPTY_DRAFT,
        answered=[],
        llm_client=llm,
        prompt_patch="STEER: prefer X",
    )

    [messages] = llm.calls
    system_message = next(m["content"] for m in messages if m["role"] == "system")
    assert "STEER: prefer X" in system_message
    assert "[MOSS-injected guidance]" in system_message


def test_clarify_prompt_requires_options_on_every_question():
    """#7: every clarification question must offer options (pickable) AND free text."""
    from pathlib import Path
    from autoresearch.agents import problem_modeling_agent as pma
    md = (Path(pma.__file__).resolve().parent / "prompts" / "clarify_system.md").read_text(encoding="utf-8")
    low = md.lower()
    assert "required on every question" in low
    assert "2–4 concrete" in md or "2-4 concrete" in md
    assert "must be `true`" in low  # allow_free_text always true


def test_clarify_prompt_never_overwrites_an_explicit_user_compute_budget():
    """A GPU/time protocol in the user's goal must outrank generic fast defaults."""
    prompt = _load_system_prompt()

    assert "NEVER overwrite an explicit user-specified" in prompt
    assert "copy that constraint into `budget`" in prompt
    assert "ALWAYS set `budget`" not in prompt
    assert "CPU-only code experiment" not in prompt


def test_clarify_prompt_classifies_repository_baseline_comparison_as_benchmark():
    prompt = _load_system_prompt()

    assert "reproducing a runnable baseline or" in prompt
    assert "comparing a method with it under a fixed evaluator" in prompt
    assert "does not make them `experiment_validation`" in prompt
