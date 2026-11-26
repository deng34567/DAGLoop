"""
Tests for HypothesisAgent's pure generate core. / HypothesisAgent 纯生成核心的测试。

No EventBus / Store / Redis / network / search backend — only a FakeLLM
(canned JSON queue). ``generate_step`` is PURE: it takes ResearchSpec,
LiteratureBrief, and an llm_client, and returns a list of schema-valid
Hypothesis dicts (validated against the whole HypothesisBank shape).
不涉及 EventBus / Store / Redis / 网络 / 检索后端 —— 只用 FakeLLM（预设 JSON
响应队列）。``generate_step`` 是纯函数：接收 ResearchSpec、LiteratureBrief 和
llm_client，返回 schema 合法的 Hypothesis dict 列表（对照整个 HypothesisBank
形状校验）。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from jsonschema import Draft202012Validator

from autoresearch.agents.hypothesis_agent.core import (
    MAX_RETRIES,
    HypothesisLLMError,
    HypothesisParseError,
    _load_system_prompt,
    generate_step,
)
from autoresearch.agents.hypothesis_agent.tests.conftest import FakeLLM

_CONTRACTS_DIR = Path(__file__).resolve().parents[1] / "data-contracts"


def _validator(name: str) -> Draft202012Validator:
    schema = yaml.safe_load((_CONTRACTS_DIR / name).read_text(encoding="utf-8"))
    return Draft202012Validator(schema)


BANK_VALIDATOR = _validator("hypothesis_bank.schema.yaml")

RESEARCH_SPEC = {
    "topic": "Retrieval-augmented generation for long-tail scientific QA",
    "research_goal": "Propose a new RAG method for long-tail scientific question answering.",
}

LITERATURE_BRIEF = {
    "summary": "The field combines dense retrieval with generative models to ground answers.",
    "related_work": [],
    "key_papers": [
        {
            "title": "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
            "id_or_url": "arxiv:2005.11401",
            "year": 2020,
        },
    ],
    "existing_benchmarks": [],
}


def _bank_json(hypotheses: list[dict]) -> str:
    return json.dumps({"hypotheses": hypotheses})


def _hypothesis(hid: str, kind: str = "idea") -> dict:
    return {
        "id": hid,
        "statement": f"Statement for {hid}: doing X improves Y by Z%.",
        "rationale": f"Rationale for {hid}, grounded in the literature brief.",
        "kind": kind,
        "verifiable": True,
        "expected_effect": "Metric improves by an estimated 5-10%.",
    }


VALID_HYPOTHESES = [
    _hypothesis("h1", "idea"),
    _hypothesis("h2", "training"),
    _hypothesis("h3", "kernel"),
    _hypothesis("h4", "code_optimization"),
]


# ---------------------------------------------------------------------------
# (a) valid path -> list of schema-valid Hypotheses, len>=1, distinct ids,
#     each kind a valid enum.
# ---------------------------------------------------------------------------


def test_valid_path_returns_schema_valid_hypotheses():
    llm = FakeLLM([_bank_json(VALID_HYPOTHESES)])

    hypotheses = generate_step(RESEARCH_SPEC, LITERATURE_BRIEF, llm, n=4)

    assert isinstance(hypotheses, list)
    assert len(hypotheses) >= 1

    errors = list(BANK_VALIDATOR.iter_errors({"hypotheses": hypotheses}))
    assert errors == [], [e.message for e in errors]

    ids = [h["id"] for h in hypotheses]
    assert len(ids) == len(set(ids))

    valid_kinds = {"code_optimization", "training", "kernel", "idea", "intervention"}
    for h in hypotheses:
        assert h["kind"] in valid_kinds

    assert len(llm.calls) == 1


def test_prior_findings_are_injected_into_the_prompt_on_loop_rounds():
    # idea<->experiment loop round 2+: the accumulated ResearchMemory steers the next
    # hypothesis toward something NOT already tried.
    from autoresearch.agents.hypothesis_agent.core import _build_user_message
    prior = {"verified_findings": ["standardization helped Euclidean k-NN"],
             "negative_results": ["z-score hurt Manhattan k-NN"],
             "excluded_hypotheses": ["global min-max scaling"]}
    msg = _build_user_message(RESEARCH_SPEC, LITERATURE_BRIEF, 4, prior)
    assert "already tried" in msg.lower()
    assert "standardization helped Euclidean k-NN" in msg
    assert "z-score hurt Manhattan k-NN" in msg
    assert "global min-max scaling" in msg
    assert "genuinely DIFFERENT" in msg
    # first-pass (no prior) prompt is unchanged -- no findings section
    first = _build_user_message(RESEARCH_SPEC, LITERATURE_BRIEF, 4, None)
    assert "already tried" not in first.lower()


# ---------------------------------------------------------------------------
# (b) malformed LLM JSON -> bounded retry -> typed error.
# ---------------------------------------------------------------------------


def test_malformed_json_retries_then_raises_typed_error():
    llm = FakeLLM(["not json at all"] * (MAX_RETRIES + 1))

    with pytest.raises(HypothesisParseError):
        generate_step(RESEARCH_SPEC, LITERATURE_BRIEF, llm, n=4)

    assert len(llm.calls) == MAX_RETRIES + 1


# ---------------------------------------------------------------------------
# (c) invalid `kind` enum in a hypothesis -> schema validation fails ->
#     retry -> typed error. The bad bank must never be returned.
# ---------------------------------------------------------------------------


def test_invalid_kind_enum_retries_then_raises_typed_error():
    bad_hypotheses = [_hypothesis("h1", "idea"), _hypothesis("h2", "not_a_real_kind")]
    llm = FakeLLM([_bank_json(bad_hypotheses)] * (MAX_RETRIES + 1))

    with pytest.raises(HypothesisParseError):
        generate_step(RESEARCH_SPEC, LITERATURE_BRIEF, llm, n=4)

    assert len(llm.calls) == MAX_RETRIES + 1


# ---------------------------------------------------------------------------
# (d) empty hypotheses list from LLM -> parse failure -> retry -> typed
#     error (never returned as success).
# ---------------------------------------------------------------------------


def test_empty_hypotheses_retries_then_raises_typed_error():
    llm = FakeLLM([_bank_json([])] * (MAX_RETRIES + 1))

    with pytest.raises(HypothesisParseError):
        generate_step(RESEARCH_SPEC, LITERATURE_BRIEF, llm, n=4)

    assert len(llm.calls) == MAX_RETRIES + 1


# ---------------------------------------------------------------------------
# (e) duplicate ids from LLM -> parse failure -> retry -> typed error.
# ---------------------------------------------------------------------------


def test_duplicate_ids_retries_then_raises_typed_error():
    dup_hypotheses = [_hypothesis("h1", "idea"), _hypothesis("h1", "training")]
    llm = FakeLLM([_bank_json(dup_hypotheses)] * (MAX_RETRIES + 1))

    with pytest.raises(HypothesisParseError):
        generate_step(RESEARCH_SPEC, LITERATURE_BRIEF, llm, n=4)

    assert len(llm.calls) == MAX_RETRIES + 1


# ---------------------------------------------------------------------------
# Recovery: a malformed first response followed by a valid one succeeds
# without exhausting all retries.
# ---------------------------------------------------------------------------


def test_recovers_after_one_bad_response():
    llm = FakeLLM(["not json at all", _bank_json(VALID_HYPOTHESES)])

    hypotheses = generate_step(RESEARCH_SPEC, LITERATURE_BRIEF, llm, n=4)

    assert len(hypotheses) == len(VALID_HYPOTHESES)
    assert len(llm.calls) == 2


# ---------------------------------------------------------------------------
# LLM client raising -> typed transient error, no internal retry loop over
# a failing chat() call.
# ---------------------------------------------------------------------------


def test_llm_client_exception_raises_typed_llm_error():
    class RaisingLLM:
        def chat(self, *, messages, response_format=None, max_tokens=None):
            raise RuntimeError("boom")

    with pytest.raises(HypothesisLLMError):
        generate_step(RESEARCH_SPEC, LITERATURE_BRIEF, RaisingLLM(), n=4)


# ---------------------------------------------------------------------------
# MOSS prompt_patch: no-patch unchanged / with-patch injected.
# ---------------------------------------------------------------------------


def test_no_prompt_patch_leaves_system_message_unchanged():
    llm = FakeLLM([_bank_json(VALID_HYPOTHESES)])

    generate_step(RESEARCH_SPEC, LITERATURE_BRIEF, llm, n=4)

    [messages] = llm.calls
    system_message = next(m["content"] for m in messages if m["role"] == "system")
    assert system_message == _load_system_prompt()


def test_prompt_patch_appended_to_system_message():
    llm = FakeLLM([_bank_json(VALID_HYPOTHESES)])

    generate_step(
        RESEARCH_SPEC, LITERATURE_BRIEF, llm, n=4, prompt_patch="STEER: prefer X"
    )

    [messages] = llm.calls
    system_message = next(m["content"] for m in messages if m["role"] == "system")
    assert "STEER: prefer X" in system_message
    assert "[MOSS-injected guidance]" in system_message
