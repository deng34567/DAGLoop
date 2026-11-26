"""
Tests for CoderAgent's pure core. / CoderAgent 纯核心的测试。

No EventBus / Store / Redis / network — only a FakeLLM (canned JSON queue).
``implement_patch`` is PURE: it takes an ExperimentPlan + VerifierSpec + an
llm_client, and returns a schema-valid ``ExperimentPatch`` dict whose
``changes[].patch`` holds actual code.

KEY DESIGN POINT (the crown jewel of these tests): unlike
experiment_planner_agent's ``plan_experiment``, ``implement_patch`` does NOT
enforce ``verifier_spec["editable_region"]`` / ``forbidden_files`` -- it
merely relays them to the LLM as guidance. A patch that touches a file
OUTSIDE editable_region (or inside forbidden_files) is still returned
successfully, with an honestly-computed ``touched_files``. The orchestrator
gate (a later milestone) is the sole enforcement point. Test (c) below proves
this directly.

``touched_files`` is always CORE-DERIVED from ``changes[].file`` -- the LLM
is never asked to supply it, and even if it did, the core would not trust it
(the core always recomputes it as ``sorted({c["file"] for c in changes})``).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from jsonschema import Draft202012Validator

from autoresearch.agents.coder_agent.core import (
    MAX_RETRIES,
    CoderParseError,
    _load_implement_patch_system_prompt,
    implement_patch,
)
from autoresearch.agents.coder_agent.tests.conftest import FakeLLM

_CONTRACTS_DIR = Path(__file__).resolve().parents[1] / "data-contracts"


def _validator(name: str) -> Draft202012Validator:
    schema = yaml.safe_load((_CONTRACTS_DIR / name).read_text(encoding="utf-8"))
    return Draft202012Validator(schema)


EXPERIMENT_PATCH_VALIDATOR = _validator("experiment_patch.schema.yaml")

EXPERIMENT_PLAN = {
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


def _json(obj: dict) -> str:
    return json.dumps(obj)


def _valid_patch(**overrides: object) -> dict:
    patch = {
        "hypothesis_id": "hyp-1",
        "summary": "Add a learned gating layer before retrieval fusion.",
        "changes": [
            {"file": "src/model.py", "patch": "+class Gate(nn.Module):\n+    ..."},
            {"file": "src/train.py", "patch": "+model.gate = Gate()\n"},
        ],
    }
    patch.update(overrides)
    return patch


def _implement_patch(llm):
    return implement_patch(EXPERIMENT_PLAN, VERIFIER_SPEC, llm)


# ---------------------------------------------------------------------------
# (a) valid -> schema-valid ExperimentPatch; touched_files core-derived.
# ---------------------------------------------------------------------------


def test_implement_patch_valid_returns_schema_valid_patch():
    llm = FakeLLM([_json(_valid_patch())])

    result = _implement_patch(llm)

    errors = list(EXPERIMENT_PATCH_VALIDATOR.iter_errors(result))
    assert errors == [], [e.message for e in errors]
    assert result["hypothesis_id"] == "hyp-1"
    assert result["touched_files"] == sorted({"src/model.py", "src/train.py"})
    assert len(llm.calls) == 1


def test_implement_patch_touched_files_is_core_derived_not_llm_supplied():
    # The LLM supplies a bogus/inconsistent touched_files -- the core must
    # ignore it and recompute from changes[].file itself.
    llm = FakeLLM([_json(_valid_patch(touched_files=["totally/bogus.py"]))])

    result = _implement_patch(llm)

    assert result["touched_files"] == sorted({"src/model.py", "src/train.py"})


def test_implement_patch_recovers_after_one_bad_response():
    llm = FakeLLM(["not json at all", _json(_valid_patch())])

    result = _implement_patch(llm)

    assert result["hypothesis_id"] == "hyp-1"
    assert len(llm.calls) == 2


# ---------------------------------------------------------------------------
# (b) a patch whose files are all in editable_region -> succeeds.
# ---------------------------------------------------------------------------


def test_implement_patch_in_region_patch_succeeds():
    in_region_patch = _valid_patch(
        changes=[
            {"file": "src/model.py", "patch": "+class Gate(nn.Module): ...\n"},
        ]
    )
    llm = FakeLLM([_json(in_region_patch)])

    result = _implement_patch(llm)

    assert result["touched_files"] == ["src/model.py"]
    assert set(result["touched_files"]) <= set(VERIFIER_SPEC["editable_region"])


# ---------------------------------------------------------------------------
# (c) CROWN JEWEL: a patch touching an OUT-of-editable-region (and even
# forbidden) file ALSO succeeds -- the core does NOT self-police
# editable_region/forbidden_files. touched_files honestly includes it.
# ---------------------------------------------------------------------------


def test_implement_patch_out_of_region_file_still_succeeds_core_does_not_gate():
    out_of_region_patch = _valid_patch(
        changes=[
            {"file": "src/model.py", "patch": "+class Gate(nn.Module): ...\n"},
            {"file": "eval/scorer.py", "patch": "+# sneak an out-of-scope tweak\n"},
        ]
    )
    llm = FakeLLM([_json(out_of_region_patch)])

    result = _implement_patch(llm)

    errors = list(EXPERIMENT_PATCH_VALIDATOR.iter_errors(result))
    assert errors == [], [e.message for e in errors]
    assert result["touched_files"] == sorted(["src/model.py", "eval/scorer.py"])
    # Honestly reported even though it is NOT a subset of editable_region.
    assert not set(result["touched_files"]) <= set(VERIFIER_SPEC["editable_region"])
    assert len(llm.calls) == 1


def test_implement_patch_forbidden_file_still_succeeds_core_does_not_gate():
    forbidden_patch = _valid_patch(
        changes=[
            {"file": "benchmark/data/held_out.jsonl", "patch": "+tampered\n"},
        ]
    )
    llm = FakeLLM([_json(forbidden_patch)])

    result = _implement_patch(llm)

    assert result["touched_files"] == ["benchmark/data/held_out.jsonl"]
    assert set(result["touched_files"]) & set(VERIFIER_SPEC["forbidden_files"])


# ---------------------------------------------------------------------------
# (d) malformed JSON / empty changes -> retry -> typed error.
# ---------------------------------------------------------------------------


def test_implement_patch_malformed_json_retries_then_raises_typed_error():
    llm = FakeLLM(["not json at all"] * (MAX_RETRIES + 1))

    with pytest.raises(CoderParseError):
        _implement_patch(llm)

    assert len(llm.calls) == MAX_RETRIES + 1


def test_implement_patch_non_dict_json_retries_then_raises_typed_error():
    llm = FakeLLM([json.dumps(["not", "an", "object"])] * (MAX_RETRIES + 1))

    with pytest.raises(CoderParseError):
        _implement_patch(llm)

    assert len(llm.calls) == MAX_RETRIES + 1


def test_implement_patch_empty_changes_retries_then_raises_typed_error():
    empty_patch = _valid_patch(changes=[])
    llm = FakeLLM([_json(empty_patch)] * (MAX_RETRIES + 1))

    with pytest.raises(CoderParseError):
        _implement_patch(llm)

    assert len(llm.calls) == MAX_RETRIES + 1


def test_implement_patch_schema_invalid_retries_then_raises_typed_error():
    bad_patch = _valid_patch()
    del bad_patch["summary"]
    llm = FakeLLM([_json(bad_patch)] * (MAX_RETRIES + 1))

    with pytest.raises(CoderParseError):
        _implement_patch(llm)

    assert len(llm.calls) == MAX_RETRIES + 1


def test_implement_patch_change_missing_patch_field_retries_then_raises_typed_error():
    bad_patch = _valid_patch(changes=[{"file": "src/model.py"}])
    llm = FakeLLM([_json(bad_patch)] * (MAX_RETRIES + 1))

    with pytest.raises(CoderParseError):
        _implement_patch(llm)

    assert len(llm.calls) == MAX_RETRIES + 1


# ---------------------------------------------------------------------------
# MOSS prompt_patch: no-patch unchanged / with-patch injected.
# ---------------------------------------------------------------------------


def test_implement_patch_no_prompt_patch_leaves_system_message_unchanged():
    llm = FakeLLM([_json(_valid_patch())])

    _implement_patch(llm)

    [messages] = llm.calls
    system_message = next(m["content"] for m in messages if m["role"] == "system")
    assert system_message == _load_implement_patch_system_prompt()


def test_implement_patch_prompt_patch_appended_to_system_message():
    llm = FakeLLM([_json(_valid_patch())])

    implement_patch(
        EXPERIMENT_PLAN, VERIFIER_SPEC, llm, prompt_patch="STEER: prefer X"
    )

    [messages] = llm.calls
    system_message = next(m["content"] for m in messages if m["role"] == "system")
    assert "STEER: prefer X" in system_message
    assert "[MOSS-injected guidance]" in system_message
