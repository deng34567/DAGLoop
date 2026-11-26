"""
Tests for CoderAgent's P1 trivial-patch branch (ImplementingExperiment).

DEFAULT-OFF: the branch fires only when AUTORESEARCH_GEN_EXPERIMENT=1. When on,
CoderAgent emits a TRIVIAL valid ExperimentPatch touching a single file inside
the frozen editable_region WITHOUT any LLM call -- bypassing the hang-prone real
coder (its patch is unused by P1, which self-writes method.py). When off, the
real core.implement_patch path runs, byte-identical (test_handler.py).
"""

from __future__ import annotations

import uuid
from pathlib import Path

import yaml
from jsonschema import Draft202012Validator

from autoresearch.agents.coder_agent.handler import CoderAgentHandler
from autoresearch.agents.coder_agent.tests.conftest import FakeLLM
from autoresearch.agents.coder_agent.tests.test_handler import (
    FakeBusClient,
    VALID_EXPERIMENT_PLAN,
    VALID_VERIFIER_SPEC,
    _make_envelope,
    _seed_implementation_inputs,
    store,  # noqa: F401  (pytest fixture)
)

_CONTRACTS_DIR = Path(__file__).resolve().parents[1] / "data-contracts"
EXPERIMENT_PATCH_VALIDATOR = Draft202012Validator(
    yaml.safe_load((_CONTRACTS_DIR / "experiment_patch.schema.yaml").read_text(encoding="utf-8"))
)


# ---------------------------------------------------------------------------
# env ON -> trivial in-editable-region patch, NO llm call.
# ---------------------------------------------------------------------------


async def test_gen_branch_emits_trivial_patch_without_llm(store, monkeypatch):
    monkeypatch.setenv("AUTORESEARCH_GEN_EXPERIMENT", "1")
    llm = FakeLLM([])  # nothing queued; any call would raise/error
    handler = CoderAgentHandler(spec_store=store, llm_client=llm)
    client = FakeBusClient()
    handler.register(client)

    sid = str(uuid.uuid4())
    payload = _seed_implementation_inputs(store, sid)
    env = _make_envelope("code.implementation_requested", sid, payload, "ExperimentPlannerAgent")

    await handler.handle_implementation_requested(client, env)

    # NO LLM call
    assert llm.calls == []

    implemented = [p for p in client.published if p.event_type == "code.implemented"]
    assert len(implemented) == 1
    patch = store.get(implemented[0].payload["experiment_patch_ref"])

    errors = list(EXPERIMENT_PATCH_VALIDATOR.iter_errors(patch))
    assert errors == [], [e.message for e in errors]

    # touched_files subset of editable_region -> the engine protocol gate passes
    editable = set(VALID_VERIFIER_SPEC["editable_region"])
    assert set(patch["touched_files"]) <= editable
    assert patch["touched_files"] == [VALID_VERIFIER_SPEC["editable_region"][0]]
    assert implemented[0].causation_id == env.event_id


# ---------------------------------------------------------------------------
# DEFAULT-OFF: env off -> real core.implement_patch path (LLM IS called).
# ---------------------------------------------------------------------------


async def test_env_off_uses_real_coder_path(store, monkeypatch):
    import json

    monkeypatch.delenv("AUTORESEARCH_GEN_EXPERIMENT", raising=False)
    valid_patch = {
        "hypothesis_id": "h1",
        "summary": "real patch",
        "changes": [{"file": "src/model.py", "patch": "+x\n"}],
    }
    llm = FakeLLM([json.dumps(valid_patch)])
    handler = CoderAgentHandler(spec_store=store, llm_client=llm)
    client = FakeBusClient()
    handler.register(client)

    sid = str(uuid.uuid4())
    payload = _seed_implementation_inputs(store, sid)
    env = _make_envelope("code.implementation_requested", sid, payload, "ExperimentPlannerAgent")

    await handler.handle_implementation_requested(client, env)

    assert len(llm.calls) == 1  # real path called the LLM
    implemented = [p for p in client.published if p.event_type == "code.implemented"]
    assert len(implemented) == 1
