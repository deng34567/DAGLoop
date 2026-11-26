"""
Tests for CoderAgent's EventBus handler.
CoderAgent EventBus handler 的测试。

FakeBusClient (captures publishes; mirrors the real EventBusClient's
publish/subscribe signature -- same shape as experiment_planner_agent's
FakeBusClient) + FakeLLM (from this package's tests/conftest.py) + a real tmp
SpecStore. The FakeBusClient does NOT auto-deliver events to subscribed
handlers -- each test calls the handler's event method directly with a
constructed Envelope.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import pytest
import yaml
from jsonschema import Draft202012Validator

from autoresearch.agents.coder_agent import core
from autoresearch.agents.coder_agent.handler import CoderAgentHandler
from autoresearch.agents.coder_agent.tests.conftest import FakeLLM
from autoresearch.spec_store import SpecStore

_CONTRACTS_DIR = Path(__file__).resolve().parents[1] / "data-contracts"


def _validator(name: str) -> Draft202012Validator:
    schema = yaml.safe_load((_CONTRACTS_DIR / name).read_text(encoding="utf-8"))
    return Draft202012Validator(schema)


EXPERIMENT_PATCH_VALIDATOR = _validator("experiment_patch.schema.yaml")

VALID_EXPERIMENT_PLAN = {
    "hypothesis_id": "h1",
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

VALID_VERIFIER_SPEC = {
    "evaluator_hash": "sha256:abc123",
    "metric": "exact_match",
    "data_split": {"in_loop": "dev", "held_out": "test"},
    "run_budget": "10 runs, 1h wall-clock",
    "seed_policy": "fixed seed=42 across all runs",
    "editable_region": ["src/model.py", "src/train.py"],
    "forbidden_files": ["benchmark/data/held_out.jsonl"],
    "result_parse_schema": {"type": "object"},
}


def _valid_patch(**overrides: object) -> dict:
    patch = {
        "hypothesis_id": "h1",
        "summary": "Add a learned gating layer before retrieval fusion.",
        "changes": [
            {"file": "src/model.py", "patch": "+class Gate(nn.Module):\n+    ..."},
            {"file": "src/train.py", "patch": "+model.gate = Gate()\n"},
        ],
    }
    patch.update(overrides)
    return patch


def _json(obj: dict) -> str:
    return json.dumps(obj)


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


@dataclass
class _Published:
    event_type: str
    session_id: str
    payload: dict
    causation_id: Optional[str]


class FakeBusClient:
    """Captures publishes; mirrors the real EventBusClient.publish/subscribe
    signature. Does NOT auto-deliver events to subscribed handlers."""

    def __init__(self) -> None:
        self.emitter_name = "CoderAgent"
        self.published: list[_Published] = []
        self.subscriptions: list[tuple[str, Any]] = []

    def subscribe(self, pattern: str, handler: Any) -> None:
        self.subscriptions.append((pattern, handler))

    async def publish(
        self,
        *,
        event_type: str,
        session_id: str,
        payload: dict,
        cost_delta: str = "0",
        causation_id: Optional[str] = None,
        auto_commit: bool = True,
    ) -> str:
        self.published.append(
            _Published(
                event_type=event_type,
                session_id=session_id,
                payload=payload,
                causation_id=causation_id,
            )
        )
        return f"evt-{len(self.published)}"


@dataclass
class _Envelope:
    """The subset of the real broker Envelope the handler reads."""

    event_id: str
    session_id: str
    event_type: str
    emitter: str
    payload: dict


def _make_envelope(event_type: str, session_id: str, payload: dict, emitter: str) -> _Envelope:
    return _Envelope(
        event_id=str(uuid.uuid4()),
        session_id=session_id,
        event_type=event_type,
        emitter=emitter,
        payload=payload,
    )


@pytest.fixture()
def store(tmp_path):
    s = SpecStore(tmp_path / "store")
    yield s
    s.close()


def _seed_implementation_inputs(
    store: SpecStore,
    session_id: str,
    *,
    experiment_plan: dict = VALID_EXPERIMENT_PLAN,
    verifier_spec: dict = VALID_VERIFIER_SPEC,
) -> dict:
    """Seed the store with the two inputs CoderAgent reads for
    ImplementingExperiment, as upstream agents would; return the
    code.implementation_requested payload (refs only)."""
    experiment_plan_ref = store.put(
        kind="experiment_plan",
        session_id=session_id,
        key="v1",
        obj=experiment_plan,
        emitter="ExperimentPlannerAgent",
    )
    verifier_spec_ref = store.put(
        kind="verifier_spec",
        session_id=session_id,
        key="v1",
        obj=verifier_spec,
        emitter="EvaluatorAgent",
    )
    return {
        "experiment_plan_ref": experiment_plan_ref,
        "verifier_spec_ref": verifier_spec_ref,
    }


# ---------------------------------------------------------------------------
# (a) code.implementation_requested -> code.implemented, ref resolves to a
# schema-valid ExperimentPatch.
# ---------------------------------------------------------------------------


async def test_implementation_requested_publishes_implemented_with_resolvable_ref(store):
    llm = FakeLLM([_json(_valid_patch())])
    handler = CoderAgentHandler(spec_store=store, llm_client=llm)
    client = FakeBusClient()
    handler.register(client)

    session_id = str(uuid.uuid4())
    payload = _seed_implementation_inputs(store, session_id)

    env = _make_envelope(
        "code.implementation_requested", session_id, payload, "ExperimentPlannerAgent"
    )

    await handler.handle_implementation_requested(client, env)

    implemented = [p for p in client.published if p.event_type == "code.implemented"]
    assert len(implemented) == 1
    result_payload = implemented[0].payload
    assert set(result_payload.keys()) == {"experiment_patch_ref"}

    patch = store.get(result_payload["experiment_patch_ref"])
    errors = list(EXPERIMENT_PATCH_VALIDATOR.iter_errors(patch))
    assert errors == [], [e.message for e in errors]
    assert patch["hypothesis_id"] == "h1"


# ---------------------------------------------------------------------------
# (b) provenance: both inputs read from the store (never from payload inline).
# ---------------------------------------------------------------------------


async def test_implementation_requested_reads_inputs_from_store_not_payload(store):
    llm = FakeLLM([_json(_valid_patch())])
    handler = CoderAgentHandler(spec_store=store, llm_client=llm)
    client = FakeBusClient()
    handler.register(client)

    session_id = str(uuid.uuid4())
    payload = _seed_implementation_inputs(store, session_id)
    assert set(payload.keys()) == {"experiment_plan_ref", "verifier_spec_ref"}

    env = _make_envelope(
        "code.implementation_requested", session_id, payload, "ExperimentPlannerAgent"
    )

    await handler.handle_implementation_requested(client, env)

    implemented = [p for p in client.published if p.event_type == "code.implemented"]
    assert len(implemented) == 1

    [messages] = llm.calls
    user_message = next(m["content"] for m in messages if m["role"] == "user")
    assert VALID_EXPERIMENT_PLAN["change_summary"] in user_message
    assert VALID_VERIFIER_SPEC["evaluator_hash"] in user_message


# ---------------------------------------------------------------------------
# (c) core failure (malformed LLM response) -> agent.failed with correct
# classification; no code.implemented; never a crash.
# ---------------------------------------------------------------------------


async def test_implementation_requested_core_error_publishes_agent_failed(store):
    llm = FakeLLM(["not json at all"] * (core.MAX_RETRIES + 1))
    handler = CoderAgentHandler(spec_store=store, llm_client=llm)
    client = FakeBusClient()
    handler.register(client)

    session_id = str(uuid.uuid4())
    payload = _seed_implementation_inputs(store, session_id)

    env = _make_envelope(
        "code.implementation_requested", session_id, payload, "ExperimentPlannerAgent"
    )

    await handler.handle_implementation_requested(client, env)

    failed = [p for p in client.published if p.event_type == "agent.failed"]
    assert len(failed) == 1
    assert failed[0].session_id == session_id
    assert failed[0].causation_id == env.event_id
    assert failed[0].payload["agent_name"] == "CoderAgent"
    assert failed[0].payload["reason_classification"] == "permanent"
    assert failed[0].payload["underlying_reason"]

    assert not [p for p in client.published if p.event_type == "code.implemented"]


# ---------------------------------------------------------------------------
# (d) causation_id set on the success path.
# ---------------------------------------------------------------------------


async def test_implementation_requested_causation_id_set(store):
    llm = FakeLLM([_json(_valid_patch())])
    handler = CoderAgentHandler(spec_store=store, llm_client=llm)
    client = FakeBusClient()
    handler.register(client)

    session_id = str(uuid.uuid4())
    payload = _seed_implementation_inputs(store, session_id)

    env = _make_envelope(
        "code.implementation_requested", session_id, payload, "ExperimentPlannerAgent"
    )

    await handler.handle_implementation_requested(client, env)

    implemented = [p for p in client.published if p.event_type == "code.implemented"]
    assert len(implemented) == 1
    assert implemented[0].causation_id == env.event_id


# ---------------------------------------------------------------------------
# register() wires the subscription.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# MOSS prompt_patch: config_root/agent_config/CoderAgent/prompt_patch.md gets
# read by self.config.prompt_patch() and injected into the system message.
# ---------------------------------------------------------------------------


async def test_implementation_requested_prompt_patch_from_config_root_injected_into_system_message(
    store, tmp_path
):
    patch_dir = tmp_path / "agent_config" / "CoderAgent"
    patch_dir.mkdir(parents=True)
    (patch_dir / "prompt_patch.md").write_text("STEER: prefer X", encoding="utf-8")

    llm = FakeLLM([_json(_valid_patch())])
    handler = CoderAgentHandler(spec_store=store, llm_client=llm, config_root=tmp_path)
    client = FakeBusClient()
    handler.register(client)

    session_id = str(uuid.uuid4())
    payload = _seed_implementation_inputs(store, session_id)

    env = _make_envelope(
        "code.implementation_requested", session_id, payload, "ExperimentPlannerAgent"
    )

    await handler.handle_implementation_requested(client, env)

    [messages] = llm.calls
    system_message = next(m["content"] for m in messages if m["role"] == "system")
    assert "STEER: prefer X" in system_message
    assert "[MOSS-injected guidance]" in system_message


def test_register_wires_subscription(store):
    llm = FakeLLM([])
    handler = CoderAgentHandler(spec_store=store, llm_client=llm)
    client = FakeBusClient()
    handler.register(client)

    patterns = [pattern for pattern, _ in client.subscriptions]
    assert "code.implementation_requested" in patterns
