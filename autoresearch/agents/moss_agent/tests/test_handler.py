"""
Tests for MOSS's EventBus handler.
MOSS EventBus handler 的测试。

FakeBusClient (captures publishes; mirrors the real EventBusClient's
publish/subscribe signature -- same shape as memory_agent's FakeBusClient) +
FakeLLM (from this package's tests/conftest.py) + a real tmp SpecStore. The
FakeBusClient does NOT auto-deliver events to subscribed handlers -- each test
calls the handler's event method directly with a constructed Envelope.

The crown jewel here: a QUALITY problem (LLM keeps proposing an illegal
action) DEGRADES inside core.decide and still produces a
moss.intervention_completed (degraded=True) -- never an agent.failed. Only an
INFRA failure (MossLLMError from the LLM client itself raising) becomes
agent.failed.
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

from autoresearch.agents.moss_agent.core import MAX_RETRIES
from autoresearch.agents.moss_agent.handler import MossAgentHandler
from autoresearch.agents.moss_agent.tests.conftest import FakeLLM, RaisingLLM
from autoresearch.orchestrator.moss_matrix import MossActionGuard, load_moss_action_matrix
from autoresearch.spec_store import SpecStore

_CONTRACTS_DIR = Path(__file__).resolve().parents[1] / "data-contracts"


def _validator(name: str) -> Draft202012Validator:
    schema = yaml.safe_load((_CONTRACTS_DIR / name).read_text(encoding="utf-8"))
    return Draft202012Validator(schema)


MOSS_DECISION_VALIDATOR = _validator("moss_decision.schema.yaml")


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
        self.emitter_name = "MOSS"
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


@pytest.fixture()
def guard() -> MossActionGuard:
    return MossActionGuard(load_moss_action_matrix())


# ---------------------------------------------------------------------------
# (a) moss.intervention_requested -> moss.intervention_completed, ref resolves
# to a schema-valid MossDecision whose action is legal for the reason.
# ---------------------------------------------------------------------------


async def test_intervention_requested_publishes_completed_with_resolvable_ref(store, guard):
    llm = FakeLLM([_json({"action": "rewrite_framing", "rationale": "thrashed twice"})])
    handler = MossAgentHandler(spec_store=store, llm_client=llm)
    client = FakeBusClient()
    handler.register(client)

    session_id = str(uuid.uuid4())
    payload = {"reason": "direction_thrashing"}
    env = _make_envelope("moss.intervention_requested", session_id, payload, "ResearchOrchestrator")

    await handler.handle_intervention_requested(client, env)

    completed = [p for p in client.published if p.event_type == "moss.intervention_completed"]
    assert len(completed) == 1
    result_payload = completed[0].payload
    assert set(result_payload.keys()) == {"reason", "action", "moss_intervention_ref", "degraded"}
    assert result_payload["reason"] == "direction_thrashing"
    assert result_payload["action"] == "rewrite_framing"
    assert result_payload["degraded"] is False

    decision = store.get(result_payload["moss_intervention_ref"])
    errors = list(MOSS_DECISION_VALIDATOR.iter_errors(decision))
    assert errors == [], [e.message for e in errors]
    assert decision["action"] in guard.legal_actions("direction_thrashing")

    assert not [p for p in client.published if p.event_type == "agent.failed"]


async def test_component_intervention_receives_producer_and_consumer_contracts(store):
    session_id = str(uuid.uuid4())
    components = [
        {"component_id": "producer", "type": "indirect_gain", "depends_on": [],
         "functional_check": "OUTPUT_PROPERTY_PROBE",
         "downstream_contract": "Consumer wires the produced representation into prediction."},
        {"component_id": "consumer", "type": "direct_gain", "depends_on": ["producer"],
         "mechanism": "CONSUMER_OWNS_PREDICTION_WIRING"},
    ]
    method_ref = store.put(
        kind="method", session_id=session_id, key="confirmed",
        obj={"approach": "STAGED_METHOD", "components": components}, emitter="MethodAgent")
    llm = FakeLLM([_json({"action": "change_plan", "rationale": "inspect ownership",
                         "method_ref": method_ref, "diagnostic_ref": None,
                         "plan_diagnosis": {"component_id": "producer",
                                            "plan_quote": "OUTPUT_PROPERTY_PROBE"}})])
    handler = MossAgentHandler(spec_store=store, llm_client=llm)
    client = FakeBusClient()
    await handler.handle_intervention_requested(client, _make_envelope(
        "moss.intervention_requested", session_id,
        {"reason": "implementation_thrashing", "method_ref": method_ref},
        "ResearchOrchestrator"))
    prompt = "\n".join(message["content"] for message in llm.calls[0])
    assert "STAGED_METHOD" in prompt
    assert "OUTPUT_PROPERTY_PROBE" in prompt
    assert "CONSUMER_OWNS_PREDICTION_WIRING" in prompt
    assert "not empirical proof" in prompt
    assert "deferred consumer wiring" in prompt
    assert store.get(method_ref)["components"] == components
    assert client.published[-1].event_type == "moss.intervention_completed"
    assert client.published[-1].payload["degraded"] is False


async def test_intervention_uses_revised_method_and_frozen_task_boundary(store):
    session_id = str(uuid.uuid4())
    old_ref = store.put(kind="method", session_id=session_id, key="v1",
                        obj={"approach": "OLD_METHOD", "components": []}, emitter="MethodAgent")
    current_ref = store.put(kind="method", session_id=session_id, key="v2",
                            obj={"approach": "METHOD_A_PLUS_B",
                                 "changes_outline": ["RETAIN_A", "DEPENDENT_B"],
                                 "components": [{"component_id": "B", "depends_on": ["A"]}]},
                            emitter="MethodAgent")
    baseline_ref = store.put(kind="baseline_result", session_id=session_id, key="v1",
                             obj={"metric_name": "loss", "lower_is_better": True,
                                  "benchmark_contract": {"editable": ["solve.py"]},
                                  "repository_contract": {"sources": [{
                                      "path": "instruction.md", "content": "NO_HIDDEN_LABELS"}]}},
                             emitter="RunnerAgent")
    spec_ref = store.put(kind="spec_interpretation", session_id=session_id, key="v1",
                         obj={"evaluation_semantics": {"comparison": "same setting"}},
                         emitter="ProblemModelingAgent")
    llm = FakeLLM([_json({"action": "change_plan", "rationale": "check dependent B",
                         "method_ref": current_ref, "diagnostic_ref": None,
                         "plan_diagnosis": {"component_id": "B", "plan_quote": "DEPENDENT_B"}})])
    handler = MossAgentHandler(spec_store=store, llm_client=llm)
    client = FakeBusClient()
    await handler.handle_intervention_requested(client, _make_envelope(
        "moss.intervention_requested", session_id,
        {"reason": "implementation_thrashing", "method_ref": current_ref,
         "baseline_result_ref": baseline_ref, "spec_interpretation_ref": spec_ref},
        "ResearchOrchestrator"))
    prompt = "\n".join(message["content"] for message in llm.calls[0])
    assert "METHOD_A_PLUS_B" in prompt and "DEPENDENT_B" in prompt
    assert "NO_HIDDEN_LABELS" in prompt and "same setting" in prompt
    assert old_ref not in prompt and "OLD_METHOD" not in prompt
    assert client.published[-1].event_type == "moss.intervention_completed"


async def test_missing_method_context_does_not_block_failure_diagnosis(store):
    llm = FakeLLM([_json({"action": "change_plan", "rationale": "need ownership evidence"})])
    handler = MossAgentHandler(spec_store=store, llm_client=llm)
    client = FakeBusClient()
    await handler.handle_intervention_requested(client, _make_envelope(
        "moss.intervention_requested", "missing-method-session",
        {"reason": "implementation_thrashing", "method_ref": "method://missing"},
        "ResearchOrchestrator"))
    assert client.published[-1].event_type == "moss.intervention_completed"


@pytest.mark.parametrize("mir,task_verdict", [
    ("insufficient_evidence", "compliant"),
    ("faithful", "insufficient_evidence"),
])
async def test_unverified_run_review_reaches_moss_discard_guard(store, mir, task_verdict):
    from autoresearch.spec_store.identity import artifact_digest
    sid = str(uuid.uuid4())
    method_ref = store.put(kind="method", session_id=sid, key="v1", emitter="MethodAgent",
        obj={"components": [{"component_id": "restore", "mechanism": "Restore accepted code"}]})
    run = {"succeeded": True, "metric_value": 1.03, "experiment_id": method_ref}
    run_ref = store.put(kind="run_record", session_id=sid, emitter="RunnerAgent", obj=run,
                        key="snapshot-" + artifact_digest(run).split(":")[1])
    candidate = {"run_record_snapshot_ref": run_ref, "method_ref": method_ref,
                 "method_metric": 1.03, "accepted": False,
                 "diagnosis": "candidate_contract_violation",
                 "component_execution": {"mode": "component", "status": "execution_failed",
                                         "execution_succeeded": True},
                 "method_implementation_review": {"verdict": mir,
                     "task_contract_review": {"verdict": task_verdict}}}
    store.put(kind="gen_experiment_result", session_id=sid, emitter="RunnerAgent", obj=candidate,
              key="snapshot-" + artifact_digest(candidate).split(":")[1])
    response = {"method_ref": method_ref, "diagnostic_ref": run_ref,
                "action": "discard_hypothesis", "rationale": "The measured gain is small.",
                "plan_diagnosis": {"component_id": "restore", "plan_quote": "Restore accepted code"}}
    llm = FakeLLM([_json(response), _json({**response, "action": "change_plan",
                                         "rationale": "Bind the missing verification evidence."})])
    handler = MossAgentHandler(spec_store=store, llm_client=llm)
    client = FakeBusClient()
    await handler.handle_intervention_requested(client, _make_envelope(
        "moss.intervention_requested", sid,
        {"reason": "implementation_thrashing", "method_ref": method_ref,
         "run_record_ref": run_ref}, "ResearchOrchestrator"))
    event = client.published[-1]
    assert event.event_type == "moss.intervention_completed"
    assert event.payload["action"] == "change_plan"
    assert event.payload["degraded"] is False
    assert len(llm.calls) == 2


async def test_intervention_uses_plan_critique_instead_of_stale_run_record(store):
    session_id = str(uuid.uuid4())
    critique_ref = store.put(
        kind="critique_report",
        session_id=session_id,
        key="plan-fail",
        obj={"verdict": "fail", "issues": ["PLAN_METRIC_MISMATCH_MARKER"]},
        emitter="CriticAgent",
    )
    run_ref = store.put(
        kind="run_record",
        session_id=session_id,
        key="stale-run",
        obj={"succeeded": False, "error": "STALE_RUN_MARKER"},
        emitter="RunnerAgent",
    )
    llm = FakeLLM([_json({"action": "change_metric", "rationale": "contract mismatch"})])
    handler = MossAgentHandler(spec_store=store, llm_client=llm)
    client = FakeBusClient()

    await handler.handle_intervention_requested(
        client,
        _make_envelope(
            "moss.intervention_requested",
            session_id,
            {
                "reason": "experiment_design_thrashing",
                "critique_report_ref": critique_ref,
                "run_record_ref": run_ref,
            },
            "ResearchOrchestrator",
        ),
    )

    prompt = "\n".join(message["content"] for message in llm.calls[0])
    assert "PLAN_METRIC_MISMATCH_MARKER" in prompt
    assert "STALE_RUN_MARKER" not in prompt


async def test_baseline_failure_evidence_reaches_moss_instead_of_stale_method_run(store):
    session_id = str(uuid.uuid4())
    baseline_ref = store.put(
        kind="baseline_result", session_id=session_id, key="failed",
        obj={"reproduced": False, "metric_value": None,
             "log_excerpt": "CLONE_FAILED_PUBLIC_REPOSITORY_UNAVAILABLE"},
        emitter="RunnerAgent",
    )
    run_ref = store.put(
        kind="run_record", session_id=session_id, key="old-method",
        obj={"error": "STALE_METHOD_RUN"}, emitter="RunnerAgent",
    )
    llm = FakeLLM([_json({"action": "change_benchmark", "rationale": "repository unavailable"})])
    handler = MossAgentHandler(spec_store=store, llm_client=llm)
    client = FakeBusClient()
    await handler.handle_intervention_requested(client, _make_envelope(
        "moss.intervention_requested", session_id,
        {"reason": "baseline_reproduction_thrashing", "baseline_attempt_ref": baseline_ref,
         "run_record_ref": run_ref}, "ResearchOrchestrator"))
    prompt = "\n".join(message["content"] for message in llm.calls[0])
    assert "CLONE_FAILED_PUBLIC_REPOSITORY_UNAVAILABLE" in prompt
    assert "STALE_METHOD_RUN" not in prompt
    assert client.published[-1].event_type == "moss.intervention_completed"


async def test_current_checkpoint_failure_replaces_stale_run_diagnostic(store):
    sid = str(uuid.uuid4())
    run_ref = store.put(
        kind="run_record", session_id=sid, key="old-component",
        obj={"succeeded": True, "stdout_excerpt": "STALE_ACCEPTED_COMPONENT"},
        emitter="RunnerAgent")
    failure = {"agent_name": "RunnerAgent", "reason_classification": "permanent",
               "underlying_reason_code": "component_plan_invalid",
               "underlying_reason": "CURRENT_CHECKPOINT_CONFLICT in shared_driver.cpp"}
    llm = FakeLLM([_json({"action": "revise_component", "rationale": "Repair the join."})])
    handler = MossAgentHandler(spec_store=store, llm_client=llm)
    await handler.handle_intervention_requested(FakeBusClient(), _make_envelope(
        "moss.intervention_requested", sid,
        {"reason": "component_ablation_failed", "run_record_ref": run_ref,
         "agent_failure": failure}, "ResearchOrchestrator"))
    prompt = "\n".join(message["content"] for message in llm.calls[0])
    assert "CURRENT_CHECKPOINT_CONFLICT" in prompt
    assert "STALE_ACCEPTED_COMPONENT" not in prompt


# ---------------------------------------------------------------------------
# (b) quality problem: FakeLLM returns an ILLEGAL action every time -> STILL a
# moss.intervention_completed with degraded=True and a LEGAL fallback action
# -- NOT an agent.failed (the never-hard-stall contract at the handler layer).
# ---------------------------------------------------------------------------


async def test_intervention_requested_quality_problem_degrades_not_fails(store, guard):
    llm = FakeLLM([_json({"action": "do_whatever"})] * (MAX_RETRIES + 1))
    handler = MossAgentHandler(spec_store=store, llm_client=llm)
    client = FakeBusClient()
    handler.register(client)

    session_id = str(uuid.uuid4())
    payload = {"reason": "direction_thrashing"}
    env = _make_envelope("moss.intervention_requested", session_id, payload, "ResearchOrchestrator")

    await handler.handle_intervention_requested(client, env)

    completed = [p for p in client.published if p.event_type == "moss.intervention_completed"]
    assert len(completed) == 1
    result_payload = completed[0].payload
    assert result_payload["degraded"] is True
    assert result_payload["action"] == guard.fallback("direction_thrashing")
    assert result_payload["action"] in guard.legal_actions("direction_thrashing")

    decision = store.get(result_payload["moss_intervention_ref"])
    errors = list(MOSS_DECISION_VALIDATOR.iter_errors(decision))
    assert errors == [], [e.message for e in errors]

    assert not [p for p in client.published if p.event_type == "agent.failed"]


# ---------------------------------------------------------------------------
# (c) infra exception (LLM client raises) -> agent.failed{agent_name:"MOSS"},
# no moss.intervention_completed.
# ---------------------------------------------------------------------------


async def test_intervention_requested_infra_failure_publishes_agent_failed(store):
    llm = RaisingLLM(RuntimeError("connection reset"))
    handler = MossAgentHandler(spec_store=store, llm_client=llm)
    client = FakeBusClient()
    handler.register(client)

    session_id = str(uuid.uuid4())
    payload = {"reason": "direction_thrashing"}
    env = _make_envelope("moss.intervention_requested", session_id, payload, "ResearchOrchestrator")

    await handler.handle_intervention_requested(client, env)

    failed = [p for p in client.published if p.event_type == "agent.failed"]
    assert len(failed) == 1
    assert failed[0].session_id == session_id
    assert failed[0].causation_id == env.event_id
    assert failed[0].payload["agent_name"] == "MOSS"
    assert failed[0].payload["reason_classification"] == "transient"
    assert failed[0].payload["underlying_reason"]

    assert not [p for p in client.published if p.event_type == "moss.intervention_completed"]


# ---------------------------------------------------------------------------
# (d) causation_id set.
# ---------------------------------------------------------------------------


async def test_intervention_requested_causation_id_set(store):
    llm = FakeLLM([_json({"action": "rewrite_framing"})])
    handler = MossAgentHandler(spec_store=store, llm_client=llm)
    client = FakeBusClient()
    handler.register(client)

    session_id = str(uuid.uuid4())
    payload = {"reason": "direction_thrashing"}
    env = _make_envelope("moss.intervention_requested", session_id, payload, "ResearchOrchestrator")

    await handler.handle_intervention_requested(client, env)

    completed = [p for p in client.published if p.event_type == "moss.intervention_completed"]
    assert len(completed) == 1
    assert completed[0].causation_id == env.event_id


# ---------------------------------------------------------------------------
# (e) the guard is built ONCE (in __init__), not per-event.
# ---------------------------------------------------------------------------


def test_guard_built_once_in_init(store, monkeypatch):
    calls: list[Any] = []
    real_load = load_moss_action_matrix

    def _spy_load(path=None):
        calls.append(path)
        return real_load(path)

    monkeypatch.setattr(
        "autoresearch.agents.moss_agent.handler.load_moss_action_matrix", _spy_load
    )

    llm = FakeLLM([_json({"action": "rewrite_framing"})] * 2)
    handler = MossAgentHandler(spec_store=store, llm_client=llm)

    assert len(calls) == 1
    assert isinstance(handler.guard, MossActionGuard)

    client = FakeBusClient()
    handler.register(client)

    for _ in range(2):
        session_id = str(uuid.uuid4())
        payload = {"reason": "direction_thrashing"}
        env = _make_envelope(
            "moss.intervention_requested", session_id, payload, "ResearchOrchestrator"
        )
        import asyncio

        asyncio.run(handler.handle_intervention_requested(client, env))

    # Still only built once, at __init__ time -- not re-loaded per event.
    assert len(calls) == 1


# ---------------------------------------------------------------------------
# register() wires the subscription.
# ---------------------------------------------------------------------------


def test_register_wires_subscription(store):
    llm = FakeLLM([])
    handler = MossAgentHandler(spec_store=store, llm_client=llm)
    client = FakeBusClient()
    handler.register(client)

    patterns = [pattern for pattern, _ in client.subscriptions]
    assert "moss.intervention_requested" in patterns


# ---------------------------------------------------------------------------
# (f) Task guidance is persisted in the immutable decision, not shared configuration.
# ---------------------------------------------------------------------------


async def test_intervention_requested_persists_scoped_patch_without_shared_write(
    store, tmp_path
):
    response = {
        "action": "rewrite_framing",
        "prompt_patch": {"agent": "CoderAgent", "content": "prefer smaller, reversible diffs"},
        "rationale": "keep changes reviewable",
    }
    llm = FakeLLM([_json(response)])
    handler = MossAgentHandler(spec_store=store, llm_client=llm, config_root=tmp_path)
    client = FakeBusClient()
    handler.register(client)

    session_id = str(uuid.uuid4())
    payload = {"reason": "direction_thrashing"}
    env = _make_envelope("moss.intervention_requested", session_id, payload, "ResearchOrchestrator")

    await handler.handle_intervention_requested(client, env)

    completed = [p for p in client.published if p.event_type == "moss.intervention_completed"]
    assert len(completed) == 1

    patch_path = tmp_path / "agent_config" / "CoderAgent" / "prompt_patch.md"
    assert not patch_path.exists()
    from autoresearch.agents.moss_agent.patch_writer import scoped_prompt_patch
    ref = completed[0].payload["moss_intervention_ref"]
    assert "prefer smaller, reversible diffs" in scoped_prompt_patch(store, ref, session_id, "CoderAgent")


# ---------------------------------------------------------------------------
# (g) Intervention completion does not call the global patch writer.
# ---------------------------------------------------------------------------


async def test_intervention_never_calls_global_patch_writer(
    store, tmp_path, monkeypatch
):
    response = {
        "action": "rewrite_framing",
        "prompt_patch": {"agent": "CoderAgent", "content": "prefer smaller, reversible diffs"},
    }
    llm = FakeLLM([_json(response)])
    handler = MossAgentHandler(spec_store=store, llm_client=llm, config_root=tmp_path)
    client = FakeBusClient()
    handler.register(client)

    def _raise(self, agent, content):
        pytest.fail("MOSS must not write a session-specific patch to shared configuration")

    monkeypatch.setattr(
        "autoresearch.agents.moss_agent.patch_writer.PatchWriter.append_prompt_patch", _raise
    )

    session_id = str(uuid.uuid4())
    payload = {"reason": "direction_thrashing"}
    env = _make_envelope("moss.intervention_requested", session_id, payload, "ResearchOrchestrator")

    await handler.handle_intervention_requested(client, env)

    completed = [p for p in client.published if p.event_type == "moss.intervention_completed"]
    assert len(completed) == 1
    assert not [p for p in client.published if p.event_type == "agent.failed"]


@pytest.mark.parametrize("configured,expected", [("0", False), ("1", True), (None, True)])
async def test_intervention_receives_actual_sandbox_tool_mode(store, monkeypatch, configured, expected):
    if configured is None:
        monkeypatch.delenv("AUTORESEARCH_SANDBOX_NATIVE_TOOL_CALLING", raising=False)
    else:
        monkeypatch.setenv("AUTORESEARCH_SANDBOX_NATIVE_TOOL_CALLING", configured)
    llm = FakeLLM([_json({"action": "shrink_patch"})])
    handler = MossAgentHandler(spec_store=store, llm_client=llm)
    env = _make_envelope("moss.intervention_requested", str(uuid.uuid4()),
                         {"reason": "implementation_thrashing"}, "ResearchOrchestrator")
    await handler.handle_intervention_requested(FakeBusClient(), env)
    message = llm.calls[0][1]["content"]
    assert f'"sandbox_native_tool_calling": {str(expected).lower()}' in message
    assert "SDK-emulated function interface" in message
    assert "does not authorize changing the execution backend" in message
