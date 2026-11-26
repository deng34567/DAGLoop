"""
Tests for ExperimentPlannerAgent's EventBus handler.
ExperimentPlannerAgent EventBus handler 的测试。

FakeBusClient (captures publishes; mirrors the real EventBusClient's
publish/subscribe signature -- same shape as evaluator_agent's FakeBusClient)
+ FakeLLM (from this package's tests/conftest.py) + a real tmp SpecStore. The
FakeBusClient does NOT auto-deliver events to subscribed handlers -- each test
calls the handler's event method directly with a constructed Envelope.
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

from autoresearch.agents.experiment_planner_agent import core
from autoresearch.agents.experiment_planner_agent.handler import (
    ExperimentPlannerAgentHandler,
    _component_memory_packet,
)
from autoresearch.agents.experiment_planner_agent.tests.conftest import FakeLLM
from autoresearch.experiments.repo_improve.component_plan import (
    build_component_plan,
    ledger_for_workroot,
)
from autoresearch.spec_store import SpecStore
from autoresearch.spec_store.identity import artifact_digest

_CONTRACTS_DIR = Path(__file__).resolve().parents[1] / "data-contracts"


def _validator(name: str) -> Draft202012Validator:
    schema = yaml.safe_load((_CONTRACTS_DIR / name).read_text(encoding="utf-8"))
    return Draft202012Validator(schema)


EXPERIMENT_PLAN_VALIDATOR = _validator("experiment_plan.schema.yaml")


def test_component_memory_packet_keeps_lineage_detailed_and_every_sibling_summarized():
    history = [
        {"sequence": 1, "component_id": "a", "run_ref": "run:a:1", "verification": "full"},
        {"sequence": 2, "component_id": "a", "run_ref": "run:a:2", "verification": "full"},
        {"sequence": 3, "component_id": "b", "run_ref": "run:b:1", "verification": "full"},
    ]
    packet = _component_memory_packet(
        {"component_history": history}, "h", ("a", "d", "e", "f", "g"))
    assert packet["current_component"] == "h"
    assert packet["dependency_closure"] == ["a", "d", "e", "f", "g"]
    assert packet["current_and_ancestor_rounds"] == history[:2]
    assert packet["other_branch_round_summaries"] == [
        {"sequence": 3, "component_id": "b", "run_ref": "run:b:1"}]
    assert packet["history_entry_count"] == len(history)

VALID_RESEARCH_SPEC = {
    "topic": "Retrieval-augmented generation for long-tail scientific QA",
    "research_goal": "Propose a new RAG method for long-tail scientific question answering.",
}

VALID_BENCHMARK_SPEC = {
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

HYP_1 = {
    "id": "h1",
    "statement": "Adding a learned gating mechanism to the retriever improves long-tail recall.",
    "rationale": "Dense retrievers underperform on long-tail entities.",
    "kind": "training",
    "verifiable": True,
    "expected_effect": "exact_match improves on the long-tail subset.",
}

HYP_2 = {
    "id": "h2",
    "statement": "Switching to a sparse retriever improves long-tail recall.",
    "rationale": "Sparse retrievers handle rare entities better via exact term matching.",
    "kind": "idea",
    "verifiable": True,
    "expected_effect": "exact_match improves on the long-tail subset via sparse retrieval.",
}

VALID_HYPOTHESIS_BANK = {"hypotheses": [HYP_1, HYP_2]}


def _valid_plan(hypothesis_id: str = "h1", **overrides: object) -> dict:
    plan = {
        "hypothesis_id": hypothesis_id,
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
        self.emitter_name = "ExperimentPlannerAgent"
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


def _seed_planning_inputs(
    store: SpecStore,
    session_id: str,
    *,
    hypothesis_bank: dict = VALID_HYPOTHESIS_BANK,
    verifier_spec: dict = VALID_VERIFIER_SPEC,
    benchmark_spec: dict = VALID_BENCHMARK_SPEC,
    research_spec: dict = VALID_RESEARCH_SPEC,
    selected_hypothesis_id: str = "h1",
) -> dict:
    """Seed the store with the four inputs ExperimentPlannerAgent reads for
    PlanningExperiment, as upstream agents would; return the
    experiment.planning_requested payload (refs + selected_hypothesis_id)."""
    hypothesis_bank_ref = store.put(
        kind="hypothesis_bank",
        session_id=session_id,
        key="final",
        obj=hypothesis_bank,
        emitter="HypothesisAgent",
    )
    verifier_spec_ref = store.put(
        kind="verifier_spec",
        session_id=session_id,
        key="v1",
        obj=verifier_spec,
        emitter="EvaluatorAgent",
    )
    benchmark_spec_ref = store.put(
        kind="benchmark_spec",
        session_id=session_id,
        key="v1",
        obj=benchmark_spec,
        emitter="EvaluatorAgent",
    )
    research_spec_ref = store.put(
        kind="research_spec",
        session_id=session_id,
        key="final",
        obj=research_spec,
        emitter="ProblemModelingAgent",
    )
    return {
        "hypothesis_bank_ref": hypothesis_bank_ref,
        "selected_hypothesis_id": selected_hypothesis_id,
        "verifier_spec_ref": verifier_spec_ref,
        "benchmark_spec_ref": benchmark_spec_ref,
        "research_spec_ref": research_spec_ref,
    }


# ---------------------------------------------------------------------------
# (a) experiment.planning_requested -> experiment.plan_ready, ref resolves to
# a schema-valid ExperimentPlan whose hypothesis_id == the selected one.
# ---------------------------------------------------------------------------


async def test_planning_requested_publishes_plan_ready_with_resolvable_ref(store):
    llm = FakeLLM([_json(_valid_plan("h1"))])
    handler = ExperimentPlannerAgentHandler(spec_store=store, llm_client=llm)
    client = FakeBusClient()
    handler.register(client)

    session_id = str(uuid.uuid4())
    payload = _seed_planning_inputs(store, session_id, selected_hypothesis_id="h1")

    env = _make_envelope(
        "experiment.planning_requested", session_id, payload, "ResearchOrchestrator"
    )

    await handler.handle_planning_requested(client, env)

    ready = [p for p in client.published if p.event_type == "experiment.plan_ready"]
    assert len(ready) == 1
    result_payload = ready[0].payload
    assert set(result_payload.keys()) == {"experiment_plan_ref"}

    plan = store.get(result_payload["experiment_plan_ref"])
    errors = list(EXPERIMENT_PLAN_VALIDATOR.iter_errors(plan))
    assert errors == [], [e.message for e in errors]
    assert plan["hypothesis_id"] == "h1"


async def test_survey_validation_plans_without_benchmark_or_hypothesis(store):
    llm = FakeLLM([])
    handler = ExperimentPlannerAgentHandler(spec_store=store, llm_client=llm)
    client = FakeBusClient()
    session_id = str(uuid.uuid4())
    research_spec_ref = store.put(
        kind="research_spec", session_id=session_id, key="final",
        obj={"topic": "publication value", "research_goal": "test one claim",
             "task_mode": "experiment_validation"},
        emitter="ProblemModelingAgent",
    )
    interpretation_ref = store.put(
        kind="spec_interpretation", session_id=session_id, key="v1",
        obj={"validation_study": {
            "kind": "survey_validation",
            "question": "Does the observable trend support the claim?",
            "sampling_frame": "A dated sample of public metadata.",
            "data_sources": ["OpenAlex"],
            "variables": ["year", "citation_count"],
            "analysis": "Descriptive trend and permutation test.",
            "falsification_condition": "No trend or an opposite trend.",
            "budget": "100 records and 10 minutes.",
        }},
        emitter="ProblemModelingAgent",
    )
    env = _make_envelope(
        "experiment.planning_requested", session_id,
        {
            "hypothesis_bank_ref": None,
            "selected_hypothesis_id": None,
            "verifier_spec_ref": None,
            "benchmark_spec_ref": None,
            "research_spec_ref": research_spec_ref,
            "method_ref": None,
            "baseline_result_ref": None,
            "plan_review_ref": None,
            "spec_interpretation_ref": interpretation_ref,
        },
        "ResearchOrchestrator",
    )

    await handler.handle_planning_requested(client, env)

    ready = [p for p in client.published if p.event_type == "experiment.plan_ready"]
    assert len(ready) == 1
    plan = store.get(ready[0].payload["experiment_plan_ref"])
    assert list(EXPERIMENT_PLAN_VALIDATOR.iter_errors(plan)) == []
    assert plan["task_class"] == "survey_validation"
    assert "baseline" not in plan["success_criteria"].lower()
    assert llm.calls == []


async def test_auxiliary_validation_study_does_not_replace_benchmark_method(store):
    llm = FakeLLM([_json(_valid_plan("method-1"))])
    handler = ExperimentPlannerAgentHandler(spec_store=store, llm_client=llm)
    client = FakeBusClient()
    session_id = str(uuid.uuid4())
    research_spec_ref = store.put(
        kind="research_spec", session_id=session_id, key="final",
        obj={"topic": "repository optimization", "research_goal": "improve official metric",
             "task_mode": "benchmark"}, emitter="ProblemModelingAgent")
    interpretation_ref = store.put(
        kind="spec_interpretation", session_id=session_id, key="v1",
        obj={"validation_study": {
            "kind": "survey_validation", "question": "Does one component help?",
            "sampling_frame": "Synthetic supporting data.", "data_sources": ["generated"],
            "variables": ["latency"], "analysis": "Compare summaries.",
            "falsification_condition": "No reduction.", "budget": "Ten minutes.",
        }}, emitter="ProblemModelingAgent")
    method_ref = store.put(
        kind="method", session_id=session_id, key="confirmed",
        obj={"method_name": "method-1", "description": "Optimize the repository."},
        emitter="MethodAgent")
    baseline_ref = store.put(
        kind="baseline_result", session_id=session_id, key="official",
        obj={"metric_value": 10.0}, emitter="RunnerAgent")
    env = _make_envelope(
        "experiment.planning_requested", session_id,
        {"hypothesis_bank_ref": None, "selected_hypothesis_id": None,
         "verifier_spec_ref": None, "benchmark_spec_ref": None,
         "research_spec_ref": research_spec_ref, "method_ref": method_ref,
         "baseline_result_ref": baseline_ref, "plan_review_ref": None,
         "spec_interpretation_ref": interpretation_ref},
        "ResearchOrchestrator")

    await handler.handle_planning_requested(client, env)

    ready = [p for p in client.published if p.event_type == "experiment.plan_ready"]
    assert len(ready) == 1
    plan = store.get(ready[0].payload["experiment_plan_ref"])
    assert plan.get("task_class") != "survey_validation"
    assert plan["hypothesis_id"] == "method-1"


# ---------------------------------------------------------------------------
# (b) provenance: all four inputs read from the store (never from payload
# inline), and the correct hypothesis (by id) is the one used -- not simply
# the first one in the bank.
# ---------------------------------------------------------------------------


async def test_planning_requested_reads_inputs_from_store_not_payload(store):
    llm = FakeLLM([_json(_valid_plan("h1"))])
    handler = ExperimentPlannerAgentHandler(spec_store=store, llm_client=llm)
    client = FakeBusClient()
    handler.register(client)

    session_id = str(uuid.uuid4())
    payload = _seed_planning_inputs(store, session_id, selected_hypothesis_id="h1")
    assert set(payload.keys()) == {
        "hypothesis_bank_ref",
        "selected_hypothesis_id",
        "verifier_spec_ref",
        "benchmark_spec_ref",
        "research_spec_ref",
    }

    env = _make_envelope(
        "experiment.planning_requested", session_id, payload, "ResearchOrchestrator"
    )

    await handler.handle_planning_requested(client, env)

    ready = [p for p in client.published if p.event_type == "experiment.plan_ready"]
    assert len(ready) == 1

    [messages] = llm.calls
    user_message = next(m["content"] for m in messages if m["role"] == "user")
    assert VALID_RESEARCH_SPEC["topic"] in user_message
    assert VALID_BENCHMARK_SPEC["benchmark_id"] in user_message
    assert VALID_VERIFIER_SPEC["evaluator_hash"] in user_message


async def test_planning_requested_selects_hypothesis_by_id_not_first_in_bank(store):
    """The bank has h1 and h2; selecting h2 must plan against h2's content,
    not h1's (which happens to be first in the bank's array)."""
    llm = FakeLLM([_json(_valid_plan("h2"))])
    handler = ExperimentPlannerAgentHandler(spec_store=store, llm_client=llm)
    client = FakeBusClient()
    handler.register(client)

    session_id = str(uuid.uuid4())
    payload = _seed_planning_inputs(store, session_id, selected_hypothesis_id="h2")

    env = _make_envelope(
        "experiment.planning_requested", session_id, payload, "ResearchOrchestrator"
    )

    await handler.handle_planning_requested(client, env)

    ready = [p for p in client.published if p.event_type == "experiment.plan_ready"]
    assert len(ready) == 1
    plan = store.get(ready[0].payload["experiment_plan_ref"])
    assert plan["hypothesis_id"] == "h2"

    [messages] = llm.calls
    user_message = next(m["content"] for m in messages if m["role"] == "user")
    assert HYP_2["statement"] in user_message
    assert HYP_1["statement"] not in user_message


# ---------------------------------------------------------------------------
# (c) selected_hypothesis_id NOT in the bank -> agent.failed; no
# experiment.plan_ready; LLM NOT called.
# ---------------------------------------------------------------------------


async def test_planning_requested_unknown_hypothesis_id_publishes_agent_failed(store):
    llm = FakeLLM([_json(_valid_plan("h1"))])
    handler = ExperimentPlannerAgentHandler(spec_store=store, llm_client=llm)
    client = FakeBusClient()
    handler.register(client)

    session_id = str(uuid.uuid4())
    payload = _seed_planning_inputs(store, session_id, selected_hypothesis_id="does-not-exist")

    env = _make_envelope(
        "experiment.planning_requested", session_id, payload, "ResearchOrchestrator"
    )

    await handler.handle_planning_requested(client, env)

    failed = [p for p in client.published if p.event_type == "agent.failed"]
    assert len(failed) == 1
    assert failed[0].session_id == session_id
    assert failed[0].causation_id == env.event_id
    assert failed[0].payload["agent_name"] == "ExperimentPlannerAgent"
    assert failed[0].payload["reason_classification"] == "permanent"
    assert failed[0].payload["underlying_reason"]

    assert not [p for p in client.published if p.event_type == "experiment.plan_ready"]
    assert llm.calls == []


# ---------------------------------------------------------------------------
# (d) core failure (malformed LLM response) -> agent.failed with correct
# classification; never a crash.
# ---------------------------------------------------------------------------


async def test_planning_requested_core_error_publishes_agent_failed(store):
    llm = FakeLLM(["not json at all"] * (core.MAX_RETRIES + 1))
    handler = ExperimentPlannerAgentHandler(spec_store=store, llm_client=llm)
    client = FakeBusClient()
    handler.register(client)

    session_id = str(uuid.uuid4())
    payload = _seed_planning_inputs(store, session_id, selected_hypothesis_id="h1")

    env = _make_envelope(
        "experiment.planning_requested", session_id, payload, "ResearchOrchestrator"
    )

    await handler.handle_planning_requested(client, env)

    failed = [p for p in client.published if p.event_type == "agent.failed"]
    assert len(failed) == 1
    assert failed[0].session_id == session_id
    assert failed[0].causation_id == env.event_id
    assert failed[0].payload["agent_name"] == "ExperimentPlannerAgent"
    assert failed[0].payload["reason_classification"] == "permanent"
    assert failed[0].payload["underlying_reason"]

    assert not [p for p in client.published if p.event_type == "experiment.plan_ready"]


# ---------------------------------------------------------------------------
# (e) causation_id set on the success path.
# ---------------------------------------------------------------------------


async def test_planning_requested_causation_id_set(store):
    llm = FakeLLM([_json(_valid_plan("h1"))])
    handler = ExperimentPlannerAgentHandler(spec_store=store, llm_client=llm)
    client = FakeBusClient()
    handler.register(client)

    session_id = str(uuid.uuid4())
    payload = _seed_planning_inputs(store, session_id, selected_hypothesis_id="h1")

    env = _make_envelope(
        "experiment.planning_requested", session_id, payload, "ResearchOrchestrator"
    )

    await handler.handle_planning_requested(client, env)

    ready = [p for p in client.published if p.event_type == "experiment.plan_ready"]
    assert len(ready) == 1
    assert ready[0].causation_id == env.event_id


async def test_live_repo_baseline_skips_unused_general_design(store, monkeypatch):
    """A repository run already has its own evaluator and must not start the
    unrelated self-written GEN design pipeline while planning its method."""
    monkeypatch.setenv("AUTORESEARCH_GEN_EXPERIMENT", "1")
    llm = FakeLLM([_json(_valid_plan("h1"))])
    handler = ExperimentPlannerAgentHandler(spec_store=store, llm_client=llm)
    client = FakeBusClient()

    session_id = str(uuid.uuid4())
    payload = _seed_planning_inputs(store, session_id, selected_hypothesis_id="h1")
    payload["baseline_result_ref"] = store.put(
        kind="baseline_result",
        session_id=session_id,
        key="v1",
        obj={"source_kind": "live_repo", "reproduced": True, "metric_value": 1.0},
        emitter="RunnerAgent",
    )
    env = _make_envelope(
        "experiment.planning_requested", session_id, payload, "ResearchOrchestrator"
    )

    await handler.handle_planning_requested(client, env)

    assert len(llm.calls) == 1
    with pytest.raises(KeyError):
        store.get(f"spec://gen_design/{session_id}/v1")
    assert [p.event_type for p in client.published] == ["experiment.plan_ready"]


async def test_component_planning_uses_same_baseline_scoped_ledger_as_runner(
    store, monkeypatch, tmp_path,
):
    """Planner must not reopen an empty unscoped ledger after Runner made progress."""
    method = {
        "approach": "Apply two ordered improvements.",
        "components": [
            {
                "component_id": component_id,
                "name": component_id,
                "type": "direct_gain",
                "depends_on": dependencies,
                "purpose": f"implement {component_id}",
                "inputs": ["parent checkpoint"],
                "outputs": [f"{component_id} output"],
                "acceptance_requires": dependencies,
                "mechanism": f"mechanism for {component_id}",
                "expected_effect": f"effect for {component_id}",
                "ablation_test": "metric delta > 0",
                "functional_check": "signal >= 0.9",
                "implementation_check": f"runtime evidence matches {component_id}",
                "optimization_space": f"improve {component_id}",
                "downstream_contract": f"provide {component_id} output",
            }
            for component_id, dependencies in (("first", []), ("second", ["first"]))
        ],
    }
    baseline = {"source_kind": "live_repo", "reproduced": True, "metric_value": 1.0}
    plan = build_component_plan(method)
    monkeypatch.setenv("AUTORESEARCH_WEB_WORKDIR", str(tmp_path))
    session_id = str(uuid.uuid4())
    workroot = tmp_path / "repo_improve_local" / session_id

    # This stale ledger reproduces the planner-side file that previously stayed all-pending.
    ledger_for_workroot(str(workroot), plan)
    runner_ledger = ledger_for_workroot(
        str(workroot), plan, execution_identity=artifact_digest(baseline))
    runner_ledger.begin("first")
    runner_ledger.record_attempt(
        "first", execution_succeeded=True, contract_passed=True,
        checkpoint="checkpoint-first", max_attempts=1)
    assert runner_ledger.next_ready().component_id == "second"

    research_spec_ref = store.put(
        kind="research_spec", session_id=session_id, key="final",
        obj=VALID_RESEARCH_SPEC, emitter="ProblemModelingAgent")
    benchmark_spec_ref = store.put(
        kind="benchmark_spec", session_id=session_id, key="v1",
        obj=VALID_BENCHMARK_SPEC, emitter="EvaluatorAgent")
    method_ref = store.put(
        kind="method", session_id=session_id, key="final",
        obj=method, emitter="MethodAgent")
    baseline_ref = store.put(
        kind="baseline_result", session_id=session_id, key="v1",
        obj=baseline, emitter="RunnerAgent")
    llm = FakeLLM([
        _json(_valid_plan(str(method_ref))),
        _json(_valid_plan(
            str(method_ref),
            change_summary="Implement the current second component in its first dependency context.",
        )),
    ])
    handler = ExperimentPlannerAgentHandler(spec_store=store, llm_client=llm)
    client = FakeBusClient()
    env = _make_envelope(
        "experiment.planning_requested", session_id,
        {
            "method_ref": method_ref,
            "hypothesis_bank_ref": None,
            "selected_hypothesis_id": None,
            "verifier_spec_ref": None,
            "benchmark_spec_ref": benchmark_spec_ref,
            "research_spec_ref": research_spec_ref,
            "baseline_result_ref": baseline_ref,
            "spec_interpretation_ref": None,
            "research_memory_ref": None,
            "plan_review_ref": None,
        },
        "ResearchOrchestrator",
    )

    await handler.handle_planning_requested(client, env)

    assert len(llm.calls) == 2
    prompt = next(
        message["content"] for message in llm.calls[0]
        if message["role"] == "user" and "Selected approach to implement" in message["content"]
    )
    assert '"component_id": "second"' in prompt
    assert '"component_id": "first"' in prompt  # dependency context remains visible
    assert '"current_component"' in prompt
    assert "current frozen component_id" in llm.calls[1][-1]["content"]


async def test_replanning_reads_previous_plan_review_from_store(store):
    llm = FakeLLM([_json(_valid_plan("h1"))])
    handler = ExperimentPlannerAgentHandler(spec_store=store, llm_client=llm)
    client = FakeBusClient()
    session_id = str(uuid.uuid4())
    payload = _seed_planning_inputs(store, session_id, selected_hypothesis_id="h1")
    payload["plan_review_ref"] = store.put(
        kind="critique_report",
        session_id=session_id,
        key="plan-rejected",
        obj={"target": "experiment_plan", "verdict": "fail",
             "issues": ["Remove the rejected measurement shortcut."]},
        emitter="CriticAgent",
    )
    env = _make_envelope(
        "experiment.planning_requested", session_id, payload, "ResearchOrchestrator"
    )

    await handler.handle_planning_requested(client, env)

    prompt = llm.calls[0][-1]["content"]
    assert "Remove the rejected measurement shortcut." in prompt


@pytest.mark.parametrize("component_plan", [False, True])
async def test_planning_reads_bound_method_feedback_from_store(
    store, monkeypatch, tmp_path, component_plan,
):
    session_id = str(uuid.uuid4())
    monkeypatch.setenv("AUTORESEARCH_WEB_WORKDIR", str(tmp_path))
    current_outline = ["Use the corrected threshold 0.6; defer downstream validation."]
    method = {"approach": "Keep the confirmed scientific treatment.",
              "changes_outline": current_outline}
    if component_plan:
        method["components"] = [{
            "component_id": "first", "name": "first", "type": "direct_gain",
            "depends_on": [], "acceptance_requires": [],
            "purpose": "implement first", "inputs": ["parent checkpoint"],
            "outputs": ["first output"], "mechanism": "first mechanism",
            "expected_effect": "improve first", "ablation_test": "metric delta > 0",
            "implementation_check": "runtime evidence matches first",
            "optimization_space": "improve first", "downstream_contract": "provide first output",
        }]
    method_ref = store.put(
        kind="method", session_id=session_id, key="revised",
        obj=method,
        emitter="MethodAgent",
    )
    feedback_ref = store.put(
        kind="method_improvement_review", session_id=session_id, key="slot-1",
        obj={
            "status": "completed",
            "trigger": "implementation_failure",
            "reason": "The prior child workspace lost required prepared inputs.",
            "decision": {
                "route": "revise_method",
                "modify": ["Bootstrap required inputs before component execution."],
                "add": ["Verify the prepared inputs are non-empty."],
                "validation_plan": "Re-run after bootstrap using the old threshold 0.8 and downstream gates.",
            },
        },
        emitter="MemoryAgent",
    )
    research_spec_ref = store.put(
        kind="research_spec", session_id=session_id, key="final",
        obj=VALID_RESEARCH_SPEC, emitter="ProblemModelingAgent",
    )
    benchmark_spec_ref = store.put(
        kind="benchmark_spec", session_id=session_id, key="v1",
        obj=VALID_BENCHMARK_SPEC, emitter="EvaluatorAgent",
    )
    baseline_ref = store.put(
        kind="baseline_result", session_id=session_id, key="v1",
        obj={"metric_value": 1.0}, emitter="RunnerAgent",
    )
    llm = FakeLLM([_json(_valid_plan(
        str(method_ref), change_summary="Implement first component: preserve current requirements."))])
    handler = ExperimentPlannerAgentHandler(spec_store=store, llm_client=llm)
    client = FakeBusClient()
    await handler.handle_planning_requested(client, _make_envelope(
        "experiment.planning_requested", session_id,
        {
            "method_ref": method_ref,
            "method_feedback_ref": feedback_ref,
            "hypothesis_bank_ref": None,
            "selected_hypothesis_id": None,
            "verifier_spec_ref": None,
            "benchmark_spec_ref": benchmark_spec_ref,
            "research_spec_ref": research_spec_ref,
            "baseline_result_ref": baseline_ref,
            "spec_interpretation_ref": None,
            "research_memory_ref": None,
            "plan_review_ref": None,
        },
        "ResearchOrchestrator",
    ))

    prompt = llm.calls[0][-1]["content"]
    assert "Bootstrap required inputs before component execution." in prompt
    assert "Verify the prepared inputs are non-empty." in prompt
    assert '"current_method_changes_outline"' in prompt
    assert current_outline[0] in prompt
    assert "current frozen Method and the rejecting plan review take precedence" in prompt
    assert "Do not import later components" in prompt
    assert "applied_method_feedback is binding" not in prompt


# ---------------------------------------------------------------------------
# register() wires the subscription.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# MOSS prompt_patch: config_root/agent_config/ExperimentPlannerAgent/
# prompt_patch.md gets read by self.config.prompt_patch() and injected into
# the system message.
# ---------------------------------------------------------------------------


async def test_planning_requested_prompt_patch_from_config_root_injected_into_system_message(
    store, tmp_path
):
    patch_dir = tmp_path / "agent_config" / "ExperimentPlannerAgent"
    patch_dir.mkdir(parents=True)
    (patch_dir / "prompt_patch.md").write_text("STEER: prefer X", encoding="utf-8")

    llm = FakeLLM([_json(_valid_plan("h1"))])
    handler = ExperimentPlannerAgentHandler(spec_store=store, llm_client=llm, config_root=tmp_path)
    client = FakeBusClient()
    handler.register(client)

    session_id = str(uuid.uuid4())
    payload = _seed_planning_inputs(store, session_id, selected_hypothesis_id="h1")

    env = _make_envelope(
        "experiment.planning_requested", session_id, payload, "ResearchOrchestrator"
    )

    await handler.handle_planning_requested(client, env)

    [messages] = llm.calls
    system_message = next(m["content"] for m in messages if m["role"] == "system")
    assert "STEER: prefer X" in system_message
    assert "[MOSS-injected guidance]" in system_message


def test_register_wires_subscription(store):
    llm = FakeLLM([])
    handler = ExperimentPlannerAgentHandler(spec_store=store, llm_client=llm)
    client = FakeBusClient()
    handler.register(client)

    patterns = [pattern for pattern, _ in client.subscriptions]
    assert "experiment.planning_requested" in patterns


async def test_exact_reconciliation_plans_full_validation_without_reimplementing_components(
    store, monkeypatch, tmp_path,
):
    from autoresearch.experiments.repo_improve.tests.test_component_plan import _component
    monkeypatch.setenv("AUTORESEARCH_WEB_WORKDIR", str(tmp_path))
    sid = str(uuid.uuid4())
    payload = _seed_planning_inputs(store, sid, selected_hypothesis_id="h1")
    retained = {"checkpoint": "checkpoint-retained", "off_method_observation": {"valid": True}}
    retained_ref = store.put(kind="gen_experiment_result", session_id=sid,
        key="snapshot-" + artifact_digest(retained).split(":", 1)[1], obj=retained, emitter="RunnerAgent")
    method = {"approach": "Adopt the exact measured implementation.",
              "components": [_component("retained")], "reconciled_result_ref": retained_ref}
    method_ref = store.put(kind="method", session_id=sid, key="final", obj=method, emitter="MethodAgent")
    baseline = {"source_kind": "live_repo", "reproduced": True, "metric_value": 2.0}
    baseline_ref = store.put(kind="baseline_result", session_id=sid, key="v1", obj=baseline, emitter="RunnerAgent")
    payload.update(method_ref=method_ref, hypothesis_bank_ref=None, selected_hypothesis_id=None,
                   baseline_result_ref=baseline_ref)
    llm = FakeLLM([_json(_valid_plan(str(method_ref)))])
    handler = ExperimentPlannerAgentHandler(spec_store=store, llm_client=llm)
    client = FakeBusClient()
    await handler.handle_planning_requested(client, _make_envelope(
        "experiment.planning_requested", sid, payload, "ResearchOrchestrator"))
    assert llm.calls == []
    assert [p.event_type for p in client.published] == ["experiment.plan_ready"]
    experiment_plan = store.get(client.published[0].payload["experiment_plan_ref"])
    assert experiment_plan["changes"] == []
    assert retained_ref in experiment_plan["change_summary"]
    assert "official evaluator" in experiment_plan["success_criteria"]
    ledger = ledger_for_workroot(str(tmp_path / "repo_improve_local" / sid),
        build_component_plan(method), execution_identity=artifact_digest(baseline))
    assert not ledger.all_accepted()
    assert ledger.entry("retained").attempts == 0
