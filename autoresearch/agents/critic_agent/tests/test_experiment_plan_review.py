"""
Tests for CriticAgent's third critique capability: ReviewingExperimentPlan.
CriticAgent 第三种批判能力的测试：ReviewingExperimentPlan。

Covers:
  (a) core.critique_experiment_plan: a plan whose changes[] are all within
      verifier_spec.editable_region -> verdict is whatever the LLM returns
      (both pass and fail canned responses), schema-valid CritiqueReport
      target=experiment_plan.
  (b) core.critique_experiment_plan: a plan with a changes[].file OUTSIDE
      editable_region (or inside forbidden_files) -> verdict FORCED to "fail"
      even when the LLM's canned response says "pass" -- the editable_region
      GATE, enforced in code, independent of the LLM.
  (c) handler.handle_experiment_plan_review: seeded ExperimentPlan +
      VerifierSpec -> experiment.plan_reviewed with verdict + a ref resolving
      to a schema-valid CritiqueReport.
  (d) provenance: both ExperimentPlan and VerifierSpec are read from the
      store, never from the event payload.
  (e) register() now wires THREE subscriptions (this file only asserts the
      third; the existing two are covered/regression-tested by
      test_handler.py, which must stay green).

No EventBus / Store / Redis / network for the core tests -- only a FakeLLM
(canned JSON queue). Handler tests use a FakeBusClient (mirrors the real
EventBusClient's publish/subscribe signature; does NOT auto-deliver events)
plus a real tmp SpecStore.
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

from autoresearch.agents.critic_agent import core
from autoresearch.agents.critic_agent.core import _load_experiment_plan_system_prompt
from autoresearch.agents.critic_agent.handler import CriticAgentHandler
from autoresearch.agents.critic_agent.tests.conftest import FakeLLM
from autoresearch.spec_store import SpecStore

_CONTRACTS_DIR = Path(__file__).resolve().parents[1] / "data-contracts"


def _validator(name: str) -> Draft202012Validator:
    schema = yaml.safe_load((_CONTRACTS_DIR / name).read_text(encoding="utf-8"))
    return Draft202012Validator(schema)


CRITIQUE_REPORT_VALIDATOR = _validator("critique_report.schema.yaml")


def _json(obj: dict) -> str:
    return json.dumps(obj)


def _validate_report(report: dict) -> None:
    errors = list(CRITIQUE_REPORT_VALIDATOR.iter_errors(report))
    assert errors == [], [e.message for e in errors]


VERIFIER_SPEC = {
    "evaluator_hash": "sha256:deadbeef",
    "metric": "accuracy",
    "data_split": {"in_loop": "dev", "held_out": "test"},
    "run_budget": "10 runs, 1h wall-clock",
    "seed_policy": "fixed seed 42 across runs",
    "editable_region": ["src/model.py", "src/retriever.py"],
    "forbidden_files": ["eval/harness.py"],
    "result_parse_schema": {"type": "object"},
}

EXPERIMENT_PLAN_IN_SCOPE = {
    "hypothesis_id": "h1",
    "change_summary": "Add hybrid sparse+dense retrieval.",
    "changes": [
        {"file": "src/retriever.py", "description": "Add a sparse+dense hybrid retriever."},
    ],
    "expected_outcome": "Higher long-tail QA accuracy vs. the dense-only baseline.",
}

EXPERIMENT_PLAN_OUT_OF_SCOPE = {
    "hypothesis_id": "h1",
    "change_summary": "Add hybrid sparse+dense retrieval, also patch the harness.",
    "changes": [
        {"file": "src/retriever.py", "description": "Add a sparse+dense hybrid retriever."},
        {"file": "eval/harness.py", "description": "Relax the evaluator's timeout."},
    ],
    "expected_outcome": "Higher long-tail QA accuracy vs. the dense-only baseline.",
}

VALID_EXPERIMENT_PLAN_REPORT_PASS = {
    "target": "experiment_plan",
    "verdict": "pass",
    "issues": [],
    "passing_hypothesis_ids": [],
}

VALID_EXPERIMENT_PLAN_REPORT_FAIL = {
    "target": "experiment_plan",
    "verdict": "fail",
    "issues": ["Budget is unrealistic for the proposed ablations."],
    "passing_hypothesis_ids": [],
}


# ---------------------------------------------------------------------------
# (a) core.critique_experiment_plan: in-scope plan -> verdict passthrough
# from the LLM (both pass and fail canned responses).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "canned_report",
    [VALID_EXPERIMENT_PLAN_REPORT_PASS, VALID_EXPERIMENT_PLAN_REPORT_FAIL],
    ids=["pass", "fail"],
)
def test_critique_experiment_plan_in_scope_passes_through_llm_verdict(canned_report):
    llm = FakeLLM([_json(canned_report)])

    result = core.critique_experiment_plan(EXPERIMENT_PLAN_IN_SCOPE, VERIFIER_SPEC, llm)

    _validate_report(result)
    assert result["target"] == "experiment_plan"
    assert result["verdict"] == canned_report["verdict"]
    assert len(llm.calls) == 1


@pytest.mark.parametrize("plan, verdict", [
    (EXPERIMENT_PLAN_IN_SCOPE, "pass"), (EXPERIMENT_PLAN_OUT_OF_SCOPE, "fail")])
@pytest.mark.parametrize("conclusion", [
    "Therefore, the verdict is pass.",
    "No issue found that blocks execution. The verdict should be 'pass'.",
    'No blocking defect. The verdict should be "pass".',
])
def test_explicit_contradictory_verdict_is_repaired_without_relaxing_scope(plan, verdict, conclusion):
    inconsistent = dict(VALID_EXPERIMENT_PLAN_REPORT_FAIL,
                        issues=[conclusion])
    llm = FakeLLM([_json(inconsistent), _json(VALID_EXPERIMENT_PLAN_REPORT_PASS)])
    result = core.critique_experiment_plan(plan, VERIFIER_SPEC, llm)
    assert result["verdict"] == verdict
    assert len(llm.calls) == 2
    assert any("contradicts an explicit verdict" in message["content"]
               for message in llm.calls[-1])


@pytest.mark.parametrize("conclusion", [
    "Therefore, the verdict is pass.",
    "No issue found that blocks execution. The verdict should be 'pass'.",
])
def test_repeated_contradictory_verdict_remains_protocol_failure(conclusion):
    inconsistent = dict(VALID_EXPERIMENT_PLAN_REPORT_FAIL,
                        issues=[conclusion])
    llm = FakeLLM([_json(inconsistent)] * (core.MAX_RETRIES + 1))
    with pytest.raises(core.CriticParseError, match="contradicts an explicit verdict"):
        core.critique_experiment_plan(EXPERIMENT_PLAN_IN_SCOPE, VERIFIER_SPEC, llm)
    assert len(llm.calls) == core.MAX_RETRIES + 1


def test_critique_empty_verifier_uses_benchmark_and_permissive_scope():
    llm = FakeLLM([_json(VALID_EXPERIMENT_PLAN_REPORT_PASS)])
    benchmark_spec = {
        "metric": "val_bpb",
        "resource_requirements": "one GPU, 300 seconds pure training",
        "evaluation_semantics": {"comparison_mode": "quality_at_fixed_time"},
    }

    result = core.critique_experiment_plan(
        EXPERIMENT_PLAN_IN_SCOPE, {}, llm, benchmark_spec=benchmark_spec)

    assert result["verdict"] == "pass"
    [messages] = llm.calls
    prompt = "\n".join(str(message["content"]) for message in messages)
    assert "val_bpb" in prompt
    assert "empty editable_region means no predeclared file boundary" in prompt
    assert "whole repository is provisionally editable" in prompt


def test_critique_plan_prompt_includes_frozen_baseline_runtime_evidence():
    llm = FakeLLM([_json(VALID_EXPERIMENT_PLAN_REPORT_PASS)])
    baseline_result = {
        "metric_name": "pure_train_time_seconds",
        "metric_value": 196.682,
        "fixed_config": "Muon + AdamW hybrid; bf16; manual all-reduce",
    }

    core.critique_experiment_plan(
        EXPERIMENT_PLAN_IN_SCOPE,
        VERIFIER_SPEC,
        llm,
        baseline_result=baseline_result,
    )

    [messages] = llm.calls
    prompt = "\n".join(str(message["content"]) for message in messages)
    assert "Frozen BaselineResult" in prompt
    assert "196.682" in prompt
    assert "Muon + AdamW hybrid" in prompt


def test_review_binds_current_method_without_requalifying_historical_attempts():
    from autoresearch.experiments.repo_improve.component_plan import method_fingerprint

    method = {"approach": "revised mechanism", "components": [{"component_id": "component-a"}]}
    historical = {
        "component_id": "component-a", "method_fingerprint": "sha256:previous-method",
        "status": "drifted", "metric": 78.0, "contract_passed": True,
        "failure_reason": "violates previous frozen mechanism",
    }
    llm = FakeLLM([_json(VALID_EXPERIMENT_PLAN_REPORT_PASS)])
    core.critique_experiment_plan(
        EXPERIMENT_PLAN_IN_SCOPE, VERIFIER_SPEC, llm, method=method,
        research_memory={"component_history": [historical]},
    )
    [messages] = llm.calls
    user = messages[1]["content"]
    progress = json.JSONDecoder().raw_decode(user.split("Research progress (JSON; stored observations and qualification, not a new task contract):\n", 1)[1])[0]
    assert progress["current_method_fingerprint"] == method_fingerprint(method)
    assert progress["component_history"] == [historical]


def test_review_keeps_component_parent_comparison_evidence():
    historical = {
        "component_id": "component-a", "metric": 0.56,
        "parent_checkpoint": "baseline", "parent_checkpoints": [],
        "acceptance_result": {
            "passed": True, "parent_metric": 0.53, "candidate_metric": 0.56,
            "parent_evidence": {"ref": "attempt-parent", "host_bound": True},
        },
    }
    llm = FakeLLM([_json(VALID_EXPERIMENT_PLAN_REPORT_PASS)])
    core.critique_experiment_plan(
        EXPERIMENT_PLAN_IN_SCOPE, VERIFIER_SPEC, llm,
        research_memory={"component_history": [historical]},
    )
    user = llm.calls[0][1]["content"]
    progress = json.JSONDecoder().raw_decode(user.split(
        "Research progress (JSON; stored observations and qualification, not a new task contract):\n",
        1)[1])[0]
    assert progress["component_history"] == [historical]


def test_experiment_plan_prompt_counts_budgeted_model_updates_as_training_compute():
    _load_experiment_plan_system_prompt.cache_clear()
    prompt = _load_experiment_plan_system_prompt().lower()

    assert "executed work and the frozen timer" in prompt
    assert "forward/backward/optimizer-update steps" in prompt
    assert "charged to that same training budget" in prompt
    assert "phase label" in prompt


def test_experiment_plan_prompt_rejects_benchmark_semantic_substitution():
    _load_experiment_plan_system_prompt.cache_clear()
    prompt = _load_experiment_plan_system_prompt().lower()

    assert "task-semantic consistency" in prompt
    assert "dataset, tokenizer, evaluator, model scale" in prompt
    assert "must fail" in prompt


def test_experiment_plan_prompt_respects_component_acceptance_contracts():
    _load_experiment_plan_system_prompt.cache_clear()
    prompt = " ".join(_load_experiment_plan_system_prompt().lower().split())

    assert "indirect-gain component may pass" in prompt
    assert "without immediately improving the final benchmark metric" in prompt
    assert "direct-gain component must improve" in prompt
    assert "follow-up improve plan" in prompt


def test_experiment_plan_review_does_not_reopen_novelty_or_invent_constraints():
    _load_experiment_plan_system_prompt.cache_clear()
    llm = FakeLLM([_json(VALID_EXPERIMENT_PLAN_REPORT_PASS)])
    core.critique_experiment_plan(
        EXPERIMENT_PLAN_IN_SCOPE, {}, llm,
        benchmark_spec={"method_freedom": ["model scale", "training algorithm"]},
    )
    [messages] = llm.calls
    prompt = " ".join(messages[0]["content"].lower().split())
    assert "this is not a second novelty audit" in prompt
    assert "explicitly permitted treatment factor" in prompt
    assert "weak, conventional, or primarily regularization" in prompt
    assert "identify the authoritative rule" in prompt
    assert "model scale" in messages[1]["content"]


# ---------------------------------------------------------------------------
# (b) THE GATE: a changes[].file outside editable_region (or in
# forbidden_files) forces verdict="fail" even when the LLM said "pass", and
# issues names the offending file.
# ---------------------------------------------------------------------------


def test_critique_experiment_plan_out_of_scope_change_forces_fail_gate():
    llm = FakeLLM([_json(VALID_EXPERIMENT_PLAN_REPORT_PASS)])

    result = core.critique_experiment_plan(EXPERIMENT_PLAN_OUT_OF_SCOPE, VERIFIER_SPEC, llm)

    _validate_report(result)
    assert result["target"] == "experiment_plan"
    assert result["verdict"] == "fail"
    assert any("eval/harness.py" in issue for issue in result["issues"])
    # the gate does not retry the LLM -- it overrides the already-valid response
    assert len(llm.calls) == 1


def test_critique_experiment_plan_file_not_in_editable_region_forces_fail_gate():
    plan = {
        "hypothesis_id": "h1",
        "change_summary": "Touch an unlisted file.",
        "changes": [
            {"file": "src/unlisted_module.py", "description": "Some change."},
        ],
        "expected_outcome": "Whatever.",
    }
    llm = FakeLLM([_json(VALID_EXPERIMENT_PLAN_REPORT_PASS)])

    result = core.critique_experiment_plan(plan, VERIFIER_SPEC, llm)

    assert result["verdict"] == "fail"
    assert any("src/unlisted_module.py" in issue for issue in result["issues"])


@pytest.mark.parametrize("file_path, expected", [
    ("ml_tasks/Robustness_openood/algorithm.py", "pass"),
    ("workspace/Robustness_openood/OpenOOD/algorithm.py", "pass"),
    ("ml_tasks/Robustness_openood/train_eval_baseline.py", "fail"),
])
def test_live_repository_target_files_gate_without_verifier_spec(file_path, expected):
    baseline = {
        "source_kind": "live_repo",
        "repository_contract": {
            "capture_complete": True,
            "sources": [{
                "path": "ml_tasks/Robustness_openood/config.json",
                "content": json.dumps({
                    "repo_dir": "workspace/Robustness_openood/OpenOOD",
                    "target_files": ["algorithm.py", "split_config.json"],
                }),
            }],
        },
    }
    plan = {**EXPERIMENT_PLAN_IN_SCOPE,
            "changes": [{"file": file_path, "description": "Diagnostic change."}]}
    llm = FakeLLM([_json(VALID_EXPERIMENT_PLAN_REPORT_PASS)])

    result = core.critique_experiment_plan(
        plan, {}, llm, baseline_result=baseline)

    assert result["verdict"] == expected
    if expected == "fail":
        assert any(file_path in issue for issue in result["issues"])


def test_critique_experiment_plan_malformed_json_retries_then_raises_typed_error():
    llm = FakeLLM(["not json at all"] * (core.MAX_RETRIES + 1))

    with pytest.raises(core.CriticParseError):
        core.critique_experiment_plan(EXPERIMENT_PLAN_IN_SCOPE, VERIFIER_SPEC, llm)

    assert len(llm.calls) == core.MAX_RETRIES + 1


def test_critique_experiment_plan_invalid_verdict_enum_retries_then_raises_typed_error():
    bad_report = {
        "target": "experiment_plan",
        "verdict": "maybe",
        "issues": [],
        "passing_hypothesis_ids": [],
    }
    llm = FakeLLM([_json(bad_report)] * (core.MAX_RETRIES + 1))

    with pytest.raises(core.CriticParseError):
        core.critique_experiment_plan(EXPERIMENT_PLAN_IN_SCOPE, VERIFIER_SPEC, llm)

    assert len(llm.calls) == core.MAX_RETRIES + 1


# ---------------------------------------------------------------------------
# Handler test doubles (self-contained; mirrors test_handler.py's
# FakeBusClient/_Envelope shape).
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
        self.emitter_name = "CriticAgent"
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


def _seed_experiment_plan_review_inputs(
    store: SpecStore, session_id: str, experiment_plan: dict
) -> dict[str, str]:
    """Seed the store with the ExperimentPlan + VerifierSpec CriticAgent
    reads for ReviewingExperimentPlan, as upstream agents would; return their
    refs."""
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
        obj=VERIFIER_SPEC,
        emitter="EvaluatorAgent",
    )
    return {
        "experiment_plan_ref": experiment_plan_ref,
        "verifier_spec_ref": verifier_spec_ref,
    }


# ---------------------------------------------------------------------------
# (c) experiment.review_requested -> experiment.plan_reviewed, ref resolves
# to a schema-valid CritiqueReport (target=experiment_plan).
# ---------------------------------------------------------------------------


async def test_experiment_review_requested_publishes_plan_reviewed_with_resolvable_ref(store):
    llm = FakeLLM([_json(VALID_EXPERIMENT_PLAN_REPORT_PASS)])
    handler = CriticAgentHandler(spec_store=store, llm_client=llm)
    client = FakeBusClient()
    handler.register(client)

    session_id = str(uuid.uuid4())
    refs = _seed_experiment_plan_review_inputs(store, session_id, EXPERIMENT_PLAN_IN_SCOPE)

    env = _make_envelope(
        "experiment.review_requested", session_id, refs, "ResearchOrchestrator"
    )

    await handler.handle_experiment_plan_review(client, env)

    reviewed = [p for p in client.published if p.event_type == "experiment.plan_reviewed"]
    assert len(reviewed) == 1
    payload = reviewed[0].payload
    assert set(payload.keys()) == {"verdict", "review_report_ref"}
    assert payload["verdict"] == "pass"

    report = store.get(payload["review_report_ref"])
    _validate_report(report)
    assert report["target"] == "experiment_plan"
    assert report["verdict"] == "pass"


@pytest.mark.parametrize("canned_report", [
    VALID_EXPERIMENT_PLAN_REPORT_PASS, VALID_EXPERIMENT_PLAN_REPORT_FAIL,
])
async def test_review_reads_frozen_component_contract_without_overriding_verdict(
        store, monkeypatch, canned_report):
    monkeypatch.delenv("AUTORESEARCH_GEN_EXPERIMENT", raising=False)
    llm = FakeLLM([_json(canned_report)])
    handler = CriticAgentHandler(spec_store=store, llm_client=llm)
    client = FakeBusClient()
    sid = str(uuid.uuid4())
    refs = _seed_experiment_plan_review_inputs(store, sid, EXPERIMENT_PLAN_IN_SCOPE)
    method = {
        "approach": "Condition features before downstream scoring",
        "components": [{
            "id": "h1", "component_type": "indirect_gain",
            "depends_on": ["features"],
            "acceptance": "orthogonality residual below declared tolerance",
        }],
    }
    refs["method_ref"] = store.put(
        kind="method", session_id=sid, key="confirmed", obj=method,
        emitter="MethodAgent")
    await handler.handle_experiment_plan_review(client, _make_envelope(
        "experiment.review_requested", sid, refs, "ResearchOrchestrator"))

    reviewed = [p for p in client.published if p.event_type == "experiment.plan_reviewed"]
    assert len(reviewed) == 1
    assert reviewed[0].payload["verdict"] == canned_report["verdict"]
    messages = llm.calls[0]
    user_prompt = messages[1]["content"]
    assert json.dumps(method, ensure_ascii=False, indent=2) in user_prompt
    assert "Do not invent a component match" in messages[0]["content"]
    assert "forbidden file edits" in messages[0]["content"]


async def test_review_reads_bound_method_feedback_from_store(store, monkeypatch):
    from autoresearch.experiments.repo_improve.component_plan import method_fingerprint

    monkeypatch.delenv("AUTORESEARCH_GEN_EXPERIMENT", raising=False)
    llm = FakeLLM([_json(VALID_EXPERIMENT_PLAN_REPORT_PASS)])
    handler = CriticAgentHandler(spec_store=store, llm_client=llm)
    client = FakeBusClient()
    sid = str(uuid.uuid4())
    refs = _seed_experiment_plan_review_inputs(store, sid, EXPERIMENT_PLAN_IN_SCOPE)
    method = {"approach": "Keep the confirmed scientific treatment."}
    refs["method_ref"] = store.put(
        kind="method", session_id=sid, key="revised",
        obj=method,
        emitter="MethodAgent",
    )
    refs["method_feedback_ref"] = store.put(
        kind="method_improvement_review", session_id=sid, key="slot-1",
        obj={
            "status": "completed",
            "method_ref": "spec://method/previous",
            "source_method_ref": "spec://method/previous",
            "trigger": "implementation_failure",
            "reason": "The prior child workspace lost required prepared inputs.",
            "decision": {
                "route": "revise_method",
                "modify": ["Bootstrap required inputs before component execution."],
                "validation_plan": "Verify the prepared inputs, then rerun.",
            },
        },
        emitter="MemoryAgent",
    )

    await handler.handle_experiment_plan_review(client, _make_envelope(
        "experiment.review_requested", sid, refs, "ResearchOrchestrator"))

    prompt = llm.calls[0][1]["content"]
    assert "Bootstrap required inputs before component execution." in prompt
    assert "Verify the prepared inputs, then rerun." in prompt
    decoder = json.JSONDecoder()
    confirmed = decoder.raw_decode(prompt.split(
        "Confirmed Method (JSON; frozen component designs, dependencies and acceptance contracts):\n",
        1)[1])[0]
    progress = decoder.raw_decode(prompt.split(
        "Research progress (JSON; stored observations and qualification, not a new task contract):\n",
        1)[1])[0]
    assert confirmed == method
    assert progress["current_method_fingerprint"] == method_fingerprint(method)
    assert '"source_method_ref": "spec://method/previous"' in prompt
    assert "not proof that its recommendations were applied" in llm.calls[0][0]["content"]


@pytest.mark.parametrize("canned_report", [
    VALID_EXPERIMENT_PLAN_REPORT_PASS, VALID_EXPERIMENT_PLAN_REPORT_FAIL,
])
async def test_review_reads_component_progress_from_bound_memory(store, monkeypatch, canned_report):
    monkeypatch.delenv("AUTORESEARCH_GEN_EXPERIMENT", raising=False)
    llm = FakeLLM([_json(canned_report)])
    handler = CriticAgentHandler(spec_store=store, llm_client=llm)
    client = FakeBusClient()
    sid = str(uuid.uuid4())
    refs = _seed_experiment_plan_review_inputs(store, sid, EXPERIMENT_PLAN_IN_SCOPE)
    memory = {
        "best_metric": 0.49,
        "best_result_ref": "spec://gen_experiment_result/retained",
        "component_history": [{
            "sequence": 2, "component_id": "h1", "status": "qualified",
            "metric": 0.60, "contract_passed": True, "mir_verdict": "faithful",
            "verification_verdict": "verified", "checkpoint": "checkpoint-a11111",
            "run_ref": "spec://run_record/measured",
            "changes_summary": "Changed lookahead from 8 to 16; this variant was measured.",
        }, {
            "sequence": 3, "component_id": "h1", "status": "invalid",
            "metric": 0.40, "contract_passed": False, "mir_verdict": "drifted",
            "failure_reason": "The candidate changed a forbidden evaluator file.",
        }],
    }
    refs["research_memory_ref"] = store.put(
        kind="research_memory", session_id=sid, key="bound", obj=memory,
        emitter="MemoryAgent")
    refs["research_memory"] = {"component_history": [{"changes_summary": "untrusted payload"}]}

    await handler.handle_experiment_plan_review(client, _make_envelope(
        "experiment.review_requested", sid, refs, "ResearchOrchestrator"))

    user_prompt = llm.calls[0][1]["content"]
    assert "Changed lookahead from 8 to 16" in user_prompt
    assert "spec://run_record/measured" in user_prompt
    assert '"contract_passed": false' in user_prompt
    assert '"mir_verdict": "drifted"' in user_prompt
    assert "untrusted payload" not in user_prompt
    reviewed = [p for p in client.published if p.event_type == "experiment.plan_reviewed"]
    assert reviewed[0].payload["verdict"] == canned_report["verdict"]


async def test_gen_mode_auto_passes_without_calling_llm(store, monkeypatch):
    """In GEN mode the state-machine plan is unused, so the review auto-passes
    (verdict=pass) WITHOUT the LLM -- a FakeLLM with an empty queue would raise if
    called, so a clean pass proves the LLM path was skipped."""
    monkeypatch.setenv("AUTORESEARCH_GEN_EXPERIMENT", "1")
    llm = FakeLLM([])  # empty -> raises on .chat if the handler ever calls it
    handler = CriticAgentHandler(spec_store=store, llm_client=llm)
    client = FakeBusClient()
    handler.register(client)

    session_id = str(uuid.uuid4())
    refs = _seed_experiment_plan_review_inputs(store, session_id, EXPERIMENT_PLAN_IN_SCOPE)
    env = _make_envelope(
        "experiment.review_requested", session_id, refs, "ResearchOrchestrator"
    )

    await handler.handle_experiment_plan_review(client, env)

    reviewed = [p for p in client.published if p.event_type == "experiment.plan_reviewed"]
    assert len(reviewed) == 1
    assert reviewed[0].payload["verdict"] == "pass"
    assert not [p for p in client.published if p.event_type == "agent.failed"]
    report = store.get(reviewed[0].payload["review_report_ref"])
    _validate_report(report)
    assert report["target"] == "experiment_plan" and report["verdict"] == "pass"


async def test_survey_validation_is_auto_reviewed_even_when_human_gen_review_is_on(
        store, monkeypatch):
    monkeypatch.setenv("AUTORESEARCH_GEN_EXPERIMENT", "1")
    monkeypatch.setenv("AUTORESEARCH_GEN_REVIEW_DESIGN", "1")
    plan = {
        "hypothesis_id": "validation-study",
        "change_summary": "Run the frozen survey study.",
        "changes": [{"file": "validation_study.py", "description": "Run analysis."}],
        "expected_outcome": "supports, contradicts, or inconclusive",
        "success_criteria": "Preserve sample IDs, retrieval date, and artifacts.",
        "task_class": "survey_validation",
    }
    handler = CriticAgentHandler(spec_store=store, llm_client=FakeLLM([]))
    client = FakeBusClient()
    session_id = str(uuid.uuid4())
    refs = _seed_experiment_plan_review_inputs(store, session_id, plan)

    await handler.handle_experiment_plan_review(
        client,
        _make_envelope(
            "experiment.review_requested", session_id, refs, "ResearchOrchestrator"),
    )

    reviewed = [p for p in client.published if p.event_type == "experiment.plan_reviewed"]
    assert len(reviewed) == 1
    assert reviewed[0].payload["verdict"] == "pass"


@pytest.mark.parametrize("human_review", ["0", "1"])
async def test_live_repo_plan_keeps_critic_review_in_gen_mode(
    store, monkeypatch, human_review
):
    """GEN's self-written design shortcut is unrelated to a live repository;
    repository plans still need the CriticAgent in both review settings."""
    monkeypatch.setenv("AUTORESEARCH_GEN_EXPERIMENT", "1")
    monkeypatch.setenv("AUTORESEARCH_GEN_REVIEW_DESIGN", human_review)
    llm = FakeLLM([_json(VALID_EXPERIMENT_PLAN_REPORT_FAIL)])
    handler = CriticAgentHandler(spec_store=store, llm_client=llm)
    client = FakeBusClient()

    session_id = str(uuid.uuid4())
    refs = _seed_experiment_plan_review_inputs(store, session_id, EXPERIMENT_PLAN_IN_SCOPE)
    refs["baseline_result_ref"] = store.put(
        kind="baseline_result",
        session_id=session_id,
        key="v1",
        obj={"source_kind": "live_repo", "reproduced": True, "metric_value": 1.0},
        emitter="RunnerAgent",
    )
    env = _make_envelope(
        "experiment.review_requested", session_id, refs, "ResearchOrchestrator"
    )

    await handler.handle_experiment_plan_review(client, env)

    assert len(llm.calls) == 1
    reviewed = [p for p in client.published if p.event_type == "experiment.plan_reviewed"]
    assert len(reviewed) == 1
    assert reviewed[0].payload["verdict"] == "fail"


async def test_experiment_review_requested_gate_forces_fail_end_to_end(store):
    """Even when the LLM says 'pass', an out-of-editable_region change must
    surface as verdict='fail' all the way through the handler."""
    llm = FakeLLM([_json(VALID_EXPERIMENT_PLAN_REPORT_PASS)])
    handler = CriticAgentHandler(spec_store=store, llm_client=llm)
    client = FakeBusClient()
    handler.register(client)

    session_id = str(uuid.uuid4())
    refs = _seed_experiment_plan_review_inputs(store, session_id, EXPERIMENT_PLAN_OUT_OF_SCOPE)

    env = _make_envelope(
        "experiment.review_requested", session_id, refs, "ResearchOrchestrator"
    )

    await handler.handle_experiment_plan_review(client, env)

    reviewed = [p for p in client.published if p.event_type == "experiment.plan_reviewed"]
    assert len(reviewed) == 1
    assert reviewed[0].payload["verdict"] == "fail"

    report = store.get(reviewed[0].payload["review_report_ref"])
    assert report["verdict"] == "fail"
    assert any("eval/harness.py" in issue for issue in report["issues"])


# ---------------------------------------------------------------------------
# (d) provenance: both ExperimentPlan and VerifierSpec read from the STORE,
# never from the event payload (which carries only refs).
# ---------------------------------------------------------------------------


async def test_experiment_review_requested_reads_inputs_from_store_not_payload(store):
    llm = FakeLLM([_json(VALID_EXPERIMENT_PLAN_REPORT_PASS)])
    handler = CriticAgentHandler(spec_store=store, llm_client=llm)
    client = FakeBusClient()
    handler.register(client)

    session_id = str(uuid.uuid4())
    refs = _seed_experiment_plan_review_inputs(store, session_id, EXPERIMENT_PLAN_IN_SCOPE)

    env = _make_envelope(
        "experiment.review_requested", session_id, refs, "ResearchOrchestrator"
    )
    assert set(env.payload.keys()) == {"experiment_plan_ref", "verifier_spec_ref"}

    await handler.handle_experiment_plan_review(client, env)

    reviewed = [p for p in client.published if p.event_type == "experiment.plan_reviewed"]
    assert len(reviewed) == 1

    [messages] = llm.calls
    user_message = next(m["content"] for m in messages if m["role"] == "user")
    assert EXPERIMENT_PLAN_IN_SCOPE["change_summary"] in user_message
    assert VERIFIER_SPEC["metric"] in user_message


async def test_experiment_review_requested_causation_id_set(store):
    llm = FakeLLM([_json(VALID_EXPERIMENT_PLAN_REPORT_PASS)])
    handler = CriticAgentHandler(spec_store=store, llm_client=llm)
    client = FakeBusClient()
    handler.register(client)

    session_id = str(uuid.uuid4())
    refs = _seed_experiment_plan_review_inputs(store, session_id, EXPERIMENT_PLAN_IN_SCOPE)

    env = _make_envelope(
        "experiment.review_requested", session_id, refs, "ResearchOrchestrator"
    )

    await handler.handle_experiment_plan_review(client, env)

    reviewed = [p for p in client.published if p.event_type == "experiment.plan_reviewed"]
    assert len(reviewed) == 1
    assert reviewed[0].causation_id == env.event_id


async def test_experiment_review_requested_core_error_publishes_agent_failed(store):
    llm = FakeLLM(["not json at all"] * (core.MAX_RETRIES + 1))
    handler = CriticAgentHandler(spec_store=store, llm_client=llm)
    client = FakeBusClient()
    handler.register(client)

    session_id = str(uuid.uuid4())
    refs = _seed_experiment_plan_review_inputs(store, session_id, EXPERIMENT_PLAN_IN_SCOPE)

    env = _make_envelope(
        "experiment.review_requested", session_id, refs, "ResearchOrchestrator"
    )

    await handler.handle_experiment_plan_review(client, env)

    failed = [p for p in client.published if p.event_type == "agent.failed"]
    assert len(failed) == 1
    assert failed[0].session_id == session_id
    assert failed[0].causation_id == env.event_id
    assert failed[0].payload["agent_name"] == "CriticAgent"
    assert failed[0].payload["reason_classification"] == "permanent"
    assert failed[0].payload["underlying_reason"]

    assert not [p for p in client.published if p.event_type == "experiment.plan_reviewed"]


# ---------------------------------------------------------------------------
# (e) register() now wires a THIRD subscription, alongside the existing two
# (which test_handler.py continues to cover/regression-test).
# ---------------------------------------------------------------------------


def test_register_wires_third_experiment_review_subscription(store):
    llm = FakeLLM([])
    handler = CriticAgentHandler(spec_store=store, llm_client=llm)
    client = FakeBusClient()
    handler.register(client)

    patterns = [pattern for pattern, _ in client.subscriptions]
    assert "direction.critique_requested" in patterns
    assert "hypothesis.critique_requested" in patterns
    assert "experiment.review_requested" in patterns
    assert len(patterns) == 3


# ---------------------------------------------------------------------------
# MOSS prompt_patch: no-patch unchanged / with-patch injected, at the core
# level (critique_experiment_plan) and at the handler level (config_root).
# ---------------------------------------------------------------------------


def test_critique_experiment_plan_no_prompt_patch_leaves_system_message_unchanged():
    llm = FakeLLM([_json(VALID_EXPERIMENT_PLAN_REPORT_PASS)])

    core.critique_experiment_plan(EXPERIMENT_PLAN_IN_SCOPE, VERIFIER_SPEC, llm)

    [messages] = llm.calls
    system_message = next(m["content"] for m in messages if m["role"] == "system")
    assert system_message == _load_experiment_plan_system_prompt()


def test_critique_experiment_plan_prompt_patch_appended_to_system_message():
    llm = FakeLLM([_json(VALID_EXPERIMENT_PLAN_REPORT_PASS)])

    core.critique_experiment_plan(
        EXPERIMENT_PLAN_IN_SCOPE, VERIFIER_SPEC, llm, prompt_patch="STEER: prefer X"
    )

    [messages] = llm.calls
    system_message = next(m["content"] for m in messages if m["role"] == "system")
    assert "STEER: prefer X" in system_message
    assert "[MOSS-injected guidance]" in system_message


async def test_experiment_review_prompt_patch_from_config_root_injected(store, tmp_path):
    patch_dir = tmp_path / "agent_config" / "CriticAgent"
    patch_dir.mkdir(parents=True)
    (patch_dir / "prompt_patch.md").write_text("STEER: prefer X", encoding="utf-8")

    llm = FakeLLM([_json(VALID_EXPERIMENT_PLAN_REPORT_PASS)])
    handler = CriticAgentHandler(spec_store=store, llm_client=llm, config_root=tmp_path)
    client = FakeBusClient()
    handler.register(client)

    session_id = str(uuid.uuid4())
    refs = _seed_experiment_plan_review_inputs(store, session_id, EXPERIMENT_PLAN_IN_SCOPE)

    env = _make_envelope(
        "experiment.review_requested", session_id, refs, "ResearchOrchestrator"
    )

    await handler.handle_experiment_plan_review(client, env)

    [messages] = llm.calls
    system_message = next(m["content"] for m in messages if m["role"] == "system")
    assert "STEER: prefer X" in system_message
    assert "[MOSS-injected guidance]" in system_message
