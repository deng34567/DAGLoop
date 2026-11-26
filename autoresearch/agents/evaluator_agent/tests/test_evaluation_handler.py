"""
Tests for EvaluatorAgent's EventBus handler -- M12: EvaluatingExperiment +
VerifyingBestResult flows.
EvaluatorAgent EventBus handler 的测试 -- M12：EvaluatingExperiment +
VerifyingBestResult 流程。

Mirrors test_handler.py's pattern: FakeBusClient (captures publishes; does
NOT auto-deliver) + FakeLLM (from this package's tests/conftest.py) + a real
tmp SpecStore. Each test calls the handler's event method directly with a
constructed Envelope.
"""

from __future__ import annotations

import json
import hashlib
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import pytest
import yaml
from jsonschema import Draft202012Validator

from autoresearch.agents.evaluator_agent import core
from autoresearch.agents.evaluator_agent.handler import EvaluatorAgentHandler
from autoresearch.agents.evaluator_agent.tests.conftest import FakeLLM
from autoresearch.spec_store import SpecStore

_CONTRACTS_DIR = Path(__file__).resolve().parents[1] / "data-contracts"


def _validator(name: str) -> Draft202012Validator:
    schema = yaml.safe_load((_CONTRACTS_DIR / name).read_text(encoding="utf-8"))
    return Draft202012Validator(schema)


EVALUATION_REPORT_VALIDATOR = _validator("evaluation_report.schema.yaml")
VERIFICATION_REPORT_VALIDATOR = _validator("verification_report.schema.yaml")


def _valid_verifier_spec(**overrides: object) -> dict:
    spec = {
        "evaluator_hash": "sha256:abc123",
        "metric": "exact_match",
        "data_split": {"in_loop": "dev", "held_out": "test"},
        "run_budget": "10 runs, 1h wall-clock",
        "seed_policy": "fixed seed=42 across all runs",
        "editable_region": ["src/model.py", "src/train.py"],
        "forbidden_files": ["benchmark/data/held_out.jsonl"],
        "result_parse_schema": {"type": "object"},
    }
    spec.update(overrides)
    return spec


VALID_VERIFIER_SPEC = _valid_verifier_spec()

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


def _valid_run_record(**overrides: object) -> dict:
    record = {
        "experiment_id": "exp-1",
        "succeeded": True,
        "exit_code": 0,
        "metric_value": 0.51,
        "stdout_excerpt": "epoch 10: exact_match=0.51",
        "stderr_excerpt": None,
        "environment": "python3.11, cuda12.1, 1x A100",
        "seed": "42",
        "runtime_seconds": 120.0,
        "artifacts": [],
    }
    record.update(overrides)
    return record


# RunRecord.metric_value (0.51) > BenchmarkSpec.baseline.reported_metric
# (0.42) -- the deterministic is_improvement guard in core.evaluate_experiment
# computes is_improvement=True, so the canned LLM verdict must be "new_best".
VALID_RUN_RECORD = _valid_run_record(metric_value=0.51)

VALID_EVALUATION_REPORT_NEW_BEST = {
    "verdict": "new_best",
    "metric_value": 0.51,
    "baseline_metric": 0.42,
    "is_improvement": True,
    "suspicious": False,
    "anti_cheat_flags": [],
    "reasoning": None,
}


@pytest.mark.asyncio
@pytest.mark.parametrize("metric", [None, 0.51])
async def test_component_runner_failure_without_measurement_is_invalid(store, metric):
    sid = str(uuid.uuid4())
    run = _valid_run_record(succeeded=False, exit_code=1, metric_value=metric,
                            environment="component runner failure",
                            stdout_excerpt="", stderr_excerpt="PermissionError: lock inaccessible")
    benchmark = dict(VALID_BENCHMARK_SPEC, evaluation_semantics={
        "comparison_statement": "Compare the same frozen evaluator and data."})
    payload = {
        "run_record_ref": store.put(kind="run_record", session_id=sid, key="v1",
                                    obj=run, emitter="RunnerAgent"),
        "benchmark_spec_ref": store.put(kind="benchmark_spec", session_id=sid, key="v1",
                                        obj=benchmark, emitter="EvaluatorAgent"),
        "baseline_result_ref": store.put(kind="baseline_result", session_id=sid, key="v1",
                                         obj={"metric_value": 0.42}, emitter="RunnerAgent"),
    }
    llm = FakeLLM([_json({"verdict": "comparison_changed", "issues": ["observed evaluator edit"],
                          "remediation": "restore evaluator", "reasoning": "actual source edit"})])
    client = FakeBusClient()
    handler = EvaluatorAgentHandler(spec_store=store, llm_client=llm)
    await handler.handle_evaluation_requested(client, _make_envelope(
        "experiment.evaluation_requested", sid, payload, "ResearchOrchestrator"))
    [event] = [e for e in client.published if e.event_type == "experiment.evaluated"]
    result = store.get(event.payload["evaluation_report_ref"])
    if metric is None:
        assert result["verdict"] == "invalid"
        assert result["metric_value"] is None and result["baseline_metric"] == 0.42
        assert result["is_improvement"] is False and result["suspicious"] is False
        assert result["anti_cheat_flags"] == [] and llm.calls == []
        assert "PermissionError" in result["reasoning"]
    else:
        assert result["verdict"] == "suspicious"
        assert result["anti_cheat_flags"] and llm.calls


def _json(obj: dict) -> str:
    return json.dumps(obj)


def _artifact_digest(obj: dict) -> str:
    canonical = json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return "sha256:" + hashlib.sha256(canonical.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Test doubles (mirrors test_handler.py's FakeBusClient / _Envelope).
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
        self.emitter_name = "EvaluatorAgent"
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


def _seed_evaluation_inputs(store: SpecStore, session_id: str) -> dict[str, str]:
    """Seed the store with the RunRecord + VerifierSpec + BenchmarkSpec
    EvaluatorAgent reads for EvaluatingExperiment, as an upstream RunnerAgent
    run + earlier handler runs would; return their refs."""
    run_record_ref = store.put(
        kind="run_record",
        session_id=session_id,
        key="v1",
        obj=VALID_RUN_RECORD,
        emitter="RunnerAgent",
    )
    verifier_spec_ref = store.put(
        kind="verifier_spec",
        session_id=session_id,
        key="v1",
        obj=VALID_VERIFIER_SPEC,
        emitter="EvaluatorAgent",
    )
    benchmark_spec_ref = store.put(
        kind="benchmark_spec",
        session_id=session_id,
        key="v1",
        obj=VALID_BENCHMARK_SPEC,
        emitter="EvaluatorAgent",
    )
    return {
        "run_record_ref": run_record_ref,
        "verifier_spec_ref": verifier_spec_ref,
        "benchmark_spec_ref": benchmark_spec_ref,
    }


def _seed_verification_inputs(store: SpecStore, session_id: str) -> dict[str, str]:
    """Seed the store with the EvaluationReport + RunRecord + VerifierSpec
    EvaluatorAgent reads for VerifyingBestResult, as an earlier handler run +
    upstream RunnerAgent run would; return their refs."""
    evaluation_report_ref = store.put(
        kind="evaluation_report",
        session_id=session_id,
        key="v1",
        obj=VALID_EVALUATION_REPORT_NEW_BEST,
        emitter="EvaluatorAgent",
    )
    run_record_ref = store.put(
        kind="run_record",
        session_id=session_id,
        key="v1",
        obj=VALID_RUN_RECORD,
        emitter="RunnerAgent",
    )
    verifier_spec_ref = store.put(
        kind="verifier_spec",
        session_id=session_id,
        key="v1",
        obj=VALID_VERIFIER_SPEC,
        emitter="EvaluatorAgent",
    )
    return {
        "evaluation_report_ref": evaluation_report_ref,
        "run_record_ref": run_record_ref,
        "verifier_spec_ref": verifier_spec_ref,
    }


# ---------------------------------------------------------------------------
# (a) experiment.evaluation_requested -> experiment.evaluated, ref resolves to
# a schema-valid EvaluationReport, inputs read from the STORE (provenance),
# never from the event payload.
# ---------------------------------------------------------------------------


async def test_evaluation_requested_publishes_evaluated_with_resolvable_ref(store):
    llm_response = {
        "verdict": "new_best",
        "suspicious": False,
        "anti_cheat_flags": [],
        "reasoning": "0.51 beats the baseline 0.42.",
    }
    llm = FakeLLM([_json(llm_response)])
    handler = EvaluatorAgentHandler(spec_store=store, llm_client=llm)
    client = FakeBusClient()
    handler.register(client)

    session_id = str(uuid.uuid4())
    refs = _seed_evaluation_inputs(store, session_id)

    env = _make_envelope(
        "experiment.evaluation_requested", session_id, refs, "ResearchOrchestrator"
    )

    await handler.handle_evaluation_requested(client, env)

    evaluated = [p for p in client.published if p.event_type == "experiment.evaluated"]
    assert len(evaluated) == 1
    payload = evaluated[0].payload
    assert set(payload.keys()) == {
        "verdict", "evaluation_report_ref", "run_record_digest"}
    assert payload["verdict"] == "new_best"
    assert payload["run_record_digest"] == _artifact_digest(VALID_RUN_RECORD)

    report = store.get(payload["evaluation_report_ref"])
    errors = list(EVALUATION_REPORT_VALIDATOR.iter_errors(report))
    assert errors == [], [e.message for e in errors]
    assert report["verdict"] == "new_best"
    assert report["is_improvement"] is True
    assert report["run_record_digest"] == payload["run_record_digest"]


async def test_repo_result_uses_runner_outcome_without_comparison_guardian_veto(store):
    """A completed repo attempt is evaluated from its recorded outcome.

    Scientific comparison semantics are frozen before execution.  Result evaluation must not
    ask another LLM to reinterpret an observed final step that slightly crosses the nominal
    timer boundary; verification only checks that the recorded metric/failure is internally
    consistent.  This catches the live NanoChat regression where 305.66 observed seconds made a
    real negative result skip verification and memory entirely.
    """
    semantics = {
        "comparison_statement": (
            "On the same GPU class, model scale, data, tokenizer and evaluator, give baseline "
            "and method 300 seconds of pure training and compare final val_bpb."
        ),
        "research_question": "Does the method lower val_bpb after 300 seconds?",
        "primary_metric": {"name": "val_bpb", "direction": "minimize"},
        "comparison_mode": "quality_at_fixed_time",
        "comparison_basis": "Compare quality at fixed pure-training time.",
        "budget_interpretation": "Each side gets 300 seconds of pure training.",
        "conditions_to_hold_constant": ["GPU class", "model scale", "data", "evaluator"],
        "method_freedom": ["training algorithm", "steps and tokens as observed outcomes"],
        "evidence": [{
            "source_type": "research_spec", "source_id": "research_spec",
            "supports": "The task asks for a five-minute quality comparison.",
        }],
        "uncertainties": [],
    }
    benchmark = dict(VALID_BENCHMARK_SPEC)
    benchmark["metric"] = "val_bpb"
    benchmark["evaluation_semantics"] = semantics
    report_line = "COMPARISON_AUDIT: requested_train_s=300 observed_train_s=305.66"
    run_record = _valid_run_record(
        metric_value=2.208178, stdout_excerpt=report_line,
        environment=(
            "repo_improve|benchmark=repo|verdict=valid_not_best|"
            "baseline=1.555459|method=2.208178|dgx"
        ),
    )
    session_id = str(uuid.uuid4())
    refs = {
        "run_record_ref": store.put(
            kind="run_record", session_id=session_id, key="v1",
            obj=run_record, emitter="RunnerAgent"),
        "verifier_spec_ref": store.put(
            kind="verifier_spec", session_id=session_id, key="v1",
            obj=VALID_VERIFIER_SPEC, emitter="EvaluatorAgent"),
        "benchmark_spec_ref": store.put(
            kind="benchmark_spec", session_id=session_id, key="v1",
            obj=benchmark, emitter="EvaluatorAgent"),
    }
    guardian_veto = {
        "verdict": "comparison_changed",
        "issues": ["observed timer crossed 300 seconds"],
        "remediation": "rerun with a stricter timer",
        "reasoning": "The final completed step crossed the nominal boundary.",
    }
    llm = FakeLLM([_json(guardian_veto)])
    handler = EvaluatorAgentHandler(spec_store=store, llm_client=llm)
    client = FakeBusClient()

    await handler.handle_evaluation_requested(client, _make_envelope(
        "experiment.evaluation_requested", session_id, refs, "ResearchOrchestrator"))

    [event] = [item for item in client.published if item.event_type == "experiment.evaluated"]
    assert event.payload["verdict"] == "valid_not_best"
    result = store.get(event.payload["evaluation_report_ref"])
    assert result["metric_value"] == 2.208178
    assert result["baseline_metric"] == 1.555459
    assert result["anti_cheat_flags"] == []
    assert llm.calls == []

    await handler.handle_verification_requested(client, _make_envelope(
        "result.verification_requested", session_id, {
            "evaluation_report_ref": event.payload["evaluation_report_ref"],
            "run_record_ref": refs["run_record_ref"],
            "verifier_spec_ref": refs["verifier_spec_ref"],
        }, "ResearchOrchestrator"))

    [verified] = [item for item in client.published if item.event_type == "result.verified"]
    assert verified.payload["verdict"] == "verified"
    verification = store.get(verified.payload["verification_report_ref"])
    assert verification["anti_cheat_flags"] == []
    assert any("metric_matches_run_record" in check for check in verification["checks"])


async def test_comparison_guardian_resolves_frozen_baseline_from_store(store):
    semantics = {
        "comparison_statement": (
            "On the same GPU, model, accepted repository data, tokenizer and evaluator, "
            "give each side 300 seconds and compare val_bpb."
        ),
        "research_question": "Does the method lower val_bpb?",
        "primary_metric": {"name": "val_bpb", "direction": "minimize"},
        "comparison_mode": "quality_at_fixed_time",
        "comparison_basis": "Fixed training time.",
        "budget_interpretation": "Each side receives 300 seconds.",
        "conditions_to_hold_constant": ["GPU", "model", "accepted repository data"],
        "method_freedom": ["training algorithm and hyperparameters"],
        "evidence": [],
        "uncertainties": [],
    }
    benchmark = dict(VALID_BENCHMARK_SPEC, metric="val_bpb",
                     evaluation_semantics=semantics)
    run_record = _valid_run_record(metric_value=0.91)
    baseline = {
        "reproduced": True, "metric_value": 0.94, "metric_name": "val_bpb",
        "fixed_config": {"data": "accepted repository data", "training_seconds": 300},
    }
    session_id = str(uuid.uuid4())
    refs = {
        "run_record_ref": store.put(
            kind="run_record", session_id=session_id, key="v1",
            obj=run_record, emitter="RunnerAgent"),
        "verifier_spec_ref": store.put(
            kind="verifier_spec", session_id=session_id, key="v1",
            obj=VALID_VERIFIER_SPEC, emitter="EvaluatorAgent"),
        "benchmark_spec_ref": store.put(
            kind="benchmark_spec", session_id=session_id, key="v1",
            obj=benchmark, emitter="EvaluatorAgent"),
        "baseline_result_ref": store.put(
            kind="baseline_result", session_id=session_id, key="v1",
            obj=baseline, emitter="RunnerAgent"),
    }
    guardian_pass = {
        "verdict": "pass", "issues": [], "remediation": None,
        "reasoning": "The method preserved the accepted baseline conditions.",
    }
    evaluator_result = {
        "verdict": "new_best", "suspicious": False,
        "anti_cheat_flags": [], "reasoning": "0.91 beats 0.94.",
    }
    llm = FakeLLM([_json(guardian_pass), _json(evaluator_result)])
    handler = EvaluatorAgentHandler(spec_store=store, llm_client=llm)
    client = FakeBusClient()

    await handler.handle_evaluation_requested(client, _make_envelope(
        "experiment.evaluation_requested", session_id, refs, "ResearchOrchestrator"))

    guardian_prompt = llm.calls[0][1]["content"]
    assert "Frozen reproduced BaselineResult" in guardian_prompt
    assert "accepted repository data" in guardian_prompt
    [event] = [p for p in client.published if p.event_type == "experiment.evaluated"]
    assert event.payload["verdict"] == "new_best"


async def test_evaluation_requested_reads_inputs_from_store_not_payload(store):
    """The event payload carries ONLY the three refs (per the
    experiment.evaluation_requested schema) -- the handler must resolve the
    full objects via the store, never expect them inline in the payload."""
    llm_response = {
        "verdict": "new_best",
        "suspicious": False,
        "anti_cheat_flags": [],
        "reasoning": "0.51 beats the baseline 0.42.",
    }
    llm = FakeLLM([_json(llm_response)])
    handler = EvaluatorAgentHandler(spec_store=store, llm_client=llm)
    client = FakeBusClient()
    handler.register(client)

    session_id = str(uuid.uuid4())
    refs = _seed_evaluation_inputs(store, session_id)

    env = _make_envelope(
        "experiment.evaluation_requested", session_id, refs, "ResearchOrchestrator"
    )
    assert set(env.payload.keys()) == {
        "run_record_ref",
        "verifier_spec_ref",
        "benchmark_spec_ref",
    }

    await handler.handle_evaluation_requested(client, env)

    evaluated = [p for p in client.published if p.event_type == "experiment.evaluated"]
    assert len(evaluated) == 1

    [messages] = llm.calls
    user_message = next(m["content"] for m in messages if m["role"] == "user")
    assert VALID_RUN_RECORD["stdout_excerpt"] in user_message
    assert VALID_VERIFIER_SPEC["evaluator_hash"] in user_message
    assert VALID_BENCHMARK_SPEC["benchmark_id"] in user_message


async def test_evaluation_causation_id_set(store):
    llm_response = {
        "verdict": "new_best",
        "suspicious": False,
        "anti_cheat_flags": [],
        "reasoning": "0.51 beats the baseline 0.42.",
    }
    llm = FakeLLM([_json(llm_response)])
    handler = EvaluatorAgentHandler(spec_store=store, llm_client=llm)
    client = FakeBusClient()
    handler.register(client)

    session_id = str(uuid.uuid4())
    refs = _seed_evaluation_inputs(store, session_id)

    env = _make_envelope(
        "experiment.evaluation_requested", session_id, refs, "ResearchOrchestrator"
    )

    await handler.handle_evaluation_requested(client, env)

    evaluated = [p for p in client.published if p.event_type == "experiment.evaluated"]
    assert len(evaluated) == 1
    assert evaluated[0].causation_id == env.event_id


async def test_evaluation_core_error_publishes_agent_failed(store):
    llm = FakeLLM(["not json at all"] * (core.MAX_RETRIES + 1))
    handler = EvaluatorAgentHandler(spec_store=store, llm_client=llm)
    client = FakeBusClient()
    handler.register(client)

    session_id = str(uuid.uuid4())
    refs = _seed_evaluation_inputs(store, session_id)

    env = _make_envelope(
        "experiment.evaluation_requested", session_id, refs, "ResearchOrchestrator"
    )

    await handler.handle_evaluation_requested(client, env)

    failed = [p for p in client.published if p.event_type == "agent.failed"]
    assert len(failed) == 1
    assert failed[0].session_id == session_id
    assert failed[0].causation_id == env.event_id
    assert failed[0].payload["agent_name"] == "EvaluatorAgent"
    assert failed[0].payload["reason_classification"] == "permanent"
    assert failed[0].payload["underlying_reason"]

    assert not [p for p in client.published if p.event_type == "experiment.evaluated"]


# ---------------------------------------------------------------------------
# (M13 Task 5) handle_evaluation_requested reads a prior ResearchMemory from
# the store IFF payload.research_memory_ref is truthy, and passes it through
# to core.evaluate_experiment as the keyword-only prior_research_memory.
# ---------------------------------------------------------------------------

PRIOR_RESEARCH_MEMORY = {
    "best_metric": 0.9,
    "best_result_ref": None,
    "verified_findings": ["long-tail retrieval never beats baseline without reranking"],
    "negative_results": [],
    "consolidated_insights": [],
    "divergence_seeds": [],
}


def _spy_on_evaluate_experiment(monkeypatch, captured: dict) -> None:
    original = core.evaluate_experiment

    def _spy(
        run_record, verifier_spec, benchmark_spec, llm_client, *,
        prior_research_memory=None, prompt_patch=None,
    ):
        captured["prior_research_memory"] = prior_research_memory
        return original(
            run_record, verifier_spec, benchmark_spec, llm_client,
            prior_research_memory=prior_research_memory,
            prompt_patch=prompt_patch,
        )

    monkeypatch.setattr(core, "evaluate_experiment", _spy)


async def test_evaluation_requested_passes_prior_research_memory_when_ref_given(store, monkeypatch):
    session_id = str(uuid.uuid4())
    refs = _seed_evaluation_inputs(store, session_id)
    research_memory_ref = store.put(
        kind="research_memory",
        session_id=session_id,
        key="v1",
        obj=PRIOR_RESEARCH_MEMORY,
        emitter="MemoryAgent",
    )
    refs = {**refs, "research_memory_ref": research_memory_ref}

    captured: dict = {}
    _spy_on_evaluate_experiment(monkeypatch, captured)

    # 0.51 (VALID_RUN_RECORD.metric_value) <= PRIOR_RESEARCH_MEMORY.best_metric
    # (0.9) -- the running-best threshold, not the BenchmarkSpec baseline
    # (0.42) -- so the computed is_improvement is False.
    llm_response = {
        "verdict": "valid_not_best",
        "suspicious": False,
        "anti_cheat_flags": [],
        "reasoning": "0.51 does not beat the running best of 0.9.",
    }
    llm = FakeLLM([_json(llm_response)])
    handler = EvaluatorAgentHandler(spec_store=store, llm_client=llm)
    client = FakeBusClient()
    handler.register(client)

    env = _make_envelope(
        "experiment.evaluation_requested", session_id, refs, "ResearchOrchestrator"
    )
    await handler.handle_evaluation_requested(client, env)

    assert captured["prior_research_memory"] == PRIOR_RESEARCH_MEMORY

    evaluated = [p for p in client.published if p.event_type == "experiment.evaluated"]
    assert len(evaluated) == 1
    assert evaluated[0].payload["verdict"] == "valid_not_best"


async def test_evaluation_requested_passes_none_when_research_memory_ref_absent(store, monkeypatch):
    session_id = str(uuid.uuid4())
    refs = _seed_evaluation_inputs(store, session_id)
    assert "research_memory_ref" not in refs

    captured: dict = {}
    _spy_on_evaluate_experiment(monkeypatch, captured)

    llm_response = {
        "verdict": "new_best",
        "suspicious": False,
        "anti_cheat_flags": [],
        "reasoning": "0.51 beats the baseline 0.42.",
    }
    llm = FakeLLM([_json(llm_response)])
    handler = EvaluatorAgentHandler(spec_store=store, llm_client=llm)
    client = FakeBusClient()
    handler.register(client)

    env = _make_envelope(
        "experiment.evaluation_requested", session_id, refs, "ResearchOrchestrator"
    )
    await handler.handle_evaluation_requested(client, env)

    assert captured["prior_research_memory"] is None

    evaluated = [p for p in client.published if p.event_type == "experiment.evaluated"]
    assert len(evaluated) == 1
    assert evaluated[0].payload["verdict"] == "new_best"


# ---------------------------------------------------------------------------
# (b) result.verification_requested -> result.verified, ref resolves to a
# schema-valid VerificationReport, inputs read from the STORE (provenance),
# never from the event payload.
# ---------------------------------------------------------------------------


async def test_verification_requested_publishes_verified_with_resolvable_ref(store):
    llm_response = {
        "verdict": "verified",
        "checks": ["cross-seed: fixed seed=42", "held-out: metric on held-out split"],
        "anti_cheat_flags": [],
        "reasoning": "Result reproduces under the frozen protocol.",
    }
    llm = FakeLLM([_json(llm_response)])
    handler = EvaluatorAgentHandler(spec_store=store, llm_client=llm)
    client = FakeBusClient()
    handler.register(client)

    session_id = str(uuid.uuid4())
    refs = _seed_verification_inputs(store, session_id)

    env = _make_envelope(
        "result.verification_requested", session_id, refs, "ResearchOrchestrator"
    )

    await handler.handle_verification_requested(client, env)

    verified = [p for p in client.published if p.event_type == "result.verified"]
    assert len(verified) == 1
    payload = verified[0].payload
    assert set(payload.keys()) == {"verdict", "verification_report_ref"}
    assert payload["verdict"] == "verified"

    report = store.get(payload["verification_report_ref"])
    errors = list(VERIFICATION_REPORT_VALIDATOR.iter_errors(report))
    assert errors == [], [e.message for e in errors]
    assert report["verdict"] == "verified"


async def test_verification_requested_reads_inputs_from_store_not_payload(store):
    """The event payload carries ONLY the three refs (per the
    result.verification_requested schema) -- the handler must resolve the
    full objects via the store, never expect them inline in the payload."""
    llm_response = {
        "verdict": "verified",
        "checks": ["cross-seed: fixed seed=42"],
        "anti_cheat_flags": [],
        "reasoning": "Result reproduces under the frozen protocol.",
    }
    llm = FakeLLM([_json(llm_response)])
    handler = EvaluatorAgentHandler(spec_store=store, llm_client=llm)
    client = FakeBusClient()
    handler.register(client)

    session_id = str(uuid.uuid4())
    refs = _seed_verification_inputs(store, session_id)

    env = _make_envelope(
        "result.verification_requested", session_id, refs, "ResearchOrchestrator"
    )
    assert set(env.payload.keys()) == {
        "evaluation_report_ref",
        "run_record_ref",
        "verifier_spec_ref",
    }

    await handler.handle_verification_requested(client, env)

    verified = [p for p in client.published if p.event_type == "result.verified"]
    assert len(verified) == 1

    report = store.get(verified[0].payload["verification_report_ref"])
    assert report["verdict"] == "verified"
    assert any("metric_matches_run_record" in check for check in report["checks"])
    assert llm.calls == []


async def test_verification_accepts_grounded_indirect_component_without_task_metric(store):
    handler = EvaluatorAgentHandler(spec_store=store, llm_client=FakeLLM([]))
    client = FakeBusClient()
    session_id = str(uuid.uuid4())
    run = _valid_run_record(
        metric_value=None,
        environment="repo_improve|verdict=valid_not_best|baseline=128|method=None",
    )
    run_ref = store.put(
        kind="run_record", session_id=session_id, key="v1", obj=run,
        emitter="RunnerAgent")
    evaluation_ref = store.put(
        kind="evaluation_report", session_id=session_id, key="v1",
        obj={"verdict": "valid_not_best", "metric_value": None,
             "baseline_metric": 128.0, "is_improvement": False,
             "suspicious": False, "anti_cheat_flags": [],
             "reasoning": "indirect component", "run_record_digest": _artifact_digest(run)},
        emitter="EvaluatorAgent")
    store.put(
        kind="gen_experiment_result", session_id=session_id, key="current",
        obj={
            "baseline_metric": 128.0,
            "method_metric": None,
            "method_implementation_review": {"verdict": "faithful"},
            "component_execution": {
                "mode": "component", "role": "indirect_gain", "status": "accepted",
                "execution_succeeded": True,
                "attempt_contract_result": {
                    "measurement": "indirect_gain_check", "passed": True,
                    "evidence": {"source": "runtime_attempt", "ref": "attempt-1"},
                },
            },
        },
        emitter="RunnerAgent")
    env = _make_envelope(
        "result.verification_requested", session_id,
        {"evaluation_report_ref": evaluation_ref, "run_record_ref": run_ref,
         "verifier_spec_ref": None},
        "ResearchOrchestrator")

    await handler.handle_verification_requested(client, env)

    [published] = [p for p in client.published if p.event_type == "result.verified"]
    report = store.get(published.payload["verification_report_ref"])
    assert report["verdict"] == "verified"
    assert "primary_task_metric_intentionally_not_claimed" in report["checks"]


@pytest.mark.parametrize("grounded,tamper", [
    (False, None), (True, None), (True, "digest"), (True, "evidence"),
])
async def test_verification_records_failed_indirect_check_only_with_evidence(
    store, grounded, tamper,
):
    handler = EvaluatorAgentHandler(spec_store=store, llm_client=FakeLLM([]))
    client = FakeBusClient()
    session_id = str(uuid.uuid4())
    run = _valid_run_record(
        metric_value=None,
        environment="repo_improve|verdict=valid_not_best|baseline=128|method=None",
    )
    run_ref = store.put(
        kind="run_record", session_id=session_id, key="v1", obj=run,
        emitter="RunnerAgent")
    evaluation_ref = store.put(
        kind="evaluation_report", session_id=session_id, key="v1",
        obj={"verdict": "valid_not_best", "metric_value": None,
             "baseline_metric": 128.0, "is_improvement": False,
             "suspicious": False, "anti_cheat_flags": [],
             "reasoning": "indirect component",
             "run_record_digest": "sha256:other-run" if tamper == "digest" else _artifact_digest(run)},
        emitter="EvaluatorAgent")
    contract = {
        "measurement": "indirect_gain_check", "passed": False,
        "evidence": {"source": "runtime_attempt", "ref": "attempt-1"},
    }
    if grounded:
        contract.update(actual=0.1, expected={"operator": ">", "value": 0.8},
                        reason="indirect-gain threshold failed")
    if tamper == "evidence":
        contract["evidence"] = {}
    gen_ref = store.put(
        kind="gen_experiment_result", session_id=session_id, key="current",
        obj={
            "baseline_metric": 128.0,
            "method_implementation_review": {"verdict": "faithful"},
            "component_execution": {
                "mode": "component", "role": "indirect_gain", "status": "contract_failed",
                "execution_succeeded": True,
                "attempt_contract_result": contract,
            },
        },
        emitter="RunnerAgent")
    env = _make_envelope(
        "result.verification_requested", session_id,
        {"evaluation_report_ref": evaluation_ref, "run_record_ref": run_ref,
         "verifier_spec_ref": None},
        "ResearchOrchestrator")

    await handler.handle_verification_requested(client, env)

    [published] = [p for p in client.published if p.event_type == "result.verified"]
    report = store.get(published.payload["verification_report_ref"])
    assert report["verdict"] == ("verified" if grounded and tamper is None else "failed")
    assert store.get(gen_ref)["component_execution"]["status"] == "contract_failed"
    if grounded and tamper is None:
        assert "indirect_component_failure_confirmed_with_grounded_evidence" in report["checks"]
        assert "indirect_component_contract_passed_with_grounded_evidence" not in report["checks"]
        assert "indirect-gain threshold failed" in report["reasoning"]


async def test_verification_causation_id_set(store):
    llm_response = {
        "verdict": "verified",
        "checks": ["cross-seed: fixed seed=42"],
        "anti_cheat_flags": [],
        "reasoning": "Result reproduces under the frozen protocol.",
    }
    llm = FakeLLM([_json(llm_response)])
    handler = EvaluatorAgentHandler(spec_store=store, llm_client=llm)
    client = FakeBusClient()
    handler.register(client)

    session_id = str(uuid.uuid4())
    refs = _seed_verification_inputs(store, session_id)

    env = _make_envelope(
        "result.verification_requested", session_id, refs, "ResearchOrchestrator"
    )

    await handler.handle_verification_requested(client, env)

    verified = [p for p in client.published if p.event_type == "result.verified"]
    assert len(verified) == 1
    assert verified[0].causation_id == env.event_id


async def test_verification_llm_unusable_falls_back_to_verified_not_agent_failed(store):
    # A (2026-08-03): an unusable verify LLM re-check must NOT fail the flow. verify falls back to a
    # code-derived verdict, so the handler publishes result.verified (with the fallback), never
    # agent.failed -- verify is a second opinion, not a blocker.
    llm = FakeLLM(["not json at all"] * (core.MAX_RETRIES + 1))
    handler = EvaluatorAgentHandler(spec_store=store, llm_client=llm)
    client = FakeBusClient()
    handler.register(client)

    session_id = str(uuid.uuid4())
    refs = _seed_verification_inputs(store, session_id)

    env = _make_envelope(
        "result.verification_requested", session_id, refs, "ResearchOrchestrator"
    )

    await handler.handle_verification_requested(client, env)

    failed = [p for p in client.published if p.event_type == "agent.failed"]
    assert len(failed) == 0                                    # no hard failure
    verified = [p for p in client.published if p.event_type == "result.verified"]
    assert len(verified) == 1                                  # flow proceeds with the fallback
    assert verified[0].causation_id == env.event_id


async def test_survey_validation_evaluates_without_benchmark(store):
    handler = EvaluatorAgentHandler(spec_store=store, llm_client=FakeLLM([]))
    client = FakeBusClient()
    session_id = str(uuid.uuid4())
    run = _valid_run_record(
        metric_value=0.03,
        environment="survey_validation|host_rerun",
        stdout_excerpt="No stable association.",
        artifacts=[f"spec://validation_study_result/{session_id}/v1"],
    )
    run_ref = store.put(
        kind="run_record", session_id=session_id, key="v1", obj=run,
        emitter="RunnerAgent")
    store.put(
        kind="validation_study_result", session_id=session_id, key="v1",
        obj={"conclusion": "inconclusive", "metric_name": "spearman_r",
             "metric_value": 0.03, "sample_ids": ["W1"],
             "retrieved_at": "2026-08-20T10:00:00Z", "data_sources": ["OpenAlex"],
             "analysis_summary": "No stable association.", "artifacts": ["raw_data.json"]},
        emitter="RunnerAgent")
    interpretation_ref = store.put(
        kind="spec_interpretation", session_id=session_id, key="v1",
        obj={"validation_study": {"kind": "survey_validation"}},
        emitter="ProblemModelingAgent")
    env = _make_envelope(
        "experiment.evaluation_requested", session_id,
        {"run_record_ref": run_ref, "verifier_spec_ref": None,
         "benchmark_spec_ref": None, "baseline_result_ref": None,
         "research_memory_ref": None, "spec_interpretation_ref": interpretation_ref},
        "ResearchOrchestrator")

    await handler.handle_evaluation_requested(client, env)

    [published] = [p for p in client.published if p.event_type == "experiment.evaluated"]
    report = store.get(published.payload["evaluation_report_ref"])
    assert report["verdict"] == "valid_not_best"
    assert report["metric_value"] == 0.03
    assert report["baseline_metric"] is None
    assert "inconclusive" in report["reasoning"]


async def test_survey_validation_verification_checks_recorded_result(store):
    handler = EvaluatorAgentHandler(spec_store=store, llm_client=FakeLLM([]))
    client = FakeBusClient()
    session_id = str(uuid.uuid4())
    run = _valid_run_record(
        metric_value=0.03, environment="survey_validation|host_rerun",
        artifacts=[f"spec://validation_study_result/{session_id}/v1"])
    run_ref = store.put(
        kind="run_record", session_id=session_id, key="v1", obj=run,
        emitter="RunnerAgent")
    evaluation_ref = store.put(
        kind="evaluation_report", session_id=session_id, key="v1",
        obj={"verdict": "valid_not_best", "metric_value": 0.03,
             "baseline_metric": None, "is_improvement": False, "suspicious": False,
             "anti_cheat_flags": [], "reasoning": "survey_validation:inconclusive",
             "run_record_digest": _artifact_digest(run)},
        emitter="EvaluatorAgent")
    store.put(
        kind="validation_study_result", session_id=session_id, key="v1",
        obj={"conclusion": "inconclusive", "metric_name": "spearman_r",
             "metric_value": 0.03, "sample_ids": ["W1"],
             "retrieved_at": "2026-08-20T10:00:00Z", "data_sources": ["OpenAlex"],
             "analysis_summary": "No stable association.", "artifacts": ["raw_data.json"]},
        emitter="RunnerAgent")
    interpretation_ref = store.put(
        kind="spec_interpretation", session_id=session_id, key="v1",
        obj={"validation_study": {"kind": "survey_validation"}},
        emitter="ProblemModelingAgent")
    env = _make_envelope(
        "result.verification_requested", session_id,
        {"evaluation_report_ref": evaluation_ref, "run_record_ref": run_ref,
         "verifier_spec_ref": None, "spec_interpretation_ref": interpretation_ref},
        "ResearchOrchestrator")

    await handler.handle_verification_requested(client, env)

    [published] = [p for p in client.published if p.event_type == "result.verified"]
    report = store.get(published.payload["verification_report_ref"])
    assert report["verdict"] == "verified"
    assert any("sample_ids" in check for check in report["checks"])


# ---------------------------------------------------------------------------
# (f) register() wires all FOUR subscriptions (the M7 pair + these two new
# ones) -- regression guard against dropping the M7 wiring.
# ---------------------------------------------------------------------------


def test_register_wires_four_subscriptions(store):
    llm = FakeLLM([])
    handler = EvaluatorAgentHandler(spec_store=store, llm_client=llm)
    client = FakeBusClient()
    handler.register(client)

    patterns = [pattern for pattern, _ in client.subscriptions]
    assert set(patterns) == {
        "benchmark.selection_requested",
        "verifier.preparation_requested",
        "experiment.evaluation_requested",
        "result.verification_requested",
    }
    assert len(patterns) == 4
