"""
Tests for RunnerAgent's EventBus handler.
RunnerAgent EventBus handler 的测试。

FakeBusClient (captures publishes; mirrors the real EventBusClient's
publish/subscribe signature -- same shape as evaluator_agent's FakeBusClient)
+ FakeRunnerBackend (from autoresearch.agents._reuse.runner_backend) + a real
tmp SpecStore. The FakeBusClient does NOT auto-deliver events to subscribed
handlers -- each test calls the handler's event method directly with a
constructed Envelope.
"""

from __future__ import annotations

import base64
import copy
import hashlib
import json
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import pytest
import yaml
from jsonschema import Draft202012Validator

from autoresearch.agents._reuse.runner_backend import FakeRunnerBackend
from autoresearch.agents.runner_agent.handler import RunnerAgentHandler
from autoresearch.agents.runner_agent.service import build_runner_service
from autoresearch.spec_store import SpecStore

_CONTRACTS_DIR = Path(__file__).resolve().parents[1] / "data-contracts"
BASELINE_RESULT_VALIDATOR = Draft202012Validator(
    yaml.safe_load((_CONTRACTS_DIR / "baseline_result.schema.yaml").read_text(encoding="utf-8"))
)

VALID_BENCHMARK_SPEC = {
    "benchmark_id": "longtailqa-v1",
    "name": "LongTailQA",
    "metric": "exact_match",
    "run_command": "python run_benchmark.py --model candidate",
    "baseline": {
        "command": "python run_benchmark.py --model baseline",
        "reported_metric": 0.5,
    },
    "resource_requirements": "1x A100 GPU",
    "forbidden_files": ["benchmark/data/held_out.jsonl"],
    "evaluation_semantics": {
        "comparison_statement": (
            "Under the LongTailQA protocol, compare exact_match while allowing method changes."
        ),
        "research_question": "Does the method improve exact match under the benchmark protocol?",
        "primary_metric": {"name": "exact_match", "direction": "maximize"},
        "comparison_mode": "repository_defined",
        "comparison_basis": "Use the repository benchmark.",
        "budget_interpretation": "Use its documented budget.",
        "conditions_to_hold_constant": ["dataset and evaluator"],
        "method_freedom": ["method implementation"],
        "evidence": [{
            "source_type": "existing_benchmark", "source_id": "longtailqa-v1",
            "supports": "The benchmark defines the evaluator.",
        }],
        "uncertainties": [],
    },
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


_EMPTY_DIFF_HASH = "sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


def _valid_host_attestation(*, metric: str = "loss", value: float = 1.0,
                            repo_commit: str = "abc123", protocol_evidence: dict | None = None,
                            protocol_parser: dict | None = None) -> dict:
    argv = ["python", "train.py", "--steps", "1000"]
    parser = {
        "stream": "stdout", "regex": r"loss=(\d+(?:\.\d+)?)", "group": 1,
    }

    measurement_spec = {
        "execution": {"argv": argv, "cwd": "."},
        "trusted_probe": {
            "kind": "external_scorer",
            "argv": ["python", "evaluate.py", "--frozen", "{artifact}"], "cwd": ".",
            "artifact": {
                "path": "metrics.json", "producer_sha256_before": None,
                "required_new": True,
            },
            "parser": parser,
            **({"protocol_parser": protocol_parser} if protocol_parser else {}),
        },
    }
    protocol_content_hash = "sha256:" + "5" * 64
    official_source_evidence = []
    for role, identity in (
        ("execution", measurement_spec["execution"]),
        ("trusted_probe", measurement_spec["trusted_probe"]),
    ):
        canonical = json.dumps(identity, sort_keys=True, separators=(",", ":"))
        official_source_evidence.append({
            "authority": "repository", "role": role, "path": "PROTOCOL.md",
            "line_start": 1, "line_end": 2,
            "file_sha256": protocol_content_hash,
            "excerpt_sha256": protocol_content_hash,
            "command": canonical,
            "identity_sha256": "sha256:" + hashlib.sha256(canonical.encode()).hexdigest(),
        })
    return {
        "host_issued": True,
        "source": "host_remeasurement",
        "phase": "baseline",
        "attempt_id": "host-attempt-1",
        "measurement_spec": measurement_spec,
        "argv": argv,
        "cwd": ".",
        "start": 100.0,
        "end": 110.0,
        "rc": 0,
        "stdout_sha256": "sha256:" + "1" * 64,
        "stderr_sha256": "sha256:" + "2" * 64,
        "repo_commit_start": repo_commit,
        "repo_commit_end": repo_commit,
        "repo_commit": repo_commit,
        "tracked_diff_hash_start": _EMPTY_DIFF_HASH,
        "tracked_diff_hash_end": _EMPTY_DIFF_HASH,
        "tracked_diff_present": False,
        "tracked_diff_paths": [],
        "immutable_file_hashes": {
            "data_config.txt": "sha256:" + "3" * 64,
            "evaluate.py": "sha256:" + "4" * 64,
            "PROTOCOL.md": protocol_content_hash,
        },
        "protocol_evidence": protocol_evidence or {
            "source": "host_elapsed", "unit": "seconds", "value": 10.0,
        },
        "official_source_evidence": official_source_evidence,
        "trusted_probe_evidence": {
            "kind": "external_scorer",
            "argv": measurement_spec["trusted_probe"]["argv"], "cwd": ".",
        },
        "artifact_evidence": {
            "path": "metrics.json", "sha256": "sha256:" + "7" * 64,
            "mtime": 101.0, "size": 42, "required_new": True,
        },
        "prior_artifact_evidence": {"state": "absent", "path": "metrics.json"},
        "agent_attempt_evidence": {
            "authority": "host_execution_broker",
            "attempt_id": "attempt-agent-1", "argv": argv,
            "normalized_argv": argv, "execution_argv": argv,
            "cwd": "/sandbox/repo",
            "start": 90.0, "end": 99.0, "rc": 0, "count": 1,
        },
        "resource_evidence": {
            "accelerator_count": 1, "device_models": ["nvidia l20"],
            "source": "nvidia-smi",
            "caveat": "precision, seed, and model scale are contract-frozen, not host-observed",
        },
        "actual_controls": {
            "version": 1,
            "execution": {"argv": argv, "cwd": ".", "source_paths": ["train.py"]},
            "immutable_file_hashes": {
                "data_config.txt": "sha256:" + "3" * 64,
                "evaluate.py": "sha256:" + "4" * 64,
                "PROTOCOL.md": protocol_content_hash,
            },
            "bindings": [],
            "caveat": (
                "Host observes canonical execution argv/cwd, resolved tracked sources, immutable "
                "file hashes, and declared structured bindings; it does not observe arbitrary "
                "internal runtime state or prove mathematical equivalence of undeclared "
                "batch/model/seed/precision controls."
            ),
        },
        "untracked_files": {
            "before": {},
            "after": {
                "ar_result.json": {
                    "kind": "regular", "sha256": "sha256:" + "8" * 64,
                    "size": 100, "executable": False,
                },
                "metrics.json": {
                    "kind": "regular", "sha256": "sha256:" + "7" * 64,
                    "size": 42, "executable": False,
                },
            },
            "added_allowed": ["ar_result.json", "metrics.json"],
            "added_other": [],
        },
        "metric_evidence": {
            "source": "host_parser",
            "stream": parser["stream"],
            "regex": parser["regex"],
            "group": parser["group"],
            "metric": metric,
            "value": value,
        },
    }


def _valid_live_artifact() -> dict:
    from autoresearch.experiments.repo_improve.benchmark_contract import contract_fingerprint

    attestation = _valid_host_attestation()
    measurement_spec = attestation["measurement_spec"]
    contract_source = [
        {key: value for key, value in claim.items()
         if key not in {"authority", "role", "identity_sha256"}}
        for claim in attestation["official_source_evidence"]
    ]
    contract = {
        "version": 2,
        "objective": {"metric": "loss", "direction": "minimize"},
        "protocol": {
            "kind": "quality_at_fixed_time", "budget": {"wall_clock_s": 10},
            "timing_scope": "process", "timer_authority": "host",
        },
        "controls": {
            "resources": {"accelerator_count": 1},
            "data": {"immutable_paths": ["data_config.txt"]},
            "evaluator": {
                "immutable_paths": ["evaluate.py", "PROTOCOL.md"],
                "output_paths": ["metrics.json"],
                "measurement_spec": measurement_spec,
                "official_source_evidence": contract_source,
            },
            "workload": {},
            "treatment": {"allowed_paths": ["train.py"]},
        },
        "allowed_treatment_axes": ["implementation"],
        "observed_outcomes": {"quality": {"metric": "loss", "value": 1.0}},
        "verified_evidence": {
            "source": "host_remeasurement", "artifact": attestation["attempt_id"],
            "observations": {
                "quality": {"metric": "loss", "value": 1.0},
                "stdout_sha256": attestation["stdout_sha256"],
                "stderr_sha256": attestation["stderr_sha256"],
                "immutable_file_hashes": copy.deepcopy(attestation["immutable_file_hashes"]),
                "protocol_evidence": copy.deepcopy(attestation["protocol_evidence"]),
                "official_source_evidence": copy.deepcopy(
                    attestation["official_source_evidence"]),
                "trusted_probe_evidence": copy.deepcopy(
                    attestation["trusted_probe_evidence"]),
                "artifact_evidence": copy.deepcopy(attestation["artifact_evidence"]),
                "agent_attempt_evidence": copy.deepcopy(
                    attestation["agent_attempt_evidence"]),
                "resource_evidence": copy.deepcopy(attestation["resource_evidence"]),
                "actual_controls": copy.deepcopy(attestation["actual_controls"]),
                "untracked_files": copy.deepcopy(attestation["untracked_files"]),
            },
        },
        "provenance": [{"source": "official_harness", "detail": "fixture"}],
    }
    return {
        "source_kind": "live_repo", "reproduced": True, "metric_value": 1.0,
        "repo_url": "https://github.com/example/project", "repo_commit": "abc123",
        "metric_name": "loss", "lower_is_better": True, "fixed_config": "batch=4",
        "environment_fingerprint": "sha256:env", "benchmark_contract": contract,
        "contract_fingerprint": contract_fingerprint(contract),
        "host_attestation": attestation,
    }


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
        self.emitter_name = "RunnerAgent"
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


class _RaisingBackend:
    def run_baseline(self, benchmark_spec: dict, verifier_spec: dict) -> dict:
        raise RuntimeError("backend blew up")


@pytest.fixture()
def store(tmp_path):
    s = SpecStore(tmp_path / "store")
    yield s
    s.close()


def _seed_reproduction_inputs(store: SpecStore, session_id: str) -> dict[str, str]:
    """Seed the store with the BenchmarkSpec + VerifierSpec RunnerAgent reads
    for ReproducingBaseline, as EvaluatorAgent would; return their refs."""
    benchmark_spec_ref = store.put(
        kind="benchmark_spec",
        session_id=session_id,
        key="v1",
        obj=VALID_BENCHMARK_SPEC,
        emitter="EvaluatorAgent",
    )
    verifier_spec_ref = store.put(
        kind="verifier_spec",
        session_id=session_id,
        key="v1",
        obj=VALID_VERIFIER_SPEC,
        emitter="EvaluatorAgent",
    )
    return {
        "benchmark_spec_ref": benchmark_spec_ref,
        "verifier_spec_ref": verifier_spec_ref,
    }


# ---------------------------------------------------------------------------
# (a) baseline.reproduction_requested -> baseline.reproduced {verdict:
# "reproduced"}, ref resolves to a schema-valid BaselineResult with
# reproduced==True.
# ---------------------------------------------------------------------------


async def test_reproduction_requested_matching_metric_publishes_reproduced(store):
    backend = FakeRunnerBackend(metric_value=0.505, exit_code=0)
    handler = RunnerAgentHandler(spec_store=store, backend=backend)
    client = FakeBusClient()
    handler.register(client)

    session_id = str(uuid.uuid4())
    refs = _seed_reproduction_inputs(store, session_id)

    env = _make_envelope(
        "baseline.reproduction_requested", session_id, refs, "ResearchOrchestrator"
    )

    await handler.handle_reproduction_requested(client, env)

    reproduced = [p for p in client.published if p.event_type == "baseline.reproduced"]
    assert len(reproduced) == 1
    payload = reproduced[0].payload
    assert payload["verdict"] == "reproduced"
    assert payload["benchmark_id"] == VALID_BENCHMARK_SPEC["benchmark_id"]

    result = store.get(payload["baseline_result_ref"])
    errors = list(BASELINE_RESULT_VALIDATOR.iter_errors(result))
    assert errors == [], [e.message for e in errors]
    assert result["reproduced"] is True


async def test_reproduction_requested_gen_mode_auto_passes(store, monkeypatch):
    """GEN honest-experiment mode: there is NO published baseline to reproduce -- the
    baseline is established by the experiment harness. So even a backend whose metric is
    FAR OFF (which would 'fail' + thrash in the benchmark path) must AUTO-PASS reproduction,
    letting the flow reach the experiment instead of dying at baseline_reproduction_thrashing."""
    monkeypatch.setenv("AUTORESEARCH_GEN_EXPERIMENT", "1")
    backend = FakeRunnerBackend(metric_value=999.0, exit_code=0)   # far off -> would 'fail'
    handler = RunnerAgentHandler(spec_store=store, backend=backend, p1_backend=object())
    client = FakeBusClient()
    handler.register(client)

    session_id = str(uuid.uuid4())
    refs = _seed_reproduction_inputs(store, session_id)
    env = _make_envelope(
        "baseline.reproduction_requested", session_id, refs, "ResearchOrchestrator")

    await handler.handle_reproduction_requested(client, env)

    reproduced = [p for p in client.published if p.event_type == "baseline.reproduced"]
    assert len(reproduced) == 1
    payload = reproduced[0].payload
    assert payload["verdict"] == "reproduced"                  # auto-pass, NOT "failed"
    assert payload["benchmark_id"] == VALID_BENCHMARK_SPEC["benchmark_id"]
    result = store.get(payload["baseline_result_ref"])
    assert list(BASELINE_RESULT_VALIDATOR.iter_errors(result)) == []   # schema-valid
    assert result["reproduced"] is True


@pytest.mark.parametrize("baseline_valid", [True, False])
async def test_repo_reproduction_runs_baseline_sandbox_without_fairness_attestation(
    store, monkeypatch, baseline_valid,
):
    """A real finite baseline advances without a contract or host attestation."""
    monkeypatch.setenv("AUTORESEARCH_GEN_EXPERIMENT", "1")
    handler = RunnerAgentHandler(
        spec_store=store, backend=object(), p1_backend=object(), p1_llm=object())
    client = FakeBusClient()
    session_id = str(uuid.uuid4())
    refs = _seed_reproduction_inputs(store, session_id)
    store.put(
        kind="research_spec", session_id=session_id, key="final",
        obj={"research_goal": "Improve https://github.com/example/nanochat"},
        emitter="ProblemModelingAgent",
    )
    calls = []

    def run_phase(sid, repo_url, question, *, execution_phase, frozen_baseline=None,
                  overrides=None, evaluation_semantics=None,
                  resource_requirements=None):
        calls.append((sid, repo_url, execution_phase, frozen_baseline, overrides,
                      evaluation_semantics))
        return ({
            "succeeded": True, "exit_code": 0, "metric_value": 1.0,
            "stdout_excerpt": "baseline=1.0", "stderr_excerpt": None,
            "environment": "gpu-worker-04", "seed": "0", "runtime_seconds": 10.0,
            "artifacts": [],
        }, {
            "baseline_metric": 1.0, "baseline_valid": baseline_valid, "method_metric": None,
            "metric_name": "loss", "lower_is_better": True,
            "config": "steps=100", "repo_commit": "abc123", "reasons": [],
        })

    monkeypatch.setattr(handler, "_run_repo_phase", run_phase)

    await handler.handle_reproduction_requested(client, _make_envelope(
        "baseline.reproduction_requested", session_id,
        {**refs, "confirmed_repo_url": "https://github.com/example/nanochat"},
        "ResearchOrchestrator",
    ))

    event = [e for e in client.published if e.event_type == "baseline.reproduced"][0]
    result = store.get(event.payload["baseline_result_ref"])
    assert event.payload["verdict"] == ("reproduced" if baseline_valid else "failed")
    assert calls == [(
        session_id, "https://github.com/example/nanochat", "baseline", None, None,
        VALID_BENCHMARK_SPEC["evaluation_semantics"])]
    assert result["reproduced"] is baseline_valid
    assert result["source_kind"] == "live_repo"
    assert result["metric_value"] == 1.0
    assert result["repo_commit"] == "abc123"
    assert result["host_attestation"] is None
    assert result["benchmark_contract"] is None
    assert list(BASELINE_RESULT_VALIDATOR.iter_errors(result)) == []


async def test_agentic_repo_reproduction_recovers_url_without_gen_mode(
    store, monkeypatch,
):
    """An explicit repository task must never fall through to the demo backend merely because
    the unrelated general-experiment feature flag is off or the auto checkpoint injected no URL."""
    monkeypatch.delenv("AUTORESEARCH_GEN_EXPERIMENT", raising=False)
    monkeypatch.setenv("AUTORESEARCH_REPO_MODE", "auto")
    monkeypatch.setenv("AUTORESEARCH_AGENTIC_IMPROVE", "1")
    handler = RunnerAgentHandler(spec_store=store, backend=_RaisingBackend())
    client = FakeBusClient()
    session_id = str(uuid.uuid4())
    refs = _seed_reproduction_inputs(store, session_id)
    store.put(
        kind="research_spec", session_id=session_id, key="final",
        obj={"research_goal": "Improve https://github.com/example/nanochat"},
        emitter="ProblemModelingAgent",
    )
    calls = []

    def run_phase(sid, repo_url, question, *, execution_phase, frozen_baseline=None,
                  overrides=None, evaluation_semantics=None,
                  resource_requirements=None):
        calls.append((sid, repo_url, execution_phase))
        return ({
            "succeeded": True, "exit_code": 0, "metric_value": 1.0,
            "stdout_excerpt": "baseline=1.0", "stderr_excerpt": None,
            "environment": "gpu-worker-04", "seed": "0", "runtime_seconds": 300.0,
            "artifacts": [],
        }, {
            "baseline_metric": 1.0, "method_metric": None,
            "metric_name": "loss", "lower_is_better": True,
            "config": "budget=300s", "repo_commit": "abc123", "reasons": [],
        })

    monkeypatch.setattr(handler, "_run_repo_phase", run_phase)
    await handler.handle_reproduction_requested(client, _make_envelope(
        "baseline.reproduction_requested", session_id, refs, "ResearchOrchestrator"))

    assert calls == [(session_id, "https://github.com/example/nanochat", "baseline")]
    event = [e for e in client.published if e.event_type == "baseline.reproduced"][0]
    result = store.get(event.payload["baseline_result_ref"])
    assert result["source_kind"] == "live_repo"
    assert result["repo_url"] == "https://github.com/example/nanochat"


async def test_agentic_repo_baseline_retry_carries_previous_failure_diagnosis(
    store, monkeypatch,
):
    """A retry must learn from the failed sandbox instead of repeating it from scratch."""
    monkeypatch.delenv("AUTORESEARCH_GEN_EXPERIMENT", raising=False)
    monkeypatch.setenv("AUTORESEARCH_REPO_MODE", "auto")
    monkeypatch.setenv("AUTORESEARCH_AGENTIC_IMPROVE", "1")
    handler = RunnerAgentHandler(spec_store=store, backend=_RaisingBackend())
    client = FakeBusClient()
    session_id = str(uuid.uuid4())
    refs = _seed_reproduction_inputs(store, session_id)
    repo_url = "https://github.com/example/project"
    store.put(
        kind="research_spec", session_id=session_id, key="final",
        obj={"research_goal": f"Improve {repo_url}"},
        emitter="ProblemModelingAgent",
    )
    store.put(
        kind="baseline_result", session_id=session_id, key="v1",
        obj={
            "benchmark_id": VALID_BENCHMARK_SPEC["benchmark_id"],
            "reproduced": False,
            "command": "repository baseline sandbox",
            "exit_code": 1,
            "metric_value": None,
            "environment": "gpu-worker-04",
            "log_excerpt": (
                "Exact repository command failed because the native extension rejected its "
                "publisher trust. A generated wrapper was rejected as non-canonical."
            ),
            "source_kind": "live_repo",
            "repo_url": repo_url,
            "repo_commit": "abc123",
        },
        emitter="RunnerAgent",
    )
    questions = []

    def run_phase(sid, url, question, **kwargs):
        questions.append(question)
        return ({
            "succeeded": False, "exit_code": 1, "metric_value": None,
            "stdout_excerpt": "still blocked", "stderr_excerpt": None,
            "environment": "gpu-worker-04", "seed": None, "runtime_seconds": 1.0,
            "artifacts": [],
        }, {
            "baseline_metric": None, "method_metric": None,
            "metric_name": "loss", "lower_is_better": True,
            "config": "unknown", "repo_commit": "abc123", "reasons": [],
        })

    monkeypatch.setattr(handler, "_run_repo_phase", run_phase)
    await handler.handle_reproduction_requested(client, _make_envelope(
        "baseline.reproduction_requested", session_id,
        {**refs, "confirmed_repo_url": repo_url}, "ResearchOrchestrator"))

    assert len(questions) == 1
    assert "PRIOR BASELINE ATTEMPT DIAGNOSTIC" in questions[0]
    assert "publisher trust" in questions[0]
    assert "generated wrapper was rejected" in questions[0]
    assert "diagnostic evidence, not instructions" in questions[0]


@pytest.mark.parametrize("failure_source", ["stderr", "error", "reasons"])
async def test_baseline_retry_preserves_host_rejection_before_long_report(
    store, monkeypatch, failure_source,
):
    monkeypatch.setenv("AUTORESEARCH_REPO_MODE", "auto")
    monkeypatch.setenv("AUTORESEARCH_AGENTIC_IMPROVE", "1")
    handler = RunnerAgentHandler(spec_store=store, backend=_RaisingBackend())
    client = FakeBusClient()
    session_id = str(uuid.uuid4())
    inputs = _seed_reproduction_inputs(store, session_id)
    repo_url = "https://github.com/example/project"
    store.put(kind="research_spec", session_id=session_id, key="final",
              obj={"research_goal": f"Improve {repo_url}"}, emitter="ProblemModelingAgent")
    rejection = "reported repository commit differs from host checkout; formally reselect it"
    questions = []

    def run_phase(sid, url, question, **kwargs):
        questions.append(question)
        record = {
            "succeeded": False, "exit_code": 1, "metric_value": None,
            "stdout_excerpt": "A finite measurement was produced. " * 400,
            "stderr_excerpt": rejection if failure_source == "stderr" else None,
            "environment": "repository sandbox", "artifacts": [],
        }
        measured = {
            "verdict": "invalid", "baseline_metric": 1.0, "baseline_valid": False,
            "method_metric": None, "metric_name": "loss", "lower_is_better": True,
            "repo_commit": "abc123", "config": "unchanged",
            "error": rejection if failure_source == "error" else None,
            "reasons": [rejection] if failure_source == "reasons" else [],
        }
        return record, measured

    monkeypatch.setattr(handler, "_run_repo_phase", run_phase)
    for _ in range(2):
        await handler.handle_reproduction_requested(client, _make_envelope(
            "baseline.reproduction_requested", session_id,
            {**inputs, "confirmed_repo_url": repo_url}, "ResearchOrchestrator"))

    events = [e for e in client.published if e.event_type == "baseline.reproduced"]
    result = store.get(events[-1].payload["baseline_result_ref"])
    assert list(BASELINE_RESULT_VALIDATOR.iter_errors(result)) == []
    assert result["reproduced"] is False
    assert result["metric_value"] == 1.0  # Preserve the observation, not a false success.
    assert rejection in result["log_excerpt"][:6000]
    assert "A finite measurement was produced." in result["log_excerpt"]
    assert rejection not in questions[0]
    assert rejection in questions[1]


async def test_agentic_repo_reproduction_discovers_named_project_url(
    store, monkeypatch,
):
    import types
    from autoresearch.experiments.repo_improve import discover as discover_module

    monkeypatch.delenv("AUTORESEARCH_GEN_EXPERIMENT", raising=False)
    monkeypatch.setenv("AUTORESEARCH_REPO_MODE", "auto")
    monkeypatch.setenv("AUTORESEARCH_AGENTIC_IMPROVE", "1")
    handler = RunnerAgentHandler(
        spec_store=store, backend=_RaisingBackend(), p1_llm=object())
    client = FakeBusClient()
    session_id = str(uuid.uuid4())
    refs = _seed_reproduction_inputs(store, session_id)
    store.put(
        kind="research_spec_draft", session_id=session_id, key="v1",
        obj={"research_goal_text": "Improve autoresearch"},
        emitter="ProblemModelingAgent",
    )
    discovered = []
    monkeypatch.setattr(
        discover_module, "discover_candidates",
        lambda question, llm, **kwargs: (
            discovered.append((question, llm))
            or [types.SimpleNamespace(repo_url="https://github.com/example/autoresearch")]),
    )
    called = []
    monkeypatch.setattr(handler, "_run_repo_phase", lambda sid, url, question, **kwargs: (
        called.append((url, question, kwargs["execution_phase"])) or ({
            "succeeded": True, "exit_code": 0, "metric_value": 1.0,
            "stdout_excerpt": "baseline=1.0", "stderr_excerpt": None,
            "environment": "gpu-worker-04", "seed": "0", "runtime_seconds": 300.0,
            "artifacts": [],
        }, {
            "baseline_metric": 1.0, "method_metric": None,
            "metric_name": "loss", "lower_is_better": True,
            "config": "budget=300s", "repo_commit": "abc123", "reasons": [],
        })))

    await handler.handle_reproduction_requested(client, _make_envelope(
        "baseline.reproduction_requested", session_id, refs, "ResearchOrchestrator"))

    assert discovered == [("Improve autoresearch", handler.p1_llm)]
    assert called == [(
        "https://github.com/example/autoresearch", "Improve autoresearch", "baseline")]


# ---------------------------------------------------------------------------
# GEN mode: prefer a PUBLISHED baseline found in the session's literature; fall
# back to the harness-established baseline when none is found (professor's rule).
# ---------------------------------------------------------------------------


@dataclass
class _FakeLLMResult:
    content: Optional[str]


class _FakeLLM:
    """Minimal LLM double: returns a single canned JSON string from .chat()."""

    def __init__(self, content: str) -> None:
        self._content = content
        self.calls: list = []

    def chat(self, *, messages, response_format=None, max_tokens=None, stream_max_s=None):
        self.calls.append(messages)
        return _FakeLLMResult(content=self._content)


def _seed_literature_brief(store: SpecStore, session_id: str) -> str:
    return store.put(
        kind="literature_brief",
        session_id=session_id,
        key="final",
        obj={
            "summary": "Prior work reports k-NN baselines on standard datasets.",
            "key_papers": [{"title": "Feature scaling for k-NN", "id_or_url": "arxiv:1"}],
            "existing_benchmarks": [{"name": "Wine (UCI)", "note": "classic heterogeneous-scale set"}],
        },
        emitter="DeepResearchAgent",
    )


async def test_gen_mode_uses_published_baseline_from_literature(store, monkeypatch):
    """When the session's LiteratureBrief reports a published baseline value, GEN mode records
    it in BaselineResult.reported_metric (a reference cross-check), not just harness-established."""
    monkeypatch.setenv("AUTORESEARCH_GEN_EXPERIMENT", "1")
    monkeypatch.setenv("AUTORESEARCH_PUBLISHED_BASELINE_LOOKUP", "1")  # lookup is opt-in now
    backend = FakeRunnerBackend(metric_value=999.0, exit_code=0)
    llm = _FakeLLM('{"found": true, "value": 0.72, "source": "Smith 2019", "note": "Wine k-NN"}')
    handler = RunnerAgentHandler(spec_store=store, backend=backend, p1_backend=object(), p1_llm=llm)
    client = FakeBusClient()
    handler.register(client)

    session_id = str(uuid.uuid4())
    inputs = _seed_reproduction_inputs(store, session_id)
    _seed_literature_brief(store, session_id)
    env = _make_envelope("baseline.reproduction_requested", session_id, inputs, "ResearchOrchestrator")

    await handler.handle_reproduction_requested(client, env)

    reproduced = [p for p in client.published if p.event_type == "baseline.reproduced"]
    result = store.get(reproduced[0].payload["baseline_result_ref"])
    assert list(BASELINE_RESULT_VALIDATOR.iter_errors(result)) == []
    assert result["reproduced"] is True
    assert result["reported_metric"] == 0.72
    assert "Published baseline found in literature" in result["environment"]
    assert len(llm.calls) == 1


async def test_gen_mode_no_published_baseline_falls_back(store, monkeypatch):
    """When the literature has no citable baseline value, GEN mode keeps the harness-established
    baseline (reported_metric null) and says so honestly."""
    monkeypatch.setenv("AUTORESEARCH_GEN_EXPERIMENT", "1")
    backend = FakeRunnerBackend(metric_value=999.0, exit_code=0)
    llm = _FakeLLM('{"found": false, "value": null}')
    handler = RunnerAgentHandler(spec_store=store, backend=backend, p1_backend=object(), p1_llm=llm)
    client = FakeBusClient()
    handler.register(client)

    session_id = str(uuid.uuid4())
    inputs = _seed_reproduction_inputs(store, session_id)
    _seed_literature_brief(store, session_id)
    env = _make_envelope("baseline.reproduction_requested", session_id, inputs, "ResearchOrchestrator")

    await handler.handle_reproduction_requested(client, env)

    reproduced = [p for p in client.published if p.event_type == "baseline.reproduced"]
    result = store.get(reproduced[0].payload["baseline_result_ref"])
    assert list(BASELINE_RESULT_VALIDATOR.iter_errors(result)) == []
    assert result["reproduced"] is True
    assert result.get("reported_metric") is None
    assert "no published baseline found" in result["environment"]


# ---------------------------------------------------------------------------
# (b) far-off metric -> baseline.reproduced {verdict: "failed"}, BaselineResult
# reproduced==False.
# ---------------------------------------------------------------------------


async def test_reproduction_requested_metric_far_off_publishes_failed_verdict(store):
    backend = FakeRunnerBackend(metric_value=0.9, exit_code=0)
    handler = RunnerAgentHandler(spec_store=store, backend=backend)
    client = FakeBusClient()
    handler.register(client)

    session_id = str(uuid.uuid4())
    refs = _seed_reproduction_inputs(store, session_id)

    env = _make_envelope(
        "baseline.reproduction_requested", session_id, refs, "ResearchOrchestrator"
    )

    await handler.handle_reproduction_requested(client, env)

    reproduced = [p for p in client.published if p.event_type == "baseline.reproduced"]
    assert len(reproduced) == 1
    payload = reproduced[0].payload
    assert payload["verdict"] == "failed"

    result = store.get(payload["baseline_result_ref"])
    errors = list(BASELINE_RESULT_VALIDATOR.iter_errors(result))
    assert errors == [], [e.message for e in errors]
    assert result["reproduced"] is False


async def test_reproduction_requested_nonzero_exit_publishes_failed_verdict(store):
    backend = FakeRunnerBackend(metric_value=0.5, exit_code=1)
    handler = RunnerAgentHandler(spec_store=store, backend=backend)
    client = FakeBusClient()
    handler.register(client)

    session_id = str(uuid.uuid4())
    refs = _seed_reproduction_inputs(store, session_id)

    env = _make_envelope(
        "baseline.reproduction_requested", session_id, refs, "ResearchOrchestrator"
    )

    await handler.handle_reproduction_requested(client, env)

    reproduced = [p for p in client.published if p.event_type == "baseline.reproduced"]
    assert len(reproduced) == 1
    assert reproduced[0].payload["verdict"] == "failed"

    result = store.get(reproduced[0].payload["baseline_result_ref"])
    assert result["reproduced"] is False


# ---------------------------------------------------------------------------
# (c) provenance -- both specs are read from the store, never from the event
# payload (which carries only refs).
# ---------------------------------------------------------------------------


async def test_reproduction_requested_reads_inputs_from_store_not_payload(store):
    backend = FakeRunnerBackend(metric_value=0.505, exit_code=0)
    handler = RunnerAgentHandler(spec_store=store, backend=backend)
    client = FakeBusClient()
    handler.register(client)

    session_id = str(uuid.uuid4())
    refs = _seed_reproduction_inputs(store, session_id)

    env = _make_envelope(
        "baseline.reproduction_requested", session_id, refs, "ResearchOrchestrator"
    )
    assert set(env.payload.keys()) == {"benchmark_spec_ref", "verifier_spec_ref"}

    await handler.handle_reproduction_requested(client, env)

    reproduced = [p for p in client.published if p.event_type == "baseline.reproduced"]
    assert len(reproduced) == 1


# ---------------------------------------------------------------------------
# (d) backend raises -> agent.failed, transient classification, never crashes.
# ---------------------------------------------------------------------------


async def test_reproduction_requested_backend_raises_publishes_agent_failed(store):
    handler = RunnerAgentHandler(spec_store=store, backend=_RaisingBackend())
    client = FakeBusClient()
    handler.register(client)

    session_id = str(uuid.uuid4())
    refs = _seed_reproduction_inputs(store, session_id)

    env = _make_envelope(
        "baseline.reproduction_requested", session_id, refs, "ResearchOrchestrator"
    )

    await handler.handle_reproduction_requested(client, env)

    failed = [p for p in client.published if p.event_type == "agent.failed"]
    assert len(failed) == 1
    assert failed[0].session_id == session_id
    assert failed[0].causation_id == env.event_id
    assert failed[0].payload["agent_name"] == "RunnerAgent"
    assert failed[0].payload["reason_classification"] == "transient"
    assert failed[0].payload["underlying_reason"]

    assert not [p for p in client.published if p.event_type == "baseline.reproduced"]


# ---------------------------------------------------------------------------
# (e) causation_id set on the success path.
# ---------------------------------------------------------------------------


async def test_reproduction_requested_causation_id_set(store):
    backend = FakeRunnerBackend(metric_value=0.505, exit_code=0)
    handler = RunnerAgentHandler(spec_store=store, backend=backend)
    client = FakeBusClient()
    handler.register(client)

    session_id = str(uuid.uuid4())
    refs = _seed_reproduction_inputs(store, session_id)

    env = _make_envelope(
        "baseline.reproduction_requested", session_id, refs, "ResearchOrchestrator"
    )

    await handler.handle_reproduction_requested(client, env)

    reproduced = [p for p in client.published if p.event_type == "baseline.reproduced"]
    assert len(reproduced) == 1
    assert reproduced[0].causation_id == env.event_id


# ---------------------------------------------------------------------------
# register() wires the single subscription.
# ---------------------------------------------------------------------------


def test_register_wires_subscription(store):
    handler = RunnerAgentHandler(spec_store=store, backend=FakeRunnerBackend())
    client = FakeBusClient()
    handler.register(client)

    patterns = [pattern for pattern, _ in client.subscriptions]
    assert "baseline.reproduction_requested" in patterns


# ---------------------------------------------------------------------------
# (f) build_runner_service with no backend arg wires a FakeRunnerBackend --
# proves the safe default (never a real backend).
# ---------------------------------------------------------------------------


def test_build_runner_service_defaults_to_fake_runner_backend(store):
    client = FakeBusClient()

    handler = build_runner_service(client=client, spec_store=store)

    assert isinstance(handler.backend, FakeRunnerBackend)
    patterns = [pattern for pattern, _ in client.subscriptions]
    assert "baseline.reproduction_requested" in patterns


# --- repo-baseline GPU branch: result mapping (repo_improve -> flow shape) ---


def test_map_repo_result_improved_maps_to_new_best():
    from autoresearch.agents.runner_agent.handler import _map_repo_result
    from autoresearch.agents.runner_agent import core as _core
    rr = {"benchmark": "nanogpt-5min-single-gpu",
          "verdict": {"verdict": "pass", "improved": True, "reason": "improved 1.01->0.99"},
          "baseline": {"metric": 1.01, "valid": True},
          "candidate": {"metric": 0.99, "valid": True},
          "report": "# repo report"}
    record, gen = _map_repo_result("improve nanochat", rr)
    assert gen["verdict"] == "new_best" and gen["accepted"] is True
    assert gen["baseline_metric"] == 1.01 and gen["method_metric"] == 0.99
    assert gen["isolated"] is True
    assert record["metric_value"] == 0.99 and record["succeeded"] is True
    assert _core._validate_run_record(record) == []


def test_map_repo_result_keeps_method_comparison_context_in_run_record():
    """The semantic guardian must see method-side evidence, not only a one-line report."""
    import json

    from autoresearch.agents.runner_agent.handler import _map_repo_result

    rr = {
        "benchmark": "generic-repo-benchmark",
        "verdict": {"verdict": "fail", "improved": False, "reason": "not better"},
        "baseline": {"metric": 1.0, "valid": True},
        "candidate": {"metric": 1.1, "valid": True},
        "report": "candidate completed",
        "config": {"device": "one accelerator", "budget_seconds": 300},
        "changes_summary": ["changed the optimizer"],
        "rationale": "test an alternative update rule",
        "bottleneck": "optimization",
        "setting_evidence": [{"setting": "budget_seconds", "actual": 300}],
        "comparison_audit": {
            "conditions_held_constant": ["same evaluator"],
            "method_budget_seconds": 300.1,
        },
    }

    record, _ = _map_repo_result("generic repository improvement", rr)

    marker = "REPO_COMPARISON_CONTEXT_JSON:\n"
    assert record["stdout_excerpt"].startswith(marker)
    context = json.loads(record["stdout_excerpt"][len(marker):])
    assert context == {
        "report": "candidate completed",
        "method_config": {"device": "one accelerator", "budget_seconds": 300},
        "method_changes": ["changed the optimizer"],
        "rationale": "test an alternative update rule",
        "bottleneck": "optimization",
        "setting_evidence": [{"setting": "budget_seconds", "actual": 300}],
        "comparison_audit": {
            "conditions_held_constant": ["same evaluator"],
            "method_budget_seconds": 300.1,
        },
    }


def test_map_repo_result_not_improved_maps_to_valid_not_best():
    from autoresearch.agents.runner_agent.handler import _map_repo_result
    rr = {"benchmark": "b", "verdict": {"verdict": "fail", "improved": False, "reason": "no improvement"},
          "baseline": {"metric": 1.01, "valid": True}, "candidate": {"metric": 1.02, "valid": True}, "report": "r"}
    _, gen = _map_repo_result("q", rr)
    assert gen["verdict"] == "valid_not_best" and gen["accepted"] is False
    assert gen["method_metric"] == 1.02


def test_map_repo_result_invalid_candidate_maps_to_invalid():
    from autoresearch.agents.runner_agent.handler import _map_repo_result
    rr = {"benchmark": "b", "verdict": {"verdict": "fail", "improved": False, "reason": "crashed"},
          "baseline": {"metric": 1.0, "valid": True}, "candidate": {"metric": None, "valid": False}, "report": "r"}
    _, gen = _map_repo_result("q", rr)
    assert gen["verdict"] == "invalid" and gen["accepted"] is False


@pytest.mark.parametrize("config", [None, {"seed": 1}, "seed=17; repeats=3"])
def test_map_repo_result_does_not_invent_seed_zero(config):
    from autoresearch.agents.runner_agent.handler import _map_repo_result

    record, gen = _map_repo_result("generic repository task", {
        "baseline": {"metric": 1.0, "valid": True},
        "candidate": {"metric": 0.9, "valid": True},
        "verdict": {"improved": True}, "config": config,
    })

    assert record["seed"] is None
    assert gen["config"] == config
    assert gen["method_metric"] == 0.9


def test_map_repo_result_records_transport_error_as_failed_execution():
    from autoresearch.agents.runner_agent.handler import _map_repo_result

    error = "Remote conversation ended with error"
    record, gen = _map_repo_result("generic repository task", {
        "error": error, "report": error,
    })
    assert record["succeeded"] is False
    assert record["exit_code"] != 0
    assert record["stderr_excerpt"] == error
    assert gen["diagnosis"] == "execution_failure"
    assert gen["diagnosis_evidence"] == [error]
    assert gen["reasons"] == [error]
    assert gen["verdict"] == "invalid"


def test_map_repo_result_keeps_completed_negative_separate_from_transport_failure():
    from autoresearch.agents.runner_agent.handler import _map_repo_result

    record, gen = _map_repo_result("generic repository task", {
        "baseline": {"metric": 1.0},
        "candidate": {"metric": 2.0, "valid": True},
        "verdict": {"improved": False},
    })
    assert record["succeeded"] is True
    assert record["exit_code"] == 0
    assert gen["verdict"] == "valid_not_best"
    assert gen["diagnosis"] is None


def test_map_repo_result_preserves_execution_evidence_without_explicit_diagnosis():
    from autoresearch.agents.runner_agent.handler import _map_repo_result

    evidence = ["workspace: /run/candidate", "attempt_ledger: /run/attempts.jsonl"]
    _, gen = _map_repo_result("generic repository task", {
        "error": "no result", "diagnosis_evidence": evidence,
    })
    assert gen["diagnosis"] == "execution_failure"
    assert gen["diagnosis_evidence"] == [*evidence, "no result"]


def test_map_repo_result_does_not_duplicate_existing_execution_error():
    from autoresearch.agents.runner_agent.handler import _map_repo_result

    _, gen = _map_repo_result("generic repository task", {
        "error": "no result", "diagnosis_evidence": ["no result"],
    })
    assert gen["diagnosis_evidence"] == ["no result"]


def test_map_repo_result_preserves_explicit_diagnosis_and_evidence():
    """Catch the phase finalizer's trusted reason being lost before MemoryAgent routing."""
    from autoresearch.agents.runner_agent.handler import _map_repo_result
    rr = {
        "benchmark": "b",
        "verdict": {"verdict": "fail", "improved": False,
                    "reason": "candidate contract violation: controls changed"},
        "baseline": {"metric": 1.0, "valid": True},
        "candidate": {"metric": None, "valid": False},
        "diagnosis": "candidate_contract_violation", "report": "r",
    }

    _, gen = _map_repo_result("q", rr)

    assert gen["diagnosis"] == "candidate_contract_violation"
    assert gen["diagnosis_evidence"] == ["candidate contract violation: controls changed"]


def test_map_repo_result_preserves_structured_setting_evidence():
    """Trusted memory classification needs the measured setting mismatch, not a label."""
    from autoresearch.agents.runner_agent.handler import _map_repo_result
    evidence = [{
        "setting": "batch_size", "actual": 64, "expected": 128,
        "evidence": {"source": "run_manifest", "ref": "manifest:1"},
    }]
    rr = {
        "benchmark": "b", "verdict": {"verdict": "fail", "improved": False},
        "baseline": {"metric": 1.0, "valid": True},
        "candidate": {"metric": 1.1, "valid": True},
        "setting_evidence": evidence,
    }

    _, gen = _map_repo_result("q", rr)

    assert gen["setting_evidence"] == evidence


def test_map_repo_result_preserves_fairness_artifacts_for_baseline_freeze():
    """Catch the remote baseline contract being dropped before BaselineResult persistence."""
    from autoresearch.agents.runner_agent.handler import _map_repo_result
    contract = {
        "version": 1, "benchmark_kind": "official",
        "objective": {"metric": "loss", "direction": "minimize", "mode": "quality"},
        "protocol": {"kind": "fixed_work", "budget": {"steps": 10}},
        "controls": {"workload": {"steps": 10}},
        "allowed_treatment_axes": ["implementation"],
        "provenance": [{"source": "official_harness", "detail": "canonical"}],
    }
    rr = {
        "benchmark": "b", "metric": "loss", "lower_is_better": True,
        "repo_commit": "abc", "config": "steps=10", "benchmark_contract": contract,
        "contract_fingerprint": "sha256:abc",
        "verdict": {"verdict": "pass", "improved": False, "reason": "baseline"},
        "baseline": {"metric": 1.0, "valid": True},
        "candidate": {"metric": None, "valid": False}, "report": "baseline",
    }

    _, gen = _map_repo_result("q", rr)

    assert gen["benchmark_contract"] == contract
    assert gen["contract_fingerprint"] == "sha256:abc"


def test_build_repo_bundle_surfaces_the_winning_deliverable():
    # the GPU repo path must expose a concrete 产物: winning train.py + change list + report.
    from autoresearch.agents.runner_agent.handler import _build_repo_bundle
    rr = {"benchmark": "nanogpt-5min-single-gpu",
          "lower_is_better": True,
          "verdict": {"verdict": "pass", "improved": True, "reason": "改进且复现 1.017->1.009"},
          "baseline": {"metric": 1.01735, "valid": True},
          "candidate": {"metric": 1.009543, "valid": True},
          "report": "# repo honest report\nrepo_url: https://github.com/karpathy/nanoGPT",
          "patch": {"rationale": "warmdown 0.5->0.2; depth 8->10",
                    "edits": [{"file": "train.py", "find": "DEPTH = 8", "replace": "DEPTH = 10"},
                              {"file": "config/gpt.py", "find": "WARMDOWN_RATIO = 0.5",
                               "replace": "WARMDOWN_RATIO = 0.2"}],
                    "patched_sources": {"train.py": "DEPTH = 10\n# ...full train.py...",
                                        "config/gpt.py": "WARMDOWN_RATIO = 0.2\n"}}}
    b = _build_repo_bundle("improve nanoGPT 5-min speedrun", rr)
    assert b["verdict"] == "pass"
    assert b["lower_is_better"] is True
    assert b["baseline_metric"] == 1.01735 and b["method_metric"] == 1.009543
    files = b["files"]
    # the WINNING source of EACH changed file, under its REAL path (multi-file, no hard-coded name)
    assert files["train.py"] == "DEPTH = 10\n# ...full train.py..."
    assert files["config/gpt.py"] == "WARMDOWN_RATIO = 0.2\n"
    # the change list shows the real diff across files (not a 'no-op' placeholder)
    assert "DEPTH = 8" in files["CHANGES.md"] and "DEPTH = 10" in files["CHANGES.md"]
    assert "config/gpt.py" in files["CHANGES.md"] and "warmdown 0.5->0.2" in files["CHANGES.md"]
    # the repo's honest report + a README with metrics + repro steps
    assert "repo honest report" in files["report.md"]
    assert "1.01735" in files["README.md"] and "复现" in files["README.md"]


def test_build_repo_bundle_tolerates_missing_patch():
    # a failed/empty patch must not crash bundle assembly (best-effort deliverable)
    from autoresearch.agents.runner_agent.handler import _build_repo_bundle
    rr = {"benchmark": "b", "verdict": {"verdict": "fail", "improved": False, "reason": "基线未跑通"},
          "baseline": {"metric": None, "valid": False}, "candidate": None, "report": "r"}
    b = _build_repo_bundle("q", rr)
    assert "train.py" not in b["files"]          # nothing to ship, but no crash
    assert "CHANGES.md" in b["files"] and "README.md" in b["files"]


def test_confirmed_repo_url_routes_to_repo_improve(monkeypatch):
    # a user-CONFIRMED repo (picked from auto-discovered candidates) routes the experiment to
    # repo-improve on THAT repo -- even when the question named no repo (self-write otherwise).
    from autoresearch.agents.runner_agent.handler import RunnerAgentHandler
    from autoresearch.spec_store import refs
    monkeypatch.setenv("AUTORESEARCH_REPO_REMOTE", "1")
    sid = "s-conf"

    class _Store:
        def get(self, ref):
            if ref == refs.encode("research_spec", sid, "final"):
                return {"research_goal": "make k-NN classification better on digits"}
            raise KeyError(ref)

    h = RunnerAgentHandler(spec_store=_Store(), backend=None, p1_llm=None)
    captured = {}
    # SPLIT: REMOTE mode -> GPU task -> DISPATCHES to the remote GPU host (not the local box)
    h._run_repo_experiment_remote = lambda session_id, question, overrides=None, repo_url=None: (
        captured.update(q=question, overrides=overrides, repo_url=repo_url)
        or ({"environment": "x"}, {"verdict": "new_best"}))

    rec, gen = h._run_gen_experiment(
        sid, confirmed_repo_url="https://github.com/owner/cool-repo",
        confirmed_boundary={"run_cmd": "python run.py -i data", "budget_s": 900})
    assert "https://github.com/owner/cool-repo" in captured["q"]   # routed to repo-improve w/ URL
    assert captured["repo_url"] == "https://github.com/owner/cool-repo"
    assert captured["overrides"] == {"run_cmd": "python run.py -i data", "budget_s": 900}  # boundary threaded
    assert gen["verdict"] == "new_best"


def test_repo_mode_auto_routes_by_needs_gpu(monkeypatch):
    # AUTORESEARCH_REPO_MODE=auto -> MOSS's _needs_gpu decides GPU/CPU PER TASK, not a manual flag.
    # While gpu-worker-02 is down BOTH run locally; a GPU task claims an L20 card (gpu=True), a CPU task
    # runs on CPU (gpu=False). That flag is what distinguishes them.
    from autoresearch.agents.runner_agent.handler import RunnerAgentHandler
    from autoresearch.spec_store import refs
    monkeypatch.delenv("AUTORESEARCH_REPO_REMOTE", raising=False)
    monkeypatch.delenv("AUTORESEARCH_REPO_LOCAL", raising=False)
    monkeypatch.setenv("AUTORESEARCH_REPO_MODE", "auto")
    sid = "s-auto"

    class _Store:
        def get(self, ref):
            if ref == refs.encode("research_spec", sid, "final"):
                return {"research_goal": "improve a repo"}
            raise KeyError(ref)

    # SPLIT: a GPU task DISPATCHES to the remote GPU host; a CPU task runs LOCALLY (gpu=False).
    for needs_gpu, expect_where in ((True, "remote"), (False, "local")):
        h = RunnerAgentHandler(spec_store=_Store(), backend=None, p1_llm=None)
        h._needs_gpu = lambda q, url, _v=needs_gpu: _v
        hit = {}
        h._run_repo_experiment_remote = lambda s, q, overrides=None, repo_url=None: (
            hit.__setitem__("where", "remote") or ({"environment": "x"}, {"verdict": "new_best"}))
        h._run_repo_experiment_local = lambda s, q, overrides=None, gpu=False, repo_url=None: (
            hit.update(where="local", gpu=gpu) or ({"environment": "x"}, {"verdict": "new_best"}))
        h._run_gen_experiment(sid, confirmed_repo_url="https://github.com/owner/repo")
        assert hit["where"] == expect_where   # GPU -> remote dispatch; CPU -> local
        if expect_where == "local":
            assert hit["gpu"] is False        # CPU task claims no card


@pytest.mark.parametrize("gpu", [False, True])
def test_local_agentic_phase_receives_confirmed_boundary(tmp_path, monkeypatch, gpu):
    """Catch local fallback applying a different fairness source than the DGX path."""
    from autoresearch.experiments.repo_improve import agentic_improve
    from autoresearch.experiments.repo_improve.runner import RunResult

    monkeypatch.setenv("AUTORESEARCH_AGENTIC_IMPROVE", "1")
    monkeypatch.setenv("AUTORESEARCH_WEB_WORKDIR", str(tmp_path))
    captured = {}

    def fake_run(request, **kwargs):
        captured.update(kwargs)
        return {
            "benchmark": "repo", "metric": "loss", "lower_is_better": True,
            "repo_commit": "abc", "config": "steps=10",
            "baseline": RunResult("loss", 1.0, True),
            "candidate": RunResult("loss", None, False),
            "verdict": {"verdict": "pass", "improved": False, "reason": "baseline"},
            "report": "baseline",
        }

    monkeypatch.setattr(agentic_improve, "run_agentic_improve", fake_run)
    runner = RunnerAgentHandler(spec_store=None, backend=object())
    boundary = {"budget_s": 300, "accelerator_count": 1}
    semantics = {
        "comparison_statement": (
            "Under the repository protocol, compare loss while allowing method changes."
        ),
        "comparison_mode": "repository_defined",
        "conditions_to_hold_constant": ["dataset and evaluator"],
        "method_freedom": ["method implementation"],
    }

    record, _ = runner._run_repo_experiment_local(
        "sid", "Improve https://github.com/example/repo", overrides=boundary,
        repo_url="https://github.com/example/repo", execution_phase="baseline",
        evaluation_semantics=semantics, gpu=gpu,
    )

    assert captured["confirmed_boundary"] == boundary
    assert captured["evaluation_semantics"] == semantics
    assert captured["enforce_fairness"] is False
    expected_where = (f"local GPU (card {captured['cuda_device']})"
                      if gpu else "local CPU")
    assert record["environment"].endswith(expected_where)


def test_local_agentic_phase_preserves_component_plan_exhaustion(tmp_path, monkeypatch):
    """The orchestrator needs the structured code to revise the exhausted component."""
    from autoresearch.agents.runner_agent import core
    from autoresearch.experiments.repo_improve import agentic_improve

    monkeypatch.setenv("AUTORESEARCH_AGENTIC_IMPROVE", "1")
    monkeypatch.setenv("AUTORESEARCH_WEB_WORKDIR", str(tmp_path))

    def fake_run(*args, **kwargs):
        raise core.ComponentPlanExhaustedError(
            "component plan has no ready node; exhausted: train-loop")

    monkeypatch.setattr(agentic_improve, "run_agentic_improve", fake_run)
    runner = RunnerAgentHandler(spec_store=None, backend=object())

    with pytest.raises(core.ComponentPlanExhaustedError):
        runner._run_repo_experiment_local(
            "sid", "Improve https://github.com/example/repo",
            repo_url="https://github.com/example/repo",
            execution_phase="method",
        )


def test_local_repo_same_session_runs_do_not_overlap(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    import threading
    import time

    monkeypatch.setenv("AUTORESEARCH_WEB_WORKDIR", str(tmp_path))
    runner = RunnerAgentHandler(spec_store=None, backend=object())
    start = threading.Barrier(3)
    guard = threading.Lock()
    active = peak = 0

    def fake_run(*args, **kwargs):
        nonlocal active, peak
        with guard:
            active += 1
            peak = max(peak, active)
        time.sleep(0.05)
        with guard:
            active -= 1
        return ({}, {})

    monkeypatch.setattr(runner, "_run_repo_experiment_local_unlocked", fake_run)

    def call():
        start.wait()
        return runner._run_repo_experiment_local("same-session", "question")

    with ThreadPoolExecutor(max_workers=2) as pool:
        left = pool.submit(call)
        right = pool.submit(call)
        start.wait()
        assert left.result(timeout=5) == ({}, {})
        assert right.result(timeout=5) == ({}, {})

    assert peak == 1


def test_gpu_card_pool_parallel_distinct_cards():
    # two concurrent GPU tasks each claim a DIFFERENT L20 card (parallel, one training per card);
    # a further task would block until one frees. Pool is restored for other tests.
    from autoresearch.agents.runner_agent.handler import _GPU_CARD_POOL, _LOCAL_GPU_CARDS
    assert len(_LOCAL_GPU_CARDS) >= 2          # two L20 cards -> two parallel slots
    n0 = _GPU_CARD_POOL.qsize()
    a = _GPU_CARD_POOL.get()
    b = _GPU_CARD_POOL.get()
    try:
        assert a != b and a in _LOCAL_GPU_CARDS and b in _LOCAL_GPU_CARDS  # distinct real cards
        assert _GPU_CARD_POOL.qsize() == n0 - 2   # two claimed -> two fewer free (3rd would wait)
    finally:
        _GPU_CARD_POOL.put(a)
        _GPU_CARD_POOL.put(b)
    assert _GPU_CARD_POOL.qsize() == n0


def test_needs_gpu_heuristic_without_llm():
    # no LLM -> keyword heuristic: CUDA/train signals -> GPU; sklearn/tabular -> CPU.
    from autoresearch.agents.runner_agent.handler import RunnerAgentHandler
    h = RunnerAgentHandler(spec_store=None, backend=None, p1_llm=None)
    assert h._needs_gpu("train a transformer with torch.compile on cuda", None) is True
    assert h._needs_gpu("improve the AcmeResearch project", None) is False
    assert h._needs_gpu("improve a scikit-learn classifier on a small tabular dataset", None) is False


@pytest.mark.parametrize("declaration", [
    "CPU only; 16 GB RAM", "4 CPU cores, 16 GiB host memory, 0 GPUs",
])
def test_frozen_resource_contract_overrides_repository_wide_gpu_guess(monkeypatch, declaration):
    import autoresearch.agents.runner_agent.handler as runner_handler
    from autoresearch.agents.runner_agent.handler import RunnerAgentHandler

    monkeypatch.setenv("AUTORESEARCH_REPO_MODE", "auto")
    monkeypatch.setattr(runner_handler, "_manifest_gpu_requirement", lambda *_: None)
    h = RunnerAgentHandler(spec_store=None, backend=None, p1_llm=None)
    h._needs_gpu = lambda *_: (_ for _ in ()).throw(
        AssertionError("an explicit resource contract must be authoritative"))
    hit = {}
    h._run_repo_experiment_remote = lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("CPU-only work must not be dispatched remotely"))
    h._run_repo_experiment_local = lambda *a, **k: (
        hit.update(gpu=k.get("gpu")) or ({"environment": "x"}, {"verdict": "pass"}))

    h._run_repo_phase(
        "sid", "https://github.com/example/multi-task-repo",
        "Improve one task in a repository that also contains CUDA benchmarks",
        execution_phase="baseline", resource_requirements=declaration,
    )

    assert hit["gpu"] is False


@pytest.mark.parametrize("question", [
    "Run the selected official task. CPU resources follow its task manifest.",
    "运行所选任务；不使用GPU；environment.gpus=0；4个CPU核，16GiB主机内存。",
])
def test_explicit_question_resource_used_when_manifest_is_unavailable(monkeypatch, question):
    import autoresearch.agents.runner_agent.handler as runner_handler

    monkeypatch.setenv("AUTORESEARCH_REPO_MODE", "auto")
    monkeypatch.setattr(runner_handler, "_manifest_gpu_requirement", lambda *_: None)
    h = runner_handler.RunnerAgentHandler(spec_store=None, backend=None, p1_llm=None)
    h._needs_gpu = lambda *_: (_ for _ in ()).throw(
        AssertionError("an explicit resource declaration must precede repository-wide inference"))
    h._run_repo_experiment_remote = lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("an explicitly CPU task must not be dispatched remotely"))
    hit = {}
    h._run_repo_experiment_local = lambda *a, **k: (
        hit.update(gpu=k.get("gpu")) or ({"environment": "x"}, {"verdict": "pass"}))

    h._run_repo_phase(
        "sid", "https://github.com/example/multi-task-repo",
        question,
        execution_phase="baseline",
    )

    assert hit["gpu"] is False


@pytest.mark.parametrize("declaration, expected", [
    ("environment.gpus=0; CPU 4", False),
    ("environment.gpus = 1; CPU 8", True),
    ("gpus: 2; CPU 16", True),
    ("gpus=10", True),
    ("4 CPU cores, 16 GiB host memory, 0 GPUs", False),
    ("0 GPU; CUDA libraries available", False),
    ("2 GPUs; 16 CPU cores", True),
    ("CPU only; 16 GB RAM", False),
])
def test_resource_gpu_count_precedes_keyword_guess(declaration, expected):
    from autoresearch.agents.runner_agent.handler import _resource_requires_gpu

    assert _resource_requires_gpu(declaration) is expected


def test_task_manifest_gpu_requirement_parses_selected_task(monkeypatch):
    import autoresearch.agents.runner_agent.handler as runner_handler

    monkeypatch.setattr(
        runner_handler, "_github_task_manifest_index",
        lambda _url: (("cpu_task", "blob:cpu"), ("gpu_task", "blob:gpu")),
    )
    monkeypatch.setattr(
        runner_handler, "_github_manifest_text",
        lambda url: "[environment]\ngpus = 0\n" if url == "blob:cpu"
        else "[environment]\ngpus = 2\n",
    )

    assert runner_handler._manifest_gpu_requirement(
        "Improve the official cpu_task", "https://github.com/example/tasks") is False
    assert runner_handler._manifest_gpu_requirement(
        "Improve the official gpu_task", "https://github.com/example/tasks") is True


def test_failed_task_manifest_lookup_is_not_cached(monkeypatch):
    import autoresearch.agents.runner_agent.handler as runner_handler

    repo_url = "https://github.com/example/transient-manifest-lookup"
    runner_handler._GITHUB_TASK_MANIFEST_INDEX_CACHE.pop(repo_url, None)
    calls = 0

    def fetch(_url, **_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("temporary GitHub failure")
        return json.dumps({
            "tree": [{
                "type": "blob",
                "path": "tasks/cpu_task/task.toml",
                "url": "blob:cpu",
            }]
        })

    monkeypatch.setattr(runner_handler, "_direct_url_text", fetch)

    assert runner_handler._github_task_manifest_index(repo_url) == ()
    assert runner_handler._github_task_manifest_index(repo_url) == (("cpu_task", "blob:cpu"),)
    assert calls == 2


def test_failed_task_manifest_blob_lookup_is_not_cached(monkeypatch):
    import autoresearch.agents.runner_agent.handler as runner_handler

    blob_url = "https://api.github.com/repos/example/tasks/git/blobs/cpu"
    runner_handler._GITHUB_MANIFEST_TEXT_CACHE.pop(blob_url, None)
    calls = 0

    def fetch(_url, **_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("temporary GitHub blob failure")
        return json.dumps({
            "content": base64.b64encode(b"[environment]\ngpus = 0\n").decode(),
        })

    monkeypatch.setattr(runner_handler, "_direct_url_text", fetch)

    assert runner_handler._github_manifest_text(blob_url) == ""
    assert runner_handler._github_manifest_text(blob_url) == "[environment]\ngpus = 0\n"
    assert calls == 2


def test_claim_cards_fails_fast_when_remote_pool_is_empty(monkeypatch):
    import queue
    import autoresearch.agents.runner_agent.handler as runner_handler

    monkeypatch.setattr(runner_handler, "_REMOTE_GPU_CARDS", [])
    monkeypatch.setattr(runner_handler, "_REMOTE_GPU_POOL", queue.Queue())

    with pytest.raises(runner_handler.core.RunnerParseError, match="no remote GPU cards"):
        runner_handler._claim_cards("a task incorrectly routed to GPU")


@pytest.mark.parametrize("phase", ["baseline", "method"])
@pytest.mark.parametrize("accelerators", [0, 1])
def test_confirmed_accelerator_boundary_keeps_both_phases_on_the_right_target(
        monkeypatch, phase, accelerators):
    import autoresearch.agents.runner_agent.handler as runner_handler

    monkeypatch.setenv("AUTORESEARCH_REPO_MODE", "auto")
    monkeypatch.setenv("AUTORESEARCH_AGENTIC_IMPROVE", "1")
    monkeypatch.setenv("AUTORESEARCH_REPO_SPLIT", "1")
    monkeypatch.setattr(runner_handler, "_manifest_gpu_requirement", lambda *_: (
        _ for _ in ()).throw(AssertionError("confirmed boundary needs no network inference")))
    h = runner_handler.RunnerAgentHandler(spec_store=None, backend=None, p1_llm=None)
    hit = {}
    h._run_repo_experiment_remote = lambda *a, **k: (
        hit.update(target="remote") or ({}, {}))
    h._run_repo_experiment_local = lambda *a, **k: (
        hit.update(target="local", gpu=k["gpu"]) or ({}, {}))

    h._run_repo_phase("sid", "https://github.com/example/repo", "GPU training topic",
        execution_phase=phase, overrides={"accelerator_count": accelerators},
        resource_requirements="1 GPU" if accelerators == 0 else "CPU only")

    assert hit["target"] == ("remote" if accelerators else "local")
    if not accelerators:
        assert hit["gpu"] is False


def test_task_manifest_overrides_inferred_resource_text(monkeypatch):
    import autoresearch.agents.runner_agent.handler as runner_handler

    monkeypatch.setenv("AUTORESEARCH_REPO_MODE", "auto")
    monkeypatch.setattr(runner_handler, "_manifest_gpu_requirement", lambda *_: False)
    h = runner_handler.RunnerAgentHandler(spec_store=None, backend=None, p1_llm=None)
    h._needs_gpu = lambda *_: (_ for _ in ()).throw(
        AssertionError("manifested task resources must be authoritative"))
    h._run_repo_experiment_remote = lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("a manifested CPU task must not be remotely dispatched"))
    hit = {}
    h._run_repo_experiment_local = lambda *a, **k: (
        hit.update(gpu=k.get("gpu")) or ({"environment": "x"}, {"verdict": "pass"}))

    h._run_repo_phase(
        "sid", "https://github.com/example/multi-task-repo", "Improve cpu_task",
        execution_phase="baseline", resource_requirements="1x A100 GPU",
    )

    assert hit["gpu"] is False


def test_frozen_gpu_resource_contract_routes_remote(monkeypatch):
    from autoresearch.agents.runner_agent.handler import RunnerAgentHandler

    monkeypatch.setenv("AUTORESEARCH_REPO_MODE", "auto")
    monkeypatch.setenv("AUTORESEARCH_AGENTIC_IMPROVE", "1")
    h = RunnerAgentHandler(spec_store=None, backend=None, p1_llm=None)
    h._needs_gpu = lambda *_: (_ for _ in ()).throw(
        AssertionError("an explicit resource contract must be authoritative"))
    hit = {}
    h._run_repo_experiment_remote = lambda *a, **k: (
        hit.update(where="remote") or ({"environment": "x"}, {"verdict": "pass"}))

    h._run_repo_phase(
        "sid", "https://github.com/example/repo", "Improve the official task",
        execution_phase="baseline", resource_requirements="1x H800 GPU",
    )

    assert hit["where"] == "remote"


def test_repo_local_flag_routes_to_local_cpu(monkeypatch):
    # AUTORESEARCH_REPO_LOCAL=1 (no REMOTE) runs the SAME confirm/discover routing but
    # in-process on CPU (_run_repo_experiment_local), not over SSH to the GPU host.
    from autoresearch.agents.runner_agent.handler import RunnerAgentHandler
    from autoresearch.spec_store import refs
    monkeypatch.delenv("AUTORESEARCH_REPO_REMOTE", raising=False)
    monkeypatch.setenv("AUTORESEARCH_REPO_LOCAL", "1")
    sid = "s-local"

    class _Store:
        def get(self, ref):
            if ref == refs.encode("research_spec", sid, "final"):
                return {"research_goal": "improve a small CPU baseline on digits"}
            raise KeyError(ref)

    h = RunnerAgentHandler(spec_store=_Store(), backend=None, p1_llm=None)
    captured = {}
    h._run_repo_experiment_remote = lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("must not call remote when only REPO_LOCAL is set"))
    h._run_repo_experiment_local = lambda session_id, question, overrides=None, gpu=False, repo_url=None: (
        captured.update(q=question, overrides=overrides, gpu=gpu) or ({"environment": "x"}, {"verdict": "new_best"}))

    rec, gen = h._run_gen_experiment(
        sid, confirmed_repo_url="https://github.com/owner/cpu-repo",
        confirmed_boundary={"run_cmd": "python train.py", "budget_s": 300})
    assert "https://github.com/owner/cpu-repo" in captured["q"]   # local repo-improve, chosen repo
    assert captured["overrides"] == {"run_cmd": "python train.py", "budget_s": 300}
    assert captured["gpu"] is False   # REPO_LOCAL -> CPU (no card claimed)
    assert gen["verdict"] == "new_best"


def test_gpu_run_lock_serializes_to_one():
    # Fix C: gpu-worker-02 card6 is single-tenant -- the GPU lock must admit exactly one training at a
    # time even under concurrent sessions.
    import threading
    import time
    from autoresearch.agents.runner_agent.handler import _GPU_RUN_LOCK

    seen = {"cur": 0, "max": 0}
    lock = threading.Lock()

    def work():
        with _GPU_RUN_LOCK:
            with lock:
                seen["cur"] += 1
                seen["max"] = max(seen["max"], seen["cur"])
            time.sleep(0.03)
            with lock:
                seen["cur"] -= 1

    threads = [threading.Thread(target=work) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert seen["max"] == 1               # strictly serialized -- never two on the card at once


def test_persist_repo_bundle_preserves_best_round():
    # in the stop-on-win loop, a later flaky round (null metrics) must NOT clobber an earlier
    # round's valid deliverable; a real win DOES replace it.
    from autoresearch.agents.runner_agent.handler import (
        RunnerAgentHandler, _bundle_rank, _GEN_BUNDLE_KIND, _RUN_RECORD_KEY)
    from autoresearch.spec_store import refs

    assert _bundle_rank({
        "verdict": "pass", "baseline_metric": 1.6,
        "method_metric": None, "candidate_valid": False,
    }) == -1
    assert _bundle_rank({
        "verdict": "fail", "baseline_metric": 1.6,
        "method_metric": 1.7, "candidate_valid": True,
    }) == 1
    assert _bundle_rank({
        "verdict": "pass", "baseline_metric": 1.6,
        "method_metric": 1.5, "candidate_valid": True,
    }) == 2

    sid = "s-preserve"
    saved = {}

    class _Store:
        def get(self, ref):
            if ref in saved:
                return saved[ref]
            raise KeyError(ref)

        def put(self, *, kind, session_id, key, obj, emitter):
            saved[refs.encode(kind, session_id, key)] = obj

    h = RunnerAgentHandler(spec_store=_Store(), backend=None, p1_llm=None)
    ref = refs.encode(_GEN_BUNDLE_KIND, sid, _RUN_RECORD_KEY)
    baseline_only = {
        "verdict": "pass", "baseline_metric": 1.6841,
        "method_metric": None, "candidate_valid": False,
    }
    good = {
        "verdict": "fail", "baseline_metric": 1.6841,
        "method_metric": 1.7, "candidate_valid": True,
    }
    dead = {
        "verdict": "fail", "baseline_metric": None,
        "method_metric": None, "candidate_valid": False,
    }
    win = {
        "verdict": "pass", "baseline_metric": 1.6,
        "method_metric": 1.5, "candidate_valid": True,
    }

    h._persist_repo_bundle(sid, baseline_only)
    assert ref not in saved
    h._persist_repo_bundle(sid, good)
    assert saved[ref] is good
    h._persist_repo_bundle(sid, dead)          # flaky later round -> preserved, not clobbered
    assert saved[ref] is good
    h._persist_repo_bundle(sid, win)           # a win replaces
    assert saved[ref] is win
    later_worse_win = {
        "verdict": "pass", "baseline_metric": 1.6,
        "method_metric": 1.55, "lower_is_better": True, "candidate_valid": True,
    }
    h._persist_repo_bundle(sid, later_worse_win)
    assert saved[ref] is win                     # same-tier win must keep the better source
    later_better_win = {**later_worse_win, "method_metric": 1.45}
    h._persist_repo_bundle(sid, later_better_win)
    assert saved[ref] is later_better_win

    higher_win = {
        "verdict": "pass", "baseline_metric": 0.5852,
        "method_metric": 0.6126, "candidate_valid": True,
    }
    saved[ref] = higher_win                      # legacy bundle without direction field
    h._persist_repo_bundle(sid, {
        **higher_win, "method_metric": 0.5981, "lower_is_better": False,
    })
    assert saved[ref] is higher_win             # observed Flux regression


def test_res_to_improve_json_maps_to_flow_shapes():
    # a local run's result dict must flatten into the same shape _map_repo_result consumes,
    # so a CPU run and a GPU run flow through evaluate -> report identically.
    from autoresearch.agents.runner_agent.handler import _map_repo_result, _res_to_improve_json

    class _RR:
        def __init__(self, m, v, e=None):
            self.metric_value, self.valid, self.error = m, v, e

    res = {
        "benchmark": "digits-knn",
        "verdict": {"verdict": "new_best", "improved": True, "reason": "acc up"},
        "baseline": _RR(0.90, True),
        "candidate": _RR(0.95, True),
        "component_attempt_id": "attempt-component-1",
        "baseline_relative_diff": "diff --git a/train.py b/train.py\n+better = True\n",
        "patch": {"rationale": "tune k", "edits": [{"find": "k=5", "replace": "k=7"}]},
        "report": "ran on cpu",
    }
    rr = _res_to_improve_json(res)
    assert rr["baseline"] == {"metric": 0.90, "valid": True, "error": None}
    assert rr["candidate"] == {"metric": 0.95, "valid": True, "error": None}
    assert rr["component_attempt_id"] == "attempt-component-1"
    assert rr["baseline_relative_diff"] == res["baseline_relative_diff"]
    record, gen = _map_repo_result("improve digits", rr)
    assert gen["verdict"] == "new_best" and gen["accepted"] is True
    assert gen["baseline_metric"] == 0.90 and gen["method_metric"] == 0.95
    assert gen["baseline_relative_diff"] == res["baseline_relative_diff"]
    assert record["metric_value"] == 0.95


def test_original_question_preserves_repo_reference_over_reinterpreted():
    # repo-improve routing must see the ORIGINAL question (explicit repo ref) rather than the
    # ReinterpretingSpec-paraphrased goal that drops it -- else the GPU repo path never fires.
    from autoresearch.agents.runner_agent.handler import RunnerAgentHandler
    from autoresearch.spec_store import refs
    sid = "sess-orig"
    final_ref = refs.encode("research_spec", sid, "final")
    reint_ref = refs.encode("research_spec", sid, "reinterpreted")

    class _Store:
        objs = {
            final_ref: {"research_goal": "Clone and improve KellerJordan/cifar10-airbench "
                                         "for higher CIFAR-10 test accuracy", "topic": "cifar"},
            reint_ref: {"research_goal": "Ablate LR schedules + augmentation on CIFAR-10 "
                                         "ResNet-18 (repo name dropped)"},
        }

        def get(self, ref):
            if ref not in self.objs:
                raise KeyError(ref)
            return self.objs[ref]

    h = RunnerAgentHandler(spec_store=_Store(), backend=None)
    # _original_question keeps the repo reference; _reconstruct_question uses the paraphrase
    assert "KellerJordan/cifar10-airbench" in h._original_question(sid)
    assert "KellerJordan/cifar10-airbench" not in h._reconstruct_question(sid)


@pytest.mark.parametrize("use_search", [False, True])
def test_repository_resolution_uses_verified_candidates_and_failure_feedback(store, monkeypatch, use_search):
    import importlib
    from types import SimpleNamespace
    from autoresearch.spec_store import refs
    from autoresearch.experiments.repo_improve.tests.test_discover import _FakeLLM
    module = importlib.import_module("autoresearch.experiments.repo_improve.discover")
    sid = str(uuid.uuid4())
    store.put(kind="research_spec_draft", session_id=sid, key="v1",
              obj={"research_goal_text": "Autonomously select a runnable baseline repository"},
              emitter="ProblemModelingAgent")
    evidence = [{"id_or_url": "https://github.com/example/runnable"}]
    store.put(kind="literature_brief", session_id=sid, key="v1",
              obj={"existing_benchmarks": [] if use_search else evidence},
              emitter="DeepResearchAgent")
    store.put(kind="baseline_result", session_id=sid, key="v1",
              obj={"reproduced": False, "repo_url": "https://github.com/example/missing",
                   "log_excerpt": "repository not found"}, emitter="RunnerAgent")
    llm = _FakeLLM([{"repos": [{"url": "https://github.com/example/missing"},
                              *([{"url": "https://github.com/example/runnable"}] if use_search else [])]}])
    monkeypatch.setattr(module, "_public_repo_status",
                        lambda url: 200 if url.endswith('/runnable') else 404)
    searches = []
    search = SimpleNamespace(search_frontier=lambda q, s: searches.append(q) or evidence)
    store.put(kind="benchmark_spec", session_id=sid, key="v1",
              obj={"name": "Task-derived algorithm benchmark"}, emitter="EvaluatorAgent")
    handler = build_runner_service(client=FakeBusClient(), spec_store=store, p1_llm=llm,
                                   repository_search_backend=search if use_search else None)
    assert handler._resolve_repo_url(sid) == "https://github.com/example/runnable"
    assert "repository not found" in llm.calls[0][1]["content"]
    assert handler._resolve_repo_url(sid, "https://github.com/example/user-choice") == \
        "https://github.com/example/user-choice"
    assert len(llm.calls) == 1
    assert len(searches) == int(use_search)
    if use_search:
        assert store.get(refs.encode("audit", sid, "repository-search"))["results"] == evidence
        assert searches == ["GitHub runnable implementation Task-derived algorithm benchmark"]


@pytest.mark.asyncio
async def test_unresolved_repository_preserves_last_execution_failure_without_nesting(store, monkeypatch):
    from autoresearch.spec_store import refs

    sid = str(uuid.uuid4())
    inputs = _seed_reproduction_inputs(store, sid)
    store.put(kind="baseline_result", session_id=sid, key="v1", obj={
        "reproduced": False, "command": "repository baseline sandbox",
        "repo_url": "https://github.com/example/incompatible",
        "log_excerpt": "Required operation is absent from the public implementation.",
    }, emitter="RunnerAgent")
    client = FakeBusClient()
    handler = RunnerAgentHandler(spec_store=store, backend=None)
    monkeypatch.setattr(handler, "_agentic_repo_branch", lambda: True)
    monkeypatch.setattr(handler, "_resolve_repo_url", lambda *args: None)
    env = _make_envelope("baseline.reproduction_requested", sid, inputs, "Orchestrator")
    logs = []
    for _ in range(3):
        await handler.handle_reproduction_requested(client, env)
        result = store.get(refs.encode("baseline_result", sid, "v1"))
        assert result["reproduced"] is False
        assert result["metric_value"] is None
        assert result.get("repo_url") is None  # No repository was executed this time.
        BASELINE_RESULT_VALIDATOR.validate(result)
        logs.append(result["log_excerpt"])
    assert "No concrete repository URL" in logs[0]
    assert "https://github.com/example/incompatible" in logs[0]
    assert "Required operation is absent" in logs[0]
    assert logs[0] == logs[1] == logs[2]


def test_live_baseline_requires_host_attestation_and_frozen_contract():
    from autoresearch.agents.runner_agent.handler import _live_baseline_valid
    base = _valid_live_artifact()

    assert _live_baseline_valid(base)
    for missing in ("benchmark_contract", "contract_fingerprint", "host_attestation"):
        candidate = dict(base)
        candidate.pop(missing)
        assert not _live_baseline_valid(candidate)
    candidate = dict(base)
    candidate["host_attestation"] = {"host_issued": False, "phase": "baseline"}
    assert not _live_baseline_valid(candidate)
    candidate = dict(base)
    candidate["contract_fingerprint"] = "sha256:tampered"
    assert not _live_baseline_valid(candidate)


@pytest.mark.parametrize(
    ("mutation", "value"),
    [
        ("source", "agent_report"),
        ("phase", "method"),
        ("argv", []),
        ("rc", 1),
        ("repo_commit", "different"),
        ("tracked_diff_hash_start", "sha256:" + "3" * 64),
        ("tracked_diff_hash_end", "sha256:" + "3" * 64),
        ("tracked_diff_present", True),
        ("measurement_spec", None),
    ],
)
def test_live_baseline_rejects_incomplete_or_inconsistent_host_attestation(
    mutation, value,
):
    """Catch agent-reported or dirty-worktree evidence being released as a live baseline."""
    from autoresearch.agents.runner_agent.handler import _live_baseline_valid
    artifact = _valid_live_artifact()
    artifact["host_attestation"][mutation] = value

    assert not _live_baseline_valid(artifact)


@pytest.mark.parametrize(
    ("field", "value"),
    [("metric", "accuracy"), ("value", 2.0), ("value", float("nan"))],
)
def test_live_baseline_rejects_metric_evidence_that_does_not_match_artifact(field, value):
    """Catch a finite artifact number detached from the host-parsed metric evidence."""
    from autoresearch.agents.runner_agent.handler import _live_baseline_valid
    artifact = _valid_live_artifact()
    artifact["host_attestation"]["metric_evidence"][field] = value

    assert not _live_baseline_valid(artifact)


@pytest.mark.parametrize("mutation", [
    "missing_hash", "escaping_hash_path", "bad_hash", "empty_protocol",
    "extra_protocol_key", "nonpositive_protocol", "wrong_protocol_unit",
    "time_to_target", "wrong_immutable_paths",
])
def test_live_baseline_enforces_immutable_and_protocol_evidence(mutation):
    from autoresearch.agents.runner_agent.handler import _live_baseline_valid
    from autoresearch.experiments.repo_improve.benchmark_contract import contract_fingerprint

    artifact = _valid_live_artifact()
    attestation = artifact["host_attestation"]
    contract = artifact["benchmark_contract"]
    if mutation == "missing_hash":
        attestation["immutable_file_hashes"].pop("evaluate.py")
    elif mutation == "escaping_hash_path":
        attestation["immutable_file_hashes"]["../evaluate.py"] = (
            "sha256:" + "4" * 64)
    elif mutation == "bad_hash":
        attestation["immutable_file_hashes"]["evaluate.py"] = "sha256:not-a-digest"
    elif mutation == "empty_protocol":
        attestation["protocol_evidence"] = {}
    elif mutation == "extra_protocol_key":
        attestation["protocol_evidence"]["extra"] = "reject"
    elif mutation == "nonpositive_protocol":
        attestation["protocol_evidence"]["value"] = 0
    elif mutation == "wrong_protocol_unit":
        attestation["protocol_evidence"]["unit"] = "steps"
    elif mutation == "time_to_target":
        contract["objective"] = {"metric": "elapsed_seconds", "direction": "minimize"}
        contract["protocol"] = {
            "kind": "time_to_target",
            "target": {"metric": "loss", "value": 1.0, "direction": "minimize"},
        }
        contract["observed_outcomes"] = {
            "quality": {"metric": "loss", "value": 1.0},
            "objective": {"metric": "elapsed_seconds", "value": 10.0},
        }
        contract["verified_evidence"]["observations"]["quality"] = {
            "metric": "loss", "value": 1.0,
        }
        contract["verified_evidence"]["observations"]["objective"] = {
            "metric": "elapsed_seconds", "value": 10.0,
        }
        artifact["contract_fingerprint"] = contract_fingerprint(contract)
    elif mutation == "wrong_immutable_paths":
        contract["controls"]["data"]["immutable_paths"] = ["other.txt"]
        artifact["contract_fingerprint"] = contract_fingerprint(contract)

    assert not _live_baseline_valid(artifact)


@pytest.mark.parametrize(
    "mutation",
    [
        "verified_source", "verified_artifact", "verified_resource", "verified_attempt",
        "verified_probe", "verified_artifact_evidence", "verified_official",
        "objective_direction", "official_contract", "resource_contract",
        "artifact_output_path",
    ],
)
def test_live_baseline_binds_host_evidence_to_frozen_contract(mutation):
    """Catch internally valid host evidence being paired with a different frozen contract."""
    from autoresearch.agents.runner_agent.handler import _live_baseline_valid
    from autoresearch.experiments.repo_improve.benchmark_contract import contract_fingerprint

    artifact = _valid_live_artifact()
    contract = artifact["benchmark_contract"]
    observations = contract["verified_evidence"]["observations"]
    if mutation == "verified_source":
        contract["verified_evidence"]["source"] = "agent proposal"
    elif mutation == "verified_artifact":
        contract["verified_evidence"]["artifact"] = "different-host-attempt"
    elif mutation == "verified_resource":
        observations["resource_evidence"] = {
            **observations["resource_evidence"], "device_models": ["nvidia a100"],
        }
    elif mutation == "verified_attempt":
        observations["agent_attempt_evidence"] = {
            **observations["agent_attempt_evidence"], "attempt_id": "different-attempt",
        }
    elif mutation == "verified_probe":
        observations["trusted_probe_evidence"] = {
            **observations["trusted_probe_evidence"], "cwd": "subdir",
        }
    elif mutation == "verified_artifact_evidence":
        observations["artifact_evidence"] = {
            **observations["artifact_evidence"], "size": 43,
        }
    elif mutation == "verified_official":
        observations["official_source_evidence"][0]["role"] = "trusted_probe"
    elif mutation == "objective_direction":
        contract["objective"]["direction"] = "maximize"
    elif mutation == "official_contract":
        contract["controls"]["evaluator"]["official_source_evidence"][0][
            "excerpt_sha256"] = "sha256:" + "9" * 64
    elif mutation == "resource_contract":
        contract["controls"]["resources"]["accelerator_count"] = 2
    elif mutation == "artifact_output_path":
        contract["controls"]["evaluator"]["output_paths"] = ["different.json"]
    artifact["contract_fingerprint"] = contract_fingerprint(contract)

    assert not _live_baseline_valid(artifact)


def test_live_baseline_accepts_user_authoritative_measurement_claims():
    from autoresearch.agents.runner_agent.handler import _live_baseline_valid
    from autoresearch.experiments.repo_improve.benchmark_contract import contract_fingerprint

    artifact = _valid_live_artifact()
    contract = artifact["benchmark_contract"]
    attestation = artifact["host_attestation"]
    spec = attestation["measurement_spec"]
    boundary_sha = "sha256:" + hashlib.sha256(json.dumps(
        spec, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    user_claims = []
    contract_claims = []
    for role, identity in (("execution", spec["execution"]),
                           ("trusted_probe", spec["trusted_probe"])):
        canonical = json.dumps(identity, sort_keys=True, separators=(",", ":"))
        user_claims.append({
            "role": role, "authority": "user", "boundary_sha256": boundary_sha,
            "identity_sha256": "sha256:" + hashlib.sha256(canonical.encode()).hexdigest(),
        })
        contract_claims.append({
            "path": ".confirmed-boundary", "line_start": 1, "line_end": 1,
            "file_sha256": boundary_sha, "excerpt_sha256": boundary_sha,
            "command": canonical,
        })
    attestation["official_source_evidence"] = user_claims
    contract["controls"]["evaluator"]["official_source_evidence"] = contract_claims
    contract["verified_evidence"]["observations"]["official_source_evidence"] = (
        copy.deepcopy(user_claims))
    artifact["contract_fingerprint"] = contract_fingerprint(contract)

    assert _live_baseline_valid(artifact)


@pytest.mark.parametrize(
    "mutation",
    [
        "agent_rc", "agent_argv", "agent_window", "resource_count", "resource_devices",
        "resource_source", "resource_unsorted", "resource_known_fallback",
        "resource_zero_nvidia", "resource_fallback_caveat",
        "artifact_path", "artifact_required", "artifact_hash", "artifact_size",
        "dirty_paths", "duplicate_role", "official_identity",
    ],
)
def test_live_baseline_rejects_malformed_round5_evidence(mutation):
    """Catch malformed Round5 evidence even when the contract fingerprint is otherwise valid."""
    from autoresearch.agents.runner_agent.handler import _live_baseline_valid

    artifact = _valid_live_artifact()
    attestation = artifact["host_attestation"]
    if mutation == "agent_rc":
        attestation["agent_attempt_evidence"]["rc"] = 1
    elif mutation == "agent_argv":
        attestation["agent_attempt_evidence"]["argv"] = []
    elif mutation == "agent_window":
        attestation["agent_attempt_evidence"]["end"] = attestation["start"] + 1
    elif mutation == "resource_count":
        attestation["resource_evidence"]["accelerator_count"] = 2
    elif mutation == "resource_devices":
        attestation["resource_evidence"]["device_models"] = []
    elif mutation == "resource_source":
        attestation["resource_evidence"]["source"] = "sandbox_launch"
    elif mutation == "resource_unsorted":
        attestation["resource_evidence"].update({
            "accelerator_count": 2,
            "device_models": ["nvidia l40", "nvidia a100"],
        })
    elif mutation == "resource_known_fallback":
        attestation["resource_evidence"].update({
            "source": "allocation_count_only", "device_models": ["nvidia l20"],
        })
    elif mutation == "resource_zero_nvidia":
        attestation["resource_evidence"].update({
            "accelerator_count": 0, "device_models": [],
        })
    elif mutation == "resource_fallback_caveat":
        attestation["resource_evidence"].update({
            "source": "allocation_count_only", "device_models": ["unknown"],
        })
    elif mutation == "artifact_path":
        attestation["artifact_evidence"]["path"] = "other.json"
    elif mutation == "artifact_required":
        attestation["artifact_evidence"]["required_new"] = False
    elif mutation == "artifact_hash":
        attestation["artifact_evidence"]["sha256"] = "sha256:bad"
    elif mutation == "artifact_size":
        attestation["artifact_evidence"]["size"] = -1
    elif mutation == "dirty_paths":
        attestation["tracked_diff_paths"] = ["train.py"]
    elif mutation == "duplicate_role":
        attestation["official_source_evidence"][0]["role"] = "trusted_probe"
    elif mutation == "official_identity":
        attestation["official_source_evidence"][0]["identity_sha256"] = (
            "sha256:" + "9" * 64)

    assert not _live_baseline_valid(artifact)


def test_live_baseline_accepts_unknown_resource_model_without_declared_model():
    """A missing nvidia-smi must not bind fairness to an unverifiable GPU ordinal."""
    from autoresearch.agents.runner_agent.handler import _live_baseline_valid
    from autoresearch.experiments.repo_improve.benchmark_contract import contract_fingerprint

    artifact = _valid_live_artifact()
    evidence = artifact["host_attestation"]["resource_evidence"]
    evidence.update({
        "device_models": ["unknown"], "source": "allocation_count_only",
        "caveat": ("precision, seed, and model scale are contract-frozen, not host-observed; "
                   "device model not verified (nvidia-smi unavailable)"),
    })
    artifact["benchmark_contract"]["verified_evidence"]["observations"][
        "resource_evidence"] = copy.deepcopy(evidence)
    artifact["contract_fingerprint"] = contract_fingerprint(
        artifact["benchmark_contract"])

    assert _live_baseline_valid(artifact)


@pytest.mark.parametrize("field", ["device_model", "resource_class"])
def test_live_baseline_fails_closed_for_unverifiable_declared_resource(field):
    from autoresearch.agents.runner_agent.handler import _live_baseline_valid
    from autoresearch.experiments.repo_improve.benchmark_contract import contract_fingerprint

    artifact = _valid_live_artifact()
    contract = artifact["benchmark_contract"]
    evidence = artifact["host_attestation"]["resource_evidence"]
    contract["controls"]["resources"][field] = "NVIDIA L20"
    evidence.update({
        "device_models": ["unknown"], "source": "allocation_count_only",
        "caveat": ("precision, seed, and model scale are contract-frozen, not host-observed; "
                   "device model not verified (nvidia-smi unavailable)"),
    })
    contract["verified_evidence"]["observations"]["resource_evidence"] = copy.deepcopy(
        evidence)
    artifact["contract_fingerprint"] = contract_fingerprint(contract)

    assert not _live_baseline_valid(artifact)


def test_live_baseline_accepts_contract_bound_structured_control():
    """A finite host-extracted binding matching the contract remains releasable."""
    from autoresearch.agents.runner_agent.handler import _live_baseline_valid
    from autoresearch.experiments.repo_improve.benchmark_contract import contract_fingerprint

    artifact = _valid_live_artifact()
    attestation = artifact["host_attestation"]
    contract = artifact["benchmark_contract"]
    digest = "sha256:" + "9" * 64
    contract["controls"]["control_bindings"] = [{
        "name": "seed", "path": "controls.json", "format": "json",
        "json_pointer": "/training/seed",
    }]
    attestation["immutable_file_hashes"]["controls.json"] = digest
    attestation["actual_controls"]["immutable_file_hashes"]["controls.json"] = digest
    attestation["actual_controls"]["bindings"] = [{
        "name": "seed", "value": 17, "treatment_axis": None,
        "source": {
            "path": "controls.json", "format": "json",
            "json_pointer": "/training/seed", "file_sha256": digest,
        },
    }]
    observations = contract["verified_evidence"]["observations"]
    observations["immutable_file_hashes"] = copy.deepcopy(
        attestation["immutable_file_hashes"])
    observations["actual_controls"] = copy.deepcopy(attestation["actual_controls"])
    artifact["contract_fingerprint"] = contract_fingerprint(contract)

    assert _live_baseline_valid(artifact)


@pytest.mark.parametrize("mutation", [
    "prior_path", "prior_stale_source", "attempt_normalized", "attempt_execution",
    "attempt_cwd", "controls_execution", "controls_hashes", "controls_nonfinite",
    "untracked_shape", "untracked_partition", "untracked_artifact", "untracked_observation",
])
def test_live_baseline_rejects_unbound_final_producer_evidence(mutation):
    """Every new producer claim must remain exact and bound to its authority."""
    from autoresearch.agents.runner_agent.handler import _live_baseline_valid

    artifact = _valid_live_artifact()
    attestation = artifact["host_attestation"]
    contract = artifact["benchmark_contract"]
    if mutation == "prior_path":
        attestation["prior_artifact_evidence"]["path"] = "other.json"
    elif mutation == "prior_stale_source":
        attestation["prior_artifact_evidence"] = {
            "state": "quarantined", "path": "metrics.json",
            "quarantine_path": "/sandbox/quarantine/metrics.json",
            "sha256": "sha256:" + "6" * 64, "size": 1, "mtime": 90.0,
            "source": "stale_pre_attempt", "attempt_id": "attempt-agent-1",
        }
    elif mutation == "attempt_normalized":
        attestation["agent_attempt_evidence"]["normalized_argv"] = ["true"]
    elif mutation == "attempt_execution":
        attestation["agent_attempt_evidence"]["execution_argv"] = ["true"]
    elif mutation == "attempt_cwd":
        attestation["agent_attempt_evidence"]["cwd"] = "."
    elif mutation == "controls_execution":
        attestation["actual_controls"]["execution"]["source_paths"] = ["../train.py"]
    elif mutation == "controls_hashes":
        attestation["actual_controls"]["immutable_file_hashes"].pop("evaluate.py")
    elif mutation == "controls_nonfinite":
        attestation["actual_controls"]["bindings"] = [{
            "name": "seed", "value": float("nan"), "treatment_axis": None,
            "source": {
                "path": "data_config.txt", "format": "json", "json_pointer": "/seed",
                "file_sha256": attestation["immutable_file_hashes"]["data_config.txt"],
            },
        }]
        contract["controls"]["control_bindings"] = [{
            "name": "seed", "path": "data_config.txt", "format": "json",
            "json_pointer": "/seed",
        }]
    elif mutation == "untracked_shape":
        attestation["untracked_files"]["after"]["metrics.json"]["extra"] = True
    elif mutation == "untracked_partition":
        attestation["untracked_files"]["added_allowed"] = ["ar_result.json"]
    elif mutation == "untracked_artifact":
        attestation["untracked_files"]["after"]["metrics.json"]["sha256"] = (
            "sha256:" + "6" * 64)
    elif mutation == "untracked_observation":
        contract["verified_evidence"]["observations"]["untracked_files"] = {
            "before": {}, "after": {}, "added_allowed": [], "added_other": [],
        }

    assert not _live_baseline_valid(artifact)


def test_live_baseline_rejects_untracked_allowed_path_not_declared_by_contract():
    """The agent cannot relabel an undeclared added file as an allowed host output."""
    from autoresearch.agents.runner_agent.handler import _live_baseline_valid
    from autoresearch.experiments.repo_improve.benchmark_contract import contract_fingerprint

    artifact = _valid_live_artifact()
    untracked = artifact["host_attestation"]["untracked_files"]
    untracked["after"]["helper.py"] = {
        "kind": "regular", "sha256": "sha256:" + "6" * 64,
        "size": 1, "executable": False,
    }
    untracked["added_allowed"] = ["ar_result.json", "helper.py", "metrics.json"]
    artifact["benchmark_contract"]["verified_evidence"]["observations"][
        "untracked_files"] = copy.deepcopy(untracked)
    artifact["contract_fingerprint"] = contract_fingerprint(
        artifact["benchmark_contract"])

    assert not _live_baseline_valid(artifact)


def test_live_baseline_recomputes_attempt_identity_from_raw_argv():
    """A producer claim cannot hide an environment-changing raw wrapper."""
    from autoresearch.agents.runner_agent.handler import _live_baseline_valid
    from autoresearch.experiments.repo_improve.benchmark_contract import contract_fingerprint

    artifact = _valid_live_artifact()
    attempt = artifact["host_attestation"]["agent_attempt_evidence"]
    attempt["argv"] = ["env", "--ignore-environment", *attempt["execution_argv"]]
    artifact["benchmark_contract"]["verified_evidence"]["observations"][
        "agent_attempt_evidence"] = copy.deepcopy(attempt)
    artifact["contract_fingerprint"] = contract_fingerprint(
        artifact["benchmark_contract"])

    assert not _live_baseline_valid(artifact)


@pytest.mark.parametrize("mutation", ["added_binary", "modified_preexisting"])
def test_live_baseline_rejects_undeclared_untracked_inputs_and_mutations(mutation):
    from autoresearch.agents.runner_agent.handler import _live_baseline_valid
    from autoresearch.experiments.repo_improve.benchmark_contract import contract_fingerprint

    artifact = _valid_live_artifact()
    untracked = artifact["host_attestation"]["untracked_files"]
    old = {
        "kind": "regular", "sha256": "sha256:" + "6" * 64,
        "size": 10, "executable": False,
    }
    new = {**old, "sha256": "sha256:" + "9" * 64}
    if mutation == "added_binary":
        untracked["after"]["weights.bin"] = new
        untracked["added_other"] = ["weights.bin"]
    else:
        untracked["before"]["weights.bin"] = old
        untracked["after"]["weights.bin"] = new
    artifact["benchmark_contract"]["verified_evidence"]["observations"][
        "untracked_files"] = copy.deepcopy(untracked)
    artifact["contract_fingerprint"] = contract_fingerprint(
        artifact["benchmark_contract"])

    assert not _live_baseline_valid(artifact)


def test_agentic_producer_attestation_passes_runner_gate(tmp_path):
    """The actual host producer output must be accepted by Runner's release gate."""
    from autoresearch.agents.runner_agent.handler import _live_baseline_valid
    from autoresearch.experiments.repo_improve.benchmark_contract import contract_fingerprint
    from autoresearch.experiments.repo_improve.agentic_improve import run_agentic_improve
    from autoresearch.experiments.repo_improve.tests.test_agentic_improve import (
        _FakeBackend, _local_git_clone, _measurement_spec, _v2_benchmark_contract,
    )

    clone, commit = _local_git_clone(tmp_path)
    contract = _v2_benchmark_contract(999.0)
    contract["controls"]["resources"]["accelerator_count"] = 1
    backend = _FakeBackend(
        result={
            "metric": "val_loss", "baseline": 999.0, "method": None,
            "lower_is_better": True, "config": "steps=1000",
                "measurement": _measurement_spec(),
                "benchmark_contract": contract,
        },
        measurement_stdout="metric=1.75 steps=1000\n",
    )
    produced = run_agentic_improve(
        "lower val loss", repo_url="https://x/repo", workroot=str(tmp_path / "run"),
        backend=backend, clone_fn=clone, execution_phase="baseline", cuda_device="0",
    )
    artifact = {
        "source_kind": "live_repo", "reproduced": True,
        "metric_value": produced["baseline"].metric_value,
        "repo_url": "https://x/repo", "repo_commit": commit,
        "metric_name": produced["metric"],
        "lower_is_better": produced["lower_is_better"], "fixed_config": produced["config"],
        "environment_fingerprint": "sha256:env",
        "benchmark_contract": produced["benchmark_contract"],
        "contract_fingerprint": contract_fingerprint(produced["benchmark_contract"]),
        "host_attestation": produced["host_attestation"],
    }

    assert _live_baseline_valid(artifact)
    schema_artifact = {
        **artifact, "benchmark_id": "repo:callable", "command": "python train.py",
        "exit_code": 0, "environment": "test sandbox", "reported_metric": None,
        "tolerance": None, "seed": "0", "runtime_seconds": 0.01,
        "log_excerpt": "callable probe", "baseline_runs": [artifact["metric_value"]],
        "artifacts": [],
    }
    assert list(BASELINE_RESULT_VALIDATOR.iter_errors(schema_artifact)) == []


def test_runner_accepts_host_hashed_uv_environment_inputs(tmp_path):
    from autoresearch.agents.runner_agent.handler import _live_baseline_valid
    from autoresearch.experiments.repo_improve.agentic_improve import run_agentic_improve
    from autoresearch.experiments.repo_improve.tests.test_agentic_improve import (
        _FakeBackend, _clone_with_protocol_command, _external_measurement, _round4_contract,
    )

    argv = [
        "uv", "run", "--env-file", "bench.env", "--with-requirements",
        "requirements-bench.txt", "python", "train.py",
    ]
    command = " ".join(argv)
    clone, _commit = _clone_with_protocol_command(tmp_path, command)
    source = tmp_path / "source"
    (source / "pyproject.toml").write_text(
        "[project]\nname='fixture'\nversion='0.1.0'\n", encoding="utf-8")
    (source / "bench.env").write_text("SEED=17\n", encoding="utf-8")
    (source / "requirements-bench.txt").write_text("pytest==8.0.0\n", encoding="utf-8")
    subprocess.run([
        "git", "-C", str(source), "add", "pyproject.toml", "bench.env",
        "requirements-bench.txt",
    ], check=True)
    subprocess.run([
        "git", "-C", str(source), "commit", "-q", "-m", "uv controls",
    ], check=True)
    measurement = _external_measurement(execution_argv=argv)
    contract = _round4_contract(
        measurement,
        protocol={"kind": "quality_at_fixed_work", "budget": {"steps": 1000}},
        command=command,
    )
    contract["controls"]["workload"] = {"steps": 1000}
    produced = run_agentic_improve(
        "lower loss", repo_url="https://x/repo", workroot=str(tmp_path / "run"),
        backend=_FakeBackend(result={
            "metric": "val_loss", "baseline": 1.75, "method": None,
            "lower_is_better": True, "config": "steps=1000",
            "measurement": measurement, "benchmark_contract": contract,
        }), clone_fn=clone, execution_phase="baseline",
    )
    assert produced["baseline"] is not None, produced
    artifact = {
        "source_kind": "live_repo", "reproduced": True,
        "metric_value": produced["baseline"].metric_value,
        "repo_url": "https://x/repo", "repo_commit": produced["repo_commit"],
        "metric_name": produced["metric"], "lower_is_better": produced["lower_is_better"],
        "fixed_config": produced["config"], "environment_fingerprint": "sha256:env",
        "benchmark_contract": produced["benchmark_contract"],
        "contract_fingerprint": produced["contract_fingerprint"],
        "host_attestation": produced["host_attestation"],
    }

    assert produced["baseline"].valid is True
    assert _live_baseline_valid(artifact)


def test_runner_accepts_producer_binding_for_joined_make_entrypoint(tmp_path):
    from autoresearch.agents.runner_agent.handler import _live_baseline_valid
    from autoresearch.experiments.repo_improve.agentic_improve import run_agentic_improve
    from autoresearch.experiments.repo_improve.benchmark_contract import contract_fingerprint
    from autoresearch.experiments.repo_improve.tests.test_agentic_improve import (
        _FakeBackend, _clone_with_protocol_command, _external_measurement, _round4_contract,
    )

    argv = ["make", "-fBench.mk", "train.py"]
    command = "make -fBench.mk train.py"
    clone, _commit = _clone_with_protocol_command(tmp_path, command)
    source = tmp_path / "source"
    (source / "Bench.mk").write_text(
        "train.py:\n\tpython train.py\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(source), "add", "Bench.mk"], check=True)
    subprocess.run([
        "git", "-C", str(source), "commit", "-q", "-m", "make entrypoint",
    ], check=True)
    measurement = _external_measurement(execution_argv=argv)
    contract = _round4_contract(
        measurement,
        protocol={"kind": "quality_at_fixed_work", "budget": {"steps": 1000}},
        command=command,
    )
    contract["controls"]["workload"] = {"steps": 1000}
    produced = run_agentic_improve(
        "lower loss", repo_url="https://x/repo", workroot=str(tmp_path / "run"),
        backend=_FakeBackend(result={
            "metric": "val_loss", "baseline": 1.75, "method": None,
            "lower_is_better": True, "config": "steps=1000",
            "measurement": measurement, "benchmark_contract": contract,
        }), clone_fn=clone, execution_phase="baseline",
    )
    assert produced["baseline"] is not None, produced
    assert produced["baseline"].valid is True, produced
    artifact = {
        "source_kind": "live_repo", "reproduced": True,
        "metric_value": produced["baseline"].metric_value,
        "repo_url": "https://x/repo", "repo_commit": produced["repo_commit"],
        "metric_name": produced["metric"], "lower_is_better": produced["lower_is_better"],
        "fixed_config": produced["config"], "environment_fingerprint": "sha256:env",
        "benchmark_contract": produced["benchmark_contract"],
        "contract_fingerprint": contract_fingerprint(produced["benchmark_contract"]),
        "host_attestation": produced["host_attestation"],
    }

    assert produced["baseline"].valid is True
    assert produced["host_attestation"]["actual_controls"]["execution"][
        "source_paths"] == ["Bench.mk"]
    assert _live_baseline_valid(artifact)


def test_runner_and_schema_accept_real_scaled_protocol_parser_attestation(tmp_path):
    from autoresearch.agents.runner_agent.handler import _live_baseline_valid
    from autoresearch.experiments.repo_improve.agentic_improve import run_agentic_improve
    from autoresearch.experiments.repo_improve.benchmark_contract import contract_fingerprint
    from autoresearch.experiments.repo_improve.tests.test_agentic_improve import (
        _FakeBackend, _clone_with_protocol_command, _external_measurement, _round4_contract,
    )

    clone, commit = _clone_with_protocol_command(tmp_path, "python train.py")
    measurement = _external_measurement(protocol_regex=r"minutes=([0-9.]+)")
    measurement["trusted_probe"]["protocol_parser"].update({
        "unit": "seconds", "multiplier": 60,
    })
    contract = _round4_contract(
        measurement,
        protocol={
            "kind": "quality_at_fixed_time", "budget": {"wall_clock_s": 300},
            "timing_scope": "training", "timer_authority": "official_harness",
        },
    )
    contract["controls"]["workload"] = {"training_time_s": 300}
    produced = run_agentic_improve(
        "lower val loss", repo_url="https://x/repo", workroot=str(tmp_path / "run"),
        backend=_FakeBackend(result={
            "metric": "val_loss", "baseline": 999.0, "method": None,
            "lower_is_better": True, "config": "training_time_s=300",
            "measurement": measurement, "benchmark_contract": contract,
        }, measurement_stdout="metric=1.75 minutes=5\n"),
        clone_fn=clone, execution_phase="baseline",
    )
    artifact = {
        "source_kind": "live_repo", "reproduced": True,
        "metric_value": produced["baseline"].metric_value,
        "repo_url": "https://x/repo", "repo_commit": commit,
        "metric_name": produced["metric"],
        "lower_is_better": produced["lower_is_better"], "fixed_config": produced["config"],
        "environment_fingerprint": "sha256:env",
        "benchmark_contract": produced["benchmark_contract"],
        "contract_fingerprint": contract_fingerprint(produced["benchmark_contract"]),
        "host_attestation": produced["host_attestation"],
    }
    schema_artifact = {
        **artifact, "benchmark_id": "repo:scaled-protocol", "command": "python train.py",
        "exit_code": 0, "environment": "test sandbox", "reported_metric": None,
        "tolerance": None, "seed": "0", "runtime_seconds": 300.0,
        "log_excerpt": "official harness minutes", "baseline_runs": [artifact["metric_value"]],
        "artifacts": [],
    }

    assert produced["baseline"].valid is True, produced
    assert produced["host_attestation"]["measurement_spec"]["trusted_probe"][
        "protocol_parser"] == {
            "stream": "stdout", "regex": r"minutes=([0-9.]+)", "group": 1,
            "unit": "seconds", "multiplier": 60,
        }
    assert _live_baseline_valid(artifact)
    assert list(BASELINE_RESULT_VALIDATOR.iter_errors(schema_artifact)) == []
    invalid_schema_artifact = copy.deepcopy(schema_artifact)
    invalid_schema_artifact["host_attestation"]["measurement_spec"]["trusted_probe"][
        "protocol_parser"]["multiplier"] = 0
    assert list(BASELINE_RESULT_VALIDATOR.iter_errors(invalid_schema_artifact))
    tampered_budget = copy.deepcopy(artifact)
    tampered_budget["benchmark_contract"]["protocol"]["budget"]["wall_clock_s"] = 1
    tampered_budget["benchmark_contract"]["verified_evidence"]["observations"][
        "protocol_evidence"] = copy.deepcopy(
            tampered_budget["host_attestation"]["protocol_evidence"])
    tampered_budget["contract_fingerprint"] = contract_fingerprint(
        tampered_budget["benchmark_contract"])
    assert not _live_baseline_valid(tampered_budget)


def test_runner_and_schema_accept_callable_with_execution_protocol_parser(tmp_path):
    from autoresearch.agents.runner_agent.handler import _live_baseline_valid
    from autoresearch.experiments.repo_improve.agentic_improve import run_agentic_improve
    from autoresearch.experiments.repo_improve.benchmark_contract import contract_fingerprint
    from autoresearch.experiments.repo_improve.tests.test_agentic_improve import (
        _LocalCommandBackend, _callable_measurement, _local_git_clone, _round4_contract,
    )

    clone, _commit = _local_git_clone(tmp_path)
    source = tmp_path / "source"
    (source / "train.py").write_text(
        "from prepare import evaluate_bpb\n"
        "from pathlib import Path\n"
        "Path('metrics.json').write_text('{}\\n')\n"
        "print('training_minutes=5')\n"
        "evaluate_bpb()\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "-C", str(source), "add", "train.py"], check=True)
    subprocess.run([
        "git", "-C", str(source), "commit", "-q", "--amend", "--no-edit",
    ], check=True)
    measurement = _callable_measurement()
    measurement["trusted_probe"]["protocol_parser"] = {
        "stream": "stdout", "regex": r"training_minutes=([0-9.]+)", "group": 1,
        "unit": "seconds", "multiplier": 60,
    }
    contract = _round4_contract(
        measurement,
        protocol={
            "kind": "quality_at_fixed_time", "budget": {"wall_clock_s": 300},
            "timing_scope": "training", "timer_authority": "official_harness",
        },
    )
    contract["controls"]["workload"] = {"training_time_s": 300}
    produced = run_agentic_improve(
        "lower val loss", repo_url="https://x/repo", workroot=str(tmp_path / "run"),
        backend=_LocalCommandBackend(result={
            "metric": "val_loss", "baseline": -999.0, "method": None,
            "lower_is_better": True, "config": "training_time_s=300",
            "measurement": measurement, "benchmark_contract": contract,
        }),
        clone_fn=clone, execution_phase="baseline",
    )
    artifact = {
        "source_kind": "live_repo", "reproduced": True,
        "metric_value": produced["baseline"].metric_value,
        "repo_url": "https://x/repo", "repo_commit": produced["repo_commit"],
        "metric_name": produced["metric"],
        "lower_is_better": produced["lower_is_better"], "fixed_config": produced["config"],
        "environment_fingerprint": "sha256:env",
        "benchmark_contract": produced["benchmark_contract"],
        "contract_fingerprint": contract_fingerprint(produced["benchmark_contract"]),
        "host_attestation": produced["host_attestation"],
    }
    schema_artifact = {
        **artifact, "benchmark_id": "repo:callable-training-time",
        "command": "python train.py", "exit_code": 0, "environment": "test sandbox",
        "reported_metric": None, "tolerance": None, "seed": "0", "runtime_seconds": 300.0,
        "log_excerpt": "callable metric and execution timer",
        "baseline_runs": [artifact["metric_value"]], "artifacts": [],
    }

    assert produced["baseline"].valid is True, produced
    assert produced["host_attestation"]["measurement_spec"]["trusted_probe"] == {
        "kind": "python_callable", "module": "prepare", "callable": "evaluate_bpb",
        "source_path": "prepare.py",
        "protocol_parser": {
            "stream": "stdout", "regex": r"training_minutes=([0-9.]+)", "group": 1,
            "unit": "seconds", "multiplier": 60,
        },
    }
    assert _live_baseline_valid(artifact)
    assert list(BASELINE_RESULT_VALIDATOR.iter_errors(schema_artifact)) == []
    for unsafe_stream in ("stderr", "combined"):
        tampered = copy.deepcopy(artifact)
        tampered_attestation = tampered["host_attestation"]
        tampered_contract = tampered["benchmark_contract"]
        attestation_probe = tampered_attestation["measurement_spec"]["trusted_probe"]
        contract_probe = tampered_contract["controls"]["evaluator"]["measurement_spec"][
            "trusted_probe"]
        attestation_probe["protocol_parser"]["stream"] = unsafe_stream
        contract_probe["protocol_parser"]["stream"] = unsafe_stream
        canonical = json.dumps(attestation_probe, sort_keys=True, separators=(",", ":"))
        trusted_claim = next(
            claim for claim in tampered_attestation["official_source_evidence"]
            if claim["role"] == "trusted_probe")
        trusted_claim["command"] = canonical
        trusted_claim["identity_sha256"] = (
            "sha256:" + hashlib.sha256(canonical.encode()).hexdigest())
        tampered_contract["controls"]["evaluator"]["official_source_evidence"][1][
            "command"] = canonical
        tampered_contract["verified_evidence"]["observations"][
            "official_source_evidence"] = copy.deepcopy(
                tampered_attestation["official_source_evidence"])
        tampered["contract_fingerprint"] = contract_fingerprint(tampered_contract)
        assert not _live_baseline_valid(tampered)

        invalid_schema = copy.deepcopy(schema_artifact)
        invalid_schema["host_attestation"]["measurement_spec"]["trusted_probe"][
            "protocol_parser"]["stream"] = unsafe_stream
        assert list(BASELINE_RESULT_VALIDATOR.iter_errors(invalid_schema))


def test_runner_binds_host_elapsed_protocol_value_to_attested_duration():
    from autoresearch.agents.runner_agent.handler import _live_baseline_valid

    artifact = _valid_live_artifact()
    artifact["host_attestation"]["end"] = 109.0

    assert not _live_baseline_valid(artifact)


@pytest.mark.parametrize("multiplier", [0, -1])
def test_runner_rejects_nonpositive_protocol_multiplier(multiplier):
    from autoresearch.agents.runner_agent.handler import _live_baseline_valid
    from autoresearch.experiments.repo_improve.benchmark_contract import contract_fingerprint

    artifact = _valid_live_artifact()
    contract = artifact["benchmark_contract"]
    attestation = artifact["host_attestation"]
    probe = attestation["measurement_spec"]["trusted_probe"]
    probe["protocol_parser"] = {
        "stream": "stdout", "regex": r"minutes=([0-9.]+)", "group": 1,
        "unit": "seconds", "multiplier": multiplier,
    }
    contract["protocol"] = {
        "kind": "quality_at_fixed_time", "budget": {"wall_clock_s": 10},
        "timing_scope": "training", "timer_authority": "official_harness",
    }
    contract["controls"]["workload"] = {"training_time_s": 10}
    protocol_evidence = {"source": "host_parser", "unit": "seconds", "value": 10.0}
    attestation["protocol_evidence"] = protocol_evidence
    trusted_claim = next(
        claim for claim in attestation["official_source_evidence"]
        if claim["role"] == "trusted_probe")
    canonical = json.dumps(probe, sort_keys=True, separators=(",", ":"))
    trusted_claim["command"] = canonical
    trusted_claim["identity_sha256"] = (
        "sha256:" + hashlib.sha256(canonical.encode()).hexdigest())
    contract_source = contract["controls"]["evaluator"]["official_source_evidence"]
    contract_source[1]["command"] = canonical
    observations = contract["verified_evidence"]["observations"]
    observations["protocol_evidence"] = copy.deepcopy(protocol_evidence)
    observations["official_source_evidence"] = copy.deepcopy(
        attestation["official_source_evidence"])
    artifact["contract_fingerprint"] = contract_fingerprint(contract)

    assert not _live_baseline_valid(artifact)


def test_structured_controls_from_actual_producer_pass_runner_gate(tmp_path):
    """Runner accepts the producer's host-extracted JSON/TOML/Python controls unchanged."""
    from autoresearch.agents.runner_agent.handler import _live_baseline_valid
    from autoresearch.experiments.repo_improve.agentic_improve import run_agentic_improve
    from autoresearch.experiments.repo_improve.tests.test_agentic_improve import (
        _FakeBackend, _clone_with_structured_control_files, _measurement_spec,
        _structured_control_bindings, _v2_benchmark_contract,
    )

    clone = _clone_with_structured_control_files(tmp_path)
    contract = _v2_benchmark_contract(999.0)
    contract["controls"]["control_bindings"] = _structured_control_bindings()
    backend = _FakeBackend(result={
        "metric": "val_loss", "baseline": 999.0, "method": None,
        "lower_is_better": True, "config": "agent prose is not control evidence",
        "measurement": _measurement_spec(), "benchmark_contract": contract,
    }, measurement_stdout="metric=1.75 steps=1000\n")
    produced = run_agentic_improve(
        "lower val loss", repo_url="https://x/repo", workroot=str(tmp_path / "run"),
        backend=backend, clone_fn=clone, execution_phase="baseline",
    )
    artifact = {
        "source_kind": "live_repo", "reproduced": True,
        "metric_value": produced["baseline"].metric_value,
        "repo_url": "https://x/repo", "repo_commit": produced["repo_commit"],
        "metric_name": produced["metric"],
        "lower_is_better": produced["lower_is_better"], "fixed_config": produced["config"],
        "environment_fingerprint": "sha256:env",
        "benchmark_contract": produced["benchmark_contract"],
        "contract_fingerprint": produced["contract_fingerprint"],
        "host_attestation": produced["host_attestation"],
    }

    assert _live_baseline_valid(artifact)


def test_callable_producer_attestation_passes_runner_gate(tmp_path):
    """Callable trusted-probe evidence from the producer is accepted unchanged."""
    from autoresearch.agents.runner_agent.handler import _live_baseline_valid
    from autoresearch.experiments.repo_improve.agentic_improve import run_agentic_improve
    from autoresearch.experiments.repo_improve.tests.test_agentic_improve import (
        _LocalCommandBackend, _callable_measurement, _local_git_clone, _round4_contract,
    )

    clone, commit = _local_git_clone(tmp_path)
    measurement = _callable_measurement()
    contract = _round4_contract(measurement)
    contract["controls"]["resources"]["accelerator_count"] = 1
    backend = _LocalCommandBackend(result={
        "metric": "val_loss", "baseline": -999.0, "method": None,
        "lower_is_better": True, "config": "seconds=0.01",
        "measurement": measurement, "benchmark_contract": contract,
    })
    produced = run_agentic_improve(
        "lower val loss", repo_url="https://x/repo", workroot=str(tmp_path / "run"),
        backend=backend, clone_fn=clone, execution_phase="baseline", cuda_device="0",
    )
    from autoresearch.experiments.repo_improve.benchmark_contract import contract_fingerprint
    artifact = {
        "source_kind": "live_repo", "reproduced": True,
        "metric_value": produced["baseline"].metric_value,
        "repo_url": "https://x/repo", "repo_commit": commit,
        "metric_name": produced["metric"],
        "lower_is_better": produced["lower_is_better"], "fixed_config": produced["config"],
        "environment_fingerprint": "sha256:env",
        "benchmark_contract": produced["benchmark_contract"],
        "contract_fingerprint": contract_fingerprint(produced["benchmark_contract"]),
        "host_attestation": produced["host_attestation"],
    }

    assert _live_baseline_valid(artifact)
    schema_artifact = {
        **artifact, "benchmark_id": "repo:callable", "command": "python train.py",
        "exit_code": 0, "environment": "test sandbox", "reported_metric": None,
        "tolerance": None, "seed": "0", "runtime_seconds": 0.01,
        "log_excerpt": "callable probe", "baseline_runs": [artifact["metric_value"]],
        "artifacts": [],
    }
    assert list(BASELINE_RESULT_VALIDATOR.iter_errors(schema_artifact)) == []


def test_callable_attestation_requires_exactly_one_observed_call(tmp_path):
    from autoresearch.agents.runner_agent.handler import _live_baseline_valid
    from autoresearch.experiments.repo_improve.agentic_improve import run_agentic_improve
    from autoresearch.experiments.repo_improve.benchmark_contract import contract_fingerprint
    from autoresearch.experiments.repo_improve.tests.test_agentic_improve import (
        _LocalCommandBackend, _callable_measurement, _local_git_clone, _round4_contract,
    )

    clone, commit = _local_git_clone(tmp_path)
    measurement = _callable_measurement()
    contract = _round4_contract(measurement)
    contract["controls"]["resources"]["accelerator_count"] = 1
    produced = run_agentic_improve(
        "lower val loss", repo_url="https://x/repo", workroot=str(tmp_path / "run"),
        backend=_LocalCommandBackend(result={
            "metric": "val_loss", "baseline": -999.0, "method": None,
            "lower_is_better": True, "config": "seconds=0.01",
            "measurement": measurement, "benchmark_contract": contract,
        }),
        clone_fn=clone, execution_phase="baseline", cuda_device="0",
    )
    attestation = produced["host_attestation"]
    attestation["trusted_probe_evidence"]["call_count"] = 2
    produced["benchmark_contract"]["verified_evidence"]["observations"][
        "trusted_probe_evidence"] = copy.deepcopy(
            attestation["trusted_probe_evidence"])
    artifact = {
        "source_kind": "live_repo", "reproduced": True,
        "metric_value": produced["baseline"].metric_value,
        "repo_url": "https://x/repo", "repo_commit": commit,
        "metric_name": produced["metric"],
        "lower_is_better": produced["lower_is_better"], "fixed_config": produced["config"],
        "environment_fingerprint": "sha256:env",
        "benchmark_contract": produced["benchmark_contract"],
        "contract_fingerprint": contract_fingerprint(produced["benchmark_contract"]),
        "host_attestation": attestation,
    }
    schema_artifact = {
        **artifact, "benchmark_id": "repo:callable", "command": "python train.py",
        "exit_code": 0, "environment": "test sandbox", "reported_metric": None,
        "tolerance": None, "seed": "0", "runtime_seconds": 0.01,
        "log_excerpt": "callable probe", "baseline_runs": [artifact["metric_value"]],
        "artifacts": [],
    }

    assert not _live_baseline_valid(artifact)
    assert list(BASELINE_RESULT_VALIDATOR.iter_errors(schema_artifact))


def test_baseline_result_schema_strictly_validates_host_attestation():
    """Catch malformed or extensible host claims being accepted by the persisted contract."""
    artifact = {
        "benchmark_id": "repo:test", "reproduced": True,
        "command": "python train.py", "exit_code": 0, "environment": "test",
        "metric_value": 1.0, "source_kind": "live_repo",
        "repo_url": "https://github.com/example/project", "repo_commit": "abc123",
        "metric_name": "loss", "lower_is_better": True, "fixed_config": "steps=1000",
        "environment_fingerprint": "sha256:env", "benchmark_contract": None,
        "contract_fingerprint": None, "host_attestation": _valid_host_attestation(),
        "baseline_runs": [1.0], "artifacts": [],
    }
    assert list(BASELINE_RESULT_VALIDATOR.iter_errors(artifact)) == []

    artifact["host_attestation"]["untrusted_note"] = "agent supplied"
    errors = list(BASELINE_RESULT_VALIDATOR.iter_errors(artifact))
    assert errors

    artifact["host_attestation"].pop("untrusted_note")
    artifact["host_attestation"].pop("protocol_evidence")
    assert list(BASELINE_RESULT_VALIDATOR.iter_errors(artifact))


@pytest.mark.parametrize("mutation", [
    "prior_extra", "attempt_missing", "controls_source_extra", "untracked_entry_extra",
])
def test_baseline_result_schema_rejects_extensible_final_attestation_claims(mutation):
    attestation = _valid_host_attestation()
    if mutation == "prior_extra":
        attestation["prior_artifact_evidence"]["note"] = "untrusted"
    elif mutation == "attempt_missing":
        attestation["agent_attempt_evidence"].pop("normalized_argv")
    elif mutation == "controls_source_extra":
        attestation["actual_controls"]["bindings"] = [{
            "name": "seed", "value": 17, "treatment_axis": None,
            "source": {
                "path": "data_config.txt", "format": "json", "json_pointer": "/seed",
                "file_sha256": "sha256:" + "3" * 64, "line": 1,
            },
        }]
    elif mutation == "untracked_entry_extra":
        attestation["untracked_files"]["after"]["metrics.json"]["owner"] = "agent"
    artifact = {
        "benchmark_id": "repo:test", "reproduced": True,
        "command": "python train.py", "exit_code": 0, "environment": "test",
        "metric_value": 1.0, "source_kind": "live_repo",
        "repo_url": "https://github.com/example/project", "repo_commit": "abc123",
        "metric_name": "loss", "lower_is_better": True, "fixed_config": "steps=1000",
        "environment_fingerprint": "sha256:env", "benchmark_contract": None,
        "contract_fingerprint": None, "host_attestation": attestation,
        "baseline_runs": [1.0], "artifacts": [],
    }

    assert list(BASELINE_RESULT_VALIDATOR.iter_errors(artifact))


@pytest.mark.parametrize("include_attestation", [False, True])
def test_nonlive_baseline_schema_allows_omitted_or_null_host_attestation(include_attestation):
    artifact = {
        "benchmark_id": "paper:test", "reproduced": True,
        "command": "published reference", "exit_code": 0, "environment": "literature",
        "source_kind": "published_reference",
    }
    if include_attestation:
        artifact["host_attestation"] = None

    assert list(BASELINE_RESULT_VALIDATOR.iter_errors(artifact)) == []
