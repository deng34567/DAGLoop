"""
Tests for RunnerAgent's P1 general-experiment branch (RunningExperiment).
RunnerAgent 的 P1 通用实验分支（RunningExperiment）测试。

DEFAULT-OFF: the branch fires only when BOTH ``AUTORESEARCH_GEN_EXPERIMENT=1``
is set AND a ``p1_backend`` is injected. When either is absent the existing
default RunnerBackend path runs, byte-identical (covered by test_run_handler.py
+ test_mc_wiring.py).

Everything is mocked: a fake p1_backend (never touched because run_pipeline is
monkeypatched), a fake p1_llm, a real tmp SpecStore, and a monkeypatched
``run_pipeline`` so no Docker/LLM/network is exercised.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any, Optional

import pytest
import yaml
from pathlib import Path
from jsonschema import Draft202012Validator

from autoresearch.agents.runner_agent import handler as handler_module
from autoresearch.agents.runner_agent.handler import RunnerAgentHandler
from autoresearch.spec_store import SpecStore
from autoresearch.spec_store import refs
from autoresearch.spec_store.identity import artifact_digest

_CONTRACTS_DIR = Path(__file__).resolve().parents[1] / "data-contracts"
RUN_RECORD_VALIDATOR = Draft202012Validator(
    yaml.safe_load((_CONTRACTS_DIR / "run_record.schema.yaml").read_text(encoding="utf-8"))
)


@dataclass
class _Published:
    event_type: str
    session_id: str
    payload: dict
    causation_id: Optional[str]


class FakeBusClient:
    def __init__(self) -> None:
        self.emitter_name = "RunnerAgent"
        self.published: list[_Published] = []
        self.subscriptions: list[tuple[str, Any]] = []

    def subscribe(self, pattern: str, handler: Any) -> None:
        self.subscriptions.append((pattern, handler))

    async def publish(self, *, event_type, session_id, payload, cost_delta="0",
                      causation_id=None, auto_commit=True) -> str:
        self.published.append(_Published(event_type, session_id, payload, causation_id))
        return f"evt-{len(self.published)}"


@dataclass
class _Envelope:
    event_id: str
    session_id: str
    event_type: str
    emitter: str
    payload: dict


def _env(session_id, payload):
    return _Envelope(str(uuid.uuid4()), session_id, "experiment.run_requested",
                     "ResearchOrchestrator", payload)


@pytest.fixture()
def store(tmp_path):
    s = SpecStore(tmp_path / "store")
    yield s
    s.close()


def _fake_pipeline_result(verdict="pass"):
    """A canned run_pipeline() return with baseline-vs-method numbers."""
    return {
        "design": {"idea": "x"},
        "harness": {},
        "method": {},
        "primary": {"baseline_metric": 100.0, "method_metric": 42.0,
                    "valid": True, "seed": 0, "isolated": True},
        "reproduce": {"baseline_metric": 100.0, "method_metric": 43.0,
                      "valid": True, "seed": 1, "isolated": True},
        "isolated": True,
        "verdict": {"verdict": verdict, "accepted": verdict == "pass",
                    "reasons": ["baseline=100 vs method=42 bins"]},
        "report": (
            "# Honest report\n\nbaseline=100 vs method=42 bins; "
            f"verdict={verdict}"
        ),
    }


def _seed_reinterpreted_spec(store, session_id, goal, key="reinterpreted"):
    return store.put(kind="research_spec", session_id=session_id, key=key,
                     obj={"research_goal": goal}, emitter="ProblemModelingAgent")


@pytest.mark.parametrize("raises", [False, True])
@pytest.mark.parametrize("error,reason_code", [
    ("pre_consume_token_quota_failed", "insufficient_quota"),
    ("Allocated quota exceeded", "insufficient_quota"),
    ("Conversation error code: BadGatewayError", "llm_provider_rate_limit"),
    ("Conversation error code: LLMServiceUnavailableError", "llm_provider_rate_limit"),
    ("Conversation error code: LLMError; llm_provider_timeout", "llm_provider_rate_limit"),
    ("ComparisonGuardianLLMError: LLM call failed: APITimeoutError: Request timed out.",
     "llm_provider_rate_limit"),
])
async def test_provider_failure_never_finalizes_a_component_attempt(
        store, monkeypatch, raises, error, reason_code):
    monkeypatch.setenv("AUTORESEARCH_GEN_EXPERIMENT", "1")
    monkeypatch.delenv("AUTORESEARCH_AGENTIC_IMPROVE", raising=False)
    h = RunnerAgentHandler(spec_store=store, backend=object(),
                           p1_backend=object(), p1_llm=object())
    sid = str(uuid.uuid4())
    _seed_reinterpreted_spec(store, sid, "Improve a numerical method")
    client = FakeBusClient()
    def run(*args, **kwargs):
        if raises:
            raise RuntimeError("Remote error: " + error)
        # Repository mappings use succeeded=True even for honest invalid results.
        return ({"succeeded": True, "stderr_excerpt": error},
                {"verdict": "invalid", "method_metric": None})
    monkeypatch.setattr(h, "_run_gen_experiment", run)
    monkeypatch.setattr(h, "_finalize_component_attempt", lambda *a, **k:
                        pytest.fail("provider outage consumed component budget"))
    monkeypatch.setattr(h, "_record_component_failure", lambda *a, **k:
                        pytest.fail("provider outage recorded a component failure"))
    await h.handle_run_requested(client, _env(sid, {}))
    assert [e.event_type for e in client.published] == ["agent.failed"]
    assert client.published[0].payload["underlying_reason_code"] == reason_code


@pytest.mark.parametrize("stale_field", [None, "stderr_excerpt", "stdout_excerpt"])
async def test_salvaged_provider_failure_reports_provider_error_not_binding_diagnostic(
        store, monkeypatch, stale_field):
    monkeypatch.setenv("AUTORESEARCH_GEN_EXPERIMENT", "1")
    monkeypatch.delenv("AUTORESEARCH_AGENTIC_IMPROVE", raising=False)
    h = RunnerAgentHandler(spec_store=store, backend=object(),
                           p1_backend=object(), p1_llm=object())
    sid = str(uuid.uuid4())
    _seed_reinterpreted_spec(store, sid, "Improve a numerical method")
    client = FakeBusClient()
    provider_error = "Conversation error code: LLMServiceUnavailableError"
    record = {"succeeded": True, "stderr_excerpt": "", "stdout_excerpt": ""}
    if stale_field:
        record[stale_field] = "component probe source missing; scalar is not traceable"

    monkeypatch.setattr(h, "_run_gen_experiment", lambda *a, **k: (
        record,
        {
            "verdict": "invalid",
            "reasons": ["runtime attempt tracked diff fingerprint does not match"],
            "report": f"[agent session ended with a transient error: {provider_error}]",
        },
    ))
    monkeypatch.setattr(h, "_finalize_component_attempt", lambda *a, **k:
                        pytest.fail("provider outage consumed component budget"))
    monkeypatch.setattr(h, "_record_component_failure", lambda *a, **k:
                        pytest.fail("provider outage recorded a component failure"))

    await h.handle_run_requested(client, _env(sid, {}))

    assert [e.event_type for e in client.published] == ["agent.failed"]
    failure = client.published[0].payload
    assert failure["underlying_reason_code"] == "llm_provider_rate_limit"
    assert provider_error in failure["underlying_reason"]
    assert "tracked diff fingerprint" not in failure["underlying_reason"]
    from autoresearch.llm.errors import is_resumable_provider_error
    assert is_resumable_provider_error(failure)


# ---------------------------------------------------------------------------
# (a) env ON + p1_backend -> run_pipeline drives; run_completed carries a
# schema-valid RunRecord whose stdout_excerpt IS the P1 report; succeeded True.
# ---------------------------------------------------------------------------


async def test_gen_branch_runs_pipeline_and_publishes_report(store, monkeypatch):
    monkeypatch.setenv("AUTORESEARCH_GEN_EXPERIMENT", "1")

    captured = {}

    def fake_run_pipeline(backend, llm, question, *, design=None):
        captured["backend"] = backend
        captured["llm"] = llm
        captured["question"] = question
        captured["design"] = design
        return _fake_pipeline_result(verdict="pass")

    monkeypatch.setattr(handler_module, "run_pipeline", fake_run_pipeline)

    p1_backend = object()
    p1_llm = object()
    handler = RunnerAgentHandler(spec_store=store, backend=object(),
                                 p1_backend=p1_backend, p1_llm=p1_llm)
    client = FakeBusClient()

    sid = str(uuid.uuid4())
    _seed_reinterpreted_spec(store, sid, "Reduce estimator variance.")

    await handler.handle_run_requested(client, _env(sid, {}))

    # run_pipeline was called with the injected backend/llm + reconstructed question
    assert captured["backend"] is p1_backend
    assert captured["llm"] is p1_llm
    assert captured["question"] == "Reduce estimator variance."
    completed = [p for p in client.published if p.event_type == "experiment.run_completed"]
    assert len(completed) == 1
    rec = store.get(completed[0].payload["run_record_ref"])
    errors = list(RUN_RECORD_VALIDATOR.iter_errors(rec))
    assert errors == [], [e.message for e in errors]
    assert rec["succeeded"] is True
    assert rec["metric_value"] == 42.0
    assert rec["stdout_excerpt"].startswith("# Honest report")
    assert rec["environment"] == "docker/openhands"
    assert completed[0].payload["verdict"] == "succeeded"


@pytest.mark.parametrize("recover,event_type", [
    (False, "agent.failed"),
    (True, "experiment.run_completed"),
])
async def test_started_run_marker_only_allows_explicit_restart_recovery(
        store, monkeypatch, recover, event_type):
    monkeypatch.setenv("AUTORESEARCH_GEN_EXPERIMENT", "1")
    handler = RunnerAgentHandler(
        spec_store=store, backend=object(), p1_backend=object(), p1_llm=object())
    sid = str(uuid.uuid4())
    _seed_reinterpreted_spec(store, sid, "Recover an interrupted experiment.")
    client = FakeBusClient()
    env = _env(sid, {"recover_completed_result": recover})
    store.put(
        kind="audit", session_id=sid, key="runner-started-" + env.event_id,
        obj={"request_digest": artifact_digest(env.payload)}, emitter="RunnerAgent")
    called = []

    def run(*args, **kwargs):
        called.append(True)
        return ({
            "succeeded": True, "exit_code": 0, "metric_value": 42.0,
            "stdout_excerpt": "recovered", "stderr_excerpt": None,
            "environment": "docker/openhands", "seed": "0",
            "runtime_seconds": 1.0, "artifacts": [],
        }, {
            "verdict": "new_best", "accepted": True, "reasons": [],
            "baseline_metric": 100.0, "method_metric": 42.0,
            "lower_is_better": True,
        })

    monkeypatch.setattr(handler, "_run_gen_experiment", run)
    await handler.handle_run_requested(client, env)

    assert [item.event_type for item in client.published] == [event_type]
    assert bool(called) is recover


async def test_survey_validation_uses_validation_runner_without_baseline(store, monkeypatch):
    monkeypatch.setenv("AUTORESEARCH_GEN_EXPERIMENT", "1")
    captured = {}

    def fake_validation(backend, llm, study):
        captured.update(backend=backend, llm=llm, study=study)
        return ({
            "succeeded": True, "exit_code": 0, "metric_value": 0.03,
            "stdout_excerpt": "No stable association.", "stderr_excerpt": None,
            "environment": "survey_validation|host_rerun", "seed": None,
            "runtime_seconds": 2.0, "artifacts": [],
        }, {
            "conclusion": "inconclusive", "metric_name": "spearman_r",
            "metric_value": 0.03, "sample_ids": ["W1"],
            "retrieved_at": "2026-08-20T10:00:00Z", "data_sources": ["OpenAlex"],
            "analysis_summary": "No stable association.", "artifacts": ["raw_data.json"],
        }, {"study": study, "result": {}, "files": {"raw_data.json": {}}})

    monkeypatch.setattr(handler_module, "run_validation_study", fake_validation)
    p1_backend = object()
    p1_llm = object()
    handler = RunnerAgentHandler(
        spec_store=store, backend=object(), p1_backend=p1_backend, p1_llm=p1_llm)
    client = FakeBusClient()
    sid = str(uuid.uuid4())
    study = {
        "kind": "survey_validation", "question": "Does the trend support the claim?",
        "sampling_frame": "A dated public sample.", "data_sources": ["OpenAlex"],
        "variables": ["year"], "analysis": "Report a trend.",
        "falsification_condition": "No trend.", "budget": "100 records.",
    }
    interpretation_ref = store.put(
        kind="spec_interpretation", session_id=sid, key="v1",
        obj={"validation_study": study}, emitter="ProblemModelingAgent")

    await handler.handle_run_requested(
        client, _env(sid, {"spec_interpretation_ref": interpretation_ref}))

    assert captured == {"backend": p1_backend, "llm": p1_llm, "study": study}
    completed = [p for p in client.published if p.event_type == "experiment.run_completed"]
    assert len(completed) == 1, client.published
    record = store.get(completed[0].payload["run_record_ref"])
    assert record["succeeded"] is True
    assert record["metric_value"] == 0.03
    assert len(record["artifacts"]) == 2
    result = store.get(refs.encode("validation_study_result", sid, "v1"))
    assert result["conclusion"] == "inconclusive"


async def test_repo_run_uses_method_sandbox_with_relaxed_frozen_baseline(
    store, monkeypatch,
):
    """The second sandbox gets the real baseline but needs no fairness attestation."""
    monkeypatch.delenv("AUTORESEARCH_GEN_EXPERIMENT", raising=False)
    monkeypatch.setenv("AUTORESEARCH_REPO_MODE", "auto")
    monkeypatch.setenv("AUTORESEARCH_AGENTIC_IMPROVE", "1")
    handler = RunnerAgentHandler(
        spec_store=store, backend=object(), p1_backend=object(), p1_llm=object())
    client = FakeBusClient()
    sid = str(uuid.uuid4())
    _seed_reinterpreted_spec(
        store, sid, "Improve https://github.com/example/nanochat", key="final")
    frozen = {
        "benchmark_id": "repo:nanochat", "command": "python train.py",
        "exit_code": 0, "environment": "gpu-worker-04", "reported_metric": None,
        "reproduced": True, "metric_value": 1.0, "metric_name": "loss",
        "lower_is_better": True, "fixed_config": "steps=100",
        "repo_commit": "abc123", "source_kind": "live_repo",
        "benchmark_contract": None, "contract_fingerprint": None,
        "host_attestation": None, "environment_fingerprint": None,
        "tolerance": None, "seed": "0", "runtime_seconds": 10.0,
        "log_excerpt": "baseline=1.0", "repo_url": "https://github.com/example/nanochat",
        "baseline_runs": [1.0], "artifacts": [],
    }
    baseline_ref = store.put(
        kind="baseline_result", session_id=sid, key="v1", obj=frozen,
        emitter="RunnerAgent")
    interpretation_ref = store.put(
        kind="spec_interpretation", session_id=sid, key="v1",
        obj={"validation_study": {
            "kind": "survey_validation", "question": "Does one component help?",
            "sampling_frame": "Synthetic supporting data.", "data_sources": ["generated"],
            "variables": ["loss"], "analysis": "Compare summaries.",
            "falsification_condition": "No reduction.", "budget": "Ten minutes.",
        }}, emitter="ProblemModelingAgent")
    semantics = {
        "comparison_statement": (
            "Under the repository protocol, compare loss while allowing method changes."
        ),
        "research_question": "Does the method lower loss under the repository protocol?",
        "primary_metric": {"name": "loss", "direction": "minimize"},
        "comparison_mode": "repository_defined",
        "comparison_basis": "Use the repository benchmark.",
        "budget_interpretation": "Use its documented budget.",
        "conditions_to_hold_constant": ["dataset and evaluator"],
        "method_freedom": ["method implementation"],
        "evidence": [{
            "source_type": "existing_benchmark", "source_id": "repo:nanochat",
            "supports": "The repository defines the evaluator.",
        }],
        "uncertainties": [],
    }
    benchmark_ref = store.put(
        kind="benchmark_spec", session_id=sid, key="v1",
        obj={"evaluation_semantics": semantics}, emitter="EvaluatorAgent")
    calls = []

    def run_phase(session_id, repo_url, question, *, execution_phase,
                  frozen_baseline=None, overrides=None, evaluation_semantics=None,
                  resource_requirements=None, memory_scope=None):
        calls.append((session_id, repo_url, execution_phase, frozen_baseline, overrides,
                      evaluation_semantics, memory_scope))
        return ({
            "succeeded": True, "exit_code": 0, "metric_value": 0.9,
            "stdout_excerpt": "baseline=1.0 method=0.9", "stderr_excerpt": None,
            "environment": "repo_improve|verdict=new_best", "seed": "0",
            "runtime_seconds": 20.0, "artifacts": [],
        }, {
            "verdict": "new_best", "accepted": True, "reasons": [],
            "baseline_metric": 1.0, "method_metric": 0.9,
            "lower_is_better": True, "isolated": True,
        })

    monkeypatch.setattr(handler, "_run_repo_phase", run_phase)
    monkeypatch.setattr(
        handler, "_run_gen_experiment",
        lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("repo method bypassed the method-only sandbox")),
    )
    monkeypatch.setattr(
        handler_module, "run_validation_study",
        lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("auxiliary validation study replaced the repository Method run")),
    )

    await handler.handle_run_requested(client, _env(sid, {
        "benchmark_spec_ref": benchmark_ref,
        "baseline_result_ref": baseline_ref,
        "method_ref": "spec://method/s1/replacement",
        "spec_interpretation_ref": interpretation_ref,
        "confirmed_boundary": {"training_budget_s": 300},
    }))

    assert len(calls) == 1
    assert calls[0][0:3] == (
        sid, "https://github.com/example/nanochat", "method")
    assert calls[0][3]["metric_value"] == 1.0
    assert calls[0][3]["host_attestation"] is None
    assert calls[0][4] == {"training_budget_s": 300}
    assert calls[0][5] == {
        **semantics,
        "primary_metric": {
            **semantics["primary_metric"],
            "measurement": "Frozen from the reproduced official repository evaluator and baseline result.",
        },
    }
    assert calls[0][6] == "spec://method/s1/replacement"
    completed = [e for e in client.published if e.event_type == "experiment.run_completed"]
    assert len(completed) == 1 and completed[0].payload["verdict"] == "succeeded"
    run_record = store.get(completed[0].payload["run_record_ref"])
    assert run_record["experiment_id"] == "spec://method/s1/replacement"


# ---------------------------------------------------------------------------
# (b) H5: a FAIL verdict still yields succeeded=True (flows FORWARD, no loop).
# ---------------------------------------------------------------------------


async def test_gen_branch_fail_verdict_still_succeeds(store, monkeypatch):
    monkeypatch.setenv("AUTORESEARCH_GEN_EXPERIMENT", "1")
    monkeypatch.setattr(handler_module, "run_pipeline",
                        lambda b, l, q, *, design=None: _fake_pipeline_result(verdict="fail"))

    handler = RunnerAgentHandler(spec_store=store, backend=object(),
                                 p1_backend=object())
    client = FakeBusClient()
    sid = str(uuid.uuid4())
    _seed_reinterpreted_spec(store, sid, "goal")

    await handler.handle_run_requested(client, _env(sid, {}))

    completed = [p for p in client.published if p.event_type == "experiment.run_completed"]
    assert len(completed) == 1
    rec = store.get(completed[0].payload["run_record_ref"])
    assert rec["succeeded"] is True  # H5: honest negative flows forward
    assert completed[0].payload["verdict"] == "succeeded"


# ---------------------------------------------------------------------------
# (c) The P1 verdict+numbers are threaded to a sibling gen_experiment_result
# ref MemoryAgent (Task 3) can recover.
# ---------------------------------------------------------------------------


async def test_gen_branch_stores_recoverable_verdict_and_metrics(store, monkeypatch):
    monkeypatch.setenv("AUTORESEARCH_GEN_EXPERIMENT", "1")
    monkeypatch.setattr(handler_module, "run_pipeline",
                        lambda b, l, q, *, design=None: _fake_pipeline_result(verdict="pass"))

    handler = RunnerAgentHandler(spec_store=store, backend=object(),
                                 p1_backend=object())
    client = FakeBusClient()
    sid = str(uuid.uuid4())
    _seed_reinterpreted_spec(store, sid, "goal")

    await handler.handle_run_requested(client, _env(sid, {}))

    completed = [p for p in client.published if p.event_type == "experiment.run_completed"]
    run_record_ref = completed[0].payload["run_record_ref"]
    from autoresearch.spec_store.identity import result_for_run
    gen = result_for_run(store, run_record_ref)
    assert gen["verdict"] == "pass"
    assert gen["baseline_metric"] == 100.0
    assert gen["method_metric"] == 42.0
    assert gen["reasons"] == ["baseline=100 vs method=42 bins"]
    snapshot_ref = gen["run_record_snapshot_ref"]
    original_record = store.get(run_record_ref)
    assert snapshot_ref == run_record_ref
    assert store.get(snapshot_ref) == original_record
    store.put(kind="run_record", session_id=sid, key="v1",
              obj=dict(original_record, metric_value=99.0), emitter="RunnerAgent")
    assert store.get(snapshot_ref) == original_record
    candidate_ref = gen["candidate_snapshot_ref"]
    saved_candidate = store.get(candidate_ref)
    assert saved_candidate == {k: v for k, v in gen.items() if k != "candidate_snapshot_ref"}
    store.put(kind="gen_experiment_result", session_id=sid, key="v1",
              obj=dict(gen, method_metric=99.0), emitter="RunnerAgent")
    assert store.get(candidate_ref) == saved_candidate


# ---------------------------------------------------------------------------
# (d) reconstruct-question falls back to key "final" when "reinterpreted" absent.
# ---------------------------------------------------------------------------


async def test_gen_branch_question_falls_back_to_final(store, monkeypatch):
    monkeypatch.setenv("AUTORESEARCH_GEN_EXPERIMENT", "1")
    captured = {}
    monkeypatch.setattr(handler_module, "run_pipeline",
                        lambda b, l, q, *, design=None: captured.update(question=q) or _fake_pipeline_result())

    handler = RunnerAgentHandler(spec_store=store, backend=object(),
                                 p1_backend=object())
    client = FakeBusClient()
    sid = str(uuid.uuid4())
    _seed_reinterpreted_spec(store, sid, "final goal", key="final")

    await handler.handle_run_requested(client, _env(sid, {}))
    assert captured["question"] == "final goal"


# ---------------------------------------------------------------------------
# (e) DEFAULT-OFF: env off -> the default backend path runs, run_pipeline never
# called.
# ---------------------------------------------------------------------------


async def test_env_off_uses_default_path(store, monkeypatch):
    monkeypatch.delenv("AUTORESEARCH_GEN_EXPERIMENT", raising=False)

    def boom(*a, **k):
        raise AssertionError("run_pipeline must NOT be called with env off")

    monkeypatch.setattr(handler_module, "run_pipeline", boom)

    from autoresearch.agents._reuse.runner_backend import FakeRunnerBackend

    handler = RunnerAgentHandler(spec_store=store, backend=FakeRunnerBackend(exit_code=0),
                                 p1_backend=object())
    client = FakeBusClient()
    sid = str(uuid.uuid4())
    pref = store.put(kind="experiment_patch", session_id=sid, key="v1",
                     obj={"summary": "s", "changes": [], "touched_files": []},
                     emitter="CoderAgent")
    vref = store.put(kind="verifier_spec", session_id=sid, key="v1",
                     obj={"evaluator_hash": "sha256:x", "metric": "m",
                          "data_split": {"in_loop": "dev", "held_out": "test"},
                          "run_budget": "1", "seed_policy": "fixed",
                          "editable_region": ["a.py"], "forbidden_files": [],
                          "result_parse_schema": {"type": "object"}},
                     emitter="EvaluatorAgent")
    await handler.handle_run_requested(
        client, _env(sid, {"experiment_patch_ref": pref, "verifier_spec_ref": vref}))

    completed = [p for p in client.published if p.event_type == "experiment.run_completed"]
    assert len(completed) == 1  # default path ran, no crash


# ---------------------------------------------------------------------------
# (f) DEFAULT-OFF: env on but NO p1_backend -> default path (run_pipeline never
# called).
# ---------------------------------------------------------------------------


async def test_env_on_but_no_p1_backend_uses_default_path(store, monkeypatch):
    monkeypatch.setenv("AUTORESEARCH_GEN_EXPERIMENT", "1")

    def boom(*a, **k):
        raise AssertionError("run_pipeline must NOT be called without p1_backend")

    monkeypatch.setattr(handler_module, "run_pipeline", boom)

    from autoresearch.agents._reuse.runner_backend import FakeRunnerBackend

    handler = RunnerAgentHandler(spec_store=store, backend=FakeRunnerBackend(exit_code=0))
    client = FakeBusClient()
    sid = str(uuid.uuid4())
    pref = store.put(kind="experiment_patch", session_id=sid, key="v1",
                     obj={"summary": "s", "changes": [], "touched_files": []},
                     emitter="CoderAgent")
    vref = store.put(kind="verifier_spec", session_id=sid, key="v1",
                     obj={"evaluator_hash": "sha256:x", "metric": "m",
                          "data_split": {"in_loop": "dev", "held_out": "test"},
                          "run_budget": "1", "seed_policy": "fixed",
                          "editable_region": ["a.py"], "forbidden_files": [],
                          "result_parse_schema": {"type": "object"}},
                     emitter="EvaluatorAgent")
    await handler.handle_run_requested(
        client, _env(sid, {"experiment_patch_ref": pref, "verifier_spec_ref": vref}))

    completed = [p for p in client.published if p.event_type == "experiment.run_completed"]
    assert len(completed) == 1


async def test_current_round_artifact_keeps_check_expectation_report(store, monkeypatch, tmp_path):
    """Best-so-far storage must not hide the current ablation round from the orchestrator."""
    monkeypatch.setenv("AUTORESEARCH_GEN_EXPERIMENT", "1")
    monkeypatch.setenv("AUTORESEARCH_WEB_WORKDIR", str(tmp_path))
    handler = RunnerAgentHandler(spec_store=store, backend=object(), p1_backend=object())
    sid = str(uuid.uuid4())
    handler._last_operator = {sid: "check-expectation"}
    record = {
        "succeeded": True, "exit_code": 0, "metric_value": 1.1,
        "stdout_excerpt": "ablation", "stderr_excerpt": None,
        "environment": "docker/openhands", "seed": "0", "runtime_seconds": 1,
        "artifacts": [],
    }
    report = [{"component": "optimizer", "type": "gain", "verdict": "unmet"}]
    gen_result = {
        "verdict": "valid_not_best", "accepted": False, "reasons": [],
        "baseline_metric": 1.0, "method_metric": 1.1, "lower_is_better": True,
        "ablation_report": report,
    }
    monkeypatch.setattr(handler, "_run_gen_experiment",
                        lambda *args, **kwargs: (record, gen_result))
    client = FakeBusClient()

    await handler.handle_run_requested(client, _env(sid, {}))

    current = store.get(refs.encode("gen_experiment_result", sid, "current"))
    assert current["operator"] == "check-expectation"
    assert current["ablation_report"] == report


async def test_new_method_does_not_keep_prior_methods_better_result(
    store, monkeypatch, tmp_path,
):
    """A fundamentally new Method starts its own best-so-far lineage."""
    monkeypatch.setenv("AUTORESEARCH_GEN_EXPERIMENT", "1")
    monkeypatch.setenv("AUTORESEARCH_WEB_WORKDIR", str(tmp_path))
    handler = RunnerAgentHandler(spec_store=store, backend=object(), p1_backend=object())
    client = FakeBusClient()
    sid = str(uuid.uuid4())
    old_method = "spec://method/s/old"
    new_method = "spec://method/s/new"
    store.put(
        kind="gen_experiment_result", session_id=sid, key="v1",
        obj={"verdict": "new_best", "accepted": True, "baseline_metric": 1.0,
             "method_metric": 0.8, "lower_is_better": True,
             "method_ref": old_method},
        emitter="RunnerAgent",
    )
    record = {
        "succeeded": True, "exit_code": 0, "metric_value": 1.2,
        "stdout_excerpt": "new method run", "stderr_excerpt": None,
        "environment": "docker/openhands", "seed": "0", "runtime_seconds": 1,
        "artifacts": [],
    }
    gen_result = {
        "verdict": "valid_not_best", "accepted": False, "reasons": [],
        "baseline_metric": 1.0, "method_metric": 1.2, "lower_is_better": True,
    }
    monkeypatch.setattr(handler, "_run_gen_experiment",
                        lambda *args, **kwargs: (record, gen_result))

    await handler.handle_run_requested(client, _env(sid, {"method_ref": new_method}))

    best = store.get(refs.encode("gen_experiment_result", sid, "v1"))
    assert best["method_ref"] == new_method
    assert best["method_metric"] == 1.2


# --- ★任务保真度(老师 2026-08-07): REAL repo 实验失败绝不掉 GEN 自写评测 ---------

async def test_repo_failure_does_not_fall_to_gen_self_write(store, monkeypatch):
    # nanochat 那个 bug: repo agentic 的 ssh 派发超时(TimeoutExpired)被当成 "repo failed",
    # 掉进 GEN 自写管道 -> agent 给成熟基准仓库自造评测(4.763/9.573)。修复后: 有真实 repo 时
    # 失败就如实返回 INVALID, GEN 只兜底"根本没 repo"的任务。
    import subprocess
    import types
    from autoresearch.experiments.repo_improve import discover as discover_mod

    monkeypatch.setenv("AUTORESEARCH_GEN_EXPERIMENT", "1")
    monkeypatch.setenv("AUTORESEARCH_REPO_MODE", "auto")
    monkeypatch.setattr(discover_mod, "discover",
                        lambda q, llm: types.SimpleNamespace(
                            repo_url="https://github.com/karpathy/nanochat"))

    def _gen_must_not_run(*a, **k):
        raise AssertionError("GEN self-write pipeline must NOT run when a REAL repo experiment fails")
    monkeypatch.setattr(handler_module, "run_pipeline", _gen_must_not_run)

    handler = RunnerAgentHandler(spec_store=store, backend=object(),
                                 p1_backend=object(), p1_llm=object())
    monkeypatch.setattr(handler, "_needs_gpu", lambda q, url: False)   # CPU 本地路径

    def _raise_timeout(*a, **k):
        raise subprocess.TimeoutExpired(cmd="ssh", timeout=3600)       # 模拟 ssh 派发超时
    monkeypatch.setattr(handler, "_run_repo_experiment_local", _raise_timeout)

    client = FakeBusClient()
    sid = str(uuid.uuid4())
    _seed_reinterpreted_spec(store, sid, "Clone and improve nanochat: lower val BPB.")
    await handler.handle_run_requested(client, _env(sid, {}))

    # 没掉 GEN(没触发 AssertionError), 且返回的是诚实 INVALID 的 repo 结果(非自造 eval)
    completed = [p for p in client.published if p.event_type == "experiment.run_completed"]
    assert len(completed) == 1
    rec = store.get(completed[0].payload["run_record_ref"])
    assert "repo-failed" in rec["environment"] and "verdict=invalid" in rec["environment"]


# --- best-so-far guard (R5 outer loop must not let a later invalid round clobber
#     an earlier valid (baseline, method) pair) ---------------------------------

def test_gen_result_quality_ranks_best_so_far():
    from autoresearch.agents.runner_agent.handler import _gen_result_quality
    win = {"verdict": "new_best", "accepted": True,
           "baseline_metric": 1.70, "method_metric": 1.65}
    non_win = {"verdict": "valid_not_best", "accepted": False,
               "baseline_metric": 1.70, "method_metric": 1.76}
    evaluator_invalid = {"verdict": "invalid", "accepted": False,
                         "baseline_metric": 1.70, "method_metric": 1.99}
    no_value = {"verdict": "invalid", "accepted": False,
                "baseline_metric": 1.71, "method_metric": None}
    # strict ordering: trusted win > honest non-win > evaluator-invalid-with-value
    #                  > ran-but-no-method-value > missing
    assert (_gen_result_quality(win) > _gen_result_quality(non_win)
            > _gen_result_quality(evaluator_invalid) > _gen_result_quality(no_value)
            > _gen_result_quality(None))
    # THE bug: a later method=None round must rank strictly below any valued round,
    # so the write-guard keeps the earlier valid pair.
    assert _gen_result_quality(no_value) < _gen_result_quality(non_win)
    assert _gen_result_quality(no_value) < _gen_result_quality(win)
    # baseline-only (method=None but a baseline WAS measured) must beat a nothing round
    # (clone-fail / crash: no baseline, no method) so the guard keeps the measured baseline.
    baseline_only = {"verdict": "invalid", "accepted": False,
                     "baseline_metric": 94.0, "method_metric": None}
    nothing = {"verdict": "fail", "accepted": False,
               "baseline_metric": None, "method_metric": None}
    assert _gen_result_quality(baseline_only) > _gen_result_quality(nothing)


# --- paper-reproduction line (opt-in AUTORESEARCH_PAPER_REPRODUCTION) ----------

def test_assess_paper_reproduction_statuses():
    from autoresearch.agents.runner_agent.handler import _assess_paper_reproduction
    # within 5% tol -> reproduced
    r = _assess_paper_reproduction(1.02, 1.00, "GPT-2 paper", None, True, 0.05)
    assert r["status"] == "reproduced" and abs(r["rel_error"] - 0.02) < 1e-9
    # outside tol -> mismatch
    r = _assess_paper_reproduction(1.30, 1.00, "GPT-2 paper", None, True, 0.05)
    assert r["status"] == "mismatch" and abs(r["delta"] - 0.30) < 1e-9
    # nothing citable in the literature -> live reproduction stands
    r = _assess_paper_reproduction(1.02, None, None, None, True, 0.05)
    assert r["status"] == "no_published_baseline"
    # our run produced no baseline value
    r = _assess_paper_reproduction(None, 1.00, "x", None, True, 0.05)
    assert r["status"] == "no_measured_baseline"


def test_res_to_improve_json_threads_metric_name_and_direction():
    from autoresearch.agents.runner_agent.handler import _res_to_improve_json
    rr = _res_to_improve_json({"benchmark": "nanoGPT", "metric": "val_loss",
                               "lower_is_better": True, "verdict": {}, "report": ""})
    assert rr["metric"] == "val_loss" and rr["lower_is_better"] is True
