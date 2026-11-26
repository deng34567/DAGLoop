"""
Tests for MemoryAgent's P1 general-mode branch (ConsolidatingMemory).

DEFAULT-OFF: under AUTORESEARCH_GEN_EXPERIMENT=1, MemoryAgent DETERMINISTICALLY
(no LLM) recovers the P1 verdict+metrics from the sibling gen_experiment_result
ref (written by RunnerAgent's gen branch), seeds a ResearchMemory grounded on the
REAL numbers, and FORCES routing_verdict="report" (H6: guarantee we reach
Reporting, no re-loop). When off, the real LLM consolidate path runs unchanged
(test_handler.py).
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import pytest
import yaml
from jsonschema import Draft202012Validator

from autoresearch.agents.memory_agent.handler import (
    MemoryAgentHandler, _append_component_history, _formal_best_eligible,
)
from autoresearch.agents.memory_agent.tests.conftest import FakeLLM
from autoresearch.spec_store import SpecStore, refs
from autoresearch.spec_store.identity import artifact_digest

_CONTRACTS_DIR = Path(__file__).resolve().parents[1] / "data-contracts"
RESEARCH_MEMORY_VALIDATOR = Draft202012Validator(
    yaml.safe_load((_CONTRACTS_DIR / "research_memory.schema.yaml").read_text(encoding="utf-8"))
)


@pytest.mark.parametrize("operator", ["improve", "debug"])
@pytest.mark.parametrize("qualified,executed,expected", [
    (2, False, "component_contract_retry"),
    (0, False, "component_expectation_failed"),
    (2, True, "component_expectation_failed"),
])
def test_unexecuted_refinement_does_not_refute_retained_component(operator, qualified, executed, expected):
    from autoresearch.agents.memory_agent.handler import _classify_component_evidence
    gen = {
        "verdict": "invalid", "method_metric": None, "accepted": False,
        "operator": operator, "component_execution_started": executed,
        "component_execution": {
            "mode": "component", "component_id": "source", "operator": operator,
            "status": "execution_failed", "execution_succeeded": executed,
            "qualified_attempts": qualified, "checkpoint": "checkpoint-qualified" if qualified else "",
        },
        "ablation_report": [{
            "component": "source", "type": "indirect_gain", "actual": 0,
            "expected": {"operator": "==", "value": 1}, "verdict": "unmet",
            "evidence": {"source": "tracked_repository_source", "ref": "proposed parameter is absent"},
        }],
    }
    assert _classify_component_evidence(gen)[0] == expected
    gen["method_ref"] = "method-current"
    gen["diagnosis"] = "component_expectation_failed"
    gen["method_implementation_review"] = {
        "method_ref": "method-current", "plan_revision": {
            "component_id": "source", "plan_quote": "actual frozen contract",
            "reason": "grounded contract contradiction", "evidence": ["bound source review"],
        },
    }
    assert _classify_component_evidence(gen)[0] == "component_expectation_failed"


@dataclass
class _Published:
    event_type: str
    session_id: str
    payload: dict
    causation_id: Optional[str]


class FakeBusClient:
    def __init__(self) -> None:
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


def test_component_history_keeps_later_attempt_when_run_ref_is_reused():
    execution = {
        "mode": "component",
        "component_id": "c1",
        "method_fingerprint": "sha256:method",
        "round": 1,
        "attempts": 1,
        "operator": "draft",
        "status": "qualified",
    }
    gen = {"component_execution": execution, "method_metric": 1.1}
    first = _append_component_history({}, gen, "spec://run_record/s/v1", None)

    later_execution = dict(execution, round=2, attempts=2, operator="improve")
    later = _append_component_history(
        {"component_history": first},
        {"component_execution": later_execution, "method_metric": 1.0},
        "spec://run_record/s/v1",
        None,
    )
    repeated = _append_component_history(
        {"component_history": later},
        {"component_execution": later_execution, "method_metric": 1.0},
        "spec://run_record/s/v1",
        None,
    )

    assert [(item["round"], item["attempt"]) for item in later] == [(1, 1), (2, 2)]
    assert repeated == later


@pytest.fixture()
def store(tmp_path):
    s = SpecStore(tmp_path / "store")
    yield s
    s.close()


def _seed_gen_result(store, sid, verdict="pass", pb=100.0, pm=42.0, accepted=None,
                     diagnosis=None, diagnosis_evidence=None, lower_is_better=None):
    """Seed the RunRecord + sibling gen_experiment_result RunnerAgent writes.

    ``accepted`` defaults to ``verdict == "pass"`` (the general path's invariant); pass it
    explicitly to mimic the repo_improve path (verdict="new_best", accepted=True)."""
    if accepted is None:
        accepted = verdict == "pass"
    run_record_ref = store.put(
        kind="run_record", session_id=sid, key="v1",
        obj={"succeeded": True, "exit_code": 0, "metric_value": pm,
             "stdout_excerpt": "report", "environment": "docker/openhands"},
        emitter="RunnerAgent")
    result = {
        "verdict": verdict, "accepted": accepted,
        "reasons": ["r1", "r2"], "baseline_metric": pb,
        "method_metric": pm, "isolated": True,
        "diagnosis": diagnosis, "diagnosis_evidence": diagnosis_evidence or [],
    }
    if lower_is_better is not None:
        result["lower_is_better"] = lower_is_better
    store.put(
        kind="gen_experiment_result", session_id=sid, key="v1",
        obj=result,
        emitter="RunnerAgent")
    return run_record_ref


# ---------------------------------------------------------------------------
# env ON + pass run_record -> real numbers in verified_findings, best_metric,
# route report, NO llm call.
# ---------------------------------------------------------------------------


def test_gen_best_result_binds_verified_snapshot_and_preserves_prior_best(store):
    handler = MemoryAgentHandler(spec_store=store, llm_client=FakeLLM([]))
    sid = str(uuid.uuid4())
    memory = None
    for index, value in enumerate((42.0, 55.0, 31.0)):
        run_ref = _seed_gen_result(store, sid, pm=value, lower_is_better=True)
        snapshot = store.put(kind="run_record", session_id=sid,
                             key=f"snapshot-{index}", obj=store.get(run_ref),
                             emitter="RunnerAgent")
        result = store.get(refs.encode("gen_experiment_result", sid, "v1"))
        result["run_record_snapshot_ref"] = snapshot
        store.put(kind="gen_experiment_result", session_id=sid, key="v1",
                  obj=result, emitter="RunnerAgent")
        prior = memory
        memory, *_ = handler._consolidate_gen(
            run_ref, prior_research_memory=prior,
            verification_report={"verdict": "verified"})
        if index == 1:
            assert memory["best_result_ref"] == prior["best_result_ref"]
            assert memory["best_metric"] == 42.0
        else:
            assert memory["best_result_ref"] == snapshot
        assert store.get(memory["best_result_ref"])["metric_value"] == memory["best_metric"]


@pytest.mark.parametrize("lower,old,new,baseline", [
    (True, 1549.0, 1543.0, 2000.0),
    (False, 1543.0, 1549.0, 1000.0),
])
def test_small_qualified_improvement_replaces_formal_best_without_relative_floor(
        store, monkeypatch, lower, old, new, baseline):
    monkeypatch.delenv("AUTORESEARCH_BESTSOFAR_EPS_REL", raising=False)
    handler = MemoryAgentHandler(spec_store=store, llm_client=FakeLLM([]))
    sid = str(uuid.uuid4())
    prior = None
    for value in (old, new):
        run_ref = _seed_gen_result(store, sid, verdict="new_best", pb=baseline,
                                   pm=value, accepted=True, lower_is_better=lower)
        run = store.get(run_ref)
        snapshot_ref = store.put(
            kind="run_record", session_id=sid,
            key="snapshot-" + artifact_digest(run).split(":", 1)[1],
            obj=run, emitter="RunnerAgent")
        result_ref = refs.encode("gen_experiment_result", sid, "v1")
        result = store.get(result_ref)
        result.update(run_record_snapshot_ref=snapshot_ref,
            measurement_evidence={"protocol_fingerprint": "same", "minimum_improvement": 0})
        candidate_ref = store.put(kind="gen_experiment_result", session_id=sid,
            key="snapshot-" + artifact_digest(result).split(":", 1)[1],
            obj=result, emitter="RunnerAgent")
        result["candidate_snapshot_ref"] = candidate_ref
        store.put(kind="gen_experiment_result", session_id=sid, key="v1",
                  obj=result, emitter="RunnerAgent")
        prior, *_ = handler._consolidate_gen(
            run_ref, prior_research_memory=prior,
            verification_report={"verdict": "verified", "run_record_digest": artifact_digest(run)})
        assert prior["best_metric"] == value
        assert prior["best_result_ref"] == candidate_ref


@pytest.mark.parametrize("execution_mode", [None, "integration"])
@pytest.mark.parametrize("review_verdict", ["partial", "drifted"])
@pytest.mark.parametrize("prior_metric", [0.001, 0.07])
def test_partial_whole_result_cannot_displace_faithful_formal_best(
        store, execution_mode, review_verdict, prior_metric):
    handler = MemoryAgentHandler(spec_store=store, llm_client=FakeLLM([]))
    sid = str(uuid.uuid4())
    run_ref = _seed_gen_result(
        store, sid, verdict="new_best", pb=7.0, pm=0.07,
        accepted=True, lower_is_better=True)
    result_ref = refs.encode("gen_experiment_result", sid, "v1")
    result = store.get(result_ref)
    result.update(
        component_execution={"mode": execution_mode, "all_components_accepted": True},
        checkpoint="/saved/qualified-whole",
        method_ref=f"spec://method/{sid}/qualified",
        measurement_evidence={"value": 0.07, "valid": True, "stdout_sha256": "bound-stdout",
                              "protocol_fingerprint": "frozen-protocol",
                              "attempt_id": "attempt-qualified-whole",
                              "protocol": {"kind": "official_single"}},
        candidate_admissibility={"verdict": "pass", "attempt_id": "attempt-qualified-whole"},
        repository_comparison_review={"verdict": "pass"},
        method_implementation_review={
            "verdict": "faithful", "method_ref": f"spec://method/{sid}/qualified",
            "task_contract_review": {"verdict": "compliant"}},
    )
    run = store.get(run_ref)
    snapshot_ref = store.put(
        kind="run_record", session_id=sid,
        key="snapshot-" + artifact_digest(run).split(":", 1)[1],
        obj=run, emitter="RunnerAgent")
    result["run_record_snapshot_ref"] = snapshot_ref
    candidate_ref = store.put(
        kind="gen_experiment_result", session_id=sid,
        key="snapshot-" + artifact_digest(result).split(":", 1)[1],
        obj=result, emitter="RunnerAgent")
    result["candidate_snapshot_ref"] = candidate_ref
    store.put(kind="gen_experiment_result", session_id=sid, key="v1",
              obj=result, emitter="RunnerAgent")

    partial = dict(result, method_metric=prior_metric,
                   method_implementation_review={
                       "verdict": review_verdict,
                       "task_contract_review": {"verdict": "compliant"}})
    partial_ref = store.put(
        kind="gen_experiment_result", session_id=sid,
        key="snapshot-" + artifact_digest(partial).split(":", 1)[1],
        obj=partial, emitter="RunnerAgent")
    prior = {"best_metric": prior_metric, "best_result_ref": partial_ref}
    memory, *_ = handler._consolidate_gen(
        run_ref, prior_research_memory=prior,
        verification_report={"verdict": "verified"})
    assert memory["best_metric"] == 0.07
    assert memory["best_result_ref"] == candidate_ref
    assert any(partial_ref in item for item in memory["negative_results"])


@pytest.mark.parametrize("review_verdict", ["partial", "drifted"])
@pytest.mark.parametrize("candidate_metric", [0.001, 0.07])
def test_incomplete_revalidation_cannot_replace_qualified_incumbent(
        store, review_verdict, candidate_metric):
    handler = MemoryAgentHandler(spec_store=store, llm_client=FakeLLM([]))
    sid = str(uuid.uuid4())
    run_ref = _seed_gen_result(
        store, sid, verdict="new_best", pb=7.0, pm=0.07,
        accepted=True, lower_is_better=True)
    result_ref = refs.encode("gen_experiment_result", sid, "v1")
    qualified = store.get(result_ref)
    qualified.update(
        checkpoint="/saved/qualified-incumbent",
        method_ref=f"spec://method/{sid}/qualified",
        measurement_evidence={"value": 0.07, "valid": True, "stdout_sha256": "bound-stdout",
                              "protocol_fingerprint": "frozen-protocol",
                              "attempt_id": "attempt-qualified-incumbent",
                              "protocol": {"kind": "official_single"}},
        candidate_admissibility={"verdict": "pass", "attempt_id": "attempt-qualified-incumbent"},
        repository_comparison_review={"verdict": "pass"},
        method_implementation_review={"verdict": "faithful",
            "method_ref": f"spec://method/{sid}/qualified",
            "task_contract_review": {"verdict": "compliant"}},
    )
    qualified_ref = store.put(
        kind="gen_experiment_result", session_id=sid,
        key="snapshot-" + artifact_digest(qualified).split(":", 1)[1],
        obj=qualified, emitter="RunnerAgent")
    run_ref = _seed_gen_result(
        store, sid, verdict="new_best", pb=7.0, pm=candidate_metric,
        accepted=True, lower_is_better=True)
    partial = store.get(result_ref)
    partial["method_implementation_review"] = {
        "verdict": review_verdict, "task_contract_review": {"verdict": "compliant"}}
    store.put(kind="gen_experiment_result", session_id=sid, key="v1",
              obj=partial, emitter="RunnerAgent")
    memory, *_ = handler._consolidate_gen(
        run_ref, prior_research_memory={"best_metric": 0.07, "best_result_ref": qualified_ref},
        verification_report={"verdict": "verified"})
    assert memory["best_metric"] == 0.07
    assert memory["best_result_ref"] == qualified_ref
    assert store.get(result_ref)["method_metric"] == candidate_metric


@pytest.mark.parametrize("incumbent_admissibility", ["insufficient_evidence", "violation"])
def test_ineligible_incumbent_cannot_block_verified_protocol_result(store, incumbent_admissibility):
    handler = MemoryAgentHandler(spec_store=store, llm_client=FakeLLM([]))
    sid = str(uuid.uuid4())
    run_ref = _seed_gen_result(
        store, sid, verdict="new_best", pb=7.0, pm=0.07,
        accepted=True, lower_is_better=True)
    result = store.get(refs.encode("gen_experiment_result", sid, "v1"))
    result.update(
        metric_name="runtime_seconds", metric_unit="s",
        component_execution={"mode": "integration", "all_components_accepted": True},
        candidate_admissibility={"verdict": "pass"},
        measurement_evidence={"valid": True, "protocol_fingerprint": "official"},
        method_implementation_review={
            "verdict": "faithful", "task_contract_review": {"verdict": "compliant"}},
    )
    snapshot_ref = store.put(
        kind="run_record", session_id=sid,
        key="snapshot-" + artifact_digest(store.get(run_ref)).split(":", 1)[1],
        obj=store.get(run_ref), emitter="RunnerAgent")
    result["run_record_snapshot_ref"] = snapshot_ref
    candidate_ref = store.put(
        kind="gen_experiment_result", session_id=sid,
        key="snapshot-" + artifact_digest(result).split(":", 1)[1],
        obj=result, emitter="RunnerAgent")
    old = dict(result, method_metric=0.24,
               candidate_admissibility={"verdict": incumbent_admissibility},
               measurement_evidence={"valid": True, "protocol_fingerprint": None})
    old_ref = store.put(
        kind="gen_experiment_result", session_id=sid, key="old",
        obj=old, emitter="RunnerAgent")
    result["candidate_snapshot_ref"] = candidate_ref
    store.put(kind="gen_experiment_result", session_id=sid, key="v1",
              obj=result, emitter="RunnerAgent")
    memory, *_ = handler._consolidate_gen(
        run_ref, prior_research_memory={"best_metric": 0.24, "best_result_ref": old_ref},
        verification_report={"verdict": "verified"})
    assert memory["best_metric"] == 0.07
    assert memory["best_result_ref"] == candidate_ref
    assert store.get(old_ref) == old
    assert any(old_ref in item for item in memory["negative_results"])


def test_formal_best_respects_admissibility_and_protocol_comparability(store):
    handler = MemoryAgentHandler(spec_store=store, llm_client=FakeLLM([]))
    sid = str(uuid.uuid4())
    run_ref = _seed_gen_result(store, sid, pm=0.07, lower_is_better=True)
    result = store.get(refs.encode("gen_experiment_result", sid, "v1"))
    result.update(
        component_execution={"mode": "integration", "all_components_accepted": True},
        method_implementation_review={
            "verdict": "faithful", "task_contract_review": {"verdict": "compliant"}},
        candidate_admissibility={"verdict": "pass"},
        measurement_evidence={"valid": True, "protocol_fingerprint": "new"},
    )
    old = dict(result, method_metric=0.24,
               measurement_evidence={"valid": True, "protocol_fingerprint": "different"})
    old_ref = store.put(kind="gen_experiment_result", session_id=sid, key="old",
                        obj=old, emitter="RunnerAgent")
    store.put(kind="gen_experiment_result", session_id=sid, key="v1",
              obj=result, emitter="RunnerAgent")
    prior = {"best_metric": 0.24, "best_result_ref": old_ref}
    memory, *_ = handler._consolidate_gen(
        run_ref, prior_research_memory=prior, verification_report={"verdict": "verified"})
    assert memory["best_metric"] == prior["best_metric"]
    assert memory["best_result_ref"] == old_ref
    assert _formal_best_eligible(dict(result, candidate_admissibility={
        "verdict": "insufficient_evidence"})) is False


def test_formal_best_rejects_revalidation_candidate_that_failed_its_adoption_gate():
    attempt_id = "attempt-paired-candidate"
    result = {
        "accepted": True,
        "method_metric": 0.9,
        "measurement_evidence": {"attempt_id": attempt_id},
        "ablation_report": [{
            "component": "fast-path",
            "type": "direct_gain",
            "actual": 0.9,
            "expected": {"operator": "<", "value": 0.8},
            "evidence": {"source": "runtime_attempt", "ref": attempt_id},
        }],
    }

    assert _formal_best_eligible(result) is False
    assert _formal_best_eligible({**result, "method_metric": 1.0}) is True


def test_partial_integration_is_not_promoted_as_first_formal_best(store):
    handler = MemoryAgentHandler(spec_store=store, llm_client=FakeLLM([]))
    sid = str(uuid.uuid4())
    run_ref = _seed_gen_result(
        store, sid, verdict="new_best", pb=7.0, pm=0.001,
        accepted=True, lower_is_better=True)
    result_ref = refs.encode("gen_experiment_result", sid, "v1")
    result = store.get(result_ref)
    result.update(
        component_execution={"mode": "integration", "all_components_accepted": True},
        method_implementation_review={
            "verdict": "partial", "task_contract_review": {"verdict": "compliant"}},
    )
    store.put(kind="gen_experiment_result", session_id=sid, key="v1",
              obj=result, emitter="RunnerAgent")
    memory, *_ = handler._consolidate_gen(
        run_ref, verification_report={"verdict": "verified"})
    assert memory["best_metric"] is None
    assert memory["best_result_ref"] is None


def test_formal_best_rechecks_explicit_import_limit_in_old_receipt():
    result = {
        "component_execution": {"mode": "integration", "all_components_accepted": True},
        "method_implementation_review": {"verdict": "faithful",
                                         "task_contract_review": {"verdict": "compliant"}},
        "repository_contract": {"sources": [{"path": "instruction.md", "content":
            "Allowed imports: `numpy`, `torch`. No other libraries."}]},
        "diff": "diff --git a/solve.py b/solve.py\n+++ b/solve.py\n@@\n+import time\n",
    }

    assert _formal_best_eligible(result) is False
    assert _formal_best_eligible({**result, "diff": result["diff"].replace(
        "import time", "import torch")}) is True


def test_memory_reclassifies_old_accepted_result_with_forbidden_import(store):
    handler = MemoryAgentHandler(spec_store=store, llm_client=FakeLLM([]))
    sid = str(uuid.uuid4())
    run_ref = _seed_gen_result(
        store, sid, verdict="new_best", pb=17988.0, pm=1217.0,
        accepted=True, lower_is_better=True)
    result_ref = refs.encode("gen_experiment_result", sid, "v1")
    result = store.get(result_ref)
    result.update(
        component_execution={"mode": "integration", "all_components_accepted": True},
        method_implementation_review={"verdict": "faithful",
                                      "task_contract_review": {"verdict": "compliant"}},
        repository_contract={"sources": [{"path": "instruction.md", "content":
            "Allowed imports: `numpy`, `torch`. No other libraries."}]},
        diff="diff --git a/solve.py b/solve.py\n+++ b/solve.py\n@@\n+import time\n",
    )
    prior_best_ref = store.put(
        kind="gen_experiment_result", session_id=sid, key="prior-best",
        obj=result, emitter="RunnerAgent")
    store.put(kind="gen_experiment_result", session_id=sid, key="v1",
              obj=result, emitter="RunnerAgent")

    memory, *_ = handler._consolidate_gen(
        run_ref, verification_report={"verdict": "verified"},
        prior_research_memory={"best_metric": 1217.0,
            "best_result_ref": prior_best_ref,
            "verified_findings": ["Method beats baseline: baseline=17988.0 vs "
                                  "method=1217.0 (metric value)."]})

    assert memory["best_result_ref"] is None
    assert memory["diagnosis"] == "candidate_contract_violation"
    assert not memory["verified_findings"]
    assert any("import time" in item for item in memory["negative_results"])


def test_memory_records_forbidden_import_in_old_component_observation(store):
    handler = MemoryAgentHandler(spec_store=store, llm_client=FakeLLM([]))
    sid = str(uuid.uuid4())
    run_ref = _seed_gen_result(
        store, sid, verdict="invalid", pb=17988.0, pm=1153.0,
        accepted=False, lower_is_better=True)
    result_ref = refs.encode("gen_experiment_result", sid, "v1")
    result = store.get(result_ref)
    result.update(
        component_execution={"mode": "component", "status": "drifted"},
        repository_contract={"sources": [{"path": "instruction.md", "content":
            "Allowed imports: `numpy`, `torch`. No other libraries."}]},
        diff="diff --git a/solve.py b/solve.py\n+++ b/solve.py\n@@\n+import sys\n",
    )
    store.put(kind="gen_experiment_result", session_id=sid, key="v1",
              obj=result, emitter="RunnerAgent")

    memory, *_ = handler._consolidate_gen(
        run_ref, verification_report={"verdict": "verified"})

    assert memory["diagnosis"] == "candidate_contract_violation"
    assert any(item.startswith("Task-contract violation") and "import sys" in item
               for item in memory["negative_results"])


@pytest.mark.parametrize("case", ["missing_verification", "failed_verification", "wrong_snapshot", "component"])
def test_gen_best_result_does_not_bind_unverified_or_unrelated_snapshot(store, case):
    handler = MemoryAgentHandler(spec_store=store, llm_client=FakeLLM([]))
    sid = str(uuid.uuid4())
    run_ref = _seed_gen_result(store, sid, pm=42.0, lower_is_better=True)
    record = store.get(run_ref)
    snapshot = store.put(kind="run_record", session_id=sid, key="snapshot",
                         obj=dict(record, metric_value=43.0) if case == "wrong_snapshot" else record,
                         emitter="RunnerAgent")
    result = store.get(refs.encode("gen_experiment_result", sid, "v1"))
    result["run_record_snapshot_ref"] = snapshot
    if case == "component":
        result["component_execution"] = {"mode": "component", "status": "accepted"}
    store.put(kind="gen_experiment_result", session_id=sid, key="v1",
              obj=result, emitter="RunnerAgent")
    verification = None if case == "missing_verification" else {
        "verdict": "failed" if case == "failed_verification" else "verified"}
    memory, *_ = handler._consolidate_gen(run_ref, verification_report=verification)
    assert memory["best_result_ref"] is None


async def test_gen_branch_win_stops_the_loop_and_reports(store, monkeypatch):
    # STOP-ON-WIN: a trusted win ends the loop immediately (routing report) -- we found an
    # improvement, no need to burn more rounds.
    monkeypatch.setenv("AUTORESEARCH_GEN_EXPERIMENT", "1")
    llm = FakeLLM([])
    handler = MemoryAgentHandler(spec_store=store, llm_client=llm)
    client = FakeBusClient()

    sid = str(uuid.uuid4())
    run_record_ref = _seed_gen_result(store, sid, verdict="pass", pb=100.0, pm=42.0)
    env = _Envelope(str(uuid.uuid4()), sid, "memory.consolidation_requested",
                    "ResearchOrchestrator", {"run_record_ref": run_record_ref})

    await handler.handle_consolidation_requested(client, env)

    assert llm.calls == []  # deterministic, no LLM

    consolidated = [p for p in client.published if p.event_type == "memory.consolidated"]
    assert len(consolidated) == 1
    assert consolidated[0].payload["routing_verdict"] == "report"

    memory = store.get(consolidated[0].payload["research_memory_ref"])
    errors = list(RESEARCH_MEMORY_VALIDATOR.iter_errors(memory))
    assert errors == [], [e.message for e in errors]
    assert memory["best_metric"] == 42.0
    assert len(memory["verified_findings"]) == 1
    finding = memory["verified_findings"][0]
    assert "100.0" in finding and "42.0" in finding  # REAL numbers verbatim
    assert memory["negative_results"] == []


@pytest.mark.parametrize("role,method_metric", [("functional", 1.1), ("gain", 0.9)])
async def test_accepted_component_advances_even_without_whole_method_win(
        store, monkeypatch, role, method_metric):
    monkeypatch.setenv("AUTORESEARCH_GEN_EXPERIMENT", "1")
    handler = MemoryAgentHandler(spec_store=store, llm_client=FakeLLM([]))
    client = FakeBusClient()
    sid = str(uuid.uuid4())
    verdict = "valid_not_best" if role == "functional" else "new_best"
    run_ref = _seed_gen_result(
        store, sid, verdict=verdict, pb=1.0, pm=method_metric,
        accepted=role == "gain", lower_is_better=True)
    result_ref = refs.encode("gen_experiment_result", sid, "v1")
    result = store.get(result_ref)
    result["component_execution"] = {
        "mode": "component", "component_id": "c1", "role": role,
        "plan_version": 2, "method_fingerprint": "sha256:method", "round": 3,
        "attempts": 4, "operator": "improve", "execution_succeeded": True,
        "dependency_closure": ["a"], "parent_checkpoints": ["/cp/a"],
        "checkpoint": "/cp/c1", "diff_ref": "/cp/c1#git-diff", "became_best": True,
        "status": "accepted", "contract_result": {"passed": True, "reason": "passed"},
    }
    store.put(kind="gen_experiment_result", session_id=sid, key="v1",
              obj=result, emitter="RunnerAgent")

    await handler.handle_consolidation_requested(
        client, _Envelope(str(uuid.uuid4()), sid, "memory.consolidation_requested",
                          "ResearchOrchestrator", {"run_record_ref": run_ref}))

    event = [item for item in client.published
             if item.event_type == "memory.consolidated"][-1]
    assert event.payload["diagnosis"] == "component_accepted"
    assert event.payload["routing_verdict"] == "need_experiment"
    memory = store.get(event.payload["research_memory_ref"])
    assert len(memory["component_history"]) == 1
    assert memory["component_history"][0]["component_id"] == "c1"
    assert memory["component_history"][0]["run_ref"] == run_ref
    assert memory["component_history"][0]["plan_version"] == 2
    assert memory["component_history"][0]["round"] == 3
    assert memory["component_history"][0]["attempt"] == 4
    assert memory["component_history"][0]["acceptance_result"]["passed"] is True
    assert memory["component_history"][0]["became_best"] is True
    assert memory["component_history"][0]["diff_ref"] == "/cp/c1#git-diff"


async def test_indirect_component_metric_stays_in_component_history_not_global_best(
        store, monkeypatch):
    monkeypatch.setenv("AUTORESEARCH_GEN_EXPERIMENT", "1")
    handler = MemoryAgentHandler(spec_store=store, llm_client=FakeLLM([]))
    client = FakeBusClient()
    sid = str(uuid.uuid4())
    run_ref = _seed_gen_result(
        store, sid, verdict="valid_not_best", pb=107.0, pm=None,
        accepted=False, lower_is_better=True)
    result_ref = refs.encode("gen_experiment_result", sid, "v1")
    result = store.get(result_ref)
    result["component_execution"] = {
        "mode": "component", "component_id": "profile-hotspot",
        "role": "indirect_gain", "status": "accepted",
        "plan_version": 1, "method_fingerprint": "sha256:method", "round": 2,
        "attempts": 2, "operator": "improve", "execution_succeeded": True,
        "checkpoint": "/cp/profile", "became_best": True,
        "attempt_contract_result": {
            "passed": True, "measurement": "indirect_gain_check",
            "actual": 0.982236892261024,
            "expected": {"operator": ">=", "value": 0.9},
        },
    }
    store.put(kind="gen_experiment_result", session_id=sid, key="v1",
              obj=result, emitter="RunnerAgent")

    await handler.handle_consolidation_requested(
        client, _Envelope(str(uuid.uuid4()), sid, "memory.consolidation_requested",
                          "ResearchOrchestrator", {"run_record_ref": run_ref}))

    event = [item for item in client.published
             if item.event_type == "memory.consolidated"][-1]
    memory = store.get(event.payload["research_memory_ref"])
    assert event.payload["diagnosis"] == "component_accepted"
    assert event.payload["routing_verdict"] == "need_experiment"
    assert memory["best_metric"] is None
    assert memory["verified_findings"] == []
    assert memory["component_history"][0]["metric"] is None
    assert memory["component_history"][0]["acceptance_result"]["actual"] == 0.982236892261024


async def test_direct_component_metric_stays_in_component_history_not_global_best(
        store, monkeypatch):
    monkeypatch.setenv("AUTORESEARCH_GEN_EXPERIMENT", "1")
    handler = MemoryAgentHandler(spec_store=store, llm_client=FakeLLM([]))
    client = FakeBusClient()
    sid = str(uuid.uuid4())
    run_ref = _seed_gen_result(
        store, sid, verdict="new_best", pb=16641.0, pm=2081.0,
        accepted=True, lower_is_better=True)
    result_ref = refs.encode("gen_experiment_result", sid, "v1")
    result = store.get(result_ref)
    result["component_execution"] = {
        "mode": "component", "component_id": "hd-compress",
        "role": "direct_gain", "status": "accepted",
        "plan_version": 1, "method_fingerprint": "sha256:method", "round": 1,
        "attempts": 1, "operator": "draft", "execution_succeeded": True,
        "checkpoint": "/cp/hd-compress", "became_best": True,
        "attempt_contract_result": {
            "passed": True, "measurement": "signed_metric_improvement",
            "parent_metric": 16641.0, "candidate_metric": 2081.0,
        },
    }
    store.put(kind="gen_experiment_result", session_id=sid, key="v1",
              obj=result, emitter="RunnerAgent")

    await handler.handle_consolidation_requested(
        client, _Envelope(str(uuid.uuid4()), sid, "memory.consolidation_requested",
                          "ResearchOrchestrator", {"run_record_ref": run_ref}))

    event = [item for item in client.published
             if item.event_type == "memory.consolidated"][-1]
    memory = store.get(event.payload["research_memory_ref"])
    assert event.payload["diagnosis"] == "component_accepted"
    assert event.payload["routing_verdict"] == "need_experiment"
    assert memory["best_metric"] is None
    assert memory["verified_findings"] == []
    assert memory["component_history"][0]["metric"] == 2081.0


@pytest.mark.parametrize("status", ["accepted", "qualified", "improving"])
def test_verified_component_observation_survives_later_component_failure(store, monkeypatch, status):
    """A later component failure must not erase an accepted official component result."""
    monkeypatch.setenv("AUTORESEARCH_GEN_EXPERIMENT", "1")
    handler = MemoryAgentHandler(spec_store=store, llm_client=FakeLLM([]))
    sid = str(uuid.uuid4())
    run_ref = _seed_gen_result(
        store, sid, verdict="new_best", pb=17988.0, pm=913.0,
        accepted=True, lower_is_better=True)
    result_ref = refs.encode("gen_experiment_result", sid, "v1")
    result = store.get(result_ref)
    result["component_execution"] = {
        "mode": "component", "component_id": "score-mlp",
        "role": "direct_gain", "status": status,
        "execution_succeeded": True,
        "attempt_contract_result": {"passed": True},
    }
    result["method_implementation_review"] = {"verdict": "faithful"}
    store.put(kind="gen_experiment_result", session_id=sid, key="v1",
              obj=result, emitter="RunnerAgent")

    memory, _, _, _ = handler._consolidate_gen(
        run_ref, verification_report={"verdict": "verified", "reasoning": "metric matches"})
    assert memory["best_metric"] is None
    assert len(memory["verified_findings"]) == 1
    assert memory["verified_findings"][0].startswith("Verified component observation:")
    assert "metric=913.0" in memory["verified_findings"][0]

    failed = dict(result, verdict="failed", accepted=False, method_metric=None)
    failed["component_execution"] = {
        "mode": "component", "component_id": "train-ce",
        "role": "direct_gain", "status": "execution_failed",
        "execution_succeeded": False,
        "attempt_contract_result": {"passed": False},
    }
    store.put(kind="gen_experiment_result", session_id=sid, key="v1",
              obj=failed, emitter="RunnerAgent")
    memory_after_failure, _, _, _ = handler._consolidate_gen(
        run_ref, prior_research_memory=memory,
        verification_report={"verdict": "failed", "checks": ["result missing"]})
    assert memory_after_failure["best_metric"] is None
    assert memory_after_failure["verified_findings"] == memory["verified_findings"]


@pytest.mark.parametrize("failed_check", [False, True])
def test_qualified_component_does_not_publish_vetoed_or_failure_verified_value(
        store, monkeypatch, failed_check):
    monkeypatch.setenv("AUTORESEARCH_GEN_EXPERIMENT", "1")
    handler = MemoryAgentHandler(spec_store=store, llm_client=FakeLLM([]))
    sid = str(uuid.uuid4())
    run_ref = _seed_gen_result(store, sid, verdict="new_best", pb=100., pm=50., accepted=True)
    ref = refs.encode("gen_experiment_result", sid, "v1")
    result = store.get(ref)
    result["component_execution"] = {
        "mode": "component", "component_id": "compress", "role": "direct_gain",
        "status": "qualified", "execution_succeeded": True,
        "attempt_contract_result": {"passed": True},
    }
    result["method_implementation_review"] = {"verdict": "faithful"}
    store.put(kind="gen_experiment_result", session_id=sid, key="v1",
              obj=result, emitter="RunnerAgent")
    eval_ref = store.put(kind="evaluation_report", session_id=sid, key="v1",
        obj={"verdict": "new_best" if failed_check else "invalid"}, emitter="EvaluatorAgent")
    memory, _, _, _ = handler._consolidate_gen(
        run_ref, evaluation_report_ref=eval_ref,
        verification_report={"verdict": "verified",
            "checks": ["failure_confirmed: invalid outcome"] if failed_check else []})
    assert memory["verified_findings"] == []
    assert memory["best_metric"] is None


async def test_qualified_component_routes_to_improve_without_advancing_dag(store, monkeypatch):
    monkeypatch.setenv("AUTORESEARCH_GEN_EXPERIMENT", "1")
    handler = MemoryAgentHandler(spec_store=store, llm_client=FakeLLM([]))
    client = FakeBusClient()
    sid = str(uuid.uuid4())
    run_ref = _seed_gen_result(
        store, sid, verdict="valid_not_best", pb=1.0, pm=1.1,
        accepted=False, lower_is_better=True)
    result_ref = refs.encode("gen_experiment_result", sid, "v1")
    result = store.get(result_ref)
    result["component_execution"] = {
        "mode": "component", "component_id": "c1", "role": "functional",
        "status": "qualified", "contract_result": {"passed": True},
    }
    store.put(kind="gen_experiment_result", session_id=sid, key="v1",
              obj=result, emitter="RunnerAgent")

    await handler.handle_consolidation_requested(
        client, _Envelope(str(uuid.uuid4()), sid, "memory.consolidation_requested",
                          "ResearchOrchestrator", {"run_record_ref": run_ref}))

    event = [item for item in client.published
             if item.event_type == "memory.consolidated"][-1]
    assert event.payload["diagnosis"] == "component_progress"
    assert event.payload["routing_verdict"] == "need_experiment"


@pytest.mark.parametrize("pm", [None, 0.8])
async def test_failed_component_routes_to_same_component_retry(store, monkeypatch, pm):
    monkeypatch.setenv("AUTORESEARCH_GEN_EXPERIMENT", "1")
    handler = MemoryAgentHandler(spec_store=store, llm_client=FakeLLM([]))
    client = FakeBusClient()
    sid = str(uuid.uuid4())
    run_ref = _seed_gen_result(
        store, sid, verdict="invalid" if pm is None else "new_best", pb=1.0, pm=pm,
        accepted=False, lower_is_better=True)
    result_ref = refs.encode("gen_experiment_result", sid, "v1")
    result = store.get(result_ref)
    result["component_execution"] = {
        "mode": "component", "component_id": "c1",
        "role": "functional" if pm is None else "direct_gain",
        "status": "execution_failed",
        "execution_succeeded": pm is not None,
        "contract_result": {"passed": False, "reason": (
            "candidate crashed" if pm is None
            else "direct-gain component lacks finite parent/candidate metrics")},
    }
    store.put(kind="gen_experiment_result", session_id=sid, key="v1",
              obj=result, emitter="RunnerAgent")

    await handler.handle_consolidation_requested(
        client, _Envelope(str(uuid.uuid4()), sid, "memory.consolidation_requested",
                          "ResearchOrchestrator", {"run_record_ref": run_ref}))

    event = [item for item in client.published
             if item.event_type == "memory.consolidated"][-1]
    assert event.payload["diagnosis"] == "component_contract_retry"
    assert event.payload["routing_verdict"] == "need_experiment"


@pytest.mark.parametrize("pm", [None, 1.1])
async def test_component_probe_success_does_not_falsify_whole_method(
        store, monkeypatch, pm):
    """A passed local probe is not a completed, non-improving Method experiment."""
    monkeypatch.setenv("AUTORESEARCH_GEN_EXPERIMENT", "1")
    handler = MemoryAgentHandler(spec_store=store, llm_client=FakeLLM([]))
    client = FakeBusClient()
    sid = str(uuid.uuid4())
    run_ref = _seed_gen_result(
        store, sid, verdict="invalid", pb=1.0, pm=pm, accepted=False)
    result = store.get(refs.encode("gen_experiment_result", sid, "v1"))
    result.update({
        "operator": "check-expectation",
        "component_execution": {
            "mode": "component", "component_id": "current-component",
            "status": "execution_failed", "all_components_accepted": False,
        },
        "ablation_report": [{
            "component": "current-component", "type": "direct_gain",
            "actual": 0.8, "expected": {"operator": "<", "value": 1.0},
            "evidence": {"source": "runtime_attempt", "ref": "candidate-vs-parent"},
            "verdict": "met",
        }],
    })
    store.put(kind="gen_experiment_result", session_id=sid, key="v1",
              obj=result, emitter="RunnerAgent")
    await handler.handle_consolidation_requested(
        client, _Envelope(str(uuid.uuid4()), sid, "memory.consolidation_requested",
                          "ResearchOrchestrator", {"run_record_ref": run_ref}))
    event = [item for item in client.published
             if item.event_type == "memory.consolidated"][-1]
    assert event.payload["diagnosis"] != "method_hypothesis_falsified"


@pytest.mark.parametrize("operator", ["draft", "debug", "improve"])
@pytest.mark.parametrize("status", ["implementation_failed", "contract_failed", "execution_failed"])
def test_measured_failed_component_keeps_local_retry_with_ablation_report(operator, status):
    from autoresearch.agents.memory_agent.handler import _classify_component_evidence

    diagnosis, evidence = _classify_component_evidence({
        "operator": operator, "verdict": "new_best", "accepted": True,
        "baseline_metric": 1.0, "method_metric": 1.2,
        "component_execution": {
            "mode": "component", "component_id": "candidate", "status": status,
            "contract_result": {
                "passed": False, "measurement": "signed_metric_improvement",
                "parent_metric": 1.5, "candidate_metric": 1.2,
                "reason": "direct-gain component did not improve over its parent checkpoint",
            },
        },
        "ablation_report": [{
            "component": "candidate", "type": "direct_gain", "actual": 1.2,
            "expected": {"operator": ">", "value": 1.0},
            "evidence": {"source": "runtime_attempt", "ref": "attempt-candidate"},
            "verdict": "met",
        }],
    })
    assert diagnosis == "component_contract_retry"
    assert "parent checkpoint" in evidence[0]


@pytest.mark.parametrize("status", ["implementation_failed", "contract_failed", "execution_failed"])
def test_passed_component_probe_with_missing_implementation_evidence_stays_local(status):
    from autoresearch.agents.memory_agent.handler import _classify_component_evidence

    diagnosis, evidence = _classify_component_evidence({
        "operator": "debug", "verdict": "invalid", "method_metric": None,
        "component_execution": {
            "mode": "component", "component_id": "inspection", "status": status,
            "role": "indirect_gain", "execution_succeeded": True,
            "contract_result": {"passed": True, "measurement": "indirect_gain_check"},
        },
        "method_implementation_review": {"verdict": "insufficient_evidence"},
        "ablation_report": [{
            "component": "inspection", "type": "indirect_gain", "actual": 0,
            "expected": {"operator": "==", "value": 0},
            "evidence": {"source": "runtime_attempt", "ref": "attempt-inspection"},
            "verdict": "met",
        }],
    })
    assert diagnosis == "component_contract_retry"
    assert evidence


async def test_component_parent_rejection_is_not_replaced_by_global_baseline_judgment(
        store, monkeypatch):
    monkeypatch.setenv("AUTORESEARCH_GEN_EXPERIMENT", "1")
    handler = MemoryAgentHandler(spec_store=store, llm_client=FakeLLM([]))
    client = FakeBusClient()
    sid = str(uuid.uuid4())
    run_ref = _seed_gen_result(store, sid, verdict="new_best", pb=1.0, pm=1.2,
                               accepted=True, lower_is_better=False)
    result_ref = refs.encode("gen_experiment_result", sid, "v1")
    result = store.get(result_ref)
    result.update({
        "operator": "draft",
        "component_execution": {
            "mode": "component", "component_id": "candidate",
            "status": "contract_failed", "execution_succeeded": True,
            "role": "direct_gain", "became_best": False,
            "contract_result": {
                "passed": False, "measurement": "signed_metric_improvement",
                "parent_metric": 1.5, "candidate_metric": 1.2,
                "reason": "direct-gain component did not improve over its parent checkpoint",
            },
        },
        "ablation_report": [{
            "component": "candidate", "type": "direct_gain", "actual": 1.2,
            "expected": {"operator": ">", "value": 1.0},
            "evidence": {"source": "runtime_attempt", "ref": "attempt-candidate"},
            "verdict": "met",
        }],
    })
    store.put(kind="gen_experiment_result", session_id=sid, key="v1",
              obj=result, emitter="RunnerAgent")
    await handler.handle_consolidation_requested(
        client, _Envelope(str(uuid.uuid4()), sid, "memory.consolidation_requested",
                          "ResearchOrchestrator", {"run_record_ref": run_ref}))
    event = [item for item in client.published
             if item.event_type == "memory.consolidated"][-1]
    assert event.payload["diagnosis"] == "component_contract_retry"
    assert event.payload["routing_verdict"] == "need_experiment"
    assert "parent checkpoint" in event.payload["diagnosis_evidence"][0]


@pytest.mark.parametrize("source,executed,expected_diagnosis", [
    ("runtime_attempt", False, "component_contract_retry"),
    ("runtime_attempt", True, "component_expectation_failed"),
    ("tracked_repository_source", False, "component_expectation_failed"),
])
def test_failed_execution_cannot_supply_measured_zero_for_component_redesign(
        store, source, executed, expected_diagnosis):
    handler = MemoryAgentHandler(spec_store=store, llm_client=FakeLLM([]))
    sid = str(uuid.uuid4())
    run_ref = _seed_gen_result(store, sid, verdict="invalid", pb=0.73,
                               pm=None, accepted=False)
    result = store.get(refs.encode("gen_experiment_result", sid, "v1"))
    result.update({
        "operator": "draft",
        "component_execution": {
            "mode": "component", "component_id": "current-component",
            "status": "execution_failed", "execution_succeeded": executed,
            "attempts": 1,
            "contract_result": {"passed": False, "reason": "no finite metric"},
        },
        "ablation_report": [{
            "component": "current-component", "type": "direct_gain",
            "actual": 0, "expected": {"operator": ">", "value": 0.73},
            "evidence": {"source": source, "ref": "attempt-or-source-evidence"},
            "verdict": "unmet",
        }],
    })
    store.put(kind="gen_experiment_result", session_id=sid, key="v1",
              obj=result, emitter="RunnerAgent")
    memory, _, diagnosis, _ = handler._consolidate_gen(
        run_ref, verification_report={"verdict": "failed", "checks": ["no metric"]})
    assert diagnosis == expected_diagnosis
    assert memory["best_metric"] is None


@pytest.mark.parametrize("receipt,expected", [
    ("attempt-check", "component_expectation_failed"),
    (None, "component_contract_retry"),
    ("attempt-other", "component_contract_retry"),
])
def test_bound_failed_check_reaches_revision_without_becoming_a_best_result(store, receipt, expected):
    handler = MemoryAgentHandler(spec_store=store, llm_client=FakeLLM([]))
    sid = str(uuid.uuid4())
    run_ref = _seed_gen_result(store, sid, verdict="invalid", pm=None, accepted=False)
    result_ref = refs.encode("gen_experiment_result", sid, "v1")
    result = store.get(result_ref)
    result.update({
        "operator": "draft", "component_failed_attempt_id": receipt,
        "component_execution": {
            "mode": "component", "component_id": "property-check", "role": "indirect_gain",
            "status": "contract_failed", "execution_succeeded": False, "attempts": 1,
            "contract_result": {"passed": False, "measurement": "indirect_gain_check"},
        },
        "ablation_report": [{
            "component": "property-check", "type": "indirect_gain", "actual": 1,
            "expected": {"operator": "==", "value": 0}, "verdict": "unmet",
            "evidence": {"source": "runtime_attempt", "ref": "attempt-check: check.json"},
        }],
    })
    store.put(kind="gen_experiment_result", session_id=sid, key="v1", obj=result, emitter="RunnerAgent")
    memory, _, diagnosis, _ = handler._consolidate_gen(
        run_ref, verification_report={"verdict": "verified", "checks": ["failure confirmed"]})
    assert diagnosis == expected
    assert memory["best_metric"] is None
    assert memory["verified_findings"] == []


@pytest.mark.parametrize("case", [
    "bound", "missing_attempt", "other_attempt", "missing_stdout", "no_failed_gate",
    "missing_metric", "wrong_metric", "binding_error", "comparison_drift", "integration",
    "evaluator_flag", "passing_gate", "nonfinite_metric", "boolean_metric",
])
def test_measured_official_gate_failure_requests_component_review_not_acceptance(store, case):
    handler = MemoryAgentHandler(spec_store=store, llm_client=FakeLLM([]))
    sid = str(uuid.uuid4())
    run_ref = _seed_gen_result(store, sid, verdict="invalid", pm=0.7, accepted=False)
    result_ref = refs.encode("gen_experiment_result", sid, "v1")
    result = store.get(result_ref)
    result.update({
        "component_attempt_id": "attempt-quality-check",
        "component_execution": {
            "mode": "component", "component_id": "regularizer",
            "status": "execution_failed", "execution_succeeded": False,
        },
        "candidate_admissibility": {
            "verdict": "fail", "attempt_id": "attempt-quality-check",
        },
        "measurement_evidence": {
            "attempt_id": "attempt-quality-check", "stdout_sha256": "a" * 64,
            "value": 0.7, "valid": False, "gates": {"/quality_passed": False},
        },
    })
    if case == "missing_attempt":
        result["component_attempt_id"] = ""
    elif case == "other_attempt":
        result["measurement_evidence"]["attempt_id"] = "attempt-old"
    elif case == "missing_stdout":
        result["measurement_evidence"].pop("stdout_sha256")
    elif case == "no_failed_gate":
        result["measurement_evidence"]["gates"] = {}
    elif case == "missing_metric":
        result["method_metric"] = None
    elif case == "wrong_metric":
        result["measurement_evidence"]["value"] = 0.8
    elif case == "binding_error":
        result["error"] = "candidate worktree changed after measurement"
    elif case == "comparison_drift":
        result["repository_comparison_review"] = {"verdict": "comparison_drift"}
    elif case == "integration":
        result["component_execution"]["mode"] = "integration"
    elif case == "passing_gate":
        result["measurement_evidence"]["gates"] = {"/quality_passed": True}
    elif case in {"nonfinite_metric", "boolean_metric"}:
        value = float("inf") if case == "nonfinite_metric" else True
        result["method_metric"] = result["measurement_evidence"]["value"] = value
    store.put(kind="gen_experiment_result", session_id=sid, key="v1",
              obj=result, emitter="RunnerAgent")
    eval_ref = store.put(kind="evaluation_report", session_id=sid, key="v1",
                         obj={"verdict": "invalid", "anti_cheat_flags": (
                             ["measurement source mismatch"] if case == "evaluator_flag" else [])},
                         emitter="EvaluatorAgent")
    memory, _, diagnosis, evidence = handler._consolidate_gen(
        run_ref, verification_report={"verdict": "verified", "checks": ["failure confirmed"]},
        evaluation_report_ref=eval_ref)
    if case == "bound":
        assert diagnosis == "component_expectation_failed"
        assert evidence[0] == "regularizer"
    else:
        assert diagnosis != "component_expectation_failed"
    assert memory["best_metric"] is None
    assert memory["verified_findings"] == []


@pytest.mark.parametrize("case", [
    "improve", "debug", "draft", "no_qualified", "no_checkpoint",
    "failed_retained_contract", "invalid_retained_measurement", "string_count",
])
def test_failed_quality_refinement_reconsiders_retained_valid_component(case):
    from autoresearch.agents.memory_agent.handler import _classify_component_evidence

    execution = {
        "mode": "component", "component_id": "feature", "role": "direct_gain",
        "operator": "improve", "status": "execution_failed", "qualified_attempts": 1,
        "checkpoint": "qualified-checkpoint",
        "selected_contract_result": {
            "passed": True, "measurement_evidence": {"valid": True}},
    }
    gen = {
        "accepted": False, "verdict": "invalid", "method_metric": 0.7,
        "component_attempt_id": "attempt-refinement", "component_execution": execution,
        "candidate_admissibility": {"verdict": "fail", "attempt_id": "attempt-refinement"},
        "measurement_evidence": {
            "value": 0.7, "valid": False, "attempt_id": "attempt-refinement",
            "stdout_sha256": "a" * 64, "gates": {"/quality_passed": False}},
    }
    if case in {"debug", "draft"}:
        execution["operator"] = case
    elif case == "no_qualified":
        execution["qualified_attempts"] = 0
    elif case == "no_checkpoint":
        execution["checkpoint"] = ""
    elif case == "failed_retained_contract":
        execution["selected_contract_result"]["passed"] = False
    elif case == "invalid_retained_measurement":
        execution["selected_contract_result"]["measurement_evidence"]["valid"] = False
    elif case == "string_count":
        execution["qualified_attempts"] = "1"
    expected = "component_progress" if case in {"improve", "debug"} else "component_expectation_failed"
    assert _classify_component_evidence(gen)[0] == expected
    assert gen["accepted"] is False and gen["measurement_evidence"]["valid"] is False


async def test_failed_component_with_preflight_evidence_routes_to_component_revision(
        store, monkeypatch):
    """A disproved component precondition is a plan problem, not another DEBUG retry."""
    monkeypatch.setenv("AUTORESEARCH_GEN_EXPERIMENT", "1")
    handler = MemoryAgentHandler(spec_store=store, llm_client=FakeLLM([]))
    client = FakeBusClient()
    sid = str(uuid.uuid4())
    run_ref = _seed_gen_result(
        store, sid, verdict="invalid", pb=1.0, pm=None,
        accepted=False, lower_is_better=True)
    result_ref = refs.encode("gen_experiment_result", sid, "v1")
    result = store.get(result_ref)
    result.update({
        "operator": "debug",
        "component_execution": {
            "mode": "component", "component_id": "requires-missing-surface",
            "role": "direct_gain", "status": "execution_failed",
            "contract_result": {
                "passed": False, "reason": "direct-gain component lacks finite metrics"},
        },
        "ablation_report": [{
            "component": "requires-missing-surface",
            "type": "direct_gain",
            "actual": 0,
            "expected": {"operator": ">", "value": 0},
            "evidence": {
                "source": "tracked_repository_source",
                "ref": "src/main.py:1-20",
            },
            "verdict": "unmet",
        }],
    })
    store.put(kind="gen_experiment_result", session_id=sid, key="v1",
              obj=result, emitter="RunnerAgent")

    await handler.handle_consolidation_requested(
        client, _Envelope(str(uuid.uuid4()), sid, "memory.consolidation_requested",
                          "ResearchOrchestrator", {"run_record_ref": run_ref}))

    event = [item for item in client.published
             if item.event_type == "memory.consolidated"][-1]
    assert event.payload["diagnosis"] == "component_expectation_failed"
    assert event.payload["diagnosis_evidence"] == ["requires-missing-surface"]


async def test_contract_failed_check_expectation_routes_to_component_revision(
        store, monkeypatch):
    """A measured failed component contract must reach CHECK-EXPECTATION routing."""
    monkeypatch.setenv("AUTORESEARCH_GEN_EXPERIMENT", "1")
    handler = MemoryAgentHandler(spec_store=store, llm_client=FakeLLM([]))
    client = FakeBusClient()
    sid = str(uuid.uuid4())
    run_ref = _seed_gen_result(
        store, sid, verdict="invalid", pb=17988.0, pm=913.0,
        accepted=False, lower_is_better=True)
    result_ref = refs.encode("gen_experiment_result", sid, "v1")
    result = store.get(result_ref)
    result.update({
        "operator": "check-expectation",
        "component_execution": {
            "mode": "component", "component_id": "train-loop",
            "role": "indirect_gain", "status": "contract_failed",
            "contract_result": {
                "passed": False, "reason": "indirect-gain threshold failed"},
        },
        "ablation_report": [{
            "component": "train-loop",
            "type": "indirect_gain",
            "actual": 1,
            "expected": {"operator": "==", "value": 0},
            "evidence": {
                "source": "runtime_attempt",
                "ref": "attempt-check-expectation",
            },
            "verdict": "unmet",
        }],
    })
    store.put(kind="gen_experiment_result", session_id=sid, key="v1",
              obj=result, emitter="RunnerAgent")

    await handler.handle_consolidation_requested(
        client, _Envelope(str(uuid.uuid4()), sid, "memory.consolidation_requested",
                          "ResearchOrchestrator", {"run_record_ref": run_ref}))

    event = [item for item in client.published
             if item.event_type == "memory.consolidated"][-1]
    assert event.payload["diagnosis"] == "component_expectation_failed"
    assert event.payload["diagnosis_evidence"] == ["train-loop"]


@pytest.mark.parametrize("case", ["grounded", "noncompliant", "missing_measurement", "finite_metric"])
def test_drifted_failed_probe_can_reach_component_revision_without_accepting_result(store, case):
    handler = MemoryAgentHandler(spec_store=store, llm_client=FakeLLM([]))
    sid = str(uuid.uuid4())
    run_ref = _seed_gen_result(store, sid, verdict="invalid", pm=None, accepted=False)
    result = store.get(refs.encode("gen_experiment_result", sid, "v1"))
    result.update({
        "operator": "debug",
        "component_execution": {
            "mode": "component", "component_id": "parser",
            "status": "drifted", "execution_succeeded": True,
            "contract_result": {"passed": False},
        },
        "method_implementation_review": {
            "verdict": "drifted",
            "task_contract_review": {"verdict": "noncompliant" if case == "noncompliant" else "compliant"},
        },
        "off_method_observation": {"valid": False},
        "ablation_report": [{
            "component": "parser", "type": "indirect_gain", "actual": 1,
            "expected": {"operator": "==", "value": 0},
            "evidence": {"source": "runtime_attempt", "ref": "attempt-parser-probe"},
            "verdict": "unmet",
        }],
    })
    if case == "missing_measurement":
        result.pop("ablation_report")
    if case == "finite_metric":
        result["method_metric"] = 0.5
    store.put(kind="gen_experiment_result", session_id=sid, key="v1", obj=result, emitter="RunnerAgent")
    memory, _, diagnosis, _ = handler._consolidate_gen(
        run_ref, verification_report={"verdict": "verified"})
    assert diagnosis == ("component_expectation_failed" if case == "grounded" else "component_contract_retry")
    assert memory["best_metric"] is None
    assert memory["verified_findings"] == []


@pytest.mark.parametrize("case", ["grounded", "violation", "missing_attempt", "other_attempt",
                                 "unverified_source", "failed_execution", "direct_component",
                                 "annotated_ref", "colon_ref", "ambiguous_ref", "id_suffix"])
def test_legal_drifted_indirect_probe_revises_without_a_primary_metric(store, case):
    handler = MemoryAgentHandler(spec_store=store, llm_client=FakeLLM([]))
    sid = str(uuid.uuid4())
    run_ref = _seed_gen_result(store, sid, verdict="invalid", pm=None, accepted=False)
    result = store.get(refs.encode("gen_experiment_result", sid, "v1"))
    result.update({
        "component_attempt_id": "" if case == "missing_attempt" else "attempt-probe",
        "component_execution": {
            "mode": "component", "component_id": "transform", "status": "drifted",
            "role": "direct_gain" if case == "direct_component" else "indirect_gain",
            "execution_succeeded": case != "failed_execution",
            "contract_result": {
                "passed": True, "measurement": "indirect_gain_check", "actual": 0.0,
                "expected": {"operator": "==", "value": 0.0},
                "evidence": {
                    "source": "self_report" if case == "unverified_source" else "runtime_attempt",
                    "ref": {
                        "other_attempt": "attempt-other",
                        "annotated_ref": "attempt-probe (probe.json; zero mismatches)",
                        "colon_ref": "attempt-probe: probe.json",
                        "ambiguous_ref": "attempt-probe: compared with attempt-other",
                        "id_suffix": "attempt-probe-extra",
                    }.get(case, "attempt-probe"),
                },
            },
        },
        "method_implementation_review": {
            "verdict": "drifted",
            "task_contract_review": {"verdict": "violation" if case == "violation" else "compliant"},
            "issues": ["legal implementation differs from frozen component"],
        },
        "off_method_observation": {"valid": False, "metric": None},
    })
    store.put(kind="gen_experiment_result", session_id=sid, key="v1", obj=result, emitter="RunnerAgent")
    memory, _, diagnosis, _ = handler._consolidate_gen(
        run_ref, verification_report={"verdict": "verified"})
    assert diagnosis == ("method_description_reconciliation" if case in {
        "grounded", "annotated_ref", "colon_ref"}
                         else "component_contract_retry")
    assert memory["best_metric"] is None
    assert memory["verified_findings"] == []


async def test_failed_component_integration_routes_to_composition_retry(store, monkeypatch):
    monkeypatch.setenv("AUTORESEARCH_GEN_EXPERIMENT", "1")
    handler = MemoryAgentHandler(spec_store=store, llm_client=FakeLLM([]))
    client = FakeBusClient()
    sid = str(uuid.uuid4())
    run_ref = _seed_gen_result(
        store, sid, verdict="valid_not_best", pb=1.0, pm=1.02,
        accepted=False, lower_is_better=True)
    result_ref = refs.encode("gen_experiment_result", sid, "v1")
    result = store.get(result_ref)
    result["component_execution"] = {
        "mode": "integration", "all_components_accepted": True,
    }
    store.put(kind="gen_experiment_result", session_id=sid, key="v1",
              obj=result, emitter="RunnerAgent")

    await handler.handle_consolidation_requested(
        client, _Envelope(str(uuid.uuid4()), sid, "memory.consolidation_requested",
                          "ResearchOrchestrator", {"run_record_ref": run_ref}))

    event = [item for item in client.published
             if item.event_type == "memory.consolidated"][-1]
    assert event.payload["diagnosis"] == "component_combination_retry"
    assert event.payload["routing_verdict"] == "need_experiment"


async def test_integration_with_wrong_parent_control_cannot_be_formal_best(store, monkeypatch):
    monkeypatch.setenv("AUTORESEARCH_GEN_EXPERIMENT", "1")
    handler = MemoryAgentHandler(spec_store=store, llm_client=FakeLLM([]))
    client = FakeBusClient()
    sid = str(uuid.uuid4())
    run_ref = _seed_gen_result(
        store, sid, verdict="new_best", pb=1.0, pm=0.8,
        accepted=True, lower_is_better=True)
    result_ref = refs.encode("gen_experiment_result", sid, "v1")
    result = store.get(result_ref)
    result["component_evidence_error"] = "control attempt is not the exact dependency parent"
    result["component_execution"] = {
        "mode": "integration", "all_components_accepted": True,
        "attempt_contract_result": {"passed": True},
    }
    result["method_implementation_review"] = {
        "verdict": "faithful", "task_contract_review": {"verdict": "compliant"},
    }
    store.put(kind="gen_experiment_result", session_id=sid, key="v1",
              obj=result, emitter="RunnerAgent")

    await handler.handle_consolidation_requested(
        client, _Envelope(str(uuid.uuid4()), sid, "memory.consolidation_requested",
                          "ResearchOrchestrator", {"run_record_ref": run_ref}))

    event = [item for item in client.published
             if item.event_type == "memory.consolidated"][-1]
    memory = store.get(event.payload["research_memory_ref"])
    assert event.payload["diagnosis"] == "component_combination_retry"
    assert memory["best_result_ref"] is None
    assert memory["best_metric"] is None


async def test_failed_component_integration_without_metric_routes_to_composition_retry(
        store, monkeypatch):
    monkeypatch.setenv("AUTORESEARCH_GEN_EXPERIMENT", "1")
    handler = MemoryAgentHandler(spec_store=store, llm_client=FakeLLM([]))
    client = FakeBusClient()
    sid = str(uuid.uuid4())
    run_ref = _seed_gen_result(
        store, sid, verdict="invalid", pb=1.0, pm=None,
        accepted=False, lower_is_better=True)
    result_ref = refs.encode("gen_experiment_result", sid, "v1")
    result = store.get(result_ref)
    result["component_execution"] = {
        "mode": "integration",
        "status": "execution_failed",
        "all_components_accepted": True,
    }
    store.put(kind="gen_experiment_result", session_id=sid, key="v1",
              obj=result, emitter="RunnerAgent")

    await handler.handle_consolidation_requested(
        client, _Envelope(str(uuid.uuid4()), sid, "memory.consolidation_requested",
                          "ResearchOrchestrator", {"run_record_ref": run_ref}))

    event = [item for item in client.published
             if item.event_type == "memory.consolidated"][-1]
    assert event.payload["diagnosis"] == "component_combination_retry"
    assert event.payload["routing_verdict"] == "need_experiment"


@pytest.mark.parametrize("error", [
    "clone failed: [Errno 122] Disk quota exceeded",
    "clone failed: source checkout could not be copied",
])
@pytest.mark.parametrize("status", ["invalid", "execution_failed"])
async def test_unstarted_integration_clone_failure_is_not_a_composition_failure(
        store, monkeypatch, error, status):
    monkeypatch.setenv("AUTORESEARCH_GEN_EXPERIMENT", "1")
    handler = MemoryAgentHandler(spec_store=store, llm_client=FakeLLM([]))
    client = FakeBusClient()
    sid = str(uuid.uuid4())
    run_ref = _seed_gen_result(
        store, sid, verdict="invalid", pb=1.0, pm=None,
        accepted=False, lower_is_better=True)
    result = store.get(refs.encode("gen_experiment_result", sid, "v1"))
    result.update(error=error, component_execution_started=False, component_execution={
        "mode": "integration", "status": status, "attempts": 0,
        "execution_succeeded": False, "all_components_accepted": True,
    })
    store.put(kind="gen_experiment_result", session_id=sid, key="v1",
              obj=result, emitter="RunnerAgent")

    await handler.handle_consolidation_requested(
        client, _Envelope(str(uuid.uuid4()), sid, "memory.consolidation_requested",
                          "ResearchOrchestrator", {"run_record_ref": run_ref}))

    event = [item for item in client.published
             if item.event_type == "memory.consolidated"][-1]
    memory = store.get(event.payload["research_memory_ref"])
    assert event.payload["diagnosis"] == "execution_failure"
    assert error in event.payload["diagnosis_evidence"]
    assert memory["best_metric"] is None
    assert not memory["verified_findings"]


@pytest.mark.parametrize("scope,key,value", [
    ("gen", "component_execution_started", True),
    ("gen", "component_attempt_id", "attempt-measured"),
    ("gen", "component_failed_attempt_id", "attempt-failed"),
    ("gen", "method_metric", 0.9),
    ("gen", "error", "evaluation failed after cloning"),
    ("execution", "execution_succeeded", True),
])
def test_integration_clone_label_cannot_hide_actual_candidate_execution(scope, key, value):
    from autoresearch.agents.memory_agent.handler import _classify_component_evidence
    gen = {
        "error": "clone failed: [Errno 122] Disk quota exceeded",
        "verdict": "invalid", "accepted": False, "method_metric": None,
        "component_execution_started": False,
        "component_execution": {
            "mode": "integration", "execution_succeeded": False,
            "all_components_accepted": True,
        },
    }
    (gen if scope == "gen" else gen["component_execution"])[key] = value
    assert _classify_component_evidence(gen)[0] == "component_combination_retry"


async def test_drifted_measured_gain_routes_to_method_description_reconciliation(
        store, monkeypatch):
    monkeypatch.setenv("AUTORESEARCH_GEN_EXPERIMENT", "1")
    handler = MemoryAgentHandler(spec_store=store, llm_client=FakeLLM([]))
    client = FakeBusClient()
    sid = str(uuid.uuid4())
    run_record_ref = _seed_gen_result(
        store, sid, verdict="valid_not_best", pb=1.0, pm=0.9, accepted=False,
        lower_is_better=True)
    result_ref = refs.encode("gen_experiment_result", sid, "v1")
    result = store.get(result_ref)
    result.update({
        "method_implementation_review": {"verdict": "drifted"},
        "off_method_observation": {
            "valid": True, "improved": True, "metric": 0.9,
            "review_verdict": "drifted",
        },
        "host_reflection": {
            "route": "revise_method",
            "verdict": "The clipping mechanism explains the measured gain.",
            "method_note": "Reconcile the method description with clipping evidence.",
        },
    })
    store.put(
        kind="gen_experiment_result", session_id=sid, key="v1",
        obj=result, emitter="RunnerAgent")

    await handler.handle_consolidation_requested(
        client, _Envelope(str(uuid.uuid4()), sid, "memory.consolidation_requested",
                          "ResearchOrchestrator", {"run_record_ref": run_record_ref}))

    consolidated = [p for p in client.published if p.event_type == "memory.consolidated"]
    assert consolidated, client.published
    event = consolidated[-1]
    assert event.payload["diagnosis"] == "method_description_reconciliation"
    memory = store.get(event.payload["research_memory_ref"])
    assert memory["diagnosis"] == "method_description_reconciliation"
    assert any("Off-method observation retained" in item
               for item in memory["consolidated_insights"])


async def test_legal_drifted_component_routes_to_formal_component_revision(store, monkeypatch):
    monkeypatch.setenv("AUTORESEARCH_GEN_EXPERIMENT", "1")
    handler = MemoryAgentHandler(spec_store=store, llm_client=FakeLLM([]))
    client = FakeBusClient()
    sid = str(uuid.uuid4())
    run_record_ref = _seed_gen_result(
        store, sid, verdict="valid_not_best", pb=1.0, pm=0.9,
        accepted=False, lower_is_better=True)
    result_ref = refs.encode("gen_experiment_result", sid, "v1")
    result = store.get(result_ref)
    result.update({
        "component_execution": {
            "mode": "component", "component_id": "feature-split", "status": "drifted"},
        "method_implementation_review": {"verdict": "drifted"},
        "off_method_observation": {
            "valid": True, "improved": True, "metric": 0.9,
            "review_verdict": "drifted"},
    })
    store.put(kind="gen_experiment_result", session_id=sid, key="v1",
              obj=result, emitter="RunnerAgent")

    await handler.handle_consolidation_requested(
        client, _Envelope(str(uuid.uuid4()), sid, "memory.consolidation_requested",
                          "ResearchOrchestrator", {"run_record_ref": run_record_ref}))

    event = [item for item in client.published
             if item.event_type == "memory.consolidated"][-1]
    assert event.payload["diagnosis"] == "method_description_reconciliation"
    assert event.payload["routing_verdict"] == "need_experiment"


@pytest.mark.parametrize("role", ["indirect_gain", "direct_gain"])
@pytest.mark.parametrize("status", ["drifted", "exhausted"])
async def test_premature_component_implementation_does_not_force_plan_revision(
        store, monkeypatch, role, status):
    monkeypatch.setenv("AUTORESEARCH_GEN_EXPERIMENT", "1")
    handler = MemoryAgentHandler(spec_store=store, llm_client=FakeLLM([]))
    client = FakeBusClient()
    sid = str(uuid.uuid4())
    run_ref = _seed_gen_result(
        store, sid, verdict="invalid", pb=10.0,
        pm=None if role == "indirect_gain" else 7.0,
        accepted=False, lower_is_better=True)
    result_ref = refs.encode("gen_experiment_result", sid, "v1")
    result = store.get(result_ref)
    result.update({
        "component_attempt_id": "attempt-current",
        "component_execution": {
            "mode": "component", "component_id": "current", "role": role,
            "status": status, "execution_succeeded": True,
            "contract_result": {
                "passed": True, "measurement": "indirect_gain_check",
                "evidence": {"source": "runtime_attempt", "ref": "attempt-current"},
            },
        },
        "method_implementation_review": {
            "verdict": "drifted", "task_contract_review": {"verdict": "compliant"},
            "boundary_reviews": [{
                "component_id": "other", "status": "prematurely_implemented",
                "reason": "Unscheduled independent component was added to this diff",
            }],
            "issues": ["Remove the prematurely implemented other component"],
        },
        "off_method_observation": {
            "valid": role == "direct_gain", "improved": role == "direct_gain",
        },
    })
    store.put(kind="gen_experiment_result", session_id=sid, key="v1",
              obj=result, emitter="RunnerAgent")
    await handler.handle_consolidation_requested(
        client, _Envelope(str(uuid.uuid4()), sid, "memory.consolidation_requested",
                          "ResearchOrchestrator", {"run_record_ref": run_ref}))
    event = [p for p in client.published if p.event_type == "memory.consolidated"][-1]
    memory = store.get(event.payload["research_memory_ref"])
    assert memory["diagnosis"] == (
        "component_contract_retry" if status == "drifted" else "component_execution_blocked")
    assert memory["best_metric"] is None


@pytest.mark.parametrize("admissible", [True, False])
async def test_exhausted_component_preserves_legal_off_method_revision(
        store, monkeypatch, admissible):
    """A final driver attempt may improve a dependency without qualifying the frozen node."""
    monkeypatch.setenv("AUTORESEARCH_GEN_EXPERIMENT", "1")
    handler = MemoryAgentHandler(spec_store=store, llm_client=FakeLLM([]))
    client = FakeBusClient()
    sid = str(uuid.uuid4())
    run_ref = _seed_gen_result(
        store, sid, verdict="valid_not_best", pb=10.0, pm=7.0,
        accepted=False, lower_is_better=True)
    result_ref = refs.encode("gen_experiment_result", sid, "v1")
    result = store.get(result_ref)
    result.update({
        "repo_path": "/candidate/final-driver",
        "method_implementation_review": {
            "verdict": "drifted",
            "issues": ["final-driver changed accepted compute-core"],
        },
        "off_method_observation": {"valid": admissible, "improved": admissible},
        "component_execution": {
            "mode": "component", "component_id": "final-driver",
            "status": "exhausted", "execution_succeeded": True,
            "attempts": 6, "qualified_attempts": 0, "checkpoint": "",
            "dependency_closure": ["compute-core"],
            "attempt_contract_result": {
                "passed": True, "measurement": "signed_metric_improvement"},
        },
    })
    store.put(kind="gen_experiment_result", session_id=sid, key="v1",
              obj=result, emitter="RunnerAgent")
    await handler.handle_consolidation_requested(
        client, _Envelope(str(uuid.uuid4()), sid, "memory.consolidation_requested",
                          "ResearchOrchestrator", {"run_record_ref": run_ref}))
    event = [p for p in client.published if p.event_type == "memory.consolidated"][-1]
    memory = store.get(event.payload["research_memory_ref"])
    assert memory["diagnosis"] == (
        "method_description_reconciliation" if admissible else "component_execution_blocked")
    assert memory["best_metric"] is None
    if admissible:
        assert "compute-core" in " ".join(memory["diagnosis_evidence"])


async def test_theory_method_consolidates_without_baseline_or_run(store):
    llm = FakeLLM([])
    handler = MemoryAgentHandler(spec_store=store, llm_client=llm)
    client = FakeBusClient()
    sid = str(uuid.uuid4())
    method_ref = store.put(
        kind="method", session_id=sid, key="v1",
        obj={
            "approach": "Use a finite verification budget model.",
            "changes_outline": ["Theorem: correlated papers have diminishing information gain."],
            "assumptions": ["Evidence correlation is observable."],
            "components": [{
                "name": "counterexample check",
                "functional_check": "Construct the independent-evidence boundary case.",
            }],
        },
        emitter="MethodAgent",
    )
    problem_ref = store.put(
        kind="problem_model", session_id=sid, key="v1",
        obj={"risks": ["The proof sketch may hide regularity assumptions."]},
        emitter="ProblemModelingAgent",
    )
    interpretation_ref = store.put(
        kind="spec_interpretation", session_id=sid, key="v1",
        obj={"disposition": "out_of_scope", "scope_reason": "No automatic theorem verifier."},
        emitter="ProblemModelingAgent",
    )
    env = _Envelope(
        str(uuid.uuid4()), sid, "memory.consolidation_requested", "ResearchOrchestrator",
        {
            "theory_method_ref": method_ref,
            "problem_model_ref": problem_ref,
            "spec_interpretation_ref": interpretation_ref,
            "run_record_ref": None,
        },
    )

    await handler.handle_consolidation_requested(client, env)

    event = next(p for p in client.published if p.event_type == "memory.consolidated")
    assert event.payload["routing_verdict"] == "report"
    memory = store.get(event.payload["research_memory_ref"])
    assert memory["best_metric"] is None
    assert memory["verified_findings"] == []
    assert any("diminishing information gain" in item
               for item in memory["consolidated_insights"])
    assert any("Unverified assumption" in item for item in memory["negative_results"])
    assert any("counterexample check" in item for item in memory["next_steps"])
    assert llm.calls == []


async def test_stale_structured_result_without_run_is_honest_failure_not_type_error(store):
    llm = FakeLLM([])
    handler = MemoryAgentHandler(spec_store=store, llm_client=llm)
    client = FakeBusClient()
    sid = str(uuid.uuid4())
    store.put(
        kind="gen_experiment_result", session_id=sid, key="current",
        obj={"verdict": "invalid", "method_metric": None},
        emitter="RunnerAgent",
    )
    env = _Envelope(
        str(uuid.uuid4()), sid, "memory.consolidation_requested", "ResearchOrchestrator",
        {"run_record_ref": None, "evaluation_report_ref": None},
    )

    await handler.handle_consolidation_requested(client, env)

    event = next(p for p in client.published if p.event_type == "memory.consolidated")
    memory = store.get(event.payload["research_memory_ref"])
    assert memory["diagnosis"] == "execution_failure"


async def test_gen_branch_evaluator_suspicious_vetoes_p1_pass(store, monkeypatch):
    # P1 says "pass", but the trusted 34-state evaluator flagged the result "suspicious"
    # (e.g. an impossible metric). The finding must NOT be verified -- it lands in
    # negative_results WITH the flags, so a MOSS-discarded result is not reported as a win.
    monkeypatch.setenv("AUTORESEARCH_GEN_EXPERIMENT", "1")
    handler = MemoryAgentHandler(spec_store=store, llm_client=FakeLLM([]))
    client = FakeBusClient()
    sid = str(uuid.uuid4())
    run_record_ref = _seed_gen_result(store, sid, verdict="pass", pb=1.0, pm=81.14)
    eval_ref = store.put(
        kind="evaluation_report", session_id=sid, key="v1",
        obj={"verdict": "suspicious", "metric_value": 81.14, "baseline_metric": 1.0,
             "is_improvement": True, "suspicious": True,
             "anti_cheat_flags": ["metric improbably high (81.14 > max_allowed 1.0)"],
             "reasoning": "impossible metric outside the verifier's valid range"},
        emitter="EvaluatorAgent")
    env = _Envelope(str(uuid.uuid4()), sid, "memory.consolidation_requested",
                    "ResearchOrchestrator",
                    {"run_record_ref": run_record_ref, "evaluation_report_ref": eval_ref})

    await handler.handle_consolidation_requested(client, env)

    consolidated = [p for p in client.published if p.event_type == "memory.consolidated"]
    memory = store.get(consolidated[0].payload["research_memory_ref"])
    assert memory["best_metric"] is None
    assert memory["verified_findings"] == []              # NOT a verified win
    assert len(memory["negative_results"]) == 1
    neg = memory["negative_results"][0]
    assert "suspicious" in neg and "improbably high" in neg
    # a SUSPICIOUS (anti-cheat) result STOPS the loop -- don't build the next idea on it.
    assert consolidated[0].payload["routing_verdict"] == "report"


async def test_gen_branch_comparison_only_suspicion_reenters_bounded_failure_loop(
        store, monkeypatch):
    """A measured method that misses the frozen comparison target needs revision, not shutdown."""
    monkeypatch.setenv("AUTORESEARCH_GEN_EXPERIMENT", "1")
    handler = MemoryAgentHandler(spec_store=store, llm_client=FakeLLM([]))
    client = FakeBusClient()
    sid = str(uuid.uuid4())
    run_record_ref = _seed_gen_result(
        store, sid, verdict="valid_not_best", pb=196.834, pm=198.319,
        accepted=False)
    eval_ref = store.put(
        kind="evaluation_report", session_id=sid, key="v1",
        obj={
            "verdict": "suspicious", "metric_value": 198.319,
            "baseline_metric": 196.834, "is_improvement": False,
            "suspicious": True,
            "anti_cheat_flags": [
                "comparison_semantics_changed:method did not satisfy the frozen target",
                "comparison_semantics_changed:time is not comparable until the target is met",
            ],
            "reasoning": "revise the method or its allowed settings and retry",
        }, emitter="EvaluatorAgent")

    await handler.handle_consolidation_requested(
        client, _Envelope(str(uuid.uuid4()), sid, "memory.consolidation_requested",
                          "ResearchOrchestrator", {
                              "run_record_ref": run_record_ref,
                              "evaluation_report_ref": eval_ref,
                          }))

    event = [p for p in client.published if p.event_type == "memory.consolidated"][-1]
    assert event.payload["diagnosis"] == "inconclusive"
    assert event.payload["routing_verdict"] == "need_experiment"


@pytest.mark.parametrize("method_metric", [None, float("nan"), float("inf")])
async def test_comparison_only_suspicion_without_finite_measurement_still_reports(
        store, monkeypatch, method_metric):
    monkeypatch.setenv("AUTORESEARCH_GEN_EXPERIMENT", "1")
    handler = MemoryAgentHandler(spec_store=store, llm_client=FakeLLM([]))
    client = FakeBusClient()
    sid = str(uuid.uuid4())
    run_record_ref = _seed_gen_result(
        store, sid, verdict="fail", pb=1.0, pm=method_metric, accepted=False)
    eval_ref = store.put(
        kind="evaluation_report", session_id=sid, key="v1",
        obj={
            "verdict": "suspicious", "metric_value": method_metric,
            "baseline_metric": 1.0, "is_improvement": False, "suspicious": True,
            "anti_cheat_flags": [
                "comparison_semantics_changed:no trustworthy method measurement"],
            "reasoning": "missing or non-finite evidence",
        }, emitter="EvaluatorAgent")

    await handler.handle_consolidation_requested(
        client, _Envelope(str(uuid.uuid4()), sid, "memory.consolidation_requested",
                          "ResearchOrchestrator", {
                              "run_record_ref": run_record_ref,
                              "evaluation_report_ref": eval_ref,
                          }))

    event = [p for p in client.published if p.event_type == "memory.consolidated"][-1]
    assert event.payload["routing_verdict"] == "report"


async def test_gen_branch_invalid_result_still_feeds_the_loop(store, monkeypatch):
    # an INVALID result (the method failed / was illegal) is an HONEST failure: the loop
    # should try a DIFFERENT next idea (continue_explore), NOT stop. Only "suspicious" stops.
    monkeypatch.setenv("AUTORESEARCH_GEN_EXPERIMENT", "1")
    handler = MemoryAgentHandler(spec_store=store, llm_client=FakeLLM([]))
    client = FakeBusClient()
    sid = str(uuid.uuid4())
    run_record_ref = _seed_gen_result(store, sid, verdict="fail", pb=1.0, pm=None, accepted=False)
    eval_ref = store.put(
        kind="evaluation_report", session_id=sid, key="v1",
        obj={"verdict": "invalid", "is_improvement": False, "suspicious": False,
             "anti_cheat_flags": []},
        emitter="EvaluatorAgent")
    env = _Envelope(str(uuid.uuid4()), sid, "memory.consolidation_requested",
                    "ResearchOrchestrator",
                    {"run_record_ref": run_record_ref, "evaluation_report_ref": eval_ref})

    await handler.handle_consolidation_requested(client, env)

    consolidated = [p for p in client.published if p.event_type == "memory.consolidated"]
    assert consolidated[0].payload["routing_verdict"] == "continue_explore"


@pytest.mark.parametrize(("diagnosis", "expected_routing"), [
    ("execution_failure", "continue_explore"),
    ("candidate_contract_violation", "continue_explore"),
    ("attempt_accounting_violation", "continue_explore"),
    ("baseline_drift", "continue_explore"),
])
async def test_gen_branch_publishes_trusted_diagnosis(
        store, monkeypatch, diagnosis, expected_routing):
    """Only trusted phase-finalizer diagnoses survive without structured measurements."""
    monkeypatch.setenv("AUTORESEARCH_GEN_EXPERIMENT", "1")
    handler = MemoryAgentHandler(spec_store=store, llm_client=FakeLLM([]))
    client = FakeBusClient()
    sid = str(uuid.uuid4())
    run_record_ref = _seed_gen_result(
        store, sid, verdict="invalid", pb=1.0, pm=None, accepted=False,
        diagnosis=diagnosis, diagnosis_evidence=["trusted phase finalizer evidence"],
    )

    await handler.handle_consolidation_requested(
        client, _Envelope(str(uuid.uuid4()), sid, "memory.consolidation_requested",
                          "ResearchOrchestrator", {"run_record_ref": run_record_ref}))

    event = [p for p in client.published if p.event_type == "memory.consolidated"][-1]
    memory = store.get(event.payload["research_memory_ref"])
    assert event.payload["diagnosis"] == diagnosis
    assert event.payload["diagnosis_evidence"] == ["trusted phase finalizer evidence"]
    assert event.payload["routing_verdict"] == expected_routing
    assert memory["diagnosis"] == diagnosis


async def test_gen_branch_generic_nonwin_is_inconclusive_not_setting_issue(store, monkeypatch):
    """Catch an ordinary measured loss consuming the experiment-setting rework budget."""
    monkeypatch.setenv("AUTORESEARCH_GEN_EXPERIMENT", "1")
    handler = MemoryAgentHandler(spec_store=store, llm_client=FakeLLM([]))
    client = FakeBusClient()
    sid = str(uuid.uuid4())
    run_record_ref = _seed_gen_result(
        store, sid, verdict="valid_not_best", pb=1.0, pm=1.1, accepted=False)

    await handler.handle_consolidation_requested(
        client, _Envelope(str(uuid.uuid4()), sid, "memory.consolidation_requested",
                          "ResearchOrchestrator", {"run_record_ref": run_record_ref}))

    event = [p for p in client.published if p.event_type == "memory.consolidated"][-1]
    assert event.payload["diagnosis"] == "inconclusive"
    assert event.payload["routing_verdict"] == "continue_explore"


@pytest.mark.parametrize("provider_code", [
    "llm_provider_timeout", "llm_provider_content_filter",
    "clone failed: git clone produced an empty/incomplete repo; Disk quota exceeded",
    "clone failed: fatal: unable to access repository; connection timed out",
])
@pytest.mark.parametrize("operator", ["draft", "improve"])
@pytest.mark.parametrize("exhausted", [False, True])
def test_unstarted_provider_failure_survives_component_consolidation(
        store, provider_code, operator, exhausted):
    handler = MemoryAgentHandler(spec_store=store, llm_client=FakeLLM([]))
    sid = str(uuid.uuid4())
    run_ref = _seed_gen_result(store, sid, verdict="invalid", pb=1.0, pm=None,
                              accepted=False, diagnosis="execution_failure")
    result = store.get(refs.encode("gen_experiment_result", sid, "v1"))
    result.update(error=(provider_code if provider_code.startswith("clone failed:")
                         else "Conversation run failed; " + provider_code),
                  component_execution_started=False,
                  component_execution={
                      "mode": "component", "component_id": "candidate", "operator": operator,
                      "status": "infrastructure_exhausted" if exhausted else "execution_failed",
                      "execution_succeeded": False, "attempts": 0,
                      "infrastructure_attempts": 8 if exhausted else 1,
                      "qualified_attempts": 1 if operator == "improve" else 0,
                      "checkpoint": "checkpoint-qualified" if operator == "improve" else "",
                      "contract_result": {"passed": False, "reason": "no finite candidate metric"},
                  })
    store.put(kind="gen_experiment_result", session_id=sid, key="current",
              obj=result, emitter="RunnerAgent")
    memory, route, diagnosis, evidence = handler._consolidate_gen(run_ref)
    assert diagnosis == ("component_execution_blocked" if exhausted else "execution_failure")
    assert route == ("report" if exhausted else "continue_explore")
    assert memory["best_metric"] is None
    assert not memory["verified_findings"]
    if not exhausted:
        assert provider_code in " ".join(evidence)


@pytest.mark.parametrize("scope,key,value", [
    ("gen", "component_execution_started", True),
    ("gen", "component_attempt_id", "attempt-launched"),
    ("gen", "component_failed_attempt_id", "attempt-failed"),
    ("gen", "method_metric", 0.9),
    ("gen", "diagnosis", "attempt_accounting_violation"),
    ("gen", "error", "compiler rejected source"),
    ("execution", "execution_succeeded", True),
    ("execution", "infrastructure_attempts", 0),
    ("execution", "infrastructure_attempts", True),
    ("execution", "infrastructure_attempts", "1"),
    ("execution", "mode", "integration"),
    ("execution", "status", "exhausted"),
])
def test_provider_routing_does_not_hide_started_or_other_component_failures(scope, key, value):
    from autoresearch.agents.memory_agent.handler import _classify_component_evidence
    gen = {
        "diagnosis": "execution_failure", "error": "llm_provider_timeout",
        "method_metric": None, "component_execution_started": False,
        "component_execution": {
            "mode": "component", "status": "execution_failed",
            "execution_succeeded": False, "infrastructure_attempts": 1,
        },
    }
    target = gen if scope == "gen" else gen["component_execution"]
    target[key] = value
    assert _classify_component_evidence(gen)[0] != "execution_failure"


@pytest.mark.parametrize("reported_metric", [0.9, 1.1])
def test_failed_component_binding_with_claimed_finite_score_keeps_failure_reason(
        store, reported_metric):
    handler = MemoryAgentHandler(spec_store=store, llm_client=FakeLLM([]))
    sid = str(uuid.uuid4())
    run_ref = _seed_gen_result(store, sid, verdict="invalid", pb=1.0,
                               pm=reported_metric, accepted=False)
    result = store.get(refs.encode("gen_experiment_result", sid, "v1"))
    failure = "execution output is missing /method"
    result.update(error=None, reasons=[failure], ablation_report=[{"component": "c1"}],
                  component_execution={"mode": "component", "component_id": "c1",
                                       "status": "execution_failed", "execution_succeeded": False})
    store.put(kind="gen_experiment_result", session_id=sid, key="current",
              obj=result, emitter="RunnerAgent")
    eval_ref = store.put(kind="evaluation_report", session_id=sid, key="v1",
                         obj={"verdict": "invalid", "anti_cheat_flags": []},
                         emitter="EvaluatorAgent")
    memory, _, diagnosis, evidence = handler._consolidate_gen(run_ref, eval_ref)
    assert diagnosis == "execution_failure"
    assert failure in " ".join(evidence)
    assert "valid comparison did not improve" not in " ".join(evidence)
    assert memory["best_metric"] is None
    assert not memory["verified_findings"]


@pytest.mark.parametrize("eval_verdict", ["invalid", "suspicious"])
@pytest.mark.parametrize("exhausted", [False, True])
def test_invalid_finite_observation_does_not_erase_exhausted_component_budget(
        store, eval_verdict, exhausted):
    handler = MemoryAgentHandler(spec_store=store, llm_client=FakeLLM([]))
    sid = str(uuid.uuid4())
    run_ref = _seed_gen_result(
        store, sid, verdict="invalid", pb=1.0, pm=0.9, accepted=False,
        diagnosis="attempt_accounting_violation", diagnosis_evidence=["control binding missing"])
    result = store.get(refs.encode("gen_experiment_result", sid, "v1"))
    result["component_execution"] = {
        "mode": "component", "component_id": "candidate", "role": "direct_gain",
        "status": "infrastructure_exhausted" if exhausted else "execution_failed",
        "execution_succeeded": False,
    }
    store.put(kind="gen_experiment_result", session_id=sid, key="current",
              obj=result, emitter="RunnerAgent")
    eval_ref = store.put(kind="evaluation_report", session_id=sid, key="v1",
                         obj={"verdict": eval_verdict, "anti_cheat_flags": []},
                         emitter="EvaluatorAgent")
    memory, route, diagnosis, evidence = handler._consolidate_gen(run_ref, eval_ref)
    assert diagnosis == ("component_execution_blocked" if exhausted else "attempt_accounting_violation")
    assert memory["best_metric"] is None
    assert not memory["verified_findings"]
    if exhausted:
        assert route == "report"
        assert any("retry budget exhausted" in item for item in evidence)


async def test_host_reflection_routes_valid_nonwin_to_method_revision(store, monkeypatch):
    monkeypatch.setenv("AUTORESEARCH_GEN_EXPERIMENT", "1")
    handler = MemoryAgentHandler(spec_store=store, llm_client=FakeLLM([]))
    client = FakeBusClient()
    sid = str(uuid.uuid4())
    run_record_ref = _seed_gen_result(
        store, sid, verdict="valid_not_best", pb=0.5, pm=0.51, accepted=False)
    store.put(
        kind="gen_experiment_result", session_id=sid, key="current",
        obj={
            "verdict": "valid_not_best", "accepted": False,
            "baseline_metric": 0.5, "method_metric": 0.51, "reasons": [],
            "host_reflection": {
                "route": "revise_method",
                "verdict": "faithful trials falsified the mechanism",
                "method_note": "replace the unsupported component",
            },
        }, emitter="RunnerAgent")

    await handler.handle_consolidation_requested(
        client, _Envelope(str(uuid.uuid4()), sid, "memory.consolidation_requested",
                          "ResearchOrchestrator", {"run_record_ref": run_record_ref}))

    event = [p for p in client.published if p.event_type == "memory.consolidated"][-1]
    assert event.payload["diagnosis"] == "method_hypothesis_falsified"
    assert event.payload["diagnosis_evidence"] == ["replace the unsupported component"]


async def test_candidate_contract_violation_takes_priority_over_method_reflection(
        store, monkeypatch):
    monkeypatch.setenv("AUTORESEARCH_GEN_EXPERIMENT", "1")
    handler = MemoryAgentHandler(spec_store=store, llm_client=FakeLLM([]))
    client = FakeBusClient()
    sid = str(uuid.uuid4())
    run_record_ref = _seed_gen_result(
        store, sid, verdict="valid_not_best", pb=9220.0, pm=1544.0,
        accepted=False, diagnosis="candidate_contract_violation",
        diagnosis_evidence=["candidate diff includes a file outside the permitted edit scope"],
    )
    current = store.get(refs.encode("gen_experiment_result", sid, "v1"))
    current["component_execution"] = {
        "mode": "integration", "component_id": "__integration__",
        "all_components_accepted": True,
    }
    current["method_implementation_review"] = {
        "verdict": "drifted",
        "issues": ["Non-permitted file analysis/search.py is added in the diff"],
        "task_contract_review": {"verdict": "compliant"},
    }
    current["off_method_observation"] = {"valid": True, "improved": True, "metric": 1544.0}
    current["repository_comparison_review"] = {"verdict": "pass"}
    current["host_reflection"] = {
        "route": "revise_method", "verdict": "the core algorithm improved",
        "method_note": "reconcile the faster implementation",
    }
    store.put(kind="gen_experiment_result", session_id=sid, key="current",
              obj=current, emitter="RunnerAgent")

    await handler.handle_consolidation_requested(
        client, _Envelope(str(uuid.uuid4()), sid, "memory.consolidation_requested",
                          "ResearchOrchestrator", {"run_record_ref": run_record_ref}))

    event = [p for p in client.published if p.event_type == "memory.consolidated"][-1]
    assert event.payload["diagnosis"] == "candidate_contract_violation"
    assert event.payload["diagnosis_evidence"] == [
        "candidate diff includes a file outside the permitted edit scope"]
    assert store.get(event.payload["research_memory_ref"])["best_result_ref"] is None


async def test_gen_branch_recomputes_component_failure_from_structured_evidence(
        store, monkeypatch):
    """Component routing is decided at consolidation, not from the sandbox's `met` label."""
    monkeypatch.setenv("AUTORESEARCH_GEN_EXPERIMENT", "1")
    handler = MemoryAgentHandler(spec_store=store, llm_client=FakeLLM([]))
    client = FakeBusClient()
    sid = str(uuid.uuid4())
    run_record_ref = _seed_gen_result(
        store, sid, verdict="valid_not_best", pb=1.0, pm=1.1, accepted=False)
    store.put(
        kind="gen_experiment_result", session_id=sid, key="current",
        obj={
            "operator": "check-expectation", "verdict": "valid_not_best",
            "accepted": False, "baseline_metric": 1.0, "method_metric": 1.1,
            "reasons": [],
            "ablation_report": [{
                "component": "router", "type": "functional", "actual": 0.4,
                "expected": {"operator": "<=", "value": 0.1},
                "evidence": {"source": "run_record", "ref": run_record_ref},
                "verdict": "met",
            }],
        }, emitter="RunnerAgent")

    await handler.handle_consolidation_requested(
        client, _Envelope(str(uuid.uuid4()), sid, "memory.consolidation_requested",
                          "ResearchOrchestrator", {"run_record_ref": run_record_ref}))

    event = [p for p in client.published if p.event_type == "memory.consolidated"][-1]
    assert event.payload["diagnosis"] == "component_expectation_failed"
    assert event.payload["diagnosis_evidence"] == ["router"]
    assert event.payload["routing_verdict"] == "continue_explore"


async def test_check_expectation_survives_legacy_whole_method_review_veto(
        store, monkeypatch):
    """Old runners misapplied whole-method fidelity review to diagnostic A/B probes."""
    monkeypatch.setenv("AUTORESEARCH_GEN_EXPERIMENT", "1")
    handler = MemoryAgentHandler(spec_store=store, llm_client=FakeLLM([]))
    client = FakeBusClient()
    sid = str(uuid.uuid4())
    run_record_ref = _seed_gen_result(
        store, sid, verdict="invalid", pb=1.0, pm=1.1, accepted=False,
        diagnosis="candidate_contract_violation",
        diagnosis_evidence=["whole-method review=drifted"],
    )
    store.put(
        kind="gen_experiment_result", session_id=sid, key="current",
        obj={
            "operator": "check-expectation", "verdict": "invalid",
            "accepted": False, "baseline_metric": 1.0, "method_metric": 1.1,
            "reasons": ["whole-method review=drifted"],
            "diagnosis": "candidate_contract_violation",
            "diagnosis_evidence": ["whole-method review=drifted"],
            "method_implementation_review": {
                "verdict": "drifted",
                "issues": ["A/B probes do not each contain the full method"],
            },
            "ablation_report": [{
                "component": "GC", "type": "gain", "actual": -0.01,
                "expected": {"operator": ">=", "value": 0.0},
                "evidence": {"source": "run_record", "ref": run_record_ref},
                "verdict": "unmet",
            }],
        }, emitter="RunnerAgent")
    eval_ref = store.put(
        kind="evaluation_report", session_id=sid, key="v1",
        obj={
            "verdict": "invalid", "is_improvement": False,
            "suspicious": False, "anti_cheat_flags": [],
        }, emitter="EvaluatorAgent")

    await handler.handle_consolidation_requested(
        client, _Envelope(str(uuid.uuid4()), sid, "memory.consolidation_requested",
                          "ResearchOrchestrator", {
                              "run_record_ref": run_record_ref,
                              "evaluation_report_ref": eval_ref,
                          }))

    event = [p for p in client.published if p.event_type == "memory.consolidated"][-1]
    assert event.payload["diagnosis"] == "component_expectation_failed"
    assert event.payload["diagnosis_evidence"] == ["GC"]


async def test_gen_branch_routes_structured_preflight_component_failure_from_debug(
        store, monkeypatch):
    """A method incompatibility found before a metric is not a retryable code crash."""
    monkeypatch.setenv("AUTORESEARCH_GEN_EXPERIMENT", "1")
    handler = MemoryAgentHandler(spec_store=store, llm_client=FakeLLM([]))
    client = FakeBusClient()
    sid = str(uuid.uuid4())
    run_record_ref = _seed_gen_result(
        store, sid, verdict="invalid", pb=1.0, pm=None, accepted=False,
        diagnosis="execution_failure", diagnosis_evidence=["candidate did not run"],
    )
    store.put(
        kind="gen_experiment_result", session_id=sid, key="current",
        obj={
            "operator": "debug", "verdict": "invalid", "accepted": False,
            "baseline_metric": 1.0, "method_metric": None,
            "reasons": ["confirmed component is incompatible with repository semantics"],
            "diagnosis": "execution_failure",
            "diagnosis_evidence": ["candidate did not run"],
            "ablation_report": {
                "component": "confirmed attention replacement",
                "type": "functional", "actual": 0,
                "expected": {"operator": "==", "value": 1},
                "evidence": {
                    "source": "tracked_repository_source",
                    "ref": "model.py:40-75",
                },
                "verdict": "unmet",
            },
        }, emitter="RunnerAgent")

    await handler.handle_consolidation_requested(
        client, _Envelope(str(uuid.uuid4()), sid, "memory.consolidation_requested",
                          "ResearchOrchestrator", {"run_record_ref": run_record_ref}))

    event = [p for p in client.published if p.event_type == "memory.consolidated"][-1]
    assert event.payload["diagnosis"] == "component_expectation_failed"
    assert event.payload["diagnosis_evidence"] == ["confirmed attention replacement"]


async def test_gen_branch_keeps_structured_preflight_failure_when_evaluator_has_no_metric(
        store, monkeypatch):
    """Missing metric may be suspicious, but a failed precondition must change the method."""
    monkeypatch.setenv("AUTORESEARCH_GEN_EXPERIMENT", "1")
    handler = MemoryAgentHandler(spec_store=store, llm_client=FakeLLM([]))
    client = FakeBusClient()
    sid = str(uuid.uuid4())
    run_record_ref = _seed_gen_result(
        store, sid, verdict="invalid", pb=1.0, pm=None, accepted=False,
        diagnosis="execution_failure", diagnosis_evidence=["candidate did not run"],
    )
    store.put(
        kind="gen_experiment_result", session_id=sid, key="current",
        obj={
            "operator": "debug", "verdict": "invalid", "accepted": False,
            "baseline_metric": 1.0, "method_metric": None,
            "reasons": ["confirmed component is already present in the repository"],
            "diagnosis": "execution_failure",
            "diagnosis_evidence": ["candidate did not run"],
            "ablation_report": [{
                "component": "confirmed compile addition",
                "type": "gain", "actual": 0,
                "expected": {"operator": "==", "value": 1},
                "evidence": {
                    "source": "tracked_repository_source",
                    "ref": "train.py:100",
                },
                "verdict": "unmet",
            }],
        }, emitter="RunnerAgent")
    eval_ref = store.put(
        kind="evaluation_report", session_id=sid, key="v1",
        obj={
            "verdict": "suspicious", "is_improvement": False,
            "suspicious": True,
            "anti_cheat_flags": [
                "comparison_semantics_changed:no method metric was produced"],
        }, emitter="EvaluatorAgent")

    await handler.handle_consolidation_requested(
        client, _Envelope(str(uuid.uuid4()), sid, "memory.consolidation_requested",
                          "ResearchOrchestrator", {
                              "run_record_ref": run_record_ref,
                              "evaluation_report_ref": eval_ref,
                          }))

    event = [p for p in client.published if p.event_type == "memory.consolidated"][-1]
    assert event.payload["diagnosis"] == "component_expectation_failed"
    assert event.payload["diagnosis_evidence"] == ["confirmed compile addition"]


async def test_gen_branch_all_gain_preconditions_failed_falsifies_whole_method(
        store, monkeypatch):
    """When no gain mechanism remains implementable, component revision cannot save the method."""
    monkeypatch.setenv("AUTORESEARCH_GEN_EXPERIMENT", "1")
    handler = MemoryAgentHandler(spec_store=store, llm_client=FakeLLM([]))
    client = FakeBusClient()
    sid = str(uuid.uuid4())
    run_record_ref = _seed_gen_result(
        store, sid, verdict="invalid", pb=1.0, pm=None, accepted=False)
    store.put(
        kind="gen_experiment_result", session_id=sid, key="current",
        obj={
            "operator": "debug", "verdict": "invalid", "accepted": False,
            "baseline_metric": 1.0, "method_metric": None,
            "reasons": ["every gain component failed preflight"],
            "ablation_report": [
                {
                    "component": "gain one", "type": "gain", "actual": 0,
                    "expected": {"operator": "==", "value": 1},
                    "evidence": {"source": "tracked_repository_source", "ref": "a.py:1"},
                    "verdict": "unmet",
                },
                {
                    "component": "gain two", "type": "gain", "actual": 0,
                    "expected": {"operator": "==", "value": 1},
                    "evidence": {"source": "tracked_repository_source", "ref": "b.py:2"},
                    "verdict": "unmet",
                },
                {
                    "component": "supporting check", "type": "functional", "actual": 1,
                    "expected": {"operator": "==", "value": 1},
                    "evidence": {"source": "tracked_repository_source", "ref": "c.py:3"},
                    "verdict": "met",
                },
            ],
        }, emitter="RunnerAgent")

    await handler.handle_consolidation_requested(
        client, _Envelope(str(uuid.uuid4()), sid, "memory.consolidation_requested",
                          "ResearchOrchestrator", {"run_record_ref": run_record_ref}))

    event = [p for p in client.published if p.event_type == "memory.consolidated"][-1]
    assert event.payload["diagnosis"] == "method_hypothesis_falsified"
    assert event.payload["diagnosis_evidence"] == ["gain one", "gain two"]


async def test_gen_branch_does_not_trust_preflight_component_label_with_finite_metric(
        store, monkeypatch):
    """Outside CHECK-EXPECTATION, a completed run remains governed by its real metric."""
    monkeypatch.setenv("AUTORESEARCH_GEN_EXPERIMENT", "1")
    handler = MemoryAgentHandler(spec_store=store, llm_client=FakeLLM([]))
    client = FakeBusClient()
    sid = str(uuid.uuid4())
    run_record_ref = _seed_gen_result(
        store, sid, verdict="valid_not_best", pb=1.0, pm=1.1, accepted=False)
    store.put(
        kind="gen_experiment_result", session_id=sid, key="current",
        obj={
            "operator": "debug", "verdict": "valid_not_best", "accepted": False,
            "baseline_metric": 1.0, "method_metric": 1.1, "reasons": [],
            "ablation_report": [{
                "component": "claimed blocker", "type": "functional", "actual": 0,
                "expected": {"operator": "==", "value": 1},
                "evidence": {"source": "sandbox", "ref": "self-report"},
                "verdict": "unmet",
            }],
        }, emitter="RunnerAgent")

    await handler.handle_consolidation_requested(
        client, _Envelope(str(uuid.uuid4()), sid, "memory.consolidation_requested",
                          "ResearchOrchestrator", {"run_record_ref": run_record_ref}))

    event = [p for p in client.published if p.event_type == "memory.consolidated"][-1]
    assert event.payload["diagnosis"] == "inconclusive"


async def test_gen_branch_discards_sandbox_setting_issue_without_structured_evidence(
        store, monkeypatch):
    monkeypatch.setenv("AUTORESEARCH_GEN_EXPERIMENT", "1")
    handler = MemoryAgentHandler(spec_store=store, llm_client=FakeLLM([]))
    client = FakeBusClient()
    sid = str(uuid.uuid4())
    run_record_ref = _seed_gen_result(
        store, sid, verdict="valid_not_best", pb=1.0, pm=1.1, accepted=False,
        diagnosis="setting_issue", diagnosis_evidence=["sandbox says tune batch size"],
    )

    await handler.handle_consolidation_requested(
        client, _Envelope(str(uuid.uuid4()), sid, "memory.consolidation_requested",
                          "ResearchOrchestrator", {"run_record_ref": run_record_ref}))

    event = [p for p in client.published if p.event_type == "memory.consolidated"][-1]
    assert event.payload["diagnosis"] == "inconclusive"
    assert event.payload["routing_verdict"] == "continue_explore"


async def test_gen_branch_rejects_untraceable_setting_evidence(store, monkeypatch):
    monkeypatch.setenv("AUTORESEARCH_GEN_EXPERIMENT", "1")
    handler = MemoryAgentHandler(spec_store=store, llm_client=FakeLLM([]))
    client = FakeBusClient()
    sid = str(uuid.uuid4())
    run_record_ref = _seed_gen_result(
        store, sid, verdict="valid_not_best", pb=1.0, pm=1.1, accepted=False,
        diagnosis="setting_issue", diagnosis_evidence=["self-report"],
    )
    store.put(
        kind="gen_experiment_result", session_id=sid, key="current",
        obj={
            "verdict": "valid_not_best", "accepted": False,
            "baseline_metric": 1.0, "method_metric": 1.1, "reasons": [],
            "setting_evidence": [{
                "setting": "batch_size", "actual": 32, "expected": 64,
                "evidence": {"claim": "sandbox says so"},
            }],
        }, emitter="RunnerAgent")

    await handler.handle_consolidation_requested(
        client, _Envelope(str(uuid.uuid4()), sid, "memory.consolidation_requested",
                          "ResearchOrchestrator", {"run_record_ref": run_record_ref}))

    event = [p for p in client.published if p.event_type == "memory.consolidated"][-1]
    assert event.payload["diagnosis"] == "inconclusive"


async def test_gen_branch_derives_setting_issue_from_structured_mismatch(store, monkeypatch):
    monkeypatch.setenv("AUTORESEARCH_GEN_EXPERIMENT", "1")
    handler = MemoryAgentHandler(spec_store=store, llm_client=FakeLLM([]))
    client = FakeBusClient()
    sid = str(uuid.uuid4())
    run_record_ref = _seed_gen_result(
        store, sid, verdict="valid_not_best", pb=1.0, pm=1.1, accepted=False)
    store.put(
        kind="gen_experiment_result", session_id=sid, key="current",
        obj={
            "verdict": "valid_not_best", "accepted": False,
            "baseline_metric": 1.0, "method_metric": 1.1, "reasons": [],
            "setting_evidence": [{
                "setting": "batch_size", "actual": 32, "expected": 64,
                "evidence": {"source": "run_manifest", "ref": run_record_ref},
            }],
        }, emitter="RunnerAgent")

    await handler.handle_consolidation_requested(
        client, _Envelope(str(uuid.uuid4()), sid, "memory.consolidation_requested",
                          "ResearchOrchestrator", {"run_record_ref": run_record_ref}))

    event = [p for p in client.published if p.event_type == "memory.consolidated"][-1]
    assert event.payload["diagnosis"] == "setting_issue"
    assert event.payload["routing_verdict"] == "need_experiment"
    assert "batch_size" in event.payload["diagnosis_evidence"][0]


@pytest.mark.parametrize(("eval_verdict", "structured_kind", "expected_routing"), [
    ("invalid", "component", "continue_explore"),
    ("suspicious", "setting", "report"),
])
async def test_evaluator_veto_blocks_component_and_setting_mutation_diagnoses(
        store, monkeypatch, eval_verdict, structured_kind, expected_routing):
    """Distrusted evaluator results may be recorded, but cannot mutate method/settings."""
    monkeypatch.setenv("AUTORESEARCH_GEN_EXPERIMENT", "1")
    handler = MemoryAgentHandler(spec_store=store, llm_client=FakeLLM([]))
    client = FakeBusClient()
    sid = str(uuid.uuid4())
    run_record_ref = _seed_gen_result(
        store, sid, verdict="valid_not_best", pb=1.0, pm=1.1, accepted=False)
    current = {
        "verdict": "valid_not_best", "accepted": False,
        "baseline_metric": 1.0, "method_metric": 1.1, "reasons": [],
    }
    if structured_kind == "component":
        current.update({
            "operator": "check-expectation",
            "ablation_report": [{
                "component": "router", "type": "functional", "actual": 0.4,
                "expected": {"operator": "<=", "value": 0.1},
                "evidence": {"source": "run_record", "ref": run_record_ref},
                "verdict": "unmet",
            }],
        })
    else:
        current["setting_evidence"] = [{
            "setting": "batch_size", "actual": 32, "expected": 64,
            "evidence": {"source": "run_manifest", "ref": run_record_ref},
        }]
    store.put(kind="gen_experiment_result", session_id=sid, key="current",
              obj=current, emitter="RunnerAgent")
    eval_ref = store.put(
        kind="evaluation_report", session_id=sid, key="v1",
        obj={
            "verdict": eval_verdict, "is_improvement": False,
            "suspicious": eval_verdict == "suspicious",
            "anti_cheat_flags": ["untrusted result"],
        }, emitter="EvaluatorAgent")

    await handler.handle_consolidation_requested(
        client, _Envelope(str(uuid.uuid4()), sid, "memory.consolidation_requested",
                          "ResearchOrchestrator", {
                              "run_record_ref": run_record_ref,
                              "evaluation_report_ref": eval_ref,
                          }))

    event = [p for p in client.published if p.event_type == "memory.consolidated"][-1]
    assert event.payload["diagnosis"] == "inconclusive"
    assert event.payload["routing_verdict"] == expected_routing


@pytest.mark.parametrize("run_status", [False, None])
async def test_failure_consolidation_without_evaluation_persists_honest_negative(
        store, monkeypatch, run_status):
    """A MOSS-discarded failed run still becomes durable ResearchMemory without an evaluator."""
    # This is the production repo loop mode too: explicit MOSS failure context wins over
    # looking for a successful gen result that cannot exist for this discarded run.
    monkeypatch.setenv("AUTORESEARCH_GEN_EXPERIMENT", "1")
    handler = MemoryAgentHandler(spec_store=store, llm_client=FakeLLM([]))
    client = FakeBusClient()
    sid = str(uuid.uuid4())
    run_ref = store.put(
        kind="run_record", session_id=sid, key="failed",
        obj={
            "succeeded": run_status, "exit_code": 1, "metric_value": None,
            "stdout_excerpt": "", "stderr_excerpt": "candidate crashed",
            "environment": "sandbox",
        }, emitter="RunnerAgent")
    baseline_ref = store.put(
        kind="baseline_result", session_id=sid, key="v1",
        obj={"benchmark_id": "bench", "metric_value": 1.0, "reproduced": True},
        emitter="RunnerAgent")

    await handler.handle_consolidation_requested(
        client, _Envelope(str(uuid.uuid4()), sid, "memory.consolidation_requested",
                          "ResearchOrchestrator", {
                              "evaluation_report_ref": None,
                              "run_record_ref": run_ref,
                              "baseline_result_ref": baseline_ref,
                              "failure_context": {
                                  "reason": "implementation_thrashing",
                                  "action": "discard_hypothesis",
                                  "artifact_ref": "moss_intervention://discard",
                              },
                          }))

    event = [p for p in client.published if p.event_type == "memory.consolidated"][-1]
    assert event.payload["routing_verdict"] == "report"
    assert isinstance(event.payload["research_memory_ref"], str)
    memory = store.get(event.payload["research_memory_ref"])
    assert memory["verified_findings"] == []
    assert any("candidate crashed" in item for item in memory["negative_results"])
    assert memory["diagnosis"] == "component_execution_blocked"
    assert event.payload["diagnosis"] == "component_execution_blocked"


async def test_evaluated_moss_discard_routes_as_method_falsification(store, monkeypatch):
    """A discard retains a negative evaluation bound to the actual run."""
    from autoresearch.spec_store.identity import artifact_digest
    monkeypatch.setenv("AUTORESEARCH_GEN_EXPERIMENT", "1")
    handler = MemoryAgentHandler(spec_store=store, llm_client=FakeLLM([]))
    client = FakeBusClient()
    sid = str(uuid.uuid4())
    run_ref = _seed_gen_result(
        store, sid, verdict="valid_not_best", pb=1.0, pm=1.1, accepted=False)
    eval_ref = store.put(
        kind="evaluation_report", session_id=sid, key="v1",
        obj={"verdict": "valid_not_best", "metric_value": 1.1,
             "baseline_metric": 1.0, "is_improvement": False,
             "suspicious": False, "anti_cheat_flags": [],
             "run_record_digest": artifact_digest(store.get(run_ref)),
             "reasoning": "candidate did not improve"},
        emitter="EvaluatorAgent")

    await handler.handle_consolidation_requested(
        client, _Envelope(str(uuid.uuid4()), sid, "memory.consolidation_requested",
                          "ResearchOrchestrator", {
                              "evaluation_report_ref": eval_ref,
                              "run_record_ref": run_ref,
                              "baseline_result_ref": None,
                              "failure_context": {
                                  "reason": "implementation_thrashing",
                                  "action": "discard_hypothesis",
                                  "artifact_ref": "moss_intervention://discard",
                              },
                          }))

    event = [p for p in client.published if p.event_type == "memory.consolidated"][-1]
    assert event.payload["diagnosis"] == "method_hypothesis_falsified"
    assert event.payload["routing_verdict"] == "report"


async def test_verified_attempt_failure_reason_is_persisted_for_next_attempt(
        store, monkeypatch):
    monkeypatch.setenv("AUTORESEARCH_GEN_EXPERIMENT", "1")
    handler = MemoryAgentHandler(spec_store=store, llm_client=FakeLLM([]))
    client = FakeBusClient()
    sid = str(uuid.uuid4())
    run_ref = _seed_gen_result(
        store, sid, verdict="invalid", pb=1.0, pm=None, accepted=False)
    eval_ref = store.put(
        kind="evaluation_report", session_id=sid, key="v1",
        obj={"verdict": "invalid", "metric_value": None,
             "baseline_metric": 1.0, "is_improvement": False,
             "suspicious": False, "anti_cheat_flags": [],
             "reasoning": "candidate did not produce a metric"},
        emitter="EvaluatorAgent")
    verification_ref = store.put(
        kind="verification_report", session_id=sid, key="v1",
        obj={"verdict": "verified",
             "checks": ["failure_confirmed: attempt recorded an invalid outcome and a cause"],
             "anti_cheat_flags": [],
             "reasoning": "Recorded failure cause: CUDA out of memory"},
        emitter="EvaluatorAgent")

    await handler.handle_consolidation_requested(
        client, _Envelope(str(uuid.uuid4()), sid, "memory.consolidation_requested",
                          "ResearchOrchestrator", {
                              "run_record_ref": run_ref,
                              "evaluation_report_ref": eval_ref,
                              "verification_report_ref": verification_ref,
                          }))

    event = [p for p in client.published if p.event_type == "memory.consolidated"][-1]
    memory = store.get(event.payload["research_memory_ref"])
    assert memory["diagnosis"] == "execution_failure"
    assert any("CUDA out of memory" in item for item in memory["diagnosis_evidence"])
    assert any("CUDA out of memory" in item for item in memory["negative_results"])


async def test_novelty_exhaustion_failure_context_is_not_an_execution_failure(
        store, monkeypatch):
    monkeypatch.setenv("AUTORESEARCH_GEN_EXPERIMENT", "1")
    handler = MemoryAgentHandler(spec_store=store, llm_client=FakeLLM([]))
    client = FakeBusClient()
    sid = str(uuid.uuid4())

    await handler.handle_consolidation_requested(
        client, _Envelope(str(uuid.uuid4()), sid, "memory.consolidation_requested",
                          "ResearchOrchestrator", {
                              "evaluation_report_ref": None,
                              "run_record_ref": None,
                              "baseline_result_ref": None,
                              "failure_context": {
                                  "reason": "literature_novelty_audit_failed",
                                  "action": "targeted_search",
                                  "artifact_ref": "moss_intervention://novelty-cap",
                              },
                          }))

    event = [p for p in client.published if p.event_type == "memory.consolidated"][-1]
    assert event.payload["routing_verdict"] == "report"
    assert event.payload["diagnosis"] == "literature_novelty_exhausted"
    memory = store.get(event.payload["research_memory_ref"])
    assert memory["diagnosis"] == "literature_novelty_exhausted"


async def test_baseline_reproduction_exhaustion_is_not_an_implementation_failure(
        store, monkeypatch):
    monkeypatch.setenv("AUTORESEARCH_GEN_EXPERIMENT", "1")
    handler = MemoryAgentHandler(spec_store=store, llm_client=FakeLLM([]))
    client = FakeBusClient()
    sid = str(uuid.uuid4())

    await handler.handle_consolidation_requested(
        client, _Envelope(str(uuid.uuid4()), sid, "memory.consolidation_requested",
                          "ResearchOrchestrator", {
                              "evaluation_report_ref": None,
                              "run_record_ref": None,
                              "baseline_result_ref": None,
                              "failure_context": {
                                  "reason": "baseline_reproduction_thrashing",
                                  "action": "fix_environment",
                                  "artifact_ref": "moss_intervention://baseline-cap",
                              },
                          }))

    event = [p for p in client.published if p.event_type == "memory.consolidated"][-1]
    assert event.payload["routing_verdict"] == "report"
    assert event.payload["diagnosis"] == "baseline_reproduction_failed"
    memory = store.get(event.payload["research_memory_ref"])
    assert memory["diagnosis"] == "baseline_reproduction_failed"


async def test_gen_branch_repo_new_best_is_verified(store, monkeypatch):
    # repo_improve (GPU) path: verdict vocabulary is "new_best" (not "pass") but accepted=True
    # for a reproduced, isolated win, and the trusted evaluator did NOT flag it. It must land
    # in verified_findings, NOT negative_results (previously "new_best" fell through to
    # negative because the check was verdict=="pass" only).
    monkeypatch.setenv("AUTORESEARCH_GEN_EXPERIMENT", "1")
    handler = MemoryAgentHandler(spec_store=store, llm_client=FakeLLM([]))
    client = FakeBusClient()
    sid = str(uuid.uuid4())
    run_record_ref = _seed_gen_result(
        store, sid, verdict="new_best", pb=1.01735, pm=1.009543, accepted=True)
    eval_ref = store.put(
        kind="evaluation_report", session_id=sid, key="v1",
        obj={"verdict": "new_best", "metric_value": 1.009543, "baseline_metric": 1.01735,
             "is_improvement": True, "suspicious": False, "anti_cheat_flags": []},
        emitter="EvaluatorAgent")
    env = _Envelope(str(uuid.uuid4()), sid, "memory.consolidation_requested",
                    "ResearchOrchestrator",
                    {"run_record_ref": run_record_ref, "evaluation_report_ref": eval_ref})

    await handler.handle_consolidation_requested(client, env)

    consolidated = [p for p in client.published if p.event_type == "memory.consolidated"]
    memory = store.get(consolidated[0].payload["research_memory_ref"])
    assert len(memory["verified_findings"]) == 1
    finding = memory["verified_findings"][0]
    assert "1.01735" in finding and "1.009543" in finding
    assert memory["negative_results"] == []


# ---------------------------------------------------------------------------
# H5: a FAIL verdict lands in negative_results (with reasons). A clean honest
# negative still FEEDS the loop (continue_explore) -- the negative informs the
# next hypothesis; the engine's round cap guarantees termination.
# ---------------------------------------------------------------------------


async def test_gen_branch_fail_goes_to_negative_results_and_feeds_loop(store, monkeypatch):
    monkeypatch.setenv("AUTORESEARCH_GEN_EXPERIMENT", "1")
    llm = FakeLLM([])
    handler = MemoryAgentHandler(spec_store=store, llm_client=llm)
    client = FakeBusClient()

    sid = str(uuid.uuid4())
    run_record_ref = _seed_gen_result(store, sid, verdict="fail", pb=100.0, pm=99.0)
    env = _Envelope(str(uuid.uuid4()), sid, "memory.consolidation_requested",
                    "ResearchOrchestrator", {"run_record_ref": run_record_ref})

    await handler.handle_consolidation_requested(client, env)
    assert llm.calls == []

    consolidated = [p for p in client.published if p.event_type == "memory.consolidated"]
    # a clean honest negative is a finding the next round can build on
    assert consolidated[0].payload["routing_verdict"] == "continue_explore"
    memory = store.get(consolidated[0].payload["research_memory_ref"])
    assert memory["verified_findings"] == []
    assert len(memory["negative_results"]) == 1
    neg = memory["negative_results"][0]
    assert "r1" in neg and "r2" in neg  # reasons carried
    assert memory["best_metric"] == 99.0


async def test_gen_branch_accumulates_findings_across_loop_rounds(store, monkeypatch):
    # Round 2's consolidation MERGES into round 1's ResearchMemory: findings from BOTH rounds
    # are present, and a later win updates best_metric.
    monkeypatch.setenv("AUTORESEARCH_GEN_EXPERIMENT", "1")
    handler = MemoryAgentHandler(spec_store=store, llm_client=FakeLLM([]))
    client = FakeBusClient()
    sid = str(uuid.uuid4())

    # round 1: an honest negative (verdict=fail) -> negative_results, continue_explore
    rr1 = _seed_gen_result(store, sid, verdict="fail", pb=100.0, pm=99.0)
    await handler.handle_consolidation_requested(
        client, _Envelope(str(uuid.uuid4()), sid, "memory.consolidation_requested",
                          "ResearchOrchestrator", {"run_record_ref": rr1}))
    rm1_ref = [p for p in client.published if p.event_type == "memory.consolidated"][-1] \
        .payload["research_memory_ref"]
    rm1 = store.get(rm1_ref)
    assert len(rm1["negative_results"]) == 1 and rm1["verified_findings"] == []

    # round 2: a win (verdict=pass) with the PRIOR memory threaded in -> accumulates
    rr2 = _seed_gen_result(store, sid, verdict="pass", pb=99.0, pm=50.0)
    await handler.handle_consolidation_requested(
        client, _Envelope(str(uuid.uuid4()), sid, "memory.consolidation_requested",
                          "ResearchOrchestrator",
                          {"run_record_ref": rr2, "research_memory_ref": rm1_ref}))
    rm2 = store.get([p for p in client.published if p.event_type == "memory.consolidated"][-1]
                    .payload["research_memory_ref"])
    # BOTH rounds' findings survive; the win set the new best
    assert len(rm2["negative_results"]) == 1        # round-1 negative carried forward
    assert len(rm2["verified_findings"]) == 1       # round-2 win added
    assert rm2["best_metric"] == 50.0


@pytest.mark.parametrize(
    ("lower_is_better", "prior_best", "later_value"),
    [(True, 1.056768, 1.138752), (False, 1.57179, 1.56097)],
)
async def test_gen_branch_preserves_better_prior_metric_across_accepted_rounds(
    store, monkeypatch, lower_is_better, prior_best, later_value,
):
    """A sandbox-relative win must not overwrite a stronger outer-loop best."""
    monkeypatch.setenv("AUTORESEARCH_GEN_EXPERIMENT", "1")
    handler = MemoryAgentHandler(spec_store=store, llm_client=FakeLLM([]))
    client = FakeBusClient()
    sid = str(uuid.uuid4())
    prior_ref = store.put(
        kind="research_memory", session_id=sid, key="v1",
        obj={
            "best_metric": prior_best, "best_result_ref": None,
            "verified_findings": ["earlier verified result"], "negative_results": [],
            "excluded_hypotheses": [], "consolidated_insights": [],
            "divergence_seeds": [], "next_steps": [],
        },
        emitter="MemoryAgent",
    )
    run_ref = _seed_gen_result(
        store, sid, verdict="new_best", pb=100.0, pm=later_value,
        accepted=True, lower_is_better=lower_is_better,
    )

    await handler.handle_consolidation_requested(
        client,
        _Envelope(
            str(uuid.uuid4()), sid, "memory.consolidation_requested",
            "ResearchOrchestrator",
            {"run_record_ref": run_ref, "research_memory_ref": prior_ref},
        ),
    )

    memory_ref = [
        item for item in client.published if item.event_type == "memory.consolidated"
    ][-1].payload["research_memory_ref"]
    assert store.get(memory_ref)["best_metric"] == prior_best


async def test_gen_branch_uses_current_round_and_demotes_failed_verification(store, monkeypatch):
    """A retained best-so-far must not overwrite the strict rerun that just failed it."""
    monkeypatch.setenv("AUTORESEARCH_GEN_EXPERIMENT", "1")
    handler = MemoryAgentHandler(spec_store=store, llm_client=FakeLLM([]))
    client = FakeBusClient()
    sid = str(uuid.uuid4())

    run_ref = _seed_gen_result(
        store, sid, verdict="new_best", pb=1.023444, pm=1.023204, accepted=True)
    store.put(
        kind="gen_experiment_result", session_id=sid, key="current",
        obj={"verdict": "valid_not_best", "accepted": False,
             "reasons": ["strict rerun did not improve"],
             "baseline_metric": 1.022782, "method_metric": 1.023369,
             "isolated": True}, emitter="RunnerAgent")
    eval_ref = store.put(
        kind="evaluation_report", session_id=sid, key="v1",
        obj={"verdict": "valid_not_best", "metric_value": 1.023369,
             "baseline_metric": 1.022782, "is_improvement": False,
             "suspicious": False, "anti_cheat_flags": []}, emitter="EvaluatorAgent")
    verification_ref = store.put(
        kind="verification_report", session_id=sid, key="v1",
        obj={"verdict": "failed", "checks": ["marginal win did not reproduce"],
             "anti_cheat_flags": [], "reasoning": "strict rerun required"},
        emitter="EvaluatorAgent")
    prior_ref = store.put(
        kind="research_memory", session_id=sid, key="v1",
        obj={"best_metric": 1.023204, "best_result_ref": None,
             "verified_findings": [
                 "Method beats baseline: baseline=1.023444 vs method=1.023204 "
                 "(metric value), reproduced, run-time isolated."
             ], "negative_results": [], "excluded_hypotheses": [],
             "consolidated_insights": [], "divergence_seeds": [], "next_steps": []},
        emitter="MemoryAgent")

    await handler.handle_consolidation_requested(
        client, _Envelope(str(uuid.uuid4()), sid, "memory.consolidation_requested",
                          "ResearchOrchestrator", {
                              "run_record_ref": run_ref,
                              "evaluation_report_ref": eval_ref,
                              "verification_report_ref": verification_ref,
                              "research_memory_ref": prior_ref,
                          }))

    event = [p for p in client.published if p.event_type == "memory.consolidated"][-1]
    memory = store.get(event.payload["research_memory_ref"])
    assert event.payload["routing_verdict"] == "continue_explore"
    assert memory["verified_findings"] == []
    assert any("1.022782" in item and "1.023369" in item
               for item in memory["negative_results"])
    assert any("verification failed" in item.lower() for item in memory["negative_results"])


# ---------------------------------------------------------------------------
# DEFAULT-OFF: env off -> real LLM consolidate path (LLM IS called).
# ---------------------------------------------------------------------------


async def test_env_off_uses_real_llm_path(store, monkeypatch):
    import json

    monkeypatch.delenv("AUTORESEARCH_GEN_EXPERIMENT", raising=False)
    memory_obj = {
        "verified_findings": [], "negative_results": [],
        "excluded_hypotheses": [], "consolidated_insights": [],
        "divergence_seeds": [], "next_steps": [],
    }
    llm = FakeLLM([json.dumps(
        {"research_memory": memory_obj, "routing_verdict": "continue_explore"})])
    handler = MemoryAgentHandler(spec_store=store, llm_client=llm)
    client = FakeBusClient()

    sid = str(uuid.uuid4())
    eval_ref = store.put(kind="evaluation_report", session_id=sid, key="v1",
                         obj={"verdict": "no_improvement"}, emitter="EvaluatorAgent")
    run_record_ref = store.put(
        kind="run_record", session_id=sid, key="v1",
        obj={"succeeded": True, "exit_code": 0, "metric_value": 1.0,
             "stdout_excerpt": "x", "environment": "e"},
        emitter="RunnerAgent")
    env = _Envelope(str(uuid.uuid4()), sid, "memory.consolidation_requested",
                    "ResearchOrchestrator",
                    {"evaluation_report_ref": eval_ref, "run_record_ref": run_record_ref})

    await handler.handle_consolidation_requested(client, env)

    assert len(llm.calls) == 1  # real path called the LLM
    consolidated = [p for p in client.published if p.event_type == "memory.consolidated"]
    assert len(consolidated) == 1
    assert consolidated[0].payload["routing_verdict"] == "continue_explore"


async def test_agentic_result_uses_deterministic_diagnosis_without_legacy_gen_flag(
        store, monkeypatch):
    """A real repo run must not lose its structured diagnosis because an unrelated old flag is off."""
    monkeypatch.delenv("AUTORESEARCH_GEN_EXPERIMENT", raising=False)
    llm = FakeLLM([])
    handler = MemoryAgentHandler(spec_store=store, llm_client=llm)
    client = FakeBusClient()
    sid = str(uuid.uuid4())
    run_ref = _seed_gen_result(
        store, sid, verdict="valid_not_best", pb=1.0, pm=1.1, accepted=False)
    store.put(
        kind="gen_experiment_result", session_id=sid, key="current",
        obj={
            "verdict": "valid_not_best", "accepted": False,
            "reasons": ["measured non-win"], "baseline_metric": 1.0,
            "method_metric": 1.1, "isolated": True,
        }, emitter="RunnerAgent")

    await handler.handle_consolidation_requested(
        client, _Envelope(str(uuid.uuid4()), sid, "memory.consolidation_requested",
                          "ResearchOrchestrator", {"run_record_ref": run_ref}))

    assert llm.calls == []
    event = [p for p in client.published if p.event_type == "memory.consolidated"][-1]
    assert event.payload["routing_verdict"] == "continue_explore"
    assert event.payload["diagnosis"] == "inconclusive"
    assert event.payload["diagnosis_evidence"] == [
        "valid comparison did not improve: baseline=1.0, method=1.1"]


@pytest.mark.asyncio
async def test_paper_reproduction_note_surfaces_in_consolidated_insights(store, monkeypatch):
    # Paper-reproduction line ON: the runner attached a paper_reproduction assessment; the
    # MemoryAgent surfaces it as a neutral consolidated insight so the report shows the check.
    monkeypatch.setenv("AUTORESEARCH_GEN_EXPERIMENT", "1")
    handler = MemoryAgentHandler(spec_store=store, llm_client=FakeLLM([]))
    client = FakeBusClient()
    sid = str(uuid.uuid4())
    run_record_ref = store.put(
        kind="run_record", session_id=sid, key="v1",
        obj={"succeeded": True, "exit_code": 0, "metric_value": 1.69,
             "stdout_excerpt": "report", "environment": "repo_improve"},
        emitter="RunnerAgent")
    store.put(
        kind="gen_experiment_result", session_id=sid, key="v1",
        obj={"verdict": "new_best", "accepted": True, "reasons": [],
             "baseline_metric": 1.02, "method_metric": 0.98, "isolated": True,
             "paper_reproduction": {"status": "reproduced", "measured": 1.02,
                                    "published": 1.00, "source": "GPT-2 paper",
                                    "note": None, "rel_error": 0.02, "tolerance": 0.05,
                                    "lower_is_better": True}},
        emitter="RunnerAgent")
    env = _Envelope(str(uuid.uuid4()), sid, "memory.consolidation_requested",
                    "ResearchOrchestrator", {"run_record_ref": run_record_ref})
    await handler.handle_consolidation_requested(client, env)

    memory = store.get(
        [p for p in client.published if p.event_type == "memory.consolidated"][0]
        .payload["research_memory_ref"])
    insights = memory["consolidated_insights"]
    assert any("论文复现线" in s and "GPT-2 paper" in s for s in insights), insights


async def test_survey_validation_consolidates_theory_and_result_then_reports(store):
    handler = MemoryAgentHandler(spec_store=store, llm_client=FakeLLM([]))
    client = FakeBusClient()
    sid = str(uuid.uuid4())
    method_ref = store.put(
        kind="method", session_id=sid, key="v1",
        obj={"approach": "Derive a falsifiable publication-value claim.",
             "changes_outline": ["Proposition 1"], "components": []},
        emitter="MethodAgent")
    interpretation_ref = store.put(
        kind="spec_interpretation", session_id=sid, key="v1",
        obj={"disposition": "in_scope", "scope_reason": "runnable survey",
             "validation_study": {"kind": "survey_validation"}},
        emitter="ProblemModelingAgent")
    run_ref = store.put(
        kind="run_record", session_id=sid, key="v1",
        obj={"succeeded": True, "exit_code": 0, "metric_value": 0.03,
             "stdout_excerpt": "No stable association.", "stderr_excerpt": None,
             "environment": "survey_validation|host_rerun", "seed": None,
             "runtime_seconds": 2.0,
             "artifacts": [f"spec://validation_study_result/{sid}/v1"]},
        emitter="RunnerAgent")
    store.put(
        kind="validation_study_result", session_id=sid, key="v1",
        obj={"conclusion": "inconclusive", "metric_name": "spearman_r",
             "metric_value": 0.03, "sample_ids": ["W1"],
             "retrieved_at": "2026-08-20T10:00:00Z", "data_sources": ["OpenAlex"],
             "analysis_summary": "No stable association.", "artifacts": ["raw_data.json"]},
        emitter="RunnerAgent")
    evaluation_ref = store.put(
        kind="evaluation_report", session_id=sid, key="v1",
        obj={"verdict": "valid_not_best", "metric_value": 0.03,
             "baseline_metric": None, "is_improvement": False, "suspicious": False,
             "anti_cheat_flags": [], "reasoning": "survey_validation:inconclusive"},
        emitter="EvaluatorAgent")
    verification_ref = store.put(
        kind="verification_report", session_id=sid, key="v1",
        obj={"verdict": "verified", "checks": ["sample_ids_present"],
             "anti_cheat_flags": [], "reasoning": "consistent"},
        emitter="EvaluatorAgent")
    env = _Envelope(
        str(uuid.uuid4()), sid, "memory.consolidation_requested", "ResearchOrchestrator",
        {"run_record_ref": run_ref, "evaluation_report_ref": evaluation_ref,
         "verification_report_ref": verification_ref,
         "theory_method_ref": method_ref, "problem_model_ref": None,
         "spec_interpretation_ref": interpretation_ref})

    await handler.handle_consolidation_requested(client, env)

    [event] = [p for p in client.published if p.event_type == "memory.consolidated"]
    assert event.payload["routing_verdict"] == "report"
    memory = store.get(event.payload["research_memory_ref"])
    assert list(RESEARCH_MEMORY_VALIDATOR.iter_errors(memory)) == []
    assert memory["best_metric"] == 0.03
    assert any("inconclusive" in item for item in memory["negative_results"])
    assert any("Proposition 1" in item for item in memory["consolidated_insights"])
