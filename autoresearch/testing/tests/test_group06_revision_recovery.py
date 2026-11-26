"""Offline counterexamples: plan conclusions, current-version recovery, and probes."""
from __future__ import annotations

import copy
import json
from contextlib import closing

import pytest

from autoresearch.agents.method_implementation_agent import core as mir
from autoresearch.agents.moss_agent import core as moss
from autoresearch.agents.moss_agent.tests.conftest import FakeLLM
from autoresearch.agents.memory_agent.handler import (
    MemoryAgentHandler, _classify_component_evidence,
)
from autoresearch.agents.runner_agent.handler import RunnerAgentHandler
from autoresearch.orchestrator.moss_matrix import MossActionGuard, load_moss_action_matrix
from autoresearch.spec_store import SpecStore


METHOD = {
    "approach": "Measure schedules and report the conclusion.",
    "components": [{
        "component_id": "summary", "name": "Conclusion", "type": "indirect_gain",
        "mechanism": "Always report no improvement; do not report a refuted bound.",
    }],
}
DIFF = """diff --git a/conclusion.py b/conclusion.py
--- a/conclusion.py
+++ b/conclusion.py
@@ -0,0 +1,2 @@
+claimed_lower_bound = 17
+conclusion = 'no improvement'  # emitted even when a valid schedule costs 16
"""


def review_reply(**revision_overrides):
    revision = {
        "component_id": "summary",
        "plan_quote": METHOD["components"][0]["mechanism"],
        "reason": "The unconditional conclusion suppresses contrary measurements.",
        "evidence": ["E2"],
        **revision_overrides,
    }
    return {
        "verdict": "faithful", "approach_evidence": ["E2"],
        "component_reviews": [
            {"name": "Conclusion", "status": "implemented", "evidence": ["E2"]}],
        "issues": [], "remediation": None, "reasoning": "The code follows the flawed plan.",
        "plan_revision": revision,
    }


def test_plan_revision_evidence_is_resolved_without_relabeling_fidelity():
    result = mir.review_implementation(METHOD, DIFF, FakeLLM([json.dumps(review_reply())]))
    assert result["verdict"] == "faithful"
    assert result["plan_revision"]["evidence"] == [
        "conclusion = 'no improvement'  # emitted even when a valid schedule costs 16"]


@pytest.mark.parametrize("override", [
    {"component_id": "unknown"}, {"plan_quote": "invented frozen requirement"},
    {"evidence": ["E999"]}, {"reason": ""}, {"component_id": None},
])
def test_invented_plan_revision_fails_closed(override):
    llm = FakeLLM([json.dumps(review_reply(**override))] * 6)
    result = mir.review_implementation(METHOD, DIFF, llm)
    assert result["verdict"] == "insufficient_evidence"
    assert not result.get("plan_revision")


def test_faithful_but_unsound_plan_cannot_be_accepted_by_runner(tmp_path):
    with closing(SpecStore(tmp_path / "store")) as store:
        ref = store.put(kind="method", session_id="s", key="confirmed",
                        obj=METHOD, emitter="MethodAgent")
        runner = object.__new__(RunnerAgentHandler)
        runner.spec_store = store
        runner.p1_llm = FakeLLM([json.dumps(review_reply())])
        runner._latest_method_ref = lambda _: ref
        measured = {
            "candidate": {"valid": True, "metric": 16.0},
            "verdict": {"improved": True}, "diff": DIFF, "repo_path": "/saved/candidate",
        }
        result = runner._review_method_implementation("s", measured, "method")
    assert result["verdict"]["improved"] is False
    assert result["candidate"]["valid"] is False
    assert result["candidate"]["metric"] == 16.0
    assert result["repo_path"] == "/saved/candidate"
    assert result["diagnosis"] == "component_expectation_failed"
    assert measured["candidate"]["valid"] is True


def test_plan_revision_precedes_accepted_component_routing():
    review = review_reply()
    review["plan_revision"]["evidence"] = [
        "conclusion = 'no improvement'  # emitted even when a valid schedule costs 16"]
    review["method_ref"] = "spec://method/s/confirmed"
    gen = {
        "accepted": False, "method_metric": 16.0,
        "method_ref": "spec://method/s/confirmed",
        "method_implementation_review": review,
        "diagnosis": "component_expectation_failed",
        "diagnosis_evidence": [review["plan_revision"]["reason"]],
        "component_execution": {"mode": "component", "component_id": "summary",
                                "status": "contract_failed"},
    }
    diagnosis, evidence = _classify_component_evidence(gen)
    assert diagnosis == "component_expectation_failed"
    assert "summary" in evidence
    # A review of another Method cannot direct this Method's repair.
    gen["method_ref"] = "spec://method/s/revised"
    assert _classify_component_evidence(gen)[0] != "component_expectation_failed"


def test_moss_retries_a_decision_bound_to_an_obsolete_method():
    guard = MossActionGuard(load_moss_action_matrix())
    current = {"method_ref": "spec://method/s/new", "approach": "Revised probe",
               "components": [{"component_id": "probe", "mechanism": "Accepted-domain input"}]}
    diagnostic = {"scope": {"ref": "spec://run_record/s/snapshot-current",
                            "relation": "current"},
                  "record": {"succeeded": False, "stderr_excerpt": "transport failure"}}
    stale = {"action": "discard_hypothesis", "rationale": "Old probe is impossible.",
             "method_ref": "spec://method/s/old",
             "diagnostic_ref": "spec://run_record/s/snapshot-old"}
    repair = {"action": "change_plan", "rationale": "Repair the current execution.",
              "method_ref": current["method_ref"],
              "plan_diagnosis": {"component_id": "probe", "plan_quote": "Accepted-domain input"},
              "diagnostic_ref": diagnostic["scope"]["ref"]}
    llm = FakeLLM([json.dumps(stale), json.dumps(repair)])
    result = moss.decide("implementation_thrashing", None, guard, llm,
                         method_context=current, diagnostic=diagnostic)
    assert result["action"] == "change_plan"
    assert len(llm.calls) == 2
    assert result["method_ref"] == current["method_ref"]
    assert result["diagnostic_ref"] == diagnostic["scope"]["ref"]


@pytest.mark.parametrize("succeeded,metric", [(False, None), (True, None), (True, 16.0)])
def test_abandoning_a_run_without_evaluated_counterexample_is_not_falsification(
        succeeded, metric):
    handler = object.__new__(MemoryAgentHandler)
    prior = {"best_metric": 16.0, "best_result_ref": "saved-best",
             "negative_results": ["old-plan failure"]}
    before = copy.deepcopy(prior)
    result = handler._consolidate_failure(
        run_record={"succeeded": succeeded, "metric_value": metric,
                    "exit_code": 0 if succeeded else 1},
        baseline_result={"metric_value": 17.0},
        failure_context={"reason": "implementation_thrashing", "action": "discard_hypothesis"},
        prior_research_memory=prior)
    assert result["diagnosis"] == "component_execution_blocked"
    assert result["best_metric"] == 16.0
    assert result["best_result_ref"] == "saved-best"
    assert result["negative_results"][0] == "old-plan failure"
    assert prior == before


def test_same_method_ref_does_not_allow_quoting_a_superseded_component():
    current = {"method_ref": "spec://method/s/revision",
               "components": [{"component_id": "probe", "mechanism": "Use a 3000-op calibration timeline."}]}
    answer = {"action": "discard_hypothesis", "rationale": "The 3-op timeline cannot run.",
              "method_ref": current["method_ref"], "diagnostic_ref": None,
              "plan_diagnosis": {"component_id": "probe", "plan_quote": "Use a 3-op calibration timeline."}}
    repair = {"action": "shrink_patch", "rationale": "Repair the transport failure.",
              "method_ref": current["method_ref"], "diagnostic_ref": None}
    llm = FakeLLM([json.dumps(answer), json.dumps(repair)])
    result = moss.decide("implementation_thrashing", None,
                         MossActionGuard(load_moss_action_matrix()), llm,
                         method_context=current)
    assert result["action"] == "shrink_patch"
    assert len(llm.calls) == 2
    assert result["degraded"] is False


def test_plan_revision_keeps_a_measured_scalar_but_never_qualifies_the_ledger(tmp_path):
    from autoresearch.experiments.repo_improve.tests.test_component_plan import _component
    from autoresearch.experiments.repo_improve.component_plan import build_component_plan, ComponentLedger
    node = _component("summary", "indirect_gain")
    node["functional_check"] = "failed_check_count == 0"
    plan = build_component_plan({"approach": "Conclude from measurements", "components": [node]})
    ledger = ComponentLedger(str(tmp_path / "ledger.json"), plan)
    ledger.begin("summary")
    runner = object.__new__(RunnerAgentHandler)
    runner._active_component_context = {
        "s": {"mode": "component", "plan": plan, "ledger": ledger, "node": plan.node("summary")}}
    review = review_reply()
    report = [{"component": "summary", "actual": 0, "_host_bound": True,
               "evidence": {"source": "runtime_attempt", "ref": "attempt-current#/failed_check_count", "json_pointer": "/failed_check_count"}}]
    gen = {"method_metric": 16.0, "baseline_metric": 17.0, "verdict": "invalid",
           "accepted": False, "component_attempt_id": "attempt-current",
           "repo_path": str(tmp_path / "candidate"), "method_implementation_review": review,
           "ablation_report": report}
    result = runner._finalize_component_attempt("s", gen)
    assert result["accepted"] is False
    assert result["method_metric"] == 16.0
    assert ledger.entry("summary").checkpoint == ""
    assert result["component_execution"]["status"] == "contract_failed"
    assert result["component_execution"]["execution_succeeded"] is True


async def test_moss_handler_binds_actual_run_and_persists_immutable_decision(tmp_path):
    from autoresearch.agents.moss_agent.handler import MossAgentHandler
    from autoresearch.agents.moss_agent.tests.test_handler import FakeBusClient, _make_envelope
    from autoresearch.spec_store.identity import artifact_digest
    from autoresearch.experiments.repo_improve.component_plan import method_fingerprint
    with closing(SpecStore(tmp_path / "store")) as store:
        method = {"approach": "Revised calibration", "components": [
            {"component_id": "probe", "mechanism": "Use the verifier accepted input domain."}]}
        current = store.put(kind="method", session_id="s", key="new", obj=method, emitter="MethodAgent")
        run = {"succeeded": False, "stderr_excerpt": "TRANSPORT_FAILURE", "stdout_excerpt": "x" * 8000}
        run_ref = store.put(kind="run_record", session_id="s",
                            key="snapshot-" + artifact_digest(run).split(":")[1],
                            obj=run, emitter="RunnerAgent")
        gen = {"method_ref": current, "run_record_snapshot_ref": run_ref,
               "component_execution": {"method_fingerprint": method_fingerprint(method),
                                       "component_id": "probe", "status": "execution_failed"},
               "diagnosis": "execution_failure", "method_metric": None}
        store.put(kind="gen_experiment_result", session_id="s",
                  key="snapshot-" + artifact_digest(gen).split(":")[1], obj=gen, emitter="RunnerAgent")
        llm = FakeLLM([json.dumps({"action": "shrink_patch", "rationale": "Repair transport.",
                                  "method_ref": current, "diagnostic_ref": run_ref})])
        handler = MossAgentHandler(spec_store=store, llm_client=llm, config_root=tmp_path)
        bus = FakeBusClient()
        await handler.handle_intervention_requested(bus, _make_envelope(
            "moss.intervention_requested", "s",
            {"reason": "implementation_thrashing", "method_ref": current, "run_record_ref": run_ref},
            "ResearchOrchestrator"))
        event = bus.published[-1]
        decision = store.get(event.payload["moss_intervention_ref"])
        assert event.payload["degraded"] is False
        assert "/snapshot-" in event.payload["moss_intervention_ref"]
        assert decision["method_ref"] == current
        assert decision["diagnostic_ref"] == run_ref
        prompt = llm.calls[0][1]["content"]
        assert '"relation": "current"' in prompt
        assert run_ref in prompt and "execution_failure" in prompt
        store.put(kind="moss_intervention", session_id="s", key="implementation_thrashing-v1",
                  obj={"action": "discard_hypothesis"}, emitter="MOSS")
        assert store.get(event.payload["moss_intervention_ref"]) == decision


@pytest.mark.parametrize("case,expected", [
    ("evaluated_integration", "method_hypothesis_falsified"),
    ("old_run", "component_execution_blocked"),
    ("old_method", "component_execution_blocked"),
    ("component_probe", "component_execution_blocked"),
    ("plan_revision", "component_execution_blocked"),
    ("suspicious", "component_execution_blocked"),
    ("missing_digest", "component_execution_blocked"),
    ("improvement", "component_execution_blocked"),
])
def test_discard_requires_a_current_evaluated_whole_method_negative(case, expected):
    from autoresearch.spec_store.identity import artifact_digest
    run = {"succeeded": True, "metric_value": 18.0, "exit_code": 0}
    evaluation = {"verdict": "valid_not_best", "metric_value": 18.0,
                  "is_improvement": False, "suspicious": False,
                  "anti_cheat_flags": [], "run_record_digest": artifact_digest(run)}
    gen = {"method_ref": "method-current", "component_execution": {"mode": "integration"}}
    if case == "old_run":
        evaluation["run_record_digest"] = artifact_digest({**run, "metric_value": 19.0})
    elif case == "old_method":
        gen["method_ref"] = "method-old"
    elif case == "component_probe":
        gen["component_execution"]["mode"] = "component"
    elif case == "plan_revision":
        gen["method_implementation_review"] = review_reply()
    elif case == "suspicious":
        evaluation["anti_cheat_flags"] = ["invalid_data_flow"]
    elif case == "missing_digest":
        evaluation.pop("run_record_digest")
    elif case == "improvement":
        evaluation["is_improvement"] = True
    result = object.__new__(MemoryAgentHandler)._consolidate_failure(
        run_record=run, baseline_result={"metric_value": 17.0},
        failure_context={"action": "discard_hypothesis", "method_ref": "method-current"},
        prior_research_memory=None, evaluation_report=evaluation, gen_result=gen)
    assert result["diagnosis"] == expected
    assert result["best_metric"] is None
    assert result["verified_findings"] == []


def test_memory_retains_rejected_measurement_and_routes_grounded_plan_revision(tmp_path):
    with closing(SpecStore(tmp_path / "store")) as store:
        run_ref = store.put(kind="run_record", session_id="s", key="v1",
                            obj={"succeeded": True, "metric_value": 16.0}, emitter="RunnerAgent")
        review = review_reply()
        review["method_ref"] = "method-current"
        gen = {"verdict": "invalid", "accepted": False, "baseline_metric": 17.0,
               "method_metric": 16.0, "method_ref": "method-current",
               "method_implementation_review": review,
               "diagnosis": "component_expectation_failed",
               "diagnosis_evidence": ["The frozen conclusion contradicts the measured result."],
               "component_execution": {"mode": "component", "component_id": "summary",
                                       "status": "contract_failed", "execution_succeeded": True}}
        store.put(kind="gen_experiment_result", session_id="s", key="current", obj=gen, emitter="RunnerAgent")
        eval_ref = store.put(kind="evaluation_report", session_id="s", key="v1",
                             obj={"verdict": "invalid", "anti_cheat_flags": []}, emitter="EvaluatorAgent")
        handler = MemoryAgentHandler(spec_store=store, llm_client=FakeLLM([]))
        memory, _, diagnosis, evidence = handler._consolidate_gen(run_ref, eval_ref)
    assert diagnosis == "component_expectation_failed"
    assert "summary" in evidence
    assert memory["best_metric"] is None
    assert memory["verified_findings"] == []
    assert memory["component_history"][0]["metric"] == 16.0
    assert memory["component_history"][0]["status"] == "contract_failed"
