import json
from types import SimpleNamespace

import pytest

from autoresearch.agents.comparison_guardian_agent.core import audit_repository_comparison
from autoresearch.agents.runner_agent.handler import RunnerAgentHandler, _map_repo_result
from autoresearch.agents.memory_agent.handler import _classify_component_evidence
from autoresearch.spec_store import SpecStore, EmitterNotAllowed
from autoresearch.spec_store import refs


class Store:
    def __init__(self):
        self.saved = []

    def put(self, **kwargs):
        self.saved.append(kwargs)
        return "spec://audit/test/v1"


def result():
    return {"baseline": {"metric": 10.0, "valid": True},
            "candidate": {"metric": 9.0, "valid": True},
            "verdict": {"improved": True}, "metric": "latency", "metric_unit": "ms",
            "config": {"problem": "task-required"},
            "report": "baseline ran a substituted problem; candidate ran the required problem"}


def test_malformed_comparison_preserves_measurement_for_review_recovery(monkeypatch):
    from autoresearch.agents.comparison_guardian_agent.core import ComparisonGuardianParseError

    error = ComparisonGuardianParseError("invalid audit schema")
    error.responses = ['{"status":"pass"}']

    def review(*args, **kwargs):
        raise error

    monkeypatch.setattr(
        "autoresearch.agents.comparison_guardian_agent.core.audit_repository_comparison", review)
    store = Store()
    handler = RunnerAgentHandler(spec_store=store, backend=None, p1_llm=object())
    monkeypatch.setattr(handler, "_original_question", lambda _: "task")
    monkeypatch.setattr(handler, "_latest_method_ref", lambda _: "spec://method/s1/current")
    measured = {**result(), "checkpoint": "/immutable/candidate", "measurement_evidence": {
        "attempt_id": "attempt-real", "value": 9.0, "stdout_sha256": "a" * 64}}

    with pytest.raises(ComparisonGuardianParseError):
        handler._review_repository_comparison("s1", measured, "method", frozen_baseline={"metric_value": 10.0})

    captured = next(item["obj"] for item in store.saved if item["kind"] == "implementation_review_input")
    pending = next(item["obj"] for item in store.saved if item["kind"] == "pending_implementation_review")
    assert captured["result"] == measured
    assert captured["comparison_review_pending"] is True
    assert captured["invalid_review_responses"] == error.responses
    assert pending["status"] == "pending"


@pytest.mark.parametrize("audit_verdict", ["pass", "insufficient_evidence"])
def test_metricless_indirect_probe_receives_measurement_identity_audit(monkeypatch, audit_verdict):
    seen = {}
    audit = {"verdict": audit_verdict,
             "issues": [] if audit_verdict == "pass" else ["Probe repeats a formula without exercising the candidate."],
             "remediation": None if audit_verdict == "pass" else "Measure the actual candidate.",
             "reasoning": "Check the measured property, separately from its numeric threshold."}

    def review(*args, **kwargs):
        seen.update(result=args[2], scope=kwargs["component_scope"])
        return audit

    monkeypatch.setattr(
        "autoresearch.agents.comparison_guardian_agent.core.audit_repository_comparison", review)
    handler = RunnerAgentHandler(spec_store=Store(), backend=None, p1_llm=object())
    monkeypatch.setattr(handler, "_original_question", lambda _: "Optimize the official metric.")
    contract = {"component_id": "routing", "implementation_check": "Measure actual routed events."}
    handler._active_component_context = {"s1": {"mode": "component", "node": SimpleNamespace(
        component_id="routing", role="indirect_gain", depends_on=(),
        acceptance_check="routed_fraction > 0", to_dict=lambda: contract)}}
    rr = {**result(), "candidate": {"metric": None, "valid": False},
          "component_attempt_id": "attempt-probe", "ablation_report": [{
              "component": "routing", "type": "indirect_gain", "actual": 0.02,
              "expected": {"operator": ">", "value": 0}, "_host_bound": True,
              "evidence": {"source": "runtime_attempt", "ref": "attempt-probe"}}],
          "attempt_review_evidence": [{"attempt_id": "attempt-probe", "command_sources": [{
              "path": "/tmp/check.py", "source": "print({'routed_fraction': 0.02})"}]}]}

    out = handler._review_repository_comparison("s1", rr, "method")

    assert out["repository_comparison_review"] == audit
    assert seen["result"]["attempt_review_evidence"] == rr["attempt_review_evidence"]
    assert seen["scope"]["validation_contract"] == contract
    assert out["candidate"]["metric"] is None
    if audit_verdict != "pass":
        assert out["error"] and out["candidate"]["valid"] is False


@pytest.mark.parametrize("phase", ["baseline", "method"])
def test_repository_comparison_persists_in_real_store(tmp_path, monkeypatch, phase):
    audit = {"verdict": "pass", "issues": [], "remediation": None,
             "reasoning": "Same task and valid measurements."}
    monkeypatch.setattr(
        "autoresearch.agents.comparison_guardian_agent.core.audit_repository_comparison",
        lambda *args, **kwargs: audit)
    store = SpecStore(tmp_path / "store")
    try:
        handler = RunnerAgentHandler(spec_store=store, backend=None, p1_llm=object())
        monkeypatch.setattr(handler, "_original_question", lambda _: "task")
        out = handler._review_repository_comparison("s1", result(), phase)
        assert out["repository_comparison_review"] == audit
        ref = refs.encode("repository_comparison_review", "s1", "v1")
        assert store.get(ref) == audit
        with pytest.raises(EmitterNotAllowed):
            store.put(kind="repository_comparison_review", session_id="s1",
                      key="v1", obj=audit, emitter="CoderAgent")
        assert store.get(ref) == audit
    finally:
        store.close()


def test_method_comparison_uses_frozen_baseline_delta(monkeypatch):
    seen = {}
    def review(original, semantics, reviewed_result, llm, **kwargs):
        seen.update(reviewed_result)
        return {"verdict": "pass", "issues": [], "remediation": None,
                "reasoning": "candidate changes are in scope"}
    monkeypatch.setattr(
        "autoresearch.agents.comparison_guardian_agent.core.audit_repository_comparison",
        review)
    handler = RunnerAgentHandler(spec_store=Store(), backend=None, p1_llm=object())
    monkeypatch.setattr(handler, "_original_question", lambda _: "task")
    rr = {**result(), "diff": "baseline-generated cache plus method edit",
          "baseline_relative_diff": "method edit"}
    out = handler._review_repository_comparison("s1", rr, "method")
    assert seen["diff"] == "method edit"
    assert seen["diff_basis"] == "frozen_baseline_checkpoint"
    assert out["diff"] == rr["diff"]


@pytest.mark.parametrize("case", ["active", "inactive", "other_session", "candidate_claim", "baseline"])
def test_comparison_receives_only_current_session_active_user_closeout(tmp_path, monkeypatch, case):
    store = SpecStore(tmp_path / "followup-store")
    idea = "Revalidate this session's own trained checkpoint; preserve original budgets and official gates."
    if case != "candidate_claim":
        store.put(kind="human_idea_seed", session_id="other" if case == "other_session" else "s1",
                  key="closeout-v1", obj={"idea": idea, "closeout": case != "inactive"}, emitter="Human")
    seen = []
    def audit(question, *args, **kwargs):
        seen.append(question)
        return {"verdict": "pass", "issues": [], "remediation": None, "reasoning": "same task"}
    monkeypatch.setattr("autoresearch.agents.comparison_guardian_agent.core.audit_repository_comparison", audit)
    handler = RunnerAgentHandler(spec_store=store, backend=None, p1_llm=object())
    monkeypatch.setattr(handler, "_original_question", lambda _: "Use a fresh independent session.")
    rr = {**result(), "user_closeout": {"idea": idea, "closeout": True}}
    handler._review_repository_comparison("s1", rr, "baseline" if case == "baseline" else "method")
    assert (idea in seen[0]) is (case == "active")
    assert seen[0].startswith("Use a fresh independent session.")
    store.close()


@pytest.mark.parametrize("case", ["valid", "other_session", "tampered", "other_baseline",
                                  "unqualified", "candidate_claim", "baseline"])
def test_revalidation_review_receives_bound_same_session_source(tmp_path, monkeypatch, case):
    from autoresearch.spec_store.identity import artifact_digest

    store = SpecStore(tmp_path / "provenance-store")
    baseline = {"metric_value": 10.0}
    producer = "other" if case == "other_session" else "s1"
    source_method = f"spec://method/{producer}/original"
    measured = {"attempt_id": "attempt-original", "value": 8.0, "valid": True,
                "stdout_sha256": "a" * 64, "protocol_fingerprint": "b" * 64,
                "protocol": {"kind": "official_single"}}
    source = {"accepted": case != "unqualified", "checkpoint": "/immutable/own-training",
              "method_metric": 8.0, "method_ref": source_method,
              "baseline_digest": artifact_digest({"metric_value": 11.0} if case == "other_baseline" else baseline),
              "measurement_evidence": measured,
              "candidate_admissibility": {"verdict": "pass", "attempt_id": "attempt-original"},
              "repository_comparison_review": {"verdict": "pass"},
              "method_implementation_review": {"verdict": "faithful", "method_ref": source_method,
                                               "task_contract_review": {"verdict": "compliant"}},
              "attempt_review_evidence": [{"attempt_id": "attempt-original", "rc": 0,
                                            "command_sources": [{"source": "train(); evaluate()"}]}]}
    source_ref = store.put(kind="gen_experiment_result", session_id=producer,
                           key="snapshot-" + artifact_digest(source).split(":", 1)[1],
                           obj=source, emitter="RunnerAgent")
    if case == "tampered":
        store._db.execute("update artifacts set json=? where ref=?",
                          (json.dumps({**source, "checkpoint": "/changed"}), source_ref))
        store._db.commit()
    method = {} if case == "candidate_claim" else {"reconciled_result_ref": source_ref}
    class LLM:
        def chat(self, **kwargs):
            self.messages = kwargs["messages"]
            return SimpleNamespace(content=json.dumps({"verdict": "pass", "issues": [],
                "remediation": None, "reasoning": "Assess source provenance and current measurement separately."}))
    llm = LLM()
    handler = RunnerAgentHandler(spec_store=store, backend=None, p1_llm=llm)
    monkeypatch.setattr(handler, "_original_question", lambda _: "Independently produce this session's candidate.")
    monkeypatch.setattr(handler, "_latest_method", lambda _: method)
    rr = {**result(), "retained_source_provenance": {"session_id": "forged"}}
    out = handler._review_repository_comparison("s1", rr,
        "baseline" if case == "baseline" else "method", frozen_baseline=baseline)
    provenance = json.loads(llm.messages[1]["content"]).get("host_retained_source")
    if case == "valid":
        assert provenance["session_id"] == "s1"
        assert provenance["result_ref"] == source_ref
        assert provenance["measurement_evidence"] == measured
        assert provenance["attempt_review_evidence"] == source["attempt_review_evidence"]
        assert "not automatic acceptance" in provenance["scope"]
    else:
        assert provenance is None
    assert out["candidate"] == rr["candidate"]
    assert out["retained_source_provenance"] == rr["retained_source_provenance"]
    store.close()


@pytest.mark.parametrize("phase,verdict,diagnosis", [
    ("baseline", "baseline_invalid", "baseline_drift"),
    ("method", "baseline_invalid", "baseline_drift"),
    ("method", "comparison_changed", "candidate_contract_violation"),
    ("method", "insufficient_evidence", "candidate_contract_violation"),
])
def test_repository_comparison_preserves_numbers_but_blocks_invalid_comparison(
        monkeypatch, phase, verdict, diagnosis):
    audit = {"verdict": verdict, "issues": ["Recorded task identity is inconsistent."],
             "remediation": "Reproduce the required task.", "reasoning": "Compare actual records."}
    seen = {}

    def review(*args, **kwargs):
        seen.update(kwargs)
        return audit

    monkeypatch.setattr(
        "autoresearch.agents.comparison_guardian_agent.core.audit_repository_comparison", review)
    handler = RunnerAgentHandler(spec_store=Store(), backend=None, p1_llm=object())
    monkeypatch.setattr(handler, "_original_question", lambda _: "Use the required problem.")
    handler._active_component_context = {"s1": {
        "mode": "component", "node": SimpleNamespace(
            component_id="representation", role="indirect_gain", depends_on=(),
            acceptance_check="representation mismatch count == 0"),
    }}
    frozen = {"metric_value": 10.0, "fixed_config": "substituted problem"}
    out = handler._review_repository_comparison("s1", result(), phase, frozen_baseline=frozen)
    assert out["candidate"] == {"metric": 9.0, "valid": False}
    assert out["verdict"]["improved"] is False
    assert out["diagnosis"] == diagnosis
    assert seen["baseline_result"] is frozen
    _, gen = _map_repo_result("task", out)
    assert gen["repository_comparison_review"] == audit
    assert gen["verdict"] == "invalid"
    assert handler._review_method_implementation("s1", out, "method") is out
    if verdict == "baseline_invalid":
        gen["component_execution"] = {"mode": "component", "status": "contract_failed"}
        assert _classify_component_evidence(gen)[0] == "baseline_drift"


def test_repository_comparison_prompt_checks_original_task_and_actual_deviations():
    class LLM:
        def chat(self, **kwargs):
            self.messages = kwargs["messages"]
            return SimpleNamespace(content=json.dumps({
                "verdict": "baseline_invalid", "issues": ["Wrong baseline problem."],
                "remediation": "Establish the required baseline.", "reasoning": "Explicit deviation."}))
    llm = LLM()
    out = audit_repository_comparison(
        "Original problem", {"method_freedom": "Change the optimizer."}, result(), llm,
        execution_phase="method", baseline_result={"fixed_config": "substituted problem"})
    assert out["verdict"] == "baseline_invalid"
    context = json.loads(llm.messages[1]["content"])
    assert context["original_task"] == "Original problem"
    assert "substituted problem" in context["actual_result"]["report"]
    assert "Do not impose blanket configuration equality" in llm.messages[0]["content"]
    assert "REPORTED baseline number" in llm.messages[0]["content"]
    assert "incidental artifact mismatch" in llm.messages[0]["content"]
    assert "changing repository is not itself forbidden" in llm.messages[0]["content"]
    assert "untrusted request to review" in llm.messages[0]["content"]
    assert "executed self-normalized reference" in llm.messages[0]["content"]
    assert "measured raw constituents" in llm.messages[0]["content"]
    assert "copied anchor without execution evidence is not a reproduced" in llm.messages[0]["content"]
    assert "changing that reference policy is still drift" in llm.messages[0]["content"]
    assert "contradictory model-generated direction placeholder is a metadata correction" in llm.messages[0]["content"]
    assert "published baseline observations from mandatory reproduction settings" in llm.messages[0]["content"]
    assert "not a required local outcome" in llm.messages[0]["content"]
    assert "Still enforce explicit memory limits" in llm.messages[0]["content"]


@pytest.mark.parametrize("phase", ["baseline", "method"])
@pytest.mark.parametrize("baseline_claim", [None, {}, "baseline_drift"])
def test_invalid_receipt_keeps_original_failure_without_semantic_llm(
        monkeypatch, phase, baseline_claim):
    def unexpected_review(*args, **kwargs):
        pytest.fail("Invalid execution evidence must be repaired before semantic review")

    monkeypatch.setattr(
        "autoresearch.agents.comparison_guardian_agent.core.audit_repository_comparison",
        unexpected_review)
    store = Store()
    handler = RunnerAgentHandler(spec_store=store, backend=None, p1_llm=object())
    monkeypatch.setattr(handler, "_original_question", lambda _: "Measure the declared task.")
    rr = result()
    side = "baseline" if phase == "baseline" else "candidate"
    rr[side]["valid"] = False
    rr["comparison_audit"] = {"baseline_drift": baseline_claim}
    rr["error"] = "execution output is missing its declared measurement field"
    rr["verdict"] = {"verdict": "fail", "improved": False, "reason": rr["error"]}
    assert handler._review_repository_comparison("s1", rr, phase) is rr
    assert rr[side]["valid"] is False
    assert rr["error"] == "execution output is missing its declared measurement field"
    assert store.saved == []


def test_comparable_negative_result_is_not_rejected(monkeypatch):
    monkeypatch.setattr(
        "autoresearch.agents.comparison_guardian_agent.core.audit_repository_comparison",
        lambda *args, **kwargs: {"verdict": "pass", "issues": [], "remediation": None,
                                "reasoning": "Permitted changes, same task."})
    handler = RunnerAgentHandler(spec_store=Store(), backend=None, p1_llm=object())
    monkeypatch.setattr(handler, "_original_question", lambda _: "task")
    rr = result()
    rr["candidate"]["metric"] = 11.0
    rr["verdict"]["improved"] = False
    out = handler._review_repository_comparison("s1", rr, "method")
    assert out["candidate"] == {"metric": 11.0, "valid": True}
    assert out["verdict"]["improved"] is False


@pytest.mark.parametrize("audit_verdict", ["baseline_invalid", "pass"])
@pytest.mark.parametrize("transport", ["local", "remote"])
def test_explicit_baseline_problem_is_reviewed_without_candidate_score(
        tmp_path, monkeypatch, audit_verdict, transport):
    from autoresearch.experiments.repo_improve.agentic_improve import _build_result_from_file
    from autoresearch.agents.runner_agent.handler import _res_to_improve_json, _parse_agentic_result

    claim = {"confirmed": True, "frozen_baseline_environment": {"device": "CPU"},
             "repaired_environment": {"device": "GPU"}}
    path = tmp_path / "ar_result.json"
    path.write_text(json.dumps({
        "baseline": 10, "method": None, "candidate_valid": False,
        "metric": "quality", "baseline_drift": claim,
        "comparison_audit": {"data": "unchanged"},
    }))
    parsed = _build_result_from_file(str(path), "generic", "quality", False,
                                     lambda *args: None, execution_phase="method")
    assert parsed["comparison_audit"]["baseline_drift"] == claim
    assert parsed["comparison_audit"]["data"] == "unchanged"
    seen = []
    audit = {"verdict": audit_verdict, "issues": ["Required device differs."]
             if audit_verdict == "baseline_invalid" else [],
             "remediation": "Reproduce baseline in the required environment."
             if audit_verdict == "baseline_invalid" else None,
             "reasoning": "Checked task requirements and frozen evidence, not just the claim."}

    def review(*args, **kwargs):
        seen.append((args, kwargs))
        return audit

    monkeypatch.setattr(
        "autoresearch.agents.comparison_guardian_agent.core.audit_repository_comparison", review)
    handler = RunnerAgentHandler(spec_store=Store(), backend=None, p1_llm=object())
    monkeypatch.setattr(handler, "_original_question", lambda _: "Use the assigned accelerator.")
    rr = _res_to_improve_json(parsed)
    if transport == "remote":
        rr = _parse_agentic_result("RESULT_JSON " + json.dumps(rr), 0)
    assert rr["comparison_audit"] == parsed["comparison_audit"]
    rr.update(diagnosis="execution_failure",
              error="no candidate score", verdict={"verdict": "fail", "improved": False})
    rr["candidate"] = {"metric": None, "valid": False}
    frozen = {"metric_value": 10, "fixed_config": "CPU"}
    out = handler._review_repository_comparison("s1", rr, "method", frozen_baseline=frozen)
    assert len(seen) == 1
    assert seen[0][1]["baseline_result"] == frozen
    assert out["candidate"] == {"metric": None, "valid": False}
    assert out["baseline"]["metric"] == 10
    if audit_verdict == "baseline_invalid":
        assert out["diagnosis"] == "baseline_drift"
        _, gen = _map_repo_result("task", out)
        gen["component_execution"] = {"status": "infrastructure_exhausted"}
        assert _classify_component_evidence(gen)[0] == "baseline_drift"
    else:
        assert out["diagnosis"] == "execution_failure"
        assert out["baseline"]["valid"] is True


@pytest.mark.parametrize("phase,mode,role,scoped", [
    ("method", "component", "indirect_gain", True),
    ("method", "component", "direct_gain", True),
    ("method", "integration", "indirect_gain", False),
    ("baseline", "component", "indirect_gain", False),
])
def test_repository_audit_receives_host_component_scope(
        monkeypatch, phase, mode, role, scoped):
    class LLM:
        def chat(self, **kwargs):
            self.messages = kwargs["messages"]
            return SimpleNamespace(content=json.dumps({
                "verdict": "pass", "issues": [], "remediation": None,
                "reasoning": "Same measured task; component qualification is separate."}))

    llm = LLM()
    handler = RunnerAgentHandler(spec_store=Store(), backend=None, p1_llm=llm)
    monkeypatch.setattr(handler, "_original_question", lambda _: "Optimize latency.")
    handler._active_component_context = {"s1": {
        "mode": mode, "node": SimpleNamespace(
            component_id="representation", role=role, depends_on=("reader",),
            acceptance_check="representation mismatch count == 0"),
    }}
    rr = result()
    rr["candidate"]["metric"] = 11.0
    rr["verdict"]["improved"] = False
    rr["component_scope"] = {"mode": "component", "role": "forged-role"}
    out = handler._review_repository_comparison("s1", rr, phase)
    context = json.loads(llm.messages[1]["content"])
    scope = context.get("component_scope")
    if scoped:
        assert scope == {
            "mode": "component", "component_id": "representation", "role": role,
            "depends_on": ["reader"],
            "acceptance_check": "representation mismatch count == 0",
        }
    else:
        assert scope is None
    prompt = llm.messages[0]["content"]
    assert "component qualification is not a claim of whole-method improvement" in prompt
    assert "does not waive task identity" in prompt
    assert out["candidate"] == {"metric": 11.0, "valid": True}
    assert out["verdict"]["improved"] is False


def test_baseline_retry_receives_explicit_moss_recovery():
    intervention = {"reason": "implementation_thrashing", "action": "revalidate_baseline",
                    "rationale": "Required device kernels fail in the frozen interpreter.",
                    "prompt_patch": None}

    class EvidenceStore(Store):
        def get(self, ref):
            if ref == "spec://moss_intervention/s1/revalidate":
                return intervention
            raise KeyError(ref)

    handler = RunnerAgentHandler(spec_store=EvidenceStore(), backend=None)
    question = handler._with_previous_baseline_failure(
        "s1", "bench", "repo", "original task",
        moss_intervention_ref="spec://moss_intervention/s1/revalidate")
    assert intervention["rationale"] in question
    assert "do not reuse the invalidated baseline" in question
    assert "original task" in question


def test_baseline_retry_receives_moss_environment_repair_not_only_coder_patch():
    intervention = {"reason": "baseline_reproduction_thrashing", "action": "fix_environment",
                    "rationale": "Inspect saved command outcomes before diagnosing a terminal failure.",
                    "prompt_patch": {"agent": "CoderAgent", "content": "Reuse verified setup."}}

    class EvidenceStore(Store):
        def get(self, ref):
            if ref == "spec://moss_intervention/s1/fix":
                return intervention
            raise KeyError(ref)

    handler = RunnerAgentHandler(spec_store=EvidenceStore(), backend=None)
    question = handler._with_previous_baseline_failure(
        "s1", "bench", "repo", "original task",
        moss_intervention_ref="spec://moss_intervention/s1/fix")
    assert intervention["rationale"] in question
    assert intervention["prompt_patch"]["content"] in question
    assert "verify the proposed diagnosis against actual tool outputs" in question
    assert "original task" in question


def test_baseline_retry_receives_the_actual_identity_failure():
    audit = {"verdict": "baseline_invalid", "issues": ["substituted problem"],
             "remediation": "Use the required problem", "reasoning": "Recorded deviation"}

    class EvidenceStore(Store):
        def get(self, ref):
            if "repository_comparison_review" in ref:
                return audit
            raise KeyError(ref)

    handler = RunnerAgentHandler(spec_store=EvidenceStore(), backend=None)
    question = handler._with_previous_baseline_failure("s1", "bench", "repo", "original task")
    assert "substituted problem" in question
    assert "Use the required problem" in question
    assert "Do not change the task to match the old baseline" in question
    assert "the review is not a new user requirement" in question
    assert "verify this diagnosis against the original task and repository sources" in question
    assert "separate raw objective from reward" in question


@pytest.mark.parametrize("phase", ["baseline", "method"])
def test_comparison_keeps_host_boundary_separate_from_candidate(monkeypatch, phase):
    seen = {}
    def review(*args, **kwargs):
        seen.update(kwargs)
        return {"verdict": "pass", "issues": [], "remediation": None, "reasoning": "Local boundary."}
    monkeypatch.setattr("autoresearch.agents.comparison_guardian_agent.core.audit_repository_comparison", review)
    handler = RunnerAgentHandler(spec_store=Store(), backend=None, p1_llm=object())
    monkeypatch.setattr(handler, "_original_question", lambda _: "Run the task.")
    boundary = {"accelerator_type": "H800", "conflicts_with_official": True}
    rr = {**result(), "confirmed_boundary": {"accelerator_type": "other"}}
    handler._review_repository_comparison("s1", rr, phase, confirmed_boundary=boundary)
    assert seen["confirmed_boundary"] == boundary
