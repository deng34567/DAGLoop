"""meeting ②: a method-SWAP regen (payload method_not_working=True) must instruct the MethodAgent
to propose a FUNDAMENTALLY DIFFERENT method (the prior one was falsified experimentally)."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from autoresearch.agents.method_agent.handler import MethodAgentHandler
from autoresearch.eventbus.broker.registry import Registry
from autoresearch.experiments.repo_improve.search_memory import (
    ExperienceCard, board_for_session)
from autoresearch.spec_store import refs


class _Store:
    def __init__(self, objs):
        self._objs = objs
        self.puts = []

    def get(self, ref):
        if ref not in self._objs:
            raise KeyError(ref)
        return self._objs[ref]

    def put(self, *, kind, session_id, key, obj, emitter):
        self.puts.append((kind, obj))
        return f"{kind}://{session_id}/{key}"

    def list_refs(self, session_id, kind):
        prefix = f"{kind}://{session_id}/"
        return [ref for ref in self._objs if ref.startswith(prefix)]


def test_reconciliation_selection_rechecks_explicit_task_import_rule():
    contract = {"sources": [{"path": "instruction.md", "content":
        "Allowed imports: `numpy`, `torch`. No other libraries."}]}
    store = _Store({
        "result://legal": {
            "repository_contract": contract,
            "method_implementation_review": {
                "task_contract_review": {"verdict": "compliant"}},
            "off_method_observation": {"valid": True, "improved": True},
            "diff": "diff --git a/solve.py b/solve.py\n+++ b/solve.py\n@@\n+import torch\n",
        },
        "result://illegal": {
            "repository_contract": contract,
            "method_implementation_review": {
                "task_contract_review": {"verdict": "compliant"}},
            "off_method_observation": {"valid": True, "improved": True},
            "diff": "diff --git a/solve.py b/solve.py\n+++ b/solve.py\n@@\n+import sys\n",
        },
    })
    handler = MethodAgentHandler(spec_store=store, llm_client=_LLM())
    memory = {"reconciliation_candidates": [
        {"result_ref": "result://legal"}, {"result_ref": "result://illegal"}]}

    assert handler._latest_admissible_reconciliation_candidate(memory) == {
        "result_ref": "result://legal"}
    assert handler._latest_admissible_reconciliation_candidate({
        "reconciliation_candidates": [{"result_ref": "result://illegal"}]}) is None


def test_task_contract_revision_can_change_illegal_core(monkeypatch):
    method_ref = "method://s1/current"
    memory_ref = "research_memory://s1/current"
    llm = _LLM(["legal static scorer"])
    store = _Store({
        "problem_model://s1": {"research_goal": "improve the official metric"},
        method_ref: {"approach": "successor-state search", "changes_outline": [],
                     "why_it_should_win": "", "assumptions": []},
        memory_ref: {"diagnosis": "candidate_contract_violation",
                     "reconciliation_candidates": []},
    })
    handler = MethodAgentHandler(spec_store=store, llm_client=llm)
    monkeypatch.setattr(handler, "_search_history", lambda _sid: "")

    asyncio.run(handler.handle_revise_requested(_Client(), _Env({
        "problem_model_ref": "problem_model://s1", "method_ref": method_ref,
        "research_memory_ref": memory_ref,
        "experiment_feedback": "Task rule prohibits successor-state search.",
    })))

    assert "change to the core mechanism" in llm.last_user
    assert "Preserve legal, qualified components" in llm.last_user
    assert "retain legal qualified component evidence" in llm.last_user
    assert "previously unsuccessful method family" not in llm.last_user
    assert "Reconcile only the implicated component" not in llm.last_user
    assert "Match declared output types literally" in llm.last_system
    assert "Keep execution time limits in the experiment runner" in llm.last_system


class _LLM:
    def __init__(self, approaches=None):
        self.last_user = ""
        self.last_system = ""
        self.calls = 0
        self.approaches = list(approaches or ["x"])

    def chat(self, *, messages, **_):
        self.calls += 1
        self.last_user = " ".join(m.get("content", "") for m in messages)
        self.last_system = messages[0].get("content", "")
        approach = self.approaches[min(self.calls - 1, len(self.approaches) - 1)]
        return type("R", (), {"content": json.dumps(
            {"approach": approach, "changes_outline": [], "why_it_should_win": "", "assumptions": []})})()


class _Client:
    def __init__(self):
        self.published = []

    async def publish(self, **kw):
        self.published.append(kw)


class _Env:
    def __init__(self, payload):
        self.session_id = "s1"
        self.event_id = "e1"
        self.payload = payload


@pytest.mark.parametrize("entry", ["revision", "deliberation"])
def test_method_entry_supplies_referenced_interface_source_to_review(tmp_path, monkeypatch, entry):
    import subprocess
    from autoresearch.experiments.repo_improve.agentic_improve import _repository_task_contract

    root = tmp_path / "repo"
    runtime = root / "runtime"
    runtime.mkdir(parents=True)
    (root / "instruction.md").write_text("Keep the public saved-model interface.")
    source = "def load_model(path):\n    return {'weights': path}\n"
    (runtime / "public_api.py").write_text(source)
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "add", "."], check=True)
    subprocess.run(["git", "-C", str(root), "-c", "user.name=Test", "-c",
                    "user.email=test@example.invalid", "commit", "-qm", "initial"], check=True)
    commit = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    baseline = {"repo_commit": commit, "metric_value": 1.0,
                "artifacts": [str(root / "result.json")],
                "repository_contract": _repository_task_contract(str(root), "Optimize the model")}
    before = json.dumps(baseline, sort_keys=True)
    prior = {"approach": "saved-model treatment", "components": [], "reference_sources": [
        {"url": "https://example.invalid/owner/repo/blob/main/runtime/public_api.py"}]}
    store = _Store({"problem_model://s1": {"key_problems": ["Improve the official metric"]},
                    "method://s1": prior, "baseline://s1": baseline})
    seen = []

    def revise(*args, **kwargs):
        seen.append(kwargs["feedback"])
        return {**prior, "approach": "revised saved-model treatment"}

    def deliberate(*args, **kwargs):
        seen.append(json.dumps(kwargs["baseline_result"]))
        raise RuntimeError("insufficient_user_quota")

    monkeypatch.setattr("autoresearch.agents.method_agent.handler.method_core.method_step", revise)
    monkeypatch.setattr(
        "autoresearch.agents.method_agent.handler.deliberation_core.deliberate_method", deliberate)
    handler = MethodAgentHandler(spec_store=store, llm_client=_LLM())
    monkeypatch.setattr(handler, "_search_history", lambda _sid: "")
    payload = {"problem_model_ref": "problem_model://s1", "method_ref": "method://s1",
               "baseline_result_ref": "baseline://s1", "experiment_feedback": "Repair the interface."}
    call = handler.handle_revise_requested if entry == "revision" else handler.handle_deliberation_requested
    asyncio.run(call(_Client(), _Env(payload)))

    assert seen and "runtime/public_api.py" in seen[0] and "def load_model(path):" in seen[0]
    assert json.dumps(store.get("baseline://s1"), sort_keys=True) == before


def _run(payload, llm, *, extra_objs=None):
    objs = {"problem_model://s1": {"research_goal": "lower val loss"}}
    objs.update(extra_objs or {})
    store = _Store(objs)
    handler = MethodAgentHandler(spec_store=store, llm_client=llm)
    client = _Client()
    asyncio.run(handler.handle_generation_requested(client, _Env(payload)))
    return client


@pytest.mark.parametrize("closeout", [False, True])
def test_explicit_closeout_survives_ordinary_method_revision(closeout):
    llm = _LLM(["retained implementation described accurately"])
    store = _Store({
        "problem_model://s1": {"research_goal": "improve the official metric"},
        "method://s1/prior": {"approach": "unqualified extension"},
        "idea://s1": {"idea": "Keep the permitted implementation boundary."},
        refs.encode("human_idea_seed", "s1", "closeout-v1"): {
            "closeout": closeout, "idea": "Finish the best measured implementation."},
    })
    handler = MethodAgentHandler(spec_store=store, llm_client=llm)
    asyncio.run(handler.handle_revise_requested(_Client(), _Env({
        "problem_model_ref": "problem_model://s1",
        "method_ref": "method://s1/prior", "human_idea_seed_ref": "idea://s1",
        "experiment_feedback": "The latest optional component failed.",
    })))
    assert "Keep the permitted implementation boundary." in llm.last_user
    assert ("No further exploratory search is authorized" in llm.last_user) == closeout
    assert ("Finish the best measured implementation." in llm.last_user) == closeout


@pytest.mark.parametrize("closeout", [False, True])
@pytest.mark.parametrize("component_plan_audit", [False, True])
def test_explicit_closeout_survives_method_deliberation(monkeypatch, closeout, component_plan_audit):
    seen = []
    def debate(*args, **kwargs):
        seen.append(args[2])
        raise RuntimeError("insufficient_user_quota")
    monkeypatch.setattr(
        "autoresearch.agents.method_agent.handler.deliberation_core.deliberate_method", debate)
    store = _Store({
        "method://s1": {"approach": "retained implementation", "components": []},
        "idea://s1": {"idea": "Keep the permitted implementation boundary."},
        refs.encode("human_idea_seed", "s1", "closeout-v1"): {
            "closeout": closeout, "idea": "Finish the best measured implementation."},
    })
    client = _Client()
    handler = MethodAgentHandler(spec_store=store, llm_client=_LLM())
    asyncio.run(handler.handle_deliberation_requested(client, _Env({
        "method_ref": "method://s1", "human_idea_seed_ref": "idea://s1",
        "component_plan_audit": component_plan_audit,
    })))
    assert "Keep the permitted implementation boundary." in seen[0]
    assert ("No further exploratory search is authorized" in seen[0]) == closeout
    assert ("Finish the best measured implementation." in seen[0]) == closeout
    assert [x["event_type"] for x in client.published] == ["agent.failed"]


def test_deliberation_quota_pauses_instead_of_accepting_draft_or_rejecting_cap(monkeypatch):
    def no_quota(*args, **kwargs):
        raise RuntimeError("insufficient_user_quota")
    monkeypatch.setattr(
        "autoresearch.agents.method_agent.handler.deliberation_core.deliberate_method", no_quota)
    for cap in (False, True):
        store = _Store({"method://s1": {"approach": "original", "components": []}})
        client = _Client()
        h = MethodAgentHandler(spec_store=store, llm_client=_LLM())
        asyncio.run(h.handle_deliberation_requested(client, _Env({
            "method_ref": "method://s1", "component_plan_audit": cap})))
        assert not store.puts
        assert [x['event_type'] for x in client.published] == ['agent.failed']
        assert client.published[0]['payload']['underlying_reason_code'] == 'insufficient_quota'


@pytest.mark.parametrize('draft_run', [None, {'succeeded': True, 'exit_code': 0,
                                           'metric_value': 0.16},
                                      {'succeeded': False, 'exit_code': 1,
                                       'metric_value': None}])
def test_revision_proposal_is_not_a_run_of_the_new_draft(monkeypatch, draft_run):
    seen = []
    draft = {'approach': 'use larger blocks', 'components': []}
    def debate(*args, **kwargs):
        seen.append(kwargs['prior_failure_context'])
        return SimpleNamespace(method=draft, report={'accepted': True})
    monkeypatch.setattr(
        'autoresearch.agents.method_agent.handler.deliberation_core.deliberate_method', debate)
    objects = {
        'method://s1/draft': draft,
        'run_record://s1/old': {'experiment_id': 'method://s1/old',
                              'succeeded': True, 'exit_code': 0, 'metric_value': 0.14},
        'result://s1/old': {'method_metric': 0.14,
                          'run_record_snapshot_ref': 'run_record://s1/old',
                          'measurement_evidence': {'valid': True}},
        'review://s1': {'trigger': 'repeated_verified_no_gain',
                       'result_ref': 'result://s1/old',
                       'decision': {'modify': ['Increase block size from 64 to 128.'],
                                    'evidence_refs': ['result://s1/old']}},
    }
    if draft_run is not None:
        objects['run_record://s1/draft'] = {
            **draft_run, 'experiment_id': 'method://s1/draft'}
    handler = MethodAgentHandler(spec_store=_Store(objects), llm_client=_LLM())
    asyncio.run(handler.handle_deliberation_requested(_Client(), _Env({
        'method_ref': 'method://s1/draft', 'method_feedback_ref': 'review://s1'})))

    evidence = seen[0]['method_revision_evidence']
    assert evidence['evaluated_result']['method_ref'] == 'method://s1/old'
    assert evidence['evaluated_result']['method_metric'] == 0.14
    assert evidence['modify'] == ['Increase block size from 64 to 128.']
    draft_evidence = evidence['draft_method_evaluation']
    assert draft_evidence['method_ref'] == 'method://s1/draft'
    assert draft_evidence['has_bound_run_record'] is (draft_run is not None)
    if draft_run is None:
        assert draft_evidence['runs'] == []
    else:
        assert draft_evidence['runs'] == [{'run_ref': 'run_record://s1/draft', **draft_run}]


def test_method_debate_receives_bound_revision_evidence_without_full_diff(monkeypatch):
    seen = []
    draft = {"approach": "tune the scorer", "components": []}
    def debate(*args, **kwargs):
        seen.append(kwargs.get("prior_failure_context"))
        return SimpleNamespace(method=draft, report={"accepted": True})
    monkeypatch.setattr(
        "autoresearch.agents.method_agent.handler.deliberation_core.deliberate_method", debate)
    store = _Store({
        "method://s1": draft,
        "review://s1": {
            "trigger": "legal_off_method_gain", "result_ref": "result://bound",
            "decision": {"primary_hypothesis": "The measured scorer passed.",
                         "evidence_refs": ["result://bound"],
                         "retain": ["Keep the measured mechanism."],
                         "modify": ["Correct the reviewer premise."]},
            "off_method_evidence": {
                "candidate": {"result_ref": "result://bound", "official_reward": 0.83,
                              "diff": "large diff must stay in the bound artifact"},
                "qualified_alternative": {"result_ref": "result://qualified",
                                          "official_reward": 0.93}},
        },
        "result://qualified": {
            "metric_name": "cost", "method_metric": 12,
            "measurement_evidence": {"valid": True},
            "candidate_admissibility": {"verdict": "pass"},
            "repository_comparison_review": {
                "reasoning": "The same official evaluator reported reward 0.93."},
        },
    })
    handler = MethodAgentHandler(spec_store=store, llm_client=_LLM())
    asyncio.run(handler.handle_deliberation_requested(_Client(), _Env({
        "method_ref": "method://s1", "method_feedback_ref": "review://s1"})))

    evidence = seen[0]["method_revision_evidence"]
    assert evidence["result_ref"] == "result://bound"
    assert "official_reward" not in evidence["candidate"]
    assert evidence["qualified_alternative"]["result_ref"] == "result://qualified"
    assert "official_reward" not in evidence["qualified_alternative"]
    assert evidence["qualified_alternative"]["method_metric"] == 12
    assert evidence["qualified_alternative"]["measurement_valid"] is True
    assert "reward 0.93" not in json.dumps(evidence)
    assert "diff" not in evidence["candidate"]


def test_method_not_working_instructs_a_different_method():
    llm = _LLM()
    client = _run({"problem_model_ref": "problem_model://s1", "method_not_working": True}, llm)
    assert "FUNDAMENTALLY DIFFERENT" in llm.last_user
    assert any(p.get("event_type") == "method.generated" for p in client.published)


def test_method_swap_includes_accumulated_research_memory():
    prior_ref = "method://s1/prior"
    memory_ref = "research_memory://s1/v1"
    prior = {"approach": "latest failed mechanism", "changes_outline": [],
             "why_it_should_win": "", "assumptions": []}
    memory = {
        "negative_results": [
            "Earlier fused optimizer family reached 196.721s and did not beat baseline 196.682s."
        ],
        "diagnosis": "method_hypothesis_falsified",
    }
    llm = _LLM(["genuinely different mechanism"])

    _run(
        {"problem_model_ref": "problem_model://s1", "method_not_working": True,
         "prior_method_ref": prior_ref, "research_memory_ref": memory_ref},
        llm,
        extra_objs={prior_ref: prior, memory_ref: memory},
    )

    assert "Earlier fused optimizer family" in llm.last_user
    assert "196.721" in llm.last_user


def test_method_swap_surfaces_bound_off_method_evidence_before_memory():
    prior_ref = "method://s1/prior"
    memory_ref = "research_memory://s1/v1"
    result_ref = "gen_experiment_result://s1/off-method"
    run_ref = "run_record://s1/off-method"
    llm = _LLM(["evidence-guided method"])
    _run(
        {"problem_model_ref": "problem_model://s1", "method_not_working": True,
         "prior_method_ref": prior_ref, "research_memory_ref": memory_ref},
        llm,
        extra_objs={
            prior_ref: {"approach": "shared scorer", "changes_outline": [],
                        "why_it_should_win": "", "assumptions": []},
            memory_ref: {"diagnosis": "component_execution_blocked",
                         "reconciliation_candidates": [{"result_ref": result_ref}]},
            result_ref: {"method_metric": 913,
                         "off_method_observation": {"valid": True, "improved": True},
                         "changes_summary": "different training pipeline cleared the gate",
                         "rationale": "The measured scorer reached hidden accuracy 0.958 and passed the quality gate.",
                         "diff": "+train_with_validated_schedule()",
                         "component_execution": {"diff_ref": "/checkpoint/off#git-diff"},
                         "run_record_snapshot_ref": run_ref,
                         "method_implementation_review": {"issues": ["premature component"]}},
            run_ref: {"stdout_excerpt": "hidden accuracy 0.958"},
        },
    )
    assert "implementation or training failure" in llm.last_user
    assert "different training pipeline cleared the gate" in llm.last_user
    assert "The measured scorer reached hidden accuracy 0.958" in llm.last_user
    assert "/checkpoint/off#git-diff" in llm.last_user
    assert "hidden accuracy 0.958" in llm.last_user
    assert llm.last_user.index("Bound result comparison") < llm.last_user.index(
        "Accumulated ResearchMemory")


def test_method_swap_includes_newer_moss_diagnosis_without_overriding_contract():
    ref = "moss_intervention://s1/v1"
    llm = _LLM(["new mechanism"])
    _run(
        {"problem_model_ref": "problem_model://s1", "method_not_working": True,
         "moss_intervention_ref": ref},
        llm,
        extra_objs={ref: {
            "action": "back_to_hypothesis",
            "rationale": "The previous treatment preserves the failed quality constraint.",
            "prompt_patch": {"content": "Address the measured failure."},
        }},
    )
    assert "FUNDAMENTALLY DIFFERENT" in llm.last_user
    assert "preserves the failed quality constraint" in llm.last_user
    assert "Address the measured failure" in llm.last_user
    assert "not permission to change the frozen" in llm.last_user


def test_method_debate_receives_prior_moss_failure_evidence(monkeypatch):
    seen = []
    draft = {"approach": "new mechanism", "components": []}
    def debate(*args, **kwargs):
        seen.append(kwargs.get("prior_failure_context"))
        return SimpleNamespace(method=draft, report={"accepted": True})
    monkeypatch.setattr(
        "autoresearch.agents.method_agent.handler.deliberation_core.deliberate_method", debate)
    store = _Store({
        "method://s1/new": draft,
        "moss_intervention://s1/failed": {
            "reason": "implementation_thrashing", "action": "discard_hypothesis",
            "rationale": "The measured workload is decode-bound, so prefill scheduling did not help.",
            "plan_diagnosis": {"component_id": "prefill"},
            "diagnostic_ref": "run_record://s1/failed",
        },
        "run_record://s1/failed": {"metric_value": 1.0051},
    })
    handler = MethodAgentHandler(spec_store=store, llm_client=_LLM())
    asyncio.run(handler.handle_deliberation_requested(_Client(), _Env({
        "method_ref": "method://s1/new",
        "moss_intervention_ref": "moss_intervention://s1/failed",
    })))
    assert seen[0]["action"] == "discard_hypothesis"
    assert seen[0]["component"] == "prefill"
    assert seen[0]["measured_metric"] == 1.0051
    assert "decode-bound" in seen[0]["rationale"]


def test_initial_generation_does_not_use_stale_moss_replacement_advice():
    ref = "moss_intervention://s1/old"
    llm = _LLM()
    _run(
        {"problem_model_ref": "problem_model://s1", "moss_intervention_ref": ref},
        llm, extra_objs={ref: {"rationale": "stale replacement advice"}},
    )
    assert "stale replacement advice" not in llm.last_user


def test_method_swap_includes_frozen_baseline_runtime_context():
    baseline_ref = "baseline_result://s1/v1"
    baseline = {
        "metric_name": "pure_train_time_seconds",
        "metric_value": 196.682,
        "fixed_config": "Muon + AdamW hybrid; bf16; manual all-reduce",
        "repo_commit": "abc123",
        "log_excerpt": "measured optimizer path: Muon + AdamW",
    }
    llm = _LLM(["source-grounded different mechanism"])

    _run(
        {"problem_model_ref": "problem_model://s1",
         "method_not_working": True,
         "baseline_result_ref": baseline_ref},
        llm,
        extra_objs={baseline_ref: baseline},
    )

    assert "Authoritative frozen BaselineResult" in llm.last_user
    assert "196.682" in llm.last_user
    assert "Muon + AdamW hybrid" in llm.last_user
    assert "measured optimizer path: Muon + AdamW" in llm.last_user
    assert "Do not assume a generic baseline" in llm.last_user


def test_initial_method_uses_bounded_real_baseline_context():
    baseline_ref = "baseline_result://s1/initial"
    llm = _LLM(["a measured-input optimization"])
    _run(
        {"problem_model_ref": "problem_model://s1", "baseline_result_ref": baseline_ref},
        llm,
        extra_objs={baseline_ref: {
            "metric_name": "runtime_ms", "metric_value": 12.5,
            "environment": "cpu", "runtime_seconds": 3.0,
            "log_excerpt": "observed run output: " + "x" * 5000,
        }},
    )
    assert "Authoritative frozen BaselineResult" in llm.last_user
    assert '"metric_value": 12.5' in llm.last_user
    assert '"environment": "cpu"' in llm.last_user
    assert "observed run output" in llm.last_user
    assert "x" * 4100 not in llm.last_user
    assert "prior method's experiments" not in llm.last_user.lower()
    assert "baseline observations are not prior Method experiments" in llm.last_user
    assert "not a mandatory algorithmic template" in llm.last_system
    assert "not required to start from the starter algorithm" in llm.last_system
    assert "complete replacement, or local improvement is allowed" in llm.last_system


def test_revision_keeps_explicit_prior_method_feedback_without_mislabeling_baseline():
    prior_ref = "method://s1/prior"
    llm = _LLM(["revised mechanism"])
    store = _Store({"problem_model://s1": {"research_goal": "lower val loss"},
                    prior_ref: {"approach": "old mechanism"},
                    "baseline_result://s1/v1": {"metric_value": 1.0}})
    handler = MethodAgentHandler(spec_store=store, llm_client=llm)
    asyncio.run(handler.handle_revise_requested(_Client(), _Env({
        "problem_model_ref": "problem_model://s1", "prior_method_ref": prior_ref,
        "baseline_result_ref": "baseline_result://s1/v1",
        "experiment_feedback": "The measured candidate missed the quality gate.",
    })))
    assert "Authoritative frozen BaselineResult" in llm.last_user
    assert "The measured candidate missed the quality gate" in llm.last_user
    assert "Prior method JSON" in llm.last_user
    assert "prior method's experiments" not in llm.last_user.lower()


def test_method_generation_preserves_frozen_comparison_semantics():
    interpretation_ref = "spec_interpretation://s1/v1"
    semantics = {
        "comparison_statement": (
            "Give baseline and method 300 seconds of pure training and compare final val_bpb."),
        "primary_metric": {"name": "val_bpb", "direction": "minimize"},
        "comparison_mode": "quality_at_fixed_time",
        "budget_interpretation": "steps and tokens are observed outcomes",
    }
    llm = _LLM(["length curriculum"])

    _run(
        {"problem_model_ref": "problem_model://s1",
         "spec_interpretation_ref": interpretation_ref},
        llm,
        extra_objs={interpretation_ref: {"evaluation_semantics": semantics}},
    )

    assert "Authoritative frozen comparison semantics" in llm.last_user
    assert "quality_at_fixed_time" in llm.last_user
    assert "steps and tokens are observed outcomes" in llm.last_user
    assert "MUST NOT replace the metric" in llm.last_user


def test_repository_task_rule_overrides_inferred_treatment_restriction():
    interpretation_ref = "spec_interpretation://s1/v1"
    baseline_ref = "baseline_result://s1/v1"
    llm = _LLM(["change the permitted clipping setting"])
    _run(
        {"problem_model_ref": "problem_model://s1",
         "spec_interpretation_ref": interpretation_ref,
         "baseline_result_ref": baseline_ref},
        llm,
        extra_objs={
            interpretation_ref: {"evaluation_semantics": {
                "comparison_mode": "repository_defined",
                "conditions_to_hold_constant": ["clipping setting fixed"],
            }},
            baseline_ref: {
                "source_kind": "live_repo", "reproduced": True,
                "metric_name": "validation_accuracy", "lower_is_better": False,
                "repository_contract": {"sources": [{
                    "path": "task/prompt.json",
                    "content": "You may modify the clipping setting.",
                }]},
            },
        },
    )
    assert "You may modify the clipping setting" in llm.last_user
    assert "take priority over inferred conditions_to_hold_constant" in llm.last_user
    assert "do not reject a Method solely because of that restriction" in llm.last_user


def test_prebaseline_generation_downgrades_repository_claims_to_hypotheses():
    llm = _LLM(["adapt the component found in the repository"])

    _run({"problem_model_ref": "problem_model://s1"}, llm)

    assert "No authoritative frozen BaselineResult exists yet" in llm.last_user
    assert "ProblemModel or literature" in llm.last_user
    assert "do not assume a named baseline component is present" in llm.last_user


def test_component_rethink_includes_prior_method_and_rejects_exact_repeat():
    prior_ref = "method://s1/prior"
    prior = {"approach": "old mechanism", "changes_outline": [],
             "why_it_should_win": "", "assumptions": []}
    llm = _LLM(["old mechanism", "different mechanism"])

    client = _run(
        {"problem_model_ref": "problem_model://s1",
         "component_ablation_failed": True,
         "ablation_failed_component": "failed component",
         "prior_method_ref": prior_ref},
        llm,
        extra_objs={prior_ref: prior},
    )

    assert llm.calls == 2
    assert "FUNDAMENTALLY DIFFERENT" in llm.last_user
    assert "old mechanism" in llm.last_user
    assert "failed component" in llm.last_user
    assert any(p.get("event_type") == "method.generated" for p in client.published)


def test_method_swap_does_not_publish_when_bounded_retry_repeats_prior_method():
    prior_ref = "method://s1/prior"
    prior = {"approach": "old mechanism", "changes_outline": [],
             "why_it_should_win": "", "assumptions": []}
    llm = _LLM(["old mechanism"])

    client = _run(
        {"problem_model_ref": "problem_model://s1", "method_not_working": True,
         "prior_method_ref": prior_ref},
        llm,
        extra_objs={prior_ref: prior},
    )

    assert llm.calls == 2
    assert not any(p.get("event_type") == "method.generated" for p in client.published)
    failures = [p for p in client.published if p.get("event_type") == "agent.failed"]
    assert failures[-1]["payload"]["underlying_reason_code"] == "method_not_replaced"


@pytest.mark.parametrize("bound", [True, False])
@pytest.mark.parametrize("trigger", ["qualified_component_closeout",
    "component_method_reconciliation", "legal_off_method_gain"])
def test_qualified_closeout_keeps_retained_mechanism_for_formal_evaluation(bound, trigger):
    prior_ref = "method://s1/prior"
    memory_ref = "research_memory://s1/current"
    review_ref = "method_improvement_review://s1/current"
    prior = {"approach": "retained measured mechanism", "changes_outline": [],
             "why_it_should_win": "", "assumptions": []}
    llm = _LLM(["retained measured mechanism"])
    payload = {
        "problem_model_ref": "problem_model://s1", "method_ref": prior_ref,
        "baseline_result_ref": "baseline://s1/current", "research_memory_ref": memory_ref,
        "experiment_feedback": "Restore the measured implementation and formally evaluate it.",
    }
    store = _Store({
        "problem_model://s1": {"research_goal": "improve the official metric"},
        prior_ref: prior,
        memory_ref: {"diagnosis": "method_improvement_proposed", "diagnosis_evidence": [review_ref]},
        review_ref: {"status": "completed", "trigger": trigger, "closeout_requested": True,
                    "method_ref": prior_ref,
                    "baseline_ref": "baseline://s1/current" if bound else "baseline://another",
                    **({"qualified_closeout_components": [{"checkpoint": "measured-source"}]}
                       if trigger == "qualified_component_closeout" else
                       {"off_method_evidence": {"candidate": {"checkpoint": "measured-source"}}}),
                    "decision": {"route": "revise_method"}},
    })
    client = _Client()
    handler = MethodAgentHandler(spec_store=store, llm_client=llm)
    asyncio.run(handler.handle_revise_requested(client, _Env(payload)))
    assert any(p.get("event_type") == "method.generated" for p in client.published) is bound
    assert llm.calls == (1 if bound else 2)
    if bound:
        assert "No new or unevaluated mechanism" in llm.last_user
        assert "formally evaluate" in llm.last_user


@pytest.mark.parametrize("bound", [True, False])
def test_incremental_closeout_contract_reuse_requires_bound_review(monkeypatch, bound):
    from autoresearch.experiments.repo_improve.tests.test_component_plan import _component

    prior_ref = "method://s1/prior"
    memory_ref = "research_memory://s1/current"
    review_ref = "method_improvement_review://s1/current"
    prior = {"approach": "unimplemented extension", "changes_outline": ["extend treatment"],
             "components": [_component("retained")]}
    seen = []

    def generate(*args, **kwargs):
        seen.append(kwargs)
        return {**prior, "approach": "retained measured treatment",
                "changes_outline": ["use measured treatment"]}

    monkeypatch.setattr("autoresearch.agents.method_agent.handler.method_core.method_step", generate)
    store = _Store({
        "problem_model://s1": {}, prior_ref: prior,
        memory_ref: {"diagnosis": "method_improvement_proposed", "diagnosis_evidence": [review_ref]},
        review_ref: {"status": "completed", "trigger": "qualified_component_closeout",
                     "method_ref": prior_ref, "baseline_ref": "baseline://s1" if bound else "other",
                     "qualified_closeout_components": [{"checkpoint": "measured-source"}],
                     "decision": {"route": "revise_method"}},
    })
    client = _Client()
    asyncio.run(MethodAgentHandler(spec_store=store, llm_client=_LLM()).handle_revise_requested(
        client, _Env({"problem_model_ref": "problem_model://s1", "method_ref": prior_ref,
                      "baseline_result_ref": "baseline://s1", "research_memory_ref": memory_ref,
                      "incremental_revision": True, "experiment_feedback": "Restore measured source"})))

    assert seen[0]["retained_treatment_revision"] is bound
    assert seen[0]["closeout_revision"] is bound
    assert any(event["event_type"] == "method.generated" for event in client.published)


def test_incremental_measured_description_reconciliation_keeps_contract_reuse(monkeypatch):
    from autoresearch.experiments.repo_improve.tests.test_component_plan import _component

    prior_ref = "method://s1/prior"
    memory_ref = "research_memory://s1/current"
    prior = {"approach": "unimplemented extension", "changes_outline": ["extend treatment"],
             "components": [_component("retained")]}
    seen = []

    def generate(*args, **kwargs):
        seen.append(kwargs)
        return {**prior, "approach": "actual measured treatment",
                "changes_outline": ["use measured treatment"]}

    monkeypatch.setattr("autoresearch.agents.method_agent.handler.method_core.method_step", generate)
    store = _Store({"problem_model://s1": {}, prior_ref: prior,
                    memory_ref: {"diagnosis": "method_description_reconciliation"}})
    client = _Client()
    asyncio.run(MethodAgentHandler(spec_store=store, llm_client=_LLM()).handle_revise_requested(
        client, _Env({"problem_model_ref": "problem_model://s1", "method_ref": prior_ref,
                      "research_memory_ref": memory_ref, "incremental_revision": True,
                      "experiment_feedback": "Describe the retained measured implementation"})))

    assert seen[0]["retained_treatment_revision"] is True
    assert any(event["event_type"] == "method.generated" for event in client.published)


def test_method_swap_does_not_treat_shared_functional_control_as_rejected_family():
    prior_ref = "method://s1/prior"
    prior = {
        "approach": "frontload optimizer work",
        "changes_outline": [],
        "why_it_should_win": "",
        "assumptions": [],
        "components": [
            {"name": "frontloaded batch schedule", "type": "gain"},
            {"name": "canonical token stream", "type": "functional"},
        ],
    }
    llm = _LLM(["Use a distinct kernel fusion mechanism while preserving canonical token stream"])

    client = _run(
        {"problem_model_ref": "problem_model://s1", "method_not_working": True,
         "prior_method_ref": prior_ref},
        llm,
        extra_objs={prior_ref: prior},
    )

    assert llm.calls == 1
    assert any(p.get("event_type") == "method.generated" for p in client.published)


def test_method_swap_does_not_treat_negative_prose_reference_as_family_reuse():
    prior = {
        "approach": "frontload batches",
        "components": [{"name": "frontloaded batch schedule", "type": "gain"}],
    }
    candidate = {
        "approach": "Abandon the frontloaded batch schedule and instead fuse optimizer kernels.",
        "changes_outline": ["Fuse the optimizer update into one kernel."],
    }

    assert not MethodAgentHandler._reuses_prior_component_family(candidate, prior)


def test_method_swap_does_not_infer_structured_gain_reuse_from_unstructured_controls():
    prior = {
        "approach": "frontload batches",
        "components": [{"name": "frontloaded batch schedule", "type": "gain"}],
    }
    candidate = {
        "approach": "Use fused communication while preserving the experiment controls.",
        "changes_outline": ["Keep the baseline batch schedule unchanged."],
    }

    assert not MethodAgentHandler._reuses_prior_component_family(candidate, prior)


def test_method_swap_places_rejected_gain_axes_at_end_of_prompt():
    prior_ref = "method://s1/prior"
    prior = {
        "approach": "old mechanism",
        "components": [
            {"name": "old gain axis", "type": "gain"},
            {"name": "shared evaluator", "type": "functional"},
        ],
    }
    llm = _LLM(["use a distinct mechanism"])

    _run(
        {"problem_model_ref": "problem_model://s1", "method_not_working": True,
         "prior_method_ref": prior_ref},
        llm,
        extra_objs={prior_ref: prior},
    )

    final_constraint = llm.last_user.split("FINAL REPLACEMENT CONSTRAINT:", 1)[1]
    assert "old gain axis" in final_constraint
    assert "shared evaluator" not in final_constraint


def test_novelty_regeneration_keeps_rejected_method_history(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTORESEARCH_WEB_WORKDIR", str(tmp_path))
    from autoresearch.experiments.repo_improve.search_memory import board_for_session

    board_for_session("s1").remember_rejected_method({"approach": "failed old mechanism"})
    llm = _LLM(["novel mechanism"])
    novelty_ref = "novelty://s1/report"

    _run(
        {"problem_model_ref": "problem_model://s1",
         "novelty_audit_report_ref": novelty_ref},
        llm,
        extra_objs={novelty_ref: {"verdict": "evidence_insufficient"}},
    )

    assert "failed old mechanism" in llm.last_user
    assert "避开" in llm.last_user


def test_novelty_regeneration_retries_exact_prior_draft():
    prior_ref = "method://s1/prior-novelty"
    novelty_ref = "novelty://s1/report"
    prior = {"approach": "same draft", "changes_outline": [],
             "why_it_should_win": "", "assumptions": []}
    llm = _LLM(["same draft", "novel revised draft"])

    client = _run(
        {"problem_model_ref": "problem_model://s1",
         "prior_method_ref": prior_ref,
         "novelty_audit_report_ref": novelty_ref},
        llm,
        extra_objs={prior_ref: prior,
                    novelty_ref: {"verdict": "evidence_insufficient"}},
    )

    assert llm.calls == 2
    assert any(p.get("event_type") == "method.generated" for p in client.published)


def test_novelty_regeneration_repeat_returns_prior_to_bounded_audit_loop():
    """A stubborn generator must not terminate before the next novelty audit/search step."""
    prior_ref = "method://s1/prior-novelty-repeat"
    novelty_ref = "novelty://s1/report-repeat"
    prior = {"approach": "same draft", "changes_outline": [],
             "why_it_should_win": "", "assumptions": []}
    llm = _LLM(["same draft"])

    client = _run(
        {"problem_model_ref": "problem_model://s1",
         "prior_method_ref": prior_ref,
         "novelty_audit_report_ref": novelty_ref},
        llm,
        extra_objs={prior_ref: prior,
                    novelty_ref: {"verdict": "evidence_insufficient"}},
    )

    assert llm.calls == 2
    generated = [p for p in client.published if p.get("event_type") == "method.generated"]
    assert generated[-1]["payload"]["method_ref"] == prior_ref
    assert not any(p.get("event_type") == "agent.failed" for p in client.published)


def test_novelty_regeneration_includes_accumulated_research_memory():
    prior_ref = "method://s1/prior-novelty-memory"
    novelty_ref = "novelty://s1/report-memory"
    memory_ref = "research_memory://s1/v1"
    prior = {"approach": "recent standard method", "changes_outline": [],
             "why_it_should_win": "", "assumptions": []}
    memory = {"negative_results": ["Earlier optimizer family was slower than baseline."]}
    llm = _LLM(["different evidence-grounded mechanism"])

    _run(
        {"problem_model_ref": "problem_model://s1",
         "prior_method_ref": prior_ref,
         "novelty_audit_report_ref": novelty_ref,
         "research_memory_ref": memory_ref},
        llm,
        extra_objs={prior_ref: prior,
                    novelty_ref: {"verdict": "evidence_insufficient"},
                    memory_ref: memory},
    )

    assert "Earlier optimizer family was slower" in llm.last_user


def test_novelty_regeneration_retries_reused_prior_component_family():
    prior_ref = "method://s1/prior-component-family"
    novelty_ref = "novelty://s1/report-family"
    prior = {
        "approach": "Combine several mechanisms.",
        "changes_outline": [], "why_it_should_win": "", "assumptions": [],
        "components": [{"name": "Quantum Router: sparse dispatch optimization"}],
    }
    llm = _LLM(["Use Quantum Router alone for dispatch", "use adaptive sequence packing"])

    client = _run(
        {"problem_model_ref": "problem_model://s1",
         "prior_method_ref": prior_ref,
         "novelty_audit_report_ref": novelty_ref},
        llm,
        extra_objs={prior_ref: prior,
                    novelty_ref: {"verdict": "evidence_insufficient"}},
    )

    assert llm.calls == 2
    assert any(p.get("event_type") == "method.generated" for p in client.published)


def test_novelty_regeneration_retries_rephrased_unstructured_method_family():
    prior_ref = "method://s1/prior-unstructured-family"
    novelty_ref = "novelty://s1/report-unstructured-family"
    prior = {"approach": "Quantum Router improves sparse dispatch", "changes_outline": [],
             "why_it_should_win": "", "assumptions": []}
    llm = _LLM(["Use Quantum Router alone for routing", "use adaptive sequence packing"])

    client = _run(
        {"problem_model_ref": "problem_model://s1",
         "prior_method_ref": prior_ref,
         "novelty_audit_report_ref": novelty_ref},
        llm,
        extra_objs={prior_ref: prior,
                    novelty_ref: {"verdict": "evidence_insufficient"}},
    )

    assert llm.calls == 2
    assert any(p.get("event_type") == "method.generated" for p in client.published)


def test_structured_components_prevent_task_context_from_becoming_method_identity():
    prior = {
        "approach": "Keep Track One FineWeb evaluation fixed and compile the training graph.",
        "changes_outline": [], "why_it_should_win": "", "assumptions": [],
        "components": [{"name": "Graph Capture: compile forward and backward"}],
    }
    candidate = {
        "approach": "Keep Track One FineWeb evaluation fixed and use low rank communication.",
        "changes_outline": [], "why_it_should_win": "", "assumptions": [],
    }

    assert not MethodAgentHandler._reuses_prior_component_family(candidate, prior)


def test_component_revision_reads_method_ref_and_retries_exact_prior_draft():
    prior_ref = "method://s1/prior-revision"
    prior = {"approach": "same component mechanism", "changes_outline": [],
             "why_it_should_win": "", "assumptions": []}
    llm = _LLM(["same component mechanism", "replaced component mechanism"])
    store = _Store({
        "problem_model://s1": {"research_goal": "lower val loss"},
        prior_ref: prior,
    })
    handler = MethodAgentHandler(spec_store=store, llm_client=llm)
    client = _Client()

    asyncio.run(handler.handle_revise_requested(client, _Env({
        "problem_model_ref": "problem_model://s1",
        "method_ref": prior_ref,
        "experiment_feedback": "component ablation failed: optimizer",
    })))

    assert llm.calls == 2
    assert "Prior method JSON" in llm.last_user
    assert "optimizer" in llm.last_user
    assert any(p.get("event_type") == "method.generated" for p in client.published)


def test_reconciliation_does_not_force_a_successful_mechanism_to_change(monkeypatch):
    prior_ref = "method://s1/prior"
    memory_ref = "research_memory://s1/current"
    prior = {"approach": "retain measured mechanism", "changes_outline": [],
             "why_it_should_win": "", "assumptions": []}
    llm = _LLM([prior["approach"]])
    store = _Store({
        "problem_model://s1": {"research_goal": "reduce latency"},
        prior_ref: prior,
        memory_ref: {"diagnosis": "method_description_reconciliation"},
    })
    handler = MethodAgentHandler(spec_store=store, llm_client=llm)
    monkeypatch.setattr(handler, "_search_history", lambda _sid: "measured legal improvement")
    client = _Client()
    asyncio.run(handler.handle_revise_requested(client, _Env({
        "problem_model_ref": "problem_model://s1", "method_ref": prior_ref,
        "research_memory_ref": memory_ref,
        "experiment_feedback": "Consume the accepted dependency output, not a duplicate implementation.",
    })))

    assert llm.calls == 1
    assert "Preserve the successful core mechanism" in llm.last_user
    assert "revise that owning component" in llm.last_user
    assert "changed dependencies must be revalidated" in llm.last_user
    assert "with a real mechanism change" not in llm.last_user
    assert "previously unsuccessful method family" not in llm.last_user
    assert "避开这些已试且未取胜" not in llm.last_user
    assert "measured legal improvement" in llm.last_user
    assert [p["event_type"] for p in client.published] == ["method.generated"]
    assert store.puts[-1][1]["approach"] == prior["approach"]


@pytest.mark.parametrize("trigger", ["legal_off_method_gain", "component_method_reconciliation"])
@pytest.mark.parametrize("binding", ["bound", "method", "baseline", "result", "core"])
def test_bound_reconciliation_review_does_not_force_new_treatment(trigger, binding):
    prior_ref = "method://s1/prior"
    memory_ref = "research_memory://s1/current"
    review_ref = "method_improvement_review://s1/current"
    result_ref = "gen_experiment_result://s1/measured"
    prior = {"approach": "retained measured mechanism", "changes_outline": [],
             "why_it_should_win": "", "assumptions": []}
    llm = _LLM([prior["approach"]])
    review = {
        "status": "completed", "trigger": trigger,
        "method_ref": "method://other" if binding == "method" else prior_ref,
        "baseline_ref": "baseline://other" if binding == "baseline" else "baseline://s1",
        "result_ref": result_ref,
        "off_method_evidence": {"candidate": {
            "result_ref": "result://other" if binding == "result" else result_ref}},
        "decision": {"route": "revise_method", "requires_core_reconstruction": binding == "core"},
    }
    store = _Store({"problem_model://s1": {}, prior_ref: prior,
        memory_ref: {"diagnosis": "method_improvement_proposed", "diagnosis_evidence": [review_ref]},
        review_ref: review})
    client = _Client()

    asyncio.run(MethodAgentHandler(spec_store=store, llm_client=llm).handle_revise_requested(
        client, _Env({"problem_model_ref": "problem_model://s1", "method_ref": prior_ref,
                      "baseline_result_ref": "baseline://s1", "research_memory_ref": memory_ref,
                      "experiment_feedback": "Reconcile the retained measured implementation."})))

    generated = any(e["event_type"] == "method.generated" for e in client.published)
    assert generated is (binding == "bound")
    assert llm.calls == (1 if binding == "bound" else 2)
    if binding == "bound":
        assert "Do not invent a different mechanism" in llm.last_user
        assert store.puts[-1][1]["approach"] == prior["approach"]


@pytest.mark.parametrize("case", ["bound", "pending", "baseline", "checkpoint",
    "admissibility", "task", "component", "execution"])
def test_reviewed_partial_integration_remains_available_for_revalidation(monkeypatch, case):
    from autoresearch.spec_store.identity import artifact_digest

    baseline = {"metric_name": "loss", "metric_value": 2.0}
    baseline_ref = refs.encode("baseline_result", "s1", "snapshot-" + artifact_digest(baseline).split(":", 1)[1])
    result = {
        "accepted": True,
        "checkpoint": "/checkpoint/measured", "method_metric": 1.0,
        "baseline_digest": artifact_digest(baseline),
        "measurement_evidence": {"value": 1.0, "valid": True,
            "attempt_id": "attempt-retained", "stdout_sha256": "a" * 64,
            "protocol_fingerprint": "b" * 64, "protocol": {"kind": "official_single"}},
        "candidate_admissibility": {"verdict": "pass", "attempt_id": "attempt-retained"},
        "method_implementation_review": {"verdict": "partial",
            "task_contract_review": {"verdict": "compliant"}},
        "repository_comparison_review": {"verdict": "pass"},
        "component_execution": {"mode": "integration", "status": "new_best",
            "execution_succeeded": True, "attempt_contract_result": {"passed": True}},
    }
    if case == "baseline":
        result["baseline_digest"] = "sha256:other"
    if case == "admissibility":
        result["candidate_admissibility"]["verdict"] = "insufficient_evidence"
    if case == "task":
        result["method_implementation_review"]["task_contract_review"]["verdict"] = "violating"
    if case == "component":
        result["component_execution"]["mode"] = "component"
    if case == "execution":
        result["component_execution"]["execution_succeeded"] = False
    result_ref = refs.encode("gen_experiment_result", "s1", "snapshot-" + artifact_digest(result).split(":", 1)[1])
    selected = {"result_ref": result_ref,
                "checkpoint": "/checkpoint/other" if case == "checkpoint" else result["checkpoint"]}
    review_ref, memory_ref, method_ref = "review://s1/current", "memory://s1", "method://s1/prior"
    memory = {"diagnosis": "method_improvement_proposed", "diagnosis_evidence": [review_ref]}
    prior = {"approach": "retained measured treatment", "changes_outline": []}
    store = _Store({baseline_ref: baseline, result_ref: result, method_ref: prior,
        memory_ref: memory, "problem_model://s1": {},
        review_ref: {"status": "pending" if case == "pending" else "completed",
            "trigger": "legal_off_method_gain", "method_ref": method_ref,
            "baseline_ref": baseline_ref, "result_ref": result_ref,
            "off_method_evidence": {"candidate": selected},
            "decision": {"route": "revise_method", "requires_core_reconstruction": False}}})
    handler = MethodAgentHandler(spec_store=store, llm_client=_LLM())
    candidate = handler._latest_admissible_reconciliation_candidate(memory)
    assert (candidate is not None) is (case == "bound")
    if case != "bound":
        return
    monkeypatch.setattr("autoresearch.agents.method_agent.handler.method_core.method_step",
                        lambda *args, **kwargs: {**prior, "reconciled_result_ref": result_ref})
    client = _Client()
    asyncio.run(handler.handle_revise_requested(client, _Env({
        "problem_model_ref": "problem_model://s1", "method_ref": method_ref,
        "baseline_result_ref": baseline_ref, "research_memory_ref": memory_ref,
        "experiment_feedback": "Revalidate the measured retained implementation."})))
    assert store.puts[-1][1]["reconciled_result_ref"] == result_ref
    from autoresearch.experiments.repo_improve.tests.test_component_plan import _component
    component = {**_component("retained"), "ablation_test": "signed_metric_improvement > 0"}
    revised_ref = "method://s1/revised"
    store._objs[revised_ref] = {**store.puts[-1][1], "components": [component]}
    seen = []
    def deliberate(*args, **kwargs):
        seen.append(kwargs.get("component_plan_revision_feedback"))
        return SimpleNamespace(method=args[3], report={"accepted": True})
    monkeypatch.setattr("autoresearch.agents.method_agent.handler.deliberation_core.deliberate_method",
                        deliberate)
    asyncio.run(handler.handle_deliberation_requested(client, _Env({
        "method_ref": revised_ref, "research_memory_ref": memory_ref,
        "baseline_result_ref": baseline_ref, "component_plan_audit": True})))
    assert "official_single" in seen[0] and result_ref in seen[0]
    saved = [obj for kind, obj in store.puts if kind == "method"][-1]
    assert saved["reconciled_result_ref"] == result_ref


@pytest.mark.parametrize("trigger", [
    "legal_off_method_gain", "component_method_reconciliation", "complete_result"])
def test_incremental_bound_reconciliation_preserves_component_contracts(monkeypatch, trigger):
    from autoresearch.experiments.repo_improve.tests.test_component_plan import _component

    prior_ref, memory_ref, review_ref = "method://s1/prior", "memory://s1", "review://s1"
    prior = {"approach": "retained mechanism", "changes_outline": [],
             "components": [_component("retained")]}
    seen = []

    def generate(*args, **kwargs):
        seen.append(kwargs)
        return dict(prior)

    monkeypatch.setattr("autoresearch.agents.method_agent.handler.method_core.method_step", generate)
    store = _Store({"problem_model://s1": {}, prior_ref: prior,
        memory_ref: {"diagnosis": "method_improvement_proposed", "diagnosis_evidence": [review_ref]},
        review_ref: {"status": "completed", "trigger": trigger, "method_ref": prior_ref,
                     "baseline_ref": "baseline://s1", "result_ref": "result://s1",
                     "off_method_evidence": {"candidate": {"result_ref": "result://s1"}},
                     "decision": {"route": "revise_method", "requires_core_reconstruction": False}}})
    client = _Client()

    asyncio.run(MethodAgentHandler(spec_store=store, llm_client=_LLM()).handle_revise_requested(
        client, _Env({"problem_model_ref": "problem_model://s1", "method_ref": prior_ref,
                      "baseline_result_ref": "baseline://s1", "research_memory_ref": memory_ref,
                      "incremental_revision": True, "experiment_feedback": "Update from bound evidence."})))

    reconciling = trigger != "complete_result"
    assert seen[0]["retained_treatment_revision"] is reconciling
    assert any(e["event_type"] == "method.generated" for e in client.published) is reconciling
    if reconciling:
        assert store.puts[-1][1]["components"] == prior["components"]


def test_revision_keeps_historical_gain_bound_to_its_implementation(monkeypatch):
    method_ref = "method://s1/prior"
    memory_ref = "research_memory://s1/current"
    historical_ref = "gen_experiment_result://s1/off-method"
    current_ref = refs.encode("gen_experiment_result", "s1", "current")
    llm = _LLM(["revised mechanism with unmeasured outcome"])
    store = _Store({
        "problem_model://s1": {"research_goal": "improve the official metric"},
        method_ref: {"approach": "old mechanism", "changes_outline": [],
                     "why_it_should_win": "", "assumptions": []},
        memory_ref: {"diagnosis": "component_expectation_failed",
                     "reconciliation_candidates": [
                         {"result_ref": historical_ref, "checkpoint": "/checkpoint/old"}]},
        historical_ref: {
            "method_metric": 1153,
            "off_method_observation": {"valid": True, "improved": True},
            "changes_summary": "used a two-ply opponent-reply feature",
            "method_implementation_review": {
                "issues": ["the two-ply feature changed the frozen mechanism"]},
        },
        current_ref: {"verdict": "invalid", "method_metric": 1153,
                      "bottleneck": "one-ply revision measured below the quality gate"},
    })
    handler = MethodAgentHandler(spec_store=store, llm_client=llm)
    monkeypatch.setattr(handler, "_search_history", lambda _sid: "")
    asyncio.run(handler.handle_revise_requested(_Client(), _Env({
        "problem_model_ref": "problem_model://s1", "method_ref": method_ref,
        "research_memory_ref": memory_ref,
        "experiment_feedback": "revise after the latest failed attempt",
    })))

    assert "two-ply opponent-reply feature" in llm.last_user
    assert "one-ply revision measured below the quality gate" in llm.last_user
    assert "its score is unmeasured" in llm.last_user
    assert "exact evaluated implementation" in llm.last_system


def test_no_change_design_feedback_can_remove_required_empty_component(monkeypatch):
    method_ref = "method://s1/confirmed"
    memory_ref = "research_memory://s1/current"
    prior = {"approach": "required pre-pass then active optimization",
             "changes_outline": ["run pre-pass", "run active optimization"],
             "why_it_should_win": "active optimization reduces cost", "assumptions": []}
    store = _Store({
        "problem_model://s1": {"research_goal": "reduce cost"},
        method_ref: prior,
        memory_ref: {"reconciliation_candidates": [{"result_ref": "result://s1/earlier"}]},
    })
    llm = _LLM(["active optimization without the empty pre-pass"])
    handler = MethodAgentHandler(spec_store=store, llm_client=llm)
    monkeypatch.setattr(handler, "_search_history", lambda _sid: "")
    asyncio.run(handler.handle_revise_requested(_Client(), _Env({
        "problem_model_ref": "problem_model://s1", "method_ref": method_ref,
        "research_memory_ref": memory_ref,
        "experiment_feedback": "COMPONENT_NO_CHANGE_DESIGN_BLOCK: bound local check passed, diff empty",
    })))
    assert "remove it as a required Method component" in llm.last_user
    assert "Reconcile only the implicated component's description" not in llm.last_user


def test_revised_method_gets_a_distinct_content_addressed_ref():
    prior_ref = "method://s1/v1"
    prior = {"approach": "old mechanism", "changes_outline": [],
             "why_it_should_win": "", "assumptions": []}
    store = _Store({
        "problem_model://s1": {"research_goal": "lower val loss"},
        prior_ref: prior,
    })
    handler = MethodAgentHandler(spec_store=store, llm_client=_LLM(["new mechanism"]))
    client = _Client()

    asyncio.run(handler.handle_revise_requested(client, _Env({
        "problem_model_ref": "problem_model://s1",
        "method_ref": prior_ref,
        "experiment_feedback": "replace the failed component",
    })))

    generated_ref = next(
        event["payload"]["method_ref"]
        for event in client.published if event["event_type"] == "method.generated"
    )
    assert generated_ref != prior_ref
    assert generated_ref.startswith("method://s1/method-")


def test_component_plan_audit_preserves_confirmed_core_and_accepts_audited_components(
        monkeypatch):
    method_ref = "method://s1/confirmed"
    original = {
        "approach": "confirmed core mechanism",
        "changes_outline": ["keep core"],
        "why_it_should_win": "confirmed reason",
        "assumptions": ["confirmed assumption"],
        "components": [{"component_id": "old", "name": "old component"}],
    }
    audited_components = [{
        "component_id": "orthogonalize",
        "name": "orthogonal representation",
        "type": "indirect_gain",
        "depends_on": [],
        "purpose": "provide the orthogonal representation required by the later loss",
        "inputs": ["coupled representation"],
        "outputs": ["orthogonal representation"],
        "acceptance_requires": [],
        "mechanism": "produce orthogonal representations for the later loss",
        "expected_effect": "orthogonality error below threshold",
        "functional_check": "orthogonality_error <= 0.01",
        "implementation_check": "diff and runtime evidence show the orthogonalization operation",
        "optimization_space": "reduce orthogonality error without changing the interface",
        "downstream_contract": "provide the orthogonal representation to the later loss",
    }]

    def fake_deliberate(*args, **kwargs):
        assert kwargs["component_plan_audit"] is True
        return SimpleNamespace(
            method={
                "approach": "silently rewritten core",
                "changes_outline": ["rewrite"],
                "components": audited_components,
            },
            report={"accepted": True},
        )

    monkeypatch.setattr(
        "autoresearch.agents.method_agent.handler.deliberation_core.deliberate_method",
        fake_deliberate,
    )
    store = _Store({
        method_ref: original,
        "problem_model://s1": {"research_goal": "lower val loss"},
        "baseline_result://s1/v1": {"metric_name": "val_loss", "metric_value": 1.0},
    })
    client = _Client()
    handler = MethodAgentHandler(spec_store=store, llm_client=_LLM())

    asyncio.run(handler.handle_deliberation_requested(client, _Env({
        "method_ref": method_ref,
        "problem_model_ref": "problem_model://s1",
        "baseline_result_ref": "baseline_result://s1/v1",
        "component_plan_audit": True,
    })))

    saved = next(obj for kind, obj in store.puts if kind == "method")
    assert saved["approach"] == original["approach"]
    assert saved["changes_outline"] == original["changes_outline"]
    assert saved["components"] == audited_components


def test_component_plan_audit_cannot_freeze_when_critic_did_not_accept(monkeypatch):
    method_ref = "method://s1/confirmed"
    original = {
        "approach": "confirmed core mechanism", "changes_outline": [],
        "why_it_should_win": "reason", "assumptions": [],
        "components": [{"component_id": "old", "name": "old"}],
    }

    monkeypatch.setattr(
        "autoresearch.agents.method_agent.handler.deliberation_core.deliberate_method",
        lambda *args, **kwargs: SimpleNamespace(
            method=original,
            report={
                "accepted": False,
                "converged": True,
                "transcript": [{
                    "critic_deep_problems": [{
                        "component": "gate",
                        "issue": "acceptance check cannot establish the claimed property",
                    }],
                }],
            }),
    )
    store = _Store({
        method_ref: original,
        "problem_model://s1": {"research_goal": "lower loss"},
        "baseline_result://s1/v1": {"metric_name": "loss", "metric_value": 1.0},
    })
    client = _Client()
    handler = MethodAgentHandler(spec_store=store, llm_client=_LLM())

    asyncio.run(handler.handle_deliberation_requested(client, _Env({
        "method_ref": method_ref,
        "problem_model_ref": "problem_model://s1",
        "baseline_result_ref": "baseline_result://s1/v1",
        "component_plan_audit": True,
    })))

    assert not [item for item in client.published if item["event_type"] == "method.deliberated"]
    failed = [item for item in client.published if item["event_type"] == "agent.failed"]
    assert failed[0]["payload"]["underlying_reason_code"] == "component_plan_audit_failed"
    assert "gate: acceptance check cannot establish" in failed[0]["payload"]["underlying_reason"]


def test_component_plan_audit_retains_legal_reconciliation_checkpoint(monkeypatch):
    method_ref = "method://s1/confirmed"
    memory_ref = "research_memory://s1/current"
    result_ref = "gen_experiment_result://s1/snapshot-legal"
    component = {
        "component_id": "fast-path", "name": "fast path", "type": "direct_gain",
        "depends_on": [], "acceptance_requires": [],
        "purpose": "reduce runtime", "inputs": ["input"], "outputs": ["output"],
        "mechanism": "use the measured fast path", "expected_effect": "lower runtime",
        "ablation_test": "signed_metric_improvement > 0",
        "implementation_check": "official evaluator and diff show the fast path",
        "optimization_space": "tune the fast path without changing the evaluator",
        "downstream_contract": "provide the evaluated output",
    }
    original = {
        "approach": "confirmed mechanism", "changes_outline": [],
        "why_it_should_win": "faster implementation", "assumptions": [],
        "reconciled_result_ref": result_ref,
        "components": [component],
    }
    seen = []

    def deliberate(*args, **kwargs):
        seen.append((args[3], kwargs.get("component_plan_revision_feedback")))
        return SimpleNamespace(method=args[3], report={"accepted": True})

    monkeypatch.setattr(
        "autoresearch.agents.method_agent.handler.deliberation_core.deliberate_method",
        deliberate,
    )
    store = _Store({
        method_ref: original,
        memory_ref: {
            "diagnosis": "method_description_reconciliation",
            "reconciliation_candidates": [
                {"result_ref": result_ref, "checkpoint": "/checkpoint/fast"}],
        },
        result_ref: {
            "checkpoint": "/checkpoint/fast",
            "off_method_observation": {"valid": True, "improved": True},
            "method_implementation_review": {
                "verdict": "drifted", "issues": ["fast path differs from Method"],
                "remediation": "revise the fast-path contract",
            },
            "component_execution": {"diff_ref": "/checkpoint/fast#git-diff"},
            "run_record_snapshot_ref": "run_record://s1/fast",
            "changes_summary": "measured fast path",
            "diff": "diff --git a/fast.py b/fast.py",
        },
        "run_record://s1/fast": {"stdout_excerpt": "official score improved"},
    })
    client = _Client()
    handler = MethodAgentHandler(spec_store=store, llm_client=_LLM())

    asyncio.run(handler.handle_deliberation_requested(client, _Env({
        "method_ref": method_ref,
        "research_memory_ref": memory_ref,
        "component_plan_audit": True,
    })))

    assert seen[0][0]["reconciled_result_ref"] == result_ref
    assert result_ref in seen[0][1]
    assert "mechanism_understanding" in seen[0][1]
    assert "score alone grants no Method attribution" in seen[0][1]
    assert '"verdict": "drifted"' in seen[0][1]
    assert "fast path differs from Method" in seen[0][1]
    assert "/checkpoint/fast#git-diff" in seen[0][1]
    assert "official score improved" in seen[0][1]
    saved = next(obj for kind, obj in store.puts if kind == "method")
    assert saved["reconciled_result_ref"] == result_ref
    assert saved["approach"] == original["approach"]


def test_component_plan_audit_skips_newer_invalid_reconciliation_candidate(monkeypatch):
    method_ref = "method://s1/confirmed"
    memory_ref = "research_memory://s1/current"
    valid_ref = "gen_experiment_result://s1/valid"
    invalid_ref = "gen_experiment_result://s1/invalid"
    original = {
        "approach": "confirmed mechanism", "changes_outline": [],
        "why_it_should_win": "faster implementation", "assumptions": [],
        "reconciled_result_ref": valid_ref,
        "components": [{
            "component_id": "fast-path", "name": "fast path", "type": "direct_gain",
            "depends_on": [], "acceptance_requires": [], "purpose": "reduce runtime",
            "inputs": ["input"], "outputs": ["output"], "mechanism": "measured fast path",
            "expected_effect": "lower runtime", "ablation_test": "metric improves",
            "implementation_check": "official evaluator and diff", "optimization_space": "tune",
            "downstream_contract": "evaluated output",
        }],
    }
    seen = []

    def deliberate(*args, **kwargs):
        seen.append(args[3])
        return SimpleNamespace(method=args[3], report={"accepted": True})

    monkeypatch.setattr(
        "autoresearch.agents.method_agent.handler.deliberation_core.deliberate_method",
        deliberate,
    )
    store = _Store({
        method_ref: original,
        memory_ref: {
            "diagnosis": "method_description_reconciliation",
            "reconciliation_candidates": [
                {"result_ref": valid_ref, "checkpoint": "/checkpoint/valid"},
                {"result_ref": invalid_ref, "checkpoint": "/checkpoint/invalid"},
            ],
        },
        valid_ref: {
            "checkpoint": "/checkpoint/valid",
            "off_method_observation": {"valid": True, "improved": True},
        },
        invalid_ref: {
            "checkpoint": "/checkpoint/invalid",
            "off_method_observation": {"valid": False, "improved": False},
        },
    })
    handler = MethodAgentHandler(spec_store=store, llm_client=_LLM())
    asyncio.run(handler.handle_deliberation_requested(_Client(), _Env({
        "method_ref": method_ref,
        "research_memory_ref": memory_ref,
        "component_plan_audit": True,
    })))

    assert seen[0]["reconciled_result_ref"] == valid_ref


def test_method_revision_keeps_legal_candidate_as_comparison_after_diagnosis_is_overwritten(monkeypatch):
    method_ref = "method://s1/confirmed"
    memory_ref = "research_memory://s1/current"
    result_ref = "gen_experiment_result://s1/legal"
    original = {
        "approach": "aspirational mechanism", "changes_outline": [],
        "why_it_should_win": "expected larger gain", "assumptions": [],
        "components": [{"component_id": "fast-path", "name": "fast path"}],
    }
    store = _Store({
        method_ref: original,
        memory_ref: {
            # The most recent failed retry overwrote the earlier reconciliation diagnosis.
            "diagnosis": "component_expectation_failed",
            "reconciliation_candidates": [
                {"result_ref": result_ref, "checkpoint": "/checkpoint/legal"}],
        },
        result_ref: {
            "checkpoint": "/checkpoint/legal",
            "off_method_observation": {"valid": True, "improved": True},
        },
        "problem_model://s1": {"research_goal": "lower runtime"},
    })
    handler = MethodAgentHandler(spec_store=store, llm_client=_LLM([original["approach"]]))
    monkeypatch.setattr(handler, "_search_history", lambda _sid: "")
    asyncio.run(handler.handle_revise_requested(_Client(), _Env({
        "method_ref": method_ref,
        "prior_method_ref": method_ref,
        "problem_model_ref": "problem_model://s1",
        "research_memory_ref": memory_ref,
        "experiment_feedback": "latest attempt missed its expected magnitude",
    })))

    saved = next(obj for kind, obj in store.puts if kind == "method")
    assert "reconciled_result_ref" not in saved
    assert result_ref in handler.llm_client.last_user
    assert "different mechanism" in handler.llm_client.last_user


def test_failed_expectation_rejects_renamed_method_despite_older_off_method_gain(monkeypatch):
    method_ref = "method://s1/confirmed"
    memory_ref = "research_memory://s1/current"
    result_ref = "gen_experiment_result://s1/earlier-gain"
    prior = {
        "approach": "shared scorer on fixed features",
        "changes_outline": ["extract features", "train shared scorer"],
        "why_it_should_win": "the features should be sufficient",
        "assumptions": ["fixed features reach the quality gate"],
        "components": [{"component_id": "scorer", "type": "direct_gain",
                        "purpose": "score the fixed features",
                        "expected_effect": "pass the quality gate",
                        "optimization_space": "hidden width"}],
    }
    renamed = json.loads(json.dumps(prior))
    renamed["components"][0]["component_id"] = "renamed-scorer"
    extended = json.loads(json.dumps(prior))
    extended["components"].append({
        "component_id": "new-features", "type": "indirect_gain",
        "purpose": "add missing discriminative information",
        "expected_effect": "improve held-out accuracy",
        "optimization_space": "feature construction",
    })
    drafts = iter((renamed, extended))
    feedbacks = []

    def generate(*_args, **kwargs):
        feedbacks.append(kwargs.get("feedback") or "")
        return next(drafts)

    monkeypatch.setattr(
        "autoresearch.agents.method_agent.handler.method_core.method_step", generate)
    store = _Store({
        "problem_model://s1": {"research_goal": "improve the official metric"},
        method_ref: prior,
        memory_ref: {"diagnosis": "component_expectation_failed",
                     "reconciliation_candidates": [{"result_ref": result_ref}]},
        result_ref: {"off_method_observation": {"valid": True, "improved": True}},
    })
    handler = MethodAgentHandler(spec_store=store, llm_client=_LLM())
    monkeypatch.setattr(handler, "_search_history", lambda _sid: "")
    asyncio.run(handler.handle_revise_requested(_Client(), _Env({
        "method_ref": method_ref,
        "problem_model_ref": "problem_model://s1",
        "research_memory_ref": memory_ref,
        "experiment_feedback": "The fixed features missed the measured quality gate.",
    })))

    assert len(feedbacks) == 2
    assert "real mechanism change" in feedbacks[0]
    assert "Reconcile only the implicated component" not in feedbacks[0]
    assert result_ref in feedbacks[0]
    assert "changed only names, checks, or evidence wording" in feedbacks[1]
    assert next(obj for kind, obj in store.puts if kind == "method") == extended


def test_method_revision_binds_candidate_only_on_explicit_adoption(monkeypatch):
    method_ref = "method://s1/confirmed"
    result_ref = "gen_experiment_result://s1/legal"
    store = _Store({
        method_ref: {"approach": "old mechanism", "changes_outline": [],
                     "why_it_should_win": "", "assumptions": []},
        "research_memory://s1/current": {
            "diagnosis": "method_description_reconciliation",
            "reconciliation_candidates": [{"result_ref": result_ref}],
        },
        result_ref: {
            "checkpoint": "/checkpoint/legal",
            "off_method_observation": {"valid": True, "improved": True},
        },
        "problem_model://s1": {"research_goal": "lower runtime"},
    })
    handler = MethodAgentHandler(spec_store=store, llm_client=_LLM())
    monkeypatch.setattr(handler, "_search_history", lambda _sid: "")
    monkeypatch.setattr(
        "autoresearch.agents.method_agent.handler.method_core.method_step",
        lambda *_args, **_kwargs: {
            "approach": "measured legal mechanism", "changes_outline": [],
            "why_it_should_win": "", "assumptions": [],
            "reconciled_result_ref": result_ref,
        },
    )
    asyncio.run(handler.handle_revise_requested(_Client(), _Env({
        "method_ref": method_ref,
        "problem_model_ref": "problem_model://s1",
        "research_memory_ref": "research_memory://s1/current",
        "experiment_feedback": "consider the legal observed mechanism",
    })))

    saved = next(obj for kind, obj in store.puts if kind == "method")
    assert saved["reconciled_result_ref"] == result_ref


def test_component_plan_audit_does_not_bind_unadopted_historical_candidate(monkeypatch):
    method_ref = "method://s1/new"
    result_ref = "gen_experiment_result://s1/old-fast"
    seen = []

    def deliberate(*args, **kwargs):
        seen.append((args[3], kwargs.get("component_plan_revision_feedback")))
        return SimpleNamespace(method=args[3], report={"accepted": True})

    monkeypatch.setattr(
        "autoresearch.agents.method_agent.handler.deliberation_core.deliberate_method",
        deliberate,
    )
    store = _Store({
        method_ref: {
            "approach": "new independent mechanism", "changes_outline": [],
            "why_it_should_win": "new reason", "assumptions": [],
            "components": [{
                "component_id": "new-path", "name": "new path", "type": "direct_gain",
                "depends_on": [], "acceptance_requires": [], "purpose": "reduce runtime",
                "inputs": ["input"], "outputs": ["output"], "mechanism": "new path",
                "expected_effect": "lower runtime", "ablation_test": "metric improves",
                "implementation_check": "check new path", "optimization_space": "tune",
                "downstream_contract": "evaluated output",
            }],
        },
        "research_memory://s1/current": {
            "diagnosis": "method_description_reconciliation",
            "reconciliation_candidates": [{"result_ref": result_ref}],
        },
        result_ref: {
            "checkpoint": "/checkpoint/old-fast",
            "off_method_observation": {"valid": True, "improved": True},
        },
    })
    handler = MethodAgentHandler(spec_store=store, llm_client=_LLM())
    asyncio.run(handler.handle_deliberation_requested(_Client(), _Env({
        "method_ref": method_ref,
        "research_memory_ref": "research_memory://s1/current",
        "component_plan_audit": True,
    })))

    assert "reconciled_result_ref" not in seen[0][0]
    assert "mechanism_understanding" not in str(seen[0][1])
    saved = next(obj for kind, obj in store.puts if kind == "method")
    assert "reconciled_result_ref" not in saved


def test_component_plan_audit_routes_confirmed_method_contract_conflict(monkeypatch):
    method_ref = "method://s1/confirmed"
    original = {
        "approach": "prohibited core mechanism", "changes_outline": [],
        "why_it_should_win": "reason", "assumptions": [],
        "components": [{"component_id": "old", "name": "old"}],
    }
    monkeypatch.setattr(
        "autoresearch.agents.method_agent.handler.deliberation_core.deliberate_method",
        lambda *args, **kwargs: SimpleNamespace(
            method=original,
            report={
                "accepted": False,
                "method_revision_required": True,
                "method_revision_reason": "repository contract prohibits the core mechanism",
            },
        ),
    )
    store = _Store({
        method_ref: original,
        "problem_model://s1": {"research_goal": "lower loss"},
        "baseline_result://s1/v1": {"metric_name": "loss", "metric_value": 1.0},
    })
    client = _Client()
    handler = MethodAgentHandler(spec_store=store, llm_client=_LLM())

    asyncio.run(handler.handle_deliberation_requested(client, _Env({
        "method_ref": method_ref,
        "problem_model_ref": "problem_model://s1",
        "baseline_result_ref": "baseline_result://s1/v1",
        "component_plan_audit": True,
    })))

    assert not [item for item in client.published if item["event_type"] == "method.deliberated"]
    failed = [item for item in client.published if item["event_type"] == "agent.failed"]
    assert failed[0]["payload"]["underlying_reason_code"] == (
        "confirmed_method_contract_conflict"
    )
    assert "prohibits" in failed[0]["payload"]["underlying_reason"]
    report_ref = failed[0]["payload"]["component_plan_checkpoint_ref"]
    assert report_ref.startswith("method_debate_report://s1/")
    report = next(obj for kind, obj in store.puts if kind == "method_debate_report")
    assert report["baseline_result_ref"] == "baseline_result://s1/v1"
    assert report["confirmed_method_ref"] == method_ref


@pytest.mark.parametrize("protocol_exception", [False, True])
def test_unavailable_cap_review_preserves_draft_without_freezing_method(monkeypatch, protocol_exception):
    method_ref = "method://s1/confirmed"
    original = {"approach": "frozen mechanism", "components": [{"name": "initial"}]}
    revised = [{"name": "repaired dependency design"}]
    seen = []
    feedback_seen = []
    last_review = {"outcome": "critic_review_unavailable",
                   "protocol_errors": [{"issue": "copy the exact resource quote"}]}

    def debate(*args, **kwargs):
        seen.append(args[3])
        feedback_seen.append(kwargs.get("component_plan_revision_feedback"))
        if protocol_exception:
            from autoresearch.agents.deliberation_agent.core import DeliberationLLMError
            error = DeliberationLLMError("Critic reply was malformed")
            error.draft_method = {"approach": "frozen mechanism", "components": revised}
            error.debate_transcript = [last_review]
            raise error
        return SimpleNamespace(
            method={"approach": "must not replace frozen mechanism", "components": revised},
            report={"accepted": False, "transcript": [last_review]} )

    monkeypatch.setattr(
        "autoresearch.agents.method_agent.handler.deliberation_core.deliberate_method", debate)
    store = _Store({method_ref: original})
    handler = MethodAgentHandler(spec_store=store, llm_client=_LLM())
    client = _Client()
    payload = {"method_ref": method_ref, "component_plan_audit": True}
    asyncio.run(handler.handle_deliberation_requested(client, _Env(payload)))
    failure = client.published[-1]["payload"]
    assert failure["underlying_reason_code"] == "component_plan_review_unavailable"
    checkpoint_ref = failure["component_plan_checkpoint_ref"]
    checkpoint = next(obj for kind, obj in store.puts if kind == "method_debate_report")
    assert checkpoint["draft_components"] == revised
    assert checkpoint["accepted"] is False
    assert not any(kind == "method" for kind, _ in store.puts)
    # Simulate a fresh handler loading the persisted failure artifact after a restart.
    resumed_store = _Store({method_ref: original, checkpoint_ref: checkpoint})
    resumed = MethodAgentHandler(spec_store=resumed_store, llm_client=_LLM())
    asyncio.run(resumed.handle_deliberation_requested(_Client(), _Env({
        **payload, "component_plan_checkpoint_ref": checkpoint_ref})))
    assert seen[-1] == {"approach": "frozen mechanism", "components": revised}
    assert "copy the exact resource quote" in feedback_seen[-1]
    # A checkpoint from another frozen Method cannot seed this audit.
    checkpoint["confirmed_method_ref"] = "method://s1/other"
    asyncio.run(resumed.handle_deliberation_requested(_Client(), _Env({
        **payload, "component_plan_checkpoint_ref": checkpoint_ref})))
    assert seen[-1] == original
    assert not feedback_seen[-1]


def test_component_plan_revision_cannot_freeze_the_unchanged_exhausted_plan(monkeypatch):
    method_ref = "method://s1/confirmed"
    component = {
        "component_id": "orthogonalize",
        "name": "orthogonal representation",
        "type": "indirect_gain",
        "depends_on": [],
        "purpose": "provide the representation required downstream",
        "inputs": ["coupled representation"],
        "outputs": ["orthogonal representation"],
        "acceptance_requires": [],
        "mechanism": "produce orthogonal representations",
        "expected_effect": "orthogonality error below threshold",
        "functional_check": "orthogonality_error <= 0.01",
        "implementation_check": "diff and runtime evidence show the operation",
        "optimization_space": "reduce the error while preserving the interface",
        "downstream_contract": "provide the representation to dependent components",
    }
    original = {
        "approach": "confirmed core mechanism", "changes_outline": [],
        "why_it_should_win": "reason", "assumptions": [], "components": [component],
    }
    monkeypatch.setattr(
        "autoresearch.agents.method_agent.handler.deliberation_core.deliberate_method",
        lambda *args, **kwargs: SimpleNamespace(
            method=original, report={"accepted": True, "transcript": []}),
    )
    store = _Store({
        method_ref: original,
        "problem_model://s1": {"research_goal": "lower loss"},
        "baseline_result://s1/v1": {"metric_name": "loss", "metric_value": 1.0},
    })
    client = _Client()
    handler = MethodAgentHandler(spec_store=store, llm_client=_LLM())

    asyncio.run(handler.handle_deliberation_requested(client, _Env({
        "method_ref": method_ref,
        "problem_model_ref": "problem_model://s1",
        "baseline_result_ref": "baseline_result://s1/v1",
        "component_plan_audit": True,
        "component_plan_revision_feedback": "orthogonalize exhausted all attempts",
    })))

    assert not [item for item in client.published if item["event_type"] == "method.deliberated"]
    failed = [item for item in client.published if item["event_type"] == "agent.failed"]
    assert failed[0]["payload"]["underlying_reason_code"] == "component_plan_audit_failed"
    assert "unchanged exhausted plan" in failed[0]["payload"]["underlying_reason"]

    # A clean CAP review may retain a legal plan when only its implementation was illegal.
    legal_plan_client = _Client()
    asyncio.run(handler.handle_deliberation_requested(legal_plan_client, _Env({
        "method_ref": method_ref,
        "problem_model_ref": "problem_model://s1",
        "baseline_result_ref": "baseline_result://s1/v1",
        "component_plan_audit": True,
        "component_plan_revision_feedback": "Implementation edited a forbidden file",
        "component_plan_allow_unchanged": True,
    })))
    assert any(item["event_type"] == "method.deliberated"
               for item in legal_plan_client.published)


def test_component_plan_conflict_requires_changing_a_conflicting_parent(monkeypatch):
    method_ref = "method://s1/confirmed"

    def component(component_id, dependencies):
        return {
            "component_id": component_id, "name": component_id,
            "type": "indirect_gain", "depends_on": dependencies,
            "purpose": "prepare input", "inputs": ["task input"],
            "outputs": [component_id], "acceptance_requires": [],
            "mechanism": "prepare input", "expected_effect": "valid output",
            "functional_check": "failed_check_count == 0",
            "implementation_check": "runtime check and diff match",
            "optimization_space": "choose a compatible implementation",
            "downstream_contract": "provide output to the next component",
        }

    original = {"approach": "combine two prepared inputs", "components": [
        component("root", []), component("left", ["root"]),
        component("right", ["root"]), component("join", ["left", "right"]),
    ]}
    cosmetic = json.loads(json.dumps(original))
    cosmetic["components"][-1]["expected_effect"] = "better output"
    monkeypatch.setattr(
        "autoresearch.agents.method_agent.handler.deliberation_core.deliberate_method",
        lambda *args, **kwargs: SimpleNamespace(
            method=cosmetic, report={"accepted": True, "transcript": []}),
    )
    a, b = "a" * 64, "b" * 64
    failure = {
        "underlying_reason_code": "component_plan_invalid",
        "underlying_reason": (
            f"dependency checkpoint conflict for solve.py: checkpoint-{a}, "
            f"checkpoint-{b} produced different states"),
    }
    store = _Store({
        method_ref: original,
        "gen_experiment_result://s1/left": {"component_execution": {
            "component_id": "left", "checkpoint": f"/work/checkpoint-{a}"}},
        "gen_experiment_result://s1/right": {"component_execution": {
            "component_id": "right", "checkpoint": f"/work/checkpoint-{b}"}},
    })
    handler = MethodAgentHandler(spec_store=store, llm_client=_LLM())
    client = _Client()
    asyncio.run(handler.handle_deliberation_requested(client, _Env({
        "method_ref": method_ref, "component_plan_audit": True,
        "component_plan_revision_feedback": failure["underlying_reason"],
        "component_plan_failure": failure,
    })))
    assert not any(event["event_type"] == "method.deliberated"
                   for event in client.published)
    failed = [event for event in client.published
              if event["event_type"] == "agent.failed"]
    assert failed[0]["payload"]["underlying_reason_code"] == "component_plan_audit_failed"
    assert "left, right" in failed[0]["payload"]["underlying_reason"]
    assert failed[0]["payload"]["component_plan_failure"] == failure

    cosmetic_parent = json.loads(json.dumps(cosmetic))
    cosmetic_parent["components"][1]["expected_effect"] = "improved description"
    assert handler._unchanged_conflicting_parents(
        "s1", failure, original["components"], cosmetic_parent["components"])

    revised = json.loads(json.dumps(cosmetic))
    revised["components"][2]["depends_on"] = ["left"]
    assert handler._unchanged_conflicting_parents(
        "s1", failure, original["components"], revised["components"]) == []
    revised_join = json.loads(json.dumps(cosmetic))
    revised_join["components"][3]["depends_on"] = ["left"]
    assert handler._unchanged_conflicting_parents(
        "s1", failure, original["components"], revised_join["components"]) == []


def test_component_plan_failure_cannot_pass_with_only_prose_changes(monkeypatch):
    method_ref = "method://s1/confirmed"
    failed = {
        "component_id": "features", "name": "features", "type": "indirect_gain",
        "depends_on": [], "inputs": ["task input"], "outputs": ["features"],
        "mechanism": "static features → scorer",
        "functional_check": "best_validation_accuracy >= 0.95",
        "implementation_check": "no board mutation",
        "optimization_space": "static feature definitions",
        "expected_effect": "accuracy ≥ 0.95",
    }
    original = {"approach": "feature scorer", "components": [failed]}
    cosmetic = json.loads(json.dumps(original))
    cosmetic["components"][0]["mechanism"] = "static features -> scorer"
    cosmetic["components"][0]["expected_effect"] = "accuracy >= 0.95"
    monkeypatch.setattr(
        "autoresearch.agents.method_agent.handler.deliberation_core.deliberate_method",
        lambda *args, **kwargs: SimpleNamespace(
            method=cosmetic, report={"accepted": True, "transcript": []}),
    )
    handler = MethodAgentHandler(spec_store=_Store({method_ref: original}),
                                 llm_client=_LLM())
    client = _Client()
    asyncio.run(handler.handle_deliberation_requested(client, _Env({
        "method_ref": method_ref, "component_plan_audit": True,
        "component_plan_revision_feedback": "features missed its validation check",
        "component_plan_failed_component": "features",
    })))
    assert not any(event["event_type"] == "method.deliberated"
                   for event in client.published)
    failures = [event for event in client.published
                if event["event_type"] == "agent.failed"]
    assert "features executable contract and dependency graph unchanged" in (
        failures[0]["payload"]["underlying_reason"])

    repaired = json.loads(json.dumps(cosmetic))
    repaired["components"][0]["functional_check"] = "feature validity on held-out inputs"
    assert not handler._failed_component_contract_unchanged(
        "features", original["components"], repaired["components"])


@pytest.mark.parametrize("changed,field,unchanged", [
    ("source", "implementation_check", False),
    ("middle", "functional_check", False),
    ("leaf", "optimization_space", False),
    ("unrelated", "implementation_check", True),
    ("source", "expected_effect", True),
    ("source", "mechanism", True),
])
def test_failed_component_revision_checks_its_dependency_contracts(changed, field, unchanged):
    previous = [
        {"component_id": "source", "depends_on": [], "implementation_check": "valid source"},
        {"component_id": "middle", "depends_on": ["source"], "functional_check": "valid output"},
        {"component_id": "leaf", "depends_on": ["middle"]},
        {"component_id": "unrelated", "depends_on": []},
    ]
    revised = json.loads(json.dumps(previous))
    next(item for item in revised if item["component_id"] == changed)[field] = "revised"
    assert MethodAgentHandler._failed_component_contract_unchanged(
        "leaf", previous, revised) is unchanged


def test_component_plan_accepts_audited_upstream_repair_for_downstream_failure(monkeypatch):
    method_ref = "method://s1/confirmed"
    source = {"component_id": "source", "name": "source", "type": "indirect_gain",
              "purpose": "build valid features", "expected_effect": "valid source",
              "acceptance_requires": [], "downstream_contract": "provide features",
              "depends_on": [], "inputs": ["task input"], "outputs": ["features"],
              "mechanism": "build source", "functional_check": "source_mismatch_count == 0",
              "implementation_check": "process all entries", "optimization_space": "source code"}
    leaf = {"component_id": "leaf", "name": "leaf", "type": "direct_gain",
            "purpose": "answer efficiently", "expected_effect": "lower task cost",
            "ablation_test": "signed_metric_improvement > 0",
            "acceptance_requires": ["source"], "downstream_contract": "provide answer",
            "depends_on": ["source"], "inputs": ["features"], "outputs": ["answer"],
            "mechanism": "consume source", "functional_check": "output_mismatch_count == 0",
            "implementation_check": "consume features", "optimization_space": "consumer code"}
    original = {"approach": "same confirmed core", "components": [source, leaf]}
    repaired = json.loads(json.dumps(original))
    repaired["components"][0]["implementation_check"] = "process only valid entries"
    monkeypatch.setattr(
        "autoresearch.agents.method_agent.handler.deliberation_core.deliberate_method",
        lambda *args, **kwargs: SimpleNamespace(
            method=repaired, report={"accepted": True, "transcript": []}),
    )
    handler = MethodAgentHandler(spec_store=_Store({method_ref: original}), llm_client=_LLM())
    client = _Client()
    asyncio.run(handler.handle_deliberation_requested(client, _Env({
        "method_ref": method_ref, "component_plan_audit": True,
        "component_plan_revision_feedback": "leaf failure originates in source bounds",
        "component_plan_failed_component": "leaf",
    })))
    assert any(event["event_type"] == "method.deliberated" for event in client.published)
    assert not any(event["event_type"] == "agent.failed" for event in client.published)


def test_component_plan_revision_cannot_drop_unrelated_confirmed_components(monkeypatch):
    method_ref = "method://s1/confirmed"
    first = {
        "component_id": "index", "name": "index", "type": "indirect_gain",
        "depends_on": [], "acceptance_requires": [],
        "purpose": "build the index", "inputs": ["documents"],
        "outputs": ["index"], "mechanism": "construct postings",
        "expected_effect": "valid postings", "functional_check": "errors == 0",
        "implementation_check": "postings exist", "optimization_space": "reduce build cost",
        "downstream_contract": "provide postings",
    }
    second = {
        **first, "component_id": "pool", "name": "pool",
        "purpose": "reuse scratch", "outputs": ["pooled scratch"],
        "mechanism": "pool buffers", "downstream_contract": "provide pooled scratch",
    }
    original = {"approach": "index and pool", "components": [first, second]}
    monkeypatch.setattr(
        "autoresearch.agents.method_agent.handler.deliberation_core.deliberate_method",
        lambda *args, **kwargs: SimpleNamespace(
            method={**original, "components": [second]}, report={"accepted": True}),
    )
    store = _Store({method_ref: original})
    client = _Client()
    handler = MethodAgentHandler(spec_store=store, llm_client=_LLM())

    asyncio.run(handler.handle_deliberation_requested(client, _Env({
        "method_ref": method_ref, "component_plan_audit": True,
        "component_plan_revision_feedback": "Repair only pool's dependency boundary",
    })))

    assert not any(kind == "method" for kind, _ in store.puts)
    failed = [item for item in client.published if item["event_type"] == "agent.failed"]
    assert failed[0]["payload"]["underlying_reason_code"] == "component_plan_audit_failed"
    assert "omitted existing components" in failed[0]["payload"]["underlying_reason"]


def test_component_plan_revision_can_explicitly_merge_confirmed_components(monkeypatch):
    method_ref = "method://s1/confirmed"
    first = {
        "component_id": "index", "name": "index", "type": "indirect_gain",
        "depends_on": [], "acceptance_requires": [],
        "purpose": "build the index", "inputs": ["documents"],
        "outputs": ["index"], "mechanism": "construct postings",
        "expected_effect": "valid postings", "functional_check": "errors == 0",
        "implementation_check": "postings exist", "optimization_space": "reduce build cost",
        "downstream_contract": "provide postings",
    }
    second = {
        **first, "component_id": "pool", "name": "pool", "depends_on": ["index"],
        "acceptance_requires": ["index"], "purpose": "reuse scratch",
        "inputs": ["index"], "outputs": ["pooled index"],
        "mechanism": "pool buffers", "downstream_contract": "provide pooled index",
    }
    original = {"approach": "indexed pooling", "components": [first, second]}
    merged = {
        **first, "component_id": "indexed-pool", "name": "indexed pool",
        "purpose": "construct postings with pooled buffers",
        "mechanism": "construct postings and pool their scratch buffers",
        "merged_from": ["index", "pool"],
    }
    monkeypatch.setattr(
        "autoresearch.agents.method_agent.handler.deliberation_core.deliberate_method",
        lambda *args, **kwargs: SimpleNamespace(
            method={**original, "components": [merged]}, report={"accepted": True}),
    )
    store = _Store({method_ref: original})
    client = _Client()
    handler = MethodAgentHandler(spec_store=store, llm_client=_LLM())

    asyncio.run(handler.handle_deliberation_requested(client, _Env({
        "method_ref": method_ref, "component_plan_audit": True,
        "component_plan_revision_feedback": "Merge components that cannot be tested separately",
    })))

    assert [item["event_type"] for item in client.published] == ["method.deliberated"]
    assert [c["component_id"] for kind, obj in store.puts if kind == "method"
            for c in obj["components"]] == ["indexed-pool"]

    # A single known prior ID is unambiguous even when the model emits it as
    # a scalar instead of a one-item list.
    scalar = {**first, "merged_from": "index"}
    monkeypatch.setattr(
        "autoresearch.agents.method_agent.handler.deliberation_core.deliberate_method",
        lambda *args, **kwargs: SimpleNamespace(
            method={**original, "components": [scalar, second]}, report={"accepted": True}),
    )
    scalar_store = _Store({method_ref: original})
    scalar_client = _Client()
    asyncio.run(MethodAgentHandler(spec_store=scalar_store, llm_client=_LLM())
                .handle_deliberation_requested(scalar_client, _Env({
                    "method_ref": method_ref, "component_plan_audit": True,
                    "component_plan_revision_feedback": "Preserve the confirmed components",
                })))
    assert scalar_client.published[-1]["event_type"] == "method.deliberated"
    assert [c["merged_from"] for kind, obj in scalar_store.puts if kind == "method"
            for c in obj["components"] if c["component_id"] == "index"] == [["index"]]

    # A resumed partial draft cannot bypass the omission guard merely because
    # its retry payload lacks an explicit revision-feedback string.
    incomplete = {key: value for key, value in merged.items() if key != "merged_from"}
    checkpoint_ref = "method_debate_report://s1/partial"
    checkpoint = {"confirmed_method_ref": method_ref,
                  "draft_components": [incomplete], "transcript": []}
    monkeypatch.setattr(
        "autoresearch.agents.method_agent.handler.deliberation_core.deliberate_method",
        lambda *args, **kwargs: SimpleNamespace(
            method={**original, "components": [incomplete]}, report={"accepted": True}),
    )
    resumed_store = _Store({method_ref: original, checkpoint_ref: checkpoint})
    resumed_client = _Client()
    resumed_handler = MethodAgentHandler(spec_store=resumed_store, llm_client=_LLM())
    asyncio.run(resumed_handler.handle_deliberation_requested(resumed_client, _Env({
        "method_ref": method_ref, "component_plan_audit": True,
        "component_plan_checkpoint_ref": checkpoint_ref,
    })))
    assert resumed_client.published[-1]["event_type"] == "agent.failed"
    assert "omitted existing components" in resumed_client.published[-1]["payload"]["underlying_reason"]


def test_component_plan_retry_restores_omitted_contracts_to_feedback(monkeypatch):
    method_ref = "method://s1/confirmed"
    checkpoint_ref = "method_debate_report://s1/partial"
    original = {"approach": "indexed pooling", "components": [
        {"component_id": "index", "purpose": "construct postings", "depends_on": []},
        {"component_id": "pool", "purpose": "reuse scratch", "depends_on": ["index"]},
    ]}
    checkpoint = {"confirmed_method_ref": method_ref, "draft_components": [
        {"component_id": "pooled-index", "purpose": "combined implementation"}],
        "transcript": []}
    seen = []

    def debate(*args, **kwargs):
        seen.append((args[3], kwargs["confirmed_component_ids"],
                     kwargs["component_plan_revision_feedback"]))
        return SimpleNamespace(method=args[3], report={"accepted": False, "transcript": []})

    monkeypatch.setattr(
        "autoresearch.agents.method_agent.handler.deliberation_core.deliberate_method", debate)
    store = _Store({method_ref: original, checkpoint_ref: checkpoint})
    handler = MethodAgentHandler(spec_store=store, llm_client=_LLM())
    asyncio.run(handler.handle_deliberation_requested(_Client(), _Env({
        "method_ref": method_ref, "component_plan_audit": True,
        "component_plan_checkpoint_ref": checkpoint_ref,
    })))

    assert [c["component_id"] for c in seen[0][0]["components"]] == ["pooled-index"]
    assert seen[0][1] == {"index", "pool"}
    assert "construct postings" in seen[0][2]
    assert "reuse scratch" in seen[0][2]
    assert "merged_from" in seen[0][2]
    assert '"index", "pool"' in seen[0][2]
    assert "interim drafts" in seen[0][2]


def test_method_prompt_requires_evidence_grounded_implementable_components():
    llm = _LLM(["bounded concrete mechanism"])

    _run({"problem_model_ref": "problem_model://s1"}, llm)

    assert "complete implementable mechanism" in llm.last_system
    assert "Do not invent literature support" in llm.last_system
    assert "observed run horizon" in llm.last_system


def test_normal_generation_has_no_swap_instruction():
    llm = _LLM()
    _run({"problem_model_ref": "problem_model://s1"}, llm)          # no method_not_working
    assert "FUNDAMENTALLY DIFFERENT" not in llm.last_user


def test_generated_revised_method_starts_a_fresh_operator_lineage(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTORESEARCH_WEB_WORKDIR", str(tmp_path))
    board = board_for_session("s1")
    board.begin_method("sha256:prior")
    board.add_card(ExperienceCard(
        round=1, operator="check-expectation", method_family="failed-family",
        metric_value=1.2, valid=True, verdict="valid_not_best"))

    _run({"problem_model_ref": "problem_model://s1"}, _LLM())

    assert board_for_session("s1").cards() == []
    assert "failed-family(best=1.2)" in board_for_session("s1").summarize_for_regen()


def test_method_failure_matches_real_eventbus_contract():
    handler = MethodAgentHandler(spec_store=_Store({}), llm_client=_LLM())
    client = _Client()

    asyncio.run(handler._fail(client, "s1", "e1", "transient", "bad_output", "invalid JSON"))

    event = client.published[-1]
    root = Path(__file__).resolve().parents[3] / "eventbus"
    registry = Registry(root / "eventbus-registry.yaml")
    assert registry.is_legitimate_emitter(event["event_type"], "MethodAgent")
    assert list(
        registry.get_payload_validator(event["event_type"], 1).iter_errors(event["payload"])
    ) == []


def test_success_feedback_allows_additive_revision_and_preserves_valid_family(monkeypatch):
    prior_ref = 'method://s1/prior'
    memory_ref = 'research_memory://s1/current'
    prior = {'approach': 'retain measured mechanism', 'changes_outline': [],
             'why_it_should_win': '', 'assumptions': []}
    llm = _LLM([prior['approach'], prior['approach'] + ' plus conditional correction'])
    store = _Store({'problem_model://s1': {'research_goal': 'reduce latency'},
                    prior_ref: prior, memory_ref: {'diagnosis': 'method_improvement_proposed'}})
    handler = MethodAgentHandler(spec_store=store, llm_client=llm)
    monkeypatch.setattr(handler, '_search_history', lambda _sid: 'measured legal improvement')
    client = _Client()
    asyncio.run(handler.handle_revise_requested(client, _Env({
        'problem_model_ref': 'problem_model://s1', 'method_ref': prior_ref,
        'research_memory_ref': memory_ref,
        'experiment_feedback': 'Keep A, add conditional B; recheck integration.'})))
    assert llm.calls == 2  # unchanged first draft still needs a real revision
    assert 'A failed candidate does not by itself reject the whole prior Method' in llm.last_user
    assert 'complete replacement of a refuted core is allowed' in llm.last_user
    assert 'Do not keep prior components merely for continuity' in llm.last_user
    assert 'a testable extension B to evidence-supported Method A' in llm.last_user
    assert 'a set of components with explicit dependencies' in llm.last_user
    assert 'compare A with A+B' in llm.last_user
    assert 'do not replace it merely to look different' in llm.last_user
    assert 'previously unsuccessful method family' not in llm.last_user
    assert 'Discard it and propose' not in llm.last_user
    assert '避开这些已试且未取胜' not in llm.last_user
    assert [p['event_type'] for p in client.published] == ['method.generated']
    assert store.puts[-1][1]['approach'] == prior['approach'] + ' plus conditional correction'


def test_second_revision_receives_the_first_revised_method_and_task_contract(monkeypatch):
    prior_ref = 'method://s1/v1'
    baseline_ref = 'baseline_result://s1/v1'
    first = {'approach': 'Method A', 'changes_outline': ['A'],
             'why_it_should_win': 'measured A gain', 'assumptions': []}
    llm = _LLM(['Method A plus extension B', 'Method A plus B plus extension C'])
    store = _Store({'problem_model://s1': {'research_goal': 'reduce loss'},
                    prior_ref: first,
                    baseline_ref: {'metric_name': 'loss',
                                   'repository_contract': {'sources': [{
                                       'path': 'instruction.md', 'content': 'EDIT_SOLVE_ONLY'}]}}})
    handler = MethodAgentHandler(spec_store=store, llm_client=llm)
    monkeypatch.setattr(handler, '_search_history', lambda _sid: '')
    client = _Client()
    asyncio.run(handler.handle_revise_requested(client, _Env({
        'problem_model_ref': 'problem_model://s1', 'method_ref': prior_ref,
        'baseline_result_ref': baseline_ref, 'experiment_feedback': 'Extend supported A.'})))
    revised_ref = client.published[-1]['payload']['method_ref']
    store._objs[revised_ref] = store.puts[-1][1]
    assert store._objs[revised_ref]['approach'] == 'Method A plus extension B'

    asyncio.run(handler.handle_revise_requested(client, _Env({
        'problem_model_ref': 'problem_model://s1', 'method_ref': revised_ref,
        'baseline_result_ref': baseline_ref, 'experiment_feedback': 'Extend supported A+B.'})))
    assert '"approach": "Method A plus extension B"' in llm.last_user
    assert 'EDIT_SOLVE_ONLY' in llm.last_user
    assert store.puts[-1][1]['approach'] == 'Method A plus B plus extension C'


def test_generation_consumes_local_repository_facts_before_baseline(tmp_path, monkeypatch):
    monkeypatch.setenv('AUTORESEARCH_EXECUTION_STRATEGY', 'efficient')
    (tmp_path / 'README.md').write_text('Use evaluator.py; latency in milliseconds; freeze validation split.')
    llm = _LLM()
    client = _run({'problem_model_ref': 'problem_model://s1', 'repository_context_request': {
        'repo_url': str(tmp_path), 'question': 'task'}}, llm)
    assert 'evaluator.py' in llm.last_user
    assert 'not a baseline' in llm.last_user
    assert 'EFFICIENT EXECUTION' in llm.last_user
    assert any(p['event_type'] == 'method.generated' for p in client.published)


@pytest.mark.parametrize('entry', ['generation', 'deliberation'])
@pytest.mark.parametrize('edited_rules', [False, True])
@pytest.mark.parametrize('captured_rules', [False, True])
def test_legacy_baseline_recovers_only_commit_pinned_task_context(
        tmp_path, monkeypatch, entry, edited_rules, captured_rules):
    import subprocess
    import hashlib
    from copy import deepcopy

    rule = 'Edit candidate.py only; score with the unchanged evaluator.'
    (tmp_path / 'instruction.md').write_text(rule)
    subprocess.run(['git', 'init', '-q', str(tmp_path)], check=True)
    subprocess.run(['git', '-C', str(tmp_path), 'add', 'instruction.md'], check=True)
    subprocess.run(['git', '-C', str(tmp_path), '-c', 'user.name=Test',
                    '-c', 'user.email=test@example.invalid', 'commit', '-qm', 'task'], check=True)
    commit = subprocess.check_output(
        ['git', '-C', str(tmp_path), 'rev-parse', 'HEAD'], text=True).strip()
    baseline = {'metric_name': 'latency', 'metric_value': 5.0, 'source_kind': 'live_repo',
                'repo_commit': commit, 'artifacts': [str(tmp_path / 'ar_result.json')],
                'repository_contract': {'sources': [], 'execution_contract': {}}}
    if captured_rules:
        baseline['repository_contract']['sources'] = [{'path': 'instruction.md',
            'content': rule, 'sha256': hashlib.sha256(rule.encode()).hexdigest()}]
    original = deepcopy(baseline)
    if edited_rules:
        (tmp_path / 'instruction.md').write_text('The candidate may rewrite every file.')
    baseline_ref = 'baseline_result://s1/frozen'
    method_ref = 'method://s1/draft'
    store = _Store({'problem_model://s1': {'research_goal': 'Reduce latency'},
                    baseline_ref: baseline,
                    method_ref: {'approach': 'x', 'components': []}})
    llm, client, observed = _LLM(), _Client(), {}

    def deliberate(_problem, *_args, **kwargs):
        observed.update(kwargs['baseline_result'])
        return SimpleNamespace(method={'approach': 'x', 'components': []},
                               report={'converged': True, 'rounds': 1})

    monkeypatch.setattr(
        'autoresearch.agents.method_agent.handler.deliberation_core.deliberate_method',
        deliberate)
    handler = MethodAgentHandler(spec_store=store, llm_client=llm)
    payload = {'problem_model_ref': 'problem_model://s1', 'method_ref': method_ref,
               'baseline_result_ref': baseline_ref}
    if entry == 'generation':
        asyncio.run(handler.handle_generation_requested(client, _Env(payload)))
        assert (rule in llm.last_user) is (captured_rules or not edited_rules)
    else:
        asyncio.run(handler.handle_deliberation_requested(client, _Env(payload)))
        sources = observed['repository_contract']['sources']
        expected = [rule] if captured_rules or not edited_rules else []
        assert [s['content'] for s in sources] == expected
        assert observed['metric_value'] == 5.0
    assert store._objs[baseline_ref] == original


def test_deliberation_consumes_persisted_repository_facts_before_baseline(monkeypatch):
    monkeypatch.setenv('AUTORESEARCH_EXECUTION_STRATEGY', 'efficient')
    method_ref = 'method://s1/draft'
    audit_ref = refs.encode('audit', 's1', 'early-repository-context')
    store = _Store({
        method_ref: {'approach': 'x', 'components': []},
        'problem_model://s1': {'research_goal': 'reduce latency'},
        audit_ref: {'entry_point': 'evaluate.py', 'metric': 'latency_ms'},
    })
    observed = {}

    def deliberate(problem_model, *_args, **_kwargs):
        observed['problem_model'] = problem_model
        return SimpleNamespace(
            method={'approach': 'x', 'components': []},
            report={'converged': True, 'rounds': 1},
        )

    monkeypatch.setattr(
        'autoresearch.agents.method_agent.handler.deliberation_core.deliberate_method',
        deliberate,
    )
    client = _Client()
    handler = MethodAgentHandler(spec_store=store, llm_client=_LLM())
    asyncio.run(handler.handle_deliberation_requested(client, _Env({
        'method_ref': method_ref,
        'problem_model_ref': 'problem_model://s1',
    })))

    assert observed['problem_model']['repository_facts'] == {
        'entry_point': 'evaluate.py', 'metric': 'latency_ms'}
    assert [event['event_type'] for event in client.published] == ['method.deliberated']


@pytest.mark.parametrize("newer_failure", [False, True])
@pytest.mark.parametrize("problem", [None, "pending", "checkpoint", "parent", "measurement",
    "attempt", "contract", "gain", "comparison", "task", "core"])
def test_reviewed_component_local_gain_keeps_its_source_for_revalidation(monkeypatch, problem, newer_failure):
    from copy import deepcopy
    from autoresearch.agents.memory_agent.tests.test_incremental_method_updates import component_observation
    from autoresearch.spec_store.identity import artifact_digest

    baseline = {"metric_value": 3.0}
    baseline_ref = refs.encode("baseline_result", "s1", "snapshot-" + artifact_digest(baseline).split(":", 1)[1])
    result = deepcopy(component_observation())
    result["baseline_digest"] = artifact_digest(baseline)
    if problem == "parent":
        result["component_execution"]["attempt_contract_result"]["parent_evidence"]["host_bound"] = False
    elif problem == "measurement":
        result["measurement_evidence"]["valid"] = False
    elif problem == "attempt":
        result["candidate_admissibility"]["attempt_id"] = "attempt-other"
    elif problem == "contract":
        result["component_execution"]["attempt_contract_result"]["passed"] = False
    elif problem == "gain":
        result["component_execution"]["attempt_contract_result"]["parent_metric"] = 1.0
    elif problem == "comparison":
        result["repository_comparison_review"]["verdict"] = "fail"
    elif problem == "task":
        result["method_implementation_review"]["task_contract_review"]["verdict"] = "violation"
    result_ref = refs.encode("gen_experiment_result", "s1", "snapshot-" + artifact_digest(result).split(":", 1)[1])
    review_ref, method_ref = "review://s1/current", "method://s1/prior"
    memory = {"diagnosis": "method_improvement_proposed", "diagnosis_evidence": [review_ref]}
    review = {"status": "pending" if problem == "pending" else "completed",
        "trigger": "component_method_reconciliation", "method_ref": method_ref,
        "baseline_ref": baseline_ref, "result_ref": result_ref,
        "off_method_evidence": {"candidate": {"result_ref": result_ref,
            "checkpoint": "/other" if problem == "checkpoint" else result["checkpoint"]}},
        "decision": {"route": "revise_method", "requires_core_reconstruction": problem == "core"}}
    prior = {"approach": "retain the measured mechanism", "changes_outline": []}
    store = _Store({baseline_ref: baseline, result_ref: result, review_ref: review,
        method_ref: prior, "memory://s1": memory, "problem_model://s1": {}})
    if newer_failure:
        later_ref = "review://s1/later-failure"
        store._objs[later_ref] = {"status": "completed", "trigger": "implementation_failure",
            "decision": {"route": "revise_method"}}
        memory["method_feedback_refs"] = [review_ref, later_ref]
        memory["diagnosis_evidence"] = [later_ref]
    handler = MethodAgentHandler(spec_store=store, llm_client=_LLM())
    candidate = handler._latest_admissible_reconciliation_candidate(memory)
    assert (candidate is not None) is (problem is None)
    if problem is not None:
        return
    monkeypatch.setattr("autoresearch.agents.method_agent.handler.method_core.method_step",
        lambda *args, **kwargs: {**prior, "reconciled_result_ref": result_ref})
    client = _Client()
    asyncio.run(handler.handle_revise_requested(client, _Env({
        "problem_model_ref": "problem_model://s1", "method_ref": method_ref,
        "baseline_result_ref": baseline_ref, "research_memory_ref": "memory://s1",
        "experiment_feedback": "Reconcile and formally evaluate the retained implementation."})))
    assert store.puts[-1][1]["reconciled_result_ref"] == result_ref
    assert result["accepted"] is False
    from autoresearch.experiments.repo_improve.tests.test_component_plan import _component
    revised_ref = "method://s1/reconciled"
    component = {**_component("retained"), "ablation_test": "signed_metric_improvement > 0"}
    store._objs[revised_ref] = {**store.puts[-1][1], "components": [component]}
    def deliberate(*args, **kwargs):
        assert result_ref in kwargs["component_plan_revision_feedback"]
        return SimpleNamespace(method=args[3], report={"accepted": True})
    monkeypatch.setattr("autoresearch.agents.method_agent.handler.deliberation_core.deliberate_method", deliberate)
    asyncio.run(handler.handle_deliberation_requested(client, _Env({
        "method_ref": revised_ref, "research_memory_ref": "memory://s1",
        "baseline_result_ref": baseline_ref, "component_plan_audit": True})))
    assert [obj for kind, obj in store.puts if kind == "method"][-1]["reconciled_result_ref"] == result_ref


@pytest.mark.parametrize("problem", [None, "checkpoint", "baseline", "unaccepted", "parent", "task", "core",
    "two_candidates", "explicit_choice", "descriptive_choice"])
def test_qualified_component_closeout_binds_its_measured_source(monkeypatch, problem):
    from copy import deepcopy
    from autoresearch.agents.memory_agent.tests.test_incremental_method_updates import component_observation
    from autoresearch.spec_store.identity import artifact_digest
    baseline = {"metric_value": 3.0, "metric_name": "latency_ms", "metric_unit": "ms",
                "lower_is_better": True}
    baseline_ref = refs.encode("baseline_result", "s1", "snapshot-" + artifact_digest(baseline).split(":", 1)[1])
    result = deepcopy(component_observation())
    result.update(accepted=problem != "unaccepted", baseline_digest=artifact_digest(baseline),
                  run_record_snapshot_ref="spec://run_record/s1/snapshot-qualified")
    result["metric_name"], result["metric_unit"] = "latency_ms", "ms"
    result["method_implementation_review"].update(verdict="faithful")
    result["component_execution"]["status"] = "qualified"
    if problem == "baseline":
        result["baseline_digest"] = "other"
    elif problem == "parent":
        result["component_execution"]["attempt_contract_result"]["parent_evidence"]["host_bound"] = False
    elif problem == "task":
        result["method_implementation_review"]["task_contract_review"]["verdict"] = "violation"
    result_ref = refs.encode("gen_experiment_result", "s1", "snapshot-" + artifact_digest(result).split(":", 1)[1])
    prior_ref, review_ref = "method://s1/prior", "review://s1/closing"
    prior = {"approach": "retain the measured component", "changes_outline": []}
    review = {"status": "completed", "trigger": "qualified_component_closeout",
        "method_ref": prior_ref, "baseline_ref": baseline_ref,
        "decision": {"route": "revise_method", "requires_core_reconstruction": problem == "core"},
        "qualified_closeout_components": [{"run_ref": result["run_record_snapshot_ref"],
            "checkpoint": "/wrong" if problem == "checkpoint" else result["checkpoint"]}]}
    memory = {"diagnosis": "method_improvement_proposed", "diagnosis_evidence": [review_ref]}
    store = _Store({prior_ref: prior, baseline_ref: baseline, result_ref: result,
        review_ref: review, "memory://s1": memory, "problem_model://s1": {}})
    resolved = {result["run_record_snapshot_ref"]: {**result, "candidate_snapshot_ref": result_ref}}
    expected_ref = result_ref
    if problem in {"two_candidates", "explicit_choice"}:
        other = deepcopy(result)
        other.update(checkpoint="/qualified-B", method_metric=2.5,
                     run_record_snapshot_ref="spec://run_record/s1/snapshot-other")
        other["measurement_evidence"]["value"] = 2.5
        contract = other["component_execution"]["attempt_contract_result"]
        contract.update(candidate_metric=2.5, actual=0.5, measurement_evidence=other["measurement_evidence"])
        other_ref = refs.encode("gen_experiment_result", "s1", "snapshot-" + artifact_digest(other).split(":", 1)[1])
        store._objs[other_ref] = other
        resolved[other["run_record_snapshot_ref"]] = {**other, "candidate_snapshot_ref": other_ref}
        review["qualified_closeout_components"].append({"run_ref": other["run_record_snapshot_ref"],
                                                       "checkpoint": other["checkpoint"]})
        if problem == "explicit_choice":
            review["decision"]["reuse"] = [other["checkpoint"]]
            expected_ref = other_ref
    elif problem == "descriptive_choice":
        review["decision"]["reuse"] = [
            "Reuse the already qualified retained implementation and official evaluator."]
    monkeypatch.setattr("autoresearch.spec_store.identity.result_for_run",
        lambda _store, run_ref: resolved[run_ref])
    handler = MethodAgentHandler(spec_store=store, llm_client=_LLM())
    candidate = handler._latest_admissible_reconciliation_candidate(memory)
    valid = problem in {None, "two_candidates", "explicit_choice", "descriptive_choice"}
    assert (candidate is not None) is valid
    if not valid:
        return
    assert candidate["result_ref"] == expected_ref
    monkeypatch.setattr("autoresearch.agents.method_agent.handler.method_core.method_step",
        lambda *args, **kwargs: {**prior, "reconciled_result_ref": expected_ref})
    asyncio.run(handler.handle_revise_requested(_Client(), _Env({
        "problem_model_ref": "problem_model://s1", "method_ref": prior_ref,
        "baseline_result_ref": baseline_ref, "research_memory_ref": "memory://s1",
        "incremental_revision": True, "experiment_feedback": "Finalize the measured component."})))
    assert [obj for kind, obj in store.puts if kind == "method"][-1]["reconciled_result_ref"] == expected_ref


@pytest.mark.parametrize("observed_metric", [None, 2.75])
@pytest.mark.parametrize("newer_whole_gain", [False, True])
def test_qualified_indirect_component_closeout_binds_source_without_claiming_score(
        monkeypatch, observed_metric, newer_whole_gain):
    from autoresearch.experiments.repo_improve.runtime_evidence import (
        can_revalidate_reconciliation_candidate,
    )
    from autoresearch.spec_store.identity import artifact_digest

    prior_ref = "method://s1/prior"
    run_ref = "spec://run_record/s1/snapshot-indirect"
    checkpoint = "/qualified-indirect"
    attempt_id = "attempt-indirect"
    baseline = {"metric_value": 3.0, "metric_name": "score", "metric_unit": "ratio",
                "lower_is_better": False}
    baseline_ref = refs.encode(
        "baseline_result", "s1", "snapshot-" + artifact_digest(baseline).split(":", 1)[1])
    result = {
        "accepted": False,
        "checkpoint": checkpoint,
        "method_metric": observed_metric,
        "method_ref": prior_ref,
        "metric_name": "score",
        "metric_unit": "ratio",
        "lower_is_better": False,
        "baseline_digest": artifact_digest(baseline),
        "run_record_snapshot_ref": run_ref,
        "candidate_admissibility": {
            "attempt_id": attempt_id, "verdict": "insufficient_evidence"},
        **({
            "measurement_evidence": {
                "value": observed_metric, "valid": True, "attempt_id": attempt_id,
                "stdout_sha256": "observed-output", "protocol": {"kind": "component_local"}},
            "repository_comparison_review": {"verdict": "pass"},
        } if observed_metric is not None else {}),
        "method_implementation_review": {
            "verdict": "faithful", "method_ref": prior_ref,
            "task_contract_review": {"verdict": "compliant"}},
        "component_execution": {
            "mode": "component", "role": "indirect_gain", "status": "qualified",
            "execution_succeeded": True, "checkpoint": checkpoint,
            "attempt_checkpoint": checkpoint, "became_best": True,
            "attempt_contract_result": {
                "passed": True,
                "evidence": {"source": "runtime_attempt", "ref": attempt_id,
                             "sha256": "abc123"},
                **({"measurement_evidence": {
                    "value": observed_metric, "valid": True, "attempt_id": attempt_id,
                    "stdout_sha256": "observed-output", "protocol": {"kind": "component_local"},
                }} if observed_metric is not None else {}),
                "measurement_review": {"verdict": "pass"},
            },
        },
    }
    result_ref = refs.encode(
        "gen_experiment_result", "s1",
        "snapshot-" + artifact_digest(result).split(":", 1)[1])
    assert not can_revalidate_reconciliation_candidate(result)
    assert can_revalidate_reconciliation_candidate(result, allow_qualified=True)
    unbound = {**result, "method_ref": None, "method_implementation_review": {
        **result["method_implementation_review"], "method_ref": None}}
    assert not can_revalidate_reconciliation_candidate(unbound, allow_qualified=True)
    review_ref = "review://s1/closing-indirect"
    review = {
        "status": "completed", "trigger": "qualified_component_closeout",
        "method_ref": prior_ref, "baseline_ref": baseline_ref,
        "decision": {"route": "revise_method", "requires_core_reconstruction": False,
                     "reuse": [checkpoint]},
        "qualified_closeout_components": [
            {"run_ref": run_ref, "checkpoint": checkpoint}],
    }
    memory = {"diagnosis": "method_improvement_proposed",
              "diagnosis_evidence": [review_ref]}
    prior = {"approach": "retain the qualified indirect component", "changes_outline": []}
    store = _Store({prior_ref: prior, baseline_ref: baseline, result_ref: result,
                    review_ref: review, "memory://s1": memory, "problem_model://s1": {}})
    monkeypatch.setattr(
        "autoresearch.spec_store.identity.result_for_run",
        lambda _store, _run_ref: {**result, "candidate_snapshot_ref": result_ref},
    )
    handler = MethodAgentHandler(spec_store=store, llm_client=_LLM())
    if newer_whole_gain:
        whole = {**result, "checkpoint": "/official-main", "method_metric": 4.5,
                 "component_execution": None,
                 "off_method_observation": {"valid": True, "improved": True},
                 "method_implementation_review": {"verdict": "drifted",
                     "task_contract_review": {"verdict": "compliant"}}}
        whole_ref = refs.encode("gen_experiment_result", "s1",
            "snapshot-" + artifact_digest(whole).split(":", 1)[1])
        newer_review_ref = "review://s1/whole-gain"
        store._objs[whole_ref] = whole
        store._objs[newer_review_ref] = {**review, "trigger": "legal_off_method_gain",
            "result_ref": whole_ref, "off_method_evidence": {
                "candidate": {"result_ref": whole_ref, "checkpoint": whole["checkpoint"]}}}
        memory["reconciliation_candidates"] = [{"result_ref": whole_ref, "checkpoint": whole["checkpoint"]}]
        memory["method_feedback_refs"] = [review_ref, newer_review_ref]
        memory["diagnosis_evidence"] = [newer_review_ref]
        assert handler._latest_admissible_reconciliation_candidate(memory) == {
            "result_ref": whole_ref, "checkpoint": whole["checkpoint"]}
        return
    candidate = handler._latest_admissible_reconciliation_candidate(memory)
    assert candidate == {
        "result_ref": result_ref, "checkpoint": checkpoint, "metric": observed_metric}
    monkeypatch.setattr(
        "autoresearch.agents.method_agent.handler.method_core.method_step",
        lambda *args, **kwargs: {**prior, "reconciled_result_ref": result_ref},
    )
    asyncio.run(handler.handle_revise_requested(_Client(), _Env({
        "problem_model_ref": "problem_model://s1", "method_ref": prior_ref,
        "baseline_result_ref": baseline_ref, "research_memory_ref": "memory://s1",
        "incremental_revision": True,
        "experiment_feedback": "Finalize the qualified indirect component.",
    })))
    saved = [obj for kind, obj in store.puts if kind == "method"][-1]
    assert saved["reconciled_result_ref"] == result_ref
    assert result["method_metric"] == observed_metric


@pytest.mark.parametrize("case", ["valid", "method", "baseline", "unbound", "component",
    "admissibility", "measurement", "attempt", "task", "unaccepted", "core", "no_prior_binding"])
def test_whole_result_revision_advances_the_retained_source(monkeypatch, case):
    from autoresearch.spec_store.identity import artifact_digest
    prior_ref, old_ref = "method://s1/prior", "result://s1/old"
    prior = {"approach": "retained mechanism", "reconciled_result_ref": old_ref}
    if case == "no_prior_binding":
        prior.pop("reconciled_result_ref")
    baseline = {"metric_value": 3.0}
    baseline_ref = refs.encode("baseline_result", "s1", "snapshot-" + artifact_digest(baseline).split(":", 1)[1])
    result = {"accepted": True, "checkpoint": "/new-whole", "method_metric": 1.0,
        "method_ref": "method://other" if case == "method" else prior_ref,
        "baseline_digest": "other" if case == "baseline" else artifact_digest(baseline),
        "measurement_evidence": {"value": 1.0, "valid": True, "attempt_id": "attempt-whole",
            "stdout_sha256": "a" * 64, "protocol_fingerprint": "b" * 64,
            "protocol": {"kind": "official_single"}},
        "candidate_admissibility": {"verdict": "fail" if case == "admissibility" else "pass",
            "attempt_id": "attempt-whole"},
        "repository_comparison_review": {"verdict": "pass"},
        "method_implementation_review": {"verdict": "faithful", "method_ref": prior_ref,
            "task_contract_review": {"verdict": "compliant"}}}
    if case == "component":
        result["component_execution"] = {"mode": "component"}
    elif case == "measurement":
        result["measurement_evidence"]["value"] = 2.0
    elif case == "attempt":
        result["candidate_admissibility"]["attempt_id"] = "attempt-other"
    elif case == "task":
        result["method_implementation_review"]["task_contract_review"]["verdict"] = "violation"
    elif case == "unaccepted":
        result["accepted"] = False
    result_ref = refs.encode("gen_experiment_result", "s1", "snapshot-" + artifact_digest(result).split(":", 1)[1])
    if case == "unbound":
        result_ref = "result://s1/current"
    review = {"status": "completed", "trigger": "complete_result", "method_ref": prior_ref,
        "baseline_ref": baseline_ref, "result_ref": result_ref,
        "decision": {"route": "revise_method", "requires_core_reconstruction": case == "core"}}
    store = _Store({prior_ref: prior, baseline_ref: baseline, result_ref: result,
        "review://s1": review, "problem_model://s1": {}, old_ref: {},
        "memory://s1": {"diagnosis": "method_improvement_proposed", "diagnosis_evidence": ["review://s1"]}})
    handler = MethodAgentHandler(spec_store=store, llm_client=_LLM())
    monkeypatch.setattr(handler, "_latest_admissible_reconciliation_candidate", lambda _: {"result_ref": old_ref})
    monkeypatch.setattr("autoresearch.agents.method_agent.handler.method_core.method_step",
        lambda *args, **kwargs: {**prior, "approach": "extend the retained mechanism"})
    asyncio.run(handler.handle_revise_requested(_Client(), _Env({
        "problem_model_ref": "problem_model://s1", "method_ref": prior_ref,
        "baseline_result_ref": baseline_ref, "research_memory_ref": "memory://s1",
        "incremental_revision": True, "experiment_feedback": "Extend the verified whole result."})))
    saved = [obj for kind, obj in store.puts if kind == "method"][-1]
    # A new Method may use the old result as comparison evidence, but the old source does
    # not implement the new treatment and therefore cannot remain its exact-source binding.
    assert "reconciled_result_ref" not in saved


@pytest.mark.parametrize("case", ["retained", "omitted", "method", "baseline", "scope",
    "not_required", "no_report", "no_prior_binding", "mechanism", "settings",
    "other_result", "unavailable_source"])
def test_contract_revision_retains_only_the_same_measured_treatment(monkeypatch, case):
    from autoresearch.experiments.repo_improve.tests.test_component_plan import _component

    prior_ref, source_ref = "method://s1/reconciled", "result://s1/retained"
    prior = {"approach": "retained measured treatment", "changes_outline": ["retained operation"],
             "components": [_component("retained")], "reconciled_result_ref": source_ref}
    if case == "no_prior_binding":
        prior.pop("reconciled_result_ref")
    draft = json.loads(json.dumps(prior))
    draft["approach"] += "; clarify the evaluation contract"
    draft["components"][0]["downstream_contract"] = "measure with the unchanged official evaluator"
    if case == "omitted":
        draft.pop("reconciled_result_ref", None)
    elif case == "mechanism":
        draft["components"][0]["mechanism"] = "a different treatment"
    elif case == "settings":
        draft["components"][0]["optimization_space"] = "a different search space"
    elif case == "other_result":
        draft["reconciled_result_ref"] = "result://s1/other"
    report_ref = refs.encode("method_debate_report", "s1", "contract-repair")
    report = {"scope": "other" if case == "scope" else "component_plan",
              "confirmed_method_ref": "method://other" if case == "method" else prior_ref,
              "baseline_result_ref": "baseline://other" if case == "baseline" else "baseline://s1",
              "method_revision_required": case != "not_required"}
    store = _Store({prior_ref: prior, report_ref: report, "problem_model://s1": {},
        source_ref: {"checkpoint": "/retained", "method_metric": 1.0},
        "memory://s1": {"diagnosis": "method_improvement_proposed",
                         "diagnosis_evidence": ["review://s1/old"]},
        "review://s1/old": {"status": "completed", "trigger": "component_method_reconciliation",
            "method_ref": "method://s1/before-reconciliation", "baseline_ref": "baseline://s1",
            "decision": {"route": "revise_method"}}})
    handler = MethodAgentHandler(spec_store=store, llm_client=_LLM())
    monkeypatch.setattr(handler, "_latest_admissible_reconciliation_candidate",
        lambda _: None if case == "unavailable_source" else {"result_ref": source_ref})
    seen = []
    def generate(*args, **kwargs):
        seen.append(kwargs)
        return json.loads(json.dumps(draft))
    monkeypatch.setattr("autoresearch.agents.method_agent.handler.method_core.method_step", generate)
    payload = {"problem_model_ref": "problem_model://s1", "method_ref": prior_ref,
               "baseline_result_ref": "baseline://s1", "research_memory_ref": "memory://s1",
               "incremental_revision": True, "experiment_feedback": "Repair the bound CAP contract."}
    if case != "no_report":
        payload["component_plan_checkpoint_ref"] = report_ref
    asyncio.run(handler.handle_revise_requested(_Client(), _Env(payload)))
    saved = [obj for kind, obj in store.puts if kind == "method"][-1]
    if case in {"retained", "omitted"}:
        assert seen[0]["reconciliation_result_ref"] == source_ref
        assert saved["reconciled_result_ref"] == source_ref
    else:
        assert "reconciled_result_ref" not in saved


@pytest.mark.parametrize("binding", ["bound", "method", "baseline", "core",
    "complete_partial", "complete_faithful", "complete_other_method"])
def test_bound_local_implementation_repair_keeps_the_frozen_method(monkeypatch, binding):
    from autoresearch.experiments.repo_improve.tests.test_component_plan import _component
    prior_ref, baseline_ref = "method://s1/prior", "baseline://s1"
    prior = {"approach": "same mechanism; repair its implementation", "changes_outline": [],
             "components": [_component("retained")]}
    review_ref = "review://s1/repair"
    review = {"status": "completed", "trigger": "implementation_failure",
        "method_ref": "method://other" if binding == "method" else prior_ref,
        "baseline_ref": "baseline://other" if binding == "baseline" else baseline_ref,
        "decision": {"route": "revise_method", "requires_core_reconstruction": binding == "core"}}
    if binding.startswith("complete_"):
        review.update(trigger="complete_result", result_ref="result://s1/current")
    partial = {"method_ref": prior_ref, "method_implementation_review": {
        "verdict": "faithful" if binding == "complete_faithful" else "partial",
        "method_ref": "method://other" if binding == "complete_other_method" else prior_ref}}
    memory = {"diagnosis": "method_improvement_proposed", "diagnosis_evidence": [review_ref]}
    store = _Store({prior_ref: prior, baseline_ref: {}, "memory://s1": memory,
        review_ref: review, "problem_model://s1": {}, "result://s1/current": partial})
    repair_allowed = binding in {"bound", "complete_partial"}
    seen = []
    def step(*args, **kwargs):
        seen.append(kwargs)
        assert kwargs.get("retained_treatment_revision") is repair_allowed
        if repair_allowed:
            return {**prior, "components": [*prior["components"], _component(
                "artifact bookkeeping", role="indirect_gain")]}
        return dict(prior)
    monkeypatch.setattr("autoresearch.agents.method_agent.handler.method_core.method_step", step)
    client = _Client()
    asyncio.run(MethodAgentHandler(spec_store=store, llm_client=_LLM()).handle_revise_requested(
        client, _Env({"problem_model_ref": "problem_model://s1", "method_ref": prior_ref,
            "baseline_result_ref": baseline_ref, "research_memory_ref": "memory://s1",
            "experiment_feedback": "Repair the failed implementation without changing its mechanism.",
            "incremental_revision": True})))
    assert any(e["event_type"] == "method.generated" for e in client.published) is repair_allowed
    if repair_allowed:
        # A bound implementation repair is already carried by ResearchMemory to the
        # next plan.  Re-generating the scientific Method here can turn bookkeeping
        # (for example, preserving a reward file) into a spurious DAG component.
        assert seen == []
        assert store.puts[-1][1] == prior


@pytest.mark.parametrize("case", ["repair", "reconstruction", "invalid_prior", "provider_error"])
def test_local_implementation_repair_format_failure_retains_only_a_valid_prior(monkeypatch, case):
    from autoresearch.agents.method_agent.core import MethodError
    from autoresearch.experiments.repo_improve.tests.test_component_plan import _component
    prior_ref, baseline_ref = "method://s1/prior", "baseline://s1"
    prior = {"approach": "repair the existing implementation", "changes_outline": [],
             "components": [_component("retained")]}
    if case == "invalid_prior":
        prior["components"][0]["type"] = "indirect_gain"
        prior["components"][0]["functional_check"] = "the infrastructure works"
    review_ref = "review://s1/repair"
    review = {"status": "completed", "trigger": "implementation_failure",
        "method_ref": prior_ref, "baseline_ref": baseline_ref,
        "decision": {"route": "revise_method", "requires_core_reconstruction": case == "reconstruction"}}
    store = _Store({prior_ref: prior, baseline_ref: {}, "problem_model://s1": {},
        "memory://s1": {"diagnosis": "method_improvement_proposed", "diagnosis_evidence": [review_ref]},
        review_ref: review})
    def step(*args, **kwargs):
        if case == "provider_error":
            raise MethodError("transient", "method_llm_error", "provider unavailable")
        raise MethodError("permanent", "method_parse_error", "new infrastructure gate has no numeric threshold")
    monkeypatch.setattr("autoresearch.agents.method_agent.handler.method_core.method_step", step)
    client = _Client()
    asyncio.run(MethodAgentHandler(spec_store=store, llm_client=_LLM()).handle_revise_requested(
        client, _Env({"problem_model_ref": "problem_model://s1", "method_ref": prior_ref,
            "baseline_result_ref": baseline_ref, "research_memory_ref": "memory://s1",
            "experiment_feedback": "Repair the implementation; preserve its intended mechanism.",
            "incremental_revision": True})))
    generated = [e for e in client.published if e["event_type"] == "method.generated"]
    assert bool(generated) is (case in {"repair", "provider_error"})
    if case in {"repair", "provider_error"}:
        methods = [obj for kind, obj in store.puts if kind == "method"]
        assert methods == [prior]


@pytest.mark.parametrize("closeout", [False, True])
@pytest.mark.parametrize("parse_failure", [False, True])
def test_implementation_repair_honors_selected_contract_revision(monkeypatch, closeout, parse_failure):
    from autoresearch.agents.method_agent.core import MethodError
    from autoresearch.experiments.repo_improve.tests.test_component_plan import _component
    prior_ref, baseline_ref, review_ref = "method://s1/prior", "baseline://s1", "review://s1/repair"
    prior = {"approach": "retained treatment with an obsolete extra gate", "changes_outline": [],
             "components": [_component("retained"), _component("obsolete-check", role="indirect_gain")]}
    revised = {**prior, "approach": "retained treatment with the authorized task gate",
               "components": [prior["components"][0]]}
    review = {"status": "completed", "trigger": "implementation_failure",
        "method_ref": prior_ref, "baseline_ref": baseline_ref,
        "closeout_requested": closeout,
        "decision": {"route": "revise_method", "requires_core_reconstruction": False,
                     "remove": ["obsolete-check"],
                     "modify": ["Remove the obsolete gate from the Method description too."]}}
    store = _Store({prior_ref: prior, baseline_ref: {}, "problem_model://s1": {},
        "memory://s1": {"diagnosis": "method_improvement_proposed", "diagnosis_evidence": [review_ref]},
        review_ref: review})
    seen = []
    def step(*args, **kwargs):
        seen.append(kwargs)
        assert kwargs["retained_treatment_revision"] is True
        if parse_failure:
            raise MethodError("permanent", "method_parse_error", "invalid revised contract")
        return json.loads(json.dumps(revised))
    monkeypatch.setattr("autoresearch.agents.method_agent.handler.method_core.method_step", step)
    client = _Client()
    asyncio.run(MethodAgentHandler(spec_store=store, llm_client=_LLM()).handle_revise_requested(
        client, _Env({"problem_model_ref": "problem_model://s1", "method_ref": prior_ref,
            "baseline_result_ref": baseline_ref, "research_memory_ref": "memory://s1",
            "experiment_feedback": "Apply the bound local contract correction; retain the treatment.",
            "incremental_revision": True})))
    assert len(seen) == 1
    methods = [obj for kind, obj in store.puts if kind == "method"]
    assert methods == ([] if parse_failure else [revised])
    assert any(e["event_type"] == "agent.failed" for e in client.published) is parse_failure
