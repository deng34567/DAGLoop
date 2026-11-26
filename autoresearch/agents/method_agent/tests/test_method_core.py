"""MethodAgent core: direction -> concrete method."""
from __future__ import annotations

import json

import pytest

from autoresearch.agents.method_agent.core import MethodError, method_step


class _Resp:
    def __init__(self, content, *, finish_reason=None):
        self.content = content
        self.finish_reason = finish_reason


class _FakeLLM:
    def __init__(self, responses):
        self._r = [r if isinstance(r, str) else json.dumps(r) for r in responses]
        self.calls = []

    def chat(self, *, messages, response_format=None, max_tokens=None):
        self.calls.append(messages)
        return _Resp(self._r.pop(0) if self._r else "")


_PM = {"key_problems": ["planning is hard"], "verifiable_paths": ["compare schedules"]}
_GOOD = {"approach": "cosine-then-restart LR schedule", "changes_outline": ["add restart"],
         "why_it_should_win": "escapes sharp minima", "assumptions": ["fixed budget"]}


def test_method_step_returns_concrete_method():
    llm = _FakeLLM([_GOOD])
    m = method_step(_PM, {"papers": []}, {"verdict": "pass"}, llm)
    assert m["approach"] == "cosine-then-restart LR schedule"
    assert m["changes_outline"] == ["add restart"] and m["assumptions"] == ["fixed budget"]
    system = llm.calls[0][0]["content"]
    assert "only candidate treatment changes" in system
    assert "control groups" in system
    assert "Repository inspection, environment preparation, and final evaluator submission" in system
    assert "no authoritative frozen BaselineResult" in system
    assert "literature context alone" in system


def test_method_step_incorporates_human_idea_and_feedback():
    llm = _FakeLLM([_GOOD])
    method_step(_PM, {}, {}, llm, human_idea="focus on curriculum learning",
                feedback="the LR-schedule method showed no gain")
    user = next(m["content"] for m in llm.calls[0] if m["role"] == "user")
    assert "curriculum learning" in user          # human redirection carried
    assert "no gain" in user and "Evidence and design guidance" in user


def test_method_revision_keeps_null_alternative_unmeasured():
    llm = _FakeLLM([_GOOD])
    feedback = {"trigger": "complete_result", "incumbent_metric": 0.174514,
                "modify": ["Try a wider vector path with fallback"],
                "qualified_alternative": {"method_metric": None,
                                          "measurement_valid": None}}
    method_step(_PM, {}, {}, llm, feedback=json.dumps(feedback))
    system, user = (message["content"] for message in llm.calls[0])
    assert "A null or missing metric is missing measurement evidence, not zero gain" in system
    assert "or proof that hardware lacks a capability" in system
    assert json.dumps(feedback) in user
    assert "bound measurements for the exact evaluated mechanism" in system


def test_method_step_retries_then_raises_on_bad_output():
    llm = _FakeLLM(["not json", {"approach": ""}, {"approach": ""}])  # never a valid approach
    with pytest.raises(MethodError):
        method_step(_PM, {}, {}, llm)


def test_method_step_repairs_malformed_long_draft_with_compact_reply():
    malformed = '{"approach": "candidate", "components": ['
    llm = _FakeLLM([malformed, _GOOD])
    method = method_step(_PM, {}, {}, llm)
    assert method == _GOOD
    assert llm.calls[1][-2] == {"role": "assistant", "content": malformed}
    assert "compact valid JSON" in llm.calls[1][-1]["content"]
    assert "Omit components" in llm.calls[1][-1]["content"]


def test_incremental_parse_repair_receives_the_malformed_draft():
    from autoresearch.experiments.repo_improve.tests.test_component_plan import _component

    malformed = '{"approach": "candidate", "components": ['
    repaired = {"approach": "candidate", "components": [_component("scheduler")]}
    llm = _FakeLLM([malformed, repaired])

    result = method_step(_PM, {}, {}, llm, incremental_revision=True)

    assert result["components"] == repaired["components"]
    assert llm.calls[1][-2] == {"role": "assistant", "content": malformed}
    assert "Repair the preceding malformed response" in llm.calls[1][-1]["content"]
    assert "fix JSON syntax/completeness only" in llm.calls[1][-1]["content"]


def test_incremental_parse_retries_keep_only_the_latest_malformed_draft():
    from autoresearch.experiments.repo_improve.tests.test_component_plan import _component

    first = '{"approach": "first-unique-draft", "components": ['
    second = '{"approach": "second-unique-draft", "components": ['
    repaired = {"approach": "candidate", "components": [_component("scheduler")]}
    llm = _FakeLLM([first, second, repaired])

    result = method_step(_PM, {}, {}, llm, incremental_revision=True)

    assert result["components"] == repaired["components"]
    third_prompt = "\n".join(message["content"] for message in llm.calls[2])
    assert "second-unique-draft" in third_prompt
    assert "first-unique-draft" not in third_prompt


def test_incremental_length_truncation_continues_only_the_missing_suffix():
    from autoresearch.experiments.repo_improve.tests.test_component_plan import _component

    complete = json.dumps({"approach": "large candidate",
                           "components": [_component("scheduler")]})
    split = complete.index('"components"') + len('"components": [')

    class LLM:
        def __init__(self):
            self.calls = []

        def chat(self, *, messages, **_):
            self.calls.append(messages)
            if len(self.calls) == 1:
                return _Resp(complete[:split], finish_reason="length")
            return _Resp(complete[split:], finish_reason="stop")

    llm = LLM()
    result = method_step(_PM, {}, {}, llm, incremental_revision=True)

    assert result["components"][0]["component_id"] == "scheduler"
    assert len(llm.calls) == 2
    assert llm.calls[1][-2] == {"role": "assistant", "content": complete[:split]}
    assert "return only the missing suffix" in llm.calls[1][-1]["content"]


def test_incremental_contract_repair_receives_the_invalid_draft():
    from autoresearch.experiments.repo_improve.tests.test_component_plan import _component
    invalid = {"approach": "candidate", "components": [_component("index", "indirect_gain")]}
    invalid["components"][0]["functional_check"] = "count == expected_count"
    repaired = {"approach": "candidate", "components": [_component("index", "indirect_gain")]}
    repaired["components"][0]["functional_check"] = "difference == 0"
    llm = _FakeLLM([invalid, repaired])
    assert method_step(_PM, {}, {}, llm, incremental_revision=True)["components"] == repaired["components"]
    assert llm.calls[1][-2] == {"role": "assistant", "content": json.dumps({
        **invalid, "changes_outline": [], "why_it_should_win": "", "assumptions": []}, ensure_ascii=False)}
    assert "numeric threshold" in llm.calls[1][-1]["content"]


@pytest.mark.parametrize("check", [
    "hidden_test_accuracy >= 0.95 and total_params == 1009",
    "signed_metric_improvement <= 0",
])
def test_incremental_revision_repairs_unexecutable_direct_gain_acceptance(check):
    from copy import deepcopy
    from autoresearch.experiments.repo_improve.tests.test_component_plan import _component

    invalid = {"approach": "candidate", "components": [_component("scheduler")]}
    invalid["components"][0]["ablation_test"] = check
    repaired = deepcopy(invalid)
    repaired["components"][0]["ablation_test"] = "signed_metric_improvement > 0"
    llm = _FakeLLM([invalid, repaired])

    result = method_step(_PM, {}, {}, llm, incremental_revision=True)

    assert len(llm.calls) == 2
    assert result["components"] == repaired["components"]
    assert check in llm.calls[1][-2]["content"]
    assert "direct-gain acceptance" in llm.calls[1][-1]["content"]


@pytest.mark.parametrize('check,canonical', [
    ('signed_metric_improvement > 0.033; or all variants measured with no gain',
     'signed_metric_improvement > 0.033'),
    ('signed_metric_improvement > 0（相对父节点）；正确性失败时回退，不宣称增益。',
     'signed_metric_improvement > 0'),
    ('same_session_parent_runtime - candidate_runtime > 0',
     'signed_metric_improvement > 0'),
])
def test_incremental_gain_format_repair_preserves_other_obligations(check, canonical):
    from copy import deepcopy
    from autoresearch.experiments.repo_improve.component_plan import (
        build_component_plan, validate_component_acceptance)
    from autoresearch.experiments.repo_improve.tests.test_component_plan import _component

    parent = _component('index', 'indirect_gain')
    prior = {'approach': 'indexed candidate', 'components': [parent]}
    before = deepcopy(prior)
    invalid = {'approach': 'refined candidate',
               'components': [parent, _component('extension', depends_on=['index'])]}
    invalid['components'][1].update(
        ablation_test=check, implementation_check='Official correctness must pass.',
        optimization_space='Measure the declared variants; retain the best on no gain.')
    repaired = deepcopy(invalid)
    repaired['components'][1]['ablation_test'] = canonical
    llm = _FakeLLM([invalid, repaired])

    result = method_step(_PM, {}, {}, llm, incremental_revision=True, prior_method=prior)

    validate_component_acceptance(build_component_plan(result))
    assert len(llm.calls) == 2
    assert result['components'] == repaired['components']
    assert prior == before
    system = llm.calls[0][0]['content']
    assert 'signed_metric_improvement <operator> <finite number>' in system
    assert 'candidate_metric <operator> <finite number>' in system
    assert 'parent - candidate' in system and 'candidate - parent' in system
    repair = llm.calls[1][-1]['content']
    assert 'implementation_check' in repair and 'optimization_space' in repair
    assert 'thresholds' in repair and 'stop' in repair


@pytest.mark.parametrize('check', [
    'accuracy >= 0.95 and total_params == 1009',
    'signed_metric_improvement > 0.033; or all variants measured with no gain',
    'signed_metric_improvement > 0（相对父节点）；正确性失败时回退，不宣称增益。',
    'same_session_parent_runtime - candidate_runtime > 0',
])
def test_incremental_revision_rejects_invalid_acceptance_after_bounded_retries(check):
    from autoresearch.experiments.repo_improve.tests.test_component_plan import _component

    invalid = {"approach": "candidate", "components": [_component("scheduler")]}
    invalid["components"][0]["ablation_test"] = check
    llm = _FakeLLM([invalid, invalid, invalid])

    with pytest.raises(MethodError, match="direct-gain acceptance"):
        method_step(_PM, {}, {}, llm, incremental_revision=True)
    assert len(llm.calls) == 3


def test_missing_prerequisite_repair_receives_bound_prior_contract_verbatim():
    from autoresearch.experiments.repo_improve.tests.test_component_plan import _component
    parent = _component("retained_parent", "indirect_gain")
    prior = {"approach": "candidate", "components": [parent]}
    child = _component("extension", depends_on=["retained_parent"])
    invalid = {"approach": "extended candidate", "components": [child]}
    repaired = {"approach": "extended candidate", "components": [parent, child]}
    before = json.dumps(prior, sort_keys=True)
    llm = _FakeLLM([invalid, repaired])
    result = method_step(_PM, {}, {}, llm, incremental_revision=True, prior_method=prior)
    assert result["components"] == [parent, child]
    assert len(llm.calls) == 2
    assert json.dumps([parent], ensure_ascii=False) in llm.calls[-1][-1]["content"]
    assert json.dumps(prior, sort_keys=True) == before


def test_incremental_treatment_change_repairs_stale_component_contracts():
    from copy import deepcopy
    from autoresearch.experiments.repo_improve.tests.test_component_plan import _component

    prior = {"approach": "batched processing", "changes_outline": ["batch complete inputs"],
             "components": [_component("scheduler")]}
    stale = {**prior, "approach": "resumable processing",
             "changes_outline": ["track per-input progress and resume partial work"]}
    repaired = deepcopy(stale)
    repaired["components"][0]["mechanism"] = "track per-input progress and resume partial work"
    llm = _FakeLLM([stale, repaired])

    result = method_step(_PM, {}, {}, llm, incremental_revision=True, prior_method=prior)

    assert len(llm.calls) == 2
    assert result["components"] == repaired["components"]
    assert "every executable component contract unchanged" in llm.calls[-1][-1]["content"]
    assert prior["components"][0]["mechanism"] != repaired["components"][0]["mechanism"]


def test_incremental_stale_contract_repair_is_bounded():
    from autoresearch.experiments.repo_improve.tests.test_component_plan import _component

    prior = {"approach": "original treatment", "changes_outline": ["original operation"],
             "components": [_component("scheduler")]}
    stale = {**prior, "approach": "new treatment", "changes_outline": ["new operation"]}
    llm = _FakeLLM([stale, stale, stale])

    with pytest.raises(MethodError, match="every executable component contract unchanged"):
        method_step(_PM, {}, {}, llm, incremental_revision=True, prior_method=prior)
    assert len(llm.calls) == 3


def test_incremental_description_only_update_keeps_component_reuse():
    from autoresearch.experiments.repo_improve.tests.test_component_plan import _component

    prior = {"approach": "batched processing", "changes_outline": ["batch complete inputs"],
             "components": [_component("scheduler")]}
    clarified = {**prior, "approach": "clarified description of the same batched processing"}
    llm = _FakeLLM([clarified])

    result = method_step(_PM, {}, {}, llm, incremental_revision=True, prior_method=prior)

    assert len(llm.calls) == 1
    assert result["components"] == prior["components"]


@pytest.mark.parametrize("retained", [False, True])
def test_description_repair_receives_exact_prior_treatment_outline(retained):
    from copy import deepcopy
    from autoresearch.experiments.repo_improve.tests.test_component_plan import _component

    prior = {"approach": "retained implementation", "changes_outline": ["use measured core"],
             "components": [_component("retained")]}
    before = deepcopy(prior)
    invalid = {**prior, "changes_outline": ["clarify the evaluation of the measured core"]}
    repaired = {**prior, "approach": "retained implementation with clarified evaluation"}
    llm = _FakeLLM([invalid, repaired])

    result = method_step(_PM, {}, {}, llm, incremental_revision=True, prior_method=prior,
                         retained_treatment_revision=retained)

    assert len(llm.calls) == 2
    assert result["changes_outline"] == prior["changes_outline"]
    assert result["components"] == prior["components"]
    assert prior == before
    repair = llm.calls[1][-1]["content"]
    assert "Frozen prior treatment changes_outline" in repair
    assert json.dumps(prior["changes_outline"], ensure_ascii=False) in repair
    assert "description or evaluation-only" in repair
    assert "affected executable component contracts" in repair


def test_incremental_treatment_update_preserves_unaffected_component_contract():
    from copy import deepcopy
    from autoresearch.experiments.repo_improve.tests.test_component_plan import _component

    prior = {"approach": "indexed scheduler", "changes_outline": ["build index and schedule"],
             "components": [_component("index", "indirect_gain"),
                            _component("scheduler", depends_on=["index"])]}
    revised = deepcopy(prior)
    revised["changes_outline"] = ["build index and resume partially scheduled work"]
    revised["components"][1]["mechanism"] = "resume partially scheduled work"
    llm = _FakeLLM([revised])

    result = method_step(_PM, {}, {}, llm, incremental_revision=True, prior_method=prior)

    assert len(llm.calls) == 1
    assert result["components"][0] == prior["components"][0]
    assert result["components"][1] == revised["components"][1]


def test_closeout_can_restore_treatment_prose_with_retained_component_contracts():
    from autoresearch.experiments.repo_improve.tests.test_component_plan import _component

    prior = {"approach": "unimplemented extension", "changes_outline": ["add untested extension"],
             "components": [_component("retained")]}
    restored = {**prior, "approach": "retained measured treatment",
                "changes_outline": ["use the retained measured implementation"]}
    llm = _FakeLLM([restored])

    result = method_step(_PM, {}, {}, llm, incremental_revision=True,
                         prior_method=prior, retained_treatment_revision=True,
                         closeout_revision=True)

    assert len(llm.calls) == 1
    assert "RETAINED TREATMENT EXCEPTION" in llm.calls[0][0]["content"]
    assert result["components"] == prior["components"]


def test_implementation_repair_cannot_change_treatment_without_component_contract():
    from copy import deepcopy
    from autoresearch.experiments.repo_improve.tests.test_component_plan import _component

    prior = {"approach": "retained mechanism", "changes_outline": ["use retained path"],
             "components": [_component("retained")]}
    stale = {**prior, "approach": "add a new mechanism",
             "changes_outline": ["add a new mechanism"]}
    repaired = deepcopy(stale)
    repaired["components"][0]["mechanism"] = "implement the new mechanism"
    llm = _FakeLLM([stale, repaired])

    result = method_step(_PM, {}, {}, llm, incremental_revision=True,
                         prior_method=prior, retained_treatment_revision=True)

    assert len(llm.calls) == 2
    assert result["components"][0]["mechanism"] == "implement the new mechanism"


@pytest.mark.parametrize('binding', [None, 'result://other', 'result://retained'])
@pytest.mark.parametrize('closing', [False, True])
def test_closeout_requires_selected_retained_result_binding(binding, closing):
    from autoresearch.experiments.repo_improve.tests.test_component_plan import _component

    prior = {'approach': 'retained mechanism', 'changes_outline': [],
             'components': [_component('retained')]}
    draft = {**prior, **({'reconciled_result_ref': binding} if binding else {})}
    llm = _FakeLLM([draft, {**prior, 'reconciled_result_ref': 'result://retained'}])
    result = method_step(_PM, {}, {}, llm, incremental_revision=True, prior_method=prior,
                         retained_treatment_revision=True,
                         reconciliation_result_ref='result://retained', closeout_revision=closing)

    assert result.get('reconciled_result_ref') == ('result://retained' if closing else binding)
    assert result['components'] == prior['components']
    assert len(llm.calls) == (2 if closing and binding != 'result://retained' else 1)
    if closing:
        assert 'reconciled_result_ref' in llm.calls[0][0]['content']


@pytest.mark.parametrize("matches", [True, False])
def test_measured_reconciliation_contract_reuse_requires_exact_result_binding(matches):
    from autoresearch.experiments.repo_improve.tests.test_component_plan import _component

    prior = {"approach": "unimplemented extension", "changes_outline": ["extend treatment"],
             "components": [_component("retained")]}
    restored = {**prior, "approach": "actual measured treatment",
                "changes_outline": ["use the measured implementation"],
                "reconciled_result_ref": "result://observed" if matches else "result://other"}
    llm = _FakeLLM([restored, restored, restored])

    if matches:
        result = method_step(_PM, {}, {}, llm, incremental_revision=True,
                             prior_method=prior, reconciliation_result_ref="result://observed")
        assert result["components"] == prior["components"]
        assert len(llm.calls) == 1
    else:
        with pytest.raises(MethodError, match="every executable component contract unchanged"):
            method_step(_PM, {}, {}, llm, incremental_revision=True,
                        prior_method=prior, reconciliation_result_ref="result://observed")


@pytest.mark.parametrize("candidate_ref", [None, "result://retained"])
def test_contract_repair_explains_the_available_retained_source_binding(candidate_ref):
    from copy import deepcopy
    from autoresearch.experiments.repo_improve.tests.test_component_plan import _component
    prior = {"approach": "measured mechanism", "changes_outline": ["original treatment"],
             "components": [_component("retained")]}
    draft = {**prior, "changes_outline": ["restore the measured implementation"]}
    class LLM:
        calls = 0
        def chat(self, *, messages, **kwargs):
            self.calls += 1
            if self.calls == 1:
                return _Resp(json.dumps(draft))
            repair = messages[-1]["content"]
            result = deepcopy(draft)
            if candidate_ref:
                assert "reconciled_result_ref=" + candidate_ref in repair
                assert "fresh validation" in repair
                result["reconciled_result_ref"] = candidate_ref
            else:
                assert "reconciled_result_ref=" not in repair
                result["components"][0]["mechanism"] = "a different measured operation"
            return _Resp(json.dumps(result))
    llm = LLM()
    result = method_step(_PM, {}, {}, llm, incremental_revision=True, prior_method=prior,
                         reconciliation_result_ref=candidate_ref)
    assert llm.calls == 2
    assert result.get("reconciled_result_ref") == candidate_ref
