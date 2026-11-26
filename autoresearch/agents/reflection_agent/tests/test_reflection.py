"""方法反思 core: 对照 method 反思搜索足迹,给下一轮方向。会议 2026-08-05, 张老师。"""
import json
import pytest
from types import SimpleNamespace

from autoresearch.agents.reflection_agent.core import (
    ReflectionResult, reflect_on_search, reflection_preamble, review_complete_method)


class _FakeLLM:
    def __init__(self, obj): self._obj = obj; self.calls = 0; self.last_call = None
    def chat(self, **kw):
        self.calls += 1
        self.last_call = kw
        return SimpleNamespace(content=json.dumps(self._obj))


class _Card:
    def __init__(self, **kw):
        self.round = kw.get("round", 1); self.operator = kw.get("operator", "improve")
        self.metric_value = kw.get("metric_value"); self.valid = kw.get("valid", True)
        self.verdict = kw.get("verdict", ""); self.changes_summary = kw.get("changes_summary", "")
        self.rationale = kw.get("rationale", ""); self.bottleneck = kw.get("bottleneck", "")
        self.error_signature = kw.get("error_signature", ""); self.lower_is_better = True


@pytest.mark.parametrize('trigger', ['component_method_reconciliation', 'legal_off_method_gain'])
@pytest.mark.parametrize('expansion', ['add', 'requires_core_reconstruction'])
def test_off_method_closeout_rejects_new_exploration(trigger, expansion):
    from autoresearch.agents.memory_agent.tests.test_method_feedback import decision
    response = {**decision('candidate'), 'add': [], 'modify': ['Reconcile retained source.'],
                'requires_core_reconstruction': False}
    response[expansion] = ['new mechanism'] if expansion == 'add' else True
    llm = _FakeLLM(response)
    result = review_complete_method({}, {}, {
        'trigger': {'kind': trigger}, 'candidate': {'checkpoint': '/saved/A'},
        'closeout_requested': True}, llm)
    assert result['route'] == 'unavailable'
    assert 'closeout permits only' in result['error_detail']
    assert result['validation_failed'] is True


@pytest.mark.parametrize('trigger', [
    'complete_result', 'complete_metric_regression', 'complete_verified_no_gain'])
@pytest.mark.parametrize('prior', [
    {'trigger': 'qualified_component_closeout'},
    {'trigger': 'component_method_reconciliation', 'closeout_requested': True},
])
def test_prior_component_closeout_does_not_end_whole_method_reflection(trigger, prior):
    from autoresearch.agents.memory_agent.tests.test_method_feedback import decision
    llm = _FakeLLM(decision('trigger'))
    evidence = {'trigger': {'kind': trigger}, 'prior_method_reflections': [
        {**prior, 'decision': {'stop_condition': 'Do not resume exploration.'}}]}

    result = review_complete_method({}, {}, evidence, llm)

    assert result['route'] == 'revise_method'
    assert llm.calls == 1
    assert '\nCLOSEOUT:' not in llm.last_call['messages'][0]['content']
    assert 'Historical reflections are diagnostic context, not current instructions.' in (
        llm.last_call['messages'][0]['content'])
    assert 'Qualification alone does not establish saturation.' in (
        llm.last_call['messages'][0]['content'])
    assert json.loads(llm.last_call['messages'][1]['content'])['evidence'] == evidence


@pytest.mark.parametrize('directive', ['closeout_requested', 'user_closeout'])
def test_current_closeout_still_prevents_whole_method_exploration(directive):
    from autoresearch.agents.memory_agent.tests.test_method_feedback import decision
    llm = _FakeLLM(decision('trigger'))
    evidence = {'trigger': {'kind': 'complete_result'},
                directive: {'closeout': True} if directive == 'user_closeout' else True}

    result = review_complete_method({}, {}, evidence, llm)

    assert result['route'] == 'unavailable'
    assert 'closeout permits only' in result['error_detail']


def test_prior_closeout_still_bounds_unfinished_implementation_repair():
    from autoresearch.agents.memory_agent.tests.test_method_feedback import decision
    llm = _FakeLLM(decision('trigger'))
    result = review_complete_method({}, {}, {
        'trigger': {'kind': 'implementation_failure'},
        'prior_method_reflections': [{'trigger': 'qualified_component_closeout'}]}, llm)

    assert result['route'] == 'unavailable'
    assert 'closeout permits only' in result['error_detail']


@pytest.mark.parametrize('error', [ValueError('provider configuration'), TimeoutError('provider timeout')])
def test_review_provider_errors_are_not_response_validation_failures(error):
    class FailingLLM:
        def chat(self, **kwargs):
            raise error
    result = review_complete_method({}, {}, {'closeout_requested': True}, FailingLLM())
    assert result['route'] == 'unavailable'
    assert result['validation_failed'] is False


@pytest.mark.parametrize('repair_succeeds', [True, False])
def test_review_repairs_its_rejected_response_without_accepting_unknown_citations(repair_succeeds):
    from autoresearch.agents.memory_agent.tests.test_method_feedback import decision
    invalid = json.dumps(decision('repo/unsupplied.py'))

    class RepairLLM:
        def __init__(self):
            self.calls = []

        def chat(self, **kwargs):
            self.calls.append(kwargs)
            messages = kwargs['messages']
            can_repair = any(message['role'] == 'assistant'
                             and message['content'] == invalid for message in messages)
            content = (json.dumps(decision('candidate'))
                       if repair_succeeds and can_repair else invalid)
            return SimpleNamespace(content=content)

    llm = RepairLLM()
    result = review_complete_method({}, {}, {'candidate': {'metric': 2.0}}, llm)

    assert len(llm.calls) == 2
    assert llm.calls[0]['messages'][1] == llm.calls[1]['messages'][1]
    if repair_succeeds:
        assert result['route'] == 'revise_method'
        assert result['evidence_refs'] == ['candidate']
    else:
        assert result['route'] == 'unavailable'
        assert 'repo/unsupplied.py' in result['error_detail']
        assert result['invalid_responses'] == [invalid, invalid]


@pytest.mark.parametrize('trigger', ['component_method_reconciliation', 'qualified_component_closeout'])
def test_component_reconciliation_repairs_report_route_before_formal_evaluation(trigger):
    class RepairLLM(_FakeLLM):
        def chat(self, **kwargs):
            if self.calls:
                assert "requires revise_method" in kwargs['messages'][0]['content']
                self._obj = {
                    'route': 'revise_method', 'dimensions': ['mechanism_understanding'],
                    'primary_hypothesis': 'The retained legal mechanism differs from its description.',
                    'evidence_refs': ['trigger'], 'retain': ['measured implementation'],
                    'modify': ['Match the Method description and evaluation contract to the checkpoint.'],
                    'add': [], 'remove': [], 'expected_observation': 'Complete official evaluation passes.',
                    'validation_plan': 'Restore and officially evaluate the retained checkpoint.',
                    'affected_dependencies': [], 'reuse': ['checkpoint'],
                    'requires_core_reconstruction': False, 'stop_condition': 'Formal result is bound.',
                }
            return super().chat(**kwargs)

    llm = RepairLLM({'route': 'report', 'reason': 'Run the official evaluator next.'})
    result = review_complete_method({}, {}, {'trigger': {'kind': trigger}}, llm)
    assert result['route'] == 'revise_method' and llm.calls == 2
    assert result['requires_core_reconstruction'] is False
    assert result['add'] == [] and result['remove'] == []


@pytest.mark.parametrize("lower,value,reference,relation,reached", [
    (True, 0.100225, 0.03, ">", False),
    (True, 1.571425, 0.37, ">", False),
    (True, 0.01, 0.03, "<", True),
    (False, 0.6, 0.7, "<", False),
    (False, 0.8, 0.7, ">", True),
    (True, 0.03, 0.03, "=", True),
])
def test_method_review_receives_direction_correct_reference_facts(lower, value, reference, relation, reached):
    llm = _FakeLLM({"route": "report", "reason": "No concrete next experiment."})
    result_ref = "spec://gen_experiment_result/session/snapshot-measured"
    source = {"path": "tasks/example/task.toml", "sha256": "captured-source",
              "content": f'[optimization]\nmetric="metric"\ndirection="{"lower" if lower else "higher"}"\n[optimization.reference]\nscore={reference}\n'}
    evidence = {result_ref: {"method_metric": value}, "task_requirements": {
        "metric_name": "metric", "lower_is_better": lower, "repository_sources": [source]}}
    review_complete_method({}, {}, evidence, llm)
    comparison = json.loads(llm.last_call["messages"][1]["content"])["numeric_reference_comparisons"][0]
    assert comparison["measured"] == value and comparison["declared_reference"] == reference
    assert comparison["relation"] == relation
    assert comparison["at_least_as_good_as_declared_reference"] is reached
    assert comparison["source"] == source["path"] and comparison["result_ref"] == result_ref


@pytest.mark.parametrize("value", [None, True, float("nan"), float("inf")])
def test_method_review_does_not_turn_missing_or_invalid_metric_into_reference_attainment(value):
    llm = _FakeLLM({"route": "report", "reason": "No concrete next experiment."})
    source = {"path": "task.toml", "content": '[optimization]\nmetric="time"\ndirection="lower"\n[optimization.reference]\nscore=0.03\n'}
    evidence = {"spec://gen_experiment_result/session/value": {"method_metric": value},
                "task_requirements": {"metric_name": "time", "lower_is_better": True,
                                      "repository_sources": [source]}}
    review_complete_method({}, {}, evidence, llm)
    assert json.loads(llm.last_call["messages"][1]["content"])["numeric_reference_comparisons"] == []


def test_no_llm_or_no_cards_is_empty():
    assert reflect_on_search({}, [], True, _FakeLLM({})).route == "stay_experiment"
    assert reflect_on_search({}, [_Card()], True, None).verdict == ""   # 无 llm -> 空


def test_parses_stay_experiment_with_bottleneck():
    llm = _FakeLLM({"route": "stay_experiment", "verdict": "round2支撑method",
                    "bottleneck_focus": "attention显存带宽", "report": "..."})
    r = reflect_on_search({"method": "用RMSNorm+SwiGLU"}, [_Card(bottleneck="慢")], True, llm)
    assert r.route == "stay_experiment" and r.bottleneck_focus == "attention显存带宽"
    assert "round2支撑method" in r.verdict and llm.calls == 1


def test_search_reflection_accepts_slow_call_with_configured_deadline():
    class ConfiguredDeadlineLLM(_FakeLLM):
        def chat(self, **kw):
            effective_deadline = kw.get("stream_max_s", 1800)
            if effective_deadline < 200:
                raise TimeoutError("configured long response was cut off")
            return super().chat(**kw)

    llm = ConfiguredDeadlineLLM({"route": "revise_method",
                                "verdict": "measured evidence supports revision",
                                "method_note": "repair the documented interface"})
    result = reflect_on_search({"method": "A"}, [_Card(metric_value=1)], True, llm)
    assert result.route == "revise_method"
    assert result.verdict == "measured evidence supports revision"
    assert llm.calls == 1


def test_reflection_receives_the_already_frozen_baseline_context():
    llm = _FakeLLM({"route": "stay_experiment", "verdict": "method is worse"})

    reflect_on_search(
        {"method": "try a new optimizer"}, [_Card(metric_value=1.2)], True, llm,
        baseline={"metric_name": "val_bpb", "metric_value": 1.0,
                  "lower_is_better": True, "reproduced": True},
    )

    user = llm.last_call["messages"][1]["content"]
    assert "FROZEN BASELINE ALREADY ESTABLISHED" in user
    assert '"metric_value":1.0' in user
    assert "Do not propose reproducing or recalibrating it" in user


def test_reflection_receives_repository_interface_even_with_long_baseline_log():
    llm = _FakeLLM({"route": "stay_experiment"})
    reflect_on_search({"method": "A"}, [_Card()], True, llm,
                      baseline={"metric_name": "reward", "lower_is_better": False,
                                "log_excerpt": "x" * 5000,
                                "repository_contract": {"sources": [{
                                    "path": "instruction.md", "content": "Only solve(x) is allowed."}]}})
    user = llm.last_call["messages"][1]["content"]
    system = llm.last_call["messages"][0]["content"]
    assert "FROZEN TASK REQUIREMENTS AND INTERFACE" in user
    assert "Only solve(x) is allowed" in user
    assert "testable extension B to an evidence-supported Method A" in system
    assert "B may be one component or a set of components with explicit dependencies" in system
    assert "complete evidence-grounded replacement is permitted" in system


def test_reflection_uses_full_revised_method_and_frozen_task_boundary():
    llm = _FakeLLM({"route": "stay_experiment"})
    method = {"approach": "A plus B", "components": [
        {"name": "A", "mechanism": "retained evidence-supported part"},
        {"name": "B", "mechanism": "dependent extension", "depends_on": ["A"]}]}
    reflect_on_search(method, [_Card()], True, llm, baseline={
        "benchmark_contract": {"allowed_file": "solve.py"},
        "repository_contract": {"sources": [{"path": "instruction.md",
            "content": "context" * 1200 + "Do not inspect hidden labels."}]}})
    user = llm.last_call["messages"][1]["content"]
    assert "dependent extension" in user
    assert "Do not inspect hidden labels." in user
    assert '"allowed_file": "solve.py"' in user


def test_reflection_treats_an_impossible_frozen_requirement_as_method_revision():
    llm = _FakeLLM({"route": "stay_experiment"})

    reflect_on_search({"method": "required component configuration"}, [_Card()], True, llm)

    system = llm.last_call["messages"][0]["content"]
    assert "empirically impossible" in system
    assert "route=revise_method" in system


def test_reflection_distinguishes_off_method_gain_from_method_validation():
    llm = _FakeLLM({"route": "stay_experiment"})
    card = _Card(metric_value=0.9, verdict="valid_not_best")
    card.implementation_verdict = "drifted"

    reflect_on_search({"method": "fused optimizer"}, [card], True, llm)

    system = llm.last_call["messages"][0]["content"]
    user = llm.last_call["messages"][1]["content"]
    assert "empirical evidence, not validation" in system
    assert "evidence-grounded replacement or reconciliation" in system
    assert "implementation=drifted" in user


def test_reflection_preserves_verified_parts_and_requires_concrete_alternative():
    llm = _FakeLLM({"route": "stay_experiment"})
    reflect_on_search({"method": "fused optimizer"}, [_Card(metric_value=0.9)], True, llm)
    system = llm.last_call["messages"][0]["content"]
    assert "Retain Method mechanisms and components supported by valid" in system
    assert "propose a concrete alternative mechanism and a measurable check" in system
    assert "route it through revise_method" in system


def test_complete_method_review_preserves_effective_parts_and_redesigns_invalid_ones():
    llm = _FakeLLM({"route": "report", "reason": "insufficient evidence"})
    review_complete_method({}, {}, {}, llm)
    system = llm.last_call["messages"][0]["content"]
    assert "Retain Method parts and components with verified" in system
    assert "give a concrete alternative mechanism, its expected observation" in system
    assert "removal without a replacement is allowed" in system
    assert "validate the revised complete Method against the retained qualified incumbent" in system
    assert "testable extension B" in system
    assert "a set of components with explicit dependencies" in system
    assert "replace the entire treatment when its core is refuted" in system
    assert "candidate_valid=true gate cannot be described as an accuracy failure" in system
    assert "Do not call a qualified component mechanism ineffective" in system
    assert "make a compliant integration of that checkpoint an early validation step" in system
    assert "A null or missing metric is missing measurement evidence, not zero gain" in system
    assert "bound measurements for the exact evaluated mechanism" in system


def test_parses_revise_method():
    llm = _FakeLLM({"route": "revise_method", "verdict": "两轮都否定了核心假设",
                    "method_note": "RMSNorm不是瓶颈,假设错了", "report": "x"})
    r = reflect_on_search({"method": "m"}, [_Card(), _Card(round=2)], True, llm)
    assert r.route == "revise_method" and "假设错了" in r.method_note


def test_revise_method_preamble_does_not_tell_current_sandbox_to_change_method():
    """Method redesign belongs to the formal RevisingMethod state.

    The current implementation sandbox may gather the diagnostic requested by
    reflection, but must not receive the proposed replacement as an instruction
    while the frozen method still names the old design.
    """
    r = ReflectionResult(
        route="revise_method",
        verdict="The optimizer assumption is falsified.",
        expected_check="Compare matrix-parameter update magnitudes.",
        suspected_module="optimizer",
        bottleneck_focus="Restore the baseline optimizer.",
        method_note="Replace AdamW with Muon now.",
    )

    preamble = reflection_preamble(r)

    assert "Compare matrix-parameter update magnitudes" in preamble
    assert "optimizer" in preamble
    assert "Replace AdamW with Muon now" not in preamble
    assert "Restore the baseline optimizer" not in preamble
    assert "不要在当前实现沙箱中修改已确认方法" in preamble


def test_bad_route_defaults_stay():
    llm = _FakeLLM({"route": "garbage", "verdict": "v"})
    assert reflect_on_search({}, [_Card()], True, llm).route == "stay_experiment"


def test_malformed_json_fail_open():
    class Bad:
        def chat(self, **kw): return SimpleNamespace(content="not json{{{")
    assert reflect_on_search({}, [_Card()], True, Bad()).route == "stay_experiment"


def test_preamble_renders_direction():
    r = ReflectionResult(route="stay_experiment", verdict="V1支撑", bottleneck_focus="显存带宽")
    p = reflection_preamble(r)
    assert "方法反思" in p and "V1支撑" in p and "瓶颈焦点" in p and "显存带宽" in p


def test_preamble_queues_method_problem_for_formal_revision():
    r = ReflectionResult(route="revise_method", method_note="假设太理想", bottleneck_focus="X")
    p = reflection_preamble(r)
    assert "RevisingMethod" in p
    assert "不要在当前实现沙箱中修改已确认方法" in p
    assert "假设太理想" not in p
    assert "一句话瓶颈焦点: X" not in p


def test_empty_reflection_empty_preamble():
    assert reflection_preamble(ReflectionResult()) == ""


# --- #3 深度瓶颈诊断(会议 2026-08-06 老师第二类回路): 中间量预期关系 + 定向修模块 ------------

def test_parses_mechanistic_diagnosis():
    llm = _FakeLLM({"route": "stay_experiment", "verdict": "method站得住",
                    "expected_check": "辅助loss应随主loss同步下降",
                    "suspected_module": "attention pooling层",
                    "bottleneck_focus": "aux loss不降"})
    r = reflect_on_search({"method": "m"}, [_Card()], True, llm)
    assert r.expected_check == "辅助loss应随主loss同步下降"
    assert r.suspected_module == "attention pooling层"


def test_preamble_includes_mechanistic_check():
    r = ReflectionResult(route="stay_experiment", verdict="v",
                         expected_check="attention entropy 应 >0.5",
                         suspected_module="第3层MLP", bottleneck_focus="熵塌缩")
    p = reflection_preamble(r)
    assert "测量并核对" in p and "attention entropy 应 >0.5" in p      # 先测中间量
    assert "定向修" in p and "第3层MLP" in p                          # 再定向修模块


def test_preamble_nonempty_with_only_expected_check():
    # 即使 verdict/瓶颈为空,只要有 expected_check 也要出提示(机制诊断是核心)
    r = ReflectionResult(expected_check="grad-norm 应 ~1.0")
    assert "grad-norm 应 ~1.0" in reflection_preamble(r)



def test_report_preserves_stopping_evidence_without_revision_requirements():
    answer = {"route": "report", "reason": "no justified next experiment", "evidence_refs": ["saved"],
              "stop_condition": "no supported gain", "unresolved": ["interaction not tested"]}
    result = review_complete_method({}, {}, {"saved": {}}, _FakeLLM(answer))
    assert result == answer


def test_report_rejects_invented_evidence_or_missing_reason():
    for answer in ({"route": "report", "reason": "stop", "evidence_refs": ["invented"]},
                   {"route": "report"}):
        result = review_complete_method({}, {}, {"saved": {}}, _FakeLLM(answer))
        assert result["route"] == "unavailable"
