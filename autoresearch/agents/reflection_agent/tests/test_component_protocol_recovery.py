import json
from types import SimpleNamespace
import pytest
from autoresearch.agents.reflection_agent.core import decide_component_progress


@pytest.mark.parametrize("invalid", [
    {"action": "improve", "reason": "worth testing"},
    {"action": "advance", "reason": True},
    {"action": "improve", "reason": "test", "next_experiment": {"change": "x"}},
    {"action": "invalid", "reason": "x"},
])
def test_protocol_errors_are_repaired_with_same_evidence(invalid):
    calls = []
    def chat(**kwargs):
        calls.append(kwargs)
        obj = invalid if len(calls) == 1 else {"action": "advance", "reason": "no worthwhile next step"}
        return SimpleNamespace(content=json.dumps(obj))
    result = decide_component_progress({}, {}, {}, [], [], SimpleNamespace(chat=chat))
    assert result["action"] == "advance"
    assert len(calls) == 2
    assert calls[0]["messages"][1] == calls[1]["messages"][1]


def test_persistent_invalid_protocol_stops_after_three_without_default_advance():
    calls = []
    def chat(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(content='{"action":"advance","reason":true}')
    result = decide_component_progress({}, {}, {}, [], [], SimpleNamespace(chat=chat))
    assert result["action"] == "unavailable"
    assert len(calls) == 3


def test_structured_next_experiment_gets_specific_type_repair_without_changing_action():
    plan = {"change": "tune permitted setting", "question": "does it help", "evaluation": "same frozen check"}
    calls = []
    def chat(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            assert 'all four fields must be JSON strings' in kwargs['messages'][0]['content']
            obj = {"action": "improve", "selected_checkpoint": "best", "reason": "observed bottleneck", "next_experiment": plan}
        else:
            correction = kwargs['messages'][-1]['content']
            assert 'next_experiment must be a string' in correction
            assert 'dict' in correction
            obj = {"action": "improve", "selected_checkpoint": "best", "reason": "observed bottleneck", "next_experiment": "tune permitted setting; test whether it helps using the same frozen check"}
        return SimpleNamespace(content=json.dumps(obj))
    result = decide_component_progress({}, {}, {}, [], [], SimpleNamespace(chat=chat))
    assert result['action'] == 'improve' and result['reason'] == 'observed bottleneck'
    assert len(calls) == 2


@pytest.mark.parametrize("field,value", [("reason", True), ("next_experiment", {}), ("selected_checkpoint", None)])
def test_invalid_field_is_named_and_still_rejected_after_three(field, value):
    calls = []
    def chat(**kwargs):
        calls.append(kwargs)
        obj = {"action": "advance", "reason": "no justified improvement", "next_experiment": "", "selected_checkpoint": "best"}
        obj[field] = value
        return SimpleNamespace(content=json.dumps(obj))
    result = decide_component_progress({}, {}, {}, [], [], SimpleNamespace(chat=chat))
    assert result['action'] == 'unavailable' and len(calls) == 3
    assert field+' must be a string' in result['reason']
