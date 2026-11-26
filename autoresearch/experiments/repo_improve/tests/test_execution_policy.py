"""Fast mode changes scheduling, never relaxes acceptance or source boundaries."""
import json
import subprocess
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from autoresearch.experiments.repo_improve.component_plan import ComponentLedger, build_component_plan
from autoresearch.experiments.repo_improve.execution_policy import efficient_execution, method_feedback_limit
from autoresearch.experiments.repo_improve.repository_context import early_repository_context
from autoresearch.experiments.repo_improve.tests.test_component_plan import _component
from autoresearch.orchestrator.engine import OrchestratorEngine
from autoresearch.orchestrator.session_state import SessionState


@pytest.mark.parametrize('force', ['0', '1'])
def test_first_qualified_checkpoint_awaits_decision_but_failed_contract_does_not(tmp_path, monkeypatch, force):
    monkeypatch.setenv('AUTORESEARCH_EXECUTION_STRATEGY', 'efficient')
    monkeypatch.setenv('AUTORESEARCH_FORCE_KEEP_SEARCHING', force)
    ledger = ComponentLedger(str(tmp_path / 'ledger.json'), build_component_plan(
        {'components': [_component('a')]}))
    ledger.begin('a')
    failed = ledger.record_attempt('a', execution_succeeded=True, contract_passed=False,
                                   checkpoint='bad', metric_value=0.1, quality_value=0.1)
    assert failed.status != 'accepted'
    passed = ledger.record_attempt('a', execution_succeeded=True, contract_passed=True,
                                   checkpoint='qualified', metric_value=0.9, quality_value=0.9)
    assert passed.status == 'active'
    assert passed.checkpoint == 'qualified'
    advanced = ledger.decide_qualified(
        'a', result_ref='result-1', action='advance', reason='no informative follow-up')
    assert advanced.status == 'accepted'


def test_efficient_indirect_choice_keeps_downstream_parent_and_is_bounded(tmp_path, monkeypatch):
    monkeypatch.setenv('AUTORESEARCH_EXECUTION_STRATEGY', 'efficient')
    plan = build_component_plan({'components': [
        _component('feature', 'indirect_gain'),
        _component('downstream', depends_on=['feature']),
    ]})
    ledger = ComponentLedger(str(tmp_path / 'ledger.json'), plan)
    ledger.begin('feature')
    first = ledger.record_attempt('feature', execution_succeeded=True, contract_passed=True,
                                  checkpoint='local-best', quality_value=0.1)
    assert first.status == 'active'
    continued = ledger.decide_qualified('feature', result_ref='r1', action='improve',
                                         reason='test downstream compatibility',
                                         next_experiment='measure a different representation')
    assert continued.status == 'active'
    second = ledger.record_attempt('feature', execution_succeeded=True, contract_passed=True,
                                   checkpoint='downstream-fit', quality_value=0.2)
    assert second.checkpoint == 'local-best'
    ledger.decide_qualified('feature', result_ref='r2', action='improve',
                            reason='test another local proxy', next_experiment='measure proxy')
    third = ledger.record_attempt('feature', execution_succeeded=True, contract_passed=True,
                                  checkpoint='proxy-best', quality_value=0.05)
    assert third.checkpoint == 'proxy-best'
    selected = ledger.decide_qualified('feature', result_ref='r3', action='advance',
                                        reason='new representation satisfies downstream interface',
                                        selected_checkpoint='downstream-fit')
    assert selected.checkpoint == 'downstream-fit'
    ledger.begin('downstream')
    assert ledger.entry('downstream').parent_checkpoint == 'downstream-fit'


def test_efficient_direct_gain_cannot_override_host_best(tmp_path, monkeypatch):
    monkeypatch.setenv('AUTORESEARCH_EXECUTION_STRATEGY', 'efficient')
    ledger = ComponentLedger(str(tmp_path / 'ledger.json'), build_component_plan(
        {'components': [_component('gain')]}))
    ledger.begin('gain')
    ledger.record_attempt('gain', execution_succeeded=True, contract_passed=True,
                          checkpoint='fast', metric_value=0.1, quality_value=0.1)
    with pytest.raises(ValueError):
        ledger.decide_qualified('gain', result_ref='r1', action='advance', reason='unsupported',
                                selected_checkpoint='fast')
    assert ledger.entry('gain').checkpoint == 'fast'


def test_default_off_and_feedback_count(monkeypatch):
    monkeypatch.delenv('AUTORESEARCH_EXECUTION_STRATEGY', raising=False)
    monkeypatch.delenv('AUTORESEARCH_METHOD_FEEDBACK', raising=False)
    assert not efficient_execution()
    assert method_feedback_limit() == 0
    monkeypatch.setenv('AUTORESEARCH_METHOD_FEEDBACK', '1')
    monkeypatch.delenv('AUTORESEARCH_METHOD_FEEDBACK_ROUNDS', raising=False)
    assert method_feedback_limit() == 5
    for setting, expected in [('1', 1), ('0', 0), ('-2', 0), ('bad', 5), ('2', 2), ('5', 5)]:
        monkeypatch.setenv('AUTORESEARCH_METHOD_FEEDBACK_ROUNDS', setting)
        assert method_feedback_limit() == expected


def test_early_facts_respect_source_policy_without_network(tmp_path, monkeypatch):
    (tmp_path / 'README.md').write_text('Official task rules. Use frozen validation split.')
    (tmp_path / 'INSTRUCTIONS.md').write_text('PRIVATE hidden answer must not be read.')
    subprocess.run(['git', 'init', '-q', str(tmp_path)], check=True)
    facts = early_repository_context({'repo_url': str(tmp_path), 'question': 'task',
        'execution_contract': {'sources': {'forbidden': ['INSTRUCTIONS.md']}}})
    assert facts['status'] == 'available'
    assert 'Official task rules' in json.dumps(facts)
    assert 'PRIVATE hidden answer' not in json.dumps(facts)
    assert 'not a measured baseline' in facts['note']


def test_no_mirror_does_not_clone(tmp_path, monkeypatch):
    monkeypatch.setenv('AUTORESEARCH_CLONE_MIRROR_DIR', str(tmp_path))
    def forbidden(*args, **kwargs):
        raise AssertionError('No subprocess/network should run without a local checkout')
    monkeypatch.setattr(subprocess, 'run', forbidden)
    assert early_repository_context({'repo_url': 'https://example.org/team/repo.git'})['status'] == 'unavailable'
    assert early_repository_context({})['status'] == 'unavailable'


def test_mirror_basename_collision_is_not_trusted(tmp_path, monkeypatch):
    root = tmp_path / 'repo'
    root.mkdir()
    subprocess.run(['git', 'init', '-q', str(root)], check=True)
    subprocess.run(['git', '-C', str(root), 'remote', 'add', 'origin', 'https://example.org/other/repo'], check=True)
    (root / 'README.md').write_text('Wrong repository')
    monkeypatch.setenv('AUTORESEARCH_CLONE_MIRROR_DIR', str(tmp_path))
    result = early_repository_context({'repo_url': 'https://example.org/team/repo'})
    assert result['status'] == 'unavailable'
    assert result['reason'] == 'mirror origin mismatch'


def test_early_context_event_schema_and_remote_strategy(monkeypatch):
    from autoresearch.agents.runner_agent.handler import _REMOTE_AGENTIC_ENV_KEYS
    assert 'AUTORESEARCH_EXECUTION_STRATEGY' in _REMOTE_AGENTIC_ENV_KEYS
    monkeypatch.setenv('AUTORESEARCH_EXECUTION_STRATEGY', 'efficient')
    state = SessionState(session_id='test', user_id='u', research_goal_text='task')
    payload = {'problem_model_ref': 'problem://test', 'literature_brief_ref': 'literature://test',
               **OrchestratorEngine._early_repository_context(state)}
    root = Path(__file__).resolve().parents[3]
    schema = json.loads((root / 'eventbus/schemas/method.generation_requested.v1.schema.json').read_text())
    Draft202012Validator(schema).validate(payload)
