"""A changed integration must not silently lose its qualified starting component's gain."""
import json

import pytest

from autoresearch.agents.memory_agent.handler import MemoryAgentHandler
from autoresearch.agents.memory_agent.tests.test_method_feedback import (
    LLM, decision, enabled, seed, store,
)


@pytest.mark.parametrize('problem', [None, 'same_source', 'metadata_only',
    'different_protocol', 'local_metric', 'different_baseline', 'not_a_parent',
    'unverified_component', 'failed_component', 'missing_source', 'within_tolerance',
    'failed_integration', 'higher_is_better', 'valid_not_best'])
async def test_component_parent_regression_uses_existing_method_reflection(
        store, enabled, monkeypatch, problem):
    monkeypatch.setenv('AUTORESEARCH_METHOD_FEEDBACK_ROUNDS', '2')
    lower = problem != 'higher_is_better'
    old_metric, new_metric = ((1.5, 2.0) if lower else (2.5, 2.0))
    source = 'diff --git a/model.py b/model.py\n-old\n+good\n'
    common = {'metric_name': 'objective', 'metric_unit': 'units',
              'lower_is_better': lower, 'baseline_digest': 'sha256:frozen',
              'baseline_relative_diff': source, 'baseline_diff_error': None,
              'measurement_evidence': {'valid': True, 'value': old_metric,
                  'stdout_sha256': 'old-output', 'attempt_id': 'attempt-old',
                  'protocol_fingerprint': 'official-evaluator'}}
    _, old_run, old_ref = seed(store, mode='component', key='old', metric=old_metric,
        feedback={**common, 'checkpoint': '/saved/component',
            'method_implementation_review': {
                'verdict': 'faithful', 'task_contract_review': {'verdict': 'compliant'}},
            'component_execution': {'mode': 'component', 'execution_succeeded': True,
                'attempt_contract_result': {'passed': True}}})
    current = {**common, 'measurement_evidence': {
        **common['measurement_evidence'], 'value': new_metric,
        'stdout_sha256': 'new-output', 'attempt_id': 'attempt-new'},
        'accepted': False, 'verdict': 'valid_not_best',
        'candidate_admissibility': {'verdict': 'pass'},
        'baseline_relative_diff': 'diff --git a/model.py b/model.py\n-old\n+new\n',
        'component_execution': {'mode': 'integration', 'all_components_accepted': True,
            'execution_succeeded': True, 'parent_checkpoints': ['/saved/component']}}
    if problem in {'same_source', 'metadata_only'}:
        current['baseline_relative_diff'] = source
    if problem == 'metadata_only':
        current['component_execution']['method_fingerprint'] = 'renamed-method'
    if problem in {'different_protocol', 'local_metric'}:
        current['measurement_evidence']['protocol_fingerprint'] = (
            None if problem == 'local_metric' else 'another-evaluator')
    if problem == 'different_baseline':
        current['baseline_digest'] = 'sha256:other'
    if problem == 'not_a_parent':
        current['component_execution']['parent_checkpoints'] = ['/saved/other']
    if problem == 'missing_source':
        current['baseline_relative_diff'] = None
    if problem == 'within_tolerance':
        current['measurement_evidence']['minimum_improvement'] = 0.6
    if problem == 'failed_integration':
        current['component_execution']['execution_succeeded'] = False
    if problem == 'valid_not_best':
        current.update(accepted=False, verdict='valid_not_best')
    _, run_ref, result_ref = seed(store, key='new', metric=new_metric, feedback=current)
    history = [{'mode': 'component', 'run_ref': old_run, 'checkpoint': '/saved/component',
        'metric': old_metric, 'contract_passed': problem != 'failed_component',
        'verification_verdict': 'unverified' if problem == 'unverified_component' else 'verified'}]
    memory = {'best_metric': new_metric, 'best_result_ref': result_ref,
              'component_history': history}
    # A complete regression shares the whole-result review budget; leave one slot here.
    store.put(kind='method_improvement_review', session_id='feedback', key='previous-review',
        obj={'status': 'completed', 'trigger': 'complete_result',
             'decision': {'route': 'report', 'reason': 'Previous method reviewed.'}},
        emitter='MemoryAgent')
    llm = LLM({**decision('integration_parent'), 'requires_core_reconstruction': False})
    handler = MemoryAgentHandler(spec_store=store, llm_client=llm)
    diagnosis = 'component_combination_retry' if problem == 'valid_not_best' else 'valid_improvement'

    review_ref = await handler._review_complete_result(
        'feedback', run_ref, memory, {'verdict': 'verified'}, diagnosis)

    expected = problem in {None, 'higher_is_better', 'valid_not_best'}
    assert (review_ref is not None) == expected
    assert len(llm.calls) == int(expected)
    if expected:
        assert store.get(review_ref)['trigger'] == 'complete_metric_regression'
        evidence = json.loads(llm.calls[0]['messages'][1]['content'])['evidence']
        assert evidence['integration_parent']['ref'] == old_ref
        assert evidence['integration_parent']['method_metric'] == old_metric
        assert memory['best_result_ref'] == result_ref
        assert store.get(result_ref)['method_metric'] == new_metric
        assert store.get(old_ref)['method_metric'] == old_metric
        assert await handler._review_complete_result(
            'feedback', run_ref, memory, {'verdict': 'verified'}, diagnosis) == review_ref
        assert len(llm.calls) == 1
