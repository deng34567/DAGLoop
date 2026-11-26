"""Reflected extensions keep their evidence and avoid reconstructing the Method."""
import json
from types import SimpleNamespace

import pytest

from autoresearch.agents.memory_agent.tests.test_method_feedback import (
    SM_ROOT, LLM, decision, enabled, fake_bus, seed, store,
)
from autoresearch.agents.method_agent.core import MethodError, method_step
from autoresearch.agents.memory_agent.handler import MemoryAgentHandler, _method_review_trigger
from autoresearch.experiments.repo_improve.tests.test_component_plan import _component
from autoresearch.experiments.repo_improve.runtime_evidence import (
    can_revalidate_reconciliation_candidate)
from autoresearch.orchestrator.engine import OrchestratorEngine
from autoresearch.orchestrator.session_state import SessionState
from autoresearch.spec_store.identity import artifact_digest


@pytest.mark.parametrize('trigger,incremental', [
    ('complete_result', True), ('legal_off_method_gain', True),
    ('repeated_verified_no_gain', True), ('cap_expectation_shortfall', True),
    ('component_method_reconciliation', True),
    ('mir_task_contract_violation', False), ('complete_metric_regression', True),
])
async def test_reflection_revision_routes_by_actual_problem(
        store, enabled, fake_bus, trigger, incremental):
    method_ref, _, result_ref = seed(store)
    benchmark_ref = store.put(kind='benchmark_spec', session_id='feedback', key='frozen',
        obj={'benchmark_id': 'benchmark-frozen'}, emitter='EvaluatorAgent')
    baseline_ref = store.put(kind='baseline_result', session_id='feedback', key='frozen',
        obj={'reproduced': True, 'metric_value': 3.0, 'benchmark_id': 'benchmark-frozen'},
        emitter='RunnerAgent')
    review_ref = store.put(kind='method_improvement_review', session_id='feedback', key='slot-1',
        obj={'status': 'completed', 'trigger': trigger, 'method_ref': method_ref,
             'decision': decision()}, emitter='MemoryAgent')
    engine = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT, spec_store=store)
    await engine.start()
    state = SessionState(session_id='feedback', user_id='u', mode='benchmark',
        current_state='RevisingMethod', method_ref=method_ref,
        baseline_result_ref=baseline_ref, benchmark_spec_ref=benchmark_ref,
        current_benchmark_id='benchmark-frozen', method_feedback_ref=review_ref,
        method_revision_round=1, method_feedback_round=1,
        method_feedback_in_progress=True)
    payload = engine._entry_actions['RevisingMethod'][1](state, None)
    assert payload.get('incremental_revision', False) is incremental
    revised = store.put(kind='method', session_id='feedback', key='revised',
        obj={'approach': 'A with B', 'components': [_component('A'), _component('B', depends_on=['A'])]},
        emitter='MethodAgent')
    await engine._on_RevisingMethod_method_generated(
        state, SimpleNamespace(payload={'method_ref': revised}))
    assert state.current_state == 'DeliberatingMethod'
    assert state.baseline_result_ref == baseline_ref
    assert store.get(result_ref)['accepted'] is True
    if incremental:
        assert state.component_plan_audit_pending
        assert state.method_feedback_in_progress
        assert fake_bus.published[-1].event_type == 'method.deliberation_requested'
        payload = fake_bus.published[-1].payload
        assert payload['component_plan_audit'] is True
        assert payload['method_ref'] == revised
        assert payload['baseline_result_ref'] == baseline_ref
        rounds = (state.method_revision_round, state.method_feedback_round)
        await engine._on_DeliberatingMethod_method_deliberated(
            state, SimpleNamespace(payload={'method_ref': revised}))
        assert state.current_state == 'PlanningExperiment'
        assert not state.component_plan_audit_pending
        assert not state.method_feedback_in_progress
        assert (state.method_revision_round, state.method_feedback_round) == rounds
        assert fake_bus.published[-1].event_type == 'experiment.planning_requested'


@pytest.mark.parametrize('core_rebuild,expected', [
    (False, 'DeliberatingMethod'), (True, 'DeliberatingMethod'),
])
async def test_regression_reflection_reconstructs_only_when_explicitly_required(
        store, enabled, fake_bus, core_rebuild, expected):
    method_ref, _, result_ref = seed(store)
    benchmark_ref = store.put(kind='benchmark_spec', session_id='feedback', key='frozen',
        obj={'benchmark_id': 'benchmark-frozen'}, emitter='EvaluatorAgent')
    baseline_ref = store.put(kind='baseline_result', session_id='feedback', key='frozen',
        obj={'reproduced': True, 'metric_value': 3.0, 'benchmark_id': 'benchmark-frozen'},
        emitter='RunnerAgent')
    review_ref = store.put(kind='method_improvement_review', session_id='feedback', key='regression',
        obj={'status': 'completed', 'trigger': 'complete_metric_regression',
             'method_ref': method_ref,
             'decision': {**decision(), 'requires_core_reconstruction': core_rebuild}},
        emitter='MemoryAgent')
    engine = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT, spec_store=store)
    await engine.start()
    state = SessionState(session_id='feedback', user_id='u', mode='benchmark',
        current_state='RevisingMethod', method_ref=method_ref,
        baseline_result_ref=baseline_ref, benchmark_spec_ref=benchmark_ref,
        current_benchmark_id='benchmark-frozen', method_feedback_ref=review_ref,
        method_feedback_in_progress=True)
    revised = store.put(kind='method', session_id='feedback', key='repair',
        obj={'approach': 'Local repair', 'components': [_component('A')]}, emitter='MethodAgent')

    await engine._on_RevisingMethod_method_generated(
        state, SimpleNamespace(payload={'method_ref': revised}))

    assert state.current_state == expected
    assert state.component_plan_audit_pending is (not core_rebuild)
    assert state.baseline_result_ref == baseline_ref
    assert store.get(result_ref)['accepted'] is True


def component_observation():
    measurement = {'valid': True, 'value': 1.5, 'attempt_id': 'attempt-bound',
                   'stdout_sha256': 'bound-output', 'minimum_improvement': 0.0,
                   'protocol': {'kind': 'component_local'}, 'official_gate_verified': False}
    return {'accepted': False, 'method_metric': 1.5, 'lower_is_better': True,
            'checkpoint': '/saved/observation', 'measurement_evidence': measurement,
            'candidate_admissibility': {'verdict': 'insufficient_evidence',
                                       'attempt_id': 'attempt-bound'},
            'repository_comparison_review': {'verdict': 'pass'},
            'method_implementation_review': {'verdict': 'drifted',
                                            'task_contract_review': {'verdict': 'compliant'}},
            'component_execution': {'mode': 'component', 'role': 'direct_gain',
                'execution_succeeded': True, 'attempt_contract_result': {
                    'passed': True, 'parent_metric': 3.0, 'candidate_metric': 1.5,
                    'parent_evidence': {'host_bound': True},
                    'measurement_evidence': measurement}}}


def unqualified_revalidation_source(*, review_verdict='drifted', role='indirect_gain',
                                    status='drifted', execution_succeeded=True):
    """Legal measured code may seed a fresh whole run without qualifying its old score."""
    attempt_id = 'attempt-bound-source'
    checkpoint = '/saved/unqualified-source'
    measurement = {'valid': True, 'value': 1.0, 'attempt_id': attempt_id,
                   'stdout_sha256': 'a' * 64, 'minimum_improvement': 0.0,
                   'gates': {'/candidate_valid': True},
                   'protocol': {'kind': 'component_local'},
                   'official_gate_verified': False}
    return {'accepted': False, 'verdict': 'invalid', 'method_metric': 1.0,
            'baseline_metric': 3.0, 'metric_name': 'runtime_seconds',
            'metric_unit': 'seconds', 'lower_is_better': True,
            'method_ref': 'spec://method/source', 'checkpoint': checkpoint,
            'component_attempt_id': attempt_id, 'measurement_evidence': measurement,
            'candidate_admissibility': {'verdict': 'insufficient_evidence',
                                        'attempt_id': attempt_id},
            'repository_comparison_review': {'verdict': 'pass'},
            'method_implementation_review': {
                'verdict': review_verdict, 'method_ref': 'spec://method/source',
                'task_contract_review': {'verdict': 'compliant'}},
            'component_evidence_error': 'parent comparison was not bound',
            'component_execution': {'mode': 'component', 'role': role, 'status': status,
                'execution_succeeded': execution_succeeded,
                'attempt_checkpoint': checkpoint, 'attempt_diff_ref': checkpoint + '#git-diff',
                'attempt_contract_result': {'passed': False,
                    'reason': 'component attribution or parent binding failed',
                    'measurement_evidence': measurement}}}


@pytest.mark.parametrize('shape', ['premature_whole_implementation', 'stale_plan_parent_binding'])
def test_legal_unqualified_checkpoint_routes_to_fresh_whole_revalidation(shape):
    if shape == 'premature_whole_implementation':
        candidate = unqualified_revalidation_source()
        diagnosis = 'candidate_contract_violation'
    else:
        candidate = unqualified_revalidation_source(
            review_verdict='partial', role='direct_gain', status='execution_failed',
            execution_succeeded=False)
        diagnosis = 'component_expectation_failed'
    incumbent = {'accepted': True, 'method_metric': 2.0, 'baseline_metric': 3.0,
                 'metric_name': 'runtime_seconds', 'metric_unit': 'seconds',
                 'lower_is_better': True}

    assert can_revalidate_reconciliation_candidate(candidate)
    assert _method_review_trigger(
        candidate, {'best_result_ref': 'spec://result/incumbent', 'best_metric': 2.0},
        diagnosis, incumbent) == 'component_method_reconciliation'
    assert candidate['accepted'] is False


@pytest.mark.parametrize('problem', [
    'failed_gate', 'task_violation', 'attempt_mismatch', 'checkpoint_mismatch',
    'failed_admissibility', 'failed_comparison', 'missing_output_binding'])
def test_unqualified_checkpoint_revalidation_keeps_safety_gates(problem):
    candidate = unqualified_revalidation_source()
    if problem == 'failed_gate':
        candidate['measurement_evidence']['gates']['/candidate_valid'] = False
    elif problem == 'task_violation':
        candidate['method_implementation_review']['task_contract_review']['verdict'] = 'violation'
    elif problem == 'attempt_mismatch':
        candidate['component_attempt_id'] = 'attempt-other'
    elif problem == 'checkpoint_mismatch':
        candidate['component_execution']['attempt_checkpoint'] = '/saved/other'
    elif problem == 'failed_admissibility':
        candidate['candidate_admissibility']['verdict'] = 'fail'
    elif problem == 'failed_comparison':
        candidate['repository_comparison_review']['verdict'] = 'fail'
    elif problem == 'missing_output_binding':
        candidate['measurement_evidence']['stdout_sha256'] = ''

    assert not can_revalidate_reconciliation_candidate(candidate)


@pytest.mark.parametrize('problem', [None, 'missing_report', 'wrong_method',
    'wrong_baseline', 'no_revision', 'method_failed', 'task_violation'])
async def test_bound_cap_contract_reflection_revises_without_reconstruction(
        store, enabled, fake_bus, problem):
    method_ref, _, result_ref = seed(store)
    benchmark_ref = store.put(kind='benchmark_spec', session_id='feedback', key='frozen',
        obj={'benchmark_id': 'benchmark-frozen'}, emitter='EvaluatorAgent')
    baseline_ref = store.put(kind='baseline_result', session_id='feedback', key='frozen',
        obj={'reproduced': True, 'metric_value': 3.0, 'benchmark_id': 'benchmark-frozen'},
        emitter='RunnerAgent')
    report = {'scope': 'component_plan', 'confirmed_method_ref': method_ref,
              'baseline_result_ref': baseline_ref,
              'method_revision_required': True}
    if problem == 'wrong_method':
        report['confirmed_method_ref'] = 'method://other'
    elif problem == 'wrong_baseline':
        report['baseline_result_ref'] = 'baseline://other'
    elif problem == 'no_revision':
        report['method_revision_required'] = False
    report_ref = store.put(kind='method_debate_report', session_id='feedback',
        key='cap-reflection', obj=report, emitter='MethodAgent')
    failure = {'underlying_reason_code': 'confirmed_method_contract_conflict'}
    if problem != 'missing_report':
        failure['component_plan_checkpoint_ref'] = report_ref
    engine = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT, spec_store=store)
    await engine.start()
    if problem == 'task_violation':
        engine._current_gen_result = lambda sid: {'method_implementation_review': {
            'task_contract_review': {'verdict': 'violation'}}}
    state = SessionState(session_id='feedback', user_id='u', mode='benchmark',
        current_state='RevisingMethod', method_ref=method_ref,
        baseline_result_ref=baseline_ref, benchmark_spec_ref=benchmark_ref,
        current_benchmark_id='benchmark-frozen', last_agent_failure=failure,
        method_not_working=problem == 'method_failed')
    incremental = problem is None
    assert engine._entry_actions['RevisingMethod'][1](state, None)[
        'incremental_revision'] is incremental
    revised = store.put(kind='method', session_id='feedback', key='corrected-contract',
        obj={'approach': 'A with corrected contract', 'components': [_component('A')]},
        emitter='MethodAgent')
    await engine._on_RevisingMethod_method_generated(
        state, SimpleNamespace(payload={'method_ref': revised}))
    assert state.current_state == 'DeliberatingMethod'
    assert state.component_plan_audit_pending is incremental
    assert state.baseline_result_ref == baseline_ref
    assert store.get(result_ref)['accepted'] is True


@pytest.mark.parametrize('problem', [None, 'missing_measurement', 'unbound_attempt',
    'failed_contract', 'unbound_parent', 'no_gain', 'evidence_error', 'failed_comparison'])
@pytest.mark.parametrize('diagnosis', ['candidate_contract_violation', 'component_contract_retry'])
def test_component_reconciliation_requires_bound_positive_evidence(problem, diagnosis):
    candidate = component_observation()
    if problem == 'missing_measurement':
        candidate['measurement_evidence']['valid'] = False
    elif problem == 'unbound_attempt':
        candidate['candidate_admissibility']['attempt_id'] = 'attempt-other'
    elif problem == 'failed_contract':
        candidate['component_execution']['attempt_contract_result']['passed'] = False
    elif problem == 'unbound_parent':
        candidate['component_execution']['attempt_contract_result']['parent_evidence']['host_bound'] = False
    elif problem == 'no_gain':
        candidate['component_execution']['attempt_contract_result']['parent_metric'] = 1.0
    elif problem == 'evidence_error':
        candidate['component_evidence_error'] = 'unbound checkpoint'
    elif problem == 'failed_comparison':
        candidate['repository_comparison_review']['verdict'] = 'fail'
    assert _method_review_trigger(candidate, {}, diagnosis, None) == (
        'component_method_reconciliation' if problem is None else None)
    assert candidate['accepted'] is False


@pytest.mark.parametrize('unbound', [None, 'method', 'run', 'digest', 'snapshot', 'anti_cheat'])
@pytest.mark.parametrize('later_method', [False, True])
@pytest.mark.parametrize('diagnosis', ['candidate_contract_violation', 'component_contract_retry'])
async def test_component_reconciliation_reflects_once_without_promoting_score(
        store, enabled, unbound, later_method, diagnosis):
    _, _, incumbent_ref = seed(store)
    observation = component_observation()
    observation['method_implementation_review']['method_ref'] = 'spec://method/feedback/observation'
    if unbound == 'method':
        observation['method_implementation_review']['method_ref'] = 'spec://method/feedback/other'
    _, run_ref, result_ref = seed(store, key='observation', metric=1.5,
                                mode='component', feedback=observation)
    run = store.get(run_ref)
    verification = {'verdict': 'verified', 'run_record_ref': run_ref,
                    'run_record_digest': artifact_digest(run), 'candidate_snapshot_ref': result_ref}
    if unbound in {'run', 'digest', 'snapshot'}:
        verification[{'run': 'run_record_ref', 'digest': 'run_record_digest',
                      'snapshot': 'candidate_snapshot_ref'}[unbound]] = 'other'
    elif unbound == 'anti_cheat':
        verification['anti_cheat_flags'] = ['independent violation']
    memory = {'best_metric': 2.0, 'best_result_ref': incumbent_ref}
    llm = LLM()
    handler = MemoryAgentHandler(spec_store=store, llm_client=llm)
    reviewed_method = None
    if later_method:
        reviewed_method = store.put(kind='method', session_id='feedback', key='later',
            obj={'name': 'A with clarified staging', 'components': []}, emitter='MethodAgent')
    for _ in range(2):
        review_ref = await handler._review_complete_result(
            'feedback', run_ref, memory, verification, diagnosis,
            review_method_ref=reviewed_method)
        if unbound:
            assert review_ref is None
            assert not llm.calls
            return
        assert review_ref
    assert len(llm.calls) == 1
    review = store.get(review_ref)
    assert review['trigger'] == 'component_method_reconciliation'
    assert review['source_method_ref'] == run['experiment_id']
    assert review['method_ref'] == (reviewed_method or run['experiment_id'])
    assert review['incumbent_ref'] == incumbent_ref
    assert memory['best_metric'] == 2.0
    assert store.get(result_ref)['accepted'] is False
    supplied = json.loads(llm.calls[0]['messages'][1]['content'])
    assert supplied['evidence']['trigger']['score_admissible'] is False
    if later_method:
        assert review['off_method_evidence']['source_method']['ref'] == run['experiment_id']


async def test_already_running_positive_deliberation_does_not_start_novelty_loop(
        store, enabled, fake_bus):
    method_ref, _, _ = seed(store)
    review_ref = store.put(kind='method_improvement_review', session_id='feedback', key='slot-1',
        obj={'status': 'completed', 'trigger': 'repeated_verified_no_gain', 'decision': decision()},
        emitter='MemoryAgent')
    engine = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT, spec_store=store)
    await engine.start()
    state = SessionState(session_id='feedback', user_id='u', mode='benchmark',
        current_state='DeliberatingMethod', method_ref=method_ref,
        baseline_result_ref='baseline://unchanged', method_feedback_ref=review_ref,
        method_feedback_in_progress=True, literature_brief_ref='literature://existing')
    await engine._on_DeliberatingMethod_method_deliberated(
        state, SimpleNamespace(payload={'method_ref': method_ref}))
    assert state.current_state == 'PlanningExperiment'
    assert not state.novelty_targeted_search_pending
    assert not any(e.event_type in {'evidence.audit_requested', 'deep_research.requested'}
                   for e in fake_bus.published)


def test_incremental_method_response_keeps_executable_components(enabled):
    method = {'approach': 'Extend A with B', 'components': [_component('A'), _component('B', depends_on=['A'])]}
    llm = LLM(method)
    assert method_step({}, {}, {}, llm, incremental_revision=True) == {
        **method, 'changes_outline': [], 'why_it_should_win': '', 'assumptions': []}
    assert len(llm.calls) == 1
    assert 'INCREMENTAL' in llm.calls[0]['messages'][0]['content']


def test_incremental_method_cannot_omit_component_contracts(enabled):
    llm = LLM({'approach': 'A with B'})
    with pytest.raises(MethodError, match='component'):
        method_step({}, {}, {}, llm, incremental_revision=True)
    assert all('Omit components' not in message['content']
               for call in llm.calls for message in call['messages'])


@pytest.mark.parametrize('reason,unchanged', [
    ('component checkpoint lookup failed: remote snapshot rename failed', True),
    ('dependency checkpoint conflict: branches changed the same source', False),
])
def test_checkpoint_infrastructure_repair_does_not_require_new_mechanism(fake_bus, reason, unchanged):
    engine = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    state = SessionState(session_id='snapshot-repair', user_id='u',
        component_plan_audit_pending=True, component_ablation_failed=True,
        last_agent_failure={'underlying_reason_code': 'component_plan_audit_failed',
            'component_plan_failure': {'underlying_reason_code': 'component_plan_invalid',
                                      'underlying_reason': reason}})
    payload = engine._entry_actions['DeliberatingMethod'][1](state, None)
    assert payload['component_plan_allow_unchanged'] is unchanged
