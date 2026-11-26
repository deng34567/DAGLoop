"""Enabled-policy tests with real SpecStore snapshots and persisted orchestration."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from jsonschema import Draft202012Validator

from autoresearch.agents.memory_agent.handler import MemoryAgentHandler, _method_review_trigger
from autoresearch.agents.memory_agent.tests.test_gen_handler import (
    FakeBusClient as MemoryBus, _Envelope, RESEARCH_MEMORY_VALIDATOR,
)
from autoresearch.agents.reflection_agent.core import (
    METHOD_DIAGNOSTIC_DIMENSIONS, decide_component_progress, review_complete_method,
)
from autoresearch.experiments.repo_improve.component_plan import (
    build_component_plan, ledger_for_workroot)
from autoresearch.experiments.repo_improve.search_memory import session_workroot
from autoresearch.experiments.repo_improve.tests.test_component_plan import _component
from autoresearch.orchestrator.engine import OrchestratorEngine, _METHOD_REVISION_CAP
from autoresearch.orchestrator.session_state import SessionState, save_session_state, load_session_state
from autoresearch.orchestrator.tests.unit.test_engine_min import fake_bus  # noqa: F401
from autoresearch.spec_store import SpecStore, refs
from autoresearch.spec_store.identity import artifact_digest

ROOT = Path(__file__).resolve().parents[3]
SM_ROOT = ROOT / 'orchestrator/state-machine'


def decision(evidence='component_history'):
    return dict(route='revise_method', dimensions=['miss', 'operating_conditions'],
                primary_hypothesis='Current mechanism does not cover the observed residual cases.',
                evidence_refs=[evidence], retain=['existing valid mechanism'],
                modify=[], add=['conditional correction'], remove=[],
                expected_observation='Residual validation error decreases.',
                validation_plan='Compare A and A+B using the frozen evaluator.',
                affected_dependencies=['integration'], reuse=['unchanged control'],
                estimated_cost='one candidate and its comparison', stop_condition='No credible gain.')


class LLM:
    def __init__(self, response=None):
        self.response = response if response is not None else decision()
        self.calls = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        if isinstance(self.response, Exception):
            raise self.response
        return SimpleNamespace(content=json.dumps(self.response))


@pytest.fixture
def store(tmp_path):
    value = SpecStore(tmp_path / 'store')
    yield value
    value.close()


@pytest.fixture
def enabled(monkeypatch):
    monkeypatch.setenv('AUTORESEARCH_METHOD_FEEDBACK', '1')
    monkeypatch.setenv('AUTORESEARCH_METHOD_FEEDBACK_ROUNDS', '1')
    monkeypatch.setenv('AUTORESEARCH_EXECUTION_STRATEGY', 'efficient')
    monkeypatch.setenv('AUTORESEARCH_GEN_EXPERIMENT', '1')


def seed(store, sid='feedback', metric=2.0, mode='integration', key='first', feedback=None):
    method_ref = store.put(kind='method', session_id=sid, key=key,
                           obj={'name': 'A', 'components': []}, emitter='MethodAgent')
    run = {'succeeded': metric is not None, 'exit_code': 0 if metric is not None else 1, 'metric_value': metric,
           'experiment_id': method_ref, 'environment': 'test'}
    run_ref = store.put(kind='run_record', session_id=sid,
                        key='snapshot-' + artifact_digest(run).split(':')[1],
                        obj=run, emitter='RunnerAgent')
    gen = dict(verdict='new_best', accepted=True, baseline_metric=3.0,
               method_metric=metric, lower_is_better=True, checkpoint='/saved/candidate',
               run_record_snapshot_ref=run_ref, isolated=True, reasons=['real comparison'],
               component_execution={'mode': mode, 'all_components_accepted': True})
    if mode == 'integration':
        gen['method_implementation_review'] = {
            'verdict': 'faithful', 'task_contract_review': {'verdict': 'compliant'}}
    gen.update(feedback or {})
    result_ref = store.put(kind='gen_experiment_result', session_id=sid,
                           key='snapshot-' + artifact_digest(gen).split(':')[1],
                           obj=gen, emitter='RunnerAgent')
    return method_ref, run_ref, result_ref


@pytest.mark.parametrize('reason', [
    'method_revision_format_exhausted',
    'component_plan_review_format_exhausted',
])
def test_format_exhaustion_is_an_inconclusive_negative_result(store, reason):
    handler = MemoryAgentHandler(spec_store=store, llm_client=LLM())
    memory = handler._consolidate_failure(
        run_record=None,
        baseline_result={'metric_value': 0.133275},
        failure_context={
            'reason': reason,
            'action': 'report_now',
            'method_ref': 'method://frozen',
            'detail': 'invalid component threshold',
        },
        prior_research_memory=None,
    )

    assert memory['diagnosis'] == 'inconclusive'
    assert len(memory['negative_results']) == 1
    assert reason in memory['negative_results'][0]
    assert 'invalid component threshold' in memory['negative_results'][0]
    assert 'frozen_baseline=0.133275' in memory['negative_results'][0]


@pytest.mark.parametrize('problem,entry', [
    (problem, 'direct') for problem in [None, 'wrong_baseline', 'unverified', 'failed_contract',
                                      'task_violation', 'wrong_verification', 'saved_report', 'partial',
                                      'qualified_equal', 'qualified_better', 'qualified_worse',
                                      'unverified_qualified', 'wrong_baseline_qualified', 'current_partial', 'current_partial_unbound',
                                      'bound_source', 'bound_source_closeout', 'bound_source_user_closeout']
] + [(None, 'failure'), ('partial', 'failure'),
     ('partial_integration', 'direct'), ('partial_integration_not_best', 'direct'),
     ('repeat_historical_closeout', 'direct'),
     ('partial_integration_repeat_closeout', 'direct')])
async def test_failed_extension_recovers_bound_drifted_component_without_complete_best(
        store, enabled, problem, entry):
    sid = 'retained-component'
    integration = problem in {'partial_integration', 'partial_integration_not_best',
                              'partial_integration_repeat_closeout'}
    repeated_closeout = problem in {'repeat_historical_closeout',
                                    'partial_integration_repeat_closeout'}
    baseline = {'reproduced': True, 'metric_value': 3.0}
    if integration:
        baseline.update(lower_is_better=True,
                        measurement_evidence={'valid': True, 'protocol_fingerprint': 'official'})
    baseline_ref = store.put(kind='baseline_result', session_id=sid, key='frozen',
                             obj=baseline, emitter='RunnerAgent')
    old_method, old_run, old_result = seed(store, sid=sid, mode='component', key='old')
    evidence = {'value': 2.0, 'valid': True, 'attempt_id': 'attempt-retained',
                'stdout_sha256': 'host-stdout', 'protocol': {'kind': 'component_local'}}
    if integration:
        evidence.update(protocol_fingerprint='official', protocol={'kind': 'official'})
    contract = {'passed': True, 'candidate_metric': 2.0, 'parent_metric': 3.0,
                'parent_evidence': {'host_bound': True}, 'measurement_evidence': evidence}
    observed = {**store.get(old_result), 'accepted': False, 'verdict': 'invalid',
        'method_ref': old_method, 'diagnosis': 'candidate_contract_violation',
        'baseline_digest': artifact_digest(baseline), 'measurement_evidence': evidence,
        'candidate_admissibility': {'verdict': 'insufficient_evidence', 'attempt_id': 'attempt-retained'},
        'repository_comparison_review': {'verdict': 'pass'},
        'method_implementation_review': {'verdict': 'drifted', 'method_ref': old_method,
                                       'task_contract_review': {'verdict': 'compliant'}},
        'component_execution': {'mode': 'component', 'role': 'direct_gain', 'status': 'drifted',
                               'execution_succeeded': True, 'attempt_contract_result': contract}}
    if problem == 'wrong_baseline':
        observed['baseline_digest'] = 'another-baseline'
    if problem == 'task_violation':
        observed['method_implementation_review']['task_contract_review']['verdict'] = 'violation'
    if problem == 'partial':
        observed['method_implementation_review'].update(verdict='partial', issues=['Frozen mechanism differs.'])
        observed['component_execution']['status'] = 'implementation_failed'
    if integration:
        observed.update(accepted=True, verdict='new_best',
                        diagnosis='method_description_reconciliation',
                        candidate_admissibility={'verdict': 'pass', 'attempt_id': 'attempt-retained'})
        observed['method_implementation_review']['verdict'] = 'partial'
        observed['component_execution'] = {
            'mode': 'integration', 'status': 'new_best', 'all_components_accepted': True,
            'execution_succeeded': True}
    old_result = store.put(kind='gen_experiment_result', session_id=sid,
        key='snapshot-' + artifact_digest(observed).split(':')[1], obj=observed, emitter='RunnerAgent')
    run = store.get(old_run)
    evaluation = {'verdict': 'valid_not_best', 'run_record_ref': old_run,
                  'run_record_digest': artifact_digest(run)}
    evaluation_ref = store.put(kind='evaluation_report', session_id=sid,
        key='snapshot-' + artifact_digest(evaluation).split(':')[1], obj=evaluation, emitter='EvaluatorAgent')
    verification = {'verdict': 'verified', 'run_record_ref': old_run,
        'run_record_digest': artifact_digest(run), 'evaluation_report_ref': evaluation_ref,
        'candidate_snapshot_ref': 'another-result' if problem == 'wrong_verification' else old_result}
    verification_ref = store.put(kind='verification_report', session_id=sid,
        key='snapshot-' + artifact_digest(verification).split(':')[1], obj=verification, emitter='EvaluatorAgent')
    _, current_run, current_ref = seed(store, sid=sid, metric=None, mode='component', key='failed',
                            feedback={'accepted': False, 'verdict': 'invalid'})
    if problem in {'bound_source', 'bound_source_closeout', 'bound_source_user_closeout'}:
        method_ref = store.get(current_run)['experiment_id']
        store.put(kind='method', session_id=sid, key=refs.decode(method_ref).key,
                  obj={**store.get(method_ref), 'reconciled_result_ref': old_result},
                  emitter='MethodAgent')
        if problem == 'bound_source_user_closeout':
            store.put(kind='human_idea_seed', session_id=sid, key='closeout-v1',
                      obj={'closeout': True}, emitter='Human')
    if problem in {'current_partial', 'current_partial_unbound'}:
        current_method = (None if problem == 'current_partial_unbound' else
                          'spec://method/' + sid + '/current')
        _, current_run, current_ref = seed(store, sid=sid, metric=2.5, mode=None, key='current',
            feedback={'method_ref': current_method, 'component_execution': None,
                'method_implementation_review': {'verdict': 'partial', 'method_ref': current_method,
                    'issues': ['The currently declared output component is missing.'],
                    'task_contract_review': {'verdict': 'compliant'}},
                'measurement_evidence': {'valid': True, 'value': 2.5, 'attempt_id': 'attempt-current',
                    'stdout_sha256': 'current-output', 'protocol_fingerprint': 'official',
                    'protocol': {'kind': 'official_single'}},
                'candidate_admissibility': {'verdict': 'pass', 'attempt_id': 'attempt-current'},
                'repository_comparison_review': {'verdict': 'pass'}})
    history = {'sequence': 1, 'component_id': 'retained', 'mode': 'component', 'attempt': 1,
               'status': 'drifted', 'contract_passed': problem != 'failed_contract',
               'verification_verdict': 'failed' if problem == 'unverified' else 'verified',
               'run_ref': old_run}
    if problem == 'partial':
        history.update(status='implementation_failed', mir_verdict='partial')
    if integration:
        history.update(mode='integration', mir_verdict='partial',
                       status='valid_not_best' if problem.endswith('not_best') else 'new_best')
    memory = {'best_metric': None, 'best_result_ref': None, 'component_history': [history]}
    if problem in {'current_partial', 'current_partial_unbound'}:
        memory.update(best_metric=2.5, best_result_ref=current_ref)
    if problem in {'qualified_equal', 'qualified_better', 'qualified_worse',
                   'unverified_qualified', 'wrong_baseline_qualified'}:
        metric = {'qualified_better': 1.0, 'qualified_worse': 2.5}.get(problem, 2.0)
        _, qualified_run, _ = seed(store, sid=sid, metric=metric,
                                   mode='component', key='reconciled')
        memory['component_history'].append({
            'component_id': 'reconciled', 'mode': 'component', 'role': 'direct_gain',
            'status': 'qualified', 'became_best': True, 'execution_succeeded': True,
            'contract_passed': True, 'mir_verdict': 'faithful', 'metric': metric,
            'verification_verdict': 'failed' if problem == 'unverified_qualified' else 'verified',
            'baseline_digest': 'other' if problem == 'wrong_baseline_qualified'
                else artifact_digest(baseline),
            'checkpoint': '/saved/candidate', 'run_ref': qualified_run,
        })
    if problem == 'saved_report':
        saved_report = store.put(kind='method_improvement_review', session_id=sid, key='slot-1',
            obj={'status': 'completed', 'trigger': 'component_method_reconciliation',
                 'result_ref': old_result, 'method_ref': store.get(current_run)['experiment_id'],
                 'incumbent_ref': baseline_ref,
                 'decision': {'route': 'report', 'reason': 'Run the official evaluation next.'}},
            emitter='MemoryAgent')
    response = decision(old_result)
    if problem in {'bound_source_closeout', 'bound_source_user_closeout'} or repeated_closeout:
        response.update(add=[], modify=['Bind the retained source for complete validation.'],
                        requires_core_reconstruction=False)
    llm = LLM(response)
    handler = MemoryAgentHandler(spec_store=store, llm_client=llm)
    if entry == 'direct':
        review_ref = await handler._review_retained_off_method(
            sid, current_run, memory, baseline_ref,
            closeout_requested=problem == 'bound_source_closeout' or repeated_closeout)
    else:
        prior_ref = store.put(kind='research_memory', session_id=sid, key='prior',
                              obj=memory, emitter='MemoryAgent')
        bus = MemoryBus()
        await handler.handle_consolidation_requested(bus, _Envelope(
            event_id='failed-extension', session_id=sid,
            event_type='memory.consolidation_requested', emitter='ResearchOrchestrator', payload={
                'failure_context': {'reason': 'implementation_thrashing', 'action': 'discard_hypothesis'},
                'run_record_ref': old_run, 'baseline_result_ref': baseline_ref,
                'research_memory_ref': prior_ref, 'verification_report_ref': verification_ref}))
        published = bus.published[-1]
        assert published.event_type == 'memory.consolidated', published.payload
        memory = store.get(published.payload['research_memory_ref'])
        review_ref = ((published.payload.get('diagnosis_evidence') or [None])[0]
                      if published.payload.get('diagnosis') == 'method_improvement_proposed' else None)
    if repeated_closeout:
        assert review_ref and len(llm.calls) == 1
        assert store.get(review_ref)['trigger'] == (
            'legal_off_method_gain' if integration else 'component_method_reconciliation')
        later_method, later_run, _ = seed(store, sid=sid, metric=None, key='later',
                                        feedback={'accepted': False, 'verdict': 'invalid'})
        store.put(kind='method', session_id=sid, key=refs.decode(later_method).key,
                  obj={'name': 'A revised Method', 'components': []}, emitter='MethodAgent')
        repeated = await handler._review_retained_off_method(
            sid, later_run, memory, baseline_ref, closeout_requested=True)
        assert repeated is None and len(llm.calls) == 1
        assert store.get(old_result) == observed
        return
    if problem == 'current_partial_unbound':
        assert review_ref is not None
        assert store.get(review_ref)['trigger'] == 'component_method_reconciliation'
        return
    if problem == 'bound_source':
        assert review_ref is None
        assert not llm.calls
        assert memory['component_history'] == [history]
        assert store.get(old_result) == observed
        return
    if problem == 'current_partial':
        assert review_ref is None
        assert not llm.calls
        from autoresearch.agents.memory_agent.handler import _method_review_trigger
        current = store.get(current_ref)
        assert _method_review_trigger(current, memory, 'valid_improvement', current, baseline) == 'complete_result'
        return
    if integration or problem in {None, 'saved_report', 'partial', 'qualified_worse',
                                  'bound_source_closeout', 'bound_source_user_closeout',
                                  'unverified_qualified', 'wrong_baseline_qualified'}:
        assert store.get(review_ref)['trigger'] == (
            'legal_off_method_gain' if integration else 'component_method_reconciliation')
        assert store.get(review_ref)['result_ref'] == old_result
        assert len(llm.calls) == 1
        if problem == 'saved_report':
            assert store.get(saved_report)['decision']['route'] == 'report'
            assert review_ref != saved_report
    else:
        assert review_ref is None and not llm.calls
    assert memory['best_metric'] is None and memory['best_result_ref'] is None
    assert store.get(old_result)['accepted'] is integration


@pytest.mark.parametrize('problem', [None, 'revision_budget', 'quota', 'not_exhausted', 'wrong_baseline',
    'failed_contract', 'unverified', 'no_checkpoint', 'complete_best'])
def test_exhausted_extension_closes_from_verified_indirect_checkpoint(problem):
    baseline = {'reproduced': True, 'metric_value': 3.0}
    historical = {'mode': 'component', 'component_id': 'A', 'status': 'accepted',
        'baseline_digest': artifact_digest(baseline), 'became_best': True,
        'execution_succeeded': True, 'contract_passed': True, 'metric': None,
        'verification_verdict': 'verified', 'mir_verdict': 'faithful',
        'checkpoint': '/saved/A', 'run_ref': 'run-A'}
    gen = {'baseline_digest': artifact_digest(baseline),
        'component_execution': {'mode': 'component', 'component_id': 'B', 'status': 'exhausted'}}
    memory = {'best_result_ref': None, 'component_history': [historical]}
    if problem in {'not_exhausted', 'revision_budget'}:
        gen['component_execution']['status'] = 'contract_failed'
    elif problem == 'wrong_baseline':
        historical['baseline_digest'] = 'other-protocol'
    elif problem == 'failed_contract':
        historical['contract_passed'] = False
    elif problem == 'unverified':
        historical['verification_verdict'] = 'failed'
    elif problem == 'no_checkpoint':
        historical['checkpoint'] = ''
    elif problem == 'complete_best':
        memory['best_result_ref'] = 'complete-result'
    elif problem == 'quota':
        gen.update(diagnosis='execution_failure', error='Allocated quota exceeded')
        gen['component_execution'].update(status='execution_failed', execution_succeeded=False)
    trigger = _method_review_trigger(gen, memory, 'component_execution_blocked', None, baseline,
                                     closeout_requested=problem in {'revision_budget', 'quota'})
    assert (trigger == 'qualified_component_closeout') is (problem in {None, 'revision_budget'})


@pytest.mark.parametrize('directive', [None, 'idea_only', 'another_session', 'closeout',
                                      'saved_premature_report'])
async def test_explicit_user_closeout_uses_existing_validation_route(store, enabled, directive):
    baseline = {'reproduced': True, 'metric_value': 3.0}
    baseline_ref = store.put(kind='baseline_result', session_id='feedback', key='frozen',
                             obj=baseline, emitter='RunnerAgent')
    _, run_ref, result_ref = seed(store, metric=None, mode='component', feedback={
        'accepted': False, 'baseline_digest': artifact_digest(baseline),
        'component_execution': {'mode': 'component', 'status': 'contract_failed'}})
    history = {'mode': 'component', 'component_id': 'A', 'status': 'accepted',
        'baseline_digest': artifact_digest(baseline), 'became_best': True,
        'execution_succeeded': True, 'contract_passed': True, 'metric': None,
        'verification_verdict': 'verified', 'mir_verdict': 'faithful',
        'checkpoint': '/saved/A', 'run_ref': 'run-A'}
    memory = {'best_result_ref': None, 'best_metric': None, 'component_history': [history]}
    if directive:
        store.put(kind='human_idea_seed',
            session_id='other' if directive == 'another_session' else 'feedback',
            key='closeout-v1', obj={'idea': 'Close on the best retained implementation.',
                                  'closeout': directive != 'idea_only'}, emitter='Human')
    llm = LLM({**decision('component_history[0]'), 'add': [], 'remove': ['B'],
               'retain': ['A'], 'reuse': ['/saved/A'], 'requires_core_reconstruction': False})
    verification = {'verdict': 'verified', 'anti_cheat_flags': [], 'run_record_ref': run_ref,
        'run_record_digest': artifact_digest(store.get(run_ref)), 'candidate_snapshot_ref': result_ref}
    handler = MemoryAgentHandler(spec_store=store, llm_client=llm)
    if directive == 'saved_premature_report':
        prior_report = store.put(kind='method_improvement_review', session_id='feedback',
            key='slot-0', obj={'status': 'completed', 'trigger': 'qualified_component_closeout',
                'method_ref': store.get(run_ref)['experiment_id'], 'result_ref': result_ref,
                'incumbent_ref': baseline_ref,
                'decision': {'route': 'report', 'reason': 'Formal evaluation is still required.'}},
            emitter='MemoryAgent')
    review_ref = await handler._review_complete_result(
        'feedback', run_ref, memory, verification, 'component_progress', baseline_ref)
    if directive in {'closeout', 'saved_premature_report'}:
        assert review_ref
        review = store.get(review_ref)
        assert review['trigger'] == 'qualified_component_closeout'
        evidence = json.loads(llm.calls[0]['messages'][1]['content'])['evidence']
        assert evidence['user_closeout']['closeout'] is True
        assert 'explicitly requested' in llm.calls[0]['messages'][0]['content']
        if directive == 'saved_premature_report':
            assert store.get(prior_report)['decision']['route'] == 'report'
            assert review_ref != prior_report
    else:
        assert review_ref is None and not llm.calls
    assert memory['best_result_ref'] is None and memory['best_metric'] is None
    assert memory['component_history'] == [history]
    assert store.get(result_ref)['accepted'] is False


@pytest.mark.parametrize('changed_method', [False, True])
async def test_closeout_can_repair_a_changed_method_without_repeating_same_method(
        store, enabled, changed_method):
    baseline = {'reproduced': True, 'metric_value': 3.0}
    baseline_ref = store.put(kind='baseline_result', session_id='feedback', key='frozen',
                             obj=baseline, emitter='RunnerAgent')
    feedback = {'accepted': False, 'baseline_digest': artifact_digest(baseline),
                'component_execution': {'mode': 'component', 'status': 'contract_failed'}}
    _, run_ref, result_ref = seed(store, metric=None, mode='component', feedback=feedback)
    history = {'mode': 'component', 'component_id': 'A', 'status': 'accepted',
        'baseline_digest': artifact_digest(baseline), 'became_best': True,
        'execution_succeeded': True, 'contract_passed': True, 'metric': None,
        'verification_verdict': 'verified', 'mir_verdict': 'faithful',
        'checkpoint': '/saved/A', 'run_ref': 'run-A'}
    memory = {'best_result_ref': None, 'best_metric': None, 'component_history': [history]}
    store.put(kind='human_idea_seed', session_id='feedback', key='closeout-v1',
              obj={'closeout': True}, emitter='Human')
    llm = LLM({**decision('component_history[0]'), 'add': [], 'remove': ['B'],
               'retain': ['A'], 'reuse': ['/saved/A'], 'requires_core_reconstruction': False})
    handler = MemoryAgentHandler(spec_store=store, llm_client=llm)
    verification = {'verdict': 'verified', 'anti_cheat_flags': [], 'run_record_ref': run_ref,
        'run_record_digest': artifact_digest(store.get(run_ref)), 'candidate_snapshot_ref': result_ref}
    first = await handler._review_complete_result(
        'feedback', run_ref, memory, verification, 'component_progress', baseline_ref)
    assert first
    method_ref, later_run, later_result = seed(
        store, metric=None, mode='component', key='later', feedback=feedback)
    if changed_method:
        store.put(kind='method', session_id='feedback', key='later',
                  obj={'name': 'Retained A with an inconsistent validation component',
                       'components': []}, emitter='MethodAgent')
    later_verification = {**verification, 'run_record_ref': later_run,
        'run_record_digest': artifact_digest(store.get(later_run)),
        'candidate_snapshot_ref': later_result}
    repaired = await handler._review_complete_result(
        'feedback', later_run, memory, later_verification, 'component_progress', baseline_ref)
    assert bool(repaired) == changed_method
    assert len(llm.calls) == (2 if changed_method else 1)
    assert store.get(first)['status'] == 'completed'
    assert memory['best_result_ref'] is None and memory['best_metric'] is None
    assert memory['component_history'] == [history]
    assert store.get(later_result)['accepted'] is False
    if repaired:
        assert store.get(repaired)['method_ref'] == method_ref
        assert store.get(repaired)['baseline_ref'] == baseline_ref


@pytest.mark.parametrize('retained_source_review', [True, False])
async def test_retained_closeout_is_not_reopened_for_method_metadata_alone(
        store, enabled, retained_source_review):
    baseline = {'reproduced': True, 'metric_value': 3.0}
    baseline_ref = store.put(kind='baseline_result', session_id='feedback', key='frozen',
                             obj=baseline, emitter='RunnerAgent')
    _, run_ref, result_ref = seed(store, metric=None, mode='component', feedback={
        'accepted': False, 'baseline_digest': artifact_digest(baseline),
        'component_execution': {'mode': 'component', 'status': 'exhausted'}})
    history = {'mode': 'component', 'component_id': 'A', 'status': 'accepted',
        'baseline_digest': artifact_digest(baseline), 'became_best': True,
        'execution_succeeded': True, 'contract_passed': True, 'metric': None,
        'verification_verdict': 'verified', 'mir_verdict': 'faithful',
        'checkpoint': '/saved/A', 'run_ref': 'run-A'}
    memory = {'best_result_ref': None, 'best_metric': None, 'component_history': [history]}
    llm = LLM({**decision('component_history[0]'), 'add': [], 'remove': ['B'],
               'retain': ['A'], 'reuse': ['/saved/A'], 'requires_core_reconstruction': False})
    handler = MemoryAgentHandler(spec_store=store, llm_client=llm)
    verification = {'verdict': 'verified', 'anti_cheat_flags': [], 'run_record_ref': run_ref,
        'run_record_digest': artifact_digest(store.get(run_ref)), 'candidate_snapshot_ref': result_ref}
    first = await handler._review_complete_result(
        'feedback', run_ref, memory, verification, 'component_progress', baseline_ref,
        closeout_requested=True, retained_source_review=retained_source_review)
    assert first and len(llm.calls) == 1
    later_method = store.put(kind='method', session_id='feedback', key='later',
        obj={'name': 'Reconciled current method', 'components': []}, emitter='MethodAgent')
    repeated = await handler._review_complete_result(
        'feedback', run_ref, memory, verification, 'component_progress', baseline_ref,
        review_method_ref=later_method, closeout_requested=True,
        retained_source_review=retained_source_review)
    assert bool(repeated) is (not retained_source_review)
    assert len(llm.calls) == (1 if retained_source_review else 2)
    assert store.get(first)['status'] == 'completed'
    assert memory['best_result_ref'] is None and memory['best_metric'] is None
    assert memory['component_history'] == [history]


@pytest.mark.parametrize('prior_kind', ['whole', 'different_source', 'changed_directive', 'same_source'])
async def test_closeout_rechecks_new_retained_source_after_prior_closeout(store, enabled, prior_kind):
    baseline = {'reproduced': True, 'metric_value': 3.0}
    baseline_ref = store.put(kind='baseline_result', session_id='feedback', key='frozen',
                             obj=baseline, emitter='RunnerAgent')
    method_ref, run_ref, result_ref = seed(store, metric=None, mode='component', feedback={
        'accepted': False, 'baseline_digest': artifact_digest(baseline),
        'component_execution': {'mode': 'component', 'status': 'exhausted'}})
    history = {'mode': 'component', 'component_id': 'A', 'status': 'accepted',
        'baseline_digest': artifact_digest(baseline), 'became_best': True,
        'execution_succeeded': True, 'contract_passed': True, 'metric': None,
        'verification_verdict': 'verified', 'mir_verdict': 'faithful',
        'checkpoint': '/saved/new-source', 'run_ref': 'new-source-run'}
    memory = {'best_result_ref': None, 'best_metric': None, 'component_history': [history]}
    closeout = {**decision('component_history[0]'), 'add': [], 'remove': ['B'],
                'retain': ['A'], 'reuse': ['/saved/new-source'], 'requires_core_reconstruction': False}
    llm = LLM(closeout)
    handler = MemoryAgentHandler(spec_store=store, llm_client=llm)
    if prior_kind == 'changed_directive':
        store.put(kind='human_idea_seed', session_id='feedback', key='closeout-v1',
                  obj={'closeout': True, 'idea': 'Validate the earlier retained source.'}, emitter='Human')
    verification = {'verdict': 'verified', 'anti_cheat_flags': [], 'run_record_ref': run_ref,
        'run_record_digest': artifact_digest(store.get(run_ref)), 'candidate_snapshot_ref': result_ref}
    first = await handler._review_complete_result('feedback', run_ref, memory, verification,
        'component_execution_blocked', baseline_ref, closeout_requested=True)
    assert first and len(llm.calls) == 1
    original = store.get(first)
    if prior_kind == 'changed_directive':
        store.put(kind='human_idea_seed', session_id='feedback', key='closeout-v1',
                  obj={'closeout': True, 'idea': 'Validate the newly audited best retained source.'}, emitter='Human')
    elif prior_kind != 'same_source':
        prior = dict(original)
        if prior_kind == 'whole':
            prior.update(trigger='complete_result', terminal_whole_closeout=True,
                         decision={'route': 'report'})
            prior.pop('qualified_closeout_components')
        else:
            prior['qualified_closeout_components'] = [
                {**item, 'checkpoint': '/saved/older-source'}
                for item in prior['qualified_closeout_components']]
        store.put(kind='method_improvement_review', session_id='feedback',
                  key=refs.decode(first).key, obj=prior, emitter='MemoryAgent')
    again = await handler._review_complete_result('feedback', run_ref, memory, verification,
        'component_execution_blocked', baseline_ref, closeout_requested=True)
    assert again is not None
    assert (again == first) is (prior_kind == 'same_source')
    assert len(llm.calls) == (1 if prior_kind == 'same_source' else 2)
    assert store.get(again)['qualified_closeout_components'] == original['qualified_closeout_components']
    assert memory['best_result_ref'] is None and memory['best_metric'] is None


@pytest.mark.parametrize('failure,expected', [
    ('partial', 'implementation_failure'),
    ('execution_failure', 'implementation_failure'),
    ('attempt_accounting_violation', 'implementation_failure'),
    (None, 'qualified_component_closeout'),
])
def test_closeout_does_not_hide_current_implementation_failure(failure, expected):
    baseline = {'reproduced': True, 'metric_value': 3.0}
    digest = artifact_digest(baseline)
    gen = {'baseline_digest': digest, 'method_ref': 'method://current',
        'component_execution': {'mode': 'component', 'status': 'exhausted',
                                'execution_succeeded': False}}
    if failure == 'partial':
        gen['component_execution'].update(status='implementation_failed', execution_succeeded=True)
        gen['method_implementation_review'] = {
            'verdict': 'partial', 'method_ref': gen['method_ref'],
            'issues': ['Retained adapter source differs from the frozen component.']}
    elif failure:
        gen['diagnosis'] = failure
        gen['component_execution']['status'] = 'execution_failed'
    history = {'mode': 'component', 'component_id': 'A', 'status': 'accepted',
        'baseline_digest': digest, 'became_best': True, 'execution_succeeded': True,
        'contract_passed': True, 'metric': None, 'verification_verdict': 'verified',
        'mir_verdict': 'faithful', 'checkpoint': '/saved/A', 'run_ref': 'run-A'}
    memory = {'best_result_ref': None, 'best_metric': None, 'component_history': [history]}
    assert _method_review_trigger(gen, memory, 'component_execution_blocked', None,
        baseline, closeout_requested=True, user_closeout_requested=True) == expected
    assert memory['best_result_ref'] is None
    assert memory['component_history'] == [history]


@pytest.mark.parametrize('revision_exhausted', [False, True])
async def test_component_closeout_reserves_validation_without_promoting_score(
        store, enabled, fake_bus, revision_exhausted):
    baseline = {'reproduced': True, 'metric_value': 3.0, 'benchmark_id': 'bench-closeout'}
    baseline_ref = store.put(kind='baseline_result', session_id='feedback', key='closeout',
                             obj=baseline, emitter='RunnerAgent')
    feedback = {'accepted': False, 'verdict': 'invalid',
        'baseline_digest': artifact_digest(baseline),
        'component_execution': {'mode': 'component', 'component_id': 'B',
                                'status': 'contract_failed' if revision_exhausted else 'exhausted',
                                'execution_succeeded': False}}
    method_ref, run_ref, result_ref = seed(store, metric=None, mode='component', feedback=feedback)
    history = {'mode': 'component', 'component_id': 'A', 'status': 'accepted',
        'baseline_digest': artifact_digest(baseline), 'became_best': True,
        'execution_succeeded': True, 'contract_passed': True, 'metric': None,
        'verification_verdict': 'verified', 'mir_verdict': 'faithful',
        'checkpoint': '/saved/A', 'run_ref': 'run-A'}
    memory = {'best_result_ref': None, 'best_metric': None, 'component_history': [history]}
    for i in range(3):
        store.put(kind='method_improvement_review', session_id='feedback', key=f'spent-{i}',
            obj={'status': 'completed', 'trigger': 'complete_result',
                 'decision': {'route': 'report'}}, emitter='MemoryAgent')
    closeout = {**decision('component_history[0]'), 'add': [], 'remove': ['B'],
                'retain': ['A'], 'reuse': ['/saved/A'], 'requires_core_reconstruction': False}
    llm = LLM(closeout)
    handler = MemoryAgentHandler(spec_store=store, llm_client=llm)
    verification = {'verdict': 'verified', 'anti_cheat_flags': [], 'run_record_ref': run_ref,
        'run_record_digest': artifact_digest(store.get(run_ref)), 'candidate_snapshot_ref': result_ref}
    if revision_exhausted:
        method_ref = store.put(kind='method', session_id='feedback', key='latest-revision',
            obj={'approach': 'A with unsuccessful extension B', 'components': []}, emitter='MethodAgent')
    review_ref = await handler._review_complete_result(
        'feedback', run_ref, memory, verification, 'component_execution_blocked', baseline_ref,
        review_method_ref=method_ref, closeout_requested=revision_exhausted)
    assert store.get(review_ref)['trigger'] == 'qualified_component_closeout'
    assert memory['best_result_ref'] is None and memory['best_metric'] is None
    memory_ref = store.put(kind='research_memory', session_id='feedback', key='closeout',
                           obj=memory, emitter='MemoryAgent')
    engine = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT, spec_store=store)
    await engine.start()
    state = SessionState(session_id='feedback', user_id='u', mode='benchmark',
        current_state='ConsolidatingMemory', method_ref=method_ref, baseline_result_ref=baseline_ref,
        benchmark_spec_ref='benchmark_spec://feedback/frozen', current_benchmark_id='bench-closeout',
        run_record_refs=[run_ref], method_feedback_round=3,
        method_revision_round=_METHOD_REVISION_CAP - 1)
    await save_session_state(fake_bus.db, state, auto_commit=True)
    await engine.tick(SimpleNamespace(event_type='memory.consolidated', session_id='feedback',
        emitter='MemoryAgent', payload={'research_memory_ref': memory_ref,
            'routing_verdict': 'need_experiment', 'diagnosis': 'method_improvement_proposed',
            'diagnosis_evidence': [review_ref]}))
    routed = await load_session_state(fake_bus.db, 'feedback')
    assert routed.current_state == 'RevisingMethod'
    assert routed.method_feedback_round == 3
    assert routed.method_revision_round == _METHOD_REVISION_CAP - 1
    assert fake_bus.published[-1].payload['incremental_revision'] is True
    revised_ref = store.put(kind='method', session_id='feedback', key='retained-A',
        obj={'approach': 'Retained A without failed extension B', 'components': [_component('A')]},
        emitter='MethodAgent')
    await engine._on_RevisingMethod_method_generated(
        routed, SimpleNamespace(payload={'method_ref': revised_ref}))
    assert routed.current_state == 'DeliberatingMethod'
    assert routed.component_plan_audit_pending is True
    assert routed.baseline_result_ref == baseline_ref
    assert store.get(result_ref)['accepted'] is False
    _, another_run, another_result = seed(store, metric=None, mode='component', key='later', feedback=feedback)
    another_verification = {**verification, 'run_record_ref': another_run,
        'run_record_digest': artifact_digest(store.get(another_run)), 'candidate_snapshot_ref': another_result}
    assert await handler._review_complete_result('feedback', another_run, memory,
        another_verification, 'component_execution_blocked', baseline_ref,
        closeout_requested=revision_exhausted) is None
    assert len(llm.calls) == 1


@pytest.mark.parametrize('applied,failure_branch', [(True, False), (False, False),
                                                   (True, True), (False, True)])
async def test_applied_feedback_does_not_replace_current_component_progress(
        store, enabled, monkeypatch, applied, failure_branch):
    _, run_ref, _ = seed(store, mode='component')
    memory = {'best_metric': None, 'best_result_ref': None, 'verified_findings': [],
              'negative_results': [], 'consolidated_insights': [],
              'excluded_hypotheses': [], 'divergence_seeds': [], 'next_steps': [],
              'diagnosis': 'execution_failure' if failure_branch else 'component_contract_retry',
              'diagnosis_evidence': ['Current attempt evidence.']}
    handler = MemoryAgentHandler(spec_store=store, llm_client=LLM())
    monkeypatch.setattr(handler, '_consolidate_gen', lambda *args, **kwargs: (
        memory, 'need_experiment', memory['diagnosis'], memory['diagnosis_evidence']))
    monkeypatch.setattr(handler, '_consolidate_failure', lambda **kwargs: memory)
    calls = []
    retained_ref = store.put(kind='method_improvement_review', session_id='feedback', key='applied',
                             obj={'status': 'completed', 'decision': {'reason': 'old'}}, emitter='MemoryAgent')
    current_ref = store.put(kind='method_improvement_review', session_id='feedback', key='new',
                            obj={'status': 'completed', 'decision': {'reason': 'new'}}, emitter='MemoryAgent')
    async def retained(*args, **kwargs):
        return retained_ref
    async def current(*args, **kwargs):
        calls.append('current_review')
        return current_ref if failure_branch else None
    async def advance(*args, **kwargs):
        calls.append('component_choice')
        return True
    monkeypatch.setattr(handler, '_review_retained_off_method', retained)
    monkeypatch.setattr(handler, '_review_complete_result', current)
    monkeypatch.setattr(handler, '_decide_efficient_component', advance)
    evaluation_ref = store.put(kind='evaluation_report', session_id='feedback', key='current',
                               obj={'verdict': 'valid_not_best'}, emitter='EvaluatorAgent')
    payload = {'run_record_ref': run_ref, 'evaluation_report_ref': evaluation_ref}
    if failure_branch:
        payload['failure_context'] = {'reason': 'Current execution failed.'}
    if applied:
        payload['applied_method_feedback_ref'] = store.snapshot(retained_ref)
    bus = MemoryBus()
    await handler.handle_consolidation_requested(bus, _Envelope(
        'component-after-revision', 'feedback', 'memory.consolidation_requested',
        'ResearchOrchestrator', payload))
    result = bus.published[-1]
    assert result.event_type == 'memory.consolidated', result.payload
    if applied and not failure_branch:
        assert result.payload['diagnosis'] == 'component_accepted'
        assert result.payload['routing_verdict'] == 'need_experiment'
        assert calls == ['current_review', 'component_choice']
    else:
        assert result.payload['diagnosis'] == 'method_improvement_proposed'
        expected = current_ref if applied else retained_ref
        assert result.payload['diagnosis_evidence'] == [expected]
        assert calls == (['current_review'] if applied else [])


async def test_failed_contract_repair_reports_without_falsifying_an_unmeasured_method(store, enabled):
    sid = 'unmeasured-contract-failure'
    baseline_ref = store.put(kind='baseline_result', session_id=sid, key='frozen',
        obj={'reproduced': True, 'metric_value': 3.0}, emitter='RunnerAgent')
    method_ref = store.put(kind='method', session_id=sid, key='frozen',
        obj={'approach': 'Contradictory requirements'}, emitter='MethodAgent')
    class NoLLM:
        def complete(self, *args, **kwargs):
            raise AssertionError('No measured candidate exists to review')
    handler = MemoryAgentHandler(spec_store=store, llm_client=NoLLM())
    bus = MemoryBus()
    await handler.handle_consolidation_requested(bus, _Envelope('failed-contract', sid,
        'memory.consolidation_requested', 'ResearchOrchestrator', {
            'baseline_result_ref': baseline_ref, 'method_ref': method_ref,
            'failure_context': {'reason': 'method_contract_repair_exhausted',
                                'action': 'report_now'},
            'qualified_component_closeout_requested': True}))
    event = bus.published[-1]
    assert event.event_type == 'memory.consolidated'
    assert event.payload['routing_verdict'] == 'report'
    assert event.payload['diagnosis'] == 'inconclusive'
    memory = store.get(event.payload['research_memory_ref'])
    assert memory['best_result_ref'] is None and memory['best_metric'] is None
    assert memory['excluded_hypotheses'] == []
    assert 'method_contract_repair_exhausted' in memory['negative_results'][-1]


@pytest.mark.parametrize('review_case', [
    'closeout', 'other_trigger', 'missing_components', 'stale_method', 'already_applied'])
@pytest.mark.parametrize('failure_reason', ['ComponentPlan audit budget exhausted',
                                           'experiment_design_thrashing',
                                           'method_contract_repair_exhausted'])
async def test_audit_budget_failure_closes_from_bound_history(
        store, enabled, fake_bus, review_case, failure_reason):
    sid = 'feedback'
    baseline = {'reproduced': True, 'metric_value': 3.0, 'benchmark_id': 'bench-closeout'}
    baseline_ref = store.put(kind='baseline_result', session_id=sid, key='frozen',
                             obj=baseline, emitter='RunnerAgent')
    method_ref, run_ref, result_ref = seed(store, metric=None, mode='component', feedback={
        'accepted': False, 'verdict': 'invalid', 'baseline_digest': artifact_digest(baseline),
        'component_execution': {'mode': 'component', 'component_id': 'B', 'status': 'contract_failed'}})
    prior = {key: [] for key in ('verified_findings', 'negative_results', 'excluded_hypotheses',
        'consolidated_insights', 'divergence_seeds', 'next_steps')}
    prior.update(best_metric=None, best_result_ref=None, component_history=[{
        'sequence': 1, 'attempt': 1,
        'mode': 'component', 'component_id': 'A', 'status': 'accepted',
        'baseline_digest': artifact_digest(baseline), 'became_best': True,
        'execution_succeeded': True, 'contract_passed': True, 'metric': None,
        'verification_verdict': 'verified', 'mir_verdict': 'faithful',
        'checkpoint': '/saved/A', 'run_ref': 'run-A'}])
    prior_ref = store.put(kind='research_memory', session_id=sid, key='before',
                          obj=prior, emitter='MemoryAgent')
    verification_ref = store.put(kind='verification_report', session_id=sid, key='bound', obj={
        'verdict': 'verified', 'anti_cheat_flags': [], 'run_record_ref': run_ref,
        'run_record_digest': artifact_digest(store.get(run_ref)), 'candidate_snapshot_ref': result_ref},
        emitter='EvaluatorAgent')
    llm = LLM({**decision('component_history[0]'), 'add': [], 'remove': ['B'],
               'retain': ['A'], 'reuse': ['/saved/A'], 'requires_core_reconstruction': False})
    handler = MemoryAgentHandler(spec_store=store, llm_client=llm)
    bus = MemoryBus()
    await handler.handle_consolidation_requested(bus, _Envelope('audit-failure', sid,
        'memory.consolidation_requested', 'ResearchOrchestrator', {
            'run_record_ref': run_ref, 'research_memory_ref': prior_ref,
            'baseline_result_ref': baseline_ref, 'verification_report_ref': verification_ref,
            'method_ref': method_ref, 'qualified_component_closeout_requested': True,
            'failure_context': {'reason': failure_reason, 'action': 'report_now'}}))
    published = bus.published[-1]
    assert published.event_type == 'memory.consolidated', published.payload
    assert published.payload['diagnosis'] == 'method_improvement_proposed'
    assert published.payload['routing_verdict'] == 'need_experiment'
    memory = store.get(published.payload['research_memory_ref'])
    assert memory['best_result_ref'] is None and memory['best_metric'] is None
    assert memory['component_history'] == prior['component_history']
    assert failure_reason in memory['negative_results'][-1]
    assert store.get(result_ref)['accepted'] is False

    # Exercise the downstream budget gate too: producing a review is not enough.
    from autoresearch.orchestrator.engine import _COMPONENT_REVISION_CAP, _METHOD_REVISION_CAP
    review_ref = published.payload['diagnosis_evidence'][0]
    if review_case in {'other_trigger', 'missing_components', 'stale_method'}:
        review = store.get(review_ref)
        if review_case == 'other_trigger':
            review['trigger'] = 'implementation_failure'
        elif review_case == 'missing_components':
            review['qualified_closeout_components'] = []
        else:
            review['method_ref'] = 'method://feedback/old'
        review_ref = store.put(kind='method_improvement_review', session_id=sid,
            key='altered-review', obj=review, emitter='MemoryAgent')
        memory['method_feedback_refs'] = [review_ref]
    memory_ref = store.put(kind='research_memory', session_id=sid, key='for-routing',
        obj=memory, emitter='MemoryAgent')
    engine = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT, spec_store=store)
    await engine.start()
    state = SessionState(session_id=sid, user_id='u', mode='benchmark',
        current_state='ConsolidatingMemory', method_ref=method_ref, baseline_result_ref=baseline_ref,
        benchmark_spec_ref='benchmark_spec://feedback/frozen', current_benchmark_id='bench-closeout',
        run_record_refs=[run_ref], research_memory_ref=memory_ref,
        component_revision_round=_COMPONENT_REVISION_CAP + 1,
        method_revision_round=_METHOD_REVISION_CAP - 1, method_feedback_round=1,
        last_agent_failure={'underlying_reason_code': 'component_plan_audit_failed'},
        method_feedback_ref=review_ref if review_case == 'already_applied' else None)
    await save_session_state(fake_bus.db, state, auto_commit=True)
    await engine.tick(SimpleNamespace(event_type='memory.consolidated', session_id=sid,
        emitter='MemoryAgent', payload={**published.payload, 'research_memory_ref': memory_ref,
                                       'diagnosis_evidence': [review_ref]}))
    routed = await load_session_state(fake_bus.db, sid)
    assert routed.current_state == ('RevisingMethod' if review_case == 'closeout' else 'Reporting')
    assert routed.component_revision_round == _COMPONENT_REVISION_CAP + 1
    assert routed.method_revision_round == _METHOD_REVISION_CAP - 1
    assert routed.method_feedback_round == 1
    assert store.get(memory_ref)['best_result_ref'] is None
    if review_case == 'closeout':
        assert fake_bus.published[-1].payload['incremental_revision'] is True
        assert routed.method_feedback_ref == review_ref
        revised_ref = store.put(kind='method', session_id=sid, key='retained-A',
            obj={'approach': 'Retained A without failed extension B',
                 'components': [_component('A')]}, emitter='MethodAgent')
        await engine._on_RevisingMethod_method_generated(
            routed, SimpleNamespace(payload={'method_ref': revised_ref}))
        assert routed.current_state == 'DeliberatingMethod'
        assert routed.component_plan_audit_pending is True
        assert routed.baseline_result_ref == baseline_ref
        assert store.get(result_ref)['accepted'] is False


async def test_verified_equal_complete_result_gets_bounded_method_review(store, enabled):
    baseline = dict(reproduced=True, metric_value=80, metric_name='count',
                    metric_unit='comparators', lower_is_better=True,
                    measurement_evidence={'valid': True})
    baseline_ref = store.put(kind='baseline_result', session_id='feedback', key='equal',
                             obj=baseline, emitter='RunnerAgent')
    feedback = {
        'verdict': 'valid_not_best', 'accepted': False, 'baseline_metric': 80,
        'metric_name': 'count', 'metric_unit': 'comparators',
        'candidate_admissibility': {'verdict': 'pass'},
        'measurement_evidence': {'valid': True, 'value': 80},
        'component_execution': {'mode': 'integration', 'all_components_accepted': True,
                                'execution_succeeded': True}}
    _, run_ref, result_ref = seed(store, metric=80, feedback=feedback)
    llm = LLM()
    handler = MemoryAgentHandler(spec_store=store, llm_client=llm)
    memory = {'best_result_ref': None, 'best_metric': None}
    verification = {'verdict': 'verified'}

    review_ref = await handler._review_complete_result(
        'feedback', run_ref, memory, verification, 'component_combination_retry', baseline_ref)

    assert review_ref
    review = store.get(review_ref)
    assert review['trigger'] == 'complete_verified_no_gain'
    assert review['incumbent_ref'] == baseline_ref and review['incumbent_metric'] == 80
    assert memory['best_result_ref'] is None and memory['best_metric'] is None
    assert await handler._review_complete_result(
        'feedback', run_ref, memory, verification, 'component_combination_retry', baseline_ref) == review_ref
    assert len(llm.calls) == 1
    _, second_run, _ = seed(store, metric=80, key='second', feedback=feedback)
    # A distinct result after the configured one review must not spend another slot.
    assert await handler._review_complete_result(
        'feedback', second_run, memory, verification, 'component_combination_retry', baseline_ref) is None


@pytest.mark.parametrize('stale_verification', [False, True])
@pytest.mark.parametrize('diagnosis', ['execution_failure', 'attempt_accounting_violation'])
async def test_implementation_failure_reuses_bound_reflection_and_keeps_incumbent(
        store, enabled, stale_verification, diagnosis):
    baseline_ref = store.put(kind='baseline_result', session_id='feedback', key='control',
        obj={'reproduced': True, 'metric_value': 3.0}, emitter='RunnerAgent')
    feedback = {'accepted': False, 'verdict': 'invalid', 'method_metric': None,
        'diagnosis': diagnosis, 'error': 'candidate failed its implementation check',
        'component_execution': {'mode': 'component', 'status': 'execution_failed',
                                'execution_succeeded': False}}
    _, run_ref, result_ref = seed(store, metric=None, mode='component', feedback=feedback)
    run = store.get(run_ref)
    verification = {'verdict': 'verified', 'anti_cheat_flags': [],
        'run_record_ref': run_ref, 'run_record_digest': artifact_digest(run),
        'candidate_snapshot_ref': 'stale' if stale_verification else result_ref}
    memory = {'best_result_ref': None, 'best_metric': None}
    llm = LLM()
    handler = MemoryAgentHandler(spec_store=store, llm_client=llm)
    previous_refs = [store.put(kind='method_improvement_review', session_id='feedback',
        key=f'prior-{i}', obj={'status': 'completed', 'method_ref': 'earlier-method',
            'result_ref': f'prior-result-{i}', 'decision': {'route': 'report'}},
        emitter='MemoryAgent') for i in range(4)]
    review_ref = await handler._review_complete_result(
        'feedback', run_ref, memory, verification, 'execution_failure', baseline_ref)
    if stale_verification:
        assert review_ref is None and not llm.calls
        return
    saved = store.get(review_ref)
    assert saved['trigger'] == 'implementation_failure'
    assert saved['incumbent_ref'] == baseline_ref and saved['incumbent_metric'] == 3.0
    assert memory['best_result_ref'] is None and memory['best_metric'] is None
    assert await handler._review_complete_result(
        'feedback', run_ref, memory, verification, 'execution_failure', baseline_ref) == review_ref
    assert len(llm.calls) == 1
    first_packet = json.loads(llm.calls[0]['messages'][1]['content'])
    assert [x['ref'] for x in first_packet['evidence']['prior_method_reflections']] == previous_refs[::-1][:3]
    _, new_run_ref, new_result_ref = seed(
        store, metric=None, mode='component', key='new-failure',
        feedback={**feedback, 'error': 'a distinct implementation failure'})
    new_verification = {**verification, 'run_record_ref': new_run_ref,
        'run_record_digest': artifact_digest(store.get(new_run_ref)),
        'candidate_snapshot_ref': new_result_ref}
    new_review_ref = await handler._review_complete_result(
        'feedback', new_run_ref, memory, new_verification, 'execution_failure', baseline_ref)
    assert new_review_ref != review_ref and len(llm.calls) == 2
    packet = json.loads(llm.calls[1]['messages'][1]['content'])
    assert packet['evidence']['prior_method_reflections'][0]['ref'] == review_ref


@pytest.mark.parametrize('error', ['receipt belongs to an earlier worktree', 'Allocated quota exceeded'])
async def test_bound_failure_consolidation_reflects_binding_error_without_promoting_score(
        store, enabled, error):
    baseline_ref = store.put(kind='baseline_result', session_id='feedback', key='control',
        obj={'reproduced': True, 'metric_value': 3.0}, emitter='RunnerAgent')
    method_ref, run_ref, result_ref = seed(store, metric=None, mode='component', feedback={
        'accepted': False, 'verdict': 'invalid', 'method_metric': None,
        'diagnosis': 'attempt_accounting_violation', 'error': error,
        'component_execution': {'mode': 'component', 'status': 'execution_failed',
                                'execution_succeeded': False}})
    verification_ref = store.put(kind='verification_report', session_id='feedback', key='bound',
        obj={'verdict': 'verified', 'anti_cheat_flags': [], 'run_record_ref': run_ref,
             'run_record_digest': artifact_digest(store.get(run_ref)),
             'candidate_snapshot_ref': result_ref}, emitter='EvaluatorAgent')
    llm = LLM()
    handler = MemoryAgentHandler(spec_store=store, llm_client=llm)
    bus = MemoryBus()
    await handler.handle_consolidation_requested(bus, _Envelope('binding-failure', 'feedback',
        'memory.consolidation_requested', 'ResearchOrchestrator', {
            'run_record_ref': run_ref, 'baseline_result_ref': baseline_ref,
            'verification_report_ref': verification_ref, 'method_ref': method_ref,
            'failure_context': {'reason': 'implementation_thrashing', 'action': 'discard_hypothesis'}}))
    published = bus.published[-1]
    expected = 'report' if error == 'Allocated quota exceeded' else 'need_experiment'
    assert published.event_type == 'memory.consolidated'
    assert published.payload['routing_verdict'] == expected
    memory = store.get(published.payload['research_memory_ref'])
    assert memory['best_metric'] is None and memory['best_result_ref'] is None
    assert store.get(result_ref)['accepted'] is False
    assert len(llm.calls) == (expected == 'need_experiment')


@pytest.mark.parametrize('error', [
    'clone failed: connection unavailable', 'llm_provider_timeout',
    'llm_provider_content_filter', 'llm_provider_rate_limit', 'Allocated quota exceeded', 'LLMRateLimitError',
])
def test_external_failure_does_not_trigger_implementation_reflection(error):
    gen = {'diagnosis': 'execution_failure', 'error': error,
        'component_execution': {'mode': 'component', 'status': 'execution_failed',
                                'execution_succeeded': False}}
    assert _method_review_trigger(gen, {}, 'execution_failure', None) is None


@pytest.mark.parametrize('problem', [None, 'stale_verification', 'stale_method', 'stale_result_method',
                                    'faithful', 'no_issues', 'quota'])
async def test_partial_implementation_reflects_once_without_promoting_measurement(
        store, enabled, problem):
    method_ref = refs.encode('method', 'feedback', 'first')
    review = {'verdict': 'partial', 'method_ref': method_ref,
              'issues': ['The implemented allocation differs from the frozen mechanism.']}
    if problem in {'stale_method', 'stale_result_method'}:
        review['method_ref'] = 'earlier-method'
    elif problem == 'faithful':
        review['verdict'] = 'faithful'
    elif problem == 'no_issues':
        review['issues'] = []
    baseline_ref = store.put(kind='baseline_result', session_id='feedback', key='control',
        obj={'reproduced': True, 'metric_value': 3.0}, emitter='RunnerAgent')
    _, run_ref, result_ref = seed(store, metric=0.58, mode='component', feedback={
        'accepted': False, 'verdict': 'invalid',
        'method_ref': 'earlier-method' if problem == 'stale_result_method' else method_ref,
        'diagnosis': None, 'method_implementation_review': review,
        'error': 'Allocated quota exceeded' if problem == 'quota' else None,
        'component_execution': {'mode': 'component', 'status': 'implementation_failed',
                                'execution_succeeded': True}})
    verification = {'verdict': 'verified', 'anti_cheat_flags': [],
        'run_record_ref': run_ref, 'run_record_digest': artifact_digest(store.get(run_ref)),
        'candidate_snapshot_ref': 'stale' if problem == 'stale_verification' else result_ref}
    memory = {'best_result_ref': None, 'best_metric': None}
    llm = LLM()
    handler = MemoryAgentHandler(spec_store=store, llm_client=llm)
    review_ref = await handler._review_complete_result(
        'feedback', run_ref, memory, verification, 'component_contract_retry', baseline_ref)
    if problem:
        assert review_ref is None and not llm.calls
    else:
        assert store.get(review_ref)['trigger'] == 'implementation_failure'
        assert await handler._review_complete_result(
            'feedback', run_ref, memory, verification,
            'component_contract_retry', baseline_ref) == review_ref
        assert len(llm.calls) == 1
    assert store.get(result_ref)['accepted'] is False
    assert memory['best_result_ref'] is None and memory['best_metric'] is None


async def test_implementation_reflection_routes_existing_local_revision(store, enabled, fake_bus):
    method_ref, run_ref, _ = seed(store)
    baseline_ref = store.put(kind='baseline_result', session_id='feedback', key='control',
        obj={'metric_value': 3.0}, emitter='RunnerAgent')
    review_ref = store.put(kind='method_improvement_review', session_id='feedback', key='failure',
        obj={'status': 'completed', 'trigger': 'implementation_failure',
             'method_ref': method_ref, 'incumbent_ref': baseline_ref, 'incumbent_metric': 3.0,
             'decision': {**decision(), 'requires_core_reconstruction': False}}, emitter='MemoryAgent')
    memory_ref = store.put(kind='research_memory', session_id='feedback', key='failure',
        obj={'best_result_ref': None, 'best_metric': None, 'method_feedback_refs': [review_ref]},
        emitter='MemoryAgent')
    engine = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT, spec_store=store)
    await engine.start()
    state = SessionState(session_id='feedback', user_id='u', mode='benchmark',
        current_state='ConsolidatingMemory', method_ref=method_ref,
        baseline_result_ref=baseline_ref, run_record_refs=[run_ref], method_feedback_round=1)
    await save_session_state(fake_bus.db, state, auto_commit=True)
    await engine.tick(SimpleNamespace(event_type='memory.consolidated', session_id='feedback',
        emitter='MemoryAgent', payload={'research_memory_ref': memory_ref,
            'routing_verdict': 'need_experiment', 'diagnosis': 'method_improvement_proposed',
            'diagnosis_evidence': [review_ref]}))
    state = await load_session_state(fake_bus.db, 'feedback')
    assert state.current_state == 'RevisingMethod' and state.method_feedback_round == 1
    request = next(e for e in fake_bus.published if e.event_type == 'method.revise_requested')
    assert request.payload['incremental_revision'] is True


@pytest.mark.parametrize('mutation', [
    {'method_metric': None}, {'candidate_admissibility': {'verdict': 'insufficient_evidence'}},
    {'measurement_evidence': {'valid': False, 'value': 80}},
    {'component_execution': {'mode': 'integration', 'all_components_accepted': False,
                             'execution_succeeded': True}},
    {'component_execution': {'mode': 'integration', 'all_components_accepted': True,
                             'execution_succeeded': False}},
])
def test_unqualified_equal_result_is_not_complete_no_gain(mutation):
    baseline = dict(reproduced=True, metric_value=80, metric_name='count',
                    metric_unit='comparators', lower_is_better=True,
                    measurement_evidence={'valid': True})
    candidate = dict(method_metric=80, baseline_metric=80, metric_name='count',
        metric_unit='comparators', lower_is_better=True, checkpoint='/saved/candidate',
        verdict='valid_not_best', candidate_admissibility={'verdict': 'pass'},
        measurement_evidence={'valid': True, 'value': 80},
        method_implementation_review={'verdict': 'faithful',
            'task_contract_review': {'verdict': 'compliant'}},
        component_execution={'mode': 'integration', 'all_components_accepted': True,
                             'execution_succeeded': True})
    assert _method_review_trigger({**candidate, **mutation}, {},
        'component_combination_retry', None, baseline) is None


@pytest.mark.parametrize('spent', [False, True])
async def test_complete_no_gain_routes_local_revision_with_baseline_and_budget(
        store, enabled, fake_bus, spent):
    method_ref, run_ref, _ = seed(store, metric=80)
    baseline_ref = store.put(kind='baseline_result', session_id='feedback', key='control',
                             obj={'metric_value': 80}, emitter='RunnerAgent')
    review_ref = store.put(kind='method_improvement_review', session_id='feedback', key='equal',
        obj={'status': 'completed', 'trigger': 'complete_verified_no_gain',
             'method_ref': method_ref, 'incumbent_ref': baseline_ref, 'incumbent_metric': 80,
             'decision': {**decision(), 'requires_core_reconstruction': False}}, emitter='MemoryAgent')
    memory_ref = store.put(kind='research_memory', session_id='feedback', key='equal',
        obj={'best_result_ref': None, 'best_metric': None, 'method_feedback_refs': [review_ref]},
        emitter='MemoryAgent')
    engine = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT, spec_store=store)
    await engine.start()
    state = SessionState(session_id='feedback', user_id='u', mode='benchmark',
        current_state='ConsolidatingMemory', method_ref=method_ref,
        baseline_result_ref=baseline_ref, run_record_refs=[run_ref], method_feedback_round=int(spent))
    await save_session_state(fake_bus.db, state, auto_commit=True)
    await engine.tick(SimpleNamespace(event_type='memory.consolidated', session_id='feedback',
        emitter='MemoryAgent', payload={'research_memory_ref': memory_ref,
            'routing_verdict': 'need_experiment', 'diagnosis': 'method_improvement_proposed',
            'diagnosis_evidence': [review_ref]}))
    state = await load_session_state(fake_bus.db, 'feedback')
    assert state.current_state == ('Reporting' if spent else 'RevisingMethod')
    assert state.method_feedback_round == 1
    assert state.baseline_result_ref == baseline_ref
    assert store.get(state.research_memory_ref)['best_result_ref'] is None
    if not spent:
        request = next(e for e in fake_bus.published if e.event_type == 'method.revise_requested')
        assert request.payload['incremental_revision'] is True


@pytest.mark.parametrize('repeat', [1, 2000])
async def test_complete_review_receives_bounded_runner_feedback(store, enabled, repeat):
    feedback = {'bottleneck': 'Remaining global-load latency. ' * repeat,
                'rationale': 'Reduce loads while preserving exact outputs. ' * repeat,
                'config': {'block_size': 256, 'notes': 'measured settings ' * repeat}}
    _, run_ref, result_ref = seed(store, feedback=feedback)
    llm = LLM({'route': 'report', 'reason': 'No supported change.'})
    handler = MemoryAgentHandler(spec_store=store, llm_client=llm)
    memory = {'best_metric': 2.0, 'best_result_ref': result_ref}

    await handler._review_complete_result(
        'feedback', run_ref, memory, {'verdict': 'verified'}, 'valid_improvement')

    packet = json.loads(llm.calls[0]['messages'][1]['content'])
    supplied = packet['evidence'][result_ref]['implementation_feedback']
    assert supplied['evidence_status'] == 'runner-reported; not independent profiling'
    if repeat == 1:
        assert supplied['bottleneck'] == feedback['bottleneck']
        assert supplied['rationale'] == feedback['rationale']
        assert json.loads(supplied['config']) == feedback['config']
    else:
        for field, limit in [('bottleneck', 4000), ('rationale', 4000), ('config', 6000)]:
            assert len(supplied[field]) < limit + 200
            assert f'[truncated; full content in {result_ref}]' in supplied[field]


@pytest.mark.parametrize('mutation', [
    {'dimensions': ['invented']}, {'evidence_refs': ['invented']},
    {'validation_plan': ''}, {'modify': [], 'add': [], 'remove': []},
    {'retain': 'not a list'}, {'expected_observation': None},
])
def test_bad_review_cannot_request_revision(mutation):
    llm = LLM({**decision(), **mutation})
    output = review_complete_method({}, {}, {'component_history': []}, llm)
    assert output['route'] == 'unavailable'
    assert output['error_detail']
    assert len(llm.calls) == 2


def test_off_method_review_accepts_explicit_nested_incumbent_reference():
    candidate_ref = 'spec://gen_experiment_result/session/candidate'
    incumbent_ref = 'spec://gen_experiment_result/session/incumbent'
    evidence = {candidate_ref: {}, 'off_method_comparison': {
        'candidate': {'result_ref': candidate_ref},
        'incumbent': {'result_ref': incumbent_ref}}}
    proposal = {**decision(candidate_ref),
                'evidence_refs': [candidate_ref, incumbent_ref]}

    output = review_complete_method({}, {'ref': incumbent_ref}, evidence, LLM(proposal))

    assert output['route'] == 'revise_method'
    assert output['evidence_refs'] == [candidate_ref, incumbent_ref]


def test_complete_review_accepts_supplied_repository_source_identity():
    source = {'path': 'repo/instruction.md', 'sha256': 'a' * 64,
              'content': 'The candidate must preserve the required interface.'}
    evidence = {'task_requirements': {'repository_sources': [source]}}
    llm = LLM(decision(source['path']))

    output = review_complete_method({}, {}, evidence, llm)

    assert output['route'] == 'revise_method'
    assert len(llm.calls) == 1


def test_complete_review_accepts_supplied_measurement_attempt_id():
    attempt_id = 'attempt-current-measurement'
    evidence = {'candidate': {'measurement_evidence': {
        'attempt_id': attempt_id, 'valid': True, 'value': 2.0}}}
    llm = LLM(decision(attempt_id))

    output = review_complete_method({}, {}, evidence, llm)

    assert output['route'] == 'revise_method'
    assert output['evidence_refs'] == [attempt_id]
    packet = json.loads(llm.calls[0]['messages'][1]['content'])
    assert attempt_id in packet['allowed_evidence_refs']
    assert len(llm.calls) == 1


def test_complete_review_binds_unique_workroot_relative_evidence_reference():
    relative_ref = (
        'repo-30/tasks/example/environment/{store.go,index.go} '
        '(byte-identical to parent; per-file sha256 in analysis/preflight-evidence.json)')
    absolute_ref = '/data/run/session/' + relative_ref
    evidence = {'candidate': {'ablation_report': [{
        'evidence': {'ref': absolute_ref}, 'verdict': 'unmet'}]}}

    output = review_complete_method({}, {}, evidence, LLM(decision(relative_ref)))

    assert output['route'] == 'revise_method'
    assert output['evidence_refs'] == [absolute_ref]


def test_complete_review_rejects_ambiguous_relative_evidence_reference():
    relative_ref = 'repo/tasks/example/result.json'
    evidence = {'candidate': {'first_ref': '/data/run-a/' + relative_ref,
                              'second_ref': '/data/run-b/' + relative_ref}}

    output = review_complete_method({}, {}, evidence, LLM(decision(relative_ref)))

    assert output['route'] == 'unavailable'
    assert relative_ref in output['error_detail']


@pytest.mark.parametrize('evidence', [
    {'candidate': {'measurement_evidence': {'attempt_id': 'attempt-actual'}}},
    {'candidate': {'stdout_excerpt': 'Runner mentioned attempt-unbound'}},
])
def test_complete_review_rejects_attempt_id_not_in_supplied_reference_fields(evidence):
    llm = LLM(decision('attempt-unbound'))

    output = review_complete_method({}, {}, evidence, llm)

    assert output['route'] == 'unavailable'
    assert len(llm.calls) == 2
    assert 'attempt-unbound' in output['error_detail']


@pytest.mark.parametrize('source', [
    {'path': 'repo/instruction.md'},
    {'path': 'repo/instruction.md', 'sha256': '', 'content': 'rules'},
    {'path': 'repo/instruction.md', 'sha256': 'invented', 'content': 'rules'},
    {'path': 'repo/instruction.md', 'sha256': 'a' * 64},
])
def test_complete_review_rejects_unbound_repository_path(source):
    evidence = {'task_requirements': {'repository_sources': [source]}}
    output = review_complete_method({}, {}, evidence, LLM(decision(source['path'])))
    assert output['route'] == 'unavailable'


def test_complete_review_rejects_source_path_absent_from_packet():
    evidence = {'task_requirements': {'repository_sources': [{
        'path': 'repo/other.md', 'sha256': 'a' * 64, 'content': 'rules'}]}}
    output = review_complete_method({}, {}, evidence, LLM(decision('repo/instruction.md')))
    assert output['route'] == 'unavailable'


def test_complete_review_explicitly_names_bound_evidence_ids():
    candidate_ref = 'spec://gen_experiment_result/session/candidate'
    class InstructionFollowingLLM(LLM):
        def chat(self, **kwargs):
            self.calls.append(kwargs)
            packet = json.loads(kwargs['messages'][1]['content'])
            supplied = packet.get('allowed_evidence_refs', [])
            reference = candidate_ref if candidate_ref in supplied else 'measurement_evidence'
            return SimpleNamespace(content=json.dumps(decision(reference)))
    llm = InstructionFollowingLLM()
    evidence = {candidate_ref: {'measurement_evidence': {'valid': True, 'value': 2.0}}}
    output = review_complete_method({}, {}, evidence, llm)
    assert output['route'] == 'revise_method'
    assert len(llm.calls) == 1
    assert 'measurement_evidence' not in json.loads(
        llm.calls[0]['messages'][1]['content'])['allowed_evidence_refs']


def test_complete_review_preserves_provider_timeout_type():
    import httpx
    from openai import APITimeoutError
    error = APITimeoutError(request=httpx.Request('POST', 'https://provider.invalid/chat'))
    output = review_complete_method({}, {}, {}, LLM(error))
    assert output['route'] == 'unavailable'
    assert 'APITimeoutError' in output['error_detail']


def test_eight_views_one_call_and_fail_open():
    assert len(METHOD_DIAGNOSTIC_DIMENSIONS) == 8
    llm = LLM()
    assert review_complete_method({}, {}, {'component_history': []}, llm)['route'] == 'revise_method'
    assert len(llm.calls) == 1
    assert all(key in llm.calls[0]['messages'][0]['content'] for key in METHOD_DIAGNOSTIC_DIMENSIONS)
    assert ('implementation_failure is an input trigger, not one of the eight diagnostic '
            'dimension keys') in llm.calls[0]['messages'][0]['content']
    assert review_complete_method({}, {}, {}, LLM(RuntimeError('offline')))['route'] == 'unavailable'


def test_malformed_review_gets_one_bounded_protocol_repair():
    class RepairLLM(LLM):
        def chat(self, **kwargs):
            self.calls.append(kwargs)
            response = {**decision(), 'dimensions': ['invented']} if len(
                self.calls) == 1 else decision()
            return SimpleNamespace(content=json.dumps(response))

    llm = RepairLLM()
    output = review_complete_method({}, {}, {'component_history': []}, llm)
    assert output['route'] == 'revise_method'
    assert len(llm.calls) == 2
    assert 'unknown diagnostic dimension' in llm.calls[1]['messages'][0]['content']
    assert 'invented' in llm.calls[1]['messages'][0]['content']
    assert 'allowed dimensions' in llm.calls[1]['messages'][0]['content']


def test_complete_review_keeps_supported_dimensions_and_audits_unknown_extras():
    response = {
        **decision(),
        'dimensions': [
            'implementation_failure',
            'operating_conditions',
            'mechanism_understanding',
            'operating_conditions',
        ],
    }
    llm = LLM(response)

    output = review_complete_method({}, {}, {'component_history': []}, llm)

    assert output['route'] == 'revise_method'
    assert output['dimensions'] == ['operating_conditions', 'mechanism_understanding']
    assert output['ignored_unsupported_dimensions'] == ['implementation_failure']
    assert len(llm.calls) == 1


def test_failed_review_keeps_both_protocol_errors_and_unknown_reference():
    class InvalidLLM(LLM):
        def chat(self, **kwargs):
            self.calls.append(kwargs)
            response = ({**decision(), 'evidence_refs': ['spec://missing']}
                        if len(self.calls) == 1 else {**decision(), 'validation_plan': ''})
            return SimpleNamespace(content=json.dumps(response))

    output = review_complete_method({}, {}, {'component_history': []}, InvalidLLM())
    assert output['route'] == 'unavailable'
    assert 'spec://missing' in output['initial_error_detail']
    assert output['error_detail'] == 'ValueError: missing validation_plan'


def test_component_choice_requires_a_specific_next_experiment():
    component = {'role': 'indirect_gain', 'downstream_contract': 'consumed by B'}
    candidates = [{'checkpoint': 'qualified-2'}]
    llm = LLM({'action': 'improve', 'selected_checkpoint': 'qualified-2',
               'next_experiment': 'measure B consumption', 'reason': 'interface uncertain'})
    choice = decide_component_progress(component, {}, {}, candidates, [], llm)
    assert choice['action'] == 'improve'
    assert choice['selected_checkpoint'] == 'qualified-2'
    assert len(llm.calls) == 1
    invalid = decide_component_progress(component, {}, {}, candidates, [], LLM({
        'action': 'improve', 'next_experiment': '', 'selected_checkpoint': 'qualified-2'}))
    assert invalid['action'] == 'unavailable'
    assert invalid['selected_checkpoint'] == 'best'


@pytest.mark.parametrize('repair_succeeds', [True, False])
def test_component_decision_repairs_json_once_without_default_advance(repair_succeeds):
    valid = json.dumps({'action': 'improve', 'selected_checkpoint': 'best',
                       'reason': 'Measure the retained mechanism',
                       'next_experiment': 'Compare the unchanged quality gates'})

    class MalformedLLM:
        def __init__(self):
            self.calls = []

        def chat(self, **kwargs):
            self.calls.append(kwargs)
            content = valid if repair_succeeds and len(self.calls) == 2 else "{'action': 'improve'}"
            return SimpleNamespace(content=content)

    llm = MalformedLLM()
    choice = decide_component_progress({'role': 'direct_gain'}, {}, {}, [], [], llm)
    assert len(llm.calls) == (2 if repair_succeeds else 3)
    assert all(call['response_format'] == 'json_object_required' for call in llm.calls)
    assert llm.calls[0]['messages'][1] == llm.calls[1]['messages'][1]
    assert llm.calls[1]['messages'][-2]['content'] == "{'action': 'improve'}"
    assert choice['action'] == ('improve' if repair_succeeds else 'unavailable')


@pytest.mark.parametrize('failure', [None, TimeoutError('decision timed out')])
def test_component_decision_failure_does_not_advance(failure):
    llm = None if failure is None else LLM(failure)
    result = decide_component_progress({}, {}, {}, [], [], llm)
    assert result['action'] == 'unavailable'
    if llm is not None:
        assert 'stream_max_s' not in llm.calls[0]


async def test_component_decision_error_blocks_consolidation_routing(
        store, enabled, monkeypatch):
    _, run_ref, _ = seed(store)
    handler = MemoryAgentHandler(spec_store=store, llm_client=LLM())
    monkeypatch.setattr(handler, '_consolidate_gen',
                        lambda *a, **k: ({}, 'need_experiment', 'component_progress', []))
    async def no_review(*a, **k):
        return None
    async def failed_decision(*a, **k):
        raise RuntimeError('component continuation decision timed out')
    async def publish_consolidated(client, session_id, causation_id, *a, **k):
        await client.publish(event_type='memory.consolidated', session_id=session_id,
                             payload={}, causation_id=causation_id)
    monkeypatch.setattr(handler, '_review_retained_off_method', no_review)
    monkeypatch.setattr(handler, '_review_complete_result', no_review)
    monkeypatch.setattr(handler, '_decide_efficient_component', failed_decision)
    monkeypatch.setattr(handler, '_publish_consolidated', publish_consolidated)
    bus = MemoryBus()
    await handler.handle_consolidation_requested(bus, _Envelope(
        event_id='decision', session_id='feedback', event_type='memory.consolidation_requested',
        emitter='VerificationAgent', payload={'run_record_ref': run_ref}))
    assert [event.event_type for event in bus.published] == ['agent.failed']
    assert 'component continuation decision timed out' in bus.published[0].payload['underlying_reason']


@pytest.mark.parametrize('action,already_accepted', [
    ('advance', False), ('improve', False), ('advance', True),
])
@pytest.mark.parametrize('failure_diagnosis', ['execution_failure', 'attempt_accounting_violation'])
async def test_unexecuted_refinement_routes_retained_component_decision(
        store, enabled, tmp_path, monkeypatch, fake_bus, action, already_accepted, failure_diagnosis):
    monkeypatch.setenv('AUTORESEARCH_WEB_WORKDIR', str(tmp_path))
    sid = 'unexecuted-refinement'
    method = {'components': [_component('feature')]}
    method_ref = store.put(kind='method', session_id=sid, key='v1',
                           obj=method, emitter='MethodAgent')
    baseline = {'reproduced': True, 'metric_value': 1.0}
    baseline_ref = store.put(kind='baseline_result', session_id=sid, key='v1',
                             obj=baseline, emitter='RunnerAgent')
    ledger = ledger_for_workroot(session_workroot(sid), build_component_plan(method),
                                 execution_identity=artifact_digest(baseline))
    contract = {'passed': True, 'measurement_evidence': {'valid': True}}
    ledger.begin('feature')
    ledger.record_attempt('feature', execution_succeeded=True, contract_passed=True,
                          checkpoint='qualified-feature', metric_value=0.8,
                          contract_result=contract)
    ledger.record_attempt('feature', execution_succeeded=False, execution_started=False,
                          error='proposed refinement has no executable premise')
    if already_accepted:
        ledger.decide_qualified('feature', result_ref='prior-choice', action='advance',
                                reason='Prior evidence-bound choice.')
    run = {'experiment_id': method_ref, 'succeeded': False, 'exit_code': 1,
           'metric_value': None, 'environment': 'test'}
    run_ref = store.put(kind='run_record', session_id=sid,
                        key='snapshot-' + artifact_digest(run).split(':')[1],
                        obj=run, emitter='RunnerAgent')
    gen = {'verdict': 'invalid', 'accepted': False, 'baseline_metric': 1.0,
           'method_metric': None, 'lower_is_better': True,
           'diagnosis': failure_diagnosis, 'error': 'refinement was not executed',
           'run_record_snapshot_ref': run_ref,
           'component_execution': {
               'mode': 'component', 'component_id': 'feature', 'role': 'direct_gain',
               'operator': 'improve', 'status': 'execution_failed',
               'execution_succeeded': False, 'qualified_attempts': 1,
               'checkpoint': 'qualified-feature', 'selected_contract_result': contract}}
    result_ref = store.put(kind='gen_experiment_result', session_id=sid,
                           key='snapshot-' + artifact_digest(gen).split(':')[1],
                           obj=gen, emitter='RunnerAgent')
    verification_ref = store.put(kind='verification_report', session_id=sid, key='v1',
        obj={'verdict': 'verified', 'run_record_ref': run_ref,
             'run_record_digest': artifact_digest(run),
             'candidate_snapshot_ref': result_ref}, emitter='EvaluatorAgent')
    llm = LLM({'action': action, 'selected_checkpoint': 'best',
               'reason': 'Retain the qualified result.',
               'next_experiment': 'One new legal diagnostic.' if action == 'improve' else ''})
    handler = MemoryAgentHandler(spec_store=store, llm_client=llm)
    bus = MemoryBus()
    await handler.handle_consolidation_requested(bus, _Envelope(
        'unexecuted-refinement', sid, 'memory.consolidation_requested',
        'ResearchOrchestrator', {'run_record_ref': run_ref,
            'verification_report_ref': verification_ref, 'baseline_result_ref': baseline_ref}))
    event = bus.published[-1]
    assert event.event_type == 'memory.consolidated'
    expected = 'component_accepted' if action == 'advance' else 'component_contract_retry'
    assert event.payload['diagnosis'] == expected
    memory = store.get(event.payload['research_memory_ref'])
    assert memory['diagnosis'] == expected
    assert memory['best_metric'] is None and not memory['verified_findings']
    assert store.get(result_ref) == gen and gen['accepted'] is False
    assert ledger.entry('feature').checkpoint == 'qualified-feature'
    assert ledger.entry('feature').status == ('accepted' if action == 'advance' else 'active')
    if action == 'advance':
        engine = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT, spec_store=store)
        await engine.start()
        state = SessionState(session_id=sid, user_id='u', mode='benchmark',
            current_state='ConsolidatingMemory', method_ref=method_ref,
            baseline_result_ref=baseline_ref, run_record_refs=[run_ref],
            implementation_fix_round={method_ref: 2})
        await save_session_state(fake_bus.db, state, auto_commit=True)
        await engine.tick(SimpleNamespace(event_type='memory.consolidated',
            session_id=sid, emitter='MemoryAgent', payload=event.payload))
        assert (await load_session_state(fake_bus.db, sid)).current_state == 'PlanningExperiment'


@pytest.mark.parametrize('timeout', [False, True])
@pytest.mark.parametrize('closeout', [False, True])
async def test_efficient_component_choice_advances_verified_checkpoint(
        store, enabled, tmp_path, monkeypatch, timeout, closeout):
    monkeypatch.setenv('AUTORESEARCH_WEB_WORKDIR', str(tmp_path))
    sid = 'component-choice'
    if closeout:
        store.put(kind='human_idea_seed', session_id=sid, key='closeout-v1',
                  obj={'closeout': True}, emitter='Human')
    method = {'components': [_component('feature', 'indirect_gain'),
                             _component('downstream', depends_on=['feature'])]}
    method_ref = store.put(kind='method', session_id=sid, key='v1',
                           obj=method, emitter='MethodAgent')
    baseline = {'reproduced': True, 'metric_value': 1.0}
    baseline_ref = store.put(kind='baseline_result', session_id=sid, key='v1',
                             obj=baseline, emitter='RunnerAgent')
    plan = build_component_plan(method)
    ledger = ledger_for_workroot(session_workroot(sid), plan,
                                 execution_identity=artifact_digest(baseline))
    ledger.begin('feature')
    ledger.record_attempt('feature', execution_succeeded=True, contract_passed=True,
                          checkpoint='feature-1', quality_value=0.1)
    run = {'experiment_id': method_ref, 'metric_value': 1.0}
    run_ref = store.put(kind='run_record', session_id=sid,
                        key='snapshot-' + artifact_digest(run).split(':')[1],
                        obj=run, emitter='RunnerAgent')
    gen = {'run_record_snapshot_ref': run_ref, 'checkpoint': 'feature-1',
           'method_metric': 1.0,
           'bottleneck': 'No matching training examples; proposed diagnostic is untestable.',
           'component_execution': {
               'mode': 'component', 'component_id': 'feature', 'status': 'qualified',
               'quality_value': 0.1, 'quality_lower_is_better': True,
               'attempt_contract_result': {'passed': True}}}
    result_ref = store.put(kind='gen_experiment_result', session_id=sid,
                           key='snapshot-' + artifact_digest(gen).split(':')[1],
                           obj=gen, emitter='RunnerAgent')
    llm = LLM({'action': 'improve' if closeout else 'advance', 'selected_checkpoint': 'best',
               'reason': 'local property is ready for downstream',
               'next_experiment': 'one more diagnostic' if closeout else ''})
    if timeout:
        llm.response = TimeoutError('decision timed out')
    handler = MemoryAgentHandler(spec_store=store, llm_client=llm)
    verification = {'verdict': 'verified', 'run_record_ref': run_ref,
                    'candidate_snapshot_ref': result_ref}
    if timeout and not closeout:
        with pytest.raises(RuntimeError, match='decision timed out'):
            await handler._decide_efficient_component(sid, run_ref, {}, verification, baseline_ref)
        retained = ledger.entry('feature')
        assert retained.status == 'active'
        assert retained.checkpoint == 'feature-1'
        assert retained.qualified_attempts == 1
        assert not retained.search_decision
        assert ledger.entry('downstream').status == 'pending'
        return
    await handler._decide_efficient_component(sid, run_ref, {}, verification, baseline_ref)
    assert ledger.entry('feature').status == 'accepted'
    assert ledger.entry('feature').checkpoint == 'feature-1'
    assert ledger.entry('feature').attempts == 1
    assert ledger.entry('feature').search_decision['next_experiment'] == ''
    assert ledger.next_ready().component_id == 'downstream'
    assert len(llm.calls) == 1
    assert json.loads(llm.calls[0]['messages'][1]['content'])['latest'][
        'observed_bottleneck'] == gen['bottleneck']
    await handler._decide_efficient_component(sid, run_ref, {}, verification, baseline_ref)
    assert len(llm.calls) == 1


@pytest.mark.parametrize('closing_review_applied', [False, True])
async def test_qualified_closeout_after_reflection_advances_before_revisiting_history(
        store, enabled, tmp_path, monkeypatch, closing_review_applied):
    monkeypatch.setenv('AUTORESEARCH_WEB_WORKDIR', str(tmp_path))
    sid = 'qualified-closeout-after-reflection'
    method = {'components': [_component('feature', 'indirect_gain')]}
    method_ref = store.put(kind='method', session_id=sid, key='v1',
                           obj=method, emitter='MethodAgent')
    baseline = {'reproduced': True, 'metric_value': 1.0}
    baseline_ref = store.put(kind='baseline_result', session_id=sid, key='v1',
                             obj=baseline, emitter='RunnerAgent')
    ledger = ledger_for_workroot(session_workroot(sid), build_component_plan(method),
                                 execution_identity=artifact_digest(baseline))
    ledger.begin('feature')
    ledger.record_attempt('feature', execution_succeeded=True, contract_passed=True,
                          checkpoint='qualified-feature', quality_value=0.1,
                          contract_result={'passed': True})
    run = {'experiment_id': method_ref, 'succeeded': True, 'exit_code': 0,
           'metric_value': 1.0, 'environment': 'test'}
    run_ref = store.put(kind='run_record', session_id=sid,
        key='snapshot-' + artifact_digest(run).split(':')[1], obj=run, emitter='RunnerAgent')
    gen = {'verdict': 'valid_not_best', 'accepted': False, 'checkpoint': 'qualified-feature',
           'baseline_metric': 1.0, 'method_metric': 1.0, 'lower_is_better': True,
           'isolated': True, 'run_record_snapshot_ref': run_ref,
           'method_implementation_review': {'verdict': 'faithful'},
           'component_execution': {'mode': 'component', 'component_id': 'feature',
               'status': 'qualified', 'role': 'indirect_gain', 'qualified_attempts': 1,
               'attempt_contract_result': {'passed': True}}}
    result_ref = store.put(kind='gen_experiment_result', session_id=sid,
        key='snapshot-' + artifact_digest(gen).split(':')[1], obj=gen, emitter='RunnerAgent')
    verification_ref = store.put(kind='verification_report', session_id=sid, key='v1',
        obj={'verdict': 'verified', 'run_record_ref': run_ref,
             'candidate_snapshot_ref': result_ref}, emitter='EvaluatorAgent')
    feedback_ref = store.put(kind='method_improvement_review', session_id=sid, key='applied',
        obj={'status': 'completed', 'user_closeout': {'closeout': closing_review_applied},
             'decision': {'route': 'revise_method'}}, emitter='MemoryAgent')
    store.put(kind='human_idea_seed', session_id=sid, key='closeout-v1',
              obj={'closeout': closing_review_applied}, emitter='Human')
    llm = LLM({'action': 'improve', 'selected_checkpoint': 'best',
               'reason': 'another diagnostic', 'next_experiment': 'repeat the local check'})
    handler = MemoryAgentHandler(spec_store=store, llm_client=llm)
    reviews = []
    async def revisit(*args, **kwargs):
        reviews.append(args)
        return None
    monkeypatch.setattr(handler, '_review_retained_off_method', revisit)
    monkeypatch.setattr(handler, '_review_complete_result', revisit)
    bus = MemoryBus()
    await handler.handle_consolidation_requested(bus, _Envelope(
        'closing-qualified', sid, 'memory.consolidation_requested', 'ResearchOrchestrator',
        {'run_record_ref': run_ref, 'baseline_result_ref': baseline_ref,
         'verification_report_ref': verification_ref, 'applied_method_feedback_ref': feedback_ref}))
    event = bus.published[-1]
    assert event.event_type == 'memory.consolidated'
    assert event.payload['routing_verdict'] == 'need_experiment'
    assert len(reviews) == (0 if closing_review_applied else 2)
    assert ledger.entry('feature').status == ('accepted' if closing_review_applied else 'active')
    assert ledger.entry('feature').checkpoint == 'qualified-feature'
    assert ledger.entry('feature').attempts == 1


async def test_failed_followup_can_advance_prior_qualified_checkpoint(
        store, enabled, tmp_path, monkeypatch):
    monkeypatch.setenv('AUTORESEARCH_WEB_WORKDIR', str(tmp_path))
    sid = 'failed-followup-choice'
    method = {'components': [_component('feature', 'indirect_gain'),
                             _component('downstream', depends_on=['feature'])]}
    method_ref = store.put(kind='method', session_id=sid, key='v1',
                           obj=method, emitter='MethodAgent')
    baseline = {'reproduced': True, 'metric_value': 1.0}
    baseline_ref = store.put(kind='baseline_result', session_id=sid, key='v1',
                             obj=baseline, emitter='RunnerAgent')
    ledger = ledger_for_workroot(session_workroot(sid), build_component_plan(method),
                                 execution_identity=artifact_digest(baseline))
    ledger.begin('feature')
    ledger.record_attempt('feature', execution_succeeded=True, contract_passed=True,
                          checkpoint='qualified-feature', quality_value=0.1)
    ledger.record_attempt('feature', execution_succeeded=False,
                          error='replay did not produce a metric')
    run = {'experiment_id': method_ref}
    run_ref = store.put(kind='run_record', session_id=sid,
                        key='snapshot-' + artifact_digest(run).split(':')[1],
                        obj=run, emitter='RunnerAgent')
    gen = {'run_record_snapshot_ref': run_ref, 'method_metric': None,
           'error': 'replay did not produce a metric', 'component_execution': {
               'mode': 'component', 'component_id': 'feature',
               'status': 'execution_failed', 'attempt_contract_result': {'passed': False}}}
    result_ref = store.put(kind='gen_experiment_result', session_id=sid,
                           key='snapshot-' + artifact_digest(gen).split(':')[1],
                           obj=gen, emitter='RunnerAgent')
    llm = LLM({'action': 'advance', 'selected_checkpoint': 'best',
               'reason': 'the qualified result is intact', 'next_experiment': ''})
    handler = MemoryAgentHandler(spec_store=store, llm_client=llm)
    verification = {'verdict': 'verified', 'run_record_ref': run_ref,
                    'candidate_snapshot_ref': result_ref}
    await handler._decide_efficient_component(sid, run_ref, {}, verification, baseline_ref)
    assert ledger.entry('feature').status == 'accepted'
    assert ledger.next_ready().component_id == 'downstream'
    assert json.loads(llm.calls[0]['messages'][1]['content'])['latest']['status'] == 'execution_failed'


@pytest.mark.parametrize('action', ['advance', 'improve'])
async def test_bound_failed_quality_followup_reaches_existing_component_choice(
        store, enabled, tmp_path, monkeypatch, fake_bus, action):
    monkeypatch.setenv('AUTORESEARCH_WEB_WORKDIR', str(tmp_path))
    sid = 'quality-followup-choice'
    method = {'components': [_component('feature'),
                             _component('downstream', depends_on=['feature'])]}
    method_ref = store.put(kind='method', session_id=sid, key='v1',
                           obj=method, emitter='MethodAgent')
    baseline = {'reproduced': True, 'metric_value': 1.0}
    baseline_ref = store.put(kind='baseline_result', session_id=sid, key='v1',
                             obj=baseline, emitter='RunnerAgent')
    ledger = ledger_for_workroot(session_workroot(sid), build_component_plan(method),
                                 execution_identity=artifact_digest(baseline))
    contract = {'passed': True, 'measurement_evidence': {'valid': True}}
    ledger.begin('feature')
    ledger.record_attempt('feature', execution_succeeded=True, contract_passed=True,
                          checkpoint='qualified-feature', metric_value=0.8,
                          contract_result=contract)
    ledger.record_attempt('feature', execution_succeeded=False,
                          error='refinement failed correctness')
    run = {'experiment_id': method_ref, 'succeeded': True, 'exit_code': 0,
           'metric_value': 0.7, 'environment': 'test'}
    run_ref = store.put(kind='run_record', session_id=sid,
                        key='snapshot-' + artifact_digest(run).split(':')[1],
                        obj=run, emitter='RunnerAgent')
    gen = {
        'verdict': 'invalid', 'accepted': False, 'baseline_metric': 1.0,
        'method_metric': 0.7, 'lower_is_better': True, 'isolated': True,
        'run_record_snapshot_ref': run_ref, 'checkpoint': 'failed-variant',
        'component_attempt_id': 'attempt-refinement',
        'candidate_admissibility': {'verdict': 'fail', 'attempt_id': 'attempt-refinement'},
        'measurement_evidence': {
            'value': 0.7, 'valid': False, 'attempt_id': 'attempt-refinement',
            'stdout_sha256': 'a' * 64, 'gates': {'/quality_passed': False}},
        'component_execution': {
            'mode': 'component', 'component_id': 'feature', 'role': 'direct_gain',
            'operator': 'improve', 'status': 'execution_failed', 'qualified_attempts': 1,
            'checkpoint': 'qualified-feature', 'selected_contract_result': contract,
            'attempt_contract_result': {'passed': False}},
    }
    result_ref = store.put(kind='gen_experiment_result', session_id=sid,
                           key='snapshot-' + artifact_digest(gen).split(':')[1],
                           obj=gen, emitter='RunnerAgent')
    verification_ref = store.put(
        kind='verification_report', session_id=sid, key='v1',
        obj={'verdict': 'verified', 'run_record_ref': run_ref,
             'candidate_snapshot_ref': result_ref}, emitter='EvaluatorAgent')
    llm = LLM({'action': action, 'selected_checkpoint': 'best',
               'reason': 'The prior valid checkpoint is intact; assess the next useful step.',
               'next_experiment': 'Verify a bounded legal alternative.' if action == 'improve' else ''})
    handler = MemoryAgentHandler(spec_store=store, llm_client=llm)
    bus = MemoryBus()
    await handler.handle_consolidation_requested(
        bus, _Envelope('quality-followup', sid, 'memory.consolidation_requested',
                       'ResearchOrchestrator', {'run_record_ref': run_ref,
                           'verification_report_ref': verification_ref,
                           'baseline_result_ref': baseline_ref}))
    event = bus.published[-1]
    assert event.event_type == 'memory.consolidated'
    assert event.payload['diagnosis'] == 'component_progress'
    retained = ledger.entry('feature')
    assert retained.checkpoint == 'qualified-feature' and retained.metric_value == 0.8
    assert retained.attempts == 2 and retained.qualified_attempts == 1
    assert retained.status == ('accepted' if action == 'advance' else 'active')
    memory = store.get(event.payload['research_memory_ref'])
    assert memory['best_metric'] is None and not memory['verified_findings']
    assert len(llm.calls) == 1

    engine = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT, spec_store=store)
    await engine.start()
    state = SessionState(
        session_id=sid, user_id='u', mode='benchmark',
        current_state='ConsolidatingMemory', method_ref=method_ref,
        baseline_result_ref=baseline_ref, run_record_refs=[run_ref],
        implementation_fix_round={method_ref: 2})
    await save_session_state(fake_bus.db, state, auto_commit=True)
    await engine.tick(SimpleNamespace(
        event_type='memory.consolidated', session_id=sid, emitter='MemoryAgent',
        payload=event.payload))
    next_state = await load_session_state(fake_bus.db, sid)
    assert next_state.current_state == 'PlanningExperiment'
    assert method_ref not in next_state.implementation_fix_round
    assert not next_state.moss_interventions
    assert ledger.entry('feature').checkpoint == 'qualified-feature'


@pytest.mark.parametrize('action', ['advance', 'improve'])
@pytest.mark.parametrize('has_downstream', [False, True])
async def test_efficient_repeated_no_gain_reaches_component_choice_before_revision(
        store, enabled, tmp_path, monkeypatch, fake_bus, action, has_downstream):
    from autoresearch.agents.memory_agent.handler import _append_component_history

    monkeypatch.setenv('AUTORESEARCH_WEB_WORKDIR', str(tmp_path))
    sid = 'repeated-no-gain-choice'
    role = 'indirect_gain' if has_downstream else 'direct_gain'
    components = [_component('feature', role)]
    if has_downstream:
        components.append(_component('downstream', depends_on=['feature']))
    method = {'components': components}
    method_ref = store.put(kind='method', session_id=sid, key='v1',
                           obj=method, emitter='MethodAgent')
    baseline = {'reproduced': True, 'metric_value': 1.0}
    baseline_ref = store.put(kind='baseline_result', session_id=sid, key='v1',
                             obj=baseline, emitter='RunnerAgent')
    plan = build_component_plan(method)
    ledger = ledger_for_workroot(session_workroot(sid), plan,
                                 execution_identity=artifact_digest(baseline))
    contract = {'passed': True, 'measurement_evidence': {'valid': True},
                'acceptance_check_result': {'passed': True}}
    ledger.begin('feature')
    for index, value in enumerate((0.8, 0.9, 0.85), 1):
        ledger.record_attempt('feature', execution_succeeded=True, contract_passed=True,
                              checkpoint=f'feature-{index}', metric_value=value,
                              quality_value=value, contract_result=contract)
    run = {'experiment_id': method_ref, 'succeeded': True, 'exit_code': 0,
           'metric_value': 0.85, 'environment': 'test'}
    run_ref = store.put(kind='run_record', session_id=sid,
                        key='snapshot-' + artifact_digest(run).split(':')[1],
                        obj=run, emitter='RunnerAgent')
    gen = {
        'verdict': 'valid_not_best', 'accepted': True, 'baseline_metric': 1.0,
        'baseline_digest': artifact_digest(baseline), 'method_metric': 0.85,
        'lower_is_better': True, 'isolated': True, 'checkpoint': 'feature-3',
        'run_record_snapshot_ref': run_ref,
        'method_implementation_review': {
            'verdict': 'faithful', 'task_contract_review': {'verdict': 'compliant'}},
        'component_execution': {
            'mode': 'component', 'component_id': 'feature', 'role': role,
            'operator': 'improve', 'status': 'qualified', 'attempts': 3,
            'method_fingerprint': plan.method_fingerprint, 'plan_version': plan.version,
            'became_best': False, 'execution_succeeded': True,
            'diff_ref': 'diff-3', 'attempt_contract_result': contract},
    }
    result_ref = store.put(kind='gen_experiment_result', session_id=sid,
                           key='snapshot-' + artifact_digest(gen).split(':')[1],
                           obj=gen, emitter='RunnerAgent')
    verification = {'verdict': 'verified', 'run_record_ref': run_ref,
                    'candidate_snapshot_ref': result_ref}
    verification_ref = store.put(kind='verification_report', session_id=sid, key='v1',
                                 obj=verification, emitter='EvaluatorAgent')
    previous = {**gen, 'method_metric': 0.9, 'component_execution': {
        **gen['component_execution'], 'attempts': 2, 'diff_ref': 'diff-2'}}
    prior = {'component_history': _append_component_history(
        {}, previous, 'previous-run', verification)}
    prior_ref = store.put(kind='research_memory', session_id=sid, key='previous',
                          obj=prior, emitter='MemoryAgent')
    llm = LLM()
    def respond(**kwargs):
        llm.calls.append(kwargs)
        packet = json.loads(kwargs['messages'][1]['content'])
        response = decision() if 'evidence' in packet else {
            'action': action, 'selected_checkpoint': 'best',
            'reason': 'Use the qualified best; another run needs a concrete question.',
            'next_experiment': 'Check one allowed configuration.' if action == 'improve' else ''}
        return SimpleNamespace(content=json.dumps(response))
    llm.chat = respond
    handler = MemoryAgentHandler(spec_store=store, llm_client=llm)
    bus = MemoryBus()
    await handler.handle_consolidation_requested(
        bus, _Envelope('no-gain', sid, 'memory.consolidation_requested',
                       'ResearchOrchestrator', {'run_record_ref': run_ref,
                           'verification_report_ref': verification_ref,
                           'baseline_result_ref': baseline_ref,
                           'research_memory_ref': prior_ref}))

    event = bus.published[-1]
    assert event.event_type == 'memory.consolidated'
    assert event.payload['diagnosis'] == 'component_progress'
    assert not store.list_refs(sid, 'method_improvement_review')
    assert len(llm.calls) == 1
    retained = ledger.entry('feature')
    assert retained.checkpoint == 'feature-1' and retained.metric_value == 0.8
    assert retained.attempts == retained.qualified_attempts == 3
    assert retained.status == ('accepted' if action == 'advance' else 'active')
    if action == 'advance':
        assert ledger.all_accepted() is (not has_downstream)
        assert (ledger.next_ready().component_id if has_downstream else None) == (
            'downstream' if has_downstream else None)
    memory = store.get(event.payload['research_memory_ref'])
    assert memory['best_metric'] is None

    engine = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT, spec_store=store)
    await engine.start()
    await save_session_state(fake_bus.db, SessionState(
        session_id=sid, user_id='u', mode='benchmark', current_state='ConsolidatingMemory',
        method_ref=method_ref, baseline_result_ref=baseline_ref,
        run_record_refs=[run_ref]), auto_commit=True)
    await engine.tick(SimpleNamespace(event_type='memory.consolidated', session_id=sid,
                                      emitter='MemoryAgent', payload=event.payload))
    next_state = await load_session_state(fake_bus.db, sid)
    assert next_state.current_state == 'PlanningExperiment'
    assert next_state.method_ref == method_ref and not next_state.method_feedback_in_progress


@pytest.mark.parametrize('efficient', [False, True])
async def test_repeated_no_gain_review_yields_only_to_enabled_component_policy(
        store, enabled, monkeypatch, efficient):
    if not efficient:
        monkeypatch.delenv('AUTORESEARCH_EXECUTION_STRATEGY')
    history = [{
        'mode': 'component', 'component_id': 'feature', 'plan_version': 1,
        'method_fingerprint': 'same-method', 'operator': 'improve',
        'verification_verdict': 'verified', 'execution_succeeded': True,
        'contract_passed': True, 'became_best': False, 'metric': 0.9,
        'diff_ref': f'diff-{index}',
    } for index in (2, 3)]
    _, run_ref, _ = seed(store, mode='component', feedback={
        'method_implementation_review': {
            'verdict': 'faithful', 'task_contract_review': {'verdict': 'compliant'}},
        'component_execution': {
            'mode': 'component', 'became_best': False,
            'attempt_contract_result': {'passed': True}}})
    llm = LLM({'route': 'report', 'reason': 'No supported Method change.'})
    handler = MemoryAgentHandler(spec_store=store, llm_client=llm)
    await handler._review_complete_result(
        'feedback', run_ref, {'component_history': history},
        {'verdict': 'verified'}, 'component_progress')
    reviews = store.list_refs('feedback', 'method_improvement_review')
    assert len(reviews) == len(llm.calls) == (0 if efficient else 1)
    if reviews:
        assert store.get(reviews[0])['trigger'] == 'repeated_verified_no_gain'


async def test_indirect_attempt_cap_still_selects_verified_downstream_checkpoint(
        store, enabled, tmp_path, monkeypatch):
    monkeypatch.setenv('AUTORESEARCH_WEB_WORKDIR', str(tmp_path))
    monkeypatch.setenv('AUTORESEARCH_EXECUTION_STRATEGY', 'efficient')
    sid = 'component-cap-choice'
    method = {'components': [_component('feature', 'indirect_gain'),
                             _component('downstream', depends_on=['feature'])]}
    method_ref = store.put(kind='method', session_id=sid, key='v1',
                           obj=method, emitter='MethodAgent')
    baseline = {'reproduced': True, 'metric_value': 5.0}
    baseline_ref = store.put(kind='baseline_result', session_id=sid, key='v1',
                             obj=baseline, emitter='RunnerAgent')
    ledger = ledger_for_workroot(
        session_workroot(sid), build_component_plan(method),
        execution_identity=artifact_digest(baseline))
    ledger.begin('feature')
    for index, quality in enumerate((0.1, 0.2, 0.3, 0.4, 0.5, 0.6), 1):
        ledger.record_attempt(
            'feature', execution_succeeded=True, contract_passed=True,
            checkpoint=f'feature-{index}', quality_value=quality,
            diff=f'diff-{index}')
    assert ledger.entry('feature').status == 'accepted'
    assert ledger.entry('feature').checkpoint == 'feature-1'
    run = {'experiment_id': method_ref, 'metric_value': 3.5}
    run_ref = store.put(kind='run_record', session_id=sid,
                        key='snapshot-' + artifact_digest(run).split(':')[1],
                        obj=run, emitter='RunnerAgent')
    gen = {'run_record_snapshot_ref': run_ref, 'checkpoint': 'feature-6',
           'method_metric': 3.5, 'component_execution': {
               'mode': 'component', 'component_id': 'feature', 'role': 'indirect_gain',
               'status': 'accepted', 'attempt_quality_value': 0.6,
               'attempt_contract_result': {'passed': True}}}
    result_ref = store.put(kind='gen_experiment_result', session_id=sid,
                           key='snapshot-' + artifact_digest(gen).split(':')[1],
                           obj=gen, emitter='RunnerAgent')
    llm = LLM({'action': 'improve', 'selected_checkpoint': 'feature-2',
               'reason': 'feature-2 matches the downstream interface',
               'next_experiment': 'try another variant'})
    handler = MemoryAgentHandler(spec_store=store, llm_client=llm)
    verification = {'verdict': 'verified', 'run_record_ref': run_ref,
                    'candidate_snapshot_ref': result_ref}
    await handler._decide_efficient_component(sid, run_ref, {}, verification, baseline_ref)
    chosen = ledger.entry('feature')
    assert (chosen.status, chosen.checkpoint, chosen.diff) == (
        'accepted', 'feature-2', 'diff-2')
    assert chosen.search_decision['action'] == 'advance'
    assert chosen.search_decision['next_experiment'] == ''
    ledger.begin('downstream')
    assert ledger.entry('downstream').parent_checkpoints == ['feature-2']
    await handler._decide_efficient_component(sid, run_ref, {}, verification, baseline_ref)
    assert len(llm.calls) == 1


def test_early_review_requires_a_better_comparable_off_method_observation():
    incumbent = {
        'accepted': True, 'method_metric': 2.0, 'baseline_metric': 3.0,
        'metric_name': 'loss', 'metric_unit': 'unit', 'lower_is_better': True,
        'measurement_evidence': {'protocol_fingerprint': 'same'},
    }
    candidate = {
        **incumbent, 'accepted': False, 'method_metric': 1.5,
        'checkpoint': '/saved/off-method',
        'measurement_evidence': {'protocol_fingerprint': 'same', 'minimum_improvement': 0.01},
        'method_implementation_review': {
            'verdict': 'drifted', 'task_contract_review': {'verdict': 'compliant'}},
        'repository_comparison_review': {'verdict': 'pass'},
        'component_execution': {
            'mode': 'integration', 'component_id': '__integration__',
            'status': 'valid_not_best', 'all_components_accepted': True,
            'execution_succeeded': True},
        'off_method_observation': {
            'valid': True, 'improved': True, 'metric': 1.5,
            'admissibility': {'verdict': 'pass'}},
    }
    memory = {'best_metric': 2.0, 'best_result_ref': 'bound-incumbent'}
    diagnosis = 'method_description_reconciliation'
    assert _method_review_trigger(candidate, memory, diagnosis, incumbent) == 'legal_off_method_gain'
    assert _method_review_trigger({**candidate, 'method_metric': 2.1,
                                   'off_method_observation': {
                                       **candidate['off_method_observation'], 'metric': 2.1}},
                                  memory, diagnosis, incumbent) is None
    assert _method_review_trigger({**candidate, 'measurement_evidence': {
        'protocol_fingerprint': 'different'}}, memory, diagnosis, incumbent) is None
    assert _method_review_trigger({**candidate, 'method_metric': 1.995,
                                   'off_method_observation': {
                                       **candidate['off_method_observation'], 'metric': 1.995}},
                                  memory, diagnosis, incumbent) is None
    assert _method_review_trigger(candidate, memory, diagnosis, None) is None

    partial = {
        **candidate, 'accepted': True, 'off_method_observation': None,
        'method_implementation_review': {
            'verdict': 'partial', 'task_contract_review': {'verdict': 'compliant'}},
        'component_execution': {**candidate['component_execution'],
                                'execution_succeeded': True},
        'candidate_admissibility': {'verdict': 'pass', 'attempt_id': 'attempt-1'},
        'measurement_evidence': {
            **candidate['measurement_evidence'], 'valid': True, 'value': 1.5,
            'attempt_id': 'attempt-1'},
    }
    assert _method_review_trigger(partial, memory, 'valid_improvement', incumbent) == (
        'legal_off_method_gain')
    assert _method_review_trigger({**partial, 'method_metric': 2.1,
                                   'measurement_evidence': {
                                       **partial['measurement_evidence'], 'value': 2.1}},
                                  memory, 'inconclusive', incumbent) is None
    assert _method_review_trigger({**partial, 'candidate_admissibility': {
        'verdict': 'fail', 'attempt_id': 'attempt-1'}}, memory,
        'valid_improvement', incumbent) is None
    assert _method_review_trigger({**partial, 'measurement_evidence': {
        **partial['measurement_evidence'], 'protocol_fingerprint': 'different'}},
        memory, 'valid_improvement', incumbent) is None


def test_early_review_distinguishes_cap_shortfall_regression_and_repeated_no_gain():
    component = {
        'accepted': True, 'checkpoint': '/saved/candidate', 'method_metric': 0.9,
        'method_implementation_review': {
            'verdict': 'faithful', 'task_contract_review': {'verdict': 'compliant'}},
        'component_execution': {
            'mode': 'component', 'role': 'direct_gain',
            'attempt_contract_result': {'passed': True,
                                        'acceptance_check_result': {'passed': False}}},
    }
    assert _method_review_trigger(component, {}, 'component_accepted', None) == 'cap_expectation_shortfall'
    assert _method_review_trigger({**component, 'component_execution': {
        **component['component_execution'], 'role': 'indirect_gain'}},
        {}, 'component_accepted', None) is None
    regression = {
        **component, 'accepted': False, 'baseline_metric': 1.0, 'method_metric': 1.2,
        'lower_is_better': True, 'verdict': 'valid_not_best',
        'candidate_admissibility': {'verdict': 'pass'},
        'component_execution': {
            'mode': 'integration', 'all_components_accepted': True,
            'execution_succeeded': True},
    }
    assert _method_review_trigger(regression, {}, 'component_combination_retry', None) == 'complete_metric_regression'
    assert _method_review_trigger({**regression, 'method_metric': 0.9}, {},
                                  'component_combination_retry', None) is None
    history = [{
        'mode': 'component', 'component_id': 'c1', 'plan_version': 1,
        'method_fingerprint': 'same', 'operator': 'improve',
        'verification_verdict': 'verified', 'execution_succeeded': True,
        'contract_passed': True, 'became_best': False, 'metric': 0.9,
        'diff_ref': f'diff-{index}',
    } for index in (1, 2)]
    repeat = {**component, 'component_execution': {
        **component['component_execution'], 'became_best': False,
        'attempt_contract_result': {'passed': True,
                                    'acceptance_check_result': {'passed': True}}}}
    assert _method_review_trigger(repeat, {'component_history': history},
                                  'component_progress', None) == 'repeated_verified_no_gain'
    history[1]['diff_ref'] = history[0]['diff_ref']
    assert _method_review_trigger(repeat, {'component_history': history},
                                  'component_progress', None) is None


def test_mir_violation_is_negative_method_evidence_without_a_score():
    candidate = {
        'accepted': False, 'method_metric': None,
        'component_execution': {'mode': 'component'},
        'method_implementation_review': {
            'verdict': 'faithful', 'task_contract_review': {'verdict': 'violation',
                                                           'violations': ['illegal interface']}},
    }
    assert _method_review_trigger(candidate, {}, 'candidate_contract_violation', None) == (
        'mir_task_contract_violation')
    assert _method_review_trigger(candidate, {}, 'component_contract_retry', None) == (
        'mir_task_contract_violation')
    candidate['method_implementation_review']['task_contract_review']['verdict'] = (
        'insufficient_evidence')
    assert _method_review_trigger(candidate, {}, 'candidate_contract_violation', None) is None


def test_old_off_method_import_violation_routes_as_contract_failure(store, fake_bus):
    sid = 'old-off-import-rule'
    candidate = {
        'method_metric': 1153.0,
        'component_execution': {'mode': 'integration'},
        'method_implementation_review': {
            'verdict': 'drifted', 'task_contract_review': {'verdict': 'compliant'}},
        'repository_contract': {'sources': [{'path': 'instruction.md', 'content':
            'Allowed imports: `numpy`, `torch`. No other libraries.'}]},
        'diff': 'diff --git a/solve.py b/solve.py\n+++ b/solve.py\n@@\n+import sys\n',
    }
    assert _method_review_trigger(candidate, {}, 'method_description_reconciliation', None) == (
        'mir_task_contract_violation')
    result_ref = store.put(kind='gen_experiment_result', session_id=sid, key='snapshot',
                           obj=candidate, emitter='RunnerAgent')
    review_ref = store.put(kind='method_improvement_review', session_id=sid, key='slot-1',
                           obj={'status': 'completed', 'trigger': 'legal_off_method_gain',
                                'result_ref': result_ref,
                                'off_method_evidence': {'candidate': {'metric': 1153.0,
                                                                       'diff': candidate['diff']}},
                                'decision': {'route': 'revise_method',
                                             'primary_hypothesis': 'Treat this as legal.'}},
                           emitter='MemoryAgent')
    state = SessionState(session_id=sid, user_id='u', current_state='RevisingMethod',
                         method_feedback_ref=review_ref, method_feedback_in_progress=True)
    engine = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT, spec_store=store)

    feedback = engine._method_revision_feedback(state)

    assert 'mir_task_contract_violation' in feedback
    assert 'import sys' in feedback
    assert 'score_admissible": false' in feedback
    assert 'Treat this as legal.' not in feedback


async def test_bound_mir_violation_reviews_contract_and_additive_revision(
        store, enabled, fake_bus):
    sid = 'mir-violation-review'
    method_ref = store.put(kind='method', session_id=sid, key='v1',
                           obj={'name': 'A', 'components': ['use legal interface']},
                           emitter='MethodAgent')
    baseline = {
        'reproduced': True, 'metric_name': 'loss', 'metric_value': 3.0,
        'lower_is_better': True, 'fixed_config': 'same data and seed',
        'benchmark_contract': {'metric': 'loss'},
        'repository_contract': {'sources': [{
            'path': 'instruction.md', 'sha256': 'abc',
            'content': 'Only edit solve.py; callable interface is solve(x).'}],
            'execution_contract': {'evaluation': {'command': 'run official verifier'}}},
    }
    baseline_ref = store.put(kind='baseline_result', session_id=sid, key='v1',
                             obj=baseline, emitter='RunnerAgent')
    store.put(kind='spec_interpretation', session_id=sid, key='v1',
              obj={'evaluation_semantics': {'primary_metric': 'loss',
                                            'comparison_mode': 'same_setting'}},
              emitter='ProblemModelingAgent')
    run = {'succeeded': False, 'exit_code': 1, 'metric_value': None,
           'experiment_id': method_ref, 'stderr_excerpt': 'unexpected interface output'}
    run_ref = store.put(kind='run_record', session_id=sid,
                        key='snapshot-' + artifact_digest(run).split(':')[1],
                        obj=run, emitter='RunnerAgent')
    candidate = {
        'accepted': False, 'verdict': 'invalid', 'method_metric': None,
        'baseline_metric': 3.0, 'run_record_snapshot_ref': run_ref,
        'diff': 'def solve(x): return illegal_interface(x)',
        'component_execution': {'mode': 'component', 'component_id': 'A',
                                'status': 'contract_failed'},
        'method_implementation_review': {
            'verdict': 'faithful', 'method_ref': method_ref,
            'task_contract_review': {'verdict': 'violation',
                                     'violations': ['illegal interface']}},
        'diagnosis': 'candidate_contract_violation',
        'diagnosis_evidence': ['illegal interface'],
    }
    candidate_ref = store.put(kind='gen_experiment_result', session_id=sid,
                              key='snapshot-' + artifact_digest(candidate).split(':')[1],
                              obj=candidate, emitter='RunnerAgent')
    verification = {'verdict': 'verified', 'run_record_ref': run_ref,
                    'run_record_digest': artifact_digest(run),
                    'candidate_snapshot_ref': candidate_ref, 'anti_cheat_flags': [],
                    'checks': ['failure_confirmed_from_exit_code_and_output']}
    llm = LLM(decision(candidate_ref))
    handler = MemoryAgentHandler(spec_store=store, llm_client=llm)
    memory = {'best_metric': None, 'best_result_ref': None}
    assert await handler._review_complete_result(
        sid, run_ref, memory, {**verification, 'candidate_snapshot_ref': 'wrong'},
        'candidate_contract_violation', baseline_ref) is None
    assert not llm.calls

    review_ref = await handler._review_complete_result(
        sid, run_ref, memory, verification, 'candidate_contract_violation', baseline_ref)
    review = store.get(review_ref)
    assert review['trigger'] == 'mir_task_contract_violation'
    assert review['incumbent_ref'] == baseline_ref
    assert review['mir_violation_evidence']['score_admissible'] is False
    assert review['mir_violation_evidence']['diff'] == candidate['diff']
    assert review['mir_violation_evidence']['stderr_excerpt'] == run['stderr_excerpt']
    assert review['task_requirements']['repository_sources'][0]['content'] == (
        baseline['repository_contract']['sources'][0]['content'])
    prompt = json.loads(llm.calls[0]['messages'][1]['content'])
    assert prompt['evidence']['task_requirements']['evaluation_semantics'][
        :1] == '{'
    assert prompt['evidence']['mir_contract_violation']['review'][
        'task_contract_review']['verdict'] == 'violation'
    assert 'When extending an evidence-supported Method A, propose a testable extension B' in (
        llm.calls[0]['messages'][0]['content'])
    assert memory['best_result_ref'] is None

    evaluation_ref = store.put(kind='evaluation_report', session_id=sid, key='v1',
                               obj={'verdict': 'invalid', 'anti_cheat_flags': []},
                               emitter='EvaluatorAgent')
    verification_ref = store.put(kind='verification_report', session_id=sid, key='v1',
                                 obj=verification, emitter='EvaluatorAgent')
    bus = MemoryBus()
    await handler.handle_consolidation_requested(bus, _Envelope(
        'event', sid, 'memory.consolidation_requested', 'ResearchOrchestrator',
        {'run_record_ref': run_ref, 'evaluation_report_ref': evaluation_ref,
         'verification_report_ref': verification_ref,
         'baseline_result_ref': baseline_ref}))
    assert bus.published[-1].payload['diagnosis'] == 'method_improvement_proposed'
    memory_ref = bus.published[-1].payload['research_memory_ref']
    saved_memory = store.get(memory_ref)
    assert saved_memory['best_result_ref'] is None
    assert saved_memory['best_metric'] is None
    engine = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT, spec_store=store)
    await engine.start()
    state = SessionState(session_id=sid, user_id='u', current_state='ConsolidatingMemory',
                         method_ref=method_ref, baseline_result_ref=baseline_ref,
                         research_memory_ref=memory_ref, run_record_refs=[run_ref])
    await save_session_state(fake_bus.db, state, auto_commit=True)
    await engine.tick(SimpleNamespace(
        event_type='memory.consolidated', session_id=sid, emitter='MemoryAgent',
        payload={'diagnosis': 'method_improvement_proposed',
                 'diagnosis_evidence': [review_ref], 'routing_verdict': 'need_experiment',
                 'research_memory_ref': memory_ref}))
    assert (await load_session_state(fake_bus.db, sid)).current_state == 'RevisingMethod'


@pytest.mark.parametrize('prior_closeout', [False, True, 'same_method'])
async def test_better_off_method_result_uses_bounded_review_without_adoption(
        store, enabled, prior_closeout):
    sid = 'off-method-review'
    method_ref = store.put(kind='method', session_id=sid, key='v1',
                           obj={'name': 'A', 'components': []}, emitter='MethodAgent')
    incumbent = {
        'accepted': True, 'method_metric': 2.0, 'baseline_metric': 3.0,
        'metric_name': 'loss', 'metric_unit': 'unit', 'lower_is_better': True,
        'checkpoint': '/saved/incumbent',
        'measurement_evidence': {'protocol_fingerprint': 'same'},
    }
    incumbent_ref = store.put(
        kind='gen_experiment_result', session_id=sid,
        key='snapshot-' + artifact_digest(incumbent).split(':')[1],
        obj=incumbent, emitter='RunnerAgent')
    run = {'succeeded': True, 'exit_code': 0, 'metric_value': 1.5,
           'experiment_id': method_ref, 'environment': 'test'}
    run_ref = store.put(kind='run_record', session_id=sid,
                        key='snapshot-' + artifact_digest(run).split(':')[1],
                        obj=run, emitter='RunnerAgent')
    candidate = {
        **incumbent, 'accepted': False, 'method_metric': 1.5,
        'checkpoint': '/saved/off-method', 'run_record_snapshot_ref': run_ref,
        'measurement_evidence': {'protocol_fingerprint': 'same', 'minimum_improvement': 0.01},
        'method_implementation_review': {
            'verdict': 'drifted', 'task_contract_review': {'verdict': 'compliant'}},
        'repository_comparison_review': {'verdict': 'pass'},
        'component_execution': {
            'mode': 'integration', 'component_id': '__integration__',
            'status': 'valid_not_best', 'all_components_accepted': True,
            'execution_succeeded': True},
        'off_method_observation': {
            'valid': True, 'improved': True, 'metric': 1.5,
            'admissibility': {'verdict': 'pass'}},
    }
    candidate_ref = store.put(
        kind='gen_experiment_result', session_id=sid,
        key='snapshot-' + artifact_digest(candidate).split(':')[1],
        obj=candidate, emitter='RunnerAgent')
    if prior_closeout:
        prior_method_ref = store.put(kind='method', session_id=sid, key='prior',
            obj={'name': 'earlier probe', 'components': []}, emitter='MethodAgent')
        if prior_closeout == 'same_method':
            prior_method_ref = method_ref
        store.put(kind='method_improvement_review', session_id=sid, key='prior-closeout',
            obj={'status': 'completed', 'trigger': 'qualified_component_closeout',
                 'method_ref': prior_method_ref, 'decision': {'route': 'revise_method'}},
            emitter='MemoryAgent')
    llm = LLM(decision(candidate_ref))
    handler = MemoryAgentHandler(spec_store=store, llm_client=llm)
    memory = {'best_metric': 2.0, 'best_result_ref': incumbent_ref}
    review_ref = await handler._review_complete_result(
        sid, run_ref, memory, {'verdict': 'verified'},
        'method_description_reconciliation')
    assert review_ref is not None
    assert store.get(review_ref)['trigger'] == 'legal_off_method_gain'
    assert memory['best_result_ref'] == incumbent_ref
    assert memory['best_metric'] == 2.0
    assert len(llm.calls) == (2 if prior_closeout else 1)
    if prior_closeout:
        review = store.get(review_ref)
        assert review['closeout_requested'] is True
        assert review['decision']['add'] == []
        assert review['decision']['reuse'] == ['/saved/off-method']
    prior_ref = store.put(kind='research_memory', session_id=sid, key='prior',
                          obj={'best_result_ref': incumbent_ref, 'best_metric': 2.0},
                          emitter='MemoryAgent')
    verification_ref = store.put(kind='verification_report', session_id=sid, key='verified',
                                 obj={'verdict': 'verified'}, emitter='EvaluatorAgent')
    bus = MemoryBus()
    await handler.handle_consolidation_requested(bus, _Envelope(
        'event', sid, 'memory.consolidation_requested', 'ResearchOrchestrator',
        {'run_record_ref': run_ref, 'research_memory_ref': prior_ref,
         'verification_report_ref': verification_ref}))
    assert bus.published[-1].payload['diagnosis'] == 'method_improvement_proposed'
    saved_memory = store.get(bus.published[-1].payload['research_memory_ref'])
    assert saved_memory['best_result_ref'] == incumbent_ref
    assert saved_memory['best_metric'] == 2.0
    assert len(llm.calls) == (2 if prior_closeout else 1)


async def test_verified_off_method_gain_before_first_accepted_result_revises_method(
        store, enabled, fake_bus):
    sid = 'baseline-off-method'
    method_ref = store.put(kind='method', session_id=sid, key='v1',
                           obj={'name': 'A', 'components': []}, emitter='MethodAgent')
    baseline = {
        'reproduced': True, 'metric_value': 3.0, 'metric_name': 'loss',
        'metric_unit': 'unit', 'lower_is_better': True, 'checkpoint': '/saved/baseline',
        'measurement_evidence': {'valid': True, 'protocol_fingerprint': 'same'},
        'log_excerpt': 'official baseline log',
    }
    baseline_ref = store.put(kind='baseline_result', session_id=sid, key='v1',
                             obj=baseline, emitter='RunnerAgent')
    run = {'succeeded': True, 'exit_code': 0, 'metric_value': 1.5,
           'experiment_id': method_ref, 'stdout_excerpt': 'official candidate log'}
    run_ref = store.put(kind='run_record', session_id=sid,
                        key='snapshot-' + artifact_digest(run).split(':')[1],
                        obj=run, emitter='RunnerAgent')
    candidate = {
        'accepted': False, 'verdict': 'valid_not_best', 'method_metric': 1.5,
        'baseline_metric': 3.0, 'metric_name': 'loss', 'metric_unit': 'unit',
        'lower_is_better': True, 'checkpoint': '/saved/off-method',
        'run_record_snapshot_ref': run_ref, 'diff': 'measured alternative diff',
        'measurement_evidence': {'protocol_fingerprint': 'same',
                                 'minimum_improvement': 0.01},
        'method_implementation_review': {
            'verdict': 'drifted', 'task_contract_review': {'verdict': 'compliant'}},
        'repository_comparison_review': {'verdict': 'pass'},
        'off_method_observation': {
            'valid': True, 'improved': True, 'metric': 1.5,
            'admissibility': {'verdict': 'pass'}},
    }
    candidate_ref = store.put(kind='gen_experiment_result', session_id=sid,
                              key='snapshot-' + artifact_digest(candidate).split(':')[1],
                              obj=candidate, emitter='RunnerAgent')
    evaluation = {'verdict': 'valid_not_best', 'run_record_ref': run_ref,
                  'run_record_digest': artifact_digest(run), 'anti_cheat_flags': []}
    evaluation_ref = store.put(kind='evaluation_report', session_id=sid,
                               key='snapshot-' + artifact_digest(evaluation).split(':')[1],
                               obj=evaluation, emitter='EvaluatorAgent')
    verification = {'verdict': 'verified', 'run_record_ref': run_ref,
                    'candidate_snapshot_ref': candidate_ref,
                    'evaluation_report_ref': evaluation_ref, 'anti_cheat_flags': []}
    llm = LLM(decision(candidate_ref))
    handler = MemoryAgentHandler(spec_store=store, llm_client=llm)
    memory = {'best_metric': None, 'best_result_ref': None}
    assert _method_review_trigger(
        candidate, memory, 'candidate_contract_violation', None,
        {**baseline, 'measurement_evidence': {
            'valid': True, 'protocol_fingerprint': 'different'}}) is None
    assert _method_review_trigger(
        {**candidate, 'method_metric': 3.1,
         'off_method_observation': {**candidate['off_method_observation'], 'metric': 3.1}},
        memory, 'candidate_contract_violation', None, baseline) is None
    assert await handler._review_complete_result(
        sid, run_ref, memory,
        {**verification, 'candidate_snapshot_ref': 'unbound-result'},
        'candidate_contract_violation', baseline_ref) is None
    assert not llm.calls
    review_ref = await handler._review_complete_result(
        sid, run_ref, memory, verification, 'candidate_contract_violation', baseline_ref)
    assert review_ref is not None
    review = store.get(review_ref)
    assert review['trigger'] == 'legal_off_method_gain'
    assert review['incumbent_ref'] == baseline_ref
    assert review['incumbent_metric'] == 3.0
    assert review['off_method_evidence']['candidate']['diff'] == 'measured alternative diff'
    assert review['off_method_evidence']['candidate']['stdout_excerpt'] == 'official candidate log'
    assert review['off_method_evidence']['incumbent']['stdout_excerpt'] == 'official baseline log'
    assert memory['best_result_ref'] is None
    assert memory['best_metric'] is None

    memory_ref = store.put(kind='research_memory', session_id=sid, key='v1',
                           obj=memory, emitter='MemoryAgent')
    engine = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT, spec_store=store)
    await engine.start()
    state = SessionState(session_id=sid, user_id='u', current_state='ConsolidatingMemory',
                         method_ref=method_ref, baseline_result_ref=baseline_ref,
                         research_memory_ref=memory_ref, run_record_refs=[run_ref])
    await save_session_state(fake_bus.db, state, auto_commit=True)
    await engine.tick(SimpleNamespace(
        event_type='memory.consolidated', session_id=sid, emitter='MemoryAgent',
        payload={'diagnosis': 'method_improvement_proposed',
                 'diagnosis_evidence': [review_ref], 'routing_verdict': 'need_experiment',
                 'research_memory_ref': memory_ref}))
    routed = await load_session_state(fake_bus.db, sid)
    assert routed.current_state == 'RevisingMethod'


@pytest.mark.parametrize('bound_verification', [True, False])
async def test_earlier_off_method_is_reviewed_after_first_accepted_incumbent(
        store, enabled, fake_bus, bound_verification):
    sid = 'retained-off-method'
    earlier_method_ref = store.put(kind='method', session_id=sid, key='v1',
                                   obj={'name': 'A', 'components': []}, emitter='MethodAgent')
    method_ref = store.put(kind='method', session_id=sid, key='v2',
                           obj={'name': 'B', 'components': []}, emitter='MethodAgent')

    def result(metric, *, accepted):
        run = {'succeeded': True, 'exit_code': 0, 'metric_value': metric,
               'experiment_id': method_ref if accepted else earlier_method_ref,
               'environment': 'test',
               'stdout_excerpt': 'accepted official log' if accepted else 'off official log'}
        run_ref = store.put(kind='run_record', session_id=sid,
                            key='snapshot-' + artifact_digest(run).split(':')[1],
                            obj=run, emitter='RunnerAgent')
        gen = {
            'accepted': accepted, 'verdict': 'new_best' if accepted else 'valid_not_best',
            'method_metric': metric, 'baseline_metric': 3.0, 'metric_name': 'loss',
            'metric_unit': 'unit', 'lower_is_better': True,
            'checkpoint': '/saved/accepted' if accepted else '/saved/off-method',
            'diff': 'accepted implementation diff' if accepted else 'off implementation diff',
            'parent_relative_diff': 'off parent diff' if not accepted else '',
            'run_record_snapshot_ref': run_ref,
            'measurement_evidence': {'protocol_fingerprint': 'same',
                                     'minimum_improvement': 0.01},
            'method_implementation_review': {
                'verdict': 'faithful' if accepted else 'drifted',
                'task_contract_review': {'verdict': 'compliant'}},
            'repository_comparison_review': {'verdict': 'pass'},
            'component_execution': {
                'mode': 'integration', 'component_id': '__integration__',
                'status': 'accepted' if accepted else 'drifted',
                'all_components_accepted': True, 'execution_succeeded': True},
        }
        if not accepted:
            gen['off_method_observation'] = {
                'valid': True, 'improved': True, 'metric': metric,
                'admissibility': {'verdict': 'pass'}}
        result_ref = store.put(kind='gen_experiment_result', session_id=sid,
                               key='snapshot-' + artifact_digest(gen).split(':')[1],
                               obj=gen, emitter='RunnerAgent')
        evaluation = {'verdict': 'pass', 'run_record_ref': run_ref,
                      'run_record_digest': artifact_digest(run)}
        evaluation_ref = store.put(kind='evaluation_report', session_id=sid,
                                   key='snapshot-' + artifact_digest(evaluation).split(':')[1],
                                   obj=evaluation, emitter='EvaluatorAgent')
        verification = {'verdict': 'verified', 'run_record_ref': run_ref,
                        'run_record_digest': artifact_digest(run),
                        'candidate_snapshot_ref': (
                            result_ref if accepted or bound_verification else 'unbound-result'),
                        'evaluation_report_ref': evaluation_ref}
        verification_ref = store.put(kind='verification_report', session_id=sid,
                                     key='snapshot-' + artifact_digest(verification).split(':')[1],
                                     obj=verification, emitter='EvaluatorAgent')
        return run_ref, result_ref, evaluation_ref, verification_ref

    earlier_run, earlier_ref, earlier_evaluation, earlier_verification = result(1.5, accepted=False)
    llm = LLM(decision(earlier_ref))
    handler = MemoryAgentHandler(spec_store=store, llm_client=llm)
    bus = MemoryBus()
    await handler.handle_consolidation_requested(bus, _Envelope(
        'early', sid, 'memory.consolidation_requested', 'ResearchOrchestrator',
        {'run_record_ref': earlier_run, 'evaluation_report_ref': earlier_evaluation,
         'verification_report_ref': earlier_verification}))
    first = store.get(bus.published[-1].payload['research_memory_ref'])
    assert first['reconciliation_candidates'][0]['result_ref'] == earlier_ref
    assert first['best_result_ref'] is None
    assert not llm.calls

    accepted_run, accepted_ref, accepted_evaluation, accepted_verification = result(2.0, accepted=True)
    await handler.handle_consolidation_requested(bus, _Envelope(
        'accepted', sid, 'memory.consolidation_requested', 'ResearchOrchestrator',
        {'run_record_ref': accepted_run, 'evaluation_report_ref': accepted_evaluation,
         'verification_report_ref': accepted_verification,
         'research_memory_ref': bus.published[-1].payload['research_memory_ref']}))
    if not bound_verification:
        assert bus.published[-1].event_type == 'agent.failed'
        assert 'unsupplied evidence' in bus.published[-1].payload['underlying_reason']
        assert store.get(accepted_ref)['accepted'] is True
        assert store.get(accepted_ref)['method_metric'] == 2.0
        review = store.get(store.list_refs(sid, 'method_improvement_review')[0])
        assert review['trigger'] == 'complete_result'
        assert review['status'] == 'failed'
        assert len(llm.calls) == 2
        return
    memory = store.get(bus.published[-1].payload['research_memory_ref'])
    assert memory['best_result_ref'] == accepted_ref
    assert memory['best_metric'] == 2.0
    review = store.get(memory['method_feedback_refs'][0])
    if bound_verification:
        assert bus.published[-1].payload['diagnosis'] == 'method_improvement_proposed'
        assert review['result_ref'] == earlier_ref
        assert review['trigger'] == 'legal_off_method_gain'
        assert review['method_ref'] == method_ref
        assert review['source_method_ref'] == earlier_method_ref
        comparison = review['off_method_evidence']
        assert comparison['candidate']['diff'] == 'off implementation diff'
        assert comparison['candidate']['parent_relative_diff'] == 'off parent diff'
        assert comparison['candidate']['stdout_excerpt'] == 'off official log'
        assert comparison['incumbent']['diff'] == 'accepted implementation diff'
        assert comparison['incumbent']['stdout_excerpt'] == 'accepted official log'
        assert comparison['source_method']['ref'] == earlier_method_ref
        review_input = json.loads(llm.calls[0]['messages'][1]['content'])
        assert review_input['method']['name'] == 'B'
        sent = review_input['evidence']
        assert sent['off_method_comparison'] == comparison
        engine = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT, spec_store=store)
        feedback = engine._method_revision_feedback(SessionState(
            session_id=sid, user_id='u', method_feedback_in_progress=True,
            method_feedback_ref=memory['method_feedback_refs'][0]))
        assert 'off implementation diff' in feedback
        assert 'off official log' in feedback
        await engine.start()
        state = SessionState(session_id=sid, user_id='u',
                             current_state='ConsolidatingMemory', method_ref=method_ref,
                             run_record_refs=[accepted_run])
        await save_session_state(fake_bus.db, state, auto_commit=True)
        await engine.tick(SimpleNamespace(
            event_type='memory.consolidated', session_id=sid, emitter='MemoryAgent',
            payload=bus.published[-1].payload))
        routed = await load_session_state(fake_bus.db, sid)
        assert routed.current_state == 'RevisingMethod'
        assert routed.method_feedback_ref == memory['method_feedback_refs'][0]
    assert len(llm.calls) == 1


async def test_memory_publishes_review_and_replay_does_not_call_again(store, enabled):
    method_ref, run_ref, result_ref = seed(store)
    llm = LLM()
    handler = MemoryAgentHandler(spec_store=store, llm_client=llm)
    verification_ref = store.put(kind='verification_report', session_id='feedback', key='v1',
                                  obj={'verdict': 'verified'}, emitter='EvaluatorAgent')
    event = _Envelope('event', 'feedback', 'memory.consolidation_requested', 'ResearchOrchestrator',
                      {'run_record_ref': run_ref, 'verification_report_ref': verification_ref})
    bus = MemoryBus()
    await handler.handle_consolidation_requested(bus, event)
    published = bus.published[-1]
    assert published.event_type == 'memory.consolidated'
    assert published.payload['diagnosis'] == 'method_improvement_proposed'
    schema = json.loads((ROOT / 'eventbus/schemas/memory.consolidated.v1.schema.json').read_text())
    Draft202012Validator(schema).validate(published.payload)
    memory = store.get(published.payload['research_memory_ref'])
    RESEARCH_MEMORY_VALIDATOR.validate(memory)
    assert memory['best_result_ref'] == result_ref
    assert memory['best_metric'] == 2.0
    review_ref = memory['method_feedback_refs'][0]
    review = store.get(review_ref)
    assert review['incumbent_ref'] == result_ref
    assert review['method_ref'] == method_ref
    assert review['status'] == 'completed'
    assert review['elapsed_seconds'] >= 0
    assert await handler._review_complete_result(
        'feedback', run_ref, memory, {'verdict': 'verified'}, 'valid_improvement') == review_ref
    assert len(llm.calls) == 1
    _, run2, _ = seed(store, metric=2.5, key='second')
    worse, *_ = handler._consolidate_gen(
        run2, prior_research_memory=memory, verification_report={'verdict': 'verified'})
    assert worse['best_metric'] == 2.0
    assert worse['best_result_ref'] == result_ref
    assert await handler._review_complete_result(
        'feedback', run2, worse, {'verdict': 'verified'}, 'valid_improvement') is None
    assert len(llm.calls) == 1


@pytest.mark.parametrize('case', ['disabled', 'component', 'failed_verification', 'no_incumbent', 'failure'])
async def test_review_gates(store, enabled, monkeypatch, case):
    _, run_ref, result_ref = seed(store, mode='component' if case == 'component' else 'integration')
    if case == 'disabled':
        monkeypatch.delenv('AUTORESEARCH_METHOD_FEEDBACK')
    llm = LLM()
    handler = MemoryAgentHandler(spec_store=store, llm_client=llm)
    memory = {'best_metric': 2.0, 'best_result_ref': None if case == 'no_incumbent' else result_ref}
    answer = await handler._review_complete_result(
        'feedback', run_ref, memory,
        {'verdict': 'failed' if case == 'failed_verification' else 'verified'},
        'execution_failure' if case == 'failure' else 'valid_improvement')
    assert answer is None
    assert not llm.calls
    assert not store.list_refs('feedback', 'method_improvement_review')


async def test_interrupted_review_slot_is_preserved_and_recovered(store, enabled):
    method_ref, run_ref, result_ref = seed(store)
    store.put(kind='method_improvement_review', session_id='feedback', key='slot-1',
              obj={'status': 'started', 'result_ref': result_ref}, emitter='MemoryAgent')
    llm = LLM()
    handler = MemoryAgentHandler(spec_store=store, llm_client=llm)
    await handler._review_complete_result(
        'feedback', run_ref, {'best_metric': 2.0, 'best_result_ref': result_ref},
        {'verdict': 'verified'}, 'valid_improvement')
    assert len(llm.calls) == 1
    assert store.get('spec://method_improvement_review/feedback/slot-1')['status'] == 'failed'
    assert len(store.list_refs('feedback', 'method_improvement_review')) == 2


@pytest.mark.parametrize('stale', [False, True])
async def test_review_routes_to_formal_revision_and_checks_binding(store, enabled, fake_bus, stale):
    method_ref, run_ref, result_ref = seed(store)
    review_ref = store.put(kind='method_improvement_review', session_id='feedback', key='slot-1',
        obj={'status': 'completed', 'method_ref': method_ref, 'incumbent_ref': result_ref,
             'incumbent_metric': 99 if stale else 2.0, 'decision': decision()}, emitter='MemoryAgent')
    memory_ref = store.put(kind='research_memory', session_id='feedback', key='v1',
        obj={'best_result_ref': result_ref, 'best_metric': 2.0, 'method_feedback_refs': [review_ref]},
        emitter='MemoryAgent')
    engine = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT, spec_store=store)
    await engine.start()
    state = SessionState(session_id='feedback', user_id='u', current_state='ConsolidatingMemory',
                         method_ref=method_ref, run_record_refs=[run_ref])
    await save_session_state(fake_bus.db, state, auto_commit=True)
    event = SimpleNamespace(event_type='memory.consolidated', session_id='feedback', emitter='MemoryAgent',
                            payload={'research_memory_ref': memory_ref, 'routing_verdict': 'need_experiment',
                                     'diagnosis': 'method_improvement_proposed', 'diagnosis_evidence': [review_ref]})
    await engine.tick(event)
    state = await load_session_state(fake_bus.db, 'feedback')
    assert state.current_state == ('Reporting' if stale else 'RevisingMethod')
    assert state.method_feedback_round == (0 if stale else 1)
    if not stale:
        assert state.method_feedback_in_progress
        assert 'conditional correction' in engine._method_revision_feedback(state)
        assert any(e.event_type == 'method.revise_requested' for e in fake_bus.published)


async def test_intermediate_revision_routes_after_complete_review_limit(
        store, enabled, fake_bus):
    method_ref, run_ref, result_ref = seed(store, sid='intermediate-route')
    review_ref = store.put(kind='method_improvement_review', session_id='intermediate-route',
        key='slot-2', obj={'status': 'completed', 'trigger': 'legal_off_method_gain',
                           'method_ref': method_ref, 'incumbent_ref': result_ref,
                           'incumbent_metric': 2.0, 'decision': decision()},
        emitter='MemoryAgent')
    memory_ref = store.put(kind='research_memory', session_id='intermediate-route', key='v1',
        obj={'best_result_ref': result_ref, 'best_metric': 2.0,
             'method_feedback_refs': [review_ref]}, emitter='MemoryAgent')
    engine = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT, spec_store=store)
    await engine.start()
    state = SessionState(session_id='intermediate-route', user_id='u',
                         current_state='ConsolidatingMemory', method_ref=method_ref,
                         method_feedback_round=1, run_record_refs=[run_ref])
    await save_session_state(fake_bus.db, state, auto_commit=True)
    await engine.tick(SimpleNamespace(
        event_type='memory.consolidated', session_id='intermediate-route',
        emitter='MemoryAgent', payload={'research_memory_ref': memory_ref,
            'routing_verdict': 'need_experiment', 'diagnosis': 'method_improvement_proposed',
            'diagnosis_evidence': [review_ref]}))
    routed = await load_session_state(fake_bus.db, 'intermediate-route')
    assert routed.current_state == 'RevisingMethod'
    assert routed.method_feedback_round == 1


async def test_incremental_revision_audits_then_searches_only_if_needed(enabled, fake_bus):
    engine = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await engine.start()
    state = SessionState(session_id='feedback', user_id='u', current_state='DeliberatingMethod',
                         mode='research', method_feedback_in_progress=True,
                         literature_brief_ref='literature://old')
    await engine._on_DeliberatingMethod_method_deliberated(
        state, SimpleNamespace(payload={'method_ref': 'method://revised'}))
    assert state.current_state == 'AuditingLiteratureNovelty'
    assert fake_bus.published[-1].event_type == 'evidence.audit_requested'
    search_id = state.novelty_targeted_search_id
    await engine._on_AuditingLiteratureNovelty_novelty_audit_ready(state, SimpleNamespace(payload={
        'verdict': 'evidence_insufficient', 'novelty_audit_report_ref': 'audit://gap',
        'targeted_search_id': search_id}))
    assert state.current_state == 'DeepResearching'
    assert state.novelty_targeted_search_pending
    state.component_plan_audit_pending = True
    await engine._on_DeliberatingMethod_method_deliberated(
        state, SimpleNamespace(payload={'method_ref': 'method://confirmed-cap'}))
    assert state.current_state == 'PlanningExperiment'
    assert not state.method_feedback_in_progress


@pytest.mark.parametrize('feedback,force,rounds,expected', [
    ('1', '1', 0, 'Reporting'), ('0', '0', 0, 'Reporting'),
    ('0', '1', 0, 'PlanningExperiment'), ('0', '1', 1, 'Reporting'),
])
async def test_efficient_report_does_not_spend_all_legacy_outer_rounds(
        enabled, monkeypatch, fake_bus, feedback, force, rounds, expected):
    monkeypatch.setenv('AUTORESEARCH_METHOD_FEEDBACK', feedback)
    monkeypatch.setenv('AUTORESEARCH_FORCE_KEEP_SEARCHING', force)
    monkeypatch.setenv('AUTORESEARCH_LOOP_MAX_ROUNDS', '20')
    engine = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await engine.start()
    state = SessionState(session_id='stop', user_id='u', current_state='ConsolidatingMemory',
                         exploration_loop_round=rounds)
    await engine._on_ConsolidatingMemory_memory_consolidated(state, SimpleNamespace(payload={
        'research_memory_ref': 'memory://test', 'routing_verdict': 'report',
        'diagnosis': 'valid_improvement'}))
    assert state.current_state == expected


async def test_failed_review_is_recorded_and_keeps_incumbent(store, enabled):
    _, run_ref, result_ref = seed(store)
    handler = MemoryAgentHandler(spec_store=store, llm_client=LLM(RuntimeError('offline')))
    memory = {'best_metric': 2.0, 'best_result_ref': result_ref}
    with pytest.raises(RuntimeError, match='offline'):
        await handler._review_complete_result(
            'feedback', run_ref, memory, {'verdict': 'verified'}, 'valid_improvement')
    review = store.get(memory['method_feedback_refs'][0])
    assert review['status'] == 'failed'
    assert review['decision']['route'] == 'unavailable'
    assert review['decision']['error_detail'] == 'RuntimeError: offline'
    assert memory['best_result_ref'] == result_ref
    assert memory['best_metric'] == 2.0


async def test_legacy_value_error_review_can_be_replayed_once(store, enabled):
    method_ref, run_ref, result_ref = seed(store)
    legacy_ref = store.put(kind='method_improvement_review', session_id='feedback', key='slot-1',
        obj={'status': 'completed', 'trigger': 'complete_result',
             'method_ref': method_ref, 'result_ref': result_ref,
             'incumbent_ref': result_ref,
             'decision': {'route': 'report', 'reason': 'review unavailable: ValueError'}},
        emitter='MemoryAgent')
    llm = LLM()
    handler = MemoryAgentHandler(spec_store=store, llm_client=llm)
    memory = {'best_metric': 2.0, 'best_result_ref': result_ref}
    repaired_ref = await handler._review_complete_result(
        'feedback', run_ref, memory, {'verdict': 'verified'}, 'valid_improvement')
    assert repaired_ref != legacy_ref
    assert store.get(legacy_ref)['decision']['reason'] == 'review unavailable: ValueError'
    assert store.get(repaired_ref)['status'] == 'completed'
    assert len(llm.calls) == 1
    assert await handler._review_complete_result(
        'feedback', run_ref, memory, {'verdict': 'verified'}, 'valid_improvement') == repaired_ref
    assert len(llm.calls) == 1


async def test_failed_protocol_review_can_be_replayed_after_repair(store, enabled):
    _, run_ref, result_ref = seed(store)
    memory = {'best_metric': 2.0, 'best_result_ref': result_ref}
    broken = MemoryAgentHandler(spec_store=store, llm_client=LLM(decision('unsupplied')))
    with pytest.raises(RuntimeError, match='unsupplied|unknown|evidence'):
        await broken._review_complete_result(
            'feedback', run_ref, memory, {'verdict': 'verified'}, 'valid_improvement')
    failed_ref = memory['method_feedback_refs'][0]
    assert store.get(failed_ref)['status'] == 'failed'

    repaired = MemoryAgentHandler(spec_store=store, llm_client=LLM())
    repaired_ref = await repaired._review_complete_result(
        'feedback', run_ref, memory, {'verdict': 'verified'}, 'valid_improvement')

    assert repaired_ref != failed_ref
    assert store.get(failed_ref)['status'] == 'failed'
    assert store.get(repaired_ref)['status'] == 'completed'


async def test_intermediate_reviews_do_not_consume_complete_review_rounds(
        store, enabled, monkeypatch):
    handler = MemoryAgentHandler(spec_store=store, llm_client=LLM())
    monkeypatch.setattr(
        'autoresearch.agents.memory_agent.handler._method_review_trigger',
        lambda *args: 'cap_expectation_shortfall')
    for index, metric in enumerate((2.0, 1.9)):
        _, run_ref, result_ref = seed(store, metric=metric, key=f'intermediate-{index}')
        review_ref = await handler._review_complete_result(
            'feedback', run_ref, {'best_metric': metric, 'best_result_ref': result_ref},
            {'verdict': 'verified'}, 'component_progress')
        assert review_ref is not None
        assert store.get(review_ref)['trigger'] == 'cap_expectation_shortfall'
    monkeypatch.setattr(
        'autoresearch.agents.memory_agent.handler._method_review_trigger',
        lambda *args: 'complete_result')
    _, run_ref, result_ref = seed(store, metric=1.8, key='full-1')
    assert await handler._review_complete_result(
        'feedback', run_ref, {'best_metric': 1.8, 'best_result_ref': result_ref},
        {'verdict': 'verified'}, 'valid_improvement') is not None
    _, run_ref, result_ref = seed(store, metric=1.7, key='full-2')
    assert await handler._review_complete_result(
        'feedback', run_ref, {'best_metric': 1.7, 'best_result_ref': result_ref},
        {'verdict': 'verified'}, 'valid_improvement') is None


@pytest.mark.parametrize('completed', [1, 2])
async def test_complete_review_budget_counts_alias_and_snapshot_once(
        store, enabled, monkeypatch, completed):
    monkeypatch.setenv('AUTORESEARCH_METHOD_FEEDBACK_ROUNDS', '2')
    for index in range(completed):
        review = {
            'status': 'completed', 'trigger': 'complete_result',
            'result_ref': f'prior-result-{index}',
            'decision': {'route': 'report', 'reason': 'No supported change.'}}
        store.put(kind='method_improvement_review', session_id='feedback',
                  key=f'prior-{index}', obj=review,
                  emitter='MemoryAgent')
        store.put(kind='method_improvement_review', session_id='feedback',
                  key='snapshot-' + artifact_digest(review).split(':')[1], obj=review,
                  emitter='MemoryAgent')
    assert len(store.list_refs('feedback', 'method_improvement_review')) == completed * 2
    _, run_ref, result_ref = seed(store)
    llm = LLM()
    handler = MemoryAgentHandler(spec_store=store, llm_client=llm)

    review_ref = await handler._review_complete_result(
        'feedback', run_ref, {'best_metric': 2.0, 'best_result_ref': result_ref},
        {'verdict': 'verified'}, 'valid_improvement')

    assert (review_ref is not None) == (completed < 2)
    assert len(llm.calls) == (1 if completed < 2 else 0)


async def test_review_preserves_older_trustworthy_best_component_evidence(store, enabled):
    _, run_ref, result_ref = seed(store)
    old_best = {
        'sequence': 1, 'component_id': 'old-best', 'mode': 'component',
        'plan_version': 1, 'method_fingerprint': 'method-a', 'round': 1,
        'role': 'direct_gain', 'status': 'accepted', 'operator': 'IMPROVE',
        'execution_succeeded': True, 'dependency_closure': ['parent'],
        'parent_checkpoint': '/saved/parent', 'parent_checkpoints': ['/saved/parent'],
        'checkpoint': '/saved/old-best', 'diff_ref': 'diff://old-best',
        'metric': 0.596453, 'contract_passed': True, 'became_best': True,
        'acceptance_result': {'passed': True, 'acceptance_check_result': {'passed': False}},
        'mir_verdict': 'partial', 'verification': 'verified',
        'run_ref': 'run://old-best', 'changes_summary': 'legal historical best',
        'failure_reason': '',
    }
    invalid_old = {
        **old_best, 'sequence': 2, 'component_id': 'invalid-old', 'metric': 0.1,
        'contract_passed': False, 'checkpoint': '/saved/invalid-old',
    }
    recent = [
        {'sequence': i + 3, 'component_id': f'recent-{i}', 'metric': 1.0 + i,
         'status': 'accepted'}
        for i in range(20)
    ]
    llm = LLM(decision('historical_best_component_evidence'))
    handler = MemoryAgentHandler(spec_store=store, llm_client=llm)
    memory = {
        'best_metric': 2.0, 'best_result_ref': result_ref,
        'component_history': [old_best, invalid_old, *recent],
    }

    review_ref = await handler._review_complete_result(
        'feedback', run_ref, memory, {'verdict': 'verified'}, 'valid_improvement')

    assert review_ref is not None
    submitted = json.loads(llm.calls[0]['messages'][1]['content'])['evidence']
    assert len(submitted['component_history']) == 20
    assert submitted['component_history'][0]['component_id'] == 'recent-0'
    assert submitted['component_history'][0]['evidence_id'] == 'component_history[2]'
    assert submitted['historical_best_component_evidence'] == [{
        **{key: old_best.get(key) for key in (
            'sequence', 'component_id', 'mode', 'plan_version', 'method_fingerprint',
            'round', 'role', 'status', 'operator', 'dependency_closure',
            'parent_checkpoint', 'parent_checkpoints', 'checkpoint', 'diff_ref',
            'metric', 'contract_passed', 'acceptance_result', 'mir_verdict', 'verification', 'run_ref',
            'changes_summary', 'failure_reason')},
        'evidence_id': 'historical_best_component_evidence[0]',
        'source_history_index': 0,
    }]


def test_review_accepts_explicit_component_history_evidence_id():
    llm = LLM(decision('component_history[24]'))
    evidence = {'component_history': [
        {'evidence_id': 'component_history[24]', 'metric': 4.37}]}

    review = review_complete_method({}, {}, evidence, llm)

    assert review['route'] == 'revise_method'
    assert len(llm.calls) == 1


def test_review_accepts_historical_best_evidence_id():
    evidence_id = 'historical_best_component_evidence[0]'
    llm = LLM(decision(evidence_id))
    evidence = {'historical_best_component_evidence': [
        {'evidence_id': evidence_id, 'source_history_index': 23, 'metric': 4.37}]}

    review = review_complete_method({}, {}, evidence, llm)

    assert review['route'] == 'revise_method'
    assert len(llm.calls) == 1


def test_complete_method_review_inherits_configured_request_deadline():
    llm = LLM({'route': 'report', 'reason': 'Evidence supports reporting.'})
    result = review_complete_method({}, {}, {}, llm)
    assert result['route'] == 'report'
    assert 'stream_max_s' not in llm.calls[0]


async def test_failed_complete_review_does_not_publish_report_routing(
        store, enabled, monkeypatch):
    _, run_ref, result_ref = seed(store)
    handler = MemoryAgentHandler(spec_store=store, llm_client=LLM())
    monkeypatch.setattr(handler, '_gen_branch', lambda: True)
    monkeypatch.setattr(handler, '_consolidate_gen', lambda *a, **k: (
        {'best_result_ref': result_ref, 'best_metric': 2.0},
        'report', 'valid_improvement', []))
    async def no_review(*a, **k):
        return None
    async def failed_review(*a, **k):
        raise RuntimeError('complete-method decision timed out')
    async def publish_consolidated(client, session_id, causation_id, *a, **k):
        await client.publish(event_type='memory.consolidated', session_id=session_id,
                             payload={'routing_verdict': 'report'}, causation_id=causation_id)
    monkeypatch.setattr(handler, '_review_retained_off_method', no_review)
    monkeypatch.setattr(handler, '_review_complete_result', failed_review)
    monkeypatch.setattr(handler, '_publish_consolidated', publish_consolidated)
    bus = MemoryBus()
    await handler.handle_consolidation_requested(bus, _Envelope(
        event_id='failed-review', session_id='feedback',
        event_type='memory.consolidation_requested', emitter='VerificationAgent',
        payload={'run_record_ref': run_ref}))
    assert [event.event_type for event in bus.published] == ['agent.failed']
    assert 'complete-method decision timed out' in bus.published[0].payload['underlying_reason']
    assert store.get(result_ref)['accepted'] is True
    assert store.get(result_ref)['method_metric'] == 2.0



async def test_live_review_owner_blocks_duplicate_review(store, enabled):
    import asyncio
    from autoresearch.agents.memory_agent.handler import _ACTIVE_COMPLETE_REVIEWS
    _, run_ref, result_ref = seed(store)
    ref = store.put(kind="method_improvement_review", session_id="feedback", key="slot-1",
                    obj={"status": "started", "result_ref": result_ref}, emitter="MemoryAgent")
    wait = asyncio.Event()
    owner = asyncio.create_task(wait.wait())
    _ACTIVE_COMPLETE_REVIEWS[ref] = owner
    llm = LLM()
    handler = MemoryAgentHandler(spec_store=store, llm_client=llm)
    try:
        with pytest.raises(RuntimeError, match="unfinished"):
            await handler._review_complete_result(
                "feedback", run_ref, {"best_metric": 2.0, "best_result_ref": result_ref},
                {"verdict": "verified"}, "valid_improvement")
        assert not llm.calls
        assert store.get(ref)["status"] == "started"
    finally:
        wait.set()
        await owner
        _ACTIVE_COMPLETE_REVIEWS.pop(ref, None)


async def test_component_decision_separates_current_history_and_used_budget(store, enabled, tmp_path, monkeypatch):
    monkeypatch.setenv('AUTORESEARCH_WEB_WORKDIR', str(tmp_path))
    monkeypatch.setenv('AUTORESEARCH_COMPONENT_MAX_ATTEMPTS', '8')
    sid = 'scoped-budget'
    method = {'components': [_component('feature', 'indirect_gain')]}
    method_ref = store.put(kind='method', session_id=sid, key='v1', obj=method, emitter='MethodAgent')
    baseline = {'reproduced':True, 'metric_value':1.0}
    baseline_ref = store.put(kind='baseline_result', session_id=sid, key='v1', obj=baseline, emitter='RunnerAgent')
    plan = build_component_plan(method)
    ledger = ledger_for_workroot(session_workroot(sid),plan,execution_identity=artifact_digest(baseline))
    ledger.begin('feature')
    ledger.record_attempt('feature',execution_succeeded=True,contract_passed=True,checkpoint='qualified',quality_value=.1,max_attempts=8)
    for _ in range(3):
        ledger.record_attempt('feature',execution_succeeded=False,error='variant failed',max_attempts=8)
    run = {'experiment_id':method_ref}
    run_ref = store.put(kind='run_record',session_id=sid,key='snapshot-'+artifact_digest(run).split(':')[1],obj=run,emitter='RunnerAgent')
    gen = {'run_record_snapshot_ref':run_ref,'component_execution':{'mode':'component','component_id':'feature','status':'execution_failed'}}
    result_ref = store.put(kind='gen_experiment_result',session_id=sid,key='snapshot-'+artifact_digest(gen).split(':')[1],obj=gen,emitter='RunnerAgent')
    current = {'component_id':'feature','method_fingerprint':plan.method_fingerprint,'plan_version':plan.version,'baseline_digest':artifact_digest(baseline)}
    history = [current, {**current,'method_fingerprint':'old-method'}, {**current,'baseline_digest':'old-baseline'}, {**current,'plan_version':99}]
    llm = LLM({'action':'improve','reason':'one concrete diagnostic can resolve remaining interface uncertainty',
               'next_experiment':'Inspect saved feature consumption without rerunning prior successful training'})
    handler = MemoryAgentHandler(spec_store=store,llm_client=llm)
    await handler._decide_efficient_component(sid,run_ref,{'component_history':history},
        {'verdict':'verified','run_record_ref':run_ref,'candidate_snapshot_ref':result_ref},baseline_ref)
    packet = json.loads(llm.calls[0]['messages'][1]['content'])
    assert packet['recent_history'] == [current]
    assert packet['best']['attempts'] == 4 and packet['best']['qualified_attempts'] == 1
    assert packet['best']['attempt_limit'] == 8 and packet['best']['remaining_attempts'] == 4
    assert ledger.entry('feature').search_decision['next_experiment'] == llm.response['next_experiment']
    assert ledger.entry('feature').checkpoint == 'qualified'


@pytest.mark.parametrize('role', ['direct_gain', 'indirect_gain'])
@pytest.mark.parametrize('limit', [6, 8])
@pytest.mark.parametrize('last_failed', [False, True])
async def test_accepted_component_at_limit_preserves_best_and_advances(
        store, enabled, tmp_path, monkeypatch, role, limit, last_failed):
    monkeypatch.setenv('AUTORESEARCH_WEB_WORKDIR', str(tmp_path))
    monkeypatch.setenv('AUTORESEARCH_COMPONENT_MAX_ATTEMPTS', str(limit))
    sid = 'accepted-at-limit'
    method = {'components': [_component('feature', role),
                             _component('downstream', depends_on=['feature'])]}
    method_ref = store.put(kind='method', session_id=sid, key='v1', obj=method, emitter='MethodAgent')
    baseline = {'reproduced': True, 'metric_value': 10.0}
    baseline_ref = store.put(kind='baseline_result', session_id=sid, key='v1', obj=baseline, emitter='RunnerAgent')
    plan = build_component_plan(method)
    ledger = ledger_for_workroot(session_workroot(sid), plan, execution_identity=artifact_digest(baseline))
    ledger.begin('feature')
    for index in range(limit):
        success = not (last_failed and index == limit - 1)
        ledger.record_attempt('feature', execution_succeeded=success, contract_passed=success,
                              checkpoint='host-best' if success else None, metric_value=8 if success else None,
                              quality_value=.1 if success else None, max_attempts=limit)
    before = ledger.entry('feature')
    assert before.status == 'accepted' and before.attempts == limit
    run = {'experiment_id': method_ref}
    run_ref = store.put(kind='run_record', session_id=sid, key='snapshot-' + artifact_digest(run).split(':')[1],
                        obj=run, emitter='RunnerAgent')
    gen = {'run_record_snapshot_ref': run_ref, 'component_execution': {
        'mode': 'component', 'component_id': 'feature',
        'status': 'execution_failed' if last_failed else 'accepted'}}
    result_ref = store.put(kind='gen_experiment_result', session_id=sid,
                           key='snapshot-' + artifact_digest(gen).split(':')[1], obj=gen, emitter='RunnerAgent')
    llm = LLM({'action': 'advance', 'reason': 'attempt limit reached; retain qualified result', 'selected_checkpoint': 'best'})
    handler = MemoryAgentHandler(spec_store=store, llm_client=llm)
    await handler._decide_efficient_component(sid, run_ref, {}, {
        'verdict': 'verified', 'run_record_ref': run_ref, 'candidate_snapshot_ref': result_ref}, baseline_ref)
    retained = ledger.entry('feature')
    assert retained.status == 'accepted' and retained.checkpoint == before.checkpoint
    assert retained.metric_value == before.metric_value and retained.attempts == limit
    assert ledger.next_ready().component_id == 'downstream'
    assert len(llm.calls) == (1 if role == 'indirect_gain' else 0)
    await handler._decide_efficient_component(sid, run_ref, {}, {
        'verdict': 'verified', 'run_record_ref': run_ref, 'candidate_snapshot_ref': result_ref}, baseline_ref)
    assert ledger.entry('feature').attempts == limit
    assert len(llm.calls) == (1 if role == 'indirect_gain' else 0)


@pytest.mark.parametrize('retained,core_rebuild,bound,expected', [
    (True, False, True, 'DeliberatingMethod'),
    (True, True, True, 'DeliberatingMethod'),
    (False, False, True, 'DeliberatingMethod'),
    (True, False, False, 'DeliberatingMethod'),
])
async def test_reflected_local_contract_repair_preserves_negative_evidence(
        store, enabled, fake_bus, retained, core_rebuild, bound, expected):
    sid = 'local-contract-repair'
    prior_ref = store.put(kind='method', session_id=sid, key='prior',
        obj={'components': [_component('core')]}, emitter='MethodAgent')
    baseline_ref = store.put(kind='baseline_result', session_id=sid, key='frozen',
        obj={'reproduced': True, 'metric_value': 3.0, 'benchmark_id': 'bench-repair'}, emitter='RunnerAgent')
    candidate = {'accepted': False, 'method_metric': None,
        'method_implementation_review': {'verdict': 'faithful',
            'task_contract_review': {'verdict': 'violation',
                'violations': ['auxiliary script imported a forbidden library']}}}
    candidate_ref = store.put(kind='gen_experiment_result', session_id=sid, key='invalid',
        obj=candidate, emitter='RunnerAgent')
    review_ref = store.put(kind='method_improvement_review', session_id=sid, key='repair',
        obj={'status': 'completed', 'trigger': 'mir_task_contract_violation',
             'method_ref': prior_ref if bound else 'other-method', 'result_ref': candidate_ref,
             'mir_violation_evidence': {'score_admissible': False},
             'decision': {**decision(candidate_ref),
                'retain': ['verified core mechanism'] if retained else [],
                'remove': ['the forbidden auxiliary import'],
                'requires_core_reconstruction': core_rebuild}}, emitter='MemoryAgent')
    revised_ref = store.put(kind='method', session_id=sid, key='revised',
        obj={'components': [_component('core')]}, emitter='MethodAgent')
    engine = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT, spec_store=store)
    engine._current_gen_result = lambda _: candidate
    await engine.start()
    state = SessionState(session_id=sid, user_id='u', current_state='RevisingMethod',
        mode='benchmark', method_ref=prior_ref, baseline_result_ref=baseline_ref,
        benchmark_spec_ref='benchmark_spec://local-contract-repair/frozen', current_benchmark_id='bench-repair',
        method_feedback_ref=review_ref, method_feedback_in_progress=True)
    await engine._on_RevisingMethod_method_generated(
        state, SimpleNamespace(payload={'method_ref': revised_ref}))
    assert state.current_state == expected
    assert state.component_plan_audit_pending is (retained and not core_rebuild and bound)
    assert state.baseline_result_ref == baseline_ref
    assert store.get(candidate_ref) == candidate and candidate['accepted'] is False
    assert fake_bus.published[-1].event_type == 'method.deliberation_requested'
    assert fake_bus.published[-1].payload['component_plan_audit'] is state.component_plan_audit_pending


@pytest.mark.parametrize('closing', ['budget', 'user', 'normal'])
@pytest.mark.parametrize('mir_verdict', ['faithful', 'partial'])
async def test_closeout_reflects_verified_whole_without_reopening_search(
        store, enabled, closing, mir_verdict):
    baseline = {'reproduced': True, 'metric_value': 3.0}
    baseline_ref = store.put(kind='baseline_result', session_id='feedback', key='frozen-closeout',
                             obj=baseline, emitter='RunnerAgent')
    _, run_ref, result_ref = seed(store, feedback={
        'baseline_digest': artifact_digest(baseline), 'component_execution': {},
        'method_implementation_review': {'verdict': mir_verdict,
            'task_contract_review': {'verdict': 'compliant'}}})
    memory = {'best_result_ref': result_ref, 'best_metric': 2.0}
    verification = {'verdict': 'verified', 'run_record_ref': run_ref,
        'run_record_digest': artifact_digest(store.get(run_ref)), 'candidate_snapshot_ref': result_ref}
    if closing == 'user':
        store.put(kind='human_idea_seed', session_id='feedback', key='closeout-v1',
                  obj={'closeout': True}, emitter='Human')
    terminal = closing != 'normal' and mir_verdict == 'faithful'
    llm = LLM({'route': 'report', 'reason': 'Analyze the verified final result and stop.',
               'evidence_refs': [result_ref]} if terminal else
              RuntimeError('Ordinary exploration review reached.'))
    handler = MemoryAgentHandler(spec_store=store, llm_client=llm)
    if terminal:
        original = store.get(result_ref)
        assert await handler._review_complete_result('feedback', run_ref, memory, verification,
            'inconclusive', baseline_ref, closeout_requested=closing == 'budget') is None
        assert len(llm.calls) == 1
        packet = json.loads(llm.calls[0]['messages'][1]['content'])
        assert packet['evidence']['terminal_whole_closeout'] is True
        review = store.get(memory['method_feedback_refs'][-1])
        assert review['status'] == 'completed' and review['decision']['route'] == 'report'
        assert review['result_ref'] == result_ref and review['terminal_whole_closeout'] is True
        assert await handler._review_complete_result('feedback', run_ref, memory, verification,
            'inconclusive', baseline_ref, closeout_requested=closing == 'budget') is None
        assert len(llm.calls) == 1 and store.get(result_ref) == original
    else:
        with pytest.raises(RuntimeError, match='Ordinary exploration review reached'):
            await handler._review_complete_result('feedback', run_ref, memory, verification,
                'inconclusive', baseline_ref, closeout_requested=closing == 'budget')
        assert llm.calls
    assert memory['best_result_ref'] == result_ref


async def test_terminal_whole_review_is_not_skipped_by_prior_component_closeout(store, enabled):
    baseline = {'reproduced': True, 'metric_value': 3.0}
    baseline_ref = store.put(kind='baseline_result', session_id='feedback', key='baseline',
                             obj=baseline, emitter='RunnerAgent')
    method_ref, run_ref, result_ref = seed(store)
    old = {'status': 'completed', 'method_ref': method_ref,
           'trigger': 'component_method_reconciliation', 'closeout_requested': True,
           'result_ref': 'old-component-result', 'incumbent_ref': result_ref,
           'decision': {'route': 'revise_method'}}
    old_ref = store.put(kind='method_improvement_review', session_id='feedback',
                        key='prior-component', obj=old, emitter='MemoryAgent')
    memory = {'best_result_ref': result_ref, 'best_metric': 2.0, 'method_feedback_refs': [old_ref]}
    verification = {'verdict': 'verified', 'run_record_ref': run_ref,
        'run_record_digest': artifact_digest(store.get(run_ref)), 'candidate_snapshot_ref': result_ref}
    llm = LLM({'route': 'report', 'reason': 'Qualified whole result is now available.',
               'evidence_refs': [result_ref]})
    handler = MemoryAgentHandler(spec_store=store, llm_client=llm)
    assert await handler._review_complete_result('feedback', run_ref, memory, verification,
        'valid_improvement', baseline_ref, closeout_requested=True) is None
    assert len(llm.calls) == 1 and len(memory['method_feedback_refs']) == 2
    assert store.get(old_ref) == old


async def test_terminal_whole_review_rejects_revision_without_host_fallback(store, enabled):
    baseline_ref = store.put(kind='baseline_result', session_id='feedback', key='baseline',
                             obj={'reproduced': True, 'metric_value': 3.0}, emitter='RunnerAgent')
    _, run_ref, result_ref = seed(store)
    original = store.get(result_ref)
    memory = {'best_result_ref': result_ref, 'best_metric': 2.0}
    verification = {'verdict': 'verified', 'run_record_ref': run_ref,
        'run_record_digest': artifact_digest(store.get(run_ref)), 'candidate_snapshot_ref': result_ref}
    llm = LLM(decision(result_ref))
    handler = MemoryAgentHandler(spec_store=store, llm_client=llm)
    with pytest.raises(RuntimeError):
        await handler._review_complete_result('feedback', run_ref, memory, verification,
            'valid_improvement', baseline_ref, closeout_requested=True)
    review = store.get(memory['method_feedback_refs'][-1])
    assert llm.calls and review['status'] == 'failed'
    assert review['decision']['route'] == 'unavailable' and 'decision_source' not in review
    assert store.get(result_ref) == original


@pytest.mark.parametrize('problem', [None, 'wrong_verification', 'exhausted_budget',
                                   'already_reflected', 'regression_reflected'])
async def test_reporting_reflects_exact_selected_whole_after_component_closeout(
        store, enabled, monkeypatch, problem):
    sid = 'selected-whole-closeout'
    baseline = {'reproduced': True, 'metric_value': 3.0}
    baseline_ref = store.put(kind='baseline_result', session_id=sid, key='frozen',
                             obj=baseline, emitter='RunnerAgent')
    method_ref, run_ref, old_result_ref = seed(store, sid=sid)
    candidate = {**store.get(old_result_ref), 'method_ref': method_ref,
        'baseline_result_ref': baseline_ref, 'baseline_digest': artifact_digest(baseline),
        'measurement_evidence': {'value': 2.0, 'valid': True, 'attempt_id': 'attempt-selected',
            'stdout_sha256': 'bound-output', 'protocol_fingerprint': 'official',
            'protocol': {'kind': 'official_single'}},
        'candidate_admissibility': {'verdict': 'pass', 'attempt_id': 'attempt-selected'},
        'repository_comparison_review': {'verdict': 'pass'},
        'method_implementation_review': {'verdict': 'faithful', 'method_ref': method_ref,
            'task_contract_review': {'verdict': 'compliant'}}}
    result_ref = store.put(kind='gen_experiment_result', session_id=sid,
        key='snapshot-' + artifact_digest(candidate).split(':')[1], obj=candidate,
        emitter='RunnerAgent')
    evaluation = {'run_record_ref': run_ref,
        'run_record_digest': artifact_digest(store.get(run_ref)), 'verdict': 'valid'}
    evaluation_ref = store.put(kind='evaluation_report', session_id=sid,
        key='snapshot-' + artifact_digest(evaluation).split(':')[1], obj=evaluation,
        emitter='EvaluatorAgent')
    verification = {'verdict': 'verified', 'run_record_ref': run_ref,
        'run_record_digest': 'wrong' if problem == 'wrong_verification' else
            artifact_digest(store.get(run_ref)),
        'candidate_snapshot_ref': result_ref, 'evaluation_report_ref': evaluation_ref}
    store.put(kind='verification_report', session_id=sid,
        key='snapshot-' + artifact_digest(verification).split(':')[1], obj=verification,
        emitter='EvaluatorAgent')
    prior = {'status': 'completed', 'method_ref': method_ref,
        'result_ref': 'other-result' if problem == 'exhausted_budget' else result_ref,
        'trigger': 'complete_result' if problem == 'exhausted_budget' else
            'qualified_component_closeout', 'decision': {'route': 'report'}}
    if problem in {'already_reflected', 'regression_reflected'}:
        prior['trigger'] = ('complete_result' if problem == 'already_reflected' else
                            'complete_metric_regression')
    prior_ref = store.put(kind='method_improvement_review', session_id=sid, key='prior',
                          obj=prior, emitter='MemoryAgent')
    memory = {'best_result_ref': result_ref, 'best_metric': 2.0,
              'method_feedback_refs': [prior_ref]}
    llm = LLM({'route': 'report', 'reason': 'Reflect on this exact final selection.',
               'evidence_refs': [result_ref]})
    handler = MemoryAgentHandler(spec_store=store, llm_client=llm)
    monkeypatch.setattr('autoresearch.agents.memory_agent.handler._qualified_closeout_components',
        lambda *args, **kwargs: [{'component_id': 'retained', 'role': 'direct_gain',
                                 'checkpoint': '/saved/retained', 'metric': 1.0}])
    bus = MemoryBus()
    # The latest Method is different; final selection still belongs to the old Method.
    handler._request_inputs = {(sid, 'closing'): {'method_ref': 'other-method'}}
    if problem == 'wrong_verification':
        with pytest.raises(RuntimeError, match='selected whole'):
            await handler._publish_consolidated(bus, sid, 'closing', memory, 'report')
        assert not bus.published and not llm.calls
        return
    if problem in {'already_reflected', 'regression_reflected'}:
        await handler._publish_consolidated(bus, sid, 'closing', memory, 'report')
        assert not llm.calls and memory['method_feedback_refs'] == [prior_ref]
        assert bus.published[-1].payload['routing_verdict'] == 'report'
        return
    await handler._publish_consolidated(bus, sid, 'closing', memory, 'report')
    assert len(llm.calls) == 1
    review = store.get(memory['method_feedback_refs'][-1])
    assert review['trigger'] == 'complete_result' and review['status'] == 'completed'
    assert review['result_ref'] == result_ref and review['method_ref'] == method_ref
    assert review['terminal_whole_closeout'] is True
    assert bus.published[-1].payload['routing_verdict'] == 'report'
    await handler._publish_consolidated(bus, sid, 'closing', memory, 'report')
    assert len(llm.calls) == 1 and store.get(result_ref) == candidate
    # The terminal reflection does not grant another exploratory review round.
    assert await handler._review_complete_result(sid, run_ref, memory, verification,
        'valid_improvement', baseline_ref) is None
    assert len(llm.calls) == 1 and store.get(prior_ref) == prior


@pytest.mark.parametrize('diagnosis', ['inconclusive', 'valid_improvement', 'component_contract_retry'])
@pytest.mark.parametrize('mode', [None, 'integration'])
@pytest.mark.parametrize('closing', [False, True])
def test_framework_closeout_considers_retained_components_after_whole_results(diagnosis, mode, closing):
    baseline = {'reproduced': True, 'metric_value': 3.0}
    current = {'accepted': True, 'method_metric': 2.0, 'checkpoint': '/saved/whole',
        'baseline_digest': artifact_digest(baseline),
        'method_implementation_review': {'verdict': 'faithful',
            'task_contract_review': {'verdict': 'compliant'}}}
    if mode:
        current['component_execution'] = {'mode': mode, 'all_components_accepted': True}
    memory = {'best_result_ref': 'whole', 'best_metric': 2.0}
    qualified = [{'component_id': 'A', 'role': 'direct_gain',
                  'checkpoint': '/saved/A', 'metric': 1.0}]
    trigger = _method_review_trigger(current, memory, diagnosis, current, baseline,
        closeout_requested=closing, closeout_components=qualified)
    assert (trigger == 'qualified_component_closeout') is closing


@pytest.mark.parametrize('problem', [None, 'wrong_unit', 'task_violation',
                                   'unbound_metric', 'failed_contract',
                                   'different_official_protocol', 'revalidated_source',
                                   'other_revalidated_source', 'partial_revalidation'])
async def test_closeout_retains_better_local_component_for_fresh_whole_validation(store, enabled, problem):
    from autoresearch.agents.memory_agent.handler import _qualified_closeout_components

    sid = 'local-source-closeout'
    baseline = {'reproduced': True, 'metric_value': 3.0}
    baseline_digest = artifact_digest(baseline)
    method_ref = store.put(kind='method', session_id=sid, key='frozen',
        obj={'approach': 'Existing measured mechanism'}, emitter='MethodAgent')
    run = {'succeeded': True, 'metric_value': 1.0, 'experiment_id': method_ref}
    run_ref = store.put(kind='run_record', session_id=sid,
        key='snapshot-' + artifact_digest(run).split(':')[1], obj=run, emitter='RunnerAgent')
    evidence = {'value': 1.0, 'valid': True, 'stdout_sha256': 'sealed-output',
                'attempt_id': 'attempt-local', 'protocol_fingerprint': None,
                'protocol': {'kind': 'component_local'}}
    contract = {'passed': True, 'candidate_metric': 1.0, 'parent_metric': 3.0,
                'parent_evidence': {'host_bound': True}, 'measurement_evidence': evidence}
    candidate = {'accepted': True, 'method_metric': 1.0, 'metric_name': 'loss',
        'metric_unit': 'unit', 'lower_is_better': True, 'baseline_digest': baseline_digest,
        'method_ref': method_ref, 'checkpoint': '/saved/component',
        'run_record_snapshot_ref': run_ref, 'measurement_evidence': evidence,
        'candidate_admissibility': {'verdict': 'insufficient_evidence', 'attempt_id': 'attempt-local'},
        'repository_comparison_review': {'verdict': 'pass'},
        'method_implementation_review': {'verdict': 'faithful', 'method_ref': method_ref,
            'task_contract_review': {'verdict': 'compliant'}},
        'component_execution': {'mode': 'component', 'role': 'direct_gain',
            'status': 'qualified', 'execution_succeeded': True,
            'attempt_contract_result': contract}}
    if problem == 'wrong_unit':
        candidate['metric_unit'] = 'another-unit'
    elif problem == 'task_violation':
        candidate['method_implementation_review']['task_contract_review']['verdict'] = 'violation'
    elif problem == 'unbound_metric':
        evidence['value'] = 0.5
    elif problem == 'failed_contract':
        contract['passed'] = False
    elif problem == 'different_official_protocol':
        evidence.update(protocol={'kind': 'official_single'}, protocol_fingerprint='other-protocol')
    candidate_ref = store.put(kind='gen_experiment_result', session_id=sid,
        key='snapshot-' + artifact_digest(candidate).split(':')[1], obj=candidate, emitter='RunnerAgent')
    incumbent = {'accepted': True, 'method_metric': 2.0, 'metric_name': 'loss',
        'metric_unit': 'unit', 'lower_is_better': True, 'baseline_digest': baseline_digest,
        'checkpoint': '/saved/whole', 'measurement_evidence': {
            'value': 2.0, 'valid': True, 'stdout_sha256': 'whole-output',
            'protocol_fingerprint': 'official-protocol'}}
    history = {'mode': 'component', 'component_id': 'A', 'role': 'direct_gain',
        'status': 'qualified', 'baseline_digest': baseline_digest, 'became_best': True,
        'execution_succeeded': True, 'contract_passed': True, 'metric': 1.0,
        'verification_verdict': 'verified', 'mir_verdict': 'faithful',
        'checkpoint': '/saved/component', 'run_ref': run_ref}
    memory = {'best_result_ref': 'existing-whole', 'best_metric': 2.0,
              'component_history': [history]}
    current = None
    if problem in {'revalidated_source', 'other_revalidated_source', 'partial_revalidation'}:
        method = {'reconciled_result_ref': candidate_ref if problem != 'other_revalidated_source' else 'other-source'}
        current_method = store.put(kind='method', session_id=sid,
            key='snapshot-' + artifact_digest(method).split(':')[1], obj=method, emitter='MethodAgent')
        run = {'succeeded': True, 'metric_value': 2.5, 'experiment_id': current_method}
        current_run = store.put(kind='run_record', session_id=sid,
            key='snapshot-' + artifact_digest(run).split(':')[1], obj=run, emitter='RunnerAgent')
        current = {**candidate, 'method_metric': 2.5, 'method_ref': current_method,
            'checkpoint': '/saved/revalidated', 'component_execution': None,
            'run_record_snapshot_ref': current_run,
            'measurement_evidence': {'value': 2.5, 'valid': True, 'attempt_id': 'attempt-whole',
                'stdout_sha256': 'whole-output', 'protocol_fingerprint': 'official-protocol',
                'protocol': {'kind': 'official_single'}},
            'candidate_admissibility': {'verdict': 'pass', 'attempt_id': 'attempt-whole'},
            'method_implementation_review': {'verdict': 'partial' if problem == 'partial_revalidation' else 'faithful',
                'method_ref': current_method, 'task_contract_review': {'verdict': 'compliant'}}}
    sources = _qualified_closeout_components(memory, baseline_digest, store=store,
                                             incumbent=incumbent, current=current)
    assert bool(sources) is (problem in {None, 'other_revalidated_source', 'partial_revalidation'})
    if sources:
        assert sources[0]['checkpoint'] == '/saved/component'
        assert _method_review_trigger(incumbent, memory, 'inconclusive', incumbent,
            baseline, closeout_requested=True, closeout_components=sources) == 'qualified_component_closeout'
    # Retaining source requests a fresh whole evaluation; it does not award its observed score.
    assert memory['best_metric'] == 2.0 and memory['best_result_ref'] == 'existing-whole'
    assert store.get(candidate_ref) == candidate
    if problem == 'revalidated_source':
        baseline_ref = store.put(kind='baseline_result', session_id=sid, key='frozen',
                                 obj=baseline, emitter='RunnerAgent')
        current_ref = store.put(kind='gen_experiment_result', session_id=sid,
            key='snapshot-' + artifact_digest(current).split(':')[1], obj=current, emitter='RunnerAgent')
        memory['best_result_ref'] = store.put(kind='gen_experiment_result', session_id=sid,
            key='snapshot-' + artifact_digest(incumbent).split(':')[1], obj=incumbent, emitter='RunnerAgent')
        verification = {'verdict': 'verified', 'anti_cheat_flags': [],
            'run_record_ref': current_run, 'run_record_digest': artifact_digest(run),
            'candidate_snapshot_ref': current_ref}
        llm = LLM({**decision(current_ref), 'route': 'report', 'add': [], 'modify': [], 'remove': [],
            'reason': 'The exact source has completed whole validation; retain the better accepted whole result.'})
        handler = MemoryAgentHandler(spec_store=store, llm_client=llm)
        review_ref = await handler._review_complete_result(sid, current_run, memory, verification,
            'valid_improvement', baseline_ref, closeout_requested=True)
        assert review_ref is None  # No further Method revision is requested.
        review = store.get(memory['method_feedback_refs'][-1])
        assert review['trigger'] == 'complete_result'
        assert review['status'] == 'completed' and review['decision']['route'] == 'report'
        assert memory['best_metric'] == 2.0


@pytest.fixture
def incomplete_native_whole():
    method_ref = refs.encode('method', 'feedback', 'first')
    return {'accepted': False, 'verdict': 'valid_not_best', 'method_metric': 4.0,
        'method_ref': method_ref, 'component_execution': None,
        'candidate_admissibility': {'verdict': 'pass'},
        'repository_comparison_review': {'verdict': 'pass'},
        'measurement_evidence': {'valid': True, 'value': 4.0,
            'stdout_sha256': 'measured-output', 'protocol_fingerprint': 'official'},
        'method_implementation_review': {'verdict': 'partial', 'method_ref': method_ref,
            'issues': ['Required frozen implementation is missing.'],
            'task_contract_review': {'verdict': 'compliant'}}}


@pytest.mark.parametrize('problem', [None, 'stale_review', 'unverified', 'inadmissible',
    'review_unavailable', 'plan_revision', 'accepted_improvement', 'unbound_metric', 'missing_protocol'])
def test_incomplete_native_whole_is_implementation_repair(incomplete_native_whole, problem):
    from autoresearch.agents.memory_agent.handler import _classify_component_evidence
    gen = incomplete_native_whole
    if problem == 'stale_review':
        gen['method_implementation_review']['method_ref'] = 'another-method'
    elif problem == 'unverified':
        gen['measurement_evidence']['valid'] = False
    elif problem == 'inadmissible':
        gen['candidate_admissibility']['verdict'] = 'insufficient_evidence'
    elif problem == 'review_unavailable':
        gen['method_implementation_review']['review_error'] = 'review transport failure'
    elif problem == 'plan_revision':
        gen['method_implementation_review']['plan_revision'] = {'reason': 'Contradicted contract'}
    elif problem == 'accepted_improvement':
        gen.update(accepted=True, verdict='new_best', method_metric=2.0)
        gen['measurement_evidence']['value'] = 2.0
    elif problem == 'unbound_metric':
        gen['measurement_evidence']['value'] = 3.0
    elif problem == 'missing_protocol':
        gen['measurement_evidence']['protocol_fingerprint'] = None
    diagnosis, _ = _classify_component_evidence(gen)
    assert (diagnosis == 'execution_failure') is (problem is None)


@pytest.mark.parametrize('prior_fixes', [0, 1, 2])
async def test_native_whole_partial_uses_existing_fix_budget_and_closing_review(
        store, enabled, fake_bus, incomplete_native_whole, prior_fixes):
    method_ref, run_ref, result_ref = seed(store, metric=4.0, feedback=incomplete_native_whole)
    gen = store.get(result_ref)
    store.put(kind='gen_experiment_result', session_id='feedback', key='current',
              obj=gen, emitter='RunnerAgent')
    baseline_ref = store.put(kind='baseline_result', session_id='feedback', key='baseline',
        obj={'reproduced': True, 'metric_value': 3.0}, emitter='RunnerAgent')
    evaluation_ref = store.put(kind='evaluation_report', session_id='feedback', key='negative-whole',
        obj={'verdict': 'valid_not_best', 'metric_value': 4.0, 'is_improvement': False,
             'run_record_digest': artifact_digest(store.get(run_ref))}, emitter='EvaluatorAgent')
    verification_ref = store.put(kind='verification_report', session_id='feedback', key='negative-whole',
        obj={'verdict': 'verified', 'run_record_ref': run_ref, 'candidate_snapshot_ref': result_ref,
             'run_record_digest': artifact_digest(store.get(run_ref))}, emitter='EvaluatorAgent')
    engine = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT, spec_store=store)
    await engine.start()
    state = SessionState(session_id='feedback', user_id='u', mode='benchmark',
        current_state='ConsolidatingMemory', method_ref=method_ref, baseline_result_ref=baseline_ref,
        current_experiment_id=method_ref, run_record_refs=[run_ref],
        evaluation_report_refs=[evaluation_ref], verification_report_refs=[verification_ref],
        implementation_fix_round={method_ref: prior_fixes})
    await save_session_state(fake_bus.db, state, auto_commit=True)
    payload = engine._memory_consolidation_payload(state)
    assert bool(payload.get('qualified_component_closeout_requested')) is (prior_fixes == 2)
    llm = LLM(RuntimeError('Ordinary implementation repair needs no new Method review.'))
    handler = MemoryAgentHandler(spec_store=store, llm_client=llm)
    bus = MemoryBus()
    await handler.handle_consolidation_requested(bus, _Envelope('partial-whole', 'feedback',
        'memory.consolidation_requested', 'ResearchOrchestrator', payload))
    consolidated = bus.published[-1]
    assert consolidated.event_type == 'memory.consolidated', consolidated.payload
    assert consolidated.payload['diagnosis'] == 'execution_failure'
    await engine.tick(SimpleNamespace(event_type='memory.consolidated', session_id='feedback',
        emitter='MemoryAgent', payload=consolidated.payload))
    restored = await load_session_state(fake_bus.db, 'feedback')
    assert restored.implementation_fix_round[method_ref] == prior_fixes + 1
    assert restored.current_state == ('MossIntervening' if prior_fixes == 2 else 'PlanningExperiment')
    assert restored.setting_rework_round == 0 and restored.method_feedback_round == 0
    assert store.get(result_ref) == gen and gen['method_metric'] == 4.0
    assert not llm.calls
