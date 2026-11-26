"""Close on retained evidence without accepting a component as a whole result."""
import json
from types import SimpleNamespace

import pytest

from autoresearch.agents.memory_agent.handler import MemoryAgentHandler
from autoresearch.agents.memory_agent.handler import (
    _formal_best_eligible, _method_review_trigger)
from autoresearch.agents.memory_agent.tests.test_method_feedback import (
    LLM, SM_ROOT, decision, enabled, fake_bus, seed, store,
    test_failed_extension_recovers_bound_drifted_component_without_complete_best as seed_off_method,
)
from autoresearch.orchestrator.engine import (
    OrchestratorEngine, _COMPONENT_REVISION_CAP, _METHOD_REVISION_CAP)
from autoresearch.orchestrator.session_state import SessionState
from autoresearch.spec_store import refs
from autoresearch.spec_store.identity import artifact_digest
from autoresearch.experiments.repo_improve.runtime_evidence import (
    can_revalidate_partial_integration)


def test_partial_integration_without_task_subreview_is_reconciled_not_promoted():
    attempt_id = 'attempt-official-partial'
    result = {
        'accepted': True,
        'checkpoint': '/saved/partial-integration',
        'method_metric': 88.0,
        'baseline_metric': 128.0,
        'lower_is_better': True,
        'measurement_evidence': {
            'valid': True, 'value': 88.0, 'attempt_id': attempt_id,
            'stdout_sha256': 'host-output', 'protocol_fingerprint': 'official-v1',
            'protocol': {'kind': 'official_single'},
        },
        'candidate_admissibility': {'verdict': 'pass', 'attempt_id': attempt_id},
        'repository_comparison_review': {'verdict': 'pass'},
        'method_implementation_review': {
            'verdict': 'partial', 'method_ref': 'spec://method/s/m',
            'issues': ['one frozen Method component is absent'],
        },
        'component_execution': {
            'mode': 'integration', 'status': 'new_best',
            'execution_succeeded': True, 'all_components_accepted': True,
            'attempt_contract_result': {'passed': True},
        },
    }
    memory = {'best_metric': None, 'best_result_ref': None, 'component_history': []}
    baseline = {'reproduced': True, 'metric_value': 128.0}

    assert can_revalidate_partial_integration(result) is True
    assert _formal_best_eligible(result) is False
    assert _method_review_trigger(
        result, memory, 'valid_improvement', None, baseline) == (
            'component_method_reconciliation')

    violation = {
        **result,
        'method_implementation_review': {
            **result['method_implementation_review'],
            'task_contract_review': {'verdict': 'violation'},
        },
    }
    assert can_revalidate_partial_integration(violation) is False
    assert _method_review_trigger(
        violation, memory, 'valid_improvement', None, baseline) == (
            'mir_task_contract_violation')


@pytest.mark.parametrize('review_mode', ['valid', 'invalid'])
@pytest.mark.parametrize('case', [
    'better', 'higher', 'worse', 'equal', 'protocol', 'baseline', 'unverified',
    'extension_failed', 'same_checkpoint', 'same_source', 'missing_source',
    'no_incumbent_extension_failed', 'partial_extension',
])
async def test_closeout_selects_comparable_better_and_independent_components(
        store, enabled, fake_bus, case, review_mode):
    sid = 'closeout-candidates'
    baseline = {'reproduced': True, 'metric_value': 4.0}
    baseline_ref = store.put(kind='baseline_result', session_id=sid, key='frozen',
                             obj=baseline, emitter='RunnerAgent')
    common = {'baseline_digest': artifact_digest(baseline), 'metric_name': 'objective',
              'metric_unit': 'units', 'lower_is_better': case != 'higher'}
    _, _, incumbent_ref = seed(store, sid=sid, metric=2.0, key='incumbent', feedback={
        **common, 'measurement_evidence': {'valid': True, 'value': 2.0,
            'protocol_fingerprint': 'official', 'stdout_sha256': 'best-output'}})
    metric = 3.0 if case in {'worse', 'higher'} else 2.0 if case == 'equal' else 1.0
    _, old_run, old_result = seed(store, sid=sid, mode='component', metric=metric, key='A', feedback={
        **common, 'checkpoint': '/saved/A', 'baseline_relative_diff': 'retained-source',
        'measurement_evidence': {'valid': True, 'value': metric, 'stdout_sha256': 'A-output',
            'protocol_fingerprint': 'local' if case == 'protocol' else 'official'},
        'method_implementation_review': {'verdict': 'faithful',
                                       'task_contract_review': {'verdict': 'compliant'}}})
    history = {'component_id': 'A', 'mode': 'component', 'role': 'direct_gain',
        'status': 'accepted', 'became_best': True, 'execution_succeeded': True,
        'contract_passed': True, 'mir_verdict': 'faithful', 'metric': metric,
        'verification_verdict': 'failed' if case == 'unverified' else 'verified',
        'baseline_digest': 'other' if case == 'baseline' else common['baseline_digest'],
        'checkpoint': '/saved/A', 'run_ref': old_run}
    failed = case in {'extension_failed', 'same_checkpoint', 'same_source', 'missing_source',
                     'no_incumbent_extension_failed', 'partial_extension'}
    current = {**common, 'accepted': False, 'verdict': 'invalid',
        'checkpoint': '/saved/A' if case == 'same_checkpoint' else '/saved/B',
        'baseline_relative_diff': (None if case == 'missing_source' else
                                  'retained-source' if case == 'same_source' else 'extension-source'),
        'diagnosis': 'execution_failure' if failed else 'component_execution_blocked',
        'component_execution': {'mode': 'component', 'component_id': 'B',
            'status': 'execution_failed' if failed else 'exhausted', 'execution_succeeded': False}}
    if case == 'partial_extension':
        current['component_execution'].update(status='implementation_failed', execution_succeeded=True)
        current['method_ref'] = refs.encode('method', sid, 'failed')
        current['method_implementation_review'] = {'verdict': 'partial',
            'method_ref': current['method_ref'], 'issues': ['Extension B was not implemented.']}
    method_ref, run_ref, result_ref = seed(store, sid=sid, mode='component', metric=None,
                                         key='failed', feedback=current)
    memory = {'best_result_ref': None if case == 'no_incumbent_extension_failed' else incumbent_ref,
              'best_metric': None if case == 'no_incumbent_extension_failed' else 2.0,
              'component_history': [history]}
    verification = {'verdict': 'verified', 'anti_cheat_flags': [], 'run_record_ref': run_ref,
        'run_record_digest': artifact_digest(store.get(run_ref)), 'candidate_snapshot_ref': result_ref}
    llm = LLM({**decision('component_history[0]'), 'add': [], 'remove': ['B'],
               'retain': ['A'], 'reuse': ['/saved/A'], 'requires_core_reconstruction': False})
    if review_mode == 'invalid':
        llm.response['add'] = ['another search component']
    handler = MemoryAgentHandler(spec_store=store, llm_client=llm)
    review_ref = await handler._review_complete_result(sid, run_ref, memory, verification,
        current['diagnosis'], baseline_ref, closeout_requested=True)
    review = store.get(review_ref) if review_ref else {}
    expected = case in {'better', 'higher', 'extension_failed',
                        'no_incumbent_extension_failed', 'partial_extension'}
    assert (review.get('trigger') == 'qualified_component_closeout') is expected
    assert memory['best_result_ref'] == (None if case == 'no_incumbent_extension_failed' else incumbent_ref)
    assert store.get(result_ref)['accepted'] is False
    assert store.get(old_result)['method_metric'] == metric
    if expected:
        assert review['qualified_closeout_components'][0]['checkpoint'] == '/saved/A'
        if review_mode == 'invalid':
            assert review['decision_source'] == 'host_closeout_policy'
            assert review['advisory_review_failure']['validation_failed'] is True
            assert review['decision']['add'] == []
            assert review['decision']['reuse'] == ['/saved/A']
            assert review['decision']['retain'] == [
                'The exact retained implementation at /saved/A and its frozen task constraints.']
        # Pre-fix closeout reviews lack the marker; replay must still reuse them.
        if case == 'better':
            review.pop('closeout_requested')
            store.put(kind='method_improvement_review', session_id=sid,
                key=refs.decode(review_ref).key, obj=review, emitter='MemoryAgent')
            assert await handler._review_complete_result(sid, run_ref, memory, verification,
                current['diagnosis'], baseline_ref, closeout_requested=True) == review_ref
            assert len(llm.calls) == (2 if review_mode == 'invalid' else 1)
        memory_ref = store.put(kind='research_memory', session_id=sid, key='routed',
                               obj=memory, emitter='MemoryAgent')
        engine = OrchestratorEngine(fake_bus, SM_ROOT, spec_store=store)
        await engine.start()
        state = SessionState(session_id=sid, mode='benchmark', current_state='ConsolidatingMemory',
            method_ref=method_ref, baseline_result_ref=baseline_ref,
            method_revision_round=_METHOD_REVISION_CAP - 1,
            component_revision_round=_COMPONENT_REVISION_CAP + 1,
            last_agent_failure={'underlying_reason_code': 'component_plan_audit_failed'})
        await engine._on_ConsolidatingMemory_memory_consolidated(state, SimpleNamespace(payload={
            'research_memory_ref': memory_ref, 'routing_verdict': 'need_experiment',
            'diagnosis': 'method_improvement_proposed', 'diagnosis_evidence': [review_ref]}))
        assert state.current_state == 'RevisingMethod'
        assert state.method_revision_round == _METHOD_REVISION_CAP - 1
        assert fake_bus.published[-1].payload['incremental_revision'] is True
        if case == 'better' and review_mode == 'valid':
            exhausted = SessionState(
                session_id=sid, mode='benchmark', current_state='ConsolidatingMemory',
                method_ref=method_ref, baseline_result_ref=baseline_ref,
                method_revision_round=_METHOD_REVISION_CAP,
                component_revision_round=_COMPONENT_REVISION_CAP,
            )
            await engine._on_ConsolidatingMemory_memory_consolidated(
                exhausted, SimpleNamespace(payload={
                    'research_memory_ref': memory_ref, 'routing_verdict': 'need_experiment',
                    'diagnosis': 'method_improvement_proposed',
                    'diagnosis_evidence': [review_ref]}))
            assert exhausted.current_state == 'RevisingMethod'
            assert exhausted.method_revision_round == _METHOD_REVISION_CAP


@pytest.mark.parametrize('problem', [None, 'partial_integration'])
@pytest.mark.parametrize('failure', ['protocol', 'provider_value_error', 'provider_timeout'])
@pytest.mark.parametrize('closing', [False, True])
async def test_closeout_policy_fallback_requires_validated_retained_evidence(
        store, enabled, fake_bus, problem, failure, closing):
    await seed_off_method(store, enabled, problem, 'failure')
    sid = 'retained-component'
    memory = store.get(refs.encode('research_memory', sid, 'v1'))
    original = store.get(memory['diagnosis_evidence'][0])
    candidate_before = store.get(original['result_ref'])
    llm = LLM({'route': 'revise_method', 'add': ['new mechanism']} if failure == 'protocol'
              else ValueError('provider configuration') if failure == 'provider_value_error'
              else TimeoutError('provider timeout'))
    handler = MemoryAgentHandler(spec_store=store, llm_client=llm)
    review_method_ref = store.put(kind='method', session_id=sid, key='review-target',
        obj={'name': 'Current frozen method', 'components': []}, emitter='MethodAgent')
    verification = next(store.get(ref) for ref in store.list_refs(sid, 'verification_report')
        if store.get(ref).get('candidate_snapshot_ref') == original['result_ref'])

    async def review():
        return await handler._review_complete_result(
            sid, candidate_before['run_record_snapshot_ref'], memory, verification,
            'method_description_reconciliation', original['baseline_ref'],
            review_method_ref=review_method_ref, closeout_requested=closing)

    if not closing or failure != 'protocol':
        with pytest.raises(RuntimeError):
            await review()
        saved = store.get(memory['method_feedback_refs'][-1])
        assert saved['status'] == 'failed'
        assert 'decision_source' not in saved
        return
    review_ref = await review()
    saved = store.get(review_ref)
    assert saved['decision_source'] == 'host_closeout_policy'
    assert saved['advisory_review_failure']['route'] == 'unavailable'
    assert saved['status'] == 'completed'
    assert saved['decision']['add'] == []
    assert saved['decision']['requires_core_reconstruction'] is False
    assert saved['decision']['reuse'] == [candidate_before['checkpoint']]
    assert memory['best_result_ref'] is None
    assert store.get(original['result_ref']) == candidate_before
    assert await review() == review_ref
    assert len(llm.calls) == 2
    memory_ref = store.put(kind='research_memory', session_id=sid, key='fallback',
                           obj=memory, emitter='MemoryAgent')
    engine = OrchestratorEngine(fake_bus, SM_ROOT, spec_store=store)
    await engine.start()
    state = SessionState(session_id=sid, mode='benchmark', current_state='ConsolidatingMemory',
        method_ref=saved['method_ref'], baseline_result_ref=original['baseline_ref'],
        component_revision_round=_COMPONENT_REVISION_CAP + 1,
        last_agent_failure={'underlying_reason_code': 'component_plan_audit_failed'})
    await engine._on_ConsolidatingMemory_memory_consolidated(state, SimpleNamespace(payload={
        'research_memory_ref': memory_ref, 'routing_verdict': 'need_experiment',
        'diagnosis': 'method_improvement_proposed', 'diagnosis_evidence': [review_ref]}))
    assert state.current_state == 'RevisingMethod'
    assert state.method_feedback_round == 0


async def test_source_closeout_cannot_discard_a_safe_revalidation_checkpoint(
        store, enabled):
    await seed_off_method(store, enabled, None, 'failure')
    sid = 'retained-component'
    memory = store.get(refs.encode('research_memory', sid, 'v1'))
    original = store.get(memory['diagnosis_evidence'][0])
    candidate = store.get(original['result_ref'])
    verification = next(store.get(ref) for ref in store.list_refs(sid, 'verification_report')
        if store.get(ref).get('candidate_snapshot_ref') == original['result_ref'])
    review_method_ref = store.put(kind='method', session_id=sid, key='review-target-discard',
        obj={'name': 'Current frozen method', 'components': []}, emitter='MethodAgent')
    llm = LLM({**decision(original['result_ref']),
        'retain': ['the slower incumbent'], 'modify': ['remove the measured source'],
        'add': [], 'reuse': ['/saved/incumbent'], 'requires_core_reconstruction': False})

    review_ref = await MemoryAgentHandler(
        spec_store=store, llm_client=llm)._review_complete_result(
            sid, candidate['run_record_snapshot_ref'], memory, verification,
            'method_description_reconciliation', original['baseline_ref'],
            review_method_ref=review_method_ref, closeout_requested=True)

    saved = store.get(review_ref)
    assert saved['decision_source'] == 'host_revalidation_closeout_policy'
    assert saved['advisory_review_failure']['reuse'] == ['/saved/incumbent']
    assert saved['decision']['reuse'] == [candidate['checkpoint']]
    assert candidate['checkpoint'] in saved['decision']['retain'][0]
    assert saved['decision']['add'] == []
    assert candidate['accepted'] is False


async def test_new_safe_candidate_is_not_hidden_by_prior_method_closeout(
        store, enabled):
    await seed_off_method(store, enabled, None, 'failure')
    sid = 'retained-component'
    memory = store.get(refs.encode('research_memory', sid, 'v1'))
    original = store.get(memory['diagnosis_evidence'][0])
    candidate = store.get(original['result_ref'])
    verification = next(store.get(ref) for ref in store.list_refs(sid, 'verification_report')
        if store.get(ref).get('candidate_snapshot_ref') == original['result_ref'])
    review_method_ref = store.put(kind='method', session_id=sid, key='terminal-method',
        obj={'name': 'Formal incumbent', 'components': []}, emitter='MethodAgent')
    prior = {
        'status': 'completed', 'trigger': 'component_method_reconciliation',
        'closeout_requested': True, 'method_ref': review_method_ref,
        'source_method_ref': review_method_ref, 'result_ref': 'spec://old/candidate',
        'decision': {**decision('old'), 'route': 'revise_method'},
    }
    store.put(kind='method_improvement_review', session_id=sid, key='prior-closeout',
              obj=prior, emitter='MemoryAgent')
    llm = LLM({**decision(original['result_ref']), 'add': [],
               'retain': [candidate['checkpoint']], 'reuse': [candidate['checkpoint']],
               'requires_core_reconstruction': False})

    review_ref = await MemoryAgentHandler(
        spec_store=store, llm_client=llm)._review_complete_result(
            sid, candidate['run_record_snapshot_ref'], memory, verification,
            'method_description_reconciliation', original['baseline_ref'],
            review_method_ref=review_method_ref, closeout_requested=True)

    assert review_ref
    assert store.get(review_ref)['result_ref'] == original['result_ref']
    assert llm.calls


async def test_closeout_retains_measured_integration_binding_over_indirect_only_components(
        store, enabled):
    await seed_off_method(store, enabled, 'partial_integration', 'failure')
    sid = 'retained-component'
    memory = store.get(refs.encode('research_memory', sid, 'v1'))
    original = store.get(memory['diagnosis_evidence'][0])
    candidate = store.get(original['result_ref'])
    _, component_run, _ = seed(store, sid=sid, key='indirect', mode='component', feedback={
        'baseline_digest': candidate['baseline_digest'], 'checkpoint': '/saved/indirect',
        'method_implementation_review': {'verdict': 'faithful'}})
    memory['component_history'].append({
        'component_id': 'indirect', 'mode': 'component', 'role': 'indirect_gain',
        'status': 'accepted', 'became_best': True, 'execution_succeeded': True,
        'contract_passed': True, 'verification_verdict': 'verified', 'mir_verdict': 'faithful',
        'baseline_digest': candidate['baseline_digest'], 'checkpoint': '/saved/indirect',
        'run_ref': component_run, 'metric': None})
    llm = LLM({**decision(original['result_ref']), 'add': [],
               'modify': ['Bind the retained measured implementation.']})
    review_ref = await MemoryAgentHandler(spec_store=store, llm_client=llm)._review_retained_off_method(
        sid, candidate['run_record_snapshot_ref'], memory, original['baseline_ref'],
        closeout_requested=True)
    review = store.get(review_ref)
    assert review['trigger'] == 'legal_off_method_gain'
    assert review['off_method_evidence']['candidate']['checkpoint'] == candidate['checkpoint']
    assert review['closeout_requested'] is True


@pytest.mark.parametrize('closing', [False, True])
async def test_retained_off_method_review_can_close_after_audit_exhaustion(
        store, enabled, fake_bus, closing):
    await seed_off_method(store, enabled, None, 'failure')
    sid = 'retained-component'
    memory = store.get(refs.encode('research_memory', sid, 'v1'))
    original = store.get(memory['diagnosis_evidence'][0])
    result = store.get(original['result_ref'])
    llm = LLM({**decision(original['result_ref']), 'add': [],
               'modify': ['Reconcile the Method with the retained implementation.'],
               'requires_core_reconstruction': False})
    handler = MemoryAgentHandler(spec_store=store, llm_client=llm)
    review_ref = await handler._review_retained_off_method(
        sid, result['run_record_snapshot_ref'], memory, original['baseline_ref'],
        closeout_requested=closing)
    review = store.get(review_ref)
    assert review['trigger'] == 'component_method_reconciliation'
    assert bool(review.get('closeout_requested')) is closing
    if closing:
        packet = json.loads(llm.calls[-1]['messages'][1]['content'])
        assert packet['evidence']['closeout_requested'] is True
        assert 'Do not add new search components' in llm.calls[-1]['messages'][0]['content']
    memory_ref = store.put(kind='research_memory', session_id=sid, key='routed',
                           obj=memory, emitter='MemoryAgent')
    engine = OrchestratorEngine(fake_bus, SM_ROOT, spec_store=store)
    await engine.start()
    state = SessionState(session_id=sid, mode='benchmark', current_state='ConsolidatingMemory',
        method_ref=review['method_ref'], baseline_result_ref=original['baseline_ref'],
        component_revision_round=_COMPONENT_REVISION_CAP + 1,
        last_agent_failure={'underlying_reason_code': 'component_plan_audit_failed'})
    await engine._on_ConsolidatingMemory_memory_consolidated(state, SimpleNamespace(payload={
        'research_memory_ref': memory_ref, 'routing_verdict': 'need_experiment',
        'diagnosis': 'method_improvement_proposed', 'diagnosis_evidence': [review_ref]}))
    assert state.current_state == ('RevisingMethod' if closing else 'Reporting')


@pytest.mark.parametrize('closing', [False, True])
@pytest.mark.parametrize('covered_by', ['current', 'incumbent'])
@pytest.mark.parametrize('case', [
    'equal', 'better_whole', 'better_retained', 'unaccepted', 'partial',
    'task_violation', 'unbound_method', 'missing_artifacts', 'metric_identity',
])
async def test_verified_whole_result_covers_retained_equal_or_worse_source(
        store, enabled, closing, covered_by, case):
    await seed_off_method(store, enabled, None, 'failure')
    sid = 'retained-component'
    memory = store.get(refs.encode('research_memory', sid, 'v1'))
    original = store.get(memory['diagnosis_evidence'][0])
    retained = store.get(original['result_ref'])
    metric = 1.0 if case == 'better_whole' else 3.0 if case == 'better_retained' else 2.0
    method_ref = refs.encode('method', sid, 'current-whole')
    attempt = 'attempt-current-whole'
    _, run_ref, result_ref = seed(store, sid=sid, key='current-whole', metric=metric, feedback={
        'accepted': case != 'unaccepted', 'method_ref': method_ref,
        'component_execution': None, 'checkpoint': '/saved/current-whole',
        'baseline_relative_diff': 'current-whole-source',
        'baseline_digest': retained['baseline_digest'],
        'metric_name': retained.get('metric_name'),
        'metric_unit': 'different-units' if case == 'metric_identity' else retained.get('metric_unit'),
        'measurement_evidence': {
            'valid': True, 'value': metric, 'attempt_id': attempt,
            'stdout_sha256': 'whole-output', 'protocol_fingerprint': 'official',
            'protocol': {'kind': 'official_single'},
        },
        'candidate_admissibility': {
            'verdict': 'insufficient_evidence' if case == 'missing_artifacts' else 'pass',
            'attempt_id': attempt,
        },
        'repository_comparison_review': {'verdict': 'pass'},
        'method_implementation_review': {
            'verdict': 'partial' if case == 'partial' else 'faithful',
            'method_ref': 'unrelated-method' if case == 'unbound_method' else method_ref,
            'task_contract_review': {'verdict': 'violation' if case == 'task_violation' else 'compliant'},
        },
    })
    llm = LLM({**decision(original['result_ref']), 'add': [],
               'modify': ['Reconcile the retained implementation.'],
               'requires_core_reconstruction': False})
    if covered_by == 'incumbent':
        memory.update(best_result_ref=result_ref, best_metric=metric)
        _, run_ref, _ = seed(store, sid=sid, key='failed-current', metric=3.2, feedback={
            'accepted': False, 'component_execution': None,
            'baseline_digest': retained['baseline_digest'],
            'baseline_relative_diff': 'failed-current-source',
            'method_implementation_review': {'verdict': 'drifted',
                'task_contract_review': {'verdict': 'compliant'},
                'issues': ['Current source does not implement the revised Method.']},
        })

    review_ref = await MemoryAgentHandler(
        spec_store=store, llm_client=llm)._review_retained_off_method(
            sid, run_ref, memory, original['baseline_ref'], closeout_requested=closing)

    expected = case not in {'equal', 'better_whole'}
    assert bool(review_ref) is expected
    assert len(llm.calls) == int(expected)
    assert store.get(result_ref)['method_metric'] == metric
    assert store.get(original['result_ref']) == retained


@pytest.mark.parametrize('status', ['qualified', 'accepted', 'improving'])
@pytest.mark.parametrize('closing', ['normal', 'budget', 'user'])
async def test_retained_history_does_not_preempt_healthy_component_progress(
        store, enabled, status, closing):
    await seed_off_method(store, enabled, None, 'failure')
    sid = 'retained-component'
    memory = store.get(refs.encode('research_memory', sid, 'v1'))
    original = store.get(memory['diagnosis_evidence'][0])
    _, run_ref, _ = seed(store, sid=sid, key='healthy', mode='component', feedback={
        'component_execution': {'mode': 'component', 'status': status,
            'execution_succeeded': True, 'attempt_contract_result': {'passed': True}}})
    if closing == 'user':
        store.put(kind='human_idea_seed', session_id=sid, key='closeout-v1',
                  obj={'closeout': True}, emitter='Human')
    llm = LLM({**decision(original['result_ref']), 'add': [],
               'modify': ['Reconcile the retained implementation.'],
               'requires_core_reconstruction': False})
    handler = MemoryAgentHandler(spec_store=store, llm_client=llm)

    review_ref = await handler._review_retained_off_method(
        sid, run_ref, memory, original['baseline_ref'],
        closeout_requested=closing == 'budget')

    assert bool(review_ref) is (closing != 'normal')
    assert len(llm.calls) == (0 if closing == 'normal' else 1)
    if review_ref:
        assert store.get(review_ref)['result_ref'] == original['result_ref']


@pytest.mark.parametrize('closing', ['normal', 'budget', 'user', 'exhausted'])
async def test_invalid_parent_repair_precedes_retained_history_until_closeout(
        store, enabled, fake_bus, closing):
    sid = 'invalid-parent'
    baseline = {'reproduced': True, 'metric_value': 3.0}
    baseline_ref = store.put(kind='baseline_result', session_id=sid, key='frozen',
                             obj=baseline, emitter='RunnerAgent')
    evidence = {'value': 2.0, 'valid': True, 'attempt_id': 'attempt-retained',
                'stdout_sha256': 'host-stdout', 'protocol': {'kind': 'component_local'}}
    _, old_run, old_result = seed(store, sid=sid, key='retained', mode='component', feedback={
        'accepted': False, 'verdict': 'invalid', 'baseline_relative_diff': 'retained-source',
        'baseline_digest': artifact_digest(baseline), 'measurement_evidence': evidence,
        'candidate_admissibility': {'verdict': 'insufficient_evidence', 'attempt_id': 'attempt-retained'},
        'repository_comparison_review': {'verdict': 'pass'},
        'method_implementation_review': {'verdict': 'drifted',
                                       'method_ref': refs.encode('method', sid, 'retained'),
                                       'task_contract_review': {'verdict': 'compliant'}},
        'component_execution': {'mode': 'component', 'role': 'direct_gain', 'status': 'drifted',
            'execution_succeeded': True, 'attempt_contract_result': {
                'passed': True, 'candidate_metric': 2.0, 'parent_metric': 3.0,
                'parent_evidence': {'host_bound': True}, 'measurement_evidence': evidence}}})
    evaluation = {'verdict': 'valid_not_best', 'run_record_ref': old_run,
                  'run_record_digest': artifact_digest(store.get(old_run))}
    evaluation_ref = store.put(kind='evaluation_report', session_id=sid,
        key='snapshot-' + artifact_digest(evaluation).split(':')[1], obj=evaluation, emitter='EvaluatorAgent')
    verification = {'verdict': 'verified', 'run_record_ref': old_run,
        'run_record_digest': artifact_digest(store.get(old_run)),
        'candidate_snapshot_ref': old_result, 'evaluation_report_ref': evaluation_ref}
    store.put(kind='verification_report', session_id=sid,
        key='snapshot-' + artifact_digest(verification).split(':')[1], obj=verification, emitter='EvaluatorAgent')
    method_ref, run_ref, result_ref = seed(store, sid=sid, key='current', mode='component', metric=1.0,
        feedback={'accepted': False, 'verdict': 'invalid', 'checkpoint': '/saved/current',
            'baseline_relative_diff': 'current-source',
            'component_execution': {'mode': 'component', 'role': 'direct_gain',
                'component_id': 'current', 'execution_succeeded': True,
                'status': 'exhausted' if closing == 'exhausted' else 'execution_failed',
                'attempt_contract_result': {'passed': False, 'parent_evidence': {
                    'host_bound': True, 'failed_gates': ['candidate_valid']}}}})
    memory = {'best_result_ref': None, 'best_metric': None, 'component_history': [],
              'reconciliation_candidates': [{'result_ref': old_result, 'checkpoint': '/saved/candidate'}]}
    if closing == 'user':
        store.put(kind='human_idea_seed', session_id=sid, key='closeout-v1',
                  obj={'closeout': True}, emitter='Human')
    llm = LLM({**decision(old_result), 'add': [],
               'modify': ['Reconcile the retained implementation.'],
               'requires_core_reconstruction': False})
    handler = MemoryAgentHandler(spec_store=store, llm_client=llm)

    review_ref = await handler._review_retained_off_method(
        sid, run_ref, memory, baseline_ref, closeout_requested=closing == 'budget')

    assert bool(review_ref) is (closing != 'normal')
    assert len(llm.calls) == (0 if closing == 'normal' else 1)
    assert store.get(result_ref)['accepted'] is False
    if closing == 'normal':
        memory_ref = store.put(kind='research_memory', session_id=sid, key='routed',
                               obj=memory, emitter='MemoryAgent')
        engine = OrchestratorEngine(fake_bus, SM_ROOT, spec_store=store)
        await engine.start()
        state = SessionState(session_id=sid, mode='benchmark', current_state='ConsolidatingMemory',
            method_ref=method_ref, baseline_result_ref=baseline_ref,
            current_experiment_id='same-experiment', run_record_refs=[run_ref],
            implementation_fix_round={'same-experiment': 2})
        await engine._on_ConsolidatingMemory_memory_consolidated(state, SimpleNamespace(payload={
            'research_memory_ref': memory_ref, 'routing_verdict': 'need_experiment',
            'diagnosis': 'component_contract_retry'}))
        assert state.current_state == 'MossIntervening'
        assert state.implementation_fix_round['same-experiment'] == 3
