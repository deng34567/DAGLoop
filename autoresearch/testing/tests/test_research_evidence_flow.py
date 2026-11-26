"""Offline regressions for candidate evidence through Runner, Memory and routing."""
import pytest

from autoresearch.agents.memory_agent.handler import (
    _append_component_history, _classify_component_evidence,
)
from autoresearch.agents.report_agent.handler import _apply_latest_verified_structured_result
from autoresearch.agents.runner_agent.tests.test_method_directive import _handler
from autoresearch.experiments.repo_improve.component_plan import ComponentLedger, build_component_plan
from autoresearch.experiments.repo_improve.search_memory import board_for_session
from autoresearch.experiments.repo_improve.tests.test_component_plan import _component
from autoresearch.orchestrator.engine import OrchestratorEngine
from autoresearch.orchestrator.session_state import load_session_state
from autoresearch.orchestrator.tests.unit.test_engine_consolidation_advance import (
    SM_ROOT, _Evt, _seed_consolidating_memory_session,
)
from autoresearch.orchestrator.tests.unit.test_engine_min import fake_bus  # noqa: F401


def _result(value=80.0, **overrides):
    return {
        'accepted': True, 'verdict': 'new_best', 'method_metric': value,
        'baseline_metric': 100.0, 'metric_name': 'runtime_s', 'lower_is_better': True,
        'method_implementation_review': {'verdict': 'faithful'}, **overrides,
    }


def test_missing_task_metric_stays_missing_in_search(tmp_path, monkeypatch):
    monkeypatch.setenv('AUTORESEARCH_WEB_WORKDIR', str(tmp_path))
    h = _handler({})
    h._record_experience_card('metrics', _result(None))
    assert board_for_session('metrics').cards()[0].metric_value is None


def test_indirect_history_keeps_task_metric_and_check_separate():
    gen = _result(80.0, component_execution={
        'mode': 'component', 'role': 'indirect_gain', 'status': 'qualified',
        'execution_succeeded': True, 'component_id': 'check',
        'contract_result': {'passed': True, 'measurement': 'indirect_gain_check', 'actual': 0.0},
    })
    item = _append_component_history({}, gen, 'spec://run_record/s/one', {})[0]
    assert item['metric'] == 80.0
    assert item['acceptance_result']['actual'] == 0.0


def test_parent_metric_keeps_original_comparison_identity(tmp_path, monkeypatch):
    monkeypatch.setenv('AUTORESEARCH_WEB_WORKDIR', str(tmp_path))
    monkeypatch.setenv('AUTORESEARCH_AGENTIC_OPERATORS', '0')
    h = _handler({'components': [_component('transform')]})
    h._with_operator('parent', 'run')
    h._finalize_component_attempt('parent', _result(repo_path='/checkpoint/first'))
    h._with_operator('parent', 'improve')
    result = h._finalize_component_attempt('parent', _result(70.0, repo_path='/checkpoint/second'))
    execution = result['component_execution']
    assert execution['attempt_contract_result']['parent_metric'] == 100.0
    assert execution['parent_checkpoint'] == 'baseline'


@pytest.mark.parametrize('failure_kind', ['transport', 'missing_parent', 'drifted', 'negative'])
@pytest.mark.parametrize('attempt_limit', [6, 8])
def test_exhaustion_requires_a_valid_negative_measurement(tmp_path, monkeypatch, failure_kind, attempt_limit):
    monkeypatch.setenv('AUTORESEARCH_COMPONENT_MAX_ATTEMPTS', str(attempt_limit))
    h = _handler({})
    plan = build_component_plan({'components': [_component('transform')]})
    ledger = ComponentLedger(str(tmp_path/'ledger.json'), plan)
    ledger.begin('transform')
    h._active_component_context = {'flow': {'mode': 'component', 'ledger': ledger,
                                           'node': plan.node('transform'), 'plan': plan}}
    result = {}
    for _ in range(attempt_limit):
        if failure_kind == 'transport':
            result = _result(None, component_execution=h._record_component_failure('flow', 'transport failed'))
        else:
            raw = _result(110.0 if failure_kind == 'negative' else 80.0,
                          repo_path='/checkpoint/candidate')
            if failure_kind == 'missing_parent': raw['baseline_metric'] = None
            if failure_kind == 'drifted': raw['method_implementation_review'] = {'verdict': 'drifted'}
            result = h._finalize_component_attempt('flow', raw)
    entry = ledger.entry('transform')
    assert entry.attempts + entry.infrastructure_attempts == attempt_limit
    assert entry.qualified_attempts == 0
    if failure_kind == 'transport':
        assert entry.attempts == 0 and entry.last_outcome == 'infrastructure_exhausted'
        assert entry.status == 'active'  # No experiment falsified the component design.
    else:
        assert entry.attempts == attempt_limit and entry.status == 'exhausted'
    diagnosis, _ = _classify_component_evidence(result)
    expected = 'component_expectation_failed' if failure_kind == 'negative' else 'component_execution_blocked'
    assert diagnosis == expected


async def test_exhausted_execution_reports_without_scientific_revision(fake_bus):
    eng = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
    await eng.start()
    await _seed_consolidating_memory_session(eng, 'blocked')
    await eng.tick(_Evt('memory.consolidated', 'blocked', 'MemoryAgent', {
        'research_memory_ref': 'spec://research_memory/blocked/v1',
        'routing_verdict': 'report', 'diagnosis': 'component_execution_blocked',
        'diagnosis_evidence': ['execution budget exhausted before a valid comparison'],
    }))
    state = await load_session_state(fake_bus.db, 'blocked')
    assert state.current_state == 'Reporting'
    assert state.component_revision_round == 0
    assert state.method_swap_round == 0
    assert not state.component_ablation_failed


def test_report_does_not_certify_an_unbound_historical_number():
    report = _apply_latest_verified_structured_result({}, {
        'diagnosis': 'valid_improvement', 'best_metric': 60.0,
    }, _result())
    assert 'method=80.0' in report['summary']
    assert 'method=60.0' not in report['summary']
    assert any('historical' in caveat.lower() for caveat in report['analysis_caveats'])


def test_report_selects_complete_verified_candidate_not_mixed_fields():
    prior = _result(60.0, baseline_metric=120.0,
                    method_implementation_review={'verdict': 'partial'})
    report = _apply_latest_verified_structured_result({}, {
        'diagnosis': 'valid_improvement', 'best_metric': 60.0,
    }, _result(), best_result=prior)
    assert 'method=60.0 beats baseline=120.0' in report['summary']
    assert 'partial implementation' in report['summary']
    assert 'latest accepted attempt measured 80.0' in report['summary'].lower()


def test_report_retains_verified_history_after_current_execution_failure():
    report = _apply_latest_verified_structured_result({}, {
        'diagnosis': 'execution_failure', 'best_metric': 60.0,
    }, _result(None, accepted=False, verdict='invalid'), best_result=_result(60.0))
    assert 'method=60.0' in report['summary']
    assert any('latest' in s.lower() for s in report['analysis_caveats'])


def _store_attempt(store, sid, value, diff, *, execution=None):
    """Persist the same content-addressed evidence objects as the Runner."""
    from autoresearch.spec_store.identity import artifact_digest
    record = {'succeeded': value is not None, 'exit_code': 0 if value is not None else 1,
              'metric_value': value, 'stdout_excerpt': str(value), 'environment': 'fake',
              'artifact_refs': [], 'stderr_excerpt': None}
    run = store.put(kind='run_record', session_id=sid, key='v1', obj=record, emitter='RunnerAgent')
    snapshot = store.put(kind='run_record', session_id=sid,
                         key='snapshot-' + artifact_digest(record).split(':', 1)[1],
                         obj=record, emitter='RunnerAgent')
    gen = _result(value, accepted=value is not None, isolated=True, diff=diff,
                  run_record_snapshot_ref=snapshot,
                  verdict='new_best' if value is not None else 'invalid', reasons=[])
    if execution is not None: gen['component_execution'] = execution
    candidate = store.put(kind='gen_experiment_result', session_id=sid,
                          key='snapshot-' + artifact_digest(gen).split(':', 1)[1],
                          obj=gen, emitter='RunnerAgent')
    gen['candidate_snapshot_ref'] = candidate
    store.put(kind='gen_experiment_result', session_id=sid, key='current', obj=gen, emitter='RunnerAgent')
    return run, candidate


@pytest.mark.parametrize('same_candidate', [False, True])
def test_memory_binds_candidate_and_scopes_verification_failure(tmp_path, same_candidate):
    from autoresearch.agents.memory_agent.handler import MemoryAgentHandler
    from autoresearch.spec_store.store import SpecStore
    with_store = SpecStore(tmp_path/'store')
    try:
        h = MemoryAgentHandler(spec_store=with_store, llm_client=None)
        run, candidate = _store_attempt(with_store, 'history', 60.0, '+source A')
        prior, *_ = h._consolidate_gen(run, verification_report={'verdict': 'verified'})
        assert prior['best_result_ref'] == candidate
        assert prior['best_metric'] == 60.0
        # A new attempt of the same candidate can invalidate that claim. A failed
        # different source must not delete the already verified candidate.
        failed, _ = _store_attempt(with_store, 'history', None,
                                    '+source A' if same_candidate else '+source B')
        current, *_ = h._consolidate_gen(failed, prior_research_memory=prior,
                                         verification_report={'verdict': 'failed'})
        assert current['best_metric'] == (None if same_candidate else 60.0)
        assert current['best_result_ref'] == (None if same_candidate else candidate)
    finally:
        with_store.close()


async def test_offline_failure_workflow_preserves_prior_result_and_reports(tmp_path, fake_bus, monkeypatch):
    """Runner failure -> Memory classification -> Engine reporting -> Report evidence selection."""
    import json
    import uuid
    from autoresearch.agents.memory_agent.handler import MemoryAgentHandler
    from autoresearch.agents.report_agent.handler import ReportAgentHandler
    from autoresearch.agents.report_agent.tests.test_handler import (
        FakeBusClient, FakeLLM, _report, _make_envelope, _seed_report_inputs,
    )
    from autoresearch.spec_store.store import SpecStore
    monkeypatch.setenv('AUTORESEARCH_COMPONENT_MAX_ATTEMPTS', '6')
    sid = str(uuid.uuid4())
    store = SpecStore(tmp_path/'workflow-store')
    try:
        memory_handler = MemoryAgentHandler(spec_store=store, llm_client=None)
        first_run, candidate = _store_attempt(store, sid, 60.0, '+working source')
        prior, *_ = memory_handler._consolidate_gen(first_run, verification_report={'verdict': 'verified'})
        plan = build_component_plan({'components': [_component('transform')]})
        ledger = ComponentLedger(str(tmp_path/'workflow-ledger.json'), plan)
        ledger.begin('transform')
        runner = _handler({})
        runner._active_component_context = {sid: {'mode': 'component', 'plan': plan,
            'ledger': ledger, 'node': plan.node('transform')}}
        for _ in range(6):
            execution = runner._record_component_failure(sid, 'transport failed')
        failed_run, _ = _store_attempt(store, sid, None, '+different source', execution=execution)
        memory, routing, diagnosis, evidence = memory_handler._consolidate_gen(
            failed_run, prior_research_memory=prior,
            verification_report={'verdict': 'verified', 'checks': ['failure_confirmed']})
        assert diagnosis == 'component_execution_blocked' and routing == 'report'
        assert memory['best_result_ref'] == candidate
        engine = OrchestratorEngine(client=fake_bus, skill_root=SM_ROOT)
        await engine.start()
        await _seed_consolidating_memory_session(engine, sid)
        payload = _seed_report_inputs(store, sid, research_memory=memory)
        await engine.tick(_Evt('memory.consolidated', sid, 'MemoryAgent', {
            'research_memory_ref': payload['research_memory_ref'], 'routing_verdict': routing,
            'diagnosis': diagnosis, 'diagnosis_evidence': evidence,
        }))
        state = await load_session_state(fake_bus.db, sid)
        assert state.current_state == 'Reporting'
        assert state.component_revision_round == 0 and state.method_swap_round == 0
        client = FakeBusClient()
        report_handler = ReportAgentHandler(spec_store=store, llm_client=FakeLLM([
            json.dumps(_report(summary='Experimental results.'))]))
        await report_handler.handle_report_requested(client,
            _make_envelope('report.requested', sid, payload, 'ResearchOrchestrator'))
        ready = [e for e in client.published if e.event_type == 'report.ready']
        assert len(ready) == 1
        report = store.get(ready[0].payload['report_ref'])
        assert 'method=60.0' in report['summary']
        assert 'baseline=100.0' in report['summary']
        assert any('latest attempt' in x.lower() for x in report['analysis_caveats'])
        assert ledger.entry('transform').attempts == 0
        assert ledger.entry('transform').infrastructure_attempts == 6
        assert ledger.entry('transform').last_outcome == 'infrastructure_exhausted'
    finally:
        store.close()
