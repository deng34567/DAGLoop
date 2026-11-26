"""Fault injection across real save/delivery boundaries, with no network or training."""
import asyncio
import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import aiosqlite
import pytest

from autoresearch.agents._reuse.runner_backend import FakeRunnerBackend
from autoresearch.agents.runner_agent.handler import RunnerAgentHandler
from autoresearch.agents.report_agent.handler import ReportAgentHandler
from autoresearch.agents.report_agent.tests.test_handler import _report
from autoresearch.agents.runner_agent.tests.test_functional import VALID_EXPERIMENT_PATCH, VALID_VERIFIER_SPEC
from autoresearch.experiments.repo_improve.runtime_evidence import comparison_epsilon
from autoresearch.orchestrator.engine import OrchestratorEngine
from autoresearch.orchestrator.session_state import SessionState, load_session_state, save_session_state
from autoresearch.spec_store import SpecStore, refs
from autoresearch.spec_store.identity import artifact_digest
from autoresearch.testing.delivering_bus import DeliveringBus

ROOT = Path(__file__).resolve().parents[2] / 'orchestrator' / 'state-machine'


@pytest.mark.parametrize('failure', ['run_record', 'experiment_completion', 'returned_receipt'])
@pytest.mark.parametrize('restart', [False, True])
async def test_saved_execution_is_not_repeated_after_save_failure(tmp_path, monkeypatch, failure, restart):
    store = SpecStore(tmp_path / 'store')
    backend = FakeRunnerBackend(metric_value=8)
    handler = RunnerAgentHandler(spec_store=store, backend=backend)
    patch = store.put(kind='experiment_patch', session_id='s', key='v1', obj=VALID_EXPERIMENT_PATCH, emitter='CoderAgent')
    verifier = store.put(kind='verifier_spec', session_id='s', key='v1', obj=VALID_VERIFIER_SPEC, emitter='EvaluatorAgent')
    env = SimpleNamespace(session_id='s', event_id='original', payload={'experiment_patch_ref':patch, 'verifier_spec_ref':verifier})
    calls = []
    from autoresearch.agents.runner_agent import core
    execute = core.run_experiment
    def counted(*args, **kwargs):
        calls.append(1)
        return execute(*args, **kwargs)
    monkeypatch.setattr(core, 'run_experiment', counted)
    real_put = store.put
    fired = []
    def put(**kw):
        match = (kw['kind'] == failure or failure == 'returned_receipt' and kw['kind'] == 'audit' and kw['key'].startswith('runner-returned-'))
        if match and not fired:
            fired.append(1)
            raise sqlite3.OperationalError('database is locked')
        return real_put(**kw)
    monkeypatch.setattr(store, 'put', put)
    async with aiosqlite.connect(':memory:') as db:
        bus = DeliveringBus(db)
        with pytest.raises(sqlite3.OperationalError):
            await handler.handle_run_requested(bus.view('RunnerAgent'), env)
        if restart:
            handler = RunnerAgentHandler(spec_store=store, backend=backend)
        await handler.handle_run_requested(bus.view('RunnerAgent'), env)
        assert calls == [1]
        if restart and failure == 'returned_receipt':
            assert bus.published[-1].event_type == 'agent.failed'
            assert bus.published[-1].payload['underlying_reason_code'] == 'execution_result_unconfirmed'
        else:
            assert bus.published[-1].event_type == 'experiment.run_completed'
            assert store.get(bus.published[-1].payload['run_record_ref'])['metric_value'] == 8
    store.close()


@pytest.mark.parametrize('broken', [None, 'missing_package', 'invalid_measurement', 'component_only', 'missing_result', 'missing_run', 'missing_verification', 'native_bundle'])
async def test_real_report_delivery_verifies_done_and_allows_valid_negative(tmp_path, monkeypatch, broken):
    store = SpecStore(tmp_path / 'store')
    base = store.put(kind='baseline_result', session_id='s', key='v1', emitter='RunnerAgent',
                     obj={'repo_url':'offline://source', 'repo_commit':'fixed-base'})
    record = {'succeeded':True, 'metric_value':11}
    run_ref = store.put(kind='run_record', session_id='s', key='snapshot-'+artifact_digest(record).split(':')[1], obj=record, emitter='RunnerAgent')
    verification_ref = store.put(kind='verification_report', session_id='s', key='v1', emitter='EvaluatorAgent',
        obj={'verdict':'verified', 'run_record_ref':run_ref, 'run_record_digest':artifact_digest(record)})
    result = {'run_record_snapshot_ref':run_ref, 'accepted':False, 'verdict':'valid_not_best', 'method_metric':11, 'baseline_metric':10,
              'lower_is_better':True, 'baseline_relative_diff':'+valid implementation',
              'baseline_result_ref':base, 'checkpoint':'saved-source',
              'measurement_evidence':{'valid':True},
              'component_execution':{'mode':'integration', 'all_components_accepted':True},
              'method_implementation_review':{'verdict':'faithful','task_contract_review':{'verdict':'compliant'}}}
    if broken == 'native_bundle':
        result.pop('baseline_relative_diff')
        result['delivery_bundle'] = {'files': {'method.py':'def solve(x): return x', 'README.md':'Saved actual source'}}
    if broken == 'invalid_measurement': result['measurement_evidence']['valid'] = False
    if broken == 'component_only': result['component_execution']['mode'] = 'component'
    result_ref = store.put(kind='gen_experiment_result', session_id='s', emitter='RunnerAgent',
        key='snapshot-'+artifact_digest(result).split(':')[1], obj=result)
    memory = store.put(kind='research_memory', session_id='s', key='v1', emitter='MemoryAgent',
                       obj={'best_result_ref':result_ref, 'best_metric':11, 'negative_results':['Valid non-improvement']})
    monkeypatch.setattr('autoresearch.agents.report_agent.core.generate_report', lambda *args, **kwargs: _report())
    async with aiosqlite.connect(':memory:') as db:
        bus = DeliveringBus(db)
        engine = OrchestratorEngine(bus.view('ResearchOrchestrator'), ROOT, spec_store=store)
        await engine.start()
        bus.subscriptions.clear()  # Deliver a deliberately altered packet before Engine consumes it.
        state = SessionState(session_id='s', current_state='Reporting', research_memory_ref=memory,
                             mode='benchmark', request_binding_required=True,
                             run_record_refs=[] if broken == 'missing_run' else [run_ref],
                             verification_report_refs=[] if broken == 'missing_verification' else [verification_ref])
        report_handler = ReportAgentHandler(spec_store=store, llm_client=object())
        # A real ReportAgent request, followed by duplicate completion delivery.
        await engine._publish_request(state, 'report.requested', {'research_memory_ref':memory})
        await db.commit()
        request = bus.published[-1]
        await report_handler.handle_report_requested(bus.view('ReportAgent'), request)
        ready = bus.published[-1]
        if broken in {'missing_package','missing_result'}:
            report = store.get(ready.payload['report_ref'])
            report.pop('delivery_bundle_ref' if broken == 'missing_package' else 'result_ref')
            ready.payload = {'report_ref':store.put(kind='report', session_id='s', emitter='ReportAgent',
                             key='snapshot-'+artifact_digest(report).split(':')[1], obj=report)}
        await engine.tick(ready)
        saved = await load_session_state(db, 's')
        assert (saved.current_state == 'Done') is (broken in {None, 'native_bundle'})
        assert saved.completion_verified is (broken in {None, 'native_bundle'})
        if broken in {None, 'native_bundle'}:
            frozen_report = store.get(saved.final_report_ref)
            bundle = store.get(frozen_report['delivery_bundle_ref'])
            assert bundle['result_ref'] == result_ref
            if broken == 'native_bundle':
                assert bundle['files'] == result['delivery_bundle']['files']
            else:
                assert bundle['files']['candidate.patch'] == '+valid implementation'
            before = saved.to_json()
            await engine.tick(ready)
            assert (await load_session_state(db,'s')).to_json() == before
            await report_handler.handle_report_requested(bus.view('ReportAgent'), request)
            assert bus.published[-1].payload['report_ref'] == saved.final_report_ref
    store.close()


def test_only_host_bound_samples_affect_noise_tolerance():
    unknown = {'method_metric':100, 'method_std':30, 'method_runs':[70,130]}
    assert comparison_epsilon(unknown) == 0
    measured = {**unknown, 'measurement_evidence':{'stdout_sha256':'host-output', 'samples':[70,130]}}
    assert comparison_epsilon(measured) > 40
    assert comparison_epsilon({'method_metric':100, 'measurement_evidence':{'protocol':{'kind':'deterministic'}}}) == 0


async def test_duplicate_failure_decision_does_not_spend_twice(tmp_path):
    store = SpecStore(tmp_path/'store')
    async with aiosqlite.connect(':memory:') as db:
        bus = DeliveringBus(db)
        engine = OrchestratorEngine(bus.view('ResearchOrchestrator'), ROOT, spec_store=store)
        await engine.start()
        state = SessionState(session_id='s', current_state='RunningExperiment', request_binding_required=True,
                             active_request={'event_id':'request', 'state':'RunningExperiment', 'payload':{}, 'consumed':False})
        await save_session_state(db,state,auto_commit=True)
        failure = dict(session_id='s',event_type='agent.failed', emitter='RunnerAgent',causation_id='request',
                       payload={'agent_name':'RunnerAgent','reason_classification':'permanent', 'underlying_reason':'cannot execute'})
        await engine.tick(SimpleNamespace(event_id='failure-a',**failure))
        before = (await load_session_state(db,'s')).to_json()
        await engine.tick(SimpleNamespace(event_id='failure-b',**failure))
        assert (await load_session_state(db,'s')).to_json() == before
    store.close()


async def test_web_job_restores_saved_handoff_with_watchdog_off(tmp_path, monkeypatch):
    from autoresearch.web.server import Engine
    from autoresearch.agents.runner_agent import core
    monkeypatch.setenv('AUTORESEARCH_WATCHDOG','0')
    store = SpecStore(tmp_path/'store')
    backend = FakeRunnerBackend(metric_value=8)
    handler = RunnerAgentHandler(spec_store=store, backend=backend)
    patch = store.put(kind='experiment_patch',session_id='s',key='v1',obj=VALID_EXPERIMENT_PATCH,emitter='CoderAgent')
    verifier = store.put(kind='verifier_spec',session_id='s',key='v1',obj=VALID_VERIFIER_SPEC,emitter='EvaluatorAgent')
    calls=[]
    execute=core.run_experiment
    def counted(*args,**kw):
        calls.append(1)
        return execute(*args,**kw)
    monkeypatch.setattr(core,'run_experiment',counted)
    real_put=store.put
    failed=[]
    def put(**kw):
        if kw['kind']=='run_record' and not failed:
            failed.append(1)
            raise sqlite3.OperationalError('database is locked')
        return real_put(**kw)
    monkeypatch.setattr(store,'put',put)
    async with aiosqlite.connect(':memory:') as db:
        bus=DeliveringBus(db)
        handler.register(bus.view('RunnerAgent'))
        worker=Engine(str(tmp_path/'web.db'))
        worker.loop=asyncio.get_running_loop()
        worker.session_id='s'
        worker.bus=bus
        future=worker.submit(lambda: bus.publish(event_type='experiment.run_requested', session_id='s',
            payload={'experiment_patch_ref':patch,'verifier_spec_ref':verifier},emitter='ResearchOrchestrator'))
        await asyncio.wrap_future(future)
        assert calls == [1]
        assert bus.published[-1].event_type == 'experiment.run_completed'
        assert not worker._watchdog_is_busy('RunningExperiment')
    store.close()


async def test_slow_startup_keeps_original_worker_and_id(tmp_path, monkeypatch):
    from autoresearch.web import server
    pool=server.EnginePool(1)
    worker=pool.workers[0]
    scheduled=[]
    async def slow_start(*args,**kwargs): return 'late-session'
    monkeypatch.setattr(worker,'eng',SimpleNamespace(start_session=slow_start),raising=False)
    monkeypatch.setattr(worker,'submit',lambda factory: scheduled.append(factory))
    monkeypatch.setattr(server,'ENGINE',pool)
    monkeypatch.setattr(server,'_START_REQUESTS',{})
    monkeypatch.setattr(server,'_record_topic',lambda *args: None)
    async def fast_sleep(*args): pass
    monkeypatch.setattr(server.asyncio,'sleep',fast_sleep)
    response=await server.start_session({'topic':'same task','request_id':'start-original'})
    assert response.status_code == 202
    assert pool.claim_worker() is None
    again=await server.start_session({'topic':'same task','request_id':'start-original'})
    assert again.status_code == 202 and len(scheduled)==1
    await scheduled[0]()
    result=server.start_status('start-original')
    assert result['session_id']=='late-session'
    assert worker.session_id=='late-session'


@pytest.mark.parametrize('quota', [False, True])
async def test_review_retry_or_quota_resume_has_new_delivery_with_same_inputs(tmp_path, quota):
    store = SpecStore(tmp_path / 'store')
    method = store.put(kind='method', session_id='s', key='v1', obj={'approach':'fixed'}, emitter='MethodAgent')
    async with aiosqlite.connect(':memory:') as db:
        bus = DeliveringBus(db)
        engine = OrchestratorEngine(bus.view('ResearchOrchestrator'), ROOT, spec_store=store)
        await engine.start()
        bus.subscriptions.clear()
        state = SessionState(session_id='s', current_state='RunningExperiment', request_binding_required=True, method_ref=method)
        await engine._publish_request(state, 'experiment.run_requested', {'method_ref':method})
        await db.commit()
        original = bus.published[-1]
        failure = SimpleNamespace(session_id='s', event_type='agent.failed', emitter='RunnerAgent',
            event_id='review-failed', causation_id=original.event_id, payload={
                'agent_name':'RunnerAgent', 'reason_classification':'transient',
                'underlying_reason_code':'insufficient_quota' if quota else 'implementation_review_unavailable',
                'underlying_reason':'insufficient_quota' if quota else 'JSON response was incomplete'})
        await engine.tick(failure)
        if quota:
            assert (await load_session_state(db,'s')).current_state == 'Suspended'
            engine._entry_actions['RunningExperiment'] = ('experiment.run_requested', lambda state, message: original.payload)
            await engine.resume_to_current_state('s')
        assert bus.published[-1].event_type == 'experiment.run_requested'
        assert bus.published[-1].event_id != original.event_id
        assert bus.published[-1].payload == original.payload
        assert (await load_session_state(db,'s')).current_state == 'RunningExperiment'
    store.close()


async def test_matching_request_cannot_accept_unbound_completion(tmp_path):
    store = SpecStore(tmp_path / 'store')
    run_ref = store.put(kind='run_record',session_id='s', key='v1', obj={'succeeded':True}, emitter='RunnerAgent')
    async with aiosqlite.connect(':memory:') as db:
        bus = DeliveringBus(db)
        engine = OrchestratorEngine(bus.view('ResearchOrchestrator'), ROOT, spec_store=store)
        await engine.start()
        state = SessionState(session_id='s', current_state='RunningExperiment', request_binding_required=True,
            active_request={'event_id':'original', 'event_type':'experiment.run_requested', 'state':'RunningExperiment', 'payload':{}, 'consumed':False})
        await save_session_state(db,state,auto_commit=True)
        await engine.tick(SimpleNamespace(session_id='s',event_type='experiment.run_completed',emitter='RunnerAgent',
            event_id='completion',causation_id='original',payload={'run_record_ref':run_ref,'verdict':'succeeded'}))
        saved = await load_session_state(db,'s')
        assert saved.current_state == 'RunningExperiment'
        assert saved.run_record_refs == [] and not saved.active_request['consumed']
    store.close()


async def test_ack_failure_after_committed_decision_does_not_repeat_effect(tmp_path, monkeypatch):
    store = SpecStore(tmp_path/'store')
    async with aiosqlite.connect(':memory:') as db:
        bus = DeliveringBus(db)
        engine = OrchestratorEngine(bus.view('ResearchOrchestrator'), ROOT, spec_store=store)
        await engine.start()
        state = SessionState(session_id='s', current_state='RunningExperiment', request_binding_required=True,
            active_request={'event_id':'original','state':'RunningExperiment','payload':{},'consumed':False})
        await save_session_state(db,state,auto_commit=True)
        execute = db.execute
        fault = []
        def inject(sql, *args, **kw):
            if sql.startswith('UPDATE web_pending_delivery') and not fault:
                fault.append(1)
                raise sqlite3.OperationalError('database is locked')
            return execute(sql,*args,**kw)
        monkeypatch.setattr(db,'execute',inject)
        with pytest.raises(sqlite3.OperationalError):
            await bus.publish(event_type='agent.failed',session_id='s',emitter='RunnerAgent',causation_id='original',
                payload={'agent_name':'RunnerAgent','reason_classification':'permanent','underlying_reason':'missing input'})
        before = (await load_session_state(db,'s')).to_json()
        assert await bus.replay_pending('s')
        assert (await load_session_state(db,'s')).to_json() == before
    store.close()
