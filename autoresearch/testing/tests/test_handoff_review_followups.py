"""Reproduce live-owner, mutable-output and accepted-budget handoff gaps."""
import asyncio
import sqlite3
import threading
from pathlib import Path
from types import SimpleNamespace

import aiosqlite
import pytest

from autoresearch.agents._reuse.runner_backend import FakeRunnerBackend
from autoresearch.agents.runner_agent import core as runner_core
from autoresearch.agents.runner_agent.handler import RunnerAgentHandler
from autoresearch.agents.runner_agent.tests.test_functional import VALID_EXPERIMENT_PATCH, VALID_VERIFIER_SPEC
from autoresearch.agents.experiment_planner_agent.handler import ExperimentPlannerAgentHandler
from autoresearch.agents.experiment_planner_agent.tests.test_handler import _seed_planning_inputs, _valid_plan
from autoresearch.agents.coder_agent.handler import CoderAgentHandler
from autoresearch.agents.critic_agent.handler import CriticAgentHandler
from autoresearch.agents.critic_agent.tests.test_experiment_plan_review import VALID_EXPERIMENT_PLAN_REPORT_PASS
from autoresearch.orchestrator.engine import OrchestratorEngine
from autoresearch.orchestrator.session_state import SessionState, load_session_state
from autoresearch.spec_store import SpecStore
from autoresearch.testing.delivering_bus import DeliveringBus

ROOT = Path(__file__).resolve().parents[2] / 'orchestrator/state-machine'


@pytest.mark.parametrize('branch', ['experiment', 'baseline'])
@pytest.mark.parametrize('cancel_duplicate', [False, True])
async def test_duplicate_live_execution_does_not_fail_or_block_completion(tmp_path, monkeypatch, branch, cancel_duplicate):
    store = SpecStore(tmp_path / 'store')
    backend = FakeRunnerBackend(metric_value=8)
    handler = RunnerAgentHandler(spec_store=store, backend=backend)
    patch = store.put(kind='experiment_patch', session_id='s', key='v1', obj=VALID_EXPERIMENT_PATCH, emitter='CoderAgent')
    verifier = store.put(kind='verifier_spec', session_id='s', key='v1', obj=VALID_VERIFIER_SPEC, emitter='EvaluatorAgent')
    entered, release = threading.Event(), threading.Event()
    calls = []
    def execute(*args, **kwargs):
        calls.append(1)
        entered.set()
        assert release.wait(5)
        record = runner_core.run_experiment(VALID_EXPERIMENT_PATCH, VALID_VERIFIER_SPEC, backend)
        return record, ({'baseline_metric': 8, 'baseline_valid': True, 'repo_commit': 'fixed'}
                        if branch == 'baseline' else None)
    monkeypatch.setattr(handler, '_gen_branch', lambda: True)
    monkeypatch.setattr(handler, '_run_gen_experiment', execute)
    monkeypatch.setattr(handler, '_resolve_repo_url', lambda *a: 'offline://source')
    monkeypatch.setattr(handler, '_run_repo_phase', execute)
    monkeypatch.setattr(handler, '_original_question', lambda *a: 'offline test question')
    async with aiosqlite.connect(':memory:') as db:
        bus = DeliveringBus(db)
        engine = OrchestratorEngine(bus.view('ResearchOrchestrator'), ROOT, spec_store=store)
        await engine.start()
        state = SessionState(session_id='s', current_state='RunningExperiment' if branch == 'experiment' else 'ReproducingBaseline',
                             request_binding_required=True)
        payload = {'experiment_patch_ref': patch, 'verifier_spec_ref': verifier}
        method = handler.handle_run_requested
        event_type = 'experiment.run_requested'
        if branch == 'baseline':
            benchmark = store.put(kind='benchmark_spec', session_id='s', key='v1',
                                  obj={'benchmark_id': 'bench', 'metric': 'metric'}, emitter='EvaluatorAgent')
            state.current_benchmark_id = 'bench'
            state.confirmed_repo_url = 'offline://source'
            payload = {'benchmark_spec_ref': benchmark, 'verifier_spec_ref': verifier,
                       'benchmark_id': 'bench', 'confirmed_repo_url': 'offline://source'}
            method = handler.handle_reproduction_requested
            event_type = 'baseline.reproduction_requested'
        await engine._publish_request(state, event_type, payload)
        await db.commit()
        request = bus.published[-1]
        original = asyncio.create_task(method(bus.view('RunnerAgent'), request))
        duplicate = None
        try:
            assert await asyncio.to_thread(entered.wait, 3)
            duplicate = asyncio.create_task(method(bus.view('RunnerAgent'), request))
            await asyncio.sleep(0)
            assert not duplicate.done()
            if cancel_duplicate:
                duplicate.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await duplicate
                duplicate = None
                assert not handler._active_requests[(request.session_id, request.event_id)].cancelled()
                assert not original.done()
        finally:
            release.set()
            await asyncio.gather(original, *([duplicate] if duplicate else []))
        assert calls == [1]
        assert not [e for e in bus.published if e.event_type == 'agent.failed']
        assert (await load_session_state(db, 's')).current_state == (
            'EvaluatingExperiment' if branch == 'experiment' else 'GeneratingMethod')
    store.close()


@pytest.mark.parametrize('branch', ['experiment', 'baseline'])
@pytest.mark.parametrize('automatic_recovery', [False, True])
async def test_concurrent_queue_replay_preserves_failed_owner_handoff(
        tmp_path, monkeypatch, branch, automatic_recovery):
    store = SpecStore(tmp_path / 'store')
    backend = FakeRunnerBackend(metric_value=8)
    handler = RunnerAgentHandler(spec_store=store, backend=backend)
    patch = store.put(kind='experiment_patch', session_id='s', key='v1', obj=VALID_EXPERIMENT_PATCH, emitter='CoderAgent')
    verifier = store.put(kind='verifier_spec', session_id='s', key='v1', obj=VALID_VERIFIER_SPEC, emitter='EvaluatorAgent')
    entered, release, duplicate_entered = threading.Event(), threading.Event(), asyncio.Event()
    executions, deliveries, injected = [], [], []
    def execute(*args, **kwargs):
        executions.append(1)
        entered.set()
        assert release.wait(5)
        record = runner_core.run_experiment(VALID_EXPERIMENT_PATCH, VALID_VERIFIER_SPEC, backend)
        return record, ({'baseline_metric': 8, 'baseline_valid': True, 'repo_commit': 'fixed'}
                        if branch == 'baseline' else None)
    monkeypatch.setattr(handler, '_gen_branch', lambda: True)
    monkeypatch.setattr(handler, '_run_gen_experiment', execute)
    monkeypatch.setattr(handler, '_resolve_repo_url', lambda *a: 'offline://source')
    monkeypatch.setattr(handler, '_run_repo_phase', execute)
    monkeypatch.setattr(handler, '_original_question', lambda *a: 'offline test question')
    real_put = store.put
    def put(**kwargs):
        if kwargs['kind'] == ('run_record' if branch == 'experiment' else 'baseline_result') and not injected:
            injected.append(1)
            raise sqlite3.OperationalError('database is locked')
        return real_put(**kwargs)
    monkeypatch.setattr(store, 'put', put)
    method_name = 'handle_run_requested' if branch == 'experiment' else 'handle_reproduction_requested'
    method = getattr(handler, method_name)
    async def receive(client, env):
        deliveries.append(env.event_id)
        if len(deliveries) == 2:
            duplicate_entered.set()
        await method(client, env)
    monkeypatch.setattr(handler, method_name, receive)
    async with aiosqlite.connect(':memory:') as db:
        bus = DeliveringBus(db)
        engine = OrchestratorEngine(bus.view('ResearchOrchestrator'), ROOT, spec_store=store)
        await engine.start()
        handler.register(bus.view('RunnerAgent'))
        state = SessionState(session_id='s', current_state='RunningExperiment' if branch == 'experiment' else 'ReproducingBaseline',
                             request_binding_required=True)
        payload = {'experiment_patch_ref': patch, 'verifier_spec_ref': verifier}
        event_type = 'experiment.run_requested'
        if branch == 'baseline':
            benchmark = store.put(kind='benchmark_spec', session_id='s', key='v1',
                                  obj={'benchmark_id': 'bench', 'metric': 'metric'}, emitter='EvaluatorAgent')
            state.current_benchmark_id, state.confirmed_repo_url = 'bench', 'offline://source'
            payload = {'benchmark_spec_ref': benchmark, 'verifier_spec_ref': verifier,
                       'benchmark_id': 'bench', 'confirmed_repo_url': 'offline://source'}
            event_type = 'baseline.reproduction_requested'
        async def start():
            await engine._publish_request(state, event_type, payload)
            await db.commit()
        if automatic_recovery:
            from autoresearch.web.server import Engine
            monkeypatch.setenv('AUTORESEARCH_WATCHDOG', '0')
            worker = Engine(str(tmp_path / 'web.db'))
            worker.loop, worker.bus, worker.session_id = asyncio.get_running_loop(), bus, 's'
            original = asyncio.wrap_future(worker.submit(start))
        else:
            original = asyncio.create_task(start())
        duplicate = None
        try:
            assert await asyncio.to_thread(entered.wait, 3)
            request_id = deliveries[0]
            duplicate = asyncio.create_task(bus.replay_pending('s'))
            await asyncio.wait_for(duplicate_entered.wait(), 3)
            await asyncio.wait([duplicate], timeout=.1)
            cursor = await db.execute('SELECT count(*) FROM web_pending_delivery WHERE event_id=?', (request_id,))
            pending_while_running = (await cursor.fetchone())[0]
        finally:
            release.set()
            outcomes = await asyncio.gather(original, *([duplicate] if duplicate else []), return_exceptions=True)
        assert executions == [1]
        assert pending_while_running == 1, 'duplicate delivery acknowledged an unfinished original request'
        assert isinstance(outcomes[1], sqlite3.OperationalError)
        if not automatic_recovery:
            assert isinstance(outcomes[0], sqlite3.OperationalError)
            cursor = await db.execute('SELECT count(*) FROM web_pending_delivery WHERE event_id=?', (request_id,))
            assert (await cursor.fetchone())[0] == 1
            assert await bus.replay_pending('s') is True
        else:
            assert outcomes[0] is None  # Web restores the saved handoff automatically.
            assert not worker._watchdog_is_busy(state.current_state)
        assert executions == [1]
        assert (await load_session_state(db, 's')).current_state == (
            'EvaluatingExperiment' if branch == 'experiment' else 'GeneratingMethod')
        assert await bus.replay_pending('s') is False
    store.close()


@pytest.mark.parametrize('producer', ['plan', 'review', 'patch'])
@pytest.mark.parametrize('fail_publish', [False, True])
async def test_producer_freezes_output_before_delivery_and_reuses_on_replay(tmp_path, monkeypatch, producer, fail_publish):
    monkeypatch.setenv('AUTORESEARCH_GEN_EXPERIMENT', '0')
    store = SpecStore(tmp_path / 'store')
    inputs = _seed_planning_inputs(store, 's', selected_hypothesis_id='h1')
    plan_ref = store.put(kind='experiment_plan', session_id='s', key='v1', obj=_valid_plan(), emitter='ExperimentPlannerAgent')
    calls = []
    if producer == 'plan':
        kind, alias, emitter = 'experiment_plan', 'v1', 'ExperimentPlannerAgent'
        field, phase, event_type = 'experiment_plan_ref', 'PlanningExperiment', 'experiment.planning_requested'
        output = _valid_plan(change_summary='current request plan')
        handler = ExperimentPlannerAgentHandler(spec_store=store, llm_client=object())
        method = handler.handle_planning_requested
        function = 'autoresearch.agents.experiment_planner_agent.core.plan_experiment'
        payload = inputs
    elif producer == 'review':
        kind, alias, emitter = 'critique_report', 'experiment_plan-v1', 'CriticAgent'
        field, phase, event_type = 'review_report_ref', 'ReviewingExperimentPlan', 'experiment.review_requested'
        output = VALID_EXPERIMENT_PLAN_REPORT_PASS
        handler = CriticAgentHandler(spec_store=store, llm_client=object())
        method = handler.handle_experiment_plan_review
        function = 'autoresearch.agents.critic_agent.core.critique_experiment_plan'
        payload = {'experiment_plan_ref': plan_ref, 'verifier_spec_ref': inputs['verifier_spec_ref']}
    else:
        kind, alias, emitter = 'experiment_patch', 'v1', 'CoderAgent'
        field, phase, event_type = 'experiment_patch_ref', 'ImplementingExperiment', 'code.implementation_requested'
        output = VALID_EXPERIMENT_PATCH
        handler = CoderAgentHandler(spec_store=store, llm_client=object())
        method = handler.handle_implementation_requested
        function = 'autoresearch.agents.coder_agent.core.implement_patch'
        payload = {'experiment_plan_ref': plan_ref, 'verifier_spec_ref': inputs['verifier_spec_ref']}
    def produce(*args, **kwargs):
        calls.append(1)
        return dict(output)
    monkeypatch.setattr(function, produce)
    async with aiosqlite.connect(':memory:') as db:
        bus = DeliveringBus(db)
        engine = OrchestratorEngine(bus.view('ResearchOrchestrator'), ROOT, spec_store=store)
        await engine.start()
        bus.subscriptions.clear()
        state = SessionState(session_id='s', current_state=phase, request_binding_required=True)
        if producer != 'plan':
            state.experiment_plan_refs = [store.snapshot(plan_ref)]
        await engine._publish_request(state, event_type, payload)
        await db.commit()
        request = bus.published[-1]
        attempted = []
        class Client:
            async def publish(self, **kwargs):
                attempted.append(kwargs['payload'])
                # A late old producer writes the same legacy slot before consumption.
                store.put(kind=kind, session_id='s', key=alias, obj={'stale': True}, emitter=emitter)
                if fail_publish and len(attempted) == 1:
                    raise RuntimeError('completion delivery interrupted')
                await bus.publish(emitter=emitter, **kwargs)
        if fail_publish:
            with pytest.raises(RuntimeError, match='delivery interrupted'):
                await method(Client(), request)
        else:
            await method(Client(), request)
        # Reconstruct the handler to test persistent, rather than in-memory, reuse.
        restored = type(handler)(spec_store=store, llm_client=object())
        await getattr(restored, method.__name__)(Client(), request)
        ready = bus.published[-1]
        assert store.get(ready.payload[field]) == output
        assert calls == [1]
        assert attempted[-1] == attempted[0]
        wrong_request = SimpleNamespace(session_id=request.session_id, event_id=request.event_id,
                                        payload={**request.payload, 'changed_input': True})
        with pytest.raises(ValueError, match='identity mismatch'):
            await getattr(restored, method.__name__)(Client(), wrong_request)
        assert calls == [1]
        await engine.tick(ready)
        saved = await load_session_state(db, 's')
        selected = (saved.experiment_plan_refs[-1] if producer == 'plan' else
                    saved.critique_report_refs[-1] if producer == 'review' else saved.experiment_patch_refs[-1])
        assert store.get(selected) == output
    store.close()
