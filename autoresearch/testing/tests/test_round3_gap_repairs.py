"""Regression coverage for the independently audited round-three gaps."""
import pytest
from pathlib import Path
from types import SimpleNamespace
from tempfile import TemporaryDirectory
from contextlib import closing
from unittest.mock import patch
import hashlib
import json
import subprocess
import sys

from autoresearch.experiments.repo_improve import runtime_evidence as evidence
from autoresearch.experiments.repo_improve.component_plan import (
    ComponentNode, ComponentPlan, ComponentLedger, evaluate_component_contract, ledger_for_workroot,
)
from autoresearch.experiments.repo_improve.checkpoint_artifacts import snapshot_checkpoint, validate_checkpoint
from autoresearch.experiments.repo_improve.agentic_improve import _repository_task_contract
from autoresearch.agents.runner_agent.handler import RunnerAgentHandler
from autoresearch.agents.memory_agent.handler import MemoryAgentHandler
from autoresearch.agents._reuse.openhands_backend import OpenHandsBackend, SandboxHandle
from autoresearch.spec_store import SpecStore
from autoresearch.spec_store.identity import artifact_digest
from autoresearch.agents.moss_agent.patch_writer import PatchWriter
from autoresearch.agents.moss_agent.core import _patch_content_leaks
from autoresearch.agents.experiment_planner_agent.kernel import ConfigLoader as PlannerConfig


def node():
    return ComponentNode('gain', 'gain', 'direct_gain', (), 'improve', ('parent',),
        ('candidate',), (), 'change implementation', 'lower loss',
        'signed_metric_improvement > 0', 'inspect implementation', 'same mechanism', 'final result')


def receipt(root, name, payload, **extra):
    return {'attempt_id': name, 'launched': True, 'finished': True, 'rc': 0,
        'cwd': '/candidate', 'phase': 'method', 'repo_commit': 'a'*40, 'repo_commit_end': 'a'*40,
        'tracked_diff_hash_start': 'same', 'tracked_diff_hash_end': 'same', 'execution_protocol': {},
        **evidence.save_output(root, name, json.dumps(payload), ''), **extra}


def pair(root, candidate_value=0.5, parent_value=0.6, gate_config=None, parent_extra=None):
    config = gate_config or {}
    parent = receipt(root, 'attempt-parent', {'method': parent_value, 'candidate_valid': True,
        **(parent_extra or {})}, cwd='/control', execution_protocol=config)
    candidate = receipt(root, 'attempt-candidate', {'method': candidate_value, 'candidate_valid': True,
        'correctness': True}, dependency_parent_cwd='/control',
        dependency_parent_state={'repo_commit': 'a'*40, 'tracked_diff_hash_start': 'same'}, execution_protocol=config)
    report = [{'component': 'gain', 'type': 'direct_gain', 'actual': candidate_value,
        'evidence': {'source': 'runtime_attempt', 'ref': 'attempt-candidate', 'json_pointer': '/method'},
        'parent_metric': parent_value,
        'parent_evidence': {'source': 'runtime_attempt', 'ref': 'attempt-parent', 'json_pointer': '/method'}}]
    return candidate, parent, report


def persist(store, kind, obj):
    return store.put(kind=kind, session_id='audit', key='snapshot-'+artifact_digest(obj).split(':')[1],
                     obj=obj, emitter='RunnerAgent')



def test_bound_parent_overrides_frozen_baseline(tmp_path):
    candidate, parent, report = pair(tmp_path, .85, .8)
    bound = evidence.bind_component_measurements(report, candidate, {parent['attempt_id']: parent})
    gate = evaluate_component_contract(node(), parent_metric=.9, candidate_metric=.85,
                                      lower_is_better=True, ablation_report=bound)
    assert gate['parent_metric'] == .8
    assert gate['passed'] is False

@pytest.mark.parametrize('gate', [False, None, 1])
def test_parent_custom_gate_cannot_certify_gain(tmp_path, gate):
    extra = {} if gate is None else {'correctness': gate}
    candidate, parent, report = pair(tmp_path, gate_config={'gate_pointers': ['/correctness']}, parent_extra=extra)
    try:
        bound = evidence.bind_component_measurements(report, candidate, {parent['attempt_id']: parent})
    except ValueError:
        return
    assert not evaluate_component_contract(node(), parent_metric=None, candidate_metric=.5,
        lower_is_better=True, ablation_report=bound)['passed']

def test_parent_roundoff_retains_host_scalar(tmp_path):
    candidate, parent, report = pair(tmp_path, .2, .30000000000000004)
    report[0]['parent_metric'] = .3
    bound = evidence.bind_component_measurements(report, candidate, {parent['attempt_id']: parent})
    assert bound[0]['parent_metric'] == .30000000000000004

def test_separate_probe_requires_same_declared_files(tmp_path):
    protocol = {'source_files': {'probe.py': 'implementation'}}
    old_files = {'probe.py': {'sha256': '1'*64, 'role': 'implementation'}}
    new_files = {'probe.py': {'sha256': '2'*64, 'role': 'implementation'}}
    old = receipt(tmp_path, 'attempt-probe', {'check': 0}, execution_protocol=protocol,
                  execution_files_start=old_files, execution_files_end=old_files)
    current = receipt(tmp_path, 'attempt-current', {'method': .5, 'candidate_valid': True}, execution_protocol=protocol,
                      execution_files_start=new_files, execution_files_end=new_files)
    with pytest.raises(ValueError, match='cannot bind'):
        evidence.bind_component_measurements([{'component': 'check', 'type': 'indirect_gain', 'actual': 0,
            'evidence': {'source': 'runtime_attempt', 'ref': 'attempt-probe', 'json_pointer': '/check'}}],
            current, {'attempt-probe': old})

def test_tiny_aggregate_does_not_allow_large_relative_error(tmp_path):
    rec = receipt(tmp_path, 'attempt-tiny', {'method': 1e-20, 'candidate_valid': True,
        'samples': [1e-30, 1e-30], 'control_samples': [1e-10, 1e-10]})
    with pytest.raises(ValueError, match='aggregate'):
        evidence.measured_result(rec, {'measurement': {'kind': 'repeated', 'repeats': 2,
                                  'aggregation': 'mean'}}, 'method')

def test_task_document_full_identity_and_truncation(tmp_path):
    doc = tmp_path/'task.md'
    doc.write_text('a'*25000+'tail rule A')
    before = _repository_task_contract(str(tmp_path), 'task.md')
    assert before['sources'][0]['sha256'] == hashlib.sha256(doc.read_bytes()).hexdigest()
    assert before['sources'][0]['truncated'] is True
    assert before['capture_complete'] is False
    doc.write_text('a'*25000+'tail rule B')
    assert _repository_task_contract(str(tmp_path), 'task.md') != before

def test_revision_does_not_inherit_missing_checkpoint(tmp_path):
    old = ledger_for_workroot(str(tmp_path), ComponentPlan('old-method', (node(),), ('gain',)))
    old.begin('gain')
    old.record_attempt('gain', execution_succeeded=True, contract_passed=True, checkpoint='/no/such/checkpoint',
                       metric_value=.5, quality_value=.5, max_attempts=1)
    new = ledger_for_workroot(str(tmp_path), ComponentPlan('new-method', (node(),), ('gain',)))
    assert not new.all_accepted()
    assert old.all_accepted()  # Historical evidence was not rewritten.

def git_repo(root):
    root.mkdir()
    subprocess.run(['git', 'init', '-q', str(root)], check=True)
    (root/'code.py').write_text('x=1')
    subprocess.run(['git', '-C', str(root), 'add', '.'], check=True)
    subprocess.run(['git', '-C', str(root), '-c', 'user.name=Test', '-c', 'user.email=test@local',
                    'commit', '-qm', 'fixture'], check=True)
    return root

def test_external_directory_link_cannot_escape_checkpoint(tmp_path):
    repo = git_repo(tmp_path/'repo')
    outside = tmp_path/'outside'; outside.mkdir(); (outside/'code.py').write_text('x=1')
    (repo/'alias').symlink_to(outside, target_is_directory=True)
    subprocess.run(['git', '-C', str(repo), 'add', 'alias'], check=True)
    subprocess.run(['git', '-C', str(repo), '-c', 'user.name=Test', '-c', 'user.email=test@local',
                    'commit', '-qm', 'alias'], check=True)
    with pytest.raises(ValueError, match='link|snapshot'):
        snapshot_checkpoint(str(repo), 'attempt-alias')

def test_attempt_transport_never_blindly_replays(tmp_path):
    calls = []
    class Workspace:
        def execute_command(self, command, **kwargs):
            calls.append(command)
            raise ConnectionResetError('connection reset by peer after execution')
    backend = OpenHandsBackend()
    from autoresearch.agents._reuse.openhands_backend import InfraError
    with pytest.raises(InfraError, match='unknown|uncertain'):
        backend.execute_attempt_command(SandboxHandle(workspace=Workspace()), 'training')
    assert calls == ['training']

@pytest.mark.parametrize('marker,lines,valid', [
    ('AR_RESULT:', '{"epoch":1}\nAR_RESULT:{"method":0.5,"candidate_valid":true}\n', True),
    ('AR_RESULT:', 'AR_RESULT:{"method":0.5}\nAR_RESULT:{"method":0.6}\n', False),
    ('AR_RESULT:', '{"method":0.5,"candidate_valid":true}\n', False),
])
def test_explicit_result_channel(tmp_path, marker, lines, valid):
    rec = receipt(tmp_path, 'attempt-marked', {}, execution_protocol={'result_marker': marker})
    Path(rec['stdout_path']).write_text(lines)
    rec['stdout_sha256'] = hashlib.sha256(lines.encode()).hexdigest()
    if valid:
        assert evidence.measured_result(rec, {}, 'method')['value'] == .5
    else:
        with pytest.raises(ValueError):
            evidence.read_output(rec)

def test_strict_contract_requires_real_configuration():
    from autoresearch.experiments.repo_improve.agentic_improve import validate_execution_contract
    with pytest.raises(ValueError, match='requires evaluation'):
        validate_execution_contract({'execution_contract': {'require_evaluation_contract': True}})
    declaration = {'require_evaluation_contract': True, 'evaluation': {
        'argv': ['python', 'evaluate.py'], 'metric_pointer': '/method',
        'gate_pointers': ['/candidate_valid'], 'required_values': {'/split': 'held_out'},
        'measurement': {'kind': 'official_single'}, 'immutable_files': {'evaluate.py': 'a'*64}}}
    validate_execution_contract({'execution_contract': declaration, 'capture_complete': True})
    with pytest.raises(ValueError, match='incomplete'):
        validate_execution_contract({'execution_contract': declaration, 'capture_complete': False})

@pytest.mark.parametrize('case', ['unexecuted', 'different', 'same'])
def test_failed_verification_is_scoped_to_candidate(tmp_path, case):
    with closing(SpecStore(tmp_path/'store')) as store:
        handler = object.__new__(MemoryAgentHandler); handler.spec_store = store
        old_run = persist(store, 'run_record', {'metric_value': .5, 'exit_code': 0})
        prior = {'method_metric': .5, 'baseline_metric': 1.0, 'lower_is_better': True,
            'metric_name': 'loss', 'metric_unit': 'unit', 'verdict': 'new_best', 'accepted': True,
            'diff': 'candidate A', 'run_record_snapshot_ref': old_run,
            'measurement_evidence': {'value': .5, 'protocol_fingerprint': 'protocol'}}
        old_ref = persist(store, 'gen_experiment_result', prior)
        new_run = persist(store, 'run_record', {'metric_value': None, 'exit_code': 1, 'stderr_excerpt': case})
        current = {**prior, 'method_metric': None, 'accepted': False, 'verdict': 'invalid',
            'diff': {'unexecuted': '', 'same': 'candidate A', 'different': 'candidate B'}[case],
            'component_execution_started': case != 'unexecuted', 'run_record_snapshot_ref': new_run}
        persist(store, 'gen_experiment_result', current)
        memory, *_ = handler._consolidate_gen(new_run,
            prior_research_memory={'best_metric': .5, 'best_result_ref': old_ref},
            verification_report={'verdict': 'failed'})
        assert memory['best_metric'] == (None if case == 'same' else .5)
        assert memory['best_result_ref'] == (None if case == 'same' else old_ref)

def test_paired_control_gain_does_not_claim_frozen_baseline_win(tmp_path):
    with closing(SpecStore(tmp_path/'store')) as store:
        handler = object.__new__(MemoryAgentHandler); handler.spec_store = store
        run = persist(store, 'run_record', {'metric_value': 1.1, 'exit_code': 0})
        gen = {'method_metric': 1.1, 'baseline_metric': 1.0, 'lower_is_better': True,
            'metric_name': 'loss', 'metric_unit': 'unit', 'verdict': 'valid_not_best', 'accepted': True,
            'component_execution': {'mode': 'integration'}, 'run_record_snapshot_ref': run,
            'measurement_evidence': {'value': 1.1, 'protocol_fingerprint': 'frozen',
                'protocol': {'kind': 'repeated'}, 'samples': [1.1,1.1], 'control_metric': 1.2}}
        persist(store, 'gen_experiment_result', gen)
        memory, *_ = handler._consolidate_gen(run, verification_report={'verdict': 'verified'})
        text = ' '.join(memory['verified_findings'])
        assert 'Method beats baseline' not in text
        assert 'does not beat' in text and 'Paired control=1.2' in text

@pytest.mark.parametrize('lower_is_better,method', [(True, 1.1), (False, .9), (True, 1.0)])
def test_partial_accepted_report_does_not_claim_unmeasured_improvement(lower_is_better, method):
    from autoresearch.agents.report_agent.handler import _apply_latest_verified_structured_result
    result = {'accepted': True, 'verdict': 'valid_not_best', 'baseline_metric': 1.0, 'method_metric': method,
              'metric_name': 'quality', 'lower_is_better': lower_is_better,
              'method_implementation_review': {'verdict': 'partial'}}
    report = _apply_latest_verified_structured_result(
        {}, {'best_metric': method}, result, best_result=result)
    assert 'does not beat baseline' in report['summary']
    assert 'measured improvement' not in report['summary']
    assert 'partial implementation' in report['summary']


def test_scoped_moss_guidance_cannot_cross_session_or_agent(tmp_path):
    from autoresearch.agents.moss_agent.patch_writer import scoped_prompt_patch
    with closing(SpecStore(tmp_path/'store')) as store:
        decision = {'action': 'shrink_patch', 'scope': {'session_id': 'A', 'method_fingerprint': None},
                    'prompt_patch': {'agent': 'CoderAgent', 'content': 'task-specific-path'}}
        ref = store.put(kind='moss_intervention', session_id='A', key='snapshot-'+artifact_digest(decision).split(':')[1],
                        obj=decision, emitter='MOSS')
        assert 'task-specific-path' in scoped_prompt_patch(store, ref, 'A', 'CoderAgent')
        assert scoped_prompt_patch(store, ref, 'A', 'ExperimentPlannerAgent') is None
        with pytest.raises(ValueError, match='this session'):
            scoped_prompt_patch(store, ref, 'B', 'CoderAgent')

@pytest.mark.asyncio
async def test_verification_events_keep_their_own_report(tmp_path, monkeypatch):
    from autoresearch.agents.evaluator_agent.handler import EvaluatorAgentHandler
    from autoresearch.agents.evaluator_agent import core
    from autoresearch.agents.evaluator_agent.tests.test_evaluation_handler import (
        FakeBusClient, _make_envelope, _seed_verification_inputs)
    with closing(SpecStore(tmp_path/'store')) as store:
        inputs = _seed_verification_inputs(store, 'audit')
        client = FakeBusClient(); handler = EvaluatorAgentHandler(spec_store=store, llm_client=None)
        verdicts = iter(['verified','failed'])
        monkeypatch.setattr(core, 'verify_best_result', lambda *a, **k: {
            'verdict': next(verdicts), 'checks': [], 'anti_cheat_flags': []})
        for _ in range(2):
            await handler.handle_verification_requested(client,
                _make_envelope('result.verification_requested','audit',inputs,'ResearchOrchestrator'))
        refs_ = [p.payload['verification_report_ref'] for p in client.published if p.event_type=='result.verified']
        assert len(refs_) == 2 and refs_[0] != refs_[1]
        assert store.get(refs_[0])['verdict'] == 'verified'
        assert store.get(refs_[1])['verdict'] == 'failed'
        assert store.get(refs_[0])['run_record_digest'] == artifact_digest(store.get(inputs['run_record_ref']))

@pytest.mark.parametrize('same_identity', [True, False])
def test_checkpoint_inheritance_keeps_execution_identity(tmp_path, same_identity):
    repo = git_repo(tmp_path/'repo')
    checkpoint = snapshot_checkpoint(str(repo), 'attempt-original')
    old = ledger_for_workroot(str(tmp_path), ComponentPlan('method-one', (node(),), ('gain',)), execution_identity='protocol-A')
    old.begin('gain')
    old.record_attempt('gain', execution_succeeded=True, contract_passed=True, checkpoint=checkpoint,
                       metric_value=.5, quality_value=.5, max_attempts=1)
    revised = ledger_for_workroot(str(tmp_path), ComponentPlan('method-two', (node(),), ('gain',)),
                                  execution_identity='protocol-A' if same_identity else 'protocol-B')
    assert revised.all_accepted() is same_identity
    assert old.all_accepted()

def test_uncertain_attempt_blocks_relaunch_and_restart(tmp_path_factory):
    from autoresearch.experiments.repo_improve.agentic_improve import _AttemptAuthority, _attempt_records, _install_attempt_guard
    root = tmp_path_factory.mktemp('uncertain')
    repo = git_repo(root/'repo')
    bindir, log = _install_attempt_guard(str(root), str(repo), str(repo/'result.json'))
    calls = []
    class Workspace:
        def execute_command(self, cmd, **kwargs):
            calls.append(cmd)
            raise ConnectionResetError('connection reset by peer after launch')
    backend = OpenHandsBackend(); handle = SandboxHandle(workspace=Workspace())
    payload = json.dumps({'argv': ['python', 'train.py'], 'cwd': str(repo)}).encode()+b'\n'
    calls_after_first = None
    for number in range(2):
        authority = _AttemptAuthority(backend, handle, str(repo), 'method', log, str(root/f'{number}.sock'),
                                      str(Path(bindir).parent/'last_attempt_id'))
        authority.start()
        try:
            result = authority._handle(SimpleNamespace(recv=lambda _: payload))
            assert result['rc'] != 0
            assert 'unknown' in result['stderr']
        finally:
            authority.stop()
        if number == 0:
            calls_after_first = len(calls)
            assert calls_after_first >= 1
        else:
            assert len(calls) == calls_after_first
    record, = _attempt_records(log).values()
    assert record['execution_uncertain'] is True
    assert record['finished'] is False

@pytest.mark.asyncio
@pytest.mark.parametrize('same_method', [True, False])
async def test_engine_routes_scoped_guidance_only_to_matching_method(tmp_path, same_method):
    from autoresearch.orchestrator.engine import OrchestratorEngine
    from autoresearch.experiments.repo_improve.component_plan import method_fingerprint
    with closing(SpecStore(tmp_path/'store')) as store:
        method = {'approach': 'current'}
        method_ref = store.put(kind='method', session_id='audit', key='v1', obj=method, emitter='MethodAgent')
        decision = {'action': 'shrink_patch', 'scope': {'session_id': 'audit',
            'method_fingerprint': method_fingerprint(method) if same_method else 'old-method'},
            'prompt_patch': {'agent': 'CoderAgent', 'content': 'current-task-only'}}
        ref = store.put(kind='moss_intervention', session_id='audit',
            key='snapshot-'+artifact_digest(decision).split(':')[1], obj=decision, emitter='MOSS')
        messages = []
        class Bus:
            async def publish(self, **kwargs): messages.append(kwargs)
        engine = object.__new__(OrchestratorEngine)
        engine.spec_store = store; engine.client = Bus()
        engine._entry_actions = {'RunningExperiment': ('experiment.run_requested', lambda state,message: {})}
        state = SimpleNamespace(current_state='RunningExperiment', session_id='audit', method_ref=method_ref, request_binding_required=False,
                                 moss_interventions=[{'moss_intervention_ref': ref}])
        await engine._execute_entry_action(state)
        assert ('moss_intervention_ref' in messages[0]['payload']) is same_method

@pytest.mark.asyncio
@pytest.mark.parametrize('kind', ['evaluation', 'verification'])
async def test_engine_ignores_delayed_report_for_another_run(tmp_path, kind):
    from autoresearch.orchestrator.engine import OrchestratorEngine
    with closing(SpecStore(tmp_path/'store')) as store:
        current = persist(store, 'run_record', {'metric_value': .5})
        obj = {'verdict': 'verified' if kind == 'verification' else 'new_best',
               'run_record_digest': artifact_digest({'metric_value': .9})}
        report = store.put(kind=kind+'_report', session_id='audit', key='snapshot-'+artifact_digest(obj).split(':')[1],
                           obj=obj, emitter='EvaluatorAgent')
        engine = object.__new__(OrchestratorEngine);engine.spec_store = store
        state = SimpleNamespace(session_id='audit',run_record_refs=[current],evaluation_report_refs=[],verification_report_refs=[])
        event = SimpleNamespace(payload={kind+'_report_ref': report,'verdict': obj['verdict']})
        handler = engine._on_EvaluatingExperiment_experiment_evaluated if kind=='evaluation' else engine._on_VerifyingBestResult_result_verified
        await handler(state,event)
        assert state.evaluation_report_refs == [] and state.verification_report_refs == []
