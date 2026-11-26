import json
import subprocess

import pytest

from autoresearch.agents.runner_agent.handler import (
    _prepare_repo_workroot,
    _recover_completed_local_baseline,
)


def _git(*args, cwd):
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)


def test_prepare_repo_workroot_precreates_world_writable_control_root(tmp_path):
    workroot = tmp_path / "repo_improve_local" / "session"
    assert _prepare_repo_workroot(str(workroot)) == str(workroot.resolve())
    control = workroot / ".sandbox-control"
    assert control.is_dir()
    assert control.stat().st_mode & 0o777 == 0o777
    assert _prepare_repo_workroot(str(workroot)) == str(workroot.resolve())


@pytest.mark.parametrize("repo_name", ["repo", "repo-3"])
@pytest.mark.parametrize("ledger_kind", ["legacy", "host_authority"])
def test_recovers_only_successful_matching_baseline_attempt(tmp_path, monkeypatch, repo_name, ledger_kind):
    monkeypatch.setenv("AUTORESEARCH_WEB_WORKDIR", str(tmp_path))
    sid = "session-1"
    root = tmp_path / "repo_improve_local" / sid
    repo = root / repo_name
    repo.mkdir(parents=True)
    _git("init", cwd=repo)
    _git("remote", "add", "origin", "https://github.com/example/project.git", cwd=repo)

    attempt_id = "attempt-1"
    (repo / "ar_result.json").write_text(json.dumps({
        "attempt_id": attempt_id,
        "metric": {"name": "latency", "unit": "s", "lower_is_better": True},
        "baseline": 3.25,
        "method": None,
        "config": {"batch_size": 1},
        "notes": "official evaluator passed",
    }), encoding="utf-8")
    control = root / ".sandbox-control" / repo_name
    control.mkdir(parents=True)
    records = [
        {"event": "launch", "attempt_id": attempt_id, "phase": "baseline",
         "cwd": str(repo), "repo_commit": "abc",
         "tracked_diff_hash_start": "sha256:empty"},
        {"event": "finish", "attempt_id": attempt_id, "phase": "baseline", "rc": 0,
         "repo_commit_end": "abc", "tracked_diff_hash_end": "sha256:empty"},
    ]
    ledger = control / "attempts.tsv"
    if ledger_kind == "host_authority":
        authority = root.parent / ("." + repo_name + "-attempt-authority-test")
        authority.mkdir()
        ledger = authority / "attempts.jsonl"
        forged = [{**records[0], "repo_commit": "forged"}, {**records[1], "repo_commit_end": "forged"}]
        (control / "attempts.tsv").write_text("\n".join(json.dumps(item) for item in forged))
    ledger.write_text(
        "\n".join(json.dumps(item) for item in records) + "\n", encoding="utf-8")

    record, measured = _recover_completed_local_baseline(
        sid, "https://github.com/example/project")
    assert record["succeeded"] is True
    assert measured == {
        "baseline_metric": 3.25,
        "repo_commit": "abc",
        "metric_name": "latency",
        "metric_unit": "s",
        "lower_is_better": True,
        "config": {"batch_size": 1},
    }
    assert _recover_completed_local_baseline(
        sid, "https://github.com/other/project") is None
    if ledger_kind == "host_authority":
        ledger.write_text(json.dumps(records[0]) + "\n")
        assert _recover_completed_local_baseline(sid, "https://github.com/example/project") is None


def test_rejects_failed_or_modified_attempt(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTORESEARCH_WEB_WORKDIR", str(tmp_path))
    sid = "session-2"
    root = tmp_path / "repo_improve_local" / sid
    repo = root / "repo-1"
    repo.mkdir(parents=True)
    _git("init", cwd=repo)
    _git("remote", "add", "origin", "https://github.com/example/project", cwd=repo)
    (repo / "ar_result.json").write_text(json.dumps({
        "attempt_id": "attempt-bad", "metric": "latency", "baseline": 1.0,
    }), encoding="utf-8")
    control = root / ".sandbox-control" / "repo-1"
    control.mkdir(parents=True)
    (control / "attempts.tsv").write_text("\n".join([
        json.dumps({"event": "launch", "attempt_id": "attempt-bad", "phase": "baseline",
                    "cwd": str(repo), "tracked_diff_hash_start": "sha256:a"}),
        json.dumps({"event": "finish", "attempt_id": "attempt-bad", "phase": "baseline",
                    "rc": 0, "repo_commit_end": "abc",
                    "tracked_diff_hash_end": "sha256:b"}),
    ]) + "\n", encoding="utf-8")

    assert _recover_completed_local_baseline(
        sid, "https://github.com/example/project") is None


def test_restart_revalidates_frozen_contract_against_saved_stdout(tmp_path, monkeypatch):
    from autoresearch.experiments.repo_improve.runtime_evidence import save_output
    monkeypatch.setenv("AUTORESEARCH_WEB_WORKDIR", str(tmp_path))
    root = tmp_path / "repo_improve_local" / "strict"
    repo = root / "repo-1"
    repo.mkdir(parents=True)
    _git("init", cwd=repo)
    _git("remote", "add", "origin", "https://github.com/example/project", cwd=repo)
    result = {"attempt_id": "a", "baseline": 7.0,
              "metric": {"name": "bits", "unit": "bits", "lower_is_better": True}}
    result_path = repo / "ar_result.json"
    result_path.write_text(json.dumps(result))
    control = root / ".sandbox-control" / "repo-1"
    control.mkdir(parents=True)
    evaluation = {"metric_name": "bits", "metric_unit": "bits", "lower_is_better": True,
                  "required_values": {"/split": "dev"},
                  "quality_gates": [{"pointer": "/accuracy", "operator": "<=", "threshold": 0.12}]}
    launch = {"event": "launch", "attempt_id": "a", "phase": "baseline", "cwd": str(repo),
              "repo_commit": "abc", "tracked_diff_hash_start": "empty", "execution_protocol": evaluation}
    finish = {"event": "finish", "attempt_id": "a", "phase": "baseline", "rc": 0,
              "repo_commit_end": "abc", "tracked_diff_hash_end": "empty"}
    log = control / "attempts.tsv"
    def recover():
        log.write_text("\n".join(json.dumps(item) for item in (launch, finish)))
        return _recover_completed_local_baseline("strict", "https://github.com/example/project",
                                                  execution_contract={"evaluation": evaluation})
    assert recover() is None  # ar_result alone cannot recover a new-contract result.
    finish.update(save_output(control, "bad-quality", json.dumps({
        "baseline": 7, "baseline_valid": True, "accuracy": 0.9, "split": "dev"}), ""))
    assert recover() is None
    finish.update(save_output(control, "good", json.dumps({
        "baseline": 7, "baseline_valid": True, "accuracy": 0.1, "split": "dev"}), ""))
    recovered = recover()
    assert recovered[1]["measurement_evidence"]["valid"] is True
    recovered_without_explicit_evaluation = _recover_completed_local_baseline(
        "strict", "https://github.com/example/project", execution_contract={"record_runtime_identity": True})
    assert recovered_without_explicit_evaluation[1]["measurement_evidence"] == recovered[1]["measurement_evidence"]
    assert recovered[1]["measurement_evidence"]["quality_observations"] == {"/accuracy": 0.1}
    result_path.write_text(json.dumps({**result, "metric": {"name": "accuracy", "unit": "%", "lower_is_better": False}}))
    assert recover() is None
    result_path.write_text(json.dumps({**result, "baseline_valid": False}))
    assert recover() is None
    result_path.write_text(json.dumps(result))
    (control / "good.stdout").write_text('{"baseline":0,"baseline_valid":true}')
    assert recover() is None


@pytest.mark.parametrize("recovery_hint", ["explicit", "replayed"])
@pytest.mark.parametrize("mismatch", [None, "attempt_id", "stdout_sha256", "protocol_fingerprint", "repo_commit", "metric_value"])
def test_recovered_baseline_rechecks_boundary_and_preserves_only_matching_host_metadata(monkeypatch, mismatch, recovery_hint):
    import asyncio
    from copy import deepcopy
    from types import SimpleNamespace
    from autoresearch.agents.runner_agent.handler import RunnerAgentHandler
    from autoresearch.agents.runner_agent import handler as module
    from autoresearch.spec_store import refs

    evidence = {"attempt_id": "attempt-original", "stdout_sha256": "a" * 64,
        "protocol_fingerprint": "b" * 64, "value": 0.55, "valid": True}
    previous = {"reproduced": False, "metric_value": 0.55, "repo_commit": "original",
        "measurement_evidence": {**evidence, "quality_observations": {"retention": 1.0}},
        "repository_contract": {"sources": [{"path": "instruction.md", "content": "Official rules."}]},
        "checkpoint": "/checkpoint/original", "source_history": [{"path": "instruction.md"}]}
    if mismatch in evidence:
        previous["measurement_evidence"][mismatch] = "different"
    elif mismatch is not None:
        previous[mismatch] = "different"
    objects = {"benchmark://s1": {"benchmark_id": "official"},
        refs.encode("baseline_result", "s1", "v1"): previous}
    class Store:
        def get(self, ref):
            return objects[ref]
    recovered = ({"succeeded": True, "exit_code": 0, "artifacts": ["ar_result.json"]},
        {"baseline_metric": 0.55, "repo_commit": "original", "metric_name": "accuracy",
         "lower_is_better": False, "measurement_evidence": evidence})
    monkeypatch.setattr(module, "_recover_completed_local_baseline", lambda *args, **kwargs: deepcopy(recovered))
    handler = RunnerAgentHandler(spec_store=Store(), backend=None, p1_llm=object())
    monkeypatch.setattr(handler, "_agentic_repo_branch", lambda: True)
    monkeypatch.setattr(handler, "_resolve_repo_url", lambda *args: "https://github.com/example/project")
    boundary = {"accelerator_count": 1, "accelerator_type": "H800", "conflicts_with_official": True}
    seen = {}
    def review(sid, rr, phase, **kwargs):
        assert kwargs["confirmed_boundary"] == boundary
        assert phase == "baseline" and rr["baseline"]["metric"] == 0.55
        if mismatch is None:
            assert rr["repository_contract"] == previous["repository_contract"]
            assert rr["measurement_evidence"]["quality_observations"] == {"retention": 1.0}
        return rr
    async def publish(client, env, obj, **kwargs):
        seen.update(obj)
    monkeypatch.setattr(handler, "_review_repository_comparison", review)
    monkeypatch.setattr(handler, "_publish_baseline_completion", publish)
    env = SimpleNamespace(session_id="s1", event_id="restart", payload={
        "benchmark_spec_ref": "benchmark://s1", "recover_completed_result": True,
        "confirmed_boundary": boundary})
    if recovery_hint == "replayed":
        from autoresearch.spec_store.identity import artifact_digest
        env.payload["recover_completed_result"] = False
        objects[refs.encode("audit", "s1", "baseline-started-restart")] = {"request_digest": artifact_digest(env.payload)}
    asyncio.run(handler._handle_reproduction_requested(None, env))
    assert seen["reproduced"] is True
    assert (seen.get("checkpoint") == previous["checkpoint"]) is (mismatch is None)


@pytest.mark.parametrize('problem', [None, 'gate_failed', 'metric_mismatch',
    'missing_protocol', 'comparison_failed', 'missing_checkpoint', 'baseline_invalid'])
def test_finished_baseline_survives_later_provider_failure(tmp_path, monkeypatch, request, problem):
    import asyncio
    from copy import deepcopy
    from types import SimpleNamespace
    from autoresearch.agents.runner_agent import handler as module
    from autoresearch.spec_store import SpecStore

    store = SpecStore(tmp_path / 'store')
    request.addfinalizer(store.close)
    benchmark_ref = store.put(kind='benchmark_spec', session_id='s', key='v1',
        obj={'benchmark_id': 'official'}, emitter='EvaluatorAgent')
    record = {'succeeded': False, 'exit_code': 1,
        'stderr_excerpt': 'LLMRateLimitError after the official measurement completed',
        'stdout_excerpt': 'Original completed baseline stdout and review retained.'}
    measured = {'verdict': 'invalid', 'baseline_metric': 0.57, 'baseline_valid': True,
        'repo_commit': 'original', 'metric_name': 'accuracy', 'lower_is_better': False,
        'checkpoint': '/original/sealed-checkpoint',
        'repository_comparison_review': {'verdict': 'pass'},
        'measurement_evidence': {'attempt_id': 'attempt-original', 'stdout_sha256': 'a' * 64,
            'protocol_fingerprint': 'b' * 64, 'value': 0.57, 'valid': True,
            'gates': {'/baseline_valid': True}, 'protocol': {'kind': 'official_single'}}}
    if problem == 'gate_failed':
        measured['measurement_evidence']['gates']['/baseline_valid'] = False
    elif problem == 'metric_mismatch':
        measured['measurement_evidence']['value'] = 0.58
    elif problem == 'missing_protocol':
        measured['measurement_evidence']['protocol_fingerprint'] = None
    elif problem == 'comparison_failed':
        measured['repository_comparison_review']['verdict'] = 'baseline_invalid'
    elif problem == 'missing_checkpoint':
        measured['checkpoint'] = None
    elif problem == 'baseline_invalid':
        measured['baseline_valid'] = False
    original = deepcopy((record, measured))
    executions, published, failures = [], [], []
    handler = module.RunnerAgentHandler(spec_store=store, backend=None)
    monkeypatch.setattr(handler, '_agentic_repo_branch', lambda: True)
    monkeypatch.setattr(handler, '_resolve_repo_url', lambda *args: 'https://github.com/example/project')
    monkeypatch.setattr(handler, '_original_question', lambda *args: 'Reproduce the official starter.')
    monkeypatch.setattr(module, '_recover_completed_local_baseline', lambda *args, **kwargs: None)
    def execute(*args, **kwargs):
        executions.append(True)
        return deepcopy((record, measured))
    monkeypatch.setattr(handler, '_run_repo_phase', execute)
    async def publish(client, env, obj, **kwargs):
        published.append(obj)
    async def fail(*args):
        failures.append(args)
    monkeypatch.setattr(handler, '_publish_baseline_completion', publish)
    monkeypatch.setattr(handler, '_fail', fail)
    env = SimpleNamespace(session_id='s', event_id='first', payload={'benchmark_spec_ref': benchmark_ref})
    asyncio.run(handler._handle_reproduction_requested(None, env))
    assert executions == [True]
    assert (record, measured) == original
    if problem is None:
        assert not failures and len(published) == 1
        baseline = published[0]
        assert baseline['reproduced'] is True and baseline['exit_code'] == 0
        assert baseline['metric_value'] == 0.57 and baseline['measurement_evidence'] == measured['measurement_evidence']
        assert 'LLMRateLimitError' in baseline['log_excerpt']
        assert baseline['checkpoint'] == '/original/sealed-checkpoint'
    else:
        assert not published and len(failures) == 1


@pytest.mark.parametrize("review_failure", ["rate_limit", "parse"])
@pytest.mark.parametrize("outcome", ["pass", "baseline_invalid", "request_mismatch"])
@pytest.mark.parametrize("fail_count", [1, 2])
def test_completed_baseline_review_resumes_without_another_execution(tmp_path, monkeypatch, request, review_failure, outcome, fail_count):
    import asyncio
    from copy import deepcopy
    from types import SimpleNamespace
    from autoresearch.agents.runner_agent import handler as module
    from autoresearch.agents.comparison_guardian_agent import core as guardian
    from autoresearch.spec_store import SpecStore, refs
    from autoresearch.agents.runner_agent.handler import _baseline_review_digest

    store = SpecStore(tmp_path / "store")
    request.addfinalizer(store.close)
    benchmark_ref = store.put(kind="benchmark_spec", session_id="s", key="v1",
                              obj={"benchmark_id": "official"}, emitter="EvaluatorAgent")
    raw = {"baseline": {"metric": 10.0, "valid": True}, "candidate": {},
           "metric": "latency", "lower_is_better": True, "repo_commit": "original",
           "measurement_evidence": {"attempt_id": "original-attempt", "stdout_sha256": "a" * 64},
           "checkpoint": "/remote/immutable-checkpoint", "report": "Official baseline completed."}
    executions, failures, published = [], [], []
    env = SimpleNamespace(session_id="s", event_id="request", payload={
        "benchmark_spec_ref": benchmark_ref, "confirmed_boundary": {"accelerator_count": 1},
        "recover_completed_result": True})
    reviews = []
    class Client:
        async def publish(self, **event):
            published.append(store.get(event["payload"]["baseline_result_ref"]))
    client = Client()
    def review(*args, **kwargs):
        reviews.append(deepcopy(args[2]))
        if len(reviews) <= fail_count:
            if review_failure == "rate_limit":
                raise guardian.ComparisonGuardianLLMError(
                    "LLMServiceUnavailableError: 429 Too many requests, tokens per minute")
            raise guardian.ComparisonGuardianParseError("invalid response")
        return {"verdict": "baseline_invalid" if outcome == "baseline_invalid" else "pass",
                "issues": ["starter drift"] if outcome == "baseline_invalid" else [],
                "remediation": "Use the official starter."}
    monkeypatch.setattr(guardian, "audit_repository_comparison", review)
    monkeypatch.setattr(module, "_recover_completed_local_baseline", lambda *args, **kwargs: None)
    def build():
        handler = module.RunnerAgentHandler(spec_store=store, backend=None, p1_llm=object())
        monkeypatch.setattr(handler, "_agentic_repo_branch", lambda: True)
        monkeypatch.setattr(handler, "_resolve_repo_url", lambda *args: "https://github.com/example/project")
        monkeypatch.setattr(handler, "_original_question", lambda *args: "Run the official baseline.")
        def execute(*args, **kwargs):
            executions.append(True)
            handler._review_repository_comparison("s", deepcopy(raw), "baseline")
            pytest.fail("A completed baseline must not execute again")
        monkeypatch.setattr(handler, "_run_repo_phase", execute)
        async def fail(*args):
            failures.append(args[3:])
        monkeypatch.setattr(handler, "_fail", fail)
        return handler
    asyncio.run(build().handle_reproduction_requested(client, env))
    assert len(executions) == 1 and not published
    receipt = store.get(refs.encode("audit", "s", "baseline-review-pending-"
                                    + _baseline_review_digest(env.payload).split(":", 1)[1]))
    assert store.get(receipt["input_ref"])["result"] == raw
    for retry in range(1, fail_count):
        again = SimpleNamespace(session_id="s", event_id=f"review-retry-{retry}",
                                payload={**env.payload, "recover_completed_result": False})
        asyncio.run(build().handle_reproduction_requested(client, again))
        assert len(executions) == 1 and not published
    if outcome == "request_mismatch":
        env.payload = {**env.payload, "confirmed_boundary": {"accelerator_count": 2}}
    else:
        env = SimpleNamespace(session_id="s", event_id="new-resume-delivery",
                              payload={**env.payload, "recover_completed_result": False})
    asyncio.run(build().handle_reproduction_requested(client, env))
    assert len(executions) == 1
    if outcome == "request_mismatch":
        assert not published and len(reviews) == fail_count
    else:
        assert len(published) == 1
        assert published[0]["reproduced"] is (outcome == "pass")
        assert published[0]["metric_value"] == 10.0
        assert published[0]["measurement_evidence"] == raw["measurement_evidence"]
        assert published[0]["checkpoint"] == raw["checkpoint"]
        review_key = "baseline-review-pending-" + _baseline_review_digest(env.payload).split(":", 1)[1]
        assert store.get(refs.encode("audit", "s", review_key))["status"] == "resolved"
        if outcome == "baseline_invalid":
            # A real failed baseline may start another scientific reproduction.
            handler = build()
            fresh = []
            def execute_retry(*args, **kwargs):
                fresh.append(True)
                return {"succeeded": True, "exit_code": 0}, {
                    "baseline_metric": 11.0, "baseline_valid": True, "repo_commit": "original"}
            monkeypatch.setattr(handler, "_run_repo_phase", execute_retry)
            again = SimpleNamespace(session_id="s", event_id="scientific-retry", payload=env.payload)
            asyncio.run(handler.handle_reproduction_requested(client, again))
            assert fresh == [True] and published[-1]["metric_value"] == 11.0
