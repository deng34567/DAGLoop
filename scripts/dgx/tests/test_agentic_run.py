from __future__ import annotations

import base64
import importlib.util
import json
from pathlib import Path
import sys

import pytest


_MODULE_PATH = Path(__file__).resolve().parents[1] / "agentic_run.py"
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
_SPEC = importlib.util.spec_from_file_location("dgx_agentic_run", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
_decode_execution_contract = _MODULE._decode_execution_contract
_decode_memory_scope = _MODULE._decode_memory_scope
_decode_prior_worktree = _MODULE._decode_prior_worktree
_select_repo_urls = _MODULE._select_repo_urls


def test_large_request_keeps_context_out_of_child_process_environment():
    import io
    env = {"AR_EXECUTION_PHASE": "method"}
    frozen = {"metric_value": 1, "notes": "x" * 200000}
    encoded = base64.b64encode(json.dumps(frozen).encode()).decode()
    question = "研究组件\n" * 30000
    q, loaded = _MODULE._read_request(["agentic_run.py", "--request-stdin"], env,
        io.StringIO(json.dumps({"question": question, "environment": {
            "AR_FROZEN_BASELINE_B64": encoded, "AR_COMPONENT_STAGE": "0"}})))
    assert q == question
    assert "AR_FROZEN_BASELINE_B64" not in env
    assert _decode_execution_contract(loaded)[1] == frozen
    assert loaded["AR_COMPONENT_STAGE"] == "0"


def test_streamed_request_cannot_inject_runtime_environment():
    import io
    with pytest.raises(ValueError, match="invalid streamed request fields"):
        _MODULE._read_request(["entry", "--request-stdin"], {}, io.StringIO(
            json.dumps({"question": "q", "environment": {"PATH": "untrusted"}})))


@pytest.mark.parametrize("layout", ["bundle", "source"])
def test_entry_imports_its_own_code_before_an_old_deployment(tmp_path, monkeypatch, layout):
    root = tmp_path / "current"
    (root / "autoresearch").mkdir(parents=True)
    entry = root / "agentic_run.py" if layout == "bundle" else root / "scripts/dgx/agentic_run.py"
    old = tmp_path / "old"
    (old / "autoresearch").mkdir(parents=True)
    monkeypatch.setattr(sys, "path", [str(old), *sys.path])
    bootstrap = _MODULE_PATH.read_text().split("from autoresearch.", 1)[0]
    exec(compile(bootstrap, str(entry), "exec"), {"__file__": str(entry)})
    assert sys.path[0] == str(root)


def test_session_lock_rejects_a_concurrent_writer(tmp_path):
    first = _MODULE._acquire_session_lock(str(tmp_path))
    try:
        with pytest.raises(BlockingIOError):
            _MODULE._acquire_session_lock(str(tmp_path), blocking=False)
    finally:
        first.close()

    resumed = _MODULE._acquire_session_lock(str(tmp_path), blocking=False)
    resumed.close()


def test_session_lock_reopens_read_only_file_without_replacing_it(tmp_path):
    lock = tmp_path / ".agentic_run.lock"
    lock.write_text("retained lock identity")
    lock.chmod(0o444)
    inode = lock.stat().st_ino
    first = _MODULE._acquire_session_lock(str(tmp_path), blocking=False)
    try:
        with pytest.raises(BlockingIOError):
            _MODULE._acquire_session_lock(str(tmp_path), blocking=False)
    finally:
        first.close()
    assert lock.stat().st_ino == inode
    assert lock.read_text() == "retained lock identity"
    _MODULE._acquire_session_lock(str(tmp_path), blocking=False).close()


def test_gpu_locks_cover_different_sessions_and_release_partial_claims(tmp_path, monkeypatch):
    monkeypatch.setattr(_MODULE, "_container_uses_cards", lambda cards: False)
    first = _MODULE._acquire_gpu_locks(str(tmp_path / "session-a"), "2", blocking=False)
    try:
        with pytest.raises(BlockingIOError):
            _MODULE._acquire_gpu_locks(str(tmp_path / "session-b"), "2,1", blocking=False)
        other = _MODULE._acquire_gpu_locks(str(tmp_path / "session-c"), "1", blocking=False)
        for handle in other:
            handle.close()
    finally:
        for handle in first:
            handle.close()
    resumed = _MODULE._acquire_gpu_locks(str(tmp_path / "session-b"), "2,1", blocking=False)
    for handle in resumed:
        handle.close()


def test_separate_campaign_roots_share_gpu_lock(tmp_path, monkeypatch):
    monkeypatch.setattr(_MODULE, "_container_uses_cards", lambda cards: False)
    monkeypatch.setenv("AUTORESEARCH_REMOTE_SESSION_ROOT", str(tmp_path / "new-round"))
    assert _MODULE._session_workroot("same-session").startswith(str(tmp_path / "new-round") + "/sess-")
    monkeypatch.setenv("AUTORESEARCH_GPU_LOCK_DIR", str(tmp_path / "old-round/.gpu-locks"))
    first = _MODULE._acquire_gpu_locks(str(tmp_path / "old-round/session-a"), "2", blocking=False)
    try:
        with pytest.raises(BlockingIOError):
            _MODULE._acquire_gpu_locks(_MODULE._session_workroot("same-session"), "2", blocking=False)
    finally:
        for handle in first:
            handle.close()


def test_shared_lock_root_does_not_contend_across_hosts(tmp_path, monkeypatch):
    monkeypatch.setattr(_MODULE, "_container_uses_cards", lambda cards: False)
    monkeypatch.setenv("AUTORESEARCH_GPU_LOCK_DIR", str(tmp_path / ".gpu-locks"))
    monkeypatch.setattr(_MODULE.socket, "gethostname", lambda: "dgx-a")
    first = _MODULE._acquire_gpu_locks(
        str(tmp_path / "session-a"), "2", blocking=False)
    try:
        monkeypatch.setattr(_MODULE.socket, "gethostname", lambda: "dgx-b")
        second = _MODULE._acquire_gpu_locks(
            str(tmp_path / "session-b"), "2", blocking=False)
        try:
            assert Path(first[0].name).parent.name == "dgx-a"
            assert Path(second[0].name).parent.name == "dgx-b"
        finally:
            for handle in second:
                handle.close()
    finally:
        for handle in first:
            handle.close()


def test_gpu_lock_waits_for_legacy_container_without_stopping_it(tmp_path, monkeypatch):
    busy = iter([True, False])
    monkeypatch.setattr(_MODULE, "_container_uses_cards", lambda cards: next(busy))
    waits = []
    monkeypatch.setattr(_MODULE.time, "sleep", waits.append)
    handles = _MODULE._acquire_gpu_locks(str(tmp_path / "session"), "3")
    assert waits == [5]
    for handle in handles:
        handle.close()


def test_gpu_pool_uses_free_alternative_without_releasing_other_session(tmp_path, monkeypatch):
    monkeypatch.setattr(_MODULE, "_container_uses_cards", lambda cards: False)
    first = _MODULE._acquire_gpu_locks(str(tmp_path / "a"), "0", blocking=False)
    second = []
    try:
        second = _MODULE._acquire_gpu_locks(
            str(tmp_path / "b"), "0", allowed_cards="0,5", blocking=False)
        assert [Path(h.name).name for h in second] == ["gpu-5.lock"]
        with pytest.raises(BlockingIOError):
            _MODULE._acquire_gpu_locks(str(tmp_path / "c"), "0", blocking=False)
        with pytest.raises(BlockingIOError):
            _MODULE._acquire_gpu_locks(
                str(tmp_path / "c"), "0", allowed_cards="0,5", blocking=False)
    finally:
        for handle in first + second:
            handle.close()


def test_gpu_affinity_keeps_later_attempts_on_same_card(tmp_path, monkeypatch):
    monkeypatch.setattr(_MODULE, "_container_uses_cards", lambda cards: False)
    workroot = tmp_path / "session"
    first = _MODULE._acquire_gpu_locks(
        str(workroot), "0", allowed_cards="0,5", blocking=False)
    try:
        assert [Path(h.name).name for h in first] == ["gpu-0.lock"]
    finally:
        for handle in first:
            handle.close()

    second = _MODULE._acquire_gpu_locks(
        str(workroot), "5", allowed_cards="0,5", blocking=False)
    try:
        assert [Path(h.name).name for h in second] == ["gpu-0.lock"]
        assert (workroot / ".gpu-card-affinity").read_text() == "0\n"
    finally:
        for handle in second:
            handle.close()


def test_gpu_affinity_does_not_fall_back_when_session_card_is_busy(tmp_path, monkeypatch):
    monkeypatch.setattr(_MODULE, "_container_uses_cards", lambda cards: False)
    workroot = tmp_path / "session"
    first = _MODULE._acquire_gpu_locks(
        str(workroot), "0", allowed_cards="0,5", blocking=False)
    try:
        with pytest.raises(BlockingIOError, match="No permitted GPU group is free"):
            _MODULE._acquire_gpu_locks(
                str(workroot), "5", allowed_cards="0,5", blocking=False)
    finally:
        for handle in first:
            handle.close()


@pytest.mark.parametrize("occupied,expected", [(set(), "0"), ({"0"}, "5")])
def test_preferred_gpu_affinity_uses_free_card_and_records_actual_assignment(
        tmp_path, monkeypatch, occupied, expected):
    monkeypatch.setenv("AUTORESEARCH_GPU_AFFINITY", "prefer")
    monkeypatch.setattr(_MODULE, "_container_uses_cards", lambda cards: bool(cards & occupied))
    workroot = tmp_path / "session"
    workroot.mkdir()
    (workroot / ".gpu-card-affinity").write_text("0\n")
    handles = _MODULE._acquire_gpu_locks(
        str(workroot), "5", allowed_cards="0,5", blocking=False)
    try:
        assert [Path(h.name).name for h in handles] == [f"gpu-{expected}.lock"]
        assert (workroot / ".gpu-card-affinity").read_text() == expected + "\n"
        with pytest.raises(BlockingIOError):
            _MODULE._acquire_gpu_locks(str(tmp_path / "other"), expected, blocking=False)
    finally:
        for handle in handles:
            handle.close()


def test_preferred_gpu_affinity_waits_when_all_allowed_cards_are_busy(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTORESEARCH_GPU_AFFINITY", "prefer")
    monkeypatch.setattr(_MODULE, "_container_uses_cards", lambda cards: True)
    workroot = tmp_path / "session"
    workroot.mkdir()
    (workroot / ".gpu-card-affinity").write_text("0\n")
    with pytest.raises(BlockingIOError):
        _MODULE._acquire_gpu_locks(str(workroot), "5", allowed_cards="0,5", blocking=False)
    assert (workroot / ".gpu-card-affinity").read_text() == "0\n"


@pytest.mark.parametrize("busy,preferred,pool,expected", [
    (set(), "5", "0,5", ["gpu-5.lock"]),
    ({"0"}, "0", "0,5", ["gpu-5.lock"]),
    ({"0"}, "0,2", "0,2,5", ["gpu-2.lock", "gpu-5.lock"]),
])
def test_gpu_pool_respects_legacy_containers_and_requested_count(
        tmp_path, monkeypatch, busy, preferred, pool, expected):
    monkeypatch.setattr(_MODULE, "_container_uses_cards", lambda cards: bool(cards & busy))
    handles = _MODULE._acquire_gpu_locks(
        str(tmp_path / "a"), preferred, allowed_cards=pool, blocking=False)
    try:
        assert [Path(h.name).name for h in handles] == expected
    finally:
        for handle in handles:
            handle.close()


@pytest.mark.parametrize("pool", ["", "1", "0,uuid", "0,"])
def test_gpu_pool_rejects_invalid_or_excluded_request(tmp_path, pool):
    with pytest.raises(ValueError, match="allowed pool"):
        _MODULE._acquire_gpu_locks(str(tmp_path), "0", allowed_cards=pool, blocking=False)


def test_gpu_pool_does_not_expand_when_all_allowed_cards_busy(tmp_path, monkeypatch):
    seen = []
    def busy(cards):
        seen.append(cards)
        return True
    monkeypatch.setattr(_MODULE, "_container_uses_cards", busy)
    with pytest.raises(BlockingIOError):
        _MODULE._acquire_gpu_locks(
            str(tmp_path), "0", allowed_cards="0,5", blocking=False)
    assert seen == [{"0"}, {"5"}]


@pytest.mark.parametrize("config,busy", [
    ({"Devices": [{"PathOnHost": "/dev/nvidia2"}]}, True),
    ({"Devices": [{"PathOnHost": "/dev/nvidia3"}]}, False),
    ({"DeviceRequests": [{"Driver": "nvidia", "DeviceIDs": ["2"]}]}, True),
    ({"DeviceRequests": [{"Driver": "nvidia", "DeviceIDs": ["3"]}]}, False),
    ({"DeviceRequests": [{"Driver": "nvidia", "Count": -1}]}, True),
    ({"DeviceRequests": [{"Driver": "nvidia", "DeviceIDs": ["GPU-uuid"]}]}, True),
])
def test_gpu_lock_checks_actual_docker_assignment(monkeypatch, config, busy):
    outputs = iter(["container-id", json.dumps([
        {"State": {"Running": True, "Paused": True}, "HostConfig": config},
    ]), ""])
    monkeypatch.setattr(_MODULE.subprocess, "check_output", lambda *a, **kw: next(outputs))
    assert _MODULE._container_uses_cards({"2"}) is busy


@pytest.mark.parametrize("survivors,busy", [
    ([{"State": {"Running": True},
       "HostConfig": {"Devices": [{"PathOnHost": "/dev/nvidia2"}]}}], True),
    ([], False),
])
def test_gpu_probe_preserves_survivors_when_an_unrelated_container_disappears(
        monkeypatch, survivors, busy):
    def output(argv, **kwargs):
        if argv[0] == "nvidia-smi":
            return ""
        if argv[:2] == ["docker", "ps"]:
            return "gone live"
        raise _MODULE.subprocess.CalledProcessError(
            1, argv, output=json.dumps(survivors),
            stderr="Error: No such object: gone\n")
    monkeypatch.setattr(_MODULE.subprocess, "check_output", output)
    assert _MODULE._container_uses_cards({"2"}) is busy


@pytest.mark.parametrize("occupied_uuid,busy", [("GPU-two", True), ("GPU-three", False)])
def test_gpu_lock_checks_native_compute_processes_without_docker(monkeypatch, occupied_uuid, busy):
    def output(argv, **kwargs):
        if argv[:2] == ["docker", "ps"]:
            return ""
        if "--query-compute-apps=gpu_uuid" in argv:
            return occupied_uuid + "\n"
        assert "--query-gpu=index,uuid" in argv
        return "2, GPU-two\n3, GPU-three\n"

    monkeypatch.setattr(_MODULE.subprocess, "check_output", output)
    assert _MODULE._container_uses_cards({"2"}) is busy


@pytest.mark.parametrize("stderr", [
    "Cannot connect to the Docker daemon\n",
    "Error: No such object: unrelated\n",
    "Error: No such object: gone\npermission denied\n",
])
def test_container_probe_does_not_hide_other_inspection_failures(monkeypatch, stderr):
    def output(argv, **kwargs):
        raise _MODULE.subprocess.CalledProcessError(1, argv, output="[]", stderr=stderr)
    monkeypatch.setattr(_MODULE.subprocess, "check_output", output)
    with pytest.raises(_MODULE.subprocess.CalledProcessError):
        _MODULE._inspect_containers(["gone"])


def test_orphan_cleanup_only_removes_same_session_agent_server(tmp_path, monkeypatch):
    workroot = tmp_path / "session"
    other = tmp_path / "other"
    outputs = iter(["same-agent other-agent evaluator", json.dumps([
        {"Id": "same-agent", "Name": "/agent-server-one", "State": {"Running": True},
         "Mounts": [{"Source": str(workroot)}]},
        {"Id": "other-agent", "Name": "/agent-server-two", "State": {"Running": True},
         "Mounts": [{"Source": str(other)}]},
        {"Id": "evaluator", "Name": "/task-evaluator", "State": {"Running": True},
         "Mounts": [{"Source": str(workroot)}]},
    ])])
    monkeypatch.setattr(_MODULE.subprocess, "check_output", lambda *a, **kw: next(outputs))
    calls = []
    monkeypatch.setattr(_MODULE.subprocess, "run", lambda *a, **kw: calls.append((a, kw)))

    _MODULE._remove_orphan_agent_containers(str(workroot))

    assert calls == [((['docker', 'rm', '-f', 'same-agent'],), {
        'capture_output': True, 'text': True, 'check': True, 'timeout': 60})]


def test_uncertain_attempt_is_archived_only_after_owning_sandbox_is_gone(
        tmp_path, monkeypatch):
    workroot = tmp_path / "session"
    workroot.mkdir()
    log = tmp_path / "attempts.jsonl"
    attempt = "attempt-123"
    log.write_text(json.dumps({"event": "launch", "attempt_id": attempt}) + "\n")
    marker = workroot / ".autoresearch-uncertain-execution.json"
    marker.write_text(json.dumps({
        "attempt_id": attempt, "cwd": str(workroot / "repo"),
        "log_path": str(log), "requires_reconciliation": True,
    }))
    outputs = iter(["active", json.dumps([{
        "Id": "active", "State": {"Running": True},
        "Mounts": [{"Source": str(workroot)}],
    }])])
    monkeypatch.setattr(_MODULE.subprocess, "check_output", lambda *a, **kw: next(outputs))
    assert not _MODULE._resolve_uncertain_attempt_without_owner(str(workroot))
    assert marker.exists()

    monkeypatch.setattr(_MODULE.subprocess, "check_output", lambda *a, **kw: "")
    assert _MODULE._resolve_uncertain_attempt_without_owner(str(workroot))
    assert not marker.exists()
    archived = workroot / f".autoresearch-uncertain-execution.resolved-{attempt}.json"
    assert archived.is_file()
    assert "exclusive session lock" in json.loads(archived.read_text())["resolution"]


def test_uncertain_attempt_without_matching_launch_remains_blocked(tmp_path, monkeypatch):
    workroot = tmp_path / "session"
    workroot.mkdir()
    log = tmp_path / "attempts.jsonl"
    log.write_text(json.dumps({"event": "launch", "attempt_id": "other"}) + "\n")
    marker = workroot / ".autoresearch-uncertain-execution.json"
    marker.write_text(json.dumps({
        "attempt_id": "attempt-missing", "cwd": str(workroot / "repo"),
        "log_path": str(log), "requires_reconciliation": True,
    }))
    monkeypatch.setattr(_MODULE.subprocess, "check_output", lambda *a, **kw: "")
    assert not _MODULE._resolve_uncertain_attempt_without_owner(str(workroot))
    assert marker.exists()


def test_gpu_lock_probe_failure_is_not_treated_as_free(tmp_path, monkeypatch):
    def fail(cards):
        raise RuntimeError("Docker unavailable")
    monkeypatch.setattr(_MODULE, "_container_uses_cards", fail)
    with pytest.raises(RuntimeError, match="Docker unavailable"):
        _MODULE._acquire_gpu_locks(str(tmp_path / "session"), "2", blocking=False)
    monkeypatch.setattr(_MODULE, "_container_uses_cards", lambda cards: False)
    handles = _MODULE._acquire_gpu_locks(str(tmp_path / "session"), "2", blocking=False)
    for handle in handles:
        handle.close()


def test_decode_execution_contract_includes_confirmed_boundary():
    """Catch the DGX driver dropping the fairness payload exported by the runner."""
    boundary = {"budget_s": 300, "accelerator_count": 1}
    encoded = base64.b64encode(json.dumps(boundary).encode()).decode()

    phase, frozen, decoded_boundary, semantics = _decode_execution_contract({
        "AR_EXECUTION_PHASE": "baseline",
        "AR_CONFIRMED_BOUNDARY_B64": encoded,
    })

    assert phase == "baseline"
    assert frozen is None
    assert decoded_boundary == boundary
    assert semantics is None


def test_decode_execution_contract_round_trips_evaluation_semantics():
    semantics = {
        "comparison_mode": "quality_at_fixed_time",
        "conditions_to_hold_constant": ["training time", "hardware"],
        "method_freedom": ["optimizer"],
    }
    encoded = base64.b64encode(json.dumps(semantics).encode()).decode()

    phase, frozen, boundary, decoded = _decode_execution_contract({
        "AR_EXECUTION_PHASE": "baseline",
        "AR_EVALUATION_SEMANTICS_B64": encoded,
    })

    assert (phase, frozen, boundary) == ("baseline", None, None)
    assert decoded == semantics


@pytest.mark.parametrize("encoded", ["not-base64!", base64.b64encode(b"[]").decode()])
def test_decode_execution_contract_rejects_invalid_evaluation_semantics(encoded):
    with pytest.raises(ValueError, match="evaluation semantics"):
        _decode_execution_contract({
            "AR_EXECUTION_PHASE": "baseline",
            "AR_EVALUATION_SEMANTICS_B64": encoded,
        })


def test_explicit_primary_repo_is_not_inferred_from_reference_urls():
    primary = "https://github.com/example/confirmed-repo"
    encoded = base64.b64encode(primary.encode()).decode()
    question = (
        "Reference https://social.example/author/status/123 and "
        "https://example.com/articles/result before running the confirmed repository"
    )

    repo_url, tool_urls = _select_repo_urls(
        question, {"AR_PRIMARY_REPO_B64": encoded})

    assert repo_url == primary
    assert tool_urls == ()


def test_decode_memory_scope_round_trips_method_identity():
    method_ref = "spec://method/session/sha256-current"
    encoded = base64.b64encode(method_ref.encode()).decode()

    assert _decode_memory_scope({"AR_METHOD_SCOPE_B64": encoded}) == method_ref


@pytest.mark.parametrize("encoded", ["not-base64!", base64.b64encode(b"   ").decode()])
def test_decode_memory_scope_rejects_invalid_or_empty_values(encoded):
    with pytest.raises(ValueError, match="method memory scope"):
        _decode_memory_scope({"AR_METHOD_SCOPE_B64": encoded})


def test_decode_prior_worktree_round_trips_path():
    path = "/srv/dagloop/ar_gpu_runs/agentic_remote/sess-abc/repo-3"
    encoded = base64.b64encode(path.encode()).decode()
    assert _decode_prior_worktree({"AR_PRIOR_WORKTREE_B64": encoded}) == path


def test_decode_prior_worktree_round_trips_json_encoded_parent_path():
    path = "/work/session/checkpoint with spaces"
    encoded = base64.b64encode(json.dumps(path).encode()).decode()
    assert _decode_prior_worktree({"AR_PRIOR_WORKTREE_B64": encoded}) == path


@pytest.mark.parametrize("path", ["", "   "])
def test_decode_prior_worktree_rejects_empty_json_encoded_path(path):
    encoded = base64.b64encode(json.dumps(path).encode()).decode()
    with pytest.raises(ValueError, match="empty value"):
        _decode_prior_worktree({"AR_PRIOR_WORKTREE_B64": encoded})


@pytest.mark.parametrize("paths", [[], ["/work/repo-e", "/work/repo-f", "/work/repo-g"]])
def test_decode_prior_worktree_round_trips_dependency_set(paths):
    encoded = base64.b64encode(json.dumps(paths).encode()).decode()
    assert _decode_prior_worktree({"AR_PRIOR_WORKTREE_B64": encoded}) == paths


def test_decode_prior_worktree_round_trips_merge_bases():
    payload = {"checkpoints": ["/work/left", "/work/right"],
               "merge_bases": ["/work/root"]}
    encoded = base64.b64encode(json.dumps(payload).encode()).decode()
    assert _decode_prior_worktree({"AR_PRIOR_WORKTREE_B64": encoded}) == payload
