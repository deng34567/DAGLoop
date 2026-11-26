"""_ssh_dispatch: connection-multiplexed, backoff-retrying SSH for the GPU dispatch (#47)."""
import threading
import time
import hashlib
import io
import os
import sys
import base64
import json
import re
import shlex
import subprocess
from types import SimpleNamespace

import pytest

from autoresearch.agents.runner_agent import handler


@pytest.fixture(autouse=True)
def _configured_remote_agentic_python(monkeypatch):
    monkeypatch.setenv(
        "AUTORESEARCH_REMOTE_PY",
        "/srv/dagloop/auto-research/.venv-py312/bin/python",
    )


def _fake_proc(rc, out="", err=""):
    return SimpleNamespace(returncode=rc, stdout=out, stderr=err)


def _patch_run(monkeypatch, results):
    """Feed _ssh_dispatch a scripted sequence of subprocess.run results; record the argv/sleeps."""
    calls, sleeps = [], []
    seq = iter(results)

    def fake_run(argv, **kw):
        calls.append((argv, kw))
        return next(seq)

    import subprocess
    import time
    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(time, "sleep", lambda s: sleeps.append(s))
    return calls, sleeps


def test_uses_connection_multiplexing(monkeypatch):
    calls, _ = _patch_run(monkeypatch, [_fake_proc(0, out="RESULT_JSON {}")])
    handler._ssh_dispatch("gpu-worker-04", "echo hi", timeout=100)
    argv = calls[0][0]
    assert "ControlMaster=auto" in argv                      # shared master -> no MaxStartups storm
    assert any(a.startswith("ControlPath=") for a in argv)
    assert any(a.startswith("ControlPersist=") for a in argv)


def test_explicit_ssh_config_covers_jump_host_and_fallback(monkeypatch):
    calls, _ = _patch_run(monkeypatch, [_fake_proc(255), _fake_proc(0, out="RESULT_JSON {}")])
    monkeypatch.setenv("AUTORESEARCH_SSH_CONFIG", "/deployment/ssh.conf")
    handler._ssh_dispatch("example-host", "run", timeout=None)
    assert len(calls) == 2
    for argv, kwargs in calls:
        assert argv[1:3] == ["-F", "/deployment/ssh.conf"]
        assert kwargs["timeout"] is None


def test_long_dispatch_uses_stdin_not_one_oversized_ssh_argument(monkeypatch):
    script = "echo " + "x" * 300000
    calls, _ = _patch_run(monkeypatch, [_fake_proc(0)])
    handler._ssh_dispatch("example-host", script, timeout=10)
    argv, kwargs = calls[0]
    assert argv[-1] == "bash -s"
    assert max(map(len, argv)) < 1024
    assert kwargs["input"] == script + "\n"


def test_retries_on_kex_drop_then_succeeds(monkeypatch):
    # Try the multiplexed and independent transports before backing off.
    calls, sleeps = _patch_run(monkeypatch, [
        _fake_proc(255, out=""), _fake_proc(255, out=""), _fake_proc(0, out="RESULT_JSON {}")])
    proc = handler._ssh_dispatch("gpu-worker-04", "run", timeout=100)
    assert proc.returncode == 0 and "RESULT_JSON" in proc.stdout
    assert len(calls) == 3          # two drops + one success
    assert sleeps == [1]


def test_mux_channel_refusal_falls_back_to_independent_connection(monkeypatch):
    calls, sleeps = _patch_run(monkeypatch, [
        _fake_proc(255, err="mux_client_request_session: session request failed: "
                            "Session open refused by peer"),
        _fake_proc(0, out="RESULT_JSON {}"),
    ])

    proc = handler._ssh_dispatch("gpu-worker-04", "run", timeout=100)

    assert proc.returncode == 0
    assert "ControlMaster=auto" in calls[0][0]
    assert "ControlMaster=no" in calls[1][0]
    assert "ControlPath=none" in calls[1][0]
    assert sleeps == []


def test_empty_transport_drop_also_tries_independent_connection(monkeypatch):
    calls, sleeps = _patch_run(monkeypatch, [
        _fake_proc(255),
        _fake_proc(0, out="RESULT_JSON {}"),
    ])

    proc = handler._ssh_dispatch("gpu-worker-04", "run", timeout=100)

    assert proc.returncode == 0
    assert "ControlMaster=no" in calls[1][0]
    assert sleeps == []


def test_no_retry_when_command_produced_output(monkeypatch):
    # exit 255 but WITH stdout == the remote command ran; retrying could double-run -> must NOT retry.
    calls, sleeps = _patch_run(monkeypatch, [_fake_proc(255, out="partial RESULT_JSON")])
    proc = handler._ssh_dispatch("gpu-worker-04", "run", timeout=100)
    assert proc.returncode == 255
    assert len(calls) == 1 and sleeps == []


def test_timeout_cleans_up_only_the_exact_remote_session(monkeypatch):
    cleanup = []

    def timeout_run(*args, **kwargs):
        raise subprocess.TimeoutExpired(kwargs.get("args", args[0]), kwargs.get("timeout"))

    monkeypatch.setattr(subprocess, "run", timeout_run)
    monkeypatch.setattr(
        handler, "_cancel_remote_session",
        lambda host, sid: cleanup.append((host, sid)),
    )

    with pytest.raises(subprocess.TimeoutExpired):
        handler._ssh_dispatch(
            "gpu-worker-03",
            "export AR_SESSION_ID='086f71eb-9f74-4688-af96-a24d52b09367'; run",
            timeout=29100,
        )

    assert cleanup == [("gpu-worker-03", "086f71eb-9f74-4688-af96-a24d52b09367")]


def test_timeout_without_session_id_does_not_run_host_cleanup(monkeypatch):
    cleanup = []
    monkeypatch.setattr(
        subprocess, "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            subprocess.TimeoutExpired(args[0], kwargs.get("timeout"))),
    )
    monkeypatch.setattr(
        handler, "_cancel_remote_session",
        lambda host, sid: cleanup.append((host, sid)),
    )

    with pytest.raises(subprocess.TimeoutExpired):
        handler._ssh_dispatch("gpu-worker-03", "cleanup", timeout=60)

    assert cleanup == []


@pytest.mark.parametrize("base", [None, "/work/current campaign/sessions"])
def test_cancel_remote_session_uses_dispatch_workroot(monkeypatch, base):
    if base is None:
        monkeypatch.delenv("AUTORESEARCH_REMOTE_SESSION_ROOT", raising=False)
    else:
        monkeypatch.setenv("AUTORESEARCH_REMOTE_SESSION_ROOT", base)
    calls, _ = _patch_run(monkeypatch, [_fake_proc(0)])
    sid = "session-to-cancel"
    handler._cancel_remote_session("example-host", sid)
    command = shlex.split(calls[0][1]["input"])
    expected_base = base or "/srv/dagloop/ar_gpu_runs/agentic_remote"
    assert command[-2:] == [sid, expected_base + "/sess-" + hashlib.sha1(sid.encode()).hexdigest()[:12]]


def test_remote_cleanup_parses_bytes_and_touches_only_owned_processes_and_containers(monkeypatch):
    monkeypatch.setenv("AUTORESEARCH_REMOTE_SESSION_ROOT", "/work/current/sessions")
    calls, _ = _patch_run(monkeypatch, [_fake_proc(0)])
    sid = "owned-session"
    handler._cancel_remote_session("example-host", sid)
    command = shlex.split(calls[0][1]["input"])
    script, root = command[2], command[-1]
    killed, removed = [], []
    monkeypatch.setattr(sys, "argv", ["cleanup", sid, root])
    monkeypatch.setattr(os, "listdir", lambda _: ["11", "22"])
    monkeypatch.setattr(os, "kill", lambda pid, signal: killed.append(pid))
    def environ(path, mode):
        value = sid if "/11/" in path else "other-session"
        return io.BytesIO(("AR_SESSION_ID=" + value + "\0OTHER=value\0").encode())
    def output(argv, **kwargs):
        if argv[:3] == ["docker", "ps", "-q"]:
            return "owned-container\nother-container\n"
        return root + "\n" if argv[-1] == "owned-container" else "/work/other/sess-other\n"
    monkeypatch.setattr(subprocess, "check_output", output)
    monkeypatch.setattr(subprocess, "run", lambda argv, **kwargs: removed.append(argv) or _fake_proc(0))

    exec(script, {"open": environ})

    assert killed == [11]
    assert removed == [["docker", "rm", "-f", "owned-container"]]


def test_gives_up_after_max_retries(monkeypatch):
    monkeypatch.setenv("AUTORESEARCH_SSH_RETRIES", "3")
    calls, sleeps = _patch_run(monkeypatch, [_fake_proc(255, out="")] * 6)
    proc = handler._ssh_dispatch("gpu-worker-04", "run", timeout=100)
    assert proc.returncode == 255   # exhausted -> returns the last failing proc (caller falls back)
    assert len(calls) == 6 and sleeps == [1, 2]   # 3 mux+direct tries, 2 backoffs


def test_ssh_subprocess_cannot_inherit_proxy(monkeypatch):
    monkeypatch.setenv("HTTP_PROXY", "http://metered-proxy")
    monkeypatch.setenv("https_proxy", "http://metered-proxy")
    calls, _ = _patch_run(monkeypatch, [_fake_proc(0, out="ok")])

    handler._ssh_dispatch("gpu-worker-04", "true", timeout=10)

    env = calls[0][1]["env"]
    assert "HTTP_PROXY" not in env and "https_proxy" not in env
    assert env["NO_PROXY"] == "*" and env["no_proxy"] == "*"


def test_remote_campaign_paths_are_forwarded(monkeypatch):
    for key in ("AUTORESEARCH_REMOTE_SESSION_ROOT", "AUTORESEARCH_GPU_LOCK_DIR"):
        monkeypatch.setenv(key, "/work/separate campaign")
        assert handler._remote_agentic_env()[key] == "/work/separate campaign"


# --- session cap + OpenRSI式僵尸清理(会议 2026-08-06 老师) --------------------------------

def test_session_cap_default_and_env(monkeypatch):
    monkeypatch.delenv("AUTORESEARCH_AGENTIC_SESSION_CAP_S", raising=False)
    assert handler._session_cap_s() == 7200          # 默认 2h(取代旧 4h 死等)
    monkeypatch.setenv("AUTORESEARCH_AGENTIC_SESSION_CAP_S", "3600")
    assert handler._session_cap_s() == 3600
    monkeypatch.setenv("AUTORESEARCH_AGENTIC_SESSION_CAP_S", "junk")
    assert handler._session_cap_s() == 7200           # 坏值兜底


def test_cleanup_sweeps_containers_and_procs(monkeypatch):
    calls = {}
    def fake_dispatch(host, script, timeout):
        calls["host"] = host; calls["script"] = script; calls["timeout"] = timeout
        return SimpleNamespace(returncode=0, stdout="", stderr="")
    monkeypatch.setattr(handler, "_ssh_dispatch", fake_dispatch)
    monkeypatch.setenv("AUTORESEARCH_GPU_CLEANUP", "1")
    for key, value in (
        ("AUTORESEARCH_AGENTIC_SESSION_CAP_S", "7200"),
        ("AUTORESEARCH_AGENTIC_HARD_CAP_S", "14400"),
        ("AUTORESEARCH_SANDBOX_CONVERSATION_TIMEOUT", "10800"),
        ("AUTORESEARCH_GPU_CLEANUP_GRACE_S", "300"),
    ):
        monkeypatch.setenv(key, value)
    handler._cleanup_gpu_host("gpu-worker-04")
    s = calls["script"]
    assert calls["host"] == "gpu-worker-04"
    assert "docker rm -f" in s and "agent-server" in s      # 清超龄容器
    assert "/srv/dagloop/ar_gpu_runs/agentic_remote/" in s  # 只清本项目 mount
    assert "/srv/dagloop/ar_gpu_runs/" in s and "agentic_run.py" in s  # 只清本项目进程
    assert "cap=14700" in s                                  # max(session/hard/conversation)+grace


def test_cleanup_disabled_by_env(monkeypatch):
    called = {"n": 0}
    monkeypatch.setattr(handler, "_ssh_dispatch", lambda *a, **k: called.__setitem__("n", called["n"]+1))
    monkeypatch.setenv("AUTORESEARCH_GPU_CLEANUP", "0")
    handler._cleanup_gpu_host("gpu-worker-04")
    assert called["n"] == 0                                  # 开关关 -> 不清


def test_cleanup_fail_open(monkeypatch):
    def boom(*a, **k): raise RuntimeError("ssh down")
    monkeypatch.setattr(handler, "_ssh_dispatch", boom)
    monkeypatch.setenv("AUTORESEARCH_GPU_CLEANUP", "1")
    handler._cleanup_gpu_host("gpu-worker-04")                     # 不抛(best-effort)


# --- multi-GPU card claim: AUTORESEARCH_REPO_ALL_GPUS drains the whole pool (8-card speedrun) ---

def test_claim_cards_single_by_default(monkeypatch):
    monkeypatch.delenv("AUTORESEARCH_REPO_ALL_GPUS", raising=False)
    claimed = handler._claim_cards()
    try:
        assert len(claimed) == 1                      # 默认每任务 1 卡(并发跨卡池并行)
    finally:
        for c in claimed:
            handler._REMOTE_GPU_POOL.put(c)


def test_gpus_for_task_uses_explicit_hardware_count_from_research_semantics(monkeypatch):
    monkeypatch.delenv("AUTORESEARCH_REPO_GPUS_MAP", raising=False)
    monkeypatch.delenv("AUTORESEARCH_REPO_ALL_GPUS", raising=False)
    monkeypatch.setattr(handler, "_REMOTE_GPU_CARDS", [str(i) for i in range(8)])

    assert handler._gpus_for_task(
        "same node; keep 8×H100 and the canonical evaluator fixed") == 8


def test_gpus_for_task_prefers_repository_contract_over_incidental_paper_hardware(monkeypatch):
    monkeypatch.delenv("AUTORESEARCH_REPO_GPUS_MAP", raising=False)
    monkeypatch.delenv("AUTORESEARCH_REPO_ALL_GPUS", raising=False)
    monkeypatch.setattr(handler, "_REMOTE_GPU_CARDS", [str(i) for i in range(8)])

    assert handler._gpus_for_task(
        "environment.cpus=8, memory_mb=32768, gpus=1, gpu_type=L40S; "
        "a cited comparison used 14 H100 accelerators") == 1


def test_gpus_for_task_rejects_unsatisfied_explicit_hardware_count(monkeypatch):
    monkeypatch.delenv("AUTORESEARCH_REPO_GPUS_MAP", raising=False)
    monkeypatch.delenv("AUTORESEARCH_REPO_ALL_GPUS", raising=False)
    monkeypatch.setattr(handler, "_REMOTE_GPU_CARDS", ["0", "1", "2", "3"])

    with pytest.raises(handler.core.RunnerParseError, match="requires 8 accelerators"):
        handler._gpus_for_task("run on 8 H100 GPUs")


def test_gpus_for_task_does_not_parse_url_hex_as_accelerator_count(monkeypatch):
    monkeypatch.delenv("AUTORESEARCH_REPO_GPUS_MAP", raising=False)
    monkeypatch.delenv("AUTORESEARCH_REPO_ALL_GPUS", raising=False)
    monkeypatch.setattr(handler, "_REMOTE_GPU_CARDS", ["0", "1"])

    assert handler._gpus_for_task(
        "Evidence: https://collections.example.edu/dl_files/1b/7a/1b7a6c407884.pdf"
    ) == 1


def test_claim_cards_all_gpus_drains_pool(monkeypatch):
    monkeypatch.setenv("AUTORESEARCH_REPO_ALL_GPUS", "1")
    claimed = handler._claim_cards()
    try:
        assert sorted(claimed) == sorted(handler._REMOTE_GPU_CARDS)   # 排空整池=独占全部卡
        assert handler._REMOTE_GPU_POOL.empty()                       # 池被抽干
    finally:
        for c in claimed:
            handler._REMOTE_GPU_POOL.put(c)


def test_multi_card_claim_is_atomic_between_threads(monkeypatch):
    """Two 2-card tasks must not interleave one card each and deadlock forever."""
    class RecordingPool:
        def __init__(self):
            self.calls = []
            self._next = 0

        def get(self):
            self.calls.append(threading.get_ident())
            if len(self.calls) == 1:
                time.sleep(0.08)  # give the competing claimant a chance to interleave
            value = str(self._next)
            self._next += 1
            return value

    pool = RecordingPool()
    monkeypatch.setattr(handler, "_REMOTE_GPU_POOL", pool)
    monkeypatch.setattr(handler, "_gpus_for_task", lambda _hint="": 2)
    threads = [threading.Thread(target=handler._claim_cards, args=("task",)) for _ in range(2)]
    threads[0].start()
    time.sleep(0.01)
    threads[1].start()
    for thread in threads:
        thread.join(timeout=1)

    assert pool.calls[:2] == [pool.calls[0], pool.calls[0]]


def test_multi_card_claim_does_not_reserve_partial_pool(monkeypatch):
    """A 2-card request must leave one available card usable while it waits."""
    import queue

    pool = queue.Queue()
    pool.put("0")
    monkeypatch.setattr(handler, "_REMOTE_GPU_POOL", pool)
    monkeypatch.setattr(handler, "_gpus_for_task", lambda _hint="": 2)
    claimed = []
    thread = threading.Thread(
        target=lambda: claimed.extend(handler._claim_cards("two-card task")))
    thread.start()
    time.sleep(0.08)

    assert pool.qsize() == 1
    assert claimed == []

    pool.put("1")
    thread.join(timeout=1)
    assert not thread.is_alive()
    assert claimed == ["0", "1"]


def test_small_claim_can_use_free_card_while_large_claim_waits(monkeypatch):
    """An unsatisfied large request must not head-of-line block a fitting job."""
    import queue

    pool = queue.Queue()
    pool.put("0")
    monkeypatch.setattr(handler, "_REMOTE_GPU_POOL", pool)
    monkeypatch.setattr(
        handler, "_gpus_for_task", lambda hint="": 2 if hint == "large" else 1)
    large, small = [], []
    large_thread = threading.Thread(
        target=lambda: large.extend(handler._claim_cards("large")))
    large_thread.start()
    time.sleep(0.08)
    small_thread = threading.Thread(
        target=lambda: small.extend(handler._claim_cards("small")))
    small_thread.start()
    small_thread.join(timeout=1)

    assert small == ["0"]
    assert large == []

    pool.put("0")
    pool.put("1")
    large_thread.join(timeout=1)
    assert not large_thread.is_alive()
    assert large == ["0", "1"]


@pytest.mark.parametrize("effort", ["default", "max"])
def test_remote_agentic_env_propagates_runtime_limits_and_direct_network(monkeypatch, effort):
    monkeypatch.setenv("AUTORESEARCH_LLM_TEMPERATURE", "0.7")
    monkeypatch.setenv("VIBE_REASONING_EFFORT", effort)
    monkeypatch.setenv("AUTORESEARCH_SANDBOX_REASONING_EFFORT", effort)
    monkeypatch.setenv("HKGAI_HTTP_TIMEOUT", "1800")
    monkeypatch.setenv("HKGAI_STREAM_MAX_S", "1800")
    monkeypatch.setenv("HKGAI_STREAM_NO_CONTENT_S", "1800")
    monkeypatch.setenv("HKGAI_STREAM_RETRY_BASE_S", "300")
    monkeypatch.setenv("HKGAI_STREAM_RETRY_CAP_S", "300")
    monkeypatch.setenv("HKGAI_STREAM_RETRIES", "3")
    monkeypatch.setenv("AUTORESEARCH_AGENTIC_PER_RUN_S", "1800")
    monkeypatch.setenv("AUTORESEARCH_AGENTIC_HARD_CAP_S", "14400")
    monkeypatch.setenv("AUTORESEARCH_SANDBOX_CONVERSATION_TIMEOUT", "10800")
    monkeypatch.setenv("AUTORESEARCH_SANDBOX_MODEL", "gpt-5.5")
    monkeypatch.setenv("AUTORESEARCH_SANDBOX_NATIVE_TOOL_CALLING", "0")
    monkeypatch.setenv("AUTORESEARCH_SANDBOX_IMAGE", "local-task-runtime:verified")
    monkeypatch.setenv("AUTORESEARCH_SANDBOX_LLM_TIMEOUT", "1200")
    monkeypatch.setenv("AUTORESEARCH_SANDBOX_LLM_RETRIES", "2")
    monkeypatch.setenv("AUTORESEARCH_SANDBOX_LLM_STREAM", "1")
    monkeypatch.setenv("AUTORESEARCH_SANDBOX_CONTROL_MEMORY_MB", "512")
    monkeypatch.setenv("AUTORESEARCH_SANDBOX_PKG_CACHE_DIR", "/srv/dagloop/ar_pkg_cache")
    monkeypatch.setenv("AUTORESEARCH_SANDBOX_DATA_ROOT", "/srv/dagloop/ar_data")

    env = handler._remote_agentic_env()
    assert env["AUTORESEARCH_LLM_TEMPERATURE"] == "0.7"
    assert env["VIBE_REASONING_EFFORT"] == effort
    assert env["AUTORESEARCH_SANDBOX_REASONING_EFFORT"] == effort
    assert env["HKGAI_HTTP_TIMEOUT"] == "1800"
    assert env["HKGAI_STREAM_MAX_S"] == "1800"
    assert env["HKGAI_STREAM_NO_CONTENT_S"] == "1800"
    assert env["HKGAI_STREAM_RETRY_BASE_S"] == "300"
    assert env["HKGAI_STREAM_RETRY_CAP_S"] == "300"
    assert env["HKGAI_STREAM_RETRIES"] == "3"

    assert env["AUTORESEARCH_AGENTIC_PER_RUN_S"] == "1800"
    assert env["AUTORESEARCH_AGENTIC_HARD_CAP_S"] == "14400"
    assert env["AUTORESEARCH_SANDBOX_CONVERSATION_TIMEOUT"] == "10800"
    assert env["AUTORESEARCH_SANDBOX_MODEL"] == "gpt-5.5"
    assert env["AUTORESEARCH_SANDBOX_NATIVE_TOOL_CALLING"] == "0"
    assert env["AUTORESEARCH_SANDBOX_IMAGE"] == "local-task-runtime:verified"
    assert env["AUTORESEARCH_SANDBOX_LLM_TIMEOUT"] == "1200"
    assert env["AUTORESEARCH_SANDBOX_LLM_RETRIES"] == "2"
    assert env["AUTORESEARCH_SANDBOX_LLM_STREAM"] == "1"
    assert env["AUTORESEARCH_SANDBOX_CONTROL_MEMORY_MB"] == "512"
    assert env["AUTORESEARCH_SANDBOX_PKG_CACHE_DIR"] == "/srv/dagloop/ar_pkg_cache"
    assert env["AUTORESEARCH_SANDBOX_DATA_ROOT"] == "/srv/dagloop/ar_data"
    assert env["NO_PROXY"] == "*" and env["no_proxy"] == "*"
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
                "http_proxy", "https_proxy", "all_proxy"):
        assert env[key] == ""


@pytest.mark.parametrize("reservation, expected_card", [
    ("", "0"),
    ("GPU_RESERVED cards=3\n", "3"),
    ("GPU_RESERVED cards=2,5\n", "2,5"),
])
def test_remote_agentic_dispatch_exports_confirmed_boundary(monkeypatch, reservation, expected_card):
    """Catch the DGX baseline phase losing user fairness requirements at SSH dispatch."""
    captured = {}
    monkeypatch.setenv("AUTORESEARCH_AGENTIC_IMPROVE", "1")
    monkeypatch.setattr(handler, "_cleanup_gpu_host", lambda _host: None)
    monkeypatch.setattr(handler, "_claim_cards", lambda _question="": ["0"])
    monkeypatch.setattr(handler, "_REMOTE_GPU_POOL", SimpleNamespace(put=lambda _card: None))

    def fake_dispatch(_host, command, timeout):
        captured["command"] = command
        result = {
            "benchmark": "repo", "metric": "loss", "lower_is_better": True,
            "repo_commit": "abc", "verdict": {"verdict": "pass", "improved": False},
            "baseline": {"metric": 1.0, "valid": True},
            "candidate": {"metric": None, "valid": False},
            "report": "baseline", "config": "steps=10",
        }
        return _fake_proc(0, out=reservation + "RESULT_JSON " + json.dumps(result))

    monkeypatch.setattr(handler, "_ssh_dispatch", fake_dispatch)
    runner = handler.RunnerAgentHandler(spec_store=None, backend=object())
    boundary = {"budget_s": 300, "accelerator_count": 1}

    record, _ = runner._run_repo_experiment_remote(
        "sid", "Improve https://github.com/example/repo", overrides=boundary,
        execution_phase="baseline",
    )

    match = re.search(r"AR_CONFIRMED_BOUNDARY_B64=([^ ;]+)", captured["command"])
    assert match is not None
    encoded = match.group(1).strip("'\"")
    assert json.loads(base64.b64decode(encoded).decode()) == boundary
    assert record["environment"].endswith("card" + expected_card)


@pytest.mark.parametrize("prior", [
    {"checkpoints": ["/work/left", "/work/right"], "merge_bases": ["/work/root"]},
    "/work/named checkpoint",
    [],  # Root component: explicitly compare with the repository baseline.
    "",  # Non-component run: no dependency-diff request.
])
@pytest.mark.parametrize("mode", ["component", "integration"])
def test_remote_dispatch_exports_dependency_merge_bases(monkeypatch, prior, mode):
    monkeypatch.setenv("AUTORESEARCH_AGENTIC_IMPROVE", "1")
    monkeypatch.setenv("AUTORESEARCH_REPO_REMOTE_DIR", "/work/versioned bundle")
    monkeypatch.setattr(handler, "_cleanup_gpu_host", lambda _host: None)
    monkeypatch.setattr(handler, "_claim_cards", lambda _question="": ["0"])
    monkeypatch.setattr(handler, "_REMOTE_GPU_POOL", SimpleNamespace(put=lambda _card: None))

    def fake_dispatch(_host, command, timeout):
        from scripts.dgx.agentic_run import _decode_prior_worktree
        assert f"AR_COMPONENT_STAGE={int(mode == 'component')}" in command
        assert "cd '/work/versioned bundle' || exit" in command
        assert "'/work/versioned bundle/agentic_run.py'" in command
        match = re.search(r"AR_PRIOR_WORKTREE_B64=([^ ;]+)", command)
        if prior:
            assert match is not None
            assert _decode_prior_worktree({
                "AR_PRIOR_WORKTREE_B64": match.group(1).strip("'\"")}) == prior
        else:
            assert match is None
        parent_match = re.search(r"AR_COMPONENT_PARENT_B64=([^ ;]+)", command)
        if prior or prior == []:
            assert parent_match is not None
            payload = parent_match.group(1).strip("'\"")
            assert json.loads(base64.b64decode(payload)) == prior
            assert _decode_prior_worktree({"AR_PRIOR_WORKTREE_B64": payload}) == prior
        else:
            assert parent_match is None
        result = {
            "benchmark": "repo", "metric": "loss", "lower_is_better": True,
            "repo_commit": "abc", "verdict": {"verdict": "pass", "improved": False},
            "baseline": {"metric": 1.0, "valid": True},
            "candidate": {"metric": None, "valid": False},
            "report": "method", "config": "steps=10",
        }
        return _fake_proc(0, out="RESULT_JSON " + json.dumps(result))

    monkeypatch.setattr(handler, "_ssh_dispatch", fake_dispatch)
    runner = handler.RunnerAgentHandler(spec_store=None, backend=object())
    runner._active_component_context = {"sid": {"mode": mode}}
    monkeypatch.setattr(runner, "_with_operator", lambda _sid, question: question)
    monkeypatch.setattr(runner, "_prior_worktree", lambda _sid, **kwargs: prior)
    runner._run_repo_experiment_remote(
        "sid", "Improve https://github.com/example/repo", execution_phase="method",
        frozen_baseline={"metric_value": 1.0})


def test_remote_dispatch_waits_for_sandbox_finalization_without_extending_run(monkeypatch):
    monkeypatch.setenv("AUTORESEARCH_AGENTIC_IMPROVE", "1")
    for key in ("AUTORESEARCH_AGENTIC_SESSION_CAP_S", "AUTORESEARCH_AGENTIC_HARD_CAP_S",
                "AUTORESEARCH_SANDBOX_CONVERSATION_TIMEOUT"):
        monkeypatch.setenv(key, "28800")
    monkeypatch.setenv("AUTORESEARCH_GPU_CLEANUP_GRACE_S", "300")
    monkeypatch.setattr(handler, "_cleanup_gpu_host", lambda _host: None)
    monkeypatch.setattr(handler, "_claim_cards", lambda _question="": ["0"])
    released = []
    monkeypatch.setattr(handler, "_REMOTE_GPU_POOL", SimpleNamespace(put=released.append))

    def fake_dispatch(_host, command, timeout):
        # The transport must still be attached while the remote 8h run finalizes.
        assert timeout == 29100
        assert released == []
        assert "AUTORESEARCH_AGENTIC_HARD_CAP_S=28800" in command
        assert "AUTORESEARCH_SANDBOX_CONVERSATION_TIMEOUT=28800" in command
        result = {
            "benchmark": "repo", "metric": "loss", "lower_is_better": True,
            "repo_commit": "abc", "verdict": {"verdict": "pass", "improved": False},
            "baseline": {"metric": 1.0, "valid": True},
            "candidate": {"metric": None, "valid": False},
            "report": "baseline", "config": "steps=10",
        }
        return _fake_proc(0, out="RESULT_JSON " + json.dumps(result))

    monkeypatch.setattr(handler, "_ssh_dispatch", fake_dispatch)
    runner = handler.RunnerAgentHandler(spec_store=None, backend=object())
    runner._run_repo_experiment_remote(
        "sid", "Improve https://github.com/example/repo", execution_phase="baseline")
    assert released == ["0"]


def test_remote_agentic_dispatch_exports_evaluation_semantics_as_data(monkeypatch):
    captured = {}
    monkeypatch.setenv("AUTORESEARCH_AGENTIC_IMPROVE", "1")
    monkeypatch.setattr(handler, "_cleanup_gpu_host", lambda _host: None)
    monkeypatch.setattr(handler, "_claim_cards", lambda _question="": ["0"])
    monkeypatch.setattr(handler, "_REMOTE_GPU_POOL", SimpleNamespace(put=lambda _card: None))

    def fake_dispatch(_host, command, timeout):
        captured["command"] = command
        result = {
            "benchmark": "repo", "metric": "loss", "lower_is_better": True,
            "repo_commit": "abc", "verdict": {"verdict": "pass", "improved": False},
            "baseline": {"metric": 1.0, "valid": True},
            "candidate": {"metric": None, "valid": False},
            "report": "baseline", "config": "steps=10",
        }
        return _fake_proc(0, out="RESULT_JSON " + json.dumps(result))

    monkeypatch.setattr(handler, "_ssh_dispatch", fake_dispatch)
    runner = handler.RunnerAgentHandler(spec_store=None, backend=object())
    semantics = {
        "comparison_mode": "quality_at_fixed_time",
        "conditions_to_hold_constant": ["training time"],
        "method_freedom": ["optimizer"],
    }

    runner._run_repo_experiment_remote(
        "sid", "Improve https://github.com/example/repo",
        execution_phase="baseline", evaluation_semantics=semantics,
    )

    match = re.search(r"AR_EVALUATION_SEMANTICS_B64=([^ ;]+)", captured["command"])
    assert match is not None
    encoded = match.group(1).strip("'\"")
    assert json.loads(base64.b64decode(encoded).decode()) == semantics


def test_remote_method_dispatch_exports_method_memory_scope(monkeypatch):
    captured = {}
    monkeypatch.setenv("AUTORESEARCH_AGENTIC_IMPROVE", "1")
    monkeypatch.setattr(handler, "_cleanup_gpu_host", lambda _host: None)
    monkeypatch.setattr(handler, "_claim_cards", lambda _question="": ["0"])
    monkeypatch.setattr(handler, "_REMOTE_GPU_POOL", SimpleNamespace(put=lambda _card: None))

    def fake_dispatch(_host, command, timeout):
        captured["command"] = command
        result = {
            "benchmark": "repo", "metric": "loss", "lower_is_better": True,
            "repo_commit": "abc", "verdict": {"verdict": "pass", "improved": False},
            "baseline": {"metric": 1.0, "valid": True},
            "candidate": {"metric": 0.9, "valid": True},
            "report": "method", "config": "steps=10",
        }
        return _fake_proc(0, out="RESULT_JSON " + json.dumps(result))

    monkeypatch.setattr(handler, "_ssh_dispatch", fake_dispatch)
    runner = handler.RunnerAgentHandler(spec_store=None, backend=object())
    method_ref = "spec://method/session/sha256-current"

    runner._run_repo_experiment_remote(
        "sid", "Improve https://github.com/example/repo", execution_phase="method",
        frozen_baseline={"metric_value": 1.0}, memory_scope=method_ref,
    )

    match = re.search(r"AR_METHOD_SCOPE_B64=([^ ;]+)", captured["command"])
    assert match is not None
    assert base64.b64decode(match.group(1).strip("'\"")).decode() == method_ref


def test_remote_agentic_dispatch_passes_confirmed_primary_repo_independently(monkeypatch):
    """Reference/blog URLs in prose must not replace the confirmed repository."""
    captured = {}
    monkeypatch.setenv("AUTORESEARCH_AGENTIC_IMPROVE", "1")
    monkeypatch.setattr(handler, "_REMOTE_GPU_CARDS", ["0", "3", "5"])
    monkeypatch.setattr(handler, "_cleanup_gpu_host", lambda _host: None)
    monkeypatch.setattr(handler, "_claim_cards", lambda _question="": ["0"])
    monkeypatch.setattr(handler, "_REMOTE_GPU_POOL", SimpleNamespace(put=lambda _card: None))

    def fake_dispatch(_host, command, timeout):
        captured["command"] = command
        result = {
            "benchmark": "repo", "metric": "loss", "lower_is_better": True,
            "repo_commit": "abc", "verdict": {"verdict": "pass", "improved": False},
            "baseline": {"metric": 1.0, "valid": True},
            "candidate": {"metric": None, "valid": False},
            "report": "baseline", "config": "steps=10",
        }
        return _fake_proc(0, out="RESULT_JSON " + json.dumps(result))

    monkeypatch.setattr(handler, "_ssh_dispatch", fake_dispatch)
    runner = handler.RunnerAgentHandler(spec_store=None, backend=object())
    primary = "https://github.com/example/confirmed-repo"
    question = (
        "Reference https://social.example/author/status/123 before using "
        "https://github.com/example/confirmed-repo"
    )

    runner._run_repo_experiment_remote(
        "sid", question, execution_phase="baseline", repo_url=primary)

    argv = shlex.split(captured["command"])
    assert "AR_ALLOWED_GPU_CARDS=0,3,5" in argv
    driver_index = next(i for i, value in enumerate(argv) if value.endswith("agentic_run.py"))
    assert argv[driver_index + 1] == question
    encoded = argv[driver_index + 2]
    assert base64.b64decode(encoded).decode() == primary


def test_remote_agentic_dispatch_rejects_missing_explicit_python(monkeypatch):
    """Catch agentic dispatch silently falling back to DGX's Python without OpenHands."""
    monkeypatch.setenv("AUTORESEARCH_AGENTIC_IMPROVE", "1")
    monkeypatch.delenv("AUTORESEARCH_REMOTE_PY", raising=False)
    monkeypatch.setattr(handler, "_cleanup_gpu_host", lambda _host: None)
    monkeypatch.setattr(handler, "_claim_cards", lambda _question="": ["0"])
    monkeypatch.setattr(handler, "_REMOTE_GPU_POOL", SimpleNamespace(put=lambda _card: None))
    monkeypatch.setattr(
        handler, "_ssh_dispatch",
        lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("missing interpreter reached SSH dispatch")),
    )
    runner = handler.RunnerAgentHandler(spec_store=None, backend=object())

    with pytest.raises(handler.core.RunnerParseError, match="AUTORESEARCH_REMOTE_PY"):
        runner._run_repo_experiment_remote(
            "sid", "Improve https://github.com/example/repo",
            execution_phase="baseline",
        )


def test_remote_agentic_dispatch_preflights_openhands_import(monkeypatch):
    """Catch an explicit but incompatible interpreter reaching agentic_run.py."""
    captured = {}
    monkeypatch.setenv("AUTORESEARCH_AGENTIC_IMPROVE", "1")
    monkeypatch.setattr(handler, "_cleanup_gpu_host", lambda _host: None)
    monkeypatch.setattr(handler, "_claim_cards", lambda _question="": ["0"])
    monkeypatch.setattr(handler, "_REMOTE_GPU_POOL", SimpleNamespace(put=lambda _card: None))

    def fake_dispatch(_host, command, timeout):
        captured["command"] = command
        result = {
            "benchmark": "repo", "metric": "loss", "lower_is_better": True,
            "repo_commit": "abc", "verdict": {"verdict": "pass", "improved": False},
            "baseline": {"metric": 1.0, "valid": True},
            "candidate": {"metric": None, "valid": False},
            "report": "baseline", "config": "steps=10",
        }
        return _fake_proc(0, out="RESULT_JSON " + json.dumps(result))

    monkeypatch.setattr(handler, "_ssh_dispatch", fake_dispatch)
    runner = handler.RunnerAgentHandler(spec_store=None, backend=object())

    runner._run_repo_experiment_remote(
        "sid", "Improve https://github.com/example/repo", execution_phase="baseline")

    assert "import openhands" in captured["command"]
    assert "AUTORESEARCH_REMOTE_PY_INVALID" in captured["command"]


@pytest.mark.parametrize("large", [False, True])
def test_remote_agentic_question_is_one_shell_quoted_data_argument(monkeypatch, large):
    """Markdown/code-shell syntax in the research question must never become remote shell code."""
    captured = {}
    monkeypatch.setenv("AUTORESEARCH_AGENTIC_IMPROVE", "1")
    monkeypatch.setattr(handler, "_cleanup_gpu_host", lambda _host: None)
    monkeypatch.setattr(handler, "_claim_cards", lambda _question="": ["0"])
    monkeypatch.setattr(handler, "_REMOTE_GPU_POOL", SimpleNamespace(put=lambda _card: None))

    def fake_dispatch(_host, command, timeout):
        captured["command"] = command
        result = {
            "benchmark": "repo", "metric": "loss", "lower_is_better": True,
            "repo_commit": "abc", "verdict": {"verdict": "pass", "improved": False},
            "baseline": {"metric": 1.0, "valid": True},
            "candidate": {"metric": None, "valid": False},
            "report": "baseline", "config": "steps=10",
        }
        return _fake_proc(0, out="RESULT_JSON " + json.dumps(result))

    monkeypatch.setattr(handler, "_ssh_dispatch", fake_dispatch)
    runner = handler.RunnerAgentHandler(spec_store=None, backend=object())
    question = 'Inspect `train.py`; keep $(touch /tmp/never) and C:\\models\\x as literal "data".'
    if large:
        question += "\n组件历史" * 40000
    semantics = {"context": "x" * 180000} if large else None

    runner._run_repo_experiment_remote(
        "sid", question, execution_phase="baseline", evaluation_semantics=semantics)

    if not large:
        assert shlex.split(captured["command"])[-1] == question
    else:
        command, payload = captured["command"].split("<<'AR_REQUEST_JSON'\n")
        assert "AR_EVALUATION_SEMANTICS_B64=" not in command
        assert len(command.encode()) < 65536
        streamed = json.loads(payload.removesuffix("\nAR_REQUEST_JSON\n"))
        assert streamed["question"] == question
        assert json.loads(base64.b64decode(
            streamed["environment"]["AR_EVALUATION_SEMANTICS_B64"])) == semantics


def test_local_dispatch_canonicalizes_recovery_workroot(tmp_path, monkeypatch):
    from autoresearch.experiments.repo_improve import agentic_improve

    original = tmp_path / "original"
    original.mkdir()
    alias = tmp_path / "recovery"
    alias.symlink_to(original, target_is_directory=True)
    monkeypatch.setenv("AUTORESEARCH_WEB_WORKDIR", str(alias))
    monkeypatch.setenv("AUTORESEARCH_AGENTIC_IMPROVE", "1")
    captured = {}

    def dispatch(question, **kwargs):
        captured.update(kwargs)
        raise handler.core.RunnerError("stop before launching an experiment")

    monkeypatch.setattr(agentic_improve, "run_agentic_improve", dispatch)
    runner = object.__new__(handler.RunnerAgentHandler)
    with pytest.raises(handler.core.RunnerError):
        runner._run_repo_experiment_local_unlocked(
            "same-session", "research", execution_phase="baseline",
            repo_url="https://example.test/repository")
    assert captured["workroot"] == str(original / "repo_improve_local" / "same-session")


def test_local_dispatch_reuses_read_only_session_lock(tmp_path, monkeypatch):
    import fcntl
    workroot = tmp_path / "repo_improve_local" / "same-session"
    workroot.mkdir(parents=True)
    lock = workroot / ".agentic_run.lock"
    lock.write_text("retained lock identity")
    lock.chmod(0o444)
    inode = lock.stat().st_ino
    monkeypatch.setenv("AUTORESEARCH_WEB_WORKDIR", str(tmp_path))
    runner = object.__new__(handler.RunnerAgentHandler)
    def dispatch(*args, **kwargs):
        with lock.open() as other:
            with pytest.raises(BlockingIOError):
                fcntl.flock(other.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        return {"retained": True}, {}
    monkeypatch.setattr(runner, "_run_repo_experiment_local_unlocked", dispatch)
    assert runner._run_repo_experiment_local("same-session", "research")[0] == {"retained": True}
    assert lock.stat().st_ino == inode
    assert lock.read_text() == "retained lock identity"
    with lock.open() as other:
        fcntl.flock(other.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
