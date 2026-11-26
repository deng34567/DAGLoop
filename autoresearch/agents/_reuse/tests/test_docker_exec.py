"""plain-docker 执行后端:给"镜像本身不是 agent-server、只需带 GPU 跑命令"的场景(SOL)用。

与 OpenHandsBackend 同接口(open_sandbox/execute_command/close_sandbox),但用 docker run 常驻 +
docker exec,不需要镜像内跑 agent-server。不真起 docker:注入假 runner 断言命令构造。
"""
from __future__ import annotations

from autoresearch.agents._reuse.docker_exec import DockerExecBackend


class _FakeDocker:
    def __init__(self, run_id="cid123"):
        self.calls = []
        self.run_id = run_id

    def __call__(self, argv, timeout=None):
        self.calls.append((argv, timeout))
        if argv[:2] == ["docker", "run"]:
            return 0, self.run_id + "\n", ""
        if argv[:2] == ["docker", "exec"]:
            return 0, "Validation ok\n", ""
        return 0, "", ""


def test_open_sandbox_runs_detached_with_gpu_and_mounts_and_keeps_alive(tmp_path):
    fake = _FakeDocker()
    be = DockerExecBackend(server_image="sol-execbench:latest", _runner=fake)
    d = tmp_path / "wr"; d.mkdir()

    h = be.open_sandbox(cuda_device=0, mounts=[str(d)])
    assert h.container_id == "cid123"

    argv = fake.calls[0][0]
    assert argv[:3] == ["docker", "run", "-d"]      # 常驻
    assert "--rm" in argv
    assert "--device" in argv and "/dev/nvidia0" in argv     # 手动 GPU 透传
    assert "--gpus" not in argv                               # 不依赖 toolkit
    assert f"{d}:{d}" in argv                                 # 同路径挂载
    # 覆盖镜像自带 entrypoint(否则它跑完就退,exec 时容器已死)
    assert "--entrypoint" in argv
    assert "sleep" in " ".join(argv)      # 保活
    assert "sol-execbench:latest" in argv


def test_open_sandbox_passes_env_vars():
    fake = _FakeDocker()
    be = DockerExecBackend(server_image="img", _runner=fake)
    be.open_sandbox(env={"FLASHINFER_TRACE_DIR": "/x/y"})
    argv = fake.calls[0][0]
    joined = " ".join(argv)
    assert "-e" in argv and "FLASHINFER_TRACE_DIR=/x/y" in joined


def test_execute_command_uses_docker_exec_with_cwd_and_returns_result():
    fake = _FakeDocker()
    be = DockerExecBackend(server_image="img", _runner=fake)
    h = be.open_sandbox()
    r = be.execute_command(h, "uv run sol-execbench --json", cwd="/sol-execbench", timeout=600)

    assert r.exit_code == 0 and "Validation ok" in r.stdout
    exec_argv = fake.calls[-1][0]
    assert exec_argv[:3] == ["docker", "exec", "cid123"]
    joined = " ".join(exec_argv)
    assert "cd /sol-execbench" in joined and "sol-execbench --json" in joined


def test_timeout_is_reported_not_raised():
    class _Timeout(_FakeDocker):
        def __call__(self, argv, timeout=None):
            if argv[:2] == ["docker", "exec"]:
                import subprocess
                raise subprocess.TimeoutExpired(argv, timeout)
            return super().__call__(argv, timeout)
    be = DockerExecBackend(server_image="img", _runner=_Timeout())
    h = be.open_sandbox()
    r = be.execute_command(h, "sleep 999", timeout=1)
    assert r.timed_out is True and r.exit_code != 0


def test_close_stops_the_container():
    fake = _FakeDocker()
    be = DockerExecBackend(server_image="img", _runner=fake)
    h = be.open_sandbox()
    assert be.close_sandbox(h) is True
    assert any(c[0][:2] == ["docker", "stop"] and h.container_id in c[0] for c in fake.calls)
