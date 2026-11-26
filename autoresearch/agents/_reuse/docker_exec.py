"""A plain-``docker`` execution backend: run commands in an image that is NOT an OpenHands
agent-server, with manual GPU passthrough.
在"镜像本身不是 agent-server、只需带 GPU 跑命令"时用的 plain-docker 执行后端。

``OpenHandsBackend`` needs the image to run the agent-server (it drives an agent inside).
SOL-ExecBench's image is a plain CUDA benchmark image, so ``docker run sol-image --host ...``
just exits and the sandbox dies. SOL also does not NEED an agent to discover its setup --
the adapter already gives a definite boundary and run command. What it needs is simply to
run those in the SOL image with a GPU. This backend does exactly that: ``docker run -d`` a
keep-alive container (manual ``--device`` passthrough, no NVIDIA Container Toolkit), then
``docker exec`` each command.

Same surface as ``OpenHandsBackend`` (``open_sandbox`` / ``execute_command`` /
``close_sandbox``) so it drops straight into ``make_sandbox_run_variant``.
"""
from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass

from autoresearch.agents._reuse.openhands_backend import (
    CommandResult,
    InfraError,
    _sandbox_extra_args,
)

logger = logging.getLogger(__name__)

#: Keep-alive: the container must outlive ``docker run`` so we can exec into it. We also
#: OVERRIDE the image's own entrypoint -- a benchmark image (SOL) has an entrypoint that runs
#: and exits, which would kill the container before any exec; we only ever exec, so skip it.
_KEEPALIVE_ENTRYPOINT = ["--entrypoint", "sleep"]
_KEEPALIVE_ARG = ["infinity"]


@dataclass
class DockerExecHandle:
    container_id: str


def _default_runner(argv, timeout=None):
    """Run a docker CLI argv; return (returncode, stdout, stderr). Raises TimeoutExpired."""
    p = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    return p.returncode, p.stdout, p.stderr


class DockerExecBackend:
    """Run commands in a plain image via ``docker run -d`` + ``docker exec``."""

    def __init__(self, *, server_image: str, command_timeout: float = 600.0, _runner=None):
        self.server_image = server_image
        self.command_timeout = command_timeout
        self._run = _runner or _default_runner

    def open_sandbox(self, *, cuda_device=None, mounts=None, env=None) -> DockerExecHandle:
        """Start a detached keep-alive container with GPU passthrough + bind mounts + env."""
        extra = _sandbox_extra_args(cuda_device=cuda_device, mounts=mounts)
        env_args = []
        for k, v in (env or {}).items():
            env_args += ["-e", f"{k}={v}"]
        argv = ["docker", "run", "-d", "--rm", "--ipc=host",
                "--ulimit", "memlock=-1", "--ulimit", "stack=67108864",
                *env_args, *_KEEPALIVE_ENTRYPOINT, *extra, self.server_image, *_KEEPALIVE_ARG]
        try:
            code, out, err = self._run(argv, timeout=120)
        except Exception as e:  # noqa: BLE001
            raise InfraError(f"docker run failed: {e}") from e
        if code != 0 or not out.strip():
            raise InfraError(f"docker run failed (exit {code}): {err[:300]}")
        return DockerExecHandle(container_id=out.strip().splitlines()[-1])

    def execute_command(self, handle: DockerExecHandle, cmd: str, *,
                        cwd: str = "/workspace", timeout=None) -> CommandResult:
        """``docker exec`` the command in a login shell at ``cwd``; a non-zero exit is a
        normal business result (returned, not raised)."""
        shell_cmd = f"cd {cwd} && {cmd}"
        argv = ["docker", "exec", handle.container_id, "bash", "-lc", shell_cmd]
        try:
            code, out, err = self._run(argv, timeout=timeout or self.command_timeout)
        except subprocess.TimeoutExpired:
            return CommandResult(exit_code=-9, stdout="", stderr="timed out", timed_out=True)
        except Exception as e:  # noqa: BLE001 -- transport/infra failure
            raise InfraError(f"docker exec transport error: {e}") from e
        return CommandResult(exit_code=code, stdout=out or "", stderr=err or "")

    def close_sandbox(self, handle: DockerExecHandle) -> bool:
        """Stop the container (``--rm`` removes it). Best-effort, never raises."""
        try:
            code, _out, _err = self._run(["docker", "stop", handle.container_id], timeout=60)
            return code == 0
        except Exception:  # noqa: BLE001
            logger.warning("docker stop failed for %s", handle.container_id, exc_info=True)
            return False


__all__ = ["DockerExecBackend", "DockerExecHandle"]
