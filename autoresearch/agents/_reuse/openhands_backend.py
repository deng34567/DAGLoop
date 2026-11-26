"""
OpenHandsBackend: an execution backend that runs an LLM-driven coding agent
inside a Docker container (OpenHands SDK ``DockerWorkspace``), then hard-judges
results by exit code (never by the model's self-assessment) and can export the
container workspace to a local directory.
在 Docker 容器里、由大模型自主写实验代码并运行的执行后端。退出码硬判(不靠
模型自评),工作区产物可导出。

Mirrors the API usage of vibe's ``vibe/agents/_openhands/adapter.py`` (read-only
reference), but ALL code lives here and this module NEVER imports the vibe tree.

Design invariants:
- **Every ``openhands`` / ``docker`` import is LAZY** (inside methods), so this
  module imports cleanly on a host with no docker / no openhands present.
- The default pytest suite NEVER starts a real container or makes a real LLM /
  network call: ``open_sandbox`` takes a ``workspace_factory`` seam that tests
  inject, and ``run_agent_task`` builds ``Conversation`` / ``get_default_agent``
  via lazy imports that tests monkeypatch.
- Business failures (non-zero exit codes) are returned, NOT raised. Only
  transport / infrastructure failures are wrapped in ``InfraError``.
"""
from __future__ import annotations

import json
import logging
import threading
import os
import time
from dataclasses import dataclass
from decimal import Decimal
from functools import wraps
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from urllib.parse import urlsplit

logger = logging.getLogger(__name__)

# 本地已有的干净上游镜像(可用 env 覆盖)。
DEFAULT_SERVER_IMAGE = os.environ.get(
    "AUTORESEARCH_SANDBOX_IMAGE", "ghcr.io/openhands/agent-server:1.19.1-python"
)

# Remote conversations outlive the local polling process unless explicitly paused.
# Keep only the conversations currently owned by this controller so a graceful web
# shutdown can stop their agents without deleting containers or workspaces.
_ACTIVE_CONVERSATIONS: dict[int, Any] = {}
_ACTIVE_CONVERSATIONS_LOCK = threading.Lock()


def _track_active_conversation(convo: Any) -> None:
    with _ACTIVE_CONVERSATIONS_LOCK:
        _ACTIVE_CONVERSATIONS[id(convo)] = convo


def _untrack_active_conversation(convo: Any) -> None:
    with _ACTIVE_CONVERSATIONS_LOCK:
        if _ACTIVE_CONVERSATIONS.get(id(convo)) is convo:
            _ACTIVE_CONVERSATIONS.pop(id(convo), None)


def pause_active_conversations() -> int:
    """Pause every remote agent currently owned by this process, best-effort.

    Returns the number whose synchronous ``pause`` request completed.  Entries stay
    registered until their owning ``run_agent_task`` exits, allowing a repeated
    shutdown hook to retry a transient pause failure.
    """
    with _ACTIVE_CONVERSATIONS_LOCK:
        active = list(_ACTIVE_CONVERSATIONS.values())
    paused = 0
    for convo in active:
        try:
            convo.pause()
            paused += 1
        except Exception:  # noqa: BLE001 -- shutdown remains best-effort
            logger.warning("failed to pause active OpenHands conversation during shutdown",
                           exc_info=True)
    return paused


# --- GPU 透传 -------------------------------------------------------------
# 这台机器没装 NVIDIA Container Toolkit(装它要 root + 重启 docker,会打断别人
# 十几个在跑的容器),所以 `docker run --gpus all` 直接失败:
#     could not select device driver "" with capabilities: [[gpu]]
# 我们改为自己把 toolkit 平时自动注入的东西显式写出来:设备节点 + 宿主驱动库。
# 已在 SOL-ExecBench 上实测通过(容器内 nvidia-smi 认出 L20,CUDA kernel 编译+计时成功)。
# 对最终用户完全透明 —— 用户只提需求,这些参数由代码自动加。
_GPU_DEVICE_NODES = ("/dev/nvidiactl", "/dev/nvidia-uvm", "/dev/nvidia-uvm-tools")
_GPU_DRIVER_LIBS = (
    "libcuda.so",                    # unversioned linker name (Triton uses -lcuda)
    "libcuda.so.1",                  # CUDA driver API(必需)
    "libnvidia-ml.so.1",             # NVML(nvidia-smi / 显存查询)
    "libnvidia-ptxjitcompiler.so.1", # PTX JIT(部分 kernel 需要)
    "libcudadebugger.so.1",
)
_DEFAULT_LIB_DIR = "/usr/lib/x86_64-linux-gnu"
_DEFAULT_SMI = "/usr/bin/nvidia-smi"


def _parse_cards(card) -> list[str]:
    """把 card(单张 int/str、逗号分隔多卡字符串 "0,1,..,7"、或 list)规整成卡号字符串列表。"""
    if isinstance(card, (list, tuple)):
        raw = [str(c).strip() for c in card]
    else:
        raw = [c.strip() for c in str(card).split(",")]
    return [c for c in raw if c != ""]


def gpu_passthrough_args(
    card: int | str, *, lib_dir: str = _DEFAULT_LIB_DIR, smi_path: str = _DEFAULT_SMI
) -> list[str]:
    """Build the ``docker run`` flags that give a container its GPU(s), without the
    NVIDIA Container Toolkit: pass the device nodes through and bind-mount the
    host driver libraries read-only.

    单卡传 int/str(如 ``3``);多卡传逗号分隔字符串("0,1,2,3,4,5,6,7")或 list ->
    透传每张卡的设备节点(speedrun 这类 8 卡分布式任务)。多卡时补 ``--shm-size`` 供
    单节点 NCCL 用(docker 默认 /dev/shm 64MB 会让 8 卡 NCCL 报共享内存不足)。

    Only paths that ACTUALLY exist are mounted -- a bogus ``-v`` for a missing
    driver lib would make ``docker run`` fail outright. NEVER emits ``--gpus``.
    透传设备节点 + 只读挂载宿主驱动库。只挂真实存在的路径;绝不发 --gpus。"""
    cards = _parse_cards(card)
    args: list[str] = []
    for c in cards:
        args += ["--device", f"/dev/nvidia{c}"]
    for node in _GPU_DEVICE_NODES:
        if os.path.exists(node):
            args += ["--device", node]
    if len(cards) > 1:
        args += ["--shm-size", "16g"]   # 单节点多卡 NCCL 走 /dev/shm,默认 64MB 不够
    if os.path.exists(smi_path):
        args += ["-v", f"{smi_path}:{smi_path}:ro"]
    for lib in _GPU_DRIVER_LIBS:
        p = os.path.join(lib_dir, lib)
        if os.path.exists(p):
            args += ["-v", f"{p}:{p}:ro"]
    return args


def _inject_docker_run_args(cmd: list[str], extra: list[str]) -> list[str]:
    """Return ``cmd`` with ``extra`` inserted right after ``docker run``.

    Docker accepts options in any order as long as they precede the IMAGE, so
    inserting at index 2 is safe and avoids having to locate the image argument.
    Any command that is not a ``docker run`` is returned unchanged (the SDK also
    shells out ``docker version`` / ``docker stop`` etc. through the same helper).
    把 extra 插到 `docker run` 之后(选项必须在 image 前;插 index 2 最安全)。
    非 `docker run` 的命令原样返回。"""
    if len(cmd) >= 2 and cmd[0] == "docker" and cmd[1] == "run":
        return cmd[:2] + list(extra) + cmd[2:]
    return cmd


def bind_mount_args(paths) -> list[str]:
    """Bind-mount each host path into the container AT THE SAME PATH.

    The repo_improve pipeline builds its variant workspaces on the host and then
    executes them in the container; mapping host path -> identical container path
    means a host path is directly usable as a container path and nothing has to be
    rewritten. Writable on purpose (the pipeline writes patched sources and the run
    writes checkpoints/outputs). Missing paths are skipped -- docker would create a
    root-owned empty dir for them, which only causes confusing failures later.
    宿主路径原样映射到容器同路径,pipeline 宿主建工作区、容器内执行即可对齐。"""
    args: list[str] = []
    for p in paths or ():
        if isinstance(p, tuple) and len(p) == 2:
            source, target = map(os.fspath, p)
        else:
            source = target = os.fspath(p)
        if source and os.path.exists(source):
            args += ["-v", f"{source}:{target}"]
    return args


def env_args(env) -> list[str]:
    """``-e KEY=VAL`` flags. Used to point a container's heavy writes (HF/torch/pip
    caches, a repo's data dir) at a bind-mounted /data path so they never land on the
    root overlay -- and so ``docker commit`` doesn't freeze GBs of data into the image."""
    args: list[str] = []
    for k, v in (env or {}).items():
        if k and v is not None:
            args += ["-e", f"{k}={v}"]
    return args


def _positive_resource_number(value, *, integer: bool = False):
    """Normalize an optional task resource value without guessing a missing limit."""
    if isinstance(value, bool) or value in (None, ""):
        return None
    try:
        number = int(value) if integer else float(value)
    except (TypeError, ValueError):
        return None
    if number <= 0:
        return None
    return number


def _sandbox_extra_args(*, cuda_device=None, mounts=None, env=None) -> list[str]:
    """All extra ``docker run`` flags a sandbox needs: resources, GPU, mounts and env."""
    args: list[str] = []
    sandbox_env = dict(env or {})
    # These values come from the selected repository task's own manifest.  Docker-enforced
    # limits keep diagnostic subprocesses honest even when an agent forgets to configure
    # OpenMP/PyTorch thread counts.  Missing/invalid values remain unlimited rather than guessed.
    cpus = _positive_resource_number(sandbox_env.get("AUTORESEARCH_TASK_CPUS"))
    memory_mb = _positive_resource_number(
        sandbox_env.get("AUTORESEARCH_TASK_MEMORY_MB"), integer=True)
    control_memory_mb = _positive_resource_number(
        sandbox_env.get("AUTORESEARCH_SANDBOX_CONTROL_MEMORY_MB",
                        os.environ.get("AUTORESEARCH_SANDBOX_CONTROL_MEMORY_MB")), integer=True)
    if cpus is not None:
        args += ["--cpus", str(cpus)]
        # A CPU quota alone still lets BLAS/OpenMP create one thread per host core,
        # which can make a small task dramatically slower through oversubscription.
        threads = str(max(1, int(cpus)))
        for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            sandbox_env.setdefault(name, threads)
    if memory_mb is not None:
        # The task budget belongs to repository code, while this outer container also
        # hosts the Agent server.  Configured control-plane headroom must not reduce the
        # task's own budget; the original task value remains available in the env and to
        # repository-owned evaluators.
        container_memory_mb = memory_mb + (control_memory_mb or 0)
        args += ["--memory", f"{container_memory_mb}m",
                 "--memory-swap", f"{container_memory_mb}m"]
    if cuda_device is not None:
        args += gpu_passthrough_args(cuda_device)
    args += bind_mount_args(mounts)
    args += env_args(sandbox_env)
    args += ["--init"]
    return args


# 启动容器时要临时替换 SDK 模块里的 execute_command;两个沙箱并行启动会互相覆盖,
# 所以整个 patch→super()→restore 串行化(启动很快,不构成瓶颈)。
_GPU_START_LOCK = threading.Lock()


def _make_sandbox_workspace(server_image: str, platform: str, extra_args: list[str]):
    """Build a ``DockerWorkspace`` whose ``docker run`` carries our extra flags
    (manual GPU passthrough and/or bind mounts).

    The SDK builds the whole ``docker run`` argv inside ``_start_container`` and has
    no hook for extra flags (its only GPU knob is ``enable_gpu`` -> ``--gpus all``,
    unusable here). Rather than copy that ~90-line method (it lives in a SDK source
    tree SHARED with the vibe project, so we must not edit it either), we subclass
    and, for the duration of container start only, wrap the module-level
    ``execute_command`` so any ``docker run`` argv gets our flags injected.
    子类 + 启动期临时包裹 execute_command 注入额外参数:既不复制 SDK 逻辑,
    也不改动与 vibe 共享的 SDK 源码。"""
    from openhands.workspace import DockerWorkspace  # lazy
    from openhands.workspace.docker import workspace as _ws_mod  # lazy
    from openhands.sdk.conversation.impl import remote_conversation as _remote_conversation
    from urllib.request import ProxyHandler, build_opener
    import httpx

    # The web process may legitimately carry proxy settings for LLM traffic, but
    # the Docker agent server is always local.  The SDK uses urllib.urlopen for
    # its health check, which otherwise inherits those proxy settings and can
    # report a healthy localhost container as unavailable.
    direct_urlopen = build_opener(ProxyHandler({})).open

    # The SDK's event stream is a WebSocket separate from the HTTP client above.
    # websockets reads proxy variables by default, so make only loopback event
    # streams explicit direct connections. External WebSockets remain unchanged.
    ws_connect = _remote_conversation.websockets.connect
    if not getattr(ws_connect, "_autoresearch_direct_loopback", False):
        original_ws_connect = ws_connect

        @wraps(original_ws_connect)
        def direct_loopback_connect(uri, *args, **kwargs):
            if urlsplit(str(uri)).hostname in {"127.0.0.1", "localhost", "::1"}:
                kwargs["proxy"] = None
            return original_ws_connect(uri, *args, **kwargs)

        direct_loopback_connect._autoresearch_direct_loopback = True
        _remote_conversation.websockets.connect = direct_loopback_connect

    class _GpuDockerWorkspace(DockerWorkspace):
        @property
        def client(self):  # type: ignore[override]
            client = self._client
            if client is None:
                timeout = httpx.Timeout(
                    connect=10.0,
                    read=self.read_timeout,
                    write=10.0,
                    pool=10.0,
                )
                client = httpx.Client(
                    base_url=self.host,
                    timeout=timeout,
                    headers=self._headers,
                    limits=httpx.Limits(max_connections=self.max_connections),
                    trust_env=False,
                )
                self._client = client
            return client

        def _start_container(self, image, context):  # type: ignore[override]
            with _GPU_START_LOCK:
                original = _ws_mod.execute_command
                original_urlopen = _ws_mod.urlopen

                def _patched(cmd, *a, **kw):
                    if isinstance(cmd, (list, tuple)):
                        cmd = _inject_docker_run_args(list(cmd), extra_args)
                    return original(cmd, *a, **kw)

                _ws_mod.execute_command = _patched
                _ws_mod.urlopen = direct_urlopen
                try:
                    return super()._start_container(image, context)
                finally:
                    _ws_mod.execute_command = original
                    _ws_mod.urlopen = original_urlopen

    return _GpuDockerWorkspace(server_image=server_image, platform=platform)


@dataclass
class SandboxHandle:
    """A live sandbox: wraps the entered ``DockerWorkspace``.
    一个已启动的沙箱句柄,持有已进入的 DockerWorkspace。"""

    workspace: Any


@dataclass
class CommandResult:
    """Result of a single in-container command; ``exit_code`` is the hard judge.
    容器内单条命令的结果;exit_code 为硬判依据。"""

    exit_code: int
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False


@dataclass
class RunResult:
    """Result of an agent coding loop: terminal status + accumulated cost.
    agent 写码 loop 的结果:终态状态 + 累计成本。"""

    status: str
    cost: str = "0"
    event_summary: str = ""


def _tool_markup_repair_message(convo, *, native_tool_calling: bool = True) -> str:
    """Recognize an unexecuted pseudo-tool reply, without interpreting its arguments."""
    events = getattr(convo.state, "events", ())
    for event in reversed(events[-32:]):
        if getattr(event, "kind", None) in ("ActionEvent", "ObservationEvent"):
            return ""
        if getattr(event, "kind", None) != "MessageEvent":
            continue
        message = getattr(event, "llm_message", None)
        if (getattr(event, "source", None) != "agent"
                or getattr(message, "role", None) != "assistant"):
            return ""
        text = "".join(getattr(part, "text", "") for part in message.content)
        # Some gateways leak a closing reasoning delimiter into the public reply.
        text = text.rsplit("</think>", 1)[-1]
        _, marker, body = text.partition("<tool_call>")
        # Detect a leaked Unicode delimiter too; never execute or rewrite arguments.
        tool_name, separator, arguments = body.replace("⟩", ">", 1).partition(">")
        tool_name = tool_name.removeprefix("function=")
        named_call = bool(marker and separator and tool_name.isidentifier()
                          and "<parameter=" in arguments)
        emulated_name = (text.lstrip().partition(">")[0].removeprefix("<function=")
                         if text.lstrip().startswith("<function=") else "")
        emulated_call = bool(emulated_name.isidentifier() and "<parameter=" in text)
        if ((text.lstrip().startswith("<tool_call>")
             and any(marker in text for marker in ("<arg_key>", "<parameter=")))
                or named_call or (not native_tool_calling and emulated_call)):
            if not native_tool_calling:
                target = emulated_name or (tool_name if named_call else "terminal")
                return (
                    "Your last reply used malformed function-call markup and did not execute. "
                    "This conversation uses the SDK's emulated function interface: begin with "
                    f"<function={target}> "
                    "(including the opening angle bracket), provide separate "
                    "<parameter=name>value</parameter> fields, and end with </function>. "
                    "Use only parameter names declared by that tool; do not add a name field. "
                    "For think, use thought and optional summary, without security_risk. "
                    "Use one function call per message, without an outer tool_call wrapper. "
                    "Retry the intended operation using that interface, preserving the existing "
                    "task and budgets. Do not claim the unexecuted command produced results. "
                    "Only when waiting for an already-running command, use:\n"
                    "<function=terminal>\n<parameter=command> </parameter>\n"
                    "<parameter=reset>false</parameter>\n"
                    "<parameter=security_risk>LOW</parameter>\n"
                    "<parameter=summary>Wait for the existing command</parameter>\n</function>"
                )
            return (
                "Your last reply was tool-call markup in a plain assistant message; "
                "it did not execute a tool. Use the provided structured tool interface "
                "to continue the existing task. Preserve its requirements and budgets; "
                "do not claim that the unexecuted command produced results. "
                "If waiting for an already-running terminal command, use separate "
                'JSON arguments {"command":" ","reset":false}: '
                "the nonempty single-space string avoids empty-value serialization, "
                "and the terminal treats it as an empty command for waiting. "
                "Omit optional fields absent from the current tool schema."
            )
        return ""
    return ""


def _terminal_wait_event(convo):
    """Use the tool's public soft-wait status, not model claims or output text."""
    for event in reversed(getattr(convo.state, "events", ())[-32:]):
        observation = getattr(event, "observation", None)
        if getattr(observation, "kind", None) == "TerminalObservation":
            metadata = getattr(observation, "metadata", None)
            suffix = (metadata.get("suffix", "") if isinstance(metadata, dict)
                      else getattr(metadata, "suffix", ""))
            # SDK rejects a new command while the prior process is still running.
            # That error describes the rejected request, not a failed prior process.
            busy_rejection = (isinstance(suffix, str)
                              and "is NOT executed" in suffix
                              and "previous command is still running" in suffix)
            pending = (getattr(observation, "exit_code", None) == -1
                    and (not getattr(observation, "is_error", False) or busy_rejection)
                    and not getattr(observation, "timeout", False))
            return event if pending else None
        if getattr(getattr(event, "action", None), "kind", None) == "TerminalAction":
            return None
    return None


def _terminal_wait_pending(convo) -> bool:
    return _terminal_wait_event(convo) is not None


def _terminal_event_summary(convo, *, error_workspace=None) -> str:
    """Keep user-facing finish evidence, never the agent's thoughts or full transcript."""
    try:
        events = list(getattr(convo.state, "events", ())[-32:])
        if error_workspace is not None and not any(
                getattr(event, "kind", None) == "ConversationErrorEvent" for event in events):
            # ERROR status may arrive before its WebSocket event. Fetch only the
            # latest error via the existing direct client, never the whole history.
            client = getattr(error_workspace, "client", None)
            if client is not None:
                from openhands.sdk.event.conversation_error import ConversationErrorEvent
                error_kind = (ConversationErrorEvent.__module__ + "."
                              + ConversationErrorEvent.__name__)
                for probe in range(5):
                    if probe:
                        # The error may not yet be published when ERROR is observed.
                        time.sleep(0.5)
                    response = client.get(
                        f"/api/conversations/{convo.id}/events/search",
                        params={"kind": error_kind, "sort_order": "TIMESTAMP_DESC", "limit": 1},
                        timeout=5.0)
                    response.raise_for_status()
                    items = response.json().get("items", [])[:1]
                    if items:
                        events.extend(SimpleNamespace(**item) for item in items)
                        break
        final_reply_allowed = getattr(
            getattr(convo.state, "execution_status", None), "name", "") == "FINISHED"
        for event in reversed(events):
            diagnostic_reply = None
            action = getattr(event, "action", None)
            if getattr(action, "kind", None) == "FinishAction":
                text = "Agent finish message (unverified): " + str(action.message)
            elif getattr(event, "kind", None) == "ConversationErrorEvent":
                text = "Conversation error code: " + str(event.code)
                from autoresearch.llm.errors import is_quota_exhaustion
                # Classify sensitive provider details without copying request headers
                # or credentials into the persisted failure evidence.
                if is_quota_exhaustion(getattr(event, "detail", "")):
                    text += "; insufficient_quota"
                elif any(marker in str(getattr(event, "detail", "")).lower()
                         for marker in ("llmtimeouterror", "litellm.timeout",
                                        "504 gateway time-out")):
                    text += "; llm_provider_timeout"
                elif "output text may contain sensitive information" in str(
                        getattr(event, "detail", "")).lower():
                    text += "; llm_provider_content_filter"
            elif (final_reply_allowed and getattr(event, "kind", None) == "MessageEvent"
                  and getattr(event, "source", None) == "agent"
                  and getattr(getattr(event, "llm_message", None), "role", None) == "assistant"):
                # SDKs may finish with a public reply instead of FinishAction. Retain
                # its explanation, not reasoning_content or an earlier progress message.
                reply = "".join(getattr(part, "text", "")
                                for part in event.llm_message.content)
                reply = reply.rsplit("</think>", 1)[-1]
                if not reply.strip() or "<think>" in reply:
                    return ""
                protocol_failure = bool(_tool_markup_repair_message(convo))
                if protocol_failure:
                    diagnostic_reply = reply
                # This summary becomes the next task's feedback. Preserve the cause,
                # not a malformed call that would become another in-context example.
                text = (
                    "Agent tool protocol failure (unexecuted tool markup): malformed "
                    "function-call syntax was returned as assistant text; no tool was executed "
                    "from this last reply. This does not negate earlier tool executions. "
                    "Inspect existing artifacts and host attempt records before repeating work; "
                    "do not infer that all prior edits or experiments failed. "
                    "Continue using the configured SDK tool protocol and preserve the task budget."
                    if protocol_failure else "Agent final reply (unverified): " + reply)
            else:
                if getattr(event, "kind", None) in {"ActionEvent", "ObservationEvent", "MessageEvent"}:
                    final_reply_allowed = False
                continue
            # These messages are agent output, not trusted telemetry. Do not persist configured
            # credentials if an agent happens to repeat one in its final explanation.
            for key, value in os.environ.items():
                if (key.upper().endswith(("_KEY", "TOKEN", "PASSWORD", "SECRET"))
                        and len(value) >= 8):
                    text = text.replace(value, "[REDACTED]")
                    if diagnostic_reply is not None:
                        diagnostic_reply = diagnostic_reply.replace(value, "[REDACTED]")
            if diagnostic_reply is not None:
                logger.warning("Unexecuted tool markup in conversation %s: %r",
                               getattr(convo, "id", "unknown"), diagnostic_reply[:16000])
            return text[:2000]
    except Exception as exc:
        # Failure to retrieve optional evidence must not discard the original execution status.
        logger.warning("Could not collect terminal event summary: %s", type(exc).__name__)
    return ""


class InfraError(RuntimeError):
    """Infrastructure / transport failure (docker gone, container unreachable).
    基础设施/传输类失败;业务失败(非零退出码)不走这里,走返回值。"""


def _check_sandbox_container(workspace) -> None:
    """Fail only on Docker-confirmed container loss, not a transport outage."""
    import subprocess

    cid = getattr(workspace, "_container_id", None)
    if not isinstance(cid, str) or not cid:
        return
    try:
        result = subprocess.run(
            ["docker", "inspect", "--format", "{{.State.Running}}", cid],
            capture_output=True, text=True, timeout=10,
            env={k: v for k, v in os.environ.items() if not k.lower().endswith("_proxy")})
    except (OSError, subprocess.TimeoutExpired):
        return  # Unavailable Docker telemetry is not evidence of a dead container.
    missing = result.returncode != 0 and any(
        marker in result.stderr.lower() for marker in ("no such object:", "no such container:"))
    if missing or (result.returncode == 0 and result.stdout.strip() == "false"):
        raise InfraError(f"sandbox container {cid} is not running")


#: How many times to re-try a container command that failed with a TRANSIENT transport
#: blip (the link dropped but the server is alive). A truly dead container just fails all
#: retries fast and the caller's own escalation retry reopens a fresh one.
_EXEC_TRANSIENT_RETRIES = 3

#: Substrings that mark a dropped-connection (retryable) rather than a real command result.
#: Deliberately NOT including plain "timeout": a command that genuinely ran out of wall-clock
#: should fail, not silently re-run for another full budget.
_TRANSIENT_MARKERS = (
    "connection reset", "server disconnected", "connection aborted", "broken pipe",
    "remoteprotocolerror", "peer closed", "connection refused", "econnreset",
)


def _is_transient_transport_error(exc: Exception) -> bool:
    """True if ``exc`` looks like a momentary link drop we should retry (not a dead
    container, not a command result). Matches on the whole exception chain's text so an
    httpx error wrapped a few layers deep is still recognised."""
    seen = set()
    cur: BaseException | None = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        text = f"{type(cur).__name__}: {cur}".lower()
        if any(m in text for m in _TRANSIENT_MARKERS):
            return True
        cur = cur.__cause__ or cur.__context__
    return False


def build_sandbox_llm(
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
    usage_id: str = "autoresearch-sandbox",
):
    """Build an OpenHands ``LLM`` pointed at OpenRouter (mirrors vibe ``build_llm``,
    but reads AutoResearch's own env). Long max-reasoning agent calls stream by
    default; the condenser gets a non-streaming copy in ``_build_agent``.
    构造指向 OpenRouter 的 OpenHands LLM；max 推理的长调用默认流式，condenser 单独禁用流式。

    Raises ``InfraError`` when no API key is configured.
    """
    api_key = api_key or os.environ.get("OPENROUTER_KEY")
    base_url = (
        base_url
        or os.environ.get("AUTORESEARCH_SANDBOX_BASE_URL")
        or os.environ.get("OPENROUTER_BASE_URL")
        or "https://openrouter.ai/api/v1"
    )
    model = model or os.environ.get("AUTORESEARCH_SANDBOX_MODEL", "moonshotai/kimi-k2")
    if not api_key:
        raise InfraError("OPENROUTER_KEY not set; cannot build sandbox LLM.")
    if not model.startswith("openai/"):
        model = "openai/" + model  # OpenRouter via litellm needs the provider prefix
    from openhands.sdk import LLM  # lazy
    from pydantic import SecretStr

    kwargs = dict(
        usage_id=usage_id,
        model=model,
        base_url=base_url,
        api_key=SecretStr(api_key),
        disable_vision=True,
    )  # 不设 stream
    temperature = os.environ.get("AUTORESEARCH_LLM_TEMPERATURE")
    if temperature:
        kwargs["temperature"] = float(temperature)
    max_tokens = (
        os.environ.get("AUTORESEARCH_SANDBOX_LLM_MAX_TOKENS")
        or os.environ.get("AUTORESEARCH_LLM_MAX_TOKENS")
    )
    if max_tokens:
        kwargs["max_output_tokens"] = int(max_tokens)
        if kwargs["max_output_tokens"] < 1:
            raise ValueError("sandbox LLM max tokens must be a positive integer")
    # CRITICAL (found live): the OpenHands SDK LLM defaults to reasoning_effort='high' + a 200k
    # extended-thinking budget, NON-streaming, with a 300s per-call timeout. A reasoning-heavy model
    # (glm) then spends minutes "thinking" and the completion EXCEEDS the timeout before it ever emits
    # a tool action -- so run_agent_task "finishes" having made ZERO edits (method=None), every round.
    # An in-container CODING agent should ACT (edit -> run -> observe), not deliberate for minutes:
    # cap the reasoning and raise the timeout so a completion actually returns and the agent can edit.
    # Both env-overridable. reasoning_effort is a top-level LLM field on the SDK model.
    #
    # Default 'none', NOT 'low': measured on this endpoint, glm at reasoning_effort='low' STILL burns
    # ~450 reasoning tokens on a trivial edit (and far more on a real multi-turn agent prompt, up to
    # the 200k budget -> timeout); only 'none' fully disables thinking (0 reasoning tokens, ~3s).
    # flash is inherently fast either way, so 'none' is harmless there. Mirrors vibe_programming's
    # "沙箱搬砖关" (VIBE_SANDBOX_REASONING_EFFORT=none) -- the deliberation agents keep high reasoning,
    # the in-sandbox coder does not.
    reasoning_setting = os.environ.get("AUTORESEARCH_SANDBOX_REASONING_EFFORT", "none")
    # Explicit 'default' leaves the provider in control instead of inheriting the SDK default.
    # The SDK enum excludes 'max'; send that explicit provider value through extra_body below.
    kwargs["reasoning_effort"] = None if reasoning_setting in ("default", "max") else reasoning_setting
    # The SDK may omit this field for model IDs absent from its capability table.
    # Preserve an explicit deployment choice on the wire; do not guess provider support
    # from a model name or force the default onto providers that did not request it.
    if "AUTORESEARCH_SANDBOX_REASONING_EFFORT" in os.environ and reasoning_setting != "default":
        kwargs["litellm_extra_body"] = {"reasoning_effort": reasoning_setting}
    stream_setting = os.environ.get("AUTORESEARCH_SANDBOX_LLM_STREAM")
    if stream_setting is not None:
        kwargs["stream"] = stream_setting == "1"
    elif reasoning_setting == "max":
        # Max reasoning can exceed a gateway's non-streaming response-body deadline before
        # emitting its first tool call. Streaming is safe for the main conversation; the
        # condenser remains non-streaming because it invokes the LLM without a token callback.
        kwargs["stream"] = True
    extra_body = json.loads(os.environ.get("AUTORESEARCH_LLM_EXTRA_BODY") or "{}")
    if not isinstance(extra_body, dict):
        raise ValueError("AUTORESEARCH_LLM_EXTRA_BODY must be a JSON object")
    if extra_body:
        kwargs["litellm_extra_body"] = {**kwargs.get("litellm_extra_body", {}), **extra_body}
    # Reuse the SDK's validated tool-call emulation for gateways with broken native calls.
    kwargs["native_tool_calling"] = os.environ.get(
        "AUTORESEARCH_SANDBOX_NATIVE_TOOL_CALLING", "1") != "0"
    try:
        kwargs["timeout"] = int(os.environ.get("AUTORESEARCH_SANDBOX_LLM_TIMEOUT", "600"))
    except (TypeError, ValueError):
        kwargs["timeout"] = 600
    # The SDK counts total attempts, so one retry requires two attempts. Its default is five; with a
    # 1200s call timeout, one stuck completion can otherwise hold the experiment state for hours
    # before Verification gets the failure and starts the repository-level retry.
    try:
        kwargs["num_retries"] = max(
            0, int(os.environ.get("AUTORESEARCH_SANDBOX_LLM_RETRIES", "1"))) + 1
    except (TypeError, ValueError):
        kwargs["num_retries"] = 2
    for env_key, field in (("HKGAI_STREAM_RETRY_BASE_S", "retry_min_wait"),
                           ("HKGAI_STREAM_RETRY_CAP_S", "retry_max_wait")):
        if env_key in os.environ:
            kwargs[field] = int(os.environ[env_key])
    return LLM(**kwargs)


def _accum_cost(convo) -> str:
    """Accumulated LLM cost from the conversation stats; ``"0"`` if unavailable.
    从 conversation stats 取累计成本,取不到返回 "0"(绝不 import vibe 的 pricing 兜底)。"""
    try:
        value = convo.conversation_stats.get_combined_metrics().accumulated_cost
        return str(value if value is not None else 0)
    except Exception:
        return "0"


def _discard_stream_token(_chunk) -> None:
    """Consume headless streaming chunks; the completed response is still retained by the SDK."""


def _build_agent(llm, cli_mode: bool):
    """Default agent, but with the terminal tool forced to ``subprocess`` mode. The default tmux
    terminal uses a pane POOL (``use_pool = terminal_type in (None, "tmux")``) that crashes under
    concurrent / loaded sandboxes -- ``RuntimeError: TmuxPanePool is not initialized or already
    closed`` -- and the SDK swallows it as an error observation, so the agent "finishes" having done
    nothing. ``subprocess`` runs each command in a plain subprocess with no pool, sidestepping the
    crash entirely (borrowed from the vibe_programming OpenHands integration analysis). Fully
    defensive: any SDK-layout mismatch falls back to the plain default agent."""
    from openhands.tools.preset.default import get_default_agent
    try:
        from openhands.sdk import Agent
        from openhands.sdk.tool import Tool
        from openhands.tools.preset.default import get_default_condenser, register_default_tools
        from openhands.tools.file_editor import FileEditorTool
        from openhands.tools.task_tracker import TaskTrackerTool
        from openhands.tools.terminal import TerminalTool
        register_default_tools(enable_browser=not cli_mode)
        tools = [
            Tool(name=TerminalTool.name, params={"terminal_type": "subprocess"}),
            Tool(name=FileEditorTool.name),
            Tool(name=TaskTrackerTool.name),
        ]
        condenser_llm = llm.model_copy(update={"usage_id": "condenser", "stream": False})
        return Agent(llm=llm, tools=tools, system_prompt_kwargs={"cli_mode": cli_mode},
                     condenser=get_default_condenser(condenser_llm))
    except Exception:  # noqa: BLE001 -- SDK layout differs -> plain default agent (may hit the pool)
        logger.warning("subprocess-terminal agent build failed; using default agent", exc_info=True)
        return get_default_agent(llm=llm, cli_mode=cli_mode)


class OpenHandsBackend:
    """Execution backend: DockerWorkspace lifecycle + in-container agent loop +
    command execution + workspace export.
    执行后端:DockerWorkspace 生命周期 + 容器内 agent loop + 命令执行 + 工作区导出。"""

    def __init__(
        self,
        *,
        server_image: str = DEFAULT_SERVER_IMAGE,
        command_timeout: float = 120.0,
    ) -> None:
        self.server_image = server_image
        self.command_timeout = command_timeout

    def open_sandbox(self, *, workspace_factory=None, cuda_device=None, mounts=None, env=None) -> SandboxHandle:
        """Start a DockerWorkspace (or a test-injected one) and enter it.
        启动 DockerWorkspace(或测试注入的假 workspace)并进入。

        ``workspace_factory`` is the test seam: tests pass a callable returning a
        mock workspace so the default suite never touches docker.

        ``mounts`` bind-mounts host paths into the container at the SAME path, so the
        pipeline can build variant workspaces on the host and execute them in here.

        ``cuda_device`` (e.g. ``3``) gives the sandbox ONE GPU via manual device
        passthrough -- see ``gpu_passthrough_args``. We deliberately do NOT use
        the SDK's ``enable_gpu`` field: it emits ``--gpus all``, which needs the
        NVIDIA Container Toolkit that this host does not have. ``None`` = CPU-only
        sandbox (unchanged behaviour).
        """
        if workspace_factory is not None:
            workspace = workspace_factory()
        else:
            workspace = _make_sandbox_workspace(
                self.server_image, "linux/amd64",
                _sandbox_extra_args(cuda_device=cuda_device, mounts=mounts, env=env),
            )
        workspace.__enter__()  # DockerWorkspace 在 __enter__ 才真正起容器
        return SandboxHandle(workspace=workspace)

    def close_sandbox(self, handle: SandboxHandle) -> bool:
        """Tear down the workspace, best-effort (never raises).
        best-effort 拆除工作区,绝不抛错。返回是否确认完成拆除。"""
        try:
            handle.workspace.__exit__(None, None, None)
            return True
        except Exception:
            logger.warning("sandbox teardown failed", exc_info=True)
            return False

    def commit_sandbox(self, handle: SandboxHandle, image_tag: str) -> bool:
        """``docker commit`` the LIVE container into a reusable image; True on success.

        A discovered setup is only valid in the environment it was verified in, and the
        agent often installs that environment at the CONTAINER level (a compiler apt-gets,
        a pip cache, a system lib) -- state that a bare base image cannot reproduce and
        that the two remembered commands cannot rebuild. Freezing the container captures
        exactly that, so a later recall runs the SAME environment (the workspace ``.venv``
        is still rebuilt by ``prepare_cmd``). Best-effort: a failed commit just means the
        next run re-discovers, so it must never raise.
        """
        import subprocess
        cid = getattr(handle.workspace, "_container_id", None)
        if not cid:
            logger.warning("commit_sandbox: no container id on handle")
            return False
        try:
            r = subprocess.run(["docker", "commit", cid, image_tag],
                               capture_output=True, text=True, timeout=600)
            if r.returncode != 0:
                logger.warning("docker commit failed: %s", (r.stderr or "").strip()[:300])
            return r.returncode == 0
        except Exception:  # noqa: BLE001 -- freezing is best-effort
            logger.warning("docker commit raised", exc_info=True)
            return False

    def execute_attempt_command(
        self, handle: SandboxHandle, cmd: str, *, cwd: str = "/workspace", timeout=None
    ) -> CommandResult:
        """Submit a non-idempotent experiment once; a lost reply is not a failed launch."""
        try:
            raw = handle.workspace.execute_command(
                cmd, cwd=cwd, timeout=timeout or self.command_timeout)
        except Exception as exc:
            error = InfraError(
                "attempt execution status unknown after transport error; do not resubmit "
                "before reconciling the original attempt: " + str(exc))
            error.execution_uncertain = True
            raise error from exc
        return CommandResult(
            exit_code=getattr(raw, "exit_code", 1), stdout=getattr(raw, "stdout", "") or "",
            stderr=getattr(raw, "stderr", "") or "",
            timed_out=bool(getattr(raw, "timeout_occurred", False)))

    def execute_command_as_root(
        self, handle: SandboxHandle, cmd: str, *, cwd: str = "/workspace", timeout=None
    ) -> CommandResult:
        """Run a controller-authored maintenance command as root in this sandbox."""
        import subprocess

        container_id = getattr(handle.workspace, "_container_id", None)
        if not container_id:
            raise InfraError("root sandbox command requires a live container id")
        command_timeout = timeout or self.command_timeout
        try:
            completed = subprocess.run([
                "docker", "exec", "--user", "0", "--workdir", cwd,
                container_id, "bash", "-lc", cmd,
            ], capture_output=True, text=True, timeout=command_timeout)
        except subprocess.TimeoutExpired as exc:
            return CommandResult(
                exit_code=124, stdout=exc.stdout or "", stderr=exc.stderr or "",
                timed_out=True)
        except OSError as exc:
            raise InfraError(f"root sandbox command transport error: {exc}") from exc
        return CommandResult(
            exit_code=completed.returncode, stdout=completed.stdout or "",
            stderr=completed.stderr or "")

    def execute_command(
        self, handle: SandboxHandle, cmd: str, *, cwd: str = "/workspace", timeout=None
    ) -> CommandResult:
        """Run a command in the container; a non-zero exit code is a normal
        business result (returned, NOT raised). Only transport/infra errors are
        wrapped in ``InfraError``.
        容器内执行命令;非零退出码是正常业务结果(返回不抛),只有传输/基础设施错才抛。

        The link to the agent-server is HTTP, and on a busy host it occasionally drops
        mid-request ("connection reset by peer", "server disconnected") even while the
        server keeps serving other calls -- a momentary blip, not a dead container. Such
        a blip during a long discovery threw away ~20 min of work (seen on cpu-tinygrad
        and nanochat). So a transient transport error is retried on a fresh connection a
        few times before we give up; a genuine non-zero exit is never retried.
        """
        import time
        attempts = _EXEC_TRANSIENT_RETRIES + 1
        for i in range(attempts):
            try:
                raw = handle.workspace.execute_command(
                    cmd, cwd=cwd, timeout=timeout or self.command_timeout
                )
            except Exception as e:  # noqa: BLE001 -- all transport failures land here
                if _is_transient_transport_error(e) and i < attempts - 1:
                    logger.warning("execute_command transport blip (retry %d/%d): %s",
                                   i + 1, attempts - 1, e)
                    time.sleep(2 * (i + 1))   # brief backoff; next call opens a new connection
                    continue
                raise InfraError(f"execute_command transport error: {e}") from e
            # SDK CommandResult 字段 = command/exit_code/stdout/stderr/timeout_occurred;
            # 把 timeout_occurred 映到我们的 timed_out(否则该字段是死的)。
            return CommandResult(
                exit_code=getattr(raw, "exit_code", 1),
                stdout=getattr(raw, "stdout", "") or "",
                stderr=getattr(raw, "stderr", "") or "",
                timed_out=bool(getattr(raw, "timeout_occurred", False)),
            )

    def run_agent_task(  # noqa: D401
        self, handle: SandboxHandle, task: str, *, llm=None, cli_mode: bool = True,
        timeout: float | None = None, stop_predicate=None, completion_check=None,
    ) -> RunResult:
        """Run one in-container agent coding loop: build the default agent, open a
        Conversation on the workspace, send the task, run to a terminal state, and
        collect the status + accumulated cost. Closes the Conversation afterwards
        but leaves the workspace open.
        容器内跑一轮 agent 写码 loop:构造默认 agent → 在 workspace 上开 Conversation →
        发 task → 跑到终态 → 收状态/累计成本;跑完关 Conversation,不关 workspace。

        ``timeout`` (seconds) is an OPTIONAL wall-clock SAFETY BACKSTOP on the
        agent's autonomous loop. SDK 1.23.0's ``Conversation.run()`` takes no
        timeout arg, so we run it on a daemon thread and stop waiting after
        ``timeout`` (status -> ``"TIMEOUT"``, conversation closed best-effort).
        Default ``None`` runs inline exactly as before, so existing callers are
        byte-identical.

        ``completion_check`` may request result repair after FINISHED. A changed
        error allows one further correction within the original timeout, not
        a separate 300-second result deadline. An unchanged error never loops; FINISHED
        is not promoted to verified success. Repeated malformed tool replies have
        at most three bounded repair continuations unless a new successful tool
        observation proves that execution resumed. Repeated malformed replies may
        use one fresh conversation in the same workspace even if no tool executed.
        All recovery shares the original task deadline. A pending terminal soft
        wait uses the remaining original deadline rather than limiting a live
        experiment to 300 seconds.
        Fresh soft-wait observations may continue that same wait; unchanged evidence
        does not authorize repeated continuations.
        """
        from openhands.sdk import Conversation  # lazy — Conversation 在 openhands.sdk

        llm = llm or build_sandbox_llm()
        agent = _build_agent(llm, cli_mode)
        # 会议 2026-08-12: 调大 agent 每轮动作步预算 500->1000(env 可配)。500 步在"装环境+实现方法+
        # 跑分"叠加下常常不够(尤其重方法), 撞 MaxIterationsReached -> round invalid。
        _max_iter = int(os.environ.get("AUTORESEARCH_AGENT_MAX_ITERS", "1000"))
        convo = Conversation(
            agent=agent,
            workspace=handle.workspace,
            token_callbacks=[_discard_stream_token],
            max_iteration_per_run=_max_iter,
            # Headless runs retain SDK events; Rich rendering of large terminal output
            # can monopolize the host event thread without advancing the experiment.
            visualizer=None,
            # Long environment setup often repeats harmless inspection/install actions.
            # The SDK's short-history heuristic has falsely killed otherwise healthy
            # AutoResearch runs; bounded iterations and wall-clock limits remain active.
            stuck_detection=False,
        )
        _track_active_conversation(convo)
        prior_cost = Decimal("0")
        try:
            started = time.monotonic()
            repair_window = (float("inf") if os.environ.get(
                "AUTORESEARCH_AGENTIC_NO_TIME_LIMIT") == "1" else 300.0)
            if not getattr(llm, "native_tool_calling", True):
                task += (
                    "\n\nRuntime tool interface: native_tool_calling=false. "
                    "This SDK uses emulated function calls, not provider-native calls. "
                    "Follow the SDK tool schema: <function=NAME> with separate "
                    "<parameter=NAME>value</parameter> fields and </function>, "
                    "without an outer <tool_call> wrapper. This actual configuration "
                    "takes precedence over prior advice about native calls or prohibiting "
                    "all XML. Wait for the SDK tool observation before claiming execution. "
                    "Keep the existing scientific task, checkpoints, and budgets unchanged."
                )
            else:
                task += (
                    "\n\nRuntime tool interface: native_tool_calling=true. "
                    "Use the provided structured tool_calls interface with separate JSON arguments. "
                    "Tool markup such as <tool_call> or <function=NAME> in ordinary assistant "
                    "text is not executable in this mode. This actual configuration takes "
                    "precedence over prior advice about emulated calls. Wait for the SDK tool "
                    "observation before claiming execution. Keep the existing scientific task, "
                    "checkpoints, and budgets unchanged."
                )
            def run_conversation(run_timeout, predicate=None):
                return self._run_convo(
                    convo, run_timeout, predicate, workspace=handle.workspace)

            convo.send_message(task)
            timed_out, stopped = run_conversation(timeout, stop_predicate)
            def check_completion():
                nonlocal timed_out
                message = completion_check()
                # FINISHED describes the conversation, not a registered background
                # experiment. Wait for its host receipt before asking for serialization.
                deadline = started + (timeout if timeout is not None else repair_window)
                while message and message.startswith("TASK EXECUTION RUNNING:"):
                    remaining_wait = deadline - time.monotonic()
                    if remaining_wait <= 0:
                        timed_out = True
                        return None
                    time.sleep(min(1.0, remaining_wait))
                    message = completion_check()
                return message
            st = getattr(convo.state, "execution_status", None)
            initial_status = getattr(st, "name", str(st))
            remaining = (max(0.0, timeout - (time.monotonic() - started))
                         if timeout is not None else repair_window)
            if (not timed_out and not stopped and initial_status == "FINISHED"
                    and remaining > 0):
                message = _tool_markup_repair_message(
                    convo, native_tool_calling=getattr(llm, "native_tool_calling", True))
                protocol_repair = bool(message)
                protocol_repairs = int(protocol_repair)
                pending_event = _terminal_wait_event(convo)
                pending_terminal = pending_event is not None
                if pending_terminal:
                    message = (message + "\n" if message else "") + (
                        "The terminal reports a still-running command (a soft wait or a "
                        "new command rejected as busy), not an execution failure. "
                        "Set command to a single space character in that same terminal "
                        "to wait, leaving input mode at its default and reset=false. "
                        "The terminal strips whitespace and treats this as an empty command; "
                        "the nonempty string avoids malformed empty-value serialization. "
                        "Use only parameters declared by the current terminal tool schema. "
                        "Check/wait for the original process within "
                        "the remaining task budget, then save its actual result and attempt ID. "
                        "Do not restart, interrupt, or duplicate the experiment, launch a new "
                        "variant, or claim success before its exit and result are observed."
                    )
                if not message and completion_check is not None:
                    message = check_completion()
                if message:
                    resume_execution = message.startswith("TASK EXECUTION INCOMPLETE:")
                    convo.send_message(message)
                    remaining = (max(0.0, timeout - (time.monotonic() - started))
                                 if timeout is not None else repair_window)
                    if remaining > 0:
                        # Result repair can require an actual evaluator/control run,
                        # not merely JSON editing. Keep the original sandbox deadline.
                        repair_budget = remaining
                        repair_deadline = time.monotonic() + repair_budget
                        repair_event_count = len(getattr(convo.state, "events", ()))
                        timed_out, stopped = run_conversation(
                            repair_budget,
                            stop_predicate if (protocol_repair or resume_execution)
                            and not pending_terminal else None)
                        while (not pending_terminal
                               and not timed_out and not stopped
                               and getattr(convo.state.execution_status, "name", "") == "FINISHED"):
                            next_protocol = _tool_markup_repair_message(
                                convo, native_tool_calling=getattr(llm, "native_tool_calling", True))
                            if not next_protocol:
                                protocol_repair = False
                                break  # Recovered execution still needs the normal result check.
                            progress = any(
                                getattr(e, "kind", None) == "ObservationEvent"
                                and getattr(getattr(e, "observation", None), "is_error", None) is False
                                for e in convo.state.events[repair_event_count:])
                            recovery_remaining = max(0.0, repair_deadline - time.monotonic())
                            # A result-repair turn can introduce its first protocol error.
                            # Give that error one repair; repeated errors require tool progress.
                            if ((protocol_repair and not progress and protocol_repairs >= 3)
                                    or recovery_remaining <= 0):
                                break
                            protocol_repair = True
                            protocol_repairs += 1
                            repair_event_count = len(convo.state.events)
                            convo.send_message(next_protocol)
                            timed_out, stopped = run_conversation(
                                recovery_remaining, stop_predicate)
                        while (pending_terminal and not timed_out and not stopped
                               and getattr(convo.state.execution_status, "name", "") == "FINISHED"):
                            latest_wait = _terminal_wait_event(convo)
                            if latest_wait is None:
                                pending_terminal = False
                                break
                            if (latest_wait is pending_event or
                                    (getattr(latest_wait, "id", None) is not None
                                     and latest_wait.id == getattr(pending_event, "id", None))):
                                # A malformed wait call cannot create a new observation.
                                # Repair only that protocol, using the existing bounded
                                # allowance; unchanged model prose cannot extend a wait.
                                if (completion_check is None or protocol_repairs >= 3
                                        or not _tool_markup_repair_message(
                                            convo, native_tool_calling=getattr(
                                                llm, "native_tool_calling", True))
                                        or not completion_check()):
                                    break
                                protocol_repairs += 1
                            remaining_wait = max(0.0, repair_deadline - time.monotonic())
                            if remaining_wait <= 0:
                                timed_out = True
                                break
                            pending_event = latest_wait
                            convo.send_message(message)
                            timed_out, stopped = run_conversation(remaining_wait)
                        if (not protocol_repair and not pending_terminal
                                and completion_check is not None
                                and not timed_out and not stopped
                                and getattr(convo.state.execution_status, "name", "") == "FINISHED"
                                and time.monotonic() < repair_deadline):
                            next_message = check_completion()
                            repair_remaining = max(0.0, repair_deadline - time.monotonic())
                            if next_message and next_message != message and repair_remaining > 0:
                                convo.send_message(next_message)
                                timed_out, stopped = run_conversation(repair_remaining)
                        # A conversation can keep copying malformed pseudo calls before
                        # or after real tool actions. Give its existing workspace one fresh,
                        # bounded context to finish the missing result; never replay a command
                        # from assistant text or grant a new wall-clock budget.
                        if (protocol_repair and not pending_terminal
                                and not timed_out and not stopped
                                and completion_check is not None):
                            missing = check_completion()
                            recovery_remaining = max(0.0, repair_deadline - time.monotonic())
                            if missing and recovery_remaining > 0:
                                old_convo = convo
                                prior_cost += Decimal(_accum_cost(old_convo))
                                convo = Conversation(
                                    agent=agent, workspace=handle.workspace,
                                    token_callbacks=[_discard_stream_token],
                                    max_iteration_per_run=_max_iter, visualizer=None,
                                    stuck_detection=False)
                                _track_active_conversation(convo)
                                try:
                                    old_convo.close()
                                except Exception:
                                    pass
                                _untrack_active_conversation(old_convo)
                                convo.send_message(
                                    task + "\n\nRECOVERY: The prior conversation ended with repeated "
                                    "unexecuted tool markup. Inspect the existing "
                                    "workspace and host attempt receipts. Do not repeat a running "
                                    "command or claim its result. Finish only the missing evidence "
                                    "within the original budget.\n" + missing)
                                timed_out, stopped = run_conversation(
                                    recovery_remaining, stop_predicate)
                    else:
                        timed_out = True
            if stopped:
                status = "PAUSED_BY_PREDICATE"
            elif timed_out:
                status = "TIMEOUT"
            else:
                st = getattr(convo.state, "execution_status", None)
                status = getattr(st, "name", str(st))
            return RunResult(
                status=str(status), cost=str(prior_cost + Decimal(_accum_cost(convo))),
                event_summary=_terminal_event_summary(convo),
            )
        except Exception as exc:
            # Conversation.run() often raises only "ended with error". Preserve the
            # provider's terminal error before closing the conversation, so an external
            # quota outage is not misclassified as a failed experiment.
            detail = _terminal_event_summary(convo, error_workspace=handle.workspace)
            if detail and detail not in str(exc):
                raise RuntimeError(f"{exc}: {detail}") from exc
            raise
        finally:
            try:
                convo.close()
            except Exception:
                pass
            _untrack_active_conversation(convo)

    @staticmethod
    def _run_convo(convo, timeout: float | None, stop_predicate=None, *, workspace=None) -> tuple[bool, bool]:
        """Run ``convo.run()``; return ``(timed_out, stopped_by_predicate)``.
        ``timeout=None`` runs inline (no thread).

        On the timeout path ``convo.run()`` runs on a daemon thread. If it CRASHES
        (e.g. ``RuntimeError: TmuxPanePool is not initialized`` on a loaded CUDA
        sandbox), the exception is captured and RE-RAISED on the caller's thread.
        Otherwise a hard agent crash is silently swallowed -- ``done.set()`` fires
        in ``finally``, the caller reads a stale status, exports an UNMODIFIED
        workspace, and the crash masquerades as an agent that "made no changes",
        defeating sandbox_propose's retry+fallback (seen live on SOL-ExecBench: two
        rounds produced no export at all, method came back None). A run still
        executing when the deadline passes returns TIMEOUT exactly as before.
        """
        import os
        _run_to = (float("inf") if os.environ.get("AUTORESEARCH_AGENTIC_NO_TIME_LIMIT") == "1"
                   else float(os.environ.get("AUTORESEARCH_SANDBOX_CONVERSATION_TIMEOUT", "10800")))

        def _run():
            # SDK RemoteConversation.run() defaults to timeout=3600s (1h) -- too short for a SLOW
            # hosted model (glm via a gateway) that must establish a baseline + edit + re-measure
            # inside ONE conversation (seen live: nanoGPT spent the whole hour on baseline+explore
            # and timed out BEFORE its first edit -> method=None). Raise it (default 3h); env-tunable.
            # Detect legacy signatures before launching: an internal TypeError may
            # occur after the SDK's POST and must never trigger a second launch.
            import inspect
            try:
                inspect.signature(convo.run).bind(timeout=_run_to)
            except TypeError:
                convo.run()
            else:
                convo.run(timeout=_run_to)

        if (timeout is None and stop_predicate is None
                and os.environ.get("AUTORESEARCH_AGENTIC_NO_TIME_LIMIT") != "1"):
            _run()
            return False, False

        done = threading.Event()
        cancelled = threading.Event()
        captured: dict = {}
        poll_status = getattr(convo, "_poll_status_once", None)
        if callable(poll_status):
            # RemoteConversation.close() closes its WebSocket but does not stop a
            # blocking REST wait. Cancel only this instance's polling after our
            # outer wait exits; never patch the shared SDK or report success.
            def cancellable_poll_status():
                if cancelled.is_set():
                    raise RuntimeError("AutoResearch sandbox wait cancelled")
                return poll_status()

            convo._poll_status_once = cancellable_poll_status

        def _target():
            try:
                _run()
            except Exception as exc:  # noqa: BLE001 -- surface on caller thread, don't swallow
                captured["exc"] = exc
            finally:
                if callable(poll_status):
                    convo._poll_status_once = poll_status
                done.set()

        runner = threading.Thread(target=_target, daemon=True)
        runner.start()
        deadline = None if timeout is None else time.monotonic() + timeout
        next_container_check = time.monotonic()
        try:
            while True:
                wait_s = 0.5
                if deadline is not None:
                    wait_s = max(0.0, min(wait_s, deadline - time.monotonic()))
                if done.wait(wait_s):
                    if "exc" in captured:
                        raise captured["exc"]
                    return False, False
                if workspace is not None and time.monotonic() >= next_container_check:
                    _check_sandbox_container(workspace)
                    next_container_check = time.monotonic() + 30
                if stop_predicate is not None:
                    try:
                        should_stop = bool(stop_predicate())
                    except Exception:  # noqa: BLE001 -- monitoring must be fail-open
                        logger.warning("agent stop predicate failed", exc_info=True)
                        should_stop = False
                    if should_stop:
                        try:
                            convo.pause()
                        except Exception:  # noqa: BLE001 -- result salvage still runs
                            logger.warning("agent pause after stop predicate failed", exc_info=True)
                        return False, True
                if deadline is not None and time.monotonic() >= deadline:
                    return True, False
        finally:
            cancelled.set()
            if callable(poll_status) and not done.is_set():
                runner.join(timeout=1.5)

    def export_workspace(
        self, handle: SandboxHandle, dest_dir, *, src: str = "/workspace"
    ) -> Path:
        """Export the container workspace ``src`` to a local ``dest_dir``.
        把容器工作区 src 导出到本地 dest_dir。

        SDK's ``file_download``/``file_upload`` are single-file (no whole-dir
        API), so we tar the dir in-container, download the tarball, and extract
        it host-side.
        SDK 只有单文件下载,没有整目录 API,故必须容器内打包 → 单文件下载 → 宿主解包。
        """
        dest = Path(dest_dir)
        dest.mkdir(parents=True, exist_ok=True)
        # 1) 容器内打包整个 src 目录;tar 非零退出(缺 src、磁盘满)必须先炸,
        #    否则下面会下载一个陈旧/不存在的 tarball,变成不透明的下游错误。
        r = self.execute_command(handle, f"tar czf /tmp/ws_export.tgz -C {src} .")
        if r.exit_code != 0:
            raise InfraError(f"workspace tar failed (exit {r.exit_code}): {r.stderr}")
        # 2) 单文件下载 tarball 到宿主
        handle.workspace.file_download("/tmp/ws_export.tgz", str(dest / "ws_export.tgz"))
        # 3) 宿主侧解包。tarball 由容器内 agent 产出,不可信 —— filter="data" 阻断
        #    路径穿越(../ 逃逸)并消除 Py3.14 弃用告警。
        import tarfile

        with tarfile.open(dest / "ws_export.tgz") as tf:
            tf.extractall(dest, filter="data")
        return dest
