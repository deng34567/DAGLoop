"""
RunnerAgent kernel: config loading + audit recording.
RunnerAgent 内核：配置加载 + 审计记录。

Mirrors autoresearch/agents/evaluator_agent/kernel.py, with
AGENT_NAME = "RunnerAgent". RunnerAgent has a single flow --
ReproducingBaseline (reads BenchmarkSpec + VerifierSpec) -- so
``AuditRecorder`` uses the same generic ``input_refs`` mapping shape as
EvaluatorAgent's kernel.
镜像 autoresearch/agents/evaluator_agent/kernel.py，AGENT_NAME 改为
"RunnerAgent"。RunnerAgent 只有一条流程 —— ReproducingBaseline（读
BenchmarkSpec + VerifierSpec）—— 因此 ``AuditRecorder`` 沿用 EvaluatorAgent
kernel 同样的通用 ``input_refs`` 映射。

- ConfigLoader: optionally loads ``agent_config/RunnerAgent/prompt_patch.md``
  and appends it to the system prompt. RunnerAgent has no LLM/system prompt
  of its own in M10; this exists only so a deployment may inject a patch
  without code changes, mirroring the other agents' kernels.
- AuditRecorder: writes one ``kind="audit"`` artifact per handler run.

- ConfigLoader：可选加载 ``agent_config/RunnerAgent/prompt_patch.md`` 追加到
  system prompt。M10 中 RunnerAgent 本身没有 LLM/system prompt；保留它仅为
  部署侧可零代码注入，与其他 agent 的 kernel 保持一致。
- AuditRecorder：每次 handler 运行写一条 ``kind="audit"`` 产物。
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

AGENT_NAME = "RunnerAgent"


def default_agent_config_root() -> Path:
    """RunnerAgent 的配置根 ``agent_config/``（未含 agent 名子目录）。

    可用 env ``VIBE_AGENT_CONFIG_ROOT`` 覆盖 base（指向 ``agent_config/`` 的
    父级）。默认相对仓库根：kernel.py -> runner_agent -> agents ->
    autoresearch -> repo root。目录不存在时 ConfigLoader 软降级为空配置。
    """
    base = os.environ.get("VIBE_AGENT_CONFIG_ROOT")
    return Path(base) if base else Path(__file__).resolve().parents[3]


class ConfigLoader:
    """Loads optional prompt patch for RunnerAgent.
    加载 RunnerAgent 可选 prompt patch。"""

    def __init__(self, config_root: str | Path | None = None) -> None:
        self.config_root = (
            Path(config_root) if config_root is not None else default_agent_config_root()
        )

    def prompt_patch(self) -> str | None:
        """Return the prompt patch text if present, else None.
        若存在则返回 prompt patch 文本，否则 None。"""
        if self.config_root is None:
            return None
        patch_path = self.config_root / "agent_config" / AGENT_NAME / "prompt_patch.md"
        if patch_path.is_file():
            text = patch_path.read_text(encoding="utf-8").strip()
            return text or None
        return None

    def apply(self, system_prompt: str) -> str:
        """Append the patch (if any) to the given prompt. / 把 patch 追加到 prompt。"""
        patch = self.prompt_patch()
        if not patch:
            return system_prompt
        return f"{system_prompt}\n\n# 部署侧补充指引\n\n{patch}"


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class AuditRecorder:
    """Writes a RunnerAgent audit_record to the Spec Store (kind="audit").
    把 RunnerAgent 审计记录写入 Spec Store（kind="audit"）。"""

    def __init__(self, spec_store: Any) -> None:
        self.spec_store = spec_store

    def _write(self, session_id: str, record: dict) -> str:
        key = f"runner/{record['timestamp']}"
        return self.spec_store.put(
            kind="audit",
            session_id=session_id,
            key=key,
            obj=record,
            emitter=AGENT_NAME,
        )

    def record_success(
        self,
        *,
        session_id: str,
        triggered_by_event: str,
        input_refs: dict[str, str | None],
        output_ref: str,
        emitted_event: str,
    ) -> str:
        """Write a success audit_record. / 写一条成功审计。

        ``input_refs`` is e.g.
        ``{"benchmark_spec_ref": ..., "verifier_spec_ref": ...}``.
        """
        record = {
            "session_id": session_id,
            "agent_name": AGENT_NAME,
            "timestamp": _iso_now(),
            "triggered_by_event": triggered_by_event,
            "input_refs": input_refs,
            "output_ref": output_ref,
            "final_outcome": "success",
            "emitted_event": emitted_event,
            "failure_context": {
                "reason_classification": None,
                "underlying_reason": None,
            },
        }
        return self._write(session_id, record)

    def record_failure(
        self,
        *,
        session_id: str,
        triggered_by_event: str,
        input_refs: dict[str, str | None],
        reason_classification: str,
        underlying_reason: str,
    ) -> str:
        """Write a failure audit_record. / 写一条失败审计。"""
        outcome = (
            "failed_permanent" if reason_classification == "permanent" else "failed_transient"
        )
        record = {
            "session_id": session_id,
            "agent_name": AGENT_NAME,
            "timestamp": _iso_now(),
            "triggered_by_event": triggered_by_event,
            "input_refs": input_refs,
            "output_ref": None,
            "final_outcome": outcome,
            "emitted_event": "agent.failed",
            "failure_context": {
                "reason_classification": reason_classification,
                "underlying_reason": underlying_reason,
            },
        }
        return self._write(session_id, record)


__all__ = ["ConfigLoader", "AuditRecorder", "AGENT_NAME", "default_agent_config_root"]
