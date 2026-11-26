"""
DataAnalysisAgent kernel: config loading + audit recording.
DataAnalysisAgent 内核：配置加载 + 审计记录。

Mirrors autoresearch/agents/ranking_judge_agent/kernel.py, with
AGENT_NAME = "DataAnalysisAgent". ``AuditRecorder`` uses a generic
``input_refs`` mapping (keyed by whichever refs the calling flow used --
e.g. an external-data-upload/protocol ref), since the exact upstream ref
shape is a later (handler) milestone's concern.
镜像 autoresearch/agents/ranking_judge_agent/kernel.py，AGENT_NAME 改为
"DataAnalysisAgent"。``AuditRecorder`` 沿用通用的 ``input_refs`` 映射（以
调用流程实际用到的 ref 为键，例如外部数据上传/协议 ref），因为具体上游
ref 形状是后续（handler）里程碑的事。

- ConfigLoader: optionally loads
  ``agent_config/DataAnalysisAgent/prompt_patch.md`` and appends it to the
  system prompt.
- AuditRecorder: writes one ``kind="audit"`` artifact per handler run.

- ConfigLoader：可选加载
  ``agent_config/DataAnalysisAgent/prompt_patch.md`` 追加到 system prompt。
- AuditRecorder：每次 handler 运行写一条 ``kind="audit"`` 产物。
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

AGENT_NAME = "DataAnalysisAgent"


def default_agent_config_root() -> Path:
    """DataAnalysisAgent 的配置根 ``agent_config/``（未含 agent 名子目录）。

    可用 env ``VIBE_AGENT_CONFIG_ROOT`` 覆盖 base（指向 ``agent_config/`` 的
    父级）。默认相对仓库根：kernel.py -> data_analysis_agent -> agents ->
    autoresearch -> repo root。目录不存在时 ConfigLoader 软降级为空配置。
    """
    base = os.environ.get("VIBE_AGENT_CONFIG_ROOT")
    return Path(base) if base else Path(__file__).resolve().parents[3]


class ConfigLoader:
    """Loads optional prompt patch for DataAnalysisAgent.
    加载 DataAnalysisAgent 可选 prompt patch。"""

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
            text = patch_path.read_text(encoding="utf-8").split("<!-- MOSS-PATCH", 1)[0].strip()
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
    """Writes a DataAnalysisAgent audit_record to the Spec Store
    (kind="audit"). 把 DataAnalysisAgent 审计记录写入 Spec Store
    （kind="audit"）。"""

    def __init__(self, spec_store: Any) -> None:
        self.spec_store = spec_store

    def _write(self, session_id: str, record: dict) -> str:
        key = f"data_analysis_agent/{record['timestamp']}"
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
        """Write a success audit_record. / 写一条成功审计。"""
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
