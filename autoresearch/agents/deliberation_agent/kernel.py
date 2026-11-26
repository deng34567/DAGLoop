"""DeliberationAgent kernel: AGENT_NAME + optional prompt-patch config loader.

Mirrors deep_research_agent/kernel.py; AGENT_NAME = "DeliberationAgent".
"""
from __future__ import annotations

import os
from pathlib import Path

AGENT_NAME = "DeliberationAgent"


def default_agent_config_root() -> Path:
    base = os.environ.get("VIBE_AGENT_CONFIG_ROOT")
    return Path(base) if base else Path(__file__).resolve().parents[3]


class ConfigLoader:
    """Loads an optional prompt patch for DeliberationAgent (soft-degrades to none)."""

    def __init__(self, config_root: str | Path | None = None) -> None:
        self.config_root = (
            Path(config_root) if config_root is not None else default_agent_config_root()
        )

    def prompt_patch(self) -> str | None:
        if self.config_root is None:
            return None
        patch_path = self.config_root / "agent_config" / AGENT_NAME / "prompt_patch.md"
        if patch_path.is_file():
            text = patch_path.read_text(encoding="utf-8").strip()
            return text or None
        return None


__all__ = ["AGENT_NAME", "ConfigLoader", "default_agent_config_root"]
