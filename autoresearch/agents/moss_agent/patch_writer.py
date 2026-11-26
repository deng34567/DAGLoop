"""
PatchWriter: writes MOSS's prompt-patch guidance to the FLAT path each
target agent's OWN ConfigLoader reads:

    config_root/agent_config/{agent}/prompt_patch.md

`config_root` is the repo-root-level dir (the parent of `agent_config`) --
the SAME value each agent's ConfigLoader is constructed with (see e.g.
autoresearch/agents/coder_agent/kernel.py's ConfigLoader). `agent` is the
raw AGENT_NAME string (e.g. "CoderAgent"), never a mapped/aliased name.

Patches are APPENDED under a dated MOSS-PATCH section marker (existing
sections are never overwritten), and the file is capped at ~5000 chars --
when exceeded, the OLDEST MOSS-PATCH section(s) are dropped first, keeping
the newest guidance. Writes are atomic (temp file + os.replace).
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

_MARKER = "<!-- MOSS-PATCH"
_PATCH_CAP_CHARS = 5000


class PatchWriter:
    def __init__(self, config_root: str | Path) -> None:
        self.config_root = Path(config_root)

    def append_prompt_patch(self, agent: str, content: str) -> None:
        """Append a dated MOSS guidance section for `agent`; cap the file
        size (dropping the oldest MOSS-PATCH sections first). Atomic."""
        path = self.config_root / "agent_config" / agent / "prompt_patch.md"
        path.parent.mkdir(parents=True, exist_ok=True)

        ts = datetime.now(timezone.utc).isoformat()
        section = f"\n\n{_MARKER} at={ts} -->\n{content}\n"

        existing = path.read_text(encoding="utf-8") if path.exists() else ""
        combined = existing + section
        if len(combined) > _PATCH_CAP_CHARS:
            combined = self._trim_oldest(combined, _PATCH_CAP_CHARS)

        self._atomic_write(path, combined)

    @staticmethod
    def _trim_oldest(text: str, cap: int) -> str:
        """Keep the most-recent MOSS-PATCH sections that fit under `cap`."""
        parts = text.split(_MARKER)
        head, sections = parts[0], parts[1:]
        kept: list[str] = []
        total = 0
        for sec in reversed(sections):
            chunk = _MARKER + sec
            if total + len(chunk) > cap:
                break
            kept.append(chunk)
            total += len(chunk)
        return head + "".join(reversed(kept))

    @staticmethod
    def _atomic_write(path: Path, text: str) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)


__all__ = ["PatchWriter"]


def scoped_prompt_patch(store, reference, session_id, agent, base=None):
    """Read explicitly routed immutable advice; never mutate shared configuration."""
    if not reference:
        return base
    from autoresearch.spec_store import refs
    from autoresearch.spec_store.identity import artifact_digest
    parts = refs.decode(reference)
    decision = store.get(reference)
    if (parts.kind != "moss_intervention" or parts.session_id != session_id
            or parts.key != "snapshot-" + artifact_digest(decision).split(":", 1)[1]
            or (decision.get("scope") or {}).get("session_id") != session_id):
        raise ValueError("MOSS guidance must be an immutable decision for this session")
    patch = decision.get("prompt_patch") or {}
    if patch.get("agent") != agent:
        return base
    content = str(patch.get("content") or "")
    return (base or "") + "\nSession-scoped recovery advice (official task rules retain priority):\n" + content
