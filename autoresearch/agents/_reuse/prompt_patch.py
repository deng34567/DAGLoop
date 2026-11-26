"""
apply_prompt_patch: shared, agent-agnostic helper that splices MOSS-injected
guidance into an agent's system prompt.

PURE -- no IO, no filesystem access. Reading the patch off disk (via each
agent's own ConfigLoader.prompt_patch()) and writing it (via MOSS's
PatchWriter) are separate concerns handled elsewhere.
"""

from __future__ import annotations


def apply_prompt_patch(system_prompt: str, patch: str | None) -> str:
    """Append `patch` under a "[MOSS-injected guidance]" marker if truthy;
    otherwise return `system_prompt` unchanged."""
    if not patch:
        return system_prompt
    return f"{system_prompt}\n\n[MOSS-injected guidance]\n{patch}"


__all__ = ["apply_prompt_patch"]
