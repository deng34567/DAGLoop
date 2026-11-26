"""
Tests for apply_prompt_patch: a pure, no-IO helper shared by every agent
that wants to splice MOSS-injected guidance into its own system prompt.
"""

from __future__ import annotations

from autoresearch.agents._reuse.prompt_patch import apply_prompt_patch


def test_apply_prompt_patch_appends_guidance_when_patch_present():
    result = apply_prompt_patch("You are CoderAgent.", "prefer smaller diffs")

    assert result == (
        "You are CoderAgent.\n\n[MOSS-injected guidance]\nprefer smaller diffs"
    )


def test_apply_prompt_patch_returns_base_unchanged_when_patch_is_none():
    assert apply_prompt_patch("You are CoderAgent.", None) == "You are CoderAgent."


def test_apply_prompt_patch_returns_base_unchanged_when_patch_is_empty_string():
    assert apply_prompt_patch("You are CoderAgent.", "") == "You are CoderAgent."
