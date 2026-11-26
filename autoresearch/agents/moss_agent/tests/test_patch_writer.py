"""
Tests for MOSS's PatchWriter: appends MOSS guidance to the FLAT path each
target agent's own ConfigLoader reads -- config_root/agent_config/{agent}/
prompt_patch.md. `agent` is the raw AGENT_NAME string (e.g. "CoderAgent"),
not a mapped/aliased name.

The path MUST match the read side exactly, or a written patch is silently
never read; test_append_prompt_patch_round_trips_through_coder_agent_config_loader
is the one that would catch a path mismatch, by reading the patch back
through CoderAgent's own kernel ConfigLoader.
"""

from __future__ import annotations

from autoresearch.agents.coder_agent.kernel import ConfigLoader as CoderConfigLoader
from autoresearch.agents.moss_agent.patch_writer import PatchWriter


def test_append_prompt_patch_creates_file_at_flat_path(tmp_path):
    writer = PatchWriter(tmp_path)

    writer.append_prompt_patch("CoderAgent", "prefer smaller diffs")

    path = tmp_path / "agent_config" / "CoderAgent" / "prompt_patch.md"
    assert path.is_file()
    assert "prefer smaller diffs" in path.read_text(encoding="utf-8")


def test_append_prompt_patch_appends_not_overwrites(tmp_path):
    writer = PatchWriter(tmp_path)

    writer.append_prompt_patch("CoderAgent", "prefer smaller diffs")
    writer.append_prompt_patch("CoderAgent", "also add tests")

    path = tmp_path / "agent_config" / "CoderAgent" / "prompt_patch.md"
    text = path.read_text(encoding="utf-8")
    assert "prefer smaller diffs" in text
    assert "also add tests" in text


def test_legacy_global_moss_patch_is_not_loaded_as_reviewed_guidance(tmp_path):
    writer = PatchWriter(tmp_path)
    writer.append_prompt_patch("CoderAgent", "prefer smaller diffs")

    loader = CoderConfigLoader(tmp_path)
    patch = loader.prompt_patch()

    assert patch is None
    assert "prefer smaller diffs" in (tmp_path / "agent_config/CoderAgent/prompt_patch.md").read_text()


def test_append_prompt_patch_write_is_atomic_no_leftover_tmp_file(tmp_path):
    writer = PatchWriter(tmp_path)
    writer.append_prompt_patch("CoderAgent", "prefer smaller diffs")

    path = tmp_path / "agent_config" / "CoderAgent" / "prompt_patch.md"
    assert not path.with_suffix(path.suffix + ".tmp").exists()


def test_append_prompt_patch_caps_total_length_and_keeps_newest(tmp_path):
    writer = PatchWriter(tmp_path)

    for i in range(50):
        writer.append_prompt_patch("CoderAgent", f"guidance chunk number {i} " * 5)

    path = tmp_path / "agent_config" / "CoderAgent" / "prompt_patch.md"
    text = path.read_text(encoding="utf-8")

    assert len(text) <= 5200  # ~5000-char cap, small tolerance for header overhead
    assert "guidance chunk number 49" in text
    assert "guidance chunk number 0" not in text
