"""Contract: every user-visible-text prompt instructs the LLM to mirror the user's language.

Chinese question was getting English clarifying options/hypotheses/reports because none of the
generation prompts told the model to answer in the user's language. This locks the directive in
place (natural-language values only -- keys/enums stay English so parsing is unaffected)."""
from __future__ import annotations

from pathlib import Path

import pytest

_AGENTS = Path(__file__).resolve().parents[1]

# prompts that emit human-readable text the user actually reads in the viewer
USER_VISIBLE_PROMPTS = [
    "problem_modeling_agent/prompts/clarify_system.md",      # clarifying questions + options
    "problem_modeling_agent/prompts/model_system.md",        # research-direction summary at checkpoint
    "problem_modeling_agent/prompts/reinterpret_system.md",
    "deep_research_agent/prompts/synthesize_system.md",      # literature brief
    "hypothesis_agent/prompts/generate_system.md",           # hypotheses
    "evaluator_agent/prompts/evaluate_experiment_system.md", # verdict reasons / report text
]

MARK = "## Output language (mirror the user)"


@pytest.mark.parametrize("rel", USER_VISIBLE_PROMPTS)
def test_user_visible_prompt_mirrors_language(rel):
    txt = (_AGENTS / rel).read_text(encoding="utf-8")
    assert MARK in txt, f"{rel} is missing the language-mirror directive"
    # must ALSO protect machine fields, or a translated JSON key/enum would break parsing
    assert "keep every JSON key" in txt and "identifier EXACTLY" in txt
