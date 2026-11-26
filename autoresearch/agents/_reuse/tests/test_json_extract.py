"""
Tests for extract_json: a pure, no-IO helper shared by every agent core that
pulls the JSON payload out of a real-LLM response so json.loads succeeds even
when the model wraps its answer in markdown ```json fences or <think> reasoning
blocks.
"""

from __future__ import annotations

import json

from autoresearch.agents._reuse.json_extract import extract_json


def test_clean_object_returned_unchanged():
    content = '{"a": 1}'

    result = extract_json(content)

    assert result == content
    assert json.loads(result) == {"a": 1}


def test_fenced_object_with_json_lang_tag_is_extracted():
    content = '```json\n{"a": 1}\n```'

    assert extract_json(content) == '{"a": 1}'


def test_fenced_object_without_lang_tag_is_extracted():
    content = '```\n{"a": 1}\n```'

    assert extract_json(content) == '{"a": 1}'


def test_think_block_is_stripped():
    content = '<think>reasoning...</think>\n{"a": 1}'

    assert extract_json(content) == '{"a": 1}'


def test_think_block_plus_fence_is_stripped_and_extracted():
    content = '<think>reasoning...</think>\n```json\n{"a": 1}\n```'

    assert extract_json(content) == '{"a": 1}'


def test_prose_around_json_is_stripped():
    content = 'Here is the result:\n{"a": 1}\nDone.'

    assert extract_json(content) == '{"a": 1}'


def test_trailing_prose_after_clean_object_is_stripped():
    content = '{"a": {"text": "a } inside a string"}}\nThis is the requested result.'

    result = extract_json(content)

    assert result == '{"a": {"text": "a } inside a string"}}'
    assert json.loads(result) == {"a": {"text": "a } inside a string"}}


def test_first_complete_object_is_selected_when_model_appends_an_example():
    content = '{"a": 1}\nExample shape: {"a": 2}'

    assert extract_json(content) == '{"a": 1}'


def test_clean_array_returned_unchanged():
    content = "[1, 2, 3]"

    assert extract_json(content) == content


def test_fenced_array_is_extracted():
    content = "```json\n[1, 2, 3]\n```"

    assert extract_json(content) == "[1, 2, 3]"


def test_empty_string_returned_unchanged():
    assert extract_json("") == ""


def test_whitespace_only_returned_unchanged():
    assert extract_json("   \n  ") == "   \n  "


def test_big_nested_object_inside_fence_is_extracted_and_parseable():
    nested = {
        "literature_brief": "some text",
        "field_trajectory_brief": "more text",
        "papers": [
            {"title": "A", "score": 1, "meta": {"year": 2024, "tags": ["x", "y"]}},
            {"title": "B", "score": 2, "meta": {"year": 2025, "tags": []}},
        ],
    }
    content = "```json\n" + json.dumps(nested) + "\n```"

    result = extract_json(content)

    assert json.loads(result) == nested
