"""
extract_json: shared, agent-agnostic helper that pulls the JSON payload out of
a real-LLM response so json.loads succeeds.

PURE -- no IO, no filesystem access. Real models (e.g. Kimi) routinely wrap
their JSON answer in markdown ```json fences and/or precede it with a
<think>...</think> reasoning block; a bare json.loads(content) crashes on
that. This helper normalizes such responses down to the raw JSON text while
leaving already-clean JSON (starting with `{` or `[`) untouched, so mocked
FakeLLM tests that feed clean JSON are unaffected.
"""

from __future__ import annotations

import json
import re

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_FENCED_JSON_RE = re.compile(r"```(?:json)?\s*(\{.*\}|\[.*\])\s*```", re.DOTALL)


def _first_complete_json(text: str) -> str | None:
    """Return the first complete JSON object/array, ignoring trailing prose."""
    starts = [index for index in (text.find("{"), text.find("[")) if index >= 0]
    if not starts:
        return None
    index = min(starts)
    decoder = json.JSONDecoder()
    try:
        value, end = decoder.raw_decode(text, index)
    except json.JSONDecodeError:
        return None
    return text[index:end] if isinstance(value, (dict, list)) else None


def extract_json(content: str) -> str:
    """Pull the JSON payload out of a real-LLM response so json.loads succeeds.
    Handles: markdown ```json...``` / ```...``` fences, <think>...</think> reasoning
    blocks, and leading/trailing prose. A response that is ALREADY clean JSON is
    returned UNCHANGED (so mocked-clean-JSON tests are unaffected)."""
    if not isinstance(content, str) or not content.strip():
        return content

    s = _THINK_RE.sub("", content)

    fenced_match = _FENCED_JSON_RE.search(s)
    if fenced_match:
        return fenced_match.group(1)

    stripped = s.strip()
    complete = _first_complete_json(stripped)
    if complete is not None:
        # Preserve an already-clean response byte-for-byte.  When a provider
        # appends prose after a complete JSON root, return only that root.
        return content if s == content and complete == content else complete

    first_obj, last_obj = stripped.find("{"), stripped.rfind("}")
    if first_obj != -1 and last_obj != -1 and last_obj > first_obj:
        return stripped[first_obj : last_obj + 1]

    first_arr, last_arr = stripped.find("["), stripped.rfind("]")
    if first_arr != -1 and last_arr != -1 and last_arr > first_arr:
        return stripped[first_arr : last_arr + 1]

    return stripped


__all__ = ["extract_json"]
