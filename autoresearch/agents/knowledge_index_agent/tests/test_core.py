"""
Tests for KnowledgeIndexAgent's pure index + retrieve core (M3 Task 2).
KnowledgeIndexAgent 纯"索引 + 检索"核心的测试。

No EventBus / Store / Redis / network / LLM -- ``index_entries`` and
``retrieve`` are pure functions. ``retrieve`` is exercised against the REAL
``KeywordBackend`` (local, deterministic lexical ranking), no fake needed.
不涉及 EventBus / Store / Redis / 网络 / LLM -- ``index_entries`` 和
``retrieve`` 都是纯函数。``retrieve`` 使用真实的 ``KeywordBackend``
（本地确定性词形排序）测试，无需 fake。
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from jsonschema import Draft202012Validator

from autoresearch.agents._reuse.knowledge_index_backend import KeywordBackend
from autoresearch.agents.knowledge_index_agent.core import (
    KnowledgeIndexError,
    index_entries,
    retrieve,
)

_CONTRACTS_DIR = Path(__file__).resolve().parents[1] / "data-contracts"
INDEX_ENTRY_VALIDATOR = Draft202012Validator(
    yaml.safe_load((_CONTRACTS_DIR / "index_entry.schema.yaml").read_text(encoding="utf-8"))
)


def _assert_schema_valid(entry: dict) -> None:
    errors = list(INDEX_ENTRY_VALIDATOR.iter_errors(entry))
    assert errors == [], [e.message for e in errors]


# ---------------------------------------------------------------------------
# (a) index_entries normalizes IndexEntryInputs into full schema-valid
#     IndexEntries (type/source/produced_by_state filled).
# ---------------------------------------------------------------------------


def test_index_entries_normalizes_input_into_schema_valid_entries():
    new_items = [
        {
            "path_or_ref": "literature_brief://sess-1/final",
            "summary": "Retrieval-augmented generation literature landscape.",
            "tags": ["rag", "literature"],
        },
    ]

    result = index_entries(
        [],
        new_items,
        produced_by_state="DeepResearching",
        source="DeepResearchAgent",
        entry_type="literature_brief",
    )

    assert len(result) == 1
    entry = result[0]
    assert entry["path_or_ref"] == "literature_brief://sess-1/final"
    assert entry["type"] == "literature_brief"
    assert entry["summary"] == new_items[0]["summary"]
    assert entry["source"] == "DeepResearchAgent"
    assert entry["produced_by_state"] == "DeepResearching"
    assert entry["tags"] == ["rag", "literature"]
    _assert_schema_valid(entry)


def test_index_entries_default_entry_type_is_paper():
    result = index_entries(
        [],
        [{"path_or_ref": "paper://p1", "summary": "A paper about graphs."}],
        produced_by_state="S1",
        source="AgentA",
    )

    assert result[0]["type"] == "paper"
    _assert_schema_valid(result[0])


# ---------------------------------------------------------------------------
# (b) tags derived from summary when absent; provided tags preserved.
# ---------------------------------------------------------------------------


def test_tags_derived_from_summary_when_absent():
    new_items = [
        {
            "path_or_ref": "evidence://sess-1/e1",
            "summary": "Docker build failed due to missing dependency.",
        },
    ]

    result = index_entries(
        [],
        new_items,
        produced_by_state="Acceptance",
        source="AcceptanceAgent",
        entry_type="evidence",
    )

    entry = result[0]
    assert "docker" in entry["tags"]
    _assert_schema_valid(entry)


def test_tags_preserved_when_present():
    new_items = [
        {
            "path_or_ref": "evidence://sess-1/e2",
            "summary": "Something happened during the build.",
            "tags": ["custom-tag"],
        },
    ]

    result = index_entries(
        [],
        new_items,
        produced_by_state="Acceptance",
        source="AcceptanceAgent",
    )

    assert result[0]["tags"] == ["custom-tag"]


# ---------------------------------------------------------------------------
# (c) dedup by path_or_ref: same ref indexed twice -> one entry (later
#     overwrites, position preserved); different refs -> both kept, order
#     preserved.
# ---------------------------------------------------------------------------


def test_dedup_same_ref_overwrites_keeping_position():
    existing = index_entries(
        [],
        [{"path_or_ref": "ref-a", "summary": "First version summary."}],
        produced_by_state="S1",
        source="AgentA",
    )
    existing = index_entries(
        existing,
        [{"path_or_ref": "ref-b", "summary": "Second entry summary."}],
        produced_by_state="S1",
        source="AgentA",
    )

    updated = index_entries(
        existing,
        [{"path_or_ref": "ref-a", "summary": "Updated version summary."}],
        produced_by_state="S2",
        source="AgentB",
    )

    assert len(updated) == 2
    assert [e["path_or_ref"] for e in updated] == ["ref-a", "ref-b"]  # order preserved
    assert updated[0]["summary"] == "Updated version summary."
    assert updated[0]["source"] == "AgentB"
    assert updated[0]["produced_by_state"] == "S2"


def test_dedup_different_refs_both_kept_order_preserved():
    result = index_entries(
        [],
        [
            {"path_or_ref": "ref-x", "summary": "X summary."},
            {"path_or_ref": "ref-y", "summary": "Y summary."},
        ],
        produced_by_state="S1",
        source="AgentA",
    )

    assert [e["path_or_ref"] for e in result] == ["ref-x", "ref-y"]


def test_dedup_within_same_batch_last_wins():
    result = index_entries(
        [],
        [
            {"path_or_ref": "ref-a", "summary": "First summary."},
            {"path_or_ref": "ref-a", "summary": "Second summary wins."},
        ],
        produced_by_state="S1",
        source="AgentA",
    )

    assert len(result) == 1
    assert result[0]["summary"] == "Second summary wins."


# ---------------------------------------------------------------------------
# (d) retrieve delegates to the (real) KeywordBackend: lexically-relevant
#     entries ranked, top_k respected, empty index -> [].
# ---------------------------------------------------------------------------


def test_retrieve_ranks_via_keyword_backend_and_respects_top_k():
    entries = index_entries(
        [],
        [
            {
                "path_or_ref": "literature_brief://sess-1/final",
                "summary": "Retrieval-augmented generation literature landscape.",
                "tags": ["rag", "literature"],
            },
            {
                "path_or_ref": "evidence://sess-1/e1",
                "summary": "Docker build failed due to missing dependency.",
                "tags": ["docker", "build"],
            },
            {
                "path_or_ref": "research_spec://sess-1/final",
                "summary": "Research spec for retrieval-augmented fine-tuning benchmarks.",
                "tags": ["retrieval", "benchmarks"],
            },
        ],
        produced_by_state="S1",
        source="AgentA",
    )

    backend = KeywordBackend()
    results = retrieve("retrieval-augmented generation", entries, backend, top_k=10)

    result_refs = [e["path_or_ref"] for e in results]
    assert "literature_brief://sess-1/final" in result_refs
    assert "research_spec://sess-1/final" in result_refs
    assert "evidence://sess-1/e1" not in result_refs

    limited = retrieve("retrieval-augmented generation", entries, backend, top_k=1)
    assert len(limited) == 1


def test_retrieve_on_empty_index_returns_empty_list():
    backend = KeywordBackend()
    assert retrieve("anything", [], backend) == []


# ---------------------------------------------------------------------------
# (e) an input item missing path_or_ref/summary -> KnowledgeIndexError (typed),
#     no crash.
# ---------------------------------------------------------------------------


def test_missing_path_or_ref_raises_typed_error():
    with pytest.raises(KnowledgeIndexError):
        index_entries(
            [],
            [{"summary": "No ref here."}],
            produced_by_state="S1",
            source="AgentA",
        )


def test_missing_summary_raises_typed_error():
    with pytest.raises(KnowledgeIndexError):
        index_entries(
            [],
            [{"path_or_ref": "ref-z"}],
            produced_by_state="S1",
            source="AgentA",
        )
