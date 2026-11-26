"""Unit tests for the KnowledgeIndexBackend adapter interface (M3 Task 1).

KeywordBackend must rank entries by lexical (token-overlap) match against the
query, exclude unrelated entries, respect top_k, return [] on no match, and
be deterministic. The opt-in real (semantic) backend stub must never actually
run -- it raises NotImplementedError.
"""
from __future__ import annotations

from autoresearch.agents._reuse.knowledge_index_backend import (
    KeywordBackend,
    KnowledgeIndexBackend,
    SemanticBackend,
)

import pytest

_ENTRIES = [
    {
        "path_or_ref": "literature_brief://sess-1/final",
        "type": "literature_brief",
        "summary": "Retrieval-augmented generation literature landscape.",
        "tags": ["rag", "literature"],
        "source": "DeepResearchAgent",
        "produced_by_state": "DeepResearching",
    },
    {
        "path_or_ref": "evidence://sess-1/e1",
        "type": "evidence",
        "summary": "Docker build failed due to missing dependency.",
        "tags": ["docker", "build"],
        "source": "AcceptanceAgent",
        "produced_by_state": "Acceptance",
    },
    {
        "path_or_ref": "research_spec://sess-1/final",
        "type": "research_spec",
        "summary": "Research spec for retrieval-augmented fine-tuning benchmarks.",
        "tags": ["retrieval", "benchmarks"],
        "source": "ProblemModelingAgent",
        "produced_by_state": "ClarifyingResearchGoal",
    },
]


def test_keyword_backend_ranks_by_lexical_overlap_and_excludes_unrelated():
    backend = KeywordBackend()

    results = backend.retrieve("retrieval-augmented generation", _ENTRIES, top_k=10)

    result_refs = [e["path_or_ref"] for e in results]
    assert "literature_brief://sess-1/final" in result_refs
    assert "research_spec://sess-1/final" in result_refs
    assert "evidence://sess-1/e1" not in result_refs

    # The entry matching more query tokens ("retrieval", "augmented",
    # "generation" all present) should rank above the one matching fewer
    # ("retrieval" only).
    assert result_refs.index("literature_brief://sess-1/final") < result_refs.index(
        "research_spec://sess-1/final"
    )


def test_keyword_backend_respects_top_k():
    backend = KeywordBackend()

    results = backend.retrieve("retrieval", _ENTRIES, top_k=1)

    assert len(results) == 1


def test_keyword_backend_returns_empty_on_no_match():
    backend = KeywordBackend()

    results = backend.retrieve("quantum entanglement teleportation", _ENTRIES, top_k=10)

    assert results == []


def test_keyword_backend_is_deterministic():
    backend = KeywordBackend()

    first = backend.retrieve("retrieval-augmented generation", _ENTRIES, top_k=10)
    second = backend.retrieve("retrieval-augmented generation", _ENTRIES, top_k=10)

    assert first == second


def test_keyword_backend_satisfies_protocol():
    backend: KnowledgeIndexBackend = KeywordBackend()
    assert hasattr(backend, "retrieve")


def test_semantic_backend_stub_raises_not_implemented():
    backend = SemanticBackend()
    with pytest.raises(NotImplementedError):
        backend.retrieve("q", _ENTRIES, top_k=5)
