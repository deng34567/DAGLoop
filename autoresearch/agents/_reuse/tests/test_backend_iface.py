"""Unit tests for the DeepResearchBackend adapter interface (M2 Task 1).

FakeBackend must return Paper-shaped dicts (matching the LiteratureBrief
`Paper` shape: title, id_or_url, optional year/note) from both search passes
and record how many times each was called. The opt-in real backend stub must
never actually run — it raises NotImplementedError.
"""
from __future__ import annotations

import pytest

from autoresearch.agents._reuse.deep_research_backend import (
    DeepResearchBackend,
    FakeBackend,
    PaperQA2Backend,
)


def _assert_paper_shape(paper: dict) -> None:
    assert isinstance(paper, dict)
    assert isinstance(paper.get("title"), str) and paper["title"]
    assert isinstance(paper.get("id_or_url"), str) and paper["id_or_url"]
    if "year" in paper:
        assert isinstance(paper["year"], int)
    if "note" in paper:
        assert isinstance(paper["note"], str)


def test_fake_backend_default_returns_paper_shaped_dicts():
    backend = FakeBackend()

    seminal = backend.search_seminal("retrieval-augmented generation", {})
    frontier = backend.search_frontier("retrieval-augmented generation", {})

    assert seminal, "default FakeBackend should return at least one seminal paper"
    assert frontier, "default FakeBackend should return at least one frontier paper"
    for paper in seminal + frontier:
        _assert_paper_shape(paper)


def test_fake_backend_records_call_counts():
    backend = FakeBackend()

    assert backend.seminal_calls == 0
    assert backend.frontier_calls == 0

    backend.search_seminal("q", {})
    backend.search_seminal("q", {})
    backend.search_frontier("q", {})

    assert backend.seminal_calls == 2
    assert backend.frontier_calls == 1


def test_fake_backend_returns_canned_lists_passed_to_constructor():
    seminal_papers = [{"title": "Seminal Paper", "id_or_url": "arxiv:1111.1111", "year": 2010}]
    frontier_papers = [{"title": "Frontier Paper", "id_or_url": "arxiv:2222.2222", "year": 2024}]
    backend = FakeBackend(seminal=seminal_papers, frontier=frontier_papers)

    assert backend.search_seminal("q", {}) == seminal_papers
    assert backend.search_frontier("q", {}) == frontier_papers


def test_fake_backend_satisfies_protocol():
    backend: DeepResearchBackend = FakeBackend()
    assert hasattr(backend, "search_seminal")
    assert hasattr(backend, "search_frontier")


def test_real_backend_stub_raises_not_implemented():
    backend = PaperQA2Backend()
    with pytest.raises(NotImplementedError):
        backend.search_seminal("q", {})
    with pytest.raises(NotImplementedError):
        backend.search_frontier("q", {})
