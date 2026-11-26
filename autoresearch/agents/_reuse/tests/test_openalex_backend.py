"""Tests for OpenAlexBackend (real search backend) with httpx MOCKED -- the
test suite never hits the network. Real connectivity is exercised manually /
opt-in only.
"""
from __future__ import annotations

from autoresearch.agents._reuse.deep_research_backend import OpenAlexBackend


class _FakeResp:
    def __init__(self, data):
        self._data = data

    def raise_for_status(self):
        pass

    def json(self):
        return self._data


def test_maps_openalex_results_to_paper_dicts(monkeypatch):
    canned = {"results": [
        {"title": "Attention Is All You Need", "doi": "https://doi.org/10.x",
         "publication_year": 2017, "cited_by_count": 90000},
        {"display_name": "A Recent Paper", "id": "https://openalex.org/W123",
         "publication_year": 2024, "cited_by_count": 5},
        {"title": None},  # no title -> skipped
    ]}
    monkeypatch.setattr("httpx.get", lambda *a, **k: _FakeResp(canned))
    papers = OpenAlexBackend().search_seminal("transformers", {})
    assert len(papers) == 2
    assert papers[0]["title"] == "Attention Is All You Need"
    assert papers[0]["id_or_url"] == "https://doi.org/10.x"
    assert papers[0]["year"] == 2017
    assert "90000 citations" in papers[0]["note"]
    # second falls back to the OpenAlex id when no doi, uses display_name for title
    assert papers[1]["title"] == "A Recent Paper"
    assert papers[1]["id_or_url"] == "https://openalex.org/W123"


def test_frontier_uses_recent_filter_and_relevance(monkeypatch):
    seen = {}

    def _get(url, params=None, **k):
        seen.update(params or {})
        return _FakeResp({"results": []})

    monkeypatch.setattr("httpx.get", _get)
    OpenAlexBackend().search_frontier("q", {})
    assert "from_publication_date" in seen.get("filter", "")
    # relevance sort (no explicit sort) so results are recent AND on-topic, not just newest
    assert "sort" not in seen


def test_seminal_uses_relevance_not_citations(monkeypatch):
    seen = {}
    monkeypatch.setattr("httpx.get", lambda url, params=None, **k: (seen.update(params or {}), _FakeResp({"results": []}))[1])
    OpenAlexBackend().search_seminal("q", {})
    # NO sort override -> OpenAlex defaults to relevance_score:desc for ?search,
    # so results are about the query (citation sort surfaced tangential hits).
    assert "sort" not in seen
    assert "filter" not in seen  # seminal is not date-filtered


def test_api_failure_returns_empty(monkeypatch):
    def _boom(*a, **k):
        raise RuntimeError("network down")

    monkeypatch.setattr("httpx.get", _boom)
    assert OpenAlexBackend().search_seminal("q", {}) == []
    assert OpenAlexBackend().search_frontier("q", {}) == []
