"""Tests for the multi-source literature backends (arXiv / Crossref / Semantic Scholar) and
MultiSourceBackend, with httpx MOCKED -- no network. Real connectivity is opt-in/manual.
"""
from __future__ import annotations

from autoresearch.agents._reuse.deep_research_backend import (
    ArxivBackend,
    CrossrefBackend,
    MultiSourceBackend,
    SemanticScholarBackend,
    _dedupe_papers,
    _sanitize_query,
)


class _Resp:
    def __init__(self, *, text: str = "", data=None, status: int = 200):
        self.text = text
        self._data = data
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._data


_ATOM = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2001.00001v1</id>
    <title>Feature Scaling for k-NN</title>
    <published>2020-01-01T00:00:00Z</published>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/2401.00002v1</id>
    <title>A Recent Preprint</title>
    <published>2024-05-01T00:00:00Z</published>
  </entry>
</feed>"""


def test_arxiv_parses_atom(monkeypatch):
    monkeypatch.setattr("httpx.get", lambda *a, **k: _Resp(text=_ATOM))
    papers = ArxivBackend().search_seminal("k-NN scaling?", {})
    assert len(papers) == 2
    assert papers[0]["title"] == "Feature Scaling for k-NN"
    assert papers[0]["id_or_url"] == "http://arxiv.org/abs/2001.00001v1"
    assert papers[0]["year"] == 2020
    assert papers[0]["note"] == "arXiv"


def test_arxiv_wildcard_stripped_from_query(monkeypatch):
    seen = {}

    def _get(url, params=None, **k):
        seen.update(params or {})
        return _Resp(text=_ATOM)

    monkeypatch.setattr("httpx.get", _get)
    ArxivBackend().search_seminal("k-NN accuracy?", {})
    assert "?" not in seen["search_query"]  # sanitized before hitting the API


def test_crossref_parses_json(monkeypatch):
    canned = {"message": {"items": [
        {"title": ["Standardization and k-NN"], "DOI": "10.1/x",
         "published": {"date-parts": [[2019]]}, "is-referenced-by-count": 42},
        {"title": [], "DOI": "10.1/y"},  # no title -> skipped
    ]}}
    monkeypatch.setattr("httpx.get", lambda *a, **k: _Resp(data=canned))
    papers = CrossrefBackend().search_seminal("q", {})
    assert len(papers) == 1
    assert papers[0]["id_or_url"] == "10.1/x"
    assert papers[0]["year"] == 2019
    assert "42 citations" in papers[0]["note"]


def test_semantic_scholar_rate_limit_fails_open(monkeypatch):
    monkeypatch.setattr("httpx.get", lambda *a, **k: _Resp(data={}, status=429))
    assert SemanticScholarBackend().search_seminal("q", {}) == []


def test_semantic_scholar_parses_json(monkeypatch):
    canned = {"data": [{"title": "S2 paper", "paperId": "abc", "year": 2021, "citationCount": 7}]}
    monkeypatch.setattr("httpx.get", lambda *a, **k: _Resp(data=canned))
    papers = SemanticScholarBackend().search_seminal("q", {})
    assert papers == [{"title": "S2 paper", "id_or_url": "abc", "year": 2021,
                       "note": "7 citations (S2)"}]


class _FakeSource:
    def __init__(self, seminal):
        self._seminal = seminal

    def search_seminal(self, query, spec):
        return list(self._seminal)

    def search_frontier(self, query, spec):
        return []


class _BoomSource:
    def search_seminal(self, query, spec):
        raise RuntimeError("source down")

    def search_frontier(self, query, spec):
        raise RuntimeError("source down")


def test_multisource_merges_and_dedupes():
    a = _FakeSource([{"title": "P1", "id_or_url": "doi:1"},
                     {"title": "P2", "id_or_url": "doi:2"}])
    b = _FakeSource([{"title": "P2 (dup)", "id_or_url": "doi:2"},  # same id -> deduped
                     {"title": "P3", "id_or_url": "doi:3"}])
    papers = MultiSourceBackend([a, b]).search_seminal("q", {})
    assert [p["id_or_url"] for p in papers] == ["doi:1", "doi:2", "doi:3"]


def test_multisource_one_bad_source_does_not_break_others():
    good = _FakeSource([{"title": "P1", "id_or_url": "doi:1"}])
    papers = MultiSourceBackend([_BoomSource(), good]).search_seminal("q", {})
    assert [p["id_or_url"] for p in papers] == ["doi:1"]


def test_multisource_caps_results():
    a = _FakeSource([{"title": f"P{i}", "id_or_url": f"doi:{i}"} for i in range(30)])
    papers = MultiSourceBackend([a], cap=5).search_seminal("q", {})
    assert len(papers) == 5


def test_sanitize_and_dedupe_helpers():
    assert _sanitize_query("k-NN accuracy? *") == "k-NN accuracy"
    deduped = _dedupe_papers([{"title": "A", "id_or_url": "x"},
                              {"title": "A2", "id_or_url": "x"}], cap=10)
    assert len(deduped) == 1
