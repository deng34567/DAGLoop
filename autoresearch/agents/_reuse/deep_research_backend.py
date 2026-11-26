"""
DeepResearchBackend: the search-backend adapter interface for DeepResearchAgent.
DeepResearchAgent 的搜索后端适配器接口。

DeepResearchAgent's DeepResearching mode (state machine V1.10 §1) runs a
two-pass literature search -- seminal papers (foundational, highly-cited work)
and frontier papers (recent work) -- through this adapter, then synthesizes
the results (via an LLM, outside this module) into a LiteratureBrief +
FieldTrajectoryBrief. Both passes return lists of Paper dicts shaped like the
LiteratureBrief `Paper` data-contract (title, id_or_url, optional year/note),
so the synthesis step can cite them directly.

M2 scope: real backend is DEFERRED. Only FakeBackend (canned, in-memory
results) is exercised by tests. PaperQA2Backend below is a pluggable, opt-in
STUB present so the interface has a shape for a future real implementation --
it is never run in M2 (its methods raise NotImplementedError).

NO network, NO LLM imports in this module.
本模块无网络调用、无 LLM 依赖。
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Protocol

logger = logging.getLogger(__name__)

_TAVILY_CONCURRENCY = max(1, int(os.environ.get("AUTORESEARCH_TAVILY_MAX_CONCURRENCY", "2")))
_TAVILY_GATE = threading.BoundedSemaphore(_TAVILY_CONCURRENCY)
_TAVILY_RATE_LOCK = threading.Lock()
_TAVILY_NEXT_START = 0.0


def _tavily_wait_for_start() -> None:
    """Globally pace Tavily request starts across concurrent research sessions."""
    global _TAVILY_NEXT_START
    interval = max(0.0, float(os.environ.get("AUTORESEARCH_TAVILY_MIN_INTERVAL_S", "0.5")))
    with _TAVILY_RATE_LOCK:
        now = time.monotonic()
        delay = max(0.0, _TAVILY_NEXT_START - now)
        if delay:
            time.sleep(delay)
            now = time.monotonic()
        _TAVILY_NEXT_START = now + interval


class DeepResearchBackend(Protocol):
    """Search-backend adapter: two passes over a research query.

    `spec` is the (dict-shaped) ResearchSpec, passed through in case a backend
    wants to use budget/task_mode/etc. to shape its search; implementations
    are free to ignore fields they don't need.
    """

    def search_seminal(self, query: str, spec: dict) -> list[dict]:
        """Return seminal (foundational) papers for `query` as Paper dicts."""
        ...

    def search_frontier(self, query: str, spec: dict) -> list[dict]:
        """Return frontier (recent) papers for `query` as Paper dicts."""
        ...


_DEFAULT_SEMINAL: list[dict] = [
    {
        "title": "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
        "id_or_url": "arxiv:2005.11401",
        "year": 2020,
        "note": "Seminal RAG paper.",
    },
]

_DEFAULT_FRONTIER: list[dict] = [
    {
        "title": "Long-Tail Scientific QA via Retrieval-Augmented Fine-Tuning",
        "id_or_url": "arxiv:2401.00000",
        "year": 2024,
        "note": "Recent frontier work on long-tail scientific QA.",
    },
]


class FakeBackend:
    """In-memory DeepResearchBackend for tests: returns canned Paper lists and
    records how many times each search pass was called.
    测试用内存后端：返回预设的 Paper 列表，并记录每个搜索通道的调用次数。
    """

    def __init__(
        self,
        *,
        seminal: list[dict] | None = None,
        frontier: list[dict] | None = None,
    ) -> None:
        self._seminal = seminal if seminal is not None else list(_DEFAULT_SEMINAL)
        self._frontier = frontier if frontier is not None else list(_DEFAULT_FRONTIER)
        self.seminal_calls = 0
        self.frontier_calls = 0

    def search_seminal(self, query: str, spec: dict) -> list[dict]:
        self.seminal_calls += 1
        return self._seminal

    def search_frontier(self, query: str, spec: dict) -> list[dict]:
        self.frontier_calls += 1
        return self._frontier


class PaperQA2Backend:
    """Opt-in, NOT-wired real DeepResearchBackend stub (e.g. PaperQA2 /
    GPT-Researcher / arXiv API). Deferred past M2: every M2 test injects
    FakeBackend instead. Present only so the adapter interface is pluggable.

    DO NOT instantiate/run this in production or tests until a real backend
    is actually wired in a later milestone.
    仅为占位/预留接口，M2 中不会被实际调用；真实实现推迟到后续里程碑。
    """

    def search_seminal(self, query: str, spec: dict) -> list[dict]:
        raise NotImplementedError("real backend deferred; opt-in, not wired in M2")

    def search_frontier(self, query: str, spec: dict) -> list[dict]:
        raise NotImplementedError("real backend deferred; opt-in, not wired in M2")


class OpenAlexBackend:
    """REAL, OPT-IN DeepResearchBackend: queries the free OpenAlex API (no key,
    https://api.openalex.org) for REAL papers -- seminal = top-cited for the
    query, frontier = most recent -- and maps them to LiteratureBrief Paper
    dicts {title, id_or_url, year?, note?} that the synthesis step cites
    (grounding-checked in the core). Best-effort: on any API/network error a
    pass returns [] (never crashes the flow). SERIAL: one HTTP GET per pass, no
    concurrency. Never the default -- callers opt in explicitly.
    """

    _URL = "https://api.openalex.org/works"

    def __init__(self, *, per_page: int = 8, timeout: float = 20.0,
                 mailto: str | None = None, from_date: str = "2023-01-01") -> None:
        self.per_page = int(per_page)
        self.timeout = float(timeout)
        self.mailto = mailto
        self.from_date = from_date

    def search_seminal(self, query: str, spec: dict) -> list[dict]:
        # RELEVANCE sort (OpenAlex default with ?search=), not cited_by_count: sorting
        # by citations discards relevance and surfaces the most-cited works loosely
        # matching ANY term (tangential ML/optimization reviews) instead of on-topic
        # papers. Relevance keeps the results actually about the query.
        return self._query(query, sort=None, recent=False)

    def search_frontier(self, query: str, spec: dict) -> list[dict]:
        # recent AND relevant (relevance sort + a from_date filter), not just the
        # newest loosely-matching works.
        return self._query(query, sort=None, recent=True)

    def _query(self, query: str, *, sort: str | None, recent: bool) -> list[dict]:
        import re
        import time

        import httpx

        # OpenAlex reads ? and * as WILDCARD operators, so a raw natural-language
        # question ("...heterogeneously-scaled features?") 400s. Strip them first.
        search = re.sub(r"\s+", " ", re.sub(r"[?*]+", " ", query or "")).strip()[:250]
        # omit ?sort so OpenAlex defaults to relevance_score:desc for the ?search match
        params: dict = {"search": search, "per_page": self.per_page}
        if sort:
            params["sort"] = sort
        if recent and self.from_date:
            params["filter"] = f"from_publication_date:{self.from_date}"
        if self.mailto:
            params["mailto"] = self.mailto
        # Retry transient failures (429 rate-limit / 5xx / network) with a short
        # exponential backoff -- as DeepMind's science-skills OpenAlex client does --
        # then fail-open to [] (never crash the flow).
        results: list = []
        for _attempt in range(3):
            try:
                r = httpx.get(self._URL, params=params, timeout=self.timeout, trust_env=False)
                r.raise_for_status()
                results = r.json().get("results", []) or []
                break
            except Exception:
                if _attempt == 2:
                    return []
                time.sleep(0.3 * (2 ** _attempt))
        out: list[dict] = []
        for w in results:
            title = w.get("title") or w.get("display_name")
            if not title:
                continue
            paper: dict = {
                "title": str(title)[:300],
                "id_or_url": w.get("doi") or w.get("id") or str(title),
            }
            if w.get("publication_year"):
                paper["year"] = w["publication_year"]
            if w.get("cited_by_count") is not None:
                paper["note"] = f"{w['cited_by_count']} citations (OpenAlex)"
            out.append(paper)
        return out


def _sanitize_query(query: str) -> str:
    """Strip OpenAlex/index wildcard operators (?/*), collapse whitespace, cap length."""
    import re

    return re.sub(r"\s+", " ", re.sub(r"[?*]+", " ", query or "")).strip()[:250]


def _dedupe_papers(papers: list[dict], cap: int) -> list[dict]:
    """Merge a list of Paper dicts, de-duplicating by id_or_url (falling back to title),
    preserving first-seen order, capped at ``cap``."""
    seen: set = set()
    out: list[dict] = []
    for p in papers:
        key = str(p.get("id_or_url") or p.get("title") or "").strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(p)
        if len(out) >= cap:
            break
    return out


class ArxivBackend:
    """REAL, OPT-IN DeepResearchBackend over the free arXiv Atom API (no key). Strong for
    CS/ML preprints. seminal = relevance-ranked, frontier = most recent. Best-effort: any
    error -> []."""

    _URL = "https://export.arxiv.org/api/query"

    def __init__(self, *, per_page: int = 6, timeout: float = 20.0) -> None:
        self.per_page = int(per_page)
        self.timeout = float(timeout)

    def search_seminal(self, query: str, spec: dict) -> list[dict]:
        return self._query(query, sort="relevance")

    def search_frontier(self, query: str, spec: dict) -> list[dict]:
        return self._query(query, sort="submittedDate")

    def _query(self, query: str, *, sort: str) -> list[dict]:
        import xml.etree.ElementTree as ET

        import httpx

        q = _sanitize_query(query)
        if not q:
            return []
        params = {"search_query": f"all:{q}", "start": 0, "max_results": self.per_page,
                  "sortBy": sort, "sortOrder": "descending"}
        try:
            r = httpx.get(self._URL, params=params, timeout=self.timeout, trust_env=False,
                          follow_redirects=True)
            r.raise_for_status()
            root = ET.fromstring(r.text)
        except Exception:
            return []
        ns = {"a": "http://www.w3.org/2005/Atom"}
        out: list[dict] = []
        for e in root.findall("a:entry", ns):
            title = " ".join((e.findtext("a:title", default="", namespaces=ns) or "").split())
            if not title:
                continue
            idu = (e.findtext("a:id", default="", namespaces=ns) or "").strip()
            pub = (e.findtext("a:published", default="", namespaces=ns) or "")[:4]
            paper: dict = {"title": title[:300], "id_or_url": idu or title, "note": "arXiv"}
            if pub.isdigit():
                paper["year"] = int(pub)
            out.append(paper)
        return out


class CrossrefBackend:
    """REAL, OPT-IN DeepResearchBackend over the free Crossref works API (no key). Broad
    cross-discipline coverage. seminal = relevance, frontier = most recent. Best-effort."""

    _URL = "https://api.crossref.org/works"

    def __init__(self, *, per_page: int = 6, timeout: float = 20.0, mailto: str | None = None) -> None:
        self.per_page = int(per_page)
        self.timeout = float(timeout)
        self.mailto = mailto

    def search_seminal(self, query: str, spec: dict) -> list[dict]:
        return self._query(query, recent=False)

    def search_frontier(self, query: str, spec: dict) -> list[dict]:
        return self._query(query, recent=True)

    def _query(self, query: str, *, recent: bool) -> list[dict]:
        import httpx

        q = _sanitize_query(query)
        if not q:
            return []
        params: dict = {"query": q, "rows": self.per_page}
        if recent:
            params["sort"] = "published"
            params["order"] = "desc"
        if self.mailto:
            params["mailto"] = self.mailto
        try:
            r = httpx.get(self._URL, params=params, timeout=self.timeout, trust_env=False)
            r.raise_for_status()
            items = (r.json().get("message") or {}).get("items", []) or []
        except Exception:
            return []
        out: list[dict] = []
        for it in items:
            titles = it.get("title") or []
            title = (titles[0] if titles else "").strip()
            if not title:
                continue
            paper: dict = {"title": title[:300], "id_or_url": it.get("DOI") or it.get("URL") or title}
            parts = (((it.get("published") or {}).get("date-parts") or [[None]])[0] or [None])
            if parts and isinstance(parts[0], int):
                paper["year"] = parts[0]
            cc = it.get("is-referenced-by-count")
            paper["note"] = f"{cc} citations (Crossref)" if cc is not None else "Crossref"
            out.append(paper)
        return out


class SemanticScholarBackend:
    """REAL, OPT-IN DeepResearchBackend over the Semantic Scholar Graph API. Strong CS/ML
    relevance, but the keyless pool rate-limits hard (429): BEST-EFFORT with NO retry, so a
    429 fails open to [] immediately (no added latency). Pass an ``api_key`` for reliability."""

    _URL = "https://api.semanticscholar.org/graph/v1/paper/search"

    def __init__(self, *, per_page: int = 6, timeout: float = 15.0, api_key: str | None = None) -> None:
        self.per_page = int(per_page)
        self.timeout = float(timeout)
        self.api_key = api_key

    def search_seminal(self, query: str, spec: dict) -> list[dict]:
        return self._query(query)

    def search_frontier(self, query: str, spec: dict) -> list[dict]:
        return self._query(query)

    def _query(self, query: str) -> list[dict]:
        import httpx

        q = _sanitize_query(query)
        if not q:
            return []
        headers = {"x-api-key": self.api_key} if self.api_key else {}
        try:
            r = httpx.get(self._URL, params={"query": q, "limit": self.per_page,
                                             "fields": "title,year,citationCount"},
                          headers=headers, timeout=self.timeout, trust_env=False)
            if r.status_code != 200:
                return []
            data = r.json().get("data", []) or []
        except Exception:
            return []
        out: list[dict] = []
        for w in data:
            title = (w.get("title") or "").strip()
            if not title:
                continue
            paper: dict = {"title": title[:300], "id_or_url": w.get("paperId") or title}
            if w.get("year"):
                paper["year"] = w["year"]
            if w.get("citationCount") is not None:
                paper["note"] = f"{w['citationCount']} citations (S2)"
            out.append(paper)
        return out


class TavilyBackend:
    """Key-backed web search plus bounded full-text extraction for literature grounding.

    Requests use ``httpx.Client(trust_env=False)`` so inherited proxy variables are never used.
    The API key is injected by the caller and is never stored in returned papers or logs.
    """

    _BASE_URL = "https://api.tavily.com"

    def __init__(self, *, api_key: str, per_page: int = 6, timeout: float = 30.0,
                 fetch_cap: int = 12, fetch_chars: int = 12_000) -> None:
        self.api_key = str(api_key or "").strip()
        self.per_page = max(1, int(per_page))
        self.timeout = float(timeout)
        self.fetch_cap = max(0, int(fetch_cap))
        self.fetch_chars = max(1000, int(fetch_chars))

    def search_seminal(self, query: str, spec: dict) -> list[dict]:
        return self._query(f"{query} foundational highly cited research paper")

    def search_frontier(self, query: str, spec: dict) -> list[dict]:
        return self._query(f"{query} latest research paper preprint")

    def _post(self, client, endpoint: str, payload: dict):
        """POST with bounded global rate and transient-only retry."""
        import httpx

        retries = max(0, int(os.environ.get("AUTORESEARCH_TAVILY_RETRIES", "3")))
        base = max(0.0, float(os.environ.get("AUTORESEARCH_TAVILY_RETRY_BASE_S", "1")))
        last = None
        for attempt in range(retries + 1):
            try:
                with _TAVILY_GATE:
                    _tavily_wait_for_start()
                    response = client.post(
                        self._BASE_URL + endpoint,
                        headers={"Authorization": f"Bearer {self.api_key}"},
                        json=payload,
                    )
                response.raise_for_status()
                return response
            except Exception as exc:
                last = exc
                status = exc.response.status_code if isinstance(exc, httpx.HTTPStatusError) else None
                transient = status is None or status == 429 or status >= 500
                if not transient or attempt >= retries:
                    raise
                retry_after = None
                if isinstance(exc, httpx.HTTPStatusError):
                    retry_after = exc.response.headers.get("Retry-After")
                try:
                    delay = float(retry_after) if retry_after is not None else base * (2 ** attempt)
                except ValueError:
                    delay = base * (2 ** attempt)
                time.sleep(min(30.0, max(0.0, delay)))
        raise last  # pragma: no cover - loop always returns or raises

    def _query(self, query: str) -> list[dict]:
        if not self.api_key:
            return []
        import httpx

        try:
            with httpx.Client(trust_env=False, timeout=self.timeout) as client:
                response = self._post(client, "/search", {
                        "query": _sanitize_query(query),
                        "search_depth": "advanced",
                        "max_results": self.per_page,
                        "include_answer": False,
                        "include_raw_content": "text",
                    })
                results = response.json().get("results", []) or []
        except Exception as exc:
            # Do not log the exception/body: providers can echo keys or queries.
            status = exc.response.status_code if isinstance(exc, httpx.HTTPStatusError) else None
            logger.warning("Tavily /search failed: http_status=%s error_type=%s; "
                           "returning no results for fallback", status, type(exc).__name__)
            return []
        papers: list[dict] = []
        for item in results:
            title = str(item.get("title") or "").strip()
            url = str(item.get("url") or "").strip()
            if not title or not url:
                continue
            content = str(item.get("raw_content") or item.get("content") or "").strip()
            paper = {
                "title": title[:300],
                "id_or_url": url,
                "note": "Tavily search result",
            }
            if content:
                paper["fetched_excerpt"] = content[:self.fetch_chars]
            papers.append(paper)
        return papers

    def fetch(self, papers: list[dict]) -> list[dict]:
        urls = [
            str(paper.get("id_or_url") or "").strip()
            for paper in papers[:self.fetch_cap]
            if str(paper.get("id_or_url") or "").startswith(("http://", "https://"))
        ]
        if not self.api_key or not urls:
            return []
        import httpx

        try:
            with httpx.Client(trust_env=False, timeout=max(self.timeout, 60.0)) as client:
                response = self._post(client, "/extract", {
                    "urls": urls, "extract_depth": "advanced", "format": "text"})
                results = response.json().get("results", []) or []
        except Exception as exc:
            status = exc.response.status_code if isinstance(exc, httpx.HTTPStatusError) else None
            logger.warning("Tavily /extract failed: http_status=%s error_type=%s; "
                           "returning no results for fallback", status, type(exc).__name__)
            return []
        by_url = {str(paper.get("id_or_url")): paper for paper in papers}
        enriched: list[dict] = []
        for item in results:
            url = str(item.get("url") or "").strip()
            original = by_url.get(url)
            if original is None:
                continue
            content = str(item.get("raw_content") or "").strip()
            enriched.append({
                **original,
                **({"fetched_excerpt": content[:self.fetch_chars]} if content else {}),
            })
        return enriched


class FailoverBackend:
    """Use a primary backend, falling back per search pass only when it returns no papers."""

    def __init__(self, primary, fallback) -> None:
        self.primary = primary
        self.fallback = fallback

    def search_seminal(self, query: str, spec: dict) -> list[dict]:
        papers = self._call(self.primary, "search_seminal", query, spec)
        return papers or self._call(self.fallback, "search_seminal", query, spec)

    def search_frontier(self, query: str, spec: dict) -> list[dict]:
        papers = self._call(self.primary, "search_frontier", query, spec)
        return papers or self._call(self.fallback, "search_frontier", query, spec)

    def fetch(self, papers: list[dict]) -> list[dict]:
        fetched = self._call(self.primary, "fetch", papers)
        if fetched:
            return fetched
        return self._call(self.fallback, "fetch", papers) or list(papers)

    @staticmethod
    def _call(backend, method: str, *args) -> list[dict]:
        fn = getattr(backend, method, None)
        if not callable(fn):
            return []
        try:
            return list(fn(*args) or [])
        except Exception:  # noqa: BLE001 -- failover is the point of this adapter
            return []


class MultiSourceBackend:
    """Compose several DeepResearchBackends into one: each search pass queries EVERY source
    (serially -- no concurrency), then merges + de-duplicates the papers (by id_or_url,
    falling back to title), capped at ``cap``. Each source is best-effort (its own fail-open
    []), so one down/rate-limited source never blocks the others."""

    def __init__(self, backends: list, *, cap: int = 20) -> None:
        self.backends = list(backends)
        self.cap = int(cap)

    def search_seminal(self, query: str, spec: dict) -> list[dict]:
        return self._merge("search_seminal", query, spec)

    def search_frontier(self, query: str, spec: dict) -> list[dict]:
        return self._merge("search_frontier", query, spec)

    def _merge(self, method: str, query: str, spec: dict) -> list[dict]:
        papers: list[dict] = []
        for b in self.backends:
            try:
                papers.extend(getattr(b, method)(query, spec) or [])
            except Exception:  # noqa: BLE001 -- one bad source must not break the merge
                continue
        return _dedupe_papers(papers, self.cap)
