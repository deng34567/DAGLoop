from __future__ import annotations

import logging

import httpx
import pytest

from autoresearch.agents._reuse.deep_research_backend import FailoverBackend, TavilyBackend


class _Response:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


def test_tavily_backend_search_and_fetch_are_direct_and_bounded(monkeypatch):
    calls = []

    class Client:
        def __init__(self, **kwargs):
            calls.append(("client", kwargs))

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def post(self, url, **kwargs):
            calls.append((url, kwargs))
            if url.endswith("/search"):
                return _Response({"results": [{
                    "title": "Paper", "url": "https://example.org/paper",
                    "raw_content": "search excerpt",
                }]})
            return _Response({"results": [{
                "url": "https://example.org/paper", "raw_content": "fetched full text",
            }]})

    monkeypatch.setattr("httpx.Client", Client)
    backend = TavilyBackend(api_key="secret", fetch_cap=1, fetch_chars=1000)

    papers = backend.search_seminal("graph learning", {})
    fetched = backend.fetch(papers)

    assert papers[0]["id_or_url"] == "https://example.org/paper"
    assert fetched[0]["fetched_excerpt"] == "fetched full text"
    assert all(kwargs["trust_env"] is False for kind, kwargs in calls if kind == "client")
    assert all("secret" not in repr(item) for item in papers + fetched)


def test_tavily_failure_uses_academic_fallback_per_search_pass():
    class EmptyPrimary:
        def search_seminal(self, query, spec):
            return []

        def search_frontier(self, query, spec):
            raise RuntimeError("service unavailable")

    class AcademicFallback:
        def search_seminal(self, query, spec):
            return [{"title": "Seminal", "id_or_url": "doi:seminal"}]

        def search_frontier(self, query, spec):
            return [{"title": "Frontier", "id_or_url": "arxiv:frontier"}]

    backend = FailoverBackend(EmptyPrimary(), AcademicFallback())

    assert backend.search_seminal("q", {})[0]["id_or_url"] == "doi:seminal"
    assert backend.search_frontier("q", {})[0]["id_or_url"] == "arxiv:frontier"


@pytest.mark.parametrize('endpoint', ['/search', '/extract'])
@pytest.mark.parametrize('status', [401, 403, 429, 432, 500, None])
def test_tavily_failure_logs_status_without_secrets(monkeypatch, caplog, endpoint, status):
    monkeypatch.setenv('AUTORESEARCH_TAVILY_RETRIES', '0')
    monkeypatch.setenv('AUTORESEARCH_TAVILY_MIN_INTERVAL_S', '0')
    secret = 'private-test-key'
    query = 'private-research-query'
    request = httpx.Request('POST', 'https://api.tavily.com' + endpoint,
                           headers={'Authorization': 'Bearer ' + secret})
    error = (httpx.HTTPStatusError(secret + query, request=request,
             response=httpx.Response(status, request=request, text=secret + query))
             if status else httpx.ReadTimeout(secret + query, request=request))

    class Client:
        def __init__(self, **kwargs):
            assert kwargs['trust_env'] is False
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def post(self, *args, **kwargs):
            raise error

    monkeypatch.setattr(httpx, 'Client', Client)
    backend = TavilyBackend(api_key=secret)
    with caplog.at_level(logging.WARNING):
        result = (backend.search_frontier(query, {}) if endpoint == '/search'
                  else backend.fetch([{'id_or_url': 'https://example.org/' + query}]))
    assert result == []
    assert endpoint in caplog.text
    assert f'http_status={status}' in caplog.text
    assert type(error).__name__ in caplog.text
    assert secret not in caplog.text
    assert query not in caplog.text


def test_tavily_retries_429_without_logging_secrets(monkeypatch):
    monkeypatch.setenv('AUTORESEARCH_TAVILY_RETRIES', '1')
    monkeypatch.setenv('AUTORESEARCH_TAVILY_RETRY_BASE_S', '0')
    monkeypatch.setenv('AUTORESEARCH_TAVILY_MIN_INTERVAL_S', '0')
    request = httpx.Request('POST', 'https://api.tavily.com/search')
    calls = 0

    class Client:
        def __init__(self, **kwargs):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def post(self, *args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                response = httpx.Response(429, request=request, headers={'Retry-After': '0'})
                raise httpx.HTTPStatusError('rate limited', request=request, response=response)
            return _Response({'results': []})

    monkeypatch.setattr(httpx, 'Client', Client)
    assert TavilyBackend(api_key='secret').search_frontier('query', {}) == []
    assert calls == 2
