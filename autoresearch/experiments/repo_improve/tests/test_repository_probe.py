import subprocess
from types import SimpleNamespace
import urllib.error
import urllib.request

import pytest

from autoresearch.experiments.repo_improve.discover import _public_repo_status


@pytest.mark.parametrize("result,expected", [
    (200, 200), (404, 404), (403, None), (429, None), (None, None),
])
def test_public_repository_probe_is_direct_bounded_and_reports_unknown(monkeypatch, result, expected):
    class Opener:
        def open(self, request, timeout):
            assert request.full_url == "https://api.github.com/repos/example/repo"
            assert timeout == 8
            if result is None:
                raise TimeoutError("temporary network failure")
            if result != 200:
                raise urllib.error.HTTPError(request.full_url, result, "failure", {}, None)
            return self

        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    def build(handler):
        assert isinstance(handler, urllib.request.ProxyHandler)
        assert handler.proxies == {}
        return Opener()

    git_calls = []

    def git_probe(args, **kwargs):
        git_calls.append(args)
        assert args[-4:] == ["ls-remote", "--exit-code", "https://github.com/example/repo.git", "HEAD"]
        assert kwargs["timeout"] == 8
        # An unavailable fallback leaves the repository unverified. Never use
        # the network to determine this unit test's result.
        return SimpleNamespace(returncode=128, stdout="", stderr="offline fixture")

    monkeypatch.setattr(urllib.request, "build_opener", build)
    monkeypatch.setattr(subprocess, "run", git_probe)
    assert _public_repo_status("https://github.com/Example/Repo.git") == expected
    assert len(git_calls) == (1 if result in (403, 429) else 0)
