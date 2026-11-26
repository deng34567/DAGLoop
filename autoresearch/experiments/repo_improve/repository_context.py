"""Read permitted task facts from an explicit checkout or existing mirror before design.

No network/downloads, no evaluation, and no baseline claims are made here.
"""
from __future__ import annotations

import os
from pathlib import Path
import subprocess


def baseline_repository_context(baseline: dict | None, question: str) -> dict | None:
    """Supply commit-pinned documents and named sources without rewriting a baseline."""
    contract = (baseline or {}).get("repository_contract")
    if not contract or not baseline.get("repo_commit"):
        return baseline
    from .agentic_improve import _contract_with_packaging_evidence, _repository_task_contract
    roots = set()
    for artifact in baseline.get("artifacts") or []:
        if not isinstance(artifact, str):
            continue
        parent = Path(artifact).parent
        root = next((p for p in (parent, *parent.parents) if (p / ".git").exists()), None)
        if root is None or root in roots:
            continue
        roots.add(root)
        captured = _repository_task_contract(
            str(root), question, contract.get("execution_contract"))
        recovered = _contract_with_packaging_evidence(str(root), baseline, captured)
        if (recovered or {}).get("sources"):
            return {**baseline, "repository_contract": recovered}
    return baseline


def early_repository_context(request: dict) -> dict:
    from .agentic_improve import _extract_repo_urls, _repository_task_contract
    question = str(request.get("question") or "")
    url = str(request.get("repo_url") or "")
    if not url:
        urls = _extract_repo_urls(question)
        url = urls[0] if len(urls) == 1 else ""
    if not url:
        return {"status": "unavailable", "reason": "no unambiguous specified repository"}
    root = Path(url.removeprefix("file://"))
    if not root.is_dir():
        name = url.rstrip("/").rsplit("/", 1)[-1].removesuffix(".git")
        mirror_root = os.environ.get("AUTORESEARCH_CLONE_MIRROR_DIR", "").strip()
        if not mirror_root:
            return {"status": "unavailable", "repo_url": url,
                    "reason": "no local checkout/mirror; inspect repository before expensive implementation"}
        root = Path(mirror_root) / name
        if not root.is_dir():
            return {"status": "unavailable", "repo_url": url,
                    "reason": "no local checkout/mirror; inspect repository before expensive implementation"}
        # A basename collision must not bind facts from another repository.
        origin = subprocess.run(["git", "-C", str(root), "remote", "get-url", "origin"],
                                capture_output=True, text=True, timeout=10)
        if origin.returncode or origin.stdout.strip().rstrip("/").removesuffix(".git") != url.rstrip("/").removesuffix(".git"):
            return {"status": "unavailable", "repo_url": url, "reason": "mirror origin mismatch"}
    contract = _repository_task_contract(str(root), question, request.get("execution_contract"))
    commit = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                            capture_output=True, text=True, timeout=10)
    return {"status": "available" if contract else "unavailable", "repo_url": url,
            "repo_commit": commit.stdout.strip() if commit.returncode == 0 else None,
            "repository_contract": contract,
            "note": "Repository facts only, not a measured baseline; revalidate if checkout changes."}
