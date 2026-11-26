"""
KnowledgeIndexBackend: the retrieval-backend adapter interface for
KnowledgeIndexAgent.
KnowledgeIndexAgent 的检索后端适配器接口。

KnowledgeIndexAgent answers knowledge.retrieve_requested by ranking its
indexed IndexEntry dicts (see agents/knowledge_index_agent/data-contracts/
index_entry.schema.yaml) against a query through this adapter.

M3 scope: KeywordBackend is the REAL DEFAULT -- local, deterministic,
lexical token-overlap ranking, NO LLM/embeddings, NO network. SemanticBackend
below is a pluggable, opt-in STUB present so the interface has a shape for a
future real implementation -- it is never run in M3 (its method raises
NotImplementedError).

NO network, NO LLM/embedding imports in this module.
本模块无网络调用、无 LLM/embedding 依赖。
"""

from __future__ import annotations

import re
from typing import Protocol

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> set[str]:
    """Lowercase and split on non-alphanumeric runs. / 小写化并按非字母数字切分。"""
    return set(_TOKEN_RE.findall(text.lower()))


class KnowledgeIndexBackend(Protocol):
    """Retrieval-backend adapter: rank `entries` against `query`."""

    def retrieve(self, query: str, entries: list[dict], top_k: int) -> list[dict]:
        """Return a ranked subset of `entries` matching `query`, at most `top_k`."""
        ...


class KeywordBackend:
    """REAL DEFAULT KnowledgeIndexBackend: local, deterministic lexical
    token-overlap ranking. NO LLM, NO embeddings, NO network.
    真实默认后端：本地确定性词形重叠排序，无 LLM/embedding/网络依赖。

    Scores each entry by the number of distinct query tokens present in the
    entry's tokens (tokenized from `summary` + `tags`). Entries with score 0
    are excluded. Ties are broken by original order (stable sort).
    """

    def retrieve(self, query: str, entries: list[dict], top_k: int) -> list[dict]:
        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        scored: list[tuple[int, int, dict]] = []
        for position, entry in enumerate(entries):
            entry_text = entry.get("summary", "") + " " + " ".join(entry.get("tags", []))
            entry_tokens = _tokenize(entry_text)
            score = len(query_tokens & entry_tokens)
            if score > 0:
                scored.append((score, position, entry))

        # Sort by score desc, stable tie-break by original position asc.
        scored.sort(key=lambda t: (-t[0], t[1]))
        return [entry for _score, _position, entry in scored[:top_k]]


class SemanticBackend:
    """Opt-in, NOT-wired real KnowledgeIndexBackend stub (e.g. a vector/
    embedding-based semantic search). Deferred past M3: every M3 test uses
    KeywordBackend instead. Present only so the adapter interface is
    pluggable.

    DO NOT instantiate/run this in production or tests until a real semantic
    backend is actually wired in a later milestone.
    仅为占位/预留接口，M3 中不会被实际调用；真实实现推迟到后续里程碑。
    """

    def retrieve(self, query: str, entries: list[dict], top_k: int) -> list[dict]:
        raise NotImplementedError("semantic backend deferred; opt-in, not wired in M3")
