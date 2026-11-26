"""
KnowledgeIndexAgent core: the pure index + retrieve logic.
KnowledgeIndexAgent 核心：与传输层无关的纯"索引 + 检索"逻辑。

``index_entries`` normalizes IndexEntryInput dicts (``{path_or_ref, summary,
tags?}``) into full IndexEntry dicts (see agents/knowledge_index_agent/
data-contracts/index_entry.schema.yaml), validates each against the schema,
and merges them into an existing index (deduped by ``path_or_ref``).
``retrieve`` delegates ranking to an injected KnowledgeIndexBackend (e.g. the
real ``KeywordBackend`` in ``agents/_reuse/knowledge_index_backend.py``).

NO EventBus / Redis / Store / network / LLM access here. The handler (a
later milestone) is responsible for resolving refs, persisting the index,
and publishing events.
本模块无 EventBus/Redis/Store/网络/LLM 依赖。handler（后续里程碑）负责解析
ref、持久化索引、发布事件。

Tag derivation rule / 标签推导规则
----------------------------------
When an IndexEntryInput has no (or an empty) ``tags`` list, tags are derived
lexically from ``summary``: lowercase, split on non-alphanumeric runs, drop
tokens shorter than 3 characters or in a small English stopword list, dedup
while preserving first-occurrence order. Deterministic, NO LLM.
当 IndexEntryInput 未提供（或提供空）``tags`` 时，从 ``summary`` 词形推导：
小写化、按非字母数字切分、丢弃长度小于 3 或属于停用词表的 token，按首次出现
顺序去重。确定性，不含 LLM。
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Protocol

import yaml
from jsonschema import Draft202012Validator

# ---------------------------------------------------------------------------
# Exceptions / 异常
# ---------------------------------------------------------------------------


class KnowledgeIndexError(Exception):
    """Base for KnowledgeIndexAgent core failures. / 核心失败基类。

    Raised when an IndexEntryInput is missing a required field
    (``path_or_ref``/``summary``) or the built IndexEntry fails schema
    validation.
    """

    #: structured error code for agent.failed.underlying_reason_code
    code: str = "knowledge_index_failed"
    #: transient | permanent
    classification: str = "permanent"


# ---------------------------------------------------------------------------
# Retrieval backend protocol / 检索后端协议
# ---------------------------------------------------------------------------


class KnowledgeIndexBackendLike(Protocol):
    """Subset of autoresearch.agents._reuse.knowledge_index_backend.KnowledgeIndexBackend."""

    def retrieve(self, query: str, entries: list[dict], top_k: int) -> list[dict]: ...


# ---------------------------------------------------------------------------
# Static resource: IndexEntry schema validator
# 静态资源：IndexEntry schema 校验器
# ---------------------------------------------------------------------------

_CONTRACTS_DIR = Path(__file__).resolve().parent / "data-contracts"


@lru_cache(maxsize=1)
def _index_entry_validator() -> Draft202012Validator:
    schema = yaml.safe_load(
        (_CONTRACTS_DIR / "index_entry.schema.yaml").read_text(encoding="utf-8")
    )
    return Draft202012Validator(schema)


def _validate_index_entry(entry: dict) -> list[str]:
    return [e.message for e in _index_entry_validator().iter_errors(entry)]


# ---------------------------------------------------------------------------
# Tag derivation / 标签推导
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_MIN_TAG_LEN = 3
_STOPWORDS = {
    "the", "and", "for", "with", "from", "that", "this", "are", "was",
    "were", "has", "have", "into", "over", "under", "about", "without",
    "not", "but", "you", "your", "all", "can", "its", "of", "in", "on",
    "to", "is", "as", "by", "or", "be", "at", "we", "our", "their",
    "due", "during",
}


def _derive_tags(summary: str) -> list[str]:
    """Lexically derive tags from ``summary`` (see module docstring for the
    rule). Deterministic, no LLM. / 从 ``summary`` 词形推导标签（规则见模块
    docstring）。确定性，不含 LLM。"""
    tags: list[str] = []
    for token in _TOKEN_RE.findall(summary.lower()):
        if len(token) < _MIN_TAG_LEN or token in _STOPWORDS:
            continue
        if token not in tags:
            tags.append(token)
    return tags


# ---------------------------------------------------------------------------
# index_entries / 索引
# ---------------------------------------------------------------------------


def index_entries(
    existing: list[dict],
    new_items: list[dict],
    *,
    produced_by_state: str,
    source: str,
    entry_type: str = "paper",
) -> list[dict]:
    """Normalize ``new_items`` (IndexEntryInput dicts) into full, schema-valid
    IndexEntry dicts and merge them into ``existing``, deduped by
    ``path_or_ref`` (a new item with the same ``path_or_ref`` OVERWRITES the
    old entry in place; order is otherwise preserved). Returns the merged
    list (``existing`` itself is not mutated).
    把 ``new_items``（IndexEntryInput）规范化为完整、schema 合法的 IndexEntry
    并合并进 ``existing``，按 ``path_or_ref`` 去重（同 ``path_or_ref``
    的新条目原位覆盖旧条目，其余顺序不变）。返回合并后的列表（不修改
    ``existing`` 本身）。

    Raises:
        KnowledgeIndexError: an input item is missing ``path_or_ref`` or
            ``summary``, or the built entry fails schema validation.
    """
    merged: list[dict] = list(existing)
    position_by_ref = {e.get("path_or_ref"): i for i, e in enumerate(merged)}

    for item in new_items:
        path_or_ref = item.get("path_or_ref")
        if not path_or_ref or not isinstance(path_or_ref, str):
            raise KnowledgeIndexError(
                f"IndexEntryInput missing required field 'path_or_ref': {item!r}"
            )

        summary = item.get("summary")
        if not summary or not isinstance(summary, str):
            raise KnowledgeIndexError(
                f"IndexEntryInput missing required field 'summary': {item!r}"
            )

        tags = item.get("tags") or _derive_tags(summary)
        entry = {
            "path_or_ref": path_or_ref,
            "type": entry_type,
            "summary": summary,
            "tags": list(tags),
            "source": source,
            "produced_by_state": produced_by_state,
        }

        errors = _validate_index_entry(entry)
        if errors:
            raise KnowledgeIndexError(
                f"built IndexEntry failed schema validation: {errors}"
            )

        if path_or_ref in position_by_ref:
            merged[position_by_ref[path_or_ref]] = entry
        else:
            position_by_ref[path_or_ref] = len(merged)
            merged.append(entry)

    return merged


# ---------------------------------------------------------------------------
# retrieve / 检索
# ---------------------------------------------------------------------------


def retrieve(
    query: str,
    entries: list[dict],
    backend: KnowledgeIndexBackendLike,
    top_k: int = 5,
) -> list[dict]:
    """Delegate ranking of ``entries`` against ``query`` to ``backend``.
    把 ``entries`` 针对 ``query`` 的排序委派给 ``backend``。"""
    return backend.retrieve(query, entries, top_k)


__all__ = [
    "KnowledgeIndexError",
    "KnowledgeIndexBackendLike",
    "index_entries",
    "retrieve",
]
