"""
KnowledgeIndexAgent: maintains a per-session knowledge index and answers
retrieval queries against it (AutoResearch state machine V1.10).
KnowledgeIndexAgent：维护每个 session 的知识索引并回答检索查询。

Public surface / 公开接口:
  - ``core.index_entries`` / ``core.retrieve``: pure index + retrieve logic
    (no transport).
  - ``handler.KnowledgeIndexAgentHandler``: EventBus wrapper over the Spec
    Store + retrieval backend.
  - ``service.build_knowledge_index_service`` / ``run_knowledge_index_agent``:
    wiring + process entry point.
"""

from __future__ import annotations

from autoresearch.agents.knowledge_index_agent import core, kernel
from autoresearch.agents.knowledge_index_agent.core import (
    KnowledgeIndexError,
    index_entries,
    retrieve,
)
from autoresearch.agents.knowledge_index_agent.handler import KnowledgeIndexAgentHandler
from autoresearch.agents.knowledge_index_agent.service import (
    build_knowledge_index_service,
    run_knowledge_index_agent,
)

__all__ = [
    "core",
    "kernel",
    "index_entries",
    "retrieve",
    "KnowledgeIndexError",
    "KnowledgeIndexAgentHandler",
    "build_knowledge_index_service",
    "run_knowledge_index_agent",
]
