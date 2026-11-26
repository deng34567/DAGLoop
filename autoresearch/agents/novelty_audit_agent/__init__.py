"""
NoveltyAuditAgent: audits a frozen (possibly reinterpreted) ResearchSpec
against prior literature (LiteratureBrief) and ProblemModelingAgent's
SpecInterpretation, producing a NoveltyAuditReport (AutoResearch state
machine V1.10 §1, AuditingNovelty).
NoveltyAuditAgent：针对冻结（可能经重新解读）的 ResearchSpec，结合此前文献
（LiteratureBrief）和 ProblemModelingAgent 的 SpecInterpretation 做审计，产出
NoveltyAuditReport。

Public surface / 公开接口:
  - ``core.audit_step``: pure citation + novelty audit logic (no transport).
  - ``handler.NoveltyAuditAgentHandler``: EventBus wrapper over the Spec
    Store + LLM.
  - ``service.build_novelty_audit_service`` / ``run_novelty_audit_agent``:
    wiring + process entry point.
"""

from __future__ import annotations

from autoresearch.agents.novelty_audit_agent import core, kernel
from autoresearch.agents.novelty_audit_agent.core import audit_step
from autoresearch.agents.novelty_audit_agent.handler import NoveltyAuditAgentHandler
from autoresearch.agents.novelty_audit_agent.service import (
    build_novelty_audit_service,
    run_novelty_audit_agent,
)

__all__ = [
    "core",
    "kernel",
    "audit_step",
    "NoveltyAuditAgentHandler",
    "build_novelty_audit_service",
    "run_novelty_audit_agent",
]
