"""
EvidenceAuditAgent: audits claim-citation support two ways -- Mode A
(literature-support audit, co-audits at AuditingLiteratureNovelty) checks
the reinterpreted ResearchSpec's claims against the LiteratureBrief +
SpecInterpretation; Mode C (citation-regression audit, a Reporting
pre-check) checks the consolidated ResearchMemory for claims lacking
citation support -- reinforcing the honest-report invariant.
EvidenceAuditAgent：以两种方式审计 claim-citation 支持度 —— Mode A
（文献支持度审计，在 AuditingLiteratureNovelty 联合审计）核对重新解读后的
ResearchSpec 的 claim 与 LiteratureBrief + SpecInterpretation；Mode C
（引用回归审计，Reporting 前置检查）核对已沉淀的 ResearchMemory 中缺乏引用
支持的 claim —— 强化诚实报告不变式。

Public surface / 公开接口:
  - ``core.audit_literature_support`` / ``core.audit_citation_regression``:
    pure core logic (no transport).
  - ``kernel``: config loading + audit recording.
  - ``handler.EvidenceAuditAgentHandler``: EventBus wrapper over the Spec
    Store + LLM.
  - ``service.build_evidence_audit_service`` / ``run_evidence_audit_agent``:
    wiring + process entry point.
"""

from __future__ import annotations

from autoresearch.agents.evidence_audit_agent import core, kernel
from autoresearch.agents.evidence_audit_agent.core import (
    audit_citation_regression,
    audit_literature_support,
)
from autoresearch.agents.evidence_audit_agent.handler import EvidenceAuditAgentHandler
from autoresearch.agents.evidence_audit_agent.service import (
    build_evidence_audit_service,
    run_evidence_audit_agent,
)

__all__ = [
    "core",
    "kernel",
    "audit_literature_support",
    "audit_citation_regression",
    "EvidenceAuditAgentHandler",
    "build_evidence_audit_service",
    "run_evidence_audit_agent",
]
