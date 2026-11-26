"""
EvidenceAuditAgent EventBus handler: the thin transport wrapper around core.
EvidenceAuditAgent EventBus handler：core 之上的薄传输层。

Subscribes TWO events:

1. ``evidence.audit_requested`` (payload ``{research_spec_ref,
   literature_brief_ref, spec_interpretation_ref}``) -- reads the
   ResearchSpec, LiteratureBrief, and SpecInterpretation from the Spec Store
   by ref (provenance -- NEVER from the event payload), runs
   ``core.audit_literature_support`` (Mode A), persists the resulting
   EvidenceAuditReport (kind="evidence_audit_report", key="v1"), and
   publishes ``evidence.audited`` with its ref.
2. ``evidence.citation_audit_requested`` (payload ``{research_memory_ref,
   research_spec_ref}``) -- reads the ResearchMemory + ResearchSpec from the
   Spec Store by ref, runs ``core.audit_citation_regression`` (Mode C),
   persists the resulting CitationAuditReport (kind="citation_audit_report",
   key="v1"), and publishes ``evidence.citation_audited`` with its ref.

Any core failure (or unexpected error) is converted into ``agent.failed``,
never a crash.

订阅两个事件：evidence.audit_requested（按 ref 从 Spec Store 读回
ResearchSpec + LiteratureBrief + SpecInterpretation，跑一次
core.audit_literature_support（Mode A），持久化 EvidenceAuditReport，发布
evidence.audited）、evidence.citation_audit_requested（按 ref 读回
ResearchMemory + ResearchSpec，跑一次 core.audit_citation_regression
（Mode C），持久化 CitationAuditReport，发布 evidence.citation_audited）。
core 的任何失败（或未预期错误）都转成 agent.failed，绝不崩溃。
"""

from __future__ import annotations

from autoresearch.agents.moss_agent.patch_writer import scoped_prompt_patch

import asyncio
import logging
from typing import Any

from autoresearch.agents.evidence_audit_agent import core
from autoresearch.agents.evidence_audit_agent.kernel import AGENT_NAME, ConfigLoader
from autoresearch.spec_store.identity import artifact_digest

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1

_EVIDENCE_AUDIT_REPORT_KEY = "v1"
_CITATION_AUDIT_REPORT_KEY = "v1"

_NO_RESEARCH_MEMORY_NOTE = (
    "No ResearchMemory available (research did not consolidate findings); "
    "citation audit skipped."
)

_CITATION_AUDIT_PARSE_FALLBACK_NOTE = (
    "Citation audit was unavailable because the model did not return a parseable "
    "schema-valid report after bounded retries. No claim was classified by this "
    "audit; the final report must rely on the existing deterministic ResearchMemory "
    "grounding checks and disclose this limitation."
)


class EvidenceAuditAgentHandler:
    """Wires core.audit_literature_support / core.audit_citation_regression
    to a Spec Store + LLM + EventBus client.
    把 core.audit_literature_support / core.audit_citation_regression 接到
    Spec Store + LLM + EventBus 客户端。"""

    def __init__(
        self,
        *,
        spec_store: Any,
        llm_client: Any,
        config_root: Any | None = None,
    ) -> None:
        self.spec_store = spec_store
        self.llm_client = llm_client
        self.config = ConfigLoader(config_root)

    # ---- EventBus registration / 注册 ----

    def register(self, client: Any) -> None:
        """Subscribe this handler's two events on an EventBusClient.
        在 EventBusClient 上订阅本 handler 的两个事件。"""
        client.subscribe(
            "evidence.audit_requested",
            self._make_eventbus_handler(client, self.handle_audit_requested),
        )
        client.subscribe(
            "evidence.citation_audit_requested",
            self._make_eventbus_handler(client, self.handle_citation_audit_requested),
        )

    def _make_eventbus_handler(self, client: Any, fn: Any):
        async def _handler(env: Any) -> None:
            await fn(client, env)

        return _handler

    # ---- evidence.audit_requested (Mode A) ----

    async def handle_audit_requested(self, client: Any, env: Any) -> None:
        """Resolve the ResearchSpec + LiteratureBrief + SpecInterpretation
        from the store, run one audit_literature_support, persist the
        EvidenceAuditReport, publish evidence.audited.
        从 store 解析 ResearchSpec + LiteratureBrief + SpecInterpretation，跑
        一次 audit_literature_support，持久化 EvidenceAuditReport，发布
        evidence.audited。"""
        session_id = env.session_id
        causation_id = env.event_id
        payload = env.payload or {}
        targeted_search_id = payload.get("targeted_search_id")

        try:
            research_spec = self.spec_store.get(payload.get("research_spec_ref"))
            literature_brief = self.spec_store.get(payload.get("literature_brief_ref"))
            spec_interpretation = self.spec_store.get(payload.get("spec_interpretation_ref"))
            report = await asyncio.to_thread(  # ④ concurrency: offload the blocking LLM call off the shared web event loop
                core.audit_literature_support,
                research_spec,
                literature_brief,
                spec_interpretation,
                self.llm_client,
                prompt_patch=scoped_prompt_patch(self.spec_store, (env.payload or {}).get("moss_intervention_ref"),
                        session_id, "EvidenceAuditAgent", self.config.prompt_patch()),
            )
        except core.EvidenceAuditError as exc:
            await self._fail(client, session_id, causation_id, exc.classification, exc.code, str(exc))
            return
        except Exception as exc:  # defensive: never crash the handler
            await self._fail(
                client, session_id, causation_id, "transient", "unexpected_error",
                f"{type(exc).__name__}: {exc}",
            )
            return

        report_key = (f"targeted-{targeted_search_id}-{artifact_digest(report)[7:]}"
                      if targeted_search_id else _EVIDENCE_AUDIT_REPORT_KEY)
        evidence_audit_report_ref = self.spec_store.put(
            kind="evidence_audit_report",
            session_id=session_id,
            key=report_key,
            obj=report,
            emitter=AGENT_NAME,
        )

        await client.publish(
            event_type="evidence.audited",
            session_id=session_id,
            payload={
                "evidence_audit_report_ref": evidence_audit_report_ref,
                **({"targeted_search_id": targeted_search_id}
                   if targeted_search_id else {}),
            },
            causation_id=causation_id,
        )

    # ---- evidence.citation_audit_requested (Mode C) ----

    async def handle_citation_audit_requested(self, client: Any, env: Any) -> None:
        """Resolve the ResearchMemory + ResearchSpec from the store, run one
        audit_citation_regression, persist the CitationAuditReport, publish
        evidence.citation_audited.
        从 store 解析 ResearchMemory + ResearchSpec，跑一次
        audit_citation_regression，持久化 CitationAuditReport，发布
        evidence.citation_audited。"""
        session_id = env.session_id
        causation_id = env.event_id
        payload = env.payload or {}

        if not payload.get("research_memory_ref"):
            # A session can reach Reporting via a decision/MOSS path (e.g. a
            # user AwaitingUserDecision action or a replication-failure
            # route) that skips ConsolidatingMemory entirely, leaving no
            # ResearchMemory to audit. Short-circuit with an empty, honest
            # CitationAuditReport instead of resolving a None ref / calling
            # the LLM.
            report = {
                "unsupported_claims": [],
                "checked_count": 0,
                "notes": _NO_RESEARCH_MEMORY_NOTE,
            }
            citation_audit_report_ref = self.spec_store.put(
                kind="citation_audit_report",
                session_id=session_id,
                key=_CITATION_AUDIT_REPORT_KEY,
                obj=report,
                emitter=AGENT_NAME,
            )
            await client.publish(
                event_type="evidence.citation_audited",
                session_id=session_id,
                payload={"citation_audit_report_ref": citation_audit_report_ref},
                causation_id=causation_id,
            )
            return

        try:
            research_memory = self.spec_store.get(payload.get("research_memory_ref"))
            research_spec = self.spec_store.get(payload.get("research_spec_ref"))
            report = await asyncio.to_thread(
                core.audit_citation_regression,
                research_memory,
                research_spec,
                self.llm_client,
                prompt_patch=scoped_prompt_patch(self.spec_store, (env.payload or {}).get("moss_intervention_ref"),
                        session_id, "EvidenceAuditAgent", self.config.prompt_patch()),
            )
        except core.EvidenceAuditParseError:
            # Citation audit is a reporting safeguard, not the experiment itself.
            # A malformed model response must not discard an already completed,
            # evidence-bound run. Preserve the missing-audit limitation explicitly;
            # infrastructure errors still take the normal retry/failure path below.
            report = {
                "unsupported_claims": [],
                "checked_count": 0,
                "notes": _CITATION_AUDIT_PARSE_FALLBACK_NOTE,
            }
        except core.EvidenceAuditError as exc:
            await self._fail(client, session_id, causation_id, exc.classification, exc.code, str(exc))
            return
        except Exception as exc:  # defensive: never crash the handler
            await self._fail(
                client, session_id, causation_id, "transient", "unexpected_error",
                f"{type(exc).__name__}: {exc}",
            )
            return

        citation_audit_report_ref = self.spec_store.put(
            kind="citation_audit_report",
            session_id=session_id,
            key=_CITATION_AUDIT_REPORT_KEY,
            obj=report,
            emitter=AGENT_NAME,
        )

        await client.publish(
            event_type="evidence.citation_audited",
            session_id=session_id,
            payload={"citation_audit_report_ref": citation_audit_report_ref},
            causation_id=causation_id,
        )

    # ---- failure path / 失败路径 ----

    async def _fail(
        self,
        client: Any,
        session_id: str,
        causation_id: str | None,
        reason_classification: str,
        underlying_reason_code: str,
        underlying_reason: str,
    ) -> None:
        await client.publish(
            event_type="agent.failed",
            session_id=session_id,
            payload={
                "agent_name": AGENT_NAME,
                "reason_classification": reason_classification,
                "underlying_reason": underlying_reason,
                "underlying_reason_code": underlying_reason_code,
            },
            causation_id=causation_id,
        )


__all__ = ["EvidenceAuditAgentHandler", "SCHEMA_VERSION"]
