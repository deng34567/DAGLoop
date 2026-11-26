"""
NoveltyAuditAgent EventBus handler: the thin transport wrapper around core.
NoveltyAuditAgent EventBus handler：core 之上的薄传输层。

Subscribes ``novelty.audit_requested`` (payload ``{research_spec_ref,
literature_brief_ref, spec_interpretation_ref, evidence_audit_report_ref?}``),
reads the frozen ResearchSpec, LiteratureBrief, and SpecInterpretation from
the Spec Store by ref (provenance — NEVER from the event payload); if
``evidence_audit_report_ref`` is present, also resolves the EvidenceAuditAgent
Mode A EvidenceAuditReport from the store and passes it to
``core.audit_step`` so the verdict is evidence-informed (absent -> None ->
unchanged behavior). Persists the resulting NoveltyAuditReport
(kind="novelty_audit_report"), and publishes ``novelty.audit_ready`` with its
verdict + ref. Any core failure (or unexpected error) is converted into
``agent.failed``, never a crash.

订阅 ``novelty.audit_requested``（payload ``{research_spec_ref,
literature_brief_ref, spec_interpretation_ref, evidence_audit_report_ref?}``），
按 ref 从 Spec Store 读回冻结的 ResearchSpec、LiteratureBrief、
SpecInterpretation（溯源——绝不读 payload）；若存在
``evidence_audit_report_ref``，还会从 store 解析 EvidenceAuditAgent 的 Mode A
EvidenceAuditReport 并传给 ``core.audit_step``，使裁定"证据知情"（缺失时为
None，行为不变）。把产出的 NoveltyAuditReport（kind="novelty_audit_report"）
写入 Spec Store，并发布 ``novelty.audit_ready``（携 verdict + ref）。core 的
任何失败（或未预期错误）都转成 ``agent.failed``，绝不崩溃。
"""

from __future__ import annotations

from autoresearch.agents.moss_agent.patch_writer import scoped_prompt_patch

import asyncio
import logging
import os
from typing import Any

from autoresearch.agents.novelty_audit_agent import core
from autoresearch.agents.novelty_audit_agent.kernel import AGENT_NAME, ConfigLoader
from autoresearch.spec_store.identity import artifact_digest
from autoresearch.spec_store.refs import decode

logger = logging.getLogger(__name__)

# P1 general honest-experiment fast-path: in GEN mode the real experiment (P1
# pipeline) re-designs from the user's question and does NOT use the literature
# novelty verdict, so a strict audit only re-cycles the DeepResearching<->audit
# loop (slow). When set, the novelty audit auto-passes (no LLM) so a GEN run
# reaches the experiment fast. Default OFF -> non-GEN behaviour is byte-identical.
_GEN_ENV_FLAG = "AUTORESEARCH_GEN_EXPERIMENT"

SCHEMA_VERSION = 1

_REPORT_KEY = "v1"


class NoveltyAuditAgentHandler:
    """Wires core.audit_step to a Spec Store + LLM + EventBus client.
    把 core.audit_step 接到 Spec Store + LLM + EventBus 客户端。"""

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
        """Subscribe this handler to its inbound event on an EventBusClient.
        在 EventBusClient 上订阅入站事件。"""
        client.subscribe(
            "novelty.audit_requested",
            self._make_eventbus_handler(client, self.handle_requested),
        )

    def _make_eventbus_handler(self, client: Any, fn: Any):
        async def _handler(env: Any) -> None:
            await fn(client, env)

        return _handler

    # ---- novelty.audit_requested ----

    async def handle_requested(self, client: Any, env: Any) -> None:
        """Resolve the three inputs from the store, run one audit_step,
        persist the report, publish novelty.audit_ready.
        从 store 解析三份输入，跑一次 audit_step，持久化报告，发布
        novelty.audit_ready。"""
        session_id = env.session_id
        causation_id = env.event_id
        payload = env.payload or {}
        targeted_search_id = payload.get("targeted_search_id")
        method_ref = payload.get("method_ref")

        try:
            method = self.spec_store.get(method_ref) if method_ref else None
            if (not targeted_search_id
                    and os.environ.get("AUTORESEARCH_NOVELTY_AUTOPASS") == "1"):
                # OPT-IN fast-path (default OFF): auto-pass (no LLM) so the literature loop can't
                # re-cycle before the real P1 experiment. By default the novelty audit now does the
                # REAL second literature search even in GEN mode (the flowchart's "新颖性·再search"
                # step) -- set AUTORESEARCH_NOVELTY_AUTOPASS=1 to restore the old skip. Schema:
                # verdict is the only required field; notes records why this is honest.
                report = {
                    "verdict": "pass",
                    "citation_sufficiency": ("GEN mode: literature novelty was not assessed; "
                                             "experiment routing does not require that claim."),
                    "novelty_status": "uncertain",
                    "stale_citations": [],
                    "unsupported_claims": [],
                    "missing_work": [],
                    "notes": "GEN mode: novelty audit skipped; experiment routing continues.",
                }
            else:
                research_spec = self.spec_store.get(payload.get("research_spec_ref"))
                literature_brief_ref = payload.get("literature_brief_ref")
                literature_brief = self.spec_store.get(literature_brief_ref)
                if (targeted_search_id
                        and literature_brief.get("searched_method_primary_core")
                        and not decode(literature_brief_ref).key.startswith(
                            f"targeted-{targeted_search_id}-")):
                    # Efficient Method revisions may reuse prior papers. Their frozen core
                    # belongs to the prior search, not the Method being audited now.
                    literature_brief = dict(literature_brief)
                    literature_brief.pop("searched_method_primary_core")
                spec_interpretation = self.spec_store.get(payload.get("spec_interpretation_ref"))
                evidence_audit_report_ref = payload.get("evidence_audit_report_ref")
                evidence_audit_report = (
                    self.spec_store.get(evidence_audit_report_ref)
                    if evidence_audit_report_ref
                    else None
                )
                # 3.3: the audit now judges the METHOD's novelty; load it when present (absent on
                # legacy/GEN paths -> None -> audits the spec as before, backward-safe).
                # ④ concurrency: offload the blocking LLM call off the shared web event loop
                report = await asyncio.to_thread(
                    core.audit_step,
                    research_spec,
                    literature_brief,
                    spec_interpretation,
                    self.llm_client,
                    prompt_patch=scoped_prompt_patch(self.spec_store, (env.payload or {}).get("moss_intervention_ref"),
                        session_id, "NoveltyAuditAgent", self.config.prompt_patch()),
                    evidence_audit_report=evidence_audit_report,
                    method=method,
                )
                # GEN / repo-improve: every normal and targeted second search still RUNS (so its
                # findings are real, not a no-op), but it is INFORMATIONAL and never BLOCKS --
                # improving an existing repo is not itself an academic novelty claim. Feasibility
                # and metric gain are established by the real experiment, not by the deprecated
                # feasibility fields in this literature report. Record the findings, then route
                # forward. AUTORESEARCH_NOVELTY_STRICT=1 restores blocking for research that
                # explicitly requires academic novelty. Non-GEN research remains strict.
                if (os.environ.get(_GEN_ENV_FLAG) == "1"
                        and os.environ.get("AUTORESEARCH_NOVELTY_STRICT") != "1"
                        and report.get("verdict") != "pass"):
                    report["notes"] = ("GEN mode: real second search done; its finding is "
                                       "retained while experiment routing continues. "
                                       + (report.get("notes") or ""))[:1500]
                    # The event controls routing; the stored report remains the auditor's
                    # actual finding for later scientific attribution.
        except core.NoveltyAuditError as exc:
            await self._fail(client, session_id, causation_id, exc.classification, exc.code, str(exc))
            return
        except Exception as exc:  # defensive: never crash the handler
            await self._fail(
                client, session_id, causation_id, "transient", "unexpected_error",
                f"{type(exc).__name__}: {exc}",
            )
            return

        if targeted_search_id:
            report_key = f"targeted-{targeted_search_id}-{artifact_digest(report)[7:]}"
        elif method is not None:
            report_key = (
                f"method-{artifact_digest(method)[7:]}-{artifact_digest(report)[7:]}")
        else:
            report_key = _REPORT_KEY
        novelty_audit_report_ref = self.spec_store.put(
            kind="novelty_audit_report",
            session_id=session_id,
            key=report_key,
            obj=report,
            emitter=AGENT_NAME,
        )

        await client.publish(
            event_type="novelty.audit_ready",
            session_id=session_id,
            payload={
                "verdict": (
                    "pass" if os.environ.get(_GEN_ENV_FLAG) == "1"
                    and os.environ.get("AUTORESEARCH_NOVELTY_STRICT") != "1"
                    else report["verdict"]
                ),
                "novelty_audit_report_ref": novelty_audit_report_ref,
                **({"targeted_search_id": targeted_search_id}
                   if targeted_search_id else {}),
            },
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


__all__ = ["NoveltyAuditAgentHandler", "SCHEMA_VERSION"]
