"""
DeepResearchAgent EventBus handler: the thin transport wrapper around core.
DeepResearchAgent EventBus handler：core 之上的薄传输层。

Subscribes ``deep_research.requested`` (payload ``{research_spec_ref}``),
reads the frozen ResearchSpec from the Spec Store by ref (provenance —
NEVER from the event payload), runs ``core.deep_research_step``, persists
the resulting ``LiteratureBrief`` (kind="literature_brief") and
``FieldTrajectoryBrief`` (kind="field_trajectory_brief") to the Spec Store,
and publishes ``literature.brief_ready`` with both refs. Any core failure
(or unexpected error) is converted into ``agent.failed``, never a crash.

订阅 ``deep_research.requested``（payload ``{research_spec_ref}``），按 ref
从 Spec Store 读回冻结的 ResearchSpec（溯源——绝不读 payload），跑一次
``core.deep_research_step``，把产出的 LiteratureBrief（kind="literature_brief"）
与 FieldTrajectoryBrief（kind="field_trajectory_brief"）写入 Spec Store，并发布
``literature.brief_ready``（携两个 ref）。core 的任何失败（或未预期错误）都
转成 ``agent.failed``，绝不崩溃。
"""

from __future__ import annotations

from autoresearch.agents.moss_agent.patch_writer import scoped_prompt_patch

import asyncio
import logging
from typing import Any

from autoresearch.agents.deep_research_agent import core
from autoresearch.agents.deep_research_agent.kernel import AGENT_NAME, ConfigLoader
from autoresearch.spec_store.identity import artifact_digest

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1

_BRIEF_KEY = "final"


class DeepResearchAgentHandler:
    """Wires core.deep_research_step to a Spec Store + search backend + LLM +
    EventBus client.
    把 core.deep_research_step 接到 Spec Store + 搜索后端 + LLM + EventBus
    客户端。"""

    def __init__(
        self,
        *,
        spec_store: Any,
        backend: Any,
        llm_client: Any,
        config_root: Any | None = None,
    ) -> None:
        self.spec_store = spec_store
        self.backend = backend
        self.llm_client = llm_client
        self.config = ConfigLoader(config_root)

    # ---- EventBus registration / 注册 ----

    def register(self, client: Any) -> None:
        """Subscribe this handler to its inbound event on an EventBusClient.
        在 EventBusClient 上订阅入站事件。"""
        client.subscribe(
            "deep_research.requested",
            self._make_eventbus_handler(client, self.handle_requested),
        )

    def _make_eventbus_handler(self, client: Any, fn: Any):
        async def _handler(env: Any) -> None:
            await fn(client, env)

        return _handler

    # ---- deep_research.requested ----

    async def handle_requested(self, client: Any, env: Any) -> None:
        """Resolve the ResearchSpec from the store, run one deep_research_step,
        persist the two briefs, publish literature.brief_ready.
        从 store 解析 ResearchSpec，跑一次 deep_research_step，持久化两份
        brief，发布 literature.brief_ready。"""
        session_id = env.session_id
        causation_id = env.event_id
        research_spec_ref = (env.payload or {}).get("research_spec_ref")
        targeted_search_id = (env.payload or {}).get("targeted_search_id")

        try:
            research_spec = self.spec_store.get(research_spec_ref)
            method_ref = (env.payload or {}).get("method_ref")
            audit_ref = (env.payload or {}).get("novelty_audit_report_ref")
            target_method = self.spec_store.get(method_ref) if method_ref else None
            novelty_audit = self.spec_store.get(audit_ref) if audit_ref else None
            # ④ concurrency: offload the blocking LLM call off the shared web event loop
            result = await asyncio.to_thread(
                core.deep_research_step,
                research_spec,
                self.backend,
                self.llm_client,
                prompt_patch=scoped_prompt_patch(self.spec_store, (env.payload or {}).get("moss_intervention_ref"),
                        session_id, "DeepResearchAgent", self.config.prompt_patch()),
                target_method=target_method,
                novelty_audit=novelty_audit,
            )
        except core.DeepResearchParseError as exc:
            # A malformed synthesis is an evidence gap, not a reason to strand the
            # whole research session.  Preserve that gap as valid, explicitly empty
            # briefs so downstream modeling can continue without inventing papers.
            # Provider/network failures still use the normal agent.failed path below.
            try:
                spec = self.spec_store.get(research_spec_ref) or {}
            except Exception:  # noqa: BLE001 -- the original failure remains the diagnosis
                spec = {}
            query = str(
                spec.get("research_goal") or spec.get("topic") or "the requested task"
            )
            gap = {
                "direction_id": "deep_research",
                "query": query,
                "reason": (
                    "DeepResearchAgent did not produce a schema-valid synthesis after "
                    "bounded retries; no literature claims were accepted."
                ),
                "attempted": True,
            }
            fallback_literature = {
                "summary": (
                    "No schema-valid literature synthesis was available after the bounded "
                    "search attempt. Treat the literature evidence as an explicit gap."
                ),
                "related_work": [],
                "key_papers": [],
                "existing_benchmarks": [],
                "evidence_gaps": [gap],
            }
            fallback_trajectory = {
                "origin": "No validated field trajectory was produced.",
                "current_state": "Field state is unknown because literature synthesis failed.",
                "main_method_families": [],
                "benchmark_evolution": "Unknown; no validated literature brief is available.",
                "open_problems": ["Recover validated literature evidence before making claims."],
            }
            suffix = targeted_search_id or "initial"
            literature_brief_ref = self.spec_store.put(
                kind="literature_brief", session_id=session_id,
                key=f"fallback-{suffix}", obj=fallback_literature, emitter=AGENT_NAME,
            )
            field_trajectory_brief_ref = self.spec_store.put(
                kind="field_trajectory_brief", session_id=session_id,
                key=f"fallback-{suffix}", obj=fallback_trajectory, emitter=AGENT_NAME,
            )
            await client.publish(
                event_type="literature.brief_ready",
                session_id=session_id,
                payload={
                    "literature_brief_ref": literature_brief_ref,
                    "field_trajectory_brief_ref": field_trajectory_brief_ref,
                    **({
                        "method_ref": method_ref,
                        "targeted_search_id": targeted_search_id,
                        **({"novelty_audit_report_ref": audit_ref} if audit_ref else {}),
                    } if targeted_search_id else {}),
                },
                causation_id=causation_id,
            )
            logger.warning(
                "DeepResearchAgent parse failure on session %s; continuing with explicit "
                "evidence-gap briefs: %s", session_id[:8], exc,
            )
            return
        except core.DeepResearchError as exc:
            await self._fail(client, session_id, causation_id, exc.classification, exc.code, str(exc))
            return
        except Exception as exc:  # defensive: never crash the handler
            await self._fail(
                client, session_id, causation_id, "transient", "unexpected_error",
                f"{type(exc).__name__}: {exc}",
            )
            return

        literature_key = (
            f"targeted-{targeted_search_id}-{artifact_digest(result.literature_brief)[7:]}"
            if targeted_search_id else _BRIEF_KEY)
        trajectory_key = (
            f"targeted-{targeted_search_id}-{artifact_digest(result.field_trajectory_brief)[7:]}"
            if targeted_search_id else _BRIEF_KEY)
        literature_brief_ref = self.spec_store.put(
            kind="literature_brief",
            session_id=session_id,
            key=literature_key,
            obj=result.literature_brief,
            emitter=AGENT_NAME,
        )
        field_trajectory_brief_ref = self.spec_store.put(
            kind="field_trajectory_brief",
            session_id=session_id,
            key=trajectory_key,
            obj=result.field_trajectory_brief,
            emitter=AGENT_NAME,
        )

        await client.publish(
            event_type="literature.brief_ready",
            session_id=session_id,
            payload={
                "literature_brief_ref": literature_brief_ref,
                "field_trajectory_brief_ref": field_trajectory_brief_ref,
                **({
                    "method_ref": method_ref,
                    "targeted_search_id": targeted_search_id,
                    **({"novelty_audit_report_ref": audit_ref} if audit_ref else {}),
                } if targeted_search_id else {}),
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


__all__ = ["DeepResearchAgentHandler", "SCHEMA_VERSION"]
