"""
HypothesisAgent EventBus handler: the thin transport wrapper around core.
HypothesisAgent EventBus handler：core 之上的薄传输层。

Subscribes ``hypothesis.generation_requested`` (payload ``{research_spec_ref,
literature_brief_ref}``), reads the frozen ResearchSpec and LiteratureBrief
from the Spec Store by ref (provenance — NEVER from the event payload), runs
``core.generate_step``, re-wraps the resulting list of hypotheses as a
HypothesisBank (``{"hypotheses": [...]}``), persists it (kind=
"hypothesis_bank"), and publishes ``hypothesis.generated`` with its ref +
count. Any core failure (or unexpected error) is converted into
``agent.failed``, never a crash.

订阅 ``hypothesis.generation_requested``（payload ``{research_spec_ref,
literature_brief_ref}``），按 ref 从 Spec Store 读回冻结的 ResearchSpec、
LiteratureBrief（溯源——绝不读 payload），跑一次 ``core.generate_step``，把
产出的假设列表重新包成 HypothesisBank（``{"hypotheses": [...]}``）写入 Spec
Store（kind="hypothesis_bank"），并发布 ``hypothesis.generated``（携 ref +
count）。core 的任何失败（或未预期错误）都转成 ``agent.failed``，绝不崩溃。
"""

from __future__ import annotations

from autoresearch.agents.moss_agent.patch_writer import scoped_prompt_patch

import asyncio
import logging
from typing import Any

from autoresearch.agents.hypothesis_agent import core
from autoresearch.agents.hypothesis_agent.kernel import AGENT_NAME, ConfigLoader

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1

_BANK_KEY = "v1"


class HypothesisAgentHandler:
    """Wires core.generate_step to a Spec Store + LLM + EventBus client.
    把 core.generate_step 接到 Spec Store + LLM + EventBus 客户端。"""

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
            "hypothesis.generation_requested",
            self._make_eventbus_handler(client, self.handle_requested),
        )

    def _make_eventbus_handler(self, client: Any, fn: Any):
        async def _handler(env: Any) -> None:
            await fn(client, env)

        return _handler

    # ---- hypothesis.generation_requested ----

    async def handle_requested(self, client: Any, env: Any) -> None:
        """Resolve the two inputs from the store, run one generate_step,
        persist the bank, publish hypothesis.generated.
        从 store 解析两份输入，跑一次 generate_step，持久化 bank，发布
        hypothesis.generated。"""
        session_id = env.session_id
        causation_id = env.event_id
        payload = env.payload or {}

        try:
            research_spec = self.spec_store.get(payload.get("research_spec_ref"))
            literature_brief = self.spec_store.get(payload.get("literature_brief_ref"))
            # idea<->experiment loop: on round 2+ the accumulated ResearchMemory is present,
            # so the next hypothesis is proposed with awareness of what's already been tried.
            prior_rm_ref = payload.get("research_memory_ref")
            prior_research_memory = self.spec_store.get(prior_rm_ref) if prior_rm_ref else None
            # ④ concurrency: offload the blocking LLM call off the shared web event loop
            hypotheses = await asyncio.to_thread(
                core.generate_step,
                research_spec,
                literature_brief,
                self.llm_client,
                prompt_patch=scoped_prompt_patch(self.spec_store, (env.payload or {}).get("moss_intervention_ref"),
                        session_id, "HypothesisAgent", self.config.prompt_patch()),
                prior_research_memory=prior_research_memory,
            )
        except core.HypothesisError as exc:
            await self._fail(client, session_id, causation_id, exc.classification, exc.code, str(exc))
            return
        except Exception as exc:  # defensive: never crash the handler
            await self._fail(
                client, session_id, causation_id, "transient", "unexpected_error",
                f"{type(exc).__name__}: {exc}",
            )
            return

        hypothesis_bank_ref = self.spec_store.put(
            kind="hypothesis_bank",
            session_id=session_id,
            key=_BANK_KEY,
            obj={"hypotheses": hypotheses},
            emitter=AGENT_NAME,
        )

        await client.publish(
            event_type="hypothesis.generated",
            session_id=session_id,
            payload={
                "hypothesis_bank_ref": hypothesis_bank_ref,
                "count": len(hypotheses),
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


__all__ = ["HypothesisAgentHandler", "SCHEMA_VERSION"]
