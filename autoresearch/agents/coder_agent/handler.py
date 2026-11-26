"""
CoderAgent EventBus handler: the thin transport wrapper around core.
CoderAgent EventBus handler：core 之上的薄传输层。

Subscribes ONE event:

``code.implementation_requested`` (payload ``{experiment_plan_ref,
verifier_spec_ref}``) -- reads the ExperimentPlan and VerifierSpec from the
Spec Store by ref (provenance -- NEVER from the event payload), runs
``core.implement_patch``, persists the resulting ExperimentPatch
(kind="experiment_patch", content-addressed snapshot), and publishes ``code.implemented`` with
its ref.

Any core failure (or unexpected error) is converted into ``agent.failed``,
never a crash.

订阅一个事件：code.implementation_requested（按 ref 从 Spec Store 读回
ExperimentPlan、VerifierSpec，跑一次 core.implement_patch，持久化
ExperimentPatch，发布 code.implemented）。core 的任何失败（或未预期错误）
都转成 agent.failed，绝不崩溃。
"""

from __future__ import annotations

from autoresearch.agents.moss_agent.patch_writer import scoped_prompt_patch

import asyncio
import logging
import os
from typing import Any

from autoresearch.agents.coder_agent import core
from autoresearch.agents.coder_agent.kernel import AGENT_NAME, ConfigLoader
from autoresearch.spec_store import refs
from autoresearch.spec_store.identity import artifact_digest

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1

_GEN_ENV_FLAG = "AUTORESEARCH_GEN_EXPERIMENT"


class CoderAgentHandler:
    """Wires core.implement_patch to a Spec Store + LLM + EventBus client.
    把 core.implement_patch 接到 Spec Store + LLM + EventBus 客户端。"""

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
        """Subscribe this handler's event on an EventBusClient.
        在 EventBusClient 上订阅本 handler 的事件。"""
        client.subscribe(
            "code.implementation_requested",
            self._make_eventbus_handler(client, self.handle_implementation_requested),
        )

    def _make_eventbus_handler(self, client: Any, fn: Any):
        async def _handler(env: Any) -> None:
            await fn(client, env)

        return _handler

    # ---- code.implementation_requested ----

    async def handle_implementation_requested(self, client: Any, env: Any) -> None:
        """Resolve the ExperimentPlan + VerifierSpec from the store, run one
        implement_patch, persist the ExperimentPatch, publish
        code.implemented.
        从 store 解析 ExperimentPlan + VerifierSpec，跑一次 implement_patch，
        持久化 ExperimentPatch，发布 code.implemented。"""
        session_id = env.session_id
        causation_id = env.event_id
        payload = env.payload or {}
        receipt_key = "coder-request-" + str(causation_id)
        try:
            completed = self.spec_store.get(refs.encode("audit", session_id, receipt_key))
        except KeyError:
            completed = None
        if completed is not None:
            if completed["request_digest"] != artifact_digest(payload):
                raise ValueError("completed implementation request identity mismatch")
            await client.publish(event_type="code.implemented", session_id=session_id,
                                 payload=completed["payload"], causation_id=causation_id)
            return

        try:
            experiment_plan = self.spec_store.get(payload.get("experiment_plan_ref"))
            verifier_spec = ((self.spec_store.get(payload.get("verifier_spec_ref")) or {}) if payload.get("verifier_spec_ref") else {})

            if self._gen_branch():
                experiment_patch = self._trivial_patch(verifier_spec)
            else:
                # ④ concurrency: offload the blocking LLM call off the shared web event loop
                experiment_patch = await asyncio.to_thread(
                    core.implement_patch,
                    experiment_plan, verifier_spec, self.llm_client,
                    prompt_patch=scoped_prompt_patch(self.spec_store, (env.payload or {}).get("moss_intervention_ref"),
                        session_id, "CoderAgent", self.config.prompt_patch()),
                )
        except core.CoderError as exc:
            await self._fail(client, session_id, causation_id, exc.classification, exc.code, str(exc))
            return
        except Exception as exc:  # defensive: never crash the handler
            await self._fail(
                client, session_id, causation_id, "transient", "unexpected_error",
                f"{type(exc).__name__}: {exc}",
            )
            return

        experiment_patch_ref = self.spec_store.put(
            kind="experiment_patch",
            session_id=session_id,
            key="snapshot-" + artifact_digest(experiment_patch).split(":", 1)[1],
            obj=experiment_patch,
            emitter=AGENT_NAME,
        )

        completion_payload = {"experiment_patch_ref": experiment_patch_ref}
        self.spec_store.put(kind="audit", session_id=session_id, key=receipt_key,
                            obj={"request_digest": artifact_digest(payload), "payload": completion_payload},
                            emitter=AGENT_NAME)
        await client.publish(
            event_type="code.implemented",
            session_id=session_id,
            payload=completion_payload,
            causation_id=causation_id,
        )

    # ---- P1 trivial-patch branch (env gated, default OFF) ----

    def _gen_branch(self) -> bool:
        """Whether to emit a trivial patch instead of calling the real (hang-prone)
        coder LLM. True only under AUTORESEARCH_GEN_EXPERIMENT=1.

        In P1 the ExperimentPatch is UNUSED (the pipeline self-writes method.py),
        but this state runs BEFORE RunningExperiment and the real coder is a live
        LLM call observed to HANG and stall the single engine thread; a trivial
        in-editable-region patch lets the protocol gate pass and advance."""
        return os.environ.get(_GEN_ENV_FLAG) == "1"

    def _trivial_patch(self, verifier_spec: Any) -> dict:
        """Build a minimal schema-valid ExperimentPatch touching a single file
        inside the frozen editable_region (so touched_files ⊆ editable_region)."""
        # meeting #1: with the pre-baseline verifier removed, there may be NO editable_region.
        # Permissive default: an absent boundary means "the whole repo is editable" (anti-cheat is
        # preserved structurally by the trusted evaluator, not by a pre-set whitelist), so fall back
        # to a conventional target file rather than failing the round.
        editable_region = (verifier_spec or {}).get("editable_region") or []
        target = editable_region[0] if editable_region else "main.py"
        patch = {
            "summary": "P1 no-op patch (method self-written by the pipeline).",
            "changes": [
                {"file": target, "patch": "# autoresearch P1: no-op edit\n"},
            ],
            "touched_files": [target],
        }
        errors = core._validate_experiment_patch(patch)
        if errors:
            raise core.CoderParseError(f"trivial ExperimentPatch failed schema validation: {errors}")
        return patch

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


__all__ = ["CoderAgentHandler", "SCHEMA_VERSION"]
