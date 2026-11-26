"""
ProblemModelingAgent EventBus handler: the thin transport wrapper around core.
ProblemModelingAgent EventBus handler：core 之上的薄传输层。

Drives the interactive ClarifyingResearchGoal dialogue:
  - ``problem_modeling.requested``: seeds a fresh clarification working state
    (draft_spec, answered, research_goal_text) and persists it to the Spec
    Store (kind="research_spec_draft"), then runs one core.clarify_step turn.
  - ``human.clarification_answer``: reads the working state back from the
    Spec Store by session_id (provenance — never from the event payload),
    appends the answer, runs one core.clarify_step turn, and re-persists.

Each turn either publishes the next ``problem_modeling.clarification_question``,
or — once core signals done — freezes the ResearchSpec into the Spec Store
(kind="research_spec") and publishes ``research_spec.ready``. Any core
failure (or unexpected error) is converted into ``agent.failed``, never a
crash.

Also drives the ReinterpretingSpec turn:
  - ``problem_modeling.reinterpret_requested``: reads the frozen ResearchSpec,
    LiteratureBrief, and FieldTrajectoryBrief back from the Spec Store by
    their refs (provenance — never from the event payload), runs one
    reinterpret_core.reinterpret_step, persists the resulting
    SpecInterpretation (kind="spec_interpretation") and revised ResearchSpec
    (kind="research_spec", key="reinterpreted" — always written regardless of
    disposition), and publishes ``spec.reinterpreted``.

Also drives the ModelingProblem turn:
  - ``problem_modeling.model_requested``: reads the frozen ResearchSpec,
    LiteratureBrief, and NoveltyAuditReport back from the Spec Store by their
    refs (provenance — never from the event payload), runs one
    model_core.model_step, persists the resulting ProblemModel
    (kind="problem_model", key="v1"), and publishes ``problem.modeled``.

驱动交互式 ClarifyingResearchGoal 对话：``problem_modeling.requested`` 播种一份
新的澄清工作状态（draft_spec/answered/research_goal_text），写入 Spec Store
（kind="research_spec_draft"）后跑一轮 core.clarify_step；
``human.clarification_answer`` 按 session_id 从 Spec Store 读回工作状态（溯
源——绝不读 payload），追加答案，跑一轮，再写回。每轮要么发下一个
``problem_modeling.clarification_question``，要么（core 判定 done）把
ResearchSpec 冻结进 Spec Store（kind="research_spec"）并发
``research_spec.ready``。core 的任何失败（或未预期错误）都转成
``agent.failed``，绝不崩溃。

同时驱动 ReinterpretingSpec 一轮：``problem_modeling.reinterpret_requested``
按 ref 从 Spec Store 读回冻结的 ResearchSpec、LiteratureBrief、
FieldTrajectoryBrief（溯源——绝不读 payload），跑一次
reinterpret_core.reinterpret_step，把产出的 SpecInterpretation
（kind="spec_interpretation"）与修订版 ResearchSpec（kind="research_spec"，
key="reinterpreted" —— 无论 disposition 如何都会写入）持久化，并发布
``spec.reinterpreted``。

同时驱动 ModelingProblem 一轮：``problem_modeling.model_requested`` 按 ref 从
Spec Store 读回冻结的 ResearchSpec、LiteratureBrief、NoveltyAuditReport
（溯源——绝不读 payload），跑一次 model_core.model_step，把产出的 ProblemModel
（kind="problem_model"，key="v1"）持久化，并发布 ``problem.modeled``。
"""

from __future__ import annotations

from autoresearch.agents.moss_agent.patch_writer import scoped_prompt_patch

import asyncio
import logging
from typing import Any

from autoresearch.agents.problem_modeling_agent import core, model_core, reinterpret_core
from autoresearch.agents.comparison_guardian_agent import core as comparison_guardian
from autoresearch.agents.problem_modeling_agent.kernel import AGENT_NAME, ConfigLoader
from autoresearch.spec_store import refs as spec_refs

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1

_DRAFT_KEY = "v1"
_FINAL_KEY = "final"
_SPEC_INTERPRETATION_KEY = "v1"
_REINTERPRETED_SPEC_KEY = "reinterpreted"
_PROBLEM_MODEL_KEY = "v1"
_MAX_SEMANTIC_RESTATEMENTS = 2


def _fresh_empty_draft() -> dict:
    """The 8 ResearchSpec keys (research_spec.schema.yaml), all null/empty —
    the starting point of every clarification dialogue.
    ResearchSpec 的 8 个字段，全部 null/empty —— 每次澄清对话的起点。"""
    return {
        "topic": None,
        "research_goal": None,
        "task_mode": None,
        "budget": None,
        "deliverable_form": None,
        "editable_scope": [],
        "forbidden_items": [],
        "open_questions": [],
    }


def _question_payload(question: dict) -> dict:
    """Trim an LLM-produced question dict down to the
    problem_modeling.clarification_question schema's exact fields
    (additionalProperties: false).
    把 LLM 产出的 question 裁剪到 clarification_question schema 的精确字段
    （additionalProperties: false）。"""
    payload = {
        "question_id": question["question_id"],
        "question_text": question["question_text"],
        "allow_free_text": question["allow_free_text"],
    }
    if question.get("options") is not None:
        payload["options"] = question["options"]
    return payload


class ProblemModelingAgentHandler:
    """Wires the ClarifyingResearchGoal clarify_step loop to a Spec Store +
    LLM + EventBus client.
    把 ClarifyingResearchGoal 的 clarify_step 循环接到 Spec Store + LLM +
    EventBus 客户端。"""

    def __init__(
        self, *, spec_store: Any, llm_client: Any,
        semantic_guardian_llm: Any | None = None,
        config_root: Any | None = None,
    ) -> None:
        self.spec_store = spec_store
        self.llm_client = llm_client
        self.semantic_guardian_llm = semantic_guardian_llm
        self.config = ConfigLoader(config_root)

    # ---- EventBus registration / 注册 ----

    def register(self, client: Any) -> None:
        """Subscribe this handler to both inbound events on an EventBusClient.
        在 EventBusClient 上订阅两个入站事件。"""
        client.subscribe(
            "problem_modeling.requested",
            self._make_eventbus_handler(client, self.handle_requested),
        )
        client.subscribe(
            "human.clarification_answer",
            self._make_eventbus_handler(client, self.handle_answer),
        )
        client.subscribe(
            "problem_modeling.reinterpret_requested",
            self._make_eventbus_handler(client, self.handle_reinterpret_requested),
        )
        client.subscribe(
            "problem_modeling.model_requested",
            self._make_eventbus_handler(client, self.handle_model_requested),
        )

    def _make_eventbus_handler(self, client: Any, fn: Any):
        async def _handler(env: Any) -> None:
            await fn(client, env)

        return _handler

    # ---- problem_modeling.requested / 首轮播种 ----

    async def handle_requested(self, client: Any, env: Any) -> None:
        """Seed a fresh working state, persist it, then run one clarify turn.
        播种一份新的工作状态、持久化，再跑一轮澄清。"""
        session_id = env.session_id
        causation_id = env.event_id
        research_goal_text = (env.payload or {}).get("research_goal_text", "")

        working_state = None
        if research_goal_text in ("", "(restarted)"):
            try:
                working_state = self._load_working_state(session_id)
            except Exception:  # noqa: BLE001 -- first start / legacy session without a draft
                pass
        if working_state is None:
            working_state = {
                "draft_spec": _fresh_empty_draft(),
                "answered": [],
                "research_goal_text": research_goal_text,
            }

        try:
            self._save_working_state(session_id, working_state)
            # ④ concurrency: offload the blocking LLM call off the shared web event loop
            decision = await asyncio.to_thread(
                core.clarify_step,
                working_state["research_goal_text"],
                working_state["draft_spec"],
                working_state["answered"],
                self.llm_client,
                prompt_patch=scoped_prompt_patch(self.spec_store, (env.payload or {}).get("moss_intervention_ref"),
                        session_id, "ProblemModelingAgent", self.config.prompt_patch()),
            )
        except core.ClarifyError as exc:
            await self._fail(client, session_id, causation_id, exc.classification, exc.code, str(exc))
            return
        except Exception as exc:  # defensive: never crash the handler
            await self._fail(
                client, session_id, causation_id, "transient", "unexpected_error",
                f"{type(exc).__name__}: {exc}",
            )
            return

        await self._branch(client, session_id, causation_id, decision)

    # ---- human.clarification_answer / 后续轮次 ----

    async def handle_answer(self, client: Any, env: Any) -> None:
        """Read working state from the Spec Store (provenance — NOT the
        payload), append the answer, run one clarify turn, re-persist.
        从 Spec Store 读工作状态（溯源——不读 payload），追加答案，跑一轮，
        再写回。"""
        session_id = env.session_id
        causation_id = env.event_id
        payload = env.payload or {}

        try:
            working_state = self._load_working_state(session_id)
            working_state["answered"].append(
                {
                    "question_id": payload.get("question_id"),
                    "answer_text": payload.get("answer_text"),
                }
            )
            decision = await asyncio.to_thread(
                core.clarify_step,
                working_state["research_goal_text"],
                working_state["draft_spec"],
                working_state["answered"],
                self.llm_client,
                prompt_patch=scoped_prompt_patch(self.spec_store, (env.payload or {}).get("moss_intervention_ref"),
                        session_id, "ProblemModelingAgent", self.config.prompt_patch()),
            )
            self._save_working_state(session_id, working_state)
        except core.ClarifyError as exc:
            await self._fail(client, session_id, causation_id, exc.classification, exc.code, str(exc))
            return
        except Exception as exc:  # defensive: never crash the handler
            await self._fail(
                client, session_id, causation_id, "transient", "unexpected_error",
                f"{type(exc).__name__}: {exc}",
            )
            return

        await self._branch(client, session_id, causation_id, decision)

    # ---- working-state persistence / 工作状态持久化 ----

    def _save_working_state(self, session_id: str, working_state: dict) -> str:
        return self.spec_store.put(
            kind="research_spec_draft",
            session_id=session_id,
            key=_DRAFT_KEY,
            obj=working_state,
            emitter=AGENT_NAME,
        )

    def _load_working_state(self, session_id: str) -> dict:
        ref = spec_refs.encode("research_spec_draft", session_id, _DRAFT_KEY)
        return self.spec_store.get(ref)

    # ---- branch on the clarify_step decision / 按决策分支 ----

    async def _branch(self, client: Any, session_id: str, causation_id: str | None, decision: Any) -> None:
        if decision.done:
            await self._finalize(client, session_id, causation_id, decision.spec)
        else:
            await self._publish_question(client, session_id, causation_id, decision.question)

    async def _publish_question(
        self, client: Any, session_id: str, causation_id: str | None, question: dict
    ) -> None:
        working_state = self._load_working_state(session_id)
        working_state["pending_question"] = _question_payload(question)
        self._save_working_state(session_id, working_state)
        await client.publish(
            event_type="problem_modeling.clarification_question",
            session_id=session_id,
            payload=_question_payload(question),
            causation_id=causation_id,
        )

    async def _finalize(
        self, client: Any, session_id: str, causation_id: str | None, spec: dict
    ) -> None:
        ref = self.spec_store.put(
            kind="research_spec",
            session_id=session_id,
            key=_FINAL_KEY,
            obj=spec,
            emitter=AGENT_NAME,
        )
        await client.publish(
            event_type="research_spec.ready",
            session_id=session_id,
            payload={"research_spec_ref": ref},
            causation_id=causation_id,
        )

    # ---- problem_modeling.reinterpret_requested / ReinterpretingSpec ----

    async def handle_reinterpret_requested(self, client: Any, env: Any) -> None:
        """Resolve the ResearchSpec + LiteratureBrief + FieldTrajectoryBrief
        from the Spec Store by their refs (provenance — NOT the payload), run
        one reinterpret_step, persist the SpecInterpretation + revised
        ResearchSpec, publish spec.reinterpreted.
        按 ref 从 Spec Store 解析 ResearchSpec + LiteratureBrief +
        FieldTrajectoryBrief（溯源——不读 payload），跑一次 reinterpret_step，
        持久化 SpecInterpretation + 修订版 ResearchSpec，发布
        spec.reinterpreted。"""
        session_id = env.session_id
        causation_id = env.event_id
        payload = env.payload or {}

        try:
            research_spec = self.spec_store.get(payload.get("research_spec_ref"))
            literature_brief = self.spec_store.get(payload.get("literature_brief_ref"))
            field_trajectory_brief = self.spec_store.get(
                payload.get("field_trajectory_brief_ref")
            )
            # New sessions retain the exact original user text in the clarification draft.
            # Legacy/imported sessions may not; preserve their old single-pass behavior.
            try:
                original_user_goal = self._load_working_state(session_id).get(
                    "research_goal_text", ""
                )
            except Exception:  # noqa: BLE001 -- legacy session without a draft artifact
                original_user_goal = ""

            guardian_guidance = ""
            result = None
            for _ in range(_MAX_SEMANTIC_RESTATEMENTS):
                patches = [p for p in (self.config.prompt_patch(), guardian_guidance) if p]
                result = await asyncio.to_thread(
                    reinterpret_core.reinterpret_step,
                    research_spec,
                    field_trajectory_brief,
                    literature_brief,
                    self.llm_client,
                    original_user_goal=original_user_goal,
                    prompt_patch="\n\n".join(patches) or None,
                )
                if not original_user_goal or self.semantic_guardian_llm is None:
                    break
                audit = await asyncio.to_thread(
                    comparison_guardian.audit_semantic_restatement,
                    original_user_goal,
                    research_spec,
                    result.spec_interpretation,
                    self.semantic_guardian_llm,
                    literature_brief=literature_brief,
                    field_trajectory_brief=field_trajectory_brief,
                )
                if audit["verdict"] == "pass":
                    break
                guardian_guidance = (
                    "[ComparisonGuardian correction]\n"
                    "The previous literature-grounded restatement changed or failed to prove "
                    "the original scientific comparison. Rewrite it without silently changing "
                    "explicit user intent.\n"
                    f"Issues: {audit['issues']}\n"
                    f"Required remediation: {audit['remediation']}"
                )
            else:
                raise reinterpret_core.ReinterpretParseError(
                    "ComparisonGuardian rejected the semantic restatement after "
                    f"{_MAX_SEMANTIC_RESTATEMENTS} attempts: {audit['reasoning']}"
                )
            assert result is not None
        except reinterpret_core.ReinterpretError as exc:
            await self._fail(client, session_id, causation_id, exc.classification, exc.code, str(exc))
            return
        except Exception as exc:  # defensive: never crash the handler
            await self._fail(
                client, session_id, causation_id, "transient", "unexpected_error",
                f"{type(exc).__name__}: {exc}",
            )
            return

        spec_interpretation_ref = self.spec_store.put(
            kind="spec_interpretation",
            session_id=session_id,
            key=_SPEC_INTERPRETATION_KEY,
            obj=result.spec_interpretation,
            emitter=AGENT_NAME,
        )
        # Always persisted regardless of disposition — disposition is only a
        # routing signal, not a gate on whether the revised spec gets written.
        revised_research_spec_ref = self.spec_store.put(
            kind="research_spec",
            session_id=session_id,
            key=_REINTERPRETED_SPEC_KEY,
            obj=result.revised_research_spec,
            emitter=AGENT_NAME,
        )

        await client.publish(
            event_type="spec.reinterpreted",
            session_id=session_id,
            payload={
                "disposition": result.disposition,
                "spec_interpretation_ref": spec_interpretation_ref,
                "revised_research_spec_ref": revised_research_spec_ref,
            },
            causation_id=causation_id,
        )

    # ---- problem_modeling.model_requested / ModelingProblem ----

    async def handle_model_requested(self, client: Any, env: Any) -> None:
        """Resolve the ResearchSpec + LiteratureBrief + NoveltyAuditReport
        from the Spec Store by their refs (provenance — NOT the payload), run
        one model_core.model_step, persist the resulting ProblemModel, and
        publish problem.modeled.
        按 ref 从 Spec Store 解析 ResearchSpec + LiteratureBrief +
        NoveltyAuditReport（溯源——不读 payload），跑一次 model_core.model_step，
        持久化产出的 ProblemModel，并发布 problem.modeled。"""
        session_id = env.session_id
        causation_id = env.event_id
        payload = env.payload or {}

        try:
            research_spec = self.spec_store.get(payload.get("research_spec_ref"))
            literature_brief = self.spec_store.get(payload.get("literature_brief_ref"))
            # 3.3: the literature-novelty audit now runs AFTER the method, so on the first pass no
            # NoveltyAuditReport exists yet -- load it only when a ref is present (was an
            # unconditional get(None) crash after the audit was moved later).
            novelty_ref = payload.get("novelty_audit_report_ref")
            novelty_audit_report = self.spec_store.get(novelty_ref) if novelty_ref else None
            # meeting 3.4: a re-model triggered by the human injecting a redirection at the
            # checkpoint carries human_idea_seed_ref -- resolve it (best-effort) and honor it so the
            # re-derived direction actually incorporates the human's idea (else it'd reproduce the
            # old direction and Fix C would skip re-confirming it).
            human_idea = None
            seed_ref = payload.get("human_idea_seed_ref")
            if seed_ref:
                try:
                    seed = self.spec_store.get(seed_ref)
                    human_idea = (seed or {}).get("idea") if isinstance(seed, dict) else None
                except Exception:  # noqa: BLE001 -- injection is best-effort, never crash modeling
                    human_idea = None
            problem_model = await asyncio.to_thread(
                model_core.model_step,
                research_spec,
                literature_brief,
                novelty_audit_report,
                self.llm_client,
                prompt_patch=scoped_prompt_patch(self.spec_store, (env.payload or {}).get("moss_intervention_ref"),
                        session_id, "ProblemModelingAgent", self.config.prompt_patch()),
                human_idea=human_idea,
            )
        except model_core.ModelError as exc:
            await self._fail(client, session_id, causation_id, exc.classification, exc.code, str(exc))
            return
        except Exception as exc:  # defensive: never crash the handler
            await self._fail(
                client, session_id, causation_id, "transient", "unexpected_error",
                f"{type(exc).__name__}: {exc}",
            )
            return

        problem_model_ref = self.spec_store.put(
            kind="problem_model",
            session_id=session_id,
            key=_PROBLEM_MODEL_KEY,
            obj=problem_model,
            emitter=AGENT_NAME,
        )

        await client.publish(
            event_type="problem.modeled",
            session_id=session_id,
            payload={"problem_model_ref": problem_model_ref},
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


__all__ = ["ProblemModelingAgentHandler", "SCHEMA_VERSION"]
