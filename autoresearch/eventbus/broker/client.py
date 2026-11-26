"""
EventBusClient: the main entry point for any process using the AutoResearch EventBus.

Each process instantiates one client:

    client = EventBusClient(
        redis_url="redis://localhost:6381",
        sqlite_path="/tmp/autoresearch-coder.db",
        emitter_name="CoderAgent",
        eventbus_root=Path(__file__).resolve().parents[1],  # .../autoresearch/eventbus
    )
    await client.start()

    # register handlers
    client.subscribe("coder.*_requested", handle_coder_command)

    # publish events
    await client.publish(
        event_type="coder.module_implemented",
        session_id="01890abc-def0-7000-8000-...",
        payload={"module_id": "...", "code_ref": "spec://code/..."},
    )

    # block until shutdown
    await client.run_forever()
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Awaitable, Callable, Optional

import aiosqlite
from redis.asyncio import Redis
from redis.exceptions import RedisError

from .dispatch import Dispatcher, HandlerType
from .discovery import SessionDiscovery
from .envelope import Envelope, EnvelopeBuilder, EnvelopeValidator
from .idempotency import (
    get_last_redis_id,
    has_processed,
    init_idempotency,
    record_processed,
)
from .outbox import OutboxWorker, init_outbox, write_outbox
from .registry import Registry
from .whitelist import PolicyViolation, check_emitter

logger = logging.getLogger(__name__)


class EventBusClient:
    """Single-instance EventBus client embedded in a subscriber/emitter process."""

    def __init__(
        self,
        redis_url: str,
        sqlite_path: str,
        emitter_name: str,
        eventbus_root: Path,
        consumer_group: Optional[str] = None,
        consumer_name: Optional[str] = None,
        session_discovery_interval_sec: float = 5.0,
    ):
        self.redis_url = redis_url
        self.sqlite_path = sqlite_path
        self.emitter_name = emitter_name
        self.eventbus_root = Path(eventbus_root)
        self.consumer_group = consumer_group or emitter_name
        self.consumer_name = consumer_name or f"{emitter_name}-{uuid.uuid4().hex[:8]}"
        self.session_discovery_interval_sec = session_discovery_interval_sec

        # Wired in start()
        self.registry: Optional[Registry] = None
        self.envelope_builder = EnvelopeBuilder(emitter=emitter_name)
        self.dispatcher = Dispatcher()
        self.redis: Optional[Redis] = None
        self.db: Optional[aiosqlite.Connection] = None
        self._outbox_worker: Optional[OutboxWorker] = None
        self._discovery: Optional[SessionDiscovery] = None
        self._subscribe_task: Optional[asyncio.Task] = None
        self._stop_event = asyncio.Event()
        self._known_streams: set[str] = set()
        self._streams_lock = asyncio.Lock()
        self._failed_deliveries: dict[tuple[str, str], dict] = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Initialize all resources. Must be called before publish/subscribe loop."""
        self.registry = Registry(self.eventbus_root / "eventbus-registry.yaml")

        self.db = await aiosqlite.connect(self.sqlite_path)
        # Better concurrency with WAL mode
        await self.db.execute("PRAGMA journal_mode=WAL")
        await init_outbox(self.db)
        await init_idempotency(self.db)

        self.redis = Redis.from_url(self.redis_url, decode_responses=True)
        await self.redis.ping()

        # Outbox worker
        async def xadd_cb(stream_key: str, envelope_json: str) -> None:
            await self.redis.xadd(stream_key, {"data": envelope_json}, maxlen=10000, approximate=True)

        self._outbox_worker = OutboxWorker(self.db, xadd_cb)
        self._outbox_worker.start()

        # Discovery
        async def on_new_stream(stream_key: str) -> None:
            async with self._streams_lock:
                self._known_streams.add(stream_key)
            # Ensure consumer group exists for this stream
            try:
                await self.redis.xgroup_create(
                    stream_key, self.consumer_group, id="0", mkstream=True
                )
                logger.info(
                    "Created consumer group %s on %s", self.consumer_group, stream_key
                )
            except RedisError as e:
                # BUSYGROUP = already exists, fine
                if "BUSYGROUP" not in str(e):
                    raise

        self._discovery = SessionDiscovery(
            self.redis,
            on_new_stream,
            interval_sec=self.session_discovery_interval_sec,
        )
        self._discovery.start()
        # Block until first scan is done so subsequent publishes / subscribes see existing streams
        await self._discovery.scan_now()

    async def stop(self) -> None:
        self._stop_event.set()
        if self._subscribe_task is not None:
            await self._subscribe_task
            self._subscribe_task = None
        if self._discovery is not None:
            await self._discovery.stop()
        if self._outbox_worker is not None:
            await self._outbox_worker.stop()
        if self.redis is not None:
            await self.redis.aclose()
        if self.db is not None:
            await self.db.close()

    # ------------------------------------------------------------------
    # Publish (emitter side)
    # ------------------------------------------------------------------

    async def publish(
        self,
        *,
        event_type: str,
        session_id: str,
        payload: dict,
        cost_delta: str = "0",
        causation_id: Optional[str] = None,
        auto_commit: bool = True,
        event_id: Optional[str] = None,
    ) -> str:
        """
        Validate, build envelope, write to outbox. Outbox worker will XADD asynchronously.

        Returns the event_id (UUID).

        Args:
            auto_commit: if True (default), the outbox row is committed immediately.
                If False, the caller is responsible for calling self.db.commit().
                Used by the ResearchOrchestrator to bundle SessionState + outbox writes in one
                atomic transaction (INV-4).

        Raises:
            PolicyViolation: if this emitter is not in the whitelist for event_type
            ValueError: if envelope or payload fails schema validation
        """
        return await self._publish_inner(
            event_type=event_type,
            session_id=session_id,
            payload=payload,
            cost_delta=cost_delta,
            causation_id=causation_id,
            emitter_override=None,
            auto_commit=auto_commit,
            event_id=event_id,
        )

    async def _publish_inner(
        self,
        *,
        event_type: str,
        session_id: str,
        payload: dict,
        cost_delta: str,
        causation_id: Optional[str],
        emitter_override: Optional[str],
        auto_commit: bool = True,
        event_id: Optional[str] = None,
    ) -> str:
        emitter = emitter_override or self.emitter_name

        # 1. Emitter whitelist check (raises PolicyViolation if not allowed)
        try:
            check_emitter(self.registry, event_type, emitter)
        except PolicyViolation as pv:
            # Convert to synthetic agent.failed event under 'system' emitter.
            # Re-emit recursively, but with emitter='system'.
            if emitter != "system":
                try:
                    await self._emit_policy_violation(pv, session_id)
                except Exception as e:
                    logger.error("Failed to emit policy violation: %s", e)
            raise

        # 2. Build envelope
        env = Envelope(
            event_id=event_id or str(uuid.uuid4()),
            session_id=session_id,
            event_type=event_type,
            emitter=emitter,
            timestamp=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            schema_version=1,
            cost_delta=cost_delta,
            payload=payload,
            causation_id=causation_id,
        )

        # 3. Validate envelope shape
        env_dict = env.to_dict()
        env_errors = list(self.registry.envelope_validator.iter_errors(env_dict))
        if env_errors:
            raise ValueError(
                f"Envelope validation failed for {event_type}: {env_errors[0].message}"
            )

        # 4. Validate payload against the per-event schema
        payload_validator = self.registry.get_payload_validator(event_type, schema_version=1)
        payload_errors = list(payload_validator.iter_errors(payload))
        if payload_errors:
            raise ValueError(
                f"Payload validation failed for {event_type}: {payload_errors[0].message}"
            )

        # 5. Write to outbox (atomic with any caller transaction)
        stream_key = f"events:{session_id}"
        await write_outbox(self.db, stream_key, env.to_json())
        if auto_commit:
            await self.db.commit()

        logger.debug("Publish enqueued: %s emitter=%s session=%s id=%s",
                     event_type, emitter, session_id, env.event_id)
        return env.event_id

    async def _emit_policy_violation(self, pv: PolicyViolation, session_id: str) -> None:
        """Emit agent.failed(policy_violation) as system emitter."""
        agent_failed_payload = {
            "agent_name": pv.offending_emitter if pv.offending_emitter in (
                "ProblemModelingAgent", "DeepResearchAgent", "KnowledgeIndexAgent",
                "NoveltyAuditAgent", "HypothesisAgent", "CriticAgent", "EvaluatorAgent",
                "ExperimentPlannerAgent", "CoderAgent", "RunnerAgent", "ReportAgent", "MOSS",
            ) else "CoderAgent",  # fallback when emitter is unknown
            "reason_classification": "policy_violation",
            "underlying_reason": str(pv),
            "violation_type": pv.violation_type,
        }
        await self._publish_inner(
            event_type="agent.failed",
            session_id=session_id,
            payload=agent_failed_payload,
            cost_delta="0",
            causation_id=None,
            emitter_override="system",
        )

    # ------------------------------------------------------------------
    # Subscribe (handler registration + main loop)
    # ------------------------------------------------------------------

    def subscribe(self, pattern: str, handler: HandlerType) -> None:
        """Register a handler for events whose event_type matches pattern (fnmatch)."""
        self.dispatcher.register(pattern, handler)

    async def run_forever(self) -> None:
        """Block, draining new events from Redis and dispatching to handlers."""
        self._subscribe_task = asyncio.create_task(self._consume_loop())
        await self._subscribe_task

    async def _consume_loop(self) -> None:
        """Loop reading events from all known streams via XREADGROUP."""
        while not self._stop_event.is_set():
            async with self._streams_lock:
                streams_snapshot = list(self._known_streams)
            if not streams_snapshot:
                try:
                    await asyncio.wait_for(self._stop_event.wait(), timeout=0.5)
                except asyncio.TimeoutError:
                    pass
                continue

            # XREADGROUP with '>' reads new messages not yet delivered to this consumer
            stream_keys = {s: ">" for s in streams_snapshot}
            try:
                # Retry only deliveries whose own handler returned an error. Idle
                # time does not establish that another consumer's training ended.
                for (stream_key, redis_id), fields in list(self._failed_deliveries.items()):
                    try:
                        await self._handle_one(stream_key, redis_id, fields)
                        self._failed_deliveries.pop((stream_key, redis_id), None)
                    except Exception:
                        logger.exception("Pending event delivery failed: %s", redis_id)
                # block=1000 ms so we wake periodically to re-check known_streams
                results = await self.redis.xreadgroup(
                    self.consumer_group,
                    self.consumer_name,
                    stream_keys,
                    count=100,
                    block=1000,
                )
            except RedisError as e:
                logger.warning("XREADGROUP error (will retry): %s", e)
                # 自愈:多流 XREADGROUP 原子失败——一条 key/组已消失的流(NOGROUP)会让整条
                # 命令报错,而 _known_streams 只增不减 → 整个循环永久锁死(一条死流饿死所有活
                # 流)。出错路径上剔除"key 已消失"的死流,下一轮只读活流即可恢复。活流不触发此
                # 路径;discovery 会重新发现真正复活的流。
                await self._prune_vanished_streams()
                await asyncio.sleep(1)
                continue

            if not results:
                continue

            for stream_key, entries in results:
                for redis_id, fields in entries:
                    try:
                        await self._handle_one(stream_key, redis_id, fields)
                    except Exception:
                        self._failed_deliveries[(stream_key, redis_id)] = fields
                        logger.exception("Event delivery failed; leaving %s pending", redis_id)

    async def _prune_vanished_streams(self) -> None:
        """Drop known streams whose Redis key no longer exists (NOGROUP source). Only runs
        on the consume-loop error path; only removes truly-gone keys (live streams have an
        existing key and are kept), so the multi-session discovery semantics are unchanged."""
        async with self._streams_lock:
            known = list(self._known_streams)
        for stream_key in known:
            try:
                if not await self.redis.exists(stream_key):
                    async with self._streams_lock:
                        self._known_streams.discard(stream_key)
                    logger.info("eventbus: pruned vanished stream %s (self-heal)", stream_key)
            except RedisError:
                # healing must never throw out of the consume loop
                continue

    async def _handle_one(self, stream_key: str, redis_id: str, fields: dict) -> None:
        try:
            env = Envelope.from_redis_fields(fields)
        except Exception as e:
            logger.error("Bad envelope in stream %s id %s: %s", stream_key, redis_id, e)
            # Ack to drop poison message; in production would dead-letter
            try:
                await self.redis.xack(stream_key, self.consumer_group, redis_id)
            except RedisError:
                pass
            return

        # ── inbound envelope 契约校验闸:入站用同一套 envelope schema 校验,
        # 绕过 publish() 直接 xadd 进流的不合法事件(如显示回放用的合成事件)
        # 直接 ack 丢弃、不派给 handler。真实事件出站时已校验过 → 必然通过,
        # 真实工作一个不漏。
        try:
            inbound_errors = list(self.registry.envelope_validator.iter_errors(env.to_dict()))
        except Exception as e:  # noqa: BLE001
            # fail-open:校验器自身异常时记日志后【照常 dispatch】,绝不静默吞掉真实事件,也绝不
            # 让异常带崩 _consume_loop(_handle_one 的调用点无 try → 抛出会终结抽流任务,正是本
            # 文件刚修过的那类循环锁死)。与上方 from_redis_fields 分支同样的防御姿态。
            logger.error(
                "eventbus: inbound validation crashed for %s id=%s: %s (fail-open → dispatch)",
                stream_key, getattr(env, "event_id", "?"), e,
            )
            inbound_errors = []
        if inbound_errors:
            logger.warning(
                "eventbus: skip malformed inbound event in %s id=%s: %s",
                stream_key, env.event_id, inbound_errors[0].message,
            )
            try:
                await self.redis.xack(stream_key, self.consumer_group, redis_id)
            except RedisError:
                pass
            return

        # Idempotency check
        if await has_processed(self.db, self.emitter_name, env.event_id):
            logger.debug("Skipping already-processed event %s", env.event_id)
            await self.redis.xack(stream_key, self.consumer_group, redis_id)
            return

        # Persist each successful subscriber separately: retrying a failed sibling
        # must not repeat an already completed training/measurement handler.
        for index, handler in enumerate(self.dispatcher.matching_handlers(env.event_type)):
            subscriber = f"{self.emitter_name}:handler:{env.event_type}:{index}"
            if await has_processed(self.db, subscriber, env.event_id):
                continue
            try:
                await handler(env)
            except Exception:
                await self.db.rollback()
                raise
            await record_processed(
                self.db, subscriber_name=subscriber, business_event_id=env.event_id,
                stream_key=stream_key, redis_stream_id=redis_id)
            await self.db.commit()

        # Record processed + ack
        await record_processed(
            self.db,
            subscriber_name=self.emitter_name,
            business_event_id=env.event_id,
            stream_key=stream_key,
            redis_stream_id=redis_id,
        )
        await self.db.commit()
        try:
            await self.redis.xack(stream_key, self.consumer_group, redis_id)
        except RedisError as e:
            logger.warning("XACK failed for %s: %s (will be redelivered, idempotency handles)", redis_id, e)
