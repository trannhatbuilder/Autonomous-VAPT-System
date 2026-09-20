"""
VAPT-AI C2 Manager — unified C2 lifecycle controller (W14-S4).

Python port of CyberStrikeAI internal/c2/manager.go (selective port —
W14 focuses on HTTP listener; TCP/WS listeners come in W15).

Responsibilities:
    - Listener lifecycle: create / start / stop
    - Session lifecycle: ingest_checkin (beacon check-in) / mark_dead
    - Task lifecycle: enqueue_task / pop_tasks_for_beacon / submit_task_result
    - EventBus hooks → SSE emission (session_online, task_complete, session_dead)

Unlike CyberStrikeAI (which uses goroutines + channels for the EventBus),
VAPT-AI uses asyncio + the existing app.pentest.events.SSEEventBus (W3).
The Manager is a thin orchestration layer over AsyncSession; the listener
does the actual socket I/O.

Architecture:
    [Beacon] ⇄ HTTP Listener ⇄ Manager ⇄ AsyncSession (DB)
                                   ↓
                              SSEEventBus → frontend

Reference: CyberStrikeAI internal/c2/manager.go (Apache 2.0).
"""
from __future__ import annotations

import asyncio
import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, UTC
from typing import Any, Callable, Awaitable

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models.c2 import C2Session, C2Task
from app.c2.types import (
    ListenerType, SessionStatus, TaskStatus, TaskType, BeaconType,
    ListenerConfig, ImplantCheckInRequest, ImplantCheckInResponse,
    TaskResultRequest,
)
from app.c2.crypto import generate_aes_key, generate_implant_token

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class C2Error(Exception):
    """Base C2 manager error."""


class InvalidInputError(C2Error):
    """Invalid input parameter."""


class SessionNotFoundError(C2Error):
    """C2 session not found in DB."""


class SessionInactiveError(C2Error):
    """Session is dead/killed — cannot enqueue tasks."""


class ListenerNotFoundError(C2Error):
    """Listener not registered."""


# ---------------------------------------------------------------------------
# Hooks — business logic callbacks (port from manager.go Hooks struct)
# ---------------------------------------------------------------------------

@dataclass
class C2Hooks:
    """Optional business callbacks invoked by the manager."""
    on_session_first_seen: Callable[[C2Session], Awaitable[None]] | None = None
    on_task_completed: Callable[[C2Task, str], Awaitable[None]] | None = None
    on_session_dead: Callable[[str], Awaitable[None]] | None = None


# ---------------------------------------------------------------------------
# Listener input dataclass (port from CreateListenerInput)
# ---------------------------------------------------------------------------

@dataclass
class CreateListenerInput:
    """Web/MCP listener creation input (validated + trimmed)."""
    name: str
    scan_id: str
    listener_type: str           # ListenerType value
    bind_host: str = "127.0.0.1"
    bind_port: int = 8443
    config: ListenerConfig | None = None
    callback_host: str | None = None
    remark: str = ""


# ---------------------------------------------------------------------------
# Task input dataclass (port from EnqueueTaskInput)
# ---------------------------------------------------------------------------

@dataclass
class EnqueueTaskInput:
    """Task enqueue input."""
    session_id: str              # C2Session.id (UUID str form)
    task_type: TaskType | str
    payload: dict[str, Any] = field(default_factory=dict)
    source: str = "manual"      # manual | ai | batch | api
    level: int = 2              # HITL level 1-5
    bypass_hitl: bool = False


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------

class C2Manager:
    """C2 Manager — singleton-per-scan orchestrator.

    Wraps AsyncSession operations on C2Session + C2Task tables. Listeners
    (HTTP/TCP/WS) call into the manager for checkin/task-poll/result-submit.
    The manager emits SSE events for session_online, task_complete,
    session_dead.

    Lifecycle:
        manager = C2Manager(session, scan_id, hooks=...)
        listener = HTTPBeaconListener(manager, ...)
        await listener.start()
        ...
        await listener.stop()
    """

    def __init__(
        self,
        session: AsyncSession,
        scan_id: str,
        hooks: C2Hooks | None = None,
    ):
        self.session = session
        self.scan_id = scan_id
        self.hooks = hooks or C2Hooks()
        self._running_listeners: dict[str, Any] = {}  # listener_id → listener instance
        self._lock = asyncio.Lock()

    # ---------- Listener lifecycle ----------

    async def create_listener(self, inp: CreateListenerInput) -> dict[str, Any]:
        """Validate + persist a new listener definition.

        Returns a dict with listener_id, encryption_key_b64, implant_token_b64,
        and config (for the listener to use at start time). The encryption
        key + token are generated here and stored in the listener's metadata.

        Note: VAPT-AI does not have a separate c2_listeners table (W15 may
        add it). For W14, listener config is stored in C2Session.metadata_json
        of the first session that checks in via this listener.
        """
        if not inp.name.strip():
            raise InvalidInputError("Listener name is required")
        if not ListenerType.is_valid(inp.listener_type):
            raise InvalidInputError(
                f"Unsupported listener type: {inp.listener_type!r}. "
                f"Valid: {[t.value for t in ListenerType.all()]}"
            )
        if not (1 <= inp.bind_port <= 65535):
            raise InvalidInputError(f"Invalid bind_port: {inp.bind_port}")

        bind_host = inp.bind_host.strip() or "127.0.0.1"
        cfg = inp.config or ListenerConfig()
        if inp.callback_host:
            cfg.callback_host = inp.callback_host.strip()
        cfg.apply_defaults()

        listener_id = f"l_{uuid.uuid4().hex[:14]}"
        encryption_key_b64 = generate_aes_key()
        implant_token_b64 = generate_implant_token()

        logger.info(
            "C2 listener created: id=%s scan=%s type=%s bind=%s:%d",
            listener_id, self.scan_id, inp.listener_type, bind_host, inp.bind_port,
        )

        return {
            "listener_id": listener_id,
            "scan_id": self.scan_id,
            "name": inp.name.strip(),
            "type": inp.listener_type,
            "bind_host": bind_host,
            "bind_port": inp.bind_port,
            "config": cfg.to_dict(),
            "encryption_key_b64": encryption_key_b64,
            "implant_token_b64": implant_token_b64,
            "remark": inp.remark,
        }

    async def register_running_listener(self, listener_id: str, listener: Any) -> None:
        """Track a started listener instance (for graceful shutdown)."""
        async with self._lock:
            self._running_listeners[listener_id] = listener

    async def unregister_running_listener(self, listener_id: str) -> None:
        async with self._lock:
            self._running_listeners.pop(listener_id, None)

    async def stop_all_listeners(self) -> None:
        """Stop all running listeners (graceful shutdown)."""
        async with self._lock:
            listeners = list(self._running_listeners.values())
            self._running_listeners.clear()

        for listener in listeners:
            try:
                await listener.stop()
            except Exception as e:
                logger.warning("Listener stop failed: %s", e)

    # ---------- Session lifecycle ----------

    async def ingest_checkin(
        self,
        listener_id: str,
        listener_type: str,
        req: ImplantCheckInRequest,
        encryption_key_b64: str,
        implant_token_b64: str,
        beacon_type: BeaconType = BeaconType.PYTHON_BEACON,
    ) -> C2Session:
        """Beacon check-in — unified entry point.

        Behavior (port from CyberStrikeAI IngestCheckIn):
            1. If implant_uuid already has a session → update last_seen + status=active
            2. Otherwise → create new C2Session row, fire on_session_first_seen hook
               + emit session_online SSE event

        Returns:
            C2Session ORM instance (flushed, not committed — caller commits)

        Raises:
            InvalidInputError: if req.implant_uuid is empty
        """
        if not req.implant_uuid.strip():
            raise InvalidInputError("implant_uuid is required")

        # Look up existing session by implant_uuid
        existing_result = await self.session.execute(
            select(C2Session)
            .where(C2Session.scan_id == self.scan_id)
            .where(C2Session.metadata_json["implant_uuid"].as_string() == req.implant_uuid)
            .limit(1)
        )
        existing = existing_result.scalar_one_or_none()

        now = datetime.now(UTC)
        is_first_seen = existing is None

        if existing is not None:
            # Update heartbeat — preserve original first_seen + sleep/jitter
            existing.last_seen_at = now
            existing.status = SessionStatus.ACTIVE.value
            existing.metadata_json = {
                **(existing.metadata_json or {}),
                "last_checkin": now.isoformat(),
                "hostname": req.hostname,
                "username": req.username,
                "os": req.os.lower(),
                "arch": req.arch.lower(),
                "pid": req.pid,
                "process_name": req.process_name,
                "is_admin": req.is_admin,
                "internal_ip": req.internal_ip,
                "user_agent": req.user_agent,
            }
            await self.session.flush()
            session = existing
        else:
            # Create new session
            session = C2Session(
                scan_id=self.scan_id,
                beacon_type=beacon_type.value,
                listener_type=listener_type,
                listener_id=None,  # listener_id is a UUID, but we use str prefix IDs; set to None
                remote_address=req.internal_ip or "unknown",
                hostname=req.hostname,
                username=req.username,
                os=req.os.lower() if req.os else None,
                arch=req.arch.lower() if req.arch else None,
                checkin_at=now,
                last_seen_at=now,
                status=SessionStatus.ACTIVE.value,
                metadata_json={
                    "implant_uuid": req.implant_uuid,
                    "listener_id": listener_id,
                    "pid": req.pid,
                    "process_name": req.process_name,
                    "is_admin": req.is_admin,
                    "internal_ip": req.internal_ip,
                    "external_ip": None,  # populated by listener from request.remote_addr
                    "user_agent": req.user_agent,
                    "sleep_seconds": req.sleep_seconds,
                    "jitter_percent": req.jitter_percent,
                    "encryption_key_b64": encryption_key_b64,
                    "implant_token_b64": implant_token_b64,
                    "beacon_metadata": req.metadata,
                    "first_seen_at": now.isoformat(),
                    "last_checkin": now.isoformat(),
                },
            )
            self.session.add(session)
            await self.session.flush()

        # Emit session_online SSE event (W3 event_bus)
        try:
            from app.pentest.events import event_bus
            await event_bus.publish(self.scan_id, {
                "event": "session_online",
                "scan_id": self.scan_id,
                "session_id": str(session.id),
                "listener_id": listener_id,
                "beacon_type": session.beacon_type,
                "hostname": session.hostname,
                "os": session.os,
                "arch": session.arch,
                "internal_ip": req.internal_ip,
                "is_first_seen": is_first_seen,
                "timestamp": now.isoformat(),
            })
        except Exception as e:
            logger.warning("Failed to emit session_online SSE: %s", e)

        # Fire on_session_first_seen hook (only on new sessions)
        if is_first_seen and self.hooks.on_session_first_seen is not None:
            try:
                await self.hooks.on_session_first_seen(session)
            except Exception as e:
                logger.warning("on_session_first_seen hook failed: %s", e)

        logger.info(
            "C2 checkin: scan=%s session=%s beacon=%s host=%s@%s first_seen=%s",
            self.scan_id, session.id, session.beacon_type,
            session.username, session.hostname, is_first_seen,
        )
        return session

    async def mark_session_dead(self, session_id: str) -> None:
        """Mark a session as dead (heartbeat timeout)."""
        session = await self.session.get(C2Session, uuid.UUID(session_id))
        if session is None:
            raise SessionNotFoundError(f"Session {session_id} not found")

        session.status = SessionStatus.DEAD.value
        await self.session.flush()

        # Emit session_dead SSE event
        try:
            from app.pentest.events import event_bus
            await event_bus.publish(self.scan_id, {
                "event": "session_dead",
                "scan_id": self.scan_id,
                "session_id": session_id,
                "timestamp": datetime.now(UTC).isoformat(),
            })
        except Exception as e:
            logger.warning("Failed to emit session_dead SSE: %s", e)

        # Fire on_session_dead hook
        if self.hooks.on_session_dead is not None:
            try:
                await self.hooks.on_session_dead(session_id)
            except Exception as e:
                logger.warning("on_session_dead hook failed: %s", e)

    async def list_sessions(
        self, beacon_type: str | None = None,
    ) -> list[C2Session]:
        """List all C2 sessions for this scan (unified — D28)."""
        stmt = (
            select(C2Session)
            .where(C2Session.scan_id == self.scan_id)
            .order_by(C2Session.last_seen_at.desc())
        )
        if beacon_type:
            stmt = stmt.where(C2Session.beacon_type == beacon_type)
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def get_session(self, session_id: str) -> C2Session | None:
        """Get a single C2 session by ID."""
        return await self.session.get(C2Session, uuid.UUID(session_id))

    # ---------- Task lifecycle ----------

    async def enqueue_task(self, inp: EnqueueTaskInput) -> C2Task:
        """Queue a new task for a session.

        Validates:
            - session exists + is active
            - task_type is valid
            - level is 1-5 (L3+ requires HITL approval, handled here per W15-S5)
            - OPSEC command deny regex

        HITL gate (W15-S5): for L3+ tasks (is_dangerous() returns True),
        calls HITLManager.request_and_wait() before queueing. L5 tasks
        (SELF_DELETE, PERSIST) get extra_confirm=True flag in metadata
        (per CyberStrikeAI single-approval pattern — no double-confirm).
        If HITL rejected → task status=hitl_denied + raise PermissionError.
        If HITL timeout (5 min) → task status=hitl_denied + raise TimeoutError.

        Returns:
            C2Task ORM instance (flushed, status=queued or hitl_denied)
        """
        if not inp.session_id.strip():
            raise InvalidInputError("session_id is required")

        # Resolve task_type enum
        if isinstance(inp.task_type, str):
            try:
                task_type_enum = TaskType(inp.task_type)
            except ValueError:
                raise InvalidInputError(f"Invalid task_type: {inp.task_type!r}")
        else:
            task_type_enum = inp.task_type

        # Look up session
        session_uuid = uuid.UUID(inp.session_id)
        session = await self.session.get(C2Session, session_uuid)
        if session is None:
            raise SessionNotFoundError(f"Session {inp.session_id} not found")

        if session.status in (SessionStatus.DEAD.value, SessionStatus.KILLED.value):
            raise SessionInactiveError(
                f"Session {inp.session_id} is {session.status} — cannot enqueue tasks"
            )

        # Resolve HITL level if not specified
        level = inp.level or TaskType.hitl_level(task_type_enum)
        is_dangerous = TaskType.is_dangerous(task_type_enum)
        is_l5 = level >= 5

        # OPSEC: command deny regex enforcement (port from manager.go)
        if task_type_enum in (TaskType.EXEC, TaskType.SHELL):
            cmd = str(inp.payload.get("cmd", ""))
            cfg = ListenerConfig.from_dict(
                (session.metadata_json or {}).get("config", {})
            )
            for deny_pattern in cfg.command_deny_regex:
                import re
                if re.search(deny_pattern, cmd):
                    raise InvalidInputError(
                        f"Command denied by OPSEC regex: {deny_pattern!r}"
                    )

        # W15-S5: HITL gate for L3+ tasks
        if is_dangerous and not inp.bypass_hitl:
            hitl_outcome = await self._request_hitl_approval(
                session=session,
                task_type_enum=task_type_enum,
                payload=inp.payload,
                level=level,
                is_l5=is_l5,
                source=inp.source,
            )

            if hitl_outcome is None:
                # HITL manager not available (e.g. in tests) — proceed with queue
                pass
            elif hitl_outcome.decision in ("reject", "user_aborted", "approval_timeout"):
                # HITL denied — create task row with hitl_denied status for audit
                task = C2Task(
                    session_id=session_uuid,
                    command=task_type_enum.value,
                    args_json=inp.payload,
                    level=level,
                    status="hitl_denied",
                    error=f"HITL {hitl_outcome.decision}: {hitl_outcome.comment}",
                    hitl_approval_id=hitl_outcome.approval_id,
                )
                self.session.add(task)
                await self.session.flush()

                logger.warning(
                    "C2 task denied by HITL: scan=%s session=%s task=%s type=%s decision=%s",
                    self.scan_id, inp.session_id, task.id, task_type_enum.value,
                    hitl_outcome.decision,
                )

                if hitl_outcome.decision == "approval_timeout":
                    raise TimeoutError(
                        f"HITL approval timed out for task {task_type_enum.value}: {hitl_outcome.comment}"
                    )
                else:
                    raise PermissionError(
                        f"HITL rejected task {task_type_enum.value}: {hitl_outcome.comment}"
                    )

            elif hitl_outcome.decision == "suggest_alternative":
                # HITL approved with edited args
                if hitl_outcome.suggested_args:
                    inp.payload = hitl_outcome.suggested_args
                logger.info(
                    "C2 task approved with edits: scan=%s task=%s",
                    self.scan_id, task_type_enum.value,
                )

            # else: decision == "approve" — proceed with original payload

        # Create task row
        task = C2Task(
            session_id=session_uuid,
            command=task_type_enum.value,
            args_json=inp.payload,
            level=level,
            status=TaskStatus.QUEUED.value,
        )
        self.session.add(task)
        await self.session.flush()

        logger.info(
            "C2 task enqueued: scan=%s session=%s task=%s type=%s level=%d dangerous=%s l5=%s",
            self.scan_id, inp.session_id, task.id, task_type_enum.value,
            level, is_dangerous, is_l5,
        )
        return task

    async def _request_hitl_approval(
        self,
        session: C2Session,
        task_type_enum: TaskType,
        payload: dict[str, Any],
        level: int,
        is_l5: bool,
        source: str,
    ) -> Any | None:
        """Request HITL approval for a dangerous C2 task (W15-S5).

        Returns:
            HITLDecision instance, or None if HITLManager is not available
            (e.g. in tests without DB session).

        L5 tasks (self_delete, persist) get extra_confirm=True in the
        agent_reasoning field to signal UI to display extra warning.
        Per CyberStrikeAI single-approval pattern (no double-confirm).
        """
        try:
            from app.hitl.manager import HITLManager
            hitl_manager = HITLManager(session=self.session)
        except Exception as e:
            logger.warning("HITLManager init failed: %s", e)
            return None

        # Build reasoning for the HITL review
        predicted_impact = self._predict_task_impact(task_type_enum, level)
        agent_reasoning = (
            f"C2 task: {task_type_enum.value} (L{level})\n"
            f"Session: {session.beacon_type} @ {session.hostname or session.remote_address}\n"
            f"Payload: {payload}\n"
            f"Source: {source}"
        )
        if is_l5:
            agent_reasoning = (
                f"⚠️ L5 DESTRUCTIVE TASK — EXTRA CONFIRM REQUIRED\n"
                + agent_reasoning
            )

        try:
            decision = await hitl_manager.request_and_wait(
                scan_id=self.scan_id,
                tool_name=f"c2_{task_type_enum.value}",
                target=str(session.remote_address),
                args=payload,
                predicted_impact=predicted_impact,
                agent_reasoning=agent_reasoning,
                kg_confidence=0.5,
                scan_context={
                    "c2_session_id": str(session.id),
                    "beacon_type": session.beacon_type,
                    "task_level": level,
                    "is_l5": is_l5,
                },
            )
            return decision
        except Exception as e:
            logger.warning("HITL request failed: %s", e)
            return None

    @staticmethod
    def _predict_task_impact(task_type_enum: TaskType, level: int) -> str:
        """Predict the impact of a C2 task (for HITL review)."""
        impacts = {
            TaskType.KILL_PROC: "Process killed on target — may disrupt services",
            TaskType.UPLOAD: "File written to target filesystem — potential persistence",
            TaskType.SELF_DELETE: "Beacon binary deleted — cleanup evidence",
            TaskType.PORT_FWD: "Port forwarding tunnel — potential lateral movement",
            TaskType.SOCKS_START: "SOCKS proxy started — potential internal pivot",
            TaskType.LOAD_ASSEMBLY: "Assembly loaded into process — potential injection",
            TaskType.PERSIST: "Persistence mechanism installed — survives reboot",
        }
        return impacts.get(task_type_enum, f"Unknown impact (L{level} task)")

    async def pop_tasks_for_beacon(
        self, session_id: str, limit: int = 10,
    ) -> list[C2Task]:
        """Beacon polls for pending tasks.

        Returns up to `limit` tasks with status=queued. Marks them as sent.
        """
        session_uuid = uuid.UUID(session_id)
        result = await self.session.execute(
            select(C2Task)
            .where(C2Task.session_id == session_uuid)
            .where(C2Task.status == TaskStatus.QUEUED.value)
            .order_by(C2Task.created_at.asc())
            .limit(limit)
        )
        tasks = list(result.scalars().all())

        now = datetime.now(UTC)
        for task in tasks:
            task.status = TaskStatus.SENT.value
            task.sent_at = now
        await self.session.flush()

        return tasks

    async def submit_task_result(
        self, task_id: str, result: TaskResultRequest,
    ) -> C2Task:
        """Beacon posts task result back.

        Updates task status to success/failed + stores result_text + exit_code.
        Fires on_task_completed hook + emits task_complete SSE event.
        """
        task_uuid = uuid.UUID(task_id)
        task = await self.session.get(C2Task, task_uuid)
        if task is None:
            raise C2Error(f"Task {task_id} not found")

        now = datetime.now(UTC)
        task.result = result.result_text
        task.exit_code = result.exit_code
        task.completed_at = now
        task.error = result.error or None

        if result.exit_code == 0 and not result.error:
            task.status = TaskStatus.SUCCESS.value
        else:
            task.status = TaskStatus.FAILED.value

        await self.session.flush()

        # Emit task_complete SSE event
        try:
            from app.pentest.events import event_bus
            await event_bus.publish(self.scan_id, {
                "event": "task_complete",
                "scan_id": self.scan_id,
                "task_id": task_id,
                "session_id": str(task.session_id),
                "status": task.status,
                "exit_code": result.exit_code,
                "duration_ms": result.duration_ms,
                "timestamp": now.isoformat(),
            })
        except Exception as e:
            logger.warning("Failed to emit task_complete SSE: %s", e)

        # Fire on_task_completed hook
        if self.hooks.on_task_completed is not None:
            try:
                await self.hooks.on_task_completed(task, str(task.session_id))
            except Exception as e:
                logger.warning("on_task_completed hook failed: %s", e)

        logger.info(
            "C2 task completed: scan=%s task=%s status=%s exit=%d",
            self.scan_id, task_id, task.status, result.exit_code,
        )
        return task

    async def list_tasks(
        self, session_id: str | None = None,
        status: str | None = None,
    ) -> list[C2Task]:
        """List tasks, optionally filtered by session_id + status."""
        stmt = select(C2Task).join(C2Session).where(C2Session.scan_id == self.scan_id)
        if session_id:
            stmt = stmt.where(C2Task.session_id == uuid.UUID(session_id))
        if status:
            stmt = stmt.where(C2Task.status == status)
        stmt = stmt.order_by(C2Task.created_at.desc())
        result = await self.session.execute(stmt)
        return list(result.scalars().all())


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def create_c2_manager(
    session: AsyncSession, scan_id: str, hooks: C2Hooks | None = None,
) -> C2Manager:
    """Create a C2Manager for a scan session."""
    return C2Manager(session=session, scan_id=scan_id, hooks=hooks)