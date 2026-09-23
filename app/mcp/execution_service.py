"""
VAPT-AI MCP ExecutionService — worker pool for tool execution.

Mirrors CyberStrikeAI `internal/mcp/execution_service.go`.

Single substrate for ALL tool calls (both MCP HTTP tools/call AND agent
loop execute_tool_call). Provides:

    - Hard timeout (per-tool — different from SubprocessExecutor's soft
      subprocess timeout; ExecutionService's hard timeout kills the entire
      asyncio.Task even if SubprocessExecutor is hung on I/O)
    - Cancellation (panic button — cancel by execution_id or scan_id)
    - Concurrency cap (default 16 — mirrors CyberStrikeAI global cap)
    - In-memory status dict (queried via get_tool_execution meta-tool)
    - Normalized result shape (all tool results funnel through one place)

P2 design notes:
    - ToolExecution rows kept in-memory (`dict[id, ToolExecution]`) for now.
      P3 will persist to PostgreSQL (new `tool_executions` table + alembic
      migration) so long-running scans survive process restart.
    - "background_running" pattern (return immediately with execution_id,
      let LLM poll via wait_tool_execution) is deferred to P3 — P2 keeps
      the synchronous wait semantics from P1.
    - DB-attribute persistence (scan_id, actor_id) is plumbed through
      already so P3 only needs to add the INSERT at start + UPDATE at end.

Architecture:

    ┌──────────────────────────────────────────────────────────┐
    │                      ExecutionService                    │
    │                                                          │
    │  ┌────────────┐  ┌────────────┐  ┌────────────┐          │
    │  │ Execution1 │  │ Execution2 │  │ Execution3 │  ...     │
    │  │ (running)  │  │ (queued)   │  │ (done)     │          │
    │  └─────┬──────┘  └─────┬───── ┘  └─────────── ┘          │
    │        │               │                                 │
    │   asyncio.Task    asyncio.Task                           │
    │        │               │                                 │
    │   ┌────▼───────────────▼────┐                            │
    │   │    Semaphore(16)        │  ← concurrency cap         │
    │   └────────────┬────────────┘                            │
    │                │                                         │
    │   ┌────────────▼────────────┐                            │
    │   │   asyncio.wait_for(     │  ← hard timeout            │
    │   │     run(cancel_event),  │                            │
    │   │     timeout=hard_timeout  │                          │
    │   │   )                       │                          │
    │   └────────────┬─────────────┘                           │
    │                │                                         │
    │            run() closure                                 │
    │            (SubprocessExecutor.execute)                  │
    └──────────────────────────────────────────────────────────┘

All status transitions:

    queued → running → completed       (success)
                    → failed           (exception)
                    → cancelled        (panic button)
                    → hard_timeout     (exceeded hard_timeout)

In P3 we'll also add:

    queued → orphaned                  (process died — detected on restart)
    queued → background_running         (returned to caller before completion)
"""
from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any, Awaitable, Callable

logger = logging.getLogger(__name__)


# ---------- Status enum ----------

class ExecutionStatus(str, Enum):
    """Mirror CyberStrikeAI ToolExecution statuses."""
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    HARD_TIMEOUT = "hard_timeout"
    # P3 additions (placeholder — not yet used in P2):
    BACKGROUND_RUNNING = "background_running"  # returned to caller before completion
    ORPHANED = "orphaned"                       # process died without reporting


# ---------- ToolExecution dataclass ----------

@dataclass
class ToolExecution:
    """One execution of one tool. Mirrors CyberStrikeAI ToolExecution struct.

    P2: in-memory only. P3 will persist to PostgreSQL `tool_executions` table.
    """
    id: str
    tool_name: str
    arguments: dict[str, Any]
    target: str
    scan_id: str | None = None
    actor_id: str = "agent"
    status: ExecutionStatus = ExecutionStatus.QUEUED
    started_at: datetime | None = None
    completed_at: datetime | None = None
    result: dict[str, Any] | None = None  # ToolResult.to_dict() on success
    error: str | None = None
    duration_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "tool_name": self.tool_name,
            "target": self.target,
            "scan_id": self.scan_id,
            "actor_id": self.actor_id,
            "status": self.status.value,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "result": self.result,
            "error": self.error,
            "duration_seconds": round(self.duration_seconds, 3),
        }


# ---------- Run-closure signature ----------

RunClosure = Callable[
    [asyncio.Event],            # cancel_event — set when cancel() is called
    Awaitable[dict[str, Any]],  # returns ToolResult.to_dict()
]


# ---------- ExecutionService ----------

class ExecutionService:
    """Worker pool for tool execution.

    Singleton via get_execution_service().

    Usage:
        svc = get_execution_service()
        execution = await svc.submit(
            tool_name="nmap",
            arguments={"target": "10.0.0.5", "ports": "1-1000"},
            target="10.0.0.5",
            run=lambda cancel_event: executor.execute(...).then(to_dict),
            scan_id="scan_abc",
            actor_id="recon_agent",
            hard_timeout=600,
        )
        if execution.status == ExecutionStatus.COMPLETED:
            tool_result = execution.result
        elif execution.status == ExecutionStatus.HARD_TIMEOUT:
            ...
    """

    def __init__(
        self,
        max_concurrent: int = 16,
        default_hard_timeout: int = 3600,
    ):
        self._executions: dict[str, ToolExecution] = {}
        self._tasks: dict[str, asyncio.Task] = {}
        self._cancel_events: dict[str, asyncio.Event] = {}
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._default_hard_timeout = default_hard_timeout
        self._max_concurrent = max_concurrent
        self._lock = asyncio.Lock()
        logger.info(
            "ExecutionService initialized | max_concurrent=%d | default_hard_timeout=%ds",
            max_concurrent, default_hard_timeout,
        )

    # ---------- Public API ----------

    async def submit(
        self,
        *,
        tool_name: str,
        arguments: dict[str, Any],
        target: str,
        run: RunClosure,
        scan_id: str | None = None,
        actor_id: str = "agent",
        hard_timeout: int | None = None,
    ) -> ToolExecution:
        """Submit a tool execution. Blocks until completion (P2 synchronous).

        P3 will add a `wait=False` mode that returns immediately with
        status=BACKGROUND_RUNNING for the agent loop.

        Args:
            tool_name, arguments, target: tool call info (for status row)
            run: async closure that does the actual work. Receives a
                cancel_event that the closure SHOULD check periodically
                (SubprocessExecutor doesn't yet — it relies on the
                asyncio.Task cancellation propagating to the subprocess
                via process-group kill).
            scan_id: scan ID for attribution (P3 will use for DB row FK)
            actor_id: agent name or "user"
            hard_timeout: max seconds — None = use self._default_hard_timeout

        Returns:
            Final ToolExecution (status=COMPLETED / FAILED / CANCELLED / HARD_TIMEOUT)
        """
        exec_id = str(uuid.uuid4())
        cancel_event = asyncio.Event()
        execution = ToolExecution(
            id=exec_id,
            tool_name=tool_name,
            arguments=arguments,
            target=target,
            scan_id=scan_id,
            actor_id=actor_id,
            status=ExecutionStatus.QUEUED,
            started_at=datetime.now(UTC),
        )
        async with self._lock:
            self._executions[exec_id] = execution
            self._cancel_events[exec_id] = cancel_event

        effective_timeout = hard_timeout or self._default_hard_timeout

        # Spawn worker task — we'll await it inline (P2 synchronous semantics)
        task = asyncio.create_task(self._run_worker(
            execution=execution,
            run=run,
            cancel_event=cancel_event,
            hard_timeout=effective_timeout,
        ))
        async with self._lock:
            self._tasks[exec_id] = task

        # Await completion — P3 will offer non-blocking mode
        try:
            await task
        except asyncio.CancelledError:
            # Task was cancelled externally — already handled in _run_worker
            pass

        async with self._lock:
            return self._executions[exec_id]

    async def _run_worker(
        self,
        *,
        execution: ToolExecution,
        run: RunClosure,
        cancel_event: asyncio.Event,
        hard_timeout: int,
    ) -> None:
        """Worker coroutine: acquire semaphore, run with timeout, finalize."""
        import time
        start = time.time()

        async with self._semaphore:
            async with self._lock:
                execution.status = ExecutionStatus.RUNNING
                execution.started_at = execution.started_at or datetime.now(UTC)

            try:
                # Run with hard timeout
                result = await asyncio.wait_for(
                    run(cancel_event),
                    timeout=hard_timeout,
                )
                async with self._lock:
                    execution.status = ExecutionStatus.COMPLETED
                    execution.result = result
                    execution.completed_at = datetime.now(UTC)
                    execution.duration_seconds = time.time() - start
                logger.info(
                    "ToolExecution %s completed | tool=%s | duration=%.2fs",
                    execution.id[:8], execution.tool_name, execution.duration_seconds,
                )

            except asyncio.TimeoutError:
                async with self._lock:
                    execution.status = ExecutionStatus.HARD_TIMEOUT
                    execution.error = f"Hard timeout after {hard_timeout}s"
                    execution.completed_at = datetime.now(UTC)
                    execution.duration_seconds = time.time() - start
                logger.warning(
                    "ToolExecution %s HARD_TIMEOUT | tool=%s | timeout=%ds",
                    execution.id[:8], execution.tool_name, hard_timeout,
                )

            except asyncio.CancelledError:
                async with self._lock:
                    execution.status = ExecutionStatus.CANCELLED
                    execution.error = "Cancelled by user or panic button"
                    execution.completed_at = datetime.now(UTC)
                    execution.duration_seconds = time.time() - start
                logger.info(
                    "ToolExecution %s CANCELLED | tool=%s",
                    execution.id[:8], execution.tool_name,
                )
                # Don't re-raise — worker should swallow CancelledError so
                # the caller (submit()) doesn't see an exception. The status
                # field is the canonical signal.

            except Exception as exc:
                async with self._lock:
                    execution.status = ExecutionStatus.FAILED
                    execution.error = str(exc)
                    execution.completed_at = datetime.now(UTC)
                    execution.duration_seconds = time.time() - start
                logger.error(
                    "ToolExecution %s FAILED | tool=%s | error=%s",
                    execution.id[:8], execution.tool_name, exc,
                )

    async def get(self, execution_id: str) -> ToolExecution | None:
        """Get execution status by ID. Non-blocking."""
        async with self._lock:
            return self._executions.get(execution_id)

    async def list_for_scan(self, scan_id: str) -> list[ToolExecution]:
        """List all executions for a scan (for panic button — abort scan)."""
        async with self._lock:
            return [
                e for e in self._executions.values()
                if e.scan_id == scan_id
            ]

    async def wait(self, execution_id: str, timeout: int = 30) -> ToolExecution | None:
        """Wait up to `timeout` seconds for an execution to complete.

        Returns the final ToolExecution (whatever its status), or the
        current ToolExecution if still running after timeout.
        """
        async with self._lock:
            task = self._tasks.get(execution_id)
            execution = self._executions.get(execution_id)
        if task is None or execution is None:
            return None
        try:
            await asyncio.wait_for(asyncio.shield(task), timeout=timeout)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            pass
        return await self.get(execution_id)

    async def cancel(self, execution_id: str) -> bool:
        """Cancel a running execution by ID.

        Returns:
            True if cancellation was initiated (status was running/queued).
            False if execution was already finished or not found.
        """
        async with self._lock:
            task = self._tasks.get(execution_id)
            execution = self._executions.get(execution_id)
            cancel_event = self._cancel_events.get(execution_id)

        if task is None or execution is None:
            return False

        terminal = {
            ExecutionStatus.COMPLETED,
            ExecutionStatus.FAILED,
            ExecutionStatus.CANCELLED,
            ExecutionStatus.HARD_TIMEOUT,
        }
        if execution.status in terminal:
            return False

        # Set the cancel event first (lets the run closure do graceful cleanup)
        if cancel_event is not None:
            cancel_event.set()

        # Then cancel the asyncio.Task (propagates CancelledError into the
        # run closure, which SubprocessExecutor's _kill_process_group handles
        # by SIGKILL-ing the entire process group)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        return True

    async def cancel_scan(self, scan_id: str) -> int:
        """Cancel all running executions for a scan. Returns count cancelled."""
        execs = await self.list_for_scan(scan_id)
        cancelled = 0
        for e in execs:
            if await self.cancel(e.id):
                cancelled += 1
        return cancelled

    async def list_all(self, limit: int = 100) -> list[ToolExecution]:
        """List recent executions (newest first). For debug/admin UI."""
        async with self._lock:
            all_execs = list(self._executions.values())
        all_execs.sort(key=lambda e: e.started_at or datetime.min, reverse=True)
        return all_execs[:limit]


# ---------- Singleton ----------

_execution_service: ExecutionService | None = None


def get_execution_service() -> ExecutionService:
    """Get the singleton ExecutionService instance."""
    global _execution_service
    if _execution_service is None:
        _execution_service = ExecutionService()
    return _execution_service


def reset_execution_service() -> None:
    """Reset the singleton (for tests)."""
    global _execution_service
    _execution_service = None


__all__ = [
    "ExecutionService",
    "ExecutionStatus",
    "ToolExecution",
    "RunClosure",
    "get_execution_service",
    "reset_execution_service",
]