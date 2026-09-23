"""
VAPT-AI MCP Execution Control Tools — 3 meta-tools for tool execution polling.

Mirrors CyberStrikeAI `internal/mcp/execution_control_tools.go`:

    - get_tool_execution(execution_id)   → non-blocking status poll
    - wait_tool_execution(execution_id, timeout=30)
                                         → block up to N sec for completion
    - cancel_tool_execution(execution_id) → panic button

When to use:
    The model uses these tools when an MCP tools/call returns status
    "background_running" (P3 feature — P2 keeps synchronous semantics).
    For now these tools are useful for:
        1. Frontend panic button (via HTTP endpoints, not via MCP)
        2. LLM to cancel a runaway tool when it realizes the scan is
           going off-track (e.g., nmap full-port scan taking too long)
        3. Debug visibility — admin can query execution status during
           development

Also registered as HTTP endpoints in app/main.py for the frontend:
    GET  /api/mcp/executions              — list recent (admin/debug)
    GET  /api/mcp/executions/{id}         — get one by ID
    POST /api/mcp/executions/{id}/wait    — wait up to ?timeout=N sec
    POST /api/mcp/executions/{id}/cancel  — panic button
    POST /api/mcp/scans/{scan_id}/abort   — cancel all tools for a scan
"""
from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


def register_execution_control_tools(mcp_server) -> None:
    """Register the 3 execution-control meta-tools on the FastMCP server.

    Idempotent — safe to call multiple times (FastMCP deduplicates by name).
    """
    from app.mcp.execution_service import get_execution_service

    svc = get_execution_service()

    @mcp_server.tool()
    async def get_tool_execution(execution_id: str) -> str:
        """Get the current status of a tool execution by ID.

        Non-blocking — returns immediately with the current status
        (queued / running / completed / failed / cancelled / hard_timeout).

        Use this when polling a long-running tool (e.g. a 600s nmap scan)
        without blocking the agent loop.

        Args:
            execution_id: The execution ID returned by a previous tool call
                when its status was "background_running" or "hard_timeout".

        Returns:
            JSON string with execution details:
                {
                  "id": "...",
                  "tool_name": "nmap",
                  "target": "...",
                  "scan_id": "...",
                  "status": "completed" | "running" | "failed" | ...,
                  "started_at": "2026-...",
                  "completed_at": "2026-..." | null,
                  "result": {stdout, stderr, exit_code, ...} | null,
                  "error": "..." | null,
                  "duration_seconds": 12.345
                }
        """
        execution = await svc.get(execution_id)
        if execution is None:
            return json.dumps({
                "error": f"Execution {execution_id!r} not found",
                "execution_id": execution_id,
            })
        return json.dumps(execution.to_dict(), indent=2, default=str)

    @mcp_server.tool()
    async def wait_tool_execution(execution_id: str, timeout: int = 30) -> str:
        """Wait up to `timeout` seconds for an execution to complete.

        Blocks the agent loop until either:
            - the execution reaches a terminal status (completed/failed/
              cancelled/hard_timeout) → returns final ToolExecution
            - timeout elapses → returns current ToolExecution (still running)

        Args:
            execution_id: The execution ID to wait for.
            timeout: Max seconds to block (default 30, max 300).

        Returns:
            Same shape as get_tool_execution. The "status" field tells
            you whether to keep waiting or proceed.
        """
        # Cap timeout at 300s to prevent runaway polling
        timeout = max(1, min(int(timeout), 300))
        execution = await svc.wait(execution_id, timeout=timeout)
        if execution is None:
            return json.dumps({
                "error": f"Execution {execution_id!r} not found",
                "execution_id": execution_id,
            })
        return json.dumps(execution.to_dict(), indent=2, default=str)

    @mcp_server.tool()
    async def cancel_tool_execution(execution_id: str) -> str:
        """Cancel a running tool execution. Panic button.

        Sends a cancellation signal to the execution. The SubprocessExecutor
        receives it and SIGKILLs the entire process group (start_new_session
        ensures the whole tool tree dies, not just the parent).

        Safe to call on already-completed executions — returns
        {"cancelled": false, "reason": "..."} in that case.

        Args:
            execution_id: The execution ID to cancel.

        Returns:
            {
              "cancelled": true | false,
              "execution_id": "...",
              "final_status": "cancelled" | "completed" | ...,
              "reason": "..."
            }
        """
        ok = await svc.cancel(execution_id)
        execution = await svc.get(execution_id)
        if execution is None:
            return json.dumps({
                "cancelled": False,
                "execution_id": execution_id,
                "final_status": None,
                "reason": "Execution not found",
            })
        return json.dumps({
            "cancelled": ok,
            "execution_id": execution_id,
            "final_status": execution.status.value,
            "reason": (
                ""
                if ok
                else f"Execution already in terminal status: {execution.status.value}"
            ),
        }, indent=2)

    logger.info(
        "Registered 3 execution-control meta-tools: get_tool_execution, "
        "wait_tool_execution, cancel_tool_execution"
    )


__all__ = ["register_execution_control_tools"]