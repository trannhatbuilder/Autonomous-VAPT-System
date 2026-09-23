"""
VAPT-AI Tool Bridge — converts YAML tool definitions into LLM-callable
tool schemas + async execution functions.

Bridges:
    app/tools/*.yaml (32 tool definitions)
    → LLM tool schemas (OpenAI function-calling format)
    → SubprocessExecutor.execute() (real subprocess)

Also provides:
    - record_vulnerability tool (LLM-callable → INSERT finding to DB)
    - Tool execution with scope guard + output cap

P1: shares scope-helper with app/tools/loader.py so MCP path + agent path
build identical ScopeGuard. Also forwards ToolDef.allowed_exit_codes to
SubprocessExecutor (nmap exits 1 when "no hosts up" but result is still
useful — was being marked as error before).
"""
from __future__ import annotations

import json
import logging
from typing import Any

from app.tools.loader import load_all_tools, ToolDef, _build_scope_for_target
from app.sandbox.executor import SubprocessExecutor, ToolResult
from app.sandbox.scope_guard import ScopeGuard

logger = logging.getLogger(__name__)


def build_tool_schemas(tool_names: list[str] | None = None) -> list[dict[str, Any]]:
    """Build OpenAI-compatible tool schemas from YAML tool definitions.

    Args:
        tool_names: optional filter — only include these tools.
            If None, includes all enabled tools.

    Returns:
        list of tool schema dicts in OpenAI function-calling format:
            [{"type": "function", "function": {"name", "description", "parameters"}}]
    """
    all_tools = load_all_tools()
    schemas: list[dict[str, Any]] = []

    for name, tool_def in all_tools.items():
        if not tool_def.enabled:
            continue
        if tool_names and name not in tool_names:
            continue

        description = tool_def.short_description or tool_def.description[:200]
        schemas.append({
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": tool_def.to_mcp_input_schema(),
            },
        })

    # Add record_vulnerability tool (always available)
    schemas.append({
        "type": "function",
        "function": {
            "name": "record_vulnerability",
            "description": (
                "Record a confirmed vulnerability finding. Call this AFTER you have "
                "verified the vulnerability with evidence (PoC, tool output). "
                "Do NOT call this for speculative or unverified findings."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "Short title (e.g. 'SQL Injection in /login')"},
                    "severity": {
                        "type": "string",
                        "enum": ["critical", "high", "medium", "low", "info"],
                        "description": "Severity level",
                    },
                    "vuln_type": {"type": "string", "description": "Vulnerability type (e.g. 'sqli', 'xss', 'rce')"},
                    "target": {"type": "string", "description": "Affected URL or IP:port"},
                    "location": {"type": "string", "description": "Specific endpoint/parameter affected"},
                    "evidence": {"type": "string", "description": "Proof — tool output, PoC command + result"},
                    "description": {"type": "string", "description": "Detailed description of the vulnerability"},
                    "remediation": {"type": "string", "description": "How to fix this vulnerability"},
                    "cvss_score": {"type": "number", "description": "CVSS v3.1 base score (0-10)"},
                },
                "required": ["title", "severity", "vuln_type", "target", "evidence", "description"],
            },
        },
    })

    # Add exit tool (LLM calls this to signal scan complete)
    schemas.append({
        "type": "function",
        "function": {
            "name": "exit",
            "description": "Signal that the scan is complete. Call this when you have finished all testing and recorded all findings.",
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {"type": "string", "description": "Final summary of scan results"},
                },
                "required": ["summary"],
            },
        },
    })

    logger.info("Built %d tool schemas (%d security tools + record_vulnerability + exit)",
                len(schemas) - 2, len(schemas) - 2)
    return schemas


async def execute_tool_call(
    tool_name: str,
    tool_args: dict[str, Any],
    target: str,
    scan_id: str,
    executor: SubprocessExecutor,
) -> str:
    """Execute a tool call and return the result as a string for LLM.

    P2: now routes through ExecutionService — the single substrate shared
    with the MCP HTTP path. Both paths converge to:
        ExecutionService.submit(run=SubprocessExecutor.execute())
    giving us unified hard timeout, cancellation, concurrency cap, and
    status tracking (visible via get_tool_execution meta-tool).

    Args:
        tool_name: tool name (e.g. "nmap", "nuclei", "record_vulnerability", "exit")
        tool_args: arguments dict from LLM tool call
        target: scan target (for scope guard)
        scan_id: scan ID for attribution
        executor: SubprocessExecutor instance (kept for backward compat
                  with P1 callers + tests; the executor's scope_guard is
                  still used inside the run closure)

    Returns:
        Tool output string (for LLM consumption). Truncated to ~4000 chars.
    """
    # Handle special tools (these don't go through ExecutionService — they
    # don't run subprocesses; they're agent-control tools)
    if tool_name == "exit":
        return json.dumps({"status": "scan_complete", "summary": tool_args.get("summary", "")})

    if tool_name == "record_vulnerability":
        return await _record_vulnerability(tool_args, scan_id)

    # Security tool — execute via ExecutionService + SubprocessExecutor
    all_tools = load_all_tools()
    tool_def = all_tools.get(tool_name)
    if tool_def is None:
        return f"Error: tool '{tool_name}' not found."

    # Build command args from tool_args
    try:
        args = tool_def.build_command_args(**tool_args)
    except Exception as exc:
        return f"Error building command args for {tool_name}: {exc}"

    # Build full command
    cmd = [tool_def.command] + args
    cmd_str = " ".join(cmd)

    logger.info("Tool execute | scan=%s | tool=%s | cmd=%s", scan_id, tool_name, cmd_str[:120])

    # P2: build run closure + submit to ExecutionService
    from app.mcp.execution_service import get_execution_service, ExecutionStatus
    svc = get_execution_service()

    async def run(cancel_event) -> dict[str, Any]:
        # Note: executor's scope_guard already configured by create_executor
        # (which calls _build_scope_for_target). We pass the same executor
        # instance to keep P1 behavior — scope rules don't change.
        result: ToolResult = await executor.execute(
            command=cmd,
            target=target,
            timeout=tool_def.timeout,
            allowed_exit_codes=tool_def.allowed_exit_codes,
            scan_id=scan_id,
            actor_id="agent",
        )
        return result.to_dict()

    execution = await svc.submit(
        tool_name=tool_def.name,
        arguments=tool_args,
        target=target,
        run=run,
        scan_id=scan_id,
        actor_id="agent",
        hard_timeout=tool_def.timeout,
    )

    # Build output string for LLM (back-compat with P1 shape — single string,
    # not JSON). The agent system prompt expects the tool output to be the
    # tool's stdout; errors are embedded as [error] lines so the LLM can
    # reason about them in the same context.
    output_parts: list[str] = []

    if execution.status == ExecutionStatus.COMPLETED and execution.result:
        result_dict = execution.result
        if result_dict.get("scope_violation"):
            output_parts.append("SCOPE VIOLATION: target not in declared scope. Tool blocked.")
        stdout = result_dict.get("stdout") or ""
        if stdout:
            output_parts.append(stdout)
        stderr = result_dict.get("stderr") or ""
        exit_code = result_dict.get("exit_code")
        if stderr and exit_code not in (0, None):
            output_parts.append(f"[stderr] {stderr[:500]}")
        error = result_dict.get("error")
        if error:
            output_parts.append(f"[error] {error}")
        # Surface execution_id for traceability (LLM doesn't need this but
        # the SSE consumer / debug logs do)
        output_parts.append(f"[execution_id] {execution.id}")
    elif execution.status == ExecutionStatus.HARD_TIMEOUT:
        output_parts.append(
            f"[error] Hard timeout after {tool_def.timeout}s. "
            f"The tool was running too long. Consider:"
        )
        output_parts.append(f"  - reducing scan scope (fewer ports, smaller CIDR)")
        output_parts.append(f"  - using a faster tool (e.g. masscan instead of nmap for full-range port scan)")
        output_parts.append(f"  - increasing timeout in the tool YAML")
        output_parts.append(f"[execution_id] {execution.id}")
    elif execution.status == ExecutionStatus.CANCELLED:
        output_parts.append(
            f"[error] Tool execution was cancelled. "
            f"This may have been triggered by the panic button or by the scan "
            f"being aborted. Do NOT retry this tool — wait for user input."
        )
        output_parts.append(f"[execution_id] {execution.id}")
    elif execution.status == ExecutionStatus.FAILED:
        output_parts.append(f"[error] Tool execution failed: {execution.error}")
        output_parts.append(f"[execution_id] {execution.id}")
    else:
        output_parts.append(
            f"[error] Unexpected execution status: {execution.status.value}"
        )
        output_parts.append(f"[execution_id] {execution.id}")

    output = "\n".join(output_parts) if output_parts else "(no output)"

    # Truncate for LLM context (keep first + last 2000 chars)
    if len(output) > 4000:
        output = output[:2000] + "\n... [truncated] ...\n" + output[-2000:]

    logger.info(
        "Tool result | scan=%s | tool=%s | status=%s | exec_id=%s | output_len=%d",
        scan_id, tool_name, execution.status.value, execution.id[:8], len(output),
    )
    return output


async def _record_vulnerability(args: dict[str, Any], scan_id: str) -> str:
    """Record a vulnerability finding to DB.

    Args:
        args: finding fields from LLM (title, severity, vuln_type, target, ...)
        scan_id: scan ID for FK

    Returns:
        Confirmation string for LLM.
    """
    try:
        from app.db.session import async_session
        from app.db.models.pentest import Finding
        from datetime import datetime, UTC

        async with async_session() as session:
            finding = Finding(
                scan_id=scan_id,
                name=args.get("title", "Unknown"),
                vuln_type=args.get("vuln_type", "unknown"),
                severity=args.get("severity", "info"),
                location=args.get("location") or args.get("target", ""),
                description=args.get("description", ""),
                remediation=args.get("remediation", ""),
                cvss_score=args.get("cvss_score", 0.0),
                poc_status="successful" if args.get("evidence") else "not_attempted",
                verified=True,
                false_positive=False,
            )
            session.add(finding)
            await session.commit()
            finding_id = str(finding.id)

        logger.info("Finding recorded | scan=%s | title=%s | severity=%s | id=%s",
                     scan_id, args.get("title"), args.get("severity"), finding_id[:8])

        return json.dumps({
            "status": "recorded",
            "finding_id": finding_id,
            "message": f"Vulnerability '{args.get('title')}' recorded with severity {args.get('severity')}.",
        })
    except Exception as exc:
        logger.error("Failed to record vulnerability: %s", exc)
        return f"Error recording vulnerability: {exc}"


def create_executor(target: str, scan_id: str) -> SubprocessExecutor:
    """Create a SubprocessExecutor configured for a scan.

    P1: uses the shared _build_scope_for_target helper so the agent path
    and the MCP path enforce identical scope rules. Subdomains of the
    scan target are auto-allowed (e.g. target=pentest-ground.com also
    allows *.pentest-ground.com).

    Args:
        target: scan target (URL/IP/hostname)
        scan_id: scan ID for attribution

    Returns:
        SubprocessExecutor with scope guard configured for the target.
    """
    scope_guard = _build_scope_for_target(target)
    executor = SubprocessExecutor(
        scope_guard=scope_guard,
        actor_id="agent",
    )
    return executor


__all__ = [
    "build_tool_schemas",
    "execute_tool_call",
    "create_executor",
]