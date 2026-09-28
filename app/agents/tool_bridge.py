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


# ---------- Schema cache (per-agent-toolset) ----------
#
# build_tool_schemas() is called once per ReAct iteration. Without a cache,
# every iteration rebuilds the OpenAI tool schema list from the cached
# ToolDefs — cheap relative to the YAML parse, but still O(N) per call.
# We memoize on (frozenset(tool_names) or None-for-all).
#
# Cache is invalidated automatically when load_all_tools() re-parses YAMLs
# (the underlying ToolDef objects change identity).

_tool_schema_cache: dict[str, list[dict[str, Any]]] = {}


def _cache_key(tool_names: list[str] | None) -> str:
    """Build a hashable cache key from the tool_names filter."""
    if tool_names is None:
        return "__all__"
    return "|".join(sorted(tool_names))


def invalidate_tool_schema_cache() -> None:
    """Drop the cached tool schemas. Call after reload_tools()."""
    _tool_schema_cache.clear()


def build_tool_schemas(tool_names: list[str] | None = None) -> list[dict[str, Any]]:
    """Build OpenAI-compatible tool schemas from YAML tool definitions.

    CACHED — the resulting schema list is memoized per `tool_names` filter
    so the per-iteration call inside the ReAct loop is O(1) after the
    first call. Cache lives for the lifetime of the process; call
    `invalidate_tool_schema_cache()` after `reload_tools()` to refresh.

    Args:
        tool_names: optional filter — only include these tools.
            If None, includes all enabled tools.

    Returns:
        list of tool schema dicts in OpenAI function-calling format:
            [{"type": "function", "function": {"name", "description", "parameters"}}]
    """
    key = _cache_key(tool_names)
    cached = _tool_schema_cache.get(key)
    if cached is not None:
        return cached

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
    _tool_schema_cache[key] = schemas
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

    Phase F5 (CyberStrikeAI-aligned): adds 3-layer cache to avoid
    wasted subprocess executions:

        1. Tool availability cache — if a binary was marked unavailable
           (Binary not found / Permission denied) earlier in this scan,
           return the cached error immediately. Skip subprocess spawn.
        2. Tool call result cache — if the same (tool_name, args) was
           already executed in this scan, return the cached output.
           Skip subprocess spawn.
        3. Scope-blocked cache — if a (tool_name, args) was blocked by
           the scope guard earlier, return the cached block message.
           Skip subprocess spawn.

    All caches are per-scan (in-memory) and cleared when the scan
    completes/aborts via clear_scan_cache(scan_id).

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

    # ── Phase F5: cache lookups BEFORE subprocess spawn ──────────────
    # All 3 caches are per-scan (in-memory). A cache HIT means we can
    # skip the subprocess entirely — saving 0.02-300s of execution time
    # + LLM tokens that would have been wasted on the same result.
    from app.agents.tool_call_cache import (
        get_cached_unavailable,
        get_cached_result,
        is_scope_blocked,
        mark_unavailable,
        cache_result,
        mark_scope_blocked,
        increment_call_count,
    )

    # ── PATCH (cache-loop fix): increment call counter ──
    # Tracks EVERY attempt for this (tool, args) signature in this scan.
    # We do this BEFORE any cache lookup so the counter captures both
    # cache hits AND cache misses (i.e. real subprocess spawns).
    # The counter is then used below to detect degenerate retry loops
    # — if the LLM calls the same (tool, args) 3+ times, we prepend a
    # warning to the cached output so the LLM understands it's stuck.
    call_count = increment_call_count(scan_id, tool_name, tool_args)

    # Layer 1: tool availability (binary missing/blocked)
    cached_unavail = get_cached_unavailable(scan_id, tool_name)
    if cached_unavail is not None:
        if call_count >= 3:
            logger.warning(
                "DUPLICATE_TOOL_CALL_WARNING | scan=%s | tool=%s | count=%d | "
                "agent may be stuck in a retry loop — appending guidance to output",
                scan_id, tool_name, call_count,
            )
            return (
                f"{cached_unavail}\n\n"
                f"⚠️ You have called '{tool_name}' {call_count} times with the same "
                f"args in this scan, and it has been marked unavailable. The tool "
                f"is NOT installed or permission-denied. STOP calling this tool — "
                f"try an alternative tool or skip this test. Calling again will "
                f"return the same cached error."
            )
        return cached_unavail

    # Layer 2: scope-blocked combo (tool + args were blocked earlier)
    cached_block = is_scope_blocked(scan_id, tool_name, tool_args)
    if cached_block is not None:
        if call_count >= 3:
            logger.warning(
                "DUPLICATE_TOOL_CALL_WARNING | scan=%s | tool=%s | count=%d | "
                "scope-blocked combo retried — appending guidance to output",
                scan_id, tool_name, call_count,
            )
            return (
                f"{cached_block}\n\n"
                f"⚠️ You have called '{tool_name}' {call_count} times with the same "
                f"args, and it was scope-blocked each time. The target is NOT in the "
                f"declared scan scope. STOP calling this combo — try a different "
                f"target or ask the user to expand the scope. Calling again will "
                f"return the same cached block."
            )
        return cached_block

    # Layer 3: same tool call with same args (result dedup)
    cached_result = get_cached_result(scan_id, tool_name, tool_args)
    if cached_result is not None:
        # ── PATCH (cache-loop fix): warn LLM about degenerate retries ──
        # When the LLM requests the SAME (tool, args) for the 3rd+ time,
        # it's a sign of a retry loop. The cached result is already in the
        # LLM's context history — calling it again wastes an iteration.
        # Append a guidance message that tells the LLM explicitly:
        #   - This is a cached result (already seen)
        #   - The agent should try a DIFFERENT tool or DIFFERENT args
        #   - Or call `exit` if there's nothing more to do
        if call_count >= 3:
            logger.warning(
                "DUPLICATE_TOOL_CALL_WARNING | scan=%s | tool=%s | count=%d | "
                "same args called %d times — appending 'try different approach' guidance",
                scan_id, tool_name, call_count, call_count,
            )
            return (
                f"{cached_result}\n\n"
                f"⚠️ DUPLICATE CALL: You have called '{tool_name}' with the SAME "
                f"args {call_count} times in this scan. The result above is cached "
                f"— calling again will return the SAME output. "
                f"To make progress, you must EITHER:\n"
                f"  1. Call '{tool_name}' with DIFFERENT args (e.g. different "
                f"target/port/severity/depth), OR\n"
                f"  2. Switch to a different tool to gather new information, OR\n"
                f"  3. If you have enough findings, call `exit` with a summary.\n"
                f"Continuing to retry the same call will burn your iteration budget "
                f"without producing new results."
            )
        return cached_result

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
    elif execution.status == ExecutionStatus.HARD_TIMEOUT:
        output_parts.append(
            f"[error] Hard timeout after {tool_def.timeout}s. "
            f"The tool was running too long. Consider:"
        )
        output_parts.append(f"  - reducing scan scope (fewer ports, smaller CIDR)")
        output_parts.append(f"  - using a faster tool (e.g. masscan instead of nmap for full-range port scan)")
        output_parts.append(f"  - increasing timeout in the tool YAML")
    elif execution.status == ExecutionStatus.CANCELLED:
        output_parts.append(
            f"[error] Tool execution was cancelled. "
            f"This may have been triggered by the panic button or by the scan "
            f"being aborted. Do NOT retry this tool — wait for user input."
        )
    elif execution.status == ExecutionStatus.FAILED:
        output_parts.append(f"[error] Tool execution failed: {execution.error}")
    else:
        output_parts.append(
            f"[error] Unexpected execution status: {execution.status.value}"
        )

    # ── PATCH (cache-loop fix): separate body from execution_id ──
    # The previous code appended `[execution_id] {uuid}` to `output_parts`
    # BEFORE calling `cache_result(...)`. The cached value therefore
    # contained a fresh UUID per call — meaning even when the LLM retried
    # with identical args, `get_cached_result()` returned a hit but the
    # string started with a stale execution_id that differed from what
    # the LLM had seen previously.
    #
    # The LLM, not understanding why the execution_id changed, would
    # sometimes retry the SAME tool call (assuming the cached result was
    # somehow "different" from what it expected). This caused degenerate
    # loops — e.g. vulnerability-triage burning 28/30 iterations calling
    # nuclei with the same args, each time spawning a fresh subprocess +
    # caching a new execution_id string.
    #
    # Fix: build the cacheable body SEPARATELY from the execution_id
    # metadata. Cache ONLY the body. Append execution_id to the LLM-facing
    # output AFTER cache lookup/store, so the cached value is stable
    # across calls.
    body = "\n".join(output_parts) if output_parts else "(no output)"

    # Truncate for LLM context (keep first + last 2000 chars).
    # NOTE: truncation must happen on `body` BEFORE caching — otherwise
    # the cached value is the un-truncated version, which won't match
    # the truncated version returned to the LLM on the next call.
    if len(body) > 4000:
        body = body[:2000] + "\n... [truncated] ...\n" + body[-2000:]

    # ── Phase F5: populate caches AFTER subprocess returns ──────────
    # Detect patterns in the output and mark the appropriate cache so
    # future calls with the same tool/args can skip the subprocess.
    # NOTE: cache the TRUNCATED `body` (no execution_id), not the
    # final `output` — this is what makes the cache stable.
    output_lower_head = body[:500].lower()  # check first 500 chars only
    if "binary not found" in output_lower_head or "[errno 13] permission denied" in output_lower_head:
        # Layer 1: tool unavailable (binary missing or no execute perms)
        # Mark for ALL future calls to this tool_name — no point retrying.
        mark_unavailable(scan_id, tool_name, body)
    elif "scope violation" in output_lower_head:
        # Layer 3: scope-blocked combo — mark for this specific (tool, args)
        # so different args of the same tool are still tried.
        mark_scope_blocked(scan_id, tool_name, tool_args, body)
    else:
        # Layer 2: successful (or non-binary/scope error) — cache body
        # so the LLM's retry with same args returns cached output.
        cache_result(scan_id, tool_name, tool_args, body)

    # Now build the LLM-facing output by appending the execution_id AFTER
    # the cache has been populated. This way:
    #   - Cached value = body (stable, deterministic)
    #   - LLM-facing output = body + "\n[execution_id] {uuid}" (variable)
    # The LLM can still see the execution_id for traceability, but the
    # cache never includes it → no false "different result" perception.
    output = body + f"\n[execution_id] {execution.id}"

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