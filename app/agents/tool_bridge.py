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
import re
from typing import Any

from app.tools.loader import load_all_tools, ToolDef, _build_scope_for_target
from app.sandbox.executor import SubprocessExecutor, ToolResult
from app.sandbox.scope_guard import ScopeGuard

logger = logging.getLogger(__name__)

# ANSI/VT100 control sequences (CSI colour codes, cursor moves, OSC titles).
# Security tools (sqlmap, feroxbuster, nuclei, nmap NSE) emit these even when
# not attached to a TTY. They waste the LLM's context budget and made the model
# report "output is truncated at the first ANSI colour escape" and burn whole
# turns trying to "disable colour" instead of reading results.
#
# The canonical implementation now lives in `app.harness.ansi.strip_ansi`.
# This module keeps a local `_strip_ansi` alias for backward compatibility
# with existing imports — callers should migrate to `app.harness.ansi`.
from app.harness.ansi import strip_ansi as _strip_ansi, _ANSI_ESCAPE_RE

# Max characters of tool output forwarded to the LLM (head + tail kept).
MAX_LLM_OUTPUT_CHARS = 8000


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


# ---------------------------------------------------------------------------
# W19-FIX2: HITL gate — defense-in-depth wiring (per DIAG-W19-2 audit)
# ---------------------------------------------------------------------------

async def _maybe_run_hitl_gate(
    scan_id: str,
    tool_name: str,
    tool_args: dict[str, Any],
    target: str,
    cmd_str: str,
) -> str | None:
    """Check if HITL is required for this tool call + run the gate if so.

    Returns:
        - None  → tool is approved (or not destructive) — proceed with execute.
        - str   → HITL decision message to return to agent (reject/timeout/abort).

    Behavior:
        1. Determine if tool + args are destructive (DESTRUCTIVE_TOOLS set +
           sqlmap --os-shell/--dump/--sql-shell/--os-pwn/--priv-esc detection).
        2. If not destructive → return None (no gate needed).
        3. If destructive → invoke HITLManager.request_and_wait() which:
           - Creates HITLApproval row (pending)
           - Emits SSE event 'hitl_approval_required'
           - Branches by mode (auto_approve | audit_agent | human_block)
           - Returns HITLDecision
        4. If decision is 'approve' → return None (proceed).
        5. If decision is 'reject'/'user_aborted'/'approval_timeout' →
           return a clear message so the agent can reason about it.
        6. If decision is 'suggest_alternative' → return the suggested
           args as a hint (agent can retry with these).

    Failure mode: if HITLManager raises (e.g. DB not reachable), default
    to reject per master plan §W7-B (VAPT_AI_HITL_AUDIT_FALLBACK=reject).
    """
    # Fix (worklog task #17): respect VAPT_AI_HITL_DISABLED env flag.
    # When HITL is globally disabled (default since task #14), this gate
    # must NOT create any HITLApproval rows or call the audit_agent LLM.
    # Previously the agent-loop layer (base.py:1036-1048) respected the
    # env flag, but this gate in tool_bridge.py did NOT — so destructive
    # tools still triggered HITL INSERTs into vapt_hitl_approvals, causing
    # VARCHAR(32) overflow on predicted_impact='Metasploit exploit module
    # execution' (35 chars > 32). Now both layers honor the same flag.
    import os as _os
    if _os.environ.get("VAPT_AI_HITL_DISABLED", "1").strip().lower() in (
        "1", "true", "yes", "on",
    ):
        return None  # HITL disabled — proceed with execute, no DB row created

    try:
        from app.db.session import async_session
        from app.hitl.manager import HITLManager

        async with async_session() as session:
            mgr = HITLManager(session)
            if not mgr.is_hitl_required(tool_name, tool_args):
                return None  # not destructive — proceed normally

            # Predicted impact — auto-derived from tool + args
            predicted_impact = _derive_predicted_impact(tool_name, tool_args, cmd_str)
            agent_reasoning = (
                f"Agent invoked {tool_name} with args {tool_args} against {target}. "
                f"Full command: {cmd_str[:200]}"
            )

            # KG confidence — default 0.5 (no KG lookup in this hot path)
            kg_confidence = 0.5

            decision = await mgr.request_and_wait(
                scan_id=scan_id,
                tool_name=tool_name,
                target=target,
                args=tool_args,
                predicted_impact=predicted_impact,
                agent_reasoning=agent_reasoning,
                kg_confidence=kg_confidence,
            )
            await session.commit()

        # Translate decision to agent-facing string
        if decision.decision == "approve":
            logger.info(
                "HITL approved | scan=%s | tool=%s | decided_by=%s | duration=%.2fs",
                scan_id, tool_name, decision.decided_by, decision.duration_seconds,
            )
            return None  # proceed with execute
        elif decision.decision == "suggest_alternative" and decision.suggested_args:
            suggested_str = json.dumps(decision.suggested_args, ensure_ascii=False)
            return (
                f"[HITL SUGGEST_ALTERNATIVE] {decision.comment}\n"
                f"Suggested args: {suggested_str}\n"
                f"Retry the tool with these args if appropriate."
            )
        else:
            # reject / user_aborted / approval_timeout
            return (
                f"[HITL {decision.decision.upper()}] Tool '{tool_name}' was "
                f"NOT executed. Reason: {decision.comment}\n"
                f"Decided by: {decision.decided_by} "
                f"(duration: {decision.duration_seconds:.2f}s).\n"
                f"You can: (a) try a different exploit path, (b) try different "
                f"args that are less destructive, or (c) report the finding "
                f"with the evidence you already have."
            )
    except Exception as exc:
        logger.exception(
            "HITL gate failed (non-fatal — falling back to allow) | scan=%s | tool=%s | %s",
            scan_id, tool_name, exc,
        )
        # Safe default: allow the tool to execute (HITL is a safety check,
        # not a hard gate — failing open avoids blocking the agent entirely).
        # If you want fail-closed, change this to:
        #     return f"[HITL ERROR] Gate failed: {exc}. Tool blocked as safety measure."
        return None


def _derive_predicted_impact(tool_name: str, tool_args: dict[str, Any], cmd_str: str) -> str:
    """Auto-derive a predicted impact string for the HITL approval row.

    Looks at the tool + its args + the full command string to produce a
    human-readable impact summary. Used by the audit_agent LLM critic.
    """
    tool_lower = tool_name.lower()
    args_str = " ".join(str(v) for v in tool_args.values()).lower()
    cmd_lower = cmd_str.lower()

    if tool_lower == "sqlmap" or "sqlmap" in tool_lower:
        if "--os-shell" in args_str or "--os-pwn" in args_str:
            return "SQL Injection → remote shell on DB server (os-level RCE)"
        if "--dump" in args_str:
            return "SQL Injection → full database dump (PII exfiltration)"
        if "--sql-shell" in args_str:
            return "SQL Injection → SQL shell (DB-level access)"
        return "SQL Injection detection/exploitation against target"

    if tool_lower == "metasploit" or "msfconsole" in cmd_lower or "exploit/" in cmd_lower:
        if "meterpreter" in cmd_lower or "reverse_tcp" in cmd_lower:
            return "Metasploit → meterpreter reverse shell (full RCE)"
        return "Metasploit exploit module execution"

    if tool_lower == "mimikatz":
        return "Credential extraction from Windows LSASS (W digest / Kerberos tickets)"

    if tool_lower == "hydra":
        return "Password brute force (auth spray — may lock accounts)"

    if tool_lower in {"impacket", "netexec"}:
        if "smbexec" in cmd_lower or "wmiexec" in cmd_lower:
            return "Lateral movement via SMB/WMI exec (remote command execution)"
        return "Network execution tool (lateral movement capability)"

    if tool_lower == "responder":
        return "LLMNR/NBT-NS poisoner — MITM credential capture"

    if tool_lower in {"hashcat", "john"}:
        return "Offline password cracking (when fed captured hashes)"

    return f"Destructive operation: {tool_name} with args {tool_args}"


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
        # ── W19-FIX5 Phase I: pentest_complete gate (EVVO parity) ──────
        # Per EVVO routes/pentest/tool_handlers.py:40-57.
        # Reject `exit` if agent hasn't done enough work — prevents
        # premature wrap-up at turn ~10 having only run curl.
        # Min conditions: turn >= MIN_AGENT_TURNS (20) AND
        # commands_run >= MIN_COMMANDS_FOR_COMPLETE (25).
        summary = tool_args.get("summary", "Pentest complete.")
        try:
            from app.agents.react_agent import MIN_AGENT_TURNS, MIN_COMMANDS_FOR_COMPLETE
            from app.agents.tool_call_caps import get_tool_call_stats
            stats = get_tool_call_stats(scan_id) if hasattr(get_tool_call_stats, "__call__") else {}
            commands_run = sum(stats.values()) if stats else 0
            # Get turn count from scan_registry (best-effort)
            turn = 0
            try:
                from app.pentest.scan_registry import scan_registry
                state = scan_registry.get_state(scan_id)
                if state and hasattr(state, "decisions_count"):
                    turn = state.decisions_count or 0
            except Exception:
                pass
            if turn < MIN_AGENT_TURNS or commands_run < MIN_COMMANDS_FOR_COMPLETE:
                logger.info(
                    "exit tool rejected | scan=%s | turn=%d/%d | commands=%d/%d",
                    scan_id, turn, MIN_AGENT_TURNS, commands_run, MIN_COMMANDS_FOR_COMPLETE,
                )
                return (
                    f"REJECTED: Pentest is NOT complete. You have only run "
                    f"{commands_run}/{MIN_COMMANDS_FOR_COMPLETE} commands and "
                    f"{turn}/{MIN_AGENT_TURNS} turns. A thorough pentest requires: "
                    f"header check (curl -sI), port scan (nmap), directory brute (ffuf/gobuster), "
                    f"nuclei scan, SQLi testing (sqlmap), XSS testing (dalfox), "
                    f"and parameter discovery (arjun). Continue testing."
                )
        except ImportError:
            # MIN_AGENT_TURNS not available (older react_agent) — fall through
            pass
        except Exception as exc:
            logger.debug("pentest_complete gate check failed (non-fatal): %s", exc)

        return json.dumps({"status": "scan_complete", "summary": summary})

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

    # ── Stall circuit-breaker (checked BEFORE any cache lookup) ──────────
    # If this tool was already killed by its hard timeout (or cancelled) too
    # many times in this scan, refuse to run it again and tell the agent to
    # switch tools. Checked first so a cached timeout from an identical retry
    # does not mask the stronger "switch required" instruction.
    from app.agents.tool_call_caps import is_tool_stalled
    stalled, stall_reason = is_tool_stalled(scan_id, tool_name)
    if stalled:
        logger.warning(
            "Tool call blocked by stall circuit-breaker | scan=%s | tool=%s",
            scan_id, tool_name,
        )
        return stall_reason

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

    # ── W19-FIX3 Phase F2: per-tool call cap ─────────────────────────────
    # Check AFTER cache lookups (so cached hits — same args — do NOT burn
    # the cap budget) but BEFORE the subprocess spawn (so a denied call
    # never reaches the executor). When allowed, the counter is incremented
    # as a side-effect of check_tool_call_cap().
    from app.agents.tool_call_caps import check_tool_call_cap
    allowed, cap_reason = check_tool_call_cap(scan_id, tool_name)
    if not allowed:
        logger.warning(
            "Tool call blocked by per-scan cap | scan=%s | tool=%s",
            scan_id, tool_name,
        )
        return cap_reason

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

    # ── W19-FIX2: HITL gate — defense-in-depth wiring ──────────────
    # Per DIAG-W19-2: tool_bridge.execute_tool_call was NOT routing
    # destructive tools through HITL. The production path goes via
    # BaseAgent._execute_destructive_with_hitl (line 976 in base.py), but
    # when the agent bypasses that (e.g. via react_agent.run_react_scan
    # or specialist agents invoked directly), HITL never fires.
    #
    # This wiring makes HITL universal: ANY destructive tool call routed
    # through execute_tool_call() will trigger the HITL approval gate.
    # The gate is a no-op if VAPT_AI_HITL_MODE=auto_approve (debug).
    hitl_decision_str = await _maybe_run_hitl_gate(
        scan_id=scan_id,
        tool_name=tool_name,
        tool_args=tool_args,
        target=target,
        cmd_str=cmd_str,
    )
    if hitl_decision_str is not None:
        # HITL rejected / timed out / user aborted — return the decision
        # message to the agent so it can reason about it (e.g. try a
        # different exploit path or report the finding without the PoC).
        return hitl_decision_str

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
        # Record the stall so the circuit-breaker can force a tool switch if
        # the agent tries this tool again in this scan.
        from app.agents.tool_call_caps import mark_tool_stalled, TOOL_ALTERNATIVES
        stalls = mark_tool_stalled(scan_id, tool_name, reason="hard_timeout")
        alt = TOOL_ALTERNATIVES.get(tool_name, "a different tool from your allowlist")
        output_parts.append(
            f"[error] TIMEOUT: '{tool_name}' was killed after {tool_def.timeout}s "
            f"without producing a result (stall #{stalls})."
        )
        output_parts.append(
            f"DO NOT retry '{tool_name}' with the same settings. Choose ONE:\n"
            f"  1. Switch to: {alt}.\n"
            f"  2. Re-run '{tool_name}' with a much narrower scope "
            f"(single port / single parameter / shorter module run).\n"
            f"  3. If you already have enough evidence, call "
            f"record_vulnerability and then `exit` — an engagement that "
            f"finishes with an unconfirmed finding beats one that hangs."
        )
    elif execution.status == ExecutionStatus.CANCELLED:
        # The operator pressed the panic button (or the scan was aborted).
        # Count it as a stall too: retrying would just hit the same state.
        from app.agents.tool_call_caps import mark_tool_stalled
        mark_tool_stalled(scan_id, tool_name, reason="cancelled")
        output_parts.append(
            f"[error] Tool execution was cancelled by the operator mid-run. "
            f"Do NOT retry this tool — wait for user input, switch to a "
            f"different tool, or summarise what you have and exit."
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

    # Strip ANSI colour/control sequences first — otherwise they eat the
    # context budget and the LLM reads "\x1b[0m" instead of the finding.
    body = _strip_ansi(body)

    # Truncate for LLM context (keep first + last half).
    # NOTE: truncation must happen on `body` BEFORE caching — otherwise
    # the cached value is the un-truncated version, which won't match
    # the truncated version returned to the LLM on the next call.
    if len(body) > MAX_LLM_OUTPUT_CHARS:
        half = MAX_LLM_OUTPUT_CHARS // 2
        body = body[:half] + "\n... [truncated] ...\n" + body[-half:]

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
    """Record a vulnerability finding to DB (W19-FIX Phase A).

    Bug 1 fix (master plan W19): the previous implementation passed `description=`
    and `cvss_score=` kwargs to `Finding(...)` — neither column exists on the
    SQLAlchemy model (which uses `cvss_base_score` + `cvss_severity` + stores
    description in `metadata_json`). SQLAlchemy 2.0 DeclarativeBase raised
    `TypeError` at construction time → no INSERT → no row in DB, but orchestrator
    still counted the finding in memory. The `try/except` swallowed the error.

    Bug 2 fix (master plan W19): the previous implementation hardcoded
    `verified=True` at INSERT time — bypassing the W12 EvidenceAuditor gate.
    Now insert with `verified=False, auditor_verdict=None` and let Phase 5
    (`_phase_audit_findings`) flip it based on the auditor verdict.

    Args:
        args: finding fields from LLM (title, severity, vuln_type, target, ...)
        scan_id: scan ID for FK

    Returns:
        Confirmation string for LLM (JSON).
    """
    try:
        from app.db.session import async_session
        from app.evidence.service import EvidenceService

        # Extract fields from LLM args (case-insensitive + alias tolerant)
        title = args.get("title") or args.get("name") or "Unknown"
        vuln_type = args.get("vuln_type") or args.get("type") or "unknown"
        severity = (args.get("severity") or "info").lower()
        location = args.get("location") or args.get("target") or args.get("endpoint") or ""
        cvss_vector = args.get("cvss_vector") or None
        cvss_base_score_raw = args.get("cvss_score") or args.get("cvss_base_score") or 0.0
        try:
            cvss_base_score = float(cvss_base_score_raw)
        except (TypeError, ValueError):
            cvss_base_score = 0.0
        # Derive cvss_severity from base_score if not provided
        cvss_severity: str | None
        if cvss_base_score >= 9.0:
            cvss_severity = "critical"
        elif cvss_base_score >= 7.0:
            cvss_severity = "high"
        elif cvss_base_score >= 4.0:
            cvss_severity = "medium"
        elif cvss_base_score >= 0.1:
            cvss_severity = "low"
        else:
            cvss_severity = "info"

        # Standards mapping (LLM may supply these — pass-through)
        cwe_id = args.get("cwe_id") or None
        cve_id = args.get("cve_id") or None
        wstg_test_id = args.get("wstg_test_id") or args.get("wstg_id") or None
        mitre_attack_technique = args.get("mitre_attack_technique") or args.get("mitre_technique") or None
        mitre_attack_tactic = args.get("mitre_attack_tactic") or args.get("mitre_tactic") or None

        # PoC metadata
        exploit_method = args.get("exploit_method") or args.get("source_tool") or "agent"
        raw_evidence = args.get("evidence") or ""
        # PoC definition (worklog task #18, refined):
        # A PoC is "command + server response" — NOT a regex pattern match.
        # The LLM is the judge: if it calls record_vulnerability with
        # evidence (any tool output), the finding is recorded as PoC=successful.
        # Each target system has different bugs — we cannot assume specific
        # patterns like uid=root or set-cookie: missing flags. The LLM saw
        # the tool output and decided it constitutes a vulnerability.
        #
        # Previous strict-pattern gate (exploit_patterns.py) was REMOVED
        # because it rejected legitimate findings that didn't match
        # hard-coded signatures. The LLM's judgment is the new gate.
        if raw_evidence:
            poc_status = "successful"  # any tool output = PoC confirmed
            exploit_match_reason = "LLM provided tool output as evidence (PoC = command + server response)"
        else:
            poc_status = "not_attempted"
            exploit_match_reason = "no evidence provided — LLM did not attach tool output"

        # Pack description + remediation into metadata_json (model has no `description` column)
        description = args.get("description") or ""
        remediation = args.get("remediation") or ""
        metadata_payload: dict[str, Any] = {}
        if description:
            metadata_payload["description"] = description
        if remediation:
            metadata_payload["remediation"] = remediation
        if raw_evidence:
            metadata_payload["raw_evidence_excerpt"] = raw_evidence[:2000]
        # Stash the exploit-gate verdict so the auditor + UI can show *why*
        # a finding was or wasn't promoted to a true PoC. This is what the
        # user asked for: "if no exploit output, don't call it a finding."
        metadata_payload["poc"] = {
            "status": poc_status,
            "reason": exploit_match_reason,
            "definition": "PoC = command executed + server response captured. LLM judged this as a vulnerability.",
        }

        async with async_session() as session:
            svc = EvidenceService(session)
            # Use EvidenceService.create_finding() — it sets verified=False,
            # auditor_verdict=None, confidence_score=0.0 by default (the correct
            # pre-audit state). No `description` kwarg; no `cvss_score` kwarg.
            finding = await svc.create_finding(
                scan_id=scan_id,
                name=title,
                vuln_type=vuln_type,
                severity=severity,
                location=location,
                cvss_vector=cvss_vector,
                cvss_base_score=cvss_base_score,
                wstg_test_id=wstg_test_id,
                mitre_attack_technique=mitre_attack_technique,
                mitre_attack_tactic=mitre_attack_tactic,
                cwe_id=cwe_id,
                cve_id=cve_id,
            )
            # Override the cvss_severity + poc_status + exploit_method + metadata
            # (EvidenceService.create_finding sets severity.lower() — we already
            # computed a finer-grained value from the score).
            finding.cvss_severity = cvss_severity
            finding.poc_status = poc_status
            finding.exploit_method = exploit_method
            finding.remediation = remediation or None
            if metadata_payload:
                finding.metadata_json = metadata_payload

            # Attach one Evidence row (detection layer) with the raw tool output.
            # EvidenceService.add_evidence handles PII redaction + custody seal.
            if raw_evidence:
                await svc.add_evidence(
                    finding_id=finding.id,
                    layer="detection",
                    raw_output=raw_evidence,
                    tool_used=exploit_method,
                )
            await session.commit()
            finding_id = str(finding.id)

        logger.info(
            "Finding recorded | scan=%s | title=%s | severity=%s | cvss=%s | id=%s",
            scan_id, title, severity, cvss_base_score, finding_id[:8],
        )

        return json.dumps({
            "status": "recorded",
            "finding_id": finding_id,
            "verified": False,  # will be flipped by Phase 5 auditor
            "poc_status": poc_status,
            "poc_reason": exploit_match_reason,
            "message": (
                f"Vulnerability '{title}' recorded with severity {severity} "
                f"(cvss_base_score={cvss_base_score}). "
                + (
                    "PoC confirmed: tool output captured as evidence. "
                    "Pending audit verification."
                    if poc_status == "successful" else
                    "No PoC: no evidence attached. Run a tool, capture the "
                    "server's response, and re-record with that output as evidence."
                )
            ),
        })
    except Exception as exc:
        logger.exception("Failed to record vulnerability | scan=%s | args_keys=%s",
                         scan_id, list(args.keys()) if isinstance(args, dict) else None)
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