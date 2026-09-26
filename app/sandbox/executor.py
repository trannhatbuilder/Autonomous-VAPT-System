"""
VAPT-AI Subprocess Executor — Python port of CyberStrikeAI executor.go.

Direct subprocess execution (NO Docker, NO iptables — D11).
Application-level scope guard validates every command before execution.

Features (ported from CyberStrikeAI internal/security/executor.go):
    - Process group isolation (start_new_session=True / setsid)
    - Timeout + cancellation via os.killpg (SIGKILL entire process tree)
    - Output cap (default 50KB — spill to disk if exceeded)
    - PTY retry on TTY-demand errors (sqlmap interactive, etc.)
    - Scope guard validation before every subprocess call
    - Non-interactive stdin (DEVNULL — prevents hangs)
    - Pager env injection (GIT_PAGER=cat, PAGER=cat — prevents pager blocking)

W7-C additions:
    - execute_with_hitl() — wraps execute() with HITL gate (AuditAgent review)
    - Subprocess PID registration with ScanRegistry (for panic-button kill)

W8-A additions:
    - Audit log persistence for scope violations (D23 chain-of-custody).
      Caller injects optional AuditLogger; executor does NOT open DB sessions
      itself (matches CyberStrikeAI pattern where the executor uses zap.Logger
      only, and the audit service is wired from the handler layer).
    - scan_id + actor_id params on execute() for attribution.

Usage:
    from app.sandbox.executor import SubprocessExecutor, ToolResult
    from app.sandbox.scope_guard import ScopeGuard, ScopeRule

    guard = ScopeGuard(declared_scope=[
        ScopeRule(host="example.com"),
        ScopeRule(cidr="192.168.1.0/24"),
    ])
    executor = SubprocessExecutor(scope_guard=guard)

    result = await executor.execute(
        command=["nmap", "-sS", "-p", "1-1000", "192.168.1.5"],
        target="192.168.1.5",
        timeout=600,
    )
    if result.success:
        print(result.stdout)
    else:
        print(f"Error: {result.stderr}")
"""
from __future__ import annotations

import asyncio
import logging
import os
import pty
import re
import signal
import struct
import fcntl
import termios
import select
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.sandbox.scope_guard import ScopeGuard, ScopeViolation, ValidationResult

logger = logging.getLogger(__name__)


# ---------- Constants ----------

DEFAULT_OUTPUT_MAX_BYTES = 50_000  # 50KB cap (D24)
DEFAULT_TIMEOUT = 300               # 5 minutes
SPILL_DIR = Path("data/evidence_spills")

# Patterns that indicate tool needs TTY (triggers PTY retry)
TTY_DEMAND_PATTERNS = [
    re.compile(r"inappropriate ioctl for device", re.IGNORECASE),
    re.compile(r"termios\.error", re.IGNORECASE),
    re.compile(r"not a tty", re.IGNORECASE),
    re.compile(r"no tty", re.IGNORECASE),
    re.compile(r"terminal required", re.IGNORECASE),
]


# ---------- Tool result ----------

@dataclass
class ToolResult:
    """Result of a tool execution."""
    success: bool
    exit_code: int | None = None
    stdout: str = ""
    stderr: str = ""
    error: str | None = None
    truncated: bool = False
    spill_path: str | None = None
    duration_seconds: float = 0.0
    command: str = ""
    target: str = ""
    scope_violation: bool = False

    @classmethod
    def error_result(cls, error: str, command: str = "", target: str = "") -> "ToolResult":
        """Create an error result."""
        return cls(
            success=False,
            error=error,
            command=command,
            target=target,
        )

    @classmethod
    def scope_violation_result(cls, reason: str, command: str = "", target: str = "") -> "ToolResult":
        """Create a scope violation result."""
        return cls(
            success=False,
            error=f"Scope violation: {reason}",
            command=command,
            target=target,
            scope_violation=True,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict (for JSON responses)."""
        return {
            "success": self.success,
            "exit_code": self.exit_code,
            "stdout": self.stdout[:500] if self.stdout else "",
            "stderr": self.stderr[:500] if self.stderr else "",
            "error": self.error,
            "truncated": self.truncated,
            "spill_path": self.spill_path,
            "duration_seconds": self.duration_seconds,
            "command": self.command,
            "target": self.target,
            "scope_violation": self.scope_violation,
        }


# ---------- Subprocess Executor ----------

class SubprocessExecutor:
    """Async subprocess executor with scope guard + output cap + PTY retry.

    Python port of CyberStrikeAI internal/security/executor.go.
    Direct subprocess — NO Docker, NO iptables.
    """

    def __init__(
        self,
        scope_guard: ScopeGuard | None = None,
        output_max_bytes: int = DEFAULT_OUTPUT_MAX_BYTES,
        audit_logger: Any = None,
        actor_id: str = "agent",
    ):
        self.scope_guard = scope_guard or ScopeGuard()
        self.output_max_bytes = output_max_bytes
        # W8-A: optional AuditLogger for persisting scope_violation events.
        # Caller (agent loop) injects this; executor never opens DB sessions
        # itself. None = audit persistence disabled (used in tests).
        self.audit_logger = audit_logger
        self.actor_id = actor_id

    async def execute(
        self,
        command: list[str] | str,
        target: str = "",
        timeout: int = DEFAULT_TIMEOUT,
        env: dict[str, str] | None = None,
        workdir: str | None = None,
        allowed_exit_codes: list[int] | None = None,
        scan_id: str | None = None,
        actor_id: str | None = None,
        agent_name: str | None = None,
    ) -> ToolResult:
        """Execute a command via subprocess.

        Args:
            command: Command list (preferred) or shell string
            target: Target IP/URL (for scope guard validation)
            timeout: Timeout in seconds
            env: Additional environment variables
            workdir: Working directory
            allowed_exit_codes: Exit codes treated as success (default: [0])
            scan_id: Scan ID (for audit log attribution when scope violation occurs).
                Pass None for non-scan contexts (CLI tools, scripts).
            actor_id: Agent name or "user" (for audit log). Falls back to
                self.actor_id if None.
            agent_name: W10-S7 — agent name for tool allowlist enforcement.
                If provided, the tool binary (command[0]) must be in the agent's
                tool_allowlist from agent_registry. Otherwise, the call is rejected
                + logged as TOOL_ALLOWLIST_VIOLATION audit event.
                If None, allowlist enforcement is skipped (W9 backward compat —
                used by single-agent scans + scripts).

        Returns:
            ToolResult with stdout, stderr, exit_code
        """
        import time
        start_time = time.time()

        # Normalize command to list
        if isinstance(command, str):
            import shlex
            cmd_list = shlex.split(command)
            cmd_str = command
        else:
            cmd_list = command
            cmd_str = " ".join(command)

        if not cmd_list:
            return ToolResult.error_result("Empty command", cmd_str, target)

        binary = cmd_list[0]

        # ---------- W10-S7: Tool allowlist enforcement ----------
        # If agent_name is provided, validate that the binary is in the agent's
        # tool_allowlist. This is the second layer of defense (the first layer
        # is in BaseAgent._validate_tool_call — this layer catches direct
        # executor.execute() calls that bypass the agent abstraction).
        if agent_name is not None:
            from app.agents.registry import agent_registry
            allowed, reason = agent_registry.validate_tool_call(agent_name, binary)
            if not allowed:
                logger.warning(
                    "TOOL_ALLOWLIST_VIOLATION | scan=%s | agent=%s | binary=%s | reason=%s",
                    scan_id, agent_name, binary, reason,
                )
                await self._log_tool_allowlist_violation(
                    scan_id=scan_id, agent_name=agent_name, binary=binary,
                    command=cmd_str, target=target, reason=reason,
                    actor_id=actor_id or self.actor_id,
                )
                return ToolResult(
                    success=False,
                    error=f"TOOL_ALLOWLIST_VIOLATION: {reason}",
                    command=cmd_str,
                    target=target,
                    duration_seconds=time.time() - start_time,
                    scope_violation=False,
                )

        # ---------- Scope guard validation (W8-A: + audit log) ----------
        validation = self.scope_guard.validate_command(cmd_str, target)
        if not validation.allowed:
            logger.warning("SCOPE_VIOLATION | scan=%s | command=%s | target=%s | reason=%s",
                          scan_id, cmd_str, target, validation.reason)
            await self._log_scope_violation(
                scan_id=scan_id, command=cmd_str, target=target,
                reason=validation.reason, severity=validation.severity,
                actor_id=actor_id or self.actor_id,
            )
            return ToolResult.scope_violation_result(validation.reason, cmd_str, target)

        allowed_exit_codes = allowed_exit_codes or [0]

        # ---------- Build environment ----------
        full_env = self._build_env(env)

        # ---------- Execute via asyncio.create_subprocess_exec ----------
        try:
            result = await self._run_subprocess(
                cmd_list, cmd_str, target, timeout, full_env, workdir,
                start_time, allowed_exit_codes,
            )
            return result

        except asyncio.TimeoutError:
            elapsed = time.time() - start_time
            logger.warning("Command timed out after %ds: %s", timeout, cmd_str)
            return ToolResult(
                success=False,
                error=f"Timeout after {timeout}s",
                command=cmd_str,
                target=target,
                duration_seconds=elapsed,
            )

        except Exception as e:
            elapsed = time.time() - start_time
            logger.error("Command failed: %s | error=%s", cmd_str, e)
            return ToolResult(
                success=False,
                error=str(e),
                command=cmd_str,
                target=target,
                duration_seconds=elapsed,
            )

    # ---------- W7-C: execute_with_hitl ----------

    async def execute_with_hitl(
        self,
        command: list[str] | str,
        target: str,
        tool_name: str,
        tool_args: dict[str, Any],
        scan_id: str,
        agent_reasoning: str,
        predicted_impact: str = "destructive",
        scan_context: dict[str, Any] | None = None,
        timeout: int = DEFAULT_TIMEOUT,
        env: dict[str, str] | None = None,
        workdir: str | None = None,
        allowed_exit_codes: list[int] | None = None,
        db_session: Any = None,
        hitl_manager: Any = None,
    ) -> ToolResult:
        """Execute a command with HITL gate (audit-agent mode by default).

        W7-C: Wraps execute() with the HITL review flow:
            1. Check if HITL is required (via HITLManager.is_hitl_required)
            2. If not required → execute directly (no LLM call, no DB write)
            3. If required → call HITLManager.request_and_wait():
               - audit_agent mode: LLM reviews in ~2-3s, returns decision
               - auto_approve mode: returns approve immediately
               - human_block mode: returns approval_timeout (deferred)
            4. Branch on decision:
               - approve              → execute with original args
               - reject               → return ToolResult.error (tool does NOT run)
               - suggest_alternative  → rebuild command from suggested_args, execute
               - user_aborted         → return ToolResult.error (panic button)
               - approval_timeout     → return ToolResult.error

        Args:
            command: Command list (preferred) or shell string
            target: Target IP/URL (for scope guard)
            tool_name: Tool name for HITL classification (metasploit, sqlmap, etc.)
            tool_args: Tool arguments dict (sent to AuditAgent for review)
            scan_id: Scan ID (for HITLApproval row + SSE events)
            agent_reasoning: Why the agent wants to run this tool
            predicted_impact: destructive / data_modification / data_exfiltration
            scan_context: {declared_scope, blackboard_facts, kg_paths,
                           destructive_ops_count, max_destructive_ops}
            timeout: Subprocess timeout in seconds
            env: Additional env vars
            workdir: Working directory
            allowed_exit_codes: Exit codes treated as success
            db_session: AsyncSession (for HITLApproval row). If None, opens a new session.
            hitl_manager: HITLManager instance (for tests). If None, creates one.

        Returns:
            ToolResult. On HITL reject/timeout/abort, success=False with
            error message explaining the HITL decision.
        """
        # 1. Resolve HITLManager (lazy init if not provided)
        if hitl_manager is None:
            if db_session is None:
                from app.db.session import async_session
                async with async_session() as session:
                    return await self.execute_with_hitl(
                        command=command, target=target, tool_name=tool_name,
                        tool_args=tool_args, scan_id=scan_id,
                        agent_reasoning=agent_reasoning,
                        predicted_impact=predicted_impact,
                        scan_context=scan_context, timeout=timeout,
                        env=env, workdir=workdir,
                        allowed_exit_codes=allowed_exit_codes,
                        db_session=session, hitl_manager=hitl_manager,
                    )
            from app.hitl.manager import HITLManager
            hitl_manager = HITLManager(db_session)

        # 2. Check if HITL is required for this tool
        if not hitl_manager.is_hitl_required(tool_name, tool_args):
            # No HITL needed — execute directly (W8-A: passes scan_id for audit attribution)
            return await self.execute(
                command=command, target=target, timeout=timeout,
                env=env, workdir=workdir, allowed_exit_codes=allowed_exit_codes,
                scan_id=scan_id,
            )

        # 3. HITL required — request + wait for decision
        decision = await hitl_manager.request_and_wait(
            scan_id=scan_id,
            tool_name=tool_name,
            target=target,
            args=tool_args,
            predicted_impact=predicted_impact,
            agent_reasoning=agent_reasoning,
            scan_context=scan_context,
        )

        # Commit the HITLApproval row (caller may also commit, but we want
        # the row persisted even if subprocess fails)
        if db_session is not None:
            try:
                await db_session.commit()
            except Exception as e:
                logger.warning("HITL commit failed (non-fatal): %s", e)

        # 4. Branch on decision
        if decision.decision == "approve":
            # Execute with original args (W8-A: passes scan_id)
            return await self.execute(
                command=command, target=target, timeout=timeout,
                env=env, workdir=workdir, allowed_exit_codes=allowed_exit_codes,
                scan_id=scan_id,
            )

        if decision.decision == "suggest_alternative":
            # Rebuild command from suggested_args + execute
            new_args = decision.suggested_args or {}
            new_command = self._rebuild_command_from_args(command, new_args)
            logger.info(
                "HITL suggested alternative args for %s: %s -> %s",
                tool_name, tool_args, new_args,
            )
            return await self.execute(
                command=new_command, target=target, timeout=timeout,
                env=env, workdir=workdir, allowed_exit_codes=allowed_exit_codes,
                scan_id=scan_id,
            )

        if decision.decision == "reject":
            return ToolResult.error_result(
                f"HITL rejected by {decision.decided_by}: {decision.comment}",
                command=" ".join(command) if isinstance(command, list) else command,
                target=target,
            )

        if decision.decision == "user_aborted":
            return ToolResult.error_result(
                f"HITL aborted by user: {decision.comment}",
                command=" ".join(command) if isinstance(command, list) else command,
                target=target,
            )

        if decision.decision == "approval_timeout":
            return ToolResult.error_result(
                f"HITL approval timeout: {decision.comment}",
                command=" ".join(command) if isinstance(command, list) else command,
                target=target,
            )

        # Unknown decision — fail safe
        return ToolResult.error_result(
            f"HITL unknown decision: {decision.decision} ({decision.comment})",
            command=" ".join(command) if isinstance(command, list) else command,
            target=target,
        )

    def _rebuild_command_from_args(
        self,
        original_command: list[str] | str,
        new_args: dict[str, Any],
    ) -> list[str]:
        """Rebuild a command list from suggested args.

        W7-C: preserves the binary + positional args from the original command,
        then appends/overrides flags from new_args. For YAML tools, the caller
        should rebuild via ToolDef.build_command_args() for proper schema-aware
        rebuilding. This method handles the common case.

        Strategy:
            - Keep binary (first element)
            - Keep positional args (non-flag args that aren't --key=value)
            - Drop original flags that are being overridden by new_args
            - Append new_args as --key value flags
        """
        if isinstance(original_command, str):
            import shlex
            cmd_list = shlex.split(original_command)
        else:
            cmd_list = list(original_command)

        if not cmd_list:
            return cmd_list

        binary = cmd_list[0]
        new_cmd = [binary]

        # Walk the original command, preserving positionals + dropping overridden flags
        new_arg_keys = set(new_args.keys())
        i = 1
        while i < len(cmd_list):
            arg = cmd_list[i]
            # Check if this is a flag (--key or -k)
            if arg.startswith("--"):
                key = arg[2:].split("=")[0]  # handle --key=value
                if key in new_arg_keys:
                    # Skip this flag (and its value if separate)
                    if "=" not in arg and i + 1 < len(cmd_list) and not cmd_list[i + 1].startswith("-"):
                        i += 2
                        continue
                    i += 1
                    continue
                new_cmd.append(arg)
                # If --key value (separate), append value too
                if "=" not in arg and i + 1 < len(cmd_list) and not cmd_list[i + 1].startswith("-"):
                    new_cmd.append(cmd_list[i + 1])
                    i += 2
                    continue
                i += 1
            elif arg.startswith("-") and len(arg) > 1 and not arg[1:].isdigit():
                # Short flag -k (but not negative number like -1)
                key = arg[1:]
                if key in new_arg_keys:
                    if i + 1 < len(cmd_list) and not cmd_list[i + 1].startswith("-"):
                        i += 2
                        continue
                    i += 1
                    continue
                new_cmd.append(arg)
                if i + 1 < len(cmd_list) and not cmd_list[i + 1].startswith("-"):
                    new_cmd.append(cmd_list[i + 1])
                    i += 2
                    continue
                i += 1
            else:
                # Positional arg — keep it
                new_cmd.append(arg)
                i += 1

        # Append new_args as flags
        for key, value in new_args.items():
            if value is None or value is False:
                continue
            if value is True:
                new_cmd.append(f"--{key}")
            elif len(key) == 1:
                new_cmd.extend([f"-{key}", str(value)])
            else:
                new_cmd.extend([f"--{key}", str(value)])

        return new_cmd

    async def _run_subprocess(
        self,
        cmd_list: list[str],
        cmd_str: str,
        target: str,
        timeout: int,
        env: dict[str, str],
        workdir: str | None,
        start_time: float,
        allowed_exit_codes: list[int],
    ) -> ToolResult:
        """Run subprocess with output capture + timeout + PTY retry."""
        import time

        # ---------- First attempt: regular subprocess ----------
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd_list,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                stdin=asyncio.subprocess.DEVNULL,
                start_new_session=True,  # setsid — process group isolation
                env=env,
                cwd=workdir,
            )
        except FileNotFoundError:
            return ToolResult.error_result(
                f"Binary not found: {cmd_list[0]}", cmd_str, target,
            )

        # Wait for completion with timeout
        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            # Kill entire process group
            self._kill_process_group(proc.pid)
            raise

        elapsed = time.time() - start_time

        stdout = stdout_bytes.decode("utf-8", errors="replace") if stdout_bytes else ""
        stderr = stderr_bytes.decode("utf-8", errors="replace") if stderr_bytes else ""

        # ---------- Check if PTY retry needed ----------
        if self._should_retry_with_pty(stderr):
            logger.info("PTY retry triggered for: %s", cmd_list[0])
            return await self._run_with_pty(
                cmd_list, cmd_str, target, timeout, env, workdir,
                start_time, allowed_exit_codes,
            )

        # ---------- Output cap + spill ----------
        stdout, truncated, spill_path = self._cap_output(stdout, cmd_str)

        # ---------- Build result ----------
        exit_code = proc.returncode
        success = exit_code in allowed_exit_codes

        return ToolResult(
            success=success,
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
            truncated=truncated,
            spill_path=spill_path,
            duration_seconds=elapsed,
            command=cmd_str,
            target=target,
        )

    async def _run_with_pty(
        self,
        cmd_list: list[str],
        cmd_str: str,
        target: str,
        timeout: int,
        env: dict[str, str],
        workdir: str | None,
        start_time: float,
        allowed_exit_codes: list[int],
    ) -> ToolResult:
        """Run command with PTY (for tools that demand TTY)."""
        import time

        # Create PTY
        master_fd, slave_fd = pty.openpty()

        # Set window size (some tools need it)
        try:
            winsize = struct.pack("HHHH", 40, 256, 0, 0)
            fcntl.ioctl(slave_fd, termios.TIOCSWINSZ, winsize)
        except Exception:
            pass

        # Start process with slave as stdin/stdout/stderr
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd_list,
                stdout=slave_fd,
                stderr=slave_fd,
                stdin=slave_fd,
                start_new_session=True,
                env=env,
                cwd=workdir,
            )
        finally:
            os.close(slave_fd)  # parent doesn't need slave

        # Read from master with timeout
        output_bytes = b""
        try:
            while True:
                try:
                    ready, _, _ = select.select([master_fd], [], [], timeout)
                    if not ready:
                        break  # timeout
                    chunk = os.read(master_fd, 4096)
                    if not chunk:
                        break
                    output_bytes += chunk
                    # Cap output
                    if len(output_bytes) > self.output_max_bytes * 2:
                        break
                except OSError:
                    break
        finally:
            os.close(master_fd)

        # Wait for process to finish
        try:
            await asyncio.wait_for(proc.wait(), timeout=5)
        except asyncio.TimeoutError:
            self._kill_process_group(proc.pid)

        elapsed = time.time() - start_time
        output = output_bytes.decode("utf-8", errors="replace")

        # Normalize \r\n → \n (PTY adds \r)
        output = output.replace("\r\n", "\n").replace("\r", "")

        # Cap output
        output, truncated, spill_path = self._cap_output(output, cmd_str)

        exit_code = proc.returncode
        success = exit_code in allowed_exit_codes if exit_code is not None else False

        return ToolResult(
            success=success,
            exit_code=exit_code,
            stdout=output,
            stderr="",  # PTY merges stdout+stderr
            truncated=truncated,
            spill_path=spill_path,
            duration_seconds=elapsed,
            command=cmd_str,
            target=target,
        )

    # ---------- W8-A: Audit log helpers ----------

    async def _log_scope_violation(
        self,
        scan_id: str | None,
        command: str,
        target: str,
        reason: str,
        severity: str = "error",
        actor_id: str = "agent",
    ) -> None:
        """Persist scope_violation event to audit log (best-effort, never raises).

        If no audit_logger was injected (None), this is a no-op.
        If audit_logger raises (DB error, etc.), we swallow the exception
        because the scope guard has already returned a ToolResult — the
        caller (agent) should not see audit-log failures as a tool failure.

        Writes entry to vapt_audit_log with:
            actor_type = "agent"
            actor_id   = agent name (e.g. "recon", "penetration")
            action     = "scope_violation"
            target_table = "vapt_scans"
            target_id  = scan_id (or None for non-scan context)
            after_json = {command, target, reason, severity}
            scan_id    = scan_id
        """
        if self.audit_logger is None:
            return
        try:
            await self.audit_logger.log(
                actor_type="agent",
                actor_id=actor_id,
                action="scope_violation",
                target_table="vapt_scans",
                target_id=scan_id,
                before=None,
                after={
                    "command": command[:500],  # cap for audit log
                    "target": target,
                    "reason": reason,
                    "severity": severity,
                },
                scan_id=scan_id,
            )
        except Exception as audit_err:
            logger.warning(
                "Failed to persist scope_violation audit entry (non-fatal): %s",
                audit_err,
            )

    async def _log_tool_allowlist_violation(
        self,
        scan_id: str | None,
        agent_name: str,
        binary: str,
        command: str,
        target: str,
        reason: str,
        actor_id: str = "agent",
    ) -> None:
        """Persist tool_allowlist_violation event to audit log (W10-S7).

        Best-effort — never raises. Same pattern as _log_scope_violation().

        Writes entry to vapt_audit_log with:
            actor_type = "agent"
            actor_id   = agent_name
            action     = "tool_allowlist_violation"
            target_table = "vapt_scans"
            target_id  = scan_id (or None for non-scan context)
            after_json = {agent_name, binary, command, target, reason}
            scan_id    = scan_id
        """
        if self.audit_logger is None:
            return
        try:
            await self.audit_logger.log(
                actor_type="agent",
                actor_id=actor_id,
                action="tool_allowlist_violation",
                target_table="vapt_scans",
                target_id=scan_id,
                before=None,
                after={
                    "agent_name": agent_name,
                    "binary": binary,
                    "command": command[:500],
                    "target": target,
                    "reason": reason,
                },
                scan_id=scan_id,
            )
        except Exception as audit_err:
            logger.warning(
                "Failed to persist tool_allowlist_violation audit entry (non-fatal): %s",
                audit_err,
            )

    # ---------- Helpers ----------

    def _build_env(self, extra: dict[str, str] | None = None) -> dict[str, str]:
        """Build environment for subprocess — inherits current + adds pager overrides.

        CRITICAL — Phase D PATH fix:
        When VAPT-AI runs as a systemd service (User=nhat), the systemd
        default PATH is minimal (`/usr/local/sbin:/usr/local/bin:/usr/sbin:/
        usr/bin:/sbin:/bin`). Go-installed security tools (httpx, nuclei,
        subfinder, katana, dalfox, gau, waybackurls, ...) live in
        `$HOME/go/bin` (or `/root/go/bin` when sudo-installed), which is
        NOT in the systemd PATH. As a result the agent saw "Binary not
        found: httpx" even though the user HAD installed httpx.

        We explicitly expand PATH to include every common install
        location for the security tools the user installed:
          - $HOME/go/bin             — go install (per-user)
          - /root/go/bin             — go install via sudo
          - /usr/local/go/bin        — go binary itself (for `go run`)
          - /usr/local/bin           — manually placed tools (e.g. mimikatz)
          - /snap/bin                — snap-installed tools
          - /opt/go/bin              — alternate go install location
          - $GOPATH/bin              — env-var go install location
          - the current PATH         — fallback so we never LOSE access
        """
        env = dict(os.environ)

        # Build the expanded PATH. Order matters — leftmost wins.
        home = env.get("HOME", "/root")
        go_path = env.get("GOPATH", f"{home}/go")
        extra_paths = [
            f"{home}/go/bin",
            f"{go_path}/bin",
            "/root/go/bin",
            "/usr/local/go/bin",
            "/usr/local/sbin",
            "/usr/local/bin",
            "/usr/sbin",
            "/usr/bin",
            "/sbin",
            "/bin",
            "/snap/bin",
            "/opt/go/bin",
            "/opt/nmap/bin",
            # Common manual install locations
            "/opt/feroxbuster",
            "/opt/sqlmap",
        ]
        # Append the existing PATH last so we never lose access to anything
        # that's already on the default systemd PATH.
        existing_path = env.get("PATH", "")
        seen: set[str] = set()
        unique_paths: list[str] = []
        for p in extra_paths:
            if p and p not in seen:
                seen.add(p)
                unique_paths.append(p)
        if existing_path:
            for p in existing_path.split(":"):
                if p and p not in seen:
                    seen.add(p)
                    unique_paths.append(p)
        env["PATH"] = ":".join(unique_paths)

        # Pager overrides (prevent tools from blocking on less/more)
        env["GIT_PAGER"] = "cat"
        env["PAGER"] = "cat"
        env["SYSTEMD_PAGER"] = "cat"
        env["DEBIAN_FRONTEND"] = "noninteractive"
        env["TERM"] = "xterm-256color"

        # Merge extra env (caller overrides win)
        if extra:
            env.update(extra)

        return env

    def _should_retry_with_pty(self, stderr: str) -> bool:
        """Check if stderr indicates TTY-demand error (triggers PTY retry)."""
        if not stderr:
            return False
        for pattern in TTY_DEMAND_PATTERNS:
            if pattern.search(stderr):
                return True
        return False

    def _cap_output(self, output: str, command: str) -> tuple[str, bool, str | None]:
        """Cap output to max_bytes. Spill to disk if exceeded.

        Returns (capped_output, truncated, spill_path).
        """
        output_bytes = output.encode("utf-8")
        if len(output_bytes) <= self.output_max_bytes:
            return output, False, None

        # Spill to disk
        SPILL_DIR.mkdir(parents=True, exist_ok=True)
        import uuid
        spill_file = SPILL_DIR / f"output_{uuid.uuid4().hex[:8]}.txt"
        spill_file.write_bytes(output_bytes)

        # Return truncated output + spill path
        truncated = output[:self.output_max_bytes].encode("utf-8", errors="replace").decode("utf-8", errors="replace")
        truncated += f"\n...[TRUNCATED — full output ({len(output_bytes)} bytes) spilled to {spill_file}]"

        logger.info("Output spilled to disk: %s (%d bytes)", spill_file, len(output_bytes))
        return truncated, True, str(spill_file)

    def _kill_process_group(self, pid: int) -> None:
        """Kill entire process group (SIGKILL — cannot be caught)."""
        try:
            os.killpg(os.getpgid(pid), signal.SIGKILL)
            logger.debug("Killed process group: pid=%d", pid)
        except ProcessLookupError:
            pass  # already dead
        except Exception as e:
            logger.warning("Failed to kill process group %d: %s", pid, e)