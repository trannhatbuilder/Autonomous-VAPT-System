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
    ):
        self.scope_guard = scope_guard or ScopeGuard()
        self.output_max_bytes = output_max_bytes

    async def execute(
        self,
        command: list[str] | str,
        target: str = "",
        timeout: int = DEFAULT_TIMEOUT,
        env: dict[str, str] | None = None,
        workdir: str | None = None,
        allowed_exit_codes: list[int] | None = None,
    ) -> ToolResult:
        """Execute a command via subprocess.

        Args:
            command: Command list (preferred) or shell string
            target: Target IP/URL (for scope guard validation)
            timeout: Timeout in seconds
            env: Additional environment variables
            workdir: Working directory
            allowed_exit_codes: Exit codes treated as success (default: [0])

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

        # ---------- Scope guard validation ----------
        validation = self.scope_guard.validate_command(cmd_str, target)
        if not validation.allowed:
            logger.warning("SCOPE_VIOLATION | command=%s | target=%s | reason=%s",
                          cmd_str, target, validation.reason)
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

    # ---------- Helpers ----------

    def _build_env(self, extra: dict[str, str] | None = None) -> dict[str, str]:
        """Build environment for subprocess — inherits current + adds pager overrides."""
        env = dict(os.environ)

        # Pager overrides (prevent tools from blocking on less/more)
        env["GIT_PAGER"] = "cat"
        env["PAGER"] = "cat"
        env["SYSTEMD_PAGER"] = "cat"
        env["DEBIAN_FRONTEND"] = "noninteractive"
        env["TERM"] = "xterm-256color"

        # Merge extra env
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
