"""
W8-A tests — verify scope guard audit log persistence in executor.

Test cases:
1. execute() with out-of-scope target → ToolResult.scope_violation=True,
   AND audit_logger.log called with action="scope_violation"
2. execute() with in-scope target → no audit log entry written
3. execute() without audit_logger injected → still returns scope_violation
   result (graceful degradation — no error from audit being None)
4. execute() with audit_logger that raises → ToolResult still returns
   scope_violation (audit failure is swallowed, doesn't block tool result)
5. execute_with_hitl() passes scan_id through to execute() for audit attribution
6. execute() with scan_id=None (non-scan context) → audit entry has target_id=None

Run:
    pytest tests/vapt/test_w8a_scope_guard_audit.py -v

These tests use mocks — no actual subprocess or DB needed.
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path

import pytest

# Ensure project root on sys.path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app.sandbox.executor import SubprocessExecutor, ToolResult
from app.sandbox.scope_guard import ScopeGuard, ScopeRule


# ---------- Fixtures ----------

@pytest.fixture
def in_scope_guard() -> ScopeGuard:
    """Scope guard that allows 192.168.1.0/24 + example.com."""
    return ScopeGuard(declared_scope=[
        ScopeRule(cidr="192.168.1.0/24"),
        ScopeRule(host="example.com"),
        ScopeRule(host="*.example.com"),
    ])


@pytest.fixture
def mock_audit_logger() -> AsyncMock:
    """Mock AuditLogger — captures .log() calls."""
    logger = AsyncMock()
    # AuditLogger.log is async — returns a coroutine that resolves to a MagicMock
    logger.log = AsyncMock(return_value=MagicMock())
    return logger


@pytest.fixture
def executor_with_audit(in_scope_guard, mock_audit_logger) -> SubprocessExecutor:
    """Executor with audit_logger injected."""
    return SubprocessExecutor(
        scope_guard=in_scope_guard,
        audit_logger=mock_audit_logger,
        actor_id="recon",
    )


@pytest.fixture
def executor_no_audit(in_scope_guard) -> SubprocessExecutor:
    """Executor WITHOUT audit_logger (legacy mode)."""
    return SubprocessExecutor(scope_guard=in_scope_guard)


# ---------- Test 1: scope violation writes audit log ----------

@pytest.mark.asyncio
async def test_scope_violation_writes_audit_log(executor_with_audit, mock_audit_logger):
    """Out-of-scope target → ToolResult.scope_violation=True AND audit log written."""
    result = await executor_with_audit.execute(
        command=["nmap", "-sS", "10.0.0.99"],  # not in 192.168.1.0/24
        target="10.0.0.99",
        scan_id="scan_test_001",
        actor_id="recon",
    )

    # ToolResult is scope violation
    assert result.scope_violation is True
    assert result.success is False
    assert "Scope violation" in result.error
    assert "10.0.0.99" in result.error or "not in declared scope" in result.error

    # Audit log was called with correct fields
    mock_audit_logger.log.assert_called_once()
    call_kwargs = mock_audit_logger.log.call_args.kwargs
    assert call_kwargs["actor_type"] == "agent"
    assert call_kwargs["actor_id"] == "recon"
    assert call_kwargs["action"] == "scope_violation"
    assert call_kwargs["target_table"] == "vapt_scans"
    assert call_kwargs["target_id"] == "scan_test_001"
    assert call_kwargs["scan_id"] == "scan_test_001"
    assert call_kwargs["before"] is None
    after = call_kwargs["after"]
    assert after["command"] == "nmap -sS 10.0.0.99"
    assert after["target"] == "10.0.0.99"
    # Reason is one of: "not in declared scope" OR "SSRF blocked" (private IP)
    # 10.0.0.99 falls in private 10.0.0.0/8 → triggers SSRF block
    assert (
        "not in declared scope" in after["reason"]
        or "SSRF" in after["reason"]
        or "private/loopback/metadata" in after["reason"]
    ), f"Unexpected reason: {after['reason']}"
    assert after["severity"] == "error"


# ---------- Test 2: in-scope target does NOT write audit log ----------

@pytest.mark.asyncio
async def test_in_scope_no_audit_log(executor_with_audit, mock_audit_logger):
    """In-scope target → no audit log entry (no violation to record)."""
    # Patch _run_subprocess to avoid actually running nmap
    with patch.object(executor_with_audit, "_run_subprocess", new_callable=AsyncMock) as mock_run:
        mock_run.return_value = ToolResult(
            success=True, exit_code=0, stdout="Nmap done", stderr="",
            command="nmap -sS 192.168.1.5", target="192.168.1.5",
        )
        result = await executor_with_audit.execute(
            command=["nmap", "-sS", "192.168.1.5"],
            target="192.168.1.5",
            scan_id="scan_test_002",
        )

    assert result.success is True
    assert result.scope_violation is False
    # No audit log entry should be written
    mock_audit_logger.log.assert_not_called()


# ---------- Test 3: no audit_logger injected → still works ----------

@pytest.mark.asyncio
async def test_no_audit_logger_graceful_degradation(executor_no_audit):
    """Executor without audit_logger still returns scope_violation result."""
    result = await executor_no_audit.execute(
        command=["nmap", "-sS", "10.0.0.99"],
        target="10.0.0.99",
        scan_id="scan_test_003",
    )
    assert result.scope_violation is True
    assert result.success is False
    assert "Scope violation" in result.error


# ---------- Test 4: audit_logger raises → ToolResult still returned ----------

@pytest.mark.asyncio
async def test_audit_logger_failure_swallowed(in_scope_guard):
    """If audit_logger.log raises, executor still returns scope_violation ToolResult."""
    failing_audit = AsyncMock()
    failing_audit.log = AsyncMock(side_effect=RuntimeError("DB connection lost"))

    executor = SubprocessExecutor(
        scope_guard=in_scope_guard,
        audit_logger=failing_audit,
        actor_id="penetration",
    )

    # Should not raise — exception is swallowed
    result = await executor.execute(
        command=["nmap", "-sS", "10.0.0.99"],
        target="10.0.0.99",
        scan_id="scan_test_004",
        actor_id="penetration",
    )

    # Tool result still returned correctly
    assert result.scope_violation is True
    assert result.success is False
    # Audit logger was attempted (even though it failed)
    failing_audit.log.assert_called_once()


# ---------- Test 5: scan_id=None for non-scan context ----------

@pytest.mark.asyncio
async def test_scan_id_none_audit_log(executor_with_audit, mock_audit_logger):
    """scan_id=None (CLI usage, no scan context) → audit entry still written with target_id=None."""
    result = await executor_with_audit.execute(
        command=["nmap", "-sS", "10.0.0.99"],
        target="10.0.0.99",
        scan_id=None,
        actor_id="cli_user",
    )

    assert result.scope_violation is True
    mock_audit_logger.log.assert_called_once()
    call_kwargs = mock_audit_logger.log.call_args.kwargs
    assert call_kwargs["target_id"] is None
    assert call_kwargs["scan_id"] is None
    assert call_kwargs["actor_id"] == "cli_user"


# ---------- Test 6: actor_id fallback ----------

@pytest.mark.asyncio
async def test_actor_id_fallback_to_self(in_scope_guard):
    """If actor_id not passed to execute(), falls back to self.actor_id."""
    mock_audit = AsyncMock()
    mock_audit.log = AsyncMock(return_value=MagicMock())

    executor = SubprocessExecutor(
        scope_guard=in_scope_guard,
        audit_logger=mock_audit,
        actor_id="vulnerability_triage",  # default
    )

    # Don't pass actor_id — should use self.actor_id
    await executor.execute(
        command=["nmap", "-sS", "10.0.0.99"],
        target="10.0.0.99",
        scan_id="scan_test_006",
    )

    mock_audit.log.assert_called_once()
    call_kwargs = mock_audit.log.call_args.kwargs
    assert call_kwargs["actor_id"] == "vulnerability_triage"


# ---------- Test 7: SSRF violation also writes audit log ----------

@pytest.mark.asyncio
async def test_ssrf_violation_writes_audit_log(executor_with_audit, mock_audit_logger):
    """SSRF-blocked target (private IP without scope rule) → audit entry written."""
    # 169.254.169.254 is cloud metadata IP — blocked by SSRF guard
    # (no scope rule matches, SSRF blocks it)
    result = await executor_with_audit.execute(
        command=["curl", "http://169.254.169.254/latest/meta-data/"],
        target="169.254.169.254",
        scan_id="scan_test_007",
        actor_id="recon",
    )

    assert result.scope_violation is True
    mock_audit_logger.log.assert_called_once()
    call_kwargs = mock_audit_logger.log.call_args.kwargs
    assert call_kwargs["action"] == "scope_violation"
    after = call_kwargs["after"]
    assert "SSRF" in after["reason"] or "private/loopback/metadata" in after["reason"]


# ---------- Test 8: destructive command also writes audit log ----------

@pytest.mark.asyncio
async def test_destructive_command_writes_audit_log(executor_with_audit, mock_audit_logger):
    """Destructive command (rm -rf /) → audit entry written with destructive reason."""
    result = await executor_with_audit.execute(
        command="rm -rf /",
        target="192.168.1.5",  # in-scope target, but command is destructive
        scan_id="scan_test_008",
        actor_id="cleanup_rollback",
    )

    assert result.scope_violation is True
    mock_audit_logger.log.assert_called_once()
    after = mock_audit_logger.log.call_args.kwargs["after"]
    assert "Destructive" in after["reason"] or "destructive" in after["reason"].lower()


# ---------- Test 9: execute_with_hitl passes scan_id to execute ----------

@pytest.mark.asyncio
async def test_execute_with_hitl_passes_scan_id(in_scope_guard, mock_audit_logger):
    """execute_with_hitl() with non-HITL tool → execute() called with scan_id."""
    executor = SubprocessExecutor(
        scope_guard=in_scope_guard,
        audit_logger=mock_audit_logger,
    )

    # Mock HITLManager — nmap is not destructive so HITL not required
    mock_hitl = MagicMock()
    mock_hitl.is_hitl_required.return_value = False

    # Mock _run_subprocess to avoid actual nmap execution
    with patch.object(executor, "_run_subprocess", new_callable=AsyncMock) as mock_run:
        mock_run.return_value = ToolResult(
            success=True, exit_code=0, stdout="done", stderr="",
            command="nmap -sS 192.168.1.5", target="192.168.1.5",
        )
        result = await executor.execute_with_hitl(
            command=["nmap", "-sS", "192.168.1.5"],
            target="192.168.1.5",
            tool_name="nmap",
            tool_args={"target": "192.168.1.5"},
            scan_id="scan_hitl_test",
            agent_reasoning="Port scan for recon",
        )

    assert result.success is True
    # No audit log written (in-scope, no violation)
    mock_audit_logger.log.assert_not_called()


# ---------- Test 10: long command is truncated in audit log ----------

@pytest.mark.asyncio
async def test_long_command_truncated_in_audit(in_scope_guard):
    """Command >500 chars is truncated to 500 in audit log 'command' field."""
    mock_audit = AsyncMock()
    mock_audit.log = AsyncMock(return_value=MagicMock())

    executor = SubprocessExecutor(
        scope_guard=in_scope_guard,
        audit_logger=mock_audit,
        actor_id="recon",
    )

    # Build a long command (nmap with many args) targeting out-of-scope IP
    long_cmd = "nmap " + " ".join([f"--script script_{i}" for i in range(100)]) + " 10.0.0.99"
    assert len(long_cmd) > 500

    await executor.execute(
        command=long_cmd,
        target="10.0.0.99",
        scan_id="scan_test_long",
    )

    mock_audit.log.assert_called_once()
    after = mock_audit.log.call_args.kwargs["after"]
    assert len(after["command"]) <= 500


if __name__ == "__main__":
    # Run tests directly when executed as a script
    pytest.main([__file__, "-v", "--tb=short"])
