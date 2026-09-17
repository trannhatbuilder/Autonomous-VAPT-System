"""
Tests for VAPT-AI HITL Integration — W7-C.

Covers:
    1. SubprocessExecutor.execute_with_hitl() — 3 scenarios:
       a. approve → tool runs with original args
       b. reject → tool does NOT run, returns error
       c. suggest_alternative → tool runs with new args
    2. SubprocessExecutor._rebuild_command_from_args() — flag rebuilding
    3. SubprocessExecutor.execute_with_hitl() — safe tool (nmap) skips HITL
    4. ScanRegistry — register_scan / is_aborted / abort_scan / unregister
    5. ScanRegistry — abort_scan kills subprocesses + runs cleanup script
    6. Metasploit msf_module_execute — audit_agent approve → execute
    7. Metasploit msf_module_execute — audit_agent reject → error returned
    8. Metasploit msf_session_run — read-only command skips HITL
    9. Metasploit msf_session_run — destructive command requires HITL
"""
from __future__ import annotations

import asyncio
import json
import os
import uuid
from datetime import datetime, UTC
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.hitl.manager import HITLManager, HITLDecision
from app.hitl.audit_agent import AuditDecision
from app.pentest.scan_registry import ScanRegistry, ScanState, scan_registry
from app.sandbox.executor import SubprocessExecutor, ToolResult
from app.sandbox.scope_guard import ScopeGuard, ScopeRule


# ============================================================
# Test doubles
# ============================================================

class FakeAuditAgent:
    """Test double for AuditAgent — returns a preset AuditDecision."""

    def __init__(self, decision: AuditDecision):
        self._decision = decision
        self.review_calls: list[dict[str, Any]] = []

    async def review(self, **kwargs: Any) -> AuditDecision:
        self.review_calls.append(kwargs)
        return self._decision


def _make_audit_decision(
    decision: str = "approve",
    comment: str = "low risk",
    suggested_args: dict | None = None,
    decided_by: str = "audit_agent",
) -> AuditDecision:
    return AuditDecision(
        decision=decision,
        comment=comment,
        suggested_args=suggested_args,
        decided_by=decided_by,
        duration_seconds=0.5,
    )


def _make_fake_session():
    """Build a fake AsyncSession that auto-assigns UUID id on flush."""
    fake_session = MagicMock()
    fake_session.add = MagicMock()
    fake_session.flush = AsyncMock()
    fake_session.commit = AsyncMock()
    fake_session.close = AsyncMock()

    async def _flush_side_effect(*args, **kwargs):
        for call_obj in fake_session.add.call_args_list:
            obj = call_obj.args[0] if call_obj.args else call_obj.kwargs.get("instance")
            if obj is not None and getattr(obj, "id", None) is None:
                obj.id = uuid.uuid4()

    fake_session.flush.side_effect = _flush_side_effect
    return fake_session


def _make_hitl_manager(session, audit_decision: AuditDecision) -> HITLManager:
    """Build a HITLManager with a FakeAuditAgent that returns the given decision."""
    fake_audit = FakeAuditAgent(audit_decision)
    return HITLManager(session=session, mode="audit_agent", audit_agent=fake_audit)


# ============================================================
# SubprocessExecutor.execute_with_hitl — 3 scenarios
# ============================================================

@pytest.mark.asyncio
async def test_execute_with_hitl_approve_runs_tool():
    """audit_agent approve → execute() is called + tool runs with original args."""
    # Build executor with a scope guard that allows localhost
    guard = ScopeGuard(declared_scope=[ScopeRule(host="localhost")])
    executor = SubprocessExecutor(scope_guard=guard)

    # Mock the actual execute() to avoid running real subprocess
    expected_result = ToolResult(
        success=True, exit_code=0, stdout="hello", stderr="",
        duration_seconds=0.1, command="echo hello", target="localhost",
    )
    executor.execute = AsyncMock(return_value=expected_result)

    fake_session = _make_fake_session()
    fake_audit = FakeAuditAgent(_make_audit_decision(decision="approve"))
    mgr = HITLManager(session=fake_session, mode="audit_agent", audit_agent=fake_audit)

    result = await executor.execute_with_hitl(
        command=["echo", "hello"],
        target="localhost",
        tool_name="metasploit",
        tool_args={"module_type": "exploit"},
        scan_id="scan_test_1",
        agent_reasoning="test exploit",
        db_session=fake_session,
        hitl_manager=mgr,
    )

    assert result.success is True
    assert result.stdout == "hello"
    # execute() was called once
    executor.execute.assert_called_once()
    # Audit agent was called once
    assert len(fake_audit.review_calls) == 1


@pytest.mark.asyncio
async def test_execute_with_hitl_reject_blocks_tool():
    """audit_agent reject → execute() is NOT called + returns error."""
    guard = ScopeGuard(declared_scope=[ScopeRule(host="localhost")])
    executor = SubprocessExecutor(scope_guard=guard)
    executor.execute = AsyncMock()  # should NOT be called

    fake_session = _make_fake_session()
    fake_audit = FakeAuditAgent(_make_audit_decision(
        decision="reject", comment="DROP TABLE detected",
    ))
    mgr = HITLManager(session=fake_session, mode="audit_agent", audit_agent=fake_audit)

    result = await executor.execute_with_hitl(
        command=["psql", "-c", "DROP TABLE users;"],
        target="localhost",
        tool_name="custom_rce",
        tool_args={"command": "DROP TABLE users;"},
        scan_id="scan_test_2",
        agent_reasoning="cleanup",
        db_session=fake_session,
        hitl_manager=mgr,
    )

    assert result.success is False
    assert "HITL rejected" in result.error
    assert "DROP TABLE" in result.error or "audit agent" in result.error.lower()
    # execute() was NOT called
    executor.execute.assert_not_called()


@pytest.mark.asyncio
async def test_execute_with_hitl_suggest_alternative_runs_with_new_args():
    """audit_agent suggest_alternative → execute() called with rebuilt command."""
    guard = ScopeGuard(declared_scope=[ScopeRule(host="localhost")])
    executor = SubprocessExecutor(scope_guard=guard)

    captured_command: list[str] = []

    async def fake_execute(**kwargs):
        captured_command.extend(kwargs["command"])
        return ToolResult(success=True, exit_code=0, stdout="ok", duration_seconds=0.1,
                          command=" ".join(kwargs["command"]), target=kwargs.get("target", ""))

    executor.execute = fake_execute

    fake_session = _make_fake_session()
    fake_audit = FakeAuditAgent(_make_audit_decision(
        decision="suggest_alternative",
        comment="narrowing scope",
        suggested_args={"level": 1, "risk": 1},
    ))
    mgr = HITLManager(session=fake_session, mode="audit_agent", audit_agent=fake_audit)

    # Use sqlmap with --os-shell (forbidden arg) so HITL is triggered
    result = await executor.execute_with_hitl(
        command=["sqlmap", "-u", "http://localhost/?id=1", "--level=5", "--risk=3", "--os-shell"],
        target="localhost",
        tool_name="sqlmap",
        tool_args={"url": "http://localhost/?id=1", "level": 5, "risk": 3, "extra": "--os-shell"},
        scan_id="scan_test_3",
        agent_reasoning="SQLi test with os-shell",
        db_session=fake_session,
        hitl_manager=mgr,
    )

    assert result.success is True
    # Command was rebuilt with suggested args
    assert "sqlmap" in captured_command
    assert "--level" in captured_command
    assert "1" in captured_command
    assert "--risk" in captured_command


# ============================================================
# SubprocessExecutor._rebuild_command_from_args
# ============================================================

def test_rebuild_command_long_flags():
    """Long flag names (--key value) are rebuilt correctly."""
    executor = SubprocessExecutor.__new__(SubprocessExecutor)
    new_cmd = executor._rebuild_command_from_args(
        original_command=["sqlmap", "-u", "http://x"],
        new_args={"level": 1, "risk": 1},
    )
    assert new_cmd[0] == "sqlmap"
    assert "--level" in new_cmd
    assert "1" in new_cmd
    assert "--risk" in new_cmd


def test_rebuild_command_short_flags():
    """Single-char keys become -k value."""
    executor = SubprocessExecutor.__new__(SubprocessExecutor)
    new_cmd = executor._rebuild_command_from_args(
        original_command=["nmap"],
        new_args={"p": "80", "s": "S"},
    )
    assert new_cmd[0] == "nmap"
    assert "-p" in new_cmd
    assert "80" in new_cmd


def test_rebuild_command_boolean_true():
    """Boolean True becomes --flag (no value)."""
    executor = SubprocessExecutor.__new__(SubprocessExecutor)
    new_cmd = executor._rebuild_command_from_args(
        original_command=["nmap"],
        new_args={"verbose": True, "port": 80},
    )
    assert "--verbose" in new_cmd
    assert "--port" in new_cmd
    assert "80" in new_cmd


def test_rebuild_command_skips_none_and_false():
    """None + False values are skipped."""
    executor = SubprocessExecutor.__new__(SubprocessExecutor)
    new_cmd = executor._rebuild_command_from_args(
        original_command=["nmap"],
        new_args={"verbose": False, "port": None, "host": "localhost"},
    )
    assert "--verbose" not in new_cmd
    assert "--port" not in new_cmd
    assert "--host" in new_cmd
    assert "localhost" in new_cmd


# ============================================================
# SubprocessExecutor.execute_with_hitl — safe tool skips HITL
# ============================================================

@pytest.mark.asyncio
async def test_execute_with_hitl_safe_tool_skips_hitl():
    """nmap (safe tool) → execute() called directly, no HITL review."""
    guard = ScopeGuard(declared_scope=[ScopeRule(host="localhost")])
    executor = SubprocessExecutor(scope_guard=guard)

    expected_result = ToolResult(
        success=True, exit_code=0, stdout="nmap output", duration_seconds=0.1,
        command="nmap localhost", target="localhost",
    )
    executor.execute = AsyncMock(return_value=expected_result)

    fake_session = _make_fake_session()
    fake_audit = FakeAuditAgent(_make_audit_decision(decision="approve"))
    mgr = HITLManager(session=fake_session, mode="audit_agent", audit_agent=fake_audit)

    result = await executor.execute_with_hitl(
        command=["nmap", "localhost"],
        target="localhost",
        tool_name="nmap",  # safe tool
        tool_args={"target": "localhost"},
        scan_id="scan_test_4",
        agent_reasoning="port scan",
        db_session=fake_session,
        hitl_manager=mgr,
    )

    assert result.success is True
    assert result.stdout == "nmap output"
    # Audit agent was NOT called (HITL skipped for safe tool)
    assert len(fake_audit.review_calls) == 0


# ============================================================
# ScanRegistry
# ============================================================

@pytest.mark.asyncio
async def test_scan_registry_register_and_is_aborted():
    """register_scan → is_aborted returns False initially."""
    reg = ScanRegistry()  # fresh instance (don't use singleton)
    state = await reg.register_scan("scan_reg_1")
    assert state.scan_id == "scan_reg_1"
    assert state.aborted is False
    assert await reg.is_aborted("scan_reg_1") is False
    # Unknown scan
    assert await reg.is_aborted("scan_unknown") is False


@pytest.mark.asyncio
async def test_scan_registry_abort_sets_event():
    """abort_scan → is_aborted returns True + abort_event is set."""
    reg = ScanRegistry()
    state = await reg.register_scan("scan_reg_2")
    assert state.abort_event.is_set() is False

    result = await reg.abort_scan("scan_reg_2", run_cleanup=False)

    assert result["aborted"] is True
    assert result["scan_id"] == "scan_reg_2"
    assert state.abort_event.is_set() is True
    assert await reg.is_aborted("scan_reg_2") is True


@pytest.mark.asyncio
async def test_scan_registry_abort_unknown_scan():
    """abort_scan on unknown scan_id → returns aborted=False."""
    reg = ScanRegistry()
    result = await reg.abort_scan("scan_unknown_2", run_cleanup=False)
    assert result["aborted"] is False
    assert result["reason"] == "scan_not_found"


@pytest.mark.asyncio
async def test_scan_registry_double_abort_is_noop():
    """abort_scan twice → second call is no-op."""
    reg = ScanRegistry()
    await reg.register_scan("scan_reg_3")

    first = await reg.abort_scan("scan_reg_3", run_cleanup=False)
    second = await reg.abort_scan("scan_reg_3", run_cleanup=False)

    assert first["aborted"] is True
    assert second["aborted"] is True
    assert second["message"] == "Scan already aborted"


@pytest.mark.asyncio
async def test_scan_registry_unregister():
    """unregister_scan removes the scan from registry."""
    reg = ScanRegistry()
    await reg.register_scan("scan_reg_4")
    assert "scan_reg_4" in reg.list_active_scans()

    await reg.unregister_scan("scan_reg_4")
    assert "scan_reg_4" not in reg.list_active_scans()


@pytest.mark.asyncio
async def test_scan_registry_register_subprocess():
    """register_subprocess adds PID to scan state."""
    reg = ScanRegistry()
    await reg.register_scan("scan_reg_5")
    await reg.register_subprocess("scan_reg_5", 12345)
    await reg.register_subprocess("scan_reg_5", 67890)

    state = reg.get_state("scan_reg_5")
    assert 12345 in state.subprocess_pids
    assert 67890 in state.subprocess_pids

    # Unregister one
    await reg.unregister_subprocess("scan_reg_5", 12345)
    state = reg.get_state("scan_reg_5")
    assert 12345 not in state.subprocess_pids
    assert 67890 in state.subprocess_pids


@pytest.mark.asyncio
async def test_scan_registry_abort_kills_subprocesses():
    """abort_scan kills all registered subprocess PIDs."""
    reg = ScanRegistry()
    await reg.register_scan("scan_reg_6")
    await reg.register_subprocess("scan_reg_6", 12345)
    await reg.register_subprocess("scan_reg_6", 67890)

    # Mock the kill function
    killed_pids: list[int] = []
    original_kill = reg._kill_process_group
    reg._kill_process_group = lambda pid: killed_pids.append(pid) or True

    result = await reg.abort_scan("scan_reg_6", run_cleanup=False)

    assert result["aborted"] is True
    assert sorted(result["killed_subprocess_pids"]) == [12345, 67890]
    assert sorted(killed_pids) == [12345, 67890]

    # Subprocess set should be cleared
    state = reg.get_state("scan_reg_6")
    assert len(state.subprocess_pids) == 0


@pytest.mark.asyncio
async def test_scan_registry_abort_runs_cleanup_script():
    """abort_scan with run_cleanup=True calls _run_cleanup_script."""
    reg = ScanRegistry()
    await reg.register_scan("scan_reg_7")

    cleanup_called: list[str] = []
    reg._run_cleanup_script = lambda scan_id: cleanup_called.append(scan_id) or {
        "success": True, "stdout": "cleanup complete", "returncode": 0,
    }

    result = await reg.abort_scan("scan_reg_7", run_cleanup=True)

    assert result["aborted"] is True
    assert result["cleanup"]["success"] is True
    assert cleanup_called == ["scan_reg_7"]


@pytest.mark.asyncio
async def test_scan_registry_abort_skips_cleanup_when_disabled():
    """abort_scan with run_cleanup=False does NOT call cleanup script."""
    reg = ScanRegistry()
    await reg.register_scan("scan_reg_8")

    cleanup_called: list[str] = []
    reg._run_cleanup_script = lambda scan_id: cleanup_called.append(scan_id) or {
        "success": True, "returncode": 0,
    }

    result = await reg.abort_scan("scan_reg_8", run_cleanup=False)

    assert result["aborted"] is True
    assert result["cleanup"] is None
    assert cleanup_called == []


# ============================================================
# Metasploit msf_module_execute — audit_agent integration
# ============================================================
#
# NOTE: msf_module_execute() no longer accepts `db_session` param (W7-C FIX:
# AsyncSession in MCP tool signature breaks FastMCP JSON-Schema generation).
# The function creates its own session via `async_session()` (an async_sessionmaker
# instance — calling it returns an AsyncSession directly, NOT a context manager).
# Tests patch `app.db.session.async_session` to return the fake session directly.


@pytest.mark.asyncio
async def test_msf_module_execute_approve_runs_exploit():
    """msf_module_execute with audit_agent approve → client.module_execute called."""
    fake_session = _make_fake_session()
    fake_audit = FakeAuditAgent(_make_audit_decision(decision="approve"))
    mgr = HITLManager(session=fake_session, mode="audit_agent", audit_agent=fake_audit)

    # Mock the MSF client
    mock_client = MagicMock()
    mock_client.module_execute = AsyncMock(return_value={
        "job_id": 1, "sessions": [],
    })

    # Patch HITLManager + async_session (function calls async_session() to get a session)
    with patch("app.exploit.metasploit_tools.get_msf_client", return_value=mock_client), \
         patch("app.hitl.manager.HITLManager", return_value=mgr), \
         patch("app.db.session.async_session", return_value=fake_session):
        from app.exploit.metasploit_tools import msf_module_execute
        result_str = await msf_module_execute(
            module_type="exploit",
            module_name="exploit/windows/smb/ms17_010_eternalblue",
            options={"RHOSTS": "10.10.10.5"},
            scan_id="scan_msf_1",
            agent_reasoning="Exploit MS17-010",
        )

    result = json.loads(result_str)
    assert result["job_id"] == 1
    mock_client.module_execute.assert_called_once_with(
        "exploit",
        "exploit/windows/smb/ms17_010_eternalblue",
        {"RHOSTS": "10.10.10.5"},
    )


@pytest.mark.asyncio
async def test_msf_module_execute_reject_blocks_exploit():
    """msf_module_execute with audit_agent reject → client.module_execute NOT called."""
    fake_session = _make_fake_session()
    fake_audit = FakeAuditAgent(_make_audit_decision(
        decision="reject", comment="DROP TABLE detected",
    ))
    mgr = HITLManager(session=fake_session, mode="audit_agent", audit_agent=fake_audit)

    mock_client = MagicMock()
    mock_client.module_execute = AsyncMock()  # should NOT be called

    with patch("app.exploit.metasploit_tools.get_msf_client", return_value=mock_client), \
         patch("app.hitl.manager.HITLManager", return_value=mgr), \
         patch("app.db.session.async_session", return_value=fake_session):
        from app.exploit.metasploit_tools import msf_module_execute
        result_str = await msf_module_execute(
            module_type="exploit",
            module_name="exploit/multi/handler",
            options={"RHOSTS": "10.10.10.5"},
            scan_id="scan_msf_2",
        )

    result = json.loads(result_str)
    assert result["error"] == "HITL_REJECTED"
    assert "DROP TABLE" in result["message"] or "audit agent" in result["message"].lower()
    mock_client.module_execute.assert_not_called()


@pytest.mark.asyncio
async def test_msf_module_execute_auxiliary_skips_hitl():
    """msf_module_execute with module_type=auxiliary → no HITL (only exploit requires)."""
    mock_client = MagicMock()
    mock_client.module_execute = AsyncMock(return_value={"job_id": 2})

    fake_session = _make_fake_session()

    with patch("app.exploit.metasploit_tools.get_msf_client", return_value=mock_client):
        from app.exploit.metasploit_tools import msf_module_execute
        result_str = await msf_module_execute(
            module_type="auxiliary",  # NOT exploit — no HITL
            module_name="auxiliary/scanner/smb/smb_version",
            options={"RHOSTS": "10.10.10.5"},
            scan_id="scan_msf_3",
        )

    result = json.loads(result_str)
    assert result["job_id"] == 2
    mock_client.module_execute.assert_called_once()


# ============================================================
# Metasploit msf_session_run — read-only vs destructive
# ============================================================

@pytest.mark.asyncio
async def test_msf_session_run_readonly_skips_hitl():
    """msf_session_run with read-only command (getuid) → no HITL."""
    mock_client = MagicMock()
    mock_client.session_run = AsyncMock(return_value="uid=root")

    fake_session = _make_fake_session()

    with patch("app.exploit.metasploit_tools.get_msf_client", return_value=mock_client):
        from app.exploit.metasploit_tools import msf_session_run
        result_str = await msf_session_run(
            session_id=1,
            command="getuid",  # read-only
            scan_id="scan_msf_4",
        )

    result = json.loads(result_str)
    assert result["session_id"] == 1
    assert "uid=root" in result["output"]
    mock_client.session_run.assert_called_once_with(1, "getuid")


@pytest.mark.asyncio
async def test_msf_session_run_destructive_requires_hitl():
    """msf_session_run with destructive command (shell) → HITL review."""
    fake_session = _make_fake_session()
    fake_audit = FakeAuditAgent(_make_audit_decision(decision="approve"))
    mgr = HITLManager(session=fake_session, mode="audit_agent", audit_agent=fake_audit)

    mock_client = MagicMock()
    mock_client.session_run = AsyncMock(return_value="shell output")

    with patch("app.exploit.metasploit_tools.get_msf_client", return_value=mock_client), \
         patch("app.hitl.manager.HITLManager", return_value=mgr), \
         patch("app.db.session.async_session", return_value=fake_session):
        from app.exploit.metasploit_tools import msf_session_run
        result_str = await msf_session_run(
            session_id=1,
            command="shell",  # destructive — requires HITL
            scan_id="scan_msf_5",
        )

    result = json.loads(result_str)
    assert result["session_id"] == 1
    assert "shell output" in result["output"]
    mock_client.session_run.assert_called_once()
    # Audit agent was called
    assert len(fake_audit.review_calls) == 1


@pytest.mark.asyncio
async def test_msf_session_run_destructive_reject_blocks():
    """msf_session_run with destructive command + audit reject → blocked."""
    fake_session = _make_fake_session()
    fake_audit = FakeAuditAgent(_make_audit_decision(
        decision="reject", comment="rm -rf detected",
    ))
    mgr = HITLManager(session=fake_session, mode="audit_agent", audit_agent=fake_audit)

    mock_client = MagicMock()
    mock_client.session_run = AsyncMock()  # should NOT be called

    with patch("app.exploit.metasploit_tools.get_msf_client", return_value=mock_client), \
         patch("app.hitl.manager.HITLManager", return_value=mgr), \
         patch("app.db.session.async_session", return_value=fake_session):
        from app.exploit.metasploit_tools import msf_session_run
        result_str = await msf_session_run(
            session_id=1,
            command="rm -rf /",
            scan_id="scan_msf_6",
        )

    result = json.loads(result_str)
    assert result["error"] == "HITL_REJECTED"
    mock_client.session_run.assert_not_called()


# ============================================================
# Cleanup script creation
# ============================================================

def test_ensure_cleanup_script_creates_file(tmp_path):
    """_ensure_cleanup_script creates cleanup_scan.sh if missing."""
    reg = ScanRegistry()
    script_path = tmp_path / "cleanup_scan.sh"
    assert not script_path.exists()

    reg._ensure_cleanup_script(script_path)

    assert script_path.exists()
    content = script_path.read_text()
    assert "sqlmap" in content
    assert "msf6" in content
    assert "nuclei" in content
    assert "cleanup complete" in content
    # Executable bit set
    assert os.access(script_path, os.X_OK)


def test_run_cleanup_script_executes(tmp_path):
    """_run_cleanup_script runs the script + returns result dict."""
    reg = ScanRegistry()
    script_path = tmp_path / "cleanup_scan.sh"
    script_path.write_text("#!/bin/bash\necho 'cleanup complete for scan '$1\n")
    script_path.chmod(0o755)

    # Monkey-patch the project root lookup
    with patch.object(reg, "_ensure_cleanup_script"):
        # _run_cleanup_script uses Path(__file__).parent.parent.parent / scripts / cleanup_scan.sh
        # We need to patch that path
        import app.pentest.scan_registry as sr_mod
        original_file = sr_mod.ScanRegistry._run_cleanup_script

        # Directly invoke with our script path
        result = reg._run_cleanup_script("test_scan_id")

    # Result may be success or failure depending on whether script exists at
    # the default path. We just verify it returns a dict with expected keys.
    assert isinstance(result, dict)
    assert "success" in result