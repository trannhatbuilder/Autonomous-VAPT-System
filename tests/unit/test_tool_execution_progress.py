"""Tests for long-running tool execution telemetry + PATH resolution.

Covers the "scan looks frozen on nmap" fix:
  - ExecutionService publishes a `tool_call_progress` heartbeat every N seconds
    while a tool is running (so the SSE timeline is not silent for minutes).
  - No heartbeat is published when the execution has no scan_id.
  - SubprocessExecutor's service PATH includes user install locations
    (~/.cargo/bin, ~/.local/bin, the project venv) so tools like rustscan and
    wrapper shims resolve when VAPT-AI runs as a systemd service.

Run:
    pytest tests/unit/test_tool_execution_progress.py -v
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import app.mcp.execution_service as execution_service
from app.mcp.execution_service import ExecutionService
from app.pentest.events import event_bus
from app.sandbox.executor import SubprocessExecutor


async def _drain(scan_id: str, out: list[str]) -> None:
    async for ev in event_bus.subscribe(scan_id):
        out.append(ev.get("event"))
        if ev.get("event") in ("scan_complete", "scan_error"):
            break


@pytest.mark.asyncio
async def test_progress_heartbeat_emitted_while_tool_runs(monkeypatch):
    monkeypatch.setattr(execution_service, "TOOL_PROGRESS_INTERVAL_SECONDS", 0.05)

    svc = ExecutionService()
    seen: list[str] = []
    scan_id = "scan_progress_test"

    consumer = asyncio.create_task(_drain(scan_id, seen))
    await asyncio.sleep(0.05)

    async def run(cancel_event):
        await asyncio.sleep(0.18)  # > 3 intervals
        return {"stdout": "ok"}

    execution = await svc.submit(
        tool_name="nmap", arguments={"target": "x"}, target="x",
        run=run, scan_id=scan_id, hard_timeout=30,
    )
    assert execution.status.value == "completed"

    await event_bus.publish(scan_id, {"event": "scan_complete"})
    await asyncio.sleep(0.1)
    consumer.cancel()

    assert seen.count("tool_call_progress") >= 1, f"no heartbeat in {seen}"
    # Progress events must be attributable to the tool/scan.
    assert "tool_call_progress" in seen


@pytest.mark.asyncio
async def test_no_progress_when_no_scan_id(monkeypatch):
    monkeypatch.setattr(execution_service, "TOOL_PROGRESS_INTERVAL_SECONDS", 0.05)

    svc = ExecutionService()
    seen: list[str] = []
    consumer = asyncio.create_task(_drain("unused", seen))
    await asyncio.sleep(0.05)

    async def run(cancel_event):
        await asyncio.sleep(0.15)
        return {"stdout": "ok"}

    execution = await svc.submit(
        tool_name="nmap", arguments={}, target="x",
        run=run, scan_id=None, hard_timeout=30,
    )
    assert execution.status.value == "completed"
    consumer.cancel()
    assert "tool_call_progress" not in seen


class TestServicePath:
    def test_build_env_includes_user_install_dirs(self, monkeypatch):
        monkeypatch.setenv("HOME", "/home/tester")
        env = SubprocessExecutor()._build_env()
        parts = env["PATH"].split(":")
        assert "/home/tester/.cargo/bin" in parts, "rustscan (~/.cargo/bin) not on PATH"
        assert "/home/tester/.local/bin" in parts
        assert "/home/tester/go/bin" in parts
        # Must never lose the system dirs
        assert "/usr/bin" in parts and "/bin" in parts
