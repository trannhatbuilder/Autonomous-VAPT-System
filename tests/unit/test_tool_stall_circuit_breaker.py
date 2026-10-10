"""Tests: stall circuit-breaker forces a tool switch instead of hanging.

From a real scan: an approved `metasploit` call ran 542s (until the operator
pressed the panic button) and the penetration phase appeared frozen — every
other tool call was blocked behind it. The agent is also prone to re-running the
same stalling tool.

This adds a per-scan stall counter: a tool killed by its hard timeout (or
cancelled mid-run) is counted; once it hits its limit it is BLOCKED for the rest
of the scan and the agent receives an explicit "switch to <other tool>" message.

Run:
    pytest tests/unit/test_tool_stall_circuit_breaker.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app.agents.tool_call_caps import (
    DEFAULT_TOOL_STALL_LIMIT,
    MAX_TOOL_STALLS_PER_SCAN,
    TOOL_ALTERNATIVES,
    clear_scan_caps,
    get_tool_stall_count,
    is_tool_stalled,
    mark_tool_stalled,
)
from app.tools.loader import load_all_tools


@pytest.fixture(autouse=True)
def _clean_state():
    yield
    clear_scan_caps("scan_stall_test")


class TestStallCounters:
    def test_metasploit_blocks_after_one_stall(self):
        scan = "scan_stall_test"
        assert is_tool_stalled(scan, "metasploit") == (False, "")

        count = mark_tool_stalled(scan, "metasploit", reason="hard_timeout")
        assert count == 1
        assert get_tool_stall_count(scan, "metasploit") == 1

        blocked, reason = is_tool_stalled(scan, "metasploit")
        assert blocked is True
        assert "TOOL_STALLED_SWITCH_REQUIRED" in reason
        assert "metasploit" in reason
        # The reason must name a concrete alternative.
        assert "sqlmap" in reason

    def test_generic_tool_allows_one_retry_then_blocks(self):
        scan = "scan_stall_test"
        assert DEFAULT_TOOL_STALL_LIMIT >= 2
        assert is_tool_stalled(scan, "nuclei") == (False, "")

        mark_tool_stalled(scan, "nuclei")
        assert is_tool_stalled(scan, "nuclei") == (False, "")

        mark_tool_stalled(scan, "nuclei")
        blocked, reason = is_tool_stalled(scan, "nuclei")
        assert blocked is True
        assert "TOOL_STALLED_SWITCH_REQUIRED" in reason

    def test_clear_resets_stalls(self):
        scan = "scan_stall_test"
        mark_tool_stalled(scan, "metasploit")
        assert is_tool_stalled(scan, "metasploit")[0] is True
        clear_scan_caps(scan)
        assert is_tool_stalled(scan, "metasploit") == (False, "")

    def test_every_blocking_tool_has_an_alternative_message(self):
        for tool in MAX_TOOL_STALLS_PER_SCAN:
            assert tool in TOOL_ALTERNATIVES, f"no switch guidance for {tool}"


class TestToolBridgeBlocksStalledTool:
    @pytest.mark.asyncio
    async def test_execute_tool_call_returns_switch_message(self):
        from app.agents.tool_bridge import execute_tool_call

        scan = "scan_stall_test"
        mark_tool_stalled(scan, "metasploit", reason="hard_timeout")

        output = await execute_tool_call(
            tool_name="metasploit",
            tool_args={"command": "use auxiliary/scanner/http/http_version; run"},
            target="https://example.com",
            scan_id=scan,
            executor=object(),  # must never be used — the call is blocked pre-spawn
        )

        assert "TOOL_STALLED_SWITCH_REQUIRED" in output
        # Guidance must tell the agent what to do instead of stalling.
        assert "sqlmap" in output or "another tool" in output


class TestMetasploitTimeout:
    def test_timeout_is_bounded_for_quick_auxiliary_modules(self):
        tool = load_all_tools(force_reload=True)["metasploit"]
        assert tool.timeout <= 180, (
            "a hanging msfconsole must not hold the penetration phase for 10+ min"
        )


