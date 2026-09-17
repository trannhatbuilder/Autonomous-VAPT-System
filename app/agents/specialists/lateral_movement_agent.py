"""
LateralMovementAgent — lateral-movement specialist.

Internal network discovery + credential/session exploitation (HITL required)

Ported from CyberStrikeAI agents/lateral-movement.md (English, W10-S1).
Uses BaseAgent infrastructure for:
    - Tool allowlist enforcement (via AgentRegistry)
    - D18 guardrails (max 30 iterations, max 2M tokens, max 4h)
    - SSE event emission
    - System prompt loading from app/agents/lateral-movement.md

W10 stub: _decide_next() returns deterministic canned response.
W11+ will replace with LiteLLM call using self.system_prompt.

Safety class: destructive
HITL required: True

Tool allowlist: ['netexec', 'impacket', 'responder']

Usage:
    from app.agents.specialists.lateral_movement_agent import LateralMovementAgent

    agent = LateralMovementAgent(
        scan_id="scan_abc123",
        target="http://example.com",
        task_description="Run nmap port scan",
    )
    result = await agent.run()
"""
from __future__ import annotations

from typing import Any

from app.agents.base import BaseAgent


class LateralMovementAgent(BaseAgent):
    """Internal network discovery + credential/session exploitation (HITL required)

    Safety class: destructive
    HITL required: True
    """

    AGENT_NAME = "lateral-movement"

    async def _decide_next(
        self,
        turn: int,
    ) -> tuple[str, str | None, dict[str, Any], str]:
        """W10 stub: return deterministic canned response.

        W11+ will replace with LiteLLM call using self.system_prompt +
        self.task_description + blackboard context.

        Returns:
            (thought, tool_name, tool_args, observation)
        """
        task_summary = self.task_description[:80] if self.task_description else "(no task description)"
        thought = "Internal network discovery + credential/session exploitation (HITL required) on " + self.target + " (W10 stub). Task: " + task_summary

        # Pick first tool from allowlist (W10 stub: just announce which tool we'd run)
        tools = self.tool_allowlist
        if tools:
            tool_name = tools[0]
            tool_args = dict(target=self.target)
        else:
            tool_name = None
            tool_args = dict()

        observation = "Lateral movement (W10 stub): netexec smb 192.168.1.0/24 found 4 hosts, 1 Pwn3d! (192.168.1.10 with creds user:Pass123). impacket-smbexec available for RCE."

        return thought, tool_name, tool_args, observation
