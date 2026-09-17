"""
ReconAgent — recon specialist.

Active port scanning + service fingerprinting

Ported from CyberStrikeAI agents/recon.md (English, W10-S1).
Uses BaseAgent infrastructure for:
    - Tool allowlist enforcement (via AgentRegistry)
    - D18 guardrails (max 30 iterations, max 2M tokens, max 4h)
    - SSE event emission
    - System prompt loading from app/agents/recon.md

W10 stub: _decide_next() returns deterministic canned response.
W11+ will replace with LiteLLM call using self.system_prompt.

Safety class: read_only
HITL required: False

Tool allowlist: ['nmap', 'httpx', 'whatweb', 'masscan', 'rustscan']

Usage:
    from app.agents.specialists.recon_agent import ReconAgent

    agent = ReconAgent(
        scan_id="scan_abc123",
        target="http://example.com",
        task_description="Run nmap port scan",
    )
    result = await agent.run()
"""
from __future__ import annotations

from typing import Any

from app.agents.base import BaseAgent


class ReconAgent(BaseAgent):
    """Active port scanning + service fingerprinting

    Safety class: read_only
    HITL required: False
    """

    AGENT_NAME = "recon"

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
        thought = "Active port scanning + service fingerprinting on " + self.target + " (W10 stub). Task: " + task_summary

        # Pick first tool from allowlist (W10 stub: just announce which tool we'd run)
        tools = self.tool_allowlist
        if tools:
            tool_name = tools[0]
            tool_args = dict(target=self.target)
        else:
            tool_name = None
            tool_args = dict()

        observation = "Recon (W10 stub): nmap found ports 22 (ssh OpenSSH 8.2p1), 80 (http Apache 2.4.41), 443 (https OpenSSL 1.1.1f); httpx confirmed HTTP tech stack."

        return thought, tool_name, tool_args, observation
