"""
OpsecEvasionAgent — opsec-evasion specialist.

Advisory: low-noise testing strategies (no WAF bypass / evasion exploits)

Ported from CyberStrikeAI agents/opsec-evasion.md (English, W10-S1).
Uses BaseAgent infrastructure for:
    - Tool allowlist enforcement (via AgentRegistry)
    - D18 guardrails (max 30 iterations, max 2M tokens, max 4h)
    - SSE event emission
    - System prompt loading from app/agents/opsec-evasion.md

W10 stub: _decide_next() returns deterministic canned response.
W11+ will replace with LiteLLM call using self.system_prompt.

Safety class: destructive
HITL required: True

Tool allowlist: []

Usage:
    from app.agents.specialists.opsec_evasion_agent import OpsecEvasionAgent

    agent = OpsecEvasionAgent(
        scan_id="scan_abc123",
        target="http://example.com",
        task_description="Run nmap port scan",
    )
    result = await agent.run()
"""
from __future__ import annotations

from typing import Any

from app.agents.base import BaseAgent


class OpsecEvasionAgent(BaseAgent):
    """Advisory: low-noise testing strategies (no WAF bypass / evasion exploits)

    Safety class: destructive
    HITL required: True
    """

    AGENT_NAME = "opsec-evasion"

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
        thought = "Advisory: low-noise testing strategies (no WAF bypass / evasion exploits) on " + self.target + " (W10 stub). Task: " + task_summary

        # Pick first tool from allowlist (W10 stub: just announce which tool we'd run)
        tools = self.tool_allowlist
        if tools:
            tool_name = tools[0]
            tool_args = dict(target=self.target)
        else:
            tool_name = None
            tool_args = dict()

        observation = "OPSEC advisory (W10 stub): recommend 1) slow nmap timing (-T2), 2) rotate User-Agent, 3) avoid peak hours. Do NOT provide WAF bypass techniques — out of scope."

        return thought, tool_name, tool_args, observation
