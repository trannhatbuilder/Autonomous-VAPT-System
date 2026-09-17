"""
ImpactExfiltrationAgent — impact-exfiltration specialist.

Business impact + data accessibility proof (HITL required, NO actual exfiltration)

Ported from CyberStrikeAI agents/impact-exfiltration.md (English, W10-S1).
Uses BaseAgent infrastructure for:
    - Tool allowlist enforcement (via AgentRegistry)
    - D18 guardrails (max 30 iterations, max 2M tokens, max 4h)
    - SSE event emission
    - System prompt loading from app/agents/impact-exfiltration.md

W10 stub: _decide_next() returns deterministic canned response.
W11+ will replace with LiteLLM call using self.system_prompt.

Safety class: destructive
HITL required: True

Tool allowlist: ['metasploit']

Usage:
    from app.agents.specialists.impact_exfiltration_agent import ImpactExfiltrationAgent

    agent = ImpactExfiltrationAgent(
        scan_id="scan_abc123",
        target="http://example.com",
        task_description="Run nmap port scan",
    )
    result = await agent.run()
"""
from __future__ import annotations

from typing import Any

from app.agents.base import BaseAgent


class ImpactExfiltrationAgent(BaseAgent):
    """Business impact + data accessibility proof (HITL required, NO actual exfiltration)

    Safety class: destructive
    HITL required: True
    """

    AGENT_NAME = "impact-exfiltration"

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
        thought = "Business impact + data accessibility proof (HITL required, NO actual exfiltration) on " + self.target + " (W10 stub). Task: " + task_summary

        # Pick first tool from allowlist (W10 stub: just announce which tool we'd run)
        tools = self.tool_allowlist
        if tools:
            tool_name = tools[0]
            tool_args = dict(target=self.target)
        else:
            tool_name = None
            tool_args = dict()

        observation = "Impact (W10 stub): metasploit post/windows/gather/credentials found 2 hashes + 1 cleartext. Accessibility proven. NO exfiltration performed. PII redacted from evidence."

        return thought, tool_name, tool_args, observation
