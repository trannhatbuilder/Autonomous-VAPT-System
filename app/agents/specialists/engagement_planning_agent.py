"""
EngagementPlanningAgent — engagement-planning specialist.

Defines engagement scope, ROE, and success criteria; produces test blueprints

Ported from CyberStrikeAI agents/engagement-planning.md (English, W10-S1).
Uses BaseAgent infrastructure for:
    - Tool allowlist enforcement (via AgentRegistry)
    - D18 guardrails (max 30 iterations, max 2M tokens, max 4h)
    - SSE event emission
    - System prompt loading from app/agents/engagement-planning.md

W10 stub: _decide_next() returns deterministic canned response.
W11+ will replace with LiteLLM call using self.system_prompt.

Safety class: read_only
HITL required: False

Tool allowlist: []

Usage:
    from app.agents.specialists.engagement_planning_agent import EngagementPlanningAgent

    agent = EngagementPlanningAgent(
        scan_id="scan_abc123",
        target="http://example.com",
        task_description="Run nmap port scan",
    )
    result = await agent.run()
"""
from __future__ import annotations

from typing import Any

from app.agents.base import BaseAgent


class EngagementPlanningAgent(BaseAgent):
    """Defines engagement scope, ROE, and success criteria; produces test blueprints

    Safety class: read_only
    HITL required: False
    """

    AGENT_NAME = "engagement-planning"

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
        thought = "Defines engagement scope, ROE, and success criteria; produces test blueprints on " + self.target + " (W10 stub). Task: " + task_summary

        # Pick first tool from allowlist (W10 stub: just announce which tool we'd run)
        tools = self.tool_allowlist
        if tools:
            tool_name = tools[0]
            tool_args = dict(target=self.target)
        else:
            tool_name = None
            tool_args = dict()

        observation = "Engagement plan (W10 stub): scope=http://example.com, success_criteria='discover exploitable vulns', ROE='non-destructive, HITL for destructive ops'. Blueprint: 1) recon 2) vuln-triage 3) penetration (HITL)."

        return thought, tool_name, tool_args, observation
