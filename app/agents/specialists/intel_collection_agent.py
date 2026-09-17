"""
IntelCollectionAgent — intel-collection specialist.

OSINT + passive intelligence: emails, subdomains, DNS, historical URLs

Ported from CyberStrikeAI agents/intel-collection.md (English, W10-S1).
Uses BaseAgent infrastructure for:
    - Tool allowlist enforcement (via AgentRegistry)
    - D18 guardrails (max 30 iterations, max 2M tokens, max 4h)
    - SSE event emission
    - System prompt loading from app/agents/intel-collection.md

W10 stub: _decide_next() returns deterministic canned response.
W11+ will replace with LiteLLM call using self.system_prompt.

Safety class: read_only
HITL required: False

Tool allowlist: ['theharvester', 'subfinder', 'amass', 'dnsenum', 'fierce', 'gau', 'waybackurls']

Usage:
    from app.agents.specialists.intel_collection_agent import IntelCollectionAgent

    agent = IntelCollectionAgent(
        scan_id="scan_abc123",
        target="http://example.com",
        task_description="Run nmap port scan",
    )
    result = await agent.run()
"""
from __future__ import annotations

from typing import Any

from app.agents.base import BaseAgent


class IntelCollectionAgent(BaseAgent):
    """OSINT + passive intelligence: emails, subdomains, DNS, historical URLs

    Safety class: read_only
    HITL required: False
    """

    AGENT_NAME = "intel-collection"

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
        thought = "OSINT + passive intelligence: emails, subdomains, DNS, historical URLs on " + self.target + " (W10 stub). Task: " + task_summary

        # Pick first tool from allowlist (W10 stub: just announce which tool we'd run)
        tools = self.tool_allowlist
        if tools:
            tool_name = tools[0]
            tool_args = dict(target=self.target)
        else:
            tool_name = None
            tool_args = dict()

        observation = "Intel collection (W10 stub): theHarvester found 3 emails; subfinder found 5 subdomains (api.example.com, dev.example.com, ...); waybackurls returned 47 historical URLs."

        return thought, tool_name, tool_args, observation
