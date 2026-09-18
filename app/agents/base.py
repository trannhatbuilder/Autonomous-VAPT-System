"""
VAPT-AI Base Agent — W10-S3 / W11-S5.

Abstract base class for all 16 agents (3 orchestrators + 13 sub-agents).

Provides:
    - Tool allowlist enforcement (via AgentRegistry)
    - Max iterations cap (D18: max 30 decisions per scan)
    - D18 guardrails (token + time budget)
    - Abstract run() interface
    - Decision recording + SSE event emission
    - Agent metadata lookup (safety_class, tool_allowlist, prompt_file)
    - System prompt loading from .md file (W10-S1)

W11-S5 additions:
    - Auto-load relevant skills per agent role (via SkillLoader + agent_mapping)
    - `loaded_skills` property exposes list of skill names mapped to this agent
    - `get_loaded_skill_manifests()` returns full SkillManifest objects
    - `load_skill_body(name)` lazily loads a skill's Markdown body

Subclasses (app/agents/specialists/*.py — W10-S4) must implement:
    - _decide_next() → (thought, tool_name, tool_args, observation)
      W10 stub: returns deterministic canned response per agent role
      W11+ will replace with real LiteLLM call using self.system_prompt

Usage:
    from app.agents.base import BaseAgent
    from app.agents.specialists.recon_agent import ReconAgent

    agent = ReconAgent(
        scan_id="scan_abc123",
        target="http://example.com",
        user_prompt="Scan for open ports",
        task_description="Run nmap port scan on http://example.com",
    )
    # Access auto-loaded skills
    print(agent.loaded_skills)  # e.g. ["attack-surface-recon", "web-fingerprinting"]
    body = agent.load_skill_body("attack-surface-recon")
    result = await agent.run()

Architecture:
    ┌──────────────────────────────────────────────────┐
    │                  BaseAgent                       │
    │                                                  │
    │  - metadata (AgentMetadata from registry)       │
    │  - system_prompt (loaded from .md file)          │
    │  - loaded_skills (auto-loaded from agent_mapping) │
    │  - decisions[] (accumulated turn log)            │
    │  - total_tokens, start_time                      │
    │                                                  │
    │  + _check_guardrails()                           │
    │  + _validate_tool_call(tool, args)               │
    │  + _record_decision(decision)                    │
    │  + _emit_progress_event(decision)                 │
    │  + run() abstract                                │
    │  + _decide_next() abstract                       │
    │  + loaded_skills property (W11-S5)              │
    │  + load_skill_body(name) (W11-S5)               │
    └──────────────────────────────────────────────────┘
                         △
                         │
    ┌────────────────────┴────────────────────┐
    │                                         │
    │   13 sub-agent subclasses               │
    │   (app/agents/specialists/*.py)         │
    │                                         │
    │   Each overrides _decide_next() with    │
    │   agent-specific stub behavior          │
    └─────────────────────────────────────────┘

D18 guardrails (always enforced via _check_guardrails):
    - Max 30 decisions per scan (agent-level cap)
    - Max 2M tokens per scan
    - Max 4 hours per scan
    - Max 16 agents (fixed — enforced at registry level)

HITL integration (W10 stub):
    - Destructive agents (safety_class="destructive") have is_destructive=True
    - W10 only marks metadata; actual HITL gate wiring is W12 (evidence auditor)
    - For now, destructive agents record decisions but don't actually execute tools

Skill integration (W11-S5):
    - On __init__, BaseAgent calls get_skills_for_agent(self.AGENT_NAME)
    - Returns list of skill names mapped to this agent (from SKILL.md frontmatter)
    - Skills are NOT loaded into memory at init — only manifest summaries
    - Skill body loaded on demand via load_skill_body(name) (progressive disclosure)
"""
from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from app.agents.registry import AgentMetadata, agent_registry
from app.pentest.events import emit_scan_progress

logger = logging.getLogger(__name__)


# ---------- Constants (D18) ----------

MAX_DECISIONS_PER_AGENT = 30  # D18 cap
MAX_TOKENS_PER_SCAN = 2_000_000
MAX_SCAN_DURATION_SECONDS = 4 * 60 * 60  # 4 hours


# ---------- Data classes ----------

@dataclass
class AgentDecision:
    """One decision made by an agent during a scan.

    Mirrors app.orchestration.base.AgentDecision but kept independent here
    so app.agents package has no circular import on app.orchestration.
    """
    turn: int
    agent_name: str
    thought: str
    tool_name: str | None       # None = pure reasoning, no tool call
    tool_args: dict[str, Any]
    observation: str = ""
    tokens_used: int = 0
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass
class AgentRunResult:
    """Final result returned by BaseAgent.run().

    Compatible with OrchestratorResult for API response uniformity.
    """
    agent_name: str
    scan_id: str
    target: str
    task_description: str
    status: str  # completed | failed | max_iterations | timeout
    decisions: list[AgentDecision] = field(default_factory=list)
    findings: list[dict[str, Any]] = field(default_factory=list)
    total_tokens: int = 0
    duration_seconds: float = 0.0
    error: str | None = None
    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    completed_at: datetime | None = None
    loaded_skills: list[str] = field(default_factory=list)  # W11-S5

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_name": self.agent_name,
            "scan_id": self.scan_id,
            "target": self.target,
            "task_description": self.task_description,
            "status": self.status,
            "decisions_count": len(self.decisions),
            "decisions": [
                {
                    "turn": d.turn,
                    "agent_name": d.agent_name,
                    "thought": d.thought,
                    "tool_name": d.tool_name,
                    "tool_args": d.tool_args,
                    "observation": d.observation[:500] if d.observation else "",
                    "tokens_used": d.tokens_used,
                    "timestamp": d.timestamp.isoformat(),
                }
                for d in self.decisions
            ],
            "findings": self.findings,
            "total_tokens": self.total_tokens,
            "duration_seconds": self.duration_seconds,
            "error": self.error,
            "loaded_skills": self.loaded_skills,  # W11-S5
        }


# ---------- BaseAgent ----------

class BaseAgent:
    """Abstract base class for all 16 agents.

    Subclasses MUST:
        1. Set class attribute `AGENT_NAME` (e.g. "recon", "penetration")
        2. Override `_decide_next(turn)` → (thought, tool_name, tool_args, observation)

    Subclasses inherit:
        - D18 guardrail enforcement (_check_guardrails)
        - Tool allowlist validation (_validate_tool_call)
        - Decision recording (_record_decision)
        - SSE event emission (_emit_progress_event)
        - Final result construction (_finalize)
        - Auto-loaded skills (W11-S5: loaded_skills property)

    W10 stub: subclasses' _decide_next returns deterministic canned response.
    W11+ will replace _decide_next with LiteLLM call using self.system_prompt
    + relevant skill bodies (load_skill_body).
    """

    # Subclasses MUST override
    AGENT_NAME: str = "base"

    def __init__(
        self,
        scan_id: str,
        target: str,
        task_description: str = "",
        user_prompt: str = "",
        max_iterations: int | None = None,
    ):
        # Lookup metadata from registry
        self.metadata: AgentMetadata = agent_registry.require_agent(self.AGENT_NAME)

        # Cap max_iterations at D18 limit if not specified
        if max_iterations is None:
            max_iterations = self.metadata.max_iterations
        self.max_iterations = min(max_iterations, MAX_DECISIONS_PER_AGENT)

        # Context
        self.scan_id = scan_id
        self.target = target
        self.task_description = task_description
        self.user_prompt = user_prompt

        # Runtime accumulators
        self.decisions: list[AgentDecision] = []
        self.findings: list[dict[str, Any]] = []
        self.total_tokens = 0
        self.start_time = time.time()

        # Load system prompt from .md file
        self.system_prompt: str = agent_registry.load_prompt(self.AGENT_NAME)

        # W11-S5: Auto-load skill names mapped to this agent (manifests only,
        # bodies loaded on demand via load_skill_body()).
        from app.skills import get_skills_for_agent
        self._loaded_skill_names: list[str] = get_skills_for_agent(self.AGENT_NAME)

        logger.info(
            "Agent initialized: scan=%s agent=%s safety=%s tools=%d prompt_len=%d skills=%d",
            self.scan_id, self.AGENT_NAME, self.metadata.safety_class,
            len(self.metadata.tool_allowlist), len(self.system_prompt),
            len(self._loaded_skill_names),
        )

    # ---------- Properties ----------

    @property
    def name(self) -> str:
        """Canonical agent name (matches registry key)."""
        return self.AGENT_NAME

    @property
    def display_name(self) -> str:
        """Human-readable name from YAML frontmatter."""
        return self.metadata.display_name

    @property
    def safety_class(self) -> str:
        """read_only | destructive | advisory."""
        return self.metadata.safety_class

    @property
    def is_destructive(self) -> bool:
        """True if HITL approval required for tool calls."""
        return self.metadata.is_destructive

    @property
    def tool_allowlist(self) -> tuple[str, ...]:
        """Tools this agent is allowed to invoke."""
        return self.metadata.tool_allowlist

    @property
    def loaded_skills(self) -> list[str]:
        """W11-S5: List of skill names mapped to this agent (auto-loaded).

        Returns a copy to prevent external mutation.
        """
        return list(self._loaded_skill_names)

    # ---------- Skill access (W11-S5) ----------

    def get_loaded_skill_manifests(self) -> list[Any]:
        """Get full SkillManifest objects for this agent's loaded skills.

        Returns:
            List of SkillManifest instances (one per loaded skill name).
            Skills that fail to load are skipped (with a warning log).
        """
        from app.skills import skill_loader
        manifests = []
        for skill_name in self._loaded_skill_names:
            m = skill_loader.get_manifest(skill_name)
            if m is None:
                logger.warning(
                    "Skill %r mapped to agent %r but not found in SkillLoader",
                    skill_name, self.AGENT_NAME,
                )
                continue
            manifests.append(m)
        return manifests

    def load_skill_body(self, skill_name: str) -> str:
        """Load the Markdown body of a skill (progressive disclosure).

        Args:
            skill_name: Skill name (must be in self.loaded_skills for proper
                        attribution, but any registered skill can be loaded).

        Returns:
            Skill body as Markdown string.

        Raises:
            KeyError: if skill_name not found in SkillLoader.
        """
        from app.skills import skill_loader
        return skill_loader.load_skill_body(skill_name)

    def get_skill_context_for_llm(self) -> str:
        """Build a context string with skill summaries for LLM prompts.

        W11 stub: returns a formatted string listing all loaded skills with
        their names + descriptions. W12+ will use this in LLM prompts to
        give the agent awareness of available playbooks.

        Returns:
            Formatted string like:
                Loaded skills:
                - web-attack-methods: OWASP WSTG web attack taxonomy...
                - post-exploitation: Post-exploitation playbook...
        """
        manifests = self.get_loaded_skill_manifests()
        if not manifests:
            return "(no skills loaded for this agent)"
        lines = ["Loaded skills:"]
        for m in manifests:
            lines.append(f"- {m.name}: {m.description}")
        return "\n".join(lines)

    # ---------- Guardrails (D18) ----------

    def _check_guardrails(self) -> str | None:
        """Check D18 guardrails. Returns error message if violated, else None.

        Called before each decision in the agent loop.
        """
        # Iteration count
        if len(self.decisions) >= self.max_iterations:
            return (
                f"Exceeded {self.max_iterations} iterations for agent "
                f"{self.AGENT_NAME!r} (D18 cap: {MAX_DECISIONS_PER_AGENT})"
            )

        # Token budget
        if self.total_tokens > MAX_TOKENS_PER_SCAN:
            return f"Exceeded {MAX_TOKENS_PER_SCAN} tokens per scan (D18)"

        # Time budget
        elapsed = time.time() - self.start_time
        if elapsed > MAX_SCAN_DURATION_SECONDS:
            return f"Exceeded {MAX_SCAN_DURATION_SECONDS}s per scan (D18)"

        return None

    # ---------- Tool allowlist enforcement ----------

    def _validate_tool_call(
        self,
        tool_name: str,
        tool_args: dict[str, Any] | None = None,
    ) -> tuple[bool, str]:
        """Validate a tool call against agent's allowlist.

        Args:
            tool_name: Tool being invoked
            tool_args: Tool arguments (for context in error message)

        Returns:
            (allowed, reason) tuple.
        """
        return agent_registry.validate_tool_call(self.AGENT_NAME, tool_name)

    def _is_tool_allowed(self, tool_name: str) -> bool:
        """Check if tool is in agent's allowlist."""
        return agent_registry.is_tool_allowed(self.AGENT_NAME, tool_name)

    # ---------- Decision recording + SSE ----------

    def _record_decision(self, decision: AgentDecision) -> None:
        """Append a decision to the log + update accumulators."""
        self.decisions.append(decision)
        self.total_tokens += decision.tokens_used
        logger.info(
            "Turn %d: agent=%s tool=%s args=%s",
            decision.turn, decision.agent_name,
            decision.tool_name, decision.tool_args,
        )

    async def _emit_progress_event(self, decision: AgentDecision) -> None:
        """Emit SSE scan_progress event for this decision."""
        await emit_scan_progress(
            scan_id=self.scan_id,
            turn=decision.turn,
            thought=decision.thought,
            tool_name=decision.tool_name,
            observation=decision.observation,
        )

    # ---------- Finalize ----------

    def _finalize(self, status: str, error: str | None = None) -> AgentRunResult:
        """Build the final AgentRunResult."""
        return AgentRunResult(
            agent_name=self.AGENT_NAME,
            scan_id=self.scan_id,
            target=self.target,
            task_description=self.task_description,
            status=status,
            decisions=self.decisions,
            findings=self.findings,
            total_tokens=self.total_tokens,
            duration_seconds=time.time() - self.start_time,
            error=error,
            completed_at=datetime.now(UTC),
            loaded_skills=list(self._loaded_skill_names),  # W11-S5
        )

    # ---------- Abstract interface ----------

    async def run(self) -> AgentRunResult:
        """Execute the agent's task. Returns final result.

        Default implementation:
            1. Check guardrails
            2. Call _decide_next() to get one decision (W10 stub: 1 decision per run)
            3. Record decision + emit SSE
            4. Finalize with status='completed'

        Subclasses can override to implement multi-step ReAct loops (W11+).
        """
        # Check guardrails before starting
        violation = self._check_guardrails()
        if violation:
            return self._finalize("failed", error=violation)

        # Get one decision from subclass
        turn = len(self.decisions)
        thought, tool_name, tool_args, observation = await self._decide_next(turn)

        # Validate tool call against allowlist
        if tool_name is not None:
            allowed, reason = self._validate_tool_call(tool_name, tool_args)
            if not allowed:
                logger.warning(
                    "Tool allowlist violation: agent=%s tool=%s reason=%s",
                    self.AGENT_NAME, tool_name, reason,
                )
                observation = f"TOOL_ALLOWLIST_VIOLATION: {reason}"
                tool_name = None  # block the call

        # Record decision
        decision = AgentDecision(
            turn=turn,
            agent_name=self.AGENT_NAME,
            thought=thought,
            tool_name=tool_name,
            tool_args=tool_args,
            observation=observation,
            tokens_used=100,  # stub — W11+ will use real token counting
        )
        self._record_decision(decision)
        await self._emit_progress_event(decision)

        return self._finalize("completed")

    async def _decide_next(
        self,
        turn: int,
    ) -> tuple[str, str | None, dict[str, Any], str]:
        """Decide the next action. Subclasses MUST override.

        Returns:
            (thought, tool_name_or_None, tool_args, observation)

            - thought: short reasoning text (2-4 sentences)
            - tool_name: tool to invoke, or None if pure reasoning
            - tool_args: arguments for the tool
            - observation: tool result or reasoning output

        W10 stub: subclasses return canned response per agent role.
        W11+ will replace with LiteLLM call using self.system_prompt +
        relevant skill bodies (self.get_skill_context_for_llm()).
        """
        raise NotImplementedError(
            f"{self.__class__.__name__}._decide_next() not implemented. "
            f"Subclasses must override this method."
        )


# ---------- Convenience: agent factory ----------

def create_agent(
    agent_name: str,
    scan_id: str,
    target: str,
    task_description: str = "",
    user_prompt: str = "",
    max_iterations: int | None = None,
) -> BaseAgent:
    """Factory: create an agent instance by name.

    Looks up the agent subclass in app.agents.specialists and instantiates it.

    Args:
        agent_name: e.g. "recon", "penetration"
        scan_id: Scan ID this agent is running under
        target: Target URL or IP
        task_description: Task description from orchestrator
        user_prompt: Original user prompt (for context)
        max_iterations: Optional override (default from registry)

    Returns:
        BaseAgent subclass instance.

    Raises:
        KeyError: if agent_name not in registry
        ImportError: if specialist module not found
    """
    # Verify agent exists in registry
    agent_registry.require_agent(agent_name)

    # Import specialist module dynamically
    # Module name: app.agents.specialists.<agent_name>_agent
    # Class name: <AgentName>Agent (CamelCase)
    module_name = f"app.agents.specialists.{_to_module_name(agent_name)}"
    class_name = _to_class_name(agent_name)

    import importlib
    try:
        module = importlib.import_module(module_name)
    except ImportError as e:
        raise ImportError(
            f"Specialist module not found for agent {agent_name!r}: {module_name}. "
            f"Expected class: {class_name}."
        ) from e

    if not hasattr(module, class_name):
        raise ImportError(
            f"Class {class_name} not found in module {module_name}. "
            f"Available: {[n for n in dir(module) if n.endswith('Agent')]}"
        )

    cls = getattr(module, class_name)
    return cls(
        scan_id=scan_id,
        target=target,
        task_description=task_description,
        user_prompt=user_prompt,
        max_iterations=max_iterations,
    )


def _to_module_name(agent_name: str) -> str:
    """Convert agent name to module name. Handles hyphens.

    e.g. "recon" → "recon_agent"
         "attack-surface-enumeration" → "attack_surface_enumeration_agent"
    """
    return agent_name.replace("-", "_") + "_agent"


def _to_class_name(agent_name: str) -> str:
    """Convert agent name to CamelCase class name.

    e.g. "recon" → "ReconAgent"
         "attack-surface-enumeration" → "AttackSurfaceEnumerationAgent"
    """
    parts = agent_name.replace("-", "_").split("_")
    return "".join(p.capitalize() for p in parts) + "Agent"