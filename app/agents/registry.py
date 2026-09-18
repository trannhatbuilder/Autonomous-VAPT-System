"""
VAPT-AI Agent Registry — W10-S2.

Central registry for all 16 agents (3 orchestrators + 13 sub-agents).
Provides lookup, validation, and metadata access for the orchestration runtime.

Each agent is defined by a .md file in app/agents/ with YAML frontmatter:
    ---
    id: recon
    name: Reconnaissance Specialist
    description: ...
    tools: ["nmap", "httpx", "whatweb"]
    max_iterations: 30
    safety_class: read_only  # or "destructive"
    ---

For orchestrators (orchestrator.md, orchestrator-plan-execute.md, orchestrator-supervisor.md),
metadata is supplemented with default values (orchestrators don't have tools/safety_class
in their YAML — they delegate, not invoke tools directly).

Usage:
    from app.agents.registry import agent_registry, AgentMetadata

    # Lookup
    agent = agent_registry.get_agent("recon")
    print(agent.safety_class)            # "read_only"
    print(agent.tool_allowlist)          # ["nmap", "httpx", "whatweb", "masscan", "rustscan"]

    # Listing
    all_agents = agent_registry.list_agents()              # 16 agents
    sub_agents = agent_registry.list_sub_agents()          # 13 agents
    orchestrators = agent_registry.list_orchestrators()    # 3 agents

    # Filtering
    destructive = agent_registry.list_by_safety_class("destructive")  # 7 agents
    read_only = agent_registry.list_by_safety_class("read_only")       # 9 agents (incl orchestrators)

    # Validation
    allowed = agent_registry.is_tool_allowed("recon", "nmap")  # True
    allowed = agent_registry.is_tool_allowed("recon", "sqlmap")  # False

    # Prompt loading
    prompt = agent_registry.load_prompt("recon")  # full .md content

Architecture:
    - Singleton pattern: module-level `agent_registry` is the canonical instance
    - Lazy load: YAML frontmatter parsed on first access (not at import time)
    - Thread-safe: read-only after first load
    - Backward compatible: W9's EXPERT_AGENTS dict + STUB_SUB_AGENT_TASKS continue to work
      (will be migrated in W10-S5)

D18 guardrails (per master plan §8.4):
    - All agents have max_iterations capped at 30 (D18 max decisions per scan)
    - Total agent count capped at 16 (3 orchestrators + 13 sub-agents — fixed, cannot create
      new at runtime)
"""
from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import frontmatter  # python-frontmatter

logger = logging.getLogger(__name__)


# ---------- Constants ----------

AGENTS_DIR = Path(__file__).resolve().parent
ORCHESTRATOR_PREFIX = "orchestrator"
DEFAULT_MAX_ITERATIONS = 30  # D18 cap

# Mapping orchestrator filename → orchestration mode
_ORCHESTRATOR_MODE_MAP: dict[str, str] = {
    "orchestrator": "deep",
    "orchestrator-plan-execute": "plan_execute",
    "orchestrator-supervisor": "supervisor",
}


# ---------- AgentMetadata dataclass ----------

@dataclass(frozen=True)
class AgentMetadata:
    """Metadata for a single agent (orchestrator or sub-agent).

    Frozen so registry entries are immutable after load — prevents accidental
    modification by callers.

    Attributes:
        name:               Unique identifier (matches .md filename without .md extension)
                            e.g. "recon", "orchestrator-supervisor"
        display_name:       Human-readable name (from YAML `name` field)
                            e.g. "Reconnaissance Specialist"
        description:        Short description (from YAML `description` field)
        safety_class:       "read_only" | "destructive" | "advisory"
                            - read_only: can run non-destructive tools (nmap, nuclei, ...)
                            - destructive: requires HITL approval for tool calls (sqlmap, metasploit, ...)
                            - advisory: orchestrator or pure-reasoning agent (no tool calls)
        tool_allowlist:     List of tool names this agent can invoke (empty = no tools)
                            e.g. ["nmap", "httpx", "whatweb"]
        max_iterations:     Cap on decisions per scan for this agent (default 30 per D18)
        prompt_file:        .md filename (relative to AGENTS_DIR)
                            e.g. "recon.md", "orchestrator-supervisor.md"
        is_orchestrator:    True if this is one of the 3 orchestrators
        orchestration_mode: "deep" | "plan_execute" | "supervisor" if is_orchestrator else None
    """
    name: str
    display_name: str
    description: str
    safety_class: str
    tool_allowlist: tuple[str, ...]  # tuple for hashability
    max_iterations: int
    prompt_file: str
    is_orchestrator: bool
    orchestration_mode: str | None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict (for FastAPI response + JSON serialization).

        W11-S7: includes `skills` field with list of skill names mapped
        to this agent (auto-loaded via SkillLoader + agent_mapping).
        """
        d = asdict(self)
        d["tool_allowlist"] = list(self.tool_allowlist)  # tuple → list for JSON
        # W11-S7: include mapped skills (lazy-loaded from agent_mapping)
        try:
            from app.skills import get_skills_for_agent
            d["skills"] = get_skills_for_agent(self.name)
        except Exception:
            # If skills package not yet loaded (e.g. during bootstrap), return empty list
            d["skills"] = []
        return d

    @property
    def is_destructive(self) -> bool:
        """True if this agent requires HITL approval for tool calls."""
        return self.safety_class == "destructive"

    @property
    def has_tools(self) -> bool:
        """True if this agent has any tools in its allowlist."""
        return len(self.tool_allowlist) > 0


# ---------- AgentRegistry ----------

class AgentRegistry:
    """Central registry for all 16 agents (3 orchestrators + 13 sub-agents).

    Singleton pattern: module-level `agent_registry` is the canonical instance.
    Lazy loads metadata from .md YAML frontmatter on first access.

    Thread-safety: read-only after first load (frozen dataclass + cached dict).
    """

    def __init__(self, agents_dir: Path | None = None):
        self.agents_dir: Path = agents_dir or AGENTS_DIR
        self._metadata: dict[str, AgentMetadata] | None = None

    # ---------- Lazy load ----------

    @property
    def metadata(self) -> dict[str, AgentMetadata]:
        """Lazy-loaded metadata dict. Parses .md files on first access."""
        if self._metadata is None:
            self._metadata = self._load_metadata()
        return self._metadata

    def _load_metadata(self) -> dict[str, AgentMetadata]:
        """Parse all .md files in agents_dir + extract YAML frontmatter.

        Returns:
            Dict mapping agent name → AgentMetadata.

        Raises:
            FileNotFoundError: if agents_dir doesn't exist.
        """
        if not self.agents_dir.is_dir():
            raise FileNotFoundError(
                f"Agents directory not found: {self.agents_dir}"
            )

        result: dict[str, AgentMetadata] = {}

        for md_path in sorted(self.agents_dir.glob("*.md")):
            try:
                post = frontmatter.load(md_path)
            except Exception as e:
                logger.error("Failed to parse %s: %s", md_path, e)
                continue

            name = md_path.stem  # filename without .md
            is_orchestrator = name.startswith(ORCHESTRATOR_PREFIX)
            mode = _ORCHESTRATOR_MODE_MAP.get(name) if is_orchestrator else None

            # Tools: list[str] from YAML (default empty)
            tools_list = post.get("tools", []) or []
            if not isinstance(tools_list, list):
                logger.warning("Agent %s has non-list tools=%r — treating as empty", name, tools_list)
                tools_list = []
            tools_tuple = tuple(str(t) for t in tools_list)

            # Safety class: orchestrators default to "advisory", sub-agents use YAML value
            if is_orchestrator:
                safety_class = "advisory"
            else:
                safety_class = str(post.get("safety_class", "read_only"))
                if safety_class not in ("read_only", "destructive"):
                    logger.warning(
                        "Agent %s has invalid safety_class=%r — defaulting to read_only",
                        name, safety_class,
                    )
                    safety_class = "read_only"

            # Max iterations: cap at D18 limit (30)
            max_iter = int(post.get("max_iterations", DEFAULT_MAX_ITERATIONS))
            if max_iter > DEFAULT_MAX_ITERATIONS:
                logger.warning(
                    "Agent %s max_iterations=%d exceeds D18 cap %d — clamping",
                    name, max_iter, DEFAULT_MAX_ITERATIONS,
                )
                max_iter = DEFAULT_MAX_ITERATIONS

            meta = AgentMetadata(
                name=name,
                display_name=str(post.get("name", name)),
                description=str(post.get("description", "")),
                safety_class=safety_class,
                tool_allowlist=tools_tuple,
                max_iterations=max_iter,
                prompt_file=md_path.name,
                is_orchestrator=is_orchestrator,
                orchestration_mode=mode,
            )
            result[name] = meta

        logger.info(
            "Loaded %d agents (%d orchestrators + %d sub-agents) from %s",
            len(result),
            sum(1 for m in result.values() if m.is_orchestrator),
            sum(1 for m in result.values() if not m.is_orchestrator),
            self.agents_dir,
        )
        return result

    def reload(self) -> None:
        """Force reload metadata on next access (useful for tests)."""
        self._metadata = None

    # ---------- Lookup ----------

    def get_agent(self, name: str) -> AgentMetadata | None:
        """Lookup agent metadata by name. Returns None if not found."""
        return self.metadata.get(name)

    def require_agent(self, name: str) -> AgentMetadata:
        """Lookup agent metadata. Raises KeyError if not found."""
        meta = self.get_agent(name)
        if meta is None:
            raise KeyError(
                f"Unknown agent: {name!r}. Available: {sorted(self.metadata.keys())}"
            )
        return meta

    # ---------- Listing ----------

    def list_agents(self) -> list[AgentMetadata]:
        """List all 16 agents (orchestrators + sub-agents)."""
        return list(self.metadata.values())

    def list_sub_agents(self) -> list[AgentMetadata]:
        """List 13 sub-agents (excludes the 3 orchestrators)."""
        return [m for m in self.metadata.values() if not m.is_orchestrator]

    def list_orchestrators(self) -> list[AgentMetadata]:
        """List 3 orchestrators."""
        return [m for m in self.metadata.values() if m.is_orchestrator]

    def list_by_safety_class(self, safety_class: str) -> list[AgentMetadata]:
        """Filter agents by safety_class.

        Args:
            safety_class: "read_only" | "destructive" | "advisory"

        Returns:
            List of matching agents.
        """
        return [m for m in self.metadata.values() if m.safety_class == safety_class]

    def list_destructive(self) -> list[AgentMetadata]:
        """Convenience: list all destructive agents (HITL required)."""
        return self.list_by_safety_class("destructive")

    def list_read_only(self) -> list[AgentMetadata]:
        """Convenience: list all read-only agents (no HITL required)."""
        return self.list_by_safety_class("read_only")

    # ---------- Validation ----------

    def is_tool_allowed(self, agent_name: str, tool_name: str) -> bool:
        """Check if a tool is in the agent's allowlist.

        Args:
            agent_name: Agent name (e.g. "recon")
            tool_name: Tool name (e.g. "nmap")

        Returns:
            True if tool is allowed, False otherwise (including if agent doesn't exist).
        """
        agent = self.get_agent(agent_name)
        if agent is None:
            logger.warning(
                "is_tool_allowed: unknown agent %r — returning False", agent_name,
            )
            return False
        return tool_name in agent.tool_allowlist

    def get_tool_allowlist(self, agent_name: str) -> list[str]:
        """Get tool allowlist for an agent. Returns empty list if agent not found."""
        agent = self.get_agent(agent_name)
        if agent is None:
            return []
        return list(agent.tool_allowlist)

    def is_destructive(self, agent_name: str) -> bool:
        """Check if agent requires HITL approval for tool calls."""
        agent = self.get_agent(agent_name)
        return agent is not None and agent.is_destructive

    def validate_tool_call(
        self,
        agent_name: str,
        tool_name: str,
    ) -> tuple[bool, str]:
        """Validate a tool call against agent's allowlist + safety class.

        Args:
            agent_name: Agent invoking the tool
            tool_name: Tool being invoked

        Returns:
            (allowed, reason) tuple:
                - (True, "ok") if allowed
                - (False, "agent not found: {name}") if agent doesn't exist
                - (False, "tool '{tool}' not in {agent}'s allowlist: {allowlist}") if not allowed
        """
        agent = self.get_agent(agent_name)
        if agent is None:
            return (False, f"agent not found: {agent_name!r}")
        if tool_name not in agent.tool_allowlist:
            return (
                False,
                f"tool {tool_name!r} not in {agent_name!r}'s allowlist: "
                f"{list(agent.tool_allowlist)}",
            )
        return (True, "ok")

    # ---------- Prompt loading ----------

    def load_prompt(self, name: str) -> str:
        """Load full .md content for an agent.

        Args:
            name: Agent name (e.g. "recon", "orchestrator-supervisor")

        Returns:
            Full file content (frontmatter + body) as string.

        Raises:
            KeyError: if agent not found.
        """
        agent = self.require_agent(name)
        prompt_path = self.agents_dir / agent.prompt_file
        return prompt_path.read_text(encoding="utf-8")

    # ---------- Counting ----------

    @property
    def total_count(self) -> int:
        """Total number of agents (should be 16 per D18)."""
        return len(self.metadata)

    @property
    def sub_agent_count(self) -> int:
        """Number of sub-agents (should be 13)."""
        return len(self.list_sub_agents())

    @property
    def orchestrator_count(self) -> int:
        """Number of orchestrators (should be 3)."""
        return len(self.list_orchestrators())

    def to_dict(self) -> dict[str, Any]:
        """Serialize entire registry (for debugging / API response)."""
        return {
            "agents_dir": str(self.agents_dir),
            "total_count": self.total_count,
            "orchestrator_count": self.orchestrator_count,
            "sub_agent_count": self.sub_agent_count,
            "agents": [m.to_dict() for m in self.list_agents()],
        }


# ---------- Singleton instance ----------

agent_registry = AgentRegistry()


# ---------- Convenience module-level functions ----------

def get_agent(name: str) -> AgentMetadata | None:
    """Module-level shortcut: agent_registry.get_agent(name)."""
    return agent_registry.get_agent(name)


def list_agents() -> list[AgentMetadata]:
    """Module-level shortcut: agent_registry.list_agents()."""
    return agent_registry.list_agents()


def list_sub_agents() -> list[AgentMetadata]:
    """Module-level shortcut: agent_registry.list_sub_agents()."""
    return agent_registry.list_sub_agents()


def list_orchestrators() -> list[AgentMetadata]:
    """Module-level shortcut: agent_registry.list_orchestrators()."""
    return agent_registry.list_orchestrators()


def is_tool_allowed(agent_name: str, tool_name: str) -> bool:
    """Module-level shortcut: agent_registry.is_tool_allowed(name, tool)."""
    return agent_registry.is_tool_allowed(agent_name, tool_name)


def is_destructive(agent_name: str) -> bool:
    """Module-level shortcut: agent_registry.is_destructive(name)."""
    return agent_registry.is_destructive(agent_name)


def load_prompt(name: str) -> str:
    """Module-level shortcut: agent_registry.load_prompt(name)."""
    return agent_registry.load_prompt(name)