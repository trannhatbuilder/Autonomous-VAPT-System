"""
VAPT-AI Orchestration Base Classes — W9-S2.

Defines the shared abstractions used by all 3 orchestration modes:

    1. OrchestratorMode  — enum (deep / plan_execute / supervisor)
    2. SharedState       — LangGraph TypedDict state shared between nodes
    3. AgentDecision     — one decision in the orchestration loop
    4. OrchestratorResult — final result returned by orchestrator.run()
    5. BaseOrchestrator  — abstract base class for all 3 modes
    6. load_orchestrator_prompt() — loads .md prompt file with YAML frontmatter

This module is dependency-light:
    - Uses stdlib only (no LangGraph import) so it can be imported in tests
      without requiring langgraph to be installed.
    - LangGraph is imported lazily inside each mode module.

D18 guardrails (always enforced — see _check_guardrails()):
    - Max 30 decisions per scan
    - Max 5 parallel sub-agents (Deep mode)
    - Max 16 agents (fixed — cannot create new at runtime)
    - Max 2M tokens per scan
    - Max 4 hours per scan
"""
from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any, TypedDict

logger = logging.getLogger(__name__)


# ---------- Constants (D18 guardrails — master plan §8.4) ----------

MAX_DECISIONS_PER_SCAN = 30
MAX_PARALLEL_SUBAGENTS = 5  # Deep mode only
MAX_AGENTS = 16  # Fixed — cannot create new at runtime
MAX_TOOLS_PER_AGENT = 30
MAX_TOKENS_PER_SCAN = 2_000_000
MAX_SCAN_DURATION_SECONDS = 4 * 60 * 60  # 4 hours


# ---------- Mode enum ----------

class OrchestratorMode(str, Enum):
    """The 3 orchestration modes per master plan §4.2.

    Selection logic (master plan §12 W9 task 6):
        single_web_app + ≤5 steps       → supervisor (default)
        network_range + subnets > 1     → deep
        full_kill_chain or post_exploit → plan_execute
        default                          → supervisor
    """
    DEEP = "deep"
    PLAN_EXECUTE = "plan_execute"
    SUPERVISOR = "supervisor"

    def __str__(self) -> str:
        return self.value


# ---------- LangGraph shared state ----------

class SharedState(TypedDict, total=False):
    """LangGraph state shared between nodes in the orchestration graph.

    `total=False` so partial states are valid (LangGraph merges state updates).

    IMPORTANT: LangGraph only tracks fields declared in this TypedDict.
    Undeclared fields in node return values are silently dropped.

    Fields:
        scan_id:              Unique scan identifier (e.g. "scan_abc123")
        target:               Target URL or IP (e.g. "http://example.com")
        user_prompt:          Natural-language prompt from user
        mode:                 OrchestratorMode value
        decisions:            Accumulating list of AgentDecision (turn log)
        findings:             Accumulating list of finding dicts
        current_agent:        Name of the agent currently in control
        next_agent:           Name of the next agent to transfer to (Supervisor)
                              or None if done
        plan:                 Structured plan (Plan-Execute only) — list of plan steps
        current_step:         Index of current plan step (Plan-Execute only)
        sub_agent_tasks:      List of sub-agent task descriptions (Deep only)
        sub_agent_results:    List of sub-agent results (Deep only)
        replanner_decision:   Plan-Execute replanner verdict ("continue"/"replan"/"exit")
        total_tokens:         Running token count
        error:                Error message (None if no error)
        status:               "running" | "completed" | "failed" | "max_iterations" | "timeout"
    """
    scan_id: str
    target: str
    user_prompt: str
    mode: str
    decisions: list[AgentDecision]
    findings: list[dict[str, Any]]
    current_agent: str | None
    next_agent: str | None
    plan: list[dict[str, Any]]
    current_step: int
    sub_agent_tasks: list[dict[str, Any]]
    sub_agent_results: list[dict[str, Any]]
    replanner_decision: str
    total_tokens: int
    error: str | None
    status: str


# ---------- Agent decision (turn log entry) ----------

@dataclass
class AgentDecision:
    """One decision in the orchestration loop.

    Mirrors app.agents.agent.AgentDecision but kept independent here so
    orchestration package has no circular import on the single-agent module.
    """
    turn: int
    thought: str
    agent_name: str           # which agent made this decision (orchestrator or sub-agent)
    tool_name: str | None     # tool called (None = pure reasoning / transfer)
    tool_args: dict[str, Any]
    observation: str = ""
    tokens_used: int = 0
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))


# ---------- Final result ----------

@dataclass
class OrchestratorResult:
    """Final result returned by BaseOrchestrator.run().

    Compatible with app.agents.agent.ScanResult for API response uniformity.
    """
    scan_id: str
    target: str
    user_prompt: str
    mode: str  # OrchestratorMode value
    status: str  # completed / failed / max_iterations / timeout
    decisions: list[AgentDecision] = field(default_factory=list)
    findings: list[dict[str, Any]] = field(default_factory=list)
    total_tokens: int = 0
    duration_seconds: float = 0.0
    error: str | None = None
    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    completed_at: datetime | None = None
    # Orchestration-specific extras
    agents_involved: list[str] = field(default_factory=list)
    sub_agent_count: int = 0
    plan_steps_total: int = 0  # Plan-Execute only
    plan_steps_completed: int = 0  # Plan-Execute only

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict (for FastAPI response)."""
        return {
            "scan_id": self.scan_id,
            "target": self.target,
            "user_prompt": self.user_prompt,
            "mode": self.mode,
            "status": self.status,
            "decisions_count": len(self.decisions),
            "decisions": [
                {
                    "turn": d.turn,
                    "thought": d.thought,
                    "agent_name": d.agent_name,
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
            "agents_involved": self.agents_involved,
            "sub_agent_count": self.sub_agent_count,
            "plan_steps_total": self.plan_steps_total,
            "plan_steps_completed": self.plan_steps_completed,
        }


# ---------- Base orchestrator ----------

class BaseOrchestrator:
    """Abstract base class for all 3 orchestration modes.

    Subclasses (langgraph_deep.py, langgraph_plan_execute.py, langgraph_supervisor.py)
    must implement:
        - build_graph()  → compiled LangGraph runnable
        - run()          → OrchestratorResult (calls build_graph() internally)

    Subclasses inherit:
        - D18 guardrail enforcement (_check_guardrails)
        - SSE event emission (emit helpers)
        - Blackboard access (Blackboard instance)
        - Scope guard + subprocess executor reuse from W3/W4

    W9 stubs LLM decisions (deterministic) — W10-W12 will wire LiteLLM.
    """

    # Subclasses override this
    MODE: OrchestratorMode = OrchestratorMode.SUPERVISOR
    PROMPT_FILE: str = "orchestrator-supervisor.md"  # default

    def __init__(
        self,
        scan_id: str,
        target: str,
        user_prompt: str,
        max_decisions: int = MAX_DECISIONS_PER_SCAN,
        max_sub_agents: int = MAX_PARALLEL_SUBAGENTS,
    ):
        # Cap by D18
        self.scan_id = scan_id
        self.target = target
        self.user_prompt = user_prompt
        self.max_decisions = min(max_decisions, MAX_DECISIONS_PER_SCAN)
        self.max_sub_agents = min(max_sub_agents, MAX_PARALLEL_SUBAGENTS)

        # Runtime accumulators
        self.decisions: list[AgentDecision] = []
        self.findings: list[dict[str, Any]] = []
        self.total_tokens = 0
        self.start_time = time.time()
        self.agents_involved: set[str] = set()

        # Load system prompt
        self.system_prompt: str = load_orchestrator_prompt(self.PROMPT_FILE)

        logger.info(
            "Orchestrator initialized: scan=%s mode=%s target=%s prompt_len=%d",
            self.scan_id, self.MODE.value, self.target, len(self.system_prompt),
        )

    # ---------- Guardrails (D18) ----------

    def _check_guardrails(self) -> str | None:
        """Check D18 guardrails. Returns error message if violated, else None.

        Called before each decision in the orchestration loop.
        """
        # Decision count
        if len(self.decisions) >= self.max_decisions:
            return f"Exceeded {self.max_decisions} decisions per scan (D18)"

        # Token budget
        if self.total_tokens > MAX_TOKENS_PER_SCAN:
            return f"Exceeded {MAX_TOKENS_PER_SCAN} tokens per scan (D18)"

        # Time budget
        elapsed = time.time() - self.start_time
        if elapsed > MAX_SCAN_DURATION_SECONDS:
            return f"Exceeded {MAX_SCAN_DURATION_SECONDS}s per scan (D18)"

        # Agent count
        if len(self.agents_involved) > MAX_AGENTS:
            return f"Exceeded {MAX_AGENTS} agents per scan (D18)"

        return None

    # ---------- Helpers used by subclasses ----------

    def record_decision(self, decision: AgentDecision) -> None:
        """Append a decision to the log + update accumulators."""
        self.decisions.append(decision)
        self.total_tokens += decision.tokens_used
        self.agents_involved.add(decision.agent_name)
        logger.info(
            "Turn %d: agent=%s tool=%s args=%s",
            decision.turn, decision.agent_name, decision.tool_name, decision.tool_args,
        )

    def finalize(self, status: str, error: str | None = None) -> OrchestratorResult:
        """Build the final OrchestratorResult."""
        return OrchestratorResult(
            scan_id=self.scan_id,
            target=self.target,
            user_prompt=self.user_prompt,
            mode=self.MODE.value,
            status=status,
            decisions=self.decisions,
            findings=self.findings,
            total_tokens=self.total_tokens,
            duration_seconds=time.time() - self.start_time,
            error=error,
            completed_at=datetime.now(UTC),
            agents_involved=sorted(self.agents_involved),
            sub_agent_count=len(self.agents_involved) - 1,  # exclude orchestrator
        )

    # ---------- Abstract interface (subclasses implement) ----------

    async def run(self) -> OrchestratorResult:
        """Execute the orchestration graph. Subclasses must implement."""
        raise NotImplementedError(
            f"{self.__class__.__name__}.run() not implemented. "
            f"Subclasses must override this method."
        )


# ---------- Prompt loader ----------

_AGENT_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "agents"


def load_orchestrator_prompt(filename: str) -> str:
    """Load an orchestrator prompt .md file (with YAML frontmatter).

    Returns the full file content (frontmatter + body). The caller is
    responsible for parsing frontmatter if needed.

    Args:
        filename: e.g. "orchestrator.md", "orchestrator-plan-execute.md",
                  "orchestrator-supervisor.md"

    Returns:
        File content as string.

    Raises:
        FileNotFoundError: if the prompt file doesn't exist.
    """
    prompt_path = _AGENT_PROMPTS_DIR / filename
    if not prompt_path.exists():
        raise FileNotFoundError(
            f"Orchestrator prompt file not found: {prompt_path}. "
            f"Expected one of: orchestrator.md, orchestrator-plan-execute.md, "
            f"orchestrator-supervisor.md"
        )
    return prompt_path.read_text(encoding="utf-8")


# ---------- Convenience: scan_id generator ----------

def generate_scan_id() -> str:
    """Generate a new scan_id (e.g. 'scan_a1b2c3d4e5f6')."""
    return f"scan_{uuid.uuid4().hex[:12]}"