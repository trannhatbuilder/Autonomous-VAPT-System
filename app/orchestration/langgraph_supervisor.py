"""
VAPT-AI Supervisor Orchestrator — W9-S3 / W10-S5.

Default orchestration mode. Uses the **transfer mechanism** to delegate
to specialized expert sub-agents (one at a time, serially).

Ported from CyberStrikeAI internal/multiagent/runner.go (supervisor mode):
    - Orchestrator has `transfer` tool to delegate to an expert
    - Orchestrator has `exit` tool to end session
    - After expert returns, orchestrator aggregates and decides next transfer or exit
    - Avoids ping-pong transfers (only transfer on new, specific goal)

W10-S5 update: now uses `agent_registry` instead of hardcoded `EXPERT_AGENTS` dict.
The 2 stub experts (recon, reporting-remediation) from W9 are still available
as default transfer targets — they're loaded dynamically from the registry now.

LangGraph state graph:
    ┌─────────────────────────┐
    │     supervisor_node     │  ← entry point
    │  (decides next agent)   │
    └───────────┬─────────────┘
                │
       ┌────────┴────────┐
       │ route_from_sup  │
       └────────┬────────┘
                │
       ┌────────┴────────┬────────────────────┬─────────────┐
       ▼                 ▼                    ▼             ▼
  ┌─────────┐      ┌─────────┐         ┌─────────┐    ┌─────────┐
  │ recon   │      │vuln_tri │         │ penetr  │    │ report  │
  │  node   │      │age_node │         │ ation_n │    │  node   │
  └────┬────┘      └────┬────┘         └────┬────┘    └────┬────┘
       │                │                   │              │
       └────────────────┴───────────────────┴──────────────┘
                │
                ▼
       (all expert nodes loop back to supervisor_node)
                │
                ▼
            ┌────────┐
            │  END   │  (when next_agent == None)
            └────────┘

W9 stub LLM decisions (deterministic sequence):
    1. supervisor → transfer to "recon" expert
    2. recon runs nmap + httpx, writes facts to blackboard, returns
    3. supervisor → transfer to "reporting-remediation" expert
    4. reporting expert reads blackboard, summarizes findings, returns
    5. supervisor → exit (next_agent = None)

W10-S5: experts now loaded from agent_registry. Default transfer targets:
    ["recon", "reporting-remediation"] (same as W9 for backward compat).
W11+ will let LiteLLM pick from all 13 sub-agents dynamically.

D18 guardrails enforced via BaseOrchestrator._check_guardrails().
"""
from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import UTC, datetime
from typing import Any

from app.orchestration.base import (
    AgentDecision,
    BaseOrchestrator,
    MAX_PARALLEL_SUBAGENTS,
    OrchestratorMode,
    OrchestratorResult,
    SharedState,
    generate_scan_id,
)
from app.agents.registry import agent_registry
from app.pentest.events import (
    emit_scan_complete,
    emit_scan_error,
    emit_scan_progress,
    emit_scan_started,
)

logger = logging.getLogger(__name__)


# ---------- Backward compatibility: legacy EXPERT_AGENTS dict ----------

# W9 used a hardcoded dict. W10-S5 derives it from agent_registry so old
# code (e.g. tests/integration/test_orchestration_modes.py) keeps working.
def _build_legacy_expert_agents() -> dict[str, dict[str, Any]]:
    """Build the legacy EXPERT_AGENTS dict from agent_registry.

    Only includes the 2 W9 stub experts (recon, reporting-remediation).
    """
    result: dict[str, dict[str, Any]] = {}
    for name in ("recon", "reporting-remediation"):
        meta = agent_registry.get_agent(name)
        if meta is None:
            continue
        result[name] = {
            "name": meta.name,
            "description": meta.description,
            "tools": list(meta.tool_allowlist),
            "safety_class": meta.safety_class,
        }
    return result


# Backward-compat alias (W9 tests reference this)
EXPERT_AGENTS: dict[str, dict[str, Any]] = _build_legacy_expert_agents()


# ---------- Default transfer targets ----------

# W10-S5: supervisor still uses 2-expert flow (backward compat with W9 stub).
# W11+ will allow LiteLLM to pick from all 13 sub-agents dynamically.
DEFAULT_TRANSFER_TARGETS: list[str] = ["recon", "reporting-remediation"]


# ---------- LangGraph import (lazy) ----------

def _import_langgraph():
    """Lazily import LangGraph — defers hard dependency until first use."""
    try:
        from langgraph.graph import StateGraph, END  # type: ignore[import-untyped]
        return StateGraph, END
    except ImportError as e:
        raise ImportError(
            "LangGraph is required for orchestration. Install with:\n"
            "  pip install 'langgraph>=0.2.0'\n"
            "Or:\n"
            "  pip install -r requirements.txt"
        ) from e


# ---------- Supervisor Orchestrator ----------

class SupervisorOrchestrator(BaseOrchestrator):
    """Supervisor mode orchestrator — transfer mechanism to expert agents.

    Default mode per master plan §12 W9 task 6:
        single_web_app + ≤5 steps       → supervisor (default)
        default                          → supervisor

    The supervisor decides which expert to transfer to next. After the
    expert returns, the supervisor aggregates the result and decides the
    next transfer or exits.

    W10-S5: experts now come from agent_registry. The orchestrator looks
    up agent metadata (safety_class, tool_allowlist) at runtime instead
    of hardcoding it.
    """

    MODE = OrchestratorMode.SUPERVISOR
    PROMPT_FILE = "orchestrator-supervisor.md"

    def __init__(
        self,
        scan_id: str,
        target: str,
        user_prompt: str,
        max_decisions: int = 30,
        max_sub_agents: int = MAX_PARALLEL_SUBAGENTS,
        experts: dict[str, dict[str, Any]] | None = None,
        transfer_targets: list[str] | None = None,
    ):
        super().__init__(scan_id, target, user_prompt, max_decisions, max_sub_agents)
        # W9 backward compat: still accept `experts` dict override
        # W10 default: derive from agent_registry
        self.experts = experts if experts is not None else EXPERT_AGENTS
        # W10-S5: configurable transfer sequence (defaults to W9 stub: recon → reporting-remediation)
        self.transfer_targets = transfer_targets or DEFAULT_TRANSFER_TARGETS
        # Filter transfer_targets to only include registered sub-agents
        self.transfer_targets = [
            t for t in self.transfer_targets
            if agent_registry.get_agent(t) is not None
        ]
        if not self.transfer_targets:
            # Fallback: use first 2 sub-agents from registry
            all_subs = agent_registry.list_sub_agents()
            self.transfer_targets = [s.name for s in all_subs[:2]]
        self._graph = None  # lazy-compiled on first run

    # ---------- Graph building ----------

    def build_graph(self):
        """Compile the LangGraph state graph for supervisor mode."""
        StateGraph, END = _import_langgraph()

        graph = StateGraph(SharedState)
        graph.add_node("supervisor", self._supervisor_node)

        # Add one node per transfer target (registry-validated)
        for expert_name in self.transfer_targets:
            graph.add_node(expert_name, self._make_expert_node(expert_name))

        graph.set_entry_point("supervisor")

        def route_from_supervisor(state: SharedState) -> str:
            next_agent = state.get("next_agent")
            if next_agent is None or next_agent == "exit":
                return END
            if next_agent not in self.transfer_targets:
                logger.warning(
                    "Supervisor routed to non-targeted expert %r — ending scan",
                    next_agent,
                )
                return END
            return next_agent

        graph.add_conditional_edges(
            "supervisor",
            route_from_supervisor,
            {name: name for name in self.transfer_targets} | {"__end__": END},
        )

        for expert_name in self.transfer_targets:
            graph.add_edge(expert_name, "supervisor")

        return graph.compile()

    # ---------- LangGraph nodes ----------

    async def _supervisor_node(self, state: SharedState) -> SharedState:
        """Supervisor decision node — decides which expert to transfer to next.

        W9 stub: deterministic sequence based on supervisor's own call count.
        W10-S5: now reads transfer_targets list (configurable).
        W11+: LiteLLM call with self.system_prompt.
        """
        violation = self._check_guardrails()
        if violation:
            return {**state, "next_agent": None, "status": "failed", "error": violation}

        supervisor_turns = sum(
            1 for d in self.decisions if d.agent_name == "supervisor"
        )
        thought, next_agent = self._stub_decide_next(supervisor_turns)

        turn = len(self.decisions)
        decision = AgentDecision(
            turn=turn,
            thought=thought,
            agent_name="supervisor",
            tool_name="transfer" if next_agent else "exit",
            tool_args={"target_agent": next_agent} if next_agent else {},
            observation=f"Transferring to {next_agent}" if next_agent else "Exit",
            tokens_used=100,
        )
        self.record_decision(decision)
        await emit_scan_progress(
            scan_id=self.scan_id, turn=turn, thought=thought,
            tool_name=decision.tool_name, observation=decision.observation,
        )

        return {
            **state,
            "next_agent": next_agent,
            "current_agent": "supervisor",
            "status": "running" if next_agent else "completed",
            "decisions": self.decisions,
        }

    def _stub_decide_next(self, supervisor_turns: int) -> tuple[str, str | None]:
        """Deterministic stub: supervisor picks next expert from transfer_targets."""
        if supervisor_turns < len(self.transfer_targets):
            next_agent = self.transfer_targets[supervisor_turns]
            return (
                f"Transferring to {next_agent} (transfer target {supervisor_turns + 1}/{len(self.transfer_targets)}).",
                next_agent,
            )
        return ("All transfer targets exhausted. Exiting.", None)

    def _make_expert_node(self, expert_name: str):
        """Factory: create an async node function for a specific expert.

        W10-S5: looks up agent metadata from registry instead of hardcoded dict.
        """
        meta = agent_registry.get_agent(expert_name)
        if meta is None:
            raise ValueError(f"Unknown expert: {expert_name!r}")

        async def expert_node(state: SharedState) -> SharedState:
            logger.info(
                "Expert node executing: scan=%s expert=%s safety=%s tools=%s",
                self.scan_id, expert_name, meta.safety_class, list(meta.tool_allowlist),
            )

            turn = len(self.decisions)
            target = state.get("target", self.target)
            observation = self._stub_expert_observe(expert_name, target)

            decision = AgentDecision(
                turn=turn,
                thought=f"Expert {expert_name} executed (W10 stub). safety_class={meta.safety_class}",
                agent_name=expert_name,
                tool_name=meta.tool_allowlist[0] if meta.tool_allowlist else None,
                tool_args={"target": target},
                observation=observation,
                tokens_used=150,
            )
            self.record_decision(decision)
            await emit_scan_progress(
                scan_id=self.scan_id, turn=turn, thought=decision.thought,
                tool_name=decision.tool_name, observation=observation[:500],
            )

            return {**state, "current_agent": expert_name, "decisions": self.decisions}

        return expert_node

    def _stub_expert_observe(self, expert_name: str, target: str) -> str:
        """W10 stub: return canned observation per expert."""
        if expert_name == "recon":
            return (
                f"Recon on {target} (W10 stub): nmap found ports 22, 80, 443; "
                f"httpx identified Apache/2.4.41 + OpenSSL/1.1.1f."
            )
        elif expert_name == "reporting-remediation":
            return (
                "Reporting (W10 stub): aggregated 0 findings from blackboard. "
                "Recommend full scan with Deep mode for thorough coverage."
            )
        # Generic fallback for any other expert_name (when transfer_targets is customized)
        meta = agent_registry.get_agent(expert_name)
        return f"Expert {expert_name} ({meta.display_name if meta else 'unknown'}) executed (W10 stub)."

    # ---------- Run ----------

    async def run(self) -> OrchestratorResult:
        """Execute the supervisor orchestration graph."""
        logger.info(
            "Supervisor orchestrator starting: scan=%s target=%s transfer_targets=%s",
            self.scan_id, self.target, self.transfer_targets,
        )

        await emit_scan_started(self.scan_id, self.target, self.user_prompt)

        if self._graph is None:
            self._graph = self.build_graph()

        initial_state: SharedState = {
            "scan_id": self.scan_id,
            "target": self.target,
            "user_prompt": self.user_prompt,
            "mode": self.MODE.value,
            "decisions": [],
            "findings": [],
            "current_agent": "supervisor",
            "next_agent": None,
            "total_tokens": 0,
            "error": None,
            "status": "running",
        }

        try:
            final_state = await self._graph.ainvoke(initial_state)
            error = final_state.get("error")
            status = final_state.get("status", "completed")

            if len(self.decisions) >= self.max_decisions and status != "failed":
                status = "max_iterations"
                error = f"Exceeded {self.max_decisions} decisions per scan (D18)"

        except Exception as e:
            logger.exception("Supervisor orchestrator failed: scan=%s", self.scan_id)
            await emit_scan_error(self.scan_id, str(e))
            return self.finalize("failed", error=str(e))

        result = self.finalize(status, error=error)
        await emit_scan_complete(
            scan_id=self.scan_id, status=result.status,
            findings_count=len(result.findings),
            duration_seconds=result.duration_seconds,
        )
        return result


# ---------- Convenience function ----------

async def run_supervisor_scan(
    target: str,
    user_prompt: str,
    scan_id: str | None = None,
    experts: dict[str, dict[str, Any]] | None = None,
    transfer_targets: list[str] | None = None,
) -> OrchestratorResult:
    """One-shot: run a scan with the Supervisor orchestrator.

    Args:
        target: Target URL or IP
        user_prompt: Natural-language prompt
        scan_id: Optional scan ID (auto-generated if None)
        experts: Optional legacy experts dict (W9 backward compat — prefer transfer_targets)
        transfer_targets: Optional list of agent names to transfer to in sequence.
            Defaults to ["recon", "reporting-remediation"] (W9 stub).
            Can include any of the 13 sub-agents in the registry, e.g.:
            ["recon", "vulnerability-triage", "penetration", "reporting-remediation"]

    Returns:
        OrchestratorResult with decisions, findings, agents_involved.
    """
    if scan_id is None:
        scan_id = generate_scan_id()

    orchestrator = SupervisorOrchestrator(
        scan_id=scan_id,
        target=target,
        user_prompt=user_prompt,
        experts=experts,
        transfer_targets=transfer_targets,
    )
    return await orchestrator.run()