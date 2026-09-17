"""
VAPT-AI Deep Orchestrator — W9-S4.

Deep mode: parallel sub-agents via task delegation.

Ported from CyberStrikeAI internal/multiagent/runner.go RunDeepAgent():
    - Orchestrator uses the `task` tool to delegate work to specialized sub-agents
    - Sub-agents run in parallel (no replanning — Deep is "fan-out + aggregate")
    - Sub-agent results are aggregated by the orchestrator at the end
    - D18 limit: max 5 parallel sub-agents per scan

LangGraph state graph:
    ┌─────────────────────────────────────┐
    │         planner_node                 │  ← entry point
    │  (creates sub_agent_tasks list)     │
    └────────────────┬────────────────────┘
                     │
                     ▼
    ┌─────────────────────────────────────┐
    │       fan_out_node                  │
    │  (asyncio.gather over sub_agent_   │
    │   tasks, max 5 in parallel per D18)│
    └────────────────┬────────────────────┘
                     │
                     ▼
    ┌─────────────────────────────────────┐
    │      aggregator_node                │
    │  (collects results, writes facts,  │
    │   emits findings SSE events)       │
    └────────────────┬────────────────────┘
                     │
                     ▼
                   ┌─────┐
                   │ END │
                   └─────┘

W9 stub: planner creates 3 sub-agent tasks deterministically:
    1. recon sub-agent — nmap + httpx on target
    2. vuln-triage sub-agent — nuclei scan on target
    3. reporting-remediation sub-agent — summarize results

W10+ will replace planner with LiteLLM call to decide sub-agent composition
dynamically based on target type + scope.

D18 guardrails enforced via BaseOrchestrator._check_guardrails().
"""
from __future__ import annotations

import asyncio
import logging
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
from app.pentest.events import (
    emit_scan_complete,
    emit_scan_error,
    emit_scan_progress,
    emit_scan_started,
)

logger = logging.getLogger(__name__)


# ---------- Sub-agent task definitions (W9 stub) ----------

# W9 stub: planner always creates these 3 sub-agent tasks.
# W10+ will use LiteLLM to decide task composition based on target + scope.
STUB_SUB_AGENT_TASKS: list[dict[str, Any]] = [
    {
        "subagent_type": "recon",
        "description": "Run nmap port scan + httpx service fingerprint on target.",
        "expected_deliverable": "list of open ports + service versions",
    },
    {
        "subagent_type": "vulnerability-triage",
        "description": "Run nuclei vulnerability scan on target.",
        "expected_deliverable": "list of detected vulnerabilities with severity",
    },
    {
        "subagent_type": "reporting-remediation",
        "description": "Aggregate recon + vuln-triage results into a summary report.",
        "expected_deliverable": "executive summary + remediation recommendations",
    },
]


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


# ---------- Deep Orchestrator ----------

class DeepOrchestrator(BaseOrchestrator):
    """Deep mode orchestrator — parallel sub-agents via task delegation.

    Per master plan §12 W9 task 4:
        "Implement task sub-agent for Deep mode delegation (parallel sub-agents)"

    Per master plan §12 W9 task 6:
        "network_range + subnets > 1 → Deep"

    Workflow:
        1. Planner node decides sub_agent_tasks list (W9: stub; W10: LiteLLM)
        2. Fan-out node runs all sub_agent_tasks in parallel (max 5, D18)
        3. Aggregator node merges results, emits findings, completes
    """

    MODE = OrchestratorMode.DEEP
    PROMPT_FILE = "orchestrator.md"

    def __init__(
        self,
        scan_id: str,
        target: str,
        user_prompt: str,
        max_decisions: int = 30,
        max_sub_agents: int = MAX_PARALLEL_SUBAGENTS,
        sub_agent_tasks: list[dict[str, Any]] | None = None,
    ):
        super().__init__(scan_id, target, user_prompt, max_decisions, max_sub_agents)
        # W9: use STUB_SUB_AGENT_TASKS; W10+ will accept override
        self.sub_agent_tasks = sub_agent_tasks or STUB_SUB_AGENT_TASKS
        # Enforce D18: max 5 parallel sub-agents
        if len(self.sub_agent_tasks) > self.max_sub_agents:
            logger.warning(
                "Truncating sub_agent_tasks from %d to %d (D18 max parallel)",
                len(self.sub_agent_tasks), self.max_sub_agents,
            )
            self.sub_agent_tasks = self.sub_agent_tasks[: self.max_sub_agents]
        self._graph = None

    # ---------- Graph building ----------

    def build_graph(self):
        """Compile the LangGraph state graph for Deep mode.

        Linear pipeline: planner → fan_out → aggregator → END
        (parallelism happens INSIDE fan_out_node via asyncio.gather)
        """
        StateGraph, END = _import_langgraph()

        graph = StateGraph(SharedState)

        graph.add_node("planner", self._planner_node)
        graph.add_node("fan_out", self._fan_out_node)
        graph.add_node("aggregator", self._aggregator_node)

        graph.set_entry_point("planner")
        graph.add_edge("planner", "fan_out")
        graph.add_edge("fan_out", "aggregator")
        graph.add_edge("aggregator", END)

        return graph.compile()

    # ---------- LangGraph nodes ----------

    async def _planner_node(self, state: SharedState) -> SharedState:
        """Planner: decide which sub-agent tasks to fan out.

        W9 stub: returns STUB_SUB_AGENT_TASKS.
        W10+: LLM call with self.system_prompt to decide task composition.
        """
        # Check D18
        violation = self._check_guardrails()
        if violation:
            return {**state, "status": "failed", "error": violation}

        turn = len(self.decisions)
        thought = (
            f"Planning {len(self.sub_agent_tasks)} sub-agent tasks in parallel "
            f"(D18 max {self.max_sub_agents})."
        )

        decision = AgentDecision(
            turn=turn,
            thought=thought,
            agent_name="deep-orchestrator",
            tool_name="plan",
            tool_args={
                "sub_agent_count": len(self.sub_agent_tasks),
                "sub_agent_types": [t["subagent_type"] for t in self.sub_agent_tasks],
            },
            observation=(
                f"Scheduled {len(self.sub_agent_tasks)} parallel sub-agents: "
                + ", ".join(t["subagent_type"] for t in self.sub_agent_tasks)
            ),
            tokens_used=200,  # stub
        )
        self.record_decision(decision)
        await emit_scan_progress(self.scan_id, turn, thought, "plan", decision.observation)

        return {
            **state,
            "current_agent": "deep-orchestrator",
            "sub_agent_tasks": self.sub_agent_tasks,
            "status": "running",
            "decisions": self.decisions,
        }

    async def _fan_out_node(self, state: SharedState) -> SharedState:
        """Fan-out: run all sub_agent_tasks in parallel via asyncio.gather.

        D18 enforces max 5 parallel sub-agents (truncated at __init__).
        Each sub-agent runs as a stub function (W9) — W10+ will spin up
        real LangGraph subgraphs per expert with their own tool allowlists.
        """
        tasks = state.get("sub_agent_tasks", [])
        if not tasks:
            return {**state, "status": "failed", "error": "No sub_agent_tasks to execute"}

        turn = len(self.decisions)
        thought = f"Fanning out {len(tasks)} sub-agents in parallel..."
        decision = AgentDecision(
            turn=turn,
            thought=thought,
            agent_name="deep-orchestrator",
            tool_name="task",
            tool_args={"parallel_count": len(tasks)},
            observation=f"Started {len(tasks)} sub-agents in parallel",
            tokens_used=100,
        )
        self.record_decision(decision)
        await emit_scan_progress(self.scan_id, turn, thought, "task", decision.observation)

        # Run all sub-agents in parallel
        coros = [self._run_sub_agent(i, task, state) for i, task in enumerate(tasks)]
        results = await asyncio.gather(*coros, return_exceptions=True)

        # Filter out exceptions
        sub_agent_results: list[dict[str, Any]] = []
        for i, res in enumerate(results):
            if isinstance(res, Exception):
                logger.error("Sub-agent %d failed: %s", i, res)
                sub_agent_results.append({
                    "subagent_type": tasks[i]["subagent_type"],
                    "status": "failed",
                    "error": str(res),
                })
            else:
                sub_agent_results.append(res)

        # Record aggregator decision (one per sub-agent result)
        for i, result in enumerate(sub_agent_results):
            sub_turn = len(self.decisions)
            sub_decision = AgentDecision(
                turn=sub_turn,
                thought=f"Sub-agent {result['subagent_type']} returned result.",
                agent_name=result["subagent_type"],
                tool_name=None,
                tool_args={"task_index": i},
                observation=result.get("observation", "")[:500],
                tokens_used=150,
            )
            self.record_decision(sub_decision)
            await emit_scan_progress(
                self.scan_id, sub_turn, sub_decision.thought,
                None, sub_decision.observation,
            )

        return {
            **state,
            "sub_agent_results": sub_agent_results,
            "current_agent": "deep-orchestrator",
            "decisions": self.decisions,
        }

    async def _run_sub_agent(
        self,
        task_index: int,
        task: dict[str, Any],
        state: SharedState,
    ) -> dict[str, Any]:
        """Run a single sub-agent task (W9 stub).

        W10+ will:
            - Look up expert agent config (system prompt, tool allowlist)
            - Spin up a LangGraph subgraph for that expert
            - Pass task['description'] as input
            - Execute tools via SubprocessExecutor + ScopeGuard
            - Write PentestFact entries to blackboard
            - Return structured result
        """
        subagent_type = task["subagent_type"]
        target = state.get("target", self.target)
        description = task.get("description", "")

        logger.info(
            "Sub-agent starting: scan=%s index=%d type=%s",
            self.scan_id, task_index, subagent_type,
        )

        # Simulate async work (real sub-agent would call LLM + tools)
        await asyncio.sleep(0.05)

        observation = self._stub_sub_agent_observe(subagent_type, target, description)

        return {
            "subagent_type": subagent_type,
            "task_index": task_index,
            "status": "completed",
            "observation": observation,
            "deliverable": task.get("expected_deliverable", ""),
        }

    def _stub_sub_agent_observe(
        self,
        subagent_type: str,
        target: str,
        description: str,
    ) -> str:
        """W9 stub: return canned observation per sub-agent type."""
        if subagent_type == "recon":
            return (
                f"Recon on {target}: nmap found ports 22, 80, 443, 8080; "
                f"httpx identified Apache/2.4.41."
            )
        elif subagent_type == "vulnerability-triage":
            return (
                f"Vuln scan on {target}: nuclei detected 2 low-severity findings "
                f"(missing security headers, server version disclosure)."
            )
        elif subagent_type == "reporting-remediation":
            return (
                "Aggregated 2 findings across recon + vuln-triage. "
                "Recommend: upgrade Apache, add HSTS + X-Content-Type-Options headers."
            )
        return f"Sub-agent {subagent_type} executed (W9 stub)."

    async def _aggregator_node(self, state: SharedState) -> SharedState:
        """Aggregator: merge sub_agent_results, finalize findings, emit complete."""
        # Check D18
        violation = self._check_guardrails()
        if violation:
            return {**state, "status": "failed", "error": violation}

        turn = len(self.decisions)
        sub_results = state.get("sub_agent_results", [])
        thought = (
            f"Aggregating {len(sub_results)} sub-agent results into final findings..."
        )

        # W9 stub: extract findings from reporting-remediation result
        findings: list[dict[str, Any]] = []
        for r in sub_results:
            if r.get("subagent_type") == "reporting-remediation":
                # W10+ will parse structured findings from observation
                findings.append({
                    "title": "Aggregated finding (W9 stub)",
                    "severity": "low",
                    "source_subagent": r.get("subagent_type"),
                    "observation": r.get("observation", ""),
                })

        self.findings.extend(findings)

        decision = AgentDecision(
            turn=turn,
            thought=thought,
            agent_name="deep-orchestrator",
            tool_name="aggregate",
            tool_args={"results_count": len(sub_results)},
            observation=f"Aggregated {len(findings)} findings from {len(sub_results)} sub-agents.",
            tokens_used=100,
        )
        self.record_decision(decision)
        await emit_scan_progress(self.scan_id, turn, thought, "aggregate", decision.observation)

        return {
            **state,
            "current_agent": "deep-orchestrator",
            "findings": self.findings,
            "status": "completed",
            "decisions": self.decisions,
        }

    # ---------- Run ----------

    async def run(self) -> OrchestratorResult:
        """Execute the Deep orchestration graph."""
        logger.info(
            "Deep orchestrator starting: scan=%s target=%s sub_agents=%d",
            self.scan_id, self.target, len(self.sub_agent_tasks),
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
            "current_agent": "deep-orchestrator",
            "sub_agent_tasks": [],
            "sub_agent_results": [],
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
            logger.exception("Deep orchestrator failed: scan=%s", self.scan_id)
            await emit_scan_error(self.scan_id, str(e))
            return self.finalize("failed", error=str(e))

        result = self.finalize(status, error=error)
        await emit_scan_complete(
            scan_id=self.scan_id,
            status=result.status,
            findings_count=len(result.findings),
            duration_seconds=result.duration_seconds,
        )
        return result


# ---------- Convenience function ----------

async def run_deep_scan(
    target: str,
    user_prompt: str,
    scan_id: str | None = None,
    sub_agent_tasks: list[dict[str, Any]] | None = None,
) -> OrchestratorResult:
    """One-shot: run a scan with the Deep orchestrator.

    Args:
        target: Target URL or IP
        user_prompt: Natural-language prompt
        scan_id: Optional scan ID (auto-generated if None)
        sub_agent_tasks: Optional override for stub tasks (defaults to STUB_SUB_AGENT_TASKS)

    Returns:
        OrchestratorResult with decisions, findings, agents_involved, sub_agent_count.
    """
    if scan_id is None:
        scan_id = generate_scan_id()

    orchestrator = DeepOrchestrator(
        scan_id=scan_id,
        target=target,
        user_prompt=user_prompt,
        sub_agent_tasks=sub_agent_tasks,
    )
    return await orchestrator.run()