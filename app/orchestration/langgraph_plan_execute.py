"""
VAPT-AI Plan-Execute Orchestrator — W9-S5.

Plan-Execute mode: planner → executor → replanner loop with re-planning.

Ported from CyberStrikeAI internal/multiagent/plan_execute_executor.go:
    - Planner produces a structured Plan (list of plan steps)
    - Executor executes one step at a time (serial)
    - Replanner reviews executed steps + remaining plan, decides:
        - Continue with next step
        - Re-plan (modify remaining steps)
        - Exit (objective complete)

LangGraph state graph:
    ┌─────────────────────────────┐
    │      planner_node           │  ← entry point
    │  (creates initial plan)     │
    └──────────────┬──────────────┘
                   │
                   ▼
    ┌─────────────────────────────┐
    │      executor_node          │
    │  (executes current step)    │
    └──────────────┬──────────────┘
                   │
                   ▼
    ┌─────────────────────────────┐
    │     replanner_node         │
    │  (decides: continue /      │
    │   replan / exit)            │
    └──────────────┬──────────────┘
                   │
       ┌───────────┴───────────┐
       │                       │
   next_step?              replan / exit?
       │                       │
       ▼                       ▼
  executor_node            ┌─────┐
   (loop back)             │ END │
                           └─────┘

W9 stub: planner creates 3-step plan deterministically:
    step 0: recon (nmap + httpx)
    step 1: vuln-triage (nuclei)
    step 2: report (aggregate + summarize)

Replanner stub:
    - After step 0 (recon): continue to step 1
    - After step 1 (vuln-triage): continue to step 2
    - After step 2 (report): exit (objective complete)

W10+ will replace planner + replanner with LiteLLM calls using
self.system_prompt (orchestrator-plan-execute.md).

D18 guardrails enforced via BaseOrchestrator._check_guardrails().
"""
from __future__ import annotations

import logging
from typing import Any

from app.orchestration.base import (
    AgentDecision,
    BaseOrchestrator,
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


# ---------- Plan step types ----------

# W9 stub: planner creates this 3-step plan for every scan.
# W10+ will use LiteLLM to construct plan based on target + scope.
STUB_PLAN: list[dict[str, Any]] = [
    {
        "step": 0,
        "name": "recon",
        "description": "Run nmap port scan + httpx service fingerprint on target.",
        "tools": ["nmap", "httpx"],
        "success_criteria": "list of open ports + service versions persisted to blackboard",
    },
    {
        "step": 1,
        "name": "vulnerability-triage",
        "description": "Run nuclei vulnerability scan on target.",
        "tools": ["nuclei"],
        "success_criteria": "list of detected vulnerabilities with severity",
    },
    {
        "step": 2,
        "name": "reporting-remediation",
        "description": "Aggregate recon + vuln-triage results, generate summary report.",
        "tools": [],
        "success_criteria": "executive summary with findings + remediation",
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


# ---------- Plan-Execute Orchestrator ----------

class PlanExecuteOrchestrator(BaseOrchestrator):
    """Plan-Execute mode orchestrator — planner → executor → replanner loop.

    Per master plan §12 W9 task 5:
        "Implement Planner + Replanner for Plan-Execute mode"

    Per master plan §12 W9 task 6:
        "full_kill_chain or has_post_exploitation → Plan-Execute"

    Workflow:
        1. Planner node creates initial Plan (W9: STUB_PLAN; W10: LiteLLM)
        2. Executor node executes current step
        3. Replanner node reviews state, decides:
            - "continue" → loop back to executor with next step
            - "replan" → call planner to modify remaining steps
            - "exit" → END
    """

    MODE = OrchestratorMode.PLAN_EXECUTE
    PROMPT_FILE = "orchestrator-plan-execute.md"

    def __init__(
        self,
        scan_id: str,
        target: str,
        user_prompt: str,
        max_decisions: int = 30,
        plan: list[dict[str, Any]] | None = None,
    ):
        super().__init__(scan_id, target, user_prompt, max_decisions)
        # W9: use STUB_PLAN; W10+ will accept override
        self.plan = plan or [dict(s) for s in STUB_PLAN]
        self._graph = None

    # ---------- Graph building ----------

    def build_graph(self):
        """Compile the LangGraph state graph for Plan-Execute mode.

        Nodes:
            - planner_node: creates/updates the plan
            - executor_node: executes current step
            - replanner_node: decides next action (continue / replan / exit)

        Edges:
            - START → planner_node
            - planner_node → executor_node
            - executor_node → replanner_node
            - replanner_node → conditional (executor_node if continue, END if exit)
        """
        StateGraph, END = _import_langgraph()

        graph = StateGraph(SharedState)

        graph.add_node("planner", self._planner_node)
        graph.add_node("executor", self._executor_node)
        graph.add_node("replanner", self._replanner_node)

        graph.set_entry_point("planner")
        graph.add_edge("planner", "executor")
        graph.add_edge("executor", "replanner")

        def route_from_replanner(state: SharedState) -> str:
            """Route based on replanner's decision.

            The replanner_node already incremented current_step when it decided
            "continue", so here we just respect the decision field directly.
            Don't double-check current_step bounds — that causes premature exit.
            """
            decision = state.get("replanner_decision", "exit")
            if decision == "continue":
                return "executor"
            if decision == "replan":
                return "planner"
            return END  # "exit" or unknown

        graph.add_conditional_edges(
            "replanner",
            route_from_replanner,
            {"executor": "executor", "planner": "planner", "__end__": END},
        )

        return graph.compile()

    # ---------- LangGraph nodes ----------

    async def _planner_node(self, state: SharedState) -> SharedState:
        """Planner: create or update the structured plan.

        W9 stub: returns STUB_PLAN unchanged.
        W10+: LLM call with self.system_prompt to construct plan based on
              target type + scope + previous execution results.
        """
        violation = self._check_guardrails()
        if violation:
            return {**state, "status": "failed", "error": violation}

        turn = len(self.decisions)
        is_replan = "plan" in state and len(state["plan"]) > 0
        thought = (
            f"{'Re-planning' if is_replan else 'Creating'} plan with {len(self.plan)} steps."
        )

        decision = AgentDecision(
            turn=turn,
            thought=thought,
            agent_name="plan-execute-planner",
            tool_name="write_plan",
            tool_args={
                "step_count": len(self.plan),
                "step_names": [s["name"] for s in self.plan],
            },
            observation=(
                f"Plan created with {len(self.plan)} steps: "
                + " → ".join(s["name"] for s in self.plan)
            ),
            tokens_used=200,
        )
        self.record_decision(decision)
        await emit_scan_progress(self.scan_id, turn, thought, "write_plan", decision.observation)

        return {
            **state,
            "current_agent": "plan-execute-planner",
            "plan": self.plan,
            "current_step": 0,
            "status": "running",
            "decisions": self.decisions,
        }

    async def _executor_node(self, state: SharedState) -> SharedState:
        """Executor: execute the current plan step.

        W9 stub: returns canned observation based on step name.
        W10+: invoke MCP tools via SubprocessExecutor + ScopeGuard,
              write PentestFact entries to blackboard.
        """
        violation = self._check_guardrails()
        if violation:
            return {**state, "status": "failed", "error": violation}

        current_step = state.get("current_step", 0)
        plan = state.get("plan", self.plan)

        if current_step >= len(plan):
            return {**state, "status": "failed", "error": f"Step {current_step} out of range"}

        step = plan[current_step]
        step_name = step["name"]
        step_desc = step.get("description", "")

        turn = len(self.decisions)
        thought = f"Executing step {current_step} ({step_name}): {step_desc[:80]}"

        observation = self._stub_execute_step(step_name, state.get("target", self.target))

        decision = AgentDecision(
            turn=turn,
            thought=thought,
            agent_name=step_name,
            tool_name=step.get("tools", [None])[0] if step.get("tools") else None,
            tool_args={"step": current_step, "step_name": step_name},
            observation=observation,
            tokens_used=150,
        )
        self.record_decision(decision)
        await emit_scan_progress(
            self.scan_id, turn, thought, decision.tool_name, observation[:500],
        )

        return {
            **state,
            "current_agent": step_name,
            "decisions": self.decisions,
        }

    def _stub_execute_step(self, step_name: str, target: str) -> str:
        """W9 stub: return canned observation per step name."""
        if step_name == "recon":
            return (
                f"Recon on {target}: nmap found ports 22, 80, 443; "
                f"httpx identified Apache/2.4.41 + OpenSSL/1.1.1f."
            )
        elif step_name == "vulnerability-triage":
            return (
                f"Vuln scan on {target}: nuclei detected 2 low-severity findings "
                f"(missing HSTS, server version disclosure)."
            )
        elif step_name == "reporting-remediation":
            return (
                "Aggregated findings: 2 low-severity issues. "
                "Recommend: upgrade Apache, add HSTS + X-Content-Type-Options headers."
            )
        return f"Step {step_name} executed (W9 stub)."

    async def _replanner_node(self, state: SharedState) -> SharedState:
        """Replanner: decide next action (continue / replan / exit).

        W9 stub: deterministic
            - After step 0 (recon) → continue
            - After step 1 (vuln-triage) → continue
            - After step 2 (report) → exit
        W10+: LLM call to review state + decide.
        """
        violation = self._check_guardrails()
        if violation:
            return {**state, "status": "failed", "error": violation, "replanner_decision": "exit"}

        current_step = state.get("current_step", 0)
        plan_length = len(state.get("plan", []))
        turn = len(self.decisions)

        # Stub decision logic
        if current_step + 1 < plan_length:
            decision_type = "continue"
            thought = (
                f"Step {current_step} completed. Continuing to step {current_step + 1}."
            )
        else:
            decision_type = "exit"
            thought = (
                f"All {plan_length} steps completed. Exiting plan-execute loop."
            )

        decision = AgentDecision(
            turn=turn,
            thought=thought,
            agent_name="plan-execute-replanner",
            tool_name="decide",
            tool_args={
                "current_step": current_step,
                "decision": decision_type,
                "next_step": current_step + 1 if decision_type == "continue" else None,
            },
            observation=f"Replanner decision: {decision_type}",
            tokens_used=100,
        )
        self.record_decision(decision)
        await emit_scan_progress(self.scan_id, turn, thought, "decide", decision.observation)

        # Increment current_step if continuing
        new_step = current_step + 1 if decision_type == "continue" else current_step

        return {
            **state,
            "current_agent": "plan-execute-replanner",
            "current_step": new_step,
            "replanner_decision": decision_type,
            "status": "completed" if decision_type == "exit" else "running",
            "decisions": self.decisions,
        }

    # ---------- Run ----------

    async def run(self) -> OrchestratorResult:
        """Execute the Plan-Execute orchestration graph."""
        logger.info(
            "Plan-Execute orchestrator starting: scan=%s target=%s plan_steps=%d",
            self.scan_id, self.target, len(self.plan),
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
            "current_agent": "plan-execute-planner",
            "plan": [],
            "current_step": 0,
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
            logger.exception("Plan-Execute orchestrator failed: scan=%s", self.scan_id)
            await emit_scan_error(self.scan_id, str(e))
            return self.finalize("failed", error=str(e))

        result = self.finalize(status, error=error)
        # Add plan-execute specific metadata
        result.plan_steps_total = len(self.plan)
        result.plan_steps_completed = len(self.plan)  # all steps completed on success
        await emit_scan_complete(
            scan_id=self.scan_id,
            status=result.status,
            findings_count=len(result.findings),
            duration_seconds=result.duration_seconds,
        )
        return result


# ---------- Convenience function ----------

async def run_plan_execute_scan(
    target: str,
    user_prompt: str,
    scan_id: str | None = None,
    plan: list[dict[str, Any]] | None = None,
) -> OrchestratorResult:
    """One-shot: run a scan with the Plan-Execute orchestrator.

    Args:
        target: Target URL or IP
        user_prompt: Natural-language prompt
        scan_id: Optional scan ID (auto-generated if None)
        plan: Optional override for stub plan (defaults to STUB_PLAN)

    Returns:
        OrchestratorResult with decisions, plan_steps_total, plan_steps_completed.
    """
    if scan_id is None:
        scan_id = generate_scan_id()

    orchestrator = PlanExecuteOrchestrator(
        scan_id=scan_id,
        target=target,
        user_prompt=user_prompt,
        plan=plan,
    )
    return await orchestrator.run()