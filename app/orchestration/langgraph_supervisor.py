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
    emit_phase_change,
    emit_scan_started,
)
from app.pentest.scan_registry import scan_registry

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
# P3: now defaults to 4-phase web pentest flow for single-target scans.
# Override via transfer_targets param to customize.
DEFAULT_TRANSFER_TARGETS: list[str] = [
    "recon",
    "vulnerability-triage",
    "penetration",
    "reporting-remediation",
]


# ---------- P3: synthetic transfer + exit tools (LLM-callable) ----------

def _build_supervisor_tools(transfer_targets: list[str]) -> list[dict[str, Any]]:
    """Build the 2 synthetic tools the supervisor LLM can call.

    The supervisor doesn't run real pentest tools — it just decides which
    specialist to transfer to next, or to exit. The tools:

        transfer(target_agent: str, task_description: str)
            → transfer control to the named specialist with a task description
        exit(summary: str)
            → end the scan with a final summary

    Mirrors CyberStrikeAI's adk.ExitTool + supervisor's transfer mechanism.
    """
    return [
        {
            "type": "function",
            "function": {
                "name": "transfer",
                "description": (
                    "Transfer control to a specialist sub-agent. The sub-agent "
                    "will execute its task (using its own tool allowlist) and "
                    "return a result. After the sub-agent returns, you (the "
                    "supervisor) will be invoked again to decide the next step."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "target_agent": {
                            "type": "string",
                            "enum": transfer_targets,
                            "description": "Name of the specialist to transfer to.",
                        },
                        "task_description": {
                            "type": "string",
                            "description": (
                                "What the specialist should do. Be specific — "
                                "include the target, the goal, and any constraints "
                                "(e.g. 'Run nmap top-1000 + httpx on https://example.com')."
                            ),
                        },
                    },
                    "required": ["target_agent", "task_description"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "exit",
                "description": (
                    "End the scan with a final summary. Call this when all "
                    "transfer targets have run and you have aggregated their "
                    "results into a coherent summary for the user."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "summary": {
                            "type": "string",
                            "description": "Final scan summary — what was found, severity count, recommendations.",
                        },
                    },
                    "required": ["summary"],
                },
            },
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
        llm_config: dict[str, Any] | None = None,
        executor: Any = None,
    ):
        super().__init__(scan_id, target, user_prompt, max_decisions, max_sub_agents)
        # W9 backward compat: still accept `experts` dict override
        # W10 default: derive from agent_registry
        self.experts = experts if experts is not None else EXPERT_AGENTS
        # P3: configurable transfer sequence (defaults to 4-phase web pentest)
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

        # P3: LLM config + executor for real ReAct loops (supervisor + experts)
        # If None, falls back to W10 stub behavior (backward compat with tests)
        self.llm_config = llm_config
        self.executor = executor

        # P3: build synthetic supervisor tools (transfer + exit)
        self.supervisor_tools = _build_supervisor_tools(self.transfer_targets)

        # P3: supervisor's chat history (kept across nodes — supervisor is
        # called multiple times during a scan, once after each expert returns)
        self._supervisor_messages: list[dict[str, Any]] = []

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

        P3: replaces W10 deterministic stub with real LLM call.
        - Builds messages: [system, user_prompt, ...prior_decisions_summary]
        - Calls chat_completion(tools=[transfer, exit])
        - Parses tool_call → next_agent + task_description
        - If LLM calls exit → set next_agent=None (route to END)

        Falls back to W10 stub when self.llm_config is None (for tests).
        """
        # ── Phase D: abort check ──────────────────────────────────
        # Honor the panic button BEFORE the next LLM call — otherwise
        # the supervisor keeps transferring to experts and burning
        # tokens after the user clicked Stop.
        if await scan_registry.is_aborted(self.scan_id):
            logger.warning(
                "Supervisor aborting — scan_registry.abort_event is set | scan=%s",
                self.scan_id,
            )
            return {**state, "next_agent": None, "status": "aborted",
                    "error": "user_panic_button"}

        violation = self._check_guardrails()
        if violation:
            return {**state, "next_agent": None, "status": "failed", "error": violation}

        # ---------- P3 dispatch ----------
        if self.llm_config is not None:
            return await self._supervisor_node_llm(state)
        # W10 stub fallback
        return await self._supervisor_node_stub(state)

    async def _supervisor_node_llm(self, state: SharedState) -> SharedState:
        """P3: real LLM-driven supervisor decision."""
        from app.agents.llm_client import chat_completion
        import json as _json

        # Build user message (only on first supervisor call — subsequent
        # calls append the prior expert's result to the running history)
        if not self._supervisor_messages:
            user_msg = (
                f"Target: {self.target}\n\n"
                f"User request: {self.user_prompt}\n\n"
                f"You are the supervisor. Available specialists:\n"
                + "\n".join(
                    f"  - {name}: {agent_registry.get_agent(name).description}"
                    for name in self.transfer_targets
                    if agent_registry.get_agent(name)
                )
                + "\n\nDecide which specialist to transfer to first. Provide a "
                "specific task description including the target."
            )
            self._supervisor_messages.append({"role": "user", "content": user_msg})
        # else: prior expert results already appended in expert_node

        # Call LLM with synthetic transfer + exit tools.
        # ── Phase D: race the LLM call against the abort_event so a
        # mid-LLM-call panic button stops the supervisor immediately
        # (instead of completing the LLM call, transferring to an
        # expert, and burning more tokens).
        try:
            import asyncio as _asyncio
            scan_state = scan_registry.get_state(self.scan_id)
            abort_event = scan_state.abort_event if scan_state else None

            llm_task = _asyncio.create_task(
                chat_completion(
                    llm_config=self.llm_config,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        *self._supervisor_messages,
                    ],
                    tools=self.supervisor_tools,
                    temperature=0.2,
                )
            )

            if abort_event is not None:
                abort_task = _asyncio.create_task(abort_event.wait())
                done, _pending = await _asyncio.wait(
                    {llm_task, abort_task},
                    return_when=_asyncio.FIRST_COMPLETED,
                )
                if abort_task in done:
                    # User pressed abort DURING the supervisor LLM call
                    logger.warning(
                        "Supervisor aborting DURING LLM call (panic button) | scan=%s",
                        self.scan_id,
                    )
                    llm_task.cancel()
                    abort_task.cancel()
                    return {**state, "next_agent": None, "status": "aborted",
                            "error": "user_panic_button"}
                # LLM finished first → cancel the abort watcher
                abort_task.cancel()
                try:
                    await abort_task
                except _asyncio.CancelledError:
                    pass

            response = await llm_task
        except Exception as exc:
            logger.exception("Supervisor LLM call failed: scan=%s", self.scan_id)
            return {**state, "next_agent": None, "status": "failed",
                    "error": f"Supervisor LLM call failed: {exc}"}

        self.total_tokens += response["usage"].get("total_tokens", 0)

        # Append assistant message to supervisor's running history
        asst_msg: dict[str, Any] = {"role": "assistant", "content": response["content"] or ""}
        if response["tool_calls"]:
            asst_msg["tool_calls"] = [
                {"id": tc["id"], "type": "function",
                 "function": {"name": tc["name"], "arguments": tc["arguments"]}}
                for tc in response["tool_calls"]
            ]
        self._supervisor_messages.append(asst_msg)

        # Parse the (single) tool call to determine next_agent + task
        tool_calls = response["tool_calls"] or []
        next_agent: str | None = None
        task_description: str = ""
        summary: str = ""
        thought: str = (response["content"] or "")[:500]

        if tool_calls:
            tc = tool_calls[0]  # supervisor makes 1 decision at a time
            try:
                args = _json.loads(tc["arguments"]) if tc["arguments"] else {}
            except Exception:
                args = {}

            if tc["name"] == "transfer":
                next_agent = args.get("target_agent")
                task_description = args.get("task_description", "")
                # Validate next_agent is in transfer_targets
                if next_agent not in self.transfer_targets:
                    logger.warning(
                        "Supervisor tried to transfer to %r (not in transfer_targets=%s). Ending scan.",
                        next_agent, self.transfer_targets,
                    )
                    next_agent = None
                else:
                    # Record the task description so expert_node can use it
                    state["next_task_description"] = task_description
                    thought = f"Transferring to {next_agent}: {task_description[:200]}"

            elif tc["name"] == "exit":
                summary = args.get("summary", "Scan complete.")
                next_agent = None
                thought = f"Exit: {summary[:200]}"
                state["final_summary"] = summary
        else:
            # LLM returned text without tool_calls — treat as implicit exit
            next_agent = None
            summary = thought or "Supervisor ended without explicit exit."
            state["final_summary"] = summary

        # OpenAI-compatible APIs (OpenAI, DeepSeek, GLM, ...) reject the NEXT
        # request unless every assistant message carrying `tool_calls` is
        # immediately followed by one `role: "tool"` message per
        # `tool_call_id`:
        #   HTTP 400 "An assistant message with 'tool_calls' must be followed
        #   by tool messages responding to each 'tool_call_id'."
        # The supervisor's `transfer`/`exit` tools are synthetic routing tools
        # with no real executor, so we append a short acknowledgement result
        # for each call to keep the running history valid across turns.
        for tc in tool_calls:
            try:
                tc_args = _json.loads(tc["arguments"]) if tc.get("arguments") else {}
            except Exception:
                tc_args = {}
            tc_name = tc.get("name", "")
            if tc_name == "transfer":
                ack = {"status": "ok", "action": "transfer",
                       "target_agent": tc_args.get("target_agent")}
            elif tc_name == "exit":
                ack = {"status": "ok", "action": "exit"}
            else:
                ack = {"status": "ok", "action": tc_name}
            self._supervisor_messages.append({
                "role": "tool",
                "tool_call_id": tc.get("id", ""),
                "content": _json.dumps(ack),
            })

        # Record decision
        turn = len(self.decisions)
        decision = AgentDecision(
            turn=turn,
            thought=thought,
            agent_name="supervisor",
            tool_name="transfer" if next_agent else "exit",
            tool_args={"target_agent": next_agent, "task_description": task_description}
            if next_agent else {"summary": summary},
            observation=(
                f"Transferring to {next_agent} with task: {task_description[:200]}"
                if next_agent else f"Exit: {summary[:200]}"
            ),
            tokens_used=response["usage"].get("total_tokens", 0),
        )
        self.record_decision(decision)
        await emit_scan_progress(
            scan_id=self.scan_id, turn=turn, thought=thought,
            tool_name=decision.tool_name, observation=decision.observation,
            agent_name="supervisor", progress=20 + turn * 2,
        )

        return {
            **state,
            "next_agent": next_agent,
            "current_agent": "supervisor",
            "status": "running" if next_agent else "completed",
            "decisions": self.decisions,
            "total_tokens": self.total_tokens,
        }

    async def _supervisor_node_stub(self, state: SharedState) -> SharedState:
        """W10 stub fallback (when llm_config is None)."""
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
            agent_name="supervisor", progress=20 + turn * 2,
        )

        return {
            **state,
            "next_agent": next_agent,
            "current_agent": "supervisor",
            "status": "running" if next_agent else "completed",
            "decisions": self.decisions,
        }

    def _stub_decide_next(self, supervisor_turns: int) -> tuple[str, str | None]:
        """W10 stub: deterministic sequence based on transfer_targets list.

        P3: kept for backward compat when llm_config is None (tests).
        Production callers should always pass llm_config.
        """
        if supervisor_turns < len(self.transfer_targets):
            next_agent = self.transfer_targets[supervisor_turns]
            return (
                f"Transferring to {next_agent} (transfer target {supervisor_turns + 1}/{len(self.transfer_targets)}).",
                next_agent,
            )
        return ("All transfer targets exhausted. Exiting.", None)

    def _make_expert_node(self, expert_name: str):
        """Factory: create an async node function for a specific expert.

        P3: invokes the real specialist agent via create_agent(name).run().
        Falls back to W10 stub when llm_config is None.
        """
        meta = agent_registry.get_agent(expert_name)
        if meta is None:
            raise ValueError(f"Unknown expert: {expert_name!r}")

        async def expert_node(state: SharedState) -> SharedState:
            target = state.get("target", self.target)
            task_description = state.get("next_task_description") or (
                f"Run {expert_name} on {target}"
            )

            # ---------- P3 dispatch ----------
            if self.llm_config is not None and self.executor is not None:
                # Real sub-agent invocation
                from app.agents.base import create_agent
                logger.info(
                    "Expert node executing (P3 real): scan=%s expert=%s safety=%s tools=%s",
                    self.scan_id, expert_name, meta.safety_class, list(meta.tool_allowlist),
                )
                agent = create_agent(
                    agent_name=expert_name,
                    scan_id=self.scan_id,
                    target=target,
                    task_description=task_description,
                    user_prompt=self.user_prompt,
                )
                result = await agent.run(
                    llm_config=self.llm_config,
                    executor=self.executor,
                )

                # Append the agent's final summary to supervisor's history so
                # the next supervisor decision has context
                if result.decisions:
                    last_obs = result.decisions[-1].observation or "(no observation)"
                else:
                    last_obs = "(agent produced no decisions)"
                self._supervisor_messages.append({
                    "role": "user",
                    "content": (
                        f"[Result from {expert_name} on {target}]\n"
                        f"Task: {task_description}\n"
                        f"Status: {result.status}\n"
                        f"Decisions: {len(result.decisions)}\n"
                        f"Findings: {len(result.findings)}\n"
                        f"Tokens: {result.total_tokens}\n"
                        f"Last observation: {last_obs[:500]}\n"
                        f"Findings list: {result.findings[:5]}"
                    ),
                })

                # Aggregate findings into the orchestrator's findings list
                for f in result.findings:
                    self.findings.append({**f, "source_agent": expert_name})

                # Record a high-level decision for the orchestrator's log
                turn = len(self.decisions)
                decision = AgentDecision(
                    turn=turn,
                    thought=f"Expert {expert_name} executed (P3 real). "
                            f"safety={meta.safety_class} decisions={len(result.decisions)} "
                            f"findings={len(result.findings)}",
                    agent_name=expert_name,
                    tool_name=meta.tool_allowlist[0] if meta.tool_allowlist else None,
                    tool_args={"target": target, "task_description": task_description[:200]},
                    observation=f"{expert_name} returned: status={result.status}, "
                                f"decisions={len(result.decisions)}, findings={len(result.findings)}",
                    tokens_used=result.total_tokens,
                )
                self.record_decision(decision)
                self.agents_involved.add(expert_name)
                await emit_scan_progress(
                    scan_id=self.scan_id, turn=turn, thought=decision.thought,
                    tool_name=decision.tool_name, observation=decision.observation[:500],
                    agent_name=expert_name, progress=30 + turn * 2,
                )

                return {**state, "current_agent": expert_name, "decisions": self.decisions}

            # ---------- W10 stub fallback ----------
            logger.info(
                "Expert node executing (W10 stub): scan=%s expert=%s",
                self.scan_id, expert_name,
            )
            turn = len(self.decisions)
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
                agent_name=expert_name, progress=30 + turn * 2,
            )

            # Append stub result to supervisor history too (so W10 path
            # still feeds the supervisor LLM if llm_config is later added)
            self._supervisor_messages.append({
                "role": "user",
                "content": f"[Stub result from {expert_name} on {target}]\n{observation}",
            })

            return {**state, "current_agent": expert_name, "decisions": self.decisions}

        return expert_node

    def _stub_expert_observe(self, expert_name: str, target: str) -> str:
        """W10 stub: return canned observation per expert. P3 deprecated."""
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
    llm_config: dict[str, Any] | None = None,
    executor: Any = None,
) -> OrchestratorResult:
    """One-shot: run a scan with the Supervisor orchestrator.

    P3: now accepts llm_config + executor for real LLM-driven execution.
    If llm_config is None, falls back to W10 stub behavior (for tests).

    Args:
        target: Target URL or IP
        user_prompt: Natural-language prompt
        scan_id: Optional scan ID (auto-generated if None)
        experts: Optional legacy experts dict (W9 backward compat — prefer transfer_targets)
        transfer_targets: Optional list of agent names to transfer to in sequence.
            Defaults to P3 4-phase web pentest flow:
                ["recon", "vulnerability-triage", "penetration", "reporting-remediation"]
        llm_config: User LLM config from DB (provider, api_key, model, ...).
            Required for real LLM execution. None = W10 stub.
        executor: SubprocessExecutor with scope guard for the scan target.
            Required for real tool execution. None = no tools run.

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
        llm_config=llm_config,
        executor=executor,
    )
    return await orchestrator.run()