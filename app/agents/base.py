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

    async def run(
        self,
        llm_config: dict[str, Any] | None = None,
        executor: Any = None,
    ) -> AgentRunResult:
        """Execute the agent's task. Returns final result.

        P3: now dispatches to _run_react_loop() — the real LLM + tool
        execution loop. Falls back to the W10 stub _decide_next() only
        when llm_config is None (backward compat with tests).

        Args:
            llm_config: User's LLM config from DB (provider, api_key, model, ...).
                Required for real LLM-driven execution. If None, falls back
                to the W10 stub behavior.
            executor: SubprocessExecutor instance configured for the scan's
                scope guard. Required for real tool execution. If None, the
                agent loop will skip tool calls and just record the LLM's
                thinking.

        Returns:
            AgentRunResult with decisions, findings, tokens, duration.
        """
        # Check guardrails before starting
        violation = self._check_guardrails()
        if violation:
            return self._finalize("failed", error=violation)

        # P3 dispatch
        if llm_config is not None:
            return await self._run_react_loop(llm_config, executor)

        # W10 fallback (for tests that haven't been updated yet)
        logger.warning(
            "Agent %s running in W10-stub mode (no llm_config). "
            "Pass llm_config to enable real LLM execution.",
            self.AGENT_NAME,
        )
        return await self._run_w10_stub()

    async def _run_react_loop(
        self,
        llm_config: dict[str, Any],
        executor: Any,
    ) -> AgentRunResult:
        """P3: real ReAct loop using LLM + filtered tool schemas.

        Mirrors CyberStrikeAI's `runEinoADKAgentLoop`:
            1. Build messages with system_prompt + task_description
            2. Build tool schemas filtered by self.tool_allowlist
            3. Loop (up to max_iterations):
                a. Call LLM via litellm.chat_completion(messages, tools)
                b. If LLM returns tool_calls:
                    - For each tool_call:
                        - Validate against tool_allowlist (defense in depth)
                        - If self.is_destructive:
                            → use executor.execute_with_hitl(...) (HITL gate)
                          else:
                            → use execute_tool_call(...) via ExecutionService
                        - Append tool_result to messages
                    - Continue loop
                c. If LLM returns content only (no tool_calls):
                    - LLM is done thinking — append assistant message
                    - Break (assume LLM calls implicit "exit" by returning text)
            4. Finalize with status + decisions + tokens

        Each iteration records an AgentDecision + emits an SSE event.

        Args:
            llm_config: User LLM config (provider, api_key, model, ...)
            executor: SubprocessExecutor with scope guard for this scan

        Returns:
            AgentRunResult with all decisions recorded
        """
        from app.agents.llm_client import chat_completion
        from app.agents.tool_bridge import build_tool_schemas, execute_tool_call

        # ---------- Build initial messages ----------
        user_msg = (
            f"Target: {self.target}\n\n"
            f"Task: {self.task_description or '(no specific task)'}\n\n"
            f"User context: {self.user_prompt or '(no user prompt)'}\n\n"
            f"Begin your task. Use the tools available to you. "
            f"When you have completed your task, return a final summary "
            f"as your last message (without tool calls)."
        )
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_msg},
        ]

        # ---------- Build tool schemas filtered by allowlist ----------
        # tool_allowlist contains binary names (e.g. "nmap", "httpx").
        # build_tool_schemas takes tool_names filter matching YAML tool names.
        allowed = list(self.tool_allowlist) if self.tool_allowlist else None
        tool_schemas = build_tool_schemas(tool_names=allowed)

        # Add `record_finding` + `exit` tools to every agent so they can
        # persist findings + signal completion (the `exit` tool is interpreted
        # by the agent loop as "stop and return").
        # build_tool_schemas already adds both of these automatically.

        logger.info(
            "Agent %s starting ReAct loop | scan=%s | tools=%d | max_iter=%d | destructive=%s",
            self.AGENT_NAME, self.scan_id,
            len(self.tool_allowlist), self.max_iterations, self.is_destructive,
        )

        # ---------- ReAct loop ----------
        for iteration in range(self.max_iterations):
            turn = len(self.decisions)

            # ---------- P5 minimal: context budget cap ----------
            # If messages list grows beyond 50 entries (each tool call adds
            # 2 messages: assistant + tool result), drop the middle ones to
            # prevent unbounded context growth. Keep:
            #   - first 4 messages (system + initial user + first 2 LLM/tool pairs)
            #   - last 30 messages (most recent context for decision-making)
            # This is a HARD cap — true CyberStrikeAI parity would use a
            # summarize middleware (LLM call to compress older context), but
            # that's expensive (extra LLM call per scan). Defer to P5-real.
            MAX_MESSAGES = 50
            KEEP_HEAD = 4
            KEEP_TAIL = 30
            if len(messages) > MAX_MESSAGES:
                dropped = len(messages) - KEEP_HEAD - KEEP_TAIL
                logger.info(
                    "Agent %s context cap: %d messages → %d (dropped %d middle)",
                    self.AGENT_NAME, len(messages), KEEP_HEAD + KEEP_TAIL, dropped,
                )
                messages = messages[:KEEP_HEAD] + messages[-KEEP_TAIL:]

            try:
                # Call LLM
                response = await chat_completion(
                    llm_config=llm_config,
                    messages=messages,
                    tools=tool_schemas,
                )

                self.total_tokens += response["usage"].get("total_tokens", 0)

                # Append assistant message to history
                assistant_msg: dict[str, Any] = {
                    "role": "assistant",
                    "content": response["content"] or "",
                }
                if response["tool_calls"]:
                    assistant_msg["tool_calls"] = [
                        {
                            "id": tc["id"],
                            "type": "function",
                            "function": {
                                "name": tc["name"],
                                "arguments": tc["arguments"],
                            },
                        }
                        for tc in response["tool_calls"]
                    ]
                messages.append(assistant_msg)

                # If LLM returned no tool_calls, treat as implicit exit
                if not response["tool_calls"]:
                    thought = (response["content"] or "")[:500]
                    logger.info(
                        "Agent %s implicit exit at iter %d | thought=%s",
                        self.AGENT_NAME, iteration + 1, thought[:80],
                    )
                    decision = AgentDecision(
                        turn=turn,
                        agent_name=self.AGENT_NAME,
                        thought=thought,
                        tool_name=None,
                        tool_args={},
                        observation="(implicit exit — LLM returned text without tool_calls)",
                        tokens_used=response["usage"].get("total_tokens", 0),
                    )
                    self._record_decision(decision)
                    await self._emit_progress_event(decision)
                    break

                # ---------- Execute each tool call ----------
                for tc in response["tool_calls"]:
                    tool_name = tc["name"]
                    try:
                        import json as _json
                        tool_args = _json.loads(tc["arguments"]) if tc["arguments"] else {}
                    except Exception:
                        tool_args = {}

                    # Defense-in-depth: re-validate against allowlist
                    # (the LLM might hallucinate a tool name not in our schema)
                    if tool_name not in ("record_vulnerability", "exit"):
                        if tool_name not in self.tool_allowlist:
                            logger.warning(
                                "TOOL_ALLOWLIST_VIOLATION | agent=%s | tool=%s | allowed=%s",
                                self.AGENT_NAME, tool_name, list(self.tool_allowlist),
                            )
                            tool_output = (
                                f"TOOL_ALLOWLIST_VIOLATION: tool {tool_name!r} is not "
                                f"allowed for agent {self.AGENT_NAME!r}. "
                                f"Allowed: {list(self.tool_allowlist)}"
                            )
                            decision = AgentDecision(
                                turn=turn, agent_name=self.AGENT_NAME,
                                thought=f"Tried to call {tool_name} (not allowed)",
                                tool_name=tool_name, tool_args=tool_args,
                                observation=tool_output,
                                tokens_used=response["usage"].get("total_tokens", 0),
                            )
                            self._record_decision(decision)
                            await self._emit_progress_event(decision)
                            messages.append({
                                "role": "tool", "tool_call_id": tc["id"],
                                "content": tool_output,
                            })
                            continue

                    # ---------- P3.2: HITL gate for destructive agents ----------
                    if self.is_destructive and tool_name not in ("record_vulnerability", "exit"):
                        # Destructive tool — route through execute_with_hitl
                        # so the HITL gate (audit_agent mode by default)
                        # reviews the tool call before execution.
                        tool_output = await self._execute_destructive_with_hitl(
                            tool_name=tool_name,
                            tool_args=tool_args,
                            executor=executor,
                            reasoning=f"Agent {self.AGENT_NAME} requested {tool_name}",
                        )
                    elif tool_name == "exit":
                        # Exit tool — break out of loop
                        final_summary = tool_args.get("summary", "Agent task complete.")
                        logger.info(
                            "Agent %s explicit exit | iter=%d | summary=%s",
                            self.AGENT_NAME, iteration + 1, final_summary[:100],
                        )
                        decision = AgentDecision(
                            turn=turn, agent_name=self.AGENT_NAME,
                            thought=f"Agent called exit: {final_summary[:200]}",
                            tool_name="exit", tool_args=tool_args,
                            observation=final_summary,
                            tokens_used=response["usage"].get("total_tokens", 0),
                        )
                        self._record_decision(decision)
                        await self._emit_progress_event(decision)
                        return self._finalize("completed")

                    elif tool_name == "record_vulnerability":
                        # Bypass executor — direct DB insert via tool_bridge
                        tool_output = await execute_tool_call(
                            tool_name=tool_name,
                            tool_args=tool_args,
                            target=self.target,
                            scan_id=self.scan_id,
                            executor=executor,
                        )
                        # Record as a finding for this agent's result
                        self.findings.append({
                            "title": tool_args.get("title"),
                            "severity": tool_args.get("severity"),
                            "vuln_type": tool_args.get("vuln_type"),
                            "target": tool_args.get("target", self.target),
                        })
                    else:
                        # Normal tool — execute via ExecutionService (P2 path)
                        tool_output = await execute_tool_call(
                            tool_name=tool_name,
                            tool_args=tool_args,
                            target=self.target,
                            scan_id=self.scan_id,
                            executor=executor,
                        )

                    # Record decision + emit SSE
                    thought_preview = (response["content"] or "")[:200]
                    decision = AgentDecision(
                        turn=turn,
                        agent_name=self.AGENT_NAME,
                        thought=thought_preview or f"Called {tool_name}",
                        tool_name=tool_name,
                        tool_args=tool_args,
                        observation=tool_output[:500],
                        tokens_used=response["usage"].get("total_tokens", 0),
                    )
                    self._record_decision(decision)
                    await self._emit_progress_event(decision)

                    # Append tool result to messages for next LLM round
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": tool_output,
                    })

            except Exception as exc:
                logger.exception(
                    "Agent %s ReAct loop error at iter %d: %s",
                    self.AGENT_NAME, iteration + 1, exc,
                )
                return self._finalize("failed", error=str(exc))

        # Reached max_iterations without explicit exit
        if len(self.decisions) >= self.max_iterations:
            logger.warning(
                "Agent %s hit max_iterations=%d",
                self.AGENT_NAME, self.max_iterations,
            )
            return self._finalize("max_iterations")

        return self._finalize("completed")

    async def _execute_destructive_with_hitl(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        executor: Any,
        reasoning: str,
    ) -> str:
        """P3.2: route destructive tool calls through execute_with_hitl.

        Mirrors CyberStrikeAI's hitl_middleware pattern. For agents with
        safety_class="destructive" (penetration, privilege-escalation,
        lateral-movement, persistence-maintenance, impact-exfiltration),
        every tool call is routed through:
            1. HITLManager.request_and_wait() — asks AuditAgent (LLM) or
               user to approve/reject
            2. If approved: SubprocessExecutor.execute() via ExecutionService
            3. If rejected: returns ToolResult.error (no subprocess runs)
            4. If suggest_alternative: rebuild command from suggested_args

        Args:
            tool_name, tool_args: the destructive tool call
            executor: SubprocessExecutor (must have execute_with_hitl method)
            reasoning: agent's reasoning for the call (sent to HITL reviewer)

        Returns:
            String observation for the LLM (tool output or rejection reason)
        """
        # Build the actual command (same as execute_tool_call does)
        from app.tools.loader import load_all_tools
        from app.mcp.execution_service import get_execution_service, ExecutionStatus

        all_tools = load_all_tools()
        tool_def = all_tools.get(tool_name)
        if tool_def is None:
            return f"Error: destructive tool {tool_name!r} not found in tool registry"

        try:
            args = tool_def.build_command_args(**tool_args)
        except Exception as exc:
            return f"Error building args for {tool_name}: {exc}"

        cmd = [tool_def.command] + args

        # Submit via ExecutionService with HITL-wrapped run closure
        svc = get_execution_service()

        async def run_with_hitl(cancel_event) -> dict[str, Any]:
            result = await executor.execute_with_hitl(
                command=cmd,
                target=self.target,
                tool_name=tool_name,
                tool_args=tool_args,
                scan_id=self.scan_id,
                agent_reasoning=reasoning,
                predicted_impact=tool_def.safety_class,  # destructive
                timeout=tool_def.timeout,
                allowed_exit_codes=tool_def.allowed_exit_codes,
            )
            return result.to_dict()

        execution = await svc.submit(
            tool_name=tool_name,
            arguments=tool_args,
            target=self.target,
            run=run_with_hitl,
            scan_id=self.scan_id,
            actor_id=self.AGENT_NAME,
            hard_timeout=tool_def.timeout,
        )

        # Format result for LLM (same shape as execute_tool_call returns)
        if execution.status == ExecutionStatus.COMPLETED and execution.result:
            r = execution.result
            parts: list[str] = []
            if r.get("scope_violation"):
                parts.append("SCOPE VIOLATION: target not in declared scope. Tool blocked.")
            if r.get("stdout"):
                parts.append(r["stdout"])
            if r.get("stderr") and r.get("exit_code") not in (0, None):
                parts.append(f"[stderr] {r['stderr'][:500]}")
            if r.get("error"):
                parts.append(f"[error] {r['error']}")
            parts.append(f"[execution_id] {execution.id}")
            parts.append(f"[hitl] approved (safety_class={tool_def.safety_class})")
            return "\n".join(parts)
        else:
            return (
                f"[error] HITL blocked execution of {tool_name}: "
                f"{execution.error or execution.status.value}\n"
                f"[execution_id] {execution.id}\n"
                f"[hitl] status={execution.status.value}"
            )

    async def _run_w10_stub(self) -> AgentRunResult:
        """W10 fallback: 1-decision stub via _decide_next().

        Kept for backward compat with tests that don't pass llm_config.
        P3+ callers should always pass llm_config.
        """
        turn = len(self.decisions)
        thought, tool_name, tool_args, observation = await self._decide_next(turn)

        if tool_name is not None:
            allowed, reason = self._validate_tool_call(tool_name, tool_args)
            if not allowed:
                logger.warning(
                    "Tool allowlist violation: agent=%s tool=%s reason=%s",
                    self.AGENT_NAME, tool_name, reason,
                )
                observation = f"TOOL_ALLOWLIST_VIOLATION: {reason}"
                tool_name = None

        decision = AgentDecision(
            turn=turn,
            agent_name=self.AGENT_NAME,
            thought=thought,
            tool_name=tool_name,
            tool_args=tool_args,
            observation=observation,
            tokens_used=100,
        )
        self._record_decision(decision)
        await self._emit_progress_event(decision)

        return self._finalize("completed")

    async def _decide_next(
        self,
        turn: int,
    ) -> tuple[str, str | None, dict[str, Any], str]:
        """W10-stub decision function. DEPRECATED in P3.

        P3 default behavior: BaseAgent.run() dispatches to _run_react_loop()
        when llm_config is provided. This _decide_next() is only called by
        the W10 fallback path (_run_w10_stub) when llm_config=None.

        Subclasses that previously overrode this with canned responses can
        keep their overrides for backward compat with the W10 path, but
        production code should always pass llm_config.

        Returns:
            (thought, tool_name_or_None, tool_args, observation)
        """
        raise NotImplementedError(
            f"{self.__class__.__name__}._decide_next() not implemented. "
            f"Subclasses must override this method (or always pass llm_config "
            f"to run() to use the P3 real ReAct loop)."
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