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

import asyncio
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

# Context-window budget for the per-agent ReAct loop. When the running message
# list exceeds MAX_MESSAGES we drop the middle, keeping the system/user head and
# the most recent tail (see _cap_message_history).
MAX_MESSAGES = 50
KEEP_HEAD = 4
KEEP_TAIL = 30


def _cap_message_history(
    messages: list[dict[str, Any]],
    keep_head: int = KEEP_HEAD,
    keep_tail: int = KEEP_TAIL,
) -> list[dict[str, Any]]:
    """Trim the middle of a ReAct history without orphaning tool calls.

    OpenAI-compatible APIs (OpenAI, DeepSeek, GLM, ...) reject a request unless
    every assistant message that carries ``tool_calls`` is immediately followed
    by exactly one ``role: "tool"`` message per ``tool_call_id``.  A naive
    ``messages[:keep_head] + messages[-keep_tail:]`` slice can cut between an
    assistant tool-call message and its tool replies, which aborts the agent
    mid-scan with::

        HTTP 400: An assistant message with 'tool_calls' must be followed by
        tool messages responding to each 'tool_call_id' ...

    This helper drops whole tool-call exchanges: an assistant ``tool_calls``
    message is kept only when ALL of its replies survived the slice, and any
    orphaned ``tool`` message (whose assistant was sliced away) is discarded.

    Args:
        messages: Full running history (system, user, assistant, tool, ...).
        keep_head: Number of leading messages to preserve.
        keep_tail: Number of trailing messages to preserve.

    Returns:
        A history that is safe to send to an OpenAI-compatible endpoint.
    """
    if len(messages) <= keep_head + keep_tail:
        return messages

    sliced = messages[:keep_head] + messages[-keep_tail:]

    out: list[dict[str, Any]] = []
    i = 0
    n = len(sliced)
    while i < n:
        msg = sliced[i]
        role = msg.get("role")

        if role == "assistant" and msg.get("tool_calls"):
            expected = [tc.get("id") for tc in msg["tool_calls"]]
            j = i + 1
            replies: list[dict[str, Any]] = []
            while j < n and sliced[j].get("role") == "tool":
                replies.append(sliced[j])
                j += 1
            replied = [r.get("tool_call_id") for r in replies]
            if replied == expected:
                out.append(msg)
                out.extend(replies)
            else:
                logger.warning(
                    "Context cap dropped an incomplete tool-call exchange "
                    "(expected %s, kept %s)", expected, replied,
                )
            i = j
            continue

        if role == "tool":
            # Orphaned reply — its assistant tool_calls message was sliced away.
            i += 1
            continue

        out.append(msg)
        i += 1

    return out


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
        """Emit SSE scan_progress event for this decision.

        Phase D: also computes a progress % from the decision turn so the
        frontend progress bar advances through each ReAct iteration.
        The mapping is intentionally coarse — specialist agents don't
        own the global progress bar (the pipeline's phase_change events
        do), so this just nudges within the current phase window.
        """
        # Coarse per-iteration nudge — stays within the current phase's
        # window because the pipeline emits phase_change events that
        # set the progress bar to the phase boundary. This just shows
        # the agent is making incremental progress within the phase.
        progress = min(95, 20 + decision.turn * 3)
        await emit_scan_progress(
            scan_id=self.scan_id,
            turn=decision.turn,
            thought=decision.thought,
            tool_name=decision.tool_name,
            observation=decision.observation,
            agent_name=self.AGENT_NAME,
            progress=progress,
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
                a. Call LLM via chat_completion(messages, tools)
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
        from app.pentest.events import (
            emit_tool_call_started, emit_tool_call_completed,
            emit_assistant_message, emit_thinking, emit_iteration,
        )
        from app.pentest.scan_registry import scan_registry
        from app.pentest.events import emit_scan_progress, emit_scan_error

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

        # ── Phase D: consecutive-failure detector ────────────────────────
        # If the LLM keeps trying tools that fail (binary not found, scope
        # violation, etc.), we don't want to burn 30 iterations × API tokens
        # before giving up. Track consecutive failures; if we exceed
        # MAX_CONSECUTIVE_FAILURES, abort with a clear message telling the
        # user to install missing tools.
        MAX_CONSECUTIVE_FAILURES = 5
        consecutive_failures = 0
        failed_tools_seen: set[str] = set()

        # ---------- ReAct loop ----------
        for iteration in range(self.max_iterations):
            turn = len(self.decisions)

            # ── Phase D: abort check ──────────────────────────────────
            # If the user clicked the panic button (POST /api/scans/{id}/abort),
            # scan_registry.abort_scan() set state.abort_event. We need to
            # check this EACH iteration BEFORE the next LLM call — otherwise
            # the abort button has no effect until max_iterations is reached,
            # burning more LLM tokens ($$).
            #
            # PHASE D-6: added verbose logging so we can verify in journalctl
            # that the check is actually firing (vs is_aborted() returning
            # False for some weird reason like scan_id mismatch).
            abort_state = scan_registry.get_state(self.scan_id)
            abort_is_set = abort_state.abort_event.is_set() if abort_state else None
            logger.info(
                "ABORT CHECK | agent=%s | scan=%s | iter=%d | scan_state_exists=%s | abort_event_is_set=%s",
                self.AGENT_NAME, self.scan_id, iteration + 1,
                abort_state is not None, abort_is_set,
            )
            if abort_is_set:
                logger.warning(
                    "Agent %s aborting — scan_registry.abort_event is set | scan=%s | iter=%d",
                    self.AGENT_NAME, self.scan_id, iteration + 1,
                )
                abort_msg = (
                    f"⛔ Agent {self.AGENT_NAME} đã bị dừng (panic button). "
                    f"Không tốn thêm tokens."
                )
                await emit_assistant_message(
                    scan_id=self.scan_id, content=abort_msg,
                    agent_name=self.AGENT_NAME, iteration=iteration + 1,
                )
                await emit_scan_progress(
                    scan_id=self.scan_id, thought=abort_msg,
                    agent_name=self.AGENT_NAME, progress=99,
                )
                return self._finalize("aborted", error="user_panic_button")

            # Emit iteration boundary — frontend renders as a timeline divider
            await emit_iteration(
                scan_id=self.scan_id,
                iteration=iteration + 1,
                scope="sub",
                agent_name=self.AGENT_NAME,
                thought=f"Iteration {iteration + 1}/{self.max_iterations}",
            )

            # Emit a brief "thinking" status so the user sees the system is
            # doing something during the (often 5-30s) LLM call.
            await emit_thinking(
                scan_id=self.scan_id,
                text=f"{self.AGENT_NAME} đang suy nghĩ... (vòng {iteration + 1}/{self.max_iterations})",
                agent_name=self.AGENT_NAME,
                iteration=iteration + 1,
            )

            # ---------- P5 minimal: context budget cap ----------
            # If messages grow beyond MAX_MESSAGES (each tool call adds 2:
            # assistant + tool result), drop the middle to prevent unbounded
            # context growth:
            #   - first KEEP_HEAD messages (system + initial user + early pairs)
            #   - last KEEP_TAIL messages (most recent context)
            # NOTE: uses _cap_message_history, which keeps tool-call/tool-result
            # pairs intact — a naive slice produced HTTP 400 "assistant message
            # with 'tool_calls' must be followed by tool messages" and aborted
            # the agent mid-scan.
            # This is a HARD cap — true CyberStrikeAI parity would use a
            # summarize middleware (LLM call to compress older context), but
            # that's expensive (extra LLM call per scan). Defer to P5-real.
            if len(messages) > MAX_MESSAGES:
                capped = _cap_message_history(messages, KEEP_HEAD, KEEP_TAIL)
                logger.info(
                    "Agent %s context cap: %d messages → %d (dropped %d middle)",
                    self.AGENT_NAME, len(messages), len(capped),
                    len(messages) - len(capped),
                )
                messages = capped

            try:
                # ── Phase D: race LLM call against abort_event ────────
                # Even though we already checked is_aborted() at the start
                # of this iteration, the user might press the panic button
                # DURING the (often 3-30s) LLM call. Without this race,
                # the LLM call would complete + the agent would execute
                # the returned tool_calls + record a decision + THEN check
                # abort at the next iteration — burning $0.001-0.003 per
                # extra LLM call.
                #
                # Solution: spin up an asyncio task waiting on the
                # abort_event alongside the LLM call. If the abort_event
                # fires first, we cancel the LLM task and exit immediately.
                scan_state = scan_registry.get_state(self.scan_id)
                abort_event = scan_state.abort_event if scan_state else None

                llm_task = asyncio.create_task(
                    chat_completion(
                        llm_config=llm_config,
                        messages=messages,
                        tools=tool_schemas,
                    )
                )

                if abort_event is not None:
                    abort_task = asyncio.create_task(abort_event.wait())
                    done, pending = await asyncio.wait(
                        {llm_task, abort_task},
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                else:
                    # No scan_state (e.g. running outside scan_pipeline) —
                    # just await the LLM call as before
                    done = {await llm_task}
                    pending = set()
                    abort_task = None

                # If abort fired during LLM call → cancel LLM + exit
                if abort_task is not None and abort_task in done:
                    logger.warning(
                        "Agent %s aborting DURING LLM call (panic button) | scan=%s | iter=%d",
                        self.AGENT_NAME, self.scan_id, iteration + 1,
                    )
                    llm_task.cancel()
                    if abort_task is not None:
                        abort_task.cancel()
                    # Try to recover the LLM result if it finished before cancel
                    try:
                        response = await llm_task
                    except (asyncio.CancelledError, Exception):
                        response = None
                    abort_msg = (
                        f"⛔ Agent {self.AGENT_NAME} đã bị dừng (panic button). "
                        f"Không tốn thêm tokens."
                    )
                    await emit_assistant_message(
                        scan_id=self.scan_id, content=abort_msg,
                        agent_name=self.AGENT_NAME, iteration=iteration + 1,
                    )
                    return self._finalize("aborted", error="user_panic_button")

                # LLM call finished first → cancel the abort watcher
                if abort_task is not None:
                    abort_task.cancel()
                    try:
                        await abort_task
                    except asyncio.CancelledError:
                        pass

                # Get the LLM response
                response = llm_task.result()

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
                    # Emit the assistant's final text as an assistant_message event
                    # so the frontend can render it as a chat bubble.
                    if thought:
                        await emit_assistant_message(
                            scan_id=self.scan_id,
                            content=thought,
                            agent_name=self.AGENT_NAME,
                            iteration=iteration + 1,
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
                total_tcs = len(response["tool_calls"])
                for tc_idx, tc in enumerate(response["tool_calls"]):
                    tool_name = tc["name"]
                    tool_call_id = tc.get("id") or f"tc_{tool_name}_{tc_idx}"
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
                            # Emit started + immediately failed tool_call event
                            await emit_tool_call_started(
                                scan_id=self.scan_id, tool_name=tool_name,
                                tool_call_id=tool_call_id, arguments=tool_args,
                                index=tc_idx + 1, total=total_tcs,
                                agent_name=self.AGENT_NAME, iteration=iteration + 1,
                            )
                            await emit_tool_call_completed(
                                scan_id=self.scan_id, tool_name=tool_name,
                                tool_call_id=tool_call_id, success=False,
                                result_preview=tool_output, error="allowlist_violation",
                                agent_name=self.AGENT_NAME, iteration=iteration + 1,
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

                    # ── Emit tool_call_started BEFORE execution ──────────
                    # (skipped for the synthetic `exit` tool — that's a
                    # control-flow signal, not a real tool)
                    if tool_name != "exit":
                        await emit_tool_call_started(
                            scan_id=self.scan_id, tool_name=tool_name,
                            tool_call_id=tool_call_id, arguments=tool_args,
                            index=tc_idx + 1, total=total_tcs,
                            agent_name=self.AGENT_NAME, iteration=iteration + 1,
                        )

                    # ── Execute ──────────────────────────────────────────
                    tool_error: str | None = None
                    try:
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
                    except Exception as exec_exc:
                        tool_output = f"Error executing {tool_name}: {exec_exc}"
                        tool_error = str(exec_exc)
                        logger.exception(
                            "Agent %s tool %s raised | iter=%d",
                            self.AGENT_NAME, tool_name, iteration + 1,
                        )

                    # ── Emit tool_call_completed AFTER execution ─────────
                    if tool_name != "exit":
                        # Success heuristic: tool_output starts with "Error"
                        # or contains "[error]" → treat as failure for UI.
                        success = (
                            tool_error is None
                            and not tool_output.startswith("Error")
                            and "[error]" not in tool_output.lower()[:200]
                            and "Binary not found" not in tool_output
                            and "scope violation" not in tool_output.lower()[:200]
                        )
                        preview = tool_output[:200].replace("\n", " ").strip()
                        await emit_tool_call_completed(
                            scan_id=self.scan_id, tool_name=tool_name,
                            tool_call_id=tool_call_id, success=success,
                            result_preview=preview, error=tool_error,
                            agent_name=self.AGENT_NAME, iteration=iteration + 1,
                        )

                        # ── Phase D: consecutive-failure detection ──────
                        if success:
                            consecutive_failures = 0
                            failed_tools_seen.discard(tool_name)
                        else:
                            consecutive_failures += 1
                            failed_tools_seen.add(tool_name)
                            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                                logger.error(
                                    "Agent %s aborting: %d consecutive tool failures | scan=%s | failed_tools=%s",
                                    self.AGENT_NAME, consecutive_failures,
                                    self.scan_id, sorted(failed_tools_seen),
                                )
                                # Emit a clear error + assistant_message so the
                                # UI surfaces the abort reason to the user.
                                abort_msg = (
                                    f"⚠️ Agent {self.AGENT_NAME} dừng sau "
                                    f"{consecutive_failures} lần tool fail liên tiếp. "
                                    f"Các tool thất bại: {sorted(failed_tools_seen)}. "
                                    f"Có thể binary chưa cài — chạy "
                                    f"`bash scripts/install_tools.sh` rồi retry."
                                )
                                await emit_assistant_message(
                                    scan_id=self.scan_id, content=abort_msg,
                                    agent_name=self.AGENT_NAME, iteration=iteration + 1,
                                )
                                return self._finalize("failed", error=abort_msg)

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