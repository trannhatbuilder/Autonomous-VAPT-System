"""
VAPT-AI Single ReAct Agent — W3-A.

A simple ReAct (Reason + Act + Observe) agent that:
    1. Takes a target URL + natural-language user prompt
    2. Uses LLM (via LiteLLM) to decide which MCP tool to call
    3. Calls the tool (currently returns placeholder — W3-D wires actual subprocess)
    4. Observes result + decides next action
    5. Loops until done or max_iterations (D18: max 30 decisions/scan)

W3-A scope: skeleton — agent can invoke 12 MCP tools (2 skeleton + 10 wrappers).
Actual subprocess execution is W3-D task.

Adapted from EVVO's routes/pentest/agent.py (2006 LOC) — extracted core ReAct
loop only. Full decomposition (Planner/Generator/Evaluator split) deferred to W9.

Usage:
    from app.agents.agent import ReActAgent, run_scan

    # One-shot
    result = await run_scan(
        target="http://example.com",
        user_prompt="Scan for SQL injection",
        scan_id="scan_abc123",
    )

Architecture:
    ┌─────────────┐
    │  User Prompt │  "Scan example.com for SQLi"
    └──────┬──────┘
           ▼
    ┌─────────────────────────────────────────┐
    │  ReActAgent                             │
    │  ┌─────────────────────────────────┐   │
    │  │  LLM (LiteLLM — multi-provider) │   │
    │  │  Reason: "I should run nmap..."  │   │
    │  └────────────┬────────────────────┘   │
    │               ▼                         │
    │  ┌─────────────────────────────────┐   │
    │  │  Tool Call (MCP)                │   │
    │  │  - nmap / nuclei / sqlmap / ...  │   │
    │  │  - 12 tools (W1-G + W2-A)       │   │
    │  └────────────┬────────────────────┘   │
    │               ▼                         │
    │  ┌─────────────────────────────────┐   │
    │  │  Observe → loop or done         │   │
    │  └─────────────────────────────────┘   │
    └─────────────────────────────────────────┘
           ▼
    ┌─────────────┐
    │  Result      │  findings[], evidence[], trace[]
    └─────────────┘

D18 guardrails (always enforced):
    - Max 30 decisions per scan
    - Max 2M tokens per scan
    - Max 4 hours per scan
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, UTC
from typing import Any

from app.tools.loader import get_tool, list_tools, ToolDef
from app.pentest.events import (
    emit_scan_started, emit_scan_progress, emit_scan_complete, emit_scan_error,
)
from app.sandbox.scope_guard import ScopeGuard, ScopeRule
from app.sandbox.executor import SubprocessExecutor, ToolResult

logger = logging.getLogger(__name__)


# ---------- Constants (D18 guardrails) ----------

MAX_DECISIONS_PER_SCAN = 30
MAX_TOKENS_PER_SCAN = 2_000_000
MAX_SCAN_DURATION_SECONDS = 4 * 60 * 60  # 4 hours

# ---------- Data classes ----------

@dataclass
class AgentDecision:
    """One decision in the ReAct loop."""
    turn: int
    thought: str           # LLM's reasoning
    tool_name: str | None  # tool to call (None = done)
    tool_args: dict[str, Any]
    observation: str = ""  # tool result
    tokens_used: int = 0
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass
class ScanResult:
    """Final result of a scan."""
    scan_id: str
    target: str
    user_prompt: str
    status: str  # completed / failed / cancelled / max_iterations
    decisions: list[AgentDecision] = field(default_factory=list)
    findings: list[dict[str, Any]] = field(default_factory=list)
    total_tokens: int = 0
    duration_seconds: float = 0.0
    error: str | None = None
    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    completed_at: datetime | None = None


# ---------- ReAct Agent ----------

class ReActAgent:
    """Single ReAct agent — Reason + Act + Observe loop.

    W3-D: actual subprocess execution via SubprocessExecutor.
    LLM integration is stubbed (W3-B/C will add LiteLLM).
    """

    def __init__(
        self,
        scan_id: str,
        target: str,
        user_prompt: str,
        max_decisions: int = MAX_DECISIONS_PER_SCAN,
        scope_guard: ScopeGuard | None = None,
    ):
        self.scan_id = scan_id
        self.target = target
        self.user_prompt = user_prompt
        self.max_decisions = min(max_decisions, MAX_DECISIONS_PER_SCAN)
        self.decisions: list[AgentDecision] = []
        self.total_tokens = 0
        self.start_time = time.time()
        self.findings: list[dict[str, Any]] = []

        # W3-D: scope guard + subprocess executor
        # Auto-create scope guard with target as the only allowed host
        if scope_guard is None:
            scope_guard = ScopeGuard(declared_scope=[
                ScopeRule(host=target),
            ])
        self.scope_guard = scope_guard
        self.executor = SubprocessExecutor(scope_guard=self.scope_guard)

    async def run(self) -> ScanResult:
        """Run the ReAct loop until done or max iterations."""
        logger.info("Agent starting: scan=%s target=%s", self.scan_id, self.target)

        # Emit scan_started event (SSE)
        await emit_scan_started(self.scan_id, self.target, self.user_prompt)

        # ---------- Phase 1: Recon (auto-run nmap + httpx) ----------
        # For W3-A skeleton: always start with recon regardless of LLM
        await self._run_recon()

        # ---------- Phase 2: ReAct loop ----------
        for turn in range(1, self.max_decisions + 1):
            # Check time budget
            elapsed = time.time() - self.start_time
            if elapsed > MAX_SCAN_DURATION_SECONDS:
                logger.warning("Scan timed out after %ds", elapsed)
                await emit_scan_error(self.scan_id, f"Exceeded {MAX_SCAN_DURATION_SECONDS}s")
                return self._finalize("timeout", error=f"Exceeded {MAX_SCAN_DURATION_SECONDS}s")

            # Check token budget
            if self.total_tokens > MAX_TOKENS_PER_SCAN:
                logger.warning("Token budget exceeded: %d", self.total_tokens)
                await emit_scan_error(self.scan_id, f"Exceeded {MAX_TOKENS_PER_SCAN} tokens")
                return self._finalize("failed", error=f"Exceeded {MAX_TOKENS_PER_SCAN} tokens")

            # LLM decides next action (W3-A: stubbed — returns "done" after recon)
            # W3-B/C will replace with actual LiteLLM call
            thought, tool_name, tool_args = await self._llm_decide(turn)

            # Record decision
            decision = AgentDecision(
                turn=turn,
                thought=thought,
                tool_name=tool_name,
                tool_args=tool_args,
                tokens_used=100,  # stubbed
            )

            # Execute tool if specified
            if tool_name:
                observation = await self._call_tool(tool_name, tool_args)
                decision.observation = observation
                # Emit progress event (SSE)
                await emit_scan_progress(
                    self.scan_id, turn, thought, tool_name, observation,
                )
            else:
                # No tool = agent is done
                decision.observation = "Agent completed."
                self.decisions.append(decision)
                # Emit progress event (SSE) — agent done
                await emit_scan_progress(
                    self.scan_id, turn, thought, None, "Agent completed.",
                )
                break

            self.decisions.append(decision)
            self.total_tokens += decision.tokens_used

            logger.info("Turn %d: tool=%s args=%s", turn, tool_name, tool_args)

        # Check if we hit max iterations
        if len(self.decisions) >= self.max_decisions:
            await emit_scan_error(self.scan_id, f"Exceeded {self.max_decisions} decisions")
            return self._finalize("max_iterations", error=f"Exceeded {self.max_decisions} decisions")

        result = self._finalize("completed")
        # Emit scan_complete event (SSE)
        await emit_scan_complete(
            self.scan_id, result.status, len(result.findings), result.duration_seconds,
        )
        return result

    async def _run_recon(self) -> None:
        """Phase 1: auto-run recon tools (nmap + httpx) — W3-D actual execution."""
        logger.info("Recon phase: running nmap + httpx on %s", self.target)

        # Run nmap
        nmap_result = await self._call_tool("nmap", {
            "target": self.target,
            "ports": "1-1000",
            "version_detection": True,
            "timing": "4",
        })

        self.decisions.append(AgentDecision(
            turn=0,
            thought=f"Auto-recon: nmap scan on {self.target}",
            tool_name="nmap",
            tool_args={"target": self.target, "ports": "1-1000"},
            observation=nmap_result[:2000] if nmap_result else "No output",
            tokens_used=0,
        ))

    async def _llm_decide(self, turn: int) -> tuple[str, str | None, dict[str, Any]]:
        """Use LLM to decide next action.

        W3-A: STUB — returns "done" after first turn.
        W3-B/C: replace with actual LiteLLM call.
        """
        # Stub: agent is done after recon
        return (
            "Recon complete. No further action needed (W3-A skeleton — LLM not wired yet).",
            None,  # no tool = done
            {},
        )

    async def _call_tool(self, tool_name: str, tool_args: dict[str, Any]) -> str:
        """Call a tool by name — W3-D: actual subprocess execution.

        Uses SubprocessExecutor with scope guard validation.
        Returns tool output as string (or error message).
        """
        tool = get_tool(tool_name)
        if tool is None:
            return f"ERROR: tool '{tool_name}' not found"

        # Build command args from tool definition
        args = tool.build_command_args(**tool_args)
        command = [tool.command] + args

        # Execute via subprocess executor
        result = await self.executor.execute(
            command=command,
            target=tool_args.get("target", tool_args.get("domain", self.target)),
            timeout=tool.timeout,
            allowed_exit_codes=tool.allowed_exit_codes,
        )

        # Format result as string
        if result.success:
            output = result.stdout
            if result.truncated:
                output += f"\n[OUTPUT TRUNCATED — full output spilled to {result.spill_path}]"
            return output
        elif result.scope_violation:
            return f"SCOPE_VIOLATION: {result.error}"
        else:
            error_parts = []
            if result.error:
                error_parts.append(result.error)
            if result.stderr:
                error_parts.append(f"stderr: {result.stderr[:500]}")
            return f"ERROR (exit code {result.exit_code}): {' | '.join(error_parts)}"

    def _finalize(self, status: str, error: str | None = None) -> ScanResult:
        """Create final ScanResult."""
        return ScanResult(
            scan_id=self.scan_id,
            target=self.target,
            user_prompt=self.user_prompt,
            status=status,
            decisions=self.decisions,
            findings=self.findings,
            total_tokens=self.total_tokens,
            duration_seconds=time.time() - self.start_time,
            error=error,
            completed_at=datetime.now(UTC),
        )


# ---------- Convenience function ----------

async def run_scan(
    target: str,
    user_prompt: str,
    scan_id: str | None = None,
    scope_guard: ScopeGuard | None = None,
) -> ScanResult:
    """One-shot: run a scan with the ReAct agent.

    Args:
        target: Target URL or IP (e.g. "http://example.com")
        user_prompt: Natural-language prompt (e.g. "Scan for SQL injection")
        scan_id: Optional scan ID (auto-generated if None)
        scope_guard: Optional scope guard (auto-creates with target as allowed host)

    Returns:
        ScanResult with decisions, findings, status
    """
    if scan_id is None:
        scan_id = f"scan_{uuid.uuid4().hex[:12]}"

    agent = ReActAgent(
        scan_id=scan_id,
        target=target,
        user_prompt=user_prompt,
        scope_guard=scope_guard,
    )
    return await agent.run()


# ---------- Tool listing helper ----------

def get_available_tools() -> list[dict[str, Any]]:
    """List all tools available to the agent (for UI + LLM context)."""
    return [
        {
            "name": t.name,
            "description": t.short_description or t.description[:200],
            "category": t.category,
            "safety_class": t.safety_class,
            "wstg_ids": t.wstg_ids,
            "parameters": [p.name for p in t.parameters],
        }
        for t in list_tools()
    ]
