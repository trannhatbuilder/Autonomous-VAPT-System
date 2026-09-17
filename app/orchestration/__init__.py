"""
VAPT-AI Orchestration Package — W9.

Multi-agent orchestration runtime built on LangGraph.

3 orchestration modes (per master plan §12 W9):
    - Deep           (langgraph_deep.py)          — parallel sub-agents via task delegation
    - Plan-Execute   (langgraph_plan_execute.py)  — planner → executor → replanner loop
    - Supervisor     (langgraph_supervisor.py)    — transfer to expert agents (default)

Architecture:
    ┌──────────────────────────────────────────────────────────────┐
    │ User submits target + prompt + mode                          │
    └────────────────────────────┬─────────────────────────────────┘
                                 ▼
    ┌──────────────────────────────────────────────────────────────┐
    │ ModeSelector.select_mode(target, scope) → OrchestratorMode   │
    └────────────────────────────┬─────────────────────────────────┘
                                 ▼
    ┌──────────────────────────────────────────────────────────────┐
    │ OrchestratorFactory.create(mode, scan_id, target, prompt)    │
    │   → BaseOrchestrator subclass instance                       │
    └────────────────────────────┬─────────────────────────────────┘
                                 ▼
    ┌──────────────────────────────────────────────────────────────┐
    │ orchestrator.run() → OrchestratorResult                      │
    │   - LangGraph state graph executes nodes                     │
    │   - Each node writes to PentestFact blackboard (DB)          │
    │   - Each node emits SSE events (scan_progress, finding_*)    │
    │   - HITL gate intercepts destructive ops (D25 mandatory)     │
    │   - Scope guard validates every subprocess target (D19)      │
    └──────────────────────────────────────────────────────────────┘

D18 guardrails (always enforced — see base.BaseOrchestrator):
    - Max 30 decisions per scan
    - Max 5 parallel sub-agents (Deep mode)
    - Max 16 agents (fixed — cannot create new at runtime)
    - Max 2M tokens per scan
    - Max 4 hours per scan

LangGraph usage:
    - StateGraph with TypedDict state (SharedState)
    - Nodes: orchestrator_node, sub_agent_node, executor_node, replanner_node
    - Edges: conditional routing based on state
    - Checkpointing: in-memory (W9) → SQLite/Postgres (W18+ with Harness Bridge)
"""
from __future__ import annotations

from app.orchestration.base import (
    AgentDecision,
    BaseOrchestrator,
    OrchestratorMode,
    OrchestratorResult,
    SharedState,
    load_orchestrator_prompt,
)

__all__ = [
    "AgentDecision",
    "BaseOrchestrator",
    "OrchestratorMode",
    "OrchestratorResult",
    "SharedState",
    "load_orchestrator_prompt",
]