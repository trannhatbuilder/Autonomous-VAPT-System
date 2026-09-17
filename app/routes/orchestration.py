"""
VAPT-AI Orchestration Routes — W9-S7 / W10-S6.

FastAPI router exposing the 3 orchestration modes (deep/plan_execute/supervisor)
via a unified endpoint + per-agent introspection + invocation endpoints.

W10-S6 update:
    - /agents now lists all 16 agents from agent_registry (3 orchestrators + 13 sub-agents)
    - Added GET  /api/orchestration/agents/{name}    — single agent metadata
    - Added POST /api/orchestration/agents/{name}/invoke — invoke single agent

Endpoints:
    GET  /api/orchestration/modes
        List 3 orchestration modes with descriptions.

    GET  /api/orchestration/agents
        List all 16 agents (orchestrators + sub-agents) with metadata.
        W10: now derived from agent_registry (not hardcoded).

    GET  /api/orchestration/agents/{name}
        Get single agent metadata (W10-S6 NEW).
        Returns: AgentMetadata.to_dict() + tool_allowlist + safety_class + prompt_file.

    POST /api/orchestration/agents/{name}/invoke
        Invoke a single agent in isolation (W10-S6 NEW).
        Body: {"target": "...", "task_description": "...", "scan_id": "..."}
        Returns: AgentRunResult.to_dict() with decisions + status.

    POST /api/orchestration/scans/start-mode
        Start a scan with a specific orchestration mode (or auto-select).

Usage:
    # In app/main.py:
    from app.routes.orchestration import router as orch_router
    app.include_router(orch_router)
"""
from __future__ import annotations

import logging
import uuid
from typing import Any

from fastapi import APIRouter, Body, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_async_session
from app.auth.manager import decode_access_token, InvalidTokenError
from app.agents.registry import agent_registry, AgentMetadata
from app.agents.base import create_agent, AgentRunResult
from app.orchestration.base import OrchestratorMode, generate_scan_id
from app.orchestration.mode_selector import (
    ModeSelectionInput,
    select_mode,
)
from app.orchestration.langgraph_supervisor import (
    EXPERT_AGENTS as SUPERVISOR_EXPERTS,
    SupervisorOrchestrator,
)
from app.orchestration.langgraph_deep import (
    STUB_SUB_AGENT_TASKS as DEEP_TASKS,
    DeepOrchestrator,
)
from app.orchestration.langgraph_plan_execute import (
    STUB_PLAN as PE_PLAN,
    PlanExecuteOrchestrator,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/orchestration", tags=["orchestration"])


# ---------- Auth dependency (reuse from app.main) ----------

async def _require_user_id(token: str | None) -> str:
    """Validate Bearer token + return user_id (stub — W9 just checks token shape).

    W10+ will use the full UserResponse Depends from app.main.
    """
    if not token:
        raise HTTPException(status_code=401, detail="Bearer token required")
    token = token.removeprefix("Bearer ").strip()
    try:
        payload = decode_access_token(token)
        if payload.get("type") != "access":
            raise HTTPException(status_code=401, detail="Invalid token type")
        return payload.get("sub", "unknown")
    except InvalidTokenError as e:
        raise HTTPException(status_code=401, detail=f"Invalid token: {e}") from e


# ---------- Request/Response models ----------

class StartModeScanRequest(BaseModel):
    """Request body for POST /api/scans/start-mode."""
    target: str = Field(..., description="Target URL or IP (e.g. http://example.com)")
    user_prompt: str = Field("", description="Natural-language prompt")
    mode: str = Field(
        "auto",
        description='Orchestration mode: "supervisor" | "deep" | "plan_execute" | "auto"',
    )
    scope_size: int = Field(
        1, ge=1, le=1000,
        description="Number of in-scope hosts (used for auto mode selection)",
    )
    has_post_exploitation: bool = Field(
        False,
        description="Whether user wants full kill-chain (used for auto mode selection)",
    )
    transfer_targets: list[str] | None = Field(
        None,
        description="Optional (supervisor mode only): list of agent names to transfer to in sequence.",
    )


class ModeInfoResponse(BaseModel):
    mode: str
    description: str
    prompt_file: str


class InvokeAgentRequest(BaseModel):
    """Request body for POST /api/orchestration/agents/{name}/invoke (W10-S6)."""
    target: str = Field(..., description="Target URL or IP")
    task_description: str = Field("", description="Task description for the agent")
    scan_id: str | None = Field(None, description="Optional scan ID (auto-generated if None)")
    user_prompt: str = Field("", description="Original user prompt (for context)")
    max_iterations: int | None = Field(
        None, ge=1, le=30,
        description="Optional max iterations cap (default from registry, capped at 30 per D18)",
    )


# ---------- Endpoints ----------

@router.get("/modes")
async def list_modes() -> dict[str, Any]:
    """List all available orchestration modes with descriptions."""
    modes = [
        {
            "mode": OrchestratorMode.SUPERVISOR.value,
            "description": (
                "Default mode. Transfer mechanism to expert sub-agents (serial). "
                "Best for single-target web app scans."
            ),
            "prompt_file": "orchestrator-supervisor.md",
        },
        {
            "mode": OrchestratorMode.DEEP.value,
            "description": (
                "Parallel sub-agents via task delegation. "
                "Best for network ranges or multi-host scans."
            ),
            "prompt_file": "orchestrator.md",
        },
        {
            "mode": OrchestratorMode.PLAN_EXECUTE.value,
            "description": (
                "Planner → Executor → Replanner loop with re-planning. "
                "Best for full kill-chain scans (exploit → privesc → lateral)."
            ),
            "prompt_file": "orchestrator-plan-execute.md",
        },
    ]
    return {"modes": modes, "default": OrchestratorMode.SUPERVISOR.value}


@router.get("/agents")
async def list_agents() -> dict[str, Any]:
    """List all 16 registered agents (W10-S6 — now from agent_registry).

    Returns orchestrators + sub-agents with full metadata:
        name, display_name, description, safety_class, tool_allowlist,
        max_iterations, prompt_file, is_orchestrator, orchestration_mode
    """
    return {
        "total_count": agent_registry.total_count,
        "orchestrator_count": agent_registry.orchestrator_count,
        "sub_agent_count": agent_registry.sub_agent_count,
        "agents": [m.to_dict() for m in agent_registry.list_agents()],
        "by_safety_class": {
            "read_only": [m.name for m in agent_registry.list_read_only()],
            "destructive": [m.name for m in agent_registry.list_destructive()],
            "advisory": [m.name for m in agent_registry.list_by_safety_class("advisory")],
        },
        "note": (
            "W10-S6: 16 agents (3 orchestrators + 13 sub-agents) loaded from agent_registry. "
            "Use GET /api/orchestration/agents/{name} for single-agent details. "
            "Use POST /api/orchestration/agents/{name}/invoke to invoke an agent in isolation."
        ),
    }


@router.get("/agents/{name}")
async def get_agent(name: str) -> dict[str, Any]:
    """Get single agent metadata by name (W10-S6 NEW).

    Args:
        name: Agent name (e.g. "recon", "penetration", "orchestrator-supervisor")

    Returns:
        AgentMetadata.to_dict() + prompt content length.

    Raises:
        404: if agent not found in registry.
    """
    meta = agent_registry.get_agent(name)
    if meta is None:
        available = sorted(agent_registry.metadata.keys())
        raise HTTPException(
            status_code=404,
            detail=f"Agent {name!r} not found. Available: {available}",
        )
    # Include prompt content length (don't return full content — too large for response)
    try:
        prompt_content = agent_registry.load_prompt(name)
        prompt_len = len(prompt_content)
    except Exception:
        prompt_len = 0
    result = meta.to_dict()
    result["prompt_content_length"] = prompt_len
    return result


@router.post("/agents/{name}/invoke")
async def invoke_agent(
    name: str,
    req: InvokeAgentRequest = Body(...),
    session: AsyncSession = Depends(get_async_session),
) -> dict[str, Any]:
    """Invoke a single agent in isolation (W10-S6 NEW).

    W10 stub: agent runs synchronously + returns full result.
    W11+ will run async via Celery + stream events via SSE.

    Args:
        name: Agent name (e.g. "recon", "penetration")
        req: InvokeAgentRequest with target + task_description + optional scan_id

    Returns:
        AgentRunResult.to_dict() with decisions, status, agents_involved.

    Raises:
        404: if agent not found in registry
        400: if agent is an orchestrator (use /scans/start-mode instead)
        500: if agent run fails
    """
    # Validate agent exists
    meta = agent_registry.get_agent(name)
    if meta is None:
        available = sorted(agent_registry.metadata.keys())
        raise HTTPException(
            status_code=404,
            detail=f"Agent {name!r} not found. Available: {available}",
        )

    # Orchestrators cannot be invoked directly — use /scans/start-mode
    if meta.is_orchestrator:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Agent {name!r} is an orchestrator — use "
                f"POST /api/orchestration/scans/start-mode instead."
            ),
        )

    # Generate scan_id if not provided
    scan_id = req.scan_id or generate_scan_id()

    logger.info(
        "Invoking agent: name=%s scan=%s target=%s task=%s",
        name, scan_id, req.target, req.task_description[:50],
    )

    # Create agent instance via factory
    try:
        agent = create_agent(
            agent_name=name,
            scan_id=scan_id,
            target=req.target,
            task_description=req.task_description,
            user_prompt=req.user_prompt,
            max_iterations=req.max_iterations,
        )
    except ImportError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Agent {name!r} has no specialist implementation: {e}",
        ) from e

    # Run agent
    try:
        result: AgentRunResult = await agent.run()
    except Exception as e:
        logger.exception("Agent run failed: name=%s scan=%s", name, scan_id)
        raise HTTPException(
            status_code=500,
            detail=f"Agent {name!r} run failed: {e}",
        ) from e

    # Build response
    response = result.to_dict()
    response["agent_metadata"] = meta.to_dict()
    return response


@router.post("/scans/start-mode")
async def start_mode_scan(
    req: StartModeScanRequest = Body(...),
    session: AsyncSession = Depends(get_async_session),
) -> dict[str, Any]:
    """Start a scan with a specific orchestration mode (or auto-select).

    W9 stub: runs synchronously and returns the full result.
    W10+ will run async via Celery + stream events via SSE.

    W10-S5: supervisor mode now accepts optional `transfer_targets` list
    to customize the agent sequence (default: ["recon", "reporting-remediation"]).

    Authentication: pass Bearer token in Authorization header.
    """
    # Auto-select mode if requested
    if req.mode == "auto":
        selection = select_mode(ModeSelectionInput(
            target=req.target,
            scope_size=req.scope_size,
            has_post_exploitation=req.has_post_exploitation,
        ))
        chosen_mode = selection.mode
        selection_info = selection.to_dict()
    else:
        try:
            chosen_mode = OrchestratorMode(req.mode)
        except ValueError as e:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid mode '{req.mode}'. Valid: deep, plan_execute, supervisor, auto.",
            ) from e
        selection_info = {
            "mode": chosen_mode.value,
            "reason": f"Explicit mode requested by user",
            "user_override_applied": True,
        }

    # Generate scan_id
    scan_id = generate_scan_id()

    logger.info(
        "Starting mode scan: scan=%s mode=%s target=%s",
        scan_id, chosen_mode.value, req.target,
    )

    # Dispatch to the right orchestrator
    if chosen_mode == OrchestratorMode.SUPERVISOR:
        orchestrator = SupervisorOrchestrator(
            scan_id=scan_id,
            target=req.target,
            user_prompt=req.user_prompt,
            transfer_targets=req.transfer_targets,
        )
    elif chosen_mode == OrchestratorMode.DEEP:
        orchestrator = DeepOrchestrator(
            scan_id=scan_id,
            target=req.target,
            user_prompt=req.user_prompt,
        )
    else:  # PLAN_EXECUTE
        orchestrator = PlanExecuteOrchestrator(
            scan_id=scan_id,
            target=req.target,
            user_prompt=req.user_prompt,
        )

    # Run orchestrator
    try:
        result = await orchestrator.run()
    except Exception as e:
        logger.exception("Orchestrator run failed: scan=%s mode=%s", scan_id, chosen_mode.value)
        raise HTTPException(
            status_code=500,
            detail=f"Orchestrator failed: {e}",
        ) from e

    # Build response
    response = result.to_dict()
    response["mode_selection"] = selection_info
    return response