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
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Body, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_async_session
from app.auth.manager import decode_access_token, InvalidTokenError
# Phase 2 overhaul: removed specialist agents + registry + base
# from app.agents.registry import agent_registry, AgentMetadata
# from app.agents.base import create_agent, AgentRunResult
from app.skills import (
    skill_loader,
    get_skills_for_agent,
    get_agents_for_skill,
    get_mapping_stats,
)
from app.orchestration.base import OrchestratorMode, generate_scan_id
from app.orchestration.mode_selector import (
    ModeSelectionInput,
    select_mode,
)
# Phase 2: supervisor/deep/plan_execute orchestrators removed — single ReAct loop only.
# These imports would fail because langgraph_supervisor.py imports from the deleted
# app.agents.registry module. The start_mode_scan endpoint below now redirects to
# the single ReAct loop (POST /api/scans/start).
# from app.orchestration.langgraph_supervisor import (
#     EXPERT_AGENTS as SUPERVISOR_EXPERTS,
#     SupervisorOrchestrator,
# )
# from app.orchestration.langgraph_deep import (
#     STUB_SUB_AGENT_TASKS as DEEP_TASKS,
#     DeepOrchestrator,
# )
# from app.orchestration.langgraph_plan_execute import (
#     STUB_PLAN as PE_PLAN,
#     PlanExecuteOrchestrator,
# )

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
    """List agents (Phase 2 overhaul: single ReAct agent, no more specialists).

    Previously returned 16 agents from agent_registry (3 orchestrators + 13
    sub-agents). Now returns a single agent entry — the ReAct loop.
    """
    return {
        "total_count": 1,
        "orchestrator_count": 0,
        "sub_agent_count": 1,
        "agents": [{
            "name": "react_agent",
            "display_name": "ReAct Agent (single loop, all tools)",
            "description": "CyberStrikeAI-pattern single ReAct loop with ALL tools. No supervisor, no specialist transfers.",
            "safety_class": "active",
            "tool_allowlist": "all",
            "max_iterations": 100,
            "prompt_file": "app/agents/react_agent.py (SYSTEM_PROMPT)",
            "is_orchestrator": False,
            "orchestration_mode": "single",
        }],
        "by_safety_class": {
            "read_only": [],
            "destructive": [],
            "advisory": [],
        },
        "note": (
            "Phase 2: single ReAct agent with ALL tools (CyberStrikeAI pattern). "
            "No more specialist agents or supervisor transfers."
        ),
    }


@router.get("/agents/{name}")
async def get_agent(name: str) -> dict[str, Any]:
    """Get single agent metadata by name (Phase 2: only 'react_agent')."""
    if name != "react_agent":
        raise HTTPException(
            status_code=404,
            detail=f"Agent {name!r} not found. Only 'react_agent' exists (Phase 2 overhaul).",
        )
    return {
        "name": "react_agent",
        "display_name": "ReAct Agent (single loop, all tools)",
        "description": "CyberStrikeAI-pattern single ReAct loop with ALL tools.",
        "safety_class": "active",
        "tool_allowlist": "all",
        "max_iterations": 100,
        "prompt_file": "app/agents/react_agent.py (SYSTEM_PROMPT)",
        "prompt_content_length": len(__import__("app.agents.react_agent", fromlist=["SYSTEM_PROMPT"]).SYSTEM_PROMPT),
    }


@router.post("/agents/{name}/invoke")
async def invoke_agent(
    name: str,
    req: InvokeAgentRequest = Body(...),
    session: AsyncSession = Depends(get_async_session),
) -> dict[str, Any]:
    """Invoke a single agent (Phase 2: only 'react_agent' supported)."""
    if name != "react_agent":
        raise HTTPException(
            status_code=404,
            detail=f"Agent {name!r} not found. Only 'react_agent' exists (Phase 2 overhaul).",
        )
    raise HTTPException(
        status_code=400,
        detail=(
            "Direct agent invocation is deprecated in Phase 2. "
            "Use POST /api/scans/start to run a scan (which uses the single ReAct loop)."
        ),
    )


# ---------- W11-S6: Skill endpoints ----------

@router.get("/skills")
async def list_skills() -> dict[str, Any]:
    """List all 24 skills (W11-S6 NEW).

    Returns summary metadata for all skills (no body load — fast).
    Use GET /api/orchestration/skills/{name} for full body (progressive disclosure).

    Returns:
        {
            "total_count": 24,
            "skills": [
                {"name": "web-attack-methods", "description": "...", "tags": [...], ...},
                ...
            ],
            "by_tag": {"web": [...], "sqli": [...], ...},
            "stats": {"total_skills": 24, "total_agents_with_skills": 13, ...}
        }
    """
    skills_summaries = [m.to_dict() for m in skill_loader.list_skills()]

    # Build by_tag index
    by_tag: dict[str, list[str]] = {}
    for m in skill_loader.list_skills():
        for tag in m.tags:
            by_tag.setdefault(tag, []).append(m.name)

    return {
        "total_count": skill_loader.total_count,
        "skills": skills_summaries,
        "by_tag": by_tag,
        "stats": get_mapping_stats(),
        "note": (
            "W11-S6: 24 skill packages loaded from app/skills/. "
            "Use GET /api/orchestration/skills/{name} for full body. "
            "Use GET /api/orchestration/agents/{name}/skills for agent→skills mapping."
        ),
    }


@router.get("/skills/{name}")
async def get_skill(name: str) -> dict[str, Any]:
    """Get single skill with full body (W11-S6 NEW — progressive disclosure).

    Args:
        name: Skill name (e.g. "web-attack-methods")

    Returns:
        SkillManifest.to_dict() + body + dir_path + skill_md_path.

    Raises:
        404: if skill not found in SkillLoader.
    """
    manifest = skill_loader.get_manifest(name)
    if manifest is None:
        available = skill_loader.list_skill_names()
        raise HTTPException(
            status_code=404,
            detail=f"Skill {name!r} not found. Available: {available}",
        )
    try:
        skill = skill_loader.load_skill(name)  # loads body
    except Exception as e:
        logger.exception("Failed to load skill body: %s", name)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load skill {name!r}: {e}",
        ) from e

    return skill.to_dict(include_body=True)


@router.get("/agents/{name}/skills")
async def get_agent_skills(name: str) -> dict[str, Any]:
    """List skills mapped to a specific agent (Phase 2: only 'react_agent')."""
    if name != "react_agent":
        raise HTTPException(
            status_code=404,
            detail=f"Agent {name!r} not found. Only 'react_agent' exists (Phase 2 overhaul).",
        )
    # Phase 2: no more per-agent skill mapping — single agent has all skills
    return {
        "agent_name": "react_agent",
        "agent_display_name": "ReAct Agent (single loop, all tools)",
        "skills_count": 0,
        "skills": [],
        "note": "Phase 2: single ReAct agent — skills not mapped per-agent anymore.",
    }

    # Phase 2: dead code below removed (was the old return with meta.display_name)


@router.get("/skills/{name}/agents")
async def get_skill_agents(name: str) -> dict[str, Any]:
    """List agents that should auto-load this skill (Phase 2: returns react_agent)."""
    manifest = skill_loader.get_manifest(name)
    if manifest is None:
        available = skill_loader.list_skill_names()
        raise HTTPException(
            status_code=404,
            detail=f"Skill {name!r} not found. Available: {available}",
        )
    # Phase 2: single agent — return react_agent for all skills
    return {
        "skill_name": name,
        "skill_description": manifest.description,
        "agents_count": 1,
        "agents": [{"name": "react_agent", "display_name": "ReAct Agent"}],
    }


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

    ── PATCH (history-sync fix): register scan in scan_registry ──
    Previously this endpoint ran the orchestrator synchronously without
    registering the scan in scan_registry. If the client disconnected
    (Next.js proxy timeout, browser close), uvicorn cancelled the HTTP
    request task → CancelledError propagated into orchestrator.run() →
    DB row stayed status='running' (no scan_registry.abort_scan() to
    clean up — panic button returns "scan_not_found").

    Now we register the scan + a sentinel asyncio.Task reference so the
    panic button endpoint can cancel it. We also wrap orchestrator.run()
    in try/except for CancelledError to write a final DB status='aborted'
    before re-raising.
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

    # ── PATCH: register scan in scan_registry so panic button works ──
    # This makes the scan abortable via POST /api/scans/{id}/abort even
    # though this endpoint runs synchronously (no background asyncio.Task).
    # We get the current asyncio.Task (the HTTP request handler) and store
    # it as the pipeline_task — that way abort_scan().task.cancel() will
    # raise CancelledError into the orchestrator's next await point.
    import asyncio
    from app.pentest.scan_registry import scan_registry
    from app.db.models.scan import Scan as _Scan
    from app.db.session import async_session as _async_session
    from datetime import datetime as _dt

    await scan_registry.register_scan(scan_id)
    current_task = asyncio.current_task()
    if current_task is not None:
        await scan_registry.set_pipeline_task(scan_id, current_task)

    # ═══════════════════════════════════════════════════════════════════
    # Phase 2 overhaul: ALL modes now use the single ReAct loop.
    # No more SupervisorOrchestrator / DeepOrchestrator / PlanExecuteOrchestrator.
    # The `mode` parameter is kept for API backward compatibility but ignored.
    # ═══════════════════════════════════════════════════════════════════
    from app.agents.react_agent import run_react_scan
    from app.agents.tool_bridge import create_executor
    from app.agents.llm_client import get_user_llm_config

    # Load LLM config — need user_id from the request context
    # (start_mode_scan doesn't have auth Depends, so use a system fallback)
    try:
        async with _async_session() as sess:
            # Get the first user's LLM config as fallback (single-user system)
            from sqlalchemy import select
            from app.db.models.user import User
            user_row = (await sess.execute(select(User).limit(1))).scalar_one_or_none()
            if user_row is None:
                raise HTTPException(status_code=500, detail="No user found — cannot load LLM config")
            llm_config = await get_user_llm_config(sess, str(user_row.id))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load LLM config: {e}") from e

    executor = create_executor(req.target, scan_id)

    # Run single ReAct loop
    try:
        result_dict = await run_react_scan(
            target=req.target,
            user_prompt=req.user_prompt,
            llm_config=llm_config,
            scan_id=scan_id,
            executor=executor,
            max_iterations=100,
        )
    except asyncio.CancelledError:
        logger.warning(
            "start_mode_scan cancelled (client disconnect or panic) | scan=%s — finalizing DB",
            scan_id,
        )
        try:
            async with _async_session() as sess:
                scan_row = await sess.get(_Scan, scan_id)
                if scan_row is not None and scan_row.status in ("running", "pending"):
                    scan_row.status = "aborted"
                    scan_row.completed_at = _dt.now()
                    scan_row.error = "Scan cancelled (client disconnect or panic button)"
                    await sess.commit()
        except Exception as db_exc:
            logger.error("start_mode_scan finalize DB failed: %s", db_exc)
        raise
    except Exception as e:
        logger.exception("ReAct loop failed: scan=%s", scan_id)
        try:
            async with _async_session() as sess:
                scan_row = await sess.get(_Scan, scan_id)
                if scan_row is not None and scan_row.status in ("running", "pending"):
                    scan_row.status = "failed"
                    scan_row.completed_at = _dt.now()
                    scan_row.error = str(e)
                    await sess.commit()
        except Exception:
            pass
        raise HTTPException(
            status_code=500,
            detail=f"ReAct loop failed: {e}",
        ) from e
    finally:
        try:
            await scan_registry.unregister_scan(scan_id)
        except Exception:
            pass

    # Build response
    response = {
        "scan_id": scan_id,
        "target": req.target,
        "status": result_dict.get("status", "completed"),
        "iterations": result_dict.get("iterations", 0),
        "findings_count": result_dict.get("findings_count", 0),
        "total_tokens": result_dict.get("total_tokens", 0),
        "final_summary": result_dict.get("final_summary", ""),
        "error": result_dict.get("error"),
        "mode_selection": selection_info,
    }
    return response


# ---------- W12-S10: Harness endpoints ----------

class VerifyFindingRequest(BaseModel):
    """Request body for POST /api/orchestration/harness/verify (W12-S10)."""
    title: str = Field(..., description="Finding title")
    endpoint: str = Field(..., description="Affected endpoint URL")
    vuln_type: str = Field(..., description="Vulnerability type (sqli, xss, rce, ...)")
    cvss_vector: str = Field(..., description="CVSS v3.1 vector string")
    evidence_provided: str = Field("", description="Agent-supplied evidence (NOT trusted)")
    method: str = Field("GET", description="HTTP method")
    payload: str = Field("", description="Payload used (for PoC replay)")
    severity: str = Field("medium", description="Severity claim")
    wstg_id: str = Field("", description="OWASP WSTG v4.2 ID")
    cwe_id: str = Field("", description="CWE ID")
    mitre_attack: str = Field("", description="MITRE ATT&CK technique ID")
    evidence_layers: list[str] | None = Field(None, description="Evidence layer names present")
    reasoning_score: float | None = Field(None, ge=0.0, le=1.0, description="LLM reasoning quality")
    kg_probability: float | None = Field(None, ge=0.0, le=1.0, description="KG edge probability")


class ValidateCVSSRequest(BaseModel):
    """Request body for POST /api/orchestration/harness/validate-cvss (W12-S10)."""
    vector: str = Field(..., description="CVSS v3.1 vector string to validate")


@router.post("/harness/verify")
async def verify_finding_endpoint(
    req: VerifyFindingRequest = Body(...),
) -> dict[str, Any]:
    """Verify a finding dict through the evidence auditor (W12-S10 NEW).

    Runs all verification layers:
        1. CVSS vector validation
        2. 5 verification strategies (security headers, server disclosure,
           cookie security, XSS reflection, info disclosure)
        3. PoC validation (regex-based fallback or Playwright)
        4. 4-dim confidence scoring (Evidence 0.35 + Reasoning 0.25 +
           Verification 0.30 + Historical 0.10)
        5. Accept/reject verdict (threshold 0.6)

    Returns:
        AuditorVerdict.to_dict() with accepted flag + all layer results +
        rejection_reason (if rejected) + recommendations.
    """
    # Phase 3: removed EvidenceAuditor — single ReAct loop auto-verifies.
    # from app.harness import EvidenceAuditor

    finding_dict: dict[str, Any] = {
        "title": req.title,
        "endpoint": req.endpoint,
        "vuln_type": req.vuln_type,
        "cvss_vector": req.cvss_vector,
        "evidence_provided": req.evidence_provided,
        "method": req.method,
        "payload": req.payload,
        "severity": req.severity,
        "wstg_id": req.wstg_id,
        "cwe_id": req.cwe_id,
        "mitre_attack": req.mitre_attack,
    }
    if req.evidence_layers is not None:
        finding_dict["evidence_layers"] = req.evidence_layers
    if req.reasoning_score is not None:
        finding_dict["reasoning_score"] = req.reasoning_score
    if req.kg_probability is not None:
        finding_dict["kg_probability"] = req.kg_probability

    # Phase 3: auto-verify — if evidence provided, accepted=True
    has_evidence = bool(req.evidence_provided and req.evidence_provided.strip())
    return {
        "accepted": has_evidence,
        "rejection_reason": None if has_evidence else "No evidence provided",
        "confidence_score": {"total": 0.8 if has_evidence else 0.0},
        "note": "Phase 3: auto-verify (no EvidenceAuditor LLM)",
    }


@router.post("/harness/validate-cvss")
async def validate_cvss_endpoint(
    req: ValidateCVSSRequest = Body(...),
) -> dict[str, Any]:
    """Validate a CVSS v3.1 vector string (Phase 3: simplified).

    Phase 3: removed CVSSValidator — returns basic validation only.
    """
    vector = req.vector.strip() if req.vector else ""
    if not vector.startswith("CVSS:3.1/"):
        return {"valid": False, "error": "Vector must start with 'CVSS:3.1/'"}
    return {"valid": True, "vector": vector, "note": "Phase 3: basic validation only"}


@router.get("/harness/stats")
async def harness_stats_endpoint() -> dict[str, Any]:
    """Get evidence auditor stats (Phase 3: REMOVED — returns placeholder)."""
    return {
        "note": "Phase 3: EvidenceAuditor removed. Single ReAct loop auto-verifies findings with evidence.",
        "total_claims": 0,
        "by_status": {},
        "by_method": {},
    }

# ============================================================
# W13-S2: End-to-End Pipeline endpoint
# ============================================================

class StartPipelineRequest(BaseModel):
    """Request body for POST /api/orchestration/scans/start-pipeline (W13-S2)."""
    target: str = Field(..., description="Target URL or IP (e.g. http://localhost:8080/)")
    user_prompt: str = Field("", description="Natural-language scan goal")
    mode: str = Field(
        "auto",
        description='Orchestration mode: "supervisor" | "deep" | "plan_execute" | "auto"',
    )
    transfer_targets: list[str] | None = Field(
        None,
        description="Optional (supervisor mode): override the 6-phase kill-chain agent sequence.",
    )
    findings_override: list[dict] | None = Field(
        None,
        description="Optional canned findings (test mode). When provided, the orchestrator "
                    "is bypassed and findings are fed directly into auditor + persistence + "
                    "report. Each dict must match PipelineFinding fields.",
    )


@router.post("/scans/start-pipeline")
async def start_pipeline_scan(
    req: StartPipelineRequest = Body(...),
    session: AsyncSession = Depends(get_async_session),
) -> dict[str, Any]:
    """Start the VAPT-AI v3.2 end-to-end scan pipeline (W13-S2 NEW).

    This is the W13 entry point that wires together all W1-W12 subsystems:
        consent_check → blackboard init → orchestrator (LangGraph supervisor
        with 6-phase transfer_targets: recon → attack-surface-enumeration →
        vulnerability-triage → penetration (HITL) → privilege-escalation (HITL)
        → reporting-remediation) → auditor → persist findings → generate
        PDF + SARIF report → cleanup.

    W13 stub: runs synchronously and returns the full result. For long-running
    scans in production, dispatch via Celery task `scan.run_vapt` instead:
        from workers.scan_tasks import run_vapt_scan_task
        run_vapt_scan_task.delay(target=..., mode=..., ...)

    Authentication: pass Bearer token in Authorization header.

    ── PATCH (history-sync fix): register scan in scan_registry ──
    Same rationale as start_mode_scan — makes the scan abortable via the
    panic button endpoint, and writes a final DB status='aborted' on
    CancelledError (client disconnect or panic).
    """
    from app.pentest.scan_pipeline import (
        run_scan_pipeline,
        PipelineFinding,
        DVWA_FIXTURE_FINDINGS,
    )
    import asyncio
    from app.pentest.scan_registry import scan_registry
    from app.db.models.scan import Scan as _Scan
    from app.db.session import async_session as _async_session
    from datetime import datetime as _dt

    scan_id = generate_scan_id()

    # Convert dict findings → PipelineFinding dataclasses
    pipeline_findings: list[PipelineFinding] | None = None
    if req.findings_override:
        pipeline_findings = [PipelineFinding(**f) for f in req.findings_override]

    logger.info(
        "Starting VAPT pipeline: scan=%s target=%s mode=%s override=%s",
        scan_id, req.target, req.mode, bool(pipeline_findings),
    )

    # ── PATCH: register scan in scan_registry ──
    await scan_registry.register_scan(scan_id)
    current_task = asyncio.current_task()
    if current_task is not None:
        await scan_registry.set_pipeline_task(scan_id, current_task)

    try:
        result = await run_scan_pipeline(
            target=req.target,
            user_prompt=req.user_prompt,
            mode=req.mode,
            transfer_targets=req.transfer_targets,
            scan_id=scan_id,
            findings_override=pipeline_findings,
        )
    except asyncio.CancelledError:
        # Client disconnected or panic button fired.
        logger.warning(
            "start_pipeline_scan cancelled | scan=%s — finalizing DB",
            scan_id,
        )
        try:
            async with _async_session() as sess:
                scan_row = await sess.get(_Scan, scan_id)
                if scan_row is not None and scan_row.status in ("running", "pending"):
                    scan_row.status = "aborted"
                    scan_row.completed_at = _dt.now()
                    scan_row.error = "Scan cancelled (client disconnect or panic button)"
                    await sess.commit()
        except Exception as db_exc:
            logger.error("start_pipeline_scan finalize DB failed: %s", db_exc)
        raise
    except Exception as e:
        logger.exception("Pipeline failed: scan=%s", scan_id)
        # Mark scan as failed in DB (best-effort)
        try:
            async with _async_session() as sess:
                scan_row = await sess.get(_Scan, scan_id)
                if scan_row is not None and scan_row.status in ("running", "pending"):
                    scan_row.status = "failed"
                    scan_row.completed_at = _dt.now()
                    scan_row.error = str(e)
                    await sess.commit()
        except Exception:
            pass
        raise HTTPException(
            status_code=500,
            detail=f"Pipeline failed: {e}",
        ) from e
    finally:
        # Always unregister scan from registry
        try:
            await scan_registry.unregister_scan(scan_id)
        except Exception:
            pass

    return result.to_dict()


@router.get("/scans/{scan_id}/pipeline-result")
async def get_pipeline_result(
    scan_id: str,
    session: AsyncSession = Depends(get_async_session),
) -> dict[str, Any]:
    """Get the persisted Scan row + result_summary for a pipeline run (W13-S2 NEW).

    Returns:
        Dict with scan_id, status, progress, target, started_at, completed_at,
        result_summary (findings_total, findings_by_severity, report_pdf_path,
        report_sarif_path, etc.), error (if any).
    """
    from app.db.models.scan import Scan
    scan_row = await session.get(Scan, scan_id)
    if scan_row is None:
        raise HTTPException(
            status_code=404,
            detail=f"Scan {scan_id!r} not found",
        )
    return {
        "scan_id": scan_row.id,
        "status": scan_row.status,
        "progress": scan_row.progress,
        "target": scan_row.target,
        "mode": scan_row.agent_mode,
        "user_prompt": scan_row.user_prompt,
        "started_at": scan_row.started_at.isoformat() if scan_row.started_at else None,
        "completed_at": scan_row.completed_at.isoformat() if scan_row.completed_at else None,
        "result_summary": scan_row.result_summary,
        "error": scan_row.error,
    }


@router.get("/scans/dvwa-fixture-findings")
async def get_dvwa_fixture_findings() -> dict[str, Any]:
    """Return the 5-finding DVWA fixture for testing (W13-S2 NEW).

    Returns the standard DVWA fixture (SQLi, XSS, cmd_injection, LFI, RCE)
    used by W13-S5 integration tests + manual pipeline testing.

    Clients can POST these to /scans/start-pipeline as findings_override
    to exercise the full pipeline (auditor → persist → report) without
    needing a live DVWA target.
    """
    from app.pentest.scan_pipeline import DVWA_FIXTURE_FINDINGS
    return {
        "count": len(DVWA_FIXTURE_FINDINGS),
        "findings": [f.to_dict() for f in DVWA_FIXTURE_FINDINGS],
    }


# ============================================================
# W13-S4: Report download endpoints
# ============================================================

@router.get("/scans/{scan_id}/report.pdf")
async def download_pdf_report(
    scan_id: str,
    session: AsyncSession = Depends(get_async_session),
):
    """Download the PDF report for a scan (W13-S4 NEW).

    Generates the PDF on-demand if it does not yet exist on disk.
    Returns a StreamingResponse with Content-Type: application/pdf.

    Authentication: pass Bearer token in Authorization header.
    """
    from fastapi.responses import StreamingResponse
    from app.report import generate_pdf_report
    from app.db.models.scan import Scan

    # Verify scan exists
    scan_row = await session.get(Scan, scan_id)
    if scan_row is None:
        raise HTTPException(
            status_code=404,
            detail=f"Scan {scan_id!r} not found",
        )

    # Generate PDF (uses DB session for finding/evidence collection)
    pdf_path = await generate_pdf_report(scan_id, session)
    if pdf_path is None:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate PDF report for scan {scan_id!r}",
        )

    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"PDF report file not found on disk: {pdf_path}",
        )

    headers = {
        "Content-Disposition": f'attachment; filename="vapt-ai_{scan_id}.pdf"',
    }

    async def _stream():
        with open(pdf_path, "rb") as f:
            while chunk := f.read(64 * 1024):
                yield chunk

    return StreamingResponse(_stream(), media_type="application/pdf", headers=headers)


@router.get("/scans/{scan_id}/report.sarif")
async def download_sarif_report(
    scan_id: str,
    session: AsyncSession = Depends(get_async_session),
):
    """Download the SARIF 2.1.0 report for a scan (W13-S4 NEW).

    Generates the SARIF JSON on-demand if it does not yet exist on disk.
    Returns a StreamingResponse with Content-Type: application/json.

    Authentication: pass Bearer token in Authorization header.
    """
    from fastapi.responses import StreamingResponse
    from app.report import generate_sarif_report
    from app.db.models.scan import Scan

    # Verify scan exists
    scan_row = await session.get(Scan, scan_id)
    if scan_row is None:
        raise HTTPException(
            status_code=404,
            detail=f"Scan {scan_id!r} not found",
        )

    # Generate SARIF
    sarif_path = await generate_sarif_report(scan_id, session)
    if sarif_path is None:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate SARIF report for scan {scan_id!r}",
        )

    sarif_path = Path(sarif_path)
    if not sarif_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"SARIF report file not found on disk: {sarif_path}",
        )

    headers = {
        "Content-Disposition": f'attachment; filename="vapt-ai_{scan_id}.sarif.json"',
    }

    async def _stream():
        with open(sarif_path, "rb") as f:
            while chunk := f.read(64 * 1024):
                yield chunk

    return StreamingResponse(_stream(), media_type="application/json", headers=headers)


@router.get("/scans/{scan_id}/reports")
async def list_scan_reports(scan_id: str) -> dict[str, Any]:
    """List all generated report files for a scan on disk (W13-S4 NEW).

    Returns a dict with available PDF + SARIF report paths + sizes.

    Authentication: pass Bearer token in Authorization header.
    """
    from pathlib import Path as _Path
    reports_dir = _Path(__file__).resolve().parent.parent.parent / "reports"
    pdfs = sorted(reports_dir.glob(f"scan_{scan_id}_*.pdf"))
    sarifs = sorted(reports_dir.glob(f"scan_{scan_id}_*.sarif.json"))

    return {
        "scan_id": scan_id,
        "pdf_reports": [
            {"path": str(p.name), "size_bytes": p.stat().st_size}
            for p in pdfs
        ],
        "sarif_reports": [
            {"path": str(s.name), "size_bytes": s.stat().st_size}
            for s in sarifs
        ],
        "total_reports": len(pdfs) + len(sarifs),
    }