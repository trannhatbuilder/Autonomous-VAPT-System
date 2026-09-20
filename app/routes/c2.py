"""
VAPT-AI C2 FastAPI Routes (W14-S8).

Endpoints for managing C2 listeners, sessions, and tasks via HTTP.

Authentication: pass Bearer token in Authorization header (W7 auth).
All endpoints are scoped to a scan_id (user provides scan_id in path).
"""
from __future__ import annotations

import logging
import uuid
from typing import Any

from fastapi import APIRouter, Body, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_async_session
from app.db.models.c2 import C2Session, C2Task
from app.c2 import (
    C2Manager, C2Hooks, CreateListenerInput, EnqueueTaskInput,
    ListenerConfig, ListenerType, TaskType, BeaconType,
    start_http_listener,
    list_unified_sessions, get_session_stats,
    InvalidInputError, SessionNotFoundError, SessionInactiveError, C2Error,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/c2", tags=["c2"])


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class CreateListenerRequest(BaseModel):
    """Request body for POST /api/c2/listeners."""
    scan_id: str = Field(..., description="Scan ID this listener belongs to")
    name: str = Field(..., description="Listener name (human-readable)")
    listener_type: str = Field(
        "http",
        description='Listener type: "http" | "https" | "tcp" | "websocket" (W14 only supports http/https)',
    )
    bind_host: str = Field("127.0.0.1", description="Bind address")
    bind_port: int = Field(8443, ge=1, le=65535, description="Bind port")
    use_tls: bool = Field(False, description="Use HTTPS (TLS) listener")
    tls_cert_path: str | None = Field(None, description="Path to TLS cert (None = auto self-sign)")
    tls_key_path: str | None = Field(None, description="Path to TLS key")
    auto_start: bool = Field(True, description="Start listener immediately after creation")


class StartListenerRequest(BaseModel):
    """Request body for POST /api/c2/listeners/{id}/start."""
    # No body needed — listener_id is in path


class EnqueueTaskRequest(BaseModel):
    """Request body for POST /api/c2/sessions/{id}/tasks."""
    task_type: str = Field(..., description="Task type (exec, shell, pwd, ls, sleep, etc.)")
    payload: dict[str, Any] = Field(default_factory=dict, description="Task payload")
    source: str = Field("manual", description="Task source: manual | ai | batch | api")
    level: int = Field(2, ge=1, le=5, description="HITL level 1-5 (L3+ requires HITL approval)")
    bypass_hitl: bool = Field(False, description="Skip HITL gate (system use only)")


# ---------------------------------------------------------------------------
# Listener endpoints
# ---------------------------------------------------------------------------

@router.post("/listeners")
async def create_listener(
    req: CreateListenerRequest = Body(...),
    session: AsyncSession = Depends(get_async_session),
) -> dict[str, Any]:
    """Create + (optionally) start a C2 HTTP listener (W14-S8 NEW).

    Returns listener metadata: listener_id, encryption_key_b64,
    implant_token_b64, bind info. The encryption_key + implant_token are
    sensitive — share them with the beacon via deploy-time injection
    (NOT over the same channel as the API response).

    Authentication: pass Bearer token in Authorization header.
    """
    # Validate listener type
    if not ListenerType.is_valid(req.listener_type):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid listener_type: {req.listener_type!r}. Valid: http, https, tcp, websocket",
        )
    if req.listener_type in ("tcp", "websocket"):
        raise HTTPException(
            status_code=400,
            detail=f"Listener type {req.listener_type!r} not implemented in W14 — see W15",
        )

    # Build ListenerConfig
    config = ListenerConfig(
        tls_cert_path=req.tls_cert_path,
        tls_key_path=req.tls_key_path,
    )
    config.apply_defaults()

    manager = C2Manager(session=session, scan_id=req.scan_id)

    try:
        if req.auto_start:
            # Use convenience function to create + start
            listener, meta = await start_http_listener(
                manager=manager,
                bind_host=req.bind_host,
                bind_port=req.bind_port,
                use_tls=req.use_tls,
                config=config,
                listener_name=req.name,
            )
            meta["status"] = "running"
            return meta
        else:
            # Just create (don't start)
            meta = await manager.create_listener(
                inp=CreateListenerInput(
                    name=req.name,
                    scan_id=req.scan_id,
                    listener_type=req.listener_type,
                    bind_host=req.bind_host,
                    bind_port=req.bind_port,
                    config=config,
                )
            )
            meta["status"] = "created"
            return meta
    except InvalidInputError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.exception("Listener creation failed")
        raise HTTPException(status_code=500, detail=f"Failed: {e}") from e


@router.get("/listeners/{listener_id}")
async def get_listener(
    listener_id: str,
    session: AsyncSession = Depends(get_async_session),
) -> dict[str, Any]:
    """Get listener metadata (W14-S8 NEW).

    Note: VAPT-AI does not have a separate c2_listeners table in W14.
    This endpoint returns a 404 — listener metadata is only persisted
    in-memory (running listeners tracked by C2Manager). W15 will add
    a c2_listeners table for full listener lifecycle management.
    """
    raise HTTPException(
        status_code=404,
        detail="Listener persistence not yet implemented (W15 will add c2_listeners table). "
               "For W14, listeners are tracked in-memory by C2Manager.",
    )


# ---------------------------------------------------------------------------
# Session endpoints (unified — D28)
# ---------------------------------------------------------------------------

@router.get("/sessions/{scan_id}")
async def list_sessions(
    scan_id: str,
    beacon_type: str | None = Query(None, description="Filter by beacon_type"),
    status: str | None = Query(None, description="Filter by status"),
    session: AsyncSession = Depends(get_async_session),
) -> dict[str, Any]:
    """List all C2 sessions for a scan — unified across all beacon types (W14-S8 NEW).

    D28: this is the single entry point for the C2 dashboard UI. Returns
    sessions from python_beacon, msf_meterpreter, msf_shell, sqlmap_webshell
    in one response, distinguished by beacon_type field.

    Authentication: pass Bearer token in Authorization header.
    """
    sessions = await list_unified_sessions(
        session, scan_id, beacon_type=beacon_type, status=status,
    )
    return {
        "scan_id": scan_id,
        "total": len(sessions),
        "sessions": [
            {
                "id": str(s.id),
                "beacon_type": s.beacon_type,
                "listener_type": s.listener_type,
                "remote_address": s.remote_address,
                "hostname": s.hostname,
                "username": s.username,
                "os": s.os,
                "arch": s.arch,
                "status": s.status,
                "checkin_at": s.checkin_at.isoformat() if s.checkin_at else None,
                "last_seen_at": s.last_seen_at.isoformat() if s.last_seen_at else None,
                "metadata": s.metadata_json or {},
            }
            for s in sessions
        ],
    }


@router.get("/sessions/{scan_id}/stats")
async def get_session_stats_endpoint(
    scan_id: str,
    session: AsyncSession = Depends(get_async_session),
) -> dict[str, Any]:
    """Get aggregate stats for C2 sessions in a scan (W14-S8 NEW).

    Returns counts by beacon_type + status.
    """
    return await get_session_stats(session, scan_id)


@router.get("/sessions/{scan_id}/{session_id}")
async def get_session(
    scan_id: str,
    session_id: str,
    session: AsyncSession = Depends(get_async_session),
) -> dict[str, Any]:
    """Get a single C2 session by ID (W14-S8 NEW)."""
    c2_session = await session.get(C2Session, uuid.UUID(session_id))
    if c2_session is None or c2_session.scan_id != scan_id:
        raise HTTPException(
            status_code=404,
            detail=f"Session {session_id!r} not found for scan {scan_id!r}",
        )
    return {
        "id": str(c2_session.id),
        "scan_id": c2_session.scan_id,
        "beacon_type": c2_session.beacon_type,
        "listener_type": c2_session.listener_type,
        "remote_address": c2_session.remote_address,
        "hostname": c2_session.hostname,
        "username": c2_session.username,
        "os": c2_session.os,
        "arch": c2_session.arch,
        "status": c2_session.status,
        "checkin_at": c2_session.checkin_at.isoformat() if c2_session.checkin_at else None,
        "last_seen_at": c2_session.last_seen_at.isoformat() if c2_session.last_seen_at else None,
        "metadata": c2_session.metadata_json or {},
    }


# ---------------------------------------------------------------------------
# Task endpoints
# ---------------------------------------------------------------------------

@router.post("/sessions/{scan_id}/{session_id}/tasks")
async def enqueue_task(
    scan_id: str,
    session_id: str,
    req: EnqueueTaskRequest = Body(...),
    session: AsyncSession = Depends(get_async_session),
) -> dict[str, Any]:
    """Queue a new task for a C2 session (W14-S8 NEW).

    If task_type is dangerous (L3+ per TaskType.is_dangerous), the task is
    queued with status="hitl_pending" and HITL approval is required before
    the beacon can pick it up. The HITL gate (W7) intercepts the enqueue.

    Authentication: pass Bearer token in Authorization header.
    """
    manager = C2Manager(session=session, scan_id=scan_id)

    try:
        task = await manager.enqueue_task(
            inp=EnqueueTaskInput(
                session_id=session_id,
                task_type=req.task_type,
                payload=req.payload,
                source=req.source,
                level=req.level,
                bypass_hitl=req.bypass_hitl,
            )
        )
        await session.commit()
    except SessionNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except SessionInactiveError as e:
        raise HTTPException(status_code=409, detail=str(e)) from e
    except InvalidInputError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.exception("Task enqueue failed")
        raise HTTPException(status_code=500, detail=f"Failed: {e}") from e

    return {
        "task_id": str(task.id),
        "session_id": str(task.session_id),
        "scan_id": scan_id,
        "task_type": task.command,
        "level": task.level,
        "status": task.status,
        "created_at": task.created_at.isoformat() if task.created_at else None,
    }


@router.get("/sessions/{scan_id}/{session_id}/tasks")
async def list_session_tasks(
    scan_id: str,
    session_id: str,
    status: str | None = Query(None, description="Filter by status"),
    session: AsyncSession = Depends(get_async_session),
) -> dict[str, Any]:
    """List all tasks for a C2 session (W14-S8 NEW)."""
    stmt = (
        select(C2Task)
        .where(C2Task.session_id == uuid.UUID(session_id))
    )
    if status:
        stmt = stmt.where(C2Task.status == status)
    stmt = stmt.order_by(C2Task.created_at.desc())

    result = await session.execute(stmt)
    tasks = list(result.scalars().all())

    return {
        "scan_id": scan_id,
        "session_id": session_id,
        "total": len(tasks),
        "tasks": [
            {
                "id": str(t.id),
                "task_type": t.command,
                "args": t.args_json or {},
                "level": t.level,
                "status": t.status,
                "result": t.result,
                "exit_code": t.exit_code,
                "error": t.error,
                "sent_at": t.sent_at.isoformat() if t.sent_at else None,
                "completed_at": t.completed_at.isoformat() if t.completed_at else None,
                "created_at": t.created_at.isoformat() if t.created_at else None,
            }
            for t in tasks
        ],
    }


@router.post("/sessions/{scan_id}/{session_id}/close")
async def close_session(
    scan_id: str,
    session_id: str,
    session: AsyncSession = Depends(get_async_session),
) -> dict[str, Any]:
    """Close a C2 session (mark as killed) (W14-S8 NEW)."""
    c2_session = await session.get(C2Session, uuid.UUID(session_id))
    if c2_session is None or c2_session.scan_id != scan_id:
        raise HTTPException(
            status_code=404,
            detail=f"Session {session_id!r} not found for scan {scan_id!r}",
        )

    from app.c2.types import SessionStatus
    c2_session.status = SessionStatus.KILLED.value
    await session.commit()

    # Emit session_dead SSE event
    try:
        from app.pentest.events import event_bus
        await event_bus.publish(scan_id, {
            "event": "session_dead",
            "scan_id": scan_id,
            "session_id": session_id,
            "reason": "closed_by_user",
            "timestamp": c2_session.last_seen_at.isoformat() if c2_session.last_seen_at else None,
        })
    except Exception as e:
        logger.warning("Failed to emit session_dead SSE: %s", e)

    return {
        "session_id": session_id,
        "scan_id": scan_id,
        "status": c2_session.status,
    }


# ============================================================
# W15-S6: C2 Dashboard HITL endpoints + Payload Builder
# ============================================================

class BuildPayloadRequest(BaseModel):
    """Request body for POST /api/c2/payloads (W15-S6)."""
    listener_id: str = Field(..., description="Listener ID this payload targets")
    listener_type: str = Field(..., description="Listener type: tcp | http | https | websocket")
    callback_host: str = Field(..., description="Attacker IP/hostname beacon connects back to")
    callback_port: int = Field(..., ge=1, le=65535, description="Attacker port")
    payload_type: str = Field(
        "python_beacon",
        description='Payload type: "python_beacon" | "bash_oneliner" | "powershell_oneliner" | "msfvenom_stager"',
    )
    os: str = Field("linux", description="Target OS: linux | windows | darwin")
    arch: str = Field("x64", description="Target arch: x64 | arm64 | x86")
    sleep_seconds: int = Field(5, ge=1, le=3600)
    jitter_percent: int = Field(0, ge=0, le=100)
    encryption_key_b64: str = Field("", description="AES-256-GCM key (base64)")
    implant_token_b64: str = Field("", description="Implant auth token (base64url)")
    output_name: str | None = Field(None, description="Custom output filename (no extension)")


@router.post("/payloads")
async def build_payload_endpoint(
    req: BuildPayloadRequest = Body(...),
) -> dict[str, Any]:
    """Generate a C2 payload (W15-S6 NEW).

    Generates one of 4 payload types:
        - python_beacon: Python script with injected constants (file output)
        - bash_oneliner: bash /dev/tcp one-liner (string output)
        - powershell_oneliner: PowerShell TCP reverse shell (UTF-16LE base64)
        - msfvenom_stager: msfvenom-generated stager (file output, requires msfvenom)

    Returns BuildResult dict. For file outputs, output_path is on disk
    (downloadable via GET /api/c2/payloads/{payload_id}/download).
    For oneliners, content field contains the one-liner string.

    Authentication: pass Bearer token in Authorization header.
    """
    from app.c2 import PayloadBuilder, PayloadBuilderInput, build_payload

    inp = PayloadBuilderInput(
        listener_id=req.listener_id,
        listener_type=req.listener_type,
        callback_host=req.callback_host,
        callback_port=req.callback_port,
        os=req.os,
        arch=req.arch,
        sleep_seconds=req.sleep_seconds,
        jitter_percent=req.jitter_percent,
        encryption_key_b64=req.encryption_key_b64,
        implant_token_b64=req.implant_token_b64,
        output_name=req.output_name,
    )

    result = await build_payload(inp, payload_type=req.payload_type)
    return result.to_dict()


@router.post("/payloads/all")
async def build_all_payloads_endpoint(
    req: BuildPayloadRequest = Body(...),
) -> dict[str, Any]:
    """Generate all compatible payloads for a listener (W15-S6 NEW).

    Generates multiple payload types in parallel:
        - For TCP listener: python TCP oneliner + bash oneliner + powershell oneliner (if Windows) + msfvenom
        - For HTTP/HTTPS/WS: python beacon script + curl_beacon + msfvenom

    Returns a list of BuildResult dicts.
    """
    from app.c2 import PayloadBuilder, PayloadBuilderInput

    inp = PayloadBuilderInput(
        listener_id=req.listener_id,
        listener_type=req.listener_type,
        callback_host=req.callback_host,
        callback_port=req.callback_port,
        os=req.os,
        arch=req.arch,
        sleep_seconds=req.sleep_seconds,
        jitter_percent=req.jitter_percent,
        encryption_key_b64=req.encryption_key_b64,
        implant_token_b64=req.implant_token_b64,
        output_name=req.output_name,
    )

    builder = PayloadBuilder()
    results = await builder.build_all(inp)
    return {
        "listener_id": req.listener_id,
        "total": len(results),
        "payloads": [r.to_dict() for r in results],
    }


@router.get("/payloads/types")
async def list_payload_types() -> dict[str, Any]:
    """List all supported payload types + their oneliner kinds (W15-S6 NEW)."""
    from app.c2 import OnelinerKind
    return {
        "payload_types": [
            {"name": "python_beacon", "description": "Python script with injected constants (file)"},
            {"name": "bash_oneliner", "description": "bash /dev/tcp reverse shell (string)"},
            {"name": "powershell_oneliner", "description": "PowerShell TCP reverse shell (UTF-16LE base64)"},
            {"name": "msfvenom_stager", "description": "msfvenom-generated meterpreter stager (file)"},
        ],
        "oneliner_kinds": OnelinerKind.all(),
    }


# ---------------------------------------------------------------------------
# C2 HITL endpoints (W15-S6)
# ---------------------------------------------------------------------------

@router.get("/hitl/pending/{scan_id}")
async def list_pending_c2_hitl(
    scan_id: str,
    session: AsyncSession = Depends(get_async_session),
) -> dict[str, Any]:
    """List pending HITL approvals for C2 tasks in a scan (W15-S6 NEW).

    Returns HITLApproval rows with tool_name starting with "c2_" + status=pending.
    Used by the C2 dashboard HITL modal to display approvals awaiting user action.
    """
    from app.db.models.hitl import HITLApproval
    from sqlalchemy import select

    stmt = (
        select(HITLApproval)
        .where(HITLApproval.scan_id == scan_id)
        .where(HITLApproval.tool_name.like("c2_%"))
        .where(HITLApproval.status == "pending")
        .order_by(HITLApproval.created_at.asc())
    )
    result = await session.execute(stmt)
    approvals = list(result.scalars().all())

    return {
        "scan_id": scan_id,
        "total": len(approvals),
        "approvals": [
            {
                "id": str(a.id),
                "tool_name": a.tool_name,
                "target": a.target,
                "args": a.args_json,
                "predicted_impact": a.predicted_impact,
                "agent_reasoning": a.agent_reasoning,
                "status": a.status,
                "created_at": a.created_at.isoformat() if a.created_at else None,
                "expires_at": a.expires_at.isoformat() if a.expires_at else None,
            }
            for a in approvals
        ],
    }


@router.post("/hitl/{approval_id}/approve")
async def approve_c2_hitl(
    approval_id: str,
    comment: str = "",
    session: AsyncSession = Depends(get_async_session),
) -> dict[str, Any]:
    """Approve a pending C2 HITL approval (W15-S6 NEW).

    Marks the HITLApproval as approved. The C2Manager.enqueue_task()
    will then proceed to queue the task.
    """
    from app.db.models.hitl import HITLApproval
    approval = await session.get(HITLApproval, uuid.UUID(approval_id))
    if approval is None:
        raise HTTPException(
            status_code=404,
            detail=f"HITL approval {approval_id!r} not found",
        )

    if approval.status != "pending":
        raise HTTPException(
            status_code=409,
            detail=f"Approval already decided: {approval.status}",
        )

    approval.status = "approved"
    approval.user_decision = "approve"
    from datetime import datetime, UTC
    approval.decided_at = datetime.now(UTC)
    await session.commit()

    return {
        "approval_id": approval_id,
        "status": approval.status,
        "decision": approval.user_decision,
        "decided_at": approval.decided_at.isoformat() if approval.decided_at else None,
    }


@router.post("/hitl/{approval_id}/abort")
async def abort_c2_hitl(
    approval_id: str,
    comment: str = "",
    session: AsyncSession = Depends(get_async_session),
) -> dict[str, Any]:
    """Abort (reject) a pending C2 HITL approval (W15-S6 NEW).

    Marks the HITLApproval as aborted. The C2Manager.enqueue_task() will
    raise PermissionError + the task will be saved with status=hitl_denied.
    """
    from app.db.models.hitl import HITLApproval
    approval = await session.get(HITLApproval, uuid.UUID(approval_id))
    if approval is None:
        raise HTTPException(
            status_code=404,
            detail=f"HITL approval {approval_id!r} not found",
        )

    if approval.status != "pending":
        raise HTTPException(
            status_code=409,
            detail=f"Approval already decided: {approval.status}",
        )

    approval.status = "aborted"
    approval.user_decision = "abort"
    from datetime import datetime, UTC
    approval.decided_at = datetime.now(UTC)
    await session.commit()

    return {
        "approval_id": approval_id,
        "status": approval.status,
        "decision": approval.user_decision,
        "decided_at": approval.decided_at.isoformat() if approval.decided_at else None,
    }


@router.get("/sessions/{scan_id}/{session_id}/terminal")
async def get_terminal_session_info(
    scan_id: str,
    session_id: str,
    session: AsyncSession = Depends(get_async_session),
) -> dict[str, Any]:
    """Get terminal session info for xterm.js (W15-S6 NEW).

    Returns session metadata needed to initialize an xterm.js terminal
    in the C2 dashboard. W19 will wire the actual WebSocket terminal
    stream; W15 just returns the session info.
    """
    c2_session = await session.get(C2Session, uuid.UUID(session_id))
    if c2_session is None or c2_session.scan_id != scan_id:
        raise HTTPException(
            status_code=404,
            detail=f"Session {session_id!r} not found for scan {scan_id!r}",
        )

    return {
        "session_id": str(c2_session.id),
        "scan_id": c2_session.scan_id,
        "beacon_type": c2_session.beacon_type,
        "hostname": c2_session.hostname,
        "username": c2_session.username,
        "os": c2_session.os,
        "arch": c2_session.arch,
        "status": c2_session.status,
        "remote_address": c2_session.remote_address,
        "listener_type": c2_session.listener_type,
        "terminal_ws_url": f"/api/c2/sessions/{scan_id}/{session_id}/terminal/ws",
        "note": "Terminal WebSocket stream will be wired in W19 (frontend dashboard).",
    }
