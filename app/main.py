"""
VAPT-AI v3.2 — FastAPI application factory.

Entry point: `uvicorn app.main:app --reload --port 8000`

Endpoints (W1):
    GET  /health              — health check (no auth)
    GET  /docs                — Swagger UI (no auth)
    GET  /openapi.json        — OpenAPI schema (no auth)
    POST /api/auth/login      — login (returns access + refresh tokens)
    POST /api/auth/refresh    — rotate refresh token
    POST /api/auth/logout     — revoke refresh token
    GET  /api/auth/me         — current user info (requires access token)

Startup:
    - Bootstraps admin user (admin@vapt-ai.local) if not exists
    - Initializes async DB engine + session factory

Shutdown:
    - Disposes async DB engine
"""
from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

import structlog
from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field
from fastapi import Body
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.manager import (
    AuthError,
    AuthManager,
    InvalidCredentialsError,
    AccountLockedError,
    InvalidTokenError,
    TokenRevokedError,
    decode_access_token,
)
from app.core.config import settings
from app.db.session import async_session, dispose_async_engine, get_async_session

# ---------- Logging setup ----------

structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer() if settings.environment == "dev" else structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(
        getattr(logging, settings.log_level, logging.INFO)
    ),
    cache_logger_on_first_use=True,
)
logger = structlog.get_logger()


# ---------- Pydantic schemas (request/response) ----------

class LoginRequest(BaseModel):
    # Use str (not EmailStr) — single-user internal tool, admin@vapt-ai.local
    # uses .local TLD which email-validator rejects as reserved.
    email: str = Field(..., min_length=3, max_length=255)
    password: str = Field(..., min_length=1, max_length=255)


class RefreshRequest(BaseModel):
    refresh_token: str = Field(..., min_length=1)


class LogoutRequest(BaseModel):
    refresh_token: str = Field(..., min_length=1)


class UserResponse(BaseModel):
    id: str
    email: str
    role: str
    display_name: str | None = None


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int
    user: UserResponse


class StartScanRequest(BaseModel):
    target: str = Field(..., description="Target URL or IP (e.g. http://example.com)")
    user_prompt: str = Field("", description="Natural-language prompt (e.g. 'Scan for SQL injection')")


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str
    environment: str
    db_connected: bool


# ---------- Auth dependency ----------

security = HTTPBearer(auto_error=False)


async def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(security),
    session: AsyncSession = Depends(get_async_session),
) -> UserResponse:
    """FastAPI dependency: extract + verify JWT, return current user.

    Usage:
        @router.get("/protected")
        async def protected(user: UserResponse = Depends(get_current_user)):
            return {"user": user.email}
    """
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid Authorization header",
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        payload = decode_access_token(credentials.credentials)
    except InvalidTokenError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e),
            headers={"WWW-Authenticate": "Bearer"},
        ) from e

    # Fetch user from DB (verify still active)
    import uuid
    user_id = uuid.UUID(payload["sub"])
    auth = AuthManager(session)
    user = await auth.get_user_by_id(user_id)

    if user is None or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or disabled",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return UserResponse(
        id=str(user.id),
        email=user.email,
        role=user.role,
        display_name=user.display_name,
    )


# ---------- Lifespan (startup + shutdown) ----------

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """App lifespan: startup + shutdown hooks."""
    # ----- Startup -----
    logger.info("VAPT-AI starting up",
                version=settings.app_version,
                environment=settings.environment)

    # Bootstrap admin user
    try:
        async with async_session() as session:
            auth = AuthManager(session)
            user = await auth.bootstrap_admin_user()
            await session.commit()
            logger.info("Admin user ready",
                        email=user.email,
                        user_id=str(user.id))
    except Exception as e:
        logger.error("Failed to bootstrap admin user", error=str(e))
        # Don't crash — allow app to start (user can retry login later)

    yield

    # ----- Shutdown -----
    logger.info("VAPT-AI shutting down")
    await dispose_async_engine()


# ---------- App factory ----------

def create_app() -> FastAPI:
    """Create + configure the FastAPI application."""
    app = FastAPI(
        title="VAPT-AI",
        description="Vulnerability Assessment & Penetration Testing AI System — v3.2",
        version=settings.app_version,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # ---------- Middleware ----------
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[settings.app_url, "http://localhost:5173", "http://localhost:8000"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ---------- Health endpoint ----------
    @app.get("/health", response_model=HealthResponse, tags=["system"])
    async def health() -> HealthResponse:
        """Health check — no auth required."""
        db_connected = True
        try:
            async with async_session() as session:
                await session.execute(__import__("sqlalchemy").text("SELECT 1"))
        except Exception:
            db_connected = False
        return HealthResponse(
            version=settings.app_version,
            environment=settings.environment,
            db_connected=db_connected,
        )

    # ---------- Auth routes ----------
    from fastapi import APIRouter
    auth_router = APIRouter(prefix="/api/auth", tags=["auth"])

    @auth_router.post("/login", response_model=TokenResponse)
    async def login(
        req: LoginRequest,
        request: Request,
        session: AsyncSession = Depends(get_async_session),
    ) -> TokenResponse:
        """Login with email + password. Returns access + refresh tokens."""
        ip = request.client.host if request.client else None
        auth = AuthManager(session)
        try:
            result = await auth.login(req.email, req.password, ip=ip)
        except AccountLockedError as e:
            raise HTTPException(status_code=e.status_code, detail=e.message) from e
        except InvalidCredentialsError as e:
            raise HTTPException(status_code=e.status_code, detail=e.message) from e
        return TokenResponse(**result)

    @auth_router.post("/refresh", response_model=TokenResponse)
    async def refresh(
        req: RefreshRequest,
        request: Request,
        session: AsyncSession = Depends(get_async_session),
    ) -> TokenResponse:
        """Rotate refresh token. Returns new access + refresh tokens."""
        ip = request.client.host if request.client else None
        auth = AuthManager(session)
        try:
            result = await auth.refresh(req.refresh_token, ip=ip)
        except (InvalidTokenError, TokenRevokedError) as e:
            raise HTTPException(status_code=e.status_code, detail=e.message) from e
        return TokenResponse(**result)

    @auth_router.post("/logout")
    async def logout(
        req: LogoutRequest,
        session: AsyncSession = Depends(get_async_session),
    ) -> dict[str, str]:
        """Revoke a refresh token (logout)."""
        auth = AuthManager(session)
        await auth.logout(req.refresh_token)
        return {"status": "ok", "message": "Logged out"}

    @auth_router.get("/me", response_model=UserResponse)
    async def me(
        user: UserResponse = Depends(get_current_user),
    ) -> UserResponse:
        """Get current user info. Requires valid access token."""
        return user

    app.include_router(auth_router)

    # ---------- Scan endpoints (W3-A) ----------
    from app.agents.agent import run_scan as agent_run_scan, get_available_tools
    from fastapi import APIRouter
    scan_router = APIRouter(prefix="/api/scans", tags=["scans"])

    class StartScanRequest(BaseModel):
        target: str = Field(..., description="Target URL or IP (e.g. http://example.com)")
        user_prompt: str = Field("", description="Natural-language prompt (e.g. 'Scan for SQL injection')")

    @scan_router.post("/start")
    async def start_scan(
        req: StartScanRequest = Body(...),
        session: AsyncSession = Depends(get_async_session),
        user: UserResponse = Depends(get_current_user),
    ) -> dict[str, Any]:
        """Start a new scan. Returns scan_id + result.

        W3-A: runs synchronously (blocking). W3-B will make it async via Celery + SSE.
        """
        import uuid as uuid_mod
        scan_id = f"scan_{uuid_mod.uuid4().hex[:12]}"

        # W4-D fix: Create Scan row in DB (so FK constraints work)
        from app.db.models.scan import Scan
        import uuid as uuid_mod2
        scan = Scan(
            id=scan_id,
            user_id=uuid_mod2.UUID(user.id),
            target=req.target,
            target_type="web_app" if req.target.startswith("http") else "single_host",
            agent_mode="supervisor",
            user_prompt=req.user_prompt,
            status="running",
            progress=0,
            scope_json={"hosts": [req.target]},
        )
        session.add(scan)
        await session.commit()

        # Run agent
        result = await agent_run_scan(
            target=req.target,
            user_prompt=req.user_prompt,
            scan_id=scan_id,
        )

        # Update scan status
        scan.status = result.status
        scan.progress = 100
        import datetime
        scan.completed_at = datetime.datetime.now(datetime.UTC)
        await session.commit()

        return {
            "scan_id": result.scan_id,
            "target": result.target,
            "user_prompt": result.user_prompt,
            "status": result.status,
            "decisions_count": len(result.decisions),
            "total_tokens": result.total_tokens,
            "duration_seconds": result.duration_seconds,
            "findings": result.findings,
            "error": result.error,
            "decisions": [
                {
                    "turn": d.turn,
                    "thought": d.thought,
                    "tool_name": d.tool_name,
                    "tool_args": d.tool_args,
                    "observation": d.observation[:500] if d.observation else "",
                    "tokens_used": d.tokens_used,
                    "timestamp": d.timestamp.isoformat(),
                }
                for d in result.decisions
            ],
        }

    @scan_router.get("/tools")
    async def list_scan_tools(
        user: UserResponse = Depends(get_current_user),
    ) -> dict[str, Any]:
        """List all tools available to the scan agent."""
        tools = get_available_tools()
        return {
            "tools_count": len(tools),
            "tools": tools,
        }

    app.include_router(scan_router)

    # ---------- SSE Event Stream (W3-B) ----------
    from app.pentest.events import event_bus
    from fastapi.responses import StreamingResponse

    @app.get("/api/scans/{scan_id}/events", tags=["scans"])
    async def scan_events(
        scan_id: str,
        user: UserResponse = Depends(get_current_user),
    ) -> StreamingResponse:
        """SSE event stream for a scan — real-time progress.

        Connect with EventSource (browser) or curl -N (CLI):
            curl -N -H "Authorization: Bearer <token>" \\
                http://localhost:8000/api/scans/scan_abc123/events

        Events:
            scan_started, scan_progress, finding_detected,
            hitl_approval_required, scan_complete, scan_error, heartbeat
        """
        async def event_generator():
            async for event in event_bus.subscribe(scan_id):
                yield f"data: {json.dumps(event)}\n\n"

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",  # disable nginx buffering
            },
        )

    # ---------- Blackboard endpoints (W3-B) ----------
    from app.pentest.blackboard import Blackboard

    @app.get("/api/scans/{scan_id}/blackboard", tags=["scans"])
    async def get_blackboard(
        scan_id: str,
        session: AsyncSession = Depends(get_async_session),
        user: UserResponse = Depends(get_current_user),
    ) -> dict[str, Any]:
        """Get blackboard summary for a scan — all PentestFact entries."""
        bb = Blackboard(session)
        return await bb.get_scan_summary(scan_id)

    @app.get("/api/scans/{scan_id}/facts", tags=["scans"])
    async def get_facts(
        scan_id: str,
        fact_type: str | None = None,
        session: AsyncSession = Depends(get_async_session),
        user: UserResponse = Depends(get_current_user),
    ) -> dict[str, Any]:
        """Get all facts for a scan (optionally filtered by type)."""
        bb = Blackboard(session)
        if fact_type:
            facts = await bb.get_facts_by_type(scan_id, fact_type)
        else:
            facts = await bb.get_facts(scan_id)
        return {
            "scan_id": scan_id,
            "facts_count": len(facts),
            "facts": [
                {
                    "id": str(f.id),
                    "fact_type": f.fact_type,
                    "fact_key": f.fact_key,
                    "fact_value": f.fact_value,
                    "source_agent": f.source_agent,
                    "confidence": f.confidence,
                    "created_at": f.created_at.isoformat(),
                }
                for f in facts
            ],
        }

    # ---------- HITL endpoints (W3-C) ----------
    from app.hitl.manager import HITLManager

    @app.get("/api/hitl/pending/{scan_id}", tags=["hitl"])
    async def get_pending_approvals(
        scan_id: str,
        session: AsyncSession = Depends(get_async_session),
        user: UserResponse = Depends(get_current_user),
    ) -> dict[str, Any]:
        """Get all pending HITL approvals for a scan."""
        mgr = HITLManager(session)
        approvals = await mgr.get_pending_approvals(scan_id)
        return {
            "scan_id": scan_id,
            "pending_count": len(approvals),
            "approvals": [
                {
                    "id": str(a.id),
                    "tool_name": a.tool_name,
                    "target": a.target,
                    "args": a.args_json,
                    "predicted_impact": a.predicted_impact,
                    "agent_reasoning": a.agent_reasoning,
                    "kg_confidence": a.kg_confidence,
                    "status": a.status,
                    "expires_at": a.expires_at.isoformat(),
                    "created_at": a.created_at.isoformat(),
                }
                for a in approvals
            ],
        }

    @app.post("/api/hitl/{approval_id}/approve", tags=["hitl"])
    async def approve_hitl(
        approval_id: str,
        time_limit_seconds: int | None = None,
        session: AsyncSession = Depends(get_async_session),
        user: UserResponse = Depends(get_current_user),
    ) -> dict[str, Any]:
        """Approve a HITL request."""
        import uuid as uuid_mod
        mgr = HITLManager(session)
        approval = await mgr.approve(
            uuid_mod.UUID(approval_id),
            user_id=uuid_mod.UUID(user.id),
            time_limit_seconds=time_limit_seconds,
        )
        if approval is None:
            raise HTTPException(status_code=404, detail="Approval not found or already decided")
        return {"status": "ok", "approval_status": approval.status}

    @app.post("/api/hitl/{approval_id}/abort", tags=["hitl"])
    async def abort_hitl(
        approval_id: str,
        session: AsyncSession = Depends(get_async_session),
        user: UserResponse = Depends(get_current_user),
    ) -> dict[str, Any]:
        """Abort (reject) a HITL request."""
        import uuid as uuid_mod
        mgr = HITLManager(session)
        approval = await mgr.abort(
            uuid_mod.UUID(approval_id),
            user_id=uuid_mod.UUID(user.id),
        )
        if approval is None:
            raise HTTPException(status_code=404, detail="Approval not found or already decided")
        return {"status": "ok", "approval_status": approval.status}

    # ---------- Evidence custody endpoints (W4-A) ----------
    from app.evidence.custody import CustodyVerifier

    @app.get("/api/evidence/{evidence_id}/verify", tags=["evidence"])
    async def verify_evidence_custody(
        evidence_id: str,
        session: AsyncSession = Depends(get_async_session),
        user: UserResponse = Depends(get_current_user),
    ) -> dict[str, Any]:
        """Verify custody seal of a single evidence row (D23 tamper detection)."""
        import uuid as uuid_mod
        verifier = CustodyVerifier(session)
        return await verifier.verify_evidence(uuid_mod.UUID(evidence_id))

    @app.get("/api/findings/{finding_id}/custody-chain", tags=["evidence"])
    async def verify_finding_custody_chain(
        finding_id: str,
        session: AsyncSession = Depends(get_async_session),
        user: UserResponse = Depends(get_current_user),
    ) -> dict[str, Any]:
        """Verify full custody seal chain for a finding's evidence."""
        import uuid as uuid_mod
        verifier = CustodyVerifier(session)
        return await verifier.verify_finding_chain(uuid_mod.UUID(finding_id))

    @app.get("/api/scans/{scan_id}/custody-verify", tags=["evidence"])
    async def verify_scan_custody(
        scan_id: str,
        session: AsyncSession = Depends(get_async_session),
        user: UserResponse = Depends(get_current_user),
    ) -> dict[str, Any]:
        """Verify custody chains for ALL findings in a scan."""
        verifier = CustodyVerifier(session)
        return await verifier.verify_scan_chain(scan_id)

    # ---------- Consent form endpoints (W4-C) ----------
    from app.consent.manager import ConsentManager

    class CreateConsentRequest(BaseModel):
        scan_id: str = Field(..., description="Scan ID this consent is for")
        asserted_owner: str = Field(..., description="Name/email/org of target owner")
        declared_scope: dict[str, Any] = Field(..., description="Scope: {hosts: [], cidrs: [], ports: []}")
        verification_method: str = Field(..., description="dns_txt / http_meta / file_upload / owned_vps / htb_machine / thm_room")
        asserted_owner_role: str = Field("owner", description="owner or authorized_rep")

    @app.post("/api/consent/create", tags=["consent"])
    async def create_consent(
        req: CreateConsentRequest = Body(...),
        session: AsyncSession = Depends(get_async_session),
        user: UserResponse = Depends(get_current_user),
    ) -> dict[str, Any]:
        """Create a consent form (D19 — mandatory before any active scan)."""
        mgr = ConsentManager(session)
        consent = await mgr.create_consent(
            scan_id=req.scan_id,
            asserted_owner=req.asserted_owner,
            declared_scope=req.declared_scope,
            verification_method=req.verification_method,
            asserted_owner_role=req.asserted_owner_role,
        )
        await session.commit()
        return {
            "consent_id": str(consent.id),
            "scan_id": consent.scan_id,
            "verification_method": consent.verification_method,
            "verification_token": consent.verification_token,
            "verified": consent.verified,
            "tos_accepted": consent.tos_accepted,
            "message": "Consent created. Accept ToS then verify ownership.",
        }

    @app.post("/api/consent/{consent_id}/accept-tos", tags=["consent"])
    async def accept_tos(
        consent_id: str,
        session: AsyncSession = Depends(get_async_session),
        user: UserResponse = Depends(get_current_user),
    ) -> dict[str, Any]:
        """Accept Terms of Service for a consent form."""
        import uuid as uuid_mod
        mgr = ConsentManager(session)
        consent = await mgr.accept_tos(uuid_mod.UUID(consent_id))
        if consent is None:
            raise HTTPException(status_code=404, detail="Consent not found")
        await session.commit()
        return {"status": "ok", "tos_accepted": True}

    @app.post("/api/consent/{consent_id}/verify", tags=["consent"])
    async def verify_consent(
        consent_id: str,
        session: AsyncSession = Depends(get_async_session),
        user: UserResponse = Depends(get_current_user),
    ) -> dict[str, Any]:
        """Verify consent ownership (DNS TXT / HTTP meta / file upload)."""
        import uuid as uuid_mod
        mgr = ConsentManager(session)
        result = await mgr.verify_consent(uuid_mod.UUID(consent_id))
        await session.commit()
        return result

    @app.get("/api/consent/scan/{scan_id}", tags=["consent"])
    async def get_consent_for_scan(
        scan_id: str,
        session: AsyncSession = Depends(get_async_session),
        user: UserResponse = Depends(get_current_user),
    ) -> dict[str, Any]:
        """Get consent form for a scan + authorization status."""
        mgr = ConsentManager(session)
        consent = await mgr.get_consent_for_scan(scan_id)
        authorized = await mgr.is_scan_authorized(scan_id)
        if consent is None:
            return {"scan_id": scan_id, "authorized": False, "consent": None}
        return {
            "scan_id": scan_id,
            "authorized": authorized,
            "consent": {
                "id": str(consent.id),
                "asserted_owner": consent.asserted_owner,
                "verification_method": consent.verification_method,
                "verified": consent.verified,
                "tos_accepted": consent.tos_accepted,
                "declared_scope": consent.declared_scope_json,
                "created_at": consent.created_at.isoformat(),
                "verified_at": consent.verified_at.isoformat() if consent.verified_at else None,
            },
        }

    # ---------- Audit log endpoints (W4-B) ----------
    from app.audit.logger import AuditLogger, sanitize_audit_data

    @app.get("/api/audit/entries", tags=["audit"])
    async def get_audit_entries(
        scan_id: str | None = None,
        actor_type: str | None = None,
        action: str | None = None,
        limit: int = 100,
        offset: int = 0,
        session: AsyncSession = Depends(get_async_session),
        user: UserResponse = Depends(get_current_user),
    ) -> dict[str, Any]:
        """Query audit log entries with optional filters (D23)."""
        audit = AuditLogger(session)
        entries = await audit.get_entries(
            scan_id=scan_id,
            actor_type=actor_type,
            action=action,
            limit=min(limit, 500),  # cap at 500
            offset=offset,
        )
        return {
            "count": len(entries),
            "entries": [
                {
                    "id": str(e.id),
                    "actor_type": e.actor_type,
                    "actor_id": e.actor_id,
                    "action": e.action,
                    "target_table": e.target_table,
                    "target_id": e.target_id,
                    "before": e.before_json,
                    "after": e.after_json,
                    "ip_address": e.ip_address,
                    "scan_id": e.scan_id,
                    "created_at": e.created_at.isoformat(),
                }
                for e in entries
            ],
        }

    @app.get("/api/audit/verify-chain", tags=["audit"])
    async def verify_audit_chain(
        session: AsyncSession = Depends(get_async_session),
        user: UserResponse = Depends(get_current_user),
    ) -> dict[str, Any]:
        """Verify HMAC seal chain of audit log (D23 tamper detection)."""
        audit = AuditLogger(session)
        return await audit.verify_chain()

    @app.get("/api/audit/stats", tags=["audit"])
    async def audit_stats(
        session: AsyncSession = Depends(get_async_session),
        user: UserResponse = Depends(get_current_user),
    ) -> dict[str, Any]:
        """Get audit log statistics."""
        audit = AuditLogger(session)
        count = await audit.count_entries()
        return {
            "total_entries": count,
            "retention_years": 7,
            "append_only": True,
            "hmac_sealed": True,
        }
    # ---------- Sanitizer endpoint (W3-C) ----------
    from app.core.sanitizer import sanitize_prompt_input, get_injection_patterns

    @app.get("/api/sanitizer/patterns", tags=["system"])
    async def get_sanitizer_patterns(
        user: UserResponse = Depends(get_current_user),
    ) -> dict[str, Any]:
        """List all prompt-injection patterns the sanitizer detects."""
        return {
            "patterns_count": len(get_injection_patterns()),
            "patterns": get_injection_patterns(),
        }

    # ---------- MCP endpoints (W1-G + W2-A) ----------
    from app.mcp.server import list_tools_sync, mcp_server
    from app.tools.loader import load_all_tools, list_tools as list_tool_defs, register_tools_with_mcp

    # W2-A: Load all 10 tool YAMLs + register as MCP tools
    load_all_tools()
    register_tools_with_mcp(mcp_server)

    # W3-D: Register YAML tool names with scope guard (so they're allowed)
    from app.sandbox.scope_guard import register_yaml_tools
    from app.tools.loader import list_tools as _list_tools
    register_yaml_tools({t.name for t in _list_tools()})

    # W6-B: Register 8 Metasploit MCP tools + include router
    from app.exploit.metasploit_tools import register_metasploit_mcp_tools, create_metasploit_router
    register_metasploit_mcp_tools(mcp_server)
    app.include_router(create_metasploit_router(get_current_user, UserResponse))

    @app.get("/mcp/tools/list", tags=["mcp"])
    async def mcp_tools_list() -> dict[str, Any]:
        """List registered MCP tools. W1-G acceptance: responds with tools/list."""
        tools = list_tools_sync()
        return {
            "server": "vapt-ai",
            "version": "3.2.1",
            "tools": tools,
            "tools_count": len(tools),
        }

    @app.get("/mcp/health", tags=["mcp"])
    async def mcp_health() -> dict[str, str]:
        """MCP server health check."""
        return {"status": "ok", "transport": "stdio+http"}

    @app.get("/mcp/tools/definitions", tags=["mcp"])
    async def mcp_tools_definitions() -> dict[str, Any]:
        """List all tool definitions with WSTG IDs + MITRE ATT&CK mapping (W2-A)."""
        tools = list_tool_defs()
        return {
            "tools_count": len(tools),
            "tools": [
                {
                    "name": t.name,
                    "command": t.command,
                    "category": t.category,
                    "short_description": t.short_description,
                    "wstg_ids": t.wstg_ids,
                    "mitre_attack": t.mitre_attack,
                    "safety_class": t.safety_class,
                    "parameters": [p.name for p in t.parameters],
                    "timeout": t.timeout,
                }
                for t in tools
            ],
        }

    # ---------- Root endpoint ----------
    @app.get("/", tags=["system"])
    async def root() -> dict[str, str]:
        """Root — basic info."""
        return {
            "name": "VAPT-AI",
            "version": settings.app_version,
            "docs": "/docs",
            "health": "/health",
        }

    logger.info("FastAPI app created", routes=[
        "/health", "/", "/docs", "/redoc", "/openapi.json",
        "/api/auth/login", "/api/auth/refresh", "/api/auth/logout", "/api/auth/me",
    ])
    return app


# ---------- Module-level app instance (for uvicorn) ----------

app = create_app()
