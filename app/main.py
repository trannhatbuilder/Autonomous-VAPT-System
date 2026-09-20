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
from sqlalchemy import select

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


class LLMSettingsRequest(BaseModel):
    provider: str
    base_url: str
    model: str
    max_total_tokens: int = 4096
    max_completion_tokens: int = 2048
    temperature: float = 0.7
    api_key: str
    hitl_audit_provider: str
    hitl_audit_base_url: str
    hitl_audit_model: str
    hitl_audit_api_key: str


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str
    environment: str
    db_connected: bool


class CreateConversationRequest(BaseModel):
    title: str | None = "New conversation"


class SendMessageRequest(BaseModel):
    content: str
    scan_id: str | None = None
    metadata: dict[str, Any] | None = None


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

    # ----- W8-E: Seed methodology catalogs (WSTG + ATT&CK) -----
    try:
        from app.methodology.seeder import seed_all_catalogs
        async with async_session() as session:
            result = await seed_all_catalogs(session)
            await session.commit()
            logger.info(
                "Methodology catalogs seeded",
                wstg_inserted=result["wstg"]["inserted"],
                wstg_updated=result["wstg"]["updated"],
                attack_inserted=result["attack"]["inserted"],
                attack_updated=result["attack"]["updated"],
            )
    except Exception as e:
        logger.error("Failed to seed methodology catalogs", error=str(e))
        # Don't crash — app can still run without catalogs (tools won't have WSTG/ATT&CK tags)

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

        # Run agent (W3-A: synchronous — W3-B will use Celery)
        result = await agent_run_scan(
            target=req.target,
            user_prompt=req.user_prompt,
            scan_id=scan_id,
        )

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
        token: str | None = None,
        session: AsyncSession = Depends(get_async_session),
    ) -> StreamingResponse:
        """SSE event stream for a scan — real-time progress.

        Authentication: Bearer token via query param `?token=<access_token>`.
        (EventSource browser API does not support custom headers, so we
        accept the token as a query parameter instead.)

        Connect with EventSource (browser):
            const es = new EventSource(`/api/scans/${scanId}/events?token=${accessToken}`);

        Or with curl (CLI):
            curl -N "http://localhost:8000/api/scans/scan_abc123/events?token=eyJ..."

        Events:
            scan_started, scan_progress, finding_detected,
            hitl_approval_required, hitl_decision_made,
            scan_complete, scan_error, heartbeat
        """
        # Verify token (query param for EventSource compatibility)
        from app.auth.manager import decode_access_token, InvalidTokenError
        if not token:
            raise HTTPException(status_code=401, detail="Token required (?token=<access_token>)")
        try:
            payload = decode_access_token(token)
            if payload.get("type") != "access":
                raise HTTPException(status_code=401, detail="Invalid token type")
        except InvalidTokenError as e:
            raise HTTPException(status_code=401, detail=f"Invalid token: {e}") from e

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

    # ---------- Conversation + Chat endpoints (W7-D) ----------
    from app.conversation.manager import ConversationManager
    from app.db.models.conversation import Conversation, ChatMessage
    from pydantic import BaseModel as PydanticBaseModel

    conversation_router = APIRouter(prefix="/api/conversations", tags=["conversations"])

    @conversation_router.post("")
    async def create_conversation(
        req: CreateConversationRequest = Body(...),
        session: AsyncSession = Depends(get_async_session),
        user: UserResponse = Depends(get_current_user),
    ) -> dict[str, Any]:
        """Create a new chat conversation."""
        import uuid as uuid_mod
        mgr = ConversationManager(session)
        conv = await mgr.create_conversation(
            user_id=uuid_mod.UUID(user.id),
            title=req.title or "New conversation",
        )
        await session.commit()
        return {
            "id": str(conv.id),
            "title": conv.title,
            "message_count": conv.message_count,
            "created_at": conv.created_at.isoformat() if conv.created_at else None,
        }

    @conversation_router.get("")
    async def list_conversations(
        include_archived: bool = False,
        limit: int = 200,
        offset: int = 0,
        session: AsyncSession = Depends(get_async_session),
        user: UserResponse = Depends(get_current_user),
    ) -> dict[str, Any]:
        """List all conversations for the current user (newest first)."""
        import uuid as uuid_mod
        mgr = ConversationManager(session)
        convs = await mgr.list_conversations(
            user_id=uuid_mod.UUID(user.id),
            include_archived=include_archived,
            limit=min(limit, 200),
            offset=offset,
        )
        return {
            "count": len(convs),
            "conversations": [
                {
                    "id": str(c.id),
                    "title": c.title,
                    "summary": c.summary,
                    "is_archived": c.is_archived,
                    "message_count": c.message_count,
                    "last_message_at": c.last_message_at.isoformat() if c.last_message_at else None,
                    "created_at": c.created_at.isoformat() if c.created_at else None,
                }
                for c in convs
            ],
        }

    @conversation_router.get("/{conversation_id}")
    async def get_conversation(
        conversation_id: str,
        session: AsyncSession = Depends(get_async_session),
        user: UserResponse = Depends(get_current_user),
    ) -> dict[str, Any]:
        """Get a conversation + all its messages."""
        import uuid as uuid_mod
        mgr = ConversationManager(session)
        result = await mgr.get_conversation_with_messages(
            uuid_mod.UUID(conversation_id),
            user_id=uuid_mod.UUID(user.id),
        )
        if result is None:
            raise HTTPException(status_code=404, detail="Conversation not found")
        return result

    @conversation_router.post("/{conversation_id}/messages")
    async def send_message(
        conversation_id: str,
        req: SendMessageRequest = Body(...),
        session: AsyncSession = Depends(get_async_session),
        user: UserResponse = Depends(get_current_user),
    ) -> dict[str, Any]:
        """Add a message to a conversation.

        Typically the frontend sends a user message here, then triggers a
        scan separately (POST /api/scans/start). The scan_id can be linked
        back to this message via the scan_id field.
        """
        import uuid as uuid_mod
        mgr = ConversationManager(session)

        # Verify conversation exists + belongs to user
        conv = await mgr.get_conversation(
            uuid_mod.UUID(conversation_id),
            user_id=uuid_mod.UUID(user.id),
        )
        if conv is None:
            raise HTTPException(status_code=404, detail="Conversation not found")

        msg = await mgr.add_message(
            conversation_id=uuid_mod.UUID(conversation_id),
            role="user",
            content=req.content,
            scan_id=req.scan_id,
            metadata=req.metadata,
        )
        await session.commit()
        return {
            "id": str(msg.id),
            "conversation_id": str(msg.conversation_id),
            "role": msg.role,
            "content": msg.content,
            "scan_id": msg.scan_id,
            "metadata": msg.metadata_json,
            "sequence": msg.sequence,
            "created_at": msg.created_at.isoformat() if msg.created_at else None,
        }

    @conversation_router.post("/{conversation_id}/assistant-message")
    async def add_assistant_message(
        conversation_id: str,
        req: SendMessageRequest = Body(...),
        session: AsyncSession = Depends(get_async_session),
        user: UserResponse = Depends(get_current_user),
    ) -> dict[str, Any]:
        """Add an assistant message to a conversation (after scan completes).

        The frontend calls this after receiving scan results to persist the
        AI's response in the chat history.
        """
        import uuid as uuid_mod
        mgr = ConversationManager(session)

        conv = await mgr.get_conversation(
            uuid_mod.UUID(conversation_id),
            user_id=uuid_mod.UUID(user.id),
        )
        if conv is None:
            raise HTTPException(status_code=404, detail="Conversation not found")

        msg = await mgr.add_message(
            conversation_id=uuid_mod.UUID(conversation_id),
            role="assistant",
            content=req.content,
            scan_id=req.scan_id,
            metadata=req.metadata,
        )
        await session.commit()
        return {
            "id": str(msg.id),
            "conversation_id": str(msg.conversation_id),
            "role": msg.role,
            "content": msg.content,
            "scan_id": msg.scan_id,
            "metadata": msg.metadata_json,
            "sequence": msg.sequence,
            "created_at": msg.created_at.isoformat() if msg.created_at else None,
        }

    @conversation_router.delete("/{conversation_id}")
    async def delete_conversation(
        conversation_id: str,
        session: AsyncSession = Depends(get_async_session),
        user: UserResponse = Depends(get_current_user),
    ) -> dict[str, Any]:
        """Delete a conversation + all its messages (CASCADE)."""
        import uuid as uuid_mod
        mgr = ConversationManager(session)
        deleted = await mgr.delete_conversation(uuid_mod.UUID(conversation_id))
        if not deleted:
            raise HTTPException(status_code=404, detail="Conversation not found")
        await session.commit()
        return {"status": "ok", "deleted": True}

    @conversation_router.patch("/{conversation_id}")
    async def update_conversation(
        conversation_id: str,
        req: CreateConversationRequest = Body(...),
        session: AsyncSession = Depends(get_async_session),
        user: UserResponse = Depends(get_current_user),
    ) -> dict[str, Any]:
        """Update conversation title."""
        import uuid as uuid_mod
        mgr = ConversationManager(session)
        conv = await mgr.update_conversation(
            uuid_mod.UUID(conversation_id),
            title=req.title,
        )
        if conv is None:
            raise HTTPException(status_code=404, detail="Conversation not found")
        await session.commit()
        return {
            "id": str(conv.id),
            "title": conv.title,
            "updated_at": conv.updated_at.isoformat() if conv.updated_at else None,
        }

    app.include_router(conversation_router)

    # ---------- W8-G: Methodology + CVE + Scope Violation endpoints ----------
    from app.routes.w8g_methodology import create_methodology_router
    app.include_router(create_methodology_router(get_current_user, UserResponse))

    # ---------- LLM Settings endpoints (W7-D-v2) ----------
    # Stores LLM provider config in vapt_users.settings JSONB column.
    # Frontend reads/writes via GET/POST /api/settings/llm.
    from app.db.models.user import User
    from pydantic import field_validator, ConfigDict

    settings_router = APIRouter(prefix="/api/settings", tags=["settings"])

    @settings_router.get("/llm")
    async def get_llm_settings(
        session: AsyncSession = Depends(get_async_session),
        user: UserResponse = Depends(get_current_user),
    ) -> dict[str, Any]:
        """Get current LLM + HITL audit agent config for the user."""
        import uuid as uuid_mod
        result = await session.execute(
            select(User).where(User.id == uuid_mod.UUID(user.id))
        )
        db_user = result.scalar_one_or_none()
        if db_user is None:
            raise HTTPException(status_code=404, detail="User not found")
        settings = db_user.settings or {}
        llm = settings.get("llm", {})
        # Never return the actual API key — return masked version
        masked = {**llm}
        if masked.get("api_key"):
            masked["api_key"] = "•" * 8 + masked["api_key"][-4:] if len(masked["api_key"]) > 4 else "****"
        if masked.get("hitl_audit_api_key"):
            masked["hitl_audit_api_key"] = "•" * 8 + masked["hitl_audit_api_key"][-4:] if len(masked["hitl_audit_api_key"]) > 4 else "****"
        return {"llm": masked}

    @settings_router.post("/llm")
    async def save_llm_settings(
        req: LLMSettingsRequest = Body(...),
        session: AsyncSession = Depends(get_async_session),
        user: UserResponse = Depends(get_current_user),
    ) -> dict[str, Any]:
        """Save LLM + HITL audit agent config for the user.

        API keys are stored in vapt_users.settings JSONB column (encrypted at
        rest via PostgreSQL-level encryption — W8+ will add Fernet encryption).
        For MVP, keys are stored as-is (single-user internal tool).
        """
        import uuid as uuid_mod
        result = await session.execute(
            select(User).where(User.id == uuid_mod.UUID(user.id))
        )
        db_user = result.scalar_one_or_none()
        if db_user is None:
            raise HTTPException(status_code=404, detail="User not found")

        # Get existing settings (preserve other keys)
        settings = dict(db_user.settings or {})

        # Get existing LLM config (to preserve API key if frontend sent masked)
        existing_llm = settings.get("llm", {})

        # Build new LLM config
        new_llm = {
            "provider": req.provider,
            "base_url": req.base_url,
            "model": req.model,
            "max_total_tokens": req.max_total_tokens,
            "max_completion_tokens": req.max_completion_tokens,
            "temperature": req.temperature,
            "hitl_audit_provider": req.hitl_audit_provider,
            "hitl_audit_base_url": req.hitl_audit_base_url,
            "hitl_audit_model": req.hitl_audit_model,
        }

        # Preserve existing API key if frontend sent a masked version (••••)
        if req.api_key and not req.api_key.startswith("•"):
            new_llm["api_key"] = req.api_key
        elif existing_llm.get("api_key"):
            new_llm["api_key"] = existing_llm["api_key"]

        if req.hitl_audit_api_key and not req.hitl_audit_api_key.startswith("•"):
            new_llm["hitl_audit_api_key"] = req.hitl_audit_api_key
        elif existing_llm.get("hitl_audit_api_key"):
            new_llm["hitl_audit_api_key"] = existing_llm["hitl_audit_api_key"]

        settings["llm"] = new_llm
        db_user.settings = settings
        await session.commit()

        return {"status": "ok", "message": "LLM settings saved"}

    @settings_router.post("/llm/test")
    async def test_llm_connection(
        req: LLMSettingsRequest = Body(...),
        user: UserResponse = Depends(get_current_user),
    ) -> dict[str, Any]:
        """Test LLM connection by making a simple API call.

        W7-D-v2: placeholder — actual test will use litellm.acompletion in W8.
        """
        if not req.api_key or req.api_key.startswith("•"):
            return {"success": False, "error": "API key required (cannot test with masked key)"}
        if not req.model:
            return {"success": False, "error": "Model name required"}
        if not req.base_url:
            return {"success": False, "error": "Base URL required"}
        # TODO W8: actual litellm test call
        return {
            "success": True,
            "message": f"Config looks valid (provider={req.provider}, model={req.model}). Actual connection test in W8.",
        }

    app.include_router(settings_router)

    # ---------- Panic button + scan registry (W7-C) ----------
    from app.pentest.scan_registry import scan_registry

    @app.post("/api/scans/{scan_id}/abort", tags=["scans"])
    async def abort_scan(
        scan_id: str,
        reason: str = "user_panic_button",
        run_cleanup: bool = True,
        user: UserResponse = Depends(get_current_user),
    ) -> dict[str, Any]:
        """Panic button — abort a running scan immediately.

        W7-C: safety net for Option C (A-in-the-Loop). Even though the audit
        agent reviews every destructive op, the user can still abort the
        entire scan mid-flight if something looks wrong.

        Flow:
            1. Set abort_event → agent loop exits on next turn check
            2. Kill all registered subprocess PIDs (SIGKILL process group)
            3. Run scripts/cleanup_scan.sh (removes sqlmap/msf/nuclei artifacts)
            4. Emit SSE event 'scan_error' with abort reason
            5. Return summary

        The agent loop should call `await scan_registry.is_aborted(scan_id)`
        each turn and exit if True.
        """
        # Emit SSE event so frontend can show "aborting..." status
        from app.pentest.events import emit_scan_error
        await emit_scan_error(scan_id, f"Scan aborted: {reason}")

        result = await scan_registry.abort_scan(
            scan_id=scan_id,
            reason=reason,
            run_cleanup=run_cleanup,
        )
        return result

    @app.get("/api/scans/active", tags=["scans"])
    async def list_active_scans(
        user: UserResponse = Depends(get_current_user),
    ) -> dict[str, Any]:
        """List all currently-active scans (registered in scan_registry)."""
        scan_ids = scan_registry.list_active_scans()
        return {
            "active_count": len(scan_ids),
            "scan_ids": scan_ids,
        }

    # ---------- W9: Orchestration endpoints (multi-agent modes) ----------
    # Exposes 3 orchestration modes: deep, plan_execute, supervisor
    # via POST /api/orchestration/scans/start-mode + introspection endpoints.
    from app.routes.orchestration import router as orch_router
    app.include_router(orch_router)

    # ---------- W14-S8: C2 Routes ----------
    from app.routes.c2 import router as c2_router
    app.include_router(c2_router)

    # ---------- Frontend SPA static mount (W7-D-v2) ----------
    from fastapi.staticfiles import StaticFiles
    from pathlib import Path as _Path

    _FRONTEND_DIR = _Path(__file__).resolve().parent.parent / "frontend"
    if _FRONTEND_DIR.is_dir():
        # Mount /static → frontend/static/ (CSS, JS, images)
        app.mount(
            "/static",
            StaticFiles(directory=str(_FRONTEND_DIR / "static")),
            name="frontend-static",
        )

        # SPA fallback: serve index.html for /, /chat, /login, /reports, etc.
        # (client-side hash router handles the rest)
        @app.get("/{full_path:path}", include_in_schema=False)
        async def spa_fallback(full_path: str):
            """Serve index.html for all non-API routes (SPA fallback).

            API routes (/api/*, /docs, /health, /openapi.json, /static/*) are
            matched first by FastAPI. Everything else → index.html.
            """
            # Block API paths from being caught by this catch-all
            if full_path.startswith(("api/", "docs", "health", "openapi.json", "redoc", "static/")):
                raise HTTPException(status_code=404, detail="Not found")
            index_path = _FRONTEND_DIR / "index.html"
            if not index_path.is_file():
                raise HTTPException(status_code=404, detail="Frontend not built")
            from fastapi.responses import FileResponse
            return FileResponse(str(index_path), media_type="text/html")

        logger.info("Frontend SPA mounted at / (serving from %s)", _FRONTEND_DIR)
    else:
        logger.warning("Frontend directory not found: %s", _FRONTEND_DIR)

    logger.info("FastAPI app created", routes=[
        "/health", "/", "/docs", "/redoc", "/openapi.json",
        "/api/auth/login", "/api/auth/refresh", "/api/auth/logout", "/api/auth/me",
    ])
    return app


# ---------- Module-level app instance (for uvicorn) ----------

app = create_app()