"""
VAPT-AI Methodology + CVE + Scope Violation API Routes (W8-G).

Provides 6 endpoints:

    GET  /api/methodology/wstg                  — list/search OWASP WSTG v4.2 catalog
    GET  /api/methodology/wstg/{wstg_id}        — get single WSTG entry by ID
    GET  /api/methodology/attack                — list/search MITRE ATT&CK catalog
    GET  /api/methodology/attack/{technique_id} — get single ATT&CK entry by ID
    GET  /api/cve/{cve_id}                      — CVE lookup via NVD REST API v2 (24h cache)
    GET  /api/audit/scope-violations             — list scope_violation audit entries

All endpoints require authentication (Depends(get_current_user)).
NVD lookup is best-effort — returns 404 if CVE not found or NVD unavailable.

Usage (in app/main.py create_app()):
    from app.routes.w8g_methodology import create_methodology_router
    app.include_router(create_methodology_router(get_current_user, UserResponse))
"""
from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.audit.logger import AuditLogger
from app.core.cve_client import get_cve
from app.db.models.attack_catalog import AttackTechniqueCatalog
from app.db.models.methodology import WSTGMethodologyCatalog
from app.db.session import get_async_session

logger = logging.getLogger(__name__)


# ---------- Pydantic response schemas ----------

class WSTGEntry(BaseModel):
    """Single OWASP WSTG v4.2 catalog entry."""
    wstg_id: str
    name: str
    category: str
    description: str | None = None
    related_mitre_attack: list[str] = Field(default_factory=list)
    related_cwe: list[str] = Field(default_factory=list)
    applicable_to_web_mvp: bool


class AttackEntry(BaseModel):
    """Single MITRE ATT&CK technique catalog entry."""
    technique_id: str
    name: str
    tactic: str
    description: str | None = None
    detection: str | None = None
    mitigation: str | None = None
    related_wstg_ids: list[str] = Field(default_factory=list)
    related_cwe: list[str] = Field(default_factory=list)
    example_uses: list[str] = Field(default_factory=list)
    applicable_to_web_mvp: bool


class CveResponse(BaseModel):
    """CVE lookup response (from NVD REST API v2)."""
    cve_id: str
    description: str = ""
    cvss_vector: str | None = None
    cvss_base_score: float | None = None
    cvss_severity: str | None = None
    cvss_version: str | None = None
    cwe_ids: list[str] = Field(default_factory=list)
    references: list[str] = Field(default_factory=list)
    published_date: str | None = None
    last_modified_date: str | None = None
    source: str = "nvd"


class ScopeViolationEntry(BaseModel):
    """Single scope_violation audit log entry (W8-A)."""
    id: str
    actor_id: str | None = None
    scan_id: str | None = None
    command: str = ""
    target: str = ""
    reason: str = ""
    severity: str = "error"
    created_at: str


# ---------- Router factory ----------

def create_methodology_router(
    get_current_user_dep,
    user_response_model,
) -> APIRouter:
    """Create the W8-G methodology + CVE + scope-violation API router.

    Args:
        get_current_user_dep: FastAPI dependency for auth (returns current user).
        user_response_model: Pydantic model for user response (unused but
            required for type consistency with other routers).

    Returns:
        APIRouter with prefix="/api" + 6 endpoints.
    """
    router = APIRouter(prefix="/api", tags=["methodology"])

    # ========================================================================
    # WSTG catalog endpoints
    # ========================================================================

    @router.get("/methodology/wstg", response_model=dict)
    async def list_wstg(
        category: str | None = Query(None, description="Filter by category (e.g. 'Input Validation')"),
        web_mvp_only: bool = Query(False, description="If true, only return entries applicable_to_web_mvp=True"),
        limit: int = Query(200, ge=1, le=500, description="Max results"),
        offset: int = Query(0, ge=0, description="Pagination offset"),
        session: AsyncSession = Depends(get_async_session),
        user=Depends(get_current_user_dep),
    ) -> dict[str, Any]:
        """List OWASP WSTG v4.2 catalog entries with optional filters.

        Returns up to `limit` entries (default 200, max 500). Each entry has
        wstg_id, name, category, description, related_mitre_attack, related_cwe,
        applicable_to_web_mvp.

        Total count is returned in `total` field for pagination UI.
        """
        # Build query
        query = select(WSTGMethodologyCatalog)
        if category:
            query = query.where(WSTGMethodologyCatalog.category == category)
        if web_mvp_only:
            query = query.where(WSTGMethodologyCatalog.applicable_to_web_mvp == True)  # noqa: E712

        # Count query (for pagination)
        count_query = select(func.count()).select_from(WSTGMethodologyCatalog)
        if category:
            count_query = count_query.where(WSTGMethodologyCatalog.category == category)
        if web_mvp_only:
            count_query = count_query.where(WSTGMethodologyCatalog.applicable_to_web_mvp == True)  # noqa: E712

        total_result = await session.execute(count_query)
        total = total_result.scalar_one()

        # Apply pagination + ordering
        query = query.order_by(WSTGMethodologyCatalog.wstg_id).limit(limit).offset(offset)
        result = await session.execute(query)
        rows = result.scalars().all()

        return {
            "count": len(rows),
            "total": total,
            "offset": offset,
            "limit": limit,
            "entries": [
                WSTGEntry(
                    wstg_id=r.wstg_id,
                    name=r.name,
                    category=r.category,
                    description=r.description,
                    related_mitre_attack=r.related_mitre_attack or [],
                    related_cwe=r.related_cwe or [],
                    applicable_to_web_mvp=r.applicable_to_web_mvp,
                ).model_dump()
                for r in rows
            ],
        }

    @router.get("/methodology/wstg/{wstg_id}", response_model=WSTGEntry)
    async def get_wstg(
        wstg_id: str,
        session: AsyncSession = Depends(get_async_session),
        user=Depends(get_current_user_dep),
    ) -> WSTGEntry:
        """Get single WSTG entry by ID (e.g. WSTG-INPV-05).

        Returns 404 if WSTG ID not found in catalog.
        """
        # Normalize to upper-case
        wstg_id = wstg_id.upper().strip()

        query = select(WSTGMethodologyCatalog).where(WSTGMethodologyCatalog.wstg_id == wstg_id)
        result = await session.execute(query)
        row = result.scalar_one_or_none()

        if row is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"WSTG entry not found: {wstg_id}",
            )

        return WSTGEntry(
            wstg_id=row.wstg_id,
            name=row.name,
            category=row.category,
            description=row.description,
            related_mitre_attack=row.related_mitre_attack or [],
            related_cwe=row.related_cwe or [],
            applicable_to_web_mvp=row.applicable_to_web_mvp,
        )

    # ========================================================================
    # ATT&CK catalog endpoints
    # ========================================================================

    @router.get("/methodology/attack", response_model=dict)
    async def list_attack(
        tactic: str | None = Query(None, description="Filter by tactic (e.g. 'Initial Access')"),
        web_mvp_only: bool = Query(False, description="If true, only return web-MVP applicable techniques"),
        limit: int = Query(100, ge=1, le=500, description="Max results"),
        offset: int = Query(0, ge=0, description="Pagination offset"),
        session: AsyncSession = Depends(get_async_session),
        user=Depends(get_current_user_dep),
    ) -> dict[str, Any]:
        """List MITRE ATT&CK catalog entries with optional filters.

        `tactic` filter uses case-insensitive substring match (so "Initial Access"
        matches entries with tactic "Initial Access, Persistence" too).
        """
        query = select(AttackTechniqueCatalog)
        if tactic:
            query = query.where(AttackTechniqueCatalog.tactic.ilike(f"%{tactic}%"))
        if web_mvp_only:
            query = query.where(AttackTechniqueCatalog.applicable_to_web_mvp == True)  # noqa: E712

        count_query = select(func.count()).select_from(AttackTechniqueCatalog)
        if tactic:
            count_query = count_query.where(AttackTechniqueCatalog.tactic.ilike(f"%{tactic}%"))
        if web_mvp_only:
            count_query = count_query.where(AttackTechniqueCatalog.applicable_to_web_mvp == True)  # noqa: E712

        total_result = await session.execute(count_query)
        total = total_result.scalar_one()

        query = query.order_by(AttackTechniqueCatalog.technique_id).limit(limit).offset(offset)
        result = await session.execute(query)
        rows = result.scalars().all()

        return {
            "count": len(rows),
            "total": total,
            "offset": offset,
            "limit": limit,
            "entries": [
                AttackEntry(
                    technique_id=r.technique_id,
                    name=r.name,
                    tactic=r.tactic,
                    description=r.description,
                    detection=r.detection,
                    mitigation=r.mitigation,
                    related_wstg_ids=r.related_wstg_ids or [],
                    related_cwe=r.related_cwe or [],
                    example_uses=r.example_uses or [],
                    applicable_to_web_mvp=r.applicable_to_web_mvp,
                ).model_dump()
                for r in rows
            ],
        }

    @router.get("/methodology/attack/{technique_id}", response_model=AttackEntry)
    async def get_attack(
        technique_id: str,
        session: AsyncSession = Depends(get_async_session),
        user=Depends(get_current_user_dep),
    ) -> AttackEntry:
        """Get single ATT&CK entry by technique ID (e.g. T1190).

        Returns 404 if technique ID not found in catalog.
        """
        technique_id = technique_id.upper().strip()

        query = select(AttackTechniqueCatalog).where(
            AttackTechniqueCatalog.technique_id == technique_id
        )
        result = await session.execute(query)
        row = result.scalar_one_or_none()

        if row is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"ATT&CK technique not found: {technique_id}",
            )

        return AttackEntry(
            technique_id=row.technique_id,
            name=row.name,
            tactic=row.tactic,
            description=row.description,
            detection=row.detection,
            mitigation=row.mitigation,
            related_wstg_ids=row.related_wstg_ids or [],
            related_cwe=row.related_cwe or [],
            example_uses=row.example_uses or [],
            applicable_to_web_mvp=row.applicable_to_web_mvp,
        )

    # ========================================================================
    # CVE lookup endpoint
    # ========================================================================

    @router.get("/cve/{cve_id}", response_model=CveResponse)
    async def lookup_cve(
        cve_id: str,
        user=Depends(get_current_user_dep),
    ) -> CveResponse:
        """Lookup CVE details via NVD REST API v2 (W8-B client).

        24h cache (memory + disk). Rate-limited per NVD policy:
        5 req/30s without API key, 50 req/30s with VAPT_AI_NVD_API_KEY.

        Returns 404 if CVE not found or NVD unavailable.
        Returns the CVE record with description, CVSS vector + score, CWE IDs,
        references, published/last-modified dates.
        """
        record = await get_cve(cve_id)

        if record is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"CVE not found or NVD lookup failed: {cve_id}",
            )

        return CveResponse(
            cve_id=record.cve_id,
            description=record.description,
            cvss_vector=record.cvss_vector,
            cvss_base_score=record.cvss_base_score,
            cvss_severity=record.cvss_severity,
            cvss_version=record.cvss_version,
            cwe_ids=record.cwe_ids,
            references=record.references,
            published_date=record.published_date.isoformat() if record.published_date else None,
            last_modified_date=record.last_modified_date.isoformat() if record.last_modified_date else None,
            source=record.source,
        )

    # ========================================================================
    # Scope violation audit log endpoint (W8-A history)
    # ========================================================================

    @router.get("/audit/scope-violations", response_model=dict)
    async def list_scope_violations(
        scan_id: str | None = Query(None, description="Filter by scan ID"),
        actor_id: str | None = Query(None, description="Filter by agent name (e.g. 'recon')"),
        limit: int = Query(100, ge=1, le=500, description="Max results"),
        offset: int = Query(0, ge=0, description="Pagination offset"),
        session: AsyncSession = Depends(get_async_session),
        user=Depends(get_current_user_dep),
    ) -> dict[str, Any]:
        """List scope_violation audit log entries (W8-A).

        Returns audit log entries with action='scope_violation' (written by
        executor._log_scope_violation when scope guard blocks a command).

        Each entry contains:
            id, actor_id, scan_id, command, target, reason, severity, created_at

        Useful for:
            - Investigating which tools were blocked during a scan
            - Auditing which agents attempted out-of-scope actions
            - Compliance reporting (D23 chain-of-custody)
        """
        audit = AuditLogger(session)
        entries = await audit.get_entries(
            scan_id=scan_id,
            actor_type="agent",
            action="scope_violation",
            limit=min(limit, 500),
            offset=offset,
        )

        # Optional: filter by actor_id (get_entries doesn't support actor_id filter directly)
        if actor_id:
            entries = [e for e in entries if e.actor_id == actor_id]

        result_entries = []
        for e in entries:
            after_data = e.after_json or {}
            result_entries.append(
                ScopeViolationEntry(
                    id=str(e.id),
                    actor_id=e.actor_id,
                    scan_id=e.scan_id,
                    command=after_data.get("command", ""),
                    target=after_data.get("target", ""),
                    reason=after_data.get("reason", ""),
                    severity=after_data.get("severity", "error"),
                    created_at=e.created_at.isoformat() if e.created_at else "",
                ).model_dump()
            )

        # Get total count of scope_violation entries (for pagination UI)
        count_query = (
            select(func.count())
            .select_from(AuditLogger.__model_class__ if hasattr(AuditLogger, "__model_class__") else None)
        ) if False else None  # skip count for now — would need AuditLog model import

        return {
            "count": len(result_entries),
            "offset": offset,
            "limit": limit,
            "entries": result_entries,
        }

    return router
