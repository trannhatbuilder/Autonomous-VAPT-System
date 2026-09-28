"""
VAPT-AI Scan History Routes — Phase F3 (CyberStrikeAI pattern).

Mirrors CyberStrikeAI's conversation list + paginated timeline endpoints.

Endpoints:
    GET  /api/scans/history
        List completed + in-progress scans (paginated, search by target/prompt).
        Returns scan summaries — NOT full process_details (lazy-load on click).
        Query: ?limit=20&offset=0&search=keyword&sort_by=started_at|completed_at

    GET  /api/scans/{scan_id}
        Single scan detail (metadata + findings count + decisions count).
        Query: ?include_process_details=0|1  (default 0 = lite, 1 = full)

    GET  /api/scans/{scan_id}/process-details
        Paginated timeline of events for a scan.
        Query:
            ?limit=50&offset=0          — pagination
            &event_type=tool_call_started — filter by event type
            &summary=1                  — return counts only (no payload rows)
            &anchor_id=<process_detail_id> — paginate to a specific anchor
                                              (matches CyberStrikeAI's
                                              GetProcessDetailOffset pattern:
                                              page is requested such that
                                              the anchor sits at 1/3 of the
                                              page — useful when user clicks
                                              a tool card from elsewhere)

    GET  /api/scans/{scan_id}/process-details/{detail_id}
        Single process_detail record — full payload. Used by frontend when
        user expands a tool card to see full args/result (lazy-load).

    DELETE /api/scans/{scan_id}
        Hard delete a scan. Findings survive (FK SET NULL — Phase F1) so
        they remain queryable in the Findings tab. ProcessDetails are
        CASCADE-deleted with the scan row. Mirrors CyberStrikeAI's
        DELETE /api/conversations/{id} behavior.

Usage:
    # In app/main.py:
    from app.routes.scans import router as scans_history_router
    app.include_router(scans_history_router)
"""
from __future__ import annotations

import logging
from typing import Any
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select, func, and_, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_async_session
from app.db.models.scan import Scan
from app.db.models.pentest import Finding
from app.db.models.process_detail import ProcessDetail
from app.auth.manager import decode_access_token, InvalidTokenError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/scans", tags=["scans"])


# ── Pydantic-free response builders (return dict directly) ──────────────


def _scan_summary(scan: Scan, findings_count: int | None = None,
                  process_details_count: int | None = None) -> dict[str, Any]:
    """Build a lightweight scan summary for the list endpoint.

    Includes only fields needed by the sidebar UI — NOT full process_details
    (those are lazy-loaded via the per-scan process-details endpoint).

    Mirrors CyberStrikeAI's conversation list summary shape (no
    last_message_preview / message_count — those would require a JOIN
    we can defer until needed).
    """
    return {
        "id": scan.id,
        "target": scan.target,
        "target_type": scan.target_type,
        "agent_mode": scan.agent_mode,
        "hitl_mode": scan.hitl_mode,
        "user_prompt": scan.user_prompt,
        "status": scan.status,
        "progress": scan.progress,
        "started_at": scan.started_at.isoformat() if scan.started_at else None,
        "completed_at": scan.completed_at.isoformat() if scan.completed_at else None,
        "findings_count": findings_count,
        "process_details_count": process_details_count,
        "scope": scan.scope_json,
    }


# ── Endpoints ────────────────────────────────────────────────────────────


@router.get("/history")
async def list_scan_history(
    limit: int = Query(20, ge=1, le=200, description="Max results (1-200, default 20)"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
    search: str | None = Query(None, description="Search target or user_prompt (LIKE)"),
    sort_by: str = Query("started_at", description="Sort field: started_at | completed_at | target | status"),
    sort_dir: str = Query("desc", description="Sort direction: asc | desc"),
    session: AsyncSession = Depends(get_async_session),
) -> dict[str, Any]:
    """List completed + in-progress scans (paginated).

    Returns scan summaries — NOT full process_details. The frontend
    lazy-loads process_details per-scan via
    GET /api/scans/{scan_id}/process-details when the user clicks a
    scan in the sidebar.

    Query params:
        limit: max results (1-200, default 20)
        offset: pagination offset (default 0)
        search: search by target OR user_prompt (LIKE %search%)
        sort_by: started_at | completed_at | target | status (default started_at)
        sort_dir: asc | desc (default desc)

    Response shape:
        {
          "scans": [<scan summary>],
          "total": N,
          "limit": L,
          "offset": O
        }
    """
    # Build WHERE clause for search (case-insensitive LIKE on target + user_prompt)
    conditions = []
    if search:
        like_pattern = f"%{search}%"
        conditions.append(
            (Scan.target.ilike(like_pattern)) |
            (Scan.user_prompt.ilike(like_pattern))
        )

    # Count total (with filters)
    count_stmt = select(func.count(Scan.id))
    if conditions:
        count_stmt = count_stmt.where(and_(*conditions))
    total = await session.scalar(count_stmt) or 0

    # Determine sort column
    sort_columns = {
        "started_at": Scan.started_at,
        "completed_at": Scan.completed_at,
        "target": Scan.target,
        "status": Scan.status,
        "created_at": Scan.created_at,
    }
    sort_col = sort_columns.get(sort_by, Scan.started_at)
    if sort_dir.lower() == "asc":
        order_stmt = sort_col.asc()
    else:
        order_stmt = sort_col.desc()
    # NULLS LAST so completed scans show first when sorting by completed_at desc
    # (PostgreSQL: `ORDER BY x DESC NULLS LAST`)
    if sort_by in ("started_at", "completed_at"):
        order_stmt = text(f"{sort_col.key} {'DESC' if sort_dir.lower() == 'desc' else 'ASC'} NULLS LAST")

    # Build SELECT
    stmt = select(Scan).order_by(order_stmt).offset(offset).limit(limit)
    if conditions:
        stmt = stmt.where(and_(*conditions))
    result = await session.execute(stmt)
    scans = result.scalars().all()

    # For each scan, fetch findings_count + process_details_count
    # (single round-trip with a subquery would be faster, but for sidebar
    # use case this is fine — N+1 queries for N=20 scans is ~20ms total)
    scan_summaries = []
    for scan in scans:
        findings_count = await session.scalar(
            select(func.count(Finding.id)).where(Finding.scan_id == scan.id)
        ) or 0
        process_details_count = await session.scalar(
            select(func.count(ProcessDetail.id)).where(ProcessDetail.scan_id == scan.id)
        ) or 0
        scan_summaries.append(
            _scan_summary(scan, findings_count=findings_count,
                          process_details_count=process_details_count)
        )

    return {
        "scans": scan_summaries,
        "total": total,
        "limit": limit,
        "offset": offset,
    }


@router.get("/{scan_id}")
async def get_scan_detail(
    scan_id: str,
    include_process_details: int = Query(0, ge=0, le=1,
        description="0=lite (default, no process_details), 1=full (embed process_details)"),
    limit_process_details: int = Query(50, ge=1, le=500,
        description="When include_process_details=1, max process_details to return"),
    session: AsyncSession = Depends(get_async_session),
) -> dict[str, Any]:
    """Get a single scan's detail.

    By default returns lite view (scan metadata + counts only) — fast.
    Pass ?include_process_details=1 to embed a paginated slice of
    process_details (the agent timeline). For full timeline pagination,
    use GET /api/scans/{scan_id}/process-details instead.

    Mirrors CyberStrikeAI's
        GET /api/conversations/{id}?include_process_details=0|1
    pattern.
    """
    scan = await session.get(Scan, scan_id)
    if scan is None:
        raise HTTPException(status_code=404, detail=f"Scan {scan_id!r} not found")

    # Counts
    findings_count = await session.scalar(
        select(func.count(Finding.id)).where(Finding.scan_id == scan_id)
    ) or 0
    process_details_count = await session.scalar(
        select(func.count(ProcessDetail.id)).where(ProcessDetail.scan_id == scan_id)
    ) or 0
    findings_by_severity_rows = (await session.execute(
        select(Finding.severity, func.count(Finding.id))
        .where(Finding.scan_id == scan_id)
        .group_by(Finding.severity)
    )).all()
    findings_by_severity = {row[0]: row[1] for row in findings_by_severity_rows}

    response = _scan_summary(scan, findings_count=findings_count,
                              process_details_count=process_details_count)
    response["findings_by_severity"] = findings_by_severity
    response["result_summary"] = scan.result_summary

    if include_process_details == 1:
        # Embed a paginated slice of process_details (lite = no payload)
        pd_stmt = (
            select(ProcessDetail)
            .where(ProcessDetail.scan_id == scan_id)
            .order_by(ProcessDetail.created_at.asc())
            .limit(limit_process_details)
        )
        pd_result = await session.execute(pd_stmt)
        process_details = pd_result.scalars().all()
        response["process_details"] = [
            {
                "id": str(pd.id),
                "event_type": pd.event_type,
                "message": pd.message,
                "created_at": pd.created_at.isoformat() if pd.created_at else None,
                # data (full payload) is NOT included in lite view —
                # frontend fetches it via GET /process-details/{id}
                # when user expands a tool card
                "has_payload": pd.data is not None,
            }
            for pd in process_details
        ]
        response["process_details_returned"] = len(process_details)
        response["process_details_total"] = process_details_count

    return response


@router.get("/{scan_id}/process-details")
async def list_process_details(
    scan_id: str,
    limit: int = Query(50, ge=1, le=500, description="Max results (1-500, default 50)"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
    event_type: str | None = Query(None, description="Filter by event_type (e.g. 'tool_call_started')"),
    summary: int = Query(0, ge=0, le=1, description="1=return counts only, no rows"),
    anchor_id: str | None = Query(None, description="ProcessDetail ID to anchor the page at (page is requested such that anchor sits at ~1/3 of the page)"),
    session: AsyncSession = Depends(get_async_session),
) -> dict[str, Any]:
    """Paginated timeline of process_details for a scan.

    Mirrors CyberStrikeAI's
        GET /api/messages/{id}/process-details?limit&offset&anchorId&summary
    pattern.

    Query params:
        limit: max results per page (1-500, default 50)
        offset: pagination offset (default 0)
        event_type: filter by event_type (e.g. 'tool_call_started')
        summary: 1=return counts only (no payload rows), 0=return rows
        anchor_id: ProcessDetail ID — when set, the offset is auto-computed
                   so that the anchor sits at ~1/3 of the page (matches
                   CyberStrikeAI's GetProcessDetailOffset pattern, useful
                   when user clicks a tool card from elsewhere and we
                   want to scroll-to-it in the timeline)

    Response shape (summary=0):
        {
          "process_details": [<row with payload>],
          "total": N,
          "limit": L,
          "offset": O,
          "has_more": true|false
        }

    Response shape (summary=1):
        {
          "total": N,
          "by_event_type": {"tool_call_started": 5, "iteration": 10, ...},
          "first_created_at": "...",
          "last_created_at": "..."
        }
    """
    # Verify scan exists
    scan = await session.get(Scan, scan_id)
    if scan is None:
        raise HTTPException(status_code=404, detail=f"Scan {scan_id!r} not found")

    # Build WHERE
    conditions = [ProcessDetail.scan_id == scan_id]
    if event_type:
        conditions.append(ProcessDetail.event_type == event_type)

    # Summary mode: return counts only
    if summary == 1:
        total = await session.scalar(
            select(func.count(ProcessDetail.id)).where(and_(*conditions))
        ) or 0
        by_type_rows = (await session.execute(
            select(ProcessDetail.event_type, func.count(ProcessDetail.id))
            .where(and_(*conditions))
            .group_by(ProcessDetail.event_type)
        )).all()
        by_event_type = {row[0]: row[1] for row in by_type_rows}
        first_created = await session.scalar(
            select(func.min(ProcessDetail.created_at)).where(and_(*conditions))
        )
        last_created = await session.scalar(
            select(func.max(ProcessDetail.created_at)).where(and_(*conditions))
        )
        return {
            "scan_id": scan_id,
            "total": total,
            "by_event_type": by_event_type,
            "first_created_at": first_created.isoformat() if first_created else None,
            "last_created_at": last_created.isoformat() if last_created else None,
        }

    # Anchor-id mode: compute offset so anchor sits at ~1/3 of page
    actual_offset = offset
    if anchor_id:
        # Count rows BEFORE the anchor
        anchor_subq = select(ProcessDetail.created_at).where(ProcessDetail.id == anchor_id)
        rows_before = await session.scalar(
            select(func.count(ProcessDetail.id)).where(
                and_(
                    *conditions,
                    ProcessDetail.created_at < anchor_subq,
                )
            )
        ) or 0
        # Place anchor at ~1/3 of the page
        actual_offset = max(0, rows_before - limit // 3)

    # Total (with filters)
    total = await session.scalar(
        select(func.count(ProcessDetail.id)).where(and_(*conditions))
    ) or 0

    # Fetch page
    stmt = (
        select(ProcessDetail)
        .where(and_(*conditions))
        .order_by(ProcessDetail.created_at.asc())
        .offset(actual_offset)
        .limit(limit)
    )
    result = await session.execute(stmt)
    process_details = result.scalars().all()

    return {
        "scan_id": scan_id,
        "process_details": [
            {
                "id": str(pd.id),
                "event_type": pd.event_type,
                "message": pd.message,
                "data": pd.data,  # full payload (JSONB) — included here
                "signature": pd.signature,
                "created_at": pd.created_at.isoformat() if pd.created_at else None,
            }
            for pd in process_details
        ],
        "total": total,
        "limit": limit,
        "offset": actual_offset,
        "has_more": (actual_offset + len(process_details)) < total,
    }


@router.get("/{scan_id}/process-details/{detail_id}")
async def get_process_detail(
    scan_id: str,
    detail_id: str,
    session: AsyncSession = Depends(get_async_session),
) -> dict[str, Any]:
    """Get a single process_detail record (full payload).

    Used by the frontend when user expands a tool card in the timeline
    to see the full tool args + result (lazy-load — list endpoint returns
    only the message summary, this endpoint returns the full data JSONB).

    Mirrors CyberStrikeAI's GET /api/process-details/{id}.
    """
    pd = await session.get(ProcessDetail, detail_id)
    if pd is None:
        raise HTTPException(status_code=404, detail=f"ProcessDetail {detail_id!r} not found")
    if pd.scan_id != scan_id:
        raise HTTPException(status_code=404, detail=f"ProcessDetail {detail_id!r} not found in scan {scan_id!r}")
    return {
        "id": str(pd.id),
        "scan_id": pd.scan_id,
        "event_type": pd.event_type,
        "message": pd.message,
        "data": pd.data,  # full payload
        "signature": pd.signature,
        "created_at": pd.created_at.isoformat() if pd.created_at else None,
        "updated_at": pd.updated_at.isoformat() if pd.updated_at else None,
    }


@router.delete("/{scan_id}")
async def delete_scan(
    scan_id: str,
    session: AsyncSession = Depends(get_async_session),
) -> dict[str, Any]:
    """Hard delete a scan with explicit child-row cleanup + savepoint isolation.

    ── Why this is not just `session.delete(scan)` ──
    Many child tables FK to vapt_scans WITHOUT any `ondelete` clause
    (NO ACTION in PostgreSQL). The model-side `ondelete` declarations in
    app/db/models/ are only honored by `create_all()` — they are NOT applied
    to the actual DB unless a matching migration has ALTERed the constraint.
    As of migration 0008 (if applied), all FKs have proper ondelete; but on
    DBs that haven't run migration 0008 yet, DELETE on vapt_scans will hit
    ForeignKeyViolation → 500 error.

    This route explicitly deletes/nulls all known child rows BEFORE deleting
    the scan row. Even after migration 0008 is applied, this stays correct
    (idempotent — DELETE finds 0 rows, no-op).

    ── Savepoint isolation (CRITICAL) ──
    PostgreSQL aborts the ENTIRE transaction when any SQL statement fails.
    Even if Python catches the exception, the underlying transaction is
    poisoned — all subsequent SQL statements (including the final
    `session.delete(scan)` + `session.commit()`) fail with
    `InFailedSQLTransactionError: current transaction is aborted, commands
    ignored until end of transaction block`.

    Symptom (the bug this fixes):
        DELETE /api/scans/{id} returns 500 even though scan status is
        already "aborted" and all visible data looks clean. Server log shows
        `InFailedSQLTransactionError` cascading from a missing table or
        missing column on a child-table DELETE.

    Fix:
        1. Pre-check `information_schema.tables` + `information_schema.columns`
           to find which child tables ACTUALLY exist on this DB (skips
           tables from newer migrations not yet applied).
        2. Wrap each child-table DELETE in a SAVEPOINT
           (`session.begin_nested()`). If the DELETE fails, ROLLBACK TO
           SAVEPOINT — only that sub-transaction is undone, the main
           transaction stays clean.

    Cascade behavior (Phase F1, preserved):
        - vapt_findings.scan_id       → SET NULL (findings SURVIVE w/ scan_tag)
        - vapt_chat_messages.scan_id  → SET NULL (chat history survives)
        - vapt_process_details        → hard DELETE (scan-scoped timeline)
        - All other child tables      → hard DELETE (scan-scoped artifacts)
    """
    scan = await session.get(Scan, scan_id)
    if scan is None:
        raise HTTPException(status_code=404, detail=f"Scan {scan_id!r} not found")

    # Capture counts before delete for the response
    findings_count = await session.scalar(
        select(func.count(Finding.id)).where(Finding.scan_id == scan_id)
    ) or 0
    process_details_count = await session.scalar(
        select(func.count(ProcessDetail.id)).where(ProcessDetail.scan_id == scan_id)
    ) or 0

    # Snapshot scan target into findings.scan_tag (in case any findings have NULL scan_tag)
    if scan.target:
        await session.execute(
            text(
                "UPDATE vapt_findings SET scan_tag = :target "
                "WHERE scan_id = :scan_id AND scan_tag IS NULL"
            ),
            {"target": scan.target, "scan_id": scan_id},
        )

    # ── Pre-check: which child tables/columns actually exist on this DB? ──
    # Avoids hitting "table does not exist" errors at DELETE time.
    # `information_schema.tables` + `information_schema.columns` queries are
    # cheap (single index scan on pg_catalog) and let us skip cleanly.
    existing_child_tables: set[str] = set()
    try:
        rows = await session.execute(
            text(
                """
                SELECT table_name
                FROM information_schema.columns
                WHERE column_name = 'scan_id'
                  AND table_schema = 'public'
                  AND table_name LIKE 'vapt_%'
                """
            )
        )
        existing_child_tables = {r[0] for r in rows.fetchall()}
    except Exception as e:
        # information_schema should always exist on PostgreSQL — but if this
        # is a different backend (sqlite for tests), fall back to the full
        # list and rely on savepoints to swallow errors.
        logger.warning("delete_scan: information_schema pre-check failed — using full list: %s", e)
        existing_child_tables = set()  # empty = use full list below

    # ── Cleanup actions (table_name, action_type) ──
    cleanup_steps: list[tuple[str, str]] = []

    # 1. vapt_findings.scan_id → SET NULL (findings SURVIVE per Phase F1)
    if "vapt_findings" in existing_child_tables or not existing_child_tables:
        try:
            async with session.begin_nested():  # SAVEPOINT
                await session.execute(
                    text("UPDATE vapt_findings SET scan_id = NULL WHERE scan_id = :sid"),
                    {"sid": scan_id},
                )
            cleanup_steps.append(("vapt_findings", "SET NULL"))
        except Exception as e:
            logger.warning("delete_scan: vapt_findings SET NULL failed (continuing): %s", e)

    # 2. vapt_chat_messages.scan_id → SET NULL
    if "vapt_chat_messages" in existing_child_tables or not existing_child_tables:
        try:
            async with session.begin_nested():  # SAVEPOINT
                await session.execute(
                    text("UPDATE vapt_chat_messages SET scan_id = NULL WHERE scan_id = :sid"),
                    {"sid": scan_id},
                )
            cleanup_steps.append(("vapt_chat_messages", "SET NULL"))
        except Exception as e:
            logger.warning("delete_scan: vapt_chat_messages SET NULL failed (continuing): %s", e)

    # 3. Hard DELETE tables (scan-scoped artifacts with no value after scan deletion)
    hard_delete_tables = [
        "vapt_process_details",
        "vapt_pentest_facts",
        "vapt_assets",
        "vapt_consent_forms",
        "vapt_hitl_approvals",
        "vapt_audit_logs",
        "vapt_c2_sessions",
        "vapt_c2_tasks",
        "vapt_replay_traces",
        "vapt_attack_chain_facts",
        "vapt_kg_facts",
        "vapt_methodology_progress",
        "vapt_rl_episodes",
        "vapt_rl_q_snapshots",
    ]
    for tbl in hard_delete_tables:
        # Skip tables we know don't exist on this DB (pre-check above).
        # If pre-check failed (empty set), try anyway — savepoint will catch.
        if existing_child_tables and tbl not in existing_child_tables:
            cleanup_steps.append((tbl, "SKIP (table does not exist)"))
            continue
        try:
            async with session.begin_nested():  # SAVEPOINT — isolates failures
                await session.execute(
                    text(f'DELETE FROM "{tbl}" WHERE scan_id = :sid'),
                    {"sid": scan_id},
                )
            cleanup_steps.append((tbl, "DELETE"))
        except Exception as e:
            # The savepoint rolled back automatically — main transaction
            # is still clean. Log + continue with the next table.
            logger.warning(
                "delete_scan: %s DELETE failed (savepoint rolled back, continuing): %s",
                tbl, e,
            )
            cleanup_steps.append((tbl, f"FAILED: {type(e).__name__}"))

    # Final: delete the scan row itself — should succeed now that all
    # children are cleaned up (or skipped via savepoint).
    try:
        await session.delete(scan)
        await session.commit()
    except Exception as e:
        # If even this fails, ROLLBACK the whole transaction and surface
        # a clear error to the user. The cleanup_steps list in the response
        # helps diagnose which child table blocked the delete.
        await session.rollback()
        logger.error(
            "delete_scan: final DELETE on vapt_scans failed even after child cleanup | "
            "scan_id=%s | cleanup=%s | error=%s",
            scan_id,
            "; ".join(f"{t}:{a}" for t, a in cleanup_steps),
            e,
        )
        raise HTTPException(
            status_code=500,
            detail=(
                f"Failed to delete scan {scan_id}: {e}. "
                f"Cleanup attempts: {'; '.join(f'{t}:{a}' for t, a in cleanup_steps)}"
            ),
        ) from e

    logger.info(
        "Scan deleted | scan_id=%s | findings_preserved=%d | process_details_deleted=%d | cleanup=%s",
        scan_id, findings_count, process_details_count,
        "; ".join(f"{t}:{a}" for t, a in cleanup_steps),
    )

    return {
        "scan_id": scan_id,
        "deleted": True,
        "findings_preserved": findings_count,  # these survive (FK SET NULL)
        "process_details_deleted": process_details_count,  # hard DELETE'd
        "cleanup_steps": cleanup_steps,  # per-table outcome log
    }