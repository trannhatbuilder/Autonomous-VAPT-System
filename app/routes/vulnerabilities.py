"""
VAPT-AI Vulnerabilities Routes — Phase F6 (CyberStrikeAI pattern).

Mirrors CyberStrikeAI's vulnerability export endpoint
(internal/handler/vulnerability.go:417-506 ExportVulnerabilities).

CyberStrikeAI treats "report" as on-demand Markdown export, NOT as
auto-generated PDF after scan. The export endpoint:
  - Queries vulnerabilities by filter (scan_id, severity, verified)
  - Groups them by `conversation_tag` (VAPT-AI's `scan_tag` — Phase F1)
  - Builds Markdown via `append_vulnerability_markdown()` helper
  - Returns inline JSON: {mode, group_by, total, files: [{filename, content}]}
  - Frontend handles the actual file download (Blob + saveAs)

This file also adds a `/api/scans/{scan_id}/results` JSON aggregation
endpoint (mirror of CyberStrikeAI's
`GET /api/conversations/:id/results` at openapi.go:6462-6503).

Endpoints:
    GET  /api/vulnerabilities/export?group_by=scan|severity|vuln_type&mode=summary|split
        Export findings as Markdown. Returns JSON with file contents
        inline — frontend uses Blob/saveAs for download.

    GET  /api/scans/{scan_id}/results
        Aggregated JSON of a scan's messages + vulnerabilities + key
        process_details. Useful for one-shot API consumers (CI/CD,
        external dashboards) that don't want to paginate through
        /process-details.

Usage:
    # In app/main.py:
    from app.routes.vulnerabilities import router as vulns_router
    app.include_router(vulns_router)
"""
from __future__ import annotations

import logging
import re
from datetime import datetime, UTC
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select, func, and_
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_async_session
from app.db.models.scan import Scan
from app.db.models.pentest import Finding
from app.db.models.process_detail import ProcessDetail

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["vulnerabilities"])


# ── Helpers ──────────────────────────────────────────────────────────────


def _sanitize_filename(name: str) -> str:
    """Sanitize a string for use as a filename.

    Replaces non-alphanumeric characters with hyphens, collapses
    consecutive hyphens, strips leading/trailing hyphens.
    Matches CyberStrikeAI's filename sanitization pattern.
    """
    # Replace anything that's not alphanumeric or dot with hyphen
    sanitized = re.sub(r"[^a-zA-Z0-9.\-]", "-", name)
    # Collapse consecutive hyphens
    sanitized = re.sub(r"-+", "-", sanitized)
    # Strip leading/trailing hyphens
    return sanitized.strip("-").lower() or "untitled"


def _severity_sort_key(severity: str | None) -> int:
    """Sort key for severity (critical first, info last)."""
    order = {"critical": 0, "high": 1, "medium": 2, "low": 3, "info": 4}
    return order.get((severity or "info").lower(), 5)


def _append_vulnerability_markdown(buf: list[str], finding: Finding, scan_tag: str | None) -> None:
    """Append a single vulnerability as Markdown to the buffer.

    Mirrors CyberStrikeAI's appendVulnerabilityMarkdown
    (internal/handler/vulnerability.go:517-583).

    Output shape:
        ### {title}
        - **Severity**: critical
        - **Type**: sqli
        - **Location**: /login.php
        - **CVSS**: 9.8 (AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H)
        - **Status**: open
        - **Scan**: scan_tag_or_id
        - **Description**: ...
        - **Reproduction**: ...
        - **Evidence**: ...
        - **Impact**: ...
        - **Recommendation**: ...

        ---
    """
    buf.append(f"### {finding.name}")
    buf.append("")
    buf.append(f"- **Severity**: `{finding.severity}`")
    buf.append(f"- **Type**: `{finding.vuln_type}`")
    buf.append(f"- **Location**: `{finding.location}`")
    if finding.cvss_base_score is not None:
        cvss_str = f"{finding.cvss_base_score}"
        if finding.cvss_vector:
            cvss_str += f" ({finding.cvss_vector})"
        buf.append(f"- **CVSS**: {cvss_str}")
    if finding.cwe_id:
        buf.append(f"- **CWE**: {finding.cwe_id}")
    if finding.cve_id:
        buf.append(f"- **CVE**: {finding.cve_id}")
    if finding.wstg_test_id:
        buf.append(f"- **WSTG**: {finding.wstg_test_id}")
    if finding.mitre_attack_technique:
        tactics = f" ({finding.mitre_attack_tactic})" if finding.mitre_attack_tactic else ""
        buf.append(f"- **MITRE ATT&CK**: {finding.mitre_attack_technique}{tactics}")
    if finding.poc_status and finding.poc_status != "not_attempted":
        buf.append(f"- **PoC Status**: {finding.poc_status}")
    buf.append(f"- **Verified**: {'yes' if finding.verified else 'no'}")
    buf.append(f"- **Scan**: `{scan_tag or finding.scan_id or '(unknown)'}`")
    if finding.description:
        buf.append("")
        buf.append("**Description**:")
        buf.append("")
        buf.append(finding.description)
    if finding.remediation:
        buf.append("")
        buf.append("**Recommendation**:")
        buf.append("")
        buf.append(finding.remediation)
    buf.append("")
    buf.append("---")
    buf.append("")


def _build_markdown_for_group(
    group_name: str,
    findings: list[Finding],
    scan_tags_by_id: dict[str | None, str | None],
) -> str:
    """Build Markdown for a single group of findings.

    Returns the markdown content (without the file header — caller
    adds that).
    """
    buf: list[str] = []
    # Group header
    buf.append(f"## {group_name} ({len(findings)} finding{'s' if len(findings) != 1 else ''})")
    buf.append("")
    # Severity breakdown
    severity_counts: dict[str, int] = {}
    for f in findings:
        sev = (f.severity or "info").lower()
        severity_counts[sev] = severity_counts.get(sev, 0) + 1
    if severity_counts:
        sev_summary = " · ".join(
            f"{sev}: {count}"
            for sev, count in sorted(
                severity_counts.items(), key=lambda x: _severity_sort_key(x[0])
            )
        )
        buf.append(f"**Severity breakdown**: {sev_summary}")
        buf.append("")

    # Sort findings by severity (critical first)
    findings_sorted = sorted(findings, key=lambda f: _severity_sort_key(f.severity))

    # Each finding as a ### block
    for finding in findings_sorted:
        scan_tag = scan_tags_by_id.get(finding.scan_id)
        _append_vulnerability_markdown(buf, finding, scan_tag)

    return "\n".join(buf)


# ── Endpoints ────────────────────────────────────────────────────────────


@router.get("/vulnerabilities/export")
async def export_vulnerabilities(
    group_by: str = Query(
        "scan",
        description="Group findings by: scan | severity | vuln_type | none",
    ),
    mode: str = Query(
        "summary",
        description="Output mode: summary (one file with all groups) | split (one file per group)",
    ),
    scan_id: str | None = Query(None, description="Filter by scan ID"),
    severity: str | None = Query(
        None, description="Filter by severity (critical|high|medium|low|info)"
    ),
    verified: str | None = Query(
        None, description="Filter by verified status (true|false)"
    ),
    limit: int = Query(
        1000, ge=1, le=10000, description="Max findings to export (1-10000, default 1000)"
    ),
    session: AsyncSession = Depends(get_async_session),
) -> dict[str, Any]:
    """Export vulnerabilities as Markdown.

    Mirrors CyberStrikeAI's GET /api/vulnerabilities/export endpoint
    (internal/handler/vulnerability.go:417-506).

    On-demand — user clicks Export button in the Findings tab. Returns
    JSON with file contents inline so the frontend can download via
    Blob + saveAs (no server-side file writing required).

    Args:
        group_by: scan | severity | vuln_type | none
            - scan (default): group findings by their origin scan (uses
              scan_tag snapshot for scans that have been deleted)
            - severity: group by critical/high/medium/low/info
            - vuln_type: group by vulnerability type (sqli, xss, rce, ...)
            - none: no grouping, all findings in one block
        mode: summary | split
            - summary (default): one Markdown file containing all groups
              with `## {group} (count)` headers
            - split: one Markdown file per group (returned as separate
              entries in the `files` array)
        scan_id: filter findings by scan ID
        severity: filter by severity
        verified: filter by verified status
        limit: max findings to export (default 1000)

    Response shape:
        {
          "mode": "summary",
          "group_by": "scan",
          "total": N,
          "files": [
            {"filename": "vulnerabilities-summary-20260926-120000.md", "content": "..."}
          ]
        }

    For mode=split, files will have one entry per group.
    """
    # Validate group_by + mode
    valid_group_by = {"scan", "severity", "vuln_type", "none"}
    if group_by not in valid_group_by:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid group_by={group_by!r}. Valid: {sorted(valid_group_by)}",
        )
    valid_modes = {"summary", "split"}
    if mode not in valid_modes:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid mode={mode!r}. Valid: {sorted(valid_modes)}",
        )

    # Build WHERE clause from filters
    conditions = []
    if scan_id:
        conditions.append(Finding.scan_id == scan_id)
    if severity:
        conditions.append(Finding.severity.ilike(f"%{severity}%"))
    if verified is not None:
        if verified.lower() == "true":
            conditions.append(Finding.verified == True)  # noqa: E712
        elif verified.lower() == "false":
            conditions.append(Finding.verified == False)  # noqa: E712

    # Query findings
    stmt = select(Finding).order_by(Finding.created_at.desc()).limit(limit)
    if conditions:
        stmt = stmt.where(and_(*conditions))
    result = await session.execute(stmt)
    findings = result.scalars().all()

    if not findings:
        return {
            "mode": mode,
            "group_by": group_by,
            "total": 0,
            "files": [],
            "message": "No findings match the filter criteria.",
        }

    # Build scan_tags lookup (for scan grouping + scan_tag column on each finding)
    scan_ids_with_findings = {f.scan_id for f in findings if f.scan_id is not None}
    scan_tags_by_id: dict[str | None, str | None] = {}
    if scan_ids_with_findings:
        scan_rows = (await session.execute(
            select(Scan.id, Scan.target).where(Scan.id.in_(scan_ids_with_findings))
        )).all()
        for sid, target in scan_rows:
            scan_tags_by_id[sid] = target or sid
    # For findings with NULL scan_id (scan was deleted), use scan_tag column
    for finding in findings:
        if finding.scan_id is None and finding.scan_tag:
            scan_tags_by_id[None] = finding.scan_tag

    # Group findings
    groups: dict[str, list[Finding]] = {}
    if group_by == "scan":
        for f in findings:
            # Prefer scan_tag for human-readable label, fall back to scan_id
            key = scan_tags_by_id.get(f.scan_id) or f.scan_tag or f.scan_id or "(no scan)"
            groups.setdefault(key, []).append(f)
    elif group_by == "severity":
        for f in findings:
            key = (f.severity or "info").lower()
            groups.setdefault(key, []).append(f)
        # Sort groups by severity (critical first)
        groups = dict(sorted(groups.items(), key=lambda x: _severity_sort_key(x[0])))
    elif group_by == "vuln_type":
        for f in findings:
            key = f.vuln_type or "unknown"
            groups.setdefault(key, []).append(f)
        # Sort groups alphabetically
        groups = dict(sorted(groups.items()))
    else:  # group_by == "none"
        groups = {"All findings": findings}

    # Generate timestamp for filenames
    timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")

    # Build file(s)
    files: list[dict[str, str]] = []
    if mode == "summary":
        # One file containing all groups
        buf: list[str] = []
        buf.append(f"# VAPT-AI Vulnerability Report")
        buf.append("")
        buf.append(f"**Generated**: {datetime.now(UTC).isoformat()}")
        buf.append(f"**Total findings**: {len(findings)}")
        buf.append(f"**Grouped by**: {group_by}")
        if scan_id:
            buf.append(f"**Filter**: scan_id={scan_id}")
        if severity:
            buf.append(f"**Filter**: severity={severity}")
        buf.append("")
        buf.append("---")
        buf.append("")
        for group_name, group_findings in groups.items():
            group_md = _build_markdown_for_group(group_name, group_findings, scan_tags_by_id)
            buf.append(group_md)
        filename = f"vulnerabilities-{group_by}-{timestamp}.md"
        files.append({"filename": filename, "content": "\n".join(buf)})
    else:  # mode == "split"
        # One file per group
        for group_name, group_findings in groups.items():
            buf = []
            buf.append(f"# VAPT-AI Vulnerability Report — {group_name}")
            buf.append("")
            buf.append(f"**Generated**: {datetime.now(UTC).isoformat()}")
            buf.append(f"**Findings in this group**: {len(group_findings)}")
            buf.append("")
            buf.append("---")
            buf.append("")
            group_md = _build_markdown_for_group(group_name, group_findings, scan_tags_by_id)
            buf.append(group_md)
            safe_group = _sanitize_filename(str(group_name))
            filename = f"vulnerabilities-{group_by}-{safe_group}-{timestamp}.md"
            files.append({"filename": filename, "content": "\n".join(buf)})

    return {
        "mode": mode,
        "group_by": group_by,
        "total": len(findings),
        "groups_count": len(groups),
        "files": files,
    }


# ── Scan results aggregation endpoint ─────────────────────────────────────
# Mirrors CyberStrikeAI's GET /api/conversations/{id}/results
# (internal/handler/openapi.go:6462-6503)


@router.get("/scans/{scan_id}/results")
async def get_scan_results(
    scan_id: str,
    include_process_details: int = Query(
        0, ge=0, le=1,
        description="0=skip process_details (default), 1=include lite process_details",
    ),
    process_details_limit: int = Query(
        100, ge=1, le=1000,
        description="When include_process_details=1, max process_details to return",
    ),
    session: AsyncSession = Depends(get_async_session),
) -> dict[str, Any]:
    """Get aggregated JSON results for a scan.

    Mirrors CyberStrikeAI's GET /api/conversations/{id}/results
    (internal/handler/openapi.go:6462-6503). Useful for one-shot API
    consumers (CI/CD pipelines, external dashboards) that don't want
    to paginate through /process-details.

    Response shape:
        {
          "scan_id": "...",
          "scan": {<scan summary>},
          "vulnerabilities": [<Finding>],
          "process_details": [<lite rows>]  # only if include_process_details=1
        }
    """
    # Verify scan exists
    scan = await session.get(Scan, scan_id)
    if scan is None:
        raise HTTPException(status_code=404, detail=f"Scan {scan_id!r} not found")

    # Fetch all findings for this scan
    findings_stmt = (
        select(Finding)
        .where(Finding.scan_id == scan_id)
        .order_by(Finding.severity.asc())  # critical first (alphabetical proximity)
    )
    findings_result = await session.execute(findings_stmt)
    findings = findings_result.scalars().all()

    # Build scan summary
    scan_summary = {
        "id": scan.id,
        "target": scan.target,
        "target_type": scan.target_type,
        "agent_mode": scan.agent_mode,
        "status": scan.status,
        "progress": scan.progress,
        "started_at": scan.started_at.isoformat() if scan.started_at else None,
        "completed_at": scan.completed_at.isoformat() if scan.completed_at else None,
        "user_prompt": scan.user_prompt,
        "result_summary": scan.result_summary,
    }

    # Build vulnerabilities list (full payload — this is the export-friendly view)
    vulns_list = [
        {
            "id": str(f.id),
            "name": f.name,
            "vuln_type": f.vuln_type,
            "severity": f.severity,
            "location": f.location,
            "cvss_vector": f.cvss_vector,
            "cvss_base_score": f.cvss_base_score,
            "cwe_id": f.cwe_id,
            "cve_id": f.cve_id,
            "wstg_test_id": f.wstg_test_id,
            "mitre_attack_technique": f.mitre_attack_technique,
            "mitre_attack_tactic": f.mitre_attack_tactic,
            "poc_status": f.poc_status,
            "verified": f.verified,
            "false_positive": f.false_positive,
            "description": f.description,
            "remediation": f.remediation,
            "confidence_score": f.confidence_score,
            "scan_id": f.scan_id,
            "scan_tag": f.scan_tag,
            "created_at": f.created_at.isoformat() if f.created_at else None,
        }
        for f in findings
    ]

    response: dict[str, Any] = {
        "scan_id": scan_id,
        "scan": scan_summary,
        "vulnerabilities": vulns_list,
        "vulnerabilities_count": len(findings),
    }

    # Optionally include lite process_details
    if include_process_details == 1:
        pd_stmt = (
            select(ProcessDetail)
            .where(ProcessDetail.scan_id == scan_id)
            .order_by(ProcessDetail.created_at.asc())
            .limit(process_details_limit)
        )
        pd_result = await session.execute(pd_stmt)
        process_details = pd_result.scalars().all()
        response["process_details"] = [
            {
                "id": str(pd.id),
                "event_type": pd.event_type,
                "message": pd.message,
                "created_at": pd.created_at.isoformat() if pd.created_at else None,
                "has_payload": pd.data is not None,
            }
            for pd in process_details
        ]
        response["process_details_returned"] = len(process_details)
        # Total count for pagination awareness
        total_pd = await session.scalar(
            select(func.count(ProcessDetail.id)).where(ProcessDetail.scan_id == scan_id)
        ) or 0
        response["process_details_total"] = total_pd

    return response