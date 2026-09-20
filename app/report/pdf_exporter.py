"""
VAPT-AI PDF Report Exporter (W13-S3).

Generates professional VAPT reports in PDF format using ReportLab.

Report structure (per W13-S2 confirmation):
    1. Cover Page (scan ID, target, date, mode)
    2. Executive Summary (findings by severity, risk score, top risks)
    3. Methodology (WSTG v4.2 coverage, agent phases, tool inventory)
    4. Findings Detail (per-finding: title, severity, CVSS, WSTG, MITRE,
       CWE, location, evidence, PoC status, recommendations)
    5. Audit Trail (chain-of-custody verification, scope violations, HITL approvals)
    6. Appendix (tool inventory, agent decisions log)

Reference sources (per master plan §10 + §5.1):
    - CyberStrikeAI: no report generator (Go codebase); VAPT-AI implements
      its own using ReportLab 4.1+ (per §5.1 tech stack).
    - EVVO Sentinel: shield_engine/report_generator.py (DOCX via template)
      + html_report_generator.py (HTML). VAPT-AI ports the severity weighting
      logic + risk score calculation, but switches output to PDF (industry
      standard for pentest reports).

Output location: reports/ directory at project root.
    reports/scan_<id>_<timestamp>.pdf

Usage:
    from app.report import generate_pdf_report
    async with async_session() as session:
        pdf_path = await generate_pdf_report(scan_id, session)
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime, UTC
from pathlib import Path
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.report.collector import (
    collect_scan_data,
    ScanReportData,
    FindingReportData,
    EvidenceReportData,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Severity → color (used in PDF tables and headers)
SEVERITY_COLORS: dict[str, tuple[float, float, float]] = {
    "critical": (0.725, 0.110, 0.110),   # #B91C1C — red
    "high":     (0.760, 0.255, 0.047),    # #C2410C — orange
    "medium":   (0.631, 0.386, 0.027),    # #A16207 — amber
    "low":      (0.082, 0.502, 0.235),    # #15803D — green
    "info":     (0.322, 0.322, 0.322),    # #525252 — gray
}

# Severity → weight (for risk score calculation)
SEVERITY_WEIGHTS: dict[str, int] = {
    "critical": 10,
    "high": 7,
    "medium": 4,
    "low": 2,
    "info": 0,
}

# Severity → rank (for sorting findings in report)
SEVERITY_RANK: dict[str, int] = {
    "critical": 5,
    "high": 4,
    "medium": 3,
    "low": 2,
    "info": 1,
}

# Default report output directory (relative to project root)
REPORTS_DIR = Path(__file__).resolve().parent.parent.parent / "reports"

# Page layout
PAGE_WIDTH, PAGE_HEIGHT = 595, 842  # A4 in points (1pt = 1/72 inch)
MARGIN = 50  # 50pt margin all sides
CONTENT_WIDTH = PAGE_WIDTH - 2 * MARGIN


# ---------------------------------------------------------------------------
# Risk scoring (ported from EVVO html_report_generator._calculate_risk_level)
# ---------------------------------------------------------------------------

def calculate_risk_level(findings: list[FindingReportData]) -> tuple[str, int]:
    """Calculate overall risk level + score from findings.

    Ported from EVVO shield_engine/html_report_generator.py.
    Score = sum of severity weights (capped at 100).

    Returns (risk_level, risk_score):
        - risk_level: "Critical" | "High" | "Medium" | "Low"
        - risk_score: 0-100
    """
    score = sum(
        SEVERITY_WEIGHTS.get(f.severity.lower(), 0)
        for f in findings if not f.false_positive
    )
    score = min(score, 100)
    if score >= 30:
        return "Critical", score
    elif score >= 15:
        return "High", score
    elif score >= 5:
        return "Medium", score
    return "Low", score


# ---------------------------------------------------------------------------
# Exporter
# ---------------------------------------------------------------------------

class PDFExporter:
    """PDF report exporter for VAPT-AI scan results.

    One exporter instance per scan. Use generate_pdf_report() convenience
    function for the common case.
    """

    def __init__(self, output_dir: Path | None = None):
        self.output_dir = output_dir or REPORTS_DIR

    async def export(
        self, scan_id: str, session: AsyncSession,
        output_path: str | Path | None = None,
    ) -> Path:
        """Generate PDF report for a scan.

        Args:
            scan_id: Scan ID
            session: Async DB session
            output_path: Optional explicit output path. If None, auto-generated
                as reports/scan_<id>_<timestamp>.pdf

        Returns:
            Path to the generated PDF file.

        Raises:
            ValueError: if scan not found.
        """
        data = await collect_scan_data(scan_id, session)
        return self._render(data, output_path)

    def _render(self, data: ScanReportData, output_path: str | Path | None) -> Path:
        """Render the PDF report from collected data."""
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import mm
        from reportlab.lib.colors import HexColor
        from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
            PageBreak, KeepTogether,
        )
        from reportlab.pdfgen import canvas
        from reportlab.lib import colors as rl_colors

        if output_path is None:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
            output_path = self.output_dir / f"scan_{data.scan_id}_{ts}.pdf"
        else:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

        doc = SimpleDocTemplate(
            str(output_path),
            pagesize=A4,
            leftMargin=MARGIN, rightMargin=MARGIN,
            topMargin=MARGIN, bottomMargin=MARGIN,
            title=f"VAPT-AI Report — Scan {data.scan_id}",
            author="VAPT-AI v3.2",
            subject=f"Pentest report for {data.target}",
            creator="VAPT-AI v3.2 (ReportLab)",
        )

        # Styles
        styles = getSampleStyleSheet()
        style_cover_title = ParagraphStyle(
            "CoverTitle", parent=styles["Title"],
            fontSize=28, leading=34, alignment=TA_CENTER, spaceAfter=20,
            textColor=HexColor("#0F172A"),
        )
        style_cover_subtitle = ParagraphStyle(
            "CoverSubtitle", parent=styles["Normal"],
            fontSize=14, leading=20, alignment=TA_CENTER, spaceAfter=10,
            textColor=HexColor("#475569"),
        )
        style_h1 = ParagraphStyle(
            "H1", parent=styles["Heading1"],
            fontSize=18, leading=24, spaceAfter=12, spaceBefore=20,
            textColor=HexColor("#0F172A"),
        )
        style_h2 = ParagraphStyle(
            "H2", parent=styles["Heading2"],
            fontSize=14, leading=18, spaceAfter=8, spaceBefore=12,
            textColor=HexColor("#1E40AF"),
        )
        style_body = ParagraphStyle(
            "Body", parent=styles["Normal"],
            fontSize=10, leading=14, alignment=TA_JUSTIFY, spaceAfter=8,
        )
        style_code = ParagraphStyle(
            "Code", parent=styles["Code"],
            fontSize=9, leading=12, leftIndent=12, spaceAfter=8,
            textColor=HexColor("#1F2937"),
            backColor=HexColor("#F3F4F6"),
            borderPadding=4,
        )
        style_finding_title = ParagraphStyle(
            "FindingTitle", parent=styles["Heading3"],
            fontSize=12, leading=15, spaceAfter=4, spaceBefore=12,
            textColor=HexColor("#0F172A"),
        )

        story: list[Any] = []

        # ============ 1. Cover Page ============
        story.append(Spacer(1, 80))
        story.append(Paragraph("VAPT-AI v3.2", style_cover_title))
        story.append(Paragraph("Vulnerability Assessment &amp; Penetration Testing Report", style_cover_subtitle))
        story.append(Spacer(1, 40))

        risk_level, risk_score = calculate_risk_level(data.findings)
        cover_table_data = [
            ["Scan ID",       data.scan_id],
            ["Target",        data.target],
            ["Target Type",   data.target_type],
            ["Agent Mode",    data.agent_mode],
            ["HITL Mode",     data.hitl_mode],
            ["Status",        data.status],
            ["Started At",    data.started_at.strftime("%Y-%m-%d %H:%M:%S UTC") if data.started_at else "N/A"],
            ["Completed At",  data.completed_at.strftime("%Y-%m-%d %H:%M:%S UTC") if data.completed_at else "N/A"],
            ["Duration",      self._format_duration(data.started_at, data.completed_at)],
            ["Findings Total", str(len(data.findings))],
            ["Risk Level",    f"{risk_level} ({risk_score}/100)"],
            ["Generated At",  datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")],
        ]
        cover_table = Table(cover_table_data, colWidths=[140, CONTENT_WIDTH - 140])
        cover_table.setStyle(TableStyle([
            ("FONT", (0, 0), (0, -1), "Helvetica-Bold", 10),
            ("FONT", (1, 0), (1, -1), "Helvetica", 10),
            ("TEXTCOLOR", (0, 0), (0, -1), HexColor("#475569")),
            ("TEXTCOLOR", (1, 0), (1, -1), HexColor("#0F172A")),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("LINEBELOW", (0, 0), (-1, -2), 0.5, HexColor("#E2E8F0")),
        ]))
        story.append(cover_table)

        if data.user_prompt:
            story.append(Spacer(1, 30))
            story.append(Paragraph("<b>User Prompt:</b>", style_body))
            story.append(Paragraph(self._escape(data.user_prompt), style_body))

        story.append(PageBreak())

        # ============ 2. Executive Summary ============
        story.append(Paragraph("1. Executive Summary", style_h1))

        findings_by_severity = self._count_by_severity(data.findings)
        total_findings = len(data.findings)
        verified_count = sum(1 for f in data.findings if f.verified)
        fp_count = sum(1 for f in data.findings if f.false_positive)

        summary_text = (
            f"This report presents the results of an automated penetration test "
            f"conducted by VAPT-AI v3.2 against the target <b>{self._escape(data.target)}</b>. "
            f"The scan was executed in <b>{data.agent_mode}</b> mode with HITL "
            f"<b>{data.hitl_mode}</b> policy. The assessment identified a total of "
            f"<b>{total_findings}</b> findings, of which <b>{verified_count}</b> were "
            f"verified through successful PoC exploitation and <b>{fp_count}</b> were "
            f"filtered as false positives by the evidence auditor."
        )
        story.append(Paragraph(summary_text, style_body))

        risk_text = (
            f"The overall risk level is assessed as <b>{risk_level}</b> "
            f"(risk score: {risk_score}/100). "
            + (
                f"This indicates a critical exposure requiring immediate remediation. "
                if risk_level == "Critical" else
                f"This indicates significant security weaknesses that should be "
                f"addressed in a timely manner."
                if risk_level in ("High", "Medium") else
                f"The target demonstrates reasonable security posture."
            )
        )
        story.append(Paragraph(risk_text, style_body))

        # Findings by severity table
        story.append(Paragraph("Findings by Severity", style_h2))
        sev_table_data = [["Severity", "Count", "Percentage", "Risk Weight"]]
        for sev in ["critical", "high", "medium", "low", "info"]:
            count = findings_by_severity.get(sev, 0)
            pct = f"{(count / total_findings * 100):.1f}%" if total_findings > 0 else "0%"
            weight = SEVERITY_WEIGHTS.get(sev, 0)
            sev_table_data.append([
                sev.upper(),
                str(count),
                pct,
                str(weight * count),
            ])
        sev_table_data.append(["TOTAL", str(total_findings), "100%", str(risk_score)])
        sev_table = Table(sev_table_data, colWidths=[120, 80, 120, 100])
        sev_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), HexColor("#1E40AF")),
            ("TEXTCOLOR", (0, 0), (-1, 0), HexColor("#FFFFFF")),
            ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 10),
            ("FONT", (0, 1), (-1, -1), "Helvetica", 10),
            ("BACKGROUND", (0, -1), (-1, -1), HexColor("#F3F4F6")),
            ("FONT", (0, -1), (-1, -1), "Helvetica-Bold", 10),
            ("ALIGN", (1, 0), (-1, -1), "CENTER"),
            ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#CBD5E1")),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
        ]))
        # Color severity rows
        for i, sev in enumerate(["critical", "high", "medium", "low", "info"], start=1):
            r, g, b = SEVERITY_COLORS.get(sev, (0.5, 0.5, 0.5))
            sev_table.setStyle(TableStyle([
                ("TEXTCOLOR", (0, i), (0, i), rl_colors.Color(r, g, b)),
                ("FONT", (0, i), (0, i), "Helvetica-Bold", 10),
            ]))
        story.append(sev_table)

        # Top 3 critical/high findings
        top_findings = sorted(
            [f for f in data.findings if not f.false_positive],
            key=lambda f: SEVERITY_RANK.get(f.severity.lower(), 0),
            reverse=True,
        )[:3]
        if top_findings:
            story.append(Spacer(1, 12))
            story.append(Paragraph("Top Risks", style_h2))
            for f in top_findings:
                story.append(Paragraph(
                    f"• <b>[{f.severity.upper()}]</b> {self._escape(f.name)} "
                    f"— <i>{self._escape(f.location)}</i> "
                    f"(CVSS: {f.cvss_base_score or 'N/A'})",
                    style_body,
                ))

        story.append(PageBreak())

        # ============ 3. Methodology ============
        story.append(Paragraph("2. Methodology", style_h1))

        methodology_text = (
            "VAPT-AI v3.2 follows the OWASP Web Security Testing Guide (WSTG) v4.2 "
            "methodology, supplemented by MITRE ATT&amp;CK Enterprise techniques for "
            "post-exploitation phases. The scan was orchestrated by a LangGraph "
            "multi-agent runtime with the following kill-chain phases:"
        )
        story.append(Paragraph(methodology_text, style_body))

        phases = [
            ("Recon", "Reconnaissance agent identifies open ports, services, and tech stack."),
            ("Attack Surface Enumeration", "Enumerates endpoints, subdomains, and parameters."),
            ("Vulnerability Triage", "Cross-references findings with WSTG v4.2 and CVE feeds."),
            ("Penetration", "Exploitation of confirmed vulnerabilities (HITL-gated)."),
            ("Privilege Escalation", "Post-exploitation lateral movement (HITL-gated)."),
            ("Reporting &amp; Remediation", "Aggregates evidence chain and generates this report."),
        ]
        phase_table_data = [["#", "Phase", "Description"]]
        for i, (name, desc) in enumerate(phases, start=1):
            phase_table_data.append([str(i), name, desc])
        phase_table = Table(phase_table_data, colWidths=[30, 160, CONTENT_WIDTH - 190])
        phase_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), HexColor("#1E40AF")),
            ("TEXTCOLOR", (0, 0), (-1, 0), HexColor("#FFFFFF")),
            ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 10),
            ("FONT", (0, 1), (-1, -1), "Helvetica", 10),
            ("ALIGN", (0, 0), (0, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#CBD5E1")),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
        ]))
        story.append(phase_table)

        story.append(Spacer(1, 12))
        story.append(Paragraph("Evidence Chain (D8)", style_h2))
        story.append(Paragraph(
            "Each finding is backed by a 5-layer evidence chain: "
            "(1) Detection — tool output, (2) Validation — verifier strategy result, "
            "(3) Exploitation — PoC execution, (4) Post-Exploitation — actions on objective, "
            "(5) Audit — chain-of-custody HMAC seal. Findings failing the 4-dim confidence "
            "threshold (0.6) are rejected as false positives.",
            style_body,
        ))

        story.append(PageBreak())

        # ============ 4. Findings Detail ============
        story.append(Paragraph("3. Findings Detail", style_h1))

        if not data.findings:
            story.append(Paragraph(
                "No findings were recorded for this scan. This may indicate either "
                "(a) the target was fully secured against the tested attack surface, or "
                "(b) the orchestrator agent stubs were used without real LLM-driven findings "
                "(W10/W11 stub mode — see W13 pipeline for the test fixture path).",
                style_body,
            ))
        else:
            sorted_findings = sorted(
                data.findings,
                key=lambda f: SEVERITY_RANK.get(f.severity.lower(), 0),
                reverse=True,
            )
            for idx, f in enumerate(sorted_findings, start=1):
                story.append(KeepTogether(self._build_finding_block(
                    f, idx, style_finding_title, style_body, style_code, style_h2,
                )))

        story.append(PageBreak())

        # ============ 5. Audit Trail ============
        story.append(Paragraph("4. Audit Trail", style_h1))

        if data.consent:
            story.append(Paragraph("Consent &amp; Authorization", style_h2))
            story.append(Paragraph(
                f"Asserted Owner: <b>{self._escape(str(data.consent.get('asserted_owner', 'N/A')))}</b><br/>"
                f"Verification Method: <b>{self._escape(str(data.consent.get('verification_method', 'N/A')))}</b><br/>"
                f"ToS Accepted At: <b>{self._escape(str(data.consent.get('tos_accepted_at', 'N/A')))}</b>",
                style_body,
            ))
            story.append(Spacer(1, 8))

        story.append(Paragraph("Chain-of-Custody Verification", style_h2))
        custody_verified = 0
        custody_failed = 0
        for f in data.findings:
            for ev in f.evidence:
                if ev.custody_seal and ev.evidence_hash:
                    custody_verified += 1
                else:
                    custody_failed += 1
        story.append(Paragraph(
            f"Total evidence entries verified: <b>{custody_verified}</b><br/>"
            f"Entries failing verification: <b>{custody_failed}</b>",
            style_body,
        ))

        story.append(Paragraph("HITL Approvals", style_h2))
        if data.hitl_approvals:
            hitl_table_data = [["Tool", "Target", "Decision", "Timestamp"]]
            for h in data.hitl_approvals:
                hitl_table_data.append([
                    self._escape(str(h.get("tool_name", "N/A")))[:30],
                    self._escape(str(h.get("target", "N/A")))[:30],
                    str(h.get("user_decision") or h.get("status", "N/A")),
                    str(h.get("decided_at") or h.get("timestamp", "N/A"))[:19],
                ])
            hitl_table = Table(hitl_table_data, colWidths=[100, 130, 80, 110])
            hitl_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), HexColor("#1E40AF")),
                ("TEXTCOLOR", (0, 0), (-1, 0), HexColor("#FFFFFF")),
                ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 9),
                ("FONT", (0, 1), (-1, -1), "Helvetica", 9),
                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#CBD5E1")),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
            ]))
            story.append(hitl_table)
        else:
            story.append(Paragraph(
                "No HITL approvals were triggered for this scan "
                "(no destructive operations requested).",
                style_body,
            ))

        story.append(Paragraph("Audit Log Entries", style_h2))
        story.append(Paragraph(
            f"Total audit log entries: <b>{len(data.audit_entries)}</b>",
            style_body,
        ))
        if data.audit_entries:
            audit_table_data = [["#", "Action", "Actor", "Timestamp"]]
            for i, a in enumerate(data.audit_entries[:20], start=1):
                audit_table_data.append([
                    str(i),
                    self._escape(str(a.get("action", "N/A")))[:40],
                    self._escape(str(a.get("actor_id", "N/A")))[:20],
                    str(a.get("timestamp", "N/A"))[:19],
                ])
            audit_table = Table(audit_table_data, colWidths=[30, 200, 120, 110])
            audit_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), HexColor("#1E40AF")),
                ("TEXTCOLOR", (0, 0), (-1, 0), HexColor("#FFFFFF")),
                ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 9),
                ("FONT", (0, 1), (-1, -1), "Helvetica", 9),
                ("ALIGN", (0, 0), (0, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#CBD5E1")),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
            ]))
            story.append(audit_table)
            if len(data.audit_entries) > 20:
                story.append(Paragraph(
                    f"<i>... showing first 20 of {len(data.audit_entries)} entries</i>",
                    style_body,
                ))

        story.append(PageBreak())

        # ============ 6. Appendix ============
        story.append(Paragraph("5. Appendix", style_h1))

        story.append(Paragraph("Tool Inventory", style_h2))
        tools_used: set[str] = set()
        for f in data.findings:
            for ev in f.evidence:
                if ev.tool_used:
                    tools_used.add(ev.tool_used)
        if tools_used:
            tool_data = [["#", "Tool", "Evidence Count"]]
            tool_counts: dict[str, int] = {}
            for f in data.findings:
                for ev in f.evidence:
                    if ev.tool_used:
                        tool_counts[ev.tool_used] = tool_counts.get(ev.tool_used, 0) + 1
            for i, (tool, count) in enumerate(sorted(tool_counts.items()), start=1):
                tool_data.append([str(i), tool, str(count)])
            tool_table = Table(tool_data, colWidths=[30, 200, 120])
            tool_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), HexColor("#1E40AF")),
                ("TEXTCOLOR", (0, 0), (-1, 0), HexColor("#FFFFFF")),
                ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 9),
                ("FONT", (0, 1), (-1, -1), "Helvetica", 9),
                ("ALIGN", (0, 0), (0, -1), "CENTER"),
                ("ALIGN", (2, 0), (2, -1), "CENTER"),
                ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#CBD5E1")),
            ]))
            story.append(tool_table)
        else:
            story.append(Paragraph(
                "No tool execution evidence recorded.",
                style_body,
            ))

        story.append(Spacer(1, 12))
        story.append(Paragraph("Scan Metadata", style_h2))
        meta_data = [
            ["VAPT-AI Version", "3.2.0"],
            ["Report Generator", "ReportLab 4.1+"],
            ["SARIF Version", "2.1.0 (OASIS)"],
            ["CVSS Version", "v3.1 (FIRST.org)"],
            ["WSTG Version", "v4.2 (OWASP)"],
            ["MITRE ATT&CK", "Enterprise v13.1"],
            ["Generated At", datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")],
        ]
        meta_table = Table(meta_data, colWidths=[160, CONTENT_WIDTH - 160])
        meta_table.setStyle(TableStyle([
            ("FONT", (0, 0), (0, -1), "Helvetica-Bold", 10),
            ("FONT", (1, 0), (1, -1), "Helvetica", 10),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("LINEBELOW", (0, 0), (-1, -2), 0.5, HexColor("#E2E8F0")),
        ]))
        story.append(meta_table)

        # Build the PDF
        doc.build(story)
        logger.info("PDF report generated: scan=%s path=%s findings=%d",
                    data.scan_id, output_path, len(data.findings))
        return output_path

    # ---------- Helpers ----------

    def _build_finding_block(
        self, f: FindingReportData, idx: int,
        style_title, style_body, style_code, style_h2,
    ) -> list[Any]:
        """Build a single finding's PDF block (kept together on one page)."""
        from reportlab.platypus import Paragraph, Spacer, Table, TableStyle
        from reportlab.lib.colors import HexColor

        block: list[Any] = []
        sev = f.severity.lower()
        r, g, b = SEVERITY_COLORS.get(sev, (0.5, 0.5, 0.5))

        # Title bar
        block.append(Paragraph(
            f"<b>#{idx} [{f.severity.upper()}]</b> {self._escape(f.name)}",
            style_title,
        ))

        # Detail table
        detail_data = [
            ["Vulnerability Type", f.vuln_type],
            ["Severity", f.severity.upper()],
            ["Location", self._escape(f.location)],
            ["CVSS Vector", f.cvss_vector or "N/A"],
            ["CVSS Base Score", f"{f.cvss_base_score:.1f}" if f.cvss_base_score else "N/A"],
            ["CVSS Severity", f.cvss_severity or "N/A"],
            ["WSTG ID", f.wstg_test_id or "N/A"],
            ["CWE ID", f.cwe_id or "N/A"],
            ["CVE ID", f.cve_id or "N/A"],
            ["MITRE ATT&CK Technique", f.mitre_attack_technique or "N/A"],
            ["MITRE ATT&CK Tactic", f.mitre_attack_tactic or "N/A"],
            ["PoC Status", f.poc_status],
            ["PoC Tier", str(f.poc_tier) if f.poc_tier else "N/A"],
            ["Verified", "Yes" if f.verified else "No"],
            ["False Positive", "Yes" if f.false_positive else "No"],
            ["Auditor Verdict", f.auditor_verdict or "N/A"],
            ["Confidence Score", f"{f.confidence_score:.2f}"],
            ["Exploit Method", f.exploit_method or "N/A"],
        ]
        detail_table = Table(detail_data, colWidths=[160, CONTENT_WIDTH - 160])
        detail_table.setStyle(TableStyle([
            ("FONT", (0, 0), (0, -1), "Helvetica-Bold", 9),
            ("FONT", (1, 0), (1, -1), "Helvetica", 9),
            ("BACKGROUND", (0, 1), (0, 1), HexColor("#FEF2F2") if sev == "critical" else HexColor("#FFFFFF")),
            ("TEXTCOLOR", (1, 1), (1, 1), HexColor("#B91C1C") if sev == "critical" else HexColor("#0F172A")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#CBD5E1")),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ]))
        block.append(detail_table)
        block.append(Spacer(1, 6))

        # Evidence
        if f.evidence:
            block.append(Paragraph("<b>Evidence Chain</b>", style_body))
            for ev in f.evidence:
                raw = ev.raw_output
                if len(raw) > 500:
                    raw = raw[:500] + "... [truncated]"
                block.append(Paragraph(
                    f"<b>[{ev.layer}]</b> {ev.tool_used}:",
                    style_body,
                ))
                block.append(Paragraph(self._escape(raw), style_code))
                captured_str = (
                    ev.captured_at.strftime("%Y-%m-%d %H:%M:%S UTC")
                    if hasattr(ev.captured_at, "strftime")
                    else str(ev.captured_at)[:19]
                )
                block.append(Paragraph(
                    f"<i>Hash: {ev.evidence_hash[:32]}... | "
                    f"Seal: {ev.custody_seal[:32]}... | "
                    f"Captured: {captured_str}</i>",
                    style_body,
                ))

        # Remediation
        if f.remediation:
            block.append(Spacer(1, 6))
            block.append(Paragraph("<b>Remediation</b>", style_body))
            block.append(Paragraph(self._escape(f.remediation), style_body))

        block.append(Spacer(1, 16))
        return block

    def _count_by_severity(self, findings: list[FindingReportData]) -> dict[str, int]:
        counts: dict[str, int] = {"critical": 0, "high": 0, "medium": 0, "low": 0, "info": 0}
        for f in findings:
            sev = f.severity.lower()
            if sev in counts:
                counts[sev] += 1
        return counts

    def _format_duration(self, started, completed) -> str:
        if not started or not completed:
            return "N/A"
        try:
            delta = completed - started
            hours, rem = divmod(int(delta.total_seconds()), 3600)
            minutes, seconds = divmod(rem, 60)
            return f"{hours}h {minutes}m {seconds}s"
        except Exception:
            return "N/A"

    @staticmethod
    def _escape(text: str) -> str:
        """Escape XML special chars for ReportLab Paragraph."""
        if text is None:
            return ""
        return (
            str(text)
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------

async def generate_pdf_report(
    scan_id: str, session: AsyncSession,
    output_path: str | Path | None = None,
) -> Path | None:
    """Generate PDF report for a scan.

    Args:
        scan_id: Scan ID
        session: Async DB session
        output_path: Optional explicit output path

    Returns:
        Path to generated PDF file, or None if scan not found / generation failed.
    """
    try:
        exporter = PDFExporter()
        return await exporter.export(scan_id, session, output_path)
    except ValueError as e:
        logger.warning("PDF export skipped: %s", e)
        return None
    except Exception as e:
        logger.exception("PDF export failed: scan=%s", scan_id)
        return None


def generate_pdf_report_sync(
    scan_data_dict: dict[str, Any],
    output_path: str | Path,
) -> Path:
    """Synchronous PDF export from a plain dict (for unit tests).

    Args:
        scan_data_dict: Scan data as plain dict (must match ScanReportData.to_dict())
        output_path: Output file path

    Returns:
        Path to generated PDF file.
    """
    from app.report.collector import ScanReportData, FindingReportData, EvidenceReportData

    # Reconstruct dataclasses from dict
    findings: list[FindingReportData] = []
    for f_dict in scan_data_dict.get("findings", []):
        ev_list: list[EvidenceReportData] = []
        for ev in f_dict.get("evidence", []):
            if isinstance(ev, str):
                continue
            ev_kwargs = dict(ev)
            if isinstance(ev_kwargs.get("captured_at"), str):
                try:
                    ev_kwargs["captured_at"] = datetime.fromisoformat(ev_kwargs["captured_at"])
                except (ValueError, TypeError):
                    ev_kwargs["captured_at"] = datetime.now(UTC)
            ev_list.append(EvidenceReportData(**ev_kwargs))
        f = FindingReportData(
            id=uuid.UUID(f_dict["id"]) if "id" in f_dict else uuid.uuid4(),
            name=f_dict.get("name", ""),
            vuln_type=f_dict.get("vuln_type", ""),
            severity=f_dict.get("severity", "info"),
            cvss_vector=f_dict.get("cvss_vector"),
            cvss_base_score=f_dict.get("cvss_base_score"),
            cvss_severity=f_dict.get("cvss_severity"),
            location=f_dict.get("location", ""),
            cwe_id=f_dict.get("cwe_id"),
            cve_id=f_dict.get("cve_id"),
            wstg_test_id=f_dict.get("wstg_test_id"),
            mitre_attack_technique=f_dict.get("mitre_attack_technique"),
            mitre_attack_tactic=f_dict.get("mitre_attack_tactic"),
            mitre_attack_subtechnique=f_dict.get("mitre_attack_subtechnique"),
            poc_status=f_dict.get("poc_status", "not_attempted"),
            poc_tier=f_dict.get("poc_tier"),
            exploit_method=f_dict.get("exploit_method"),
            remediation=f_dict.get("remediation"),
            verified=f_dict.get("verified", False),
            false_positive=f_dict.get("false_positive", False),
            auditor_verdict=f_dict.get("auditor_verdict"),
            confidence_score=f_dict.get("confidence_score", 0.0),
            evidence=ev_list,
        )
        findings.append(f)

    data = ScanReportData(
        scan_id=scan_data_dict.get("scan_id", "unknown"),
        target=scan_data_dict.get("target", ""),
        target_type=scan_data_dict.get("target_type", ""),
        agent_mode=scan_data_dict.get("agent_mode", "supervisor"),
        hitl_mode=scan_data_dict.get("hitl_mode", "audit_agent"),
        user_prompt=scan_data_dict.get("user_prompt"),
        status=scan_data_dict.get("status", "completed"),
        progress=scan_data_dict.get("progress", 100),
        started_at=None,
        completed_at=None,
        result_summary=scan_data_dict.get("result_summary"),
        error=scan_data_dict.get("error"),
        findings=findings,
        audit_entries=scan_data_dict.get("audit_entries", []),
        hitl_approvals=scan_data_dict.get("hitl_approvals", []),
        consent=scan_data_dict.get("consent"),
    )

    exporter = PDFExporter()
    return exporter._render(data, output_path)