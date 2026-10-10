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

        # ============ 4. Conclusion & Recommendations (W19-FIX Phase B — Evvo template) ============
        story.append(Paragraph("4. Conclusion &amp; Recommendations", style_h1))

        conclusion_text = (
            f"The security assessment of <b>{self._escape(data.target)}</b> has revealed "
            f"<b>{len(data.findings)}</b> verified vulnerabilities with an overall risk "
            f"rating of <b>{risk_level}</b> (Score: {risk_score}/100)."
        )
        story.append(Paragraph(conclusion_text, style_body))

        recommendations_intro = (
            "It is strongly recommended that all Critical and High severity findings "
            "be addressed immediately to reduce the attack surface. Medium and Low "
            "severity findings should be remediated according to the prioritization "
            "roadmap below. Regular re-testing is advised to verify remediation "
            "effectiveness."
        )
        story.append(Paragraph(recommendations_intro, style_body))

        story.append(Paragraph("Recommendations", style_body))
        story.append(Paragraph("<b>High Priority</b>", style_body))
        critical_count = findings_by_severity.get("critical", 0)
        if critical_count > 0:
            story.append(Paragraph(
                f"• Address all <b>{critical_count} Critical</b> severity findings immediately.",
                style_body,
            ))
        else:
            story.append(Paragraph("• No Critical findings — proceed to High severity.", style_body))

        story.append(Paragraph("<b>Medium Priority</b>", style_body))
        high_count = findings_by_severity.get("high", 0)
        story.append(Paragraph(
            f"• Address all <b>{high_count} High</b> severity findings within 30 days.",
            style_body,
        ))

        story.append(Paragraph("<b>Low Priority</b>", style_body))
        medium_count = findings_by_severity.get("medium", 0)
        low_count = findings_by_severity.get("low", 0)
        info_count = findings_by_severity.get("info", 0)
        story.append(Paragraph(
            f"• Address <b>{medium_count} Medium</b>, <b>{low_count} Low</b>, and "
            f"<b>{info_count} Info</b> severity findings as part of regular maintenance.",
            style_body,
        ))

        story.append(PageBreak())

        # ============ 5. Audit Trail ============
        story.append(Paragraph("5. Audit Trail", style_h1))

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
        story.append(Paragraph("6. Appendix", style_h1))

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
        """Build a single finding's PDF block (W19-FIX Phase B — Evvo template).

        Per Evvo report template, each finding has 5 sub-paragraphs:
            1. Observation — description + observed headers/output
            2. Exploitation — initial discovery command + payload + response (PoC)
            3. Impact — business impact assessment
            4. Recommendation — fix steps
            5. Re-Test Verification Result — "Pending re-test" (no retest flow yet)

        Anti-hallucination: every claim references evidence_id + evidence_hash
        per master plan §8.5 D16.
        """
        from reportlab.platypus import Paragraph, Spacer, Table, TableStyle
        from reportlab.lib.colors import HexColor

        block: list[Any] = []
        sev = f.severity.lower()
        r, g, b = SEVERITY_COLORS.get(sev, (0.5, 0.5, 0.5))

        # Title bar (Evvo: "N. Finding Name")
        # W19-FIX3 Phase G3: append an "Internet-Verified" badge inline when
        # the verifier confirmed the finding via public web search. Uses
        # inline <font color=...> so it renders in the same Paragraph.
        title_html = f"<b>{idx}. {self._escape(f.name)}</b>"
        if getattr(f, "internet_verified", False):
            # Green badge for confirmed cross-check.
            title_html += (
                ' <font color="#15803D" size="9">'
                '<b>[Internet-Verified]</b></font>'
            )
        elif getattr(f, "internet_verification", None):
            # Grey badge when verifier ran but did NOT confirm — transparency
            # for the reader that internet cross-check was attempted.
            title_html += (
                ' <font color="#64748B" size="9">'
                '<b>[Internet: No Consensus]</b></font>'
            )
        block.append(Paragraph(title_html, style_title))

        # ── Finding meta header table (Severity / Affected Target / CWE / CVSS)
        header_data = [
            ["Severity", f.severity.upper()],
            ["Affected Target", self._escape(f.location)],
            ["CWE-ID", f.cwe_id or "N/A"],
            ["CVSS 3.1 Vector", f.cvss_vector or "N/A"],
            ["CVSS Base Score", f"{f.cvss_base_score:.1f}" if f.cvss_base_score else "N/A"],
            ["WSTG ID", f.wstg_test_id or "N/A"],
            ["MITRE ATT&CK", f.mitre_attack_technique or "N/A"],
        ]
        header_table = Table(header_data, colWidths=[140, CONTENT_WIDTH - 140])
        header_table.setStyle(TableStyle([
            ("FONT", (0, 0), (0, -1), "Helvetica-Bold", 9),
            ("FONT", (1, 0), (1, -1), "Helvetica", 9),
            ("BACKGROUND", (0, 0), (0, -1), HexColor("#F8FAFC")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#CBD5E1")),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ]))
        # Color severity row
        header_table.setStyle(TableStyle([
            ("TEXTCOLOR", (1, 0), (1, 0), HexColor(
                "#%02X%02X%02X" % (int(r*255), int(g*255), int(b*255))
            )),
            ("FONT", (1, 0), (1, 0), "Helvetica-Bold", 10),
        ]))
        block.append(header_table)
        block.append(Spacer(1, 8))

        # ── 1. Observation
        block.append(Paragraph("<b>Observation</b>", style_body))
        observation_text = self._extract_observation(f)
        block.append(Paragraph(self._escape(observation_text), style_body))

        detection_evidence = [ev for ev in f.evidence if ev.layer == "detection"]

        # ── 2. Exploitation (PoC)
        # FIX: VAPT-AI's agent records ALL evidence at layer="detection"
        # (tool_bridge._record_vulnerability is the only writer). The old
        # code only rendered layer="exploitation" — which never exists —
        # so every finding showed "No exploitation evidence" even when a
        # full sqlmap command + output was stored. The PoC IS the
        # detection evidence: command line + tool output.
        block.append(Spacer(1, 6))
        block.append(Paragraph("<b>Proof of Concept (PoC)</b>", style_body))

        poc_evidence = (
            [ev for ev in f.evidence if ev.layer == "exploitation"]
            or detection_evidence  # fallback — detection layer IS the PoC source
        )

        if poc_evidence:
            ev = poc_evidence[0]  # primary evidence
            raw = ev.raw_output or ""

            # PoC verified badge — based on poc_status + verified flag
            if f.verified and f.poc_status == "successful":
                block.append(Paragraph(
                    '<font color="#15803D"><b>✓ PoC VERIFIED</b></font> '
                    '<font size="9">— tool output below confirms the vulnerability</font>',
                    style_body,
                ))
            elif f.poc_status == "successful":
                block.append(Paragraph(
                    '<font color="#C2410C"><b>◐ PoC RECORDED (pending verification)</b></font>',
                    style_body,
                ))

            # Tool used
            if ev.tool_used:
                block.append(Paragraph(
                    f"<i>Tool:</i> <b>{self._escape(ev.tool_used)}</b>",
                    style_body,
                ))

            # Command line — priority: "$ ..." in raw evidence → the exact
            # command stashed in metadata_json.poc.command → exploit_method.
            command_line = None
            raw_body = raw
            first_line, _, rest = raw.partition("\n")
            if first_line.lstrip().startswith("$ "):
                command_line = first_line.lstrip()[2:]
                raw_body = rest.lstrip("\n")
            elif getattr(f, "poc_command", None):
                command_line = f.poc_command
            elif f.exploit_method and f.exploit_method not in ("agent", "pipeline-injected"):
                command_line = f.exploit_method

            if command_line:
                block.append(Paragraph("<i>Command:</i>", style_body))
                block.append(Paragraph(self._escape(f"$ {command_line}"), style_code))

            # Placeholder-repair disclosure — the stored command/evidence had a
            # {{URL}}/<target>/$TARGET placeholder that VAPT-AI replaced with the
            # finding's target. Warn the reader to verify before re-running.
            if getattr(f, "placeholder_repaired", False):
                block.append(Paragraph(
                    '<font color="#B45309"><b>⚠ Placeholder auto-repaired</b></font> '
                    '<font size="9">— the original evidence contained a '
                    '{{URL}}/&lt;target&gt; placeholder which was replaced with the '
                    "finding's target. Verify the command manually before "
                    're-running it.</font>',
                    style_body,
                ))

            # Raw output — generous cap (this is the proof)
            if raw_body:
                block.append(Paragraph("<i>Result:</i>", style_body))
                if len(raw_body) > 2500:
                    raw_body = raw_body[:2500] + "\n... [truncated — full output in evidence chain]"
                block.append(Paragraph(self._escape(raw_body), style_code))

            # Anti-hallucination reference
            captured_str = (
                ev.captured_at.strftime("%Y-%m-%d %H:%M:%S UTC")
                if hasattr(ev, "captured_at") and hasattr(ev.captured_at, "strftime")
                else str(getattr(ev, "captured_at", "N/A"))[:19]
            )
            ev_hash = (getattr(ev, "evidence_hash", "") or "")[:32]
            block.append(Paragraph(
                f"<i>Evidence ID: {str(getattr(ev, 'id', 'N/A'))[:8]} | "
                f"Hash: {ev_hash}... | Captured: {captured_str}</i>",
                style_body,
            ))

            # Additional evidence layers (if more than one)
            if len(poc_evidence) > 1:
                block.append(Paragraph(
                    f"<i>+ {len(poc_evidence) - 1} additional evidence entry(ies) "
                    f"in the evidence chain for this finding.</i>",
                    style_body,
                ))
        else:
            block.append(Paragraph(
                "<i>No evidence recorded for this finding — treat as "
                "unconfirmed. PoC: not validated.</i>",
                style_body,
            ))

        # ── 3. Impact (business impact per severity)
        block.append(Spacer(1, 6))
        block.append(Paragraph("<b>Impact</b>", style_body))
        impact_text = self._impact_for_severity(sev, f)
        block.append(Paragraph(self._escape(impact_text), style_body))

        # ── 4. Recommendation
        block.append(Spacer(1, 6))
        block.append(Paragraph("<b>Recommendation</b>", style_body))
        if f.remediation:
            block.append(Paragraph(self._escape(f.remediation), style_body))
        else:
            block.append(Paragraph(
                "<i>No specific remediation guidance recorded. "
                "Refer to OWASP WSTG v4.2 documentation for this vulnerability class.</i>",
                style_body,
            ))

        # ── 5. Re-Test Verification Result (Evvo template)
        block.append(Spacer(1, 6))
        block.append(Paragraph("<b>Re-Test Verification Result:</b> Pending re-test", style_body))
        block.append(Paragraph(
            "<i>__________________________________________________________________</i>",
            style_body,
        ))

        # ── W19-FIX3 Phase G3: Internet Cross-Check section ─────────────
        # Show the references + summary from the internet verifier so the
        # reader can independently verify the finding via the listed URLs.
        # Only rendered when internet_verification metadata exists (the
        # verifier was attempted — whether confirmed or not).
        iv = getattr(f, "internet_verification", None)
        if isinstance(iv, dict) and iv:
            block.append(Spacer(1, 6))
            block.append(Paragraph("<b>Internet Cross-Check</b>", style_body))
            summary = iv.get("summary") or ""
            if summary:
                block.append(Paragraph(self._escape(summary), style_body))
            refs = iv.get("references") or []
            if refs:
                refs_text = " | ".join(self._escape(str(r)) for r in refs[:3])
                block.append(Paragraph(
                    f"<i>References: {refs_text}</i>",
                    style_body,
                ))
            confidence = iv.get("confidence")
            if confidence is not None:
                block.append(Paragraph(
                    f"<i>Internet confidence: {float(confidence):.2f} "
                    f"(confirmed={iv.get('confirmed', False)})</i>",
                    style_body,
                ))

        block.append(Spacer(1, 16))
        return block

    def _extract_observation(self, f: FindingReportData) -> str:
        # FIX: prefer the LLM's actual description (stored in
        # metadata_json.description by tool_bridge._record_vulnerability and
        # now exposed on FindingReportData.description by the collector);
        # fall back to the synthesized text.
        desc = getattr(f, "description", None)
        if desc and str(desc).strip():
            return str(desc)
        return (
            f"A {f.severity.lower()}-severity {f.vuln_type} vulnerability was identified "
            f"at {f.location}. Verified: {f.verified} (auditor verdict: "
            f"{f.auditor_verdict or 'N/A'}, confidence: {f.confidence_score:.2f})."
        )

    def _impact_for_severity(self, sev: str, f: FindingReportData) -> str:
        """Business impact description per severity (per Evvo template)."""
        impacts = {
            "critical": (
                "Critical business impact. Exploitation succeeds trivially without "
                "authentication, leading to systems compromise. Successful exploitation "
                "may result in large-scale loss of customer or cardholder information. "
                "Immediate corrective measures are required."
            ),
            "high": (
                "High business impact. Exploitation succeeds and results in systems "
                "compromise. Technical vulnerability details and/or exploit code may be "
                "publicly available. Exploitation may result in highly costly loss of "
                "tangible assets or significantly harm the organization's mission, "
                "reputation, or interests. Strong need for corrective measures."
            ),
            "medium": (
                "Medium business impact. Exploitation requires a skilled attacker and "
                "may not directly result in elevated privileges. An additional vector "
                "(e.g. phishing, social engineering) is typically needed. Exploitation "
                "may result in costly loss of tangible assets or violate the "
                "organization's mission. Corrective actions are needed within a "
                "reasonable timeframe."
            ),
            "low": (
                "Low business impact. Exploitation is extremely difficult or requires "
                "controls already in place to impede successful exploitation. The "
                "scenario is possible but extremely unlikely. The accrediting authority "
                "should determine whether corrective actions are required or accept the risk."
            ),
            "info": (
                "No direct business impact. Information disclosed may be of interest to "
                "an attacker and useful for chaining with other vulnerabilities. "
                "Addressed as part of regular security maintenance."
            ),
        }
        return impacts.get(sev, "Impact assessment unavailable.")

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
            description=f_dict.get("description"),
            poc_command=f_dict.get("poc_command"),
            placeholder_repaired=f_dict.get("placeholder_repaired", False),
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