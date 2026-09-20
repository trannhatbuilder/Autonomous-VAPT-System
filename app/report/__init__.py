"""
VAPT-AI Report Generation Subsystem (W13-S3).

Generates two report formats per scan:

    1. PDF report via ReportLab (app/report/pdf_exporter.py)
       - Cover page
       - Executive Summary (findings by severity + risk score)
       - Methodology (WSTG v4.2 coverage + agent phases)
       - Findings Detail (per-finding: title, severity, CVSS, WSTG, MITRE,
         CWE, location, evidence, PoC status, recommendations)
       - Audit Trail (chain-of-custody verification, HITL approvals)
       - Appendix (tool inventory, agent decisions log)

    2. SARIF 2.1.0 JSON via jsonschema-validated exporter
       (app/report/sarif_exporter.py)
       - OASIS SARIF 2.1.0 spec compliant
       - One `result` per finding with ruleId (WSTG ID), level (severity),
         message, locations[], partialFingerprints, codeFlows[]

Reference sources (per master plan §10):
    - CyberStrikeAI: no report generator (Go codebase); VAPT-AI implements
      its own per master plan §5.1 tech stack (ReportLab 4.1+).
    - EVVO Sentinel: shield_engine/report_generator.py (DOCX) +
      html_report_generator.py (HTML). VAPT-AI ports the structure +
      severity weighting logic, but switches output format to PDF + SARIF
      (industry standards for pentest reports).

Usage:
    from app.report import generate_pdf_report, generate_sarif_report

    async with async_session() as session:
        pdf_path = await generate_pdf_report(scan_id, session)
        sarif_path = await generate_sarif_report(scan_id, session)

Output location: reports/ directory at project root.
    - reports/scan_<id>_<timestamp>.pdf
    - reports/scan_<id>_<timestamp>.sarif.json
"""
from __future__ import annotations

from app.report.pdf_exporter import (
    generate_pdf_report,
    generate_pdf_report_sync,
    PDFExporter,
)
from app.report.sarif_exporter import (
    generate_sarif_report,
    generate_sarif_report_sync,
    SARIFExporter,
    SARIF_VERSION,
    SARIF_SCHEMA,
)
from app.report.collector import (
    collect_scan_data,
    ScanReportData,
    FindingReportData,
    EvidenceReportData,
)

__all__ = [
    # PDF
    "generate_pdf_report",
    "generate_pdf_report_sync",
    "PDFExporter",
    # SARIF
    "generate_sarif_report",
    "generate_sarif_report_sync",
    "SARIFExporter",
    "SARIF_VERSION",
    "SARIF_SCHEMA",
    # Data collection
    "collect_scan_data",
    "ScanReportData",
    "FindingReportData",
    "EvidenceReportData",
]