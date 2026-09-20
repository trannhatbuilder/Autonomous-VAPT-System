"""
W13-S6 — Unit tests for VAPT-AI Report Generator (PDF + SARIF).

These tests exercise the report exporters with plain-dict scan data (no
DB session required) so they can run in any environment without DB setup.

Test fixtures use the 5-finding DVWA_FIXTURE_FINDINGS to mirror the W13
acceptance criteria.

Coverage:
    TestPDFExporter
        - PDF file is generated and is a valid PDF
        - PDF magic bytes (%PDF-) at start
        - PDF size > 1KB
        - PDF metadata (title, author) correct
        - PDF contains expected section headers
        - Risk level calculation (Critical/High/Medium/Low)
        - Severity weighting (Critical=10, High=7, Medium=4, Low=2, Info=0)
        - HTML escaping in finding names (XSS payloads)

    TestSARIFExporter
        - SARIF JSON is valid (parses, has version + runs)
        - SARIF version == "2.1.0"
        - Schema URI correct
        - One result per finding
        - ruleId == WSTG ID
        - level mapping (critical/high → error, medium → warning, low → note)
        - locations[0] has the finding URL
        - partialFingerprints has cvss_vector + cvss_base_score
        - properties has all standard mappings (CWE, CVE, MITRE ATT&CK)
        - Code flows for evidence chain
        - Schema validation passes

    TestRiskScoring
        - Risk score = sum of severity weights (capped at 100)
        - Critical findings → risk level "Critical"
        - Empty findings → risk level "Low"
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import uuid
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_finding_dict():
    """Single sample finding for unit tests."""
    return {
        "id": str(uuid.uuid4()),
        "name": "SQL Injection in /login",
        "vuln_type": "sqli",
        "severity": "critical",
        "cvss_vector": "AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H",
        "cvss_base_score": 9.8,
        "cvss_severity": "critical",
        "location": "/login",
        "cwe_id": "CWE-89",
        "cve_id": "CVE-2024-1234",
        "wstg_test_id": "WSTG-INPV-05",
        "mitre_attack_technique": "T1190",
        "mitre_attack_tactic": "TA0001",
        "mitre_attack_subtechnique": None,
        "poc_status": "successful",
        "poc_tier": 1,
        "exploit_method": "sqlmap --os-shell",
        "remediation": "Use parameterized queries / prepared statements.",
        "verified": True,
        "false_positive": False,
        "auditor_verdict": "confirmed",
        "confidence_score": 0.92,
        "evidence": [
            {
                "layer": "detection",
                "raw_output": "[sqlmap] sqlmap identified the following injection point(s)",
                "tool_used": "sqlmap",
                "custody_seal": "abc123def456",
                "evidence_hash": "sha256:abcdef0123456789",
                "captured_at": "2026-09-20T10:00:00+00:00",
                "spill_path": None,
            },
            {
                "layer": "exploitation",
                "raw_output": "[sqlmap] id=1 UNION SELECT 1,2,3-- → returned data",
                "tool_used": "sqlmap",
                "custody_seal": "def789ghi012",
                "evidence_hash": "sha256:0123456789abcdef",
                "captured_at": "2026-09-20T10:01:00+00:00",
                "spill_path": None,
            },
        ],
    }


@pytest.fixture
def sample_scan_data(sample_finding_dict):
    """Full scan data dict with 1 sample finding."""
    return {
        "scan_id": "scan_test_unit_001",
        "target": "http://localhost:8080/",
        "target_type": "web_app",
        "agent_mode": "supervisor",
        "hitl_mode": "audit_agent",
        "user_prompt": "Test scan",
        "status": "completed",
        "progress": 100,
        "started_at": None,
        "completed_at": None,
        "result_summary": {"findings_total": 1},
        "error": None,
        "findings": [sample_finding_dict],
        "audit_entries": [],
        "hitl_approvals": [],
        "consent": None,
    }


@pytest.fixture
def dvwa_5findings_scan_data():
    """Full scan data with the standard DVWA 5-finding fixture."""
    from app.pentest.scan_pipeline import DVWA_FIXTURE_FINDINGS
    return {
        "scan_id": "scan_dvwa_unit_001",
        "target": "http://localhost:8080/",
        "target_type": "web_app",
        "agent_mode": "supervisor",
        "hitl_mode": "audit_agent",
        "user_prompt": "DVWA full pentest",
        "status": "completed",
        "progress": 100,
        "started_at": None,
        "completed_at": None,
        "result_summary": {"findings_total": 5},
        "error": None,
        "findings": [f.to_dict() for f in DVWA_FIXTURE_FINDINGS],
        "audit_entries": [],
        "hitl_approvals": [],
        "consent": None,
    }


# ---------------------------------------------------------------------------
# PDF Exporter Tests
# ---------------------------------------------------------------------------

class TestPDFExporter:
    """Unit tests for app/report/pdf_exporter.py."""

    def test_pdf_file_generated(self, sample_scan_data, tmp_path):
        """Test that a PDF file is generated at the expected path."""
        from app.report.pdf_exporter import generate_pdf_report_sync

        output_path = tmp_path / "test_report.pdf"
        result = generate_pdf_report_sync(sample_scan_data, output_path)

        assert result == output_path
        assert output_path.exists()

    def test_pdf_magic_bytes(self, sample_scan_data, tmp_path):
        """Test that the generated PDF starts with %PDF- magic bytes."""
        from app.report.pdf_exporter import generate_pdf_report_sync

        output_path = tmp_path / "magic_test.pdf"
        generate_pdf_report_sync(sample_scan_data, output_path)

        with open(output_path, "rb") as f:
            magic = f.read(5)
        assert magic == b"%PDF-", f"Invalid PDF magic: {magic!r}"

    def test_pdf_size_nontrivial(self, dvwa_5findings_scan_data, tmp_path):
        """Test that the PDF is non-trivial in size (> 5KB with 5 findings)."""
        from app.report.pdf_exporter import generate_pdf_report_sync

        output_path = tmp_path / "size_test.pdf"
        generate_pdf_report_sync(dvwa_5findings_scan_data, output_path)

        size = output_path.stat().st_size
        assert size > 5000, f"PDF too small ({size} bytes) — likely empty/broken"

    def test_pdf_metadata_correct(self, sample_scan_data, tmp_path):
        """Test that the PDF has correct metadata (title, author)."""
        from app.report.pdf_exporter import generate_pdf_report_sync

        output_path = tmp_path / "meta_test.pdf"
        generate_pdf_report_sync(sample_scan_data, output_path)

        # Read PDF metadata via pypdf if available, else skip
        try:
            from pypdf import PdfReader
            reader = PdfReader(str(output_path))
            meta = reader.metadata
            assert meta is not None
            assert "VAPT-AI" in (meta.get("/Title", "") or "")
            assert "VAPT-AI" in (meta.get("/Author", "") or "")
        except ImportError:
            pytest.skip("pypdf not installed — skipping metadata check")

    def test_pdf_contains_section_headers(self, dvwa_5findings_scan_data, tmp_path):
        """Test that the PDF contains the 6 expected section headers."""
        from app.report.pdf_exporter import generate_pdf_report_sync

        output_path = tmp_path / "sections_test.pdf"
        generate_pdf_report_sync(dvwa_5findings_scan_data, output_path)

        # Extract text via pypdf if available
        try:
            from pypdf import PdfReader
            reader = PdfReader(str(output_path))
            full_text = ""
            for page in reader.pages:
                full_text += page.extract_text() + "\n"

            expected_sections = [
                "Executive Summary",
                "Methodology",
                "Findings Detail",
                "Audit Trail",
                "Appendix",
            ]
            for section in expected_sections:
                assert section in full_text, f"Section {section!r} not found in PDF text"
        except ImportError:
            pytest.skip("pypdf not installed — skipping section check")

    def test_pdf_contains_all_5_vuln_types(self, dvwa_5findings_scan_data, tmp_path):
        """Test that the PDF contains all 5 vulnerability types from the fixture."""
        from app.report.pdf_exporter import generate_pdf_report_sync

        output_path = tmp_path / "vuln_types_test.pdf"
        generate_pdf_report_sync(dvwa_5findings_scan_data, output_path)

        try:
            from pypdf import PdfReader
            reader = PdfReader(str(output_path))
            full_text = ""
            for page in reader.pages:
                full_text += page.extract_text() + "\n"

            expected_vulns = ["SQL Injection", "XSS", "Command Injection", "LFI", "RCE"]
            for vuln in expected_vulns:
                # Some abbreviations may be in the fixture — check both
                # full name and vuln_type code
                if vuln not in full_text:
                    # Try the vuln_type code (lowercase)
                    vtype = vuln.lower().replace(" ", "_")
                    assert vtype in full_text.lower(), \
                        f"Neither {vuln!r} nor {vtype!r} found in PDF text"
        except ImportError:
            pytest.skip("pypdf not installed — skipping vuln type check")

    def test_pdf_handles_empty_findings(self, tmp_path):
        """Test that PDF generation works with 0 findings."""
        from app.report.pdf_exporter import generate_pdf_report_sync

        empty_scan = {
            "scan_id": "scan_empty_001",
            "target": "http://localhost:8080/",
            "target_type": "web_app",
            "agent_mode": "supervisor",
            "hitl_mode": "audit_agent",
            "user_prompt": "",
            "status": "completed",
            "progress": 100,
            "started_at": None,
            "completed_at": None,
            "result_summary": {"findings_total": 0},
            "error": None,
            "findings": [],
            "audit_entries": [],
            "hitl_approvals": [],
            "consent": None,
        }

        output_path = tmp_path / "empty_test.pdf"
        generate_pdf_report_sync(empty_scan, output_path)
        assert output_path.exists()
        assert output_path.stat().st_size > 1000

    def test_pdf_escapes_xss_payload_in_finding_name(self, tmp_path):
        """Test that XSS payloads in finding names are properly escaped in PDF."""
        from app.report.pdf_exporter import generate_pdf_report_sync

        xss_scan = {
            "scan_id": "scan_xss_001",
            "target": "http://localhost:8080/",
            "target_type": "web_app",
            "agent_mode": "supervisor",
            "hitl_mode": "audit_agent",
            "user_prompt": "",
            "status": "completed",
            "progress": 100,
            "started_at": None,
            "completed_at": None,
            "result_summary": {"findings_total": 1},
            "error": None,
            "findings": [{
                "id": str(uuid.uuid4()),
                "name": "XSS via <script>alert('XSS')</script> payload",
                "vuln_type": "xss",
                "severity": "high",
                "cvss_vector": "AV:N/AC:L/PR:N/UI:R/S:C/C:H/I:L/A:N",
                "cvss_base_score": 7.4,
                "cvss_severity": "high",
                "location": "/search?q=test",
                "cwe_id": "CWE-79",
                "cve_id": None,
                "wstg_test_id": "WSTG-INPV-02",
                "mitre_attack_technique": "T1059.007",
                "mitre_attack_tactic": "TA0002",
                "mitre_attack_subtechnique": None,
                "poc_status": "successful",
                "poc_tier": 1,
                "exploit_method": None,
                "remediation": "Output encoding + CSP.",
                "verified": True,
                "false_positive": False,
                "auditor_verdict": "confirmed",
                "confidence_score": 0.85,
                "evidence": [],
            }],
            "audit_entries": [],
            "hitl_approvals": [],
            "consent": None,
        }

        output_path = tmp_path / "xss_escape_test.pdf"
        # Should not raise (test fails if XML escape doesn't happen)
        generate_pdf_report_sync(xss_scan, output_path)
        assert output_path.exists()
        assert output_path.stat().st_size > 1000


# ---------------------------------------------------------------------------
# SARIF Exporter Tests
# ---------------------------------------------------------------------------

class TestSARIFExporter:
    """Unit tests for app/report/sarif_exporter.py."""

    def test_sarif_file_generated(self, sample_scan_data, tmp_path):
        """Test that a SARIF file is generated at the expected path."""
        from app.report.sarif_exporter import generate_sarif_report_sync

        output_path = tmp_path / "test_report.sarif.json"
        result = generate_sarif_report_sync(sample_scan_data, output_path)

        assert result == output_path
        assert output_path.exists()

    def test_sarif_parses_as_json(self, sample_scan_data, tmp_path):
        """Test that the SARIF output is valid JSON."""
        from app.report.sarif_exporter import generate_sarif_report_sync

        output_path = tmp_path / "json_test.sarif.json"
        generate_sarif_report_sync(sample_scan_data, output_path)

        with open(output_path, "r", encoding="utf-8") as f:
            sarif = json.load(f)  # will raise if not valid JSON

        assert isinstance(sarif, dict)

    def test_sarif_version_correct(self, sample_scan_data, tmp_path):
        """Test that the SARIF version is 2.1.0."""
        from app.report.sarif_exporter import generate_sarif_report_sync

        output_path = tmp_path / "version_test.sarif.json"
        generate_sarif_report_sync(sample_scan_data, output_path)

        with open(output_path) as f:
            sarif = json.load(f)
        assert sarif["version"] == "2.1.0"

    def test_sarif_schema_uri_present(self, sample_scan_data, tmp_path):
        """Test that the SARIF $schema URI is present + correct."""
        from app.report.sarif_exporter import generate_sarif_report_sync, SARIF_SCHEMA

        output_path = tmp_path / "schema_test.sarif.json"
        generate_sarif_report_sync(sample_scan_data, output_path)

        with open(output_path) as f:
            sarif = json.load(f)
        assert sarif["$schema"] == SARIF_SCHEMA

    def test_sarif_has_runs(self, sample_scan_data, tmp_path):
        """Test that the SARIF has at least one run."""
        from app.report.sarif_exporter import generate_sarif_report_sync

        output_path = tmp_path / "runs_test.sarif.json"
        generate_sarif_report_sync(sample_scan_data, output_path)

        with open(output_path) as f:
            sarif = json.load(f)
        assert "runs" in sarif
        assert len(sarif["runs"]) >= 1

    def test_sarif_run_tool_driver_correct(self, sample_scan_data, tmp_path):
        """Test that the SARIF tool.driver.name is VAPT-AI."""
        from app.report.sarif_exporter import generate_sarif_report_sync

        output_path = tmp_path / "tool_test.sarif.json"
        generate_sarif_report_sync(sample_scan_data, output_path)

        with open(output_path) as f:
            sarif = json.load(f)
        run = sarif["runs"][0]
        assert run["tool"]["driver"]["name"] == "VAPT-AI"
        assert run["tool"]["driver"]["version"] == "3.2.0"

    def test_sarif_one_result_per_finding(self, dvwa_5findings_scan_data, tmp_path):
        """Test that there is exactly one result per finding (5 → 5)."""
        from app.report.sarif_exporter import generate_sarif_report_sync

        output_path = tmp_path / "results_test.sarif.json"
        generate_sarif_report_sync(dvwa_5findings_scan_data, output_path)

        with open(output_path) as f:
            sarif = json.load(f)
        results = sarif["runs"][0]["results"]
        assert len(results) == 5

    def test_sarif_ruleid_is_wstg_id(self, sample_scan_data, tmp_path):
        """Test that result.ruleId matches the WSTG ID."""
        from app.report.sarif_exporter import generate_sarif_report_sync

        output_path = tmp_path / "ruleid_test.sarif.json"
        generate_sarif_report_sync(sample_scan_data, output_path)

        with open(output_path) as f:
            sarif = json.load(f)
        result = sarif["runs"][0]["results"][0]
        assert result["ruleId"] == "WSTG-INPV-05"  # from sample_finding_dict

    @pytest.mark.parametrize("severity,expected_level", [
        ("critical", "error"),
        ("high", "error"),
        ("medium", "warning"),
        ("low", "note"),
        ("info", "none"),
    ])
    def test_sarif_severity_to_level_mapping(
        self, severity, expected_level, tmp_path,
    ):
        """Test SARIF severity → level mapping (per SARIF spec §3.27.10)."""
        from app.report.sarif_exporter import (
            SEVERITY_TO_SARIF_LEVEL,
            generate_sarif_report_sync,
        )
        # Direct mapping test
        assert SEVERITY_TO_SARIF_LEVEL[severity] == expected_level

        # Integration test — generate SARIF + verify
        scan_data = {
            "scan_id": f"scan_sev_{severity}",
            "target": "http://localhost/",
            "target_type": "web_app",
            "agent_mode": "supervisor",
            "hitl_mode": "audit_agent",
            "user_prompt": "",
            "status": "completed",
            "progress": 100,
            "started_at": None,
            "completed_at": None,
            "result_summary": {},
            "error": None,
            "findings": [{
                "id": str(uuid.uuid4()),
                "name": f"Test {severity} finding",
                "vuln_type": "test",
                "severity": severity,
                "cvss_vector": "AV:N/AC:L/PR:N/UI:N/S:U/C:N/I:N/A:N",
                "cvss_base_score": 0.0,
                "cvss_severity": severity,
                "location": "/test",
                "cwe_id": None,
                "cve_id": None,
                "wstg_test_id": "WSTG-INFO-01",
                "mitre_attack_technique": None,
                "mitre_attack_tactic": None,
                "mitre_attack_subtechnique": None,
                "poc_status": "not_attempted",
                "poc_tier": None,
                "exploit_method": None,
                "remediation": None,
                "verified": False,
                "false_positive": False,
                "auditor_verdict": None,
                "confidence_score": 0.5,
                "evidence": [],
            }],
            "audit_entries": [],
            "hitl_approvals": [],
            "consent": None,
        }
        output_path = tmp_path / f"sev_{severity}.sarif.json"
        generate_sarif_report_sync(scan_data, output_path)

        with open(output_path) as f:
            sarif = json.load(f)
        result = sarif["runs"][0]["results"][0]
        assert result["level"] == expected_level

    def test_sarif_locations_present(self, sample_scan_data, tmp_path):
        """Test that result.locations[0] has the finding URL."""
        from app.report.sarif_exporter import generate_sarif_report_sync

        output_path = tmp_path / "locations_test.sarif.json"
        generate_sarif_report_sync(sample_scan_data, output_path)

        with open(output_path) as f:
            sarif = json.load(f)
        result = sarif["runs"][0]["results"][0]
        assert len(result["locations"]) >= 1
        loc = result["locations"][0]
        assert "physicalLocation" in loc
        assert loc["physicalLocation"]["artifactLocation"]["uri"] == "/login"

    def test_sarif_partial_fingerprints_have_cvss(self, sample_scan_data, tmp_path):
        """Test that partialFingerprints contains CVSS vector + score."""
        from app.report.sarif_exporter import generate_sarif_report_sync

        output_path = tmp_path / "fingerprints_test.sarif.json"
        generate_sarif_report_sync(sample_scan_data, output_path)

        with open(output_path) as f:
            sarif = json.load(f)
        result = sarif["runs"][0]["results"][0]
        pf = result["partialFingerprints"]
        assert "vapt_ai/cvss_vector" in pf
        assert pf["vapt_ai/cvss_vector"] == "AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H"
        assert "vapt_ai/cvss_base_score" in pf
        assert "vapt_ai/finding_id" in pf

    def test_sarif_properties_have_all_mappings(self, sample_scan_data, tmp_path):
        """Test that result.properties has all standard mappings."""
        from app.report.sarif_exporter import generate_sarif_report_sync

        output_path = tmp_path / "props_test.sarif.json"
        generate_sarif_report_sync(sample_scan_data, output_path)

        with open(output_path) as f:
            sarif = json.load(f)
        result = sarif["runs"][0]["results"][0]
        props = result["properties"]

        expected_props = [
            "severity", "vuln_type", "cwe_id", "cve_id", "wstg_test_id",
            "mitre_attack_technique", "mitre_attack_tactic",
            "poc_status", "verified", "false_positive",
            "auditor_verdict", "confidence_score", "scan_id",
        ]
        for prop in expected_props:
            assert prop in props, f"Missing property: {prop}"

        # Specific values from sample_finding_dict
        assert props["cwe_id"] == "CWE-89"
        assert props["cve_id"] == "CVE-2024-1234"
        assert props["wstg_test_id"] == "WSTG-INPV-05"
        assert props["mitre_attack_technique"] == "T1190"
        assert props["mitre_attack_tactic"] == "TA0001"

    def test_sarif_code_flows_for_evidence(self, sample_scan_data, tmp_path):
        """Test that codeFlows are present when a finding has evidence."""
        from app.report.sarif_exporter import generate_sarif_report_sync

        output_path = tmp_path / "codeflows_test.sarif.json"
        generate_sarif_report_sync(sample_scan_data, output_path)

        with open(output_path) as f:
            sarif = json.load(f)
        result = sarif["runs"][0]["results"][0]
        # sample_finding_dict has 2 evidence rows
        assert "codeFlows" in result
        assert len(result["codeFlows"]) == 2

    def test_sarif_rules_built_from_findings(self, dvwa_5findings_scan_data, tmp_path):
        """Test that rules[] entries are built from unique WSTG IDs."""
        from app.report.sarif_exporter import generate_sarif_report_sync

        output_path = tmp_path / "rules_test.sarif.json"
        generate_sarif_report_sync(dvwa_5findings_scan_data, output_path)

        with open(output_path) as f:
            sarif = json.load(f)
        rules = sarif["runs"][0]["tool"]["driver"]["rules"]

        # 5 findings → 5 unique WSTG IDs → 5 rules
        assert len(rules) == 5
        rule_ids = {r["id"] for r in rules}
        assert "WSTG-INPV-05" in rule_ids   # SQLi
        assert "WSTG-INPV-02" in rule_ids   # XSS
        assert "WSTG-INPV-11" in rule_ids   # LFI
        assert "WSTG-INPV-12" in rule_ids   # cmd_injection
        assert "WSTG-INPV-03" in rule_ids   # RCE

    def test_sarif_schema_validation_passes(self, sample_scan_data, tmp_path):
        """Test that SARIF passes the minimal schema validation."""
        from app.report.sarif_exporter import (
            SARIFExporter,
            generate_sarif_report_sync,
        )
        from app.report.collector import ScanReportData, FindingReportData

        # Reconstruct + validate directly
        from app.pentest.scan_pipeline import DVWA_FIXTURE_FINDINGS
        exporter = SARIFExporter()

        # Use the sample scan data to build SARIF, then validate
        output_path = tmp_path / "validation_test.sarif.json"
        generate_sarif_report_sync(sample_scan_data, output_path)

        with open(output_path) as f:
            sarif = json.load(f)

        # Should not raise
        exporter._validate_schema(sarif)


# ---------------------------------------------------------------------------
# Risk Scoring Tests
# ---------------------------------------------------------------------------

class TestRiskScoring:
    """Unit tests for risk score calculation (ported from EVVO)."""

    def test_critical_findings_yield_critical_risk(self):
        """3 critical findings (3×10=30) → risk level "Critical"."""
        from app.report.pdf_exporter import calculate_risk_level, SEVERITY_WEIGHTS
        from app.report.collector import FindingReportData

        findings = [
            FindingReportData(
                id=uuid.uuid4(), name=f"Critical {i}", vuln_type="sqli",
                severity="critical", cvss_vector=None, cvss_base_score=None,
                cvss_severity=None, location="/", cwe_id=None, cve_id=None,
                wstg_test_id=None, mitre_attack_technique=None,
                mitre_attack_tactic=None, mitre_attack_subtechnique=None,
                poc_status="successful", poc_tier=None, exploit_method=None,
                remediation=None, verified=True, false_positive=False,
                auditor_verdict=None, confidence_score=0.9, evidence=[],
            )
            for i in range(3)
        ]
        level, score = calculate_risk_level(findings)
        assert level == "Critical"
        assert score == 30  # 3 × 10

    def test_high_findings_yield_high_risk(self):
        """3 high findings (3×7=21) → risk level "High" (>=15)."""
        from app.report.pdf_exporter import calculate_risk_level
        from app.report.collector import FindingReportData

        findings = [
            FindingReportData(
                id=uuid.uuid4(), name=f"High {i}", vuln_type="xss",
                severity="high", cvss_vector=None, cvss_base_score=None,
                cvss_severity=None, location="/", cwe_id=None, cve_id=None,
                wstg_test_id=None, mitre_attack_technique=None,
                mitre_attack_tactic=None, mitre_attack_subtechnique=None,
                poc_status="successful", poc_tier=None, exploit_method=None,
                remediation=None, verified=True, false_positive=False,
                auditor_verdict=None, confidence_score=0.8, evidence=[],
            )
            for i in range(3)
        ]
        level, score = calculate_risk_level(findings)
        assert level == "High"
        assert score == 21  # 3 × 7

    def test_empty_findings_yield_low_risk(self):
        """0 findings → risk level "Low", score 0."""
        from app.report.pdf_exporter import calculate_risk_level

        level, score = calculate_risk_level([])
        assert level == "Low"
        assert score == 0

    def test_false_positive_findings_excluded(self):
        """False positive findings should not contribute to risk score."""
        from app.report.pdf_exporter import calculate_risk_level
        from app.report.collector import FindingReportData

        findings = [
            FindingReportData(
                id=uuid.uuid4(), name="FP critical", vuln_type="sqli",
                severity="critical", cvss_vector=None, cvss_base_score=None,
                cvss_severity=None, location="/", cwe_id=None, cve_id=None,
                wstg_test_id=None, mitre_attack_technique=None,
                mitre_attack_tactic=None, mitre_attack_subtechnique=None,
                poc_status="not_attempted", poc_tier=None, exploit_method=None,
                remediation=None, verified=False, false_positive=True,
                auditor_verdict="rejected", confidence_score=0.2, evidence=[],
            ),
            FindingReportData(
                id=uuid.uuid4(), name="Real medium", vuln_type="info",
                severity="medium", cvss_vector=None, cvss_base_score=None,
                cvss_severity=None, location="/", cwe_id=None, cve_id=None,
                wstg_test_id=None, mitre_attack_technique=None,
                mitre_attack_tactic=None, mitre_attack_subtechnique=None,
                poc_status="successful", poc_tier=None, exploit_method=None,
                remediation=None, verified=True, false_positive=False,
                auditor_verdict=None, confidence_score=0.7, evidence=[],
            ),
        ]
        level, score = calculate_risk_level(findings)
        # FP critical (10) excluded → only medium (4) counts
        assert score == 4
        assert level == "Low"  # score 4 < 5

    def test_score_capped_at_100(self):
        """Risk score should be capped at 100 even with many critical findings."""
        from app.report.pdf_exporter import calculate_risk_level
        from app.report.collector import FindingReportData

        findings = [
            FindingReportData(
                id=uuid.uuid4(), name=f"Critical {i}", vuln_type="sqli",
                severity="critical", cvss_vector=None, cvss_base_score=None,
                cvss_severity=None, location="/", cwe_id=None, cve_id=None,
                wstg_test_id=None, mitre_attack_technique=None,
                mitre_attack_tactic=None, mitre_attack_subtechnique=None,
                poc_status="successful", poc_tier=None, exploit_method=None,
                remediation=None, verified=True, false_positive=False,
                auditor_verdict=None, confidence_score=0.9, evidence=[],
            )
            for i in range(20)  # 20 × 10 = 200, capped at 100
        ]
        level, score = calculate_risk_level(findings)
        assert score == 100
        assert level == "Critical"


# ---------------------------------------------------------------------------
# Constants tests
# ---------------------------------------------------------------------------

class TestConstants:
    """Verify exported constants are correct."""

    def test_sarif_version_constant(self):
        from app.report import SARIF_VERSION
        assert SARIF_VERSION == "2.1.0"

    def test_sarif_schema_constant(self):
        from app.report import SARIF_SCHEMA
        assert "sarif" in SARIF_SCHEMA.lower()
        assert "2.1.0" in SARIF_SCHEMA

    def test_severity_weights_correct(self):
        from app.report.pdf_exporter import SEVERITY_WEIGHTS
        assert SEVERITY_WEIGHTS["critical"] == 10
        assert SEVERITY_WEIGHTS["high"] == 7
        assert SEVERITY_WEIGHTS["medium"] == 4
        assert SEVERITY_WEIGHTS["low"] == 2
        assert SEVERITY_WEIGHTS["info"] == 0

    def test_severity_colors_defined(self):
        from app.report.pdf_exporter import SEVERITY_COLORS
        for sev in ["critical", "high", "medium", "low", "info"]:
            assert sev in SEVERITY_COLORS
            color = SEVERITY_COLORS[sev]
            assert len(color) == 3  # RGB tuple
            for component in color:
                assert 0.0 <= component <= 1.0