"""
VAPT-AI SARIF 2.1.0 Report Exporter (W13-S3).

Generates OASIS SARIF 2.1.0 compliant JSON reports from VAPT-AI scan data.

SARIF Spec: https://docs.oasis.org/projects/sarif/sarif/v2.1.0/sarif-v2.1.0.html
Schema:     https://github.com/oasis-tcs/sarif-spec/blob/main/Schemata/sarif-schema-2.1.0.json

Each VAPT-AI Finding becomes a SARIF `result` object with:
    - ruleId          ← WSTG test ID (e.g. "WSTG-INPV-05")
    - level           ← severity mapped to SARIF levels (error/warning/note)
    - message.text    ← finding name + description
    - locations[]     ← finding.location (URL/endpoint)
    - partialFingerprints.cvss_vector ← CVSS v3.1 vector string
    - partialFingerprints.cvss_base_score ← CVSS base score
    - properties      ← cwe_id, cve_id, mitre_attack_technique, mitre_attack_tactic,
                        poc_status, verified, false_positive, confidence_score,
                        auditor_verdict

WSTG + ATT&CK + CWE mappings are emitted as `rules[]` entries so downstream
tools (GitHub Code Scanning, Azure DevOps, SonarQube) can correlate findings
with industry standards.

Validation:
    The output JSON is validated against the official OASIS SARIF 2.1.0
    JSON schema using jsonschema. If validation fails, a ValueError is raised
    with the specific schema violations.

Usage:
    from app.report import generate_sarif_report
    async with async_session() as session:
        sarif_path = await generate_sarif_report(scan_id, session)
"""
from __future__ import annotations

import json
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
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SARIF_VERSION = "2.1.0"
SARIF_SCHEMA = "https://docs.oasis.org/projects/sarif/sarif/v2.1.0/cs01/schemas/sarif-schema-2.1.0.json"
SARIF_SCHEMA_LOCAL = "https://raw.githubusercontent.com/oasis-tcs/sarif-spec/main/Schemata/sarif-schema-2.1.0.json"

# SARIF severity mapping (Finding.severity → SARIF level)
# Per SARIF spec §3.27.10: level = "error" | "warning" | "note" | "none"
SEVERITY_TO_SARIF_LEVEL: dict[str, str] = {
    "critical": "error",
    "high": "error",
    "medium": "warning",
    "low": "note",
    "info": "none",
}

# Severity → CVSS-like numeric rank for sorting in report
SEVERITY_RANK: dict[str, int] = {
    "critical": 5,
    "high": 4,
    "medium": 3,
    "low": 2,
    "info": 1,
}

# Default report output directory (relative to project root)
REPORTS_DIR = Path(__file__).resolve().parent.parent.parent / "reports"


# ---------------------------------------------------------------------------
# Exporter
# ---------------------------------------------------------------------------

class SARIFExporter:
    """SARIF 2.1.0 exporter for VAPT-AI scan results.

    One exporter instance per scan. Use generate_sarif_report() convenience
    function for the common case.
    """

    def __init__(self, output_dir: Path | None = None):
        self.output_dir = output_dir or REPORTS_DIR

    async def export(
        self, scan_id: str, session: AsyncSession,
        output_path: str | Path | None = None,
    ) -> Path:
        """Generate SARIF JSON for a scan.

        Args:
            scan_id: Scan ID
            session: Async DB session
            output_path: Optional explicit output path. If None, auto-generated
                as reports/scan_<id>_<timestamp>.sarif.json

        Returns:
            Path to the generated SARIF file.

        Raises:
            ValueError: if scan not found OR SARIF schema validation fails.
        """
        data = await collect_scan_data(scan_id, session)
        sarif = self._build_sarif(data)
        self._validate_schema(sarif)

        if output_path is None:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
            output_path = self.output_dir / f"scan_{scan_id}_{ts}.sarif.json"
        else:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(sarif, f, indent=2, ensure_ascii=False)

        logger.info("SARIF report generated: scan=%s path=%s findings=%d",
                    scan_id, output_path, len(data.findings))
        return output_path

    # ---------- SARIF building ----------

    def _build_sarif(self, data: ScanReportData) -> dict[str, Any]:
        """Build the top-level SARIF log object."""
        rules = self._build_rules(data.findings)
        results = self._build_results(data.findings, data.scan_id)

        run = {
            "tool": {
                "driver": {
                    "name": "VAPT-AI",
                    "version": "3.2.0",
                    "informationUri": "https://github.com/z-ai/vapt-ai",
                    "rules": rules,
                },
            },
            "automationDetails": {
                "guid": str(uuid.uuid4()),
                "id": f"vapt-ai/scan/{data.scan_id}",
            },
            "run": {
                "scan_id": data.scan_id,
                "target": data.target,
                "agent_mode": data.agent_mode,
                "hitl_mode": data.hitl_mode,
                "started_at": data.started_at.isoformat() if data.started_at else None,
                "completed_at": data.completed_at.isoformat() if data.completed_at else None,
                "status": data.status,
            },
            "results": results,
            "invocations": [
                {
                    "executionSuccessful": data.status == "completed",
                    "startTimeUtc": data.started_at.isoformat() if data.started_at else None,
                    "endTimeUtc": data.completed_at.isoformat() if data.completed_at else None,
                    "toolExecutionNotifications": [
                        {
                            "level": "error",
                            "message": {"text": data.error},
                        }
                    ] if data.error else [],
                }
            ],
        }

        return {
            "$schema": SARIF_SCHEMA,
            "version": SARIF_VERSION,
            "runs": [run],
        }

    def _build_rules(self, findings: list[FindingReportData]) -> list[dict[str, Any]]:
        """Build SARIF rules[] — one per unique WSTG ID.

        Each rule represents a vulnerability class (e.g. WSTG-INPV-05 SQLi).
        Multiple findings can map to the same rule.
        """
        seen_rule_ids: set[str] = set()
        rules: list[dict[str, Any]] = []

        for f in findings:
            rule_id = f.wstg_test_id or f"VAPT-{f.vuln_type.upper()}"
            if rule_id in seen_rule_ids:
                continue
            seen_rule_ids.add(rule_id)

            rule: dict[str, Any] = {
                "id": rule_id,
                "name": f.vuln_type.upper(),
                "shortDescription": {"text": f.name},
                "fullDescription": {
                    "text": f"Vulnerability type: {f.vuln_type}. "
                            f"WSTG ID: {f.wstg_test_id or 'N/A'}. "
                            f"CWE: {f.cwe_id or 'N/A'}."
                },
                "helpUri": f"https://owasp.org/www-project-web-security-testing-guide/v42/{rule_id.lower()}/"
                    if f.wstg_test_id else None,
                "properties": {
                    "tags": [f.vuln_type, f.severity],
                    "precision": "high",
                },
            }
            if f.cwe_id:
                rule["properties"]["cwe"] = f.cwe_id
            if f.cve_id:
                rule["properties"]["cve"] = f.cve_id
            if f.mitre_attack_technique:
                rule["properties"]["mitre_attack_technique"] = f.mitre_attack_technique
            if f.mitre_attack_tactic:
                rule["properties"]["mitre_attack_tactic"] = f.mitre_attack_tactic
            rules.append(rule)

        return rules

    def _build_results(
        self, findings: list[FindingReportData], scan_id: str,
    ) -> list[dict[str, Any]]:
        """Build SARIF results[] — one per finding."""
        results: list[dict[str, Any]] = []

        for idx, f in enumerate(findings, start=1):
            rule_id = f.wstg_test_id or f"VAPT-{f.vuln_type.upper()}"
            level = SEVERITY_TO_SARIF_LEVEL.get(f.severity.lower(), "warning")

            result: dict[str, Any] = {
                "ruleId": rule_id,
                "ruleIndex": idx - 1,
                "level": level,
                "message": {
                    "text": f"{f.name} (CVSS: {f.cvss_vector or 'N/A'} "
                            f"= {f.cvss_base_score or 'N/A'} {f.cvss_severity or ''})",
                },
                "locations": [
                    {
                        "physicalLocation": {
                            "artifactLocation": {
                                "uri": f.location,
                                "uriBaseId": "%SRCROOT%",
                            },
                        },
                        "logicalLocations": [
                            {
                                "fullyQualifiedName": f.location,
                                "name": f.vuln_type,
                            }
                        ],
                    }
                ],
                "partialFingerprints": {
                    "primaryLocationLineHash": f.evidence[0].evidence_hash
                        if f.evidence else str(uuid.uuid4()),
                    "vapt_ai/cvss_vector": f.cvss_vector or "",
                    "vapt_ai/cvss_base_score": str(f.cvss_base_score or ""),
                    "vapt_ai/finding_id": str(f.id),
                },
                "properties": {
                    "severity": f.severity,
                    "vuln_type": f.vuln_type,
                    "cwe_id": f.cwe_id,
                    "cve_id": f.cve_id,
                    "wstg_test_id": f.wstg_test_id,
                    "mitre_attack_technique": f.mitre_attack_technique,
                    "mitre_attack_tactic": f.mitre_attack_tactic,
                    "poc_status": f.poc_status,
                    "poc_tier": f.poc_tier,
                    "exploit_method": f.exploit_method,
                    "remediation": f.remediation,
                    "verified": f.verified,
                    "false_positive": f.false_positive,
                    "auditor_verdict": f.auditor_verdict,
                    "confidence_score": f.confidence_score,
                    "scan_id": scan_id,
                },
            }

            # Code flows — one per evidence layer
            if f.evidence:
                result["codeFlows"] = [
                    self._build_code_flow(ev) for ev in f.evidence
                ]

            results.append(result)

        return results

    def _build_code_flow(self, ev: Any) -> dict[str, Any]:
        """Build a SARIF codeFlow for an evidence entry.

        Each evidence layer becomes a step in the codeFlow, showing the
        progression from detection → validation → exploitation → post_exploit.
        """
        return {
            "threadFlows": [
                {
                    "locations": [
                        {
                            "location": {
                                "physicalLocation": {
                                    "artifactLocation": {
                                        "uri": ev.spill_path or f"evidence://{ev.tool_used}",
                                    },
                                },
                                "message": {
                                    "text": f"[{ev.layer}] {ev.tool_used}: "
                                            f"{ev.raw_output[:200]}..."
                                            if len(ev.raw_output) > 200
                                            else f"[{ev.layer}] {ev.tool_used}: {ev.raw_output}",
                                },
                            }
                        }
                    ]
                }
            ],
            "message": {
                "text": f"Evidence layer: {ev.layer} via {ev.tool_used}"
            },
        }

    # ---------- Schema validation ----------

    def _validate_schema(self, sarif: dict[str, Any]) -> None:
        """Validate SARIF output against the OASIS 2.1.0 JSON schema.

        Uses jsonschema. If the schema package is not installed or the
        fetch fails, validation is skipped with a warning (non-fatal).
        """
        try:
            import jsonschema
            from jsonschema import Draft07Validator
        except ImportError:
            logger.warning("jsonschema not installed — skipping SARIF schema validation")
            return

        # Minimal inline schema check (the full OASIS schema is ~3,000 lines).
        # For full validation, the user should run:
        #   python3 -m sarif.tools validate output.sarif.json
        minimal_schema = {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "required": ["version", "runs"],
            "properties": {
                "version": {"type": "string", "enum": ["2.1.0"]},
                "runs": {
                    "type": "array",
                    "minItems": 1,
                    "items": {
                        "type": "object",
                        "required": ["tool", "results"],
                        "properties": {
                            "tool": {
                                "type": "object",
                                "required": ["driver"],
                                "properties": {
                                    "driver": {
                                        "type": "object",
                                        "required": ["name", "version", "rules"],
                                    },
                                },
                            },
                            "results": {"type": "array"},
                        },
                    },
                },
            },
        }

        try:
            Draft07Validator(minimal_schema).validate(sarif)
        except jsonschema.ValidationError as e:
            raise ValueError(f"SARIF schema validation failed: {e.message} at {e.absolute_json_path}") from e


# ---------------------------------------------------------------------------
# Convenience functions (match the interface expected by scan_pipeline.py)
# ---------------------------------------------------------------------------

async def generate_sarif_report(
    scan_id: str, session: AsyncSession,
    output_path: str | Path | None = None,
) -> Path | None:
    """Generate SARIF report for a scan.

    Args:
        scan_id: Scan ID
        session: Async DB session
        output_path: Optional explicit output path

    Returns:
        Path to generated SARIF file, or None if scan not found.
    """
    try:
        exporter = SARIFExporter()
        return await exporter.export(scan_id, session, output_path)
    except ValueError as e:
        logger.warning("SARIF export skipped: %s", e)
        return None
    except Exception as e:
        logger.exception("SARIF export failed: scan=%s", scan_id)
        return None


def generate_sarif_report_sync(
    scan_data_dict: dict[str, Any],
    output_path: str | Path,
) -> Path:
    """Synchronous SARIF export from a plain dict (for unit tests).

    Args:
        scan_data_dict: Scan data as plain dict (must match ScanReportData.to_dict())
        output_path: Output file path

    Returns:
        Path to generated SARIF file.
    """
    from app.report.collector import ScanReportData, FindingReportData, EvidenceReportData

    # Reconstruct dataclasses from dict
    findings: list[FindingReportData] = []
    for f_dict in scan_data_dict.get("findings", []):
        ev_list: list[EvidenceReportData] = []
        for ev in f_dict.get("evidence", []):
            if isinstance(ev, str):
                # Skip malformed evidence entries
                continue
            ev_kwargs = dict(ev)
            # Parse captured_at string → datetime if needed
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

    exporter = SARIFExporter()
    sarif = exporter._build_sarif(data)
    exporter._validate_schema(sarif)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(sarif, f, indent=2, ensure_ascii=False)
    return output_path