"""
VAPT-AI Report Data Collector (W13-S3).

Reads scan + findings + evidence + audit log from DB and assembles a
plain-data snapshot that the PDF + SARIF exporters consume. This decouples
report rendering from ORM details, so exporters can be unit-tested with
plain dicts (W13-S6).

Output: ScanReportData dataclass tree (scan → findings → evidence[]).
"""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models.scan import Scan, ConsentForm
from app.db.models.pentest import Finding, Evidence
from app.db.models.audit import AuditLog
from app.db.models.hitl import HITLApproval

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataclasses — plain-data snapshot of a scan for report generation
# ---------------------------------------------------------------------------

@dataclass
class EvidenceReportData:
    """Single evidence row in a report."""
    layer: str               # detection / validation / exploitation / post_exploitation / audit
    raw_output: str
    tool_used: str
    custody_seal: str
    evidence_hash: str
    captured_at: datetime
    spill_path: str | None = None

    @classmethod
    def from_orm(cls, ev: Evidence) -> "EvidenceReportData":
        return cls(
            layer=ev.layer,
            raw_output=ev.raw_output,
            tool_used=ev.tool_used,
            custody_seal=ev.custody_seal,
            evidence_hash=ev.evidence_hash,
            captured_at=ev.captured_at,
            spill_path=ev.spill_path,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "layer": self.layer,
            "raw_output": self.raw_output,
            "tool_used": self.tool_used,
            "custody_seal": self.custody_seal,
            "evidence_hash": self.evidence_hash,
            "captured_at": self.captured_at.isoformat(),
            "spill_path": self.spill_path,
        }


@dataclass
class FindingReportData:
    """Single finding in a report."""
    id: uuid.UUID
    name: str
    vuln_type: str
    severity: str
    cvss_vector: str | None
    cvss_base_score: float | None
    cvss_severity: str | None
    location: str
    cwe_id: str | None
    cve_id: str | None
    wstg_test_id: str | None
    mitre_attack_technique: str | None
    mitre_attack_tactic: str | None
    mitre_attack_subtechnique: str | None
    poc_status: str
    poc_tier: int | None
    exploit_method: str | None
    remediation: str | None
    verified: bool
    false_positive: bool
    auditor_verdict: str | None
    confidence_score: float
    evidence: list[EvidenceReportData] = field(default_factory=list)

    @classmethod
    def from_orm(cls, f: Finding, evidence: list[Evidence]) -> "FindingReportData":
        return cls(
            id=f.id,
            name=f.name,
            vuln_type=f.vuln_type,
            severity=f.severity,
            cvss_vector=f.cvss_vector,
            cvss_base_score=f.cvss_base_score,
            cvss_severity=f.cvss_severity,
            location=f.location,
            cwe_id=f.cwe_id,
            cve_id=f.cve_id,
            wstg_test_id=f.wstg_test_id,
            mitre_attack_technique=f.mitre_attack_technique,
            mitre_attack_tactic=f.mitre_attack_tactic,
            mitre_attack_subtechnique=f.mitre_attack_subtechnique,
            poc_status=f.poc_status,
            poc_tier=f.poc_tier,
            exploit_method=f.exploit_method,
            remediation=f.remediation,
            verified=f.verified,
            false_positive=f.false_positive,
            auditor_verdict=f.auditor_verdict,
            confidence_score=f.confidence_score,
            evidence=[EvidenceReportData.from_orm(ev) for ev in evidence],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": str(self.id),
            "name": self.name,
            "vuln_type": self.vuln_type,
            "severity": self.severity,
            "cvss_vector": self.cvss_vector,
            "cvss_base_score": self.cvss_base_score,
            "cvss_severity": self.cvss_severity,
            "location": self.location,
            "cwe_id": self.cwe_id,
            "cve_id": self.cve_id,
            "wstg_test_id": self.wstg_test_id,
            "mitre_attack_technique": self.mitre_attack_technique,
            "mitre_attack_tactic": self.mitre_attack_tactic,
            "mitre_attack_subtechnique": self.mitre_attack_subtechnique,
            "poc_status": self.poc_status,
            "poc_tier": self.poc_tier,
            "exploit_method": self.exploit_method,
            "remediation": self.remediation,
            "verified": self.verified,
            "false_positive": self.false_positive,
            "auditor_verdict": self.auditor_verdict,
            "confidence_score": self.confidence_score,
            "evidence": [ev.to_dict() for ev in self.evidence],
        }


@dataclass
class ScanReportData:
    """Top-level scan report data — assembled by collect_scan_data()."""
    scan_id: str
    target: str
    target_type: str
    agent_mode: str
    hitl_mode: str
    user_prompt: str | None
    status: str
    progress: int
    started_at: datetime | None
    completed_at: datetime | None
    result_summary: dict[str, Any] | None
    error: str | None
    findings: list[FindingReportData] = field(default_factory=list)
    audit_entries: list[dict[str, Any]] = field(default_factory=list)
    hitl_approvals: list[dict[str, Any]] = field(default_factory=list)
    consent: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "scan_id": self.scan_id,
            "target": self.target,
            "target_type": self.target_type,
            "agent_mode": self.agent_mode,
            "hitl_mode": self.hitl_mode,
            "user_prompt": self.user_prompt,
            "status": self.status,
            "progress": self.progress,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "result_summary": self.result_summary,
            "error": self.error,
            "findings": [f.to_dict() for f in self.findings],
            "audit_entries": self.audit_entries,
            "hitl_approvals": self.hitl_approvals,
            "consent": self.consent,
        }


# ---------------------------------------------------------------------------
# Collector
# ---------------------------------------------------------------------------

async def collect_scan_data(scan_id: str, session: AsyncSession) -> ScanReportData:
    """Assemble the full scan report data from DB.

    Args:
        scan_id: Scan ID (string form, e.g. "scan_abc123")
        session: Async SQLAlchemy session

    Returns:
        ScanReportData with scan row + all findings + evidence chain +
        audit log + HITL approvals + consent form (if any).

    Raises:
        ValueError: if scan_id not found in vapt_scans table.
    """
    scan_row = await session.get(Scan, scan_id)
    if scan_row is None:
        raise ValueError(f"Scan {scan_id!r} not found")

    # Findings + evidence chain
    findings_result = await session.execute(
        select(Finding).where(Finding.scan_id == scan_id)
        .order_by(Finding.severity.asc(), Finding.created_at.asc())
    )
    findings_orm = findings_result.scalars().all()

    findings: list[FindingReportData] = []
    for f in findings_orm:
        evidence_result = await session.execute(
            select(Evidence).where(Evidence.finding_id == f.id)
            .order_by(Evidence.captured_at.asc())
        )
        evidence_orm = evidence_result.scalars().all()
        findings.append(FindingReportData.from_orm(f, list(evidence_orm)))

    # Audit log entries for this scan
    audit_result = await session.execute(
        select(AuditLog).where(AuditLog.scan_id == scan_id)
        .order_by(AuditLog.created_at.asc())
    )
    audit_entries = [
        {
            "id": str(a.id),
            "action": a.action,
            "actor_type": a.actor_type,
            "actor_id": a.actor_id,
            "target_table": a.target_table,
            "target_id": a.target_id,
            "ip_address": a.ip_address,
            "tamper_seal": a.tamper_seal,
            "timestamp": a.created_at.isoformat() if a.created_at else None,
        }
        for a in audit_result.scalars().all()
    ]

    # HITL approvals for this scan
    hitl_result = await session.execute(
        select(HITLApproval).where(HITLApproval.scan_id == scan_id)
        .order_by(HITLApproval.created_at.asc())
    )
    hitl_approvals = [
        {
            "id": str(h.id),
            "tool_name": h.tool_name,
            "target": h.target,
            "status": h.status,
            "user_decision": h.user_decision,
            "predicted_impact": h.predicted_impact,
            "agent_reasoning": h.agent_reasoning,
            "kg_confidence": h.kg_confidence,
            "timestamp": h.created_at.isoformat() if h.created_at else None,
            "decided_at": h.decided_at.isoformat() if h.decided_at else None,
        }
        for h in hitl_result.scalars().all()
    ]

    # Consent form (if linked)
    consent_row = await session.execute(
        select(ConsentForm).where(ConsentForm.scan_id == scan_id).limit(1)
    )
    consent_orm = consent_row.scalar_one_or_none()
    consent_dict = None
    if consent_orm is not None:
        consent_dict = {
            "asserted_owner": getattr(consent_orm, "asserted_owner", None),
            "declared_scope": getattr(consent_orm, "declared_scope_json", {}) or {},
            "verification_method": getattr(consent_orm, "verification_method", None),
            "tos_accepted_at": getattr(consent_orm, "tos_accepted_at", None).isoformat()
                if getattr(consent_orm, "tos_accepted_at", None) else None,
        }

    return ScanReportData(
        scan_id=scan_row.id,
        target=scan_row.target,
        target_type=scan_row.target_type,
        agent_mode=scan_row.agent_mode,
        hitl_mode=scan_row.hitl_mode,
        user_prompt=scan_row.user_prompt,
        status=scan_row.status,
        progress=scan_row.progress,
        started_at=scan_row.started_at,
        completed_at=scan_row.completed_at,
        result_summary=scan_row.result_summary,
        error=scan_row.error,
        findings=findings,
        audit_entries=audit_entries,
        hitl_approvals=hitl_approvals,
        consent=consent_dict,
    )
