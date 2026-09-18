"""
VAPT-AI Harness Types — W12-S1.

Core dataclasses + enums for the evidence auditor layer.
Port of EVVO shield_engine/vulnerability_verifier.py lines 29-75.

Defines:
    - VerificationStatus: enum (verified, unverified, false_positive, inconclusive)
    - Severity: enum (critical, high, medium, low, info)
    - VerificationResult: dataclass (status, confidence, method, evidence)
    - VulnClaim: dataclass (claim made by agent — to be independently verified)

The verifier layer is OPINIONATED about independence:
    - Agent supplies a VulnClaim (assertion)
    - Verifier independently re-probes via HTTP to produce VerificationResult
    - Agent's verification_output is NEVER trusted (D17 anti-hallucination)
    - Only verifier-confirmed results can seed the KnowledgeGraph (W17)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ---------- Enums ----------

class VerificationStatus(str, Enum):
    """Status of a verification attempt.

    Order of preference (most → least definitive):
        VERIFIED: confirmed by independent test (accept finding)
        FALSE_POSITIVE: evidence contradicts claim (reject finding)
        INCONCLUSIVE: cannot determine (network error, timeout — needs retry)
        UNVERIFIED: no strategy could handle this claim type (needs manual review)
    """
    VERIFIED = "verified"
    UNVERIFIED = "unverified"
    FALSE_POSITIVE = "false_positive"
    INCONCLUSIVE = "inconclusive"

    def __str__(self) -> str:
        return self.value


class Severity(str, Enum):
    """CVSS-aligned severity levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"

    def __str__(self) -> str:
        return self.value


# ---------- Dataclasses ----------

@dataclass
class VerificationResult:
    """Result of attempting to verify a vulnerability claim.

    Attributes:
        status: VerificationStatus (verified/unverified/false_positive/inconclusive)
        confidence: 0.0 - 1.0 (how confident the verifier is in the result)
        method: Which verification strategy produced this result (e.g. "verify_xss_reflected")
        evidence: Raw evidence from verification attempt (HTTP response, regex match, etc.)
        details: Optional structured metadata (e.g. matched patterns, status codes)
    """
    status: VerificationStatus
    confidence: float  # 0.0 - 1.0
    method: str        # Which verification method was used
    evidence: str      # Raw evidence from verification attempt
    details: dict[str, Any] = field(default_factory=dict)

    def is_acceptable(self, min_confidence: float = 0.6) -> bool:
        """Check if this result meets acceptance criteria.

        A result is acceptable if:
            - status == VERIFIED
            - confidence >= min_confidence (default 0.6 per master plan §12 W12)

        Args:
            min_confidence: Minimum confidence threshold (default 0.6).

        Returns:
            True if finding should be accepted, False if rejected as FP.
        """
        return (
            self.status == VerificationStatus.VERIFIED
            and self.confidence >= min_confidence
        )

    def is_false_positive(self) -> bool:
        """Check if this result definitively marks the claim as false positive."""
        return self.status == VerificationStatus.FALSE_POSITIVE

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict (for FastAPI response + JSON)."""
        return {
            "status": self.status.value,
            "confidence": round(self.confidence, 4),
            "method": self.method,
            "evidence": self.evidence[:500] if self.evidence else "",
            "details": self.details,
            "acceptable": self.is_acceptable(),
            "false_positive": self.is_false_positive(),
        }


@dataclass
class VulnClaim:
    """A vulnerability claim made by an agent — to be independently verified.

    The verifier layer treats this as an ASSERTION, not fact. The verifier
    independently re-probes via HTTP to produce a VerificationResult.

    Attributes:
        title: Short title (e.g. "SQL Injection in /api/users")
        endpoint: Affected URL (e.g. "http://target/api/users?id=1")
        vuln_type: Vulnerability type (sqli, xss, rce, lfi, idor, csrf, ssrf, ...)
        method: HTTP method (GET, POST, PUT, DELETE)
        payload: Payload used (if any) — for replay during verification
        evidence_provided: Agent-supplied evidence (NOT trusted — for context only)
        severity: Claimed severity (critical/high/medium/low/info)
        cvss_vector: CVSS v3.1 vector string (validated separately by CVSSValidator)
        wstg_id: OWASP WSTG v4.2 ID (e.g. "WSTG-INPV-05" for SQLi)
        cwe_id: CWE ID (e.g. "CWE-89" for SQLi)
        mitre_attack: MITRE ATT&CK technique (e.g. "T1190")
        expected_status_code: Expected HTTP status code if vuln exists (optional)
        expected_pattern: Expected regex pattern in response if vuln exists (optional)
        headers: Additional headers to send during verification (optional)
    """
    title: str
    endpoint: str
    vuln_type: str
    method: str = "GET"
    payload: str = ""
    evidence_provided: str = ""  # NOT trusted — agent's claim
    severity: str = "medium"
    cvss_vector: str = ""
    wstg_id: str = ""
    cwe_id: str = ""
    mitre_attack: str = ""
    expected_status_code: int | None = None
    expected_pattern: str = ""
    headers: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict."""
        return {
            "title": self.title,
            "endpoint": self.endpoint,
            "vuln_type": self.vuln_type,
            "method": self.method,
            "payload": self.payload,
            "evidence_provided": self.evidence_provided[:500] if self.evidence_provided else "",
            "severity": self.severity,
            "cvss_vector": self.cvss_vector,
            "wstg_id": self.wstg_id,
            "cwe_id": self.cwe_id,
            "mitre_attack": self.mitre_attack,
            "expected_status_code": self.expected_status_code,
            "expected_pattern": self.expected_pattern,
            "headers": self.headers,
        }

    @classmethod
    def from_finding_dict(cls, finding: dict[str, Any]) -> VulnClaim:
        """Build a VulnClaim from a finding dict (agent output format).

        Maps common finding fields to VulnClaim fields. Missing fields
        default to empty string / None.

        Args:
            finding: Finding dict with keys like 'title', 'endpoint', 'vuln_type', etc.

        Returns:
            VulnClaim instance.
        """
        return cls(
            title=finding.get("title", "Untitled finding"),
            endpoint=finding.get("endpoint") or finding.get("location") or finding.get("url", ""),
            vuln_type=finding.get("vuln_type") or finding.get("type", "unknown"),
            method=finding.get("method", "GET"),
            payload=finding.get("payload", ""),
            # W12-S1: try multiple keys for evidence (different agents use different conventions)
            evidence_provided=(
                finding.get("evidence_provided")
                or finding.get("evidence")
                or finding.get("verification_output")
                or finding.get("detection_output")
                or ""
            ),
            severity=finding.get("severity", "medium"),
            cvss_vector=finding.get("cvss_vector", ""),
            wstg_id=finding.get("wstg_id", ""),
            cwe_id=finding.get("cwe_id", ""),
            mitre_attack=finding.get("mitre_attack", ""),
            expected_status_code=finding.get("expected_status_code"),
            expected_pattern=finding.get("expected_pattern", ""),
            headers=finding.get("headers", {}),
        )