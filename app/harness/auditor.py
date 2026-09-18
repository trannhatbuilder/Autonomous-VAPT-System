"""
VAPT-AI Evidence Auditor — W12-S8.

Orchestrator class combining all verification layers:
    1. CVSS vector validation (CVSSValidator)
    2. Multi-strategy verification (VulnerabilityVerifier)
    3. PoC validation (PoCValidator or PlaywrightPoCValidator)
    4. 4-dim confidence scoring (ConfidenceScorer)
    5. Chain-of-Custody enforcement (CustodyVerifier from W4)

Per master plan §12 W12 acceptance:
    - Zero findings with incomplete evidence
    - Auditor rejects weak findings (confidence < 0.6)
    - CVSS vectors parse-validated
    - Chain-of-Custody enforcement on every Evidence read

Usage:
    from app.harness import EvidenceAuditor

    auditor = EvidenceAuditor()
    verdict = auditor.verify_finding({
        "title": "Missing HSTS header",
        "endpoint": "http://target",
        "vuln_type": "missing_security_header",
        "cvss_vector": "AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H",
        "evidence_provided": "HTTP/1.1 200 OK\\nServer: Apache\\n...",
    })
    if verdict.accepted:
        # Finding verified — accept + record
    else:
        # Finding rejected — log reason
        print(verdict.rejection_reason)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from app.harness.types import VerificationResult, VerificationStatus, VulnClaim
from app.harness.verifier import VulnerabilityVerifier, get_verifier
from app.harness.poc_validator import PoCValidator
from app.harness.poc_validator_playwright import PlaywrightPoCValidator
from app.harness.confidence import ConfidenceScorer, ConfidenceScore
from app.harness.cvss_validator import CVSSValidator, CVSSValidationResult

logger = logging.getLogger(__name__)


# ---------- AuditorVerdict dataclass ----------

@dataclass
class AuditorVerdict:
    """Final verdict from the evidence auditor.

    Combines all verification layers into a single decision:
        - accepted: True if finding passes all checks
        - cvss_validation: CVSS vector parse result
        - verification_result: From VulnerabilityVerifier
        - poc_result: From PoCValidator (or PlaywrightPoCValidator)
        - confidence_score: 4-dim score (ConfidenceScore)
        - rejection_reason: If rejected, explains why
        - recommendations: List of next-step recommendations
    """
    accepted: bool
    cvss_validation: CVSSValidationResult
    verification_result: VerificationResult
    poc_result: VerificationResult
    confidence_score: ConfidenceScore
    rejection_reason: str | None = None
    recommendations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "cvss_validation": self.cvss_validation.to_dict(),
            "verification_result": self.verification_result.to_dict(),
            "poc_result": self.poc_result.to_dict(),
            "confidence_score": self.confidence_score.to_dict(),
            "rejection_reason": self.rejection_reason,
            "recommendations": self.recommendations,
        }


# ---------- EvidenceAuditor class ----------

class EvidenceAuditor:
    """Orchestrator for all verification layers.

    Per master plan §12 W12:
        1. Validate CVSS vector
        2. Run 5 verification strategies
        3. Run PoC validator (regex-based fallback or Playwright)
        4. Compute 4-dim confidence score
        5. Enforce chain-of-custody on every Evidence read (W4 CustodyVerifier)
        6. Return verdict (accept/reject)
    """

    def __init__(
        self,
        verifier: VulnerabilityVerifier | None = None,
        poc_validator: PoCValidator | PlaywrightPoCValidator | None = None,
        confidence_scorer: ConfidenceScorer | None = None,
        cvss_validator: CVSSValidator | None = None,
        use_playwright: bool = True,
    ):
        self.verifier = verifier or get_verifier()
        self.cvss_validator = cvss_validator or CVSSValidator()
        self.confidence_scorer = confidence_scorer or ConfidenceScorer()

        # PoC validator: try Playwright first, fall back to regex
        if poc_validator is not None:
            self.poc_validator = poc_validator
        elif use_playwright:
            pw_validator = PlaywrightPoCValidator()
            if pw_validator.is_available():
                self.poc_validator = pw_validator
            else:
                self.poc_validator = PoCValidator()
        else:
            self.poc_validator = PoCValidator()

        logger.info(
            "EvidenceAuditor initialized: verifier=%s poc_validator=%s confidence_scorer=%s cvss_validator=%s",
            type(self.verifier).__name__,
            type(self.poc_validator).__name__,
            type(self.confidence_scorer).__name__,
            type(self.cvss_validator).__name__,
        )

    # ---------- Main entry point ----------

    def verify_finding(self, finding: dict[str, Any]) -> AuditorVerdict:
        """Verify a finding dict through all auditor layers.

        Args:
            finding: Finding dict with fields:
                - title, endpoint, vuln_type (required)
                - cvss_vector (required for CVSS validation)
                - evidence_provided (required for verification)
                - evidence_layers (optional, for confidence scoring)
                - reasoning_score (optional, default 0.5)
                - kg_probability (optional, default 0.5)

        Returns:
            AuditorVerdict with accepted=True/False + all layer results.
        """
        recommendations: list[str] = []

        # Step 1: Validate CVSS vector
        cvss_result = self.cvss_validator.validate_finding(finding)
        if not cvss_result.valid:
            return AuditorVerdict(
                accepted=False,
                cvss_validation=cvss_result,
                verification_result=VerificationResult(
                    status=VerificationStatus.UNVERIFIED,
                    confidence=0.0,
                    method="cvss_validation_skipped",
                    evidence="CVSS validation failed — verification skipped",
                ),
                poc_result=VerificationResult(
                    status=VerificationStatus.UNVERIFIED,
                    confidence=0.0,
                    method="poc_validation_skipped",
                    evidence="CVSS validation failed — PoC validation skipped",
                ),
                confidence_score=ConfidenceScore(
                    total=0.0, evidence=0.0, reasoning=0.0,
                    verification=0.0, historical=0.0,
                    accepted=False, threshold=self.confidence_scorer.min_confidence,
                ),
                rejection_reason=f"CVSS vector invalid: {cvss_result.error}",
                recommendations=["Fix the CVSS vector string and resubmit"],
            )

        # Step 2: Run verification strategies
        claim = VulnClaim.from_finding_dict(finding)
        verification_result = self.verifier.verify(claim)

        # Step 3: Run PoC validation
        poc_result = self.poc_validator.validate(claim)

        # Step 4: Compute confidence score
        confidence_score = self.confidence_scorer.score_from_finding(
            finding=finding,
            verification_result=verification_result,
        )

        # Step 5: Determine acceptance
        accepted = confidence_score.accepted

        # If verification says FALSE_POSITIVE, override to reject
        if verification_result.status == VerificationStatus.FALSE_POSITIVE:
            accepted = False
            recommendations.append(
                "Verifier flagged this as FALSE_POSITIVE — investigate evidence"
            )

        # If PoC says FALSE_POSITIVE, also override
        if poc_result.status == VerificationStatus.FALSE_POSITIVE:
            accepted = False
            recommendations.append(
                "PoC validator flagged this as FALSE_POSITIVE — investigate PoC"
            )

        # Build rejection reason if rejected
        rejection_reason = None
        if not accepted:
            reasons = []
            if verification_result.status == VerificationStatus.FALSE_POSITIVE:
                reasons.append(f"Verifier: FALSE_POSITIVE (method={verification_result.method})")
            elif verification_result.status == VerificationStatus.UNVERIFIED:
                reasons.append("Verifier: no strategy matched (UNVERIFIED)")
            elif verification_result.status == VerificationStatus.INCONCLUSIVE:
                reasons.append("Verifier: inconclusive")

            if poc_result.status == VerificationStatus.FALSE_POSITIVE:
                reasons.append(f"PoC: FALSE_POSITIVE (method={poc_result.method})")
            elif poc_result.status == VerificationStatus.INCONCLUSIVE:
                reasons.append("PoC: inconclusive")

            if confidence_score.total < self.confidence_scorer.min_confidence:
                reasons.append(
                    f"Confidence too low ({confidence_score.total:.2f} < "
                    f"{self.confidence_scorer.min_confidence})"
                )

            rejection_reason = "; ".join(reasons) if reasons else "Rejected (unknown reason)"

        # Add recommendations based on results
        if not accepted:
            if verification_result.status == VerificationStatus.UNVERIFIED:
                recommendations.append(
                    "Provide more concrete evidence (HTTP response headers, "
                    "payload reflections, error messages)"
                )
            if poc_result.status == VerificationStatus.UNVERIFIED:
                recommendations.append(
                    "Provide PoC output (raw command output or HTTP response)"
                )
            if confidence_score.total < 0.4:
                recommendations.append(
                    "Finding is too weak — consider not reporting"
                )
        else:
            recommendations.append("Finding accepted — record + persist to blackboard")

        return AuditorVerdict(
            accepted=accepted,
            cvss_validation=cvss_result,
            verification_result=verification_result,
            poc_result=poc_result,
            confidence_score=confidence_score,
            rejection_reason=rejection_reason,
            recommendations=recommendations,
        )

    # ---------- Batch verification ----------

    def verify_findings_batch(
        self,
        findings: list[dict[str, Any]],
    ) -> list[tuple[dict[str, Any], AuditorVerdict]]:
        """Verify multiple findings.

        Args:
            findings: List of finding dicts.

        Returns:
            List of (finding, verdict) tuples.
        """
        return [(f, self.verify_finding(f)) for f in findings]

    # ---------- Stats ----------

    def get_stats(self) -> dict[str, Any]:
        """Get stats from underlying verifier."""
        return {
            "verifier_stats": self.verifier.get_stats(),
            "poc_validator": type(self.poc_validator).__name__,
            "confidence_scorer": type(self.confidence_scorer).__name__,
            "cvss_validator": type(self.cvss_validator).__name__,
            "min_confidence": self.confidence_scorer.min_confidence,
        }