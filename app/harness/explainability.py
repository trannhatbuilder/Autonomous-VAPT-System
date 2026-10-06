"""
VAPT-AI Explainability Engine — Phase L v2 (W19-FIX5).

Ported from EVVO shield_engine/explainability.py (683 LOC).

Provides per-finding confidence breakdown (4-dim: Evidence 35% +
Reasoning 25% + Verification 30% + Historical 10%) + "Why we're confident"
+ "How to Disprove" + "Recommended Next Steps" — matches EVVO report
template structure.

Public API:
    from app.harness.explainability import explain_finding, ExplainabilityEngine

    engine = ExplainabilityEngine()
    explanation = engine.explain_finding({
        "name": "SQL Injection in /login",
        "vuln_type": "sqli",
        "severity": "critical",
        "location": "https://target/login",
        "cvss_score": 9.8,
        "verification_output": "[INFO] GET parameter 'user' appears to be '...SQLi' injectable",
        "evidence": [{"source": "sqlmap", "content": "..."}],
        "confidence_score": 0.95,
        "auditor_verdict": "confirmed",
    })
    # explanation.confidence_breakdown → list of 4 ConfidenceBreakdown
    # explanation.why_confident → markdown text
    # explanation.how_to_disprove → str
    # explanation.recommended_next_steps → list[str]

Integration:
    - Called by app/pentest/scan_pipeline.py after auditor.verify_finding() accepts
    - Result stored in Finding.metadata_json["explanation"]
    - Exposed to frontend via app/report/collector.py FindingReportData.explanation
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# ── Confidence breakdown dimensions (per EVVO shield_engine/explainability.py:275-329) ──

# Evidence Strength: weight 0.35 — based on evidence count + relevance
# Reasoning Quality: weight 0.25 — based on reasoning step confidence avg
# Verification Status: weight 0.30 — from auditor verify_finding result
# Historical Accuracy: weight 0.10 — from past confirmed/FP ratio (KG)

EVIDENCE_WEIGHT = 0.35
REASONING_WEIGHT = 0.25
VERIFICATION_WEIGHT = 0.30
HISTORICAL_WEIGHT = 0.10


@dataclass
class ConfidenceBreakdown:
    """One dimension of the 4-dim confidence breakdown.

    Matches EVVO ConfidenceBreakdown dataclass.
    """
    component: str        # "Evidence Strength" / "Reasoning Quality" / etc.
    score: float          # 0.0 - 1.0
    weight: float         # 0.35 / 0.25 / 0.30 / 0.10
    description: str      # Human-readable explanation of the score

    def to_dict(self) -> dict[str, Any]:
        return {
            "component": self.component,
            "score": round(self.score, 3),
            "weight": self.weight,
            "description": self.description,
        }


@dataclass
class FindingExplanation:
    """Full explanation for a finding (per EVVO report template).

    Fields:
        confidence_breakdown: list of 4 ConfidenceBreakdown
        overall_confidence: weighted sum (0.0 - 1.0)
        why_confident: markdown text explaining reasoning
        how_to_disprove: command user can run to verify/disprove
        recommended_next_steps: list of actionable steps
    """
    confidence_breakdown: list[ConfidenceBreakdown]
    overall_confidence: float
    why_confident: str
    how_to_disprove: str
    recommended_next_steps: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "confidence_breakdown": [b.to_dict() for b in self.confidence_breakdown],
            "overall_confidence": round(self.overall_confidence, 3),
            "why_confident": self.why_confident,
            "how_to_disprove": self.how_to_disprove,
            "recommended_next_steps": self.recommended_next_steps,
        }


class ExplainabilityEngine:
    """Per-finding explanation generator (4-dim confidence + text).

    Ported from EVVO shield_engine/explainability.py.
    """

    def explain_finding(
        self,
        finding: dict[str, Any],
        verification_result: dict[str, Any] | None = None,
        historical_accuracy: float = 0.5,
    ) -> FindingExplanation:
        """Generate full explanation for a finding.

        Args:
            finding: dict with name, vuln_type, severity, location,
                cvss_score, verification_output, evidence, confidence_score,
                auditor_verdict (optional)
            verification_result: dict from auditor.verify_finding().to_dict()
                (status, confidence, method, evidence). If None, falls back
                to finding.confidence_score + auditor_verdict.
            historical_accuracy: 0.0-1.0 ratio of similar findings that were
                confirmed true (default 0.5 = no data).

        Returns:
            FindingExplanation with 4-dim breakdown + texts.
        """
        # Compute the 4 dimensions
        evidence_strength = self._compute_evidence_strength(finding)
        reasoning_quality = self._compute_reasoning_quality(finding, verification_result)
        verification_score = self._compute_verification_status(finding, verification_result)
        historical = max(0.0, min(1.0, historical_accuracy))

        breakdown = [
            ConfidenceBreakdown(
                component="Evidence Strength",
                score=evidence_strength,
                weight=EVIDENCE_WEIGHT,
                description=self._describe_evidence_strength(evidence_strength),
            ),
            ConfidenceBreakdown(
                component="Reasoning Quality",
                score=reasoning_quality,
                weight=REASONING_WEIGHT,
                description=self._describe_reasoning_quality(reasoning_quality),
            ),
            ConfidenceBreakdown(
                component="Verification Status",
                score=verification_score,
                weight=VERIFICATION_WEIGHT,
                description=self._describe_verification_status(verification_score),
            ),
            ConfidenceBreakdown(
                component="Historical Accuracy",
                score=historical,
                weight=HISTORICAL_WEIGHT,
                description=f"{historical:.0%} of similar findings were confirmed true",
            ),
        ]

        overall = self._compute_overall_confidence(breakdown)
        why_confident = self._generate_why_confident(finding, breakdown, overall)
        how_to_disprove = self._generate_falsifiability_test(finding)
        next_steps = self._generate_next_steps(finding, overall)

        return FindingExplanation(
            confidence_breakdown=breakdown,
            overall_confidence=overall,
            why_confident=why_confident,
            how_to_disprove=how_to_disprove,
            recommended_next_steps=next_steps,
        )

    # ── Dimension computation ─────────────────────────────────────────

    def _compute_evidence_strength(self, finding: dict[str, Any]) -> float:
        """Evidence Strength dimension (0.0-1.0).

        Based on:
        - Number of evidence items (more = stronger)
        - Relevance of evidence (verified PoC > raw detection > none)
        - Diversity (multiple tools = stronger)
        """
        evidence = finding.get("evidence") or []
        verification_output = finding.get("verification_output") or ""
        detection_output = finding.get("detection_output") or ""

        score = 0.0
        # Base: 0.3 if any evidence at all
        if evidence or verification_output or detection_output:
            score += 0.3
        # +0.2 per additional evidence item (cap at +0.4)
        score += min(0.4, len(evidence) * 0.2)
        # +0.2 if verification output present (independent confirmation)
        if verification_output:
            score += 0.2
        # +0.1 if multiple tools used (diversity)
        tools_used = {ev.get("tool_used") for ev in evidence if isinstance(ev, dict) and ev.get("tool_used")}
        if len(tools_used) >= 2:
            score += 0.1

        return min(1.0, score)

    def _compute_reasoning_quality(
        self,
        finding: dict[str, Any],
        verification_result: dict[str, Any] | None,
    ) -> float:
        """Reasoning Quality dimension (0.0-1.0).

        Based on:
        - Whether finding has auditor verdict (confirmed > unverified)
        - Whether verification method is specific (not just "manual")
        - Confidence score from auditor
        """
        score = 0.5  # default if no reasoning info

        # +0.2 if auditor confirmed (not just record_vulnerability raw)
        auditor_verdict = (finding.get("auditor_verdict") or "").lower()
        if auditor_verdict == "confirmed":
            score += 0.2
        elif auditor_verdict == "rejected":
            score -= 0.3
        elif auditor_verdict == "inconclusive":
            score -= 0.1

        # +0.2 if verification method is specific (sqlmap_confirmation, etc.)
        if verification_result and verification_result.get("method"):
            method = verification_result.get("method", "").lower()
            if "sqlmap" in method or "nuclei" in method or "playwright" in method:
                score += 0.2

        # Use existing confidence_score as tiebreaker
        confidence = finding.get("confidence_score")
        if isinstance(confidence, (int, float)):
            score = (score + float(confidence)) / 2  # blend

        return max(0.0, min(1.0, score))

    def _compute_verification_status(
        self,
        finding: dict[str, Any],
        verification_result: dict[str, Any] | None,
    ) -> float:
        """Verification Status dimension (0.0-1.0).

        From EVVO _build_confidence_breakdown verification_score logic:
        - verified status + confidence → up to 1.0
        - false_positive → 0.1
        - inconclusive → 0.4
        - default 0.5
        """
        if not verification_result:
            # Fall back to finding.verified flag + confidence_score
            if finding.get("verified"):
                return min(1.0, float(finding.get("confidence_score") or 0.5))
            return 0.5

        v_status = (verification_result.get("status") or "").lower()
        v_confidence = float(verification_result.get("confidence") or 0.0)

        if v_status == "verified":
            return min(1.0, max(0.0, v_confidence))
        elif v_status == "false_positive":
            return 0.1
        elif v_status == "inconclusive":
            return 0.4
        # Default
        return 0.5

    def _compute_overall_confidence(self, breakdown: list[ConfidenceBreakdown]) -> float:
        """Weighted sum of breakdown scores."""
        total = sum(b.score * b.weight for b in breakdown)
        total_weight = sum(b.weight for b in breakdown)
        return round(total / total_weight, 3) if total_weight > 0 else 0.5

    # ── Description helpers ──────────────────────────────────────────

    def _describe_evidence_strength(self, score: float) -> str:
        if score >= 0.8:
            return "Strong evidence with independent verification"
        elif score >= 0.5:
            return "Moderate evidence — single source or limited diversity"
        elif score >= 0.3:
            return "Weak evidence — detection-only, no PoC"
        return "Insufficient evidence"

    def _describe_reasoning_quality(self, score: float) -> str:
        if score >= 0.8:
            return "Sound reasoning with clear logical steps + auditor confirmation"
        elif score >= 0.5:
            return "Reasonable reasoning — auditor verdict present"
        elif score >= 0.3:
            return "Limited reasoning — manual assessment only"
        return "Insufficient reasoning"

    def _describe_verification_status(self, score: float) -> str:
        if score >= 0.8:
            return "Independently verified with high confidence"
        elif score >= 0.5:
            return "Verified — confidence medium"
        elif score >= 0.3:
            return "Inconclusive — needs more verification"
        return "Not verified or false positive suspected"

    # ── Text generation ──────────────────────────────────────────────

    def _generate_why_confident(
        self,
        finding: dict[str, Any],
        breakdown: list[ConfidenceBreakdown],
        overall: float,
    ) -> str:
        """Generate 'Why we're confident' markdown text.

        Ported from EVVO _generate_detailed_explanation.
        """
        parts: list[str] = []
        name = finding.get("name", "Unknown")
        location = finding.get("location", "")
        severity = finding.get("severity", "info")

        parts.append(f"## Finding: {name}")
        parts.append("")
        parts.append("### What was observed")
        verification_output = finding.get("verification_output") or finding.get("evidence", [{}])[0].get("raw_output") if finding.get("evidence") else ""
        if verification_output:
            parts.append(f"The agent observed the following evidence:")
            parts.append(f"> {str(verification_output)[:500]}...")
        else:
            parts.append("No specific evidence output recorded.")
        parts.append("")
        parts.append("### Reasoning process")
        # Build reasoning from breakdown
        for b in breakdown:
            parts.append(f"1. **{b.component}** ({b.score:.0%}, weight {b.weight:.0%}) — {b.description}")
        parts.append("")
        parts.append(f"### Overall confidence: {overall:.0%}")
        parts.append("This finding is well-supported by evidence and verification." if overall >= 0.7
                     else "This finding has limited support — review before acting.")
        return "\n".join(parts)

    def _generate_falsifiability_test(self, finding: dict[str, Any]) -> str:
        """Generate a command user can run to verify/disprove.

        Ported from EVVO _generate_falsifiability_test.
        """
        vuln_type = (finding.get("vuln_type") or "").lower()
        location = finding.get("location", "")

        if "header" in vuln_type or "hsts" in vuln_type or "csp" in vuln_type or "x-frame" in vuln_type:
            return f"Run `curl -sI {location}` and check if the reported missing header is actually present."
        elif "xss" in vuln_type:
            return f"Run `curl -s '{location}?param=<b>test</b>'` and check if HTML tags are escaped in response."
        elif "sqli" in vuln_type:
            return f"Run `curl -s '{location}?id=1 AND 1=2'` and verify the response differs from `id=1 AND 1=1`."
        elif "cookie" in vuln_type:
            return f"Run `curl -sI -c /tmp/cookies.txt {location}` and inspect Set-Cookie headers for security flags."
        elif "cors" in vuln_type:
            return f"Run `curl -sI -H 'Origin: https://evil.example' {location}` and check Access-Control-Allow-Origin response."
        elif "rce" in vuln_type or "werkzeug" in vuln_type:
            return f"Try accessing {location}/console — if debug console opens, RCE is confirmed."
        elif "access" in vuln_type or "auth" in vuln_type:
            return f"Try accessing {location} without authentication (clear cookies + use incognito)."
        else:
            return f"Re-run the verification command with a modified payload and compare results."

    def _generate_next_steps(self, finding: dict[str, Any], confidence: float) -> list[str]:
        """Generate recommended next steps.

        Ported from EVVO _generate_next_steps.
        """
        steps: list[str] = []

        if confidence < 0.8:
            steps.append("Run additional verification commands to confirm the finding")

        steps.append("Review the affected endpoint for business impact")
        steps.append("Check if the vulnerability is exploitable in the current context")

        severity = (finding.get("severity") or "info").lower()
        if severity in ("critical", "high"):
            steps.append("Prioritize remediation due to high severity")

        steps.append("Document the finding with clear reproduction steps")

        return steps


# ── Convenience function ─────────────────────────────────────────────

_engine: ExplainabilityEngine | None = None


def get_explainability_engine() -> ExplainabilityEngine:
    """Get the singleton ExplainabilityEngine instance."""
    global _engine
    if _engine is None:
        _engine = ExplainabilityEngine()
    return _engine


def explain_finding(
    finding: dict[str, Any],
    verification_result: dict[str, Any] | None = None,
    historical_accuracy: float = 0.5,
) -> FindingExplanation:
    """Convenience wrapper — see ExplainabilityEngine.explain_finding."""
    return get_explainability_engine().explain_finding(
        finding=finding,
        verification_result=verification_result,
        historical_accuracy=historical_accuracy,
    )


__all__ = [
    "ExplainabilityEngine",
    "FindingExplanation",
    "ConfidenceBreakdown",
    "explain_finding",
    "get_explainability_engine",
    "EVIDENCE_WEIGHT",
    "REASONING_WEIGHT",
    "VERIFICATION_WEIGHT",
    "HISTORICAL_WEIGHT",
]
