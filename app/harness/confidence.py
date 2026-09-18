"""
VAPT-AI Confidence Scorer — W12-S6.

4-dim confidence scoring per master plan §12 W12:
    Evidence    0.35  (5-layer evidence chain completeness)
    Reasoning   0.25  (LLM reasoning quality — heuristic)
    Verification 0.30 (Verifier strategy result)
    Historical  0.10  (KnowledgeGraph edge probability — W17)

Findings below threshold 0.6 → REJECTED.

Usage:
    from app.harness.confidence import ConfidenceScorer, ConfidenceScore
    from app.harness.types import VerificationResult, VerificationStatus

    scorer = ConfidenceScorer()
    score = scorer.score(
        evidence_score=0.8,       # 5-layer chain completeness (0.0-1.0)
        reasoning_score=0.7,      # LLM reasoning quality (0.0-1.0)
        verification_result=result,  # from VulnerabilityVerifier
        historical_score=0.5,      # KG edge probability (0.0-1.0)
    )
    print(score.total)         # 0.0-1.0
    print(score.accepted)      # True if >= 0.6 threshold
    print(score.breakdown)     # dict with per-dim contributions
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.harness.types import VerificationResult, VerificationStatus


# ---------- Constants (per master plan §12 W12) ----------

# 4-dim weights — must sum to 1.0
EVIDENCE_WEIGHT = 0.35       # 5-layer evidence chain completeness
REASONING_WEIGHT = 0.25      # LLM reasoning quality
VERIFICATION_WEIGHT = 0.30   # Verifier strategy result
HISTORICAL_WEIGHT = 0.10     # KnowledgeGraph edge probability (W17)

# Acceptance threshold — findings below this are rejected
DEFAULT_MIN_CONFIDENCE = 0.6  # per master plan §12 W12


# ---------- ConfidenceScore dataclass ----------

@dataclass(frozen=True)
class ConfidenceScore:
    """4-dim confidence score for a finding.

    Attributes:
        total: Weighted sum (0.0-1.0)
        evidence: Evidence dim raw score (0.0-1.0) × EVIDENCE_WEIGHT
        reasoning: Reasoning dim raw score × REASONING_WEIGHT
        verification: Verification dim raw score × VERIFICATION_WEIGHT
        historical: Historical dim raw score × HISTORICAL_WEIGHT
        accepted: True if total >= DEFAULT_MIN_CONFIDENCE (0.6)
        threshold: The min_confidence threshold used
    """
    total: float
    evidence: float
    reasoning: float
    verification: float
    historical: float
    accepted: bool
    threshold: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": round(self.total, 4),
            "evidence": round(self.evidence, 4),
            "reasoning": round(self.reasoning, 4),
            "verification": round(self.verification, 4),
            "historical": round(self.historical, 4),
            "accepted": self.accepted,
            "threshold": self.threshold,
        }


# ---------- ConfidenceScorer class ----------

class ConfidenceScorer:
    """4-dim confidence scorer.

    Combines evidence chain completeness, LLM reasoning quality, verifier
    strategy result, and historical KG probability into a single confidence
    score. Findings below 0.6 threshold are rejected (FP filtering).
    """

    def __init__(self, min_confidence: float = DEFAULT_MIN_CONFIDENCE):
        self.min_confidence = min_confidence

    def score(
        self,
        evidence_score: float,
        reasoning_score: float,
        verification_result: VerificationResult,
        historical_score: float = 0.5,  # default: no KG history yet (W17)
    ) -> ConfidenceScore:
        """Compute 4-dim confidence score.

        Args:
            evidence_score: 5-layer evidence chain completeness (0.0-1.0).
                            Heuristic: how many of (detection, validation,
                            exploitation, post-exploitation, audit) layers
                            have evidence?
            reasoning_score: LLM reasoning quality (0.0-1.0).
                            Heuristic: based on thought length, references
                            to evidence, etc. (W12 stub: agent-supplied)
            verification_result: From VulnerabilityVerifier.
                            Converted to 0.0-1.0 score based on status +
                            confidence.
            historical_score: KG edge probability (0.0-1.0).
                            W17 will populate from KGEdge.probability.
                            Default 0.5 (uncertain) for W12.

        Returns:
            ConfidenceScore with total + breakdown + accepted flag.
        """
        # Validate inputs
        evidence_score = self._clamp(evidence_score)
        reasoning_score = self._clamp(reasoning_score)
        historical_score = self._clamp(historical_score)

        # Convert VerificationResult to 0.0-1.0 score
        verification_score = self._verification_to_score(verification_result)

        # Compute weighted contributions
        evidence_contribution = evidence_score * EVIDENCE_WEIGHT
        reasoning_contribution = reasoning_score * REASONING_WEIGHT
        verification_contribution = verification_score * VERIFICATION_WEIGHT
        historical_contribution = historical_score * HISTORICAL_WEIGHT

        total = (
            evidence_contribution
            + reasoning_contribution
            + verification_contribution
            + historical_contribution
        )

        # Check acceptance threshold
        accepted = total >= self.min_confidence

        return ConfidenceScore(
            total=total,
            evidence=evidence_contribution,
            reasoning=reasoning_contribution,
            verification=verification_contribution,
            historical=historical_contribution,
            accepted=accepted,
            threshold=self.min_confidence,
        )

    def score_from_finding(
        self,
        finding: dict[str, Any],
        verification_result: VerificationResult,
    ) -> ConfidenceScore:
        """Score a finding dict (convenience method).

        Args:
            finding: Finding dict with optional fields:
                - evidence_layers: list of layer names with evidence
                - reasoning_score: agent-supplied reasoning quality (0.0-1.0)
                - kg_probability: historical KG edge probability (0.0-1.0)
            verification_result: From VulnerabilityVerifier.

        Returns:
            ConfidenceScore.
        """
        # Evidence score: based on how many of 5 layers have evidence
        evidence_layers = finding.get("evidence_layers", [])
        if isinstance(evidence_layers, list):
            evidence_score = len(evidence_layers) / 5.0
        else:
            evidence_score = 0.5  # default if not provided

        # Reasoning score: agent-supplied or default
        reasoning_score = finding.get("reasoning_score", 0.5)

        # Historical score: KG probability or default
        historical_score = finding.get("kg_probability", 0.5)

        return self.score(
            evidence_score=evidence_score,
            reasoning_score=reasoning_score,
            verification_result=verification_result,
            historical_score=historical_score,
        )

    def _verification_to_score(self, result: VerificationResult) -> float:
        """Convert VerificationResult to 0.0-1.0 score.

        - VERIFIED: use result.confidence (0.7-1.0 typically)
        - UNVERIFIED: 0.2 (low — no strategy matched)
        - INCONCLUSIVE: 0.4 (medium — couldn't determine)
        - FALSE_POSITIVE: 0.0 (definitive rejection)
        """
        if result.status == VerificationStatus.VERIFIED:
            return result.confidence
        if result.status == VerificationStatus.FALSE_POSITIVE:
            return 0.0
        if result.status == VerificationStatus.INCONCLUSIVE:
            return 0.4
        # UNVERIFIED
        return 0.2

    def _clamp(self, value: float) -> float:
        """Clamp value to 0.0-1.0 range."""
        return max(0.0, min(1.0, float(value)))

    def get_weights(self) -> dict[str, float]:
        """Return the 4-dim weights (for introspection / API)."""
        return {
            "evidence": EVIDENCE_WEIGHT,
            "reasoning": REASONING_WEIGHT,
            "verification": VERIFICATION_WEIGHT,
            "historical": HISTORICAL_WEIGHT,
        }