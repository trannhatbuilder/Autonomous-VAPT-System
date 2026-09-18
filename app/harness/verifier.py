"""
VAPT-AI Vulnerability Verifier — W12-S3.

Multi-strategy vulnerability claim verifier.
Port of EVVO shield_engine/vulnerability_verifier.py (lines 671-790).

Architecture:
    - Takes a VulnClaim (agent's assertion)
    - Runs 5 verification strategies in priority order
    - Returns first VERIFIED result with confidence >= 0.8 (early exit)
    - Otherwise returns most definitive result (VERIFIED > FP > INCONCLUSIVE > UNVERIFIED)
    - Never trusts agent-supplied verification_output (D17 anti-hallucination)

W12 stub: strategies analyze agent-supplied evidence (claim.evidence_provided)
without making HTTP requests. W13+ will add live HTTP re-probing.

Usage:
    from app.harness import VulnerabilityVerifier, VulnClaim

    verifier = VulnerabilityVerifier()
    claim = VulnClaim(
        title="Missing HSTS header",
        endpoint="http://target",
        vuln_type="missing_security_header",
        evidence_provided="HTTP/1.1 200 OK\\nServer: Apache\\nContent-Type: text/html",
    )
    result = verifier.verify(claim)
    if result.is_acceptable():
        # Accept the finding
    else:
        # Reject or queue for manual review
"""
from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any, Callable

from app.harness.types import VerificationResult, VerificationStatus, VulnClaim
from app.harness.verifier_strategies import DEFAULT_STRATEGIES

logger = logging.getLogger(__name__)


# ---------- Verifier class ----------

class VulnerabilityVerifier:
    """Multi-strategy vulnerability claim verifier.

    Runs 5 verification strategies (DEFAULT_STRATEGIES) in priority order.
    Returns the most definitive result.

    Configuration:
        - strategies: list of callables (VulnClaim → VerificationResult | None)
        - min_confidence: minimum confidence for acceptance (default 0.7)
        - require_verification: if True, findings without VERIFIED status are rejected
    """

    DEFAULT_STRATEGIES: list[Callable[[VulnClaim], VerificationResult | None]] = (
        DEFAULT_STRATEGIES.copy()
    )

    def __init__(
        self,
        strategies: list[Callable[[VulnClaim], VerificationResult | None]] | None = None,
        min_confidence: float = 0.7,
        require_verification: bool = True,
    ):
        self.strategies = strategies or self.DEFAULT_STRATEGIES.copy()
        self.min_confidence = min_confidence
        self.require_verification = require_verification
        # Stats tracking — flat dict, not nested defaultdict (avoid TypeError on +=)
        self._stats: dict[str, Any] = {
            "total_claims": 0,
            "by_status": defaultdict(int),
            "by_method": defaultdict(int),
        }

    # ---------- Verify ----------

    def verify(self, claim: VulnClaim) -> VerificationResult:
        """Verify a vulnerability claim using all registered strategies.

        Returns the first VERIFIED result with confidence >= 0.8 (early exit),
        otherwise the most definitive non-verified result.

        Args:
            claim: VulnClaim to verify.

        Returns:
            VerificationResult with status, confidence, method, evidence.
        """
        logger.debug("Verifying claim: %s on %s", claim.title, claim.endpoint)

        self._stats["total_claims"] += 1
        results: list[VerificationResult] = []

        for strategy in self.strategies:
            strategy_name = strategy.__name__
            try:
                result = strategy(claim)
                if result:
                    results.append(result)
                    self._stats["by_method"][strategy_name] += 1
                    logger.debug(
                        "Strategy %s returned: %s (confidence=%.2f)",
                        strategy_name, result.status, result.confidence,
                    )

                    # Early exit on strong verification
                    if (
                        result.status == VerificationStatus.VERIFIED
                        and result.confidence >= 0.8
                    ):
                        self._update_stats(result)
                        return result
            except Exception as exc:
                logger.warning(
                    "Verification strategy %s failed: %s", strategy_name, exc
                )

        # Determine final result
        final_result = self._select_best_result(results)
        self._update_stats(final_result)
        return final_result

    def verify_batch(
        self,
        claims: list[VulnClaim],
    ) -> list[tuple[VulnClaim, VerificationResult]]:
        """Verify multiple claims and return results.

        Args:
            claims: List of VulnClaim instances.

        Returns:
            List of (claim, result) tuples.
        """
        return [(claim, self.verify(claim)) for claim in claims]

    def verify_finding_dict(self, finding: dict[str, Any]) -> VerificationResult:
        """Convenience: verify a finding dict (agent output format).

        Args:
            finding: Finding dict with keys like 'title', 'endpoint', 'vuln_type', etc.

        Returns:
            VerificationResult.
        """
        claim = VulnClaim.from_finding_dict(finding)
        return self.verify(claim)

    # ---------- Selection logic ----------

    def _select_best_result(
        self,
        results: list[VerificationResult],
    ) -> VerificationResult:
        """Select the most definitive result from a list.

        Priority: VERIFIED > FALSE_POSITIVE > INCONCLUSIVE > UNVERIFIED.
        Within each status, prefer higher confidence.

        Args:
            results: List of VerificationResult instances.

        Returns:
            Best result, or UNVERIFIED if no results.
        """
        if not results:
            return VerificationResult(
                status=VerificationStatus.UNVERIFIED,
                confidence=0.0,
                method="no_strategy_applicable",
                evidence="No verification strategy could process this claim type",
            )

        # Prefer VERIFIED > FALSE_POSITIVE > INCONCLUSIVE > UNVERIFIED
        verified = [r for r in results if r.status == VerificationStatus.VERIFIED]
        if verified:
            return max(verified, key=lambda x: x.confidence)

        false_pos = [r for r in results if r.status == VerificationStatus.FALSE_POSITIVE]
        if false_pos:
            return max(false_pos, key=lambda x: x.confidence)

        inconclusive = [r for r in results if r.status == VerificationStatus.INCONCLUSIVE]
        if inconclusive:
            return max(inconclusive, key=lambda x: x.confidence)

        # Fallback to first result
        return results[0]

    # ---------- Acceptance ----------

    def is_acceptable(self, result: VerificationResult) -> bool:
        """Check if a result meets acceptance criteria.

        Args:
            result: VerificationResult to check.

        Returns:
            True if finding should be accepted, False if rejected as FP.
        """
        if self.require_verification:
            return result.is_acceptable(self.min_confidence)
        # If verification not required, accept anything that's not a confirmed FP
        return result.status != VerificationStatus.FALSE_POSITIVE

    def should_accept_finding(
        self,
        result: VerificationResult,
    ) -> tuple[bool, str]:
        """Check if finding should be accepted + return reason.

        Args:
            result: VerificationResult to check.

        Returns:
            (accepted, reason) tuple. Reason explains why accepted/rejected.
        """
        if result.status == VerificationStatus.FALSE_POSITIVE:
            return (False, f"False positive (confidence={result.confidence:.2f}, method={result.method})")
        if result.status == VerificationStatus.VERIFIED:
            if result.confidence >= self.min_confidence:
                return (True, f"Verified (confidence={result.confidence:.2f}, method={result.method})")
            return (False, f"Verified but confidence too low ({result.confidence:.2f} < {self.min_confidence})")
        if result.status == VerificationStatus.INCONCLUSIVE:
            return (False, f"Inconclusive (confidence={result.confidence:.2f}, method={result.method})")
        # UNVERIFIED
        return (False, f"Unverified (no strategy matched, method={result.method})")

    # ---------- Stats ----------

    def _update_stats(self, result: VerificationResult) -> None:
        """Update internal stats counter based on result status."""
        status_key = result.status.value
        self._stats["by_status"][status_key] = self._stats["by_status"].get(status_key, 0) + 1
        self._stats["by_method"][result.method] = self._stats["by_method"].get(result.method, 0) + 1

    def get_stats(self) -> dict[str, Any]:
        """Get verifier statistics.

        Returns:
            Dict with: total_claims, by_status (verified/fp/inconclusive/unverified counts),
            by_method (count per strategy method).
        """
        return {
            "total_claims": self._stats.get("total_claims", 0),
            "by_status": dict(self._stats.get("by_status", {})),
            "by_method": dict(self._stats.get("by_method", {})),
            "min_confidence": self.min_confidence,
            "require_verification": self.require_verification,
            "strategies_count": len(self.strategies),
        }

    def reset_stats(self) -> None:
        """Reset stats counters (useful for tests)."""
        self._stats = {
            "total_claims": 0,
            "by_status": defaultdict(int),
            "by_method": defaultdict(int),
        }


# ---------- Module-level singleton ----------

_verifier_singleton: VulnerabilityVerifier | None = None


def get_verifier() -> VulnerabilityVerifier:
    """Get the singleton VulnerabilityVerifier instance.

    Uses default strategies + min_confidence=0.7.
    """
    global _verifier_singleton
    if _verifier_singleton is None:
        _verifier_singleton = VulnerabilityVerifier()
    return _verifier_singleton