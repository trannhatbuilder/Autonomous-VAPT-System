"""
VAPT-AI Agent Self-Correction Engine.

Python port of EVVO Sentinel shield_engine/self_correction.py (selective port).

When the Vulnerability Verifier rejects a finding, this engine:
  1. Analyzes WHY the finding was rejected
  2. Generates a corrective learning signal
  3. Updates the agent's behavior via:
     a) Immediate prompt injection (corrective system message)
     b) RL reward shaping (negative reward for rejected patterns)
     c) Pattern blacklist (after 3+ rejections of same pattern)

Implements the "agent self-correction" principle: verifier reject →
agent learns from error.

Adaptations for VAPT-AI:
  - Verifier imports: from app.harness.types (not shield_engine.vulnerability_verifier)
  - DB persistence: in-memory pattern_counts (no SQL table — defer to Phase 2+)
  - FeedbackLoop: deferred (EVVO feature, not core to VAPT-AI MVP)
  - Logger: app.harness.self_correction
"""
from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field
from typing import Any

from app.harness.types import VerificationResult, VerificationStatus, VulnClaim

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# Rejection Analysis
# ═══════════════════════════════════════════════════════════════════

@dataclass
class RejectionAnalysis:
    """Structured analysis of why a finding was rejected."""

    rejection_type: str  # "insufficient_evidence" | "false_positive" | "inconclusive" | "unverified"
    root_cause: str
    corrective_action: str
    suggested_verification: str
    confidence_penalty: float  # 0.0 - 1.0
    pattern_signature: str  # SHA-256 hash of (vuln_type, endpoint_pattern)
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "rejection_type": self.rejection_type,
            "root_cause": self.root_cause,
            "corrective_action": self.corrective_action,
            "suggested_verification": self.suggested_verification,
            "confidence_penalty": round(self.confidence_penalty, 4),
            "pattern_signature": self.pattern_signature,
            "metadata": self.metadata,
        }


class RejectionAnalyzer:
    """Analyzes verifier rejection reasons and produces structured learning signals."""

    def analyze(self, claim: VulnClaim, result: VerificationResult) -> RejectionAnalysis:
        """Analyze why a finding was rejected and what to learn.

        Args:
            claim: the agent's original vulnerability claim
            result: verifier's independent verification result

        Returns:
            RejectionAnalysis with rejection_type, root_cause, corrective_action,
            suggested_verification, confidence_penalty, pattern_signature.
        """
        # Determine rejection type based on verifier status
        if result.status == VerificationStatus.FALSE_POSITIVE:
            rejection_type = "false_positive"
            root_cause = self._analyze_false_positive(claim, result)
            confidence_penalty = 0.3
        elif result.status == VerificationStatus.INCONCLUSIVE:
            rejection_type = "inconclusive"
            root_cause = self._analyze_inconclusive(claim, result)
            confidence_penalty = 0.2
        elif result.status == VerificationStatus.UNVERIFIED:
            rejection_type = "unverified"
            root_cause = self._analyze_unverified(claim, result)
            confidence_penalty = 0.4
        else:
            rejection_type = "insufficient_evidence"
            root_cause = "Verification failed with insufficient evidence"
            confidence_penalty = 0.35

        pattern_sig = self._compute_pattern_signature(claim)
        corrective_action = self._generate_corrective_action(rejection_type, claim, result)
        suggested_verification = self._generate_suggested_verification(rejection_type, claim)

        return RejectionAnalysis(
            rejection_type=rejection_type,
            root_cause=root_cause,
            corrective_action=corrective_action,
            suggested_verification=suggested_verification,
            confidence_penalty=confidence_penalty,
            pattern_signature=pattern_sig,
            metadata={
                "claim_title": claim.title,
                "claim_endpoint": claim.endpoint,
                "claim_vuln_type": claim.vuln_type,
                "verification_method": result.method,
                "verification_confidence": result.confidence,
                "verification_evidence": (result.evidence or "")[:500],
            },
        )

    # ── root cause analyzers ────────────────────────────────────────

    def _analyze_false_positive(self, claim: VulnClaim, result: VerificationResult) -> str:
        """Analyze why a finding was a false positive."""
        evidence = (result.evidence or "").lower()
        if "not found" in evidence or "absent" in evidence:
            return "Claimed vulnerability was not present in the target response"
        if "present" in evidence and "is" in evidence:
            return "Claimed missing security control was actually present"
        if "timeout" in evidence or "error" in evidence:
            return "Verification failed due to network issues, not actual vulnerability"
        if "pattern" in evidence and "not match" in evidence:
            return "Pattern-based detection triggered on benign content"
        return "Independent verification contradicted the claimed vulnerability"

    def _analyze_inconclusive(self, claim: VulnClaim, result: VerificationResult) -> str:
        """Analyze why verification was inconclusive."""
        evidence = (result.evidence or "").lower()
        if "no" in evidence and ("set-cookie" in evidence or "cookie" in evidence):
            return "Could not verify cookie security — no cookies in response"
        if "network" in evidence or "connection" in evidence:
            return "Network error prevented verification"
        return "Verification could not determine vulnerability status"

    def _analyze_unverified(self, claim: VulnClaim, result: VerificationResult) -> str:
        """Analyze why finding was unverified."""
        if not claim.payload or len(claim.payload) < 3:
            return "No payload provided — agent must re-verify with explicit test payload"
        if not claim.evidence_provided or len(claim.evidence_provided) < 40:
            return "Evidence too short or missing — need concrete proof"
        if claim.evidence_provided.lower() in {"n/a", "none", "see above", "same as poc"}:
            return "Evidence was placeholder text, not actual verification output"
        return "No verification strategy could process this claim type"

    # ── corrective action + suggested verification ──────────────────

    def _generate_corrective_action(
        self,
        rejection_type: str,
        claim: VulnClaim,
        result: VerificationResult,
    ) -> str:
        """Generate specific corrective action for the agent."""
        actions = {
            "false_positive": (
                "STOP reporting this type of finding without stronger evidence. "
                "The previous claim was incorrect. Before reporting again: "
                "(1) Run a DIFFERENT verification command, (2) Capture the ACTUAL output, "
                "(3) Confirm the vulnerability is real and not a benign pattern."
            ),
            "inconclusive": (
                "Your verification was insufficient. Improve by: "
                "(1) Using a more specific test command, (2) Checking multiple endpoints, "
                "(3) Trying different payloads or parameters."
            ),
            "unverified": (
                "You MUST provide a real payload and concrete evidence_provided. "
                "'See above' or 'N/A' is NOT acceptable. Run the test again and paste FRESH output."
            ),
            "insufficient_evidence": (
                "The evidence you provided did not support your claim. "
                "Gather more specific proof before reporting."
            ),
        }
        return actions.get(rejection_type, actions["insufficient_evidence"])

    def _generate_suggested_verification(self, rejection_type: str, claim: VulnClaim) -> str:
        """Generate a suggested verification approach based on vuln type."""
        vuln_type = (claim.vuln_type or "").lower()
        if "sql" in vuln_type or "sqli" in vuln_type:
            return (
                "Suggested: Use time-based blind SQLi test. "
                "Run: `curl -s 'URL?id=1 AND (SELECT * FROM (SELECT(SLEEP(5)))a)' -w '\\n%{time_total}'` "
                "Compare response time with baseline."
            )
        if "xss" in vuln_type:
            return (
                "Suggested: Use a unique nonce payload. "
                "Run: `curl -s 'URL?param=<script>alert(\"VAPT-TEST-12345\")</script>'` "
                "Check if the nonce appears unescaped in response."
            )
        if "header" in vuln_type:
            return (
                "Suggested: Explicitly check for the header. "
                "Run: `curl -sI URL | grep -i 'header-name'` "
                "Confirm absence before reporting."
            )
        if "cookie" in vuln_type:
            return (
                "Suggested: Check Set-Cookie headers explicitly. "
                "Run: `curl -sI -c /tmp/cookies.txt URL && cat /tmp/cookies.txt`"
            )
        return (
            "Suggested: Run the original detection command again with a VARIED payload, "
            "then run a SECOND independent command to confirm. Paste BOTH outputs."
        )

    # ── pattern signature ───────────────────────────────────────────

    def _compute_pattern_signature(self, claim: VulnClaim) -> str:
        """Compute a signature for this (vuln_type, endpoint_pattern) combo.

        Used to track repeated rejections of the same pattern — after 3+
        rejections, the pattern is blacklisted.
        """
        endpoint_pattern = self._normalize_endpoint(claim.endpoint)
        key = f"{claim.vuln_type}:{endpoint_pattern}"
        return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]

    @staticmethod
    def _normalize_endpoint(endpoint: str) -> str:
        """Normalize endpoint for pattern matching.

        Removes query params, replaces IDs with {id} placeholder.
        """
        if not endpoint:
            return ""
        if "?" in endpoint:
            endpoint = endpoint.split("?")[0]
        # Replace numeric IDs with {id}
        endpoint = re.sub(r"/\d+", "/{id}", endpoint)
        # Replace query param values with {value}
        endpoint = re.sub(r"=[^&]+", "={value}", endpoint)
        return endpoint.lower()


# ═══════════════════════════════════════════════════════════════════
# Self-Correction Engine
# ═══════════════════════════════════════════════════════════════════

class SelfCorrectionEngine:
    """Main engine that processes verifier rejections and produces:
      1. Corrective system messages for immediate agent feedback
      2. RL reward signals for long-term learning
      3. Pattern blacklist (after 3+ rejections of same pattern)
    """

    def __init__(self) -> None:
        self.analyzer = RejectionAnalyzer()
        # Per-session rejection history (for debugging + escalation)
        self._rejection_history: dict[str, list[RejectionAnalysis]] = {}
        # Pattern rejection counts (in-memory; Phase 2+ can persist to SQL)
        self._pattern_counts: dict[str, int] = {}

    def process_rejection(
        self,
        claim: VulnClaim,
        result: VerificationResult,
        session_id: str,
        session: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Process a verifier rejection and return all correction signals.

        Args:
            claim: the agent's original vulnerability claim
            result: verifier's verification result (rejected)
            session_id: session ID for per-session tracking
            session: optional live session dict (for RL state vector)

        Returns:
            dict with keys:
                - corrective_message: str (to inject into agent prompt)
                - rl_reward: float (negative reward for RL)
                - pattern_blacklisted: bool
                - rejection_analysis: RejectionAnalysis
                - pattern_rejection_count: int
        """
        # 1. Analyze the rejection
        analysis = self.analyzer.analyze(claim, result)

        # 2. Track pattern rejections (escalation)
        pattern_count = self._track_pattern_rejection(analysis.pattern_signature)

        # 3. Generate corrective message
        corrective_message = self._build_corrective_message(analysis, pattern_count)

        # 4. Compute RL reward (negative for rejection)
        rl_reward = self._compute_rejection_reward(analysis, pattern_count)

        # 5. Check if pattern should be blacklisted
        pattern_blacklisted = pattern_count >= 3

        # 6. Store in session history
        if session_id not in self._rejection_history:
            self._rejection_history[session_id] = []
        self._rejection_history[session_id].append(analysis)

        logger.info(
            "Self-correction processed | session=%s | type=%s | pattern=%s | count=%d | reward=%.2f",
            session_id, analysis.rejection_type, analysis.pattern_signature[:8],
            pattern_count, rl_reward,
        )

        return {
            "corrective_message": corrective_message,
            "rl_reward": rl_reward,
            "pattern_blacklisted": pattern_blacklisted,
            "rejection_analysis": analysis,
            "pattern_rejection_count": pattern_count,
        }

    def _build_corrective_message(
        self,
        analysis: RejectionAnalysis,
        pattern_count: int,
    ) -> str:
        """Build a corrective system message for the agent."""
        messages = [
            "CORRECTION: Your finding was rejected by independent verification.",
            "",
            f"Reason: {analysis.root_cause}",
            "",
            f"Corrective Action: {analysis.corrective_action}",
            "",
            f"{analysis.suggested_verification}",
        ]
        if pattern_count >= 2:
            messages.extend([
                "",
                f"WARNING: This is rejection #{pattern_count} for this pattern. "
                f"You are developing a bias toward incorrect findings of this type. "
                f"Be EXTRA skeptical before reporting {analysis.metadata.get('claim_vuln_type', 'this type of')} issues.",
            ])
        if pattern_count >= 3:
            messages.extend([
                "",
                f"PATTERN BLACKLISTED: You have been rejected 3+ times for this pattern. "
                f"Focus on OTHER vulnerability types. This pattern is unreliable for you.",
            ])
        return "\n".join(messages)

    def _compute_rejection_reward(
        self,
        analysis: RejectionAnalysis,
        pattern_count: int,
    ) -> float:
        """Compute RL reward for a rejection (always negative).

        Escalating penalty for repeated rejections of same pattern.
        """
        base_penalty = -analysis.confidence_penalty * 10.0  # Scale to RL reward range
        repeat_penalty = -pattern_count * 2.0
        # Bonus for learning (less negative if agent is early in learning)
        learning_bonus = 0.5 if pattern_count == 1 else 0.0
        reward = base_penalty + repeat_penalty + learning_bonus
        return float(max(-20.0, min(-1.0, reward)))

    def _track_pattern_rejection(self, pattern_signature: str) -> int:
        """Track how many times a pattern has been rejected.

        Returns:
            new rejection count for this pattern (1-indexed).
        """
        current = self._pattern_counts.get(pattern_signature, 0)
        new_count = current + 1
        self._pattern_counts[pattern_signature] = new_count
        return new_count

    def get_stats(self) -> dict[str, Any]:
        """Get engine statistics for debugging / UI."""
        return {
            "total_patterns_tracked": len(self._pattern_counts),
            "total_rejections": sum(self._pattern_counts.values()),
            "blacklisted_patterns": sum(1 for c in self._pattern_counts.values() if c >= 3),
            "sessions_with_rejections": len(self._rejection_history),
            "pattern_counts": dict(self._pattern_counts),
        }


# ── Module-level singleton ───────────────────────────────────────────────

_engine_singleton: SelfCorrectionEngine | None = None


def get_self_correction_engine() -> SelfCorrectionEngine:
    """Get the module-level SelfCorrectionEngine singleton."""
    global _engine_singleton
    if _engine_singleton is None:
        _engine_singleton = SelfCorrectionEngine()
    return _engine_singleton


def reset_self_correction_singleton() -> None:
    """Reset the singleton — for tests only."""
    global _engine_singleton
    _engine_singleton = None


# ── Convenience function ─────────────────────────────────────────────────

def process_verification_rejection(
    finding: dict[str, Any],
    verification_result: VerificationResult,
    session_id: str,
    session: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Convenience function for scan_pipeline integration.

    Called when the verifier rejects a finding. Returns correction signals
    to inject back into the agent loop.

    Args:
        finding: finding dict (agent output format with name, severity,
            location, description, poc, vuln_type, etc.)
        verification_result: VerificationResult from the verifier
        session_id: session ID for per-session tracking
        session: optional live session dict

    Returns:
        dict with corrective_message, rl_reward, pattern_blacklisted,
        rejection_analysis, pattern_rejection_count.
    """
    # Convert finding dict → VulnClaim
    claim = VulnClaim.from_finding_dict(finding)

    engine = get_self_correction_engine()
    return engine.process_rejection(claim, verification_result, session_id, session)


__all__ = [
    # Dataclasses
    "RejectionAnalysis",
    # Classes
    "RejectionAnalyzer",
    "SelfCorrectionEngine",
    # Singleton
    "get_self_correction_engine",
    "reset_self_correction_singleton",
    # Convenience
    "process_verification_rejection",
]