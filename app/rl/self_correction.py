"""
VAPT-AI RL — Self-Correction on Verifier Rejection (Phase 5).

When record_vulnerability returns status="rejected", the agent made a
mistake — it claimed a vulnerability that doesn't actually exist (or
doesn't have enough evidence). This module:

    1. Computes a pattern signature from (vuln_type, normalized_endpoint)
       so repeated rejections of the same pattern can be tracked.
    2. Maintains in-memory pattern rejection counts (reset on restart).
    3. Renders a 3-tier corrective message for the LLM:
         Tier 1 (count=1): corrective action + suggested verification
         Tier 2 (count=2): + warning about developing bias
         Tier 3 (count>=3): + blacklist warning (advisory, not hard block)
    4. Computes an escalating RL reward:
         count=1  -> -5.0
         count=2  -> -7.0
         count>=3 -> -10.0
       The harness pushes this into the PER store with priority = |reward| * 2
       so the policy learns to avoid the rejected pattern.

Pattern signature
-----------------
    signature = sha256(f"{vuln_type}:{normalized_endpoint}").hexdigest()[:16]

    Endpoint normalization:
        - strip query string
        - replace /123 with /{id}
        - replace =value with ={value}
        - lowercase

    This groups "sqli on /users/1", "sqli on /users/2", "sqli on /users/999"
    into one pattern, so the agent gets penalised for the BEHAVIOR pattern,
    not the exact URL.

Integration
-----------
    Called by HarnessBridge.process_rejection() when record_vulnerability
    returns rejected. The harness:
        1. Calls process_verification_rejection(finding, reason)
        2. Reads result["corrective_message"] + result["rl_reward"]
        3. Stashes corrective_message for next-turn LLM injection
        4. Pushes (state, action, rl_reward, next_state, done) into PER
           with priority = |rl_reward| * 2

All enforcement is advisory — the LLM is free to ignore the blacklist
warning. Hard block is intentionally deferred (risk of false-positive
pattern matching blocking legitimate findings).
"""
from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# ── defaults ──────────────────────────────────────────────────────────
DEFAULT_BLACKLIST_THRESHOLD: int = 3      # pattern_count >= 3 → blacklist warning
DEFAULT_WARNING_THRESHOLD: int = 2        # pattern_count >= 2 → bias warning

# Escalating RL reward schedule (always negative — rejections are bad).
REWARD_TIER_1: float = -5.0    # 1st rejection
REWARD_TIER_2: float = -7.0    # 2nd rejection
REWARD_TIER_3: float = -10.0   # 3rd+ rejection (blacklisted)

# PER priority multiplier — higher priority = sampled more often during
# training, so the policy learns from rejections faster than from
# regular transitions.
PER_PRIORITY_MULTIPLIER: float = 2.0


# ── pattern signature ─────────────────────────────────────────────────

# Pre-compile regexes for endpoint normalization (called on every rejection).
_ID_PATTERN = re.compile(r"/\d+")
_VALUE_PATTERN = re.compile(r"=[^&]+")


def normalize_endpoint(endpoint: str) -> str:
    """Normalize an endpoint for pattern matching.

    Strips query string, replaces numeric IDs with /{id}, replaces
    parameter values with ={value}, lowercases. This groups similar
    endpoints into one pattern:
        /users/1?q=x  →  /users/{id}?q={value}
        /users/2?q=y  →  /users/{id}?q={value}
    """
    if not endpoint:
        return ""
    # Strip query string first (keep path only for signature).
    if "?" in endpoint:
        endpoint = endpoint.split("?", 1)[0]
    # Replace /123 with /{id}
    endpoint = _ID_PATTERN.sub("/{id}", endpoint)
    # Replace =value with ={value} (handles remaining query-ish fragments)
    endpoint = _VALUE_PATTERN.sub("={value}", endpoint)
    return endpoint.lower().strip()


def compute_pattern_signature(vuln_type: str, endpoint: str) -> str:
    """Compute a stable signature for (vuln_type, normalized_endpoint).

    Returns a 16-char hex string. Two findings with the same vuln_type
    and similar endpoints get the same signature, so repeated rejections
    of the same pattern accumulate.
    """
    normalized = normalize_endpoint(endpoint)
    key = f"{(vuln_type or '').lower()}:{normalized}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


# ── rejection analysis ────────────────────────────────────────────────


@dataclass
class RejectionAnalysis:
    """Result of analyzing one rejected finding.

    The root_cause / corrective_action / suggested_verification strings
    are heuristic — derived from the rejection reason + finding fields.
    They are NOT LLM-generated (that would be slow + costly). The goal
    is to give the agent a concrete nudge in the right direction.
    """

    pattern_signature: str
    pattern_count: int                # how many times this pattern has been rejected
    vuln_type: str
    endpoint: str
    root_cause: str                   # human-readable reason for rejection
    corrective_action: str            # what the agent should do differently
    suggested_verification: str       # how to verify before re-reporting
    rl_reward: float                  # escalating negative reward
    tier: int                         # 1, 2, or 3 (blacklist)


def _derive_root_cause(reason: str, finding: dict[str, Any]) -> str:
    """Heuristically derive a root-cause string from the rejection reason."""
    reason_lower = (reason or "").lower()
    if "evidence" in reason_lower or "poc" in reason_lower:
        return "Missing or insufficient evidence — the PoC did not prove the vulnerability."
    if "false positive" in reason_lower or "fp" in reason_lower:
        return "False positive — the scanner output was misinterpreted as a vulnerability."
    if "scope" in reason_lower:
        return "Scope violation — the tested endpoint is out of authorized scope."
    if "duplicate" in reason_lower:
        return "Duplicate finding — this vulnerability was already reported."
    if "severity" in reason_lower:
        return "Severity mismatch — the claimed severity does not match the evidence."
    return reason or "Unknown — the verifier rejected this finding without a specific reason."


def _derive_corrective_action(analysis_root: str, vuln_type: str) -> str:
    """Derive a corrective action string from the root cause."""
    if "evidence" in analysis_root.lower():
        return (
            f"Re-run the detection tool, capture its FULL output as evidence, "
            f"and only call record_vulnerability when you have concrete proof "
            f"(command + server response) for the {vuln_type or 'vulnerability'}."
        )
    if "false positive" in analysis_root.lower():
        return (
            f"Verify the finding manually before reporting. Scanner warnings "
            f"are NOT proof — you need to reproduce the {vuln_type or 'issue'} "
            f"with an active test (e.g. sqlmap for SQLi, dalfox for XSS)."
        )
    if "scope" in analysis_root.lower():
        return "Check the scope before testing. Only test in-scope endpoints."
    if "duplicate" in analysis_root.lower():
        return "Check the findings list before reporting — this was already recorded."
    return "Gather stronger evidence and verify the finding before re-reporting."


def _derive_suggested_verification(vuln_type: str) -> str:
    """Suggest a verification command based on vuln type."""
    vt = (vuln_type or "").lower()
    suggestions = {
        "sqli": "Run: sqlmap -u <url> --batch --dbs  (confirm with database enumeration output)",
        "xss": "Run: dalfox url <url>  (confirm with reflected payload in response)",
        "lfi": "Run: curl <url>?file=../../../../etc/passwd  (confirm with passwd contents)",
        "rce": "Run: curl <url>;cmd=id  (confirm with command output in response)",
        "information-disclosure": "Run: httpx -u <url> -status-code -title -tech-detect  (confirm with exposed data)",
        "security_misconfiguration": "Run: nikto -h <url>  (confirm with misconfig finding)",
    }
    return suggestions.get(vt, "Re-run the detection tool and capture concrete proof before re-reporting.")


# ── self-correction engine ────────────────────────────────────────────


class SelfCorrectionEngine:
    """Tracks rejection patterns + renders corrective feedback.

    One instance per scan (held by HarnessBridge). Pattern counts are
    in-memory only — they reset when the process restarts. This is
    intentional for Phase 5: DB persistence is deferred to avoid
    coupling self-correction to a new migration.
    """

    def __init__(
        self,
        *,
        blacklist_threshold: int = DEFAULT_BLACKLIST_THRESHOLD,
        warning_threshold: int = DEFAULT_WARNING_THRESHOLD,
    ) -> None:
        self.blacklist_threshold = blacklist_threshold
        self.warning_threshold = warning_threshold
        # pattern_signature → rejection count
        self._pattern_counts: dict[str, int] = {}
        # Last corrective message — stashed by the harness for next-turn injection.
        self._last_corrective_message: str | None = None
        self._total_rejections: int = 0
        self._unique_patterns: int = 0

    def process_rejection(
        self,
        finding: dict[str, Any],
        reason: str,
        *,
        session_id: str = "",
    ) -> dict[str, Any]:
        """Process one rejected finding.

        Args:
            finding: the finding dict the agent tried to record (with
                title, vuln_type, severity, location/endpoint, etc.).
            reason: the rejection reason from record_vulnerability
                (e.g. "missing evidence", "false positive").
            session_id: scan ID for logging.

        Returns:
            dict with:
                - pattern_signature: 16-char hex
                - pattern_count: updated count for this pattern
                - root_cause: human-readable reason
                - corrective_action: what to do differently
                - suggested_verification: how to verify
                - corrective_message: full text for LLM injection
                - rl_reward: escalating negative reward
                - tier: 1, 2, or 3
                - blacklisted: True if pattern_count >= blacklist_threshold
        """
        vuln_type = str(finding.get("vuln_type") or finding.get("vuln") or "")
        endpoint = str(
            finding.get("location")
            or finding.get("endpoint")
            or finding.get("target")
            or ""
        )

        signature = compute_pattern_signature(vuln_type, endpoint)
        self._pattern_counts[signature] = self._pattern_counts.get(signature, 0) + 1
        pattern_count = self._pattern_counts[signature]
        self._total_rejections += 1
        self._unique_patterns = len(self._pattern_counts)

        # Derive analysis fields.
        root_cause = _derive_root_cause(reason, finding)
        corrective_action = _derive_corrective_action(root_cause, vuln_type)
        suggested_verification = _derive_suggested_verification(vuln_type)

        # Determine tier + reward.
        if pattern_count >= self.blacklist_threshold:
            tier = 3
            rl_reward = REWARD_TIER_3
        elif pattern_count >= self.warning_threshold:
            tier = 2
            rl_reward = REWARD_TIER_2
        else:
            tier = 1
            rl_reward = REWARD_TIER_1

        # Build the corrective message.
        corrective_message = self._build_corrective_message(
            root_cause=root_cause,
            corrective_action=corrective_action,
            suggested_verification=suggested_verification,
            pattern_count=pattern_count,
            vuln_type=vuln_type,
            tier=tier,
        )
        self._last_corrective_message = corrective_message

        logger.info(
            "sc_rejection | scan=%s | sig=%s | vuln=%s | endpoint=%s | "
            "count=%d | tier=%d | reward=%.1f | blacklisted=%s",
            session_id, signature, vuln_type, normalize_endpoint(endpoint),
            pattern_count, tier, rl_reward,
            pattern_count >= self.blacklist_threshold,
        )

        return {
            "pattern_signature": signature,
            "pattern_count": pattern_count,
            "vuln_type": vuln_type,
            "endpoint": normalize_endpoint(endpoint),
            "root_cause": root_cause,
            "corrective_action": corrective_action,
            "suggested_verification": suggested_verification,
            "corrective_message": corrective_message,
            "rl_reward": rl_reward,
            "tier": tier,
            "blacklisted": pattern_count >= self.blacklist_threshold,
            "per_priority": abs(rl_reward) * PER_PRIORITY_MULTIPLIER,
        }

    @staticmethod
    def _build_corrective_message(
        *,
        root_cause: str,
        corrective_action: str,
        suggested_verification: str,
        pattern_count: int,
        vuln_type: str,
        tier: int,
    ) -> str:
        """Build the 3-tier corrective system message for the LLM.

        Tier 1: corrective action + suggested verification
        Tier 2: + bias warning
        Tier 3: + blacklist warning
        """
        lines: list[str] = [
            "## Self-Correction Feedback (advisory — your judgment wins)",
            "",
            "Your last record_vulnerability call was REJECTED by the verifier.",
            "",
            f"**Reason:** {root_cause}",
            "",
            f"**Corrective action:** {corrective_action}",
            "",
            f"**Suggested verification:** {suggested_verification}",
        ]

        if tier >= 2:
            lines.extend([
                "",
                f"⚠️ WARNING: This is rejection #{pattern_count} for this pattern "
                f"({vuln_type or 'unknown vuln type'} on this endpoint class). "
                f"You are developing a bias toward incorrect findings of this type. "
                f"Be EXTRA skeptical before reporting this kind of issue again.",
            ])

        if tier >= 3:
            lines.extend([
                "",
                f"🚫 PATTERN BLACKLISTED: You have been rejected {pattern_count}+ times "
                f"for this pattern. Focus on OTHER vulnerability types. This pattern "
                f"is unreliable for you — continuing to report it will waste turns.",
            ])

        lines.extend([
            "",
            "Advisory — your judgment wins, but repeated rejections of the same "
            "pattern indicate a systematic issue with your evidence-gathering for "
            "this vuln type.",
        ])
        return "\n".join(lines)

    def get_pending_corrective_message(self) -> str | None:
        """Return the last corrective message, then clear it.

        The harness calls this at the start of the next turn to inject
        the message into the LLM prompt. Once consumed, the message is
        cleared so it's not re-injected on subsequent turns.
        """
        msg = self._last_corrective_message
        self._last_corrective_message = None
        return msg

    def get_stats(self) -> dict[str, Any]:
        """Snapshot for the rl_stats.sc sub-dict."""
        return {
            "enabled": True,
            "total_rejections": self._total_rejections,
            "unique_patterns": self._unique_patterns,
            "blacklisted_patterns": sum(
                1 for c in self._pattern_counts.values()
                if c >= self.blacklist_threshold
            ),
            "pattern_counts": dict(list(self._pattern_counts.items())[:20]),
            "pending_message": self._last_corrective_message is not None,
        }


# ── module-level singleton (one engine per process) ──────────────────

_engine: SelfCorrectionEngine | None = None


def get_self_correction_engine() -> SelfCorrectionEngine:
    """Get the module-level SelfCorrectionEngine singleton.

    The harness creates its own per-scan instance in __init__ — this
    singleton is only for the convenience function below.
    """
    global _engine
    if _engine is None:
        _engine = SelfCorrectionEngine()
    return _engine


def process_verification_rejection(
    finding: dict[str, Any],
    reason: str,
    *,
    session_id: str = "",
    engine: SelfCorrectionEngine | None = None,
) -> dict[str, Any]:
    """Convenience function — delegates to the engine singleton.

    Args:
        finding: the rejected finding dict.
        reason: rejection reason from record_vulnerability.
        session_id: scan ID for logging.
        engine: optional engine instance (defaults to singleton).
    """
    e = engine or get_self_correction_engine()
    return e.process_rejection(finding, reason, session_id=session_id)


__all__ = [
    # Constants
    "DEFAULT_BLACKLIST_THRESHOLD", "DEFAULT_WARNING_THRESHOLD",
    "REWARD_TIER_1", "REWARD_TIER_2", "REWARD_TIER_3",
    "PER_PRIORITY_MULTIPLIER",
    # Functions
    "normalize_endpoint", "compute_pattern_signature",
    "process_verification_rejection", "get_self_correction_engine",
    # Types
    "RejectionAnalysis",
    # Class
    "SelfCorrectionEngine",
]
