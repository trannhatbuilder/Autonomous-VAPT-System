"""
VAPT-AI PoC Validator (regex-based fallback) — W12-S4.

Port of EVVO shield_engine/poc_validator.py (194 LOC).
Regex-based PoC validator — fallback when Playwright is not available.

W12 stub: validates PoC output against expected patterns.
W13+ will use PlaywrightPoCValidator as default for XSS/SQLi/SSRF/IDOR/CSRF.

Usage:
    from app.harness.poc_validator import PoCValidator
    from app.harness.types import VulnClaim

    validator = PoCValidator()
    result = validator.validate(claim)
    if result.is_acceptable():
        # PoC confirmed
"""
from __future__ import annotations

import logging
import re
from typing import Any

from app.harness.types import VerificationResult, VerificationStatus, VulnClaim

logger = logging.getLogger(__name__)


# ---------- PoC patterns per vuln type ----------

# Each vuln_type has a list of regex patterns that, if matched in the PoC output,
# indicate the vulnerability is confirmed.
POC_PATTERNS: dict[str, list[re.Pattern[str]]] = {
    "sqli": [
        re.compile(r"sql syntax.*mysql", re.IGNORECASE),
        re.compile(r"warning.*\bsql\b", re.IGNORECASE),
        re.compile(r"ORA-\d{5}", re.IGNORECASE),  # Oracle
        re.compile(r"PostgreSQL.*ERROR", re.IGNORECASE),
        re.compile(r"Microsoft SQL Server.*error", re.IGNORECASE),
        re.compile(r"SQLite3?::query", re.IGNORECASE),
        re.compile(r"sqlite_query", re.IGNORECASE),
        re.compile(r"unterminated quoted string", re.IGNORECASE),
        re.compile(r"' or '1'='1", re.IGNORECASE),  # Classic SQLi payload
        re.compile(r"\bunion\s+select\b", re.IGNORECASE),
    ],
    "xss": [
        re.compile(r"<script[^>]*>[\s\S]*?</script>", re.IGNORECASE),
        re.compile(r"javascript:", re.IGNORECASE),
        re.compile(r"on\w+\s*=", re.IGNORECASE),
        re.compile(r"<iframe", re.IGNORECASE),
        re.compile(r"alert\s*\(", re.IGNORECASE),
        re.compile(r"prompt\s*\(", re.IGNORECASE),
        re.compile(r"document\.cookie", re.IGNORECASE),
        re.compile(r"document\.location", re.IGNORECASE),
    ],
    "rce": [
        re.compile(r"uid=\d+\([\w-]+\)", re.IGNORECASE),  # Linux id output
        re.compile(r"whoami", re.IGNORECASE),
        re.compile(r"\broot\b.*\bx\b.*\t", re.IGNORECASE),  # /etc/passwd format
        re.compile(r"Windows IP Configuration", re.IGNORECASE),  # ipconfig
        re.compile(r"Volume in drive", re.IGNORECASE),  # dir
    ],
    "lfi": [
        re.compile(r"root:x:0:0", re.IGNORECASE),  # /etc/passwd
        re.compile(r"\[boot loader\]", re.IGNORECASE),  # boot.ini
        re.compile(r"win\.ini", re.IGNORECASE),
        re.compile(r"\[fonts\]", re.IGNORECASE),
    ],
    "ssrf": [
        re.compile(r"169\.254\.169\.254", re.IGNORECASE),  # AWS metadata
        re.compile(r"meta-data", re.IGNORECASE),
        re.compile(r"instance-id", re.IGNORECASE),
        re.compile(r"ami-id", re.IGNORECASE),
        re.compile(r"localhost", re.IGNORECASE),
        re.compile(r"127\.0\.0\.1", re.IGNORECASE),
    ],
    "idor": [
        # IDOR: response differs for different user IDs (hard to regex)
        # Default to None — needs HTTP comparison logic
    ],
    "csrf": [
        # CSRF: typically verified by checking token absence
        # Hard to regex — needs HTTP session comparison
    ],
    "open_redirect": [
        re.compile(r"location:\s*https?://", re.IGNORECASE),  # HTTP redirect
        re.compile(r"window\.location\s*=", re.IGNORECASE),
        re.compile(r"document\.location\s*=", re.IGNORECASE),
    ],
}


# ---------- PoCValidator class ----------

class PoCValidator:
    """Regex-based PoC validator.

    Validates PoC output against expected patterns per vuln_type.
    Used as fallback when Playwright is not available.

    Per D17 anti-hallucination: PoC output is NOT trusted blindly.
    Patterns must match concrete indicators (error messages, payloads reflected, etc.).
    """

    def __init__(self, patterns: dict[str, list[re.Pattern[str]]] | None = None):
        self.patterns = patterns or POC_PATTERNS

    def validate(self, claim: VulnClaim) -> VerificationResult:
        """Validate a PoC claim against expected patterns.

        Args:
            claim: VulnClaim with vuln_type + evidence_provided (PoC output).

        Returns:
            VerificationResult. If vuln_type has no patterns or none match,
            returns UNVERIFIED (cannot determine).
        """
        vuln_type = claim.vuln_type.lower().strip()
        output = claim.evidence_provided or ""

        if not output:
            return VerificationResult(
                status=VerificationStatus.UNVERIFIED,
                confidence=0.0,
                method="poc_validator_no_evidence",
                evidence="No PoC output provided for validation",
            )

        # Find matching patterns for this vuln_type
        patterns = self.patterns.get(vuln_type, [])
        if not patterns:
            # Try fuzzy match (e.g. "sql_injection" → "sqli")
            for key in self.patterns:
                if key in vuln_type or vuln_type in key:
                    patterns = self.patterns[key]
                    break

        if not patterns:
            return VerificationResult(
                status=VerificationStatus.UNVERIFIED,
                confidence=0.0,
                method="poc_validator_no_patterns",
                evidence=f"No PoC patterns registered for vuln_type={vuln_type!r}",
            )

        # Check each pattern
        matched: list[str] = []
        for pattern in patterns:
            if pattern.search(output):
                matched.append(pattern.pattern[:80])

        if matched:
            return VerificationResult(
                status=VerificationStatus.VERIFIED,
                confidence=0.75,
                method="poc_pattern_match",
                evidence=(
                    f"PoC confirmed: {len(matched)} pattern(s) matched for "
                    f"vuln_type={vuln_type!r}"
                ),
                details={
                    "vuln_type": vuln_type,
                    "matched_patterns": matched[:5],
                    "output_length": len(output),
                },
            )

        # No patterns matched — could be FP or just insufficient patterns
        return VerificationResult(
            status=VerificationStatus.INCONCLUSIVE,
            confidence=0.3,
            method="poc_validator_no_match",
            evidence=(
                f"PoC output did not match any expected patterns for "
                f"vuln_type={vuln_type!r} (output length={len(output)})"
            ),
            details={
                "vuln_type": vuln_type,
                "patterns_checked": len(patterns),
                "output_length": len(output),
            },
        )

    def list_supported_vuln_types(self) -> list[str]:
        """List vuln_types that have registered PoC patterns."""
        return sorted(self.patterns.keys())