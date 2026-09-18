"""
VAPT-AI Playwright PoC Validator — W12-S5.

Port of EVVO shield_engine/poc_validator_playwright.py (366 LOC).
Playwright-based PoC validator for XSS/SQLi-time/SSRF/IDOR/CSRF/Open-Redirect.

W12 stub: class structure only. Playwright actual integration deferred to W13
(DVWA E2E test). If Playwright is not installed, falls back to regex-based
PoCValidator.

Per master plan §12 W12: "Wire Playwright-based PoC validator as default for
XSS/SQLi-time/SSRF/IDOR/CSRF/Open-Redirect".

Usage:
    from app.harness.poc_validator_playwright import PlaywrightPoCValidator

    validator = PlaywrightPoCValidator()
    if validator.is_available():
        result = validator.validate(claim)
    else:
        # Fallback to regex-based PoCValidator
        from app.harness.poc_validator import PoCValidator
        result = PoCValidator().validate(claim)
"""
from __future__ import annotations

import logging
from typing import Any

from app.harness.types import VerificationResult, VerificationStatus, VulnClaim

logger = logging.getLogger(__name__)


# ---------- Playwright availability check ----------

def is_playwright_available() -> bool:
    """Check if Playwright is installed + browsers are available.

    Returns:
        True if Playwright can be used, False otherwise.
    """
    try:
        import playwright  # type: ignore[import-untyped]
        return True
    except ImportError:
        return False


# ---------- PlaywrightPoCValidator class ----------

class PlaywrightPoCValidator:
    """Playwright-based PoC validator.

    W12 stub: validates claims using headless browser (Playwright).
    Currently a stub that falls back to PoCValidator (regex-based).

    W13+ will implement actual browser-based validation:
        - XSS: load page with payload, check if alert fires
        - SQLi-time: send request with sleep payload, measure response time
        - SSRF: load page, check if internal resource is fetched
        - IDOR: load same URL with different user session, compare responses
        - CSRF: check if token is absent + SameSite cookie is lax/none
        - Open-Redirect: check if Location header redirects to external domain

    Per D17 anti-hallucination: browser session is fresh for each claim
    (no cached state, no shared cookies).
    """

    def __init__(self, headless: bool = True, timeout_ms: int = 10000):
        self.headless = headless
        self.timeout_ms = timeout_ms
        self._available = is_playwright_available()
        if not self._available:
            logger.warning(
                "Playwright not installed — PlaywrightPoCValidator will fall back "
                "to regex-based PoCValidator. Install with: pip install playwright && "
                "playwright install chromium"
            )

    def is_available(self) -> bool:
        """Check if Playwright is available for use."""
        return self._available

    def validate(self, claim: VulnClaim) -> VerificationResult:
        """Validate a PoC claim using Playwright.

        W12 stub: returns INCONCLUSIVE if Playwright not installed.
        W13+ will implement actual browser-based validation.

        Args:
            claim: VulnClaim with endpoint, vuln_type, payload.

        Returns:
            VerificationResult. Currently INCONCLUSIVE (Playwright not wired yet).
        """
        if not self._available:
            return VerificationResult(
                status=VerificationStatus.INCONCLUSIVE,
                confidence=0.0,
                method="playwright_not_available",
                evidence=(
                    "Playwright not installed — install with: "
                    "pip install playwright && playwright install chromium"
                ),
            )

        # W13+ TODO: implement actual browser-based validation
        # For now, return INCONCLUSIVE (stub)
        vuln_type = claim.vuln_type.lower().strip()
        supported = self.list_supported_vuln_types()

        if vuln_type not in supported:
            return VerificationResult(
                status=VerificationStatus.UNVERIFIED,
                confidence=0.0,
                method="playwright_no_strategy",
                evidence=f"No Playwright strategy for vuln_type={vuln_type!r}",
            )

        return VerificationResult(
            status=VerificationStatus.INCONCLUSIVE,
            confidence=0.0,
            method="playwright_stub",
            evidence=(
                "PlaywrightPoCValidator is a stub in W12 — actual browser-based "
                "validation will be implemented in W13 (DVWA E2E test)."
            ),
        )

    def list_supported_vuln_types(self) -> list[str]:
        """List vuln_types that have Playwright validation strategies (W13+)."""
        # W13+ will implement these
        return ["xss", "sqli-time", "ssrf", "idor", "csrf", "open-redirect"]

    async def validate_async(self, claim: VulnClaim) -> VerificationResult:
        """Async variant of validate() for use in async contexts.

        W13+ will use asyncio + Playwright async API.
        """
        # W12 stub: just call sync version
        return self.validate(claim)