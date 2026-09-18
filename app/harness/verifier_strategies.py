"""
VAPT-AI Verifier Strategies — W12-S2.

5 verification strategies ported from EVVO shield_engine/vulnerability_verifier.py
(lines 107-325). Each strategy takes a VulnClaim and returns VerificationResult | None.

Strategies (per master plan §12 W12):
    1. verify_security_header_missing — HSTS, X-Frame-Options, X-Content-Type-Options, CSP
    2. verify_server_disclosure — Server header version leak
    3. verify_cookie_security — HttpOnly, Secure, SameSite attributes
    4. verify_xss_reflected — XSS payload reflection in response
    5. verify_info_disclosure — .git, .env, backup files, error messages

Adaptations for VAPT-AI:
    - Use claim.evidence_provided instead of EVVO's verification_output/detection_output
      (VAPT-AI VulnClaim schema uses evidence_provided)
    - claim.title + claim.vuln_type (not description) for keyword matching
    - All strategies async-compatible (return VerificationResult | None synchronously,
      but VulnerabilityVerifier can wrap with asyncio.to_thread if needed)

W12 stub: strategies analyze agent-supplied evidence (claim.evidence_provided) without
making HTTP requests. W13+ will add live HTTP re-probing via httpx (breaking the
confirmation bias loop fully — see EVVO's verify() method which does live HTTP).

Per D17 anti-hallucination rule: agent-supplied evidence is NOT trusted blindly.
Strategies only verify if the evidence contains concrete indicators (header absence,
payload reflection, version patterns). If evidence is too vague → return None (no
strategy can verify) → finding marked UNVERIFIED.
"""
from __future__ import annotations

import re
from typing import Any

from app.harness.types import VerificationResult, VerificationStatus, VulnClaim


# ---------- Header parsing helpers (port from EVVO lines 77-104) ----------

def _normalize_header_name(name: str) -> str:
    """Normalize header name for comparison."""
    return name.lower().strip().replace("-", "_")


def _extract_headers_from_output(output: str) -> dict[str, str]:
    """Parse headers from curl -I output or HTTP response.

    Looks for lines like "Header-Name: value" and normalizes keys.
    """
    headers: dict[str, str] = {}
    for line in output.split("\n"):
        if ":" in line and not line.startswith("HTTP/"):
            key, val = line.split(":", 1)
            headers[_normalize_header_name(key)] = val.strip()
    return headers


def _header_present(headers: dict[str, str], hyphenated_name: str) -> bool:
    """Check if a header is present using normalized form."""
    return _normalize_header_name(hyphenated_name) in headers


# ---------- Security headers to check ----------

# Map of canonical header name → keywords that hint the claim is about this header
_SECURITY_HEADERS: dict[str, list[str]] = {
    "content-security-policy": ["content-security-policy", "csp"],
    "strict-transport-security": ["strict-transport-security", "hsts"],
    "x-frame-options": ["x-frame-options", "clickjacking"],
    "x-content-type-options": ["x-content-type-options", "nosniff"],
    "referrer-policy": ["referrer-policy"],
    "permissions-policy": ["permissions-policy"],
}


# ---------- Strategy 1: Security header missing ----------

def verify_security_header_missing(claim: VulnClaim) -> VerificationResult | None:
    """Verify claims about missing security headers.

    Two modes:
        1. Specific header named in title (e.g. "Missing HSTS header"):
           Confirm that header is absent from response headers in evidence.
        2. Generic "missing security headers" claim:
           Require at least 2 of 4 core headers (CSP, HSTS, X-Frame, X-Content-Type) missing.

    Returns None if claim is not about security headers.
    """
    target_header: str | None = None
    # VAPT-AI adaptation: use title + vuln_type instead of EVVO's title + description
    title_desc = (claim.title + " " + claim.vuln_type).lower()

    for header_name, keywords in _SECURITY_HEADERS.items():
        if any(kw in title_desc for kw in keywords):
            target_header = header_name
            break

    output = claim.evidence_provided or ""

    # Generic "missing security headers" claim (no specific header named)
    if not target_header:
        generic_keywords = ("security header", "missing header", "harden", "header hardening")
        if not any(kw in title_desc for kw in generic_keywords):
            return None  # Not a security-header claim
        if not output:
            return None
        headers = _extract_headers_from_output(output)
        if not headers:
            return None
        absent = [h for h in _SECURITY_HEADERS if not _header_present(headers, h)]
        # Require at least 2 missing core headers
        core_headers = {
            "content-security-policy", "strict-transport-security",
            "x-frame-options", "x-content-type-options",
        }
        core_missing = [h for h in absent if h in core_headers]
        if len(core_missing) >= 2:
            return VerificationResult(
                status=VerificationStatus.VERIFIED,
                confidence=0.8,
                method="security_header_multi_check",
                evidence=f"Missing security headers: {', '.join(core_missing)}",
                details={"missing": core_missing, "checked_headers": list(headers.keys())},
            )
        if not core_missing:
            return VerificationResult(
                status=VerificationStatus.FALSE_POSITIVE,
                confidence=0.75,
                method="security_header_multi_check",
                evidence="All core security headers (HSTS, CSP, X-Frame, X-Content-Type) present",
            )
        return None  # Some but < 2 missing — inconclusive

    # Specific header check
    if output:
        headers = _extract_headers_from_output(output)
        if not _header_present(headers, target_header):
            return VerificationResult(
                status=VerificationStatus.VERIFIED,
                confidence=0.85,
                method="header_absence_check",
                evidence=f"Header '{target_header}' not found in response headers",
                details={"checked_headers": list(headers.keys()), "target": target_header},
            )
        else:
            value = headers[_normalize_header_name(target_header)]
            return VerificationResult(
                status=VerificationStatus.FALSE_POSITIVE,
                confidence=0.9,
                method="header_presence_check",
                evidence=f"Header '{target_header}' IS present: {value[:100]}",
                details={"header_value": value},
            )

    return None


# ---------- Strategy 2: Server disclosure ----------

def verify_server_disclosure(claim: VulnClaim) -> VerificationResult | None:
    """Verify server version disclosure in headers.

    Triggers if claim title or vuln_type mentions 'server' or 'version'.
    Looks for Server header in evidence with version number.
    """
    title_lower = (claim.title + " " + claim.vuln_type).lower()
    if "server" not in title_lower and "version" not in title_lower:
        return None

    output = claim.evidence_provided or ""
    headers = _extract_headers_from_output(output)
    server_header = headers.get("server", "")

    if server_header and any(c.isdigit() for c in server_header):
        # Has version number
        return VerificationResult(
            status=VerificationStatus.VERIFIED,
            confidence=0.8,
            method="server_header_version_check",
            evidence=f"Server header contains version: {server_header[:100]}",
            details={"server_header": server_header},
        )

    if server_header:
        # Server header present but no clear version
        return VerificationResult(
            status=VerificationStatus.VERIFIED,
            confidence=0.6,
            method="server_header_present",
            evidence=f"Server header present: {server_header[:100]}",
            details={"server_header": server_header},
        )

    return None


# ---------- Strategy 3: Cookie security ----------

def verify_cookie_security(claim: VulnClaim) -> VerificationResult | None:
    """Verify cookie security issues (missing HttpOnly, Secure, SameSite).

    Triggers if claim title or vuln_type mentions 'cookie'.
    Inspects Set-Cookie headers in evidence.
    """
    title_lower = (claim.title + " " + claim.vuln_type).lower()
    if "cookie" not in title_lower:
        return None

    output = claim.evidence_provided or ""

    # Find Set-Cookie lines
    cookie_lines: list[str] = []
    for line in output.split("\n"):
        if "set-cookie" in line.lower():
            cookie_lines.append(line)

    if not cookie_lines:
        return VerificationResult(
            status=VerificationStatus.INCONCLUSIVE,
            confidence=0.5,
            method="cookie_check",
            evidence="No Set-Cookie headers found in evidence",
        )

    issues_found: list[str] = []
    for line in cookie_lines:
        line_lower = line.lower()
        if "httponly" not in line_lower:
            issues_found.append("Missing HttpOnly")
        if "secure" not in line_lower:
            issues_found.append("Missing Secure")
        if "samesite" not in line_lower:
            issues_found.append("Missing SameSite")

    if issues_found:
        return VerificationResult(
            status=VerificationStatus.VERIFIED,
            confidence=0.75,
            method="cookie_security_check",
            evidence=(
                f"Cookie issues found: {', '.join(issues_found)} in "
                f"{len(cookie_lines)} Set-Cookie header(s)"
            ),
            details={
                "cookies_checked": len(cookie_lines),
                "issues": issues_found,
            },
        )

    # All cookies have all 3 attributes — claim is FP
    return VerificationResult(
        status=VerificationStatus.FALSE_POSITIVE,
        confidence=0.7,
        method="cookie_security_check",
        evidence="All cookies have HttpOnly, Secure, and SameSite attributes",
    )


# ---------- Strategy 4: XSS reflected ----------

# Common XSS payload patterns to look for in reflected output
_XSS_PATTERNS = [
    re.compile(r"<script[^>]*>[\s\S]*?</script>", re.IGNORECASE),
    re.compile(r"javascript:", re.IGNORECASE),
    re.compile(r"on\w+\s*=", re.IGNORECASE),  # onload=, onerror=, etc.
    re.compile(r"<iframe", re.IGNORECASE),
    re.compile(r"<object", re.IGNORECASE),
    re.compile(r"<embed", re.IGNORECASE),
    re.compile(r"<img[^>]+onerror", re.IGNORECASE),
    re.compile(r"alert\s*\(", re.IGNORECASE),
    re.compile(r"prompt\s*\(", re.IGNORECASE),
    re.compile(r"document\.cookie", re.IGNORECASE),
]


def verify_xss_reflected(claim: VulnClaim) -> VerificationResult | None:
    """Basic verification for reflected XSS claims.

    Looks for common XSS payload patterns in the evidence_provided text.
    W13+ will add live HTTP re-probing with the claim.payload.
    """
    title_lower = (claim.title + " " + claim.vuln_type).lower()
    if "xss" not in title_lower and "cross-site scripting" not in title_lower:
        return None

    output = claim.evidence_provided or ""
    if not output:
        return None

    matched_patterns: list[str] = []
    for pattern in _XSS_PATTERNS:
        if pattern.search(output):
            matched_patterns.append(pattern.pattern[:50])

    if matched_patterns:
        return VerificationResult(
            status=VerificationStatus.VERIFIED,
            confidence=0.7,
            method="xss_pattern_match",
            evidence=(
                f"Potential XSS reflection detected: "
                f"{len(matched_patterns)} pattern(s) matched"
            ),
            details={"matched_patterns": matched_patterns[:5]},  # cap at 5 for brevity
        )

    return None


# ---------- Strategy 5: Info disclosure ----------

# Indicators for different info disclosure types
_INFO_DISCLOSURE_INDICATORS: dict[str, list[str]] = {
    ".git": ["ref:", "head"],
    ".env": ["=", "db_", "api_key", "secret", "password"],
    "backup": [".sql", ".zip", ".tar", ".gz", ".bak"],
    "error": ["stack trace", "exception", "error", "traceback", "warning"],
    "debug": ["debug", "var_dump", "print_r", "console.log"],
    "config": ["config", "configuration", "settings"],
}


def verify_info_disclosure(claim: VulnClaim) -> VerificationResult | None:
    """Verify information disclosure claims (.git, .env, error messages, etc.).

    Checks claim title for indicator type, then searches evidence for
    corresponding keywords.
    """
    title_lower = (claim.title + " " + claim.vuln_type).lower()
    output = claim.evidence_provided or ""

    for indicator_type, keywords in _INFO_DISCLOSURE_INDICATORS.items():
        if indicator_type in title_lower:
            output_lower = output.lower()
            matches = [kw for kw in keywords if kw in output_lower]
            if matches:
                return VerificationResult(
                    status=VerificationStatus.VERIFIED,
                    confidence=0.75,
                    method=f"info_disclosure_{indicator_type}",
                    evidence=f"Found indicators: {', '.join(matches)}",
                    details={
                        "indicator_type": indicator_type,
                        "matched_keywords": matches,
                    },
                )

    # Also check vuln_type for info_disclosure
    if "info" in claim.vuln_type.lower() and "disclos" in claim.vuln_type.lower():
        # Generic info disclosure — check all indicator types
        output_lower = output.lower()
        for indicator_type, keywords in _INFO_DISCLOSURE_INDICATORS.items():
            matches = [kw for kw in keywords if kw in output_lower]
            if matches:
                return VerificationResult(
                    status=VerificationStatus.VERIFIED,
                    confidence=0.65,
                    method=f"info_disclosure_generic_{indicator_type}",
                    evidence=f"Found indicators ({indicator_type}): {', '.join(matches)}",
                    details={
                        "indicator_type": indicator_type,
                        "matched_keywords": matches,
                    },
                )

    return None


# ---------- Strategy registry ----------

# Master plan §12 W12 specifies 5 strategies (from EVVO's 10).
# W12 ships these 5; additional strategies (sqli, wp_user_enum, csp_weak_directives,
# stack_trace_disclosure, version_disclosure) deferred to W13+ when live HTTP
# re-probing is added (they require actual HTTP requests to verify).
DEFAULT_STRATEGIES = [
    verify_security_header_missing,
    verify_server_disclosure,
    verify_cookie_security,
    verify_xss_reflected,
    verify_info_disclosure,
]