"""
VAPT-AI Verifier Strategies — EVVO Port (W19-FIX5 / Phase J).

Five verification strategies ported from EVVO
shield_engine/vulnerability_verifier.py (902 LOC), adapted to VAPT-AI's async
harness. These strategies analyse the agent-supplied `evidence_provided`
text on a VulnClaim and return a VerificationResult | None.

Strategies (per EVVO-ANALYSIS report, priority order from
EVVO `DEFAULT_STRATEGIES`):

    1. verify_sqli (EVVO line 559-664) — 5 channels:
         - sqlmap proof regex `_SQLMAP_PROOF_PATTERNS`
           ("is vulnerable", "back-end DBMS", "Payload:", etc.)
         - DBMS error patterns (MySQL, PostgreSQL, SQLite, MSSQL, Oracle)
         - Time-based detection (SLEEP / BENCHMARK / pg_sleep / WAITFOR DELAY)
         - UNION banner detection (version banners, information_schema)
         - Boolean diff detection (TRUE/FALSE response diff)
       Triggers: `_SQLI_TITLE_TRIGGERS` (regex match on title+vuln_type).

    2. verify_security_header_missing (EVVO line 107-189) — detect:
         - HSTS, X-Frame-Options, X-Content-Type-Options, CSP missing
         - Requires ≥ 2 core headers absent (avoid FP on single missing)
         - Generic "missing security headers" claim scans all 6 known
           headers and reports which of the 4 core ones are absent.

    3. verify_server_disclosure (EVVO line 192-218) — detect:
         - Server header contains version digits (e.g. "nginx/1.31.6",
           "Apache/2.4.41") → confidence 0.8
         - X-Powered-By header leaks PHP/Python version (e.g. "PHP/8.1.2",
           "Python/3.10.4") → confidence 0.75
         - Server header present but no version → confidence 0.6

    4. verify_cookie_security (EVVO line 220-260) — check Set-Cookie flags:
         - HttpOnly missing
         - Secure missing (on HTTPS)
         - SameSite missing / None
       Returns confidence 0.75 if any flag is missing; 0.7 if all present.

    5. verify_xss_reflected (EVVO line 269-295) — regex patterns:
         - `<script>` tag reflection
         - `javascript:` URI
         - `on\\w+=` event handler (onload=, onerror=, etc.)
         - `<iframe>`, `<object>`, `<embed>`
       Returns confidence 0.7 if any pattern matches.

Adaptations for VAPT-AI:
    - Uses VAPT-AI's `VerificationResult` / `VerificationStatus` / `VulnClaim`
      (see `app/harness/types.py`).
    - VAPT-AI's `VulnClaim.evidence_provided` carries the verification_output
      text (see `VulnClaim.from_finding_dict`).
    - All strategies are sync functions returning `VerificationResult | None`
      (matching the existing `app/harness/verifier_strategies.py` pattern).
      `VulnerabilityVerifier.verify()` is async-safe via the existing
      thread-pool wrapper in `app/harness/verifier.py`.
    - Patterns are compiled at module level for performance.

Anti-hallucination rule (D17): agent-supplied evidence is NOT trusted blindly.
Strategies only verify if the evidence contains concrete indicators (header
absence, payload reflection, version patterns). If evidence is too vague →
return None (no strategy can verify) → finding marked UNVERIFIED.
"""
from __future__ import annotations

import re
from typing import Any

from app.harness.types import VerificationResult, VerificationStatus, VulnClaim


# ──────────────────────────────────────────────────────────────────────
# Header parsing helpers (port from EVVO lines 77-104)
# ──────────────────────────────────────────────────────────────────────

def _normalize_header_name(name: str) -> str:
    """Normalize header name for comparison (lowercase + underscores)."""
    return name.lower().strip().replace("-", "_")


def _extract_headers_from_output(output: str) -> dict[str, str]:
    """Parse headers from curl -I / curl -sI / HTTP response.

    Looks for lines like "Header-Name: value" and normalizes keys to
    lowercase with underscores. Skips the HTTP status line and body.
    """
    headers: dict[str, str] = {}
    if not output:
        return headers
    for line in output.split("\n"):
        # Skip the "HTTP/1.1 200 OK" status line and blank lines
        if not line or line.startswith("HTTP/"):
            continue
        # Header lines must contain a colon
        if ":" in line:
            key, val = line.split(":", 1)
            headers[_normalize_header_name(key)] = val.strip()
    return headers


def _header_present(headers: dict[str, str], hyphenated_name: str) -> bool:
    """Check if a header is present using normalized form."""
    return _normalize_header_name(hyphenated_name) in headers


# ──────────────────────────────────────────────────────────────────────
# Set-Cookie parser — handles multiple Set-Cookie lines + cookie attributes
# ──────────────────────────────────────────────────────────────────────

def _extract_set_cookie_lines(output: str) -> list[str]:
    """Find all Set-Cookie header lines in curl output.

    Multiple Set-Cookie headers may appear in a single response — each
    needs to be evaluated separately for HttpOnly/Secure/SameSite flags.
    """
    if not output:
        return []
    return [
        line for line in output.split("\n")
        if "set-cookie" in line.lower() and ":" in line
    ]


# ──────────────────────────────────────────────────────────────────────
# Security headers catalogue (port from EVVO line 92-99)
# ──────────────────────────────────────────────────────────────────────

# Canonical header name → keywords that hint the claim is about this header
_SECURITY_HEADERS: dict[str, list[str]] = {
    "content-security-policy": ["content-security-policy", "csp"],
    "strict-transport-security": ["strict-transport-security", "hsts"],
    "x-frame-options": ["x-frame-options", "clickjacking"],
    "x-content-type-options": ["x-content-type-options", "nosniff"],
    "referrer-policy": ["referrer-policy"],
    "permissions-policy": ["permissions-policy"],
}

# Core headers that must be present for "header hardening" claims
_CORE_SECURITY_HEADERS: set[str] = {
    "content-security-policy",
    "strict-transport-security",
    "x-frame-options",
    "x-content-type-options",
}


# ──────────────────────────────────────────────────────────────────────
# Strategy 1: SQL Injection (EVVO line 559-664)
# ──────────────────────────────────────────────────────────────────────

# Title / vuln-type keywords that trigger SQLi verification
_SQLI_TITLE_TRIGGERS: tuple[str, ...] = (
    "sql injection", "sqli", "sql-injection", "blind sql",
    "time-based sql", "error-based sql", "boolean-based sql", "union sql",
)

# sqlmap output patterns that PROVE the endpoint is injectable
_SQLMAP_PROOF_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"is vulnerable\b", re.IGNORECASE),
    re.compile(r"\binjectable\b", re.IGNORECASE),
    re.compile(r"\bback-end DBMS\b", re.IGNORECASE),
    re.compile(r"sqlmap identified the following injection point", re.IGNORECASE),
    re.compile(r"\bPayload:\s", re.IGNORECASE),
    re.compile(
        r"\bType:\s+(?:boolean-blind|time-blind|error-based|union-query|stacked)",
        re.IGNORECASE,
    ),
    re.compile(r"\bavailable databases\b", re.IGNORECASE),
    re.compile(r"\bcurrent database\b.*\bcurrent user\b", re.IGNORECASE | re.DOTALL),
]

# DBMS error strings — strong error-based SQLi evidence
_DBMS_ERROR_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # MySQL
    (re.compile(r"SQL syntax.*?MySQL", re.IGNORECASE), "mysql_syntax"),
    (re.compile(r"Warning.*?\bmysqli?_", re.IGNORECASE), "mysql_warning"),
    (re.compile(r"MySQLSyntaxErrorException", re.IGNORECASE), "mysql_java"),
    # PostgreSQL
    (re.compile(r"PostgreSQL.*?ERROR", re.IGNORECASE), "postgres_error"),
    (re.compile(r"pg_query\(|pg_exec\(", re.IGNORECASE), "postgres_func"),
    (re.compile(r"PSQLException", re.IGNORECASE), "postgres_java"),
    # SQLite
    (re.compile(r"SQLite3?::(?:query|exec)", re.IGNORECASE), "sqlite_php"),
    (re.compile(r"sqlite3\.OperationalError", re.IGNORECASE), "sqlite_python"),
    (re.compile(r"SQLite error", re.IGNORECASE), "sqlite_generic"),
    # MSSQL
    (re.compile(r"Microsoft SQL Server.*?\[SQL Server\]", re.IGNORECASE), "mssql_error"),
    (re.compile(r"ODBC SQL Server Driver", re.IGNORECASE), "mssql_odbc"),
    (re.compile(r"Unclosed quotation mark after the character string", re.IGNORECASE), "mssql_unclosed"),
    # Oracle
    (re.compile(r"ORA-\d{5}", re.IGNORECASE), "oracle_error"),
    # Generic
    (re.compile(r"Unclosed quotation mark", re.IGNORECASE), "generic_unclosed_quote"),
    (re.compile(r"you have an error in your sql syntax", re.IGNORECASE), "generic_syntax_error"),
]

# Time-based SQLi: SLEEP / BENCHMARK / pg_sleep / WAITFOR DELAY
_SQLI_TIME_PAYLOAD_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"SLEEP\s*\(\s*\d+\s*\)", re.IGNORECASE),
    re.compile(r"BENCHMARK\s*\(", re.IGNORECASE),
    re.compile(r"pg_sleep\s*\(", re.IGNORECASE),
    re.compile(r"WAITFOR\s+DELAY", re.IGNORECASE),
]

# UNION-based / data exfiltration banners
_SQLI_VERSION_BANNERS: list[re.Pattern[str]] = [
    re.compile(r"\b5\.\d+\.\d+-MariaDB", re.IGNORECASE),
    re.compile(r"\b5\.\d+\.\d+-MySQL", re.IGNORECASE),
    re.compile(r"\bPostgreSQL\s+\d+\.\d+", re.IGNORECASE),
    re.compile(r"\bMicrosoft\s+SQL\s+Server\s+\d+", re.IGNORECASE),
    re.compile(r"\binformation_schema\.(?:tables|columns)\b", re.IGNORECASE),
    re.compile(r"\b(?:root|postgres|sa)@(?:localhost|%\d+\.\d+\.\d+\.\d+)\b", re.IGNORECASE),
]


def verify_sqli(claim: VulnClaim) -> VerificationResult | None:
    """Verify SQL injection claims via 5 independent evidence channels.

    Channel 1: sqlmap proof (output contains "is vulnerable", "back-end DBMS",
               "Payload:", "Type: boolean-blind", etc.) — confidence 0.95.
    Channel 2: DBMS error strings leaked in output — confidence 0.85.
    Channel 3: Time-based (SLEEP/BENCHMARK/pg_sleep + measurable delay) —
               confidence 0.9 (with explicit delay) or 0.75 (payload only).
    Channel 4: UNION-based (version banner / information_schema leak) — 0.9.
    Channel 5: Boolean diff (TRUE/FALSE response length differs) — 0.8.

    Returns None if claim is not about SQL injection.
    """
    title_desc = (claim.title + " " + claim.vuln_type).lower()
    if not any(trig in title_desc for trig in _SQLI_TITLE_TRIGGERS):
        return None

    output = claim.evidence_provided or ""
    if not output or len(output) < 20:
        return None

    # VAPT-AI adaptation: VAPT-AI's VulnClaim does not have verification_command
    # or detection_command fields. Use payload + endpoint as proxy for the
    # SQLi command context.
    cmd_lower = (claim.payload + " " + claim.endpoint).lower()

    # ── Channel 1: sqlmap confirmation ───────────────────────────────
    for pattern in _SQLMAP_PROOF_PATTERNS:
        m = pattern.search(output)
        if m:
            return VerificationResult(
                status=VerificationStatus.VERIFIED,
                confidence=0.95,  # sqlmap is the gold standard
                method="sqli_sqlmap_confirmation",
                evidence=f"sqlmap proof pattern matched: {m.group(0)!r}",
                details={"pattern": m.group(0), "tool": "sqlmap"},
            )

    # ── Channel 2: DBMS error strings ────────────────────────────────
    for pattern, label in _DBMS_ERROR_PATTERNS:
        m = pattern.search(output)
        if m:
            return VerificationResult(
                status=VerificationStatus.VERIFIED,
                confidence=0.85,
                method="sqli_dbms_error_leak",
                evidence=f"DBMS error leaked ({label}): {m.group(0)[:120]}",
                details={"dbms": label, "match": m.group(0)[:200]},
            )

    # ── Channel 3: Time-based SQLi ───────────────────────────────────
    has_time_payload = any(
        p.search(cmd_lower + " " + output) for p in _SQLI_TIME_PAYLOAD_PATTERNS
    )
    if has_time_payload:
        # Look for explicit delay evidence in the output
        delay_match = re.search(
            r"(?:took|elapsed|delay|response\s+time)[^\d]{0,15}(\d+(?:\.\d+)?)\s*(s|sec|seconds|ms)",
            output,
            re.IGNORECASE,
        )
        if delay_match:
            try:
                val = float(delay_match.group(1))
                unit = delay_match.group(2).lower()
                seconds = val if unit in ("s", "sec", "seconds") else val / 1000.0
                if seconds >= 4.0:  # SLEEP(5) → ≥4s is strong evidence
                    return VerificationResult(
                        status=VerificationStatus.VERIFIED,
                        confidence=0.9,
                        method="sqli_time_based_delay",
                        evidence=(
                            f"Time-based SQLi confirmed: observed {seconds:.1f}s delay "
                            f"({delay_match.group(0)})"
                        ),
                        details={"delay_seconds": seconds, "raw": delay_match.group(0)},
                    )
            except ValueError:
                pass
        # Even without explicit delay number, presence of SLEEP/BENCHMARK
        # payload in the command AND a non-empty response is moderate evidence.
        if claim.payload and any(p.search(cmd_lower) for p in _SQLI_TIME_PAYLOAD_PATTERNS):
            return VerificationResult(
                status=VerificationStatus.VERIFIED,
                confidence=0.75,
                method="sqli_time_based_payload_present",
                evidence=(
                    "Time-based SQLi payload executed (SLEEP/BENCHMARK/pg_sleep) "
                    "— see claim payload"
                ),
                details={"payload": claim.payload[:200]},
            )

    # ── Channel 4: UNION-based / data exfiltration ──────────────────
    for pat in _SQLI_VERSION_BANNERS:
        m = pat.search(output)
        if m:
            return VerificationResult(
                status=VerificationStatus.VERIFIED,
                confidence=0.9,
                method="sqli_data_exfiltrated",
                evidence=f"DB version/data leaked via SQLi: {m.group(0)}",
                details={"banner": m.group(0)},
            )

    # ── Channel 5: Boolean-based diff ───────────────────────────────
    if re.search(
        r"(?:true|false).{0,80}(?:diff|differ|length|bytes).{0,80}(?:false|true)",
        output,
        re.IGNORECASE | re.DOTALL,
    ):
        return VerificationResult(
            status=VerificationStatus.VERIFIED,
            confidence=0.8,
            method="sqli_boolean_based_diff",
            evidence=(
                "Boolean-based SQLi: TRUE/FALSE payloads produced measurably "
                "different responses"
            ),
            details={},
        )

    return None


# ──────────────────────────────────────────────────────────────────────
# Strategy 2: Security header missing (EVVO line 107-189)
# ──────────────────────────────────────────────────────────────────────

def verify_security_header_missing(claim: VulnClaim) -> VerificationResult | None:
    """Verify claims about missing security headers.

    Two modes:
        1. Specific header named in title (e.g. "Missing HSTS header"):
           Confirm that header is absent from response headers in evidence.
        2. Generic "missing security headers" claim:
           Require at least 2 of 4 core headers (CSP, HSTS, X-Frame,
           X-Content-Type) missing — avoids FP on hosts that already hardened
           most.

    Cookie attribute parsing (HttpOnly/Secure/SameSite on Set-Cookie lines)
    is performed by `verify_cookie_security`, not here. We only inspect the
    named security headers.
    """
    target_header: str | None = None
    title_desc = (claim.title + " " + claim.vuln_type).lower()

    for header_name, keywords in _SECURITY_HEADERS.items():
        if any(kw in title_desc for kw in keywords):
            target_header = header_name
            break

    output = claim.evidence_provided or ""

    # Generic "missing security headers" claim (no specific header named)
    if not target_header:
        generic_keywords = (
            "security header", "missing header", "harden", "header hardening",
        )
        if not any(kw in title_desc for kw in generic_keywords):
            return None  # Not a security-header claim
        if not output:
            return None
        headers = _extract_headers_from_output(output)
        if not headers:
            return None
        absent = [h for h in _SECURITY_HEADERS if not _header_present(headers, h)]
        core_missing = [h for h in absent if h in _CORE_SECURITY_HEADERS]
        if len(core_missing) >= 2:
            return VerificationResult(
                status=VerificationStatus.VERIFIED,
                confidence=0.8,
                method="security_header_multi_check",
                evidence=f"Missing security headers: {', '.join(core_missing)}",
                details={
                    "missing": core_missing,
                    "checked_headers": list(headers.keys()),
                },
            )
        if not core_missing:
            return VerificationResult(
                status=VerificationStatus.FALSE_POSITIVE,
                confidence=0.75,
                method="security_header_multi_check",
                evidence=(
                    "All core security headers (HSTS, CSP, X-Frame, "
                    "X-Content-Type) present"
                ),
                details={},
            )
        # Some but < 2 missing — inconclusive
        return None

    # Specific header check
    if output:
        headers = _extract_headers_from_output(output)
        if not _header_present(headers, target_header):
            return VerificationResult(
                status=VerificationStatus.VERIFIED,
                confidence=0.85,
                method="header_absence_check",
                evidence=f"Header '{target_header}' not found in response headers",
                details={
                    "checked_headers": list(headers.keys()),
                    "target": target_header,
                },
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


# ──────────────────────────────────────────────────────────────────────
# Strategy 3: Server disclosure (EVVO line 192-218)
# ──────────────────────────────────────────────────────────────────────

# Known server-banner patterns that leak version digits. Confidence bonus if
# the disclosed product+version has known CVEs (rough heuristic — full CVE
# lookup is deferred to W13+ where live enrichment is available).
_KNOWN_CVE_VERSION_HINTS: list[tuple[re.Pattern[str], str]] = [
    # (regex matching Server header, CVE-family hint label)
    (re.compile(r"nginx/(\d+\.\d+\.\d+)", re.IGNORECASE), "nginx_version_disclosed"),
    (re.compile(r"apache/(\d+\.\d+\.\d+)", re.IGNORECASE), "apache_version_disclosed"),
    (re.compile(r"microsoft-iis/(\d+\.\d+)", re.IGNORECASE), "iis_version_disclosed"),
    (re.compile(r"jetty\((\d+\.\d+\.\d+)", re.IGNORECASE), "jetty_version_disclosed"),
]

# X-Powered-By patterns leaking framework/language version
_X_POWERED_BY_VERSION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"PHP/(\d+\.\d+\.\d+)", re.IGNORECASE),
    re.compile(r"Python/(\d+\.\d+\.\d+)", re.IGNORECASE),
    re.compile(r"ASP\.NET", re.IGNORECASE),
    re.compile(r"Express", re.IGNORECASE),
    re.compile(r"Servlet", re.IGNORECASE),
]


def verify_server_disclosure(claim: VulnClaim) -> VerificationResult | None:
    """Verify server version disclosure in headers.

    Triggers if claim title or vuln_type mentions 'server' or 'version'.
    Looks for:
        - Server header containing version digits → confidence 0.8
          (with extra +0.05 if known product+version pattern matches a
          CVE-family hint — total capped at 0.95)
        - X-Powered-By header leaking PHP/Python/etc. version → 0.75
        - Server header present but no version digits → 0.6
    """
    title_lower = (claim.title + " " + claim.vuln_type).lower()
    if "server" not in title_lower and "version" not in title_lower:
        return None

    output = claim.evidence_provided or ""
    headers = _extract_headers_from_output(output)

    server_header = headers.get("server", "")
    x_powered_by = headers.get("x_powered_by", "")

    # Server header with version digits
    if server_header and any(c.isdigit() for c in server_header):
        confidence = 0.8
        cve_hint = None
        for pat, label in _KNOWN_CVE_VERSION_HINTS:
            if pat.search(server_header):
                cve_hint = label
                confidence = min(0.95, confidence + 0.05)
                break
        details: dict[str, Any] = {"server_header": server_header}
        if cve_hint:
            details["cve_hint"] = cve_hint
        return VerificationResult(
            status=VerificationStatus.VERIFIED,
            confidence=confidence,
            method="server_header_version_check",
            evidence=f"Server header contains version: {server_header[:100]}",
            details=details,
        )

    # X-Powered-By header leaks framework version
    if x_powered_by and any(c.isdigit() for c in x_powered_by):
        for pat in _X_POWERED_BY_VERSION_PATTERNS:
            m = pat.search(x_powered_by)
            if m:
                return VerificationResult(
                    status=VerificationStatus.VERIFIED,
                    confidence=0.75,
                    method="x_powered_by_version_check",
                    evidence=f"X-Powered-By leaks version: {x_powered_by[:100]}",
                    details={
                        "x_powered_by": x_powered_by,
                        "matched": m.group(0),
                    },
                )

    # Server header present but no clear version
    if server_header:
        return VerificationResult(
            status=VerificationStatus.VERIFIED,
            confidence=0.6,
            method="server_header_present",
            evidence=f"Server header present: {server_header[:100]}",
            details={"server_header": server_header},
        )

    return None


# ──────────────────────────────────────────────────────────────────────
# Strategy 4: Cookie security (EVVO line 220-260)
# ──────────────────────────────────────────────────────────────────────

def verify_cookie_security(claim: VulnClaim) -> VerificationResult | None:
    """Verify cookie security issues (missing HttpOnly, Secure, SameSite).

    Triggers if claim title or vuln_type mentions 'cookie'.
    Inspects Set-Cookie headers in evidence — handles multiple Set-Cookie
    lines (one per cookie). For each cookie, checks:
        - HttpOnly flag present
        - Secure flag present (mandatory on HTTPS responses)
        - SameSite flag present (and not 'None' unless explicitly allowed)
    """
    title_lower = (claim.title + " " + claim.vuln_type).lower()
    if "cookie" not in title_lower:
        return None

    output = claim.evidence_provided or ""

    cookie_lines = _extract_set_cookie_lines(output)

    if not cookie_lines:
        return VerificationResult(
            status=VerificationStatus.INCONCLUSIVE,
            confidence=0.5,
            method="cookie_check",
            evidence="No Set-Cookie headers found in evidence",
            details={},
        )

    issues_found: list[str] = []
    cookies_with_issues: list[dict[str, Any]] = []
    for line in cookie_lines:
        line_lower = line.lower()
        # Extract the cookie name for diagnostic detail
        # Set-Cookie: name=value; Path=/; HttpOnly; Secure; SameSite=Lax
        cookie_name = "unknown"
        try:
            value_part = line.split(":", 1)[1].strip()
            cookie_name = value_part.split("=", 1)[0].strip()
        except (IndexError, ValueError):
            pass

        cookie_issues: list[str] = []
        if "httponly" not in line_lower:
            cookie_issues.append("Missing HttpOnly")
        if "secure" not in line_lower:
            cookie_issues.append("Missing Secure")
        if "samesite" not in line_lower:
            cookie_issues.append("Missing SameSite")
        elif "samesite=none" in line_lower:
            # SameSite=None is explicit but weak (cross-site allowed)
            cookie_issues.append("SameSite=None")

        if cookie_issues:
            issues_found.extend(cookie_issues)
            cookies_with_issues.append({
                "cookie_name": cookie_name,
                "issues": cookie_issues,
            })

    if issues_found:
        # Confidence scales with number of missing flags (more missing → higher)
        base_confidence = 0.75
        # Slight boost if many cookies have issues (systemic problem)
        if len(cookies_with_issues) >= 2:
            base_confidence = min(0.85, base_confidence + 0.05)
        return VerificationResult(
            status=VerificationStatus.VERIFIED,
            confidence=base_confidence,
            method="cookie_security_check",
            evidence=(
                f"Cookie issues found: {', '.join(issues_found)} in "
                f"{len(cookie_lines)} Set-Cookie header(s)"
            ),
            details={
                "cookies_checked": len(cookie_lines),
                "issues": issues_found,
                "cookies_with_issues": cookies_with_issues[:10],
            },
        )

    # All cookies have all 3 attributes — claim is FP
    return VerificationResult(
        status=VerificationStatus.FALSE_POSITIVE,
        confidence=0.7,
        method="cookie_security_check",
        evidence="All cookies have HttpOnly, Secure, and SameSite attributes",
        details={"cookies_checked": len(cookie_lines)},
    )


# ──────────────────────────────────────────────────────────────────────
# Strategy 5: XSS reflected (EVVO line 269-295)
# ──────────────────────────────────────────────────────────────────────

# XSS reflection patterns compiled at module-level for performance
_XSS_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"<script[^>]*>[\s\S]*?</script>", re.IGNORECASE),
    re.compile(r"javascript:", re.IGNORECASE),
    re.compile(r"on\w+\s*=", re.IGNORECASE),  # onload=, onerror=, etc.
    re.compile(r"<iframe", re.IGNORECASE),
    re.compile(r"<object", re.IGNORECASE),
    re.compile(r"<embed", re.IGNORECASE),
]


def verify_xss_reflected(claim: VulnClaim) -> VerificationResult | None:
    """Basic verification for reflected XSS claims.

    Looks for common XSS payload patterns in the evidence_provided text.
    Returns confidence 0.7 if any pattern matches (single pattern match
    is sufficient — full PoC validator with payload replay is in W13+).
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
            # Use the source pattern string for diagnostic detail
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
            details={"matched_patterns": matched_patterns[:5]},
        )

    return None



# ──────────────────────────────────────────────────────────────────────
# Strategy 6: Info disclosure (legacy W12 — not in EVVO port)
# ──────────────────────────────────────────────────────────────────────

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


# ──────────────────────────────────────────────────────────────────────
# Strategy registry — merged EVVO + legacy (6 strategies)
# ──────────────────────────────────────────────────────────────────────

DEFAULT_STRATEGIES = [
    verify_sqli,
    verify_security_header_missing,
    verify_server_disclosure,
    verify_cookie_security,
    verify_xss_reflected,
    verify_info_disclosure,
]
