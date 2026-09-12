"""
VAPT-AI PII Redactor — D24.

Redacts PII (Personally Identifiable Information) from tool outputs BEFORE
persistence to database. This is a defense-in-depth layer — even if the DB
is compromised, PII is not exposed.

Architecture:
    Layer 1: scrubadub (main) — standard PII types:
        - emails, phones, SSNs, credit cards, IBANs, URLs, names (optional)
        - scrubadub uses regex + NLP (spaCy) for high accuracy
    Layer 2: regex patterns (fallback) — security-specific PII scrubadub misses:
        - JWT tokens (eyJ...)
        - API keys (Bearer xxx, api_key=xxx, Authorization: xxx)
        - AWS access keys (AKIA...)
        - AWS secret keys (40-char base64)
        - GitHub tokens (ghp_..., gho_..., ghs_...)
        - Slack tokens (xox[abp]-...)
        - Stripe keys (sk_live_..., pk_live_...)
        - Private keys (-----BEGIN ... PRIVATE KEY-----)
        - Passwords in URLs (https://user:pass@host)
        - Generic secrets (secret=xxx, password=xxx, token=xxx)
        - IPv4 addresses (optional — configurable)
        - MAC addresses

Redaction format:
    PII is replaced with deterministic placeholder:
        {{EMAIL:hash}}    — e.g. {{EMAIL:a1b2c3}}
        {{PHONE:hash}}
        {{JWT:hash}}
        {{API_KEY:hash}}
        {{AWS_KEY:hash}}
        {{GITHUB_TOKEN:hash}}
        {{PRIVATE_KEY:hash}}
        {{PASSWORD:hash}}
        {{IP:hash}}

    The hash is first 6 chars of SHA-256 of the original value — deterministic
    so the same PII always gets the same placeholder (useful for log correlation
    without revealing the PII).

Usage:
    from app.pii.redactor import redact_pii
    cleaned = redact_pii("Contact me at john@example.com or call 555-123-4567")
    # → "Contact me at {{EMAIL:a1b2c3}} or call {{PHONE:d4e5f6}}"

Performance:
    scrubadub is slow (NLP). For high-throughput paths, set use_scrubadub=False
    to use regex-only mode (faster, less accurate for names).

D24 requirements:
    - PII redaction on ALL raw tool output before persistence
    - Patterns: emails, phones, JWTs, API keys, CCs, SSNs, IBANs, AWS keys,
      private keys, passwords in URLs
"""
from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# ---------- Regex patterns (security-specific PII) ----------

@dataclass
class PIIPattern:
    """A single PII detection pattern."""
    name: str                # e.g. "EMAIL", "JWT", "AWS_KEY"
    pattern: str             # regex pattern
    replacement_template: str  # e.g. "{{EMAIL|%s}}"
    custom_replacer: bool = False  # True = use custom logic (preserves parts of match)


# Order matters: more specific patterns first (e.g. JWT before PASSWORD_KV)
PATTERNS: list[PIIPattern] = [
    # ---------- PEM private keys (multi-line, must run first before other patterns eat parts) ----------
    PIIPattern(
        name="PRIVATE_KEY",
        pattern=r"-----BEGIN (?:RSA |EC |DSA |OPENSSH |PGP |ENCRYPTED )?PRIVATE KEY-----[\s\S]*?-----END (?:RSA |EC |DSA |OPENSSH |PGP |ENCRYPTED )?PRIVATE KEY-----",
        replacement_template="{{PRIVATE_KEY|%s}}",
    ),

    # ---------- Specific tokens (before generic PASSWORD_KV) ----------
    PIIPattern(
        name="JWT",
        pattern=r"eyJ[A-Za-z0-9_-]+\.eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+",
        replacement_template="{{JWT|%s}}",
    ),
    PIIPattern(
        name="GITHUB_TOKEN",
        # GitHub PAT: ghp_/gho_/ghs_/ghr_ + 36+ chars (some tokens have 38+)
        pattern=r"gh[posr]_[A-Za-z0-9]{36,}",
        replacement_template="{{GITHUB_TOKEN|%s}}",
    ),
    PIIPattern(
        name="SLACK_TOKEN",
        pattern=r"xox[abp]-[A-Za-z0-9-]+",
        replacement_template="{{SLACK_TOKEN|%s}}",
    ),
    PIIPattern(
        name="STRIPE_KEY",
        pattern=r"(?:sk|pk)_(?:live|test)_[A-Za-z0-9]+",
        replacement_template="{{STRIPE_KEY|%s}}",
    ),
    PIIPattern(
        name="AWS_ACCESS_KEY",
        pattern=r"AKIA[A-Z0-9]{16}",
        replacement_template="{{AWS_ACCESS_KEY|%s}}",
    ),

    # ---------- Passwords in URLs + auth headers (before PASSWORD_KV) ----------
    # NOTE: These patterns use custom replacement logic (see _apply_pattern below)
    # because they need to preserve parts of the match (URL scheme, header name)
    # while redacting only the secret portion.
    PIIPattern(
        name="PASSWORD_IN_URL",
        # Match: https://user:password@host — group 1=scheme://user, group 2=password
        pattern=r"(https?://[^:/@\s]+):([^/@\s]+)@",
        replacement_template="{{PASSWORD_IN_URL:%s}}",  # special handling
        # Custom replacer: keeps group 1 + redacts group 2
        custom_replacer=True,
    ),
    PIIPattern(
        name="AUTH_HEADER",
        # Match: Authorization: Bearer xxx — group 1=header name, group 2=token
        pattern=r"(Authorization|Bearer)([\s:=]+)([\"']?)([A-Za-z0-9_\-\.]+)([\"']?)",
        replacement_template="{{API_KEY|%s}}",  # special handling
        # Custom replacer: keeps groups 1-3 + redacts group 4 + keeps group 5
        custom_replacer=True,
    ),

    # ---------- Standard PII (regex fallback for scrubadub) ----------
    PIIPattern(
        name="EMAIL",
        pattern=r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
        replacement_template="{{EMAIL|%s}}",
    ),
    PIIPattern(
        name="PHONE_INTL",
        pattern=r"\+\d{1,3}[\s.-]?\(?\d{1,4}\)?[\s.-]?\d{1,4}[\s.-]?\d{1,9}",
        replacement_template="{{PHONE|%s}}",
    ),
    PIIPattern(
        name="PHONE_US",
        pattern=r"\(?\b\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
        replacement_template="{{PHONE|%s}}",
    ),
    PIIPattern(
        name="SSN",
        pattern=r"\b(?!000|666|9\d{2})\d{3}-(?!00)\d{2}-(?!0000)\d{4}\b",
        replacement_template="{{SSN|%s}}",
    ),
    PIIPattern(
        name="IBAN",
        pattern=r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b",
        replacement_template="{{IBAN|%s}}",
    ),

    # ---------- AWS secret key (after access key) ----------
    PIIPattern(
        name="AWS_SECRET_KEY",
        pattern=r"(?:aws_secret|secret_access_key|aws_secret_access_key)[\"']?\s*[:=]\s*[\"']([A-Za-z0-9/+=]{40})[\"']",
        replacement_template="{{AWS_SECRET_KEY|%s}}",
    ),

    # ---------- Generic password/key=value (LAST — least specific) ----------
    PIIPattern(
        name="PASSWORD_KV",
        # Match: password=xxx, passwd:xxx, pwd=xxx, secret=xxx, api_key=xxx
        # group 1 = key name, group 2 = value
        pattern=r"(?i)(password|passwd|pwd|secret|api[_-]?key)[\"']?\s*[:=]\s*[\"']?([^\s\"',;]+)",
        replacement_template="{{PASSWORD|%s}}",
        custom_replacer=True,  # preserves group 1 (key name), redacts group 2 (value)
    ),

    # ---------- Credit card (after SSN — both are digit patterns) ----------
    PIIPattern(
        name="CREDIT_CARD",
        pattern=r"\b(?:\d[ -]*?){13,19}\b",
        replacement_template="{{CREDIT_CARD|%s}}",
    ),

    # ---------- Network (optional — configurable) ----------
    PIIPattern(
        name="IPV4",
        pattern=r"\b(?:(?!10\.|127\.|192\.168\.|172\.(?:1[6-9]|2\d|3[01])\.|0\.0\.0\.0|255\.255\.255\.255)(?:\d{1,3}\.){3}\d{1,3})\b",
        replacement_template="{{IP|%s}}",
    ),
    PIIPattern(
        name="MAC_ADDRESS",
        pattern=r"\b(?:[0-9A-Fa-f]{2}[:-]){5}[0-9A-Fa-f]{2}\b",
        replacement_template="{{MAC|%s}}",
    ),
]


# ---------- Redactor ----------

def _hash_pii(value: str) -> str:
    """Generate deterministic 6-char hash for PII value.

    Same PII → same hash → same placeholder (useful for log correlation).
    """
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:6]


@dataclass
class RedactionResult:
    """Result of PII redaction."""
    redacted_text: str
    patterns_matched: dict[str, int] = field(default_factory=dict)
    total_redactions: int = 0

    def add_match(self, pattern_name: str) -> None:
        self.patterns_matched[pattern_name] = self.patterns_matched.get(pattern_name, 0) + 1
        self.total_redactions += 1


def redact_pii(
    text: str,
    use_scrubadub: bool = True,
    redact_ips: bool = False,
) -> str:
    """Redact PII from text. Returns redacted text.

    Args:
        text: Input text potentially containing PII
        use_scrubadub: If True, use scrubadub (NLP-based) as first pass.
                      Set False for high-throughput paths (regex-only).
        redact_ips: If True, redact IPv4 addresses (default False — IPs are
                   often needed for security analysis)

    Returns:
        Redacted text with PII replaced by {{TYPE:hash}} placeholders.
    """
    if not text or not isinstance(text, str):
        return text

    result = RedactionResult(redacted_text=text)

    # Layer 1: scrubadub (NLP-based, slow but accurate for names/emails)
    if use_scrubadub:
        try:
            import scrubadub
            scrubber = scrubadub.Scrubber()
            result.redacted_text = scrubber.clean(result.redacted_text)
        except ImportError:
            logger.debug("scrubadub not installed — using regex only")
        except Exception as e:
            logger.warning("scrubadub failed (%s) — falling back to regex", e)

    # Layer 2: regex patterns (security-specific PII)
    for pattern in PATTERNS:
        # Skip IP redaction unless explicitly requested
        if pattern.name == "IPV4" and not redact_ips:
            continue

        # Find all matches first (for counting)
        matches = re.findall(pattern.pattern, result.redacted_text)
        if matches:
            match_count = len(matches)
            if match_count > 0:
                result.add_match(pattern.name)

        # Apply replacement
        if pattern.custom_replacer:
            # Custom logic: preserve parts of match, redact only secret portion
            if pattern.name == "PASSWORD_IN_URL":
                # group 1 = scheme://user, group 2 = password
                def replacer(m):
                    prefix = m.group(1)  # https://user
                    password = m.group(2)  # password
                    h = _hash_pii(password)
                    return f"{prefix}:{{{{PASSWORD|{h}}}}}@"
                result.redacted_text = re.sub(pattern.pattern, replacer, result.redacted_text)
            elif pattern.name == "AUTH_HEADER":
                # group 1 = header name, group 2 = separator, group 3 = quote1, group 4 = token, group 5 = quote2
                def replacer(m):
                    header = m.group(1)
                    sep = m.group(2)
                    quote1 = m.group(3)
                    token = m.group(4)
                    quote2 = m.group(5)
                    h = _hash_pii(token)
                    return f"{header}{sep}{quote1}{{{{API_KEY|{h}}}}}{quote2}"
                result.redacted_text = re.sub(pattern.pattern, replacer, result.redacted_text)
            elif pattern.name == "PASSWORD_KV":
                # group 1 = key name (password/secret/etc.), group 2 = value
                def replacer(m):
                    key = m.group(1)
                    value = m.group(2)
                    h = _hash_pii(value)
                    return f"{key}={{{{PASSWORD|{h}}}}}"
                result.redacted_text = re.sub(pattern.pattern, replacer, result.redacted_text)
        else:
            # Standard: hash full match, replace with {{TYPE:hash}}
            def make_standard_replacer(pat_name: str, template: str):
                def replacer(m):
                    full_match = m.group(0)
                    h = _hash_pii(full_match)
                    return template.replace("%s", h)
                return replacer
            try:
                result.redacted_text = re.sub(
                    pattern.pattern,
                    make_standard_replacer(pattern.name, pattern.replacement_template),
                    result.redacted_text,
                )
            except re.error as e:
                logger.error("Regex error in pattern %s: %s", pattern.name, e)

    if result.total_redactions > 0:
        logger.debug("Redacted %d PII items: %s",
                     result.total_redactions, result.patterns_matched)

    return result.redacted_text


def redact_dict(data: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
    """Recursively redact PII from a dict (e.g. JSON tool output).

    Walks dict + lists, redacts string values.
    """
    if isinstance(data, dict):
        return {k: redact_dict(v, **kwargs) for k, v in data.items()}
    elif isinstance(data, list):
        return [redact_dict(item, **kwargs) for item in data]
    elif isinstance(data, str):
        return redact_pii(data, **kwargs)
    else:
        return data


# ---------- Audit helpers ----------

def audit_redactions(text: str, **kwargs: Any) -> dict[str, Any]:
    """Redact PII + return audit info (what was redacted, not the PII itself).

    Useful for logging: "Redacted 5 PII items: {EMAIL: 2, JWT: 1, AWS_KEY: 2}"
    without revealing the actual PII.
    """
    if not text:
        return {"redacted": False, "patterns": {}, "total": 0}

    # Run redaction + capture result
    redacted = text
    patterns_matched: dict[str, int] = {}

    for pattern in PATTERNS:
        if pattern.name == "IPV4" and not kwargs.get("redact_ips", False):
            continue
        matches = re.findall(pattern.pattern, redacted)
        if matches:
            patterns_matched[pattern.name] = len(matches)

    return {
        "redacted": len(patterns_matched) > 0,
        "patterns": patterns_matched,
        "total": sum(patterns_matched.values()),
    }


if __name__ == "__main__":
    # Quick test
    test_text = """
    Contact: john.doe@example.com, phone: +1-555-123-4567
    JWT: eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c
    AWS: AKIAIOSFODNN7EXAMPLE, secret=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY
    GitHub: ghp_1234567890abcdefghijklmnopqrstuvwxyzAB
    URL: https://admin:SuperSecret123@internal.example.com/api
    Server: 192.168.1.100
    """
    print("=== Original ===")
    print(test_text)
    print("\n=== Redacted ===")
    print(redact_pii(test_text, redact_ips=True))
    print("\n=== Audit ===")
    print(audit_redactions(test_text, redact_ips=True))
