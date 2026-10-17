"""Deterministic evidence verifier — the ONLY source of a finding's
verification level.

Why this module exists
----------------------
``tool_bridge._record_vulnerability`` used to hand the ReAct loop back a
finding with ``"verified": True`` hardcoded — for *every* successfully
recorded finding, whatever the evidence contained. ``tda.evidence_score_from_finding()``
then scored each one at ``CONFIRMED_VULN`` (0.8), so:

  * the 0.5 / 0.8 / 1.0 evidence ladder collapsed to two rungs;
  * ``replay_reproduced`` / ``human_verified`` were never set, so
    ``VERIFIED_EXPLOIT`` (1.0) was dead code inside the loop;
  * and the "evidence-gated" TDA EXPLOIT override was, in practice,
    ``len(findings) > 0`` with extra steps — the gate was built and
    immediately undermined by the value it was supposed to gate on.

This module restores a real ladder. It is DELIBERATELY rule-based: no
second LLM call. Asking another model "is this real?" only moves the
tautology one level down (and burns tokens on every record).

Levels
------
``plausible``       (0.5) — default. Evidence is present but contains no
                            *outcome assertion* for the claimed class.
``confirmed``       (0.8) — the evidence contains a verdict the TOOL
                            ITSELF emitted for the claimed class (sqlmap
                            "is vulnerable", dalfox "[V]", a leaked .env
                            body, a readable /etc/passwd, ...).
``replay_verified`` (1.0) — an independent replay executed the payload
                            and reproduced it. Never inferred here; only
                            set when the caller states it.

The three levels are also what the TDA evidence ladder consumes —
see ``app.rl.tda.evidence_score_from_finding()``.
"""

from __future__ import annotations

import re
from typing import Iterable

__all__ = [
    "VERIFICATION_PLAUSIBLE",
    "VERIFICATION_CONFIRMED",
    "VERIFICATION_REPLAY_VERIFIED",
    "VERIFICATION_LEVELS",
    "verification_rank",
    "classify_vuln",
    "assess_verification",
    "parse_verification",
]

VERIFICATION_PLAUSIBLE = "plausible"
VERIFICATION_CONFIRMED = "confirmed"
VERIFICATION_REPLAY_VERIFIED = "replay_verified"

VERIFICATION_LEVELS: tuple[str, ...] = (
    VERIFICATION_PLAUSIBLE,
    VERIFICATION_CONFIRMED,
    VERIFICATION_REPLAY_VERIFIED,
)

_RANK: dict[str, int] = {
    VERIFICATION_PLAUSIBLE: 1,
    VERIFICATION_CONFIRMED: 2,
    VERIFICATION_REPLAY_VERIFIED: 3,
}


def verification_rank(level: str | None) -> int:
    """Ordinal rank of a verification level (0 for unknown/None)."""
    if not isinstance(level, str):
        return 0
    return _RANK.get(level.strip().lower(), 0)


def parse_verification(value: object, *, default: str = VERIFICATION_PLAUSIBLE) -> str:
    """Coerce an arbitrary value into a known verification level.

    Unknown strings default to `default` — a typo must never be able to
    promote a finding to CONFIRMED.
    """
    if isinstance(value, str):
        key = value.strip().lower()
        if key in _RANK:
            return key
    return default


# ---------------------------------------------------------------------------
# vuln_type -> coarse class
# ---------------------------------------------------------------------------

# (class key, pattern matched against the lower-cased vuln_type)
_CLASS_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("sqli", re.compile(r"sql[\s_-]?inj|sqli|sql[\s_-]?query|blind[\s_-]?sql|error[\s_-]?based")),
    ("xss", re.compile(r"\bxss\b|cross[\s_-]?site[\s_-]?script|dom[\s_-]?xss")),
    ("ssti", re.compile(r"\bssti\b|template[\s_-]?inj|server[\s_-]?side[\s_-]?template")),
    ("lfi", re.compile(r"\blfi\b|local[\s_-]?file|path[\s_-]?traversal|directory[\s_-]?traversal|\.\./")),
    ("rce", re.compile(r"\brce\b|remote[\s_-]?code|command[\s_-]?inj|os[\s_-]?command|shell[\s_-]?inj")),
    ("xxe", re.compile(r"\bxxe\b|xml[\s_-]?external")),
    ("ssrf", re.compile(r"\bssrf\b|server[\s_-]?side[\s_-]?request")),
    ("open_redirect", re.compile(r"open[\s_-]?redirect")),
    ("exposure", re.compile(
        r"information[\s_-]?disclos|sensitive[\s_-]?file|exposure|exposed|leak|"
        r"backup[\s_-]?file|\.env|directory[\s_-]?listing|debug[\s_-]?page|phpinfo"
    )),
    ("default_creds", re.compile(r"default[\s_-]?(?:cred|password|login)|weak[\s_-]?password|brute")),
    ("takeover", re.compile(r"subdomain[\s_-]?takeover|dangling[\s_-]?cname")),
    ("file_upload", re.compile(r"file[\s_-]?upload|unrestricted[\s_-]?upload")),
    ("idor", re.compile(r"\bidor\b|insecure[\s_-]?direct|broken[\s_-]?access|access[\s_-]?control|authoriz")),
    ("csrf", re.compile(r"\bcsrf\b|cross[\s_-]?site[\s_-]?request")),
)


def classify_vuln(vuln_type: str | None) -> str:
    """Map a free-form ``vuln_type`` to a coarse class key.

    Returns ``""`` when no class matches (the caller then falls back to
    the generic tool-verdict markers).
    """
    if not vuln_type or not isinstance(vuln_type, str):
        return ""
    text = vuln_type.strip().lower()
    for key, pattern in _CLASS_RULES:
        if pattern.search(text):
            return key
    return ""


# ---------------------------------------------------------------------------
# class -> tool verdict markers
# ---------------------------------------------------------------------------
# Each pattern must be something the TOOL printed as a conclusion, not
# something the LLM wrote as a claim. Keep them tight: a false CONFIRMED
# tunnels the whole scan onto a vector that may not exist.

_CONFIRM_MARKERS: dict[str, tuple[re.Pattern[str], ...]] = {
    "sqli": (
        re.compile(r"\bis vulnerable\b", re.I),
        re.compile(r"identified the following injection point", re.I),
        re.compile(r"back-end DBMS:", re.I),
        re.compile(r"\bType:\s*(?:boolean|time|error|UNION|stacked|inline)", re.I),
        re.compile(r"You have an error in your SQL syntax", re.I),
        re.compile(r"SQLSTATE\[\w+\]", re.I),
        re.compile(r"unterminated quoted string", re.I),
        re.compile(r"XPATH syntax error", re.I),
    ),
    "xss": (
        re.compile(r"\[V\]"),                     # dalfox verified marker
        re.compile(r"\[POC\]", re.I),             # dalfox PoC marker
        re.compile(r"Vulnerable to XSS", re.I),
        re.compile(r"found\s+\d+\s+(?:reflected\s+)?XSS", re.I),
        re.compile(r"<script>\s*alert\s*\(", re.I),
        re.compile(r"onerror\s*=\s*alert\s*\(", re.I),
    ),
    "ssti": (
        re.compile(r"\{\{\s*7\s*\*\s*7\s*\}\}.*\b49\b", re.I | re.S),
        re.compile(r"\$\{\s*7\s*\*\s*7\s*\}.*\b49\b", re.S),
        re.compile(r"template (?:injection|engine).*(?:49|executed)", re.I | re.S),
    ),
    "lfi": (
        re.compile(r"root:x:0:0:"),               # /etc/passwd
        re.compile(r"\[boot loader\]", re.I),      # boot.ini
        re.compile(r"; for 16-bit app support", re.I),  # win.ini
        re.compile(r"#\s*/etc/passwd"),
        re.compile(r"<\?php\b"),
    ),
    "rce": (
        re.compile(r"uid=\d+\(\w+\)\s+gid=\d+"),  # `id`
        re.compile(r"\bnt authority\\system", re.I),
        re.compile(r"Volume in drive [A-Z] has no label", re.I),
        re.compile(r"uid=\d+\("),
    ),
    "xxe": (
        re.compile(r"root:x:0:0:"),
        re.compile(r"&xxe;.*(?:root:|file://)", re.I | re.S),
    ),
    "ssrf": (
        re.compile(r"169\.254\.169\.254"),
        re.compile(r"\bami-id\b", re.I),
        re.compile(r"\binstance-id\b", re.I),
        re.compile(r"metadata\.google\.internal", re.I),
    ),
    "open_redirect": (
        re.compile(r"^location:\s*https?://(?!.*(?:\bthe same\b|localhost))", re.I | re.M),
        re.compile(r"redirect(?:ed)? to\s+https?://", re.I),
    ),
    "exposure": (
        re.compile(r"(?:DB_PASSWORD|DB_USERNAME|SECRET_KEY|APP_KEY|API_KEY|AWS_SECRET_ACCESS_KEY)\s*=", re.I),
        re.compile(r"aws_access_key_id", re.I),
        re.compile(r"BEGIN (?:RSA |EC |OPENSSH |PGP )?PRIVATE KEY"),
        re.compile(r"ref:\s*refs/heads/", re.I),           # .git/HEAD
        re.compile(r"repositoryformatversion\s*=", re.I),  # .git/config
        re.compile(r"<title>\s*Index of /", re.I),         # directory listing
        re.compile(r"phpinfo\(\)|Loaded Configuration File", re.I),
        re.compile(r"^\s*<h1>\s*(?:Dashboard|Welcome to nginx)", re.I | re.M),
    ),
    "default_creds": (
        re.compile(r"(?:login|authentication)\s+(?:successful|succeeded)", re.I),
        re.compile(r"Set-Cookie:.*(?:PHPSESSID|JSESSIONID|session)", re.I),
        re.compile(r"Welcome,?\s+(?:admin|administrator)", re.I),
    ),
    "takeover": (
        re.compile(r"NoSuchBucket", re.I),
        re.compile(r"There isn't a GitHub Pages site here", re.I),
        re.compile(r"Sorry, this shop is currently unavailable", re.I),
    ),
    "file_upload": (
        re.compile(r"(?:upload|uploaded).*(?:\.php|\.jsp|\.aspx).*(?:success|200)", re.I | re.S),
    ),
    "idor": (
        re.compile(r"(?:another|other|victim)\s+user'?s?\s+(?:data|profile|order)", re.I),
        re.compile(r"\"(?:email|username)\"\s*:\s*\"[^\"]+@", re.I),
    ),
    "csrf": (
        re.compile(r"no (?:anti-)?csrf token", re.I),
        re.compile(r"csrf (?:token )?missing", re.I),
    ),
}

# Tool verdicts that are class-agnostic: a scanner template matched, or
# the tool literally printed an exploit result.
_GENERIC_CONFIRM_MARKERS: tuple[re.Pattern[str], ...] = (
    # nuclei  ->  [template-id] [http] [high] https://target/path
    re.compile(
        r"\[[a-z0-9][a-z0-9_.:-]{2,60}\]\s+\[(?:http|https|network|dns|file|tcp|ssl|websocket)"
        r"[^\]]*\]\s+\[(?:critical|high|medium|low)\]",
        re.I,
    ),
    # nikto   ->  + OSVDB-1234: /path: description
    re.compile(r"^\s*\+\s+(?:OSVDB|CVE)-\d+", re.I | re.M),
    # whatweb -> [200 OK] ... [PHP/8.1] ... [Apache/2.4]
    re.compile(r"\[(?:cpe:/a|CVE-\d{4}-\d{4,})\]", re.I),
)

# Below this length the "evidence" cannot plausibly contain a tool verdict.
_MIN_EVIDENCE_CHARS = 16

# Only the head of the evidence is scanned — findings can carry megabytes
# of raw tool output and the verdict is always near the top.
_SCAN_CHARS = 20_000


def _first_match(text: str, patterns: Iterable[re.Pattern[str]]) -> re.Pattern[str] | None:
    for pattern in patterns:
        if pattern.search(text):
            return pattern
    return None


def assess_verification(
    *,
    vuln_type: str | None,
    evidence: str | None,
    tool: str = "",
    replay_reproduced: bool = False,
    human_verified: bool = False,
) -> tuple[str, str]:
    """Decide the verification level of a finding.

    Args:
        vuln_type: the finding's claimed vulnerability class (free-form).
        evidence: the raw tool output attached as evidence.
        tool: the tool that produced the evidence (informational only —
            used in the reason string, not for gating).
        replay_reproduced: an independent replay reproduced the payload.
        human_verified: a human marked the finding as verified.

    Returns:
        ``(level, reason)`` where level is one of ``plausible`` /
        ``confirmed`` / ``replay_verified``.
    """
    # Explicit external signals outrank everything derived from text.
    if replay_reproduced:
        return VERIFICATION_REPLAY_VERIFIED, "independent replay reproduced the payload"
    if human_verified:
        return VERIFICATION_REPLAY_VERIFIED, "marked verified by a human reviewer"

    text = (evidence or "").strip()
    if not text:
        return VERIFICATION_PLAUSIBLE, "no evidence attached"
    if len(text) < _MIN_EVIDENCE_CHARS:
        return VERIFICATION_PLAUSIBLE, (
            f"evidence too short to assert an outcome ({len(text)} chars)"
        )

    haystack = text[:_SCAN_CHARS]
    cls = classify_vuln(vuln_type)

    if cls:
        matched = _first_match(haystack, _CONFIRM_MARKERS.get(cls, ()))
        if matched is not None:
            via = f" via {tool}" if tool else ""
            return VERIFICATION_CONFIRMED, (
                f"{cls} verdict present in evidence (/{matched.pattern}/){via}"
            )

    generic = _first_match(haystack, _GENERIC_CONFIRM_MARKERS)
    if generic is not None:
        via = f" via {tool}" if tool else ""
        return VERIFICATION_CONFIRMED, (
            f"tool reported a concrete result (/{generic.pattern}/){via}"
        )

    if cls:
        return VERIFICATION_PLAUSIBLE, (
            f"evidence attached but contains no {cls} verdict — treat as hypothesis"
        )
    return VERIFICATION_PLAUSIBLE, (
        "evidence attached but contains no recognisable tool verdict — treat as hypothesis"
    )
