"""
VAPT-AI KG — recon output parsers.

Regex-based parsers that extract canonical tech-stack tokens from the
output of common recon tools. The tokens are then mapped to KG
NodeType.TECHNOLOGY node names by the advisor layer.

Supported tools:
    - whatweb: "https://example.com [200 OK] [PHP, nginx, MySQL, jQuery]"
    - nmap:    "4280/tcp open http Apache httpd 2.4.41"
    - httpx:   "https://example.com [200] [cloudflare] [PHP]"

All tokens are returned lowercase + trimmed. Duplicate tokens are
removed (preserving first-seen order) so the caller can directly feed
the list into KG find_node() without deduplication.

The parsers are intentionally conservative — they prefer to return a
SHORT high-confidence list over a long noisy one. A false negative
(missing a tech) just means the KG has fewer stack nodes to consult;
a false positive (wrong tech) sends the KG down the wrong path.
"""
from __future__ import annotations

import re
from typing import Any

# ─── canonical token normalization ────────────────────────────────────
# Map common variants to a single canonical name. Keys are matched
# case-insensitively against raw tokens. Values are the canonical form
# that gets returned (and that should exist as a Technology node in the
# KG — see app/kg/seeder.py for the canonical tech list).
_TOKEN_ALIASES: dict[str, str] = {
    # web servers
    "apache": "apache",
    "apache-httpd": "apache",
    "httpd": "apache",
    "nginx": "nginx",
    "iis": "iis",
    "microsoft-iis": "iis",
    "lighttpd": "lighttpd",
    "caddy": "caddy",
    # languages / runtimes
    "php": "php",
    "php/7": "php",
    "php/8": "php",
    "asp.net": "asp.net",
    "aspnet": "asp.net",
    "jsp": "jsp",
    "java": "java",
    "tomcat": "tomcat",
    "node.js": "node.js",
    "nodejs": "node.js",
    "node": "node.js",
    "express": "express",
    "python": "python",
    "django": "django",
    "flask": "flask",
    "ruby": "ruby",
    "rails": "rails",
    "ruby-on-rails": "rails",
    # databases
    "mysql": "mysql",
    "mariadb": "mariadb",
    "postgresql": "postgresql",
    "postgres": "postgresql",
    "sqlite": "sqlite",
    "mongodb": "mongodb",
    "mongo": "mongodb",
    "redis": "redis",
    "mssql": "mssql",
    "microsoft-sql-server": "mssql",
    # CDNs / WAFs / proxies
    "cloudflare": "cloudflare",
    "akamai": "akamai",
    "fastly": "fastly",
    "incapsula": "incapsula",
    "sucuri": "sucuri",
    "aws": "aws",
    "cloudfront": "cloudfront",
    "varnish": "varnish",
    "haproxy": "haproxy",
    # CMS / frameworks
    "wordpress": "wordpress",
    "wp": "wordpress",
    "drupal": "drupal",
    "joomla": "joomla",
    "laravel": "laravel",
    "symfony": "symfony",
    "react": "react",
    "vue": "vue",
    "vue.js": "vue",
    "angular": "angular",
    "jquery": "jquery",
    # OS fingerprints
    "ubuntu": "ubuntu",
    "debian": "debian",
    "centos": "centos",
    "rhel": "rhel",
    "windows": "windows",
    "linux": "linux",
    # services
    "openssh": "openssh",
    "ssh": "openssh",
    "ftp": "ftp",
    "vsftpd": "vsftpd",
    "proftpd": "proftpd",
    "smb": "smb",
    "samba": "samba",
    "smtp": "smtp",
    "postfix": "postfix",
    "sendmail": "sendmail",
    "dns": "dns",
    "bind": "bind",
    "dnsmasq": "dnsmasq",
    "imap": "imap",
    "dovecot": "dovecot",
}


def _normalize_token(raw: str) -> str | None:
    """Lowercase + strip + alias-resolve a raw tech token.

    Returns None when the token doesn't match any known technology —
    this lets the parsers silently drop noise (HTTP status codes, IP
    addresses, response sizes, etc.) without polluting the tech list.
    """
    if not raw:
        return None
    tok = raw.strip().lower().rstrip(".,;:")
    if not tok:
        return None
    # Strip common prefixes like "v" before version (e.g. "v1.2.3")
    # but only if the rest looks like a known tech.
    if tok.startswith("v") and len(tok) > 1:
        stripped = tok[1:]
        if stripped in _TOKEN_ALIASES:
            return _TOKEN_ALIASES[stripped]
    return _TOKEN_ALIASES.get(tok)


def _dedup_preserve_order(tokens: list[str | None]) -> list[str]:
    """Drop None + remove duplicates, preserving first-seen order."""
    seen: set[str] = set()
    out: list[str] = []
    for t in tokens:
        if t is None or t in seen:
            continue
        seen.add(t)
        out.append(t)
    return out


# ─── whatweb parser ───────────────────────────────────────────────────
# whatweb output format:
#   https://example.com [200 OK] [PHP, nginx, MySQL, jQuery]
#   https://example.com [301 Moved Permanently] [Redirect,nginx]
#   https://example.com [200] [PHP, Apache, OpenSSL]
#
# The tech list is inside the LAST [bracketed] group on each line (the
# first bracket is the HTTP status, subsequent ones are plugin outputs).
_WHATWEB_TECH_RE = re.compile(
    r"\[([^\[\]]*)\]"  # match a [group] with no nested brackets
)


def parse_whatweb(output: str) -> list[str]:
    """Extract tech tokens from whatweb output.

    whatweb emits one line per URL, with tech names inside square
    brackets. We collect tech tokens from ALL bracket groups except
    the first (which is the HTTP status line).
    """
    if not output:
        return []
    tokens: list[str | None] = []
    for line in output.splitlines():
        if not line.strip():
            continue
        groups = _WHATWEB_TECH_RE.findall(line)
        if len(groups) <= 1:
            # Only the HTTP status group (or no groups at all) — no tech.
            continue
        # Tech groups are groups[1:] (skip the status group).
        for grp in groups[1:]:
            # Tech group is comma-separated.
            for raw in grp.split(","):
                tokens.append(_normalize_token(raw))
    return _dedup_preserve_order(tokens)


# ─── nmap parser ──────────────────────────────────────────────────────
# nmap output format (the part we care about):
#   PORT     STATE SERVICE  VERSION
#   4280/tcp open  http     Apache httpd 2.4.41 ((Ubuntu))
#   22/tcp   open  ssh      OpenSSH 8.2p1 Ubuntu 4ubuntu0.5
#   3306/tcp open  mysql    MySQL 8.0.27
#
# The 4th column (VERSION) is what we parse. We split it on whitespace
# and try to normalize each token. We don't use a regex here because
# nmap's table format is whitespace-aligned and a simple split is more
# robust to version-string quirks (parenthetical OS hints, etc.).


def parse_nmap(output: str) -> list[str]:
    """Extract tech tokens from nmap output.

    We only look at lines that match the standard nmap service-table
    format: "<port>/<proto> open <service> <version>". The version
    column is parsed for known tech tokens.
    """
    if not output:
        return []
    tokens: list[str | None] = []
    for line in output.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # Quick filter: must contain "/tcp" or "/udp" + "open"
        if ("/tcp" not in line and "/udp" not in line) or " open " not in line:
            continue
        parts = line.split(None, 4)  # split on whitespace, max 5 fields
        if len(parts) < 4:
            continue
        # parts[2] is "open", parts[3] is service name, parts[4] (if present) is version.
        service = parts[3]
        tokens.append(_normalize_token(service))
        if len(parts) >= 5:
            version_str = parts[4]
            # Strip trailing parentheses content like "((Ubuntu))".
            version_str = re.sub(r"\([^)]*\)", "", version_str).strip()
            for raw in version_str.split():
                tokens.append(_normalize_token(raw))
    return _dedup_preserve_order(tokens)


# ─── httpx parser ─────────────────────────────────────────────────────
# httpx (projectdiscovery) output format:
#   https://example.com [200] [title] [PHP, nginx] [cloudflare]
#   https://example.com:4280 [200] [Login Page] [PHP, MySQL]
#
# Tech tokens live in bracket groups after the URL + status code. We
# treat groups[0] as status (skip), groups[1] as title (skip — not a
# tech), groups[2:] as tech lists.
_HTTPX_BRACKET_RE = re.compile(r"\[([^\[\]]*)\]")


def parse_httpx(output: str) -> list[str]:
    """Extract tech tokens from httpx (projectdiscovery) output.

    httpx emits URL + bracket groups. The first bracket is the status
    code, the second is the page title, and any subsequent brackets are
    tech / CDN / WAF fingerprints.
    """
    if not output:
        return []
    tokens: list[str | None] = []
    for line in output.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        groups = _HTTPX_BRACKET_RE.findall(line)
        if len(groups) <= 2:
            # Only status + title (or fewer) — no tech info.
            continue
        # Tech groups are groups[2:] (skip status + title).
        for grp in groups[2:]:
            for raw in grp.split(","):
                tokens.append(_normalize_token(raw))
    return _dedup_preserve_order(tokens)


# ─── dispatcher ───────────────────────────────────────────────────────

# Map tool name → parser function. Tools not in this map return [].
_PARSERS: dict[str, Any] = {
    "whatweb": parse_whatweb,
    "nmap": parse_nmap,
    "httpx": parse_httpx,
}


def parse_recon_output(tool_name: str, output: str) -> list[str]:
    """Dispatch to the right parser based on tool name.

    Returns [] when the tool is not a recognized recon tool or when the
    output contains no parseable tech tokens.
    """
    parser = _PARSERS.get(tool_name.lower())
    if parser is None:
        return []
    try:
        return parser(output or "")
    except Exception:
        # Parser failures should never break the scan — return empty.
        return []


__all__ = [
    "parse_whatweb",
    "parse_nmap",
    "parse_httpx",
    "parse_recon_output",
]