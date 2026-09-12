"""
VAPT-AI Scope Guard — application-level scope enforcement (NO iptables, D11).

Ported from EVVO Sentinel's shield_engine/scope_guard.py (135 LOC) +
shield_engine/_agent_tools.py (288 LOC).

3 layers of protection:

Layer 1: Destructive command blocklist
    Regex patterns that block dangerous commands regardless of target:
    - rm -rf, DROP TABLE, fork bombs, mkfs, dd if=, shutdown, reboot
    - High-risk tools: masscan, zmap, hping3 --flood, slowloris, LOIC

Layer 2: Binary allowlist
    Only YAML-defined tools can run (CyberStrikeAI parity).
    21-binary allowlist for direct shell execution by specialist agents.

Layer 3: Target scope validation
    Before every subprocess call, validate target IP/URL against the scan's
    declared scope set (hosts, IP ranges, ports).
    - Hostname glob matching (*.example.com)
    - ipaddress.ip_network containment (10.0.0.0/24 contains 10.0.0.5)
    - URL extraction + host validation
    - SSRF guard: block private/loopback/link-local/cloud-metadata IPs

Usage:
    from app.sandbox.scope_guard import ScopeGuard, ScopeRule

    guard = ScopeGuard(declared_scope=[
        ScopeRule(host="example.com"),
        ScopeRule(host="*.example.com"),
        ScopeRule(cidr="192.168.1.0/24"),
        ScopeRule(host="10.10.10.5", port=80),
    ])

    # Check before running a command
    result = guard.validate_command("nmap -sS 192.168.1.5", target="192.168.1.5")
    if not result.allowed:
        raise ScopeViolation(result.reason)

    # Check target is in scope
    result = guard.validate_target("192.168.1.5")
    if not result.allowed:
        raise ScopeViolation(result.reason)
"""
from __future__ import annotations

import ipaddress
import logging
import re
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


# ---------- Layer 1: Destructive command blocklist ----------

# Regex patterns for destructive commands — blocked regardless of target
DESTRUCTIVE_CMD_PATTERNS: list[re.Pattern[str]] = [
    # Filesystem destruction
    re.compile(r"\brm\s+-rf\s+/(?:\s|$)"),               # rm -rf /
    re.compile(r"\brm\s+-rf\s+~"),                        # rm -rf ~
    re.compile(r"\brm\s+-rf\s+\*"),                       # rm -rf *
    re.compile(r"\bmkfs\b"),                               # mkfs
    re.compile(r"\bdd\s+if=\S+\s+of=/dev/[sh]d"),         # dd if=X of=/dev/sdX
    re.compile(r">\s*/dev/[sh]d"),                         # > /dev/sdX

    # Database destruction
    re.compile(r"\bDROP\s+(?:TABLE|DATABASE|SCHEMA)\b", re.IGNORECASE),
    re.compile(r"\bTRUNCATE\s+TABLE\b", re.IGNORECASE),
    re.compile(r"\bDELETE\s+FROM\s+\w+\s*;\s*$", re.IGNORECASE),  # DELETE without WHERE

    # Fork bombs
    re.compile(r":\(\)\s*\{\s*:\|:\s*&\s*\}\s*;:"),       # :(){ :|:& };:
    re.compile(r"\bfork\s+bomb\b", re.IGNORECASE),

    # System shutdown/reboot
    re.compile(r"\bshutdown\b", re.IGNORECASE),
    re.compile(r"\breboot\b", re.IGNORECASE),
    re.compile(r"\bhalt\b", re.IGNORECASE),
    re.compile(r"\bpoweroff\b", re.IGNORECASE),

    # Privilege escalation attempts (not via sudo)
    re.compile(r"\bsu\s+root\b"),
    re.compile(r"\bchown\s+-R\s+root\b"),
]

# High-risk tools — blocked entirely (too dangerous for automated use)
HIGH_RISK_TOOLS: set[str] = {
    "masscan",       # can saturate network
    "zmap",          # can saturate network
    "hping3",        # if used with --flood
    "slowloris",     # DoS tool
    "LOIC",          # DoS tool
    "hydra",         # brute-force (HITL required — W6)
}

# Check for hping3 --flood specifically
HPING_FLOOD_PATTERN = re.compile(r"hping3.*--flood", re.IGNORECASE)


# ---------- Layer 2: Binary allowlist ----------

# Tools that can be run directly via shell by specialist agents
# (CyberStrikeAI parity — from internal/security/executor.go)
BINARY_ALLOWLIST: set[str] = {
    # Security tools (W2-A core — 10)
    "nmap", "nuclei", "sqlmap", "nikto", "whatweb",
    "gobuster", "feroxbuster", "ffuf", "subfinder", "httpx",
    "dalfox", "wpscan", "wafw00f", "arjun", "dirsearch",
    # Security tools (W5-B extended — 22)
    "msfconsole", "msfvenom", "msfrpcd",          # Metasploit
    "impacket-smbexec", "impacket-psexec", "impacket-wmiexec",
    "impacket-secretsdump", "impacket-GetUserSPNs", # Impacket
    "netexec",                                      # NetExec
    "responder",                                    # Responder
    "hashcat", "john", "hydra",                     # Password cracking
    "linpeas", "winpeas.exe", "mimikatz.exe",       # Post-exploit
    "masscan", "rustscan", "fscan",                 # Network scanning
    "katana", "gau", "waybackurls",                 # Web fuzzing (ffuf already in core)
    "amass", "dnsenum", "fierce",                   # DNS/Subdomain
    "theHarvester",                                 # OSINT (note: capital H)
    # Network utilities
    "curl", "dig", "openssl", "timeout",
    # Safe read-only utilities (for agent shell commands)
    "echo", "cat", "ls", "grep", "find", "head", "tail", "sleep",
    "wc", "sort", "uniq", "cut", "tr", "base64", "xxd",
    # Safe write utilities (for evidence/cleanup)
    "mkdir", "cp", "mv", "rm", "touch", "tee", "sed", "awk",
}

# Tools defined in YAML (app/tools/*.yaml) — loaded dynamically
YAML_TOOLS: set[str] = set()  # populated by loader


# ---------- Layer 3: SSRF protection ----------

# IP ranges blocked by SSRF guard (prevents agent from attacking internal infra)
SSRF_BLOCKED_RANGES: list[ipaddress.IPv4Network] = [
    ipaddress.ip_network("127.0.0.0/8"),        # loopback
    ipaddress.ip_network("10.0.0.0/8"),         # private
    ipaddress.ip_network("172.16.0.0/12"),      # private
    ipaddress.ip_network("192.168.0.0/16"),     # private
    ipaddress.ip_network("169.254.0.0/16"),     # link-local + cloud metadata
    ipaddress.ip_network("0.0.0.0/8"),          # "this" network
    ipaddress.ip_network("100.64.0.0/10"),      # CGNAT
    ipaddress.ip_network("::1/128"),            # IPv6 loopback
    ipaddress.ip_network("fc00::/7"),           # IPv6 ULA
    ipaddress.ip_network("fe80::/10"),          # IPv6 link-local
]


# ---------- Scope rules ----------

@dataclass
class ScopeRule:
    """A single scope rule — defines what targets are authorized."""
    host: str | None = None       # hostname (supports *.example.com glob)
    cidr: str | None = None       # CIDR notation (e.g. "192.168.1.0/24")
    port: int | None = None       # specific port (None = any port)
    ip_network: ipaddress.IPv4Network | ipaddress.IPv6Network | None = field(default=None, init=False)

    def __post_init__(self):
        if self.cidr:
            try:
                self.ip_network = ipaddress.ip_network(self.cidr, strict=False)
            except ValueError as e:
                raise ValueError(f"Invalid CIDR '{self.cidr}': {e}") from e


# ---------- Validation result ----------

@dataclass
class ValidationResult:
    """Result of scope validation."""
    allowed: bool
    reason: str = ""
    severity: str = "info"  # info, warning, error
    target: str = ""
    command: str = ""


# ---------- Scope Guard ----------

class ScopeGuard:
    """Application-level scope guard — validates commands + targets before execution.

    NO iptables, NO kernel-level firewall. Pure Python validation.
    """

    def __init__(
        self,
        declared_scope: list[ScopeRule] | None = None,
        enforce_ssrf: bool = True,
        block_destructive: bool = True,
    ):
        self.declared_scope = declared_scope or []
        self.enforce_ssrf = enforce_ssrf
        self.block_destructive = block_destructive

    # ---------- Layer 1: Destructive command check ----------

    def check_destructive_command(self, command: str) -> ValidationResult:
        """Check if command matches destructive patterns."""
        if not self.block_destructive:
            return ValidationResult(allowed=True, command=command)

        # Check destructive patterns
        for pattern in DESTRUCTIVE_CMD_PATTERNS:
            if pattern.search(command):
                return ValidationResult(
                    allowed=False,
                    reason=f"Destructive command pattern matched: {pattern.pattern}",
                    severity="error",
                    command=command,
                )

        # Check high-risk tools
        for tool in HIGH_RISK_TOOLS:
            if re.search(rf"\b{re.escape(tool)}\b", command, re.IGNORECASE):
                return ValidationResult(
                    allowed=False,
                    reason=f"High-risk tool blocked: {tool}",
                    severity="error",
                    command=command,
                )

        # Check hping3 --flood
        if HPING_FLOOD_PATTERN.search(command):
            return ValidationResult(
                allowed=False,
                reason="hping3 --flood is blocked (DoS)",
                severity="error",
                command=command,
            )

        return ValidationResult(allowed=True, command=command)

    # ---------- Layer 2: Binary allowlist check ----------

    def check_binary_allowed(self, binary: str) -> ValidationResult:
        """Check if binary is in allowlist."""
        # Check YAML tools first (dynamically loaded)
        all_allowed = BINARY_ALLOWLIST | YAML_TOOLS
        if binary in all_allowed:
            return ValidationResult(allowed=True, target=binary)

        return ValidationResult(
            allowed=False,
            reason=f"Binary '{binary}' not in allowlist (allowed: {sorted(all_allowed)[:10]}...)",
            severity="error",
            target=binary,
        )

    # ---------- Layer 3: Target scope validation ----------

    def validate_target(self, target: str) -> ValidationResult:
        """Validate that target (IP/hostname/URL) is in declared scope.

        Order: declared scope check FIRST, then SSRF (only if NOT in scope).
        This allows pentest of internal IPs (192.168.x.x, 10.x.x.x) when
        explicitly declared in scope. SSRF only blocks UNTRUSTED targets
        that aren't in the declared scope.
        """
        if not target:
            return ValidationResult(allowed=True, target=target, reason="No target specified")

        # Extract host from URL if needed
        host = self._extract_host(target)
        if not host:
            return ValidationResult(
                allowed=False,
                reason=f"Could not extract host from target: {target}",
                severity="error",
                target=target,
            )

        # Check if any scope rule matches FIRST
        for rule in self.declared_scope:
            if self._match_rule(rule, host):
                return ValidationResult(allowed=True, target=target, reason=f"Matched scope rule: {rule}")

        # Not in declared scope — run SSRF check (blocks private/loopback/metadata)
        if self.enforce_ssrf:
            ssrf_result = self._check_ssrf(host)
            if not ssrf_result.allowed:
                return ssrf_result

        # No scope rule matched + not SSRF blocked
        return ValidationResult(
            allowed=False,
            reason=f"Target '{host}' not in declared scope ({len(self.declared_scope)} rules)",
            severity="error",
            target=target,
        )

    def validate_command(self, command: str, target: str = "") -> ValidationResult:
        """Full validation: destructive check + binary check + target check."""
        # Layer 1: destructive command
        destructive_result = self.check_destructive_command(command)
        if not destructive_result.allowed:
            return destructive_result

        # Layer 2: binary allowlist
        binary = self._extract_binary(command)
        if binary:
            binary_result = self.check_binary_allowed(binary)
            if not binary_result.allowed:
                return binary_result

        # Layer 3: target scope
        if target:
            target_result = self.validate_target(target)
            if not target_result.allowed:
                return target_result

        return ValidationResult(allowed=True, command=command, target=target)

    # ---------- Internal helpers ----------

    def _extract_host(self, target: str) -> str | None:
        """Extract hostname/IP from target string (URL or raw host)."""
        if not target:
            return None

        # Try parsing as URL
        if "://" in target:
            try:
                parsed = urlparse(target)
                return parsed.hostname
            except Exception:
                pass

        # Try as raw IP
        try:
            ip = ipaddress.ip_address(target)
            return str(ip)
        except ValueError:
            pass

        # Treat as hostname
        return target

    def _check_ssrf(self, host: str) -> ValidationResult:
        """SSRF check — block private/loopback/metadata IPs."""
        try:
            ip = ipaddress.ip_address(host)
        except ValueError:
            # Not an IP — hostname. Resolve is deferred to executor (W3-D).
            # For W3-C, allow hostnames (SSRF check happens at connection time)
            return ValidationResult(allowed=True, target=host)

        for network in SSRF_BLOCKED_RANGES:
            if ip in network:
                return ValidationResult(
                    allowed=False,
                    reason=f"SSRF blocked: {host} is in {network} (private/loopback/metadata)",
                    severity="error",
                    target=host,
                )

        return ValidationResult(allowed=True, target=host)

    def _match_rule(self, rule: ScopeRule, host: str) -> bool:
        """Check if a scope rule matches a host."""
        # CIDR match
        if rule.ip_network:
            try:
                ip = ipaddress.ip_address(host)
                if ip in rule.ip_network:
                    return True
            except ValueError:
                pass  # host is not an IP

        # Hostname glob match
        if rule.host:
            if self._glob_match(rule.host, host):
                return True

        return False

    def _glob_match(self, pattern: str, host: str) -> bool:
        """Glob match — supports *.example.com syntax."""
        if pattern == host:
            return True

        if pattern.startswith("*."):
            suffix = pattern[1:]  # .example.com
            return host.endswith(suffix) or host == pattern[2:]

        return False

    def _extract_binary(self, command: str) -> str | None:
        """Extract the binary name from a command string."""
        if not command:
            return None
        # First token (after any env vars / sudo)
        parts = command.strip().split()
        for part in parts:
            if "=" in part:
                continue  # skip env vars (FOO=bar)
            if part in ("sudo", "timeout", "nohup", "env"):
                continue  # skip prefixes
            # Extract basename (e.g. /usr/bin/nmap → nmap)
            return part.rsplit("/", 1)[-1]
        return None


# ---------- Scope violation exception ----------

class ScopeViolation(Exception):
    """Raised when a command or target violates scope rules."""

    def __init__(self, reason: str, severity: str = "error"):
        self.reason = reason
        self.severity = severity
        super().__init__(reason)


# ---------- Default scope guard (no rules — blocks everything) ----------

def get_default_scope_guard() -> ScopeGuard:
    """Get a default scope guard with no declared scope.

    This blocks ALL targets (nothing in scope). Use for testing.
    Production code should create ScopeGuard with actual declared_scope.
    """
    return ScopeGuard(declared_scope=[], enforce_ssrf=True, block_destructive=True)


# ---------- Register YAML tools (called by loader at startup) ----------

def register_yaml_tools(tool_names: set[str]) -> None:
    """Register YAML-defined tools so scope guard allows them.

    Called by app.tools.loader.load_all_tools() at startup.
    """
    YAML_TOOLS.update(tool_names)
    logger.info("Registered %d YAML tools with scope guard: %s",
                len(tool_names), sorted(YAML_TOOLS))
