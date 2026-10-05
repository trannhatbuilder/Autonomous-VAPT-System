"""
VAPT-AI Tool Call Caps — Phase F1 (W19-FIX3-FG-BACKEND).

Hard per-scan cap on how many times each tool may be invoked. This is the
behavioral guardrail against the "agent calls nmap → httpx → nmap → httpx →
nmap → ... (9 times) then stops" loop reported by the user.

Design (mirrors `app/agents/tool_call_cache.py` patterns):

    - State is per-scan, in-memory, keyed by scan_id.
    - Caps are tool-specific — `MAX_TOOL_CALLS_PER_SCAN` defaults are tuned
      per tool's role (nmap=5, nuclei=3, sqlmap=3, httpx=2, whatweb=1, ...).
      Generic fallback cap = 5.
    - `check_tool_call_cap(scan_id, tool_name)` is called from
      `tool_bridge.execute_tool_call()` AFTER the result cache lookups (so
      cached hits — same args — do NOT burn the cap budget) but BEFORE the
      subprocess spawn (so a denied call never reaches the executor).
    - When allowed → counter is incremented and (True, "") is returned.
    - When denied (count >= cap) → (False, reason) returned. The reason is a
      human/LLM-readable instruction: switch tool or summarize and exit.
    - Cache is cleared by `clear_scan_caps(scan_id)` at scan end
      (wired into `_phase_cleanup` in scan_pipeline.py).

Why per-tool caps (not a single global cap):

    A single global cap (e.g. "max 30 tool calls per scan") does NOT prevent
    the pathological case where the LLM hammers ONE tool 9 times in a row
    and exits because it ran out of total calls. Per-tool caps force the
    agent to make each call count and switch tools when it cannot get more
    info out of one.

Why cap values are what they are:

    - nmap=5: initial fast scan + up to 4 targeted re-scans (per port,
      per host in a /24, per NSE script group). Re-scans are useful; 9 is
      the reported runaway count — 5 leaves headroom for legit work.
    - nuclei=3: severity high+critical (default), then optionally medium,
      then a specific tag set if needed. Anything more is redundant noise.
    - sqlmap=3: per-endpoint calls (different endpoints = different tests).
      Same endpoint 4+ times means stuck.
    - httpx=2: initial probe + a re-probe with -follow-redirects if needed.
    - whatweb=1: single fingerprint call per host — re-calling is silly.
    - gobuster/feroxbuster/ffuf=2: medium wordlist first, large second.
    - subfinder/amass=1: passive enum — single call.
    - nikto=1: slow + noisy — single call per host.
    - metasploit=5: different exploit modules — reasonable for kill-chain.
    - mimikatz=1: LSASS dump is one-shot per host.
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


# ── Per-tool call caps (default values; tunable per scan via env in future) ──
MAX_TOOL_CALLS_PER_SCAN: dict[str, int] = {
    # Network scanners
    "nmap": 5,           # initial fast scan + up to 4 targeted re-scans
    "masscan": 2,        # 1 for full-range port sweep + 1 for targeted re-sweep
    "rustscan": 1,       # fast port discovery — single pass
    "fscan": 2,          # internal network sweep + targeted re-sweep
    "netexec": 3,        # SMB/WinRM probe + 2 targeted re-runs

    # Web fingerprinters
    "httpx": 2,          # initial probe + 1 re-probe (e.g. with -follow-redirects)
    "whatweb": 1,        # single fingerprint call per host
    "nikto": 1,          # slow + noisy — single pass

    # Vuln scanners
    "nuclei": 3,         # severity high+critical, then optionally medium, then tag-specific
    "dalfox": 2,         # 1 quick scan + 1 with custom payloads
    "wpscan": 1,         # single pass — WordPress enumeration
    "fscan": 2,

    # Directory brutes
    "gobuster": 2,       # medium wordlist first, large second
    "feroxbuster": 2,    # medium wordlist first, large second
    "ffuf": 2,           # medium wordlist first, large second

    # Subdomain enum
    "subfinder": 1,      # passive — single call
    "amass": 1,          # passive — single call
    "theharvester": 1,   # passive — single call

    # SQLi
    "sqlmap": 3,         # per-endpoint calls (different endpoints)

    # Exploitation
    "metasploit": 5,     # different exploit modules
    "impacket": 3,       # different lateral-movement modules

    # Creds / privesc
    "hydra": 2,          # different protocols
    "hashcat": 1,        # offline — single call (different rule sets handled in 1 call)
    "john": 1,           # offline — single call
    "mimikatz": 1,       # LSASS dump — one-shot per host
    "linpeas": 1,        # single privesc enum
    "winpeas": 1,        # single privesc enum

    # Misc / recon
    "gau": 1,            # single fetch from archive APIs
    "waybackurls": 1,    # single fetch
    "katana": 2,         # 1 passive + 1 active crawl
    "dnsenum": 1,        # single DNS enum
    "fierce": 1,         # single DNS enum
    "responder": 1,      # LLMNR poisoner — single capture session
}

# Generic fallback cap — used when a tool name isn't in MAX_TOOL_CALLS_PER_SCAN.
# Set to 5 (moderate) so unknown tools have reasonable headroom but cannot
# dominate the scan.
DEFAULT_TOOL_CALL_CAP: int = 5


# ── Per-scan cap state ─────────────────────────────────────────────────────
# Keyed by scan_id so each scan has its own isolated counter dict.
# Each scan's state is a single dict: { tool_name -> call_count }.
_SCAN_CAPS: dict[str, dict[str, int]] = {}


def _get_scan_state(scan_id: str) -> dict[str, int]:
    """Get or create the per-scan tool call counter dict."""
    if scan_id not in _SCAN_CAPS:
        _SCAN_CAPS[scan_id] = {}
    return _SCAN_CAPS[scan_id]


def _resolve_cap(tool_name: str) -> int:
    """Resolve the cap for a tool name (falls back to DEFAULT_TOOL_CALL_CAP)."""
    return MAX_TOOL_CALLS_PER_SCAN.get(tool_name, DEFAULT_TOOL_CALL_CAP)


def check_tool_call_cap(scan_id: str, tool_name: str) -> tuple[bool, str]:
    """Check whether a tool may still be called in this scan.

    Side-effect: when the call is ALLOWED, the per-(scan_id, tool_name)
    counter is INCREMENTED by this function. Callers must place this check
    AFTER any cache hits (so cached results do not burn the cap budget)
    but BEFORE the actual subprocess spawn (so a denied call never reaches
    the executor).

    Args:
        scan_id: scan ID (string, e.g. "scan_abc123")
        tool_name: tool name (e.g. "nmap", "nuclei")

    Returns:
        Tuple (allowed, reason):
            - allowed=True, reason="" → caller may proceed with the call.
            - allowed=False, reason=<msg> → caller must NOT call the tool;
              the reason string is suitable for direct LLM consumption
              (explains why + what to do instead).

    Reason format mirrors the cache-miss warning strings already produced
    by tool_bridge (e.g. DUPLICATE_TOOL_CALL_WARNING) so the agent's
    prompt-engineered handling of "switch tool or summarize and exit"
    applies uniformly.

    Examples:
        >>> check_tool_call_cap("scan_1", "nmap")
        (True, "")
        >>> check_tool_call_cap("scan_1", "nmap")  # call 6
        (False, "Tool 'nmap' đã gọi 6 lần (cap=5). Switch tool hoặc
                 summarize findings và exit.")
    """
    state = _get_scan_state(scan_id)
    count = state.get(tool_name, 0)
    cap = _resolve_cap(tool_name)
    if count >= cap:
        # Cap exhausted — return denied with a clear, actionable reason.
        reason = (
            f"TOOL_CALL_CAP_EXHAUSTED: Tool '{tool_name}' đã gọi {count} lần "
            f"(cap={cap}). Switch sang tool khác hoặc summarize findings "
            f"và exit. Calling lại '{tool_name}' sẽ không yield thêm thông tin "
            f"mới — đã đạt giới hạn gọi công cụ cho scan này."
        )
        logger.warning(
            "TOOL_CALL_CAP_EXHAUSTED | scan=%s | tool=%s | count=%d | cap=%d",
            scan_id, tool_name, count, cap,
        )
        return (False, reason)

    # Allowed — increment + return green light.
    new_count = count + 1
    state[tool_name] = new_count
    logger.info(
        "TOOL_CALL_CAP_TICK | scan=%s | tool=%s | count=%d/%d | allowed",
        scan_id, tool_name, new_count, cap,
    )
    return (True, "")


def get_tool_call_stats(scan_id: str) -> dict[str, int]:
    """Return per-tool call counts for a scan.

    Useful for debugging + the audit trail (Phase 7 PDF / scan summary).
    Returns a copy so callers cannot mutate internal state.

    Args:
        scan_id: scan ID

    Returns:
        Dict mapping tool_name -> call_count. Empty dict if scan has no
        recorded calls (e.g. scan_id not registered).
    """
    state = _SCAN_CAPS.get(scan_id, {})
    return dict(state)


def clear_scan_caps(scan_id: str) -> None:
    """Drop the per-scan cap state. Called at scan end (cleanup phase).

    Mirrors `tool_call_cache.clear_scan_cache(scan_id)`. Safe to call
    multiple times — no-op if scan_id not in registry.
    """
    if scan_id in _SCAN_CAPS:
        state = _SCAN_CAPS[scan_id]
        total_calls = sum(state.values())
        logger.info(
            "Tool call caps cleared | scan=%s | tools_tracked=%d | total_calls=%d",
            scan_id, len(state), total_calls,
        )
        del _SCAN_CAPS[scan_id]


__all__ = [
    "MAX_TOOL_CALLS_PER_SCAN",
    "DEFAULT_TOOL_CALL_CAP",
    "check_tool_call_cap",
    "get_tool_call_stats",
    "clear_scan_caps",
]
