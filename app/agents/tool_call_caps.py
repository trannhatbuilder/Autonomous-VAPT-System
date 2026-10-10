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
import re
from typing import Any

logger = logging.getLogger(__name__)


# ── Per-tool call caps (default values; tunable per scan via env in future) ──
# DISABLED (user decision, worklog task #16): the user has authorization to
# scan their targets and reported that per-tool caps (sqlmap=3, metasploit=5,
# etc.) were preventing the agent from completing a thorough pentest. All caps
# are now effectively unlimited. To re-enable, restore the dict below.
MAX_TOOL_CALLS_PER_SCAN: dict[str, int] = {}

# Generic fallback cap — used when a tool name isn't in MAX_TOOL_CALLS_PER_SCAN.
# DISABLED (worklog task #16): set to a very high number so unknown tools are
# effectively unlimited. To re-enable, restore DEFAULT_TOOL_CALL_CAP = 5.
DEFAULT_TOOL_CALL_CAP: int = 9999


# ── Per-tool STALL limits ──────────────────────────────────────────────────
# A "stall" = the tool was killed by its hard timeout, or cancelled mid-run.
# Retrying a stalling tool is the fastest way to burn a whole scan: a
# metasploit module that waited for a session burned 542s before the operator
# hit the panic button. After this many stalls the tool is BLOCKED for the rest
# of the scan and the agent is told to switch to a different tool.
#
# Exploitation tools get a limit of 1: a second attempt with different args is
# unlikely to be cheaper than switching (sqlmap <-> metasploit), and it protects
# the scan's wall-clock budget.
#
# RE-ENABLED (W22): a tool that hard-times-out will not succeed on the next
# attempt with the same conditions. This is NOT a "thoroughness" cap (which the
# worklog #16 decision disabled) — it is a stuck-tool guard, so it stays on.
MAX_TOOL_STALLS_PER_SCAN: dict[str, int] = {
    "metasploit": 1,   # the concrete incident: an msf module hung for 542s
    "sqlmap": 1,
    "hydra": 1,
}
# Generic fallback — one retry is tolerated, the 2nd stall blocks the tool.
DEFAULT_TOOL_STALL_LIMIT: int = 2


# ── Per-scan cap state ─────────────────────────────────────────────────────
# Keyed by scan_id so each scan has its own isolated counter dict.
# Each scan's state is a single dict: { tool_name -> call_count }.
_SCAN_CAPS: dict[str, dict[str, int]] = {}
# Per-scan stall counters: { scan_id -> { tool_name -> stall_count } }
_SCAN_STALLS: dict[str, dict[str, int]] = {}
# Per-scan invocation-failure counters: { scan_id -> { tool_name -> count } }
_SCAN_INVOCATION_FAILURES: dict[str, dict[str, int]] = {}


def _get_scan_state(scan_id: str) -> dict[str, int]:
    """Get or create the per-scan tool call counter dict."""
    if scan_id not in _SCAN_CAPS:
        _SCAN_CAPS[scan_id] = {}
    return _SCAN_CAPS[scan_id]


def _get_stall_state(scan_id: str) -> dict[str, int]:
    """Get or create the per-scan tool stall counter dict."""
    if scan_id not in _SCAN_STALLS:
        _SCAN_STALLS[scan_id] = {}
    return _SCAN_STALLS[scan_id]


def _resolve_stall_limit(tool_name: str) -> int:
    """Resolve the stall limit for a tool name."""
    return MAX_TOOL_STALLS_PER_SCAN.get(tool_name, DEFAULT_TOOL_STALL_LIMIT)


# Human/LLM-readable alternatives so the agent knows WHICH tool to switch to
# instead of re-trying the stalling one.
TOOL_ALTERNATIVES: dict[str, str] = {
    "metasploit": "sqlmap (if an injectable parameter exists), or report the "
                  "finding with the evidence already gathered",
    "sqlmap": "metasploit (a known-CVE module), or report the finding as "
              "unconfirmed with the evidence already gathered",
    "hydra": "netexec for credentialed checks, or report a weak-auth finding "
             "without brute-forcing",
    "impacket": "netexec, or switch to a read-only verification tool",
    "responder": "netexec for SMB/relay checks, or drop the poisoning attempt "
                 "and report the credential-exposure finding without capture",
    "hashcat": "john (same wordlist/rule set), or report the weak-hash finding "
               "without cracking it",
    "john": "hashcat (GPU cracking), or report the weak-hash finding without "
            "cracking it",
    "mimikatz": "it only works on a Windows host you already control — on any "
                "other target stop calling it and record the finding without "
                "credential dumping",
    "nmap": "rustscan / masscan for a faster sweep",
    "nuclei": "nikto / dalfox / wpscan for template-independent checks",
    "nikto": "nuclei with a focused tag set",
    "feroxbuster": "ffuf / gobuster with the bundled wordlist",
    "gobuster": "ffuf / feroxbuster",
    "ffuf": "gobuster / feroxbuster",
}


def mark_tool_stalled(scan_id: str, tool_name: str, reason: str = "") -> int:
    """Record that a tool was killed by its timeout / cancelled mid-run.

    Args:
        scan_id: scan ID
        tool_name: tool that stalled
        reason: short cause (for logging) — e.g. "hard_timeout", "cancelled"

    Returns:
        The updated stall count for (scan_id, tool_name).
    """
    state = _get_stall_state(scan_id)
    state[tool_name] = state.get(tool_name, 0) + 1
    limit = _resolve_stall_limit(tool_name)
    logger.warning(
        "TOOL_STALLED | scan=%s | tool=%s | stalls=%d/%d | reason=%s",
        scan_id, tool_name, state[tool_name], limit, reason or "?",
    )
    return state[tool_name]


def get_tool_stall_count(scan_id: str, tool_name: str) -> int:
    """Return how many times this tool has stalled in this scan."""
    return _SCAN_STALLS.get(scan_id, {}).get(tool_name, 0)


def is_tool_stalled(scan_id: str, tool_name: str) -> tuple[bool, str]:
    """Whether a tool has stalled too many times and must not be retried.

    Returns:
        (blocked, reason). reason is empty when blocked=False.
    """
    count = get_tool_stall_count(scan_id, tool_name)
    limit = _resolve_stall_limit(tool_name)
    if count == 0 or count < limit:
        return (False, "")

    alt = TOOL_ALTERNATIVES.get(tool_name, "another tool from your allowlist")
    reason = (
        f"TOOL_STALLED_SWITCH_REQUIRED: '{tool_name}' was killed by timeout "
        f"{count} times in this scan (limit={limit}) — it is consuming "
        f"wall-clock without returning results. STOP calling '{tool_name}'. "
        f"Switch to: {alt}. If you already have enough evidence, call record_vulnerability "
        f"with what you have collected and then call `exit` — an engagement "
        f"completed with an unconfirmed finding is still better than an infinite scan hang."
    )
    logger.warning(
        "TOOL_STALLED_SWITCH_REQUIRED | scan=%s | tool=%s | stalls=%d/%d",
        scan_id, tool_name, count, limit,
    )
    return (True, reason)


# ── Per-tool INVOCATION-FAILURE limits ─────────────────────────────────────
# An "invocation failure" = the tool rejected its own COMMAND LINE: unknown
# flag, unparsable value, missing required arg ("invalid value ... for flag
# -to"). Retrying with different VALUES cannot fix these — the flag MAPPING is
# wrong and the LLM has no reliable way to discover the right one. After N
# invocation failures the tool is BLOCKED for the rest of the scan.
#
# This is deliberately NOT disabled by worklog task #16: that decision removed
# "thoroughness" caps (how many times a WORKING tool may run). A tool that
# refuses to start is broken, not overused — so it stays capped.
MAX_TOOL_INVOCATION_FAILURES: dict[str, int] = {}
DEFAULT_TOOL_INVOCATION_FAILURE_LIMIT: int = 2

# Patterns that mark a COMMAND-LINE rejection (not a runtime/target error).
# Checked against the HEAD of the tool output, where arg parsers print their
# complaints.
#
# These are REGEX, not raw substrings, on purpose: a bare "invalid value"
# shows up in plenty of TARGET error pages, but `invalid value ... for flag`
# / `invalid value ... for --x` is unmistakably a CLI-parser complaint. Keeping
# the value-case anchored lets us still catch gobuster/katana/feroxbuster bad
# VALUES without false-positiving on target content.
INVOCATION_ERROR_PATTERNS: tuple[str, ...] = (
    r"incorrect usage",                      # gobuster (cobra)
    r"flag provided but not defined",          # Go pflag — gobuster, katana, httpx
    r"unknown shorthand flag",                 # Go pflag
    r"unknown flag",                           # misc Go/C
    r"unexpected argument",                    # feroxbuster (clap)
    r"unrecognized argument",                  # argparse — "unrecognized arguments:"
    r"no such option",                         # getopt-style
    r"invalid option",                         # getopt — "invalid option -- 'x'"
    r"invalid value\b.*\bfor\s+(?:flag|--)",   # bad VALUE: gobuster/katana/feroxbuster
)
_INVOCATION_ERROR_RES = tuple(re.compile(p) for p in INVOCATION_ERROR_PATTERNS)

# How much of the output head to scan (arg parsers print usage errors up front).
_INVOCATION_ERROR_SCAN_CHARS = 800


# Human/LLM-readable alternatives to suggest when a tool is blocked as broken.
INVOCATION_ERROR_ALTERNATIVES: dict[str, str] = {
    "gobuster": "ffuf or feroxbuster for directory brute-force",
    "feroxbuster": "ffuf or gobuster",
    "ffuf": "gobuster or feroxbuster",
    "httpx": "whatweb for fingerprinting",
    "whatweb": "httpx -tech-detect",
    "katana": "gau / waybackurls for URL discovery",
    "nikto": "nuclei with misconfig tags",
    "nuclei": "nikto",
    "nmap": "rustscan or masscan",
}


def looks_like_invocation_error(text: str) -> bool:
    """True if `text` looks like a command-line rejection from a tool.

    Only the first _INVOCATION_ERROR_SCAN_CHARS characters are inspected —
    arg parsers print usage errors up front, so this keeps the check cheap and
    (together with the anchored regexes) keeps TARGET-supplied text from
    matching.
    """
    if not text:
        return False
    head = text[:_INVOCATION_ERROR_SCAN_CHARS].lower()
    return any(r.search(head) for r in _INVOCATION_ERROR_RES)


def _get_invocation_failure_state(scan_id: str) -> dict[str, int]:
    """Get or create the per-scan invocation-failure counter dict."""
    if scan_id not in _SCAN_INVOCATION_FAILURES:
        _SCAN_INVOCATION_FAILURES[scan_id] = {}
    return _SCAN_INVOCATION_FAILURES[scan_id]


def _resolve_invocation_failure_limit(tool_name: str) -> int:
    """Resolve the invocation-failure limit for a tool name."""
    return MAX_TOOL_INVOCATION_FAILURES.get(
        tool_name, DEFAULT_TOOL_INVOCATION_FAILURE_LIMIT
    )


def mark_tool_invocation_failure(scan_id: str, tool_name: str) -> int:
    """Record that a tool call failed at the invocation level (bad flags/args).

    Returns the updated failure count for (scan_id, tool_name).
    """
    state = _get_invocation_failure_state(scan_id)
    state[tool_name] = state.get(tool_name, 0) + 1
    limit = _resolve_invocation_failure_limit(tool_name)
    logger.warning(
        "TOOL_INVOCATION_FAILURE | scan=%s | tool=%s | failures=%d/%d",
        scan_id, tool_name, state[tool_name], limit,
    )
    return state[tool_name]


def get_tool_invocation_failure_count(scan_id: str, tool_name: str) -> int:
    """Return how many invocation failures this tool has in this scan."""
    return _SCAN_INVOCATION_FAILURES.get(scan_id, {}).get(tool_name, 0)


def is_tool_broken(scan_id: str, tool_name: str) -> tuple[bool, str]:
    """Whether a tool has failed at invocation level too many times.

    Returns (blocked, reason). reason is empty when blocked=False.
    """
    count = get_tool_invocation_failure_count(scan_id, tool_name)
    limit = _resolve_invocation_failure_limit(tool_name)
    if count < limit:
        return (False, "")

    alt = INVOCATION_ERROR_ALTERNATIVES.get(
        tool_name, "a different tool from your allowlist"
    )
    reason = (
        f"TOOL_BROKEN_SWITCH_REQUIRED: '{tool_name}' failed with "
        f"command-line/invocation errors {count} times in this scan "
        f"(limit={limit}). The flag mapping is wrong — retrying with different "
        f"VALUES will NOT help (the tool rejects the syntax itself, not the "
        f"target). Do NOT call '{tool_name}' again this scan. Switch to: {alt}. "
        f"If you already have enough evidence, call record_vulnerability and then `exit`."
    )
    logger.warning(
        "TOOL_BROKEN_SWITCH_REQUIRED | scan=%s | tool=%s | failures=%d/%d",
        scan_id, tool_name, count, limit,
    )
    return (True, reason)


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
        (False, "Tool 'nmap' was called 6 times (cap=5). Switch tool or
                 summarize findings and exit.")
    """
    state = _get_scan_state(scan_id)
    count = state.get(tool_name, 0)
    cap = _resolve_cap(tool_name)
    if count >= cap:
        # Cap exhausted — return denied with a clear, actionable reason.
        reason = (
            f"TOOL_CALL_CAP_EXHAUSTED: Tool '{tool_name}' was called {count} times "
            f"(cap={cap}). Switch to a different tool or summarize findings "
            f"and exit. Calling '{tool_name}' again will not yield new "
            f"information — the tool call limit for this scan has been reached."
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
    """Drop the per-scan cap + stall state. Called at scan end (cleanup phase).

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
    _SCAN_STALLS.pop(scan_id, None)
    _SCAN_INVOCATION_FAILURES.pop(scan_id, None)


def get_tool_stall_stats(scan_id: str) -> dict[str, int]:
    """Return per-tool stall counts for a scan (copy)."""
    return dict(_SCAN_STALLS.get(scan_id, {}))


__all__ = [
    "MAX_TOOL_CALLS_PER_SCAN",
    "DEFAULT_TOOL_CALL_CAP",
    "MAX_TOOL_STALLS_PER_SCAN",
    "DEFAULT_TOOL_STALL_LIMIT",
    "MAX_TOOL_INVOCATION_FAILURES",
    "DEFAULT_TOOL_INVOCATION_FAILURE_LIMIT",
    "INVOCATION_ERROR_PATTERNS",
    "INVOCATION_ERROR_ALTERNATIVES",
    "looks_like_invocation_error",
    "TOOL_ALTERNATIVES",
    "check_tool_call_cap",
    "mark_tool_stalled",
    "get_tool_stall_count",
    "is_tool_stalled",
    "mark_tool_invocation_failure",
    "get_tool_invocation_failure_count",
    "is_tool_broken",
    "get_tool_call_stats",
    "get_tool_stall_stats",
    "clear_scan_caps",
]