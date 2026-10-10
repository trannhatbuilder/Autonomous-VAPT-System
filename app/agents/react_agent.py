"""
VAPT-AI Real ReAct Agent — LLM + Tools + SSE events.

This is the core scan engine. Replaces the W9 stub with a real ReAct loop
that mirrors CyberStrikeAI's agent flow:

    user message → LLM (with tools) → tool_calls → execute tools →
    tool_results → LLM → ... → exit tool → done

Each iteration emits SSE events (scan_progress) so the frontend chat
shows real-time AI thinking + tool calls + results.

Architecture:
    1. Build system prompt (structured 5-phase pentest workflow)
    2. Build tool schemas (32 security tools + record_vulnerability + exit)
    3. P1 Preflight: 1-shot LLM ping to verify OpenAI API key works
       (fail fast instead of cryptic failure mid-loop)
    4. Loop:
        a. Call LLM via chat_completion with messages + tools
        b. If LLM returns tool_calls:
            - Emit SSE: scan_progress (thought + tool_name)
            - Execute each tool via SubprocessExecutor
            - Hard-block tools that already failed N times (retry guard)
            - Emit SSE: scan_progress (observation)
            - Append tool results to messages
            - Prune old tool results (context-window management)
            - Continue loop
        c. If LLM returns content only (no tool_calls):
            - LLM is done thinking — append assistant message
            - Continue (LLM may call exit tool next)
        d. If LLM calls "exit" tool:
            - Scan complete — return summary
        e. Max iterations reached:
            - Return what we have

W19-FIX5 Phase I (EVVO parity):
    - MAX_AGENT_TURNS = 100
    - Token budget tracking: soft warning + hard cancel (ENFORCED in loop)
    - Phase tracking is MONOTONIC (1→2→3→4→5, never backwards)

Post-W19 fixes (scan_e943a1ec / scan_6c9afacc / scan_1942c8e5 findings):
    - FIX-A: removed phantom 'execute'/curl tool from system prompt (LLM
      called non-existent tool 3x per scan)
    - FIX-B: per-tool retry guard — a tool that fails twice is BLOCKED
      with an explicit message (invocation errors cannot be fixed by
      retrying with different values)
    - FIX-C: progress cap — was 20 + iteration*2 = up to 218/100
    - FIX-D: token budget soft/hard limits ENFORCED (docstring previously
      promised this without code)
    - FIX-E: malformed tool-call JSON surfaced to the LLM (was silently
      replaced with {} — agent never knew its args were broken)
    - FIX-F: defensive usage parsing (response["usage"] may be None)
    - FIX-G: context pruning — old tool results truncated, full outputs
      preserved via spill files (see tool_bridge)
    - FIX-H: WAF/empty-output rules in system prompt — agent must NOT
      conclude "secure" when results are empty (Cloudflare-fronted targets)
    - FIX-I: record_vulnerability removed from PHASE_TOOLS (it fires in
      ANY phase — mapping it to 5 caused Phase 3→5→4 UI jumps)
    - FIX-J: findings_count only counts status=="recorded" (rejected
      findings — e.g. missing evidence — no longer inflated the count)
    - FIX-K: phase 5 label renamed "Reporting & Remediation"; prompt now
      says record immediately when confirmed (matches real agent behavior)
"""
from __future__ import annotations

import json
import logging
from collections import defaultdict
from typing import Any

from app.agents.llm_client import chat_completion
from app.agents.tool_bridge import build_tool_schemas, execute_tool_call, create_executor

logger = logging.getLogger(__name__)

# ── W19-FIX5 Phase I — Agent stop conditions (EVVO parity) ──────────
MAX_AGENT_TURNS = 100
# DISABLED (worklog task #16): exit gate prevented agents from completing
# scans when they had fewer than 20 turns / 25 commands.
MIN_AGENT_TURNS = 0   # DISABLED — was 20
MIN_COMMANDS_FOR_COMPLETE = 0  # DISABLED — was 25

# Backward-compat alias (old code may reference MAX_ITERATIONS)
MAX_ITERATIONS = MAX_AGENT_TURNS  # D18 guardrail: max 100 turns per scan

# ── FIX-B: retry guard ───────────────────────────────────────────────
# A tool that failed MAX_TOOL_FAILURES times in one scan is hard-blocked.
# Key insight: INVOCATION failures (bad flag, unknown option, parse error)
# cannot be fixed by retrying with different VALUES — the LLM doesn't know
# the correct flag, and each retry burns an iteration + subprocess spawn.
MAX_TOOL_FAILURES = 2

# Substrings (lowercased) that mark a tool output as a failure. Kept
# conservative — broad markers like "error" alone would false-positive
# on nikto/nuclei findings text.
_TOOL_FAIL_MARKERS: tuple[str, ...] = (
    "binary not found",
    "tool execution error",
    "error executing",
    "no such option",
    "unknown flag",
    "flag provided but not defined",
    "incorrect usage",
    "parse error",
    "invalid value",        # gobuster: invalid value "15" for flag -to
    "tool '",
)

# ── FIX-D: token budget (enforced) ───────────────────────────────────
# Raw token counts (the $ equivalent depends on the model's pricing).
# Defaults sized for ~$5 soft / ~$20 hard on a mid-tier model — tune per
# deployment via env or per llm_config in future.
TOKEN_SOFT_LIMIT = 2_500_000   # soft warning
TOKEN_HARD_LIMIT = 10_000_000  # hard cancel

# ── FIX-G: context management ────────────────────────────────────────
# Old tool results are truncated in the message history. The FULL output
# is preserved by tool_bridge (50KB cap + disk spill), so evidence is
# never lost — only the LLM's working context is bounded.
MAX_TOOL_RESULT_CHARS = 15_000   # keep recent results up to this size
KEEP_RECENT_RESULTS = 6          # number of most-recent results kept full
PRUNE_TO_CHARS = 1_500           # old results truncated to this + pointer

_TOOL_RESULT_PRUNED_NOTE = (
    "\n... [truncated {n} chars to save context — full output preserved in "
    "scan artifacts / spill file. Reference it if needed.]"
)


def _prune_old_tool_results(messages: list[dict[str, Any]]) -> int:
    """FIX-G: truncate OLD tool results in the message history (in place).

    Keeps the most recent KEEP_RECENT_RESULTS tool messages intact and
    truncates older ones to PRUNE_TO_CHARS. Returns number pruned.

    Called once per iteration after all tool results are appended.
    """
    tool_indices = [i for i, m in enumerate(messages) if m.get("role") == "tool"]
    if len(tool_indices) <= KEEP_RECENT_RESULTS:
        return 0

    pruned = 0
    for idx in tool_indices[:-KEEP_RECENT_RESULTS]:
        msg = messages[idx]
        content = msg.get("content") or ""
        if len(content) > PRUNE_TO_CHARS * 2:
            removed = len(content) - PRUNE_TO_CHARS
            msg["content"] = (
                content[:PRUNE_TO_CHARS]
                + _TOOL_RESULT_PRUNED_NOTE.format(n=removed)
            )
            pruned += 1
    return pruned


def _calc_progress(iteration: int, max_iterations: int) -> int:
    """FIX-C: progress inside the loop — 20%..90%, never exceeds 95."""
    frac = iteration / max(max_iterations, 1)
    return min(95, 20 + int(frac * 70))


# ── P1: preflight LLM check ──────────────────────────────────────────────

async def _preflight_llm_check(llm_config: dict[str, Any]) -> tuple[bool, str]:
    """P1: verify the OpenAI API key works before entering the ReAct loop.

    Sends a 1-token "ping" message to the configured model. If the call
    succeeds (any 2xx response), the API key + base URL + model name are
    all valid. If it fails, we return a human-readable error instead of
    crashing mid-loop after spending tokens on the system prompt.

    Returns:
        (ok, error_message) — error_message is "" when ok=True.
    """
    try:
        response = await chat_completion(
            llm_config=llm_config,
            messages=[
                {"role": "user", "content": "ping"},
            ],
            tools=None,  # no tools — just a minimal call
            temperature=0.0,
        )
        # A 2xx response proves the key/base_url/model are valid. Reasoning
        # models may leave `content` empty (their answer sits in
        # `reasoning_content`), so accept reasoning output and usage too.
        content = (response.get("content") or "").strip()
        reasoning = (response.get("reasoning_content") or "").strip()
        if content or reasoning or response.get("tool_calls") or response.get("usage"):
            return True, ""
        return False, "LLM returned empty response — check provider config"
    except Exception as exc:
        msg = str(exc)
        # Common error patterns → friendlier messages
        if "401" in msg or "Invalid API key" in msg or "Incorrect API key" in msg:
            return False, (
                f"OpenAI API key invalid or unauthorized. "
                f"Go to Settings → AI Channels and re-enter the key. "
                f"Underlying error: {msg}"
            )
        if "404" in msg or "model_not_found" in msg or "does not exist" in msg:
            return False, (
                f"Model {llm_config.get('model', '?')} not found on this "
                f"provider. Check the model name in Settings. "
                f"Underlying error: {msg}"
            )
        if "Connection" in msg or "timeout" in msg.lower() or "connect" in msg.lower():
            base_url = llm_config.get("base_url", "default")
            return False, (
                f"Cannot reach LLM endpoint ({base_url}). "
                f"Check network + base_url setting. "
                f"Underlying error: {msg}"
            )
        return False, f"LLM preflight failed: {msg}"


# ── P1: install hint for missing binaries ────────────────────────────────

# Maps binary names → install instructions for the LLM observation message.
# When SubprocessExecutor.execute() returns error="Binary not found: X",
# we replace the raw error with this friendly hint so the LLM knows the tool
# isn't installed on the host and can switch to a different available tool
# (or stop attempting this tool family).
BINARY_INSTALL_HINTS: dict[str, str] = {
    "nmap":         "nmap is not installed on the VAPT-AI host. Install with: `apt install nmap` (Debian/Ubuntu).",
    "nuclei":       "nuclei is not installed. Install with: `go install github.com/projectdiscovery/nuclei/v3/cmd/nuclei@latest` and add $GOPATH/bin to PATH.",
    "sqlmap":       "sqlmap is not installed. Install with: `apt install sqlmap` or `git clone --depth 1 https://github.com/sqlmapproject/sqlmap.git /opt/sqlmap`.",
    "ffuf":         "ffuf is not installed. Install with: `go install github.com/ffuf/ffuf/v2@latest`.",
    "gobuster":     "gobuster is not installed. Install with: `apt install gobuster`.",
    "feroxbuster":  "feroxbuster is not installed. Install from: https://github.com/epi052/feroxbuster/releases.",
    "httpx":        "httpx (projectdiscovery) is not installed. Install with: `go install github.com/projectdiscovery/httpx/cmd/httpx@latest`.",
    "whatweb":      "whatweb is not installed. Install with: `apt install whatweb`.",
    "nikto":        "nikto is not installed. Install with: `apt install nikto` or `git clone https://github.com/sullo/nikto.git`.",
    "dalfox":       "dalfox is not installed. Install with: `go install github.com/hahwul/dalfox@latest`.",
    "wpscan":       "wpscan is not installed. Install from: https://github.com/wpscanner/wpscan#installers (gem install wpscan).",
    "subfinder":    "subfinder is not installed. Install with: `go install github.com/projectdiscovery/subfinder/v2/cmd/subfinder@latest`.",
    "katana":       "katana is not installed. Install with: `go install github.com/projectdiscovery/katana/cmd/katana@latest`.",
    "amass":        "amass is not installed. Install with: `go install -v github.com/owasp-amass/amass/v4/...@master`.",
    "masscan":      "masscan is not installed. Install with: `apt install masscan` (needs root for raw sockets).",
    "rustscan":     "rustscan is not installed. Install with: `cargo install rustscan` or use the docker image `rustscan/rustscan:latest`.",
    "gau":          "gau is not installed. Install with: `go install github.com/lc/gau/v2/cmd/gau@latest`.",
    "waybackurls":  "waybackurls is not installed. Install with: `go install github.com/tomnomnom/waybackurls@latest`.",
    "hydra":        "hydra is not installed. Install with: `apt install hydra`.",
    "hashcat":      "hashcat is not installed. Install with: `apt install hashcat`.",
    "john":         "john (John the Ripper) is not installed. Install with: `apt install john`.",
    "fierce":       "fierce is not installed. Install with: `apt install fierce` or `gem install fierce`.",
    "dnsenum":      "dnsenum is not installed. Install with: `apt install dnsenum`.",
    "impacket-smbexec":       "impacket is not installed. Install with: `pipx install impacket` (or `pip install impacket`).",
    "impacket-psexec":         "impacket is not installed. Install with: `pipx install impacket`.",
    "impacket-wmiexec":        "impacket is not installed. Install with: `pipx install impacket`.",
    "impacket-secretsdump":    "impacket is not installed. Install with: `pipx install impacket`.",
    "impacket-GetUserSPNs":   "impacket is not installed. Install with: `pipx install impacket`.",
    "netexec":      "netexec (a.k.a. nxc) is not installed. Install with: `pipx install netexec` or `pip install netexec`.",
    "responder":    "responder is not installed. Install with: `apt install responder` or `git clone https://github.com/lgandx/Responder.git`.",
    "linpeas":      "linpeas is not installed. Download from: https://github.com/peass-ng/PEASS-ng/releases/latest (LinPEAS).",
    "winpeas.exe":  "winpeas.exe is not installed. Download from: https://github.com/peass-ng/PEASS-ng/releases/latest (WinPEAS).",
    "mimikatz.exe": "mimikatz.exe is not installed. Download from: https://github.com/gentilkiwi/mimikatz/releases.",
}


def _format_binary_not_found_hint(error_or_output: str) -> str:
    """If input contains 'Binary not found: X', return output with install hint.

    Used in two flows:
        1. Exception path: error = "Binary not found: nmap" (raised)
        2. Tool-result path: output = "...\\n[error] Binary not found: nmap"
           (returned by execute_tool_call embedding SubprocessExecutor's
           ToolResult.error string)

    For flow 2, preserves the original output text + appends the install
    hint as a clear "HINT" block so the LLM understands both what happened
    and what to do next.

    If no "Binary not found" marker, returns the input unchanged.
    """
    if "Binary not found" not in error_or_output:
        return error_or_output

    # Extract binary name from "Binary not found: <name>"
    marker_idx = error_or_output.find("Binary not found:")
    after_marker = error_or_output[marker_idx + len("Binary not found:"):]
    # Take first whitespace-delimited token (could have leading space)
    try:
        binary = after_marker.strip().split()[0].rstrip(",.;:")
    except Exception:
        return error_or_output

    hint = BINARY_INSTALL_HINTS.get(binary)
    if not hint:
        hint = (
            f"binary '{binary}' is not installed on the VAPT-AI host. "
            f"Ask the operator to install it, or try a different available tool. "
            f"Do NOT retry the same tool."
        )

    return (
        f"{error_or_output}\n\n"
        f"───── INSTALL HINT ─────\n"
        f"{hint}\n"
        f"───── END HINT ─────\n"
    )


# ── FIX-B: failure classification for the retry guard ───────────────

def _looks_like_failure(tool_output: str) -> bool:
    """True if the tool output indicates the call itself failed.

    Conservative: only matches invocation/runtime failure markers, never
    plain words like 'error' that appear inside normal scan findings.
    """
    head = tool_output[:800].lower()
    return any(marker in head for marker in _TOOL_FAIL_MARKERS)


# ── System prompt — structured pentest workflow (5 phases) ───────────────

SYSTEM_PROMPT = """You are VAPT-AI — an autonomous pentest agent. You have pre-authorized access to the target. Do NOT ask for permission.

You have access to ALL tools. Use them directly — there is no supervisor, no specialist transfer, no HITL gate.

## CRITICAL: no shell, no arbitrary commands

There is NO shell/execute tool. You CANNOT run `curl`, `echo`, `cat`, or any arbitrary command. If you need raw HTTP requests, use httpx / whatweb / katana / nuclei — they cover the same use cases. NEVER call tools that are not in your Available Tools list — they do not exist and calling them wastes iterations.

## MANDATORY: Follow this 5-phase workflow IN ORDER

Do NOT skip phases. Do NOT jump ahead. Complete each phase before moving to the next.

IMPORTANT — recording findings: call `record_vulnerability` IMMEDIATELY when you confirm a vulnerability with evidence, in whichever phase you confirmed it. Do NOT batch recordings to the end. Phase 4 is only for vulnerabilities that need ACTIVE exploitation to confirm (SQLi, XSS, command injection...). Facts verifiable by observation (exposed panel, missing header, info disclosure) are recorded as soon as seen.

### Phase 1: Reconnaissance (gather info about target)
Goal: Understand the target — what ports are open, what services run, what tech stack is used.

Steps:
1. Run `nmap` against the target host (use -Pn — most targets block ping; skip if target is a single URL on a known port)
2. Run `httpx` to probe HTTP/HTTPS services (note the [cdn] field — it tells you if the target is behind Cloudflare/Akamai/etc.)
3. Run `whatweb` to identify tech stack (PHP, ASP.NET, nginx, Apache, etc.)
4. Run `subfinder` and/or `amass` if target is a domain (find subdomains)

Output expected: list of open ports, services, tech stack, subdomains, CDN/WAF presence.
DO NOT move to Phase 2 until you know what the target is running.

### Phase 2: Scanning & Enumeration (map the attack surface)
Goal: Discover endpoints, directories, hidden files, and configurations.

Steps:
1. Run `katana` (depth 2, max_time 2m) to crawl the site and discover pages/parameters
2. Run `ffuf` or `feroxbuster` with a wordlist to find hidden directories/files (/admin, /backup.zip, /.env, /setup.php, /phpmyadmin/)
3. Run `nikto` for web server misconfiguration checks
4. Record interesting endpoints found (login pages, admin panels, API endpoints, config files)

Output expected: list of discovered URLs, directories, files, parameters.
DO NOT move to Phase 3 until you have mapped the attack surface.

### Phase 3: Vulnerability Analysis / Triage (identify potential vulns)
Goal: From the mapped surface, identify what MIGHT be vulnerable. Prioritize by severity.

Steps:
1. Run `nuclei` with severity `high,critical,medium` against discovered URLs
2. Probe key endpoints with `httpx` to check for exposed panels and information disclosure (phpMyAdmin, phpinfo(), error_log, server-status, .env)
3. Use `httpx`/`whatweb` to check for outdated components (old jQuery, PHP version via phpinfo, etc.)
4. List potential vulnerabilities with severity ranking (critical → low)

Output expected: list of potential vulnerabilities with severity.
DO NOT move to Phase 4 until you have a list of potential vulns to test.

### Phase 4: Exploitation (prove the vulnerability is real)
Goal: For each potential vulnerability from Phase 3 that needs active exploitation, run an exploit to PROVE it is real. A scanner warning is NOT enough — you need a PoC (command + server response).

Steps:
- SQL Injection: Run `sqlmap` on suspected injectable parameters (batch mode). The sqlmap output IS the PoC.
- Parameter discovery: test interesting parameters found in Phase 2 with sqlmap/dalfox as appropriate.
- Missing Security Headers: tool output showing the response headers IS the PoC.
- Exposed Panel: probe output showing 200 OK on /phpmyadmin/ IS the PoC.
- Misconfiguration: `nikto` or `nuclei` output showing the misconfig IS the PoC.

For EACH confirmed vulnerability, IMMEDIATELY call `record_vulnerability` with:
- `evidence`: the EXACT tool output (command + server response) that proves the vuln
- `tool`: the NAME of the tool that produced the evidence (`sqlmap`, `dalfox`, `nuclei`, `nmap`, `httpx`, ...). This is REQUIRED — it is shown as "Tool used" in the report. Do NOT write `agent`.
- `command`: the EXACT command you ran, with REAL values already substituted
- `vuln_type`: sqli, xss, lfi, rce, information-disclosure, security_misconfiguration, etc.
- `severity`: critical, high, medium, low, info
- `cvss_score`: estimate based on impact (critical=9-10, high=7-8.9, medium=4-6.9, low=0.1-3.9)
- `cwe_id`: if known (e.g. CWE-89 for SQLi, CWE-79 for XSS)
- `wstg_test_id`: if known (e.g. WSTG-INPV-05 for SQLi)
- `description`: what + where + why exploitable
- `remediation`: how to fix (specific, not generic)

DO NOT call record_vulnerability without evidence. The call will be REJECTED.

### Phase 5: Reporting & Remediation (wrap up)
Goal: Summarize all findings and exit.

Steps:
1. Review all recorded vulnerabilities
2. Call `exit` with a summary:
   - Total findings count
   - List of findings with severity
   - Overall assessment of the target's security posture
   - SCAN LIMITATIONS: if the target is behind a WAF/CDN/challenge page, state explicitly what could NOT be tested and why the results are limited. A scan that saw only a CDN edge is NOT evidence that the origin is secure.

## Evidence Requirement (CRITICAL)

record_vulnerability REQUIRES the `evidence` parameter containing raw tool output.
PoC = command you ran + server's response. Without evidence, the call is REJECTED.

Example — Exposed phpMyAdmin:
```json
{
  "title": "Exposed phpMyAdmin Panel",
  "vuln_type": "security_misconfiguration",
  "severity": "high",
  "cvss_score": 7.5,
  "cwe_id": "CWE-16",
  "wstg_test_id": "WSTG-CONF-04",
  "target": "https://target.com",
  "location": "/phpMyAdmin/",
  "evidence": "$ httpx -u https://target.com/phpMyAdmin/\\nhttps://target.com/phpMyAdmin/ [200] [phpMyAdmin] [MySQL,PHP,jQuery]",
  "description": "phpMyAdmin is publicly accessible without IP restriction at /phpMyAdmin/. An attacker can attempt brute-force or exploit known phpMyAdmin CVEs against this instance.",
  "remediation": "Restrict phpMyAdmin access to trusted IPs (allowlist) or place it behind VPN. Keep phpMyAdmin updated to the latest version."
}
Rules
ALWAYS follow the 5-phase workflow in order. Do NOT skip phases.
ALWAYS run a tool first, capture its output, THEN call record_vulnerability with that output as evidence.
NEVER write placeholders in `evidence` or `command`. The fields must contain the EXACT command with REAL values — `https://target.com/?q=x`, not `{{URL}}?q=x`, not `<target>`, not `$URL`. A placeholder makes the PoC non-reproducible and the finding looks fabricated.
ALWAYS set the `tool` parameter to the real tool name (`sqlmap`, `dalfox`, `nuclei`, `nmap`, `httpx`...). NEVER set it to `agent`, `curl`, or the exploit phase name — the report displays it as "Tool used".
metasploit: use `auxiliary/scanner` modules FIRST (version checks, non-destructive probes). Use `exploit/` modules ONLY for vulnerabilities you already confirmed with evidence (sqlmap output, nuclei match). NEVER guess exploits against unconfirmed targets.
Do NOT fabricate findings — only report what tools actually detected.
Call record_vulnerability for EACH confirmed vulnerability (1 call per finding), IMMEDIATELY when confirmed.
When done with all 5 phases, call exit with a summary of all findings.
If a tool fails, try a different tool — do NOT call the same failed tool more than twice. The system will hard-block tools that keep failing.
If a tool returns EMPTY output, do NOT conclude the target is down or secure. Empty output usually means: (a) WAF/challenge page, (b) your IP is rate-limited, (c) wrong flags. Try different flags or a different tool ONCE, then note the limitation in your final report.
If the target is behind a CDN/WAF (check httpx [cdn] field), ALL port-scan and web-scan results reflect the CDN edge, NOT the origin server. State this limitation in the final summary.
Available Tools
Recon: nmap, httpx, whatweb, masscan, rustscan, subfinder, amass
Enumerate: ffuf, gobuster, feroxbuster, katana, gau, waybackurls
Vuln Scan: nuclei, nikto, dalfox, wpscan, fscan
Exploit: sqlmap, metasploit, hydra
Utility: record_vulnerability, exit

(That is the COMPLETE list. There is no execute/shell/curl tool. Do not attempt to call anything else.)

Actually RUN the tools — do NOT just describe what you would do. Each tool call produces real output that becomes evidence for findings.
"""

async def run_react_scan(
    target: str,
    user_prompt: str,
    llm_config: dict[str, Any],
    scan_id: str,
    max_iterations: int = 100,
    executor: Any = None,
) -> dict[str, Any]:
    """Run a real ReAct scan loop (CyberStrikeAI pattern — single agent, all tools).

    This is the ONLY execution path after the Phase 1 overhaul.
    No LangGraph supervisor, no specialist transfers, no HITL gate.
    One LLM + all tools + record_vulnerability + exit.

    Args:
        target: target URL or IP (e.g. "http://example.com")
        user_prompt: user's natural-language scan request
        llm_config: LLM config from DB (provider, api_key, model, ...)
        scan_id: scan ID for SSE events + DB attribution
        max_iterations: max ReAct iterations (default 100)
        executor: SubprocessExecutor instance (created if None)

    Returns:
        dict with: status, findings_count, iterations, total_tokens, final_summary, error
    """
    from app.pentest.events import emit_scan_progress
    from app.pentest.events import emit_phase_change

    # ── P1: preflight LLM check ──────────────────────────────────
    await emit_scan_progress(scan_id, thought="Checking LLM config (OpenAI API key, model, base_url)...",
                             agent_name="orchestrator", progress=5)

    ok, err = await _preflight_llm_check(llm_config)
    if not ok:
        logger.error("LLM preflight failed | scan=%s | error=%s", scan_id, err)
        await emit_scan_progress(scan_id, thought=f"❌ LLM preflight failed: {err}",
                                 agent_name="orchestrator", progress=5)
        return {
            "status": "error",
            "findings_count": 0,
            "iterations": 0,
            "total_tokens": 0,
            "final_summary": f"LLM preflight failed: {err}",
            "error": err,
        }

    logger.info("LLM preflight OK | scan=%s | model=%s",
                scan_id, llm_config.get("model"))
    await emit_scan_progress(scan_id, thought="✅ LLM config OK. Starting ReAct loop...",
                             agent_name="orchestrator", progress=10)

    # Build tool schemas — ALL tools, no allowlist (CyberStrikeAI pattern)
    tool_schemas = build_tool_schemas()

    # Create executor for this scan (if not passed in)
    if executor is None:
        executor = create_executor(target, scan_id)

    # Build initial messages
    user_message = f"Target: {target}\n\nRequest: {user_prompt}\n\nStart scanning this target."
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

    total_tokens = 0
    findings_count = 0
    iterations = 0
    final_summary = ""
    budget_warned = False  # FIX-D

    # ── FIX-B: per-tool failure tracking (retry guard) ───────────────
    tool_fail_counts: dict[str, int] = defaultdict(int)

    # ── Phase tracking: map tool name → pentest phase ──────────────────
    # FIX-I: record_vulnerability REMOVED — it fires in ANY phase (the agent
    # records as soon as evidence exists, which is correct behavior).
    # Mapping it to 5 caused Phase 3→5→4 jumps in the UI.
    PHASE_TOOLS = {
        # Phase 1: Reconnaissance
        "nmap": 1, "httpx": 1, "whatweb": 1, "masscan": 1, "rustscan": 1,
        "subfinder": 1, "amass": 1, "dnsenum": 1, "fierce": 1,
        "theharvester": 1, "gau": 1, "waybackurls": 1,
        # Phase 2: Scanning & Enumeration
        "ffuf": 2, "gobuster": 2, "feroxbuster": 2, "katana": 2,
        "nikto": 2,
        # Phase 3: Vulnerability Analysis
        "nuclei": 3, "dalfox": 3, "wpscan": 3, "fscan": 2,
        # Phase 4: Exploitation
        "sqlmap": 4, "metasploit": 4, "hydra": 4,
        # Phase 5: Reporting (exit only — record_vulnerability is phase-agnostic)
        "exit": 5,
    }
    PHASE_NAMES = {
        1: "recon",
        2: "enumeration",
        3: "vulnerability-analysis",
        4: "exploitation",
        5: "reporting",
    }
    PHASE_PROGRESS = {1: 15, 2: 35, 3: 55, 4: 75, 5: 90}
    current_phase = 1  # set below via emit

    # Emit Phase 1 start
    await emit_phase_change(
        scan_id, phase="recon", progress=15,
        message="Phase 1: Reconnaissance — gathering target info",
        agent_name="orchestrator",
    )

    logger.info("ReAct scan started | scan=%s | target=%s | max_iter=%d",
                scan_id, target, max_iterations)

    # ── ReAct loop ──────────────────────────────────────────────
    for iteration in range(max_iterations):
        iterations = iteration + 1
        loop_progress = _calc_progress(iterations, max_iterations)  # FIX-C

        # ── Phase D: abort check ──────────────────────────────────
        # Honor the panic button BEFORE the next LLM call — without this,
        # aborting a scan only takes effect after max_iterations.
        from app.pentest.scan_registry import scan_registry
        if await scan_registry.is_aborted(scan_id):
            logger.warning(
                "ReAct scan aborting — scan_registry.abort_event is set | scan=%s | iter=%d",
                scan_id, iterations,
            )
            return {
                "status": "aborted",
                "findings_count": findings_count,
                "iterations": iterations,
                "total_tokens": total_tokens,
                "final_summary": f"Scan aborted at iteration {iterations} (panic button).",
                "error": "user_panic_button",
            }

        try:
            # Call LLM
            await emit_scan_progress(scan_id, thought=f"Thinking... (iteration {iterations}/{max_iterations})",
                                     agent_name="orchestrator", progress=loop_progress)

            response = await chat_completion(
                llm_config=llm_config,
                messages=messages,
                tools=tool_schemas,
            )

            # FIX-F: defensive usage parsing (usage may be None/missing)
            total_tokens += (response.get("usage") or {}).get("total_tokens", 0)

            # ── FIX-D: token budget enforcement ────────────────────
            if total_tokens > TOKEN_SOFT_LIMIT and not budget_warned:
                budget_warned = True
                await emit_scan_progress(
                    scan_id,
                    thought=f"⚠️ Token budget warning: {total_tokens:,} tokens used (soft limit {TOKEN_SOFT_LIMIT:,}). Wrap up efficiently.",
                    agent_name="orchestrator", progress=loop_progress,
                )
            if total_tokens > TOKEN_HARD_LIMIT:
                logger.warning("Token hard limit exceeded | scan=%s | tokens=%d", scan_id, total_tokens)
                await emit_scan_progress(
                    scan_id,
                    thought=f"🛑 Token budget exceeded ({total_tokens:,}). Scan cancelled.",
                    agent_name="orchestrator", progress=95,
                )
                return {
                    "status": "error",
                    "findings_count": findings_count,
                    "iterations": iterations,
                    "total_tokens": total_tokens,
                    "final_summary": f"Token budget exceeded ({total_tokens:,} tokens). Scan cancelled.",
                    "error": "token_budget_exceeded",
                }

            # Append assistant message to history
            assistant_msg: dict[str, Any] = {"role": "assistant", "content": response["content"]}
            if response["tool_calls"]:
                assistant_msg["tool_calls"] = [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": tc["arguments"],
                        },
                    }
                    for tc in response["tool_calls"]
                ]
            messages.append(assistant_msg)

            # If LLM is thinking (content but no tool_calls), continue to next iteration
            if not response["tool_calls"]:
                if response["content"]:
                    logger.info("LLM thinking (no tool calls) | scan=%s | iter=%d | content=%s",
                                scan_id, iterations, response["content"][:100])
                    await emit_scan_progress(scan_id, thought=response["content"][:200],
                                             agent_name="orchestrator", progress=loop_progress)
                continue

            # ── Execute tool calls ───────────────────────────────
            for tc in response["tool_calls"]:
                tool_name = tc["name"]

                # ── FIX-E: surface malformed JSON args to the LLM ──
                try:
                    tool_args = json.loads(tc["arguments"]) if tc["arguments"] else {}
                except json.JSONDecodeError as e:
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": (
                            f"ERROR: your arguments for '{tool_name}' were not valid JSON ({e}). "
                            f"Raw arguments received: {str(tc['arguments'])[:200]}. "
                            f"Retry with correctly formatted JSON arguments."
                        ),
                    })
                    await emit_scan_progress(
                        scan_id,
                        thought=f"⚠️ Malformed arguments for {tool_name} — asking LLM to retry with valid JSON",
                        agent_name="orchestrator", tool_name=tool_name,
                        progress=loop_progress,
                    )
                    continue

                # ── FIX-B: hard-block tools that keep failing ──────
                if (tool_name not in ("record_vulnerability", "exit")
                        and tool_fail_counts[tool_name] >= MAX_TOOL_FAILURES):
                    blocked_msg = (
                        f"🚫 Tool '{tool_name}' is BLOCKED — it already failed "
                        f"{tool_fail_counts[tool_name]} times in this scan (limit {MAX_TOOL_FAILURES}). "
                        f"Retrying will be rejected. Switch to a different tool or move to the next step."
                    )
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": blocked_msg,
                    })
                    await emit_scan_progress(
                        scan_id,
                        thought=f"🚫 Blocked repeat-failed tool: {tool_name}",
                        agent_name="orchestrator", tool_name=tool_name,
                        progress=loop_progress,
                    )
                    continue

                # ── Phase tracking: MONOTONIC — only increase ──────
                # Once we enter Phase N, we never go back to Phase < N.
                # Prevents "Phase 3 → Phase 1" jumps when the LLM calls
                # httpx (a Phase 1 tool) during Phase 3 to verify endpoints.
                tool_phase = PHASE_TOOLS.get(tool_name, 0)
                if tool_phase > current_phase:
                    current_phase = tool_phase
                    phase_name = PHASE_NAMES.get(tool_phase, "unknown")
                    phase_progress = PHASE_PROGRESS.get(tool_phase, 50)
                    await emit_phase_change(
                        scan_id, phase=phase_name, progress=phase_progress,
                        message=f"Phase {tool_phase}: {phase_name.replace('-', ' ').title()}",
                        agent_name="orchestrator",
                    )
                    logger.info("Phase change | scan=%s | phase=%d (%s) | tool=%s",
                                scan_id, tool_phase, phase_name, tool_name)

                # Emit SSE: tool_call
                await emit_scan_progress(scan_id, thought=f"🔧 Running: {tool_name}({json.dumps(tool_args, ensure_ascii=False)[:100]})",
                                         agent_name="orchestrator", tool_name=tool_name,
                                         progress=loop_progress)

                logger.info("Tool call | scan=%s | iter=%d | tool=%s | args=%s",
                            scan_id, iterations, tool_name,
                            json.dumps(tool_args, ensure_ascii=False)[:80])

                # Check for exit tool
                if tool_name == "exit":
                    final_summary = tool_args.get("summary", "Scan complete.")
                    logger.info("Scan exit | scan=%s | iter=%d | summary=%s",
                                scan_id, iterations, final_summary[:100])
                    await emit_scan_progress(scan_id, thought=f"✅ Done: {final_summary[:150]}",
                                             agent_name="orchestrator", progress=95)

                    # Count findings from DB (source of truth — replaces the
                    # in-loop counter which could drift from rejections)
                    try:
                        from app.db.session import async_session
                        from app.db.models.pentest import Finding
                        from sqlalchemy import select, func
                        async with async_session() as session:
                            count = await session.scalar(
                                select(func.count(Finding.id)).where(Finding.scan_id == scan_id)
                            ) or 0
                            findings_count = count
                    except Exception:
                        pass

                    return {
                        "status": "completed",
                        "findings_count": findings_count,
                        "iterations": iterations,
                        "total_tokens": total_tokens,
                        "final_summary": final_summary,
                        "error": None,
                    }

                # Execute the tool
                try:
                    tool_output = await execute_tool_call(
                        tool_name=tool_name,
                        tool_args=tool_args,
                        target=target,
                        scan_id=scan_id,
                        executor=executor,
                    )

                    # P1: detect "Binary not found" inside the returned
                    # tool_output → inject install hint so the LLM can recover.
                    if "Binary not found" in tool_output:
                        tool_output = _format_binary_not_found_hint(tool_output)

                    if tool_name == "record_vulnerability":
                        # FIX-J: only count SUCCESSFULLY recorded findings —
                        # _record_vulnerability may return status "rejected"
                        # (e.g. missing evidence); those are NOT findings.
                        counted = False
                        try:
                            result_json = json.loads(tool_output)
                            counted = result_json.get("status") == "recorded"
                        except (json.JSONDecodeError, AttributeError):
                            counted = False
                        if counted:
                            findings_count += 1
                            await emit_scan_progress(
                                scan_id, thought=f"📋 Recorded vulnerability #{findings_count}",
                                agent_name="orchestrator", tool_name="record_vulnerability",
                                progress=loop_progress)
                        else:
                            await emit_scan_progress(
                                scan_id, thought=f"⚠️ record_vulnerability rejected (missing/invalid evidence)",
                                agent_name="orchestrator", tool_name="record_vulnerability",
                                progress=loop_progress)
                    else:
                        # Emit observation (truncated for SSE)
                        obs_preview = tool_output[:200].replace("\n", " ")
                        await emit_scan_progress(scan_id, thought=f"📤 Result {tool_name}: {obs_preview}",
                                                 agent_name="orchestrator", tool_name=tool_name,
                                                 progress=loop_progress)

                        # FIX-B: track failures for the retry guard
                        if _looks_like_failure(tool_output):
                            tool_fail_counts[tool_name] += 1
                            if tool_fail_counts[tool_name] >= MAX_TOOL_FAILURES:
                                # Append the block notice to THIS result too,
                                # so the LLM sees it in the same context.
                                tool_output += (
                                    f"\n\n🚫 NOTE: '{tool_name}' has now failed "
                                    f"{tool_fail_counts[tool_name]} times. It will be "
                                    f"BLOCKED on the next attempt. Switch tools."
                                )
                        else:
                            tool_fail_counts[tool_name] = 0  # success resets

                except Exception as exc:
                    tool_output = f"Error executing {tool_name}: {exc}"
                    err_str = str(exc)
                    if "Binary not found" in err_str or "FileNotFoundError" in err_str:
                        tool_output = _format_binary_not_found_hint(err_str)
                    tool_fail_counts[tool_name] += 1
                    logger.error("Tool execution error | scan=%s | tool=%s | error=%s",
                                 scan_id, tool_name, exc)
                    await emit_scan_progress(scan_id, thought=f"❌ Error {tool_name}: {str(exc)[:100]}",
                                             agent_name="orchestrator", tool_name=tool_name,
                                             progress=loop_progress)

                # Append tool result to messages
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": tool_output,
                })

            # FIX-G: prune old tool results after this iteration's results
            # are appended — bounds the message history across 100 turns.
            pruned = _prune_old_tool_results(messages)
            if pruned:
                logger.debug("Context pruning | scan=%s | pruned=%d old tool results",
                             scan_id, pruned)

        except Exception as exc:
            logger.error("ReAct loop error | scan=%s | iter=%d | error=%s",
                         scan_id, iterations, exc)
            return {
                "status": "error",
                "findings_count": findings_count,
                "iterations": iterations,
                "total_tokens": total_tokens,
                "final_summary": f"Scan failed at iteration {iterations}: {exc}",
                "error": str(exc),
            }

    # Max iterations reached
    logger.warning("Max iterations reached | scan=%s | iter=%d", scan_id, iterations)
    await emit_scan_progress(scan_id, thought=f"⚠️ Reached limit of {max_iterations} iterations. Ending scan.",
                             agent_name="orchestrator", progress=90)

    return {
        "status": "max_iterations",
        "findings_count": findings_count,
        "iterations": iterations,
        "total_tokens": total_tokens,
        "final_summary": f"Scan reached max iterations ({max_iterations}). Partial results may be available.",
        "error": None,
    }


__all__ = [
    "run_react_scan",
    "SYSTEM_PROMPT",
    "MAX_ITERATIONS",
]