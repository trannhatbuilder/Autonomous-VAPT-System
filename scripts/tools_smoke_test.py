#!/usr/bin/env python3
"""VAPT-AI tool-wrapper smoke test — run at deploy/startup.

WHY THIS EXISTS
---------------
A YAML wrapper can *parse* fine and still generate a broken CLI. The incident
that motivated this script: `whatweb.yaml` declared `quiet: {flag: "-q",
default: true}`, but whatweb's `-q` suppresses the brief logging that IS the
result line. Every scan therefore returned 0 bytes and the orchestrator
displayed `[empty_result]` — with exit code 0, so nothing looked wrong.

Unlike a hardcoded bash smoke test (which types the "correct" command by hand
and thus only tests the *tools*), this script drives the SAME code path the
agent uses — `ToolDef.build_command_args()` with the YAML's own defaults — so
it actually validates the *wrapper → CLI mapping* and auto-covers new tools.

THE HEADLINE CHECK is "exit 0 but EMPTY stdout": that is the exact signature of
a bad flag default (the whatweb bug) and is invisible to a normal exit-code
check.

CONTENT-DISCOVERY TOOLS (gobuster/ffuf/feroxbuster) are special-cased: on a bare
example.com probe they are empty no matter what, so the empty-check cannot tell a
broken flag from a clean scan. They are instead run against a local fixture
server that serves one known file — the file being found IS the pass condition.

USAGE
-----
    cd ~/VAPT-AI
    python3 scripts/tools_smoke_test.py            # safe subset
    python3 scripts/tools_smoke_test.py --all      # include slow/active tools
    python3 scripts/tools_smoke_test.py --only httpx,whatweb,nmap
    python3 scripts/tools_smoke_test.py --list     # show tiers, run nothing
    python3 scripts/tools_smoke_test.py --binary-only   # presence check only

Exit code is 0 when nothing FAILed, 1 otherwise (fail-fast for CI / a deploy
hook). SKIP (binary missing) does NOT fail the run.

WIRING IT IN
------------
Add to your deploy script / systemd ExecStartPre, e.g.:
    ExecStartPre=/home/nhat/VAPT-AI/venv314/bin/python3 /home/nhat/VAPT-AI/scripts/tools_smoke_test.py --only httpx,whatweb,nmap
Or run it as part of `scripts/install_tools.sh` after installation.
"""
from __future__ import annotations

import argparse
import contextlib
import functools
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

# ── Make `app.*` importable when run as a standalone script ──────────────
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from app.tools.loader import load_all_tools, BUNDLED_WORDLIST  # noqa: E402

# ── Test targets ────────────────────────────────────────────────────────
URL_TARGET = "https://example.com"
HOST_TARGET = "example.com"

# Tools whose "target" is a host/IP (port scanners) rather than a URL.
_HOST_TARGET_TOOLS = {"nmap", "rustscan", "masscan", "fscan"}

# ── Tiers ───────────────────────────────────────────────────────────────
# "safe"     — run by default. Passive/light recon against example.com.
# "extended" — only with --all. Slow, actively intrusive, or needs root.
# "manual"   — never auto-run: needs credentials / hash files / a live host /
#              a local shell (linpeas/winpeas). Validated for CLI shape only.
_SAFE = {
    "httpx", "whatweb", "nmap", "rustscan", "katana", "nikto",
    "gobuster", "ffuf", "feroxbuster", "subfinder", "dnsenum",
    "fierce", "gau", "waybackurls",
}
_EXTENDED = {
    "nuclei", "sqlmap", "dalfox", "wpscan", "fscan", "masscan",
    "amass", "theharvester",
}
_MANUAL = {
    "hashcat", "john", "hydra", "netexec", "impacket", "responder",
    "mimikatz", "linpeas", "winpeas", "metasploit",
}

# Extra args needed to satisfy *required* params or to keep a run bounded.
# Applied on top of the YAML defaults (so defaults are still exercised).
_EXTRA_PARAMS: dict[str, dict] = {
    # web — keep probes bounded
    "nmap":        {"ports": "80,443", "host_timeout": "20s"},
    "rustscan":    {"ports": "80,443"},
    "nikto":       {"timeout": 15, "tuning": "b"},
    "nuclei":      {"severity": "info", "timeout": 10, "rate_limit": 50},
    "katana":      {"depth": 2},
    # fuzzers need a wordlist + a mode
    "gobuster":    {"mode": "dir", "wordlist": str(BUNDLED_WORDLIST), "threads": 10},
    "ffuf":        {"wordlist": str(BUNDLED_WORDLIST), "threads": 10},
    "feroxbuster": {"wordlist": str(BUNDLED_WORDLIST), "threads": 10, "depth": 1,
                    "additional_args": "--no-state"},  # don't litter .state files
    # extended
    "wpscan":      {"enumerate": "vp"},
    "dalfox":      {"silence": False},
    # manual (CLI-shape only; a safe no-op command where possible)
    "metasploit":  {"command": "version"},
    "masscan":     {"ports": "80,443", "rate": 100},
}

# Tools known to legitimately produce empty output on a bare example.com probe.
# Reported as WARN (not FAIL) to avoid false alarms — but the empty check is
# still shown so a wrapper regression is visible.
_EXPECT_EMPTY_OK = {"dnsenum", "fierce", "gau", "waybackurls", "amass", "theharvester"}

# Content-discovery / fuzzer tools. On a bare example.com probe they return
# EMPTY no matter what (the target simply has no matching paths), which makes the
# "exit 0 + empty stdout" heuristic above useless for them — a broken flag and a
# clean empty scan look identical. So they are exercised against a LOCAL fixture
# server that serves one known file: found -> PASS, not found -> FAIL.
_CONTENT_DISCOVERY_TOOLS = {"gobuster", "ffuf", "feroxbuster"}
_FIXTURE_FILENAME = "vapt-smoke-fixture.html"

_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")


def _strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text or "")


class _QuietHTTPHandler(SimpleHTTPRequestHandler):
    """Serve files without spamming the smoke-test output with request logs."""

    def log_message(self, *args):  # silence only
        pass


@contextlib.contextmanager
def _fixture_server():
    """Serve a temp dir holding one known file, for content-discovery tools.

    Yields ``(base_url, root_dir, wordlist_path)``. The wordlist contains the
    fixture filename plus a couple of NON-existent decoys, so the scan is
    deterministic and the expected hit is unambiguous.

    Why decoys: ffuf's default ``-ac`` (auto-calibrate) needs more than one
    entry to calibrate correctly — with a single-entry wordlist ffuf emits only
    its calibration probe line and never prints the real hit, so the fixture
    would be reported "NOT found" even though the wrapper mapping is correct.
    """
    with tempfile.TemporaryDirectory(prefix="vapt-smoke-") as tmp:
        root = Path(tmp) / "root"
        root.mkdir()
        (root / _FIXTURE_FILENAME).write_text("vapt smoke fixture\n")
        wordlist = Path(tmp) / "wordlist.txt"
        wordlist.write_text(
            _FIXTURE_FILENAME + "\n"
            "vapt-smoke-decoy-a.html\n"
            "vapt-smoke-decoy-b.html\n"
        )

        handler = functools.partial(_QuietHTTPHandler, directory=str(root))
        httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        port = httpd.server_address[1]
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        try:
            yield f"http://127.0.0.1:{port}", str(root), str(wordlist)
        finally:
            httpd.shutdown()
            httpd.server_close()
            thread.join(timeout=5)


def _err(msg: str) -> None:
    print(f"error: {msg}", file=sys.stderr)


def _build_params(tool_name: str, tool_def, target: str | None = None,
                  wordlist: str | None = None) -> dict:
    """Fill required params + inject the test target, leaving defaults intact.

    ``target`` / ``wordlist``, when given, FORCE the value — the fixture path
    uses them so a content-discovery tool scans the local server.
    """
    params: dict = dict(_EXTRA_PARAMS.get(tool_name, {}))

    for p in tool_def.parameters:
        lname = p.name.lower()
        # Target injection — only when not already supplied by _EXTRA_PARAMS
        if lname in ("target", "domain") and p.name not in params:
            params[p.name] = HOST_TARGET if (lname == "domain" or tool_name in _HOST_TARGET_TOOLS) else URL_TARGET
            continue
        if p.name in params:
            continue
        # Required params with no default need a value or the tool aborts.
        if not p.required:
            continue
        if lname == "wordlist":
            params[p.name] = str(BUNDLED_WORDLIST)
        elif lname == "ports":
            params[p.name] = "80,443"
        elif lname == "mode":
            params[p.name] = "dir"
        # Anything else required (hash_file, service, interface, command, ...)
        # is intentionally left unset → the tool will be treated as "manual".

    # Forced overrides (fixture path) win over _EXTRA_PARAMS and defaults.
    if target is not None:
        for p in tool_def.parameters:
            if p.name.lower() in ("target", "domain"):
                params[p.name] = target
    if wordlist is not None:
        for p in tool_def.parameters:
            if p.name.lower() == "wordlist":
                params[p.name] = wordlist
    return params


def _classify_stdout(stdout: str, code: int, allowed_exits: list[int], expect_empty_ok: bool) -> tuple[str, str]:
    """Return (verdict, detail). verdict ∈ {PASS, FAIL, WARN}."""
    out = _strip_ansi(stdout).strip()
    if not out:
        if expect_empty_ok:
            return "WARN", "empty stdout (allowed for this tool)"
        return "FAIL", "EMPTY stdout with exit 0 — flag mapping likely wrong"
    return "PASS", ""


def run_tool(tool_name: str, tool_def, target_timeout: int, binary_only: bool,
             target_override: str | None = None,
             wordlist_override: str | None = None,
             expect_fixture: str | None = None) -> dict:
    result = {"tool": tool_name, "verdict": "PASS", "detail": "", "secs": 0.0, "bytes": 0}

    binary = tool_def.command.split()[0]
    exe = shutil.which(binary)
    if not exe:
        result.update(verdict="SKIP", detail=f"binary '{binary}' not found")
        return result

    if binary_only:
        result.update(verdict="PASS", detail=f"found at {exe}")
        return result

    params = _build_params(tool_name, tool_def,
                           target=target_override, wordlist=wordlist_override)
    # Missing required param → can't safely auto-run
    missing = [p.name for p in tool_def.parameters
               if p.required and p.name not in params and p.default in (None, "")]
    if missing:
        result.update(verdict="MANUAL", detail=f"needs {', '.join(missing)} — CLI shape only")
        return result

    try:
        args = tool_def.build_command_args(**params)
    except Exception as exc:  # wrapper parse/arg-build error is a real FAIL
        result.update(verdict="FAIL", detail=f"build_command_args raised: {exc}")
        return result

    cmd = [binary] + args
    timeout = min(tool_def.timeout, target_timeout)
    t0 = time.monotonic()
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True,
            timeout=timeout, cwd=str(_REPO_ROOT),
            stdin=subprocess.DEVNULL,   # critical: tools like gau/waybackurls
            # read stdin by default — without DEVNULL they block forever and
            # the per-tool timeout never fires (SIGTERM can't interrupt a
            # blocking read on an inherited tty).
            env={**os.environ, "NO_COLOR": "1"},
        )
    except subprocess.TimeoutExpired:
        result.update(verdict="FAIL", secs=time.monotonic() - t0,
                      detail=f"TIMEOUT after {timeout}s")
        result["cmd"] = " ".join(cmd)
        return result
    except Exception as exc:
        result.update(verdict="FAIL", secs=time.monotonic() - t0,
                      detail=f"subprocess error: {exc}")
        result["cmd"] = " ".join(cmd)
        return result

    result["secs"] = time.monotonic() - t0
    result["bytes"] = len(proc.stdout.encode("utf-8", "ignore"))
    result["cmd"] = " ".join(cmd)

    allowed = set(tool_def.allowed_exit_codes or [0])
    if proc.returncode not in allowed:
        err_tail = _strip_ansi(proc.stderr).strip().splitlines()
        tail = err_tail[-1][:120] if err_tail else ""
        result.update(verdict="FAIL",
                      detail=f"exit {proc.returncode} (allowed {sorted(allowed)})"
                             + (f" | {tail}" if tail else ""))
        return result

    if expect_fixture is not None:
        # Content-discovery tool: success == the known fixture file was found.
        if expect_fixture in _strip_ansi(proc.stdout):
            result.update(verdict="PASS", detail=f"found fixture '{expect_fixture}'")
        else:
            result.update(
                verdict="FAIL",
                detail=f"fixture '{expect_fixture}' NOT found in output — "
                       "flag mapping likely wrong")
        return result

    verdict, detail = _classify_stdout(
        proc.stdout, proc.returncode, list(allowed),
        expect_empty_ok=tool_name in _EXPECT_EMPTY_OK,
    )
    result.update(verdict=verdict, detail=detail)
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description="VAPT-AI tool-wrapper smoke test")
    ap.add_argument("--all", action="store_true",
                    help="include slow/active tools (extended tier)")
    ap.add_argument("--only", default="",
                    help="comma-separated tool names to test")
    ap.add_argument("--timeout", type=int, default=60,
                    help="per-tool wall-clock cap in seconds (default 60)")
    ap.add_argument("--list", action="store_true",
                    help="list tier classification and exit")
    ap.add_argument("--binary-only", action="store_true",
                    help="only check the binary exists (no execution)")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    args = ap.parse_args()

    tools = load_all_tools()

    if args.list:
        for name in sorted(tools):
            tier = ("safe" if name in _SAFE else
                    "extended" if name in _EXTENDED else
                    "manual" if name in _MANUAL else "unclassified")
            print(f"{tier:11} {name}")
        return 0

    if args.only:
        selected = [t.strip() for t in args.only.split(",") if t.strip()]
    else:
        allowed_tiers = set(_SAFE) | (set(_EXTENDED) if args.all else set())
        if args.all:
            allowed_tiers |= set(_MANUAL)
        selected = sorted(allowed_tiers & set(tools))

    if not selected:
        _err("no tools selected")
        return 1

    if not args.json:
        print("VAPT-AI tool-wrapper smoke test")
        print(f"  target : {URL_TARGET}  (host {HOST_TARGET})")
        print(f"  timeout: {args.timeout}s per tool")
        print(f"  mode   : {'binary-only' if args.binary_only else 'execute'}"
              f"{'  (--all)' if args.all else ''}")
        print("-" * 68)

    results = []
    with _fixture_server() as (fixture_url, _fixture_root, fixture_wl):
        for name in selected:
            if name not in tools:
                _err(f"unknown tool '{name}' — skipping")
                continue
            if not args.binary_only and name in _CONTENT_DISCOVERY_TOOLS:
                # ffuf needs FUZZ in the URL; gobuster/feroxbuster take a base URL.
                tgt = fixture_url + ("/FUZZ" if name == "ffuf" else "")
                r = run_tool(name, tools[name], args.timeout, False,
                             target_override=tgt, wordlist_override=fixture_wl,
                             expect_fixture=_FIXTURE_FILENAME)
            else:
                r = run_tool(name, tools[name], args.timeout, args.binary_only)
            results.append(r)
            if not args.json:
                icon = {"PASS": "✅", "FAIL": "❌", "SKIP": "⏭", "WARN": "⚠️", "MANUAL": "🔒"}[r["verdict"]]
                stat = f"{r['secs']:5.1f}s {r['bytes']:>7d}B" if r["secs"] else " " * 14
                line = f"{icon} {name:15} {stat}  {r['detail']}"
                print(line)
                if r["verdict"] == "FAIL" and r.get("cmd"):
                    print(f"   ↳ {r['cmd']}")

    counts: dict[str, int] = {}
    for r in results:
        counts[r["verdict"]] = counts.get(r["verdict"], 0) + 1

    if args.json:
        print(json.dumps({"results": results, "summary": counts}, indent=2))
    else:
        print("-" * 68)
        summary = ", ".join(f"{v} {k}" for k, v in sorted(counts.items()))
        print(f"summary: {summary}")

    # Fail-fast: FAIL (and WARN-on-empty counts as attention but not a hard fail)
    return 1 if counts.get("FAIL") else 0


if __name__ == "__main__":
    raise SystemExit(main())
