"""
Shared ANSI / VT100 escape code stripper.

Why this exists:
----------------
Security tools (sqlmap, metasploit/msfconsole, nuclei, feroxbuster, nmap NSE)
emit ANSI color codes, cursor-move sequences, OSC title sequences, and bare
CR (carriage return) bytes even when stdout is NOT a TTY. Without stripping,
these leak into:

  - LLM context (wastes tokens + confuses the model into thinking the output
    was truncated at the first color escape)
  - SSE event payloads (tool_call_progress observation, tool_call_completed
    result_preview) → frontend renders raw `\x1b[0m`, `\x1b[1m[33m` as
    literal garbled text inside <pre> blocks
  - SQLite process_details.data.result column (permanent garbage in DB)

This regex covers the four ANSI families that show up in real tool output:

  1. CSI sequences        — `\x1b[<params>;<intermediate><final>`
                            final byte in [@-~], covers colors, cursor moves,
                            erase, scroll, SGR (the most common)
  2. OSC sequences        — `\x1b]<data>\x07` or `\x1b]<data>\x1b\\`
                            (terminal title set — common in bash prompts)
  3. 2-char escapes       — `\x1b<@-Z\\-_>`  (rare, but real)
  4. Bare CR              — `\r` not followed by `\n` (progress bars,
                            e.g. `\rdone 12%\rdone 50%`)

Usage:
    from app.harness.ansi import strip_ansi
    clean = strip_ansi(raw_subprocess_stdout)

Idempotent: re-running on already-clean text is a no-op.
"""
from __future__ import annotations

import re

# Single compiled regex — compiled once at import time.
_ANSI_ESCAPE_RE = re.compile(
    r"""
    \x1b\[[0-?]*[ -/]*[@-~]              # CSI ... final byte (colors, cursor, SGR)
    | \x1b\][^\x07\x1b]*(?:\x07|\x1b\\)  # OSC ... BEL or ST
    | \x1b[@-Z\\-_]                       # 2-char escape
    | \r(?=[^\n])                         # bare CR (progress bars) — not CRLF
    """,
    re.VERBOSE,
)


def strip_ansi(text: str | None) -> str:
    """Remove ANSI escape sequences from `text`.

    Returns an empty string if `text` is None or empty.
    Safe to call on non-string input — returns "" (defensive).

    Idempotent: applying twice produces the same output as applying once.
    """
    if not text:
        return ""
    return _ANSI_ESCAPE_RE.sub("", text)


__all__ = ["strip_ansi", "_ANSI_ESCAPE_RE"]