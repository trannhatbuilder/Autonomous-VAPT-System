"""Tests: tool output sent to the LLM is ANSI-clean and adequately sized.

Rationale (from a real scan): sqlmap/feroxbuster emit ANSI colour codes even
without a TTY. The LLM reported "output is truncated at the first ANSI colour
escape" and burned multiple turns trying to "disable colour" instead of reading
the results. We now strip ANSI/VT100 sequences and keep up to 8k chars of output
instead of 4k.

Run:
    pytest tests/unit/test_tool_output_sanitization.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app.agents.tool_bridge import MAX_LLM_OUTPUT_CHARS, _strip_ansi


class TestStripAnsi:
    def test_csi_colour_codes_removed(self):
        assert _strip_ansi("\x1b[31mHIGH\x1b[0m risk") == "HIGH risk"

    def test_cursor_and_erase_sequences_removed(self):
        assert _strip_ansi("\x1b[2J\x1b[1;1Hclean") == "clean"

    def test_osc_title_sequence_removed(self):
        assert _strip_ansi("a\x1b]0;window title\x07b") == "ab"

    def test_bare_carriage_return_progress_removed(self):
        assert _strip_ansi("10%\r90%\n") == "10%90%\n"

    def test_crlf_preserved(self):
        assert _strip_ansi("line1\r\nline2") == "line1\r\nline2"

    def test_plain_text_untouched(self):
        text = "nmap: 4280/tcp open  ssl/http"
        assert _strip_ansi(text) == text

    def test_empty_string(self):
        assert _strip_ansi("") == ""

    def test_realistic_sqlmap_banner(self):
        raw = "\x1b[33m[\x1b[0m\x1b[36minfo\x1b[0m] testing connection"
        assert _strip_ansi(raw) == "[info] testing connection"


class TestOutputBudget:
    def test_budget_is_larger_than_legacy_4k(self):
        assert MAX_LLM_OUTPUT_CHARS >= 8000
