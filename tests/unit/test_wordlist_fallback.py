"""Tests: missing wordlists fall back to the bundled list.

From a real scan: ffuf/gobuster/feroxbuster all failed because SecLists/dirb
wordlists were not installed (`/usr/share/wordlists/dirb/common.txt` missing).
The agent then burned many turns probing the filesystem for wordlists instead of
fuzzing. `build_command_args` now substitutes the bundled
`data/wordlists/common.txt` for any `wordlist` param whose path does not exist.

Run:
    pytest tests/unit/test_wordlist_fallback.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app.tools.loader import BUNDLED_WORDLIST, _resolve_wordlist, load_all_tools


class TestResolveWordlist:
    def test_missing_path_substitutes_bundled(self):
        assert BUNDLED_WORDLIST.is_file(), "bundled wordlist must ship with the repo"
        assert _resolve_wordlist("/usr/share/seclists/does/not/exist.txt") == str(BUNDLED_WORDLIST)

    def test_existing_path_kept(self):
        assert _resolve_wordlist("/etc/hostname") == "/etc/hostname"

    def test_empty_value_returns_bundled(self):
        # No value at all still resolves to something usable.
        assert _resolve_wordlist(None) == str(BUNDLED_WORDLIST)


class TestBuildCommandArgs:
    def test_ffuf_default_wordlist_is_resolved(self):
        tools = load_all_tools(force_reload=True)
        tool = tools.get("ffuf")
        assert tool is not None
        cmd = tool.build_command_args(target="https://example.com/FUZZ")
        assert str(BUNDLED_WORDLIST) in cmd

    def test_gobuster_default_wordlist_is_resolved(self):
        tools = load_all_tools(force_reload=True)
        tool = tools.get("gobuster")
        assert tool is not None
        cmd = tool.build_command_args(target="https://example.com")
        assert str(BUNDLED_WORDLIST) in cmd

    def test_caller_supplied_missing_wordlist_is_resolved(self):
        tools = load_all_tools(force_reload=True)
        tool = tools.get("ffuf")
        assert tool is not None
        cmd = tool.build_command_args(
            target="https://example.com/FUZZ",
            wordlist="/usr/share/wfuzz/wordlist/general/common.txt",
        )
        assert str(BUNDLED_WORDLIST) in cmd
