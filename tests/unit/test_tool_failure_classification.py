"""Tests: three failure classifiers that decide whether a tool call "worked".

Bug #1 — the retry guard could not see two REAL failures from a live scan log:

    whatweb  `/usr/bin/whatweb: unrecognized option `--output-format'`   exit 0 (!)
    sqlmap   `sqlmap: error: missing a mandatory option (-d, -u, ...)`   exit 2

  1. `_TOOL_FAIL_MARKERS` (react_agent) had neither "unrecognized option" nor
     "missing a mandatory option", so `_looks_like_failure()` returned False.
  2. Its scan window was 800 chars; sqlmap's banner + disclaimer pushes the
     error line past it, so even a matching marker could be missed.
  3. Nothing used wall-time, so a <1s banner-only run (whatweb exits 0, 149 B)
     looked like a SUCCESS — which RESET `tool_fail_counts` to 0. A phantom
     run therefore not only escaped the block, it refreshed the tool's credit.

Layers under test:
    react_agent._looks_like_failure          — retry guard (per-tool counter)
    tool_bridge._looks_like_short_run_failure — short-run/banner-only classifier
    tool_call_caps.looks_like_invocation_error — W22 hard block (invocation budget)

Run:
    ./venv/bin/python -m pytest tests/unit/test_tool_failure_classification.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app.agents.react_agent import _looks_like_failure
from app.agents.tool_bridge import (
    _SHORT_RUN_MAX_CHARS,
    _SHORT_RUN_SECONDS,
    _SHORT_RUN_WARNING,
    _looks_like_short_run_failure,
)
from app.agents.tool_call_caps import looks_like_invocation_error


# ── Captured VERBATIM from the real binaries on this host ─────────────
WHATWEB_UNRECOGNIZED_OPTION = (
    "/usr/bin/whatweb: unrecognized option `--output-format'\n"
    "Error in processing commandline options - unrecognized option `--output-format'\n"
)

SQLMAP_MISSING_TARGET = (
    "        ___\n"
    "       __H__\n"
    " ___ ___[(]_____ ___ ___  {1.10.4#stable}\n"
    "|_ -| . [(]     | .'| . |\n"
    "|___|_  [(]_|_|_|__,|  _|\n"
    "      |_|V...       |_|   https://sqlmap.org\n"
    "\n"
    "Usage: python3 sqlmap [options]\n"
    "\n"
    "sqlmap: error: missing a mandatory option (-d, -u, -l, -m, -r, -g, -c, "
    "--wizard, --shell, --update, --purge, --list-tampers or --dependencies). "
    "Use -h for basic and -hh for advanced help\n"
)

# A normal, successful result — must never be classified as a failure.
NORMAL_OUTPUT = (
    "[smoke-test-detector] [http] [high] http://example.com/x.html\n"
    "http://example.com/ [200 OK] Title[Example Domain] Country[US]\n"
) * 40


class TestRetryGuardMarkers:
    """react_agent._looks_like_failure — the per-tool retry counter."""

    def test_whatweb_unrecognized_option(self):
        assert _looks_like_failure(WHATWEB_UNRECOGNIZED_OPTION) is True

    def test_sqlmap_missing_mandatory_option(self):
        assert _looks_like_failure(SQLMAP_MISSING_TARGET) is True

    def test_sqlmap_error_after_800_char_banner(self):
        """The old 800-char window would have classified this as SUCCESS."""
        padded = ("_" * 900) + SQLMAP_MISSING_TARGET
        assert _looks_like_failure(padded) is True

    def test_argparse_unrecognized_arguments(self):
        assert _looks_like_failure("dalfox: error: unrecognized arguments: --x") is True

    def test_usage_banner_is_trusted_when_short(self):
        assert _looks_like_failure("Usage: dalfox [options]") is True

    def test_usage_marker_ignored_in_long_output(self):
        """A crawled docs page contains 'usage:' but is not a CLI error."""
        noisy = NORMAL_OUTPUT + "usage: run the tool like this\n"
        assert len(noisy) > 4_000
        assert _looks_like_failure(noisy) is False

    def test_normal_findings_not_a_failure(self):
        assert _looks_like_failure(NORMAL_OUTPUT) is False

    def test_empty_output_not_a_failure(self):
        # Emptiness is handled by the [empty_result] branch, not this guard.
        assert _looks_like_failure("") is False


class TestShortRunClassifier:
    """tool_bridge._looks_like_short_run_failure — wall-time + size + signal."""

    def test_whatweb_phantom_run_is_flagged(self):
        """Fast, tiny, exit 0 — the case exit-code checks cannot catch."""
        assert _looks_like_short_run_failure(
            WHATWEB_UNRECOGNIZED_OPTION, duration_s=0.3, exit_code=0
        ) is True

    def test_sqlmap_phantom_run_is_flagged(self):
        assert _looks_like_short_run_failure(
            SQLMAP_MISSING_TARGET, duration_s=1.1, exit_code=2
        ) is True

    def test_legitimate_fast_small_result_not_flagged(self):
        """httpx on one host answers in ~1s with a few hundred bytes."""
        body = "http://example.com [200] [Example Domain] [nginx]\n" * 8
        assert _looks_like_short_run_failure(body, duration_s=1.2, exit_code=0) is False

    def test_slow_run_not_flagged(self):
        assert _looks_like_short_run_failure(
            WHATWEB_UNRECOGNIZED_OPTION, duration_s=30.0, exit_code=0
        ) is False

    def test_large_output_not_flagged(self):
        body = WHATWEB_UNRECOGNIZED_OPTION + ("x" * (_SHORT_RUN_MAX_CHARS + 1))
        assert _looks_like_short_run_failure(body, duration_s=0.3, exit_code=0) is False

    def test_unknown_duration_not_flagged(self):
        """duration_seconds == 0 means "never measured" — stay silent."""
        assert _looks_like_short_run_failure(
            WHATWEB_UNRECOGNIZED_OPTION, duration_s=0.0, exit_code=0
        ) is False

    def test_empty_result_keeps_its_own_explanation(self):
        body = "[empty_result] stdout: 0 bytes, stderr: 0 bytes, exit_code: 0."
        assert _looks_like_short_run_failure(body, duration_s=0.4, exit_code=0) is False

    def test_constants_sanity(self):
        assert _SHORT_RUN_SECONDS == 5.0
        assert _SHORT_RUN_MAX_CHARS == 2048
        assert "Treat as NO RESULT" in _SHORT_RUN_WARNING
        assert "check target/-u" in _SHORT_RUN_WARNING


class TestInvocationErrorPatterns:
    """tool_call_caps.looks_like_invocation_error — W22 hard-block budget."""

    def test_whatweb_unrecognized_option(self):
        assert looks_like_invocation_error(WHATWEB_UNRECOGNIZED_OPTION) is True

    def test_sqlmap_missing_mandatory_option(self):
        assert looks_like_invocation_error(SQLMAP_MISSING_TARGET) is True

    def test_sqlmap_error_after_800_char_banner(self):
        padded = ("_" * 900) + SQLMAP_MISSING_TARGET
        assert looks_like_invocation_error(padded) is True

    def test_normal_output_not_an_invocation_error(self):
        assert looks_like_invocation_error(NORMAL_OUTPUT) is False

    def test_pre_existing_signatures_still_match(self):
        for text in (
            "Incorrect Usage: flag provided but not defined: -nc",
            "error: unexpected argument '--x' found",
            'invalid value "240s" for flag -maxtime: parse error',
            "unknown shorthand flag: 'x' in -x",
        ):
            assert looks_like_invocation_error(text) is True, text
