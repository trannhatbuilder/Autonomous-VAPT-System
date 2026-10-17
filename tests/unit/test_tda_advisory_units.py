"""
Regression guard — the TDA C dimension must not decode to 0 late in a scan.

C's numerator is `prompt_tokens(last request) - advisory_tokens`. The
advisory term exists so the harness's own hint text cannot inflate the
difficulty signal it feeds. That only works if BOTH terms describe the SAME
request.

The bug this file pins: the loop passed the CUMULATIVE advisory total
(`advisory_tokens_injected`, grown with `+=` every turn) while the prompt
tokens were per-request (bounded by history pruning). The cumulative total
wins after a few dozen turns, `max(0, prompt - advisory)` becomes 0, and C
reads 0 for the rest of the scan — silently disabling the dimension it was
meant to protect.

The per-turn counter (`advisory_tokens_last_turn`) is the fix.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

from app.rl.harness import MODEL_CONTEXT_WINDOW_DEFAULT, HarnessBridge

REPO_ROOT = Path(__file__).resolve().parents[2]
AGENT_FILE = REPO_ROOT / "app" / "agents" / "react_agent.py"

PROMPT_PER_REQUEST = 15_000     # bounded by history pruning
ADVISORY_PER_TURN = 1_000


class TestCumulativeAdvisoryWouldKillC:
    """Why the distinction matters, in arithmetic."""

    @staticmethod
    def _c(prompt_tokens: int, advisory_tokens: int) -> float:
        # Mirrors HarnessBridge.tda_compute's C computation.
        content = max(0, prompt_tokens - max(0, advisory_tokens))
        if MODEL_CONTEXT_WINDOW_DEFAULT <= 0 or content <= 0:
            return 0.0
        return min(1.0, content / MODEL_CONTEXT_WINDOW_DEFAULT)

    def test_cumulative_counter_collapses(self):
        cumulative = ADVISORY_PER_TURN * 20          # turn 20
        assert self._c(PROMPT_PER_REQUEST, cumulative) == 0.0

    def test_per_turn_counter_survives(self):
        c = self._c(PROMPT_PER_REQUEST, ADVISORY_PER_TURN)
        assert c == pytest.approx((PROMPT_PER_REQUEST - ADVISORY_PER_TURN) / 128_000)


def _loop_body() -> ast.AsyncFunctionDef:
    """Find `run_react_scan` without importing it (it needs an LLM)."""
    tree = ast.parse(AGENT_FILE.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "run_react_scan":
            return node
    raise AssertionError("run_react_scan not found")


def _tda_call_kwarg(name: str) -> ast.expr | None:
    for node in ast.walk(_loop_body()):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "tda_compute"
        ):
            for keyword in node.keywords:
                if keyword.arg == name:
                    return keyword.value
    return None


class TestCallSiteUsesPerTurnAdvisory:
    def test_advisory_kwarg_is_not_the_cumulative_counter(self):
        value = _tda_call_kwarg("advisory_tokens")
        assert value is not None, "tda_compute() must receive advisory_tokens"
        assert isinstance(value, ast.Name)
        assert value.id != "advisory_tokens_injected", (
            "tda_compute(advisory_tokens=) must receive the PER-TURN value; "
            "the cumulative counter drives C to 0 late in the scan"
        )
        assert value.id == "advisory_tokens_last_turn"

    def test_per_turn_counter_is_assigned_each_turn(self):
        """`+=` on the cumulative counter must be paired with a per-turn set."""
        src = AGENT_FILE.read_text(encoding="utf-8")
        assert "advisory_tokens_last_turn = turn_advisory_tokens" in src
        assert "advisory_tokens_injected += turn_advisory_tokens" in src

    def test_prompt_tokens_is_still_wired(self):
        """B2's concern: the C numerator must actually reach tda_compute."""
        assert _tda_call_kwarg("prompt_tokens") is not None
        assert _tda_call_kwarg("model_context_window") is not None


class TestHarnessSubtractsOnlyGivenValue:
    """The harness does exact arithmetic — it trusts the caller's unit."""

    def test_single_request_cancels_cleanly(self):
        c = TestCumulativeAdvisoryWouldKillC._c(64_000, 1_200)
        assert c == pytest.approx((64_000 - 1_200) / 128_000)
