"""
Regression guard — the `harness.tda_compute()` CALL SITE vs its SIGNATURE.

Why this file exists
--------------------
Round 4 renamed a `tda_compute()` parameter (`context_window` ->
`model_context_window`) and added `prompt_tokens`. ALL of its parameters are
keyword-only (`*`), so a caller left behind raises

    TypeError: tda_compute() got an unexpected keyword argument 'context_window'

...every single turn. That exception is caught by a broad `except Exception`
in the agent loop, which only logs `tda_compute_failed` and sets
`tda_hint_msg = None`. The damage is therefore INVISIBLE:

  * the TDA hint is never injected -> the "tda" round-robin slot is an empty
    turn,
  * `harness.tdi_log` stays empty forever -> the RL state encoder's
    `tdi_log[-1]` read always falls back to its default,
  * EGATS `record_outcome` uses its 0.5 TDI fallback for every path,
  * i.e. every TDA fix silently does not exist in production.

Static type checkers and the existing behavioural tests do NOT catch this:
the tests call `tda_compute()` directly with correct kwargs, and the call
site is a plain attribute access on a loosely-typed `Any`. So we assert the
two sides agree by READING THE SOURCE.

If you rename or add a parameter, these tests fail until the call site is
updated too — which is exactly the point.
"""
from __future__ import annotations

import ast
import asyncio
import contextlib
import inspect
from pathlib import Path

import pytest
import pytest_asyncio

from app.rl.harness import MODEL_CONTEXT_WINDOW_DEFAULT, HarnessBridge

REPO_ROOT = Path(__file__).resolve().parents[2]
CALL_SITE_FILE = REPO_ROOT / "app" / "agents" / "react_agent.py"


def _tda_call_site() -> ast.Call:
    """Locate the production `harness.tda_compute(...)` call (must be unique)."""
    tree = ast.parse(CALL_SITE_FILE.read_text(encoding="utf-8"))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "tda_compute"
    ]
    assert len(calls) == 1, (
        f"expected exactly 1 tda_compute() call site in {CALL_SITE_FILE.name}, "
        f"found {len(calls)} at lines {[c.lineno for c in calls]}"
    )
    return calls[0]


def _call_site_kwargs() -> list[str]:
    call = _tda_call_site()
    names = []
    for keyword in call.keywords:
        assert keyword.arg is not None, (
            f"{CALL_SITE_FILE.name}:{call.lineno} expands **kwargs into "
            "tda_compute(); this guard cannot verify it statically"
        )
        names.append(keyword.arg)
    return names


def _accepted_params() -> list[str]:
    """tda_compute's parameter names, WITHOUT `self`.

    `inspect.signature()` on the unbound function keeps `self`, so every
    assertion below would otherwise have to special-case it.
    """
    sig = inspect.signature(HarnessBridge.tda_compute)
    return [name for name in sig.parameters if name != "self"]


def _required_params() -> set[str]:
    sig = inspect.signature(HarnessBridge.tda_compute)
    return {
        p.name
        for p in sig.parameters.values()
        if p.default is inspect.Parameter.empty
        and p.kind in (p.KEYWORD_ONLY, p.POSITIONAL_OR_KEYWORD)
        and p.name != "self"
    }


class TestCallSiteMatchesSignature:
    def test_no_unexpected_kwargs(self):
        """The exact TypeError from the partial refactor."""
        params = set(_accepted_params())
        unexpected = [n for n in _call_site_kwargs() if n not in params]
        assert unexpected == [], (
            f"tda_compute() call site passes unknown kwargs {unexpected}. "
            f"Accepted: {sorted(params)}"
        )

    def test_all_required_kwargs_present(self):
        missing = _required_params() - set(_call_site_kwargs())
        assert missing == set(), f"call site omits required kwargs {missing}"

    def test_signature_binds_with_call_site_kwargs(self):
        """The definitive check: what TypeError would be raised at runtime."""
        sig = inspect.signature(HarnessBridge.tda_compute)
        provided = {name: 1 for name in _required_params()}
        provided.update({name: 1 for name in _call_site_kwargs()})
        sig.bind(None, **provided)  # None stands in for `self`

    def test_removed_kwarg_is_gone(self):
        """`context_window` was renamed; passing it must never come back."""
        assert "context_window" not in _call_site_kwargs()
        assert "context_window" not in _accepted_params()

    def test_call_site_wires_the_context_signal(self):
        """C's inputs must actually reach tda_compute from the loop."""
        kwargs = _call_site_kwargs()
        assert "prompt_tokens" in kwargs
        assert "model_context_window" in kwargs


class TestCallSiteKwargsExecute:
    """Same kwarg NAMES as the call site, executed for real.

    This is what makes the guard stronger than a static bind: it proves the
    body runs without raising, so the broad `except Exception` in the agent
    loop has nothing to swallow.
    """

    @pytest_asyncio.fixture
    async def bridge(self, monkeypatch):
        monkeypatch.setenv("VAPT_AI_TDA", "1")
        instance = HarnessBridge(scan_id="scan_call_site", target="http://t.test")
        try:
            yield instance
        finally:
            task = getattr(instance, "_train_worker_task", None)
            if task is not None:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await task

    def test_kwargs_execute_and_populate_tdi_log(self, bridge):
        values = {
            "turn": 5,
            "max_turns": 100,
            "findings": [],
            "prompt_tokens": 64_000,
            "model_context_window": MODEL_CONTEXT_WINDOW_DEFAULT,
            "tokens_used": 120_000,
            "successes": 0,
            "attempts": 7,
            "advisory_tokens": 1_200,
        }
        kwargs = _call_site_kwargs()
        unmapped = [n for n in kwargs if n not in values]
        assert unmapped == [], (
            f"call site gained kwargs {unmapped} — add sample values here so the "
            "runtime guard keeps covering them"
        )

        # Patch the fixture's bridge into the dict we invoke.
        result = bridge.tda_compute(**{name: values[name] for name in kwargs})

        assert result is not None
        # C = (prompt_tokens - advisory_tokens) / window, per Round 4.
        assert result.c == pytest.approx((64_000 - 1_200) / 128_000)
        assert len(bridge.tdi_log) == 1
        assert bridge.tdi_log[-1]["mode"] == result.mode.value
        # The hint the loop would inject must render.
        assert "TDA" in bridge.tda_render_hint(result)
