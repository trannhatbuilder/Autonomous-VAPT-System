"""
Unit tests — `HarnessBridge.tda_compute` wiring for the context dimension.

Verifies the bridge actually passes the provider's LAST-request prompt
tokens (minus the harness's own advisory injections) instead of the
cumulative scan spend, and that no other TDI dimension is disturbed.
"""
from __future__ import annotations

import asyncio
import contextlib

import pytest
import pytest_asyncio

from app.rl.harness import MODEL_CONTEXT_WINDOW_DEFAULT, HarnessBridge

# HarnessBridge spawns a background training worker in __init__, so it must
# be constructed inside a running event loop — hence the async fixture.


@pytest_asyncio.fixture
async def bridge(monkeypatch):
    monkeypatch.setenv("VAPT_AI_TDA", "1")
    instance = HarnessBridge(scan_id="scan_tdi_ctx", target="http://t.test")
    try:
        yield instance
    finally:
        task = getattr(instance, "_train_worker_task", None)
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task


def _compute(bridge, **kwargs):
    base = dict(
        turn=10,
        max_turns=100,
        findings=[],
        successes=0,
        attempts=10,
    )
    base.update(kwargs)
    return bridge.tda_compute(**base)


class TestPromptTokensDriveC:
    def test_half_window(self, bridge):
        result = _compute(bridge, prompt_tokens=64_000, model_context_window=128_000)
        assert result is not None
        assert result.c == pytest.approx(0.5)

    def test_default_window_is_128k(self, bridge):
        assert MODEL_CONTEXT_WINDOW_DEFAULT == 128_000
        result = _compute(bridge, prompt_tokens=12_800)
        assert result.c == pytest.approx(0.1)

    def test_no_prompt_tokens_yields_zero_c(self, bridge):
        """A provider that omits usage must not fabricate a full context."""
        result = _compute(bridge, prompt_tokens=0)
        assert result is not None
        assert result.c == 0.0

    def test_advisory_tokens_are_excluded(self, bridge):
        """The TDA hint must not inflate the difficulty signal it reads."""
        base = _compute(bridge, prompt_tokens=64_000, model_context_window=128_000)
        with_advisory = _compute(
            bridge,
            prompt_tokens=64_000,
            model_context_window=128_000,
            advisory_tokens=32_000,
        )
        assert base.c == pytest.approx(0.5)
        assert with_advisory.c == pytest.approx(0.25)

    def test_advisory_tokens_cannot_go_negative(self, bridge):
        result = _compute(bridge, prompt_tokens=100, advisory_tokens=5_000)
        assert result.c == 0.0

    def test_spend_does_not_move_c(self, bridge):
        """`tokens_used` is a billing counter only."""
        cheap = _compute(bridge, prompt_tokens=12_800, tokens_used=10_000)
        expensive = _compute(bridge, prompt_tokens=12_800, tokens_used=8_000_000)
        assert cheap.c == pytest.approx(expensive.c)
        assert cheap.tdi == pytest.approx(expensive.tdi)


class TestOtherDimensionsUnchanged:
    def test_horizon_still_drives_tdi(self, bridge):
        """H is REMAINING horizon: a fresh scan carries more uncertainty."""
        early = _compute(bridge, turn=5, prompt_tokens=1_000)
        late = _compute(bridge, turn=95, prompt_tokens=1_000)
        assert early.h > late.h
        assert early.tdi > late.tdi

    def test_dimensions_match_recorded_values(self, bridge):
        """C is the only thing the redefinition touched."""
        result = _compute(bridge, prompt_tokens=64_000)
        assert result is not None
        assert result.c == pytest.approx(0.5)
        assert result.tdi == pytest.approx(
            0.30 * result.h + 0.30 * (1 - result.e) + 0.20 * result.c + 0.20 * (1 - result.s)
        )
