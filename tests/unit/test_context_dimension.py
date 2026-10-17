"""
Unit tests — TDA C dimension measures CONTEXT SATURATION, not spend.

The bug: C was `total_tokens_so_far / TOKEN_HARD_LIMIT`. That measures the
scan's budget (money), not how full the model's context window is, and it
rises with the number of turns even when every prompt is tiny. The result
was an inverted incentive — a long, expensive scan pushed C up, pushed the
TDI past TDI_BROADEN, and forced BROADEN, i.e. it broadened reconnaissance
exactly when the scan was already the most expensive it had ever been.

C must be `prompt_tokens(latest request) / model_context_window`.
"""
from __future__ import annotations

import pytest

from app.rl.tda import (
    MODEL_WEIGHTS_SUM,
    Mode,
    compute_tdi,
    context_load,
    context_saturation,
)


class TestContextSaturation:
    def test_zero_tokens_is_zero(self):
        assert context_saturation(0, 128_000) == 0.0

    def test_half_full_window(self):
        assert context_saturation(64_000, 128_000) == pytest.approx(0.5)

    def test_clips_at_one(self):
        """A prompt larger than the window must not return C > 1."""
        assert context_saturation(500_000, 128_000) == 1.0

    def test_unknown_window_is_zero(self):
        """A misconfigured window must not fake a saturated context."""
        assert context_saturation(50_000, 0) == 0.0
        assert context_saturation(50_000, -1) == 0.0

    def test_negative_tokens_is_zero(self):
        assert context_saturation(-10, 128_000) == 0.0


class TestComputeTdiUsesSaturation:
    def _tdi(self, **kwargs):
        base = dict(
            remaining_steps=50,
            path_evidence=[0.5],
            successes=0,
            attempts=10,
            max_horizon=100,
        )
        base.update(kwargs)
        return compute_tdi(**base)

    def test_utilisation_drives_c(self):
        result = self._tdi(context_utilisation=0.5)
        assert result.c == pytest.approx(0.5)

    def test_utilisation_clipped(self):
        assert self._tdi(context_utilisation=3.0).c == 1.0
        assert self._tdi(context_utilisation=-1.0).c == 0.0

    def test_legacy_pair_still_supported(self):
        """Older callers pass tokens_used + context_window — still works."""
        result = self._tdi(tokens_used=1_000, context_window=10_000)
        assert result.c == pytest.approx(0.1)
        assert context_load(1_000, 10_000) == pytest.approx(0.1)

    def test_no_context_signal_raises(self):
        """Silently defaulting C=0 would hide a wiring bug."""
        with pytest.raises(ValueError):
            self._tdi()

    def test_utilisation_wins_over_legacy_pair(self):
        result = self._tdi(
            context_utilisation=0.25,
            tokens_used=5_000,
            context_window=10_000,   # legacy would say 0.5
        )
        assert result.c == pytest.approx(0.25)


class TestSpendNoLongerMovesC:
    def test_long_scan_with_small_prompts_stays_flat(self):
        """THE regression: same prompt size, wildly different spend → same TDI.

        Before the fix these two calls produced different C (and therefore
        different TDI): the expensive one looked "harder" purely because it
        had spent more money.
        """
        common = dict(
            remaining_steps=50,
            path_evidence=[0.5],
            successes=0,
            attempts=10,
            max_horizon=100,
            context_utilisation=0.1,   # 12.8K prompt on a 128K window
        )
        cheap = compute_tdi(**common)
        # 8M tokens spent — but the LAST prompt was still only 12.8K.
        expensive = compute_tdi(tokens_used=8_000_000, context_window=10_000_000, **common)
        assert cheap.c == expensive.c == pytest.approx(0.1)
        assert cheap.tdi == pytest.approx(expensive.tdi)

    def test_stale_definition_would_have_forced_broaden(self):
        """Show the old definition's failure mode explicitly.

        C=0.8 (80% of the 10M budget burnt) with the other dimensions mild
        pushed the TDI over the broaden threshold. The same scan with a
        modest prompt (C=0.05) does not.
        """
        common = dict(
            remaining_steps=80,
            path_evidence=[0.5],
            successes=1,
            attempts=10,
            max_horizon=100,
        )
        stale = compute_tdi(tokens_used=8_000_000, context_window=10_000_000, **common)
        fixed = compute_tdi(context_utilisation=0.05, **common)
        assert stale.c == pytest.approx(0.8)
        assert fixed.c == pytest.approx(0.05)
        assert stale.tdi > fixed.tdi
        # The saturation-based value keeps the scan out of BROADEN here.
        assert fixed.mode is Mode.LLM_PICK
        assert stale.mode is Mode.BROADEN


class TestWeightsUnchanged:
    def test_weights_still_sum_to_one(self):
        assert MODEL_WEIGHTS_SUM == pytest.approx(1.0)
