"""
Regression tests for app.rl.tda.

Covers two bugs found in the TDA integration:

Bug #1 — Horizon saturation:
    DEFAULT_MAX_HORIZON was 20 while the agent loop runs 100 turns.
    remaining_steps / 20 clamped H to 1.0 for ~80% of the scan,
    effectively deleting one of the four TDI dimensions. The horizon
    denominator must be the scan's real budget (resolve_max_horizon).

Bug #2 — Formula-driven EXPLOIT was dead code:
    With no confirmed finding, E <= 0.5 and S -> 0, so

        TDI >= 0.30*(1-0.5) + 0.20*(1-0) = 0.35 > 0.30

    which made `TDI < 0.30 -> EXPLOIT` mathematically unreachable;
    once a finding exists the critical override fires first. The formula
    branch was removed — EXPLOIT is override-only.

Bug #3 — Override was gated on finding COUNT:
    `len(findings) > 0` let a single plausible/hallucinated finding
    (evidence 0.5) force EXPLOIT, tunnelling the whole scan onto a
    vector that may not exist. The gate is now `should_force_exploit()`:
    evidence >= CONFIRMED_VULN (0.8).

Bug #4 — The override fabricated tdi=0.20 / e=0.7 / s=0.7:
    After the first finding the RL state encoder saw a constant fake
    difficulty, so it could no longer distinguish states. The override
    now preserves the raw tdi/h/e/c/s and only sets mode + overridden.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure project root is on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Set test env BEFORE importing app modules
os.environ.setdefault("VAPT_AI_POSTGRES_DB", "postgresql://vapt:vapt@localhost:5432/vapt_ai_test")
os.environ.setdefault("VAPT_AI_ENVIRONMENT", "dev")

import pytest

from app.rl.tda import (
    CONFIRMED_VULN,
    Mode,
    PLAUSIBLE_HYPOTHESIS,
    SPECULATIVE_HYPOTHESIS,
    TDI_BROADEN,
    TDIResult,
    VERIFIED_EXPLOIT,
    W_E,
    W_S,
    compute_tdi,
    evidence_score_from_finding,
    historical_success,
    horizon_normalised,
    max_evidence_score,
    render_tda_hint,
    resolve_max_horizon,
    should_force_exploit,
)


# ── Bug #1: horizon must use the real scan budget ──────────────────────


class TestHorizonNormalisation:
    def test_horizon_tracks_real_scan_budget(self):
        """A 100-turn scan must decay H linearly from ~1.0 to 0.0."""
        assert horizon_normalised(99, 100) == pytest.approx(0.99)
        assert horizon_normalised(50, 100) == pytest.approx(0.50)
        assert horizon_normalised(1, 100) == pytest.approx(0.01)
        assert horizon_normalised(0, 100) == 0.0

    def test_old_fixed_horizon_saturates(self):
        """Documents bug #1: horizon 20 clamped H for 80 of 100 turns."""
        assert horizon_normalised(99, 20) == 1.0
        assert horizon_normalised(20, 20) == 1.0
        assert horizon_normalised(19, 20) == pytest.approx(0.95)

    def test_non_positive_horizon_raises(self):
        """A misconfigured budget must fail loudly, not look 'easy'."""
        with pytest.raises(ValueError):
            horizon_normalised(10, 0)
        with pytest.raises(ValueError):
            horizon_normalised(10, -1)


class TestResolveMaxHorizon:
    def test_auto_uses_max_turns(self):
        assert resolve_max_horizon(100, 0) == 100
        assert resolve_max_horizon(100, None) == 100

    def test_smaller_override_is_clamped_to_budget(self):
        """Never allow a denominator below the real budget (saturation)."""
        assert resolve_max_horizon(100, 20) == 100
        assert resolve_max_horizon(50, 1) == 50

    def test_larger_override_is_honoured(self):
        assert resolve_max_horizon(100, 200) == 200

    def test_invalid_max_turns_raises(self):
        with pytest.raises(ValueError):
            resolve_max_horizon(0, 100)


# ── Bug #2: EXPLOIT is override-only ───────────────────────────────────


class TestExploitUnreachableWithoutFindings:
    def test_theoretical_floor_is_035(self):
        """Reproduce the proof: E <= 0.5, S -> 0 => TDI >= 0.35."""
        floor = 0.5 * W_E + W_S
        assert floor == pytest.approx(0.35)

        result = compute_tdi(
            remaining_steps=0,
            path_evidence=[0.5],       # best evidence without a finding
            tokens_used=0,
            context_window=1_000_000,  # C = 0
            successes=0,
            attempts=10_000,           # S -> 0
            max_horizon=100,
        )
        assert result.mode is Mode.LLM_PICK
        assert result.tdi == pytest.approx(0.35, abs=1e-3)

    def test_empty_path_floor_is_041(self):
        """No evidence at all (E=0.3) makes the floor even higher."""
        result = compute_tdi(
            remaining_steps=0,
            path_evidence=[],
            tokens_used=0,
            context_window=1_000_000,
            successes=0,
            attempts=10_000,
            max_horizon=100,
        )
        assert result.mode is Mode.LLM_PICK
        assert result.tdi == pytest.approx(0.41, abs=1e-3)

    def test_compute_tdi_never_returns_exploit(self):
        """Sweep no-finding states: BROADEN/LLM_PICK only."""
        for evidence in ([], [0.3], [0.5], [0.3, 0.5]):
            for attempts in (0, 1, 10, 1_000):
                for remaining in (0, 1, 50, 99):
                    result = compute_tdi(
                        remaining_steps=remaining,
                        path_evidence=evidence,
                        tokens_used=0,
                        context_window=1_000_000,
                        successes=0,
                        attempts=attempts,
                        max_horizon=100,
                    )
                    assert result.mode in (Mode.BROADEN, Mode.LLM_PICK)
                    assert result.mode is not Mode.EXPLOIT

    def test_override_renders_forced_exploit(self):
        """The critical override still produces an EXPLOIT advisory."""
        result = TDIResult(
            tdi=0.20,
            h=0.50,
            e=0.80,
            c=0.10,
            s=0.70,
            mode=Mode.EXPLOIT,
            overridden=True,
        )
        text = render_tda_hint(result)
        assert "EXPLOIT" in text
        assert "forced" in text
        assert "confirmed finding" in text

    def test_broaden_threshold_unchanged(self):
        result = compute_tdi(
            remaining_steps=99,
            path_evidence=[],
            tokens_used=0,
            context_window=1_000_000,
            successes=0,
            attempts=0,
            max_horizon=100,
        )
        assert result.tdi > TDI_BROADEN
        assert result.mode is Mode.BROADEN


# ── Bug #3: override gate is EVIDENCE, not count ────────────────


class TestShouldForceExploit:
    def test_empty_findings_does_not_force(self):
        assert should_force_exploit([]) is False

    def test_plausible_only_finding_does_not_force(self):
        """The phantom-SQLi case: recorded but unverified (0.5)."""
        phantom = {"id": "F1", "verified": False}
        assert evidence_score_from_finding(phantom) == PLAUSIBLE_HYPOTHESIS
        assert should_force_exploit([phantom]) is False

    def test_verified_finding_forces(self):
        verified = {"id": "F1", "verified": True}
        assert evidence_score_from_finding(verified) == CONFIRMED_VULN
        assert should_force_exploit([verified]) is True

    def test_one_verified_among_plausible_forces(self):
        findings = [
            {"verified": False},
            {"verified": True},
            {"verified": False},
        ]
        assert should_force_exploit(findings) is True

    def test_verified_false_positive_does_not_force(self):
        fp = {"verified": True, "false_positive": True}
        # A refutation ranks at the FLOOR of the ladder, below an
        # unexamined hypothesis (0.5) — `false_positive` says the claim is
        # FALSE, which is strictly the weakest evidence available.
        assert evidence_score_from_finding(fp) == SPECULATIVE_HYPOTHESIS
        assert should_force_exploit([fp]) is False

    def test_replay_reproduced_forces(self):
        assert should_force_exploit([{"replay_reproduced": True}]) is True
        assert should_force_exploit([{"human_verified": True}]) is True

    def test_custom_threshold(self):
        plausible = {"verified": False}
        assert should_force_exploit([plausible], threshold=PLAUSIBLE_HYPOTHESIS) is True
        assert should_force_exploit([plausible], threshold=VERIFIED_EXPLOIT) is False

    def test_max_evidence_score(self):
        assert max_evidence_score([]) == 0.0
        assert max_evidence_score(
            [{"verified": False}, {"verified": True}]
        ) == CONFIRMED_VULN


# ── Bug #4: the override must not fabricate dimensions ──────────


class TestOverridePreservesRawTDI:
    def test_override_result_keeps_raw_values(self):
        raw = compute_tdi(
            remaining_steps=40,
            path_evidence=[CONFIRMED_VULN],
            tokens_used=1_000,
            context_window=10_000,
            successes=1,
            attempts=10,
            max_horizon=100,
        )
        overridden = TDIResult(
            tdi=raw.tdi,
            h=raw.h,
            e=raw.e,
            c=raw.c,
            s=raw.s,
            mode=Mode.EXPLOIT,
            overridden=True,
            override_evidence=CONFIRMED_VULN,
        )
        assert overridden.tdi == raw.tdi
        assert overridden.h == raw.h
        assert overridden.e == raw.e
        assert overridden.c == raw.c
        assert overridden.s == raw.s
        assert overridden.mode is Mode.EXPLOIT
        assert overridden.overridden is True

    def test_as_dict_reports_override_evidence(self):
        result = TDIResult(
            tdi=0.42, h=0.4, e=0.8, c=0.1, s=0.5,
            mode=Mode.EXPLOIT, overridden=True,
            override_evidence=CONFIRMED_VULN,
        )
        payload = result.as_dict()
        assert payload["mode"] == "exploit"
        assert payload["overridden"] is True
        assert payload["override_evidence"] == CONFIRMED_VULN

    def test_as_dict_override_evidence_none_by_default(self):
        result = TDIResult(
            tdi=0.42, h=0.4, e=0.5, c=0.1, s=0.5, mode=Mode.LLM_PICK,
        )
        assert result.as_dict()["override_evidence"] is None


# ── S: unit consistency + clamping ────────────────


class TestHistoricalSuccess:
    def test_fresh_branch_is_half(self):
        assert historical_success(0, 0) == pytest.approx(0.5)

    def test_monotonic_in_successes(self):
        assert historical_success(0, 10) < historical_success(5, 10)

    def test_clamped_when_successes_exceed_attempts(self):
        """One tool call may yield several findings -> S was > 1.0."""
        s = historical_success(5, 3)
        assert 0.0 <= s <= 1.0
        # successes clamped to attempts: (3+1) / (3+1+1) = 0.8
        assert s == pytest.approx(0.8)

    def test_negative_inputs_do_not_break_bounds(self):
        s = historical_success(-2, 5)
        assert 0.0 <= s <= 1.0
