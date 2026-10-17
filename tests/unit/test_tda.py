"""Unit tests for app/rl/tda.py (VAPT-AI).

Drop into your tests package (e.g. tests/test_tda.py) and run:
    pytest tests/test_tda.py -q

NOTE: test_label_map_replay_verified intentionally encodes the DESIRED
behaviour — it FAILS until you add "replay_verified" to
_EVIDENCE_LABEL_MAP in app/rl/tda.py (review issue #2).
"""
import pytest

from app.rl.tda import (
    CONFIRMED_VULN,
    PLAUSIBLE_HYPOTHESIS,
    SPECULATIVE_HYPOTHESIS,
    VERIFIED_EXPLOIT,
    W_C,
    W_E,
    W_H,
    W_S,
    Mode,
    compute_tdi,
    context_load,
    context_saturation,
    evidence_confidence,
    evidence_score_from_finding,
    evidence_score_from_label,
    historical_success,
    horizon_normalised,
    max_evidence_score,
    resolve_max_horizon,
    should_force_exploit,
)


# ── weight invariant ──────────────────────────────────────────────────
def test_weights_sum_to_one():
    assert abs((W_H + W_E + W_C + W_S) - 1.0) < 1e-9


# ── horizon ───────────────────────────────────────────────────────────
def test_horizon_normalised_basic():
    assert horizon_normalised(99, 100) == pytest.approx(0.99)
    assert horizon_normalised(0, 100) == 0.0


def test_horizon_clamps_over_one():
    assert horizon_normalised(150, 100) == 1.0


def test_horizon_raises_on_bad_budget():
    with pytest.raises(ValueError):
        horizon_normalised(10, 0)


def test_resolve_max_horizon():
    assert resolve_max_horizon(100) == 100
    assert resolve_max_horizon(100, 0) == 100      # 0 = auto
    assert resolve_max_horizon(100, 20) == 100     # never below real budget
    assert resolve_max_horizon(100, 200) == 200
    with pytest.raises(ValueError):
        resolve_max_horizon(0)


# ── evidence confidence ───────────────────────────────────────────────
def test_evidence_confidence_empty_is_speculative():
    assert evidence_confidence([]) == pytest.approx(SPECULATIVE_HYPOTHESIS)


def test_evidence_confidence_clips():
    assert evidence_confidence([1.5, -0.2]) == pytest.approx(0.5)


# ── context ───────────────────────────────────────────────────────────
def test_context_saturation_basic():
    assert context_saturation(20_000, 128_000) == pytest.approx(20_000 / 128_000)


def test_context_saturation_guards():
    assert context_saturation(100, 0) == 0.0       # unknown window
    assert context_saturation(0, 128_000) == 0.0   # no data yet
    assert context_saturation(500_000, 128_000) == 1.0


def test_context_load_legacy():
    assert context_load(50_000, 1_000_000) == pytest.approx(0.05)


# ── historical success ────────────────────────────────────────────────
def test_historical_success_fresh_branch_is_half():
    assert historical_success(0, 0) == pytest.approx(0.5)


def test_historical_success_clamps_successes_over_attempts():
    # 3 findings from 1 tool call: clamp to 1/1 -> (1+1)/(1+2) = 2/3
    assert historical_success(3, 1) == pytest.approx(2 / 3)


def test_historical_success_decays_without_success():
    assert historical_success(0, 50) < 0.05


# ── compute_tdi ───────────────────────────────────────────────────────
def test_tdi_turn1_no_findings_is_broaden():
    # h=0.99 e=0.3 c=0.0 s=0.5 -> 0.297 + 0.21 + 0 + 0.10 = 0.607
    r = compute_tdi(
        remaining_steps=99, path_evidence=[], successes=0, attempts=0,
        max_horizon=100, context_utilisation=0.0,
    )
    assert r.mode == Mode.BROADEN
    assert r.tdi == pytest.approx(0.607, abs=1e-6)
    assert r.overridden is False


def test_tdi_late_scan_no_findings_is_llm_pick():
    # h=0.20 e=0.3 c=0.02 s≈0.012 -> ≈0.472 <= 0.60
    r = compute_tdi(
        remaining_steps=20, path_evidence=[], successes=0, attempts=80,
        max_horizon=100, context_utilisation=0.02,
    )
    assert r.mode == Mode.LLM_PICK


def test_tdi_raises_without_context_signal():
    with pytest.raises(ValueError):
        compute_tdi(
            remaining_steps=10, path_evidence=[], successes=0, attempts=0,
            max_horizon=100,
        )


def test_tdi_legacy_context_pair_still_works():
    r = compute_tdi(
        remaining_steps=10, path_evidence=[], successes=0, attempts=0,
        max_horizon=100, tokens_used=10_000, context_window=1_000_000,
    )
    assert 0.0 <= r.c <= 1.0


# ── evidence ladder ───────────────────────────────────────────────────
def test_ladder_false_positive_wins_over_everything():
    # Design decision (review round 3): a verifier-rejected finding is
    # noise, not evidence — SPECULATIVE (0.3), not PLAUSIBLE (0.5).
    # Keep the tda.py docstring (evidence ladder run 0) in sync with this.
    f = {"verification": "replay_verified", "false_positive": True}
    assert evidence_score_from_finding(f) == SPECULATIVE_HYPOTHESIS


def test_ladder_verification_field_beats_legacy_booleans():
    f = {"verification": "plausible", "verified": True}
    assert evidence_score_from_finding(f) == PLAUSIBLE_HYPOTHESIS


def test_ladder_replay_verified():
    assert evidence_score_from_finding(
        {"verification": "replay_verified"}
    ) == VERIFIED_EXPLOIT


def test_ladder_confirmed():
    assert evidence_score_from_finding({"verification": "confirmed"}) == CONFIRMED_VULN


def test_ladder_legacy_replay_and_human():
    assert evidence_score_from_finding({"replay_reproduced": True}) == VERIFIED_EXPLOIT
    assert evidence_score_from_finding({"human_verified": True}) == VERIFIED_EXPLOIT


def test_ladder_legacy_verified():
    assert evidence_score_from_finding({"verified": True}) == CONFIRMED_VULN


def test_ladder_bare_dict_is_plausible():
    assert evidence_score_from_finding({"title": "SQLi"}) == PLAUSIBLE_HYPOTHESIS


def test_ladder_non_dict_is_speculative():
    assert evidence_score_from_finding("sql-injection") == SPECULATIVE_HYPOTHESIS


def test_label_map_replay_verified():
    # ⚠️ DESIRED behaviour — fails until "replay_verified" is added to
    # _EVIDENCE_LABEL_MAP (currently falls through to 0.3).
    assert evidence_score_from_label("replay_verified") == VERIFIED_EXPLOIT


# ── override gate ─────────────────────────────────────────────────────
def test_force_exploit_confirmed_only():
    assert should_force_exploit([{"verification": "confirmed"}]) is True


def test_force_exploit_rejects_plausible_only():
    assert should_force_exploit([{"verification": "plausible"}]) is False


def test_force_exploit_rejects_false_positive_only():
    assert should_force_exploit([{"verified": True, "false_positive": True}]) is False


def test_force_exploit_mixed():
    assert should_force_exploit([
        {"verification": "plausible"},
        {"verification": "confirmed"},
    ]) is True


def test_max_evidence_score_empty():
    assert max_evidence_score([]) == 0.0