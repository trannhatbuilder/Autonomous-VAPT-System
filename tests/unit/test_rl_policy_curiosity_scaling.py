"""
Regression guards for PL1 — the curiosity bonus must not floor epsilon.

The bug
-------
`_select_epsilon_greedy` used to do::

    effective_epsilon = min(1.0, self.epsilon + curiosity_bonus)

`curiosity_bonus` is `bonus_scale / sqrt(count + 1)` and does NOT decay with
epsilon. Measured on the stock config it sits at 0.14–0.50 depending on how
familiar the state is, so the sum is dominated by the bonus once epsilon has
decayed. Driving the real policy through 200 scans:

    scan 1    eps 0.9851   effective 1.0000
    scan 50   eps 0.4729   effective 0.6161
    scan 100  eps 0.2236   effective 0.3779
    scan 150  eps 0.1057   effective 0.2790
    scan 200  eps 0.0500   effective 0.2047   <-- 4x the W16 target

`epsilon_end = 0.05` was reached exactly, and then ignored. `use_curiosity`
defaults to True and the production default mode is `epsilon_greedy`, so this
was live in every run.

The fix
-------
The contribution is scaled by how much of the schedule is still unspent
(`_scaled_bonus`), so it reaches exactly 0.0 at `epsilon_end`:

    scan 200  eps 0.0500   effective 0.0500

These tests pin: the scaling contract, the end-to-end trajectory, that the
bonus is still fully active early, that Boltzmann scales by T (not eps), and
that the source cannot regress to raw addition.
"""
from __future__ import annotations

import ast
import os
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

os.environ.setdefault("VAPT_AI_POSTGRES_DB", "postgresql://vapt:vapt@localhost:5432/vapt_ai_test")
os.environ.setdefault("VAPT_AI_ENVIRONMENT", "dev")

import numpy as np
import pytest

from app.rl.policy import PentestPolicy

POLICY = _PROJECT_ROOT / "app" / "rl" / "policy.py"


# ── helpers ──────────────────────────────────────────────────────────────

def _method_src(name: str) -> str:
    """Return the source text of one method in policy.py."""
    src = POLICY.read_text(encoding="utf-8")
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_source_segment(src, node) or ""
    raise AssertionError(f"method {name!r} not found in {POLICY}")


def _state(turn: int = 50) -> np.ndarray:
    """A plausible 337-dim state vector."""
    v = np.zeros(337, dtype=np.float32)
    v[0] = turn / 100.0          # turn / max_turns
    v[1] = 0.35                  # tdi
    v[2] = 0.5                   # mode
    v[6] = turn / 100.0          # commands_run / max_commands
    v[7] = 1.0 - turn / 100.0    # token_budget_remaining
    return v


def _decayed_policy(scans: int = 200) -> PentestPolicy:
    """A policy that has been driven through `scans` episodes.

    Reward 1.0 keeps the adaptive decay on its nominal branch, which is the
    documented W16 schedule (1.0 -> 0.05 over 200 scans).
    """
    policy = PentestPolicy()
    for _ in range(scans):
        policy.end_episode(1.0)
    return policy


# ── 1. the scaling contract itself ───────────────────────────────────────

class TestScheduleProgress:
    """`_schedule_progress` maps [end, start] onto [0, 1] and never escapes."""

    def test_start_of_schedule_is_unspent(self):
        assert PentestPolicy._schedule_progress(1.0, 1.0, 0.05) == 1.0

    def test_end_of_schedule_is_fully_spent(self):
        assert PentestPolicy._schedule_progress(0.05, 1.0, 0.05) == 0.0

    def test_below_end_is_clamped_to_zero(self):
        """ε below ε_end must not yield a negative weight."""
        assert PentestPolicy._schedule_progress(0.0, 1.0, 0.05) == 0.0
        assert PentestPolicy._schedule_progress(-5.0, 1.0, 0.05) == 0.0

    def test_above_start_is_clamped_to_one(self):
        assert PentestPolicy._schedule_progress(2.0, 1.0, 0.05) == 1.0

    def test_midpoint_is_half(self):
        assert PentestPolicy._schedule_progress(0.525, 1.0, 0.05) == pytest.approx(0.5)

    def test_degenerate_schedule_returns_zero(self):
        """ε_start == ε_end: no decaying budget exists, so nothing may extend it."""
        assert PentestPolicy._schedule_progress(0.05, 0.05, 0.05) == 0.0
        assert PentestPolicy._schedule_progress(0.0, 0.0, 0.0) == 0.0

    def test_inverted_schedule_returns_zero(self):
        """ε_start < ε_end is a config error; it must not produce weight > 1."""
        assert PentestPolicy._schedule_progress(0.5, 0.05, 0.5) == 0.0

    def test_weight_is_always_in_unit_interval(self):
        for value in np.linspace(-2.0, 2.0, 41):
            weight = PentestPolicy._schedule_progress(float(value), 1.0, 0.05)
            assert 0.0 <= weight <= 1.0


class TestScaledBonus:
    """The bonus is multiplied by the remaining budget, not added raw."""

    def test_full_bonus_while_epsilon_is_at_start(self):
        policy = PentestPolicy()
        assert policy._scaled_bonus(0.5) == pytest.approx(0.5)

    def test_zero_contribution_at_epsilon_end(self):
        policy = _decayed_policy()
        assert policy.epsilon == pytest.approx(policy.epsilon_end, abs=1e-6)
        assert policy._scaled_bonus(0.5) == 0.0

    def test_contribution_never_exceeds_raw_bonus(self):
        policy = _decayed_policy(scans=100)
        assert policy._scaled_bonus(0.5) <= 0.5

    def test_contribution_never_negative(self):
        policy = PentestPolicy()
        for eps in (0.0, 0.01, 0.05, 0.5, 1.0):
            policy.epsilon = eps
            assert policy._scaled_bonus(0.5) >= 0.0

    def test_zero_bonus_stays_zero(self):
        assert PentestPolicy()._scaled_bonus(0.0) == 0.0


# ── 2. the schedule must actually be reachable ───────────────────────────

class TestEpsilonScheduleIsNotFloored:
    """End-to-end: driving the policy must converge on epsilon_end."""

    def test_effective_epsilon_equals_epsilon_end_at_scan_200(self):
        """The whole bug in one assertion.

        Before the fix this was 0.2047 — the schedule said 0.05 and the
        policy explored 4x more than configured.
        """
        policy = _decayed_policy()
        bonus = policy.curiosity.compute_bonus(_state())
        effective = min(1.0, policy.epsilon + policy._scaled_bonus(bonus))
        assert effective == pytest.approx(policy.epsilon_end, abs=1e-6)

    def test_effective_epsilon_never_exceeds_old_formula(self):
        """Scaling can only reduce the contribution, never inflate it."""
        policy = PentestPolicy()
        for _ in range(200):
            bonus = policy.curiosity.compute_bonus(_state())
            raw = min(1.0, policy.epsilon + bonus)
            scaled = min(1.0, policy.epsilon + policy._scaled_bonus(bonus))
            assert scaled <= raw + 1e-12
            policy.end_episode(1.0)

    def test_effective_epsilon_decreases_monotonically(self):
        """For a fixed bonus, effective ε must track ε downward, not plateau."""
        policy = _decayed_policy(scans=0)
        bonus = 0.5  # a maximal/near-maximal bonus, held constant
        previous = float("inf")
        for _ in range(200):
            effective = min(1.0, policy.epsilon + policy._scaled_bonus(bonus))
            # Once the clamp at 1.0 stops biting, the sequence must be monotone.
            if policy.epsilon < 0.5:
                assert effective <= previous + 1e-12
            previous = effective
            policy.end_episode(1.0)

    def test_curiosity_is_still_active_early(self):
        """The fix must not neuter curiosity — it only stops it outliving ε."""
        policy = PentestPolicy()
        assert policy._scaled_bonus(0.5) == pytest.approx(0.5)
        effective = min(1.0, policy.epsilon + policy._scaled_bonus(0.5))
        assert effective == 1.0  # ε_start is already maximum exploration

    def test_old_arithmetic_is_gone(self):
        """The exact numbers from the bug report, asserted by value.

        epsilon 0.05 + bonus 0.34 = 0.39 under the old formula; the correct
        answer at epsilon_end is 0.05 no matter what the bonus is.
        """
        policy = PentestPolicy()
        policy.epsilon = policy.epsilon_end
        old = min(1.0, policy.epsilon + 0.34)
        new = min(1.0, policy.epsilon + policy._scaled_bonus(0.34))
        assert old == pytest.approx(0.39)
        assert new == pytest.approx(0.05)


# ── 3. Boltzmann scales by T, not by ε ───────────────────────────────────

class TestBoltzmannScalesByTemperature:
    """`select_action_boltzmann` can run in ε-greedy mode, so it must use T."""

    def test_contribution_uses_temperature_schedule(self):
        policy = PentestPolicy(exploration_mode="epsilon_greedy")
        policy.epsilon = policy.epsilon_end   # ε fully spent
        policy.temperature = policy.temperature_start
        # ε says 0, T says 1 — Boltzmann must read T.
        assert policy._scaled_bonus(0.5, boltzmann=True) == pytest.approx(0.5)

    def test_contribution_is_zero_at_temperature_end(self):
        policy = PentestPolicy()
        policy.temperature = policy.temperature_end
        assert policy._scaled_bonus(0.5, boltzmann=True) == 0.0

    def test_boltzmann_weight_shrinks_as_temperature_decays(self):
        policy = PentestPolicy()
        policy.temperature = policy.temperature_start
        start = policy._scaled_bonus(0.5, boltzmann=True)
        policy.temperature = (policy.temperature_start + policy.temperature_end) / 2
        mid = policy._scaled_bonus(0.5, boltzmann=True)
        policy.temperature = policy.temperature_end
        end = policy._scaled_bonus(0.5, boltzmann=True)
        assert start > mid > end == 0.0


# ── 4. the selector actually uses the scaled value ───────────────────────

class TestSelectorUsesScaledBonus:
    def test_effective_epsilon_reported_equals_epsilon_at_end(self):
        policy = _decayed_policy()
        policy.curiosity.compute_bonus = lambda _v: 0.5  # pin the raw bonus
        _, _, meta = policy.select_action(_state())
        assert meta["epsilon"] == pytest.approx(policy.epsilon_end, abs=1e-4)
        assert meta["effective_epsilon"] == pytest.approx(policy.epsilon_end, abs=1e-4)

    def test_meta_reports_raw_and_applied_separately(self):
        """A log reader must be able to tell the two apart."""
        policy = _decayed_policy(scans=100)
        policy.curiosity.compute_bonus = lambda _v: 0.5
        _, _, meta = policy.select_action(_state())
        assert "curiosity_bonus" in meta
        assert "curiosity_contribution" in meta
        assert meta["curiosity_bonus"] == pytest.approx(0.5, abs=1e-4)
        assert meta["curiosity_contribution"] < meta["curiosity_bonus"]

    def test_meta_contribution_plus_epsilon_is_effective(self):
        policy = _decayed_policy(scans=150)
        policy.curiosity.compute_bonus = lambda _v: 0.5
        _, _, meta = policy.select_action(_state())
        assert meta["epsilon"] + meta["curiosity_contribution"] == pytest.approx(
            meta["effective_epsilon"], abs=1e-3
        )

    def test_boltzmann_meta_reports_contribution(self):
        policy = _decayed_policy(scans=150)
        policy.curiosity.compute_bonus = lambda _v: 0.5
        _, _, meta = policy.select_action_boltzmann(_state())
        assert "curiosity_contribution" in meta
        assert meta["curiosity_contribution"] < 0.5

    def test_disabled_curiosity_contributes_nothing(self):
        policy = PentestPolicy(use_curiosity=False)
        _, _, meta = policy.select_action(_state())
        assert meta["curiosity_bonus"] == 0.0
        assert meta["curiosity_contribution"] == 0.0

    def test_effective_epsilon_never_below_epsilon(self):
        """Scaling can reduce the bonus but must never subtract from ε."""
        policy = _decayed_policy(scans=50)
        for _ in range(20):
            _, _, meta = policy.select_action(_state())
            assert meta["effective_epsilon"] >= meta["epsilon"]


# ── 5. source guards: the class of bug, not the instance ─────────────────

class TestSourceCannotRegress:
    def test_epsilon_greedy_does_not_add_raw_bonus(self):
        src = _method_src("_select_epsilon_greedy")
        assert "self.epsilon + curiosity_bonus" not in src, (
            "_select_epsilon_greedy adds the raw curiosity bonus again — that "
            "is exactly the PL1 bug: a constant added to ε becomes a floor "
            "that survives the whole decay schedule."
        )
        assert "_scaled_bonus(curiosity_bonus" in src

    def test_boltzmann_does_not_add_raw_bonus(self):
        src = _method_src("_select_boltzmann")
        assert "self.temperature + curiosity_bonus" not in src, (
            "_select_boltzmann adds the raw curiosity bonus again."
        )
        assert "_scaled_bonus(curiosity_bonus" in src
        assert "boltzmann=True" in src

    def test_both_selectors_report_both_bonus_values(self):
        for name in ("_select_epsilon_greedy", "_select_boltzmann"):
            src = _method_src(name)
            assert '"curiosity_bonus"' in src
            assert '"curiosity_contribution"' in src

    def test_scaling_helper_exists_and_has_a_docstring(self):
        src = _method_src("_scaled_bonus")
        assert "epsilon_end" in src or "temperature_end" in src
        assert '"""' in src

    def test_dead_decay_batch_size_is_gone(self):
        """It was declared 'Decay at most 100 entries per call' and never used.

        The decay loop always walked the whole dict. A declared-but-unused cap
        is worse than no cap: it documents behaviour that does not happen.
        """
        text = POLICY.read_text(encoding="utf-8")
        assert "self._decay_batch_size" not in text, (
            "_decay_batch_size is back — either implement the cap or leave it out."
        )

    def test_hash_cardinality_measurement_is_documented(self):
        """The measured bucket count must stay written down.

        Anyone proposing to change `_hash_state` needs to know the reachable
        space is ~767 buckets over 20,000 turns, not the nominal 33**16.
        """
        src = _method_src("_hash_state")
        assert "767" in src
        assert "reachable" in src.lower()

    def test_decay_comment_does_not_claim_a_batch_cap(self):
        src = _method_src("compute_bonus")
        assert "at most 100 entries" not in src
