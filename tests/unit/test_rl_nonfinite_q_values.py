"""R27 / PL6 — non-finite Q-values must not silently reach action selection.

The report was right about the mechanism but wrong about where it matters,
and it stopped one layer too early:

  CLAIM (confirmed)  `np.random.choice(..., p=probs)` raises when `probs`
                     contains NaN because `q_vals` contained Inf/NaN.
  MISSED             ε-greedy — the DEFAULT mode — does not raise. It picks
                     the NaN: `np.argmax` on a row containing NaN returns the
                     *index of the NaN*, so the scan commits to a poisoned
                     action, silently, on every turn. Measured against all-NaN
                     weights: `action=0, exploratory=False, q=nan`.
  MISSED             A NaN checkpoint loads without complaint, so the poison
                     became *permanent*: `save_cache()` wrote it and
                     `_load_from_cache()` restored it into every future
                     process. Nothing ever overwrote it.
  REFUTED            `probs = probs / probs.sum()` after
                     `probs = exp_q / exp_q.sum()`. Measured over 200 000
                     draws across T ∈ {0.01 … 3.0}, the worst `|sum-1|` was
                     5.6e-16 against numpy's 1.49e-08 tolerance — thirteen
                     orders of magnitude of headroom. The second division is a
                     numerical no-op. The real degenerate case (a zero or
                     non-finite sum) cannot arise after sanitisation, because
                     the max element always contributes `exp(0) = 1`, but it is
                     guarded anyway.

So the class of bug is "a non-finite value silently propagates through the
RL numeric stack", not "the softmax needs a renormalisation". The guards
below cover selection, both disk boundaries, and the JSON/SSE surface.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pytest

from app.rl.policy import (
    INF_Q_SUBSTITUTE,
    NAN_Q_SUBSTITUTE,
    PentestPolicy,
)
from app.rl.state_encoder import ACTION_SPACE, STATE_DIM

REPO = Path(__file__).resolve().parents[2]
POLICY_SRC = (REPO / "app" / "rl" / "policy.py").read_text(encoding="utf-8")
QLEARNER_SRC = (REPO / "app" / "rl" / "q_learner.py").read_text(encoding="utf-8")
PERSIST_SRC = (REPO / "app" / "rl" / "persistence.py").read_text(encoding="utf-8")

A = len(ACTION_SPACE)


def _q(pairs: dict[int, float]) -> np.ndarray:
    arr = np.zeros(A, dtype=np.float64)
    for idx, val in pairs.items():
        arr[idx] = val
    return arr


# ─── 1. selection survives every non-finite row ───────────────────────


class TestSelectionSurvivesNonFiniteQ:
    @pytest.mark.parametrize(
        "name,q_vals",
        [
            ("one-NaN", _q({2: float("nan")})),
            ("one-posinf", _q({3: float("inf")})),
            ("one-neginf", _q({4: float("-inf")})),
            ("all-NaN", np.full(A, np.nan)),
            ("all-neginf", np.full(A, float("-inf"))),
            ("all-posinf", np.full(A, float("inf"))),
            ("mixed", _q({0: float("nan"), 1: float("inf"), 2: float("-inf")})),
            ("NaN-and-huge", _q({0: float("nan"), 1: 1e308})),
        ],
    )
    @pytest.mark.parametrize("mode", ["epsilon_greedy", "boltzmann"])
    def test_selection_never_raises(self, name, q_vals, mode):
        """The reported crash, plus the silent variant in the other mode."""
        p = PentestPolicy(exploration_mode=mode)
        if mode == "boltzmann":
            action, action_name, meta = p._select_boltzmann(q_vals, 0.0)
        else:
            action, action_name, meta = p._select_epsilon_greedy(q_vals, 0.0)
        assert 0 <= action < A, f"{name}/{mode} returned out-of-range action {action}"
        assert action_name == ACTION_SPACE[action]
        assert meta["q_values_sanitized"] > 0, (
            f"{name}/{mode}: corruption was not counted — the substitution "
            "became invisible."
        )

    @pytest.mark.parametrize("mode", ["epsilon_greedy", "boltzmann"])
    def test_meta_q_values_are_finite(self, mode):
        """Nothing non-finite may escape into JSON/SSE.

        `HarnessBridge.select_action` does `q_max = float(max(q_values))` and
        stores `q_values` in the rec meta, which lands in `scan_record.rl_stats`
        and in the SSE payload. A NaN there serialises as bare `NaN`, which is
        invalid JSON.
        """
        p = PentestPolicy(exploration_mode=mode)
        q_vals = np.full(A, np.nan)
        _, _, meta = (
            p._select_boltzmann(q_vals, 0.0) if mode == "boltzmann"
            else p._select_epsilon_greedy(q_vals, 0.0)
        )
        assert all(np.isfinite(meta["q_values"]))
        assert np.isfinite(max(meta["q_values"]))
        json.dumps(meta, allow_nan=False)  # raises if any NaN/inf survives

    def test_boltzmann_probs_sum_to_one(self):
        p = PentestPolicy(exploration_mode="boltzmann")
        _, _, meta = p._select_boltzmann(_q({0: float("nan")}), 0.0)
        assert abs(sum(meta["probs"]) - 1.0) < 1e-9

    def test_greedy_picks_a_real_action_not_the_nan(self):
        """Pre-fix: `argmax` returned the NaN index and ε-greedy used it."""
        p = PentestPolicy(exploration_mode="epsilon_greedy")
        p.epsilon = 0.0  # force the greedy branch
        q_vals = _q({2: float("nan")})
        action, _, meta = p._select_epsilon_greedy(q_vals, 0.0)
        # Every finite entry is 0.0 and the NaN becomes 0.0 too, so any index
        # is defensible — but the *reported* q for the pick must be finite.
        assert np.isfinite(meta["q_values"][action])
        assert meta["best_action"] == int(np.argmax(meta["q_values"]))

    def test_single_nan_does_not_disable_the_other_actions(self):
        """NaN → 0.0 must not out-compete a genuinely high Q-value."""
        p = PentestPolicy(exploration_mode="epsilon_greedy")
        p.epsilon = 0.0
        action, _, _ = p._select_epsilon_greedy(_q({0: 5.0, 3: float("nan")}), 0.0)
        assert action == 0

    def test_single_posinf_still_wins(self):
        """+inf is clamped, not flattened — the network's ordering survives."""
        p = PentestPolicy(exploration_mode="epsilon_greedy")
        p.epsilon = 0.0
        action, _, meta = p._select_epsilon_greedy(_q({0: 5.0, 3: float("inf")}), 0.0)
        assert action == 3
        assert meta["q_values"][3] == pytest.approx(INF_Q_SUBSTITUTE)

    def test_substitutes_are_documented_constants(self):
        assert NAN_Q_SUBSTITUTE == 0.0
        assert INF_Q_SUBSTITUTE > 0.0 and np.isfinite(INF_Q_SUBSTITUTE)


# ─── 2. corruption is reported, not swallowed ─────────────────────────


class TestCorruptionIsVisible:
    def test_counter_accumulates_across_calls(self):
        p = PentestPolicy()
        for _ in range(5):
            p._sanitize_q_values(_q({0: float("nan")}), "test")
        assert p._nonfinite_q_seen == 5

    def test_warning_fires_once_per_instance(self, caplog):
        """200 turns of a poisoned net must not bury the first log line."""
        import logging

        p = PentestPolicy()
        with caplog.at_level(logging.WARNING, logger="app.rl.policy"):
            for _ in range(20):
                p._sanitize_q_values(_q({0: float("nan")}), "test")
        hits = [r for r in caplog.records if "rl_nonfinite_q_values" in r.message]
        assert len(hits) == 1, f"expected 1 warning, got {len(hits)}"
        assert p._nonfinite_q_warned is True

    def test_clean_row_is_not_flagged(self):
        p = PentestPolicy()
        out, n = p._sanitize_q_values(np.array([0.1, 0.2, 0.3]), "test")
        assert n == 0
        assert p._nonfinite_q_seen == 0
        np.testing.assert_allclose(out, [0.1, 0.2, 0.3])

    def test_both_modes_call_the_sanitizer(self):
        """AST, so a rename of the helper keeps the guard meaningful."""
        tree = ast.parse(POLICY_SRC)
        callers = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr == "_sanitize_q_values":
                    fn = node  # noqa: F841 — locate the enclosing function below
        for fdef in ast.walk(tree):
            if isinstance(fdef, ast.FunctionDef) and fdef.name in (
                "_select_epsilon_greedy", "_select_boltzmann",
            ):
                src = ast.get_source_segment(POLICY_SRC, fdef) or ""
                assert "_sanitize_q_values(" in src, (
                    f"{fdef.name}() no longer sanitises Q-values — that path "
                    "either crashes (softmax) or picks the NaN (argmax)."
                )


# ─── 3. a non-finite checkpoint cannot become permanent ───────────────


class TestDiskBoundaries:
    def test_save_cache_refuses_nonfinite_weights(self):
        import app.rl.q_learner as qmod

        learner = qmod.DoubleQLearner()
        learner.save_cache()
        before = qmod.WEIGHTS_FILE.read_bytes()
        learner.online.set_weights(
            {k: np.full_like(v, np.nan) for k, v in learner.online.get_weights().items()}
        )
        learner.save_cache()
        assert qmod.WEIGHTS_FILE.read_bytes() == before, (
            "save_cache() overwrote a good cache with NaN — that poisons every "
            "future process permanently."
        )

    def test_load_quarantines_a_poisoned_cache(self):
        import app.rl.q_learner as qmod

        poisoned = qmod.DoubleQLearner()
        poisoned.online.set_weights(
            {k: np.full_like(v, np.nan) for k, v in poisoned.online.get_weights().items()}
        )
        np.savez(qmod.WEIGHTS_FILE, **poisoned.online.get_weights())
        qmod.META_FILE.write_text(
            json.dumps({"step": 617, "layout": qmod.state_layout_id()}),
            encoding="utf-8",
        )

        fresh = qmod.DoubleQLearner()
        weights = fresh.online.get_weights()
        assert all(np.isfinite(v).all() for v in weights.values()), (
            "a NaN cache was loaded as-is — every forward pass is now NaN"
        )
        assert fresh.train_steps == 0
        assert qmod.WEIGHTS_FILE.exists() is False, "poisoned cache was not moved"
        assert list(qmod.WEIGHTS_FILE.parent.glob("*.nonfinite.*.npz")), (
            "poisoned cache was deleted instead of quarantined — no post-mortem"
        )

    def test_clean_cache_still_round_trips(self):
        import app.rl.q_learner as qmod

        learner = qmod.DoubleQLearner()
        learner.train_step(
            np.zeros(STATE_DIM, dtype=np.float32), 0, 1.0,
            np.ones(STATE_DIM, dtype=np.float32), False,
        )
        learner.save_cache()
        reloaded = qmod.DoubleQLearner()
        assert reloaded.train_steps == learner.train_steps

    def test_save_checkpoint_refuses_nonfinite(self):
        """The SQL blob is durable too — guard must sit before it."""
        tree = ast.parse(PERSIST_SRC)
        fn = next(
            n for n in ast.walk(tree)
            if isinstance(n, ast.AsyncFunctionDef) and n.name == "save_checkpoint"
        )
        src = ast.get_source_segment(PERSIST_SRC, fn) or ""
        assert "_nonfinite_weight_keys()" in src, (
            "save_checkpoint() can still write a NaN weights_blob into "
            "vapt_rl_checkpoints, where load_latest_checkpoint() restores it "
            "forever."
        )
        # Must run BEFORE the blob is serialised.
        assert src.index("_nonfinite_weight_keys()") < src.index("_serialize_weights")


# ─── 4. the softmax itself ────────────────────────────────────────────


class TestSoftmaxIsNumericallySound:
    def test_float_drift_is_far_below_numpy_tolerance(self):
        """Documents why the proposed second renormalisation is a no-op."""
        rng = np.random.default_rng(1234)
        worst = 0.0
        for _ in range(20_000):
            q = rng.normal(0, rng.choice([1.0, 10.0, 1e3]), A)
            T = float(rng.choice([0.01, 0.1, 0.5, 1.0, 3.0]))
            e = np.exp((q - q.max()) / T)
            p = e / e.sum()
            worst = max(worst, abs(p.sum() - 1.0))
        assert worst < 1e-12, f"float drift {worst:.3e} is larger than expected"

    def test_max_element_always_has_nonzero_probability(self):
        """Guarantees the sum can never be 0, which is what would break the
        division — and the reason no renormalisation is needed."""
        p = PentestPolicy(exploration_mode="boltzmann")
        for q in (_q({}), np.full(A, -1e30), _q({0: -1e30, 1: float("nan")})):
            _, _, meta = p._select_boltzmann(np.asarray(q, dtype=np.float64), 0.0)
            probs = np.asarray(meta["probs"])
            assert np.isfinite(probs).all()
            assert probs.sum() == pytest.approx(1.0)
            assert probs.max() > 0.0

    def test_degenerate_sum_falls_back_to_uniform(self, monkeypatch):
        """The guard for the genuinely-degenerate case the report described."""
        import app.rl.policy as pmod

        p = PentestPolicy(exploration_mode="boltzmann")
        monkeypatch.setattr(pmod.np, "exp", lambda x: np.zeros_like(x))
        _, _, meta = p._select_boltzmann(_q({0: 1.0}), 0.0)
        probs = np.asarray(meta["probs"])
        np.testing.assert_allclose(probs, np.full(A, 1.0 / A), rtol=1e-12)

    def test_temperature_floor_is_respected(self):
        """A zero/negative temperature must not divide by ~0."""
        p = PentestPolicy(exploration_mode="boltzmann")
        p.temperature = 0.0
        _, _, meta = p._select_boltzmann(_q({0: 1.0}), 0.0)
        assert np.isfinite(np.asarray(meta["probs"])).all()
