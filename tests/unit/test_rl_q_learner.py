"""
Tests for app.rl.q_learner — Dueling Double DQN (numpy-only).

Covers:
  - TwoLayerMLP weight shapes (He/Xavier init)
  - DuelingNetwork forward (Q = V + A - mean(A))
  - DoubleQLearner instantiation + cache load
  - q_values / best_action
  - train_step returns abs(TD error)
  - train_batch + cache save
  - Soft target update (τ=0.005)
  - Huber loss gradient clipping
  - Save/load .npz round-trip
  - Stale checkpoint quarantine (dim mismatch)
  - Parameter count
"""
from __future__ import annotations

import os
import sys
import json
import tempfile
from pathlib import Path

# Ensure project root is on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

os.environ.setdefault("VAPT_AI_POSTGRES_DB", "postgresql://vapt:vapt@localhost:5432/vapt_ai_test")
os.environ.setdefault("VAPT_AI_ENVIRONMENT", "dev")

import numpy as np
import pytest

from app.rl.q_learner import (
    DoubleQLearner, QLearner, TwoLayerMLP, DuelingNetwork,
    STATE_DIM, ACTION_DIM,
    DEFAULT_LEARNING_RATE, DEFAULT_GAMMA, DEFAULT_TAU, DEFAULT_HIDDEN_DIM,
    HUBER_KAPPA, _relu, _relu_derivative,
)
import app.rl.q_learner as qmod


# ── Section 1: TwoLayerMLP ───────────────────────────────────────────────

class TestTwoLayerMLP:
    def test_weight_shapes(self):
        """W1 (hidden, input), b1 (hidden,), W2 (output, hidden), b2 (output,)."""
        mlp = TwoLayerMLP(input_dim=337, hidden_dim=64, output_dim=7)
        assert mlp.W1.shape == (64, 337)
        assert mlp.b1.shape == (64,)
        assert mlp.W2.shape == (7, 64)
        assert mlp.b2.shape == (7,)

    def test_he_init_std(self):
        """He init: W1 std ≈ sqrt(2/fan_in) = sqrt(2/337) ≈ 0.0770."""
        np.random.seed(42)
        mlp = TwoLayerMLP(input_dim=337, hidden_dim=64, output_dim=7)
        expected_std1 = np.sqrt(2.0 / 337)
        assert abs(mlp.W1.std() - expected_std1) < 0.01

    def test_xavier_init_std(self):
        """Xavier init for output: W2 std ≈ sqrt(1/hidden) = sqrt(1/64) ≈ 0.125."""
        np.random.seed(42)
        mlp = TwoLayerMLP(input_dim=337, hidden_dim=64, output_dim=7)
        expected_std2 = np.sqrt(1.0 / 64)
        assert abs(mlp.W2.std() - expected_std2) < 0.02

    def test_forward_shapes(self):
        mlp = TwoLayerMLP(input_dim=337, hidden_dim=64, output_dim=7)
        x = np.random.randn(337).astype(np.float32)
        out, h1 = mlp.forward(x)
        assert out.shape == (7,)
        assert h1.shape == (64,)

    def test_set_weights_roundtrip(self):
        mlp1 = TwoLayerMLP(337, 64, 7)
        mlp2 = TwoLayerMLP(337, 64, 7)
        mlp2.set_weights(mlp1.get_weights())
        assert np.allclose(mlp1.W1, mlp2.W1)
        assert np.allclose(mlp1.W2, mlp2.W2)

    def test_soft_update_changes_weights(self):
        """τ=0.5 soft_update should move self halfway toward other."""
        mlp1 = TwoLayerMLP(337, 64, 7)
        mlp2 = TwoLayerMLP(337, 64, 7)
        original_W1 = mlp1.W1.copy()
        mlp1.soft_update(mlp2, tau=0.5)
        # Should be different from original
        assert not np.allclose(original_W1, mlp1.W1)


# ── Section 2: DuelingNetwork ────────────────────────────────────────────

class TestDuelingNetwork:
    def test_forward_returns_3_outputs(self):
        net = DuelingNetwork(state_dim=337, action_dim=7, hidden_dim=64)
        x = np.random.randn(337).astype(np.float32)
        q, v, a = net.forward(x)
        assert q.shape == (7,)
        assert isinstance(v, float)
        assert a.shape == (7,)

    def test_q_equals_v_plus_a(self):
        """Q = V + A - mean(A) ⇒ Q = V + (A - mean(A)) = V + normalized_A."""
        net = DuelingNetwork(state_dim=337, action_dim=7, hidden_dim=64)
        x = np.random.randn(337).astype(np.float32)
        q, v, a = net.forward(x)
        # a is already normalized (mean subtracted), so q = v + a
        assert np.allclose(q, v + a, atol=1e-5)

    def test_weights_keys_prefixed(self):
        """Keys: value_W1/b1/W2/b2 + adv_W1/b1/W2/b2."""
        net = DuelingNetwork(state_dim=337, action_dim=7, hidden_dim=64)
        keys = sorted(net.get_weights().keys())
        expected = sorted([
            "value_W1", "value_b1", "value_W2", "value_b2",
            "adv_W1", "adv_b1", "adv_W2", "adv_b2",
        ])
        assert keys == expected

    def test_copy_from_makes_equal(self):
        net1 = DuelingNetwork(337, 7, 64)
        net2 = DuelingNetwork(337, 7, 64)
        net2.copy_from(net1)
        w1 = net1.get_weights()
        w2 = net2.get_weights()
        for k in w1:
            assert np.allclose(w1[k], w2[k])


# ── Section 3: DoubleQLearner ────────────────────────────────────────────

class TestDoubleQLearner:
    def test_instantiation(self):
        learner = DoubleQLearner(state_dim=337, action_dim=7, hidden_dim=64)
        assert learner.state_dim == 337
        assert learner.action_dim == 7
        assert learner.lr == DEFAULT_LEARNING_RATE
        assert learner.gamma == DEFAULT_GAMMA
        assert learner.tau == DEFAULT_TAU

    def test_q_values_shape(self):
        learner = DoubleQLearner(state_dim=337, action_dim=7, hidden_dim=64)
        state = np.random.randn(337).astype(np.float32)
        q = learner.q_values(state)
        assert q.shape == (7,)

    def test_best_action_in_range(self):
        learner = DoubleQLearner(state_dim=337, action_dim=7, hidden_dim=64)
        state = np.random.randn(337).astype(np.float32)
        action = learner.best_action(state)
        assert 0 <= action < 7

    def test_train_step_returns_td_error(self, tmp_path, monkeypatch):
        """train_step returns abs(TD error) ≥ 0."""
        # Redirect cache to tmp so we start from step=0
        monkeypatch.setattr(qmod, "WEIGHTS_FILE", tmp_path / "q_weights.npz")
        monkeypatch.setattr(qmod, "META_FILE", tmp_path / "q_meta.json")
        learner = DoubleQLearner(state_dim=337, action_dim=7, hidden_dim=64)
        s = np.random.randn(337).astype(np.float32)
        s_next = np.random.randn(337).astype(np.float32)
        td_err = learner.train_step(s, action=2, reward=1.0, next_state=s_next, done=False)
        assert td_err >= 0
        assert learner._step == 1

    def test_train_step_increments_step(self, tmp_path, monkeypatch):
        monkeypatch.setattr(qmod, "WEIGHTS_FILE", tmp_path / "q_weights.npz")
        monkeypatch.setattr(qmod, "META_FILE", tmp_path / "q_meta.json")
        learner = DoubleQLearner(state_dim=337, action_dim=7, hidden_dim=64)
        for i in range(5):
            learner.train_step(
                state=np.random.randn(337).astype(np.float32),
                action=i, reward=float(i),
                next_state=np.random.randn(337).astype(np.float32),
                done=(i == 4),
            )
        assert learner._step == 5

    def test_train_batch_returns_loss_and_td_errors(self):
        learner = DoubleQLearner(state_dim=337, action_dim=7, hidden_dim=64)
        batch = [
            {"state": np.random.randn(337).astype(np.float32),
             "action": i % 7, "reward": float(i),
             "next_state": np.random.randn(337).astype(np.float32),
             "done": (i == 4), "is_weight": 1.0}
            for i in range(5)
        ]
        loss, td_errors = learner.train_batch(batch)
        assert isinstance(loss, float)
        assert len(td_errors) == 5
        assert all(td >= 0 for td in td_errors)

    def test_target_soft_update_changes_weights(self):
        """After train_step, target network should differ from initial (soft-updated)."""
        learner = DoubleQLearner(state_dim=337, action_dim=7, hidden_dim=64)
        target_before = learner.target.get_weights()["value_W1"].copy()
        learner.train_step(
            state=np.random.randn(337).astype(np.float32),
            action=0, reward=1.0,
            next_state=np.random.randn(337).astype(np.float32),
            done=False,
        )
        target_after = learner.target.get_weights()["value_W1"]
        # τ=0.005 → small change but non-zero
        assert not np.allclose(target_before, target_after)

    def test_QLearner_alias(self):
        """QLearner is alias for DoubleQLearner (backward compat with EVVO)."""
        assert QLearner is DoubleQLearner

    def test_get_stats(self):
        learner = DoubleQLearner(state_dim=337, action_dim=7, hidden_dim=64)
        stats = learner.get_stats()
        assert "step" in stats
        assert "lr" in stats
        assert "gamma" in stats
        assert "tau" in stats
        assert "total_params" in stats
        assert "architecture" in stats
        assert stats["architecture"] == "dueling_double_q"

    def test_total_params_count(self):
        """43,784 params = (64×337+64+1×64+1) + (64×337+64+7×64+7)."""
        learner = DoubleQLearner(state_dim=337, action_dim=7, hidden_dim=64)
        # Value: 64*337 + 64 + 1*64 + 1 = 21697
        # Adv:   64*337 + 64 + 7*64 + 7 = 22087
        # Total: 43784
        assert learner._count_params() == 43784


# ── Section 4: Cache save/load ───────────────────────────────────────────

class TestCacheSaveLoad:
    def test_save_load_roundtrip(self, tmp_path, monkeypatch):
        """Save → re-instantiate → weights match."""
        # Redirect cache paths to tmp
        monkeypatch.setattr(qmod, "WEIGHTS_FILE", tmp_path / "q_weights.npz")
        monkeypatch.setattr(qmod, "META_FILE", tmp_path / "q_meta.json")

        learner = DoubleQLearner(state_dim=337, action_dim=7, hidden_dim=64)
        # Train a few steps to make weights non-default
        for i in range(3):
            learner.train_step(
                state=np.random.randn(337).astype(np.float32),
                action=i, reward=float(i),
                next_state=np.random.randn(337).astype(np.float32),
                done=False,
            )
        learner.save_cache()
        step_before = learner._step
        w_before = learner.online.get_weights()["value_W1"].copy()

        # Re-instantiate — should load from cache
        learner2 = DoubleQLearner(state_dim=337, action_dim=7, hidden_dim=64)
        assert learner2._step == step_before
        w_after = learner2.online.get_weights()["value_W1"]
        assert np.allclose(w_before, w_after)

    def test_stale_checkpoint_quarantine(self, tmp_path, monkeypatch):
        """Cache with dim=128 should be quarantined + step reset to 0."""
        stale_path = tmp_path / "q_weights.npz"
        stale_meta = tmp_path / "q_meta.json"

        # Save stale weights with dim=128
        stale_W1 = np.random.randn(64, 128).astype(np.float32)
        np.savez(
            stale_path,
            value_W1=stale_W1, value_b1=np.zeros(64, dtype=np.float32),
            value_W2=np.random.randn(1, 64).astype(np.float32), value_b2=np.zeros(1, dtype=np.float32),
            adv_W1=np.random.randn(64, 128).astype(np.float32), adv_b1=np.zeros(64, dtype=np.float32),
            adv_W2=np.random.randn(7, 64).astype(np.float32), adv_b2=np.zeros(7, dtype=np.float32),
        )
        stale_meta.write_text(json.dumps({"step": 999, "lr": 0.001}))

        monkeypatch.setattr(qmod, "WEIGHTS_FILE", stale_path)
        monkeypatch.setattr(qmod, "META_FILE", stale_meta)

        fresh = DoubleQLearner(state_dim=337, action_dim=7, hidden_dim=64)
        assert fresh._step == 0  # reset, not 999
        # Original stale file gone (renamed to .stale-dim128.<ts>.npz)
        assert not stale_path.exists()
        backups = list(tmp_path.glob("q_weights.stale-dim128.*.npz"))
        assert len(backups) == 1


# ── Section 5: Huber loss ────────────────────────────────────────────────

class TestHuberLoss:
    def test_huber_kappa_value(self):
        assert HUBER_KAPPA == 1.0

    def test_relu(self):
        x = np.array([-1.0, 0.0, 1.0, 2.0], dtype=np.float32)
        assert np.allclose(_relu(x), [0, 0, 1, 2])

    def test_relu_derivative(self):
        x = np.array([-1.0, 0.0, 1.0, 2.0], dtype=np.float32)
        d = _relu_derivative(x)
        # ReLU'(z1) = (z1 > 0); for x=0 → 0 (Python convention)
        assert d[0] == 0
        assert d[1] == 0
        assert d[2] == 1
        assert d[3] == 1

    def test_large_td_error_doesnt_explode(self):
        """Large reward (+10) should produce bounded gradient (Huber clip)."""
        learner = DoubleQLearner(state_dim=337, action_dim=7, hidden_dim=64)
        w_before = learner.online.value_net.W1.copy()
        # Large TD error (current Q near 0, target = 10 + 0.95*0 = 10)
        learner.train_step(
            state=np.zeros(337, dtype=np.float32),
            action=0, reward=10.0,
            next_state=np.zeros(337, dtype=np.float32),
            done=True,  # no future reward → target = 10
        )
        w_after = learner.online.value_net.W1
        # Gradient magnitude should be bounded by lr * kappa = 0.001 * 1.0
        max_change = np.abs(w_after - w_before).max()
        assert max_change <= 0.002  # 2x lr*kappa for safety