"""
Double Q-Learner with dueling architecture, target network, and PER support.

Python port of EVVO Sentinel shield_engine/rl/q_learner.py (post-audit-fix).

Implements:
  - Double Q-Learning (Hasselt et al. 2015): decouples action selection
    from action evaluation to reduce overestimation bias.
  - Target network: stable target computation with soft updates (τ=0.005).
  - Dueling architecture: Q(s,a) = V(s) + A(s,a) - mean(A(s,:)).
  - Prioritized Experience Replay (PER): caller passes is_weight per sample.
  - Huber / SmoothL1 loss: bounds gradient magnitude under large early-training
    TD errors (audit P2-17 from EVVO).
  - He (Kaiming) init for ReLU hidden, Xavier init for linear output.

For lightweight CPU deployment, uses a 2-layer MLP with analytical
backpropagation (no PyTorch/TensorFlow dependency).

Persistence (VAPT-AI adaptation):
  - .npz cache at settings.rl_dir / "q_weights.npz"  (fast-load)
  - q_meta.json alongside for step / lr / gamma / tau
  - Full SQL checkpoint persistence is implemented in app/rl/persistence.py (W16-S7)
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

from app.core.config import settings

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────
# Module-level path resolution (settings is a cached singleton at import time).
DATA_DIR: Path = settings.rl_dir
WEIGHTS_FILE: Path = DATA_DIR / "q_weights.npz"
META_FILE: Path = DATA_DIR / "q_meta.json"

# Must match app.rl.state_encoder.STATE_DIM (filled in W16-S3).
# Hard-coded here to avoid circular import: state_encoder imports from q_learner
# for ACTION_DIM, and q_learner needs STATE_DIM at class-instantiation time.
STATE_DIM: int = 337
ACTION_DIM: int = 7

DEFAULT_LEARNING_RATE: float = 0.001
DEFAULT_GAMMA: float = 0.95
DEFAULT_TAU: float = 0.005
DEFAULT_HIDDEN_DIM: int = 64

# Huber loss kappa (SmoothL1 threshold). EVVO audit P2-17.
HUBER_KAPPA: float = 1.0


# ── Activation helpers ────────────────────────────────────────────────────
def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)


def _relu_derivative(x: np.ndarray) -> np.ndarray:
    # x is the pre-activation z1 (we pass h1 into backward, but ReLU'(z1) = (z1>0))
    # In EVVO's port the caller passes h1 (post-activation) — ReLU'(z1) = (h1 > 0)
    # is equivalent because h1 = max(0, z1) ⇒ h1 > 0 ⟺ z1 > 0.
    return (x > 0).astype(np.float32)


# ── TwoLayerMLP ───────────────────────────────────────────────────────────
class TwoLayerMLP:
    """
    2-layer MLP: state → hidden (ReLU) → output (linear).

    Weights:
      W1: (hidden_dim, state_dim)
      b1: (hidden_dim,)
      W2: (output_dim, hidden_dim)
      b2: (output_dim,)

    Forward:  z1 = x @ W1.T + b1 ; h1 = relu(z1) ; z2 = h1 @ W2.T + b2
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
    ) -> None:
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self._init_weights()

    def _init_weights(self) -> None:
        # He (Kaiming) init for ReLU hidden layer: std = sqrt(2/fan_in).
        # Xavier under-scales for ReLU, slowing early learning (EVVO audit fix).
        std1 = np.sqrt(2.0 / self.input_dim)
        self.W1 = (np.random.randn(self.hidden_dim, self.input_dim).astype(np.float32) * std1)
        self.b1 = np.zeros(self.hidden_dim, dtype=np.float32)

        # Output layer is linear → Xavier (sqrt(1/fan_in)) is appropriate.
        std2 = np.sqrt(1.0 / self.hidden_dim)
        self.W2 = (np.random.randn(self.output_dim, self.hidden_dim).astype(np.float32) * std2)
        self.b2 = np.zeros(self.output_dim, dtype=np.float32)

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Forward pass.

        Args:
            x: input vector (state_dim,).

        Returns:
            (output, hidden) where hidden is post-activation h1 (for backprop).
        """
        z1 = x @ self.W1.T + self.b1
        h1 = _relu(z1)
        z2 = h1 @ self.W2.T + self.b2
        return z2, h1

    def backward(
        self,
        x: np.ndarray,
        h1: np.ndarray,
        grad_output: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """
        Analytical backpropagation.

        Args:
            x: input (state_dim,).
            h1: hidden activation from forward().
            grad_output: dL/dz2 (output_dim,).

        Returns:
            grads dict with keys W1, b1, W2, b2.
        """
        # dL/dW2 = grad_output[:, None] * h1[None, :]   (outer product)
        # dL/db2 = grad_output
        dW2 = np.outer(grad_output, h1)
        db2 = grad_output

        # dL/dh1 = grad_output @ W2
        dh1 = grad_output @ self.W2

        # dL/dz1 = dh1 * relu'(z1). We use h1 (post-activation) for the mask
        # because h1 > 0 ⟺ z1 > 0 ⟺ relu'(z1) = 1.
        dz1 = dh1 * _relu_derivative(h1)

        # dL/dW1 = dz1[:, None] * x[None, :]
        # dL/db1 = dz1
        dW1 = np.outer(dz1, x)
        db1 = dz1

        return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}

    def get_weights(self) -> dict[str, np.ndarray]:
        return {"W1": self.W1, "b1": self.b1, "W2": self.W2, "b2": self.b2}

    def set_weights(self, weights: dict[str, np.ndarray]) -> None:
        self.W1 = weights["W1"].astype(np.float32).copy()
        self.b1 = weights["b1"].astype(np.float32).copy()
        self.W2 = weights["W2"].astype(np.float32).copy()
        self.b2 = weights["b2"].astype(np.float32).copy()

    def copy_from(self, other: "TwoLayerMLP") -> None:
        self.set_weights(other.get_weights())

    def soft_update(self, other: "TwoLayerMLP", tau: float) -> None:
        """Polyak averaging: self ← τ·other + (1-τ)·self."""
        for key in self.get_weights():
            self_w = self.get_weights()[key]
            other_w = other.get_weights()[key]
            self_w[:] = tau * other_w + (1.0 - tau) * self_w


# ── DuelingNetwork ────────────────────────────────────────────────────────
class DuelingNetwork:
    """
    Dueling Q-Network using two 2-layer MLPs:
      - Value stream:      state → V(s)       (scalar)
      - Advantage stream:  state → A(s, a)    (action_dim,)
      - Combined:          Q(s, a) = V(s) + A(s, a) - mean(A(s, :))

    The advantage normalization (subtract mean) makes the decomposition
    identifiable: only the relative advantage of one action over another
    matters, and V(s) absorbs the absolute scale.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = DEFAULT_HIDDEN_DIM,
    ) -> None:
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.value_net = TwoLayerMLP(state_dim, hidden_dim, 1)
        self.advantage_net = TwoLayerMLP(state_dim, hidden_dim, action_dim)

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
        """
        Args:
            x: state vector (state_dim,).

        Returns:
            (q_values, v_value, advantage)
              - q_values: (action_dim,)  — Q(s, a) for each action
              - v_value:  scalar         — V(s)
              - advantage: (action_dim,) — A(s, a) - mean(A(s, :))
        """
        v, _ = self.value_net.forward(x)
        v = float(v[0])
        a, _ = self.advantage_net.forward(x)
        a = a - np.mean(a)  # advantage normalization (identifiability)
        q = v + a
        return q, v, a

    def get_weights(self) -> dict[str, np.ndarray]:
        w: dict[str, np.ndarray] = {}
        for k, v in self.value_net.get_weights().items():
            w[f"value_{k}"] = v
        for k, v in self.advantage_net.get_weights().items():
            w[f"adv_{k}"] = v
        return w

    def set_weights(self, weights: dict[str, np.ndarray]) -> None:
        vw = {k.replace("value_", "", 1): v for k, v in weights.items() if k.startswith("value_")}
        aw = {k.replace("adv_", "", 1): v for k, v in weights.items() if k.startswith("adv_")}
        self.value_net.set_weights(vw)
        self.advantage_net.set_weights(aw)

    def copy_from(self, other: "DuelingNetwork") -> None:
        self.set_weights(other.get_weights())

    def soft_update(self, other: "DuelingNetwork", tau: float) -> None:
        self.value_net.soft_update(other.value_net, tau)
        self.advantage_net.soft_update(other.advantage_net, tau)


# ── DoubleQLearner ────────────────────────────────────────────────────────
class DoubleQLearner:
    """
    Double Q-Learning with dueling architecture, target network, and PER support.

    Algorithm (per Hasselt et al. 2015):
        a* = argmax_a Q_online(s', a)            # online selects best next action
        y  = r + γ · Q_target(s', a*)             # target evaluates it
        L  = Huber(Q_online(s, a) - y)

    The decoupling reduces overestimation bias that plagues vanilla DQN:
    if Q_online is optimistic about a*, Q_target (a separate network) usually
    disagrees and pulls the target down.

    Target network updates: soft (Polyak) every step with τ=0.005.
    """

    def __init__(
        self,
        state_dim: int = STATE_DIM,
        action_dim: int = ACTION_DIM,
        lr: float = DEFAULT_LEARNING_RATE,
        gamma: float = DEFAULT_GAMMA,
        tau: float = DEFAULT_TAU,
        hidden_dim: int = DEFAULT_HIDDEN_DIM,
    ) -> None:
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr
        self.gamma = gamma
        self.tau = tau
        self.hidden_dim = hidden_dim
        self._step = 0

        self.online = DuelingNetwork(state_dim, action_dim, hidden_dim)
        self.target = DuelingNetwork(state_dim, action_dim, hidden_dim)
        self.target.copy_from(self.online)

        # Try to load .npz cache. SQL-backed restore is handled by
        # app.rl.persistence.load_latest_checkpoint() in W16-S7.
        self._load_from_cache()

    # ── cache load / save ──────────────────────────────────────────────
    def _load_from_cache(self) -> None:
        """Load weights + meta from .npz cache if present and dims match.

        If the cached weights were saved with a different state_dim (e.g.
        EVVO legacy 128-dim), quarantine the stale file and start fresh.
        This is the EVVO audit P0-6 fix, preserved.
        """
        if not WEIGHTS_FILE.exists():
            return
        try:
            data = dict(np.load(WEIGHTS_FILE, allow_pickle=False))
            w1 = data.get("value_W1")
            expected_in = self.state_dim
            if w1 is not None and w1.shape[1] != expected_in:
                saved_dim = int(w1.shape[1])
                # Quarantine stale checkpoint so we can post-mortem, then reset
                # the step counter so eval/A-B reporting reflects the fresh
                # (untrained) network honestly.
                try:
                    backup = WEIGHTS_FILE.with_suffix(
                        f".stale-dim{saved_dim}.{int(time.time())}.npz"
                    )
                    WEIGHTS_FILE.replace(backup)
                    logger.warning(
                        "RL_CHECKPOINT_RESET | saved_dim=%d | current_dim=%d | quarantined=%s",
                        saved_dim, expected_in, backup,
                    )
                except Exception as exc:
                    logger.warning(
                        "RL_CHECKPOINT_RESET | saved_dim=%d | current_dim=%d | quarantine_failed=%s",
                        saved_dim, expected_in, exc,
                    )
                self._step = 0
                return

            self.online.set_weights(data)
            self.target.copy_from(self.online)
            if META_FILE.exists():
                meta = json.loads(META_FILE.read_text())
                self._step = meta.get("step", 0)
                self.lr = meta.get("lr", self.lr)
                self.gamma = meta.get("gamma", self.gamma)
                self.tau = meta.get("tau", self.tau)
            logger.info(
                "RL weights loaded from cache | step=%d | dueling=True | double_q=True | params=%d",
                self._step, self._count_params(),
            )
        except Exception as exc:
            logger.warning("RL weight load from cache failed: %s", exc)

    def save_cache(self) -> None:
        """Persist weights + meta to .npz cache (fast-load).

        For durable cross-restart persistence, use app.rl.persistence.save_checkpoint()
        which writes to vapt_rl_checkpoints table.
        """
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        np.savez(WEIGHTS_FILE, **self.online.get_weights())
        META_FILE.write_text(json.dumps({
            "step": self._step,
            "lr": self.lr,
            "gamma": self.gamma,
            "tau": self.tau,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "hidden_dim": self.hidden_dim,
            "architecture": "dueling_double_q",
            "total_params": self._count_params(),
        }))

    def _count_params(self) -> int:
        return sum(w.size for w in self.online.get_weights().values())

    # ── prediction ─────────────────────────────────────────────────────
    def q_values(self, state_vec: np.ndarray) -> np.ndarray:
        """Return Q(s, a) for all actions — shape (action_dim,)."""
        q, _, _ = self.online.forward(state_vec.astype(np.float32))
        return q

    def best_action(self, state_vec: np.ndarray) -> int:
        return int(np.argmax(self.q_values(state_vec)))

    def action_value(self, state_vec: np.ndarray, action: int) -> float:
        return float(self.q_values(state_vec)[action])

    # ── training ───────────────────────────────────────────────────────
    def _compute_target(
        self,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> float:
        """Double Q target: r + γ · Q_target(s', argmax_a Q_online(s', a))."""
        if done:
            return reward
        q_online_next, _, _ = self.online.forward(next_state.astype(np.float32))
        best_a = int(np.argmax(q_online_next))
        q_target_next, _, _ = self.target.forward(next_state.astype(np.float32))
        return reward + self.gamma * float(q_target_next[best_a])

    def train_step(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        is_weight: float = 1.0,
    ) -> float:
        """
        Single Double Q-Learning update with analytical backprop.

        Args:
            state:      s            — (state_dim,)
            action:     a            — int in [0, action_dim)
            reward:     r            — scalar
            next_state: s'           — (state_dim,)
            done:       terminal flag
            is_weight:  PER importance-sampling weight (1.0 for uniform)

        Returns:
            abs(TD error) — used by PER to update priorities.
        """
        s = state.astype(np.float32)
        s_next = next_state.astype(np.float32)

        # Forward pass — get current Q(s, a)
        q_online, _, _ = self.online.forward(s)
        current_q = float(q_online[action])

        # Double Q target
        target = self._compute_target(reward, s_next, done)
        td_error = current_q - target

        # Huber / SmoothL1: ∂L/∂Q = δ if |δ| ≤ κ else κ·sign(δ).
        # Bounds gradient magnitude under large early-training TD errors
        # which would otherwise blow MSE up (audit P2-17).
        if abs(td_error) <= HUBER_KAPPA:
            huber_grad = td_error
        else:
            huber_grad = HUBER_KAPPA * (1.0 if td_error > 0 else -1.0)

        # ∂L/∂Q is non-zero only at the taken action.
        grad_q = np.zeros(self.action_dim, dtype=np.float32)
        grad_q[action] = huber_grad * is_weight

        # Q = V + A - mean(A)  ⇒
        #   ∂L/∂V = Σ_a ∂L/∂Q_a          (V contributes to all Q_a equally)
        #   ∂L/∂A_a = ∂L/∂Q_a − mean(∂L/∂Q)  (because A is centered)
        grad_v = np.array([np.sum(grad_q)], dtype=np.float32)
        grad_a = grad_q - np.mean(grad_q)

        # Backprop through value and advantage streams independently.
        _, h1_v = self.online.value_net.forward(s)
        grads_v = self.online.value_net.backward(s, h1_v, grad_v)

        _, h1_a = self.online.advantage_net.forward(s)
        grads_a = self.online.advantage_net.backward(s, h1_a, grad_a)

        # SGD update on online network.
        for k in grads_v:
            w = getattr(self.online.value_net, k)
            w -= self.lr * grads_v[k]
        for k in grads_a:
            w = getattr(self.online.advantage_net, k)
            w -= self.lr * grads_a[k]

        self._step += 1

        # Soft (Polyak) target update: target ← τ·online + (1-τ)·target
        self.target.soft_update(self.online, self.tau)

        return float(abs(td_error))

    def train_batch(self, batch: list[dict[str, Any]]) -> tuple[float, list[float]]:
        """
        Train on a batch of transitions. Each item may include 'is_weight'
        for PER.

        Args:
            batch: list of dicts with keys: state, action, reward,
                   next_state, done, [is_weight].

        Returns:
            (mean_squared_loss, list_of_abs_td_errors)
            mean_squared_loss is mean(td_error²) over the batch — useful
            for logging / convergence plots.
        """
        td_errors: list[float] = []
        for t in batch:
            td_error = self.train_step(
                state=t["state"],
                action=t["action"],
                reward=t["reward"],
                next_state=t["next_state"],
                done=t["done"],
                is_weight=t.get("is_weight", 1.0),
            )
            td_errors.append(td_error)

        if td_errors:
            self.save_cache()

        mean_loss = float(np.mean(np.square(td_errors))) if td_errors else 0.0
        return mean_loss, td_errors

    # ── introspection ──────────────────────────────────────────────────
    def get_stats(self) -> dict[str, Any]:
        return {
            "step": self._step,
            "lr": self.lr,
            "gamma": self.gamma,
            "tau": self.tau,
            "total_params": self._count_params(),
            "architecture": "dueling_double_q",
            "hidden_dim": self.hidden_dim,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "weights_cache_path": str(WEIGHTS_FILE),
            "cache_loaded": WEIGHTS_FILE.exists(),
        }


# Backward-compat alias (matches EVVO naming — QLearner = DoubleQLearner).
QLearner = DoubleQLearner