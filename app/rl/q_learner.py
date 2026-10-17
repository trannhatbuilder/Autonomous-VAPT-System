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

# Dimensions are DERIVED, not restated.
#
# These used to be `STATE_DIM: int = 337` and `ACTION_DIM: int = 7`, justified
# by a comment claiming a circular import between this module and
# `state_encoder`. That cycle did not exist — `state_encoder` imports nothing
# from `app` — so the duplicate constants bought nothing and cost a real
# desync hazard: adding an eighth action to `ACTION_SPACE` would leave the
# Q-network with seven outputs. See `app/rl/actions.py`.
from app.rl.actions import ACTION_DIM
from app.rl.state_encoder import STATE_DIM, state_layout_id

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────
# Module-level path resolution (settings is a cached singleton at import time).
DATA_DIR: Path = settings.rl_dir
WEIGHTS_FILE: Path = DATA_DIR / "q_weights.npz"
META_FILE: Path = DATA_DIR / "q_meta.json"

DEFAULT_LEARNING_RATE: float = 0.001
DEFAULT_GAMMA: float = 0.95
DEFAULT_TAU: float = 0.005
DEFAULT_HIDDEN_DIM: int = 64

# Huber loss kappa (SmoothL1 threshold). EVVO audit P2-17.
HUBER_KAPPA: float = 1.0

# Cache write throttle for `train_batch` (explicit `save_cache()` is never
# throttled — it is the durability anchor at episode end / shutdown).
#
# `train_batch` used to write the .npz after EVERY batch. Measured cost: the
# state dict is 177,126 bytes (~173 KiB) and `save_cache()` takes ~3.9 ms
# (np.savez + json.dumps + two file writes). With `train_from_memory`
# running every few agent turns, that is a disk write per turn to protect at
# most one batch of progress — the harness already calls `save_cache()`
# explicitly at episode end (harness.py), which is what actually bounds the
# loss window.
#
# The first call after start-up always writes (`_last_cache_save` starts at
# 0.0, so the elapsed check is satisfied), which keeps "one batch ⇒ cache
# exists" true for tests and for a process that trains once and exits.
CACHE_SAVE_MIN_BATCHES: int = 25
CACHE_SAVE_MIN_SECONDS: float = 30.0

# Weight-source problems already reported (one per kind, not per batch).
# Training runs every few turns; a per-batch warning would flood the log.
_WARNED_WEIGHT_ISSUES: set[str] = set()


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
        q, v, a, _, _ = self.forward_detailed(x)
        return q, v, a

    def forward_detailed(
        self, x: np.ndarray,
    ) -> tuple[np.ndarray, float, np.ndarray, np.ndarray, np.ndarray]:
        """Forward pass that also returns the hidden activations.

        `train_step` needs `h1_v` / `h1_a` for the backward pass. It used to
        obtain them by running `value_net.forward()` and
        `advantage_net.forward()` a second time on the same `s`, i.e. two
        redundant forward passes per gradient step — measured at ~68 µs, 13.6%
        of a 502 µs `train_step` (≈4.4 ms per batch of 64). Returning the
        activations from the pass that already computed them costs nothing.

        Returns:
            (q_values, v_value, advantage, h1_value, h1_advantage)
        """
        v_out, h1_v = self.value_net.forward(x)
        v = float(v_out[0])
        a_out, h1_a = self.advantage_net.forward(x)
        a = a_out - np.mean(a_out)  # advantage normalization (identifiability)
        q = v + a
        return q, v, a, h1_v, h1_a

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

        # Throttle state for `_save_cache_throttled()` (see train_batch).
        # 0.0 makes the first save unconditional: monotonic() - 0.0 > 30s.
        self._batches_since_cache_save = 0
        self._last_cache_save: float = 0.0

        self.online = DuelingNetwork(state_dim, action_dim, hidden_dim)
        self.target = DuelingNetwork(state_dim, action_dim, hidden_dim)
        self.target.copy_from(self.online)

        # Try to load .npz cache. SQL-backed restore is handled by
        # app.rl.persistence.load_latest_checkpoint() in W16-S7.
        self._load_from_cache()

    @property
    def train_steps(self) -> int:
        """Gradient steps this learner has actually performed.

        0 means the network is still at its random initialisation, so its
        Q-values carry no information and any argmax over them is noise.
        `HarnessBridge` uses this to decide whether RL guidance is fit to
        be shown to the LLM at all.
        """
        return int(self._step)

    # ── cache load / save ──────────────────────────────────────────────
    def load_cache(self) -> bool:
        """Public re-load of the .npz cache. Returns True if weights loaded.

        `HarnessBridge._load_checkpoint_safe()` calls this. It used to call
        a method that did not exist, so the AttributeError was swallowed by
        the surrounding `except` and the harness logged
        `rl_checkpoint_load_skipped` on EVERY scan even though
        `__init__` -> `_load_from_cache()` had already restored the weights.
        """
        if not WEIGHTS_FILE.exists():
            return False
        self._load_from_cache()
        return True

    def _load_from_cache(self) -> None:
        """Load weights + meta from .npz cache if present and dims match.

        Validates **every** cached array against the shape the live network
        expects — state_dim, hidden_dim, action_dim, and any future dimension —
        and quarantines the file on any difference. `TwoLayerMLP.set_weights()`
        copies blindly with no shape check, so an unchecked mismatch does not
        raise; it installs the wrong-width weights and the network keeps
        running.

        Two silent failure modes this closes (both were live before):

        * `action_dim` 7 → 8: a cached `adv_W2` of shape (7, hidden) is adopted
          by a network declaring 8 actions. `forward()` then returns 7
          Q-values, `argmax` is always in [0, 7), and **action 7 can never be
          selected** — for every scan, while the load reports success.
        * `hidden_dim` 64 → 128: a cached `W1` of shape (64, state) is adopted.
          This does NOT crash (the whole cached set is self-consistent), so the
          network quietly runs 64-wide while `get_stats()` advertises 128.
          The original guard assumed this case would "crash loud"; it does not.

        Only `state_dim` was ever validated, because it is the one dimension
        that also mismatches the *input data* and would raise at forward time.
        Expected shapes are derived from `self.online` so the guard cannot
        drift from the architecture it protects.
        """
        if not WEIGHTS_FILE.exists():
            return
        try:
            data = dict(np.load(WEIGHTS_FILE, allow_pickle=False))

            expected_shapes = {
                key: arr.shape for key, arr in self.online.get_weights().items()
            }
            mismatched = {
                key: (tuple(int(d) for d in data[key].shape), exp)
                for key, exp in expected_shapes.items()
                if key in data and tuple(int(d) for d in data[key].shape) != exp
            }
            missing = sorted(k for k in expected_shapes if k not in data)

            # A matching SHAPE does not identify a matching LAYOUT.
            #
            # SE4 replaced Sections 3/4 of the state vector (hash buckets →
            # direct indices) without changing one array dimension, so the
            # shape guard above is blind to it. Adopting such a cache produces
            # a network that runs perfectly and reads dimensions that no
            # longer mean what it learned: no exception, no warning, just
            # quietly degraded Q-values for the rest of the process's life.
            # `state_layout_id()` is stamped by `save_cache()`, so a cache from
            # this build always matches; one from an older build — or one with
            # no stamp at all, which is exactly what a pre-SE4 cache looks
            # like — is quarantined rather than reinterpreted.
            layout_ok = True
            cached_layout: Any = None
            if META_FILE.exists():
                try:
                    cached_layout = json.loads(META_FILE.read_text()).get("layout")
                except Exception:
                    cached_layout = None
                layout_ok = cached_layout == state_layout_id()

            if mismatched or missing or not layout_ok:
                parts = [
                    f"{key}: cached={got} expected={exp}"
                    for key, (got, exp) in sorted(mismatched.items())
                ]
                if missing:
                    parts.append(f"missing keys: {', '.join(missing)}")
                if not layout_ok:
                    parts.append(
                        f"layout: cached={cached_layout!r} "
                        f"expected={state_layout_id()!r}"
                    )
                detail = " | ".join(parts)
                # Quarantine stale checkpoint so we can post-mortem, then reset
                # the step counter so eval/A-B reporting reflects the fresh
                # (untrained) network honestly. The suffix names the REASON, so
                # a directory full of quarantined caches says why they went.
                reason = "stale-shape" if (mismatched or missing) else "stale-layout"
                try:
                    backup = WEIGHTS_FILE.with_suffix(
                        f".{reason}.{int(time.time())}.npz"
                    )
                    WEIGHTS_FILE.replace(backup)
                    logger.warning(
                        "RL_CHECKPOINT_RESET | %s | quarantined=%s", detail, backup,
                    )
                except Exception as exc:
                    logger.warning(
                        "RL_CHECKPOINT_RESET | %s | quarantine_failed=%s", detail, exc,
                    )
                self._step = 0
                return

            self.online.set_weights(data)
            self.target.copy_from(self.online)

            # A cache written by an older build (before save_cache() refused
            # non-finite weights), or edited by hand, may itself be poisoned.
            # Loading it would make every subsequent forward pass NaN and
            # every action substitution — silently, forever, because nothing
            # ever overwrites the file again once writes are refused. Quarantine
            # it instead so the failure is visible and one-shot.
            bad = self._nonfinite_weight_keys()
            if bad:
                logger.error(
                    "RL_CHECKPOINT_NONFINITE | keys=%s — cache quarantined, "
                    "starting from fresh weights so action selection is not "
                    "silently poisoned.",
                    ", ".join(sorted(bad)),
                )
                try:
                    backup = WEIGHTS_FILE.with_suffix(
                        f".nonfinite.{int(time.time())}.npz"
                    )
                    WEIGHTS_FILE.replace(backup)
                    logger.error("RL_CHECKPOINT_NONFINITE | quarantined=%s", backup)
                except Exception as exc:
                    logger.warning(
                        "RL_CHECKPOINT_NONFINITE | quarantine_failed=%s", exc,
                    )
                self._reset_weights()
                self._step = 0
                return

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

        Unconditional (subject only to the non-finite refusal below) — this is
        the durability anchor called at episode end / shutdown. `train_batch`
        uses `_save_cache_throttled()` instead so a per-batch training loop
        does not write 173 KiB to disk on every batch.

        For durable cross-restart persistence, use app.rl.persistence.save_checkpoint()
        which writes to vapt_rl_checkpoints table.

        Refuses to write non-finite weights. A NaN/Inf checkpoint is not a
        degraded checkpoint, it is a *permanent* one: `_load_from_cache()`
        would restore it into every future process, so one bad training step
        would brick action selection until someone deleted the file by hand.
        Refusing the write leaves the last known-good cache in place, which
        is strictly better than replacing it with poison.
        """
        bad = self._nonfinite_weight_keys()
        if bad:
            logger.error(
                "RL_CACHE_WRITE_REFUSED | non-finite weights in %s — existing "
                "cache left untouched. This is a training blow-up, not a "
                "disk problem.",
                ", ".join(sorted(bad)),
            )
            return
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
            # Feature-layout fingerprint. See `_load_from_cache()`: the array
            # shapes above cannot detect a re-mapping of the state vector.
            "layout": state_layout_id(),
        }))

    def _count_params(self) -> int:
        return sum(w.size for w in self.online.get_weights().values())

    def _save_cache_throttled(self) -> None:
        """Write the cache at most every N batches or M seconds.

        Trade-off, stated explicitly: between saves, up to
        `CACHE_SAVE_MIN_BATCHES` batches (or `CACHE_SAVE_MIN_SECONDS`) of
        gradient progress exist only in memory. That is bounded and small,
        and `save_cache()` is called unconditionally at episode end, so the
        durable-state window is unchanged in practice. What it buys: no disk
        write per `train_batch` (173 KiB, ~3.9 ms each).

        A non-finite blow-up bypasses the throttle: `save_cache()` refuses to
        write and logs loudly, and that signal must not be delayed by up to 25
        batches just because a save happened recently.
        """
        self._batches_since_cache_save += 1
        now = time.monotonic()
        due = (
            self._batches_since_cache_save >= CACHE_SAVE_MIN_BATCHES
            or (now - self._last_cache_save) >= CACHE_SAVE_MIN_SECONDS
            or bool(self._nonfinite_weight_keys())  # never defer a poison report
        )
        if not due:
            return
        self.save_cache()
        self._batches_since_cache_save = 0
        self._last_cache_save = now

    # ── prediction ─────────────────────────────────────────────────────
    def q_values(self, state_vec: np.ndarray) -> np.ndarray:
        """Return Q(s, a) for all actions — shape (action_dim,)."""
        q, _, _ = self.online.forward(state_vec.astype(np.float32))
        return q

    def _nonfinite_weight_keys(self) -> list[str]:
        """Names of weight arrays containing NaN or ±inf.

        Kept as a method (not inline) because both disk boundaries need the
        same test and a divergence between them is exactly how a poisoned
        cache gets written by one path and accepted by the other.
        """
        return [
            key for key, arr in self.online.get_weights().items()
            if not np.all(np.isfinite(arr))
        ]

    def _reset_weights(self) -> None:
        """Discard current weights and re-initialise both networks.

        Used only when a non-finite cache has been quarantined: there is no
        known-good state to fall back to, and keeping the NaN in memory
        would make every forward pass NaN for the rest of the process.
        """
        self.online = DuelingNetwork(self.state_dim, self.action_dim, self.hidden_dim)
        self.target = DuelingNetwork(self.state_dim, self.action_dim, self.hidden_dim)
        self.target.copy_from(self.online)

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

        # Single forward — returns the hidden activations as well, so the
        # backward pass below does not re-run both streams on the same input.
        q_online, _, _, h1_v, h1_a = self.online.forward_detailed(s)
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

        # Backprop through value and advantage streams independently, reusing
        # the h1 activations from the forward pass above.
        grads_v = self.online.value_net.backward(s, h1_v, grad_v)
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

    def train_batch(
        self,
        batch: list[dict[str, Any]],
        is_weights: list[float] | None = None,
    ) -> tuple[float, list[float]]:
        """
        Train on a batch of transitions, with PER importance weighting.

        PER corrects the bias of non-uniform sampling by weighting each
        sample's contribution by w_i = (N·P(i))^-β. This method applies that
        weight to BOTH the gradient and the reported loss.

        Args:
            batch: list of dicts with keys: state, action, reward,
                   next_state, done, [is_weight].
            is_weights: optional per-sample importance weights, positionally
                   aligned with `batch`. When omitted, each item's own
                   `is_weight` key is used (1.0 for uniform replay). When
                   provided, it is authoritative — the two sources are never
                   multiplied together.

        Returns:
            (loss, list_of_abs_td_errors)

            `loss` is the **weighted** mean of td_error², i.e. the quantity
            whose gradient is being followed. It is NOT the Huber objective
            — `train_step` clips the gradient at HUBER_KAPPA, so for
            |td| > kappa this number is larger than the loss actually being
            minimised. It is a monitoring metric; do not compare it across
            code versions that changed kappa or the weighting.

            `td_errors` are **raw** |TD| values, never multiplied by w.
            PER priority updates must use raw TD error — that is an
            invariant of this function, not an implementation detail.

        Weight source contract (why this parameter exists):
            `ExperienceStore.sample()` also injects `is_weight` into every
            transition dict, so the gradient was already weighted even when
            the returned `is_weights` list was ignored. That made the list a
            dead value that looked like a discarded one, and it made
            correctness depend on a silent injection: if that key ever
            stopped being written, PER would degrade to uniform replay with
            no error anywhere. Passing the weights explicitly removes both
            failure modes.
        """
        if is_weights is not None and len(is_weights) != len(batch):
            if "length" not in _WARNED_WEIGHT_ISSUES:
                _WARNED_WEIGHT_ISSUES.add("length")
                logger.warning(
                    "train_batch: is_weights has %d entries but batch has %d — "
                    "falling back to each sample's own is_weight key",
                    len(is_weights), len(batch),
                )
            is_weights = None

        td_errors: list[float] = []
        weighted_sq: list[float] = []
        for i, t in enumerate(batch):
            if is_weights is not None:
                weight = float(is_weights[i])
            else:
                weight = float(t.get("is_weight", 1.0))
            td_error = self.train_step(
                state=t["state"],
                action=t["action"],
                reward=t["reward"],
                next_state=t["next_state"],
                done=t["done"],
                is_weight=weight,
            )
            td_errors.append(td_error)
            weighted_sq.append(weight * td_error * td_error)

        if td_errors:
            self._save_cache_throttled()

        mean_loss = float(np.mean(weighted_sq)) if weighted_sq else 0.0
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