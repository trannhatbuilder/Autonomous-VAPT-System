"""
Pentest Policy: ε-greedy + Boltzmann + curiosity-driven exploration.

Python port of EVVO Sentinel shield_engine/rl/policy.py, adapted with
VAPT-AI master plan §12 W16 epsilon decay schedule:
    ε_start = 1.0  →  ε_end = 0.05  over 200 scans

Three exploration modes (per master plan §12 W16):
  1. epsilon_greedy (default) — random action with prob ε, else argmax Q
  2. boltzmann                 — softmax(Q/T) with T decay 1.0 → 0.1
  3. curiosity                 — ε-greedy + visit-count bonus for novel states

Action resolution translates high-level RL actions into concrete pentest
commands, skills, or KG consultations.

Integration:
  - select_action(state_vec) → (action_idx, action_name, meta)
  - train_from_memory(store) → (loss, td_errors) — sync, SumTree-sampled
  - train_from_db(store)     → (loss, td_errors) — async, DB-sampled
  - end_episode(reward)      → decay ε + T
"""
from __future__ import annotations

import logging
import shlex
from collections import deque
from typing import Any

import numpy as np

from app.rl.q_learner import DoubleQLearner
from app.rl.state_encoder import ACTION_SPACE

logger = logging.getLogger(__name__)

# ── VAPT-AI W16 epsilon schedule (master plan §12 W16) ───────────────────
# ε decays from 1.0 → 0.05 over 200 episodes (one episode = one scan).
#
# The decay factor is DERIVED, never typed by hand:
#     0.05 ** (1/200) = 0.9851329607687279  →  ≈ 0.98513
# This comment used to read "≈ 0.98566", which is wrong: 0.98566 ** 200 =
# 0.055645, i.e. ε would only fall to 0.0556 and never reach ε_end. Keep the
# decimal at 5 places so a mismatch with the constant below is obvious.
DEFAULT_EPSILON_START: float = 1.0
DEFAULT_EPSILON_END: float = 0.05
DEFAULT_EPSILON_DECAY_SCANS: int = 200
# Compute per-episode decay so that ε_end is reached at exactly scan #200
DEFAULT_EPSILON_DECAY: float = (
    (DEFAULT_EPSILON_END / DEFAULT_EPSILON_START) ** (1.0 / DEFAULT_EPSILON_DECAY_SCANS)
)  # ≈ 0.98513

# Boltzmann temperature schedule (per master plan §12 W16)
DEFAULT_TEMPERATURE_START: float = 1.0
DEFAULT_TEMPERATURE_END: float = 0.1
DEFAULT_TEMPERATURE_DECAY: float = 0.995  # per episode

# ── Non-finite Q-value handling ──────────────────────────────────────────
# Substitutes used when the Q-network emits NaN / ±inf (corrupt weights, a
# non-finite state vector, or a numerical blow-up). Chosen so a single bad
# action can neither dominate nor be silently preferred:
#   NaN  → 0.0  — Q-values are centred on the reward range, so 0.0 means
#                 "no estimate", which keeps the other actions comparable.
#   ±inf → ±1e6  — preserves the ordering the network asserted while staying
#                 finite, so `q - q.max()` cannot become `inf - inf = nan`.
NAN_Q_SUBSTITUTE: float = 0.0
INF_Q_SUBSTITUTE: float = 1e6

# Directory name used only in the corruption warning (avoids importing
# q_learner here purely for a path — that would be a circular import).
DATA_DIR_NAME: str = "data/rl"


# ── CuriosityModule ──────────────────────────────────────────────────────
class CuriosityModule:
    """
    Intrinsic curiosity module for exploration bonus.

    Uses a hash-based state visitation count:
      - Novel states get high curiosity bonus
      - Visited states decay in novelty over time
      - Encourages exploration of uncharted attack paths

    The bonus is added to the effective epsilon (ε-greedy mode) or to the
    Boltzmann temperature (Boltzmann mode) — novel states trigger more
    exploratory action selection.
    """

    def __init__(
        self,
        state_dim: int = 337,
        hash_dim: int = 32,
        bonus_scale: float = 0.5,
        decay_rate: float = 0.99,
    ) -> None:
        self.state_dim = state_dim
        self.hash_dim = hash_dim
        self._visit_counts: dict[str, int] = {}
        self._decay_rate = decay_rate
        self._bonus_scale = bonus_scale
        self._decay_counter = 0

    def _hash_state(self, state_vec: np.ndarray) -> str:
        """Hash state vector into a discrete bucket.

        Uses first 16 dims (after digitizing into `hash_dim` bins) for speed.
        Full 337-dim hash would be too sparse to track meaningfully.

        MEASURED, so nobody has to re-derive it: over a 200-scan / 20,000-turn
        run this hash produces only ~767 distinct buckets. The theoretical
        space (33**16) is irrelevant — the reachable space is tiny because
        dims 0-7 are the normalised scalars (turn/max_turns and
        commands_run/max_commands are near-duplicates) and dims 8-15 are the
        first 8 slots of the 256-dim technology multi-hot, which are zero
        whenever the stack is short. Hashing `bins[:8]` instead of `bins[:16]`
        yields the *identical* 767 buckets — dims 8-15 contribute nothing.
        """
        bins = np.digitize(state_vec, np.linspace(-1, 1, self.hash_dim))
        return "_".join(map(str, bins[:16]))

    def compute_bonus(self, state_vec: np.ndarray) -> float:
        """
        Compute curiosity bonus for a state — MUTATES tracker state.

        This is not a pure function. Every call:
          1. increments `_visit_counts[h]` for the hashed state, and
          2. advances `_decay_counter`; every 10th call decays EVERY tracked
             bucket by `_decay_rate` and drops buckets below 0.01.

        So two calls with the same `state_vec` return different values, and the
        second call makes the whole table decay sooner. Use it once per state
        per turn (i.e. once per `select_action*()` call), never as a probe.

        Returns:
            bonus: high for novel states (count=0 → bonus=0.5), low for
            familiar ones (count=100 → bonus=0.05).
        """
        h = self._hash_state(state_vec)
        count = self._visit_counts.get(h, 0)

        # Lazy decay: every 10th call, decay EVERY tracked bucket. An earlier
        # `_decay_batch_size` attribute claimed to cap this, but nothing ever
        # read it — the loop below always walked the whole dict. The attribute
        # is gone; if a cap is ever wanted it has to be implemented, not just
        # declared. Decaying all ~750 buckets every 10 calls costs nothing
        # next to the LLM call this bonus feeds into.
        self._decay_counter += 1
        if self._decay_counter >= 10:
            self._decay_counter = 0
            for k in list(self._visit_counts.keys()):
                self._visit_counts[k] *= self._decay_rate
                if self._visit_counts[k] < 0.01:
                    del self._visit_counts[k]
            # Re-read count after decay so bonus reflects decayed value
            count = self._visit_counts.get(h, 0)

        # Bonus = c / sqrt(N+1) — higher for less-visited states
        bonus = self._bonus_scale / np.sqrt(count + 1.0)
        self._visit_counts[h] = count + 1

        return float(bonus)

    def get_stats(self) -> dict[str, Any]:
        return {
            "unique_states_visited": len(self._visit_counts),
            "total_visits": sum(self._visit_counts.values()),
            "bonus_scale": self._bonus_scale,
            "decay_rate": self._decay_rate,
        }


# ── PentestPolicy ────────────────────────────────────────────────────────
class PentestPolicy:
    """
    ε-greedy policy with curiosity-driven exploration over Double Q-Learner.

    Three exploration modes (selectable at construction time or via
    settings.rl_exploration_mode):

      1. "epsilon_greedy" (default)
           - Random action with prob ε
           - ε decays from 1.0 → 0.05 over 200 scans (per master plan §12 W16)
           - Curiosity bonus added to ε for novel states, SCALED by how much
             of the ε schedule is still unspent (see `_scaled_bonus`)

      2. "boltzmann"
           - softmax(Q/T) action selection
           - T decays from 1.0 → 0.1 over episodes
           - Curiosity bonus added to T for novel states, scaled by how much
             of the T schedule is still unspent

      3. "curiosity"
           - Same as ε-greedy but with stronger curiosity bonus (×2)
           - Useful for exploration-heavy phases (early training)

    Why the bonus is scaled: an unscaled bonus is a CONSTANT added to ε, so
    it becomes a floor that survives the decay schedule. Measured on the
    default config, `epsilon + curiosity_bonus` bottomed out at ~0.20 at
    scan 200 instead of the 0.05 the W16 schedule specifies — a 4× jump in
    random actions that no configuration knob could turn off. Scaling makes
    the contribution vanish exactly as the schedule it augments runs out.
    """

    def __init__(
        self,
        q_learner: DoubleQLearner | None = None,
        epsilon_start: float = DEFAULT_EPSILON_START,
        epsilon_end: float = DEFAULT_EPSILON_END,
        epsilon_decay: float = DEFAULT_EPSILON_DECAY,
        temperature_start: float = DEFAULT_TEMPERATURE_START,
        temperature_end: float = DEFAULT_TEMPERATURE_END,
        temperature_decay: float = DEFAULT_TEMPERATURE_DECAY,
        use_curiosity: bool = True,
        exploration_mode: str = "epsilon_greedy",
        curiosity_bonus_scale: float = 0.5,
    ) -> None:
        self.q = q_learner or DoubleQLearner()
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.temperature = temperature_start
        self.temperature_start = temperature_start
        self.temperature_end = temperature_end
        self.temperature_decay = temperature_decay
        self.exploration_mode = exploration_mode
        self._episode = 0
        self._step = 0

        self.use_curiosity = use_curiosity
        bonus_scale = curiosity_bonus_scale * (2.0 if exploration_mode == "curiosity" else 1.0)
        self.curiosity = CuriosityModule(bonus_scale=bonus_scale) if use_curiosity else None

        # Track episode rewards for adaptive epsilon decay
        self._episode_rewards: deque[float] = deque(maxlen=10)

        # Non-finite Q-value tally. `_nonfinite_q_seen` accumulates every
        # substitution; `_nonfinite_q_warned` gates the warning to once per
        # policy instance so 200 turns of a poisoned net don't bury the
        # first (actionable) log line.
        self._nonfinite_q_seen: int = 0
        self._nonfinite_q_warned: bool = False

    # ── curiosity scaling ─────────────────────────────────────────────

    @staticmethod
    def _schedule_progress(value: float, start: float, end: float) -> float:
        """Fraction of an exploration schedule that is still unspent, in [0, 1].

        Returns 1.0 at `start`, 0.0 at `end`, clamped outside `[end, start]`.
        A degenerate or inverted schedule (`start <= end`) returns 0.0: there
        is no decaying budget to spend, so nothing is allowed to extend it.
        """
        span = start - end
        if span <= 0:
            return 0.0
        return float(np.clip((value - end) / span, 0.0, 1.0))

    def _scaled_bonus(self, bonus: float, *, boltzmann: bool = False) -> float:
        """Scale a curiosity bonus by the schedule that is actually in play.

        ε-greedy scales by ε, Boltzmann by T. Scaling the Boltzmann bonus by
        ε would tie it to a knob nobody is reading in that mode, and vice
        versa — so the caller says which schedule it means.

        Contract: at `epsilon == epsilon_end` the contribution is exactly 0.0,
        so `effective_epsilon == epsilon_end`. That is the whole point — the
        W16 schedule must be reachable.
        """
        if boltzmann:
            weight = self._schedule_progress(
                self.temperature, self.temperature_start, self.temperature_end
            )
        else:
            weight = self._schedule_progress(
                self.epsilon, self.epsilon_start, self.epsilon_end
            )
        return float(bonus) * weight

    # ── action selection ──────────────────────────────────────────────

    def select_action(self, state_vec: np.ndarray) -> tuple[int, str, dict[str, Any]]:
        """
        Select action using the configured exploration mode.

        CALLER CONTRACT — call exactly ONE selection method per state, per
        turn. `select_action()`, `select_action_boltzmann()` and a direct
        `curiosity.compute_bonus()` all record a visit for the hashed state as
        a SIDE EFFECT (`_CuriosityTracker.compute_bonus()` increments
        `_visit_counts[h]` and advances the lazy-decay counter). Calling two of
        them for the same `state_vec` in one turn therefore counts the visit
        twice and decays every bucket twice as fast — the bonus collapses
        toward 0 while the state is still novel, which is the exact opposite of
        what the signal is for. Nothing raises or logs; the schedule just runs
        at double speed. If a caller needs to inspect the bonus without
        consuming a visit, it must not use these methods.

        Returns:
            (action_index, action_name, meta)

            Both modes emit the same core keys, so the caller does not have
            to branch on `mode`:

              q_values, selected_action, best_action, epsilon,
              exploratory, curiosity_bonus, curiosity_contribution, mode

            Mode-specific extras: `effective_epsilon` (ε-greedy),
            `probs` / `temperature` / `effective_temp` (Boltzmann).
            `epsilon` is the exploration rate ACTUALLY in play — the ε
            schedule for ε-greedy, and `1 - P(argmax)` for Boltzmann.

            `q_values_sanitized` is the count of NaN/±inf entries replaced
            before selection (see `_sanitize_q_values`); non-zero means the
            network emitted non-finite output.
        """
        self._step += 1
        q_vals = self.q.q_values(state_vec)

        # Curiosity bonus (computed once per state visit)
        if self.curiosity is not None:
            curiosity_bonus = self.curiosity.compute_bonus(state_vec)
        else:
            curiosity_bonus = 0.0

        if self.exploration_mode == "boltzmann":
            return self._select_boltzmann(q_vals, curiosity_bonus)
        # Default: epsilon_greedy or curiosity (both use ε-greedy with bonus)
        return self._select_epsilon_greedy(q_vals, curiosity_bonus)

    def _sanitize_q_values(
        self, q_vals: np.ndarray, where: str
    ) -> tuple[np.ndarray, int]:
        """Replace non-finite Q(s,·) with finite substitutes.

        A Dueling net with NaN or ±inf in its parameters (or a state vector
        that is itself non-finite) returns a non-finite Q row. Nothing in
        the stack reacted to that:

          - `np.argmax` on a NaN row returns the index of the NaN, so
            ε-greedy would *silently* commit to a poisoned action on every
            turn — measured against all-NaN weights: `action=0,
            exploratory=False, q=nan`. This is the DEFAULT mode.
          - the Boltzmann softmax propagated the NaN into `probs`, so
            `np.random.choice(..., p=probs)` raised
            `ValueError: probabilities contain NaN` and killed the scan.

        The substitution keeps the ordering the network asserted where it
        asserted one: NaN means "no estimate" and becomes 0.0 (the centre of
        the reward range, neither preferred nor avoided); ±inf is clamped to
        a large finite value so an infinitely-preferred action stays
        preferred without producing `inf - inf = nan` in the stabiliser.

        Returns `(sanitized, n_replaced)`. `n_replaced` is surfaced in meta
        so the corruption is visible in logs and scan records rather than
        being silently absorbed.
        """
        arr = np.asarray(q_vals, dtype=np.float64).reshape(-1)
        bad = ~np.isfinite(arr)
        n_bad = int(bad.sum())
        if n_bad == 0:
            return arr, 0

        arr = np.nan_to_num(
            arr, nan=NAN_Q_SUBSTITUTE, posinf=INF_Q_SUBSTITUTE,
            neginf=-INF_Q_SUBSTITUTE,
        )
        # One warning per policy instance: a poisoned network fires this on
        # every turn of every scan, and 200 identical tracebacks bury the
        # first (useful) one. The counter in meta carries the full tally.
        self._nonfinite_q_seen += n_bad
        if not self._nonfinite_q_warned:
            self._nonfinite_q_warned = True
            logger.warning(
                "rl_nonfinite_q_values | mode=%s | where=%s | replaced=%d/%d "
                "| q=%s — Q-network produced non-finite output; action "
                "selection is using substitutes. Check the checkpoint in "
                "%s for NaN/Inf weights.",
                self.exploration_mode, where, n_bad, arr.size,
                np.asarray(q_vals).tolist(), DATA_DIR_NAME,
            )
        else:
            logger.debug(
                "rl_nonfinite_q_values | where=%s | replaced=%d | total=%d",
                where, n_bad, self._nonfinite_q_seen,
            )
        return arr, n_bad

    def _select_epsilon_greedy(
        self,
        q_vals: np.ndarray,
        curiosity_bonus: float,
    ) -> tuple[int, str, dict[str, Any]]:
        """ε-greedy with curiosity-modulated exploration.

        Adding a scalar to ALL q_vals is a no-op for argmax, so we boost
        the exploration probability directly — novel states trigger more
        random action selection, which is the intended effect of curiosity.

        The boost is SCALED (`_scaled_bonus`), not added raw. Raw addition
        makes the bonus a floor that survives the whole decay schedule.
        """
        curiosity_contribution = self._scaled_bonus(curiosity_bonus)
        effective_epsilon = min(1.0, self.epsilon + curiosity_contribution)
        q_vals, n_sanitized = self._sanitize_q_values(
            q_vals, "epsilon_greedy"
        )
        best_a = int(np.argmax(q_vals))

        if np.random.random() < effective_epsilon:
            action = int(np.random.randint(0, len(ACTION_SPACE)))
            exploratory = True
        else:
            action = best_a
            exploratory = False

        meta = {
            "q_values": q_vals.tolist(),
            "q_values_sanitized": n_sanitized,
            "selected_action": action,
            "best_action": best_a,
            "exploratory": exploratory,
            "epsilon": round(self.epsilon, 4),
            "effective_epsilon": round(effective_epsilon, 4),
            # `curiosity_bonus` is the RAW bonus; `curiosity_contribution` is
            # what was actually added to ε. Both are reported so the two can
            # never be confused in a log or a post-mortem.
            "curiosity_bonus": round(curiosity_bonus, 4),
            "curiosity_contribution": round(curiosity_contribution, 4),
            "mode": self.exploration_mode,
        }
        return action, ACTION_SPACE[action], meta

    def _select_boltzmann(
        self,
        q_vals: np.ndarray,
        curiosity_bonus: float,
    ) -> tuple[int, str, dict[str, Any]]:
        """Boltzmann (softmax) exploration.

        Novel states: increase temperature → more uniform action distribution.
        Familiar states: low temperature → sharp argmax.

        The boost is scaled by how much of the TEMPERATURE schedule is left,
        not the ε one — this method can be forced even in ε-greedy mode, and
        scaling by ε there would tie the bonus to an unused knob.

        Two meta fields are MEASUREMENTS, not constants:

          exploratory — True iff the sampled action differs from argmax(Q).
              Softmax has no "random branch" the way ε-greedy does: every
              action is drawn from a distribution, so the only honest
              question is whether the draw disagreed with the greedy
              choice. Hardcoding True (what this used to do) made the field
              unusable: at `temperature == temperature_end` the softmax is
              effectively deterministic — measured 2000/2000 picks equal to
              argmax — and it still reported 100% exploratory. That killed
              the B3 shadow-mode gate (`not rec.exploratory`) on every turn,
              and inflated `exploratory_count` in `/rl/stats` to the turn
              count.

          epsilon — `1 - P(argmax)`, the realised probability that this
              pick was NOT the greedy one, i.e. the softmax equivalent of
              ε-greedy's exploration rate. It is reported because
              `HarnessBridge.select_action()` reads `meta.get("epsilon",
              policy.epsilon)` for the guidance text, the `rl_select` log and
              `TurnRecommendation.epsilon`. Omitting the key made that
              fallback fire, so Boltzmann runs displayed the ε-greedy knob
              (a decaying schedule this mode never reads) as if it
              described the draw.

        Note the intentional asymmetry with `_select_epsilon_greedy`: there
        `exploratory` means "the random branch was taken" and can be True
        even when the random draw happens to equal argmax (≈1/A of such
        turns). That is the conservative direction — it suppresses guidance
        for an action that happens to match the greedy pick — so it is left
        alone rather than redefined.

        Non-finite `q_vals` are substituted by `_sanitize_q_values()` before
        the softmax. Without it the NaN reached `probs` and
        `np.random.choice(..., p=probs)` raised
        `ValueError: probabilities contain NaN`, taking the whole scan down.

        The distribution is NOT renormalised a second time. `exp_q /
        exp_q.sum()` already sums to 1 by construction, and after
        sanitisation `max(q) - max(q) = 0` guarantees `exp_q` contains at
        least one 1.0, so the sum can never be 0. Measured over 200 000
        draws across T ∈ {0.01 … 3.0}, the worst `|sum(p) - 1|` was
        5.6e-16 — thirteen orders of magnitude below numpy's 1.49e-08
        tolerance. A second division is a numerical no-op; what actually
        needs guarding is a non-finite sum, which cannot arise after
        sanitisation but is asserted anyway.
        """
        curiosity_contribution = self._scaled_bonus(curiosity_bonus, boltzmann=True)
        effective_temp = self.temperature + curiosity_contribution * 0.2
        q_vals, n_sanitized = self._sanitize_q_values(q_vals, "boltzmann")
        best_a = int(np.argmax(q_vals))
        # Softmax with temperature (numerically stable: subtract max before exp)
        exp_q = np.exp((q_vals - q_vals.max()) / max(effective_temp, 0.01))
        total = float(exp_q.sum())
        if not np.isfinite(total) or total <= 0.0:
            # Unreachable after sanitisation (max element ⇒ exp(0) = 1), but
            # a uniform draw is the honest "no information" answer if some
            # future change reintroduces a degenerate row.
            logger.warning(
                "rl_softmax_degenerate | sum=%r | temp=%r — falling back to "
                "a uniform draw",
                total, effective_temp,
            )
            probs = np.full(len(ACTION_SPACE), 1.0 / len(ACTION_SPACE))
        else:
            probs = exp_q / total
        action = int(np.random.choice(len(ACTION_SPACE), p=probs))

        meta = {
            "q_values": q_vals.tolist(),
            "q_values_sanitized": n_sanitized,
            "selected_action": action,
            "best_action": best_a,
            "probs": probs.tolist(),
            "temperature": round(self.temperature, 4),
            "effective_temp": round(effective_temp, 4),
            # Softmax equivalent of ε: P(this pick is not the greedy one).
            "epsilon": round(1.0 - float(probs[best_a]), 4),
            "curiosity_bonus": round(curiosity_bonus, 4),
            "curiosity_contribution": round(curiosity_contribution, 4),
            "exploratory": action != best_a,
            "mode": "boltzmann",
        }
        return action, ACTION_SPACE[action], meta

    def select_action_boltzmann(self, state_vec: np.ndarray) -> tuple[int, str, dict[str, Any]]:
        """Force Boltzmann exploration regardless of configured mode.

        Useful when ε is low but we still want some exploration.

        CALLER CONTRACT — see `select_action()`: one selection method per
        state per turn. This method calls `curiosity.compute_bonus()`, which
        RECORDS A VISIT as a side effect, so calling it *and* `select_action()`
        on the same `state_vec` double-counts the visit and double-advances the
        lazy-decay counter.

        Unlike `select_action()`, this method does not increment `self._step`
        (the counter reported as `get_stats()["steps"]`), so selections made
        through this entry point do not advance the policy's step count.
        """
        q_vals = self.q.q_values(state_vec)
        if self.curiosity is not None:
            curiosity_bonus = self.curiosity.compute_bonus(state_vec)
        else:
            curiosity_bonus = 0.0
        return self._select_boltzmann(q_vals, curiosity_bonus)

    # ── episode management ────────────────────────────────────────────

    def end_episode(self, episode_reward: float) -> None:
        """
        Call at end of each scan (episode) to decay ε + temperature.

        Uses reward-based adaptive decay:
          - High reward (>5)   → faster ε decay (exploit more)
          - Low reward (<-5)   → slower ε decay (needs more exploration)
          - Otherwise          → nominal decay (0.98513 per scan = 0.05^(1/200))
        """
        self._episode += 1
        self._episode_rewards.append(episode_reward)

        # Adaptive decay based on recent performance
        avg_reward = float(np.mean(self._episode_rewards)) if self._episode_rewards else 0.0

        if avg_reward > 5.0:
            effective_decay = self.epsilon_decay ** 1.2  # faster decay
        elif avg_reward < -5.0:
            effective_decay = self.epsilon_decay ** 0.8  # slower decay
        else:
            effective_decay = self.epsilon_decay

        self.epsilon = max(
            self.epsilon_end,
            self.epsilon * effective_decay,
        )

        # Decay Boltzmann temperature (1.0 → 0.1 over episodes)
        self.temperature = max(
            self.temperature_end,
            self.temperature * self.temperature_decay,
        )

    def reset_exploration(self, epsilon: float | None = None) -> None:
        """Reset exploration rate (e.g. for evaluation mode)."""
        if epsilon is not None:
            self.epsilon = epsilon
        else:
            self.epsilon = self.epsilon_start
        self.temperature = self.temperature_start

    # ── cross-scan persistence of the exploration schedule ────────────

    def state_dict(self) -> dict[str, Any]:
        """Exploration state that MUST outlive the process/scan boundary.

        `end_episode()` moves ε and T along a schedule that is defined in
        *scans*, not in turns (W16: 1.0 → 0.05 over 200 scans), so the
        values only mean anything if the next scan starts where the last
        one stopped. Saving Q weights alone silently reset the schedule to
        ε_start on every scan: measured, three consecutive scans all began
        at ε=1.000000 and all ended at 0.970487, so the schedule never
        advanced and every turn of every scan was a random pick.

        `_episode_rewards` is included because `end_episode()` uses it for
        the reward-adaptive branch (avg > 5 → faster decay, < -5 → slower).
        Without it, the deque is empty on the first turn of each scan and
        the adaptive branch can never fire.
        """
        return {
            "epsilon": float(self.epsilon),
            "temperature": float(self.temperature),
            "episode": int(self._episode),
            "episode_rewards": [float(r) for r in self._episode_rewards],
        }

    def load_state_dict(self, state: dict[str, Any]) -> bool:
        """Restore state produced by `state_dict()`. Returns True if applied.

        Clamps ε into `[epsilon_end, epsilon_start]` and T into
        `[temperature_end, temperature_start]`: a hand-edited or
        mismatched-config file must not be able to move ε outside the
        configured schedule. Rejects the whole payload on any bad value so
        a partially-valid file can't produce a half-restored policy.
        """
        try:
            eps = float(state["epsilon"])
            temp = float(state["temperature"])
            episode = int(state["episode"])
            rewards = [float(r) for r in state.get("episode_rewards", [])]
        except (KeyError, TypeError, ValueError) as exc:
            logger.warning("policy_state_reject | err=%s | payload=%r", exc, state)
            return False

        if not np.isfinite(eps) or not np.isfinite(temp):
            logger.warning("policy_state_reject | non-finite eps=%r T=%r", eps, temp)
            return False

        self.epsilon = float(
            np.clip(eps, self.epsilon_end, self.epsilon_start)
        )
        self.temperature = float(
            np.clip(temp, self.temperature_end, self.temperature_start)
        )
        self._episode = max(0, episode)
        self._episode_rewards.clear()
        self._episode_rewards.extend(rewards[-self._episode_rewards.maxlen:])
        return True

    # ── mapping to concrete tool / skill / strategy ───────────────────

    def resolve_action(
        self,
        action_name: str,
        session: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Translate a high-level RL action into concrete pentest parameters
        that the agent loop can execute.

        Args:
            action_name: one of ACTION_SPACE
            session: live session dict (must contain 'target_url' or 'target')

        Returns:
            dict with `type` and action-specific fields. Caller dispatches:
              - "execute_command"  → SubprocessExecutor.execute(...)
              - "select_skill"     → SkillLoader.load(...) + agent context
              - "consult_kg"       → KG.query(tech_stack=...)
              - "report_vulnerability" → agent emits finding
              - "pentest_complete" → scan ends

        Round-trip contract (credit assignment):
            For every action that produces a command, the FIRST TOKEN of that
            command must map back to the SAME action in
            `app.agents.react_agent._TOOL_TO_ACTION`. Otherwise the agent loop
            credits the reward to a different action than the one the policy
            chose, and `train_step` puts the gradient on `grad_q[action]` for
            the wrong `action` — a systematic, not random, wrong label.

            This is why the exploit branch only emits `sqlmap`: the branch
            used to emit `nuclei` (credited as `execute_command_fuzz`) and a
            `curl` pipeline (not a tool at all, so the transition was skipped
            entirely). `tests/unit/test_rl_action_mapping_consistency.py`
            enforces the invariant, so a future edit to this function cannot
            silently re-break credit assignment.

        Unknown actions:
            Substituted with recon and logged at WARNING. The previous
            behaviour fell through to a bare `curl -sI` with no log line, so a
            caller passing `action_name` from a drifted enum produced a
            plausible-looking command that silently did the wrong thing.
            The returned dict carries `requested_action` so the caller can
            tell a substitution happened; `rl_action` always reports what was
            actually chosen, never the rejected input.
        """
        target = session.get("target_url") or session.get("target") or ""
        recon = session.get("recon_signals") or {}
        tech = recon.get("technologies") or []

        requested_action = action_name
        if action_name not in ACTION_SPACE:
            logger.warning(
                "rl_resolve_action_unknown | action=%r | known=%s — "
                "substituting execute_command_recon. A caller is passing an "
                "action name the policy cannot produce; check the enum.",
                action_name, list(ACTION_SPACE),
            )
            action_name = "execute_command_recon"

        if action_name == "execute_command_recon":
            turn = session.get("_turn", 0)
            if turn <= 3:
                cmd = f"whatweb {shlex.quote(target)} --color=never"
            else:
                cmd = f"httpx -silent -title -tech-detect {shlex.quote(target)}"
            out: dict[str, Any] = {
                "type": "execute_command",
                "command": cmd,
                "rl_action": action_name,
            }
            if requested_action != action_name:
                # Substituted: say so, and say what we were asked for.
                out["requested_action"] = requested_action
                out["fallback"] = True
            return out

        if action_name == "execute_command_exploit":
            # ONLY binaries that `_TOOL_TO_ACTION` credits as exploit. See the
            # round-trip contract in the docstring: emitting `nuclei` here made
            # the reward land on `execute_command_fuzz`, teaching the policy
            # that exploitation had been attempted when a scan had.
            if "wordpress" in [t.lower() for t in tech]:
                cmd = (
                    f"sqlmap -u {shlex.quote(target)} --batch --level 3 "
                    f"--risk 2 --crawl 2"
                )
            elif any("graphql" in t.lower() or "graphiql" in t.lower() for t in tech):
                cmd = (
                    f"sqlmap -u {shlex.quote(target)}/graphql --batch "
                    f"--level 2 --risk 2 "
                    f"--data '{{\"query\":\"{{__typename}}\"}}'"
                )
            else:
                cmd = (
                    f"sqlmap -u {shlex.quote(target)} --batch "
                    f"--level 2 --risk 2"
                )
            return {"type": "execute_command", "command": cmd, "rl_action": action_name}

        if action_name == "execute_command_fuzz":
            cmd = (
                f"ffuf -u {shlex.quote(target)}/FUZZ "
                f"-w /usr/share/wordlists/dirb/common.txt -t 50 "
                f"-mc 200,301,302,403 -s"
            )
            return {"type": "execute_command", "command": cmd, "rl_action": action_name}

        if action_name == "select_skill":
            return {"type": "select_skill", "rl_action": action_name, "tech_stack": tech}

        if action_name == "consult_kg":
            return {"type": "consult_kg", "tech_stack": tech, "rl_action": action_name}

        if action_name == "report_vulnerability":
            return {"type": "report_vulnerability", "rl_action": action_name}

        if action_name == "pentest_complete":
            return {"type": "pentest_complete", "rl_action": action_name}

        # Unreachable: the unknown-action guard at the top of this method has
        # already substituted recon, and every ACTION_SPACE member is handled
        # above. Kept as a total function so a missing branch degrades to a
        # real tool instead of raising mid-scan — but it must never be silent.
        logger.error(
            "rl_resolve_action_unhandled | action=%r — this is a bug: the "
            "action is in ACTION_SPACE but has no branch.",
            action_name,
        )
        return {
            "type": "execute_command",
            "command": f"whatweb {shlex.quote(target)} --color=never",
            "rl_action": "execute_command_recon",
            "requested_action": requested_action,
            "fallback": True,
        }

    # ── training helpers ──────────────────────────────────────────────

    def train_from_memory(
        self,
        store: Any,  # ExperienceStore
        batch_size: int = 64,
    ) -> tuple[float, list[float]]:
        """Sample from PER store and run one gradient update (sync).

        The sampled `is_weights` are passed to `train_batch` **explicitly**
        and are what corrects the bias of prioritized sampling. Note the two
        distinct outputs and what each one is for:

          - `is_weights`  → the gradient, via train_batch. Bias correction.
          - `td_errors`   → `update_priorities`. These must stay RAW
            (|TD| without w); weighting them again would square the
            correction and bias priorities toward already-high-priority
            transitions.

        (The store also injects `is_weight` per transition, so the gradient
        was weighted even before this argument existed — but relying on that
        meant a silent injection failure would turn PER into uniform replay
        with no error. The list is authoritative now.)

        Args:
            store: ExperienceStore instance
            batch_size: PER batch size (default 64)

        Returns:
            (weighted_loss, raw_td_errors) — td_errors used by caller to
            update SumTree priorities.

        BLOCKING — this is a synchronous CPU-bound call. Called from inside an
        `async def` (which `HarnessBridge._train_one_batch()` does, under
        `_q_lock`), it stalls the event loop for the whole forward + backward
        pass: no SSE flush, no other scan's turn, no health check.

        Measured on this machine, DuelingNetwork hidden_dim=64, plain numpy:

            batch=  64  →  31.7 ms
            batch= 256  → 123.2 ms
            batch=1024  → 470.9 ms

        (Linear in batch, ~0.46 ms/sample.) 31 ms at the default batch size is
        tolerable next to a multi-second LLM call *per scan*, but it is not
        "a few ms", and a single process serves every concurrent scan. If
        TRAIN_BATCH_SIZE grows past ~256, or the worker runs more than a few
        times per turn, wrap the call site in `asyncio.to_thread(...)` — the
        body touches only numpy and the SumTree, so it is thread-safe as long
        as `_q_lock` is held.
        """
        batch, is_weights, indices = store.sample(batch_size)
        if not batch:
            return 0.0, []

        loss, td_errors = self.q.train_batch(batch, is_weights=is_weights)

        # Update priorities in PER store (sync, SumTree only)
        store.update_priorities(indices, td_errors)

        logger.info(
            "RL train_step | batch=%d loss=%.4f ε=%.4f β=%.3f",
            len(batch), loss, self.epsilon,
            # `beta` is a public property on ExperienceStore. Test doubles /
            # stubs that do not model PER simply have no β, and 1.0 (no
            # correction) is the right thing to report for them — but they
            # must not need a private attribute to say so.
            getattr(store, "beta", 1.0),
        )
        return loss, td_errors

    async def train_from_db(
        self,
        store: Any,
        batch_size: int = 256,
        scan_id: str | None = None,
        session: Any = None,
    ) -> tuple[float, list[float]]:
        """Pull a larger batch from Postgres for offline training (async).

        Sampling is uniform (`ORDER BY random()` in `sample_from_db`), which
        is why `train_batch` is called WITHOUT `is_weights`: every sample
        carries `is_weight = 1.0` and uniform sampling needs no importance
        correction. If `sample_from_db` ever goes back to priority-ordered
        sampling, this call must pass real IS weights — otherwise the
        off-policy bias is trained on uncorrected.

        Args:
            store: ExperienceStore instance
            batch_size: DB batch size (default 256 — larger than memory batch)
            scan_id: optional filter (only train on transitions from this scan)
            session: optional caller-owned AsyncSession, forwarded to
                `sample_from_db` (it used to be dropped on the floor)

        Returns:
            (mean_squared_loss, td_errors)
        """
        batch = await store.sample_from_db(
            batch_size, scan_id=scan_id, session=session,
        )
        if not batch:
            return 0.0, []
        loss, td_errors = self.q.train_batch(batch)
        logger.info(
            "RL offline_train | batch=%d loss=%.4f scan_id=%s",
            len(batch), loss, scan_id or "(all)",
        )
        return loss, td_errors

    # ── introspection ─────────────────────────────────────────────────

    def get_stats(self) -> dict[str, Any]:
        stats: dict[str, Any] = {
            "epsilon": round(self.epsilon, 4),
            "epsilon_start": self.epsilon_start,
            "epsilon_end": self.epsilon_end,
            "epsilon_decay": round(self.epsilon_decay, 6),
            "temperature": round(self.temperature, 4),
            "episodes": self._episode,
            "steps": self._step,
            "use_curiosity": self.use_curiosity,
            "exploration_mode": self.exploration_mode,
        }
        if self.curiosity:
            stats["curiosity"] = self.curiosity.get_stats()
        stats.update(self.q.get_stats())
        return stats