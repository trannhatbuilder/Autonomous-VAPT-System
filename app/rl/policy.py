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
# ε decays from 1.0 → 0.05 over 200 scans.
# Per-episode decay factor: 0.05^(1/200) ≈ 0.98566
DEFAULT_EPSILON_START: float = 1.0
DEFAULT_EPSILON_END: float = 0.05
DEFAULT_EPSILON_DECAY_SCANS: int = 200
# Compute per-episode decay so that ε_end is reached at exactly scan #200
import math
DEFAULT_EPSILON_DECAY: float = (
    (DEFAULT_EPSILON_END / DEFAULT_EPSILON_START) ** (1.0 / DEFAULT_EPSILON_DECAY_SCANS)
)  # ≈ 0.98566

# Boltzmann temperature schedule (per master plan §12 W16)
DEFAULT_TEMPERATURE_START: float = 1.0
DEFAULT_TEMPERATURE_END: float = 0.1
DEFAULT_TEMPERATURE_DECAY: float = 0.995  # per episode


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
        self._decay_batch_size = 100  # Decay at most 100 entries per call

    def _hash_state(self, state_vec: np.ndarray) -> str:
        """Hash state vector into a discrete bucket.

        Uses first 16 dims (after digitizing into `hash_dim` bins) for speed.
        Full 337-dim hash would be too sparse to track meaningfully.
        """
        bins = np.digitize(state_vec, np.linspace(-1, 1, self.hash_dim))
        return "_".join(map(str, bins[:16]))

    def compute_bonus(self, state_vec: np.ndarray) -> float:
        """
        Compute curiosity bonus for a state.

        Returns:
            bonus: high for novel states (count=0 → bonus=0.5), low for
            familiar ones (count=100 → bonus=0.05).
        """
        h = self._hash_state(state_vec)
        count = self._visit_counts.get(h, 0)

        # Lazy batch decay: only decay a subset every 10 calls (perf)
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
           - Curiosity bonus added to ε for novel states

      2. "boltzmann"
           - softmax(Q/T) action selection
           - T decays from 1.0 → 0.1 over episodes
           - Curiosity bonus added to T for novel states

      3. "curiosity"
           - Same as ε-greedy but with stronger curiosity bonus (×2)
           - Useful for exploration-heavy phases (early training)
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

    # ── action selection ──────────────────────────────────────────────

    def select_action(self, state_vec: np.ndarray) -> tuple[int, str, dict[str, Any]]:
        """
        Select action using the configured exploration mode.

        Returns:
            (action_index, action_name, meta)
            meta contains: q_values, selected_action, best_action, exploratory,
                          epsilon/effective_epsilon, curiosity_bonus, mode
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

    def _select_epsilon_greedy(
        self,
        q_vals: np.ndarray,
        curiosity_bonus: float,
    ) -> tuple[int, str, dict[str, Any]]:
        """ε-greedy with curiosity-modulated exploration.

        Adding a scalar to ALL q_vals is a no-op for argmax, so we boost
        the exploration probability directly — novel states trigger more
        random action selection, which is the intended effect of curiosity.
        """
        effective_epsilon = min(1.0, self.epsilon + curiosity_bonus)
        best_a = int(np.argmax(q_vals))

        if np.random.random() < effective_epsilon:
            action = int(np.random.randint(0, len(ACTION_SPACE)))
            exploratory = True
        else:
            action = best_a
            exploratory = False

        meta = {
            "q_values": q_vals.tolist(),
            "selected_action": action,
            "best_action": best_a,
            "exploratory": exploratory,
            "epsilon": round(self.epsilon, 4),
            "effective_epsilon": round(effective_epsilon, 4),
            "curiosity_bonus": round(curiosity_bonus, 4),
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
        """
        effective_temp = self.temperature + curiosity_bonus * 0.2
        # Softmax with temperature (numerically stable: subtract max before exp)
        exp_q = np.exp((q_vals - q_vals.max()) / max(effective_temp, 0.01))
        probs = exp_q / exp_q.sum()
        action = int(np.random.choice(len(ACTION_SPACE), p=probs))

        meta = {
            "q_values": q_vals.tolist(),
            "selected_action": action,
            "probs": probs.tolist(),
            "temperature": round(self.temperature, 4),
            "effective_temp": round(effective_temp, 4),
            "curiosity_bonus": round(curiosity_bonus, 4),
            "exploratory": True,
            "mode": "boltzmann",
        }
        return action, ACTION_SPACE[action], meta

    def select_action_boltzmann(self, state_vec: np.ndarray) -> tuple[int, str, dict[str, Any]]:
        """Force Boltzmann exploration regardless of configured mode.

        Useful when ε is low but we still want some exploration.
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
          - Otherwise          → nominal decay (0.98566 per scan)
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
        """
        target = session.get("target_url") or session.get("target") or ""
        recon = session.get("recon_signals") or {}
        tech = recon.get("technologies") or []

        if action_name == "execute_command_recon":
            turn = session.get("_turn", 0)
            if turn <= 3:
                cmd = f"whatweb {shlex.quote(target)} --color=never"
            else:
                cmd = f"httpx -silent -title -tech-detect {shlex.quote(target)}"
            return {"type": "execute_command", "command": cmd, "rl_action": action_name}

        if action_name == "execute_command_exploit":
            if "wordpress" in [t.lower() for t in tech]:
                cmd = f"nuclei -u {shlex.quote(target)} -t wordpress -silent"
            elif any("graphql" in t.lower() or "graphiql" in t.lower() for t in tech):
                cmd = (
                    f"curl -s {shlex.quote(target)}/graphql -X POST "
                    f"-H 'Content-Type: application/json' "
                    f"-d '{{\"query\":\"{{__typename}}\"}}'"
                )
            else:
                cmd = (
                    f"nuclei -u {shlex.quote(target)} "
                    f"-severity medium,high,critical -silent"
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

        # Fallback (shouldn't reach here — action_name ∈ ACTION_SPACE)
        return {
            "type": "execute_command",
            "command": f"curl -sI {shlex.quote(target)}",
            "rl_action": action_name,
        }

    # ── training helpers ──────────────────────────────────────────────

    def train_from_memory(
        self,
        store: Any,  # ExperienceStore
        batch_size: int = 64,
    ) -> tuple[float, list[float]]:
        """Sample from PER store and run one gradient update (sync).

        Args:
            store: ExperienceStore instance
            batch_size: PER batch size (default 64)

        Returns:
            (mean_squared_loss, td_errors) — td_errors used by caller to
            update SumTree priorities.
        """
        batch, is_weights, indices = store.sample(batch_size)
        if not batch:
            return 0.0, []

        loss, td_errors = self.q.train_batch(batch)

        # Update priorities in PER store (sync, SumTree only)
        store.update_priorities(indices, td_errors)

        logger.info(
            "RL train_step | batch=%d loss=%.4f ε=%.4f β=%.3f",
            len(batch), loss, self.epsilon,
            store._beta if hasattr(store, "_beta") else 1.0,
        )
        return loss, td_errors

    async def train_from_db(
        self,
        store: Any,
        batch_size: int = 256,
        scan_id: str | None = None,
    ) -> tuple[float, list[float]]:
        """Pull a larger batch from Postgres for offline training (async).

        Args:
            store: ExperienceStore instance
            batch_size: DB batch size (default 256 — larger than memory batch)
            scan_id: optional filter (only train on transitions from this scan)

        Returns:
            (mean_squared_loss, td_errors)
        """
        batch = await store.sample_from_db(batch_size, scan_id=scan_id)
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