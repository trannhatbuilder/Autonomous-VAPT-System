"""
VAPT-AI Reinforcement Learning persistence models.

Tables:
    rl_checkpoints   — RL model checkpoint (Dueling Double DQN weights + metadata)
    rl_transitions   — RL experience tuple (state, action, reward, next_state, done)

Ported from EVVO Sentinel's shield_engine/rl/ — but migrated from .npz file
persistence to SQL persistence (with .npz kept as fast-load cache).

RL architecture (from EVVO, post-audit-fix):
    - Dueling Double DQN (Q = V + A - mean(A))
    - State: 337-dim float32 vector
    - Action: 7 discrete (execute_command_recon, execute_command_exploit,
              execute_command_fuzz, select_skill, consult_kg,
              report_vulnerability, pentest_complete)
    - PER (Prioritized Experience Replay) SumTree: buffer 50,000, alpha=0.6,
      beta=0.4→1.0 annealed over 100k steps
    - Soft target update: tau=0.005
    - Huber loss (SmoothL1)
    - He init for ReLU hidden, Xavier for linear output

Reward shaping (D26):
    +5  confirmed exploit Tier-1 (shell obtained)
    +3  confirmed exploit Tier-2 (partial)
    +1  confirmed finding (verifier accepts)
    +0.5 successful recon (asset discovered)
    -1  timeout / no useful output
    -2  verifier-rejected FP
    -3  scope violation (blocked)
    -5  user_aborted (HITL denied)
    -10 destructive op without HITL (should never happen)
    +10 scan completed successfully (terminal bonus)
"""
from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, TimestampMixin, UUIDPrimaryKey


class RLCheckpoint(Base, UUIDPrimaryKey, TimestampMixin):
    """RL model checkpoint — Dueling Double DQN weights + training metadata.

    Saved after every scan (per plan §11). Loaded at startup to resume training.

    Storage strategy:
        - weights_blob: serialized numpy weights (np.save + zlib compress)
        - q_meta_json: training metadata (epsilon, scan_count, total_reward, etc.)
        - .npz cache files at data/rl/ kept for fast load (regenerated from this)
    """

    __tablename__ = "vapt_rl_checkpoints"

    scan_id: Mapped[str | None] = mapped_column(
        String(64),
        ForeignKey("vapt_scans.id"),
        nullable=True,  # nullable — initial checkpoint has no scan
        index=True,
    )

    # Weights — serialized Dueling Double DQN (shared trunk + value stream + advantage stream)
    # Format: zlib.compress(np.save(np_weights_dict))
    weights_blob: Mapped[bytes] = mapped_column(nullable=False)

    # Training metadata (JSONB)
    # {
    #   "epsilon": 0.15,           # current epsilon (decays 1.0 → 0.05 over 200 scans)
    #   "scan_count": 42,          # total scans trained on
    #   "total_reward": 187.5,     # cumulative reward
    #   "state_dim": 337,
    #   "action_dim": 7,
    #   "hidden_dim": 64,
    #   "learning_rate": 0.001,
    #   "gamma": 0.95,
    #   "tau": 0.005,              # soft update coefficient
    #   "huber_loss_delta": 1.0,
    #   "exploration_mode": "epsilon_greedy",  # or "boltzmann" / "curiosity"
    #   "boltzmann_temp": 1.0,     # if boltzmann
    #   "curiosity_enabled": true,
    #   "trained_at_step": 1234,
    # }
    q_meta_json: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)

    # Checkpoint metadata
    is_initial: Mapped[bool] = mapped_column(
        nullable=False, default=False, server_default="false"
    )  # true for seed checkpoint (no scan, just initial weights)

    # Checkpoint hash (SHA-256 of weights_blob — for integrity verification)
    weights_hash: Mapped[str] = mapped_column(String(128), nullable=False)

    # Adaptive encoder config (from EVVO's adaptive_encoder.json)
    adaptive_encoder_config: Mapped[dict | None] = mapped_column(JSONB, nullable=True)


class RLTransition(Base, UUIDPrimaryKey):
    """RL experience tuple — one (s, a, r, s', done) per scan turn.

    Used by Prioritized Experience Replay (PER) SumTree for offline training.
    Retention: 90 days (per §7.4) — weights persist indefinitely in RLCheckpoint.

    NOTE: No TimestampMixin — only created_at (append-only log). No updated_at.
    """

    __tablename__ = "vapt_rl_transitions"

    scan_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("vapt_scans.id"),
        nullable=False,
        index=True,
    )

    # Turn number within scan (0, 1, 2, ...)
    turn: Mapped[int] = mapped_column(Integer, nullable=False, index=True)

    # State — 337-dim float32 vector (JSONB array of floats)
    # Encoded by PentestState.to_vector() in shield_engine/rl/state_encoder.py
    state_vector: Mapped[list] = mapped_column(JSONB, nullable=False)

    # Action taken (0-6 — see ACTION_SPACE in shield_engine/rl/state_encoder.py)
    action: Mapped[int] = mapped_column(Integer, nullable=False)

    # Reward received (per D26 reward shaping)
    reward: Mapped[float] = mapped_column(Float, nullable=False)

    # Next state — 337-dim float32 vector (JSONB array of floats)
    next_state_vector: Mapped[list] = mapped_column(JSONB, nullable=False)

    # Terminal flag (true if scan ended after this turn)
    done: Mapped[bool] = mapped_column(nullable=False, default=False)

    # PER (Prioritized Experience Replay) fields
    # priority: |TD-error| + epsilon — used by SumTree for proportional sampling
    # td_error: temporal difference error from last training step
    # (alpha=0.6 applied when computing SumTree priority)
    priority: Mapped[float] = mapped_column(
        Float, nullable=False, default=1.0, server_default="1.0", index=True
    )
    td_error: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Action metadata (for debugging / replay)
    action_name: Mapped[str | None] = mapped_column(String(64), nullable=True)
    # execute_command_recon / execute_command_exploit / etc.

    # Reward breakdown (JSONB — multi-signal reward per EVVO's reward.py)
    # {
    #   "rule_verified": 1.0,
    #   "human_label": null,
    #   "replay_reproduced": null,
    #   "false_positive_penalty": 0.0,
    #   "explore_bonus": 0.5,
    #   "efficiency_penalty": -0.1,
    #   "total": 1.4
    # }
    reward_breakdown: Mapped[dict | None] = mapped_column(JSONB, nullable=True)

    # Timestamp (append-only — no updated_at)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now(), index=True
    )