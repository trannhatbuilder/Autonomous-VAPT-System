"""
VAPT-AI Reinforcement Learning module.

Ported from EVVO Sentinel shield_engine/rl/ (post-audit-fix quality).
Architecture: Dueling Double DQN (numpy-only, no torch/tensorflow).

Components:
  - q_learner.py        DoubleQLearner — dueling arch + Double Q + Huber loss
  - state_encoder.py    StateEncoder   — 337-dim state vector from session
  - experience_store.py ExperienceStore — PER SumTree + SQL persistence
  - reward.py           compute_reward  — D26 multi-signal reward shaping
  - policy.py           PentestPolicy   — 3 exploration modes (ε-greedy/Boltzmann/curiosity)
  - persistence.py      save_checkpoint / load_latest_checkpoint (SQL + .npz cache)
  - training_loop.py    train_one_step / train_from_scan / train_multi_step

Action space (7 discrete):
    execute_command_recon, execute_command_exploit, execute_command_fuzz,
    select_skill, consult_kg, report_vulnerability, pentest_complete

State dim: 337 (= 8 scalars + 256 tech multi-hot + 32 last-vector one-hot
              + 32 last-tool one-hot + 9 flags)

Persistence:
    - SQL tables (vapt_rl_checkpoints, vapt_rl_transitions) — durable
    - .npz cache at data/rl/q_weights.npz — fast-load

Reward shaping (D26 — master plan §12 W16):
    +5   confirmed exploit Tier-1 (shell obtained via C2Session)
    +3   confirmed exploit Tier-2 (partial PoC)
    +1   confirmed finding (verifier accepts)
    +0.5 successful recon (asset discovered)
    -1   timeout / no useful output
    -2   verifier-rejected FP
    -3   scope violation (blocked)
    -5   user_aborted (HITL denied)
    -10  destructive op without HITL (should never happen — failsafe)
    +10  scan completed successfully (terminal bonus)

Epsilon schedule (master plan §12 W16):
    ε_start = 1.0  →  ε_end = 0.05  over 200 scans (decay ≈ 0.98566 per scan)

PER parameters (master plan §12 W16):
    buffer = 50,000
    alpha  = 0.6  (priority exponent)
    beta   = 0.4 → 1.0  annealed over 100,000 steps
"""
from __future__ import annotations

# Core Q-learning
from app.rl.q_learner import (
    DoubleQLearner,
    QLearner,  # backward-compat alias
    TwoLayerMLP,
    DuelingNetwork,
    STATE_DIM,
    ACTION_DIM,
    DEFAULT_LEARNING_RATE,
    DEFAULT_GAMMA,
    DEFAULT_TAU,
    DEFAULT_HIDDEN_DIM,
    HUBER_KAPPA,
)

# State encoding
from app.rl.state_encoder import (
    StateEncoder,
    PentestState,
    StateEncodingConfig,
    DEFAULT_STATE_CONFIG,
    ACTION_SPACE,
    ACTION_DIM as ACTION_DIM_FROM_ENCODER,  # alias (same value)
    TECH_STACK_VOCAB,
    VECTOR_VOCAB,
    TOOL_VOCAB,
    encode_session,
)

# Experience replay
from app.rl.experience_store import (
    ExperienceStore,
    SumTree,
    MAX_MEMORY_BUFFER,
    PER_ALPHA,
    PER_EPSILON,
    PER_BETA_START,
    PER_BETA_END,
    PER_BETA_ANNEAL_STEPS,
)

# Reward shaping
from app.rl.reward import (
    compute_reward,
    d26_event,
    R_EXPLOIT_TIER1,
    R_EXPLOIT_TIER2,
    R_VERIFIED_FINDING,
    R_RECON_SUCCESS,
    R_TIMEOUT,
    R_FP_REJECTED,
    R_SCOPE_VIOLATION,
    R_USER_ABORTED,
    R_DESTRUCTIVE_NO_HITL,
    R_SCAN_COMPLETE,
    R_NOVEL_FINDING,
    R_REDUNDANT_FINDING,
    R_WASTED_TURN,
    R_EFFICIENT,
    R_TOKEN_PENALTY_PER_1K,
    R_FOLLOWED_VERIFIED,
    R_FOLLOWED_NO_FINDING,
    WEIGHT_RULE_VERIFIER,
    WEIGHT_HUMAN_LABEL,
    WEIGHT_REPLAY_REPRODUCED,
    R_MIN_CLIP,
    R_MAX_CLIP,
)

# Policy + exploration
from app.rl.policy import (
    PentestPolicy,
    CuriosityModule,
    DEFAULT_EPSILON_START,
    DEFAULT_EPSILON_END,
    DEFAULT_EPSILON_DECAY,
    DEFAULT_EPSILON_DECAY_SCANS,
    DEFAULT_TEMPERATURE_START,
    DEFAULT_TEMPERATURE_END,
    DEFAULT_TEMPERATURE_DECAY,
)

# Persistence (SQL checkpoint + .npz cache)
from app.rl.persistence import (
    save_checkpoint,
    load_latest_checkpoint,
    list_checkpoints,
)

# Training loop orchestration
from app.rl.training_loop import (
    train_one_step,
    train_multi_step,
    train_from_scan,
    train_from_recent_scans,
    get_training_stats,
)

__all__ = [
    # Core
    "DoubleQLearner", "QLearner", "TwoLayerMLP", "DuelingNetwork",
    "STATE_DIM", "ACTION_DIM", "DEFAULT_LEARNING_RATE", "DEFAULT_GAMMA",
    "DEFAULT_TAU", "DEFAULT_HIDDEN_DIM", "HUBER_KAPPA",

    # State encoder
    "StateEncoder", "PentestState", "StateEncodingConfig", "DEFAULT_STATE_CONFIG",
    "ACTION_SPACE", "TECH_STACK_VOCAB", "VECTOR_VOCAB", "TOOL_VOCAB",
    "encode_session",

    # Experience store
    "ExperienceStore", "SumTree", "MAX_MEMORY_BUFFER",
    "PER_ALPHA", "PER_EPSILON", "PER_BETA_START", "PER_BETA_END",
    "PER_BETA_ANNEAL_STEPS",

    # Reward
    "compute_reward", "d26_event",
    "R_EXPLOIT_TIER1", "R_EXPLOIT_TIER2", "R_VERIFIED_FINDING",
    "R_RECON_SUCCESS", "R_TIMEOUT", "R_FP_REJECTED", "R_SCOPE_VIOLATION",
    "R_USER_ABORTED", "R_DESTRUCTIVE_NO_HITL", "R_SCAN_COMPLETE",
    "R_NOVEL_FINDING", "R_REDUNDANT_FINDING", "R_WASTED_TURN",
    "R_EFFICIENT", "R_TOKEN_PENALTY_PER_1K",
    "R_FOLLOWED_VERIFIED", "R_FOLLOWED_NO_FINDING",
    "WEIGHT_RULE_VERIFIER", "WEIGHT_HUMAN_LABEL", "WEIGHT_REPLAY_REPRODUCED",
    "R_MIN_CLIP", "R_MAX_CLIP",

    # Policy
    "PentestPolicy", "CuriosityModule",
    "DEFAULT_EPSILON_START", "DEFAULT_EPSILON_END", "DEFAULT_EPSILON_DECAY",
    "DEFAULT_EPSILON_DECAY_SCANS",
    "DEFAULT_TEMPERATURE_START", "DEFAULT_TEMPERATURE_END",
    "DEFAULT_TEMPERATURE_DECAY",

    # Persistence
    "save_checkpoint", "load_latest_checkpoint", "list_checkpoints",

    # Training loop
    "train_one_step", "train_multi_step", "train_from_scan",
    "train_from_recent_scans", "get_training_stats",
]
