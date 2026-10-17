"""
VAPT-AI Reinforcement Learning module.

Architecture: Dueling Double DQN (numpy-only, no torch / tensorflow).

Components
----------
- q_learner.py        DoubleQLearner — dueling arch + Double Q + Huber loss
- state_encoder.py    StateEncoder   — 337-dim state vector from session
- experience_store.py ExperienceStore — PER SumTree + SQL persistence
- reward.py           compute_reward  — multi-signal reward shaping
- policy.py           PentestPolicy   — 3 exploration modes
                       (epsilon_greedy / boltzmann / curiosity)
- persistence.py      save_checkpoint / load_latest_checkpoint
                       (SQL + .npz cache)
- training_loop.py    train_one_step / train_from_scan / train_multi_step
- harness.py          HarnessBridge — per-scan bridge connecting the
                       ReAct agent loop to the policy, experience store,
                       and training worker (Phase 1)
- _session_builder.py build_session — assembles the in-memory session
                       dict consumed by StateEncoder

Action space (7 discrete)
--------------------------
    execute_command_recon      — nmap / httpx / whatweb / subfinder / ...
    execute_command_exploit    — sqlmap / metasploit / hydra
    execute_command_fuzz       — ffuf / gobuster / feroxbuster / katana /
                                 nikto / nuclei / dalfox / wpscan
    select_skill               — load an additional skill playbook
    consult_kg                 — query the knowledge graph (Phase 2+)
    report_vulnerability       — record a confirmed finding
    pentest_complete           — wrap up + exit

State dim: 337 (= 8 scalars + 256 tech multi-hot + 32 last-vector one-hot
            + 32 last-tool one-hot + 9 flags)

Persistence
-----------
- SQL tables (vapt_rl_checkpoints, vapt_rl_transitions) — durable
- .npz cache at data/rl/q_weights.npz — fast-load

Reward shaping (multi-signal, D26)
----------------------------------
    +5    confirmed exploit Tier-1 (shell obtained via C2Session)
    +3    confirmed exploit Tier-2 (partial PoC)
    +1    confirmed finding (verifier accepts)
    +0.5  successful recon (asset discovered)
    -1    timeout / no useful output
    -2    verifier-rejected false positive
    -3    scope violation (blocked)
    -5    user_aborted (HITL denied)
    -10   destructive op without HITL (failsafe — should never fire)
    +10   scan completed successfully (terminal bonus)

Epsilon schedule
----------------
    epsilon_start = 1.0  ->  epsilon_end = 0.05  over 200 scans
    (adaptive: high-reward episodes decay faster, low-reward slower)

PER parameters
--------------
    buffer = 50,000
    alpha  = 0.6   (priority exponent)
    beta   = 0.4 -> 1.0  annealed over 100,000 steps
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
    VECTOR_LABELS,
    VECTOR_VOCAB,
    TOOL_LABELS,
    TOOL_VOCAB,
    STATE_LAYOUT_VERSION,
    state_layout_id,
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
    D26_EVENT_REWARDS,
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
    USEFUL_OUTPUT_MIN_CHARS,
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

# Per-scan Harness Bridge (Phase 1) — connects the ReAct agent loop
# to the policy / experience store / training worker.
from app.rl.harness import (
    HarnessBridge,
    TurnRecommendation,
    get_harness,
    release_harness,
    is_rl_harness_enabled,
    TRAIN_INTERVAL_TURNS,
    TRAIN_BATCH_SIZE,
    MIN_BUFFER_FOR_TRAIN,
    # Phase 2: KG integration
    KG_ENABLED_DEFAULT,
    KG_TOP_K_DEFAULT,
    # Phase 3: TDA integration
    TDA_ENABLED_DEFAULT,
    TDA_FORCE_EXPLOIT_ON_FINDING,
    TDA_MAX_HORIZON_DEFAULT,
    MODEL_CONTEXT_WINDOW_DEFAULT,
    # Phase 4: EGATS integration
    EGATS_ENABLED_DEFAULT,
    # Phase 5: Self-Correction integration
    SC_ENABLED_DEFAULT,
    # Phase 6: Per-scan KG Fork
    KG_ISOLATION_DEFAULT,
)

# Session-dict builder (Phase 1) — packs agent-loop locals into the
# shape StateEncoder.from_session() expects.
from app.rl._session_builder import build_session

# Task Difficulty Assessment (Phase 3) — TDI computation + mode hint rendering.
from app.rl.tda import (
    Mode as TDAMode,
    TDIResult,
    compute_tdi,
    horizon_normalised,
    resolve_max_horizon,
    evidence_confidence,
    context_load,
    context_saturation,
    historical_success,
    evidence_score_from_label,
    evidence_score_from_finding,
    max_evidence_score,
    should_force_exploit,
    render_tda_hint,
    VERIFIED_EXPLOIT,
    CONFIRMED_VULN,
    PLAUSIBLE_HYPOTHESIS,
    SPECULATIVE_HYPOTHESIS,
    TDI_BROADEN,
)

# Evidence-Guided Attack Tree Search (Phase 4) — UCB1 bandit over KG paths.
from app.rl.egats import (
    PathStat,
    EGATSResult,
    EGATS,
    render_egats_hint,
    DEFAULT_UCB_C,
    DEFAULT_LAMBDA,
    DEFAULT_K_MIN_PRUNE,
    DEFAULT_EMA_ALPHA,
    DEFAULT_MU_SPECIFICITY,
)

# Self-Correction on verifier rejection (Phase 5) — pattern tracking +
# 3-tier corrective feedback + escalating RL reward.
from app.rl.self_correction import (
    SelfCorrectionEngine,
    RejectionAnalysis,
    process_verification_rejection,
    get_self_correction_engine,
    normalize_endpoint,
    compute_pattern_signature,
    DEFAULT_BLACKLIST_THRESHOLD,
    DEFAULT_WARNING_THRESHOLD,
    REWARD_TIER_1,
    REWARD_TIER_2,
    REWARD_TIER_3,
    PER_PRIORITY_MULTIPLIER,
)

__all__ = [
    # Core
    "DoubleQLearner", "QLearner", "TwoLayerMLP", "DuelingNetwork",
    "STATE_DIM", "ACTION_DIM", "DEFAULT_LEARNING_RATE", "DEFAULT_GAMMA",
    "DEFAULT_TAU", "DEFAULT_HIDDEN_DIM", "HUBER_KAPPA",

    # State encoder
    "StateEncoder", "PentestState", "StateEncodingConfig", "DEFAULT_STATE_CONFIG",
    "ACTION_SPACE", "TECH_STACK_VOCAB", "VECTOR_LABELS", "VECTOR_VOCAB", "TOOL_VOCAB",
    "TOOL_LABELS", "STATE_LAYOUT_VERSION", "state_layout_id",
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
    "USEFUL_OUTPUT_MIN_CHARS",
    "D26_EVENT_REWARDS",
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

    # Harness Bridge (Phase 1 + 2 + 3 + 4 + 5 + 6)
    "HarnessBridge", "TurnRecommendation", "get_harness", "release_harness",
    "is_rl_harness_enabled",
    "TRAIN_INTERVAL_TURNS", "TRAIN_BATCH_SIZE", "MIN_BUFFER_FOR_TRAIN",
    "KG_ENABLED_DEFAULT", "KG_TOP_K_DEFAULT",
    "TDA_ENABLED_DEFAULT", "TDA_FORCE_EXPLOIT_ON_FINDING", "TDA_MAX_HORIZON_DEFAULT",
    "MODEL_CONTEXT_WINDOW_DEFAULT",
    "EGATS_ENABLED_DEFAULT",
    "SC_ENABLED_DEFAULT",
    "KG_ISOLATION_DEFAULT",

    # Session builder (Phase 1)
    "build_session",

    # TDA (Phase 3)
    "TDAMode", "TDIResult", "compute_tdi", "render_tda_hint",
    "horizon_normalised", "resolve_max_horizon", "evidence_confidence",
    "context_load", "context_saturation", "historical_success",
    "evidence_score_from_label", "evidence_score_from_finding",
    "max_evidence_score", "should_force_exploit",
    "VERIFIED_EXPLOIT", "CONFIRMED_VULN", "PLAUSIBLE_HYPOTHESIS",
    "SPECULATIVE_HYPOTHESIS",
    "TDI_BROADEN",

    # EGATS (Phase 4)
    "PathStat", "EGATSResult", "EGATS", "render_egats_hint",
    "DEFAULT_UCB_C", "DEFAULT_LAMBDA", "DEFAULT_K_MIN_PRUNE",
    "DEFAULT_EMA_ALPHA", "DEFAULT_MU_SPECIFICITY",
    "EGATS_ENABLED_DEFAULT",

    # Self-Correction (Phase 5)
    "SelfCorrectionEngine", "RejectionAnalysis",
    "process_verification_rejection", "get_self_correction_engine",
    "normalize_endpoint", "compute_pattern_signature",
    "DEFAULT_BLACKLIST_THRESHOLD", "DEFAULT_WARNING_THRESHOLD",
    "REWARD_TIER_1", "REWARD_TIER_2", "REWARD_TIER_3",
    "PER_PRIORITY_MULTIPLIER",
    "SC_ENABLED_DEFAULT",
]