"""
VAPT-AI configuration — single source of truth for runtime settings.

Loads from (in priority order):
    1. Process environment variables (highest priority — for systemd, Docker, CI)
    2. `.env` file in project root (via python-dotenv)
    3. `config.yaml` in project root (lowest priority — for non-secret defaults)

Env var naming convention:
    VAPT_AI_*       — VAPT-AI-specific settings (canonical)
    EVVO_*          — Legacy EVVO env vars (kept as aliases for backward compat
                      with existing 482 tests; will be migrated in later weeks)

All secrets MUST come from environment variables or .env — NEVER from
config.yaml (which is committed to git).
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Load .env early so BaseSettings can see the values
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(_PROJECT_ROOT / ".env")


class Settings(BaseSettings):
    """VAPT-AI runtime settings.

    All fields have sensible defaults for local development. Production
    deployments MUST override via environment variables or .env.
    """

    model_config = SettingsConfigDict(
        env_file=str(_PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",  # tolerate unknown env vars (legacy EVVO_*)
    )

    # ---------- App ----------
    app_name: str = "VAPT-AI"
    app_version: str = "3.2.1"
    environment: Literal["dev", "staging", "prod"] = "dev"
    debug: bool = False
    log_level: str = "INFO"

    # ---------- Database (PostgreSQL) ----------
    # Canonical VAPT-AI env var
    postgres_dsn: str = Field(
        default="postgresql://vapt:vapt@localhost:5432/vapt_ai",
        description="PostgreSQL DSN. Use postgresql+asyncpg:// for async runtime, "
                    "postgresql+psycopg2:// for sync (alembic).",
        validation_alias="VAPT_AI_POSTGRES_DB",
    )
    # Legacy EVVO alias (backward compat)
    evvo_postgres_db: str | None = Field(
        default=None,
        description="Legacy EVVO DB URL alias. If set, overrides postgres_dsn.",
        validation_alias="EVVO_POSTGRES_DB",
    )
    db_pool_min_conn: int = Field(default=2, validation_alias="VAPT_AI_DB_POOL_MIN_CONN")
    db_pool_max_conn: int = Field(default=20, validation_alias="VAPT_AI_DB_POOL_MAX_CONN")
    db_pool_timeout: float = Field(default=30.0, description="Seconds to wait for a connection from the pool")

    # ---------- Redis ----------
    redis_url: str = Field(
        default="redis://localhost:6379/0",
        validation_alias="VAPT_AI_REDIS_URL",
    )
    evvo_redis_url: str | None = Field(default=None, validation_alias="EVVO_REDIS_URL")

    # ---------- Auth ----------
    admin_email: str = Field(default="admin@vapt-ai.local", validation_alias="VAPT_AI_ADMIN_EMAIL")
    admin_password: SecretStr = Field(
        default=SecretStr("change-me-in-prod"),
        validation_alias="VAPT_AI_ADMIN_PASSWORD",
    )
    encryption_key: SecretStr = Field(
        default=SecretStr("change-me-in-prod-32-bytes-base64-aaaa"),
        validation_alias="VAPT_AI_ENCRYPTION_KEY",
    )
    jwt_secret: SecretStr = Field(
        default=SecretStr("change-me-in-prod-jwt-secret-32-bytes"),
        validation_alias="VAPT_AI_JWT_SECRET",
    )
    jwt_algorithm: str = "HS256"
    jwt_access_token_ttl_minutes: int = 30
    jwt_refresh_token_ttl_days: int = 7
    cookie_secure: bool = Field(default=True, validation_alias="VAPT_AI_COOKIE_SECURE")
    auth_max_failed_attempts: int = 5
    auth_lockout_minutes: int = 15

    # ---------- App URLs ----------
    app_url: str = Field(default="http://localhost:8000", validation_alias="VAPT_AI_APP_URL")
    api_url: str = Field(default="http://localhost:8000/api", validation_alias="VAPT_AI_API_URL")

    # ---------- LLM Provider (default) ----------
    model_provider: str = Field(default="openai", validation_alias="VAPT_AI_MODEL_PROVIDER")
    model_enabled: bool = Field(default=True, validation_alias="VAPT_AI_MODEL_ENABLED")
    model_timeout_seconds: int = Field(default=120, validation_alias="VAPT_AI_MODEL_TIMEOUT_SECONDS")

    # ---------- MCP ----------
    mcp_enabled: bool = Field(default=True, validation_alias="VAPT_AI_MCP_ENABLED")
    mcp_metasploit_enabled: bool = Field(default=True, validation_alias="VAPT_AI_MCP_METASPLOIT_ENABLED")
    mcp_scope_enforce: bool = Field(default=True, validation_alias="VAPT_AI_MCP_SCOPE_ENFORCE")

    # ---------- HITL (W7-B — Option C: A-in-the-Loop) ----------
    # Default mode for new scans. Per-scan override via vapt_scans.hitl_mode column.
    #   audit_agent (default, MVP) — LLM critic reviews destructive ops, no human blocking
    #   human_block                — blocking channel waits for human decision (deferred)
    #   auto_approve (debug only)  — skip review entirely
    hitl_mode: str = Field(default="audit_agent", validation_alias="VAPT_AI_HITL_MODE")

    # ---------- Metasploit RPC ----------
    msf_rpc_host: str = Field(default="127.0.0.1", validation_alias="MSF_RPC_HOST")
    msf_rpc_port: int = Field(default=55553, validation_alias="MSF_RPC_PORT")
    msf_rpc_user: str = Field(default="msfuser", validation_alias="MSF_RPC_USER")
    msf_rpc_password: SecretStr = Field(
        default=SecretStr("change-me"),
        validation_alias="MSF_RPC_PASSWORD",
    )
    msf_rpc_ssl: bool = Field(default=False, validation_alias="MSF_RPC_SSL")
    msf_rpc_timeout: int = Field(default=60, validation_alias="MSF_RPC_TIMEOUT")

    # ---------- RL (W16 — Dueling Double DQN) ----------
    # Master plan §12 W16: ε 1.0 → 0.05 over 200 scans; PER α=0.6 β=0.4→1.0/100k.
    rl_enabled: bool = Field(
        default=True,
        validation_alias="VAPT_AI_RL_ENABLED",
        description="Master switch for RL policy. When False, agent uses rule-based fallback.",
    )
    rl_state_dim: int = Field(
        default=337,
        validation_alias="VAPT_AI_RL_STATE_DIM",
        description="State vector dimension. Must match StateEncoder. Default 337 "
                    "(8 scalars + 256 tech multi-hot + 32 last-vector + 32 last-tool + 9 flags).",
    )
    rl_action_dim: int = Field(
        default=7,
        validation_alias="VAPT_AI_RL_ACTION_DIM",
        description="Action space size. Must match ACTION_SPACE. Default 7.",
    )
    rl_hidden_dim: int = Field(
        default=64,
        validation_alias="VAPT_AI_RL_HIDDEN_DIM",
        description="Hidden layer size for value + advantage MLPs. Default 64.",
    )
    rl_learning_rate: float = Field(
        default=0.001,
        validation_alias="VAPT_AI_RL_LEARNING_RATE",
        description="SGD learning rate for Q-network updates.",
    )
    rl_gamma: float = Field(
        default=0.95,
        validation_alias="VAPT_AI_RL_GAMMA",
        description="Discount factor γ for future rewards.",
    )
    rl_tau: float = Field(
        default=0.005,
        validation_alias="VAPT_AI_RL_TAU",
        description="Soft target update coefficient (Polyak averaging).",
    )
    rl_epsilon_start: float = Field(
        default=1.0,
        validation_alias="VAPT_AI_RL_EPSILON_START",
        description="Initial exploration rate. Default 1.0 (fully random).",
    )
    rl_epsilon_end: float = Field(
        default=0.05,
        validation_alias="VAPT_AI_RL_EPSILON_END",
        description="Final exploration rate. Default 0.05 (95% greedy).",
    )
    rl_epsilon_decay_scans: int = Field(
        default=200,
        validation_alias="VAPT_AI_RL_EPSILON_DECAY_SCANS",
        description="Number of scans to decay ε from start → end. Default 200.",
    )
    rl_exploration_mode: str = Field(
        default="epsilon_greedy",
        validation_alias="VAPT_AI_RL_EXPLORATION_MODE",
        description="Exploration strategy: 'epsilon_greedy' (default), 'boltzmann', or 'curiosity'.",
    )
    rl_boltzmann_temp_start: float = Field(
        default=1.0,
        validation_alias="VAPT_AI_RL_BOLTZMANN_TEMP_START",
        description="Initial Boltzmann temperature (only used if exploration_mode='boltzmann').",
    )
    rl_per_alpha: float = Field(
        default=0.6,
        validation_alias="VAPT_AI_RL_PER_ALPHA",
        description="PER priority exponent. 0=uniform, 1=full prioritization. Default 0.6.",
    )
    rl_per_beta_start: float = Field(
        default=0.4,
        validation_alias="VAPT_AI_RL_PER_BETA_START",
        description="PER importance-sampling exponent start. Default 0.4.",
    )
    rl_buffer_size: int = Field(
        default=50_000,
        validation_alias="VAPT_AI_RL_BUFFER_SIZE",
        description="In-memory PER SumTree capacity. Default 50,000 transitions.",
    )
    rl_batch_size: int = Field(
        default=64,
        validation_alias="VAPT_AI_RL_BATCH_SIZE",
        description="PER batch size for training. Default 64.",
    )

    # ---------- KG (W17 — Dynamic Knowledge Graph) ----------
    # Master plan §12 W17: 6 node types, 5 edge types, EGATS UCB1 bandit.
    kg_enabled: bool = Field(
        default=True,
        validation_alias="VAPT_AI_KG_ENABLED",
        description="Master switch for Knowledge Graph. When False, agent has no KG memory.",
    )
    kg_max_depth: int = Field(
        default=4,
        validation_alias="VAPT_AI_KG_MAX_DEPTH",
        description="Max path length for get_attack_paths BFS. Default 4.",
    )
    kg_top_k: int = Field(
        default=10,
        validation_alias="VAPT_AI_KG_TOP_K",
        description="Max paths returned by get_attack_paths. Default 10.",
    )
    kg_ucb_c: float = Field(
        default=1.4142135623730951,  # sqrt(2)
        validation_alias="VAPT_AI_KG_UCB_C",
        description="EGATS UCB exploration constant. Default sqrt(2) ≈ 1.4142.",
    )
    kg_lambda_penalty: float = Field(
        default=0.5,
        validation_alias="VAPT_AI_KG_LAMBDA_PENALTY",
        description="EGATS TDI penalty weight. Higher = more penalty for hard paths. Default 0.5.",
    )
    kg_k_min_prune: int = Field(
        default=3,
        validation_alias="VAPT_AI_KG_K_MIN_PRUNE",
        description="EGATS min attempts before pruning. Paths with TDI > 0.6 after this many "
                    "attempts are pruned. Default 3.",
    )
    kg_ema_alpha: float = Field(
        default=0.3,
        validation_alias="VAPT_AI_KG_EMA_ALPHA",
        description="EGATS EMA smoothing for TDI. 0..1, higher = faster adaptation. Default 0.3.",
    )
    kg_mu_specificity: float = Field(
        default=0.2,
        validation_alias="VAPT_AI_KG_MU_SPECIFICITY",
        description="EGATS inventory specificity boost. Small nudge for paths matching target "
                    "inventory. Default 0.2.",
    )

    # ---------- NVD / CVE (W8-B) ----------
    nvd_enabled: bool = Field(default=True, validation_alias="VAPT_AI_NVD_ENABLED")
    nvd_api_key: SecretStr | None = Field(
        default=None,
        validation_alias="VAPT_AI_NVD_API_KEY",
        description="Optional NVD API key (raises rate limit from 5 to 50 req/30s). "
                    "Request one at https://nvd.nist.gov/developers/request-an-api-key",
    )
    nvd_cache_ttl_hours: int = Field(
        default=24,
        validation_alias="VAPT_AI_NVD_CACHE_TTL_HOURS",
        description="Cache TTL in hours. NVD updates daily, so 24h is reasonable.",
    )
    nvd_timeout_seconds: int = Field(
        default=15,
        validation_alias="VAPT_AI_NVD_TIMEOUT_SECONDS",
        description="Per-request HTTP timeout (NVD can be slow).",
    )
    
    # ---------- Paths ----------
    @property
    def project_root(self) -> Path:
        return _PROJECT_ROOT

    @property
    def data_dir(self) -> Path:
        return _PROJECT_ROOT / "data"

    @property
    def traces_dir(self) -> Path:
        return self.data_dir / "traces"

    @property
    def rl_dir(self) -> Path:
        return self.data_dir / "rl"

    @property
    def kg_dir(self) -> Path:
        return self.data_dir / "vapt_kg"

    @property
    def reports_dir(self) -> Path:
        return self.data_dir / "reports"

    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / "uploads"

    # ---------- Derived ----------
    @property
    def effective_postgres_dsn(self) -> str:
        """Return the effective PostgreSQL DSN, preferring EVVO_ legacy alias if set."""
        if self.evvo_postgres_db:
            return self.evvo_postgres_db
        return self.postgres_dsn

    @property
    def async_postgres_dsn(self) -> str:
        """DSN for asyncpg driver (SQLAlchemy 2.0 async)."""
        dsn = self.effective_postgres_dsn
        # Normalize: postgresql:// → postgresql+asyncpg://
        if dsn.startswith("postgresql://"):
            return dsn.replace("postgresql://", "postgresql+asyncpg://", 1)
        if dsn.startswith("postgresql+psycopg2://"):
            return dsn.replace("postgresql+psycopg2://", "postgresql+asyncpg://", 1)
        return dsn  # already has +asyncpg or is unknown scheme

    @property
    def sync_postgres_dsn(self) -> str:
        """DSN for psycopg2 driver (sync — Alembic, scripts)."""
        dsn = self.effective_postgres_dsn
        if dsn.startswith("postgresql://"):
            return dsn.replace("postgresql://", "postgresql+psycopg2://", 1)
        if dsn.startswith("postgresql+asyncpg://"):
            return dsn.replace("postgresql+asyncpg://", "postgresql+psycopg2://", 1)
        return dsn

    @property
    def effective_redis_url(self) -> str:
        return self.evvo_redis_url or self.redis_url

    # ---------- Validators ----------
    @field_validator("environment")
    @classmethod
    def _validate_env(cls, v: str) -> str:
        if v not in ("dev", "staging", "prod"):
            raise ValueError(f"environment must be dev/staging/prod, got {v!r}")
        return v

    @field_validator("log_level")
    @classmethod
    def _validate_log_level(cls, v: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        v_upper = v.upper()
        if v_upper not in allowed:
            raise ValueError(f"log_level must be one of {allowed}, got {v!r}")
        return v_upper


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached settings singleton.

    Use this everywhere instead of constructing Settings() directly —
    avoids re-parsing .env on every call.
    """
    return Settings()


# Convenience module-level instance (lazy via @lru_cache)
settings = get_settings()