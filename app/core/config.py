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