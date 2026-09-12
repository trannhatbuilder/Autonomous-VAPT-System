"""VAPT-AI v3.2 — Alembic migration environment.

DB URL resolution:
    1. EVVO_POSTGRES_DB  (legacy EVVO env var — kept for backward compat with
                          existing tests and scripts)
    2. VAPT_AI_POSTGRES_DB (canonical VAPT-AI env var)
    3. DATABASE_URL      (fallback for tests/conftest.py)

If none is set, raise a clear error rather than silently falling back to the
alembic.ini placeholder (which would migrate the wrong DB).

Target metadata:
    Base.metadata from app.db.base — so `alembic revision --autogenerate`
    detects all VAPT-AI models defined under app/db/models/.

    NOTE: Legacy EVVO tables (users, scans, agent_states, shield_metrics,
    scan_feedback, llm_models, detection_patterns, engagements, findings,
    attack_graphs, scan_quota_plans, approval_gates, approval_policies,
    handoff_reports, user_token_usage, user_token_quota_overrides,
    compliance_reports, knowledge_entries, agent_sessions, scan_metrics,
    knowledge_sync_log) are NOT defined as SQLAlchemy models — they were
    created via raw SQL in alembic/versions/0001_baseline_snapshot.py and
    migrations/00X_*.sql. Alembic autogenerate will mark them as "removed"
    unless we tell it to ignore them. We do that via include_object() below.
"""
from __future__ import annotations

import os
import sys
from logging.config import fileConfig
from pathlib import Path

# Load .env so alembic can be run from CLI without manually sourcing .env first.
try:
    from dotenv import load_dotenv
    _PROJECT_ROOT = Path(__file__).resolve().parent.parent
    load_dotenv(_PROJECT_ROOT / ".env")
except ImportError:
    pass

# Ensure project root is on sys.path so `app` package is importable
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from alembic import context  # noqa: E402
from sqlalchemy import engine_from_config, pool  # noqa: E402

# Import Base + ALL model modules so metadata is fully populated before autogenerate
from app.db.base import Base  # noqa: E402
# W1-C models (18 tables) + W1-D models (6 tables) = 24 VAPT-AI tables total
from app.db.models import (  # noqa: F401, E402
    # W1-C
    User, RefreshToken,
    Scan, ConsentForm, Asset,
    PentestFact, Finding, Evidence, PoCResult,
    C2Session, C2Task,
    HITLApproval,
    AuditLog, Quarantine,
    AttackChain, RetestRequest,
    WSTGMethodologyCatalog, SkillExecution,
    # W1-D
    RLCheckpoint, RLTransition,
    KGNode, KGEdge, KGScanSnapshot,
    ReplayTrace,
)

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# ---------- DB URL resolution ----------
db_url = (
    os.getenv("EVVO_POSTGRES_DB")
    or os.getenv("VAPT_AI_POSTGRES_DB")
    or os.getenv("DATABASE_URL")
)
if not db_url:
    print(
        "ERROR: None of EVVO_POSTGRES_DB, VAPT_AI_POSTGRES_DB, or DATABASE_URL is set. "
        "Set VAPT_AI_POSTGRES_DB in .env (preferred), or export DATABASE_URL.",
        file=sys.stderr,
    )
    sys.exit(2)

# Alembic runs synchronously — use psycopg2 driver
if db_url.startswith("postgresql+asyncpg://"):
    db_url = db_url.replace("postgresql+asyncpg://", "postgresql+psycopg2://", 1)
elif db_url.startswith("postgresql://"):
    db_url = db_url.replace("postgresql://", "postgresql+psycopg2://", 1)

# Escape % to %% for configparser — passwords with % (URL-encoded @ as %40)
# trigger "invalid interpolation syntax" error because configparser treats %
# as the start of variable interpolation (%(...)s).
db_url_escaped = db_url.replace('%', '%%')

config.set_main_option("sqlalchemy.url", db_url_escaped)

# ---------- Target metadata ----------
target_metadata = Base.metadata


# ---------- Legacy table allowlist ----------
# These tables were created by EVVO's alembic/0001_baseline_snapshot.py and
# migrations/*.sql. We must NOT let autogenerate drop them.
LEGACY_TABLES: set[str] = {
    "users", "scans", "agent_states", "shield_metrics", "scan_feedback",
    "llm_models", "detection_patterns", "engagements", "findings",
    "attack_graphs", "scan_quota_plans", "approval_gates", "approval_policies",
    "handoff_reports", "user_token_usage", "user_token_quota_overrides",
    "compliance_reports", "knowledge_entries", "agent_sessions", "scan_metrics",
    "knowledge_sync_log", "rl_experiences",
    # Views
    "findings_summary", "engagement_phase_progress",
}


def include_object(object, name, type_, reflected, compare_to):
    """Tell Alembic to ignore legacy EVVO tables (don't drop them on autogenerate)."""
    if type_ == "table" and name in LEGACY_TABLES:
        return False
    return True


def run_migrations_offline() -> None:
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        compare_type=True,
        compare_server_default=True,
        include_object=include_object,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
            compare_server_default=True,
            include_object=include_object,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()