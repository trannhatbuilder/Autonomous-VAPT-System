"""add user_token_usage + quota overrides

Daily per-user LLM cost rollup + optional override cap so an admin can
raise/lower limits per user without redeploying.

Revision ID: 0002
Revises: 0001
Create Date: 2026-04-23
"""
from __future__ import annotations

from alembic import op


revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


_UP_SQL = """
CREATE TABLE IF NOT EXISTS user_token_usage (
    user_id    UUID         NOT NULL,
    day        DATE         NOT NULL,
    cost_usd   NUMERIC(14, 6) NOT NULL DEFAULT 0,
    input_tokens   BIGINT   NOT NULL DEFAULT 0,
    output_tokens  BIGINT   NOT NULL DEFAULT 0,
    cache_read_tokens  BIGINT NOT NULL DEFAULT 0,
    cache_write_tokens BIGINT NOT NULL DEFAULT 0,
    calls      INTEGER      NOT NULL DEFAULT 0,
    updated_at TIMESTAMPTZ  NOT NULL DEFAULT now(),
    PRIMARY KEY (user_id, day)
);

CREATE INDEX IF NOT EXISTS ix_user_token_usage_user_id ON user_token_usage (user_id);
CREATE INDEX IF NOT EXISTS ix_user_token_usage_day     ON user_token_usage (day);

CREATE TABLE IF NOT EXISTS user_token_quota_overrides (
    user_id         UUID PRIMARY KEY,
    daily_cap_usd   NUMERIC(12, 4),
    monthly_cap_usd NUMERIC(12, 4),
    reason          TEXT,
    updated_by      UUID,
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
"""


_DOWN_SQL = """
DROP TABLE IF EXISTS user_token_quota_overrides;
DROP TABLE IF EXISTS user_token_usage;
"""


def upgrade() -> None:
    op.execute(_UP_SQL)


def downgrade() -> None:
    op.execute(_DOWN_SQL)
