"""W7-B: Add hitl_mode column to vapt_scans

Revision ID: 0004_w7b_hitl_mode
Revises: 0003_vapt_ai_baseline
Create Date: 2026-09-12

Adds the `hitl_mode` column to `vapt_scans` to support per-scan HITL mode
override (Option C — A-in-the-Loop).

Column:
    vapt_scans.hitl_mode VARCHAR(32) NOT NULL DEFAULT 'audit_agent'

Valid values:
    audit_agent (default, MVP) — LLM critic reviews destructive ops, no human blocking
    human_block                — blocking channel waits for human decision (deferred)
    auto_approve (debug only)  — skip review entirely
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "0004_w7b_hitl_mode"
down_revision = "0003_vapt_ai_baseline"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Add hitl_mode column with default 'audit_agent'
    op.add_column(
        "vapt_scans",
        sa.Column(
            "hitl_mode",
            sa.String(length=32),
            nullable=False,
            server_default="audit_agent",
        ),
    )

    # Backfill existing rows (if any) to 'audit_agent'
    op.execute("UPDATE vapt_scans SET hitl_mode = 'audit_agent' WHERE hitl_mode IS NULL")

    # Optional: index for filtering scans by hitl_mode (low cardinality — skip
    # index; queries will be cheap on table scan since most scans use default)


def downgrade() -> None:
    op.drop_column("vapt_scans", "hitl_mode")
