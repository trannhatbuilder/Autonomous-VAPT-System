"""PATCH (worklog task #17): widen VARCHAR columns in vapt_hitl_approvals.

Revision ID: 0009_hitl_widen_varchar
Revises: 0008_scan_fk_ondelete
Create Date: 2026-10-07

Background:
    The vapt_hitl_approvals table had three columns declared as VARCHAR
    with lengths too small for their actual values:

      - predicted_impact  VARCHAR(32)  — values like
        "Metasploit exploit module execution" (35 chars) caused
        asyncpg.exceptions.StringDataRightTruncationError on INSERT.
      - status             VARCHAR(16)  — values like
        "approved_with_time_limit" (24 chars) would also overflow.
      - user_decision      VARCHAR(16)  — values like
        "approve_with_time_limit" (23 chars) would also overflow.

    The user reported this error during a scan that called metasploit
    (a destructive tool that triggers the HITL gate in tool_bridge.py).

    The HITL gate is now disabled by default (VAPT_AI_HITL_DISABLED=1)
    per worklog task #14, so this INSERT should not fire in production
    anymore. But to be defensive (in case HITL is re-enabled later, or
    the env flag is not set), this migration widens the columns so any
    reasonable human-readable value fits.

Strategy:
    ALTER TABLE vapt_hitl_approvals
        ALTER COLUMN predicted_impact TYPE TEXT,
        ALTER COLUMN status TYPE VARCHAR(64),
        ALTER COLUMN user_decision TYPE VARCHAR(64);

    TEXT and VARCHAR(64) are large enough for any reasonable impact /
    status / decision string. The conversion is non-destructive (existing
    rows are preserved, just the column type changes).

    Downgrade restores the original sizes — but will FAIL if any existing
    row has a value longer than the original limit. This is intentional
    (downgrade is destructive in that case).

Revision ID: 0009_hitl_widen_varchar
Revises: 0008_scan_fk_ondelete
Create Date: 2026-10-07
"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "0009_hitl_widen_varchar"
down_revision = "0008_scan_fk_ondelete"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Widen VARCHAR columns in vapt_hitl_approvals to prevent overflow."""
    # predicted_impact: VARCHAR(32) → TEXT
    # (any human-readable impact string fits — no practical length limit)
    op.alter_column(
        table_name="vapt_hitl_approvals",
        column_name="predicted_impact",
        type_=sa.Text(),
        existing_type=sa.String(length=32),
        existing_nullable=False,
    )

    # status: VARCHAR(16) → VARCHAR(64)
    # (accommodates "approved_with_time_limit" = 24 chars + future values)
    op.alter_column(
        table_name="vapt_hitl_approvals",
        column_name="status",
        type_=sa.String(length=64),
        existing_type=sa.String(length=16),
        existing_nullable=False,
    )

    # user_decision: VARCHAR(16) → VARCHAR(64)
    # (accommodates "approve_with_time_limit" = 23 chars + future values)
    op.alter_column(
        table_name="vapt_hitl_approvals",
        column_name="user_decision",
        type_=sa.String(length=64),
        existing_type=sa.String(length=16),
        existing_nullable=True,
    )


def downgrade() -> None:
    """Restore original VARCHAR sizes.

    WARNING: this will FAIL if any existing row has a value longer than
    the original limit. Delete or truncate such rows before downgrading.
    """
    op.alter_column(
        table_name="vapt_hitl_approvals",
        column_name="predicted_impact",
        type_=sa.String(length=32),
        existing_type=sa.Text(),
        existing_nullable=False,
    )
    op.alter_column(
        table_name="vapt_hitl_approvals",
        column_name="status",
        type_=sa.String(length=16),
        existing_type=sa.String(length=64),
        existing_nullable=False,
    )
    op.alter_column(
        table_name="vapt_hitl_approvals",
        column_name="user_decision",
        type_=sa.String(length=16),
        existing_type=sa.String(length=64),
        existing_nullable=True,
    )