"""F1: CyberStrikeAI pattern migration — process_details table + Finding FK change

Revision ID: 0007_csa_pattern
Revises: 0006_w8e_attack_catalog
Create Date: 2026-09-26

This migration aligns VAPT-AI's DB schema with CyberStrikeAI's pattern:

1. NEW TABLE `vapt_process_details` — single source of truth for the agent
   timeline. Every event (tool_call_started, tool_call_completed, iteration,
   thinking, phase_change, scan_started, scan_complete, scan_error, finding_detected,
   hitl_approval_required, hitl_decision_made, cancelled, error, timeout) gets
   persisted here. Mirrors CyberStrikeAI's `process_details` table
   (internal/database/database.go:222).

   Without this table, when the SSE subscriber disconnected (page refresh,
   browser close), all events were lost — scan history UI had no way to
   reconstruct what happened during a past scan.

2. ALTER `vapt_findings`:
   - Change `scan_id` FK from CASCADE → SET NULL (findings SURVIVE scan delete)
   - Make `scan_id` nullable
   - Add `scan_tag` column (snapshot of scan target — so findings remain
     labeled after scan deletion)

   Mirrors CyberStrikeAI's `vulnerabilities` table where `conversation_id`
   has FK SET NULL and `conversation_tag` snapshots the conversation title.

Rollback:
    alembic downgrade -1    # drops vapt_process_details + reverts Finding changes
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB, UUID


# revision identifiers, used by Alembic.
revision = "0007_csa_pattern"
down_revision = "0006_w8e_attack_catalog"
branch_labels = None
depends_on = None


# Canonical name for the FK constraint we (re)create. Used as a fallback when
# no existing FK is found in the catalog.
FINDINGS_SCAN_ID_FK = "fk_vapt_findings_scan_id_vapt_scans"


def _find_fk_constraint_name(table: str, column: str) -> str | None:
    """Return the actual FK constraint name on ``table.column``, or None.

    The baseline migration (``0003_vapt_ai_baseline``) created this FK with an
    explicit name (``fk_vapt_findings_scan_id_vapt_scans``), NOT PostgreSQL's
    default ``<table>_<column>_fkey``. Depending on how a given database was
    provisioned (Alembic baseline vs. SQLAlchemy ``create_all``), either name
    may be in play — so query the catalog instead of hard-coding.
    """
    row = op.get_bind().execute(
        sa.text(
            """
            SELECT con.conname
            FROM pg_constraint con
            JOIN pg_class rel ON rel.oid = con.conrelid
            JOIN pg_attribute att
              ON att.attrelid = con.conrelid
             AND att.attnum = ANY (con.conkey)
            WHERE rel.relname = :table
              AND att.attname = :column
              AND con.contype = 'f'
            ORDER BY con.conname
            LIMIT 1
            """
        ),
        {"table": table, "column": column},
    ).scalar()
    return row


def upgrade() -> None:
    """Apply the CyberStrikeAI pattern migration."""

    # ── Step 1: CREATE vapt_process_details table ─────────────────────
    # Mirrors CyberStrikeAI's `process_details` table — single source of
    # truth for the agent timeline (events persisted to DB, not just SSE).
    op.create_table(
        "vapt_process_details",
        # UUID PK (matches UUIDPrimaryKey mixin)
        sa.Column("id", UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),

        # FK to vapt_chat_messages (PRIMARY timeline linkage)
        # Nullable: some events fire before any message exists (e.g. scan_started)
        sa.Column(
            "message_id",
            UUID(as_uuid=True),
            sa.ForeignKey("vapt_chat_messages.id", ondelete="CASCADE"),
            nullable=True,
        ),

        # Denormalized FK to vapt_conversations (for fast WHERE conv_id=? queries)
        sa.Column(
            "conversation_id",
            UUID(as_uuid=True),
            sa.ForeignKey("vapt_conversations.id", ondelete="CASCADE"),
            nullable=True,
        ),

        # Denormalized FK to vapt_scans (for fast WHERE scan_id=? queries)
        # VAPT-AI uses scan_id everywhere (events.py, tool_bridge.py, etc.)
        sa.Column(
            "scan_id",
            sa.String(64),
            sa.ForeignKey("vapt_scans.id", ondelete="CASCADE"),
            nullable=True,
        ),

        # Event identity (e.g. "tool_call_started", "iteration", "phase_change")
        sa.Column("event_type", sa.String(64), nullable=False),

        # Human-readable summary (shown in timeline list view)
        sa.Column("message", sa.Text(), nullable=True),

        # Full payload (JSONB): tool args, result, agent_name, iteration, etc.
        sa.Column("data", JSONB, nullable=True),

        # Idempotency signature (for dedup; format: "event_type|tool_call_id")
        sa.Column("signature", sa.String(255), nullable=True),

        # Timestamps (matches TimestampMixin)
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
    )

    # Indexes for fast queries
    op.create_index("ix_vapt_process_details_message_id", "vapt_process_details", ["message_id"])
    op.create_index("ix_vapt_process_details_conversation_id", "vapt_process_details", ["conversation_id"])
    op.create_index("ix_vapt_process_details_scan_id", "vapt_process_details", ["scan_id"])
    op.create_index("ix_vapt_process_details_event_type", "vapt_process_details", ["event_type"])
    op.create_index("ix_vapt_process_details_signature", "vapt_process_details", ["signature"])
    op.create_index("ix_vapt_process_details_created_at", "vapt_process_details", ["created_at"])

    # ── Step 2: ALTER vapt_findings — FK CASCADE → SET NULL + add scan_tag ─
    #
    # PostgreSQL ALTER COLUMN for an FK change requires:
    #   1. Drop the existing FK constraint (looked up from the catalog — the
    #      0003 baseline named it `fk_vapt_findings_scan_id_vapt_scans`, not
    #      the PostgreSQL default `vapt_findings_scan_id_fkey`)
    #   2. Drop the existing NOT NULL constraint on scan_id
    #   3. Re-add the FK with ondelete="SET NULL"
    #   4. Add the new scan_tag column

    # 2a. Drop the existing FK constraint on vapt_findings.scan_id (if present)
    fk_name = _find_fk_constraint_name("vapt_findings", "scan_id")
    if fk_name is not None:
        op.drop_constraint(fk_name, "vapt_findings", type_="foreignkey")
    fk_name = fk_name or FINDINGS_SCAN_ID_FK

    # 2b. Make scan_id nullable (was NOT NULL)
    op.alter_column(
        "vapt_findings",
        "scan_id",
        existing_type=sa.String(64),
        nullable=True,
    )

    # 2c. Re-add FK with ON DELETE SET NULL (reusing the original name)
    op.create_foreign_key(
        fk_name,
        "vapt_findings",
        "vapt_scans",
        ["scan_id"],
        ["id"],
        ondelete="SET NULL",
    )

    # 2d. Add scan_tag column (snapshot of scan target for context after scan delete)
    op.add_column(
        "vapt_findings",
        sa.Column("scan_tag", sa.String(255), nullable=True),
    )
    op.create_index("ix_vapt_findings_scan_tag", "vapt_findings", ["scan_tag"])

    # 2e. Backfill scan_tag for existing findings (snapshot current scan target)
    # Uses the scan's target column — best-effort, doesn't fail if scan was deleted.
    op.execute(
        """
        UPDATE vapt_findings f
        SET scan_tag = COALESCE(
            (SELECT s.target FROM vapt_scans s WHERE s.id = f.scan_id),
            '(deleted scan)'
        )
        WHERE f.scan_tag IS NULL
        """
    )


def downgrade() -> None:
    """Revert the migration."""
    # 1. Revert Finding changes
    op.drop_index("ix_vapt_findings_scan_tag", table_name="vapt_findings")
    op.drop_column("vapt_findings", "scan_tag")

    fk_name = _find_fk_constraint_name("vapt_findings", "scan_id")
    if fk_name is not None:
        op.drop_constraint(fk_name, "vapt_findings", type_="foreignkey")
    fk_name = fk_name or FINDINGS_SCAN_ID_FK

    op.alter_column(
        "vapt_findings",
        "scan_id",
        existing_type=sa.String(64),
        nullable=False,
    )
    op.create_foreign_key(
        fk_name,
        "vapt_findings",
        "vapt_scans",
        ["scan_id"],
        ["id"],
    )

    # 2. Drop vapt_process_details
    op.drop_index("ix_vapt_process_details_created_at", table_name="vapt_process_details")
    op.drop_index("ix_vapt_process_details_signature", table_name="vapt_process_details")
    op.drop_index("ix_vapt_process_details_event_type", table_name="vapt_process_details")
    op.drop_index("ix_vapt_process_details_scan_id", table_name="vapt_process_details")
    op.drop_index("ix_vapt_process_details_conversation_id", table_name="vapt_process_details")
    op.drop_index("ix_vapt_process_details_message_id", table_name="vapt_process_details")
    op.drop_table("vapt_process_details")