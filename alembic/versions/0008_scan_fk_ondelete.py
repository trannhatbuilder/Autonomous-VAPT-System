"""PATCH (history-sync fix): add ondelete=SET NULL/CASCADE to all vapt_scans FKs.

Revision ID: 0008_scan_fk_ondelete
Revises: 0007_csa_pattern
Create Date: 2026-09-28

Background:
    Migration 0007 only added ondelete clauses to 2 tables:
      - vapt_findings.scan_id       → SET NULL
      - vapt_process_details.scan_id → CASCADE

    All OTHER tables that FK to vapt_scans were left at PostgreSQL's default
    NO ACTION, meaning PostgreSQL BLOCKS any DELETE on vapt_scans when those
    child rows exist. This caused DELETE /api/scans/{id} to fail with
    ForeignKeyViolation (500 Internal Server Error) on any scan that had:
      - pentest_facts (every scan with blackboard)
      - assets (every recon scan)
      - consent_forms (every scan)
      - hitl_approvals (every scan with destructive ops)
      - audit_logs (every scan)
      - c2_sessions (post-exploit scans)
      - replay_traces (every scan)
      - attack_chain_facts (scans with attack graph)
      - kg_facts (scans with KG)
      - methodology_progress (every scan)
      - rl_episodes (every scan — RL records every episode)
      - rl_q_snapshots (RL training)

    The ORM models in app/db/models/ declare ondelete in some places but
    the actual DB schema was never altered to match — model/DB drift.

Strategy:
    For each child table, ALTER the FK constraint to add the appropriate
    ondelete behavior:
      - vapt_findings          → SET NULL (preserve findings — already done in 0007)
      - vapt_process_details   → CASCADE  (already done in 0007)
      - vapt_chat_messages     → SET NULL (preserve chat history)
      - All other child tables → CASCADE  (scan-scoped artifacts, no value
                                          after scan deletion)

    The route handler app/routes/scans.py:delete_scan also now does explicit
    child-row cleanup BEFORE the DELETE — so even if this migration hasn't
    been applied on a given DB, delete_scan will still work. This migration
    is the long-term fix to make the schema actually enforce the intended
    cascade semantics.

Rollback:
    alembic downgrade -1    # removes the ondelete clauses (reverts to NO ACTION)

NOTE: This migration is IDEMPOTENT — running it twice is a no-op because
      we drop the FK first if it exists, then re-create it with ondelete.
      PostgreSQL does not support ALTER CONSTRAINT directly — we must
      DROP + CREATE CONSTRAINT.
"""
from __future__ import annotations

import logging
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "0008_scan_fk_ondelete"
down_revision = "0007_csa_pattern"
branch_labels = None
depends_on = None


logger = logging.getLogger("alembic.0008_scan_fk_ondelete")


def _find_fk_constraint_name(bind, table: str, column: str) -> str | None:
    """Return the actual FK constraint name on ``table.column``, or None.

    Mirrors the helper in migration 0007 — queries the pg_catalog instead
    of guessing the name (which differs between Alembic baseline naming
    convention `fk_<table>_<col>_<ref>` vs PostgreSQL's default
    `<table>_<col>_fkey`).
    """
    row = bind.execute(
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
            """
        ),
        {"table": table, "column": column},
    ).fetchone()
    return row[0] if row else None


def _alter_fk_ondelete(
    bind,
    table: str,
    column: str,
    ref_table: str,
    ref_column: str,
    ondelete: str,
) -> bool:
    """Drop + re-create an FK constraint with the given ondelete behavior.

    Returns True if the FK was altered, False if the FK was not found
    (table may not exist on this DB).
    """
    fk_name = _find_fk_constraint_name(bind, table, column)
    if fk_name is None:
        # Try the canonical VAPT-AI naming convention as a fallback
        fk_name = f"fk_{table}_{column}_{ref_table}"
        # Verify it exists — if not, skip
        exists = bind.execute(
            sa.text(
                """
                SELECT 1 FROM pg_constraint
                WHERE conname = :name AND contype = 'f'
                """
            ),
            {"name": fk_name},
        ).fetchone()
        if not exists:
            logger.info(
                "  SKIP %s.%s → no FK constraint found (table missing?)",
                table, column,
            )
            return False

    # Drop existing FK
    try:
        op.drop_constraint(fk_name, table, type_="foreignkey")
    except Exception as e:
        logger.warning("  drop_constraint failed for %s on %s: %s", fk_name, table, e)
        return False

    # Re-create with ondelete clause
    try:
        op.create_foreign_key(
            fk_name,
            table,
            ref_table,
            [column],
            [ref_column],
            ondelete=ondelete,
        )
        logger.info("  OK %s.%s → ondelete=%s", table, column, ondelete)
        return True
    except Exception as e:
        logger.error(
            "  FAILED to re-create FK %s on %s with ondelete=%s: %s — "
            "the FK is now DROPPED. Manual recovery needed.",
            fk_name, table, ondelete, e,
        )
        return False


def upgrade() -> None:
    """Add ondelete clauses to all vapt_scans FK constraints."""
    bind = op.get_bind()
    logger.info("0008_scan_fk_ondelete: starting upgrade")

    # ── Tables that should SET NULL on scan delete (preserve child rows) ──
    # These tables hold cross-scan value (findings, chat history) — keep them.
    set_null_tables = [
        ("vapt_findings", "scan_id", "vapt_scans", "id"),
        # NOTE: vapt_findings was already done in 0007, but we re-run here
        # for idempotency (DROP + CREATE is a no-op if the same constraint
        # already exists with the same ondelete).
        ("vapt_chat_messages", "scan_id", "vapt_scans", "id"),
    ]

    # ── Tables that should CASCADE on scan delete (scan-scoped artifacts) ──
    # These tables hold per-scan state with no value after scan deletion.
    cascade_tables = [
        ("vapt_pentest_facts", "scan_id", "vapt_scans", "id"),
        ("vapt_process_details", "scan_id", "vapt_scans", "id"),
        # NOTE: vapt_process_details was already done in 0007 — re-run here
        # for idempotency.
        ("vapt_assets", "scan_id", "vapt_scans", "id"),
        ("vapt_consent_forms", "scan_id", "vapt_scans", "id"),
        ("vapt_hitl_approvals", "scan_id", "vapt_scans", "id"),
        ("vapt_audit_logs", "scan_id", "vapt_scans", "id"),
        ("vapt_c2_sessions", "scan_id", "vapt_scans", "id"),
        ("vapt_replay_traces", "scan_id", "vapt_scans", "id"),
        ("vapt_methodology_progress", "scan_id", "vapt_scans", "id"),
        ("vapt_rl_episodes", "scan_id", "vapt_scans", "id"),
        ("vapt_rl_q_snapshots", "scan_id", "vapt_scans", "id"),
    ]

    # ── Tables that may not exist on all DBs (added in later migrations) ──
    # Try to ALTER, but skip silently if table doesn't exist.
    optional_cascade_tables = [
        ("vapt_attack_chain_facts", "scan_id", "vapt_scans", "id"),
        ("vapt_kg_facts", "scan_id", "vapt_scans", "id"),
        ("vapt_c2_tasks", "scan_id", "vapt_scans", "id"),
    ]

    logger.info("Applying ondelete=SET NULL to preserve-tables:")
    for table, col, ref_table, ref_col in set_null_tables:
        _alter_fk_ondelete(bind, table, col, ref_table, ref_col, "SET NULL")

    logger.info("Applying ondelete=CASCADE to scan-scoped tables:")
    for table, col, ref_table, ref_col in cascade_tables:
        _alter_fk_ondelete(bind, table, col, ref_table, ref_col, "CASCADE")

    logger.info("Applying ondelete=CASCADE to optional tables (may skip if missing):")
    for table, col, ref_table, ref_col in optional_cascade_tables:
        _alter_fk_ondelete(bind, table, col, ref_table, ref_col, "CASCADE")

    logger.info("0008_scan_fk_ondelete: upgrade complete")


def downgrade() -> None:
    """Revert: drop + recreate FKs WITHOUT ondelete (back to NO ACTION)."""
    bind = op.get_bind()
    logger.info("0008_scan_fk_ondelete: starting downgrade")

    all_tables = [
        ("vapt_findings", "scan_id", "vapt_scans", "id"),
        ("vapt_chat_messages", "scan_id", "vapt_scans", "id"),
        ("vapt_pentest_facts", "scan_id", "vapt_scans", "id"),
        ("vapt_process_details", "scan_id", "vapt_scans", "id"),
        ("vapt_assets", "scan_id", "vapt_scans", "id"),
        ("vapt_consent_forms", "scan_id", "vapt_scans", "id"),
        ("vapt_hitl_approvals", "scan_id", "vapt_scans", "id"),
        ("vapt_audit_logs", "scan_id", "vapt_scans", "id"),
        ("vapt_c2_sessions", "scan_id", "vapt_scans", "id"),
        ("vapt_replay_traces", "scan_id", "vapt_scans", "id"),
        ("vapt_methodology_progress", "scan_id", "vapt_scans", "id"),
        ("vapt_rl_episodes", "scan_id", "vapt_scans", "id"),
        ("vapt_rl_q_snapshots", "scan_id", "vapt_scans", "id"),
        ("vapt_attack_chain_facts", "scan_id", "vapt_scans", "id"),
        ("vapt_kg_facts", "scan_id", "vapt_scans", "id"),
        ("vapt_c2_tasks", "scan_id", "vapt_scans", "id"),
    ]

    for table, col, ref_table, ref_col in all_tables:
        # Re-create WITHOUT ondelete (back to NO ACTION)
        _alter_fk_ondelete(bind, table, col, ref_table, ref_col, None)

    logger.info("0008_scan_fk_ondelete: downgrade complete")
