"""W8-E: Add vapt_attack_technique_catalog table

Revision ID: 0006_w8e_attack_catalog
Revises: 0005_w7d_chat
Create Date: 2026-09-15

Creates 1 new table for MITRE ATT&CK Enterprise catalog (curated subset
for web/network pentest MVP — 44 techniques seeded from
data/attack_mapping.yaml at app startup).

This is the counterpart of vapt_wstg_methodology_catalog (W1-C, 119 WSTG
test IDs). Together they form the bidirectional WSTG ↔ ATT&CK ↔ CWE
cross-reference network used by:
    - Finding tagging (mitre_attack_technique column on vapt_findings)
    - PDF report "ATT&CK Heatmap" section (W20)
    - PDF report "Detection" + "Mitigation" text per finding (W20)
    - SARIF 2.1.0 export properties (W20)
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB, UUID


# revision identifiers, used by Alembic.
revision = "0006_w8e_attack_catalog"
down_revision = "0005_w7d_chat"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "vapt_attack_technique_catalog",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("technique_id", sa.String(32), nullable=False, unique=True, index=True),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("tactic", sa.String(255), nullable=False, index=True),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("detection", sa.Text(), nullable=True),
        sa.Column("mitigation", sa.Text(), nullable=True),
        sa.Column("related_wstg_ids", JSONB, nullable=False, server_default=sa.text("'[]'::jsonb")),
        sa.Column("related_cwe", JSONB, nullable=False, server_default=sa.text("'[]'::jsonb")),
        sa.Column("example_uses", JSONB, nullable=False, server_default=sa.text("'[]'::jsonb")),
        sa.Column("applicable_to_web_mvp", sa.Boolean(), nullable=False, server_default=sa.text("true")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )


def downgrade() -> None:
    op.drop_index(
        "ix_vapt_attack_technique_catalog_applicable_to_web_mvp",
        table_name="vapt_attack_technique_catalog",
    )
    op.drop_index(
        "ix_vapt_attack_technique_catalog_tactic",
        table_name="vapt_attack_technique_catalog",
    )
    op.drop_table("vapt_attack_technique_catalog")
