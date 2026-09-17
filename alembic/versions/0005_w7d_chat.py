"""W7-D: Add vapt_conversations + vapt_chat_messages tables

Revision ID: 0005_w7d_chat
Revises: 0004_w7b_hitl_mode
Create Date: 2026-09-12

Creates 2 new tables for the CyberStrikeAI-pattern chat UI:
    vapt_conversations   — chat sessions (1 user → many conversations)
    vapt_chat_messages   — individual messages within a conversation

Each user message in a conversation may trigger a Scan. The scan_id FK on
vapt_chat_messages links the message to the scan it triggered.

This enables:
    - Sidebar with conversation history (sorted by last_message_at)
    - Click a conversation → load all its messages
    - Each message shows: role (user/assistant/system), content, scan_id link
    - HITL decisions for the linked scan displayed inline in the chat
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB, UUID


# revision identifiers, used by Alembic.
revision = "0005_w7d_chat"
down_revision = "0004_w7b_hitl_mode"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # 1. vapt_conversations
    op.create_table(
        "vapt_conversations",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column(
            "user_id",
            UUID(as_uuid=True),
            sa.ForeignKey("vapt_users.id"),
            nullable=False,
            index=True,
        ),
        sa.Column("title", sa.String(255), nullable=False, server_default="New conversation"),
        sa.Column("summary", sa.Text(), nullable=True),
        sa.Column(
            "is_archived",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("false"),
            index=True,
        ),
        sa.Column("last_message_at", sa.DateTime(timezone=True), nullable=True, index=True),
        sa.Column("message_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )
    op.create_index("ix_vapt_conversations_user_last_msg", "vapt_conversations", ["user_id", "last_message_at"])

    # 2. vapt_chat_messages
    op.create_table(
        "vapt_chat_messages",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column(
            "conversation_id",
            UUID(as_uuid=True),
            sa.ForeignKey("vapt_conversations.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        ),
        sa.Column("role", sa.String(16), nullable=False, index=True),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column(
            "scan_id",
            sa.String(64),
            sa.ForeignKey("vapt_scans.id", ondelete="SET NULL"),
            nullable=True,
            index=True,
        ),
        sa.Column(
            "metadata_json",
            JSONB,
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column("sequence", sa.Integer(), nullable=False, server_default="0", index=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )
    op.create_index(
        "ix_vapt_chat_messages_conv_seq",
        "vapt_chat_messages",
        ["conversation_id", "sequence"],
    )


def downgrade() -> None:
    op.drop_index("ix_vapt_chat_messages_conv_seq", table_name="vapt_chat_messages")
    op.drop_table("vapt_chat_messages")
    op.drop_index("ix_vapt_conversations_user_last_msg", table_name="vapt_conversations")
    op.drop_table("vapt_conversations")
