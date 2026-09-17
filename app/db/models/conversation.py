"""
VAPT-AI Conversation + Chat Message models — W7-D.

Tables:
    vapt_conversations   — chat conversations (1 user has many conversations)
    vapt_chat_messages   — individual messages within a conversation

Design (CyberStrikeAI-pattern chat UI):
    - User opens chat page → sees sidebar with conversation history
    - User clicks "New chat" → creates new Conversation row
    - User types message in chat input → creates ChatMessage (role=user)
    - Backend triggers a Scan (each user message = 1 new scan)
    - Scan result + agent decisions stream back via SSE → saved as ChatMessage (role=assistant)
    - Conversation persists in DB for later review / reporting

Each ChatMessage may reference a vapt_scans.id (FK) — links the chat message
to the scan it triggered. This allows:
    - Clicking a message in the sidebar → jump to scan results
    - HITL decisions for that scan displayed inline in the chat
"""
from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, TimestampMixin, UUIDPrimaryKey


class Conversation(Base, UUIDPrimaryKey, TimestampMixin):
    """A chat conversation — contains multiple ChatMessage rows.

    One user has many conversations. Each conversation is a "chat session"
    that the user can return to later.
    """

    __tablename__ = "vapt_conversations"

    # Owner
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("vapt_users.id"),
        nullable=False,
        index=True,
    )

    # Display title (auto-generated from first user message, truncated to 60 chars)
    title: Mapped[str] = mapped_column(String(255), nullable=False, default="New conversation")

    # Optional: summary of what was discussed (for sidebar preview)
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Soft delete
    is_archived: Mapped[bool] = mapped_column(
        nullable=False, default=False, server_default="false", index=True,
    )

    # Last message timestamp (denormalized for fast sidebar sorting without joining messages)
    last_message_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True, index=True,
    )

    # Message count (denormalized for fast sidebar display)
    message_count: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, server_default="0",
    )


class ChatMessage(Base, UUIDPrimaryKey, TimestampMixin):
    """A single message in a conversation (user or assistant).

    Roles:
        user      — message typed by the human
        assistant — response from the AI agent (may include scan results)
        system    — system messages (e.g. "Scan started", "Scan completed")

    Each user message typically triggers a Scan. The scan_id FK links the
    message to the scan it triggered (for user messages) or the scan it
    reports on (for assistant messages).
    """

    __tablename__ = "vapt_chat_messages"

    # Conversation this message belongs to
    conversation_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("vapt_conversations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Role: user / assistant / system
    role: Mapped[str] = mapped_column(String(16), nullable=False, index=True)

    # Message content (natural language)
    content: Mapped[str] = mapped_column(Text, nullable=False)

    # Optional: scan triggered by this message (user message → scan_id; assistant message → same scan_id)
    scan_id: Mapped[str | None] = mapped_column(
        String(64),
        ForeignKey("vapt_scans.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )

    # Optional: metadata (JSONB — flexible per role)
    # Examples:
    #   user message: {"target": "10.10.10.5", "agent_mode": "supervisor"}
    #   assistant message: {"findings_count": 3, "exploit_success": 1, "duration_seconds": 45.2}
    #   system message: {"event": "scan_started", "scan_id": "scan_abc123"}
    metadata_json: Mapped[dict] = mapped_column(
        JSONB, nullable=False, default=dict, server_default="{}",
    )

    # Message ordering within conversation (for stable sort)
    sequence: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, server_default="0", index=True,
    )
