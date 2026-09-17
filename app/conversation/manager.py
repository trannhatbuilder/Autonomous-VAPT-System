"""
VAPT-AI Conversation Manager — W7-D.

CRUD operations for chat conversations + messages.

Pattern (CyberStrikeAI-style chat):
    1. User opens chat page → list_conversations() populates sidebar
    2. User clicks "New chat" → create_conversation() → empty conversation
    3. User types message → add_message(role="user") → triggers scan
    4. Scan completes → add_message(role="assistant", content=result, scan_id=...)
    5. User returns later → get_conversation(id) loads all messages

Each user message typically triggers a Scan. The scan_id FK on ChatMessage
links the message to the scan it triggered (for user messages) or the scan
it reports on (for assistant messages).
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime, UTC
from typing import Any

from sqlalchemy import select, update, func, desc
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models.conversation import Conversation, ChatMessage

logger = logging.getLogger(__name__)


class ConversationManager:
    """Async CRUD for conversations + chat messages."""

    def __init__(self, session: AsyncSession):
        self.session = session

    # ---------- Conversation CRUD ----------

    async def create_conversation(
        self,
        user_id: uuid.UUID,
        title: str = "New conversation",
    ) -> Conversation:
        """Create a new conversation for a user."""
        conv = Conversation(
            user_id=user_id,
            title=title,
            message_count=0,
        )
        self.session.add(conv)
        await self.session.flush()
        logger.info("Conversation created: id=%s user=%s title=%r", conv.id, user_id, title)
        return conv

    async def get_conversation(
        self,
        conversation_id: uuid.UUID,
        user_id: uuid.UUID | None = None,
    ) -> Conversation | None:
        """Get a conversation by ID. If user_id provided, verify ownership."""
        stmt = select(Conversation).where(Conversation.id == conversation_id)
        if user_id is not None:
            stmt = stmt.where(Conversation.user_id == user_id)
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def list_conversations(
        self,
        user_id: uuid.UUID,
        include_archived: bool = False,
        limit: int = 50,
        offset: int = 0,
    ) -> list[Conversation]:
        """List conversations for a user, newest first."""
        stmt = (
            select(Conversation)
            .where(Conversation.user_id == user_id)
            .order_by(desc(Conversation.last_message_at).nullslast(), desc(Conversation.created_at))
            .limit(limit)
            .offset(offset)
        )
        if not include_archived:
            stmt = stmt.where(Conversation.is_archived == False)  # noqa: E712
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def update_conversation(
        self,
        conversation_id: uuid.UUID,
        title: str | None = None,
        summary: str | None = None,
        is_archived: bool | None = None,
    ) -> Conversation | None:
        """Update conversation fields."""
        conv = await self.get_conversation(conversation_id)
        if conv is None:
            return None
        if title is not None:
            conv.title = title[:255]
        if summary is not None:
            conv.summary = summary
        if is_archived is not None:
            conv.is_archived = is_archived
        await self.session.flush()
        return conv

    async def delete_conversation(self, conversation_id: uuid.UUID) -> bool:
        """Delete a conversation + all its messages (CASCADE)."""
        conv = await self.get_conversation(conversation_id)
        if conv is None:
            return False
        await self.session.delete(conv)
        await self.session.flush()
        logger.info("Conversation deleted: id=%s", conversation_id)
        return True

    async def archive_conversation(self, conversation_id: uuid.UUID) -> Conversation | None:
        """Soft-delete (archive) a conversation."""
        return await self.update_conversation(conversation_id, is_archived=True)

    # ---------- Message CRUD ----------

    async def add_message(
        self,
        conversation_id: uuid.UUID,
        role: str,
        content: str,
        scan_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ChatMessage:
        """Add a message to a conversation + update conversation stats.

        Args:
            conversation_id: Conversation UUID
            role: "user" | "assistant" | "system"
            content: Message text
            scan_id: Optional scan ID this message triggered/reports on
            metadata: Optional JSONB metadata (findings_count, duration, etc.)

        Returns:
            The created ChatMessage.
        """
        # Get current sequence number
        seq_stmt = (
            select(func.max(ChatMessage.sequence))
            .where(ChatMessage.conversation_id == conversation_id)
        )
        result = await self.session.execute(seq_stmt)
        current_max = result.scalar() or 0
        next_seq = current_max + 1

        msg = ChatMessage(
            conversation_id=conversation_id,
            role=role,
            content=content,
            scan_id=scan_id,
            metadata_json=metadata or {},
            sequence=next_seq,
        )
        self.session.add(msg)
        await self.session.flush()

        # Update conversation stats (denormalized)
        conv_stmt = (
            update(Conversation)
            .where(Conversation.id == conversation_id)
            .values(
                message_count=Conversation.message_count + 1,
                last_message_at=datetime.now(UTC),
            )
        )
        await self.session.execute(conv_stmt)

        # Auto-generate title from first user message if title is still default
        if role == "user":
            conv = await self.get_conversation(conversation_id)
            if conv and (conv.title == "New conversation" or not conv.title):
                new_title = content[:60].strip()
                if len(content) > 60:
                    new_title += "..."
                conv.title = new_title

        await self.session.flush()
        logger.info(
            "Chat message added: conv=%s role=%s seq=%d scan=%s",
            conversation_id, role, next_seq, scan_id,
        )
        return msg

    async def list_messages(
        self,
        conversation_id: uuid.UUID,
        limit: int = 100,
        offset: int = 0,
    ) -> list[ChatMessage]:
        """List messages in a conversation, ordered by sequence (chronological)."""
        stmt = (
            select(ChatMessage)
            .where(ChatMessage.conversation_id == conversation_id)
            .order_by(ChatMessage.sequence.asc())
            .limit(limit)
            .offset(offset)
        )
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def get_message(self, message_id: uuid.UUID) -> ChatMessage | None:
        """Get a single message by ID."""
        result = await self.session.execute(
            select(ChatMessage).where(ChatMessage.id == message_id)
        )
        return result.scalar_one_or_none()

    # ---------- Helpers ----------

    async def get_conversation_with_messages(
        self,
        conversation_id: uuid.UUID,
        user_id: uuid.UUID | None = None,
    ) -> dict[str, Any] | None:
        """Get a conversation + all its messages (for chat page load)."""
        conv = await self.get_conversation(conversation_id, user_id=user_id)
        if conv is None:
            return None
        messages = await self.list_messages(conversation_id)
        return {
            "id": str(conv.id),
            "title": conv.title,
            "summary": conv.summary,
            "is_archived": conv.is_archived,
            "message_count": conv.message_count,
            "last_message_at": conv.last_message_at.isoformat() if conv.last_message_at else None,
            "created_at": conv.created_at.isoformat() if conv.created_at else None,
            "messages": [
                {
                    "id": str(m.id),
                    "role": m.role,
                    "content": m.content,
                    "scan_id": m.scan_id,
                    "metadata": m.metadata_json,
                    "sequence": m.sequence,
                    "created_at": m.created_at.isoformat() if m.created_at else None,
                }
                for m in messages
            ],
        }
