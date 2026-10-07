"""
VAPT-AI ProcessDetail model — Phase F1 (CyberStrikeAI pattern migration).

Mirrors CyberStrikeAI's `process_details` table (internal/database/database.go:222).

This table is the SINGLE SOURCE OF TRUTH for the agent timeline — every
event that happens during a scan (tool_call_started, tool_call_completed,
iteration boundary, thinking, phase_change, finding_detected, scan_started,
scan_complete, scan_error, hitl_approval_required, hitl_decision_made,
cancelled, error, timeout) gets persisted here.

Why this matters:
    Before Phase F1, VAPT-AI only emitted events to the in-memory SSE
    bus (app/pentest/events.py). When the SSE subscriber disconnected
    (page refresh, browser close), all events were lost forever — the
    scan history UI had no way to reconstruct what happened.

    With `process_details` persisted:
      - Replay: open any old scan → fetch its messages → for each
        message, fetch its process_details → reconstruct timeline
      - Audit: full chain of custody for any decision
      - Debug: see exactly what the agent did at each iteration
        (even if the SSE stream was lost)

Design (mirrors CyberStrikeAI):
    - FK CASCADE on message_id + conversation_id: when conversation or
      message is deleted, all its process_details go with it
    - FK CASCADE on scan_id: when scan is deleted, all process_details
      for that scan go with it
    - Denormalized conversation_id + scan_id columns: enables fast
      `WHERE scan_id = ?` queries without joining through messages
    - JSONB data column: full event payload (tool args, result, etc.)
    - event_type indexed: enables filtering by type (e.g. all tool_calls)

Event types (alignment with app/pentest/events.py emit_* functions):
    lifecycle: scan_started, scan_progress, phase_change, iteration,
               scan_complete, scan_error
    react:     thinking, assistant_message
    tools:     tool_call_started, tool_call_completed, tool_call_failed
    findings:  finding_detected
    hitl:      hitl_approval_required, hitl_decision_made
    terminal:  cancelled, error, timeout
"""
from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, TimestampMixin, UUIDPrimaryKey


class ProcessDetail(Base, UUIDPrimaryKey, TimestampMixin):
    """A single timeline event persisted to DB.

    Each row corresponds to one SSE event that was emitted during a scan
    (tool_call_started, tool_call_completed, iteration, thinking, phase_change,
    scan_started, scan_complete, scan_error, finding_detected, hitl_*, etc.).

    Linked to:
        - vapt_chat_messages.id (FK CASCADE) — timeline of a specific message
        - vapt_conversations.id (FK CASCADE) — denormalized for fast queries
        - vapt_scans.id (FK CASCADE) — denormalized for fast queries
    """

    __tablename__ = "vapt_process_details"

    # ── FK to message (PRIMARY timeline linkage) ──
    # Each process_detail belongs to ONE message (the assistant message
    # being generated when the event fired). CyberStrikeAI does the same.
    message_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("vapt_chat_messages.id", ondelete="CASCADE"),
        nullable=True,  # nullable: some events fire before any message exists
        index=True,
    )

    # ── Denormalized conversation_id + scan_id for fast queries ──
    # CyberStrikeAI keeps both conversation_id and message_id. VAPT-AI
    # also keeps scan_id since scan_id is used everywhere (events.py,
    # tool_bridge.py, scan_registry.py).
    conversation_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("vapt_conversations.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )

    scan_id: Mapped[str | None] = mapped_column(
        String(64),
        ForeignKey("vapt_scans.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )

    # ── Event identity ──
    event_type: Mapped[str] = mapped_column(
        String(64), nullable=False, index=True,
    )
    # See module docstring for the full list of valid event_type values.
    # Indexed for `WHERE event_type = 'tool_call_started'` queries.

    # ── Human-readable summary (shown in timeline list view) ──
    # Mirrors CyberStrikeAI's `message` field in StreamEvent.
    # Examples:
    #   "🔧 nmap(target=pentest-ground.com, ports=4280)"
    #   "🤔 recon is thinking... (round 3/30)"
    #   "✅ Complete: 5 findings"
    message: Mapped[str | None] = mapped_column(Text, nullable=True)

    # ── Full payload (JSONB) ──
    # Stores the complete event data: tool_name, arguments, tool_call_id,
    # result_preview, success, agent_name, iteration, progress, etc.
    # Mirrors CyberStrikeAI's `data` field.
    data: Mapped[dict | None] = mapped_column(JSONB, nullable=True)

    # ── Idempotency signature (for dedup) ──
    # CyberStrikeAI computes a payload signature (eventType + message + JSON(data))
    # to dedupe events that fire multiple times. We do the same — but only
    # populate this for tool_call/tool_result events (the ones that are
    # most prone to double-firing due to retry logic).
    # Format: f"{event_type}|{tool_call_id or message_id}"
    signature: Mapped[str | None] = mapped_column(
        String(255), nullable=True, index=True,
    )

    # created_at comes from TimestampMixin (indexed by default)


__all__ = ["ProcessDetail"]