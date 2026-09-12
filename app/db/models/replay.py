"""
VAPT-AI replay trace metadata model.

Tables:
    replay_traces — replay trace metadata (one row per scan's trace)

Replay traces are JSONL files at data/traces/scan_<id>.jsonl — each line is
one turn: {turn, state, action, observation, reward, kg_edge_updates}.

This table stores the METADATA (trace file path, turn count, total reward,
KG edges updated, final epsilon, final Q-values). The actual JSONL content
lives on disk (regenerated from RLTransition table if needed).

Retention: 365 days for JSONL files (per §7.4); metadata row kept for audit.

Ported from EVVO Sentinel's shield_engine/replay_trace.py (96 LOC).
"""
from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, TimestampMixin, UUIDPrimaryKey


class ReplayTrace(Base, UUIDPrimaryKey, TimestampMixin):
    """Replay trace metadata — one row per scan.

    The actual JSONL trace content is on disk at trace_file_path. This table
    enables fast queries ("show me scans with reward > 5", "show me scans
    that updated KG edge X") without parsing JSONL files.

    Replay viewer (W21 task) reads this table + the JSONL file to render
    per-turn state/action/reward timeline in the UI.
    """

    __tablename__ = "vapt_replay_traces"

    scan_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("vapt_scans.id"),
        nullable=False,
        unique=True,  # one trace per scan
        index=True,
    )

    # JSONL file path (relative to project root)
    # e.g. "data/traces/scan_abc123.jsonl"
    trace_file_path: Mapped[str] = mapped_column(Text, nullable=False)

    # Statistics
    turn_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    total_reward: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)

    # Timing
    start_timestamp: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    end_timestamp: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    # KG edges updated during this scan (JSONB — list of edge_id + before/after probability)
    # [
    #   {"edge_id": "uuid", "edge_type": "resulted_in", "before": 0.5, "after": 0.67, "success": true},
    #   ...
    # ]
    kg_edges_updated_json: Mapped[list] = mapped_column(
        JSONB, nullable=False, default=list, server_default="[]"
    )

    # RL state at end of scan (for convergence analysis — W27 task)
    final_epsilon: Mapped[float | None] = mapped_column(Float, nullable=True)
    # final Q-values (JSONB — 7-dim array for the 7 actions, sampled from final state)
    final_q_values_json: Mapped[list | None] = mapped_column(JSONB, nullable=True)

    # Agent mode used (deep / plan_execute / supervisor)
    agent_mode: Mapped[str | None] = mapped_column(String(32), nullable=True)

    # Final scan status (for filtering — completed/failed/cancelled)
    scan_status: Mapped[str | None] = mapped_column(String(16), nullable=True, index=True)

    # Notes (optional — user can annotate interesting scans for later review)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Export path (if user exported this trace as PDF for legal record — W21 task)
    pdf_export_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    pdf_exported_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)