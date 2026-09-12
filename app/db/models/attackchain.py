"""
VAPT-AI attack chain + retest models.

Tables:
    attack_chains     — attack path graph nodes (kill-chain visualization)
    retest_requests   — user-requested retest of a finding ("I've Fixed This")

Attack chain graph mirrors CyberStrikeAI's project_fact_edges pattern
(internal/project/fact_edges.go). Each row = one edge in the attack graph.
"""
from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, TimestampMixin, UUIDPrimaryKey


class AttackChain(Base, UUIDPrimaryKey, TimestampMixin):
    """Attack path graph edge — for kill-chain visualization.

    Each row represents one edge in the attack graph. Nodes are referenced
    by (node_type, node_id) — node_id points to vapt_assets / vapt_findings /
    vapt_pentest_facts / vapt_c2_sessions.

    Edge types (from CyberStrikeAI project_fact_edges):
        discovered_on  — asset was discovered via this fact
        leads_to       — fact leads to another fact
        enables        — fact enables an action
        depends_on     — fact depends on another fact
        exploits       — fact exploits a vulnerability
        contains       — fact contains sub-facts
        part_of        — fact is part of a larger fact
        supports       — fact supports another fact
    """

    __tablename__ = "vapt_attack_chains"

    scan_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("vapt_scans.id"),
        nullable=False,
        index=True,
    )

    # Source node
    source_node_type: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    # asset / finding / fact / session
    source_node_id: Mapped[str] = mapped_column(String(64), nullable=False)  # FK to source table

    # Target node
    target_node_type: Mapped[str] = mapped_column(String(32), nullable=False)
    target_node_id: Mapped[str] = mapped_column(String(64), nullable=False)

    # Edge type
    edge_type: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    # discovered_on / leads_to / enables / depends_on / exploits / contains / part_of / supports

    # Edge metadata (JSONB — edge-specific info)
    metadata_json: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict, server_default="{}")


class RetestRequest(Base, UUIDPrimaryKey, TimestampMixin):
    """User-requested retest of a finding — "I've Fixed This" button.

    State machine:
        pending → scheduled → running → verified (fixed) / still_vulnerable / failed
    """

    __tablename__ = "vapt_retest_requests"

    finding_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("vapt_findings.id"),
        nullable=False,
        index=True,
    )

    scan_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("vapt_scans.id"),
        nullable=False,
        index=True,
    )

    # Status lifecycle
    status: Mapped[str] = mapped_column(
        String(16), nullable=False, default="pending", server_default="pending", index=True
    )  # pending / scheduled / running / verified / still_vulnerable / failed

    # Scheduling
    scheduled_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    # Retest result
    retest_scan_id: Mapped[str | None] = mapped_column(String(64), nullable=True)  # FK to vapt_scans.id (new scan)
    retest_finding_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), nullable=True)
    # if still_vulnerable: retest_finding_id points to new finding in retest_scan_id

    # User comment (why they think it's fixed)
    user_comment: Mapped[str | None] = mapped_column(Text, nullable=True)

    # ISO 29147 disclosure window tracking
    disclosure_window_expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True, index=True)