"""
VAPT-AI audit log + quarantine — D23 chain of custody.

Tables:
    audit_log   — append-only, HMAC-sealed audit log (every action by user/agent/system)
    quarantine  — dangerous output quarantined for review (encrypted at rest)

D23: Audit log is append-only (no UPDATE/DELETE). HMAC-SHA256 tamper seals
link each entry to the previous one (blockchain-style chain). 7-year retention.
"""
from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Integer, LargeBinary, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, TimestampMixin, UUIDPrimaryKey


class AuditLog(Base, UUIDPrimaryKey):
    """Append-only audit log — D23.

    EVERY action by user / agent / system is logged here. HMAC-SHA256 tamper
    seal links each entry to the previous one (chain). 7-year retention
    (regulatory minimum).

    NOTE: This table does NOT use TimestampMixin — created_at is the only
    timestamp, and it's set server-side. No updated_at (append-only).
    """

    __tablename__ = "vapt_audit_log"

    # Actor
    actor_type: Mapped[str] = mapped_column(String(16), nullable=False, index=True)
    # user / agent / system
    actor_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    # user UUID / agent name / "system"

    # Action
    action: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    # login / logout / scan_start / scan_complete / tool_execute / hitl_approve / hitl_abort / ...

    # Target
    target_table: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    target_id: Mapped[str | None] = mapped_column(String(64), nullable=True)

    # Before/after state (for UPDATE-like actions — but stored as new row, not in-place update)
    before_json: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    after_json: Mapped[dict | None] = mapped_column(JSONB, nullable=True)

    # Request context
    ip_address: Mapped[str | None] = mapped_column(String(64), nullable=True)
    user_agent: Mapped[str | None] = mapped_column(String(512), nullable=True)
    scan_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)

    # HMAC-SHA256 tamper seal — chain to previous entry
    # seal = HMAC(encryption_key, f"{prev_seal}|{actor_type}|{actor_id}|{action}|{target}|{created_at}")
    prev_seal: Mapped[str | None] = mapped_column(String(128), nullable=True)  # previous entry's seal
    tamper_seal: Mapped[str] = mapped_column(String(128), nullable=False)  # this entry's seal

    # Timestamp (server-side)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now(), index=True
    )


class Quarantine(Base, UUIDPrimaryKey, TimestampMixin):
    """Quarantined dangerous output — encrypted at rest.

    When tool output contains potentially dangerous content (e.g. full DB dump,
    credential file, source code), it's quarantined instead of stored in
    Evidence.raw_output. User must explicitly review + approve to release.
    """

    __tablename__ = "vapt_quarantine"

    scan_id: Mapped[str | None] = mapped_column(
        String(64),
        ForeignKey("vapt_scans.id"),
        nullable=True,
        index=True,
    )

    # Source of quarantined content
    source: Mapped[str] = mapped_column(String(32), nullable=False)
    # tool_output / finding / evidence / upload

    source_id: Mapped[str | None] = mapped_column(String(64), nullable=True)  # FK to source row

    # Reason for quarantine
    reason: Mapped[str] = mapped_column(String(64), nullable=False)
    # contains_credentials / contains_pii / contains_secrets / full_db_dump / source_code / ...

    # Quarantined content (encrypted with Fernet — VAPT_AI_ENCRYPTION_KEY)
    quarantined_content: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)

    # Original SHA-256 hash (for integrity verification after decryption)
    original_hash: Mapped[str] = mapped_column(String(128), nullable=False)

    # Review status
    reviewed: Mapped[bool] = mapped_column(nullable=False, default=False, server_default="false")
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    reviewed_by_user_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), nullable=True)
    review_decision: Mapped[str | None] = mapped_column(String(16), nullable=True)
    # release / destroy / keep_quarantined

    # Auto-expire (90 days per §7.4)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)