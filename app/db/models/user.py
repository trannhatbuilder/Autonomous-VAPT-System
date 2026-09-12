"""
VAPT-AI single-user authentication models.

Tables:
    users           — single user (admin). Replaces EVVO's `users` table
                      (which was created via raw SQL in alembic/0001).
                      This model is for NEW VAPT-AI auth flow (W1-F).
    refresh_tokens  — JWT refresh tokens with rotation + revocation.

NOTE: EVVO's legacy `users` table is NOT dropped (alembic env include_object
filter ignores it). W1-F will bootstrap the new user from VAPT_AI_ADMIN_EMAIL
+ VAPT_AI_ADMIN_PASSWORD env vars on first startup.
"""
from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import Boolean, DateTime, Integer, String, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, TimestampMixin, UUIDPrimaryKey, utcnow


class User(Base, UUIDPrimaryKey, TimestampMixin):
    """Single VAPT-AI user (admin).

    EVVO had multi-user RBAC; VAPT-AI v3.2 is single-user per D29.
    This table still supports multiple rows for future migration but the
    auth flow (W1-F) treats the first user as the canonical admin.
    """

    __tablename__ = "vapt_users"

    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    role: Mapped[str] = mapped_column(String(32), nullable=False, default="admin", server_default="admin")
    display_name: Mapped[str | None] = mapped_column(String(255), nullable=True)

    # Auth state
    failed_login_attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=0, server_default="0")
    locked_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_login_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    # User settings (JSONB for flexibility — UI theme, default scan mode, etc.)
    settings: Mapped[dict] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        server_default="{}",
    )

    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True, server_default="true")


class RefreshToken(Base, UUIDPrimaryKey, TimestampMixin):
    """JWT refresh token with rotation + revocation.

    W1-F will issue refresh tokens on login. Each refresh rotates the token
    (old token revoked, new one issued). Revoked tokens are kept for audit
    (7-year retention per §7.4).
    """

    __tablename__ = "vapt_refresh_tokens"

    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        nullable=False,
        index=True,
    )
    token_hash: Mapped[str] = mapped_column(
        String(128),  # SHA-256 hex = 64 chars, leave room for prefix
        unique=True,
        nullable=False,
        index=True,
    )
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    revoked: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False, server_default="false")
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    revoked_reason: Mapped[str | None] = mapped_column(String(255), nullable=True)

    # Track which token replaced this one (rotation chain)
    replaced_by_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), nullable=True
    )

    # Device/IP tracking (for security audit)
    issued_from_ip: Mapped[str | None] = mapped_column(String(64), nullable=True)
    issued_user_agent: Mapped[str | None] = mapped_column(String(512), nullable=True)