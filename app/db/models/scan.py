"""
VAPT-AI scan lifecycle models.

Tables:
    scans           — one scan per user request (replaces EVVO's `scans` table)
    consent_forms   — signed consent + scope declaration per scan (NEW — D19)
    assets          — assets discovered during scan (NEW — was JSON in EVVO)

NOTE: EVVO's legacy `scans` table is NOT dropped (alembic env include_object
filter ignores it). This new `vapt_scans` table is for the VAPT-AI scan flow.
"""
from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, StringPrimaryKey, TimestampMixin, UUIDPrimaryKey


class Scan(Base, StringPrimaryKey, TimestampMixin):
    """One pentest scan session.

    Uses string PK (e.g. 'scan_abc123') for human-readable IDs in URLs/logs.
    """

    __tablename__ = "vapt_scans"

    # Foreign key to vapt_users.id (NOT legacy users table)
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("vapt_users.id"),
        nullable=False,
        index=True,
    )

    # Target
    target: Mapped[str] = mapped_column(Text, nullable=False)  # URL or IP
    target_type: Mapped[str] = mapped_column(String(32), nullable=False)  # web_app / api / network_range / single_host

    # Mode selection
    agent_mode: Mapped[str] = mapped_column(
        String(32), nullable=False, default="supervisor", server_default="supervisor"
    )  # deep / plan_execute / supervisor

    # Natural-language prompt (flexible — AI auto-interprets intent)
    user_prompt: Mapped[str | None] = mapped_column(Text, nullable=True)
    # AI-parsed intent breakdown (mode + scope + goal + estimated_steps)
    parsed_intent: Mapped[dict | None] = mapped_column(JSONB, nullable=True)

    # Status lifecycle
    status: Mapped[str] = mapped_column(
        String(32), nullable=False, default="pending", server_default="pending", index=True
    )  # pending / authorized / running / paused / completed / failed / cancelled

    # Progress (0-100)
    progress: Mapped[int] = mapped_column(Integer, nullable=False, default=0, server_default="0")

    # Scope (JSONB: declared hosts/IP ranges/ports)
    scope_json: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict, server_default="{}")

    # Timing
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    # Final result summary (counts: findings_by_severity, exploit_success, etc.)
    result_summary: Mapped[dict | None] = mapped_column(JSONB, nullable=True)

    # Error (if status=failed)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)


class ConsentForm(Base, UUIDPrimaryKey, TimestampMixin):
    """Signed consent form for a scan — D19 mandatory before any active scan.

    Captures asserted ownership + declared scope + ToS acceptance + verification
    method (DNS TXT / HTTP meta-tag / file upload).
    """

    __tablename__ = "vapt_consent_forms"

    scan_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("vapt_scans.id"),
        nullable=False,
        index=True,
    )

    # Asserted ownership
    asserted_owner: Mapped[str] = mapped_column(String(255), nullable=False)  # name / email / org
    asserted_owner_role: Mapped[str] = mapped_column(String(64), nullable=False)  # owner / authorized_rep

    # Scope declaration (JSONB: hosts, ip_ranges, ports, exclusions)
    declared_scope_json: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)

    # ToS acceptance
    tos_accepted: Mapped[bool] = mapped_column(nullable=False, default=False)
    tos_accepted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    # Verification method (one of: dns_txt / http_meta / file_upload / owned_vps / htb_machine / thm_room)
    verification_method: Mapped[str] = mapped_column(String(32), nullable=False)
    verification_token: Mapped[str | None] = mapped_column(String(128), nullable=True)  # vapt_<32hex>_<userid>_<exp>
    verified: Mapped[bool] = mapped_column(nullable=False, default=False, server_default="false")
    verified_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    # Raw verification response (DNS lookup result / HTTP body / file content)
    verification_response: Mapped[str | None] = mapped_column(Text, nullable=True)


class Asset(Base, UUIDPrimaryKey, TimestampMixin):
    """Asset discovered during a scan (host, domain, URL, API endpoint).

    NEW in VAPT-AI v3.2 — EVVO stored assets only in the KG (InventoryItem
    dataclass). This table provides SQL-queryable asset inventory + feeds the KG.
    """

    __tablename__ = "vapt_assets"

    scan_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("vapt_scans.id"),
        nullable=False,
        index=True,
    )

    # Asset type
    type: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    # host / domain / url / api_endpoint / port / service / subdomain

    # Asset value (e.g. "10.10.10.5", "example.com", "https://example.com/api/login")
    value: Mapped[str] = mapped_column(Text, nullable=False, index=True)

    # Metadata (JSONB — flexible per type)
    # Examples:
    #   host: {hostname, ip, reverse_dns, os_guess, services: [...]}
    #   url:  {method, status_code, content_type, title, tech_stack: [...]}
    #   port: {port, protocol, service, version, banner}
    metadata_json: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict, server_default="{}")

    # Source agent that discovered this asset
    discovered_by_agent: Mapped[str | None] = mapped_column(String(64), nullable=True)

    # Confidence score (0.0-1.0) — for fuzzy discoveries (e.g. subdomain brute-force)
    confidence: Mapped[float] = mapped_column(nullable=False, default=1.0, server_default="1.0")