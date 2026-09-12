"""
VAPT-AI unified Command & Control models — D28.

Tables:
    c2_sessions  — unified session table (Python beacon + MSF meterpreter + sqlmap webshell)
    c2_tasks     — tasks sent to beacons (with HITL gate for L3+)

D28 decision: NO separate WebShellConnection/WebShellCommand tables.
MSF meterpreter sessions + sqlmap --os-shell webshells + custom Python beacons
ALL sync into the same c2_sessions table (distinguished by beacon_type).
"""
from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, TimestampMixin, UUIDPrimaryKey


class C2Session(Base, UUIDPrimaryKey, TimestampMixin):
    """Unified C2 session — D28.

    One row per active beacon/meterpreter/webshell session. Sessions are
    created when a beacon checks in (HTTP/TCP/WS) or when Metasploit
    establishes a session. All session types share this table.
    """

    __tablename__ = "vapt_c2_sessions"

    scan_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("vapt_scans.id"),
        nullable=False,
        index=True,
    )

    # Beacon type — distinguishes session source
    beacon_type: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    # python_beacon / msf_meterpreter / msf_shell / sqlmap_webshell / custom

    # Listener that accepted this session
    listener_type: Mapped[str] = mapped_column(String(32), nullable=False)
    # http / https / tcp / websocket
    listener_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), nullable=True)

    # Remote target info
    remote_address: Mapped[str] = mapped_column(String(64), nullable=False, index=True)  # IP:port
    hostname: Mapped[str | None] = mapped_column(String(255), nullable=True)
    username: Mapped[str | None] = mapped_column(String(255), nullable=True)
    os: Mapped[str | None] = mapped_column(String(255), nullable=True)  # "Linux ubuntu 5.15.0-91-generic"
    arch: Mapped[str | None] = mapped_column(String(16), nullable=True)  # x64 / x86 / arm64

    # Session timing
    checkin_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=func.now(), server_default=func.now())
    last_seen_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=func.now(), server_default=func.now())
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)  # for time-limited sessions

    # Status
    status: Mapped[str] = mapped_column(
        String(16), nullable=False, default="active", server_default="active", index=True
    )  # active / dead / closed / expired

    # Session metadata (JSONB — beacon-specific info)
    # Examples:
    #   python_beacon:    {beacon_version, sleep_interval, jitter, capabilities: [...]}
    #   msf_meterpreter:  {msf_session_id, transport, payload_type, ...}
    #   sqlmap_webshell:  {webshell_url, webshell_type (php/asp/jsp), ...}
    metadata_json: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict, server_default="{}")


class C2Task(Base, UUIDPrimaryKey, TimestampMixin):
    """Task sent to a C2 session beacon.

    Tasks are queued by the agent (or manually by user via UI). Beacon polls
    for pending tasks and posts results back.

    Task levels (D25 HITL gate):
        L1 — read-only (ls, cat, whoami, id) — no HITL
        L2 — write non-destructive (mkdir, touch, upload benign file) — no HITL
        L3 — write destructive (rm, mv, chmod, upload executable) — HITL required
        L4 — exec (run command, run binary) — HITL required
        L5 — high-impact (privilege escalation, lateral movement, persistence) — HITL + extra confirm
    """

    __tablename__ = "vapt_c2_tasks"

    session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("vapt_c2_sessions.id"),
        nullable=False,
        index=True,
    )

    # Task content
    command: Mapped[str] = mapped_column(Text, nullable=False)  # shell command or task type
    args_json: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    # Examples:
    #   {cmd: "ls -la /tmp"}
    #   {action: "upload", local_path: "...", remote_path: "..."}
    #   {action: "download", remote_path: "...", local_path: "..."}

    # Task level (drives HITL gate)
    level: Mapped[int] = mapped_column(Integer, nullable=False, index=True)  # 1-5

    # HITL approval (required for L3+)
    hitl_approval_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), nullable=True
    )

    # Status lifecycle
    status: Mapped[str] = mapped_column(
        String(16), nullable=False, default="queued", server_default="queued", index=True
    )  # queued / sent / running / completed / failed / cancelled / hitl_pending / hitl_denied

    # Result
    result: Mapped[str | None] = mapped_column(Text, nullable=True)  # stdout/stderr (PII-redacted)
    exit_code: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Timing
    sent_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    # Error (if status=failed)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)