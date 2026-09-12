"""
VAPT-AI HITL (Human-in-the-Loop) approval gate — D25.

Tables:
    hitl_approvals  — pending/approved/aborted HITL approval requests

D25: HITL approval gate is MANDATORY for destructive operations:
    - Metasploit exploit execution
    - sqlmap --os-shell / --dump
    - Custom RCE PoC
    - Command injection execution
    - C2 L3+ tasks (file access, change exec, destructive)
    - Payload generation (msfvenom, custom C2 payload)
    - WebShell deploy/exec

D18 guardrail: HITL cannot be bypassed.
"""
from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, TimestampMixin, UUIDPrimaryKey


class HITLApproval(Base, UUIDPrimaryKey, TimestampMixin):
    """HITL approval request for a destructive operation.

    Flow:
        1. Agent decides to execute destructive op
        2. HITL middleware intercepts → creates HITLApproval (status=pending)
        3. SSE event "hitl_approval_required" sent to UI
        4. User sees modal: tool_name + target + args + predicted_impact +
           agent_reasoning + KG_confidence + 5-min countdown
        5. User clicks Approve / Approve-with-time-limit / Abort / Edit-args
        6. status updated → tool execution proceeds (approved) or
           agent receives user_aborted (aborted) or approval_timeout (5min expiry)
    """

    __tablename__ = "vapt_hitl_approvals"

    scan_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("vapt_scans.id"),
        nullable=False,
        index=True,
    )

    # What the agent wants to do
    tool_name: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    # metasploit / sqlmap / custom_rce / c2_task / payload_gen / webshell_deploy / ...

    target: Mapped[str] = mapped_column(Text, nullable=False)  # URL / IP / endpoint

    # Tool arguments (JSONB — tool-specific)
    args_json: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)

    # Predicted impact (agent's assessment)
    predicted_impact: Mapped[str] = mapped_column(String(32), nullable=False)
    # destructive / data_modification / data_exfiltration / persistence / lateral / priv_esc

    # Agent reasoning (why it wants to do this)
    agent_reasoning: Mapped[str] = mapped_column(Text, nullable=False)

    # KG confidence (0.0-1.0) — how confident the KG is this attack path works
    kg_confidence: Mapped[float] = mapped_column(Float, nullable=False, default=0.5)

    # Status lifecycle
    status: Mapped[str] = mapped_column(
        String(16), nullable=False, default="pending", server_default="pending", index=True
    )  # pending / approved / approved_with_time_limit / aborted / timeout / edited

    # User decision
    user_decision: Mapped[str | None] = mapped_column(String(16), nullable=True)
    # approve / approve_with_time_limit / abort / edit_args

    # Edited args (if user clicked Edit-args then approve)
    edited_args_json: Mapped[dict | None] = mapped_column(JSONB, nullable=True)

    # Time limit (if approved_with_time_limit — tool auto-aborts after N seconds)
    time_limit_seconds: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Timing
    decided_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)  # 5-min auto-timeout

    # User who decided (for audit — single-user VAPT-AI, but track for future)
    decided_by_user_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), nullable=True)

    # Result reference (FK to whichever table the approved action created)
    # e.g. if approved → C2Task created, this points to vapt_c2_tasks.id
    result_table: Mapped[str | None] = mapped_column(String(64), nullable=True)
    result_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), nullable=True)