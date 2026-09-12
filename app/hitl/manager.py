"""
VAPT-AI HITL (Human-in-the-Loop) middleware — D25.

Intercepts destructive operations + creates HITLApproval row + emits SSE event.
W3-C: skeleton (creates row + event, does NOT block tool execution).
W7: full implementation (blocks until user approves/aborts, 5-min auto-timeout).

D25 destructive op classification:
    Read-only (nmap default, nuclei passive) — ❌ No HITL
    Active probe (sqlmap detect, ZAP active) — ❌ No HITL
    Exploit attempt (metasploit exploit, sqlmap --os-shell, custom RCE) — ✅ HITL required
    C2 task L3+ (file access, change exec, destructive) — ✅ HITL required
    Payload gen (msfvenom, custom C2 payload) — ✅ HITL required

Usage:
    from app.hitl.manager import HITLManager
    from app.db.session import async_session

    async with async_session() as session:
        mgr = HITLManager(session)
        approval = await mgr.request_approval(
            scan_id="scan_abc123",
            tool_name="metasploit",
            target="http://target.com",
            args={"module": "exploit/multi/handler"},
            predicted_impact="destructive",
            agent_reasoning="Exploiting RCE to get shell",
            kg_confidence=0.85,
        )
        await session.commit()
        # approval.status == "pending" — W7 will block here until user decides
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime, timedelta, UTC
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.db.models.hitl import HITLApproval
from app.pentest.events import emit_hitl_required

logger = logging.getLogger(__name__)


# ---------- Destructive op classification ----------

DESTRUCTIVE_TOOLS = {
    "metasploit",
    "sqlmap_exploit",
    "custom_rce",
    "payload_gen",
    "webshell_deploy",
}

SAFE_TOOLS = {
    "nmap", "nuclei", "subfinder", "httpx", "whatweb",
    "nikto", "gobuster", "feroxbuster", "dalfox",
    "sqlmap",
}

HITL_REQUIRED_C2_LEVELS = {3, 4, 5}

HITL_TIMEOUT_SECONDS = 300


class HITLManager:
    """HITL approval gate — creates approval requests for destructive ops.

    W3-C: skeleton — creates HITLApproval row + emits SSE event.
    Does NOT block tool execution (W7 will add blocking).
    """

    def __init__(self, session: AsyncSession):
        self.session = session

    def is_hitl_required(
        self,
        tool_name: str,
        args: dict[str, Any] | None = None,
        c2_task_level: int | None = None,
    ) -> bool:
        """Check if HITL is required for a given tool + args."""
        if c2_task_level is not None and c2_task_level in HITL_REQUIRED_C2_LEVELS:
            return True

        if tool_name in DESTRUCTIVE_TOOLS:
            return True

        if tool_name == "sqlmap" and args:
            args_str = " ".join(str(v) for v in args.values()).lower()
            forbidden = ["--os-shell", "--dump", "--sql-shell", "--os-pwn", "--priv-esc"]
            if any(f in args_str for f in forbidden):
                return True

        return False

    async def request_approval(
        self,
        scan_id: str,
        tool_name: str,
        target: str,
        args: dict[str, Any],
        predicted_impact: str,
        agent_reasoning: str,
        kg_confidence: float = 0.5,
    ) -> HITLApproval:
        """Create a HITL approval request.

        W3-C: creates row + emits SSE event. Does NOT block.
        W7: will block until user approves/aborts/timeout.
        """
        now = datetime.now(UTC)
        expires_at = now + timedelta(seconds=HITL_TIMEOUT_SECONDS)

        approval = HITLApproval(
            scan_id=scan_id,
            tool_name=tool_name,
            target=target,
            args_json=args,
            predicted_impact=predicted_impact,
            agent_reasoning=agent_reasoning,
            kg_confidence=kg_confidence,
            status="pending",
            expires_at=expires_at,
        )
        self.session.add(approval)
        await self.session.flush()

        logger.info("HITL approval requested: scan=%s tool=%s target=%s impact=%s id=%s",
                     scan_id, tool_name, target, predicted_impact, approval.id)

        await emit_hitl_required(
            scan_id=scan_id,
            hitl_id=str(approval.id),
            tool_name=tool_name,
            target=target,
            predicted_impact=predicted_impact,
        )

        return approval

    async def get_approval(self, approval_id: uuid.UUID) -> HITLApproval | None:
        """Get an approval by ID."""
        result = await self.session.execute(
            select(HITLApproval).where(HITLApproval.id == approval_id)
        )
        return result.scalar_one_or_none()

    async def get_pending_approvals(self, scan_id: str) -> list[HITLApproval]:
        """Get all pending approvals for a scan."""
        result = await self.session.execute(
            select(HITLApproval)
            .where(HITLApproval.scan_id == scan_id)
            .where(HITLApproval.status == "pending")
            .order_by(HITLApproval.created_at.asc())
        )
        return list(result.scalars().all())

    async def approve(
        self,
        approval_id: uuid.UUID,
        user_id: uuid.UUID | None = None,
        time_limit_seconds: int | None = None,
        edited_args: dict[str, Any] | None = None,
    ) -> HITLApproval | None:
        """Approve a HITL request."""
        approval = await self.get_approval(approval_id)
        if approval is None or approval.status != "pending":
            return None

        approval.status = "approved" if not time_limit_seconds else "approved_with_time_limit"
        approval.user_decision = "approve" if not time_limit_seconds else "approve_with_time_limit"
        approval.decided_at = datetime.now(UTC)
        approval.decided_by_user_id = user_id
        approval.time_limit_seconds = time_limit_seconds
        approval.edited_args_json = edited_args

        await self.session.flush()
        logger.info("HITL approved: id=%s user=%s", approval_id, user_id)
        return approval

    async def abort(
        self,
        approval_id: uuid.UUID,
        user_id: uuid.UUID | None = None,
    ) -> HITLApproval | None:
        """Abort (reject) a HITL request."""
        approval = await self.get_approval(approval_id)
        if approval is None or approval.status != "pending":
            return None

        approval.status = "aborted"
        approval.user_decision = "abort"
        approval.decided_at = datetime.now(UTC)
        approval.decided_by_user_id = user_id

        await self.session.flush()
        logger.info("HITL aborted: id=%s user=%s", approval_id, user_id)
        return approval

    async def check_timeout(self, approval_id: uuid.UUID) -> HITLApproval | None:
        """Check if a pending approval has timed out. If so, mark as timeout."""
        approval = await self.get_approval(approval_id)
        if approval is None or approval.status != "pending":
            return None

        if approval.expires_at < datetime.now(UTC):
            approval.status = "timeout"
            approval.user_decision = "timeout"
            approval.decided_at = datetime.now(UTC)
            await self.session.flush()
            logger.warning("HITL timed out: id=%s", approval_id)
            return approval

        return None
