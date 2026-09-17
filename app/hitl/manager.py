"""
VAPT-AI HITL (Human-in-the-Loop) middleware — D25.

W7-B: Refactored to support 3 modes (Option C — A-in-the-Loop):
    - audit_agent (default, MVP) — LLM critic reviews destructive ops, decides
      approve/reject/suggest_alternative in ~2-3s. No human blocking.
    - human_block (deferred)    — blocking channel waits for human decision via
      SSE event + UI modal. NotImplementedError in W7-B; will be implemented
      only if user switches to this mode later.
    - auto_approve (debug only) — skip review entirely, approve immediately.

D25 destructive op classification:
    Read-only (nmap default, nuclei passive) — ❌ No HITL
    Active probe (sqlmap detect, ZAP active) — ❌ No HITL
    Exploit attempt (metasploit exploit, sqlmap --os-shell, custom RCE) — ✅ HITL required
    C2 task L3+ (file access, change exec, destructive) — ✅ HITL required
    Payload gen (msfvenom, custom C2 payload) — ✅ HITL required

Main entry point:
    async def request_and_wait(
        scan_id, tool_name, target, args, predicted_impact,
        agent_reasoning, kg_confidence, scan_context,
    ) -> HITLDecision

    Returns a HITLDecision with one of:
        decision ∈ {approve, reject, suggest_alternative, user_aborted, approval_timeout}
        decided_by ∈ {audit_agent, audit_agent_fallback, auto, human, timeout, NotImplementedError-pending}

Usage (from agent / SubprocessExecutor):
    from app.hitl.manager import HITLManager
    from app.db.session import async_session

    async with async_session() as session:
        mgr = HITLManager(session, mode="audit_agent")
        decision = await mgr.request_and_wait(
            scan_id="scan_abc123",
            tool_name="metasploit",
            target="http://target.com",
            args={"module": "exploit/multi/handler"},
            predicted_impact="destructive",
            agent_reasoning="Exploiting RCE to get shell",
            kg_confidence=0.85,
            scan_context={
                "declared_scope": ["10.10.10.0/24"],
                "blackboard_facts": [...],
                "destructive_ops_count": 1,
                "max_destructive_ops": 5,
            },
        )
        await session.commit()
        if decision.decision == "approve":
            # proceed with original args
        elif decision.decision == "reject":
            # skip tool; feed decision.comment back to agent
        elif decision.decision == "suggest_alternative":
            # use decision.suggested_args instead
"""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, UTC
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.db.models.hitl import HITLApproval
from app.pentest.events import emit_hitl_required, emit_hitl_decision

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

HITL_TIMEOUT_SECONDS = 300  # only used in human_block mode


# ---------- HITL modes ----------

VALID_MODES = {"audit_agent", "human_block", "auto_approve"}
DEFAULT_MODE = "audit_agent"


def _resolve_mode(explicit: str | None = None) -> str:
    """Resolve the HITL mode.

    Priority:
        1. explicit param (caller can override per-scan)
        2. settings / env var VAPT_AI_HITL_MODE
        3. DEFAULT_MODE (audit_agent)
    """
    if explicit and explicit in VALID_MODES:
        return explicit
    # Lazy import to avoid circular dependency at module load time
    from app.core.config import settings as _settings
    env_mode = getattr(_settings, "hitl_mode", None)
    if env_mode and env_mode in VALID_MODES:
        return env_mode
    return DEFAULT_MODE


# ---------- HITLDecision dataclass ----------

@dataclass
class HITLDecision:
    """Outcome of a HITL review.

    Fields:
        decision: approve | reject | suggest_alternative | user_aborted | approval_timeout
        comment: human-readable rationale (always prefixed with 'audit agent:' for audit_agent mode)
        suggested_args: dict | None — only set when decision == 'suggest_alternative'
        risk_assessment: str | None — LLM's risk classification (low/medium/high)
        confidence: float — 0.0-1.0 (LLM's self-reported confidence; 0.0 for fallback)
        decided_by: audit_agent | audit_agent_fallback | auto | human | timeout
        duration_seconds: float — wall-clock time of the review
        approval_id: uuid.UUID | None — the vapt_hitl_approvals row this decision applies to
        raw_llm_response: str | None — debug only; never expose to user
    """

    decision: str
    comment: str
    suggested_args: dict[str, Any] | None = None
    risk_assessment: str | None = None
    confidence: float = 0.0
    decided_by: str = "audit_agent"
    duration_seconds: float = 0.0
    approval_id: uuid.UUID | None = None
    raw_llm_response: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision,
            "comment": self.comment,
            "suggested_args": self.suggested_args,
            "risk_assessment": self.risk_assessment,
            "confidence": self.confidence,
            "decided_by": self.decided_by,
            "duration_seconds": self.duration_seconds,
            "approval_id": str(self.approval_id) if self.approval_id else None,
        }


# ---------- HITLManager ----------

class HITLManager:
    """HITL approval gate — reviews destructive ops per the configured mode.

    W7-B: full 3-mode implementation (audit_agent / human_block / auto_approve).
    human_block mode raises NotImplementedError for the blocking wait; all
    other code paths are production-ready.
    """

    def __init__(
        self,
        session: AsyncSession,
        mode: str | None = None,
        audit_agent: Any = None,  # inject for tests; lazy-load in production
    ):
        self.session = session
        self.mode = _resolve_mode(mode)
        self._audit_agent = audit_agent  # lazy init via get_audit_agent() on first use

    # ---------- Audit agent access ----------

    def _get_audit_agent(self):
        """Lazy-load the AuditAgent singleton (or use injected instance for tests)."""
        if self._audit_agent is not None:
            return self._audit_agent
        from app.hitl.audit_agent import get_audit_agent
        self._audit_agent = get_audit_agent()
        return self._audit_agent

    # ---------- Destructive op classification ----------

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

    # ---------- Main entry: request_and_wait ----------

    async def request_and_wait(
        self,
        scan_id: str,
        tool_name: str,
        target: str,
        args: dict[str, Any],
        predicted_impact: str,
        agent_reasoning: str,
        kg_confidence: float = 0.5,
        scan_context: dict[str, Any] | None = None,
    ) -> HITLDecision:
        """Request HITL review and wait for a decision (mode-dependent).

        This is the main entry point for W7-B. It:
            1. Creates a HITLApproval row (status=pending)
            2. Emits SSE event 'hitl_approval_required'
            3. Branches by self.mode:
               - auto_approve    → return approve immediately
               - audit_agent     → call AuditAgent.review() (2-3s LLM call)
               - human_block     → NotImplementedError (deferred)
            4. Updates HITLApproval row with decision
            5. Emits SSE event 'hitl_decision_made'
            6. Returns HITLDecision

        Never raises — always returns a HITLDecision (even on failure, falls
        back to reject with a clear comment).
        """
        from time import time
        start = time()

        # 1. Create pending approval row
        approval = await self._create_pending(
            scan_id=scan_id,
            tool_name=tool_name,
            target=target,
            args=args,
            predicted_impact=predicted_impact,
            agent_reasoning=agent_reasoning,
            kg_confidence=kg_confidence,
        )
        await self.session.flush()

        # 2. Emit SSE event: approval required (for UI feed)
        await emit_hitl_required(
            scan_id=scan_id,
            hitl_id=str(approval.id),
            tool_name=tool_name,
            target=target,
            predicted_impact=predicted_impact,
        )

        # 3. Branch by mode
        try:
            if self.mode == "auto_approve":
                decision = HITLDecision(
                    decision="approve",
                    comment="audit agent: auto-approved (debug mode — VAPT_AI_HITL_MODE=auto_approve)",
                    decided_by="auto",
                    approval_id=approval.id,
                )

            elif self.mode == "human_block":
                # W7-B: defer blocking channel implementation.
                # When user switches to this mode, implement asyncio.Event
                # + 5-min timeout + SSE 'hitl_decision_required' (UI modal).
                decision = await self._wait_for_human_decision(approval)

            else:  # audit_agent (default)
                audit = self._get_audit_agent()
                audit_decision = await audit.review(
                    tool_name=tool_name,
                    target=target,
                    args=args,
                    predicted_impact=predicted_impact,
                    agent_reasoning=agent_reasoning,
                    scan_context=scan_context,
                )
                decision = HITLDecision(
                    decision=audit_decision.decision,
                    comment=audit_decision.comment,
                    suggested_args=audit_decision.suggested_args,
                    risk_assessment=audit_decision.risk_assessment,
                    confidence=audit_decision.confidence,
                    decided_by=audit_decision.decided_by,
                    approval_id=approval.id,
                    raw_llm_response=audit_decision.raw_llm_response,
                )

        except NotImplementedError as exc:
            # human_block mode not yet implemented — mark as timeout + return
            decision = HITLDecision(
                decision="approval_timeout",
                comment=f"audit agent: human_block mode not implemented ({exc})",
                decided_by="timeout",
                approval_id=approval.id,
            )
        except Exception as exc:
            # Any unexpected error — fail safe (reject)
            logger.exception("HITL review failed unexpectedly (tool=%s)", tool_name)
            decision = HITLDecision(
                decision="reject",
                comment=f"audit agent: review failed ({type(exc).__name__}: {exc})",
                decided_by="audit_agent_fallback",
                approval_id=approval.id,
            )

        decision.duration_seconds = time() - start

        # 4. Update HITLApproval row with decision
        await self._apply_decision(approval, decision)
        await self.session.flush()

        # 5. Emit SSE event: decision made
        await emit_hitl_decision(
            scan_id=scan_id,
            hitl_id=str(approval.id),
            tool_name=tool_name,
            target=target,
            decision=decision.decision,
            decided_by=decision.decided_by,
            comment=decision.comment,
            suggested_args=decision.suggested_args,
        )

        logger.info(
            "HITL decision: scan=%s tool=%s decision=%s decided_by=%s duration=%.2fs",
            scan_id, tool_name, decision.decision, decision.decided_by,
            decision.duration_seconds,
        )

        return decision

    # ---------- Internal: create pending approval row ----------

    async def _create_pending(
        self,
        scan_id: str,
        tool_name: str,
        target: str,
        args: dict[str, Any],
        predicted_impact: str,
        agent_reasoning: str,
        kg_confidence: float,
    ) -> HITLApproval:
        """Insert a HITLApproval row with status=pending + 5-min expiry."""
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
        logger.info(
            "HITL approval requested: scan=%s tool=%s target=%s impact=%s id=%s",
            scan_id, tool_name, target, predicted_impact, approval.id,
        )
        return approval

    # ---------- Internal: apply decision to approval row ----------

    async def _apply_decision(self, approval: HITLApproval, decision: HITLDecision) -> None:
        """Update the HITLApproval row with the decision outcome."""
        now = datetime.now(UTC)
        approval.decided_at = now

        # Map HITLDecision.decision → HITLApproval.status
        status_map = {
            "approve": "approved",
            "reject": "aborted",  # rejected by audit_agent → status=aborted (agent aborted by gate)
            "suggest_alternative": "approved",  # approved with edited args
            "user_aborted": "aborted",
            "approval_timeout": "timeout",
        }
        approval.status = status_map.get(decision.decision, "aborted")

        # Map user_decision field
        user_decision_map = {
            "approve": "approve",
            "reject": "abort",
            "suggest_alternative": "approve_with_edit",
            "user_aborted": "abort",
            "approval_timeout": "timeout",
        }
        approval.user_decision = user_decision_map.get(decision.decision, "abort")

        # Edited args (if suggest_alternative)
        if decision.suggested_args:
            approval.edited_args_json = decision.suggested_args

        # Store the audit comment in agent_reasoning field? No — that's the
        # agent's original reasoning. The decision comment is logged via SSE
        # + AuditLog (W7-C will wire AuditLogger for full D23 chain).

    # ---------- Internal: human_block mode (deferred) ----------

    async def _wait_for_human_decision(self, approval: HITLApproval) -> HITLDecision:
        """Block until human decides or 5-min timeout.

        W7-B: NOT YET IMPLEMENTED.
        When user switches to human_block mode, implement:
            1. Register asyncio.Event in app.pentest.scan_registry
            2. Emit SSE 'hitl_decision_required' (UI shows modal)
            3. await asyncio.wait_for(event.wait(), timeout=300)
            4. On timeout → return approval_timeout
            5. On signal → fetch decision from approval row + return
        """
        raise NotImplementedError(
            "human_block mode is not yet implemented in W7-B. "
            "Default mode is 'audit_agent' (Option C — A-in-the-Loop). "
            "To enable human_block, implement _wait_for_human_decision() "
            "in app/hitl/manager.py + add UI modal in frontend."
        )

    # ---------- Legacy: request_approval (W3-C API — kept for backward compat) ----------

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
        """[DEPRECATED — W3-C skeleton] Create a HITL approval request.

        Kept for backward compat with W6-D Metasploit tools that call this
        directly. W7-C will refactor Metasploit tools to use request_and_wait()
        instead. New code should use request_and_wait().
        """
        approval = await self._create_pending(
            scan_id=scan_id,
            tool_name=tool_name,
            target=target,
            args=args,
            predicted_impact=predicted_impact,
            agent_reasoning=agent_reasoning,
            kg_confidence=kg_confidence,
        )
        await emit_hitl_required(
            scan_id=scan_id,
            hitl_id=str(approval.id),
            tool_name=tool_name,
            target=target,
            predicted_impact=predicted_impact,
        )
        return approval

    # ---------- CRUD helpers (kept from W3-C) ----------

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
        """[LEGACY — human_block mode] Approve a HITL request.

        Used by HTTP endpoint POST /api/hitl/{id}/approve.
        Only meaningful when mode=human_block (W7-C will wire panic button +
        human_block UI). In audit_agent mode, this is a no-op (decision is
        already made by AuditAgent).
        """
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
        logger.info("HITL approved (legacy): id=%s user=%s", approval_id, user_id)
        return approval

    async def abort(
        self,
        approval_id: uuid.UUID,
        user_id: uuid.UUID | None = None,
    ) -> HITLApproval | None:
        """[LEGACY — human_block mode / panic button] Abort a HITL request."""
        approval = await self.get_approval(approval_id)
        if approval is None or approval.status != "pending":
            return None

        approval.status = "aborted"
        approval.user_decision = "abort"
        approval.decided_at = datetime.now(UTC)
        approval.decided_by_user_id = user_id

        await self.session.flush()
        logger.info("HITL aborted (legacy): id=%s user=%s", approval_id, user_id)
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
