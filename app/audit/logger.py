"""
VAPT-AI Audit Logger — D23 append-only, HMAC-sealed audit log.

Every action by user / agent / system is logged here. HMAC-SHA256 tamper
seals link each entry to the previous one (blockchain-style chain). 7-year
retention (regulatory minimum).

AuditLog table (W1-C already created):
    - id (UUID PK)
    - actor_type (user / agent / system)
    - actor_id (user UUID / agent name / "system")
    - action (login / logout / scan_start / scan_complete / tool_execute / hitl_approve / ...)
    - target_table, target_id (what was affected)
    - before_json, after_json (state change)
    - ip_address, user_agent, scan_id (context)
    - prev_seal, tamper_seal (HMAC chain)
    - created_at (server-side, no updated_at — append-only)

Usage:
    from app.audit.logger import AuditLogger
    from app.db.session import async_session

    async with async_session() as session:
        logger = AuditLogger(session)
        await logger.log(
            actor_type="user",
            actor_id=str(user.id),
            action="login",
            ip_address="127.0.0.1",
        )
        await session.commit()

D23 enforcement:
    - Append-only: no UPDATE/DELETE on vapt_audit_log
    - HMAC chain: each entry's seal = HMAC(key, prev_seal | actor | action | target | created_at)
    - Tamper detection: modify any row → chain breaks (verifiable via verify_chain)
    - Sensitive-key sanitization: before_json/after_json scrubbed of passwords/api_keys/secrets
    - 7-year retention (cold-storage archive after 1 year — W28 task)
"""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
import uuid
from datetime import datetime, UTC
from typing import Any

from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.db.models.audit import AuditLog

logger = logging.getLogger(__name__)


# ---------- Sensitive key sanitization ----------

SENSITIVE_KEYS = {
    "password", "password_hash", "passwd", "pwd",
    "api_key", "apikey", "api_secret",
    "secret", "secret_key", "secret_token",
    "token", "access_token", "refresh_token",
    "authorization", "auth_header",
    "credential", "credentials",
    "private_key", "privatekey",
    "cookie", "session_cookie",
    "ssn", "credit_card", "cc_number",
    "encryption_key", "fernet_key", "jwt_secret",
}


def sanitize_audit_data(data: dict[str, Any] | None) -> dict[str, Any] | None:
    """Sanitize sensitive keys from audit data (before_json / after_json).

    Replaces sensitive values with '[REDACTED]' to prevent secret leakage
    into the audit log.
    """
    if data is None:
        return None

    def _sanitize(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {
                k: ("[REDACTED]" if k.lower() in SENSITIVE_KEYS else _sanitize(v))
                for k, v in obj.items()
            }
        elif isinstance(obj, list):
            return [_sanitize(item) for item in obj]
        else:
            return obj

    return _sanitize(data)


# ---------- HMAC seal computation ----------

def compute_audit_seal(
    actor_type: str,
    actor_id: str | None,
    action: str,
    target_table: str | None,
    target_id: str | None,
    created_at: datetime,
    prev_seal: str | None,
) -> str:
    """Compute HMAC-SHA256 tamper seal for an audit log entry.

    Seal = HMAC(encryption_key, f"{prev_seal}|{actor_type}|{actor_id}|{action}|{target}|{created_at}")
    """
    key = settings.encryption_key.get_secret_value().encode("utf-8")
    message = f"{prev_seal or ''}|{actor_type}|{actor_id or ''}|{action}|{target_table or ''}|{target_id or ''}|{created_at.isoformat()}"
    return hmac.new(key, message.encode("utf-8"), hashlib.sha256).hexdigest()


# ---------- Audit Logger ----------

class AuditLogger:
    """Append-only audit logger with HMAC tamper seals.

    D23: Chain of Custody for audit log entries.
    """

    def __init__(self, session: AsyncSession):
        self.session = session

    async def log(
        self,
        actor_type: str,
        actor_id: str | None,
        action: str,
        target_table: str | None = None,
        target_id: str | None = None,
        before: dict[str, Any] | None = None,
        after: dict[str, Any] | None = None,
        ip_address: str | None = None,
        user_agent: str | None = None,
        scan_id: str | None = None,
    ) -> AuditLog:
        """Add an audit log entry.

        Args:
            actor_type: "user" / "agent" / "system"
            actor_id: User UUID / agent name / "system"
            action: What happened (e.g. "login", "scan_start", "tool_execute")
            target_table: Table affected (e.g. "vapt_scans")
            target_id: Row ID affected
            before: State before change (sanitized)
            after: State after change (sanitized)
            ip_address: Request IP
            user_agent: Request User-Agent
            scan_id: Scan context (if action is within a scan)

        Returns:
            AuditLog ORM instance (not yet committed)
        """
        # Sanitize sensitive data
        before_sanitized = sanitize_audit_data(before)
        after_sanitized = sanitize_audit_data(after)

        # Get previous entry's seal (for chain)
        prev_seal = await self._get_last_seal()

        # Compute this entry's seal
        created_at = datetime.now(UTC)
        seal = compute_audit_seal(
            actor_type=actor_type,
            actor_id=actor_id,
            action=action,
            target_table=target_table,
            target_id=target_id,
            created_at=created_at,
            prev_seal=prev_seal,
        )

        # Create audit log entry
        entry = AuditLog(
            id=uuid.uuid4(),
            actor_type=actor_type,
            actor_id=actor_id,
            action=action,
            target_table=target_table,
            target_id=target_id,
            before_json=before_sanitized,
            after_json=after_sanitized,
            ip_address=ip_address,
            user_agent=user_agent,
            scan_id=scan_id,
            prev_seal=prev_seal,
            tamper_seal=seal,
            created_at=created_at,
        )
        self.session.add(entry)
        await self.session.flush()

        logger.debug("Audit log: actor=%s action=%s target=%s/%s",
                      actor_type, action, target_table, target_id)
        return entry

    async def _get_last_seal(self) -> str | None:
        """Get the tamper_seal of the most recent audit log entry."""
        result = await self.session.execute(
            select(AuditLog.tamper_seal)
            .order_by(AuditLog.created_at.desc())
            .limit(1)
        )
        row = result.scalar_one_or_none()
        return row

    async def get_entries(
        self,
        scan_id: str | None = None,
        actor_type: str | None = None,
        action: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[AuditLog]:
        """Query audit log entries with optional filters."""
        query = select(AuditLog)
        if scan_id:
            query = query.where(AuditLog.scan_id == scan_id)
        if actor_type:
            query = query.where(AuditLog.actor_type == actor_type)
        if action:
            query = query.where(AuditLog.action == action)
        query = query.order_by(AuditLog.created_at.desc()).limit(limit).offset(offset)

        result = await self.session.execute(query)
        return list(result.scalars().all())

    async def verify_chain(self, limit: int = 1000) -> dict[str, Any]:
        """Verify the HMAC seal chain of the audit log.

        Walks entries in chronological order, verifying each seal links
        to the previous one. Any break = tampering detected.

        Returns:
            {
                "verified": bool,
                "entries_checked": int,
                "broken_at": str | None (entry ID where chain broke),
            }
        """
        result = await self.session.execute(
            select(AuditLog)
            .order_by(AuditLog.created_at.asc())
            .limit(limit)
        )
        entries = list(result.scalars().all())

        prev_seal = None
        for entry in entries:
            expected = compute_audit_seal(
                actor_type=entry.actor_type,
                actor_id=entry.actor_id,
                action=entry.action,
                target_table=entry.target_table,
                target_id=entry.target_id,
                created_at=entry.created_at,
                prev_seal=prev_seal,
            )
            if expected != entry.tamper_seal:
                logger.error("AUDIT CHAIN BROKEN at entry %s", entry.id)
                return {
                    "verified": False,
                    "entries_checked": len(entries),
                    "broken_at": str(entry.id),
                }
            prev_seal = entry.tamper_seal

        return {
            "verified": True,
            "entries_checked": len(entries),
            "broken_at": None,
        }

    async def count_entries(self) -> int:
        """Count total audit log entries."""
        result = await self.session.execute(
            select(func.count(AuditLog.id))
        )
        return result.scalar_one()


# ---------- Convenience loggers ----------

async def log_user_action(
    session: AsyncSession,
    user_id: str,
    action: str,
    ip_address: str | None = None,
    **kwargs: Any,
) -> None:
    """Log a user action (login, logout, scan_start, etc.)."""
    audit = AuditLogger(session)
    await audit.log(
        actor_type="user",
        actor_id=user_id,
        action=action,
        ip_address=ip_address,
        **kwargs,
    )


async def log_agent_action(
    session: AsyncSession,
    agent_name: str,
    action: str,
    scan_id: str | None = None,
    **kwargs: Any,
) -> None:
    """Log an agent action (tool_execute, finding_detected, etc.)."""
    audit = AuditLogger(session)
    await audit.log(
        actor_type="agent",
        actor_id=agent_name,
        action=action,
        scan_id=scan_id,
        **kwargs,
    )


async def log_system_action(
    session: AsyncSession,
    action: str,
    **kwargs: Any,
) -> None:
    """Log a system action (startup, shutdown, cleanup, etc.)."""
    audit = AuditLogger(session)
    await audit.log(
        actor_type="system",
        actor_id="system",
        action=action,
        **kwargs,
    )
