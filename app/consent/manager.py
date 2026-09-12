"""
VAPT-AI Consent Form Manager — D19.

D19: Scope authorization + signed consent form MANDATORY before any active scan.
Even internal use needs consent trail.

Consent flow:
    1. User submits target + asserted_owner + declared_scope (hosts/IP ranges/ports)
    2. User accepts ToS
    3. System generates verification token: vapt_<32hex>_<userid>_<expiry>
    4. User verifies ownership via 1 of 3 methods:
       a. DNS TXT: add vapt-ai-verify=<token> TXT record
       b. HTTP meta-tag: add <meta name="vapt-ai-verify" content="<token>"> to root URL
       c. File upload: upload vapt-ai-<token>.txt to web root
    5. System verifies → ConsentForm.verified = True
    6. Scan can proceed (scope guard uses declared_scope_json)

Re-authorization cadence:
    - First scan of new target: ✅ Mandatory
    - Subsequent scan within 30 days: ❌ Cached
    - After 30 days: ✅ Re-verify
    - DNS records change: ✅ Re-verify (weekly background check)

Usage:
    from app.consent.manager import ConsentManager
    from app.db.session import async_session

    async with async_session() as session:
        mgr = ConsentManager(session)
        consent = await mgr.create_consent(
            scan_id="scan_abc123",
            asserted_owner="John Doe <john@example.com>",
            declared_scope={"hosts": ["example.com", "*.example.com"], "cidrs": ["192.168.1.0/24"]},
            verification_method="dns_txt",
        )
        await session.commit()
        # User adds DNS TXT record, then:
        verified = await mgr.verify_consent(consent.id)
        await session.commit()
"""
from __future__ import annotations

import hashlib
import logging
import secrets
import uuid
from datetime import datetime, timedelta, UTC
from typing import Any

import httpx
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.db.models.scan import ConsentForm
from app.sandbox.scope_guard import ScopeGuard, ScopeRule

logger = logging.getLogger(__name__)


# ---------- Constants ----------

VERIFICATION_TOKEN_TTL_HOURS = 24  # token valid 24 hours
VERIFICATION_METHODS = ("dns_txt", "http_meta", "file_upload", "owned_vps", "htb_machine", "thm_room")


# ---------- Consent Manager ----------

class ConsentManager:
    """Consent form + scope authorization manager.

    D19: Mandatory consent before any active scan.
    """

    def __init__(self, session: AsyncSession):
        self.session = session

    async def create_consent(
        self,
        scan_id: str,
        asserted_owner: str,
        declared_scope: dict[str, Any],
        verification_method: str,
        asserted_owner_role: str = "owner",
    ) -> ConsentForm:
        """Create a new consent form (pending verification).

        Args:
            scan_id: FK to vapt_scans.id
            asserted_owner: Name/email/org of owner (e.g. "John Doe <john@example.com>")
            declared_scope: {hosts: [...], cidrs: [...], ports: [...], exclusions: [...]}
            verification_method: One of VERIFICATION_METHODS
            asserted_owner_role: "owner" or "authorized_rep"

        Returns:
            ConsentForm ORM instance (verified=False, pending verification)
        """
        if verification_method not in VERIFICATION_METHODS:
            raise ValueError(f"Invalid verification_method: {verification_method}. Must be one of {VERIFICATION_METHODS}")

        # Generate verification token: vapt_<32hex>_<userid>_<expiry>
        token_hex = secrets.token_hex(16)  # 32 hex chars
        expiry = int((datetime.now(UTC) + timedelta(hours=VERIFICATION_TOKEN_TTL_HOURS)).timestamp())
        # Use a placeholder user_id=0 for now (single-user system)
        token = f"vapt_{token_hex}_0_{expiry}"

        consent = ConsentForm(
            scan_id=scan_id,
            asserted_owner=asserted_owner,
            asserted_owner_role=asserted_owner_role,
            declared_scope_json=declared_scope,
            tos_accepted=False,
            verification_method=verification_method,
            verification_token=token,
            verified=False,
        )
        self.session.add(consent)
        await self.session.flush()

        logger.info("Consent created: scan=%s method=%s token=%s...",
                     scan_id, verification_method, token[:20])
        return consent

    async def accept_tos(self, consent_id: uuid.UUID) -> ConsentForm | None:
        """Mark ToS as accepted."""
        consent = await self._get_consent(consent_id)
        if consent is None:
            return None
        consent.tos_accepted = True
        consent.tos_accepted_at = datetime.now(UTC)
        await self.session.flush()
        return consent

    async def verify_consent(self, consent_id: uuid.UUID) -> dict[str, Any]:
        """Verify consent ownership via the selected method.

        Returns:
            {
                "consent_id": str,
                "verified": bool,
                "method": str,
                "reason": str,
            }
        """
        consent = await self._get_consent(consent_id)
        if consent is None:
            return {"consent_id": str(consent_id), "verified": False, "reason": "Consent not found"}

        if not consent.tos_accepted:
            return {"consent_id": str(consent_id), "verified": False, "reason": "ToS not accepted"}

        if consent.verified:
            return {"consent_id": str(consent_id), "verified": True, "method": consent.verification_method, "reason": "Already verified"}

        # Verify based on method
        method = consent.verification_method
        token = consent.verification_token or ""

        if method in ("owned_vps", "htb_machine", "thm_room"):
            # These methods don't need external verification — user asserts ownership
            consent.verified = True
            consent.verified_at = datetime.now(UTC)
            consent.verification_response = f"Auto-verified (method={method})"
            await self.session.flush()
            return {"consent_id": str(consent_id), "verified": True, "method": method, "reason": "Auto-verified (owned/rented target)"}

        # For dns_txt / http_meta / file_upload — need target URL to verify
        # W4-C: stub verification (always succeeds for now)
        # W8: implement actual DNS TXT lookup + HTTP meta-tag check + file upload check
        consent.verified = True
        consent.verified_at = datetime.now(UTC)
        consent.verification_response = f"[W4-C stub] Verification method={method} token={token[:20]}... — actual verification is W8 task"
        await self.session.flush()

        logger.info("Consent verified: id=%s method=%s", consent_id, method)
        return {"consent_id": str(consent_id), "verified": True, "method": method, "reason": "Verified (W4-C stub — actual verification W8)"}

    async def is_scan_authorized(self, scan_id: str) -> bool:
        """Check if a scan has valid verified consent.

        A scan is authorized if:
            1. Has a ConsentForm
            2. ConsentForm.verified == True
            3. ToS accepted
            4. Verification not expired (within 30 days)
        """
        result = await self.session.execute(
            select(ConsentForm)
            .where(ConsentForm.scan_id == scan_id)
            .order_by(ConsentForm.created_at.desc())
            .limit(1)
        )
        consent = result.scalar_one_or_none()
        if consent is None:
            return False

        if not consent.verified or not consent.tos_accepted:
            return False

        # Check 30-day re-authorization window
        if consent.verified_at:
            age = datetime.now(UTC) - consent.verified_at
            if age > timedelta(days=30):
                logger.warning("Consent expired (>%d days): scan=%s", 30, scan_id)
                return False

        return True

    def build_scope_guard(self, consent: ConsentForm) -> ScopeGuard:
        """Build a ScopeGuard from a consent form's declared scope.

        Converts declared_scope_json into ScopeRule list:
            {"hosts": ["example.com", "*.example.com"], "cidrs": ["192.168.1.0/24"]}
            → [ScopeRule(host="example.com"), ScopeRule(host="*.example.com"), ScopeRule(cidr="192.168.1.0/24")]
        """
        scope_data = consent.declared_scope_json or {}
        rules: list[ScopeRule] = []

        for host in scope_data.get("hosts", []):
            rules.append(ScopeRule(host=host))

        for cidr in scope_data.get("cidrs", []):
            try:
                rules.append(ScopeRule(cidr=cidr))
            except ValueError as e:
                logger.warning("Invalid CIDR in scope: %s — %s", cidr, e)

        for ip in scope_data.get("ips", []):
            rules.append(ScopeRule(host=ip))

        return ScopeGuard(declared_scope=rules)

    async def get_consent_for_scan(self, scan_id: str) -> ConsentForm | None:
        """Get the most recent consent form for a scan."""
        result = await self.session.execute(
            select(ConsentForm)
            .where(ConsentForm.scan_id == scan_id)
            .order_by(ConsentForm.created_at.desc())
            .limit(1)
        )
        return result.scalar_one_or_none()

    async def _get_consent(self, consent_id: uuid.UUID) -> ConsentForm | None:
        """Get a consent form by ID."""
        result = await self.session.execute(
            select(ConsentForm).where(ConsentForm.id == consent_id)
        )
        return result.scalar_one_or_none()
