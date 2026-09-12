"""
VAPT-AI Chain of Custody Verifier — D23.

Verifies HMAC-SHA256 tamper seals on Evidence rows. Each evidence links to
the previous one via prev_seal — creating a blockchain-style chain.

Tampering with ANY evidence row breaks the chain (detectable by recomputing).

W2-B already has compute_custody_seal() + verify_custody_seal() in service.py.
W4-A adds:
    - CustodyVerifier class (higher-level API)
    - Graceful verification (returns result dict instead of raising)
    - Full chain verification for a finding
    - Endpoint GET /api/evidence/{id}/verify

Usage:
    from app.evidence.custody import CustodyVerifier
    from app.db.session import async_session

    async with async_session() as session:
        verifier = CustodyVerifier(session)
        result = await verifier.verify_evidence(evidence_id)
        if result["verified"]:
            print("Evidence intact — no tampering detected")
        else:
            print(f"TAMPER DETECTED: {result['reason']}")

        # Verify entire chain for a finding
        chain_result = await verifier.verify_finding_chain(finding_id)
        print(f"Chain: {chain_result['evidence_count']} evidence, verified={chain_result['verified']}")
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models.pentest import Evidence
from app.evidence.service import compute_custody_seal, verify_custody_seal

logger = logging.getLogger(__name__)


class CustodyVerifier:
    """Verify HMAC-SHA256 custody seals on evidence rows.

    D23: Chain of Custody with tamper detection.
    Every Evidence read MUST verify custody seal before returning data.
    """

    def __init__(self, session: AsyncSession):
        self.session = session

    async def verify_evidence(self, evidence_id: uuid.UUID) -> dict[str, Any]:
        """Verify a single evidence row's custody seal.

        Returns:
            {
                "evidence_id": str,
                "verified": bool,
                "reason": str (if not verified),
                "evidence_hash": str,
                "custody_seal": str,
                "layer": str,
                "tool_used": str,
                "captured_at": str,
            }
        """
        # Fetch evidence
        result = await self.session.execute(
            select(Evidence).where(Evidence.id == evidence_id)
        )
        evidence = result.scalar_one_or_none()

        if evidence is None:
            return {
                "evidence_id": str(evidence_id),
                "verified": False,
                "reason": "Evidence not found",
            }

        # Get previous evidence's seal (for chain verification)
        prev_result = await self.session.execute(
            select(Evidence)
            .where(Evidence.finding_id == evidence.finding_id)
            .where(Evidence.captured_at < evidence.captured_at)
            .order_by(Evidence.captured_at.desc())
            .limit(1)
        )
        prev_evidence = prev_result.scalar_one_or_none()
        prev_seal = prev_evidence.custody_seal if prev_evidence else None

        # Verify seal
        is_verified = verify_custody_seal(
            raw_output=evidence.raw_output,
            evidence_id=evidence.id,
            captured_at=evidence.captured_at,
            custody_seal=evidence.custody_seal,
            prev_seal=prev_seal,
        )

        return {
            "evidence_id": str(evidence.id),
            "verified": is_verified,
            "reason": "" if is_verified else "Custody seal mismatch — possible tampering",
            "evidence_hash": evidence.evidence_hash,
            "custody_seal": evidence.custody_seal[:16] + "...",
            "layer": evidence.layer,
            "tool_used": evidence.tool_used,
            "captured_at": evidence.captured_at.isoformat(),
            "prev_seal": (prev_seal[:16] + "...") if prev_seal else None,
        }

    async def verify_finding_chain(self, finding_id: uuid.UUID) -> dict[str, Any]:
        """Verify the full custody seal chain for a finding's evidence.

        Walks all evidence for a finding in chronological order, verifying
        each seal links to the previous one. Any break = tampering detected.

        Returns:
            {
                "finding_id": str,
                "verified": bool,
                "evidence_count": int,
                "broken_at": str | None (evidence_id where chain broke),
                "evidence": [list of per-evidence verification results],
            }
        """
        # Fetch all evidence for finding, ordered by time
        result = await self.session.execute(
            select(Evidence)
            .where(Evidence.finding_id == finding_id)
            .order_by(Evidence.captured_at.asc())
        )
        evidence_list = list(result.scalars().all())

        if not evidence_list:
            return {
                "finding_id": str(finding_id),
                "verified": True,
                "evidence_count": 0,
                "broken_at": None,
                "evidence": [],
            }

        prev_seal = None
        evidence_results = []
        verified = True
        broken_at = None

        for ev in evidence_list:
            expected_seal = compute_custody_seal(
                raw_output=ev.raw_output,
                evidence_id=ev.id,
                captured_at=ev.captured_at,
                prev_seal=prev_seal,
            )

            is_ok = expected_seal == ev.custody_seal
            evidence_results.append({
                "evidence_id": str(ev.id),
                "layer": ev.layer,
                "tool_used": ev.tool_used,
                "verified": is_ok,
                "captured_at": ev.captured_at.isoformat(),
            })

            if not is_ok and verified:
                verified = False
                broken_at = str(ev.id)
                logger.error("CUSTODY CHAIN BROKEN at evidence %s (finding %s)",
                             ev.id, finding_id)

            prev_seal = ev.custody_seal

        return {
            "finding_id": str(finding_id),
            "verified": verified,
            "evidence_count": len(evidence_list),
            "broken_at": broken_at,
            "evidence": evidence_results,
        }

    async def verify_scan_chain(self, scan_id: str) -> dict[str, Any]:
        """Verify custody chains for ALL findings in a scan.

        Returns summary:
            {
                "scan_id": str,
                "verified": bool (True only if ALL findings verified),
                "findings_checked": int,
                "findings_verified": int,
                "findings_failed": int,
                "total_evidence": int,
                "details": [list of per-finding results],
            }
        """
        # Get all findings for scan
        from app.db.models.pentest import Finding
        result = await self.session.execute(
            select(Finding.id).where(Finding.scan_id == scan_id)
        )
        finding_ids = [row[0] for row in result.all()]

        if not finding_ids:
            return {
                "scan_id": scan_id,
                "verified": True,
                "findings_checked": 0,
                "findings_verified": 0,
                "findings_failed": 0,
                "total_evidence": 0,
                "details": [],
            }

        all_verified = True
        total_evidence = 0
        details = []

        for fid in finding_ids:
            chain = await self.verify_finding_chain(fid)
            total_evidence += chain["evidence_count"]
            details.append({
                "finding_id": chain["finding_id"],
                "verified": chain["verified"],
                "evidence_count": chain["evidence_count"],
                "broken_at": chain["broken_at"],
            })
            if not chain["verified"]:
                all_verified = False

        return {
            "scan_id": scan_id,
            "verified": all_verified,
            "findings_checked": len(finding_ids),
            "findings_verified": sum(1 for d in details if d["verified"]),
            "findings_failed": sum(1 for d in details if not d["verified"]),
            "total_evidence": total_evidence,
            "details": details,
        }
