"""
VAPT-AI Evidence Service — 5-layer evidence chain (D8) + chain of custody (D23).

5 layers (per plan §5):
    1. detection          — tool detected potential vuln (nmap/nuclei output)
    2. validation         — verifier confirmed (HTTP re-probe / regex match)
    3. exploitation       — PoC executed (sqlmap --os-shell / metasploit exploit)
    4. post_exploitation  — actions on objective (data dump / screenshot)
    5. audit              — audit trail (who/what/when)

Each Evidence row has:
    - raw_output (PII-redacted via app.pii.redactor before persistence)
    - custody_seal (HMAC-SHA256 — tamper detection)
    - evidence_hash (SHA-256 of raw_output — integrity)
    - tool_used (which CLI tool produced this evidence)
    - captured_at (timestamp)
    - spill_path (if raw_output > 50KB, full content stored on disk)

Usage:
    from app.evidence.service import EvidenceService
    from app.db.session import async_session

    async with async_session() as session:
        svc = EvidenceService(session)
        evidence = await svc.add_evidence(
            finding_id=uuid.uuid4(),
            layer="detection",
            raw_output="...",
            tool_used="nmap",
        )
        await session.commit()

CVSS v3.1 (D22):
    Every Finding has cvss_vector + cvss_base_score. The CVSS calculator
    (app/evidence/cvss.py — W2-C task) parses the vector string and computes
    the base score using the official FIRST.org formula.

D17 (failed PoC excluded):
    Evidence with layer="exploitation" where success=False is NOT linked to
    a Finding. The Finding is only created AFTER successful PoC.
"""
from __future__ import annotations

import hashlib
import hmac
import logging
import uuid
from datetime import datetime, UTC
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.db.models.pentest import Evidence, Finding
from app.pii.redactor import redact_pii, audit_redactions

logger = logging.getLogger(__name__)


# ---------- Constants ----------

EVIDENCE_LAYERS = (
    "detection",
    "validation",
    "exploitation",
    "post_exploitation",
    "audit",
)

MAX_RAW_OUTPUT_BYTES = 50_000  # 50KB cap — spill to disk if exceeded
SPILL_DIR = settings.data_dir / "evidence_spills"


# ---------- Chain of custody (D23) ----------

def compute_evidence_hash(raw_output: str) -> str:
    """SHA-256 hash of raw_output (for integrity verification)."""
    return hashlib.sha256(raw_output.encode("utf-8")).hexdigest()


def compute_custody_seal(
    raw_output: str,
    evidence_id: uuid.UUID,
    captured_at: datetime,
    prev_seal: str | None = None,
) -> str:
    """HMAC-SHA256 tamper seal — links this evidence to the previous one.

    Seal = HMAC(encryption_key, f"{prev_seal}|{evidence_id}|{captured_at}|{hash}")

    This creates a blockchain-style chain: tampering with any evidence row
    breaks the seal chain (verifiable by recomputing).
    """
    key = settings.encryption_key.get_secret_value().encode("utf-8")
    evidence_hash = compute_evidence_hash(raw_output)
    message = f"{prev_seal or ''}|{evidence_id}|{captured_at.isoformat()}|{evidence_hash}"
    return hmac.new(key, message.encode("utf-8"), hashlib.sha256).hexdigest()


def verify_custody_seal(
    raw_output: str,
    evidence_id: uuid.UUID,
    captured_at: datetime,
    custody_seal: str,
    prev_seal: str | None = None,
) -> bool:
    """Verify that the custody seal matches (tamper detection)."""
    expected = compute_custody_seal(raw_output, evidence_id, captured_at, prev_seal)
    return hmac.compare_digest(expected, custody_seal)


# ---------- Evidence service ----------

class EvidenceService:
    """Service for creating + retrieving evidence with PII redaction + custody seals."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def add_evidence(
        self,
        finding_id: uuid.UUID,
        layer: str,
        raw_output: str,
        tool_used: str,
        spill_path: str | None = None,
    ) -> Evidence:
        """Add a new evidence row with PII redaction + custody seal.

        Args:
            finding_id: FK to vapt_findings.id
            layer: One of EVIDENCE_LAYERS (detection/validation/exploitation/post_exploitation/audit)
            raw_output: Raw tool output (will be PII-redacted before persistence)
            tool_used: Name of CLI tool that produced this (nmap/nuclei/sqlmap/...)
            spill_path: If raw_output > 50KB, full content stored on disk at this path

        Returns:
            Evidence ORM instance (not yet committed — caller must commit)
        """
        if layer not in EVIDENCE_LAYERS:
            raise ValueError(f"Invalid layer: {layer}. Must be one of {EVIDENCE_LAYERS}")

        # Step 1: PII redaction (D24)
        redacted_output = redact_pii(raw_output)
        audit = audit_redactions(raw_output)
        if audit["total"] > 0:
            logger.info("Evidence PII redaction: %d items (%s)",
                        audit["total"], audit["patterns"])

        # Step 2: Spill to disk if too large
        actual_output = redacted_output
        actual_spill_path = spill_path
        if len(redacted_output.encode("utf-8")) > MAX_RAW_OUTPUT_BYTES:
            SPILL_DIR.mkdir(parents=True, exist_ok=True)
            evidence_id = uuid.uuid4()
            spill_file = SPILL_DIR / f"evidence_{evidence_id}.txt"
            spill_file.write_text(redacted_output, encoding="utf-8")
            actual_output = f"[SPILLED TO DISK: {spill_file}]\n{redacted_output[:1000]}..."
            actual_spill_path = str(spill_file)
            logger.info("Evidence spilled to disk: %s (%d bytes)",
                        spill_file, len(redacted_output))
        else:
            evidence_id = uuid.uuid4()

        # Step 3: Compute hash + custody seal
        evidence_hash = compute_evidence_hash(actual_output)
        captured_at = datetime.now(UTC)

        # Get previous evidence's seal for chain (most recent for this finding)
        prev_evidence = await self.session.execute(
            select(Evidence)
            .where(Evidence.finding_id == finding_id)
            .order_by(Evidence.captured_at.desc())
            .limit(1)
        )
        prev_seal = prev_evidence.scalar_one_or_none().custody_seal if prev_evidence.scalar_one_or_none() else None

        custody_seal = compute_custody_seal(actual_output, evidence_id, captured_at, prev_seal)

        # Step 4: Create evidence row
        evidence = Evidence(
            id=evidence_id,
            finding_id=finding_id,
            layer=layer,
            raw_output=actual_output,
            tool_used=tool_used,
            custody_seal=custody_seal,
            evidence_hash=evidence_hash,
            captured_at=captured_at,
            spill_path=actual_spill_path,
        )
        self.session.add(evidence)
        await self.session.flush()

        logger.info("Evidence added: finding=%s layer=%s tool=%s hash=%s...",
                    finding_id, layer, tool_used, evidence_hash[:12])

        return evidence

    async def get_evidence(self, evidence_id: uuid.UUID) -> Evidence | None:
        """Get an evidence row by ID. Verifies custody seal before returning."""
        result = await self.session.execute(
            select(Evidence).where(Evidence.id == evidence_id)
        )
        evidence = result.scalar_one_or_none()
        if evidence is None:
            return None

        # Verify custody seal (tamper detection)
        # Get previous evidence's seal
        prev_result = await self.session.execute(
            select(Evidence)
            .where(Evidence.finding_id == evidence.finding_id)
            .where(Evidence.captured_at < evidence.captured_at)
            .order_by(Evidence.captured_at.desc())
            .limit(1)
        )
        prev_evidence = prev_result.scalar_one_or_none()
        prev_seal = prev_evidence.custody_seal if prev_evidence else None

        if not verify_custody_seal(
            evidence.raw_output,
            evidence.id,
            evidence.captured_at,
            evidence.custody_seal,
            prev_seal,
        ):
            logger.error("CUSTODY SEAL VERIFICATION FAILED: evidence=%s", evidence_id)
            raise RuntimeError(f"Evidence {evidence_id} custody seal verification failed — possible tampering")

        return evidence

    async def list_evidence_for_finding(
        self,
        finding_id: uuid.UUID,
        layer: str | None = None,
    ) -> list[Evidence]:
        """List all evidence for a finding, optionally filtered by layer."""
        query = select(Evidence).where(Evidence.finding_id == finding_id)
        if layer:
            query = query.where(Evidence.layer == layer)
        query = query.order_by(Evidence.captured_at.asc())

        result = await self.session.execute(query)
        return list(result.scalars().all())

    async def verify_finding_evidence_chain(self, finding_id: uuid.UUID) -> dict[str, Any]:
        """Verify the full custody seal chain for a finding's evidence.

        Returns dict with verification results:
            {
                "verified": True/False,
                "evidence_count": N,
                "broken_at": evidence_id or None,
            }
        """
        evidence_list = await self.list_evidence_for_finding(finding_id)

        prev_seal = None
        for ev in evidence_list:
            expected = compute_custody_seal(ev.raw_output, ev.id, ev.captured_at, prev_seal)
            if not hmac.compare_digest(expected, ev.custody_seal):
                return {
                    "verified": False,
                    "evidence_count": len(evidence_list),
                    "broken_at": str(ev.id),
                }
            prev_seal = ev.custody_seal

        return {
            "verified": True,
            "evidence_count": len(evidence_list),
            "broken_at": None,
        }

    # ---------- Finding helpers ----------

    async def create_finding(
        self,
        scan_id: str,
        name: str,
        vuln_type: str,
        severity: str,
        location: str,
        cvss_vector: str | None = None,
        cvss_base_score: float | None = None,
        wstg_test_id: str | None = None,
        mitre_attack_technique: str | None = None,
        mitre_attack_tactic: str | None = None,
        cwe_id: str | None = None,
        cve_id: str | None = None,
    ) -> Finding:
        """Create a new finding with CVSS + standards mapping.

        CVSS v3.1 vector format: "AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H"
        (Attack Vector / Attack Complexity / Privileges Required /
         User Interaction / Scope / Confidentiality / Integrity / Availability)

        Use app.evidence.cvss.calculate_cvss_base_score(vector) to compute
        base_score from vector (W2-C task).
        """
        finding = Finding(
            scan_id=scan_id,
            name=name,
            vuln_type=vuln_type,
            severity=severity,
            cvss_vector=cvss_vector,
            cvss_base_score=cvss_base_score,
            cvss_severity=severity.lower() if severity else None,
            location=location,
            wstg_test_id=wstg_test_id,
            mitre_attack_technique=mitre_attack_technique,
            mitre_attack_tactic=mitre_attack_tactic,
            cwe_id=cwe_id,
            cve_id=cve_id,
            poc_status="not_attempted",
            verified=False,
            false_positive=False,
            confidence_score=0.0,
        )
        self.session.add(finding)
        await self.session.flush()

        logger.info("Finding created: scan=%s vuln=%s severity=%s wstg=%s",
                    scan_id, vuln_type, severity, wstg_test_id)
        return finding
