"""
VAPT-AI MITRE ATT&CK technique catalog model.

W8-E: Stores curated ATT&CK Enterprise techniques (44 entries) seeded
from data/attack_mapping.yaml at app startup.

This table is the counterpart of vapt_wstg_methodology_catalog — together
they form the bidirectional WSTG ↔ ATT&CK ↔ CWE cross-reference network.

Used to:
    - Tag every Finding with mitre_attack_technique
    - Generate ATT&CK heatmap in PDF/SARIF report (W20)
    - Provide "Detection" + "Mitigation" text for each finding's report section
    - Cross-link to WSTG catalog (related_wstg_ids JSONB)
"""
from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import Boolean, DateTime, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, TimestampMixin, UUIDPrimaryKey


class AttackTechniqueCatalog(Base, UUIDPrimaryKey, TimestampMixin):
    """MITRE ATT&CK Enterprise catalog (curated subset for web/network pentest MVP).

    One row per ATT&CK technique ID (e.g. T1190 = Exploit Public-Facing App).
    Seeded from data/attack_mapping.yaml at app startup.

    Used to:
        - Tag every Finding with mitre_attack_technique (W12 verifier)
        - Generate "ATT&CK Heatmap" in PDF report (W20)
        - Provide "Detection" text per finding (W20 report)
        - Provide "Mitigation" text per finding (W20 report)
        - Cross-reference WSTG catalog (related_wstg_ids JSONB)
    """

    __tablename__ = "vapt_attack_technique_catalog"

    technique_id: Mapped[str] = mapped_column(
        String(32), unique=True, nullable=False, index=True
    )
    # e.g. "T1190" or "T1059.004" (sub-technique)

    name: Mapped[str] = mapped_column(String(255), nullable=False)
    # e.g. "Exploit Public-Facing Application"

    tactic: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    # e.g. "Initial Access" or "Execution, Persistence" (multi-tactic comma-separated)

    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    # Practical pentest context (>=50 chars in catalog YAML)

    detection: Mapped[str | None] = mapped_column(Text, nullable=True)
    # How defenders detect this technique (for report's Detection section)

    mitigation: Mapped[str | None] = mapped_column(Text, nullable=True)
    # How to fix / defend (for report's Remediation section)

    # Cross-references
    related_wstg_ids: Mapped[list] = mapped_column(
        JSONB, nullable=False, default=list, server_default="[]"
    )
    # ["WSTG-INPV-05", "WSTG-INPV-13"] — cross-references to vapt_wstg_methodology_catalog

    related_cwe: Mapped[list] = mapped_column(
        JSONB, nullable=False, default=list, server_default="[]"
    )
    # ["CWE-89", "CWE-78"]

    example_uses: Mapped[list] = mapped_column(
        JSONB, nullable=False, default=list, server_default="[]"
    )
    # ["sqlmap -u 'https://target.com/login' --batch", ...] — practical pentest scenarios

    applicable_to_web_mvp: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=True, server_default="true"
    )
    # False = network-only technique (e.g. T1133 External Remote Services, T1486 Ransomware)
