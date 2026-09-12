"""
VAPT-AI methodology catalog + skill execution tracking.

Tables:
    wstg_methodology_catalog  — OWASP WSTG v4.2 catalog (80+ test IDs)
    skill_executions          — skill invocation log (which skill ran when, by which agent)

WSTG catalog is seeded from data/wstg_catalog.yaml (W8 task).
Skill library is loaded from skills/<name>/SKILL.md (W11 task — 24 skills from CyberStrikeAI).
"""
from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import DateTime, Boolean, ForeignKey, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, TimestampMixin, UUIDPrimaryKey


class WSTGMethodologyCatalog(Base, UUIDPrimaryKey, TimestampMixin):
    """OWASP WSTG v4.2 methodology catalog.

    One row per WSTG test ID (e.g. WSTG-INPV-05 = Testing for SQL Injection).
    Seeded from data/wstg_catalog.yaml at app startup.

    Used to:
        - Tag every tool wrapper with WSTG ID (W2 task)
        - Tag every Finding with wstg_test_id
        - Generate "Methodology Coverage" table in PDF report
    """

    __tablename__ = "vapt_wstg_methodology_catalog"

    wstg_id: Mapped[str] = mapped_column(String(32), unique=True, nullable=False, index=True)
    # e.g. "WSTG-INPV-05"

    name: Mapped[str] = mapped_column(String(255), nullable=False)
    # e.g. "Testing for SQL Injection"

    category: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    # e.g. "Input Validation Testing"

    description: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Related standards mapping
    related_mitre_attack: Mapped[list] = mapped_column(JSONB, nullable=False, default=list, server_default="[]")
    # ["T1190"]
    related_cwe: Mapped[list] = mapped_column(JSONB, nullable=False, default=list, server_default="[]")
    # ["CWE-89"]

    # Is this test applicable to web pentest MVP? (some WSTG tests are network-only)
    applicable_to_web_mvp: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=True, server_default="true"
    )


class SkillExecution(Base, UUIDPrimaryKey, TimestampMixin):
    """Skill invocation log — tracks every time a skill was loaded + used by an agent.

    Skills are markdown playbook packages (e.g. web-attack-methods, post-exploitation).
    Loaded from skills/<name>/SKILL.md (24 skills from CyberStrikeAI in W11).
    """

    __tablename__ = "vapt_skill_executions"

    scan_id: Mapped[str | None] = mapped_column(
        String(64),
        ForeignKey("vapt_scans.id"),
        nullable=True,
        index=True,
    )

    skill_name: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    # e.g. "web-attack-methods", "post-exploitation"

    agent_name: Mapped[str] = mapped_column(String(64), nullable=False)
    # which agent invoked this skill (recon / penetration / privilege_escalation / ...)

    invoked_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=func.now(), server_default=func.now())
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    # Did the skill succeed?
    success: Mapped[bool] = mapped_column(nullable=False, default=False, server_default="false")

    # Output summary (what the skill produced — NOT full output, that goes to Evidence)
    output_summary: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Skill version (from SKILL.md frontmatter)
    skill_version: Mapped[str | None] = mapped_column(String(32), nullable=True)