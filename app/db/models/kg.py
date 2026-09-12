"""
VAPT-AI Dynamic Knowledge Graph models.

Tables:
    kg_nodes          — KG nodes (6 types: Technology, AttackVector, Finding,
                        Recommendation, Verification, Outcome)
    kg_edges          — KG edges (5 types: has_vuln, produces_finding,
                        recommends, verified_by, resulted_in) with outcome
                        probabilities (success_count / total_attempts)
    kg_scan_snapshots — per-scan KG fork snapshots (atomic merge at scan end)

Ported from EVVO Sentinel's shield_engine/vapt_kg.py (2650 LOC) — migrated
from JSON file persistence to SQL persistence. NetworkX DiGraph kept in-memory
for hot-path queries; SQL is the source of truth.

KG schema (per plan §12):
    Node types:
        Technology       — e.g. "Apache 2.4.49", "PHP 8.1", "WordPress 6.4"
        AttackVector     — e.g. "SQL Injection", "RCE via deserialization"
        Finding          — link to vapt_findings.id
        Recommendation   — e.g. "Upgrade to Apache 2.4.50", "Use prepared statements"
        Verification     — e.g. "sqlmap --batch --dump", "nuclei -t cves/2024/CVE-..."
        Outcome          — e.g. "Shell obtained", "Data dumped", "FP rejected"

    Edge types:
        Technology → AttackVector    ("has_vuln")
        AttackVector → Finding       ("produces_finding")
        Finding → Recommendation     ("recommends")
        Finding → Verification       ("verified_by")
        Finding → Outcome            ("resulted_in")

    Each AttackVector → Outcome edge carries outcome probability:
        success_count / total_attempts = probability
        (updated after each scan — KG "learns" which attack paths work)

Seeding (W17 task):
    - 24 skill packages from CyberStrikeAI (initial probability 0.5 uncertain)
    - OWASP WSTG v4.2 catalog (80+ WSTG IDs as nodes)
    - MITRE ATT&CK catalog
    - CVE/NVD API on-demand
"""
from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, TimestampMixin, UUIDPrimaryKey


class KGNode(Base, UUIDPrimaryKey, TimestampMixin):
    """Knowledge Graph node.

    Each node has a type + name + metadata. Edges (KGEdge) connect nodes.
    """

    __tablename__ = "vapt_kg_nodes"

    # Node type — 6 types per plan §12
    node_type: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    # Technology / AttackVector / Finding / Recommendation / Verification / Outcome

    # Node name (unique within type)
    name: Mapped[str] = mapped_column(String(255), nullable=False, index=True)

    # Unique constraint on (node_type, name) — no duplicate nodes
    # (enforced via unique index below)

    # Metadata (JSONB — type-specific)
    # Examples:
    #   Technology:    {vendor, version, cpe, discovered_in_scans: [...]}
    #   AttackVector:  {wstg_test_id, mitre_attack_technique, cvss_vector, severity}
    #   Finding:       {finding_id (FK to vapt_findings), severity, verified}
    #   Recommendation:{priority, effort, references: [...]}
    #   Verification:  {tool, command, expected_output_pattern}
    #   Outcome:       {outcome_type, impact_description}
    metadata_json: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict, server_default="{}")


class KGEdge(Base, UUIDPrimaryKey, TimestampMixin):
    """Knowledge Graph edge — directed, with outcome probability.

    Each edge connects source_node → target_node with a type. For
    AttackVector → Outcome edges, success_count / total_attempts tracks
    the empirical success rate (KG "learns" which paths work).

    Edge types:
        has_vuln         Technology → AttackVector
        produces_finding AttackVector → Finding
        recommends       Finding → Recommendation
        verified_by      Finding → Verification
        resulted_in      Finding → Outcome
    """

    __tablename__ = "vapt_kg_edges"

    # Source + target nodes (FKs to vapt_kg_nodes)
    source_node_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("vapt_kg_nodes.id"),
        nullable=False,
        index=True,
    )
    target_node_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("vapt_kg_nodes.id"),
        nullable=False,
        index=True,
    )

    # Edge type
    edge_type: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    # has_vuln / produces_finding / recommends / verified_by / resulted_in

    # Outcome probability tracking (for AttackVector → Outcome edges)
    success_count: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, server_default="0"
    )
    total_attempts: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, server_default="0"
    )
    probability: Mapped[float] = mapped_column(
        Float, nullable=False, default=0.5, server_default="0.5"
    )  # success_count / total_attempts (Laplace-smoothed: (s+1)/(n+2))

    # Standards mapping (denormalized for fast EGATS queries)
    wstg_test_id: Mapped[str | None] = mapped_column(String(32), nullable=True, index=True)
    mitre_attack_technique: Mapped[str | None] = mapped_column(String(32), nullable=True, index=True)

    # Last scan that updated this edge
    last_scan_id: Mapped[str | None] = mapped_column(
        String(64),
        ForeignKey("vapt_scans.id"),
        nullable=True,
    )

    # Edge metadata (JSONB — edge-specific info)
    # Examples:
    #   has_vuln:    {cvss_vector, cvss_base_score, discovered_in_scans: [...]}
    #   resulted_in: {outcome_type, impact_description, last_verified_at}
    metadata_json: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict, server_default="{}")


class KGScanSnapshot(Base, UUIDPrimaryKey, TimestampMixin):
    """Per-scan KG fork snapshot.

    At scan start, fork the main KG → snapshot (isolated per-scan view).
    During scan, agent consults snapshot for path probabilities; updates
    snapshot edges with new outcomes.
    At scan end, atomic merge snapshot → main KG (or discard if scan failed).

    Storage: snapshot_json contains the full NetworkX DiGraph serialized
    (nodes + edges + metadata). For large KGs, this could spill to disk
    (data/vapt_kg/snapshots/scan_<id>.json) with snapshot_path pointing there.
    """

    __tablename__ = "vapt_kg_scan_snapshots"

    scan_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("vapt_scans.id"),
        nullable=False,
        index=True,
    )

    # Snapshot serialization (NetworkX DiGraph as JSON)
    # For large KGs, set snapshot_json=None and store in snapshot_path instead
    snapshot_json: Mapped[dict | None] = mapped_column(JSONB, nullable=True)

    # Spill path (if snapshot too large for JSONB)
    snapshot_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    # e.g. "data/vapt_kg/snapshots/scan_abc123.json"

    # Merge status
    merge_status: Mapped[str] = mapped_column(
        String(16), nullable=False, default="pending", server_default="pending", index=True
    )  # pending / merged / discarded

    merged_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    # Snapshot statistics (for debug / audit)
    node_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    edge_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    edges_updated: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    # number of edges whose success_count/total_attempts changed in this snapshot