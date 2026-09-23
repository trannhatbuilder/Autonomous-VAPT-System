"""
VAPT-AI Knowledge Graph type definitions.

Enums + dataclasses shared by app/kg/graph.py, app/kg/persistence.py,
app/kg/seeder.py, and app/harness/egats.py.

Node types (6 — per master plan §12 W17 + W1-D schema):
    Technology      — e.g. "Apache 2.4.49", "PHP 8.1", "WordPress 6.4"
    AttackVector    — e.g. "SQL Injection", "RCE via deserialization"
    Finding         — link to vapt_findings.id
    Recommendation  — e.g. "Upgrade to Apache 2.4.50"
    Verification    — e.g. "sqlmap --batch --dump"
    Outcome         — e.g. "Shell obtained", "Data dumped", "FP rejected"

Edge types (5 — per W1-D schema):
    has_vuln         Technology → AttackVector
    produces_finding AttackVector → Finding
    recommends       Finding → Recommendation
    verified_by      Finding → Verification
    resulted_in      Finding → Outcome
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ── Node types ───────────────────────────────────────────────────────────

class NodeType(str, Enum):
    """KG node type — 6 values per master plan §12 W17."""

    TECHNOLOGY = "Technology"
    ATTACK_VECTOR = "AttackVector"
    FINDING = "Finding"
    RECOMMENDATION = "Recommendation"
    VERIFICATION = "Verification"
    OUTCOME = "Outcome"


# ── Edge types ───────────────────────────────────────────────────────────

class EdgeType(str, Enum):
    """KG edge type — 5 values per W1-D schema.

    Each edge type has a canonical source → target node type pattern:
        has_vuln         Technology → AttackVector
        produces_finding AttackVector → Finding
        recommends       Finding → Recommendation
        verified_by      Finding → Verification
        resulted_in      Finding → Outcome
    """

    HAS_VULN = "has_vuln"
    PRODUCES_FINDING = "produces_finding"
    RECOMMENDS = "recommends"
    VERIFIED_BY = "verified_by"
    RESULTED_IN = "resulted_in"


# Canonical source/target NodeType for each EdgeType (for validation)
EDGE_TYPE_NODE_CONSTRAINTS: dict[EdgeType, tuple[NodeType, NodeType]] = {
    EdgeType.HAS_VULN: (NodeType.TECHNOLOGY, NodeType.ATTACK_VECTOR),
    EdgeType.PRODUCES_FINDING: (NodeType.ATTACK_VECTOR, NodeType.FINDING),
    EdgeType.RECOMMENDS: (NodeType.FINDING, NodeType.RECOMMENDATION),
    EdgeType.VERIFIED_BY: (NodeType.FINDING, NodeType.VERIFICATION),
    EdgeType.RESULTED_IN: (NodeType.FINDING, NodeType.OUTCOME),
}


# ── Dataclasses ──────────────────────────────────────────────────────────

@dataclass
class KGNodeData:
    """In-memory representation of a KG node (NetworkX node attrs).

    Mirrors vapt_kg_nodes SQL row but uses Python-native types.
    """

    id: str  # UUID string (or deterministic hash for seeded nodes)
    node_type: NodeType
    name: str
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str | None = None  # ISO 8601
    updated_at: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "node_type": self.node_type.value,
            "name": self.name,
            "metadata": self.metadata,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "KGNodeData":
        return cls(
            id=str(d["id"]),
            node_type=NodeType(d.get("node_type") or d.get("kind") or "Technology"),
            name=str(d["name"]),
            metadata=dict(d.get("metadata") or d.get("metadata_json") or {}),
            created_at=d.get("created_at"),
            updated_at=d.get("updated_at"),
        )


@dataclass
class KGEdgeData:
    """In-memory representation of a KG edge (NetworkX edge attrs).

    Mirrors vapt_kg_edges SQL row. Outcome probability is computed
    lazily from success_count / total_attempts (Laplace-smoothed).
    """

    source_node_id: str
    target_node_id: str
    edge_type: EdgeType
    success_count: int = 0
    total_attempts: int = 0
    wstg_test_id: str | None = None
    mitre_attack_technique: str | None = None
    last_scan_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str | None = None
    updated_at: str | None = None

    @property
    def probability(self) -> float:
        """Laplace-smoothed success probability: (s + 1) / (n + 2).

        Fresh edges (0/0) → 0.5 (uncertain prior). Converges to true
        rate as attempts accumulate.
        """
        if self.total_attempts < 0:
            return 0.5
        return (self.success_count + 1.0) / (self.total_attempts + 2.0)

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_node_id": self.source_node_id,
            "target_node_id": self.target_node_id,
            "edge_type": self.edge_type.value,
            "success_count": self.success_count,
            "total_attempts": self.total_attempts,
            "probability": round(self.probability, 4),
            "wstg_test_id": self.wstg_test_id,
            "mitre_attack_technique": self.mitre_attack_technique,
            "last_scan_id": self.last_scan_id,
            "metadata": self.metadata,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "KGEdgeData":
        return cls(
            source_node_id=str(d["source_node_id"]),
            target_node_id=str(d["target_node_id"]),
            edge_type=EdgeType(d["edge_type"]),
            success_count=int(d.get("success_count", 0)),
            total_attempts=int(d.get("total_attempts", 0)),
            wstg_test_id=d.get("wstg_test_id"),
            mitre_attack_technique=d.get("mitre_attack_technique"),
            last_scan_id=d.get("last_scan_id"),
            metadata=dict(d.get("metadata") or d.get("metadata_json") or {}),
            created_at=d.get("created_at"),
            updated_at=d.get("updated_at"),
        )


@dataclass
class KGPath:
    """A path through the KG — sequence of nodes + edges with a score.

    Returned by KnowledgeGraph.get_attack_paths() and consumed by EGATS.
    """

    nodes: list[str]  # node IDs in path order
    edges: list[tuple[str, str, str]]  # (source_id, target_id, edge_type)
    score: float  # path score (severity * probability product)
    match_specificity: float = 0.0  # 0..1 — how well path matches target inventory

    def as_dict(self) -> dict[str, Any]:
        return {
            "nodes": self.nodes,
            "edges": [
                {"source": s, "target": t, "edge_type": e}
                for s, t, e in self.edges
            ],
            "score": round(self.score, 4),
            "match_specificity": round(self.match_specificity, 4),
        }


@dataclass
class ConsultResult:
    """Result of KnowledgeGraph.consult() — attack path recommendations.

    Returned to the agent loop. The agent picks a path, executes it,
    then reports outcome back via record_consult_hit() or record_consult_miss().
    """

    consult_id: str  # UUID for tracking (agent quotes back when reporting)
    paths: list[KGPath]  # ranked attack paths (highest score first)
    inventory_summary: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "consult_id": self.consult_id,
            "paths": [p.as_dict() for p in self.paths],
            "inventory_summary": self.inventory_summary,
            "notes": self.notes,
        }


# ── Helpers ──────────────────────────────────────────────────────────────

def make_node_id(node_type: NodeType, name: str) -> str:
    """Deterministic node ID from (type, name) — enables idempotent seeding.

    Uses SHA-256 hash of "type:name" so the same node always gets the same ID,
    regardless of insertion order. Format: "kg_<first16hex>".
    """
    import hashlib
    h = hashlib.sha256(f"{node_type.value}:{name}".encode("utf-8")).hexdigest()
    return f"kg_{h[:16]}"


def validate_edge_types(
    source_type: NodeType,
    target_type: NodeType,
    edge_type: EdgeType,
) -> bool:
    """Check that the (source_type, target_type, edge_type) tuple is valid.

    Returns True if the edge type's canonical source/target node types
    match the provided source/target. False otherwise (caller should
    log + reject).
    """
    expected_src, expected_tgt = EDGE_TYPE_NODE_CONSTRAINTS.get(
        edge_type, (None, None)
    )
    if expected_src is None:
        return False
    return source_type == expected_src and target_type == expected_tgt


__all__ = [
    # Enums
    "NodeType", "EdgeType",
    # Constants
    "EDGE_TYPE_NODE_CONSTRAINTS",
    # Dataclasses
    "KGNodeData", "KGEdgeData", "KGPath", "ConsultResult",
    # Helpers
    "make_node_id", "validate_edge_types",
]