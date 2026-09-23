"""
VAPT-AI Dynamic Knowledge Graph module.

Ported from EVVO Sentinel shield_engine/vapt_kg.py — adapted for
VAPT-AI schema (6 node types + 5 edge types per W1-D) and async SQL
persistence (vapt_kg_nodes + vapt_kg_edges + vapt_kg_scan_snapshots).

Components:
    - types.py         NodeType, EdgeType, KGNodeData, KGEdgeData, KGPath, ConsultResult
    - graph.py         KnowledgeGraph class — NetworkX in-memory + SQL bridge
    - persistence.py   load_kg_from_db, persist_kg_to_db, save/load/merge snapshots
    - seeder.py        seed_all — populate from skills + WSTG + ATT&CK catalogs
    - (EGATS)          app/harness/egats.py — UCB1 bandit over KG attack paths
    - (TDA)            app/harness/tda.py — Task Difficulty Index for EGATS

Node types (6 — per master plan §12 W17):
    Technology, AttackVector, Finding, Recommendation, Verification, Outcome

Edge types (5 — per W1-D schema):
    has_vuln         Technology → AttackVector
    produces_finding AttackVector → Finding
    recommends       Finding → Recommendation
    verified_by      Finding → Verification
    resulted_in      Finding → Outcome

Outcome probability (Laplace-smoothed):
    P(success) = (success_count + 1) / (total_attempts + 2)
    Fresh edges (0/0) → 0.5 (uncertain prior). Converges to true rate.

Seeding (W17-S6):
    - 24 skill packages → Technology + AttackVector nodes (P=0.5)
    - 119 OWASP WSTG v4.2 IDs → AttackVector nodes
    - 44 MITRE ATT&CK v15.1 techniques → AttackVector nodes
"""
from __future__ import annotations

# Core types
from app.kg.types import (
    NodeType, EdgeType, EDGE_TYPE_NODE_CONSTRAINTS,
    KGNodeData, KGEdgeData, KGPath, ConsultResult,
    make_node_id, validate_edge_types,
)

# KnowledgeGraph class + singleton
from app.kg.graph import (
    KnowledgeGraph, get_kg, reset_kg_singleton,
    DEFAULT_MAX_DEPTH, DEFAULT_TOP_K, SEED_PROBABILITY,
)

# Persistence (SQL ↔ NetworkX)
from app.kg.persistence import (
    load_kg_from_db, persist_node, persist_edge, persist_kg_to_db,
    save_snapshot, load_snapshot, merge_snapshot, discard_snapshot,
)

# Seeder
from app.kg.seeder import (
    SeedResult, seed_from_skills, seed_from_wstg_catalog,
    seed_from_attack_catalog, seed_all,
)

__all__ = [
    # Types
    "NodeType", "EdgeType", "EDGE_TYPE_NODE_CONSTRAINTS",
    "KGNodeData", "KGEdgeData", "KGPath", "ConsultResult",
    "make_node_id", "validate_edge_types",
    # Graph
    "KnowledgeGraph", "get_kg", "reset_kg_singleton",
    "DEFAULT_MAX_DEPTH", "DEFAULT_TOP_K", "SEED_PROBABILITY",
    # Persistence
    "load_kg_from_db", "persist_node", "persist_edge", "persist_kg_to_db",
    "save_snapshot", "load_snapshot", "merge_snapshot", "discard_snapshot",
    # Seeder
    "SeedResult", "seed_from_skills", "seed_from_wstg_catalog",
    "seed_from_attack_catalog", "seed_all",
]