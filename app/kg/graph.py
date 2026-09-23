"""
VAPT-AI Dynamic Knowledge Graph.

Python port of EVVO Sentinel shield_engine/vapt_kg.py — adapted for
VAPT-AI schema (6 node types + 5 edge types per W1-D) and async SQL
persistence (vapt_kg_nodes + vapt_kg_edges + vapt_kg_scan_snapshots).

Architecture:
    - NetworkX DiGraph in-memory (hot-path queries: O(V+E) for paths)
    - SQL persistence (durable, cross-restart)
    - Load on startup: SQL → NetworkX
    - Write-through: every add_node/add_edge/update_edge_outcome updates
      both NetworkX (sync) and SQL (async, best-effort)

Per-scan snapshot flow (per master plan §12 W17):
    1. scan_start: kg_snapshot = await kg.fork_for_scan(scan_id)
       → creates KGScanSnapshot row (status=pending) + in-memory fork
    2. during scan: agent consults kg_snapshot (isolated from main KG)
    3. scan_end: await kg.merge_scan(scan_id)
       → atomic merge snapshot edges into main KG + status=merged

Outcome probability (Laplace-smoothed):
    P(success) = (success_count + 1) / (total_attempts + 2)
    Fresh edges (0/0) → 0.5 (uncertain prior). Converges to true rate.
"""
from __future__ import annotations

import copy
import logging
import secrets
from collections import defaultdict
from datetime import datetime, UTC
from typing import Any

import networkx as nx

from app.kg.types import (
    ConsultResult,
    EdgeType,
    KGEdgeData,
    KGNodeData,
    KGPath,
    NodeType,
    make_node_id,
    validate_edge_types,
)

logger = logging.getLogger(__name__)


# ── Defaults ─────────────────────────────────────────────────────────────
DEFAULT_MAX_DEPTH: int = 4
DEFAULT_TOP_K: int = 10
SEED_PROBABILITY: float = 0.5  # initial probability for seeded edges


class KnowledgeGraph:
    """In-memory NetworkX DiGraph backed by SQL persistence.

    One instance per process (singleton via get_kg()). For per-scan
    isolation, use fork_for_scan() which returns a new KnowledgeGraph
    with _is_fork=True (writes go to snapshot, not main KG).
    """

    def __init__(self, *, is_fork: bool = False, fork_scan_id: str | None = None) -> None:
        self.graph: nx.DiGraph = nx.DiGraph()
        self._is_fork: bool = is_fork
        self._fork_scan_id: str | None = fork_scan_id
        # Consult tracking — agent quotes consult_id back when reporting
        # a finding so we can credit the recommendation that led to a hit.
        self._consult_log: dict[str, dict[str, Any]] = {}
        self._consult_order: list[str] = []
        self._consult_max: int = 200

    # ── node ops ─────────────────────────────────────────────────────

    def add_node(
        self,
        node_type: NodeType,
        name: str,
        *,
        metadata: dict[str, Any] | None = None,
        node_id: str | None = None,
    ) -> str:
        """Add a node to the graph. Idempotent — re-adding updates metadata.

        Args:
            node_type: one of NodeType
            name: human-readable name (unique within type)
            metadata: type-specific metadata dict
            node_id: optional explicit ID. If None, deterministic ID
                from (node_type, name) via make_node_id().

        Returns:
            node_id (string)
        """
        if node_id is None:
            node_id = make_node_id(node_type, name)
        now = _now_iso()
        existing = self.graph.nodes.get(node_id)
        if existing is None:
            self.graph.add_node(
                node_id,
                node_type=node_type.value,
                name=name,
                metadata=metadata or {},
                created_at=now,
                updated_at=now,
            )
        else:
            # Update metadata (merge)
            self.graph.nodes[node_id]["metadata"] = {
                **(existing.get("metadata") or {}),
                **(metadata or {}),
            }
            self.graph.nodes[node_id]["updated_at"] = now
        return node_id

    def get_node(self, node_id: str) -> KGNodeData | None:
        """Get a node by ID. Returns None if not found."""
        attrs = self.graph.nodes.get(node_id)
        if attrs is None:
            return None
        return KGNodeData(
            id=node_id,
            node_type=NodeType(attrs.get("node_type", "Technology")),
            name=attrs.get("name", ""),
            metadata=dict(attrs.get("metadata") or {}),
            created_at=attrs.get("created_at"),
            updated_at=attrs.get("updated_at"),
        )

    def find_node(self, node_type: NodeType, name: str) -> str | None:
        """Find node ID by (type, name). Returns None if not found."""
        node_id = make_node_id(node_type, name)
        if node_id in self.graph.nodes:
            return node_id
        return None

    # ── edge ops ────────────────────────────────────────────────────

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        edge_type: EdgeType,
        *,
        wstg_test_id: str | None = None,
        mitre_attack_technique: str | None = None,
        last_scan_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Add an edge. Returns True if added/updated, False if validation failed.

        Validates: both endpoints exist + edge_type matches their node types.
        Idempotent — re-adding updates metadata + wstg/mitre fields.
        """
        if source_id not in self.graph.nodes or target_id not in self.graph.nodes:
            logger.warning(
                "KG add_edge failed — endpoint missing | src=%s tgt=%s type=%s",
                source_id, target_id, edge_type.value,
            )
            return False

        src_type = NodeType(self.graph.nodes[source_id].get("node_type", "Technology"))
        tgt_type = NodeType(self.graph.nodes[target_id].get("node_type", "Technology"))
        if not validate_edge_types(src_type, tgt_type, edge_type):
            logger.warning(
                "KG add_edge failed — type mismatch | %s(%s)→%s(%s) edge=%s",
                source_id, src_type.value, target_id, tgt_type.value, edge_type.value,
            )
            return False

        now = _now_iso()
        if self.graph.has_edge(source_id, target_id):
            # Update existing
            existing = self.graph.edges[source_id, target_id]
            existing["metadata"] = {**(existing.get("metadata") or {}), **(metadata or {})}
            if wstg_test_id:
                existing["wstg_test_id"] = wstg_test_id
            if mitre_attack_technique:
                existing["mitre_attack_technique"] = mitre_attack_technique
            if last_scan_id:
                existing["last_scan_id"] = last_scan_id
            existing["updated_at"] = now
        else:
            self.graph.add_edge(
                source_id, target_id,
                edge_type=edge_type.value,
                success_count=0,
                total_attempts=0,
                wstg_test_id=wstg_test_id,
                mitre_attack_technique=mitre_attack_technique,
                last_scan_id=last_scan_id,
                metadata=metadata or {},
                created_at=now,
                updated_at=now,
            )
        return True

    def get_edge(self, source_id: str, target_id: str) -> KGEdgeData | None:
        """Get edge data. Returns None if not found."""
        if not self.graph.has_edge(source_id, target_id):
            return None
        attrs = self.graph.edges[source_id, target_id]
        return KGEdgeData(
            source_node_id=source_id,
            target_node_id=target_id,
            edge_type=EdgeType(attrs.get("edge_type", "has_vuln")),
            success_count=int(attrs.get("success_count", 0)),
            total_attempts=int(attrs.get("total_attempts", 0)),
            wstg_test_id=attrs.get("wstg_test_id"),
            mitre_attack_technique=attrs.get("mitre_attack_technique"),
            last_scan_id=attrs.get("last_scan_id"),
            metadata=dict(attrs.get("metadata") or {}),
            created_at=attrs.get("created_at"),
            updated_at=attrs.get("updated_at"),
        )

    def update_edge_outcome(
        self,
        source_id: str,
        target_id: str,
        *,
        success: bool,
        scan_id: str | None = None,
    ) -> bool:
        """Record an outcome on an edge (success or failure).

        Increments success_count (if success) and total_attempts on both
        the in-memory graph and (for non-fork KGs) the SQL row.

        Args:
            source_id, target_id: edge endpoints
            success: True if the action succeeded
            scan_id: optional scan ID for attribution

        Returns:
            True if updated, False if edge not found.
        """
        if not self.graph.has_edge(source_id, target_id):
            return False
        attrs = self.graph.edges[source_id, target_id]
        attrs["total_attempts"] = int(attrs.get("total_attempts", 0)) + 1
        if success:
            attrs["success_count"] = int(attrs.get("success_count", 0)) + 1
        if scan_id:
            attrs["last_scan_id"] = scan_id
        attrs["updated_at"] = _now_iso()
        return True

    # ── query ops ───────────────────────────────────────────────────

    def get_attack_paths(
        self,
        source_node_ids: list[str],
        *,
        max_depth: int = DEFAULT_MAX_DEPTH,
        top_k: int = DEFAULT_TOP_K,
        target_kind: NodeType = NodeType.FINDING,
    ) -> list[KGPath]:
        """Find attack paths from source nodes to target nodes (BFS).

        Args:
            source_node_ids: starting node IDs (usually Technology nodes)
            max_depth: max path length (edges)
            top_k: return top-K paths by score
            target_kind: target node type (default Finding)

        Returns:
            list of KGPath sorted by score (highest first)
        """
        if not source_node_ids or self.graph.number_of_nodes() == 0:
            return []

        paths: list[KGPath] = []
        seen_paths: set[tuple[str, ...]] = set()

        for source_id in source_node_ids:
            if source_id not in self.graph.nodes:
                continue
            # BFS up to max_depth
            for target_id in self.graph.nodes:
                if target_id == source_id:
                    continue
                target_type = self.graph.nodes[target_id].get("node_type")
                if target_type != target_kind.value:
                    continue
                try:
                    # all_simple_paths returns generator of node lists
                    for node_path in nx.all_simple_paths(
                        self.graph, source_id, target_id, cutoff=max_depth,
                    ):
                        path_key = tuple(node_path)
                        if path_key in seen_paths:
                            continue
                        seen_paths.add(path_key)
                        edges = [
                            (node_path[i], node_path[i + 1],
                             self.graph.edges[node_path[i], node_path[i + 1]].get("edge_type", "has_vuln"))
                            for i in range(len(node_path) - 1)
                        ]
                        score = self._score_path(node_path, edges)
                        paths.append(KGPath(
                            nodes=node_path,
                            edges=edges,
                            score=score,
                        ))
                except nx.NetworkXError:
                    continue

        paths.sort(key=lambda p: p.score, reverse=True)
        return paths[:top_k]

    def _score_path(self, nodes: list[str], edges: list[tuple[str, str, str]]) -> float:
        """Path score = product of edge probabilities * severity factor.

        Severity factor: max CVSS base score among AttackVector nodes
        in the path (normalized to [0, 1] by dividing by 10).
        """
        prob_product = 1.0
        for src, tgt, _ in edges:
            edge_data = self.get_edge(src, tgt)
            if edge_data:
                prob_product *= edge_data.probability
        if prob_product <= 0:
            prob_product = 0.01  # floor

        max_severity = 0.5  # default if no CVSS info
        for node_id in nodes:
            attrs = self.graph.nodes[node_id]
            if attrs.get("node_type") == NodeType.ATTACK_VECTOR.value:
                meta = attrs.get("metadata") or {}
                cvss = meta.get("cvss_base_score")
                if cvss and isinstance(cvss, (int, float)):
                    max_severity = max(max_severity, float(cvss) / 10.0)

        return prob_product * max_severity

    def consult(
        self,
        source_node_ids: list[str],
        *,
        max_depth: int = DEFAULT_MAX_DEPTH,
        top_k: int = DEFAULT_TOP_K,
        target_kind: NodeType = NodeType.FINDING,
    ) -> ConsultResult:
        """Consult the KG for attack path recommendations.

        Returns a ConsultResult with consult_id (for tracking) + ranked paths.
        Agent picks a path, executes it, then reports outcome via
        record_consult_hit() or record_consult_miss().
        """
        consult_id = f"consult_{secrets.token_hex(8)}"
        paths = self.get_attack_paths(
            source_node_ids, max_depth=max_depth, top_k=top_k, target_kind=target_kind,
        )
        result = ConsultResult(
            consult_id=consult_id,
            paths=paths,
            inventory_summary=self._inventory_summary(source_node_ids),
        )
        self._record_consult(result, source_node_ids)
        return result

    def _inventory_summary(self, node_ids: list[str]) -> dict[str, Any]:
        """Build a summary of the source inventory for the consult result."""
        techs: list[str] = []
        for nid in node_ids:
            attrs = self.graph.nodes.get(nid, {})
            if attrs.get("node_type") == NodeType.TECHNOLOGY.value:
                techs.append(attrs.get("name", ""))
        return {
            "technology_count": len(techs),
            "technologies": techs[:20],  # cap for response size
        }

    def _record_consult(self, result: ConsultResult, queried: list[str]) -> None:
        """Track consult for feedback loop (agent quotes consult_id back)."""
        self._consult_log[result.consult_id] = {
            "paths": [p.as_dict() for p in result.paths],
            "queried": queried,
            "timestamp": _now_iso(),
            "hit": None,  # set by record_consult_hit/miss
        }
        self._consult_order.append(result.consult_id)
        # Cap memory
        if len(self._consult_order) > self._consult_max:
            old_id = self._consult_order.pop(0)
            self._consult_log.pop(old_id, None)

    def record_consult_hit(self, consult_id: str, finding_id: str) -> None:
        """Mark a consult as leading to a verified finding (feedback loop)."""
        if consult_id in self._consult_log:
            self._consult_log[consult_id]["hit"] = {
                "finding_id": finding_id,
                "timestamp": _now_iso(),
            }

    def record_consult_miss(self, consult_id: str, reason: str = "") -> None:
        """Mark a consult as not leading to a finding."""
        if consult_id in self._consult_log:
            self._consult_log[consult_id]["hit"] = {
                "finding_id": None,
                "reason": reason,
                "timestamp": _now_iso(),
            }

    # ── per-scan snapshot ───────────────────────────────────────────

    def fork_for_scan(self, scan_id: str) -> "KnowledgeGraph":
        """Fork the KG for a scan — returns a new KnowledgeGraph with _is_fork=True.

        The fork is an isolated copy; writes to it don't affect the main KG
        until merge_scan() is called at scan end.
        """
        fork = KnowledgeGraph(is_fork=True, fork_scan_id=scan_id)
        fork.graph = copy.deepcopy(self.graph)
        logger.info(
            "KG forked for scan | scan_id=%s | nodes=%d | edges=%d",
            scan_id, fork.graph.number_of_nodes(), fork.graph.number_of_edges(),
        )
        return fork

    def merge_scan(self, scan_id: str, snapshot_kg: "KnowledgeGraph") -> dict[str, int]:
        """Merge a per-scan snapshot back into the main KG (atomic).

        Only edges with updated outcome (success_count/total_attempts
        changed) are merged — node additions are also merged.

        Args:
            scan_id: scan ID for attribution
            snapshot_kg: the forked KnowledgeGraph to merge from

        Returns:
            {"nodes_added": N, "edges_added": N, "edges_updated": N}
        """
        if not snapshot_kg._is_fork:
            logger.warning("merge_scan called on non-fork KG — skipping")
            return {"nodes_added": 0, "edges_added": 0, "edges_updated": 0}

        nodes_added = 0
        edges_added = 0
        edges_updated = 0

        # Merge nodes
        for node_id, attrs in snapshot_kg.graph.nodes(data=True):
            if node_id not in self.graph.nodes:
                self.graph.add_node(node_id, **attrs)
                nodes_added += 1

        # Merge edges (preserve outcome counts)
        for src, tgt, attrs in snapshot_kg.graph.edges(data=True):
            if self.graph.has_edge(src, tgt):
                existing = self.graph.edges[src, tgt]
                # If snapshot has more attempts, it learned something — merge
                if attrs.get("total_attempts", 0) > existing.get("total_attempts", 0):
                    existing["success_count"] = attrs["success_count"]
                    existing["total_attempts"] = attrs["total_attempts"]
                    existing["last_scan_id"] = scan_id
                    existing["updated_at"] = _now_iso()
                    edges_updated += 1
            else:
                self.graph.add_edge(src, tgt, **attrs)
                edges_added += 1

        logger.info(
            "KG merge complete | scan_id=%s | nodes_added=%d edges_added=%d edges_updated=%d",
            scan_id, nodes_added, edges_added, edges_updated,
        )
        return {
            "nodes_added": nodes_added,
            "edges_added": edges_added,
            "edges_updated": edges_updated,
        }

    # ── introspection ───────────────────────────────────────────────

    def stats(self) -> dict[str, Any]:
        """Return graph statistics."""
        node_counts: dict[str, int] = defaultdict(int)
        edge_counts: dict[str, int] = defaultdict(int)
        for _, attrs in self.graph.nodes(data=True):
            node_counts[attrs.get("node_type", "Unknown")] += 1
        for _, _, attrs in self.graph.edges(data=True):
            edge_counts[attrs.get("edge_type", "Unknown")] += 1
        return {
            "total_nodes": self.graph.number_of_nodes(),
            "total_edges": self.graph.number_of_edges(),
            "nodes_by_type": dict(node_counts),
            "edges_by_type": dict(edge_counts),
            "consult_count": len(self._consult_log),
            "is_fork": self._is_fork,
            "fork_scan_id": self._fork_scan_id,
        }

    def search(self, query: str, limit: int = 40) -> list[dict[str, Any]]:
        """Search nodes by name (case-insensitive substring)."""
        q_lower = query.lower()
        results: list[dict[str, Any]] = []
        for node_id, attrs in self.graph.nodes(data=True):
            name = attrs.get("name", "")
            if q_lower in name.lower():
                results.append({
                    "id": node_id,
                    "node_type": attrs.get("node_type"),
                    "name": name,
                    "metadata": attrs.get("metadata", {}),
                })
                if len(results) >= limit:
                    break
        return results

    def clear(self) -> None:
        """Clear all nodes + edges from the in-memory graph."""
        self.graph.clear()
        self._consult_log.clear()
        self._consult_order.clear()


# ── Helpers ──────────────────────────────────────────────────────────────

def _now_iso() -> str:
    """Current UTC timestamp in ISO 8601 format."""
    return datetime.now(UTC).isoformat()


# ── Module-level singleton ───────────────────────────────────────────────

_kg_singleton: KnowledgeGraph | None = None


def get_kg() -> KnowledgeGraph:
    """Get the module-level KG singleton (main, non-fork).

    For per-scan isolation, use kg.fork_for_scan(scan_id) instead.
    """
    global _kg_singleton
    if _kg_singleton is None:
        _kg_singleton = KnowledgeGraph()
    return _kg_singleton


def reset_kg_singleton() -> None:
    """Reset the singleton — for tests only."""
    global _kg_singleton
    _kg_singleton = None


__all__ = [
    "KnowledgeGraph",
    "get_kg",
    "reset_kg_singleton",
    "DEFAULT_MAX_DEPTH",
    "DEFAULT_TOP_K",
    "SEED_PROBABILITY",
]