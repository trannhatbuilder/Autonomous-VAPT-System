"""
SQL ↔ NetworkX persistence bridge for VAPT-AI Knowledge Graph.

Bridges the in-memory NetworkX DiGraph (in app/kg/graph.py) with the
SQL tables defined in app/db/models/kg.py (W1-D):
    vapt_kg_nodes          — KGNode
    vapt_kg_edges          — KGEdge (with outcome probability tracking)
    vapt_kg_scan_snapshots — KGScanSnapshot (per-scan fork snapshots)

All functions are async (use SQLAlchemy 2.0 async session).

Load strategy (startup):
    1. load_kg_from_db(session) → KnowledgeGraph
       SELECT all kg_nodes + kg_edges → build NetworkX DiGraph
    2. If KG empty → seeder.seed_all(session, kg) populates from
       skills + WSTG + ATT&CK catalogs

Save strategy (write-through):
    - persist_node(session, kg, node_id) — upsert single node
    - persist_edge(session, kg, src_id, tgt_id) — upsert single edge
    - persist_kg_to_db(session, kg) — bulk upsert (for shutdown / periodic)

Snapshot strategy:
    - save_snapshot(session, scan_id, kg) — INSERT KGScanSnapshot row
      with serialized NetworkX graph (JSON)
    - load_snapshot(session, scan_id) → KnowledgeGraph (fork instance)
    - merge_snapshot(session, scan_id, snapshot_kg, main_kg) — atomic
      merge snapshot edges into main KG + UPDATE snapshot merge_status
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, UTC
from typing import Any

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.kg.graph import KnowledgeGraph
from app.kg.types import EdgeType, KGEdgeData, KGNodeData, NodeType

logger = logging.getLogger(__name__)


# ── Load ─────────────────────────────────────────────────────────────────

async def load_kg_from_db(session: AsyncSession) -> KnowledgeGraph:
    """Load all KG nodes + edges from SQL into a new KnowledgeGraph.

    Args:
        session: SQLAlchemy AsyncSession

    Returns:
        KnowledgeGraph populated with all SQL rows. Empty if DB has no KG data.
    """
    from app.db.models.kg import KGNode, KGEdge

    kg = KnowledgeGraph()

    # Load nodes
    result = await session.execute(select(KGNode))
    nodes = result.scalars().all()
    for node in nodes:
        kg.graph.add_node(
            str(node.id),
            node_type=node.node_type,
            name=node.name,
            metadata=dict(node.metadata_json or {}),
            created_at=node.created_at.isoformat() if node.created_at else None,
            updated_at=node.updated_at.isoformat() if node.updated_at else None,
        )

    # Load edges
    result = await session.execute(select(KGEdge))
    edges = result.scalars().all()
    for edge in edges:
        src_id = str(edge.source_node_id)
        tgt_id = str(edge.target_node_id)
        if src_id not in kg.graph.nodes or tgt_id not in kg.graph.nodes:
            # Stale edge — endpoint was deleted. Skip + log.
            logger.warning(
                "KG load: skipping edge with missing endpoint | src=%s tgt=%s",
                src_id, tgt_id,
            )
            continue
        kg.graph.add_edge(
            src_id, tgt_id,
            edge_type=edge.edge_type,
            success_count=int(edge.success_count or 0),
            total_attempts=int(edge.total_attempts or 0),
            wstg_test_id=edge.wstg_test_id,
            mitre_attack_technique=edge.mitre_attack_technique,
            last_scan_id=edge.last_scan_id,
            metadata=dict(edge.metadata_json or {}),
            created_at=edge.created_at.isoformat() if edge.created_at else None,
            updated_at=edge.updated_at.isoformat() if edge.updated_at else None,
        )

    logger.info(
        "KG loaded from DB | nodes=%d edges=%d",
        kg.graph.number_of_nodes(), kg.graph.number_of_edges(),
    )
    return kg


# ── Persist (write-through) ──────────────────────────────────────────────

async def persist_node(
    session: AsyncSession,
    kg: KnowledgeGraph,
    node_id: str,
) -> bool:
    """Upsert a single node to SQL. Returns True on success."""
    from app.db.models.kg import KGNode
    from sqlalchemy.dialects.postgresql import insert as pg_insert

    attrs = kg.graph.nodes.get(node_id)
    if attrs is None:
        return False

    node_type_str = attrs.get("node_type", "Technology")
    name = attrs.get("name", "")
    metadata = attrs.get("metadata") or {}

    # Use UUID string as id (must be valid UUID for SQL cast)
    # The kg_ prefix produces non-UUID strings; we let DB generate UUID
    # on insert, but track the in-memory ID via a metadata field.
    # For seeded nodes with deterministic IDs, we store the kg_id in metadata.
    stmt = pg_insert(KGNode).values(
        node_type=node_type_str,
        name=name,
        metadata_json={**metadata, "_kg_id": node_id},
    )
    stmt = stmt.on_conflict_do_update(
        index_elements=["node_type", "name"],  # unique constraint
        set_=dict(
            metadata_json=stmt.excluded.metadata_json,
            updated_at=datetime.now(UTC),
        ),
    )
    try:
        await session.execute(stmt)
        return True
    except Exception as exc:
        logger.warning("persist_node failed for %s: %s", node_id, exc)
        await session.rollback()
        return False


async def persist_edge(
    session: AsyncSession,
    kg: KnowledgeGraph,
    source_id: str,
    target_id: str,
) -> bool:
    """Upsert a single edge to SQL. Returns True on success."""
    from app.db.models.kg import KGEdge, KGNode

    if not kg.graph.has_edge(source_id, target_id):
        return False

    attrs = kg.graph.edges[source_id, target_id]
    src_attrs = kg.graph.nodes.get(source_id, {})
    tgt_attrs = kg.graph.nodes.get(target_id, {})

    # Look up SQL UUIDs for source + target nodes
    src_sql = await _lookup_node_uuid(session, src_attrs.get("name", ""),
                                       src_attrs.get("node_type", "Technology"))
    tgt_sql = await _lookup_node_uuid(session, tgt_attrs.get("name", ""),
                                       tgt_attrs.get("node_type", "Technology"))
    if src_sql is None or tgt_sql is None:
        logger.warning("persist_edge: endpoint not in SQL | src=%s tgt=%s", source_id, target_id)
        return False

    from sqlalchemy.dialects.postgresql import insert as pg_insert
    stmt = pg_insert(KGEdge).values(
        source_node_id=src_sql,
        target_node_id=tgt_sql,
        edge_type=attrs.get("edge_type", "has_vuln"),
        success_count=int(attrs.get("success_count", 0)),
        total_attempts=int(attrs.get("total_attempts", 0)),
        wstg_test_id=attrs.get("wstg_test_id"),
        mitre_attack_technique=attrs.get("mitre_attack_technique"),
        last_scan_id=attrs.get("last_scan_id"),
        metadata_json={**(attrs.get("metadata") or {}), "_kg_src_id": source_id, "_kg_tgt_id": target_id},
    )
    stmt = stmt.on_conflict_do_update(
        index_elements=["source_node_id", "target_node_id", "edge_type"],
        set_=dict(
            success_count=stmt.excluded.success_count,
            total_attempts=stmt.excluded.total_attempts,
            wstg_test_id=stmt.excluded.wstg_test_id,
            mitre_attack_technique=stmt.excluded.mitre_attack_technique,
            last_scan_id=stmt.excluded.last_scan_id,
            metadata_json=stmt.excluded.metadata_json,
            updated_at=datetime.now(UTC),
        ),
    )
    try:
        await session.execute(stmt)
        return True
    except Exception as exc:
        logger.warning("persist_edge failed for %s→%s: %s", source_id, target_id, exc)
        await session.rollback()
        return False


async def _lookup_node_uuid(
    session: AsyncSession,
    name: str,
    node_type: str,
) -> Any:
    """Look up a node's SQL UUID by (name, node_type). Returns None if not found."""
    from app.db.models.kg import KGNode

    try:
        stmt = select(KGNode.id).where(
            KGNode.name == name,
            KGNode.node_type == node_type,
        ).limit(1)
        result = await session.execute(stmt)
        row = result.scalar_one_or_none()
        return row
    except Exception:
        await session.rollback()
        return None


async def persist_kg_to_db(session: AsyncSession, kg: KnowledgeGraph) -> dict[str, int]:
    """Bulk persist all nodes + edges to SQL. For shutdown / periodic sync.

    Returns:
        {"nodes_upserted": N, "edges_upserted": N}
    """
    nodes_count = 0
    edges_count = 0

    # Persist all nodes first
    for node_id in list(kg.graph.nodes):
        if await persist_node(session, kg, node_id):
            nodes_count += 1

    # Then edges (endpoints must exist)
    for src, tgt in list(kg.graph.edges):
        if await persist_edge(session, kg, src, tgt):
            edges_count += 1

    logger.info(
        "KG persisted to DB | nodes_upserted=%d edges_upserted=%d",
        nodes_count, edges_count,
    )
    return {"nodes_upserted": nodes_count, "edges_upserted": edges_count}


# ── Snapshot persistence ─────────────────────────────────────────────────

async def save_snapshot(
    session: AsyncSession,
    scan_id: str,
    kg: KnowledgeGraph,
) -> str | None:
    """Save a per-scan KG snapshot to SQL.

    Serializes the NetworkX graph as JSON in snapshot_json column.
    For large graphs (> 1MB), spill to disk + store path instead.

    Args:
        session: SQLAlchemy AsyncSession
        scan_id: scan ID (FK to vapt_scans.id)
        kg: the forked KnowledgeGraph to snapshot

    Returns:
        snapshot_id (UUID string) on success, None on failure.
    """
    from app.db.models.kg import KGScanSnapshot

    # Serialize graph
    nodes_data = [
        {"id": n, **kg.graph.nodes[n]}
        for n in kg.graph.nodes
    ]
    edges_data = [
        {"source": u, "target": v, **kg.graph.edges[u, v]}
        for u, v in kg.graph.edges
    ]
    snapshot_payload = {
        "nodes": nodes_data,
        "edges": edges_data,
        "stats": kg.stats(),
        "saved_at": datetime.now(UTC).isoformat(),
    }
    snapshot_json_str = json.dumps(snapshot_payload, default=str)

    # Check size — if > 1MB, spill to disk
    snapshot_path = None
    snapshot_json_for_db = snapshot_payload
    if len(snapshot_json_str) > 1_000_000:
        # Spill to disk
        from app.core.config import settings
        spill_dir = settings.kg_dir / "snapshots"
        spill_dir.mkdir(parents=True, exist_ok=True)
        spill_path = spill_dir / f"scan_{scan_id}.json"
        spill_path.write_text(snapshot_json_str, encoding="utf-8")
        snapshot_path = str(spill_path)
        snapshot_json_for_db = None  # don't store in DB
        logger.info("KG snapshot spilled to disk | scan=%s | path=%s | size=%d bytes",
                    scan_id, snapshot_path, len(snapshot_json_str))

    row = KGScanSnapshot(
        scan_id=scan_id,
        snapshot_json=snapshot_json_for_db,
        snapshot_path=snapshot_path,
        merge_status="pending",
        node_count=kg.graph.number_of_nodes(),
        edge_count=kg.graph.number_of_edges(),
        edges_updated=0,  # will be set by merge_snapshot()
    )
    session.add(row)
    try:
        await session.flush()
        snapshot_id = str(row.id)
        logger.info(
            "KG snapshot saved | scan=%s | snapshot_id=%s | nodes=%d edges=%d",
            scan_id, snapshot_id, kg.graph.number_of_nodes(), kg.graph.number_of_edges(),
        )
        return snapshot_id
    except Exception as exc:
        logger.warning("save_snapshot failed for scan %s: %s", scan_id, exc)
        return None


async def load_snapshot(
    session: AsyncSession,
    scan_id: str,
) -> KnowledgeGraph | None:
    """Load a per-scan snapshot from SQL → forked KnowledgeGraph.

    Args:
        session: AsyncSession
        scan_id: scan ID

    Returns:
        KnowledgeGraph (is_fork=True) or None if no snapshot found.
    """
    from app.db.models.kg import KGScanSnapshot

    stmt = (
        select(KGScanSnapshot)
        .where(KGScanSnapshot.scan_id == scan_id)
        .order_by(KGScanSnapshot.created_at.desc())
        .limit(1)
    )
    result = await session.execute(stmt)
    row = result.scalars().first()
    if row is None:
        return None

    # Load payload
    if row.snapshot_json is not None:
        payload = row.snapshot_json
        if isinstance(payload, str):
            payload = json.loads(payload)
    elif row.snapshot_path:
        from pathlib import Path
        path = Path(row.snapshot_path)
        if not path.is_file():
            logger.warning("Snapshot spill file missing: %s", path)
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
    else:
        return None

    # Rebuild KnowledgeGraph
    kg = KnowledgeGraph(is_fork=True, fork_scan_id=scan_id)
    for node in payload.get("nodes", []):
        node_id = node.pop("id")
        kg.graph.add_node(node_id, **node)
    for edge in payload.get("edges", []):
        src = edge.pop("source")
        tgt = edge.pop("target")
        kg.graph.add_edge(src, tgt, **edge)

    logger.info(
        "KG snapshot loaded | scan=%s | nodes=%d edges=%d",
        scan_id, kg.graph.number_of_nodes(), kg.graph.number_of_edges(),
    )
    return kg


async def merge_snapshot(
    session: AsyncSession,
    scan_id: str,
    snapshot_kg: KnowledgeGraph,
    main_kg: KnowledgeGraph,
) -> dict[str, int]:
    """Merge a snapshot into the main KG (atomic).

    1. Calls main_kg.merge_scan(scan_id, snapshot_kg) — updates in-memory graph
    2. Persists updated edges to SQL (write-through)
    3. Updates KGScanSnapshot.merge_status = 'merged' + merged_at = now()

    Args:
        session: AsyncSession
        scan_id: scan ID
        snapshot_kg: the fork to merge from
        main_kg: the main KG to merge into

    Returns:
        {"nodes_added": N, "edges_added": N, "edges_updated": N}
    """
    from app.db.models.kg import KGScanSnapshot

    # In-memory merge
    merge_result = main_kg.merge_scan(scan_id, snapshot_kg)

    # Persist updated edges to SQL
    edges_persisted = 0
    for src, tgt in main_kg.graph.edges:
        # Only persist edges that were updated by the merge
        edge_attrs = main_kg.graph.edges[src, tgt]
        if edge_attrs.get("last_scan_id") == scan_id:
            if await persist_edge(session, main_kg, src, tgt):
                edges_persisted += 1

    # Update snapshot row merge_status
    stmt = (
        update(KGScanSnapshot)
        .where(KGScanSnapshot.scan_id == scan_id)
        .values(
            merge_status="merged",
            merged_at=datetime.now(UTC),
            edges_updated=merge_result["edges_updated"],
        )
    )
    try:
        await session.execute(stmt)
    except Exception as exc:
        logger.warning("merge_snapshot: status update failed for scan %s: %s", scan_id, exc)

    logger.info(
        "KG snapshot merged | scan=%s | in_memory=%s | edges_persisted=%d",
        scan_id, merge_result, edges_persisted,
    )
    return {**merge_result, "edges_persisted": edges_persisted}


async def discard_snapshot(
    session: AsyncSession,
    scan_id: str,
) -> bool:
    """Discard a snapshot (mark as 'discarded' — scan failed, don't merge).

    Args:
        session: AsyncSession
        scan_id: scan ID

    Returns:
        True on success, False if no snapshot found.
    """
    from app.db.models.kg import KGScanSnapshot

    stmt = (
        update(KGScanSnapshot)
        .where(KGScanSnapshot.scan_id == scan_id)
        .values(merge_status="discarded", merged_at=datetime.now(UTC))
    )
    try:
        result = await session.execute(stmt)
        return result.rowcount > 0
    except Exception as exc:
        logger.warning("discard_snapshot failed for scan %s: %s", scan_id, exc)
        return False


__all__ = [
    "load_kg_from_db",
    "persist_node",
    "persist_edge",
    "persist_kg_to_db",
    "save_snapshot",
    "load_snapshot",
    "merge_snapshot",
    "discard_snapshot",
]