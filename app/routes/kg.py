"""
VAPT-AI Knowledge Graph API routes.

7 FastAPI endpoints (all require auth) for KG introspection + admin:

    GET  /api/kg/stats                       — graph statistics
    GET  /api/kg/graph?limit=&offset=&type=  — paginated nodes + edges
    GET  /api/kg/node/{node_id}              — single node detail
    GET  /api/kg/search?q=&limit=            — search nodes by name
    POST /api/kg/ingest                      — admin: ingest scan outcome
    GET  /api/kg/paths?source=&max_depth=&top_k=  — attack paths from source
    POST /api/kg/snapshot/{scan_id}/merge    — admin: merge scan snapshot

All endpoints read from the in-memory KnowledgeGraph singleton (get_kg())
populated at startup by _bootstrap_kg() in app/main.py.
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from app.kg import (
    KnowledgeGraph, NodeType, EdgeType,
    get_kg, seed_all,
)
from app.kg.types import KGNodeData, KGEdgeData

router = APIRouter(prefix="/api/kg", tags=["kg"])


# ── Pydantic response schemas ────────────────────────────────────────────

class KGStatsResponse(BaseModel):
    total_nodes: int
    total_edges: int
    nodes_by_type: dict[str, int]
    edges_by_type: dict[str, int]
    consult_count: int
    is_fork: bool
    fork_scan_id: str | None = None


class KGNodeResponse(BaseModel):
    id: str
    node_type: str
    name: str
    metadata: dict[str, Any] = {}
    created_at: str | None = None
    updated_at: str | None = None


class KGEdgeResponse(BaseModel):
    source_node_id: str
    target_node_id: str
    edge_type: str
    success_count: int
    total_attempts: int
    probability: float
    wstg_test_id: str | None = None
    mitre_attack_technique: str | None = None
    last_scan_id: str | None = None
    metadata: dict[str, Any] = {}


class KGGraphResponse(BaseModel):
    nodes: list[KGNodeResponse]
    edges: list[KGEdgeResponse]
    total_nodes: int
    total_edges: int
    offset: int
    limit: int


class KGPathResponse(BaseModel):
    nodes: list[str]
    edges: list[dict[str, str]]
    score: float
    match_specificity: float = 0.0


class KGPathsResponse(BaseModel):
    source_node_ids: list[str]
    paths: list[KGPathResponse]
    count: int


class KGSearchResponse(BaseModel):
    query: str
    results: list[KGNodeResponse]
    count: int


class KGIngestRequest(BaseModel):
    """Request body for POST /api/kg/ingest — record scan outcome on edges."""
    scan_id: str = Field(..., description="Scan ID for attribution")
    edges: list[dict[str, Any]] = Field(
        ..., description="List of {source_id, target_id, success} tuples"
    )


class KGIngestResponse(BaseModel):
    scan_id: str
    edges_updated: int
    edges_failed: int


class KGSnapshotMergeResponse(BaseModel):
    scan_id: str
    merge_status: str
    nodes_added: int
    edges_added: int
    edges_updated: int
    edges_persisted: int = 0


# ── Helpers ──────────────────────────────────────────────────────────────

def _get_kg() -> KnowledgeGraph:
    """Get the module-level KG singleton. Raises 503 if not bootstrapped."""
    kg = get_kg()
    if kg.graph.number_of_nodes() == 0:
        raise HTTPException(
            status_code=503,
            detail="KG not bootstrapped (empty graph — check startup logs)",
        )
    return kg


def _node_to_response(node_data: KGNodeData) -> KGNodeResponse:
    return KGNodeResponse(
        id=node_data.id,
        node_type=node_data.node_type.value,
        name=node_data.name,
        metadata=node_data.metadata,
        created_at=node_data.created_at,
        updated_at=node_data.updated_at,
    )


def _edge_to_response(edge_data: KGEdgeData) -> KGEdgeResponse:
    return KGEdgeResponse(
        source_node_id=edge_data.source_node_id,
        target_node_id=edge_data.target_node_id,
        edge_type=edge_data.edge_type.value,
        success_count=edge_data.success_count,
        total_attempts=edge_data.total_attempts,
        probability=round(edge_data.probability, 4),
        wstg_test_id=edge_data.wstg_test_id,
        mitre_attack_technique=edge_data.mitre_attack_technique,
        last_scan_id=edge_data.last_scan_id,
        metadata=edge_data.metadata,
    )


# ── Endpoints ────────────────────────────────────────────────────────────

@router.get("/stats", response_model=KGStatsResponse)
async def get_kg_stats() -> KGStatsResponse:
    """Get KG statistics — node/edge counts by type, consult count."""
    kg = _get_kg()
    stats = kg.stats()
    return KGStatsResponse(**stats)


@router.get("/graph", response_model=KGGraphResponse)
async def get_kg_graph(
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    node_type: str | None = Query(None, description="Filter by NodeType"),
) -> KGGraphResponse:
    """Get paginated KG nodes + edges.

    Args:
        limit: max nodes to return (1-1000, default 100)
        offset: pagination offset
        node_type: optional NodeType filter (Technology/AttackVector/Finding/...)
    """
    kg = _get_kg()
    nodes: list[KGNodeResponse] = []
    skipped = 0
    taken = 0
    for node_id, attrs in kg.graph.nodes(data=True):
        if node_type and attrs.get("node_type") != node_type:
            continue
        if skipped < offset:
            skipped += 1
            continue
        if taken >= limit:
            break
        node_data = kg.get_node(node_id)
        if node_data:
            nodes.append(_node_to_response(node_data))
            taken += 1

    # Edges: only include edges where both endpoints are in the returned node set
    node_ids_set = {n.id for n in nodes}
    edges: list[KGEdgeResponse] = []
    for src, tgt, attrs in kg.graph.edges(data=True):
        if src in node_ids_set and tgt in node_ids_set:
            edge_data = kg.get_edge(src, tgt)
            if edge_data:
                edges.append(_edge_to_response(edge_data))

    return KGGraphResponse(
        nodes=nodes,
        edges=edges,
        total_nodes=kg.graph.number_of_nodes(),
        total_edges=kg.graph.number_of_edges(),
        offset=offset,
        limit=limit,
    )


@router.get("/node/{node_id}", response_model=KGNodeResponse)
async def get_kg_node(node_id: str) -> KGNodeResponse:
    """Get a single KG node by ID."""
    kg = _get_kg()
    node_data = kg.get_node(node_id)
    if node_data is None:
        raise HTTPException(status_code=404, detail=f"Node {node_id} not found")
    return _node_to_response(node_data)


@router.get("/search", response_model=KGSearchResponse)
async def search_kg(
    q: str = Query(..., min_length=1, description="Search query (case-insensitive substring)"),
    limit: int = Query(40, ge=1, le=200),
) -> KGSearchResponse:
    """Search KG nodes by name (case-insensitive substring match)."""
    kg = _get_kg()
    results = kg.search(q, limit=limit)
    nodes = [
        KGNodeResponse(
            id=r["id"],
            node_type=r["node_type"],
            name=r["name"],
            metadata=r.get("metadata", {}),
        )
        for r in results
    ]
    return KGSearchResponse(query=q, results=nodes, count=len(nodes))


@router.get("/paths", response_model=KGPathsResponse)
async def get_kg_paths(
    source: str = Query(..., description="Source node ID (comma-separated for multiple)"),
    max_depth: int = Query(4, ge=1, le=10),
    top_k: int = Query(10, ge=1, le=50),
    target_kind: str = Query("Finding", description="Target NodeType"),
) -> KGPathsResponse:
    """Get attack paths from source node(s) to target nodes.

    Args:
        source: source node ID, or comma-separated list for multiple sources
        max_depth: max path length (1-10, default 4)
        top_k: max paths to return (1-50, default 10)
        target_kind: target NodeType (default Finding)
    """
    kg = _get_kg()
    source_ids = [s.strip() for s in source.split(",") if s.strip()]
    if not source_ids:
        raise HTTPException(status_code=400, detail="At least one source node ID required")

    try:
        target_node_type = NodeType(target_kind)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid target_kind: {target_kind}. Valid: {[t.value for t in NodeType]}",
        )

    paths = kg.get_attack_paths(
        source_ids, max_depth=max_depth, top_k=top_k, target_kind=target_node_type,
    )
    path_responses = [
        KGPathResponse(
            nodes=p.nodes,
            edges=[
                {"source": s, "target": t, "edge_type": e}
                for s, t, e in p.edges
            ],
            score=round(p.score, 4),
            match_specificity=round(p.match_specificity, 4),
        )
        for p in paths
    ]
    return KGPathsResponse(
        source_node_ids=source_ids,
        paths=path_responses,
        count=len(path_responses),
    )


@router.post("/ingest", response_model=KGIngestResponse)
async def ingest_kg_outcomes(req: KGIngestRequest) -> KGIngestResponse:
    """Ingest scan outcomes — update edge success_count/total_attempts.

    Admin endpoint for recording what worked/didn't in a scan. Called
    by the Harness Bridge (W18) at end of each scan.

    Request body:
        scan_id: scan ID for attribution
        edges: list of {source_id, target_id, success (bool)}
    """
    kg = _get_kg()
    updated = 0
    failed = 0
    for edge_req in req.edges:
        src = edge_req.get("source_id")
        tgt = edge_req.get("target_id")
        success = bool(edge_req.get("success", False))
        if not src or not tgt:
            failed += 1
            continue
        ok = kg.update_edge_outcome(src, tgt, success=success, scan_id=req.scan_id)
        if ok:
            updated += 1
        else:
            failed += 1
    return KGIngestResponse(scan_id=req.scan_id, edges_updated=updated, edges_failed=failed)


@router.post("/snapshot/{scan_id}/merge", response_model=KGSnapshotMergeResponse)
async def merge_scan_snapshot(scan_id: str) -> KGSnapshotMergeResponse:
    """Merge a per-scan KG snapshot into the main KG.

    Admin endpoint. Loads the snapshot from SQL, merges into main KG
    in-memory, persists updated edges to SQL.

    Args:
        scan_id: scan ID whose snapshot to merge
    """
    from app.db.session import async_session
    from app.kg import load_snapshot, merge_snapshot

    kg = _get_kg()
    try:
        async with async_session() as session:
            snapshot_kg = await load_snapshot(session, scan_id)
            if snapshot_kg is None:
                raise HTTPException(
                    status_code=404,
                    detail=f"No snapshot found for scan {scan_id}",
                )
            result = await merge_snapshot(session, scan_id, snapshot_kg, kg)
            await session.commit()
        return KGSnapshotMergeResponse(
            scan_id=scan_id,
            merge_status="merged",
            nodes_added=result["nodes_added"],
            edges_added=result["edges_added"],
            edges_updated=result["edges_updated"],
            edges_persisted=result.get("edges_persisted", 0),
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Merge failed: {exc}")


__all__ = ["router"]