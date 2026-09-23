"""
VAPT-AI Replay Trace API routes.

4 FastAPI endpoints (all require auth) for trace introspection + download:

    GET  /api/trace/{scan_id}            — list trace events (paginated)
    GET  /api/trace/{scan_id}/download   — download JSONL file
    GET  /api/trace/{scan_id}/stats      — trace summary stats
    GET  /api/trace                      — list all trace files

Reads from data/traces/scan_<id>.jsonl (written by ReplayTraceRecorder).
"""
from __future__ import annotations

import logging
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse, PlainTextResponse
from pydantic import BaseModel

from app.core.config import settings
from app.trace import read_trace, get_trace_stats, trace_path

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/trace", tags=["trace"])


# ── Pydantic response schemas ────────────────────────────────────────────

class TraceEventResponse(BaseModel):
    ts: str
    trace_id: str
    event: str
    turn: int | None = None
    scan_id: str | None = None
    session_id: str | None = None
    data: dict = {}


class TraceListResponse(BaseModel):
    scan_id: str
    events: list[TraceEventResponse]
    count: int
    total_in_file: int


class TraceStatsResponse(BaseModel):
    trace_id: str
    event_count: int
    turn_count: int
    first_event_ts: str | None = None
    last_event_ts: str | None = None
    file_path: str
    file_size_bytes: int


class TraceFileInfoResponse(BaseModel):
    scan_id: str
    file_path: str
    file_size_bytes: int
    event_count: int


class TraceListAllResponse(BaseModel):
    traces: list[TraceFileInfoResponse]
    count: int
    trace_dir: str


# ── Endpoints ────────────────────────────────────────────────────────────

@router.get("/{scan_id}", response_model=TraceListResponse)
async def list_trace_events(
    scan_id: str,
    limit: int = Query(100, ge=1, le=10000),
    offset: int = Query(0, ge=0),
) -> TraceListResponse:
    """List trace events for a scan (paginated).

    Args:
        scan_id: scan ID (trace file is data/traces/scan_<id>.jsonl)
        limit: max events to return (1-10000, default 100)
        offset: skip first N events
    """
    # Read all events first to get total count
    all_events = read_trace(scan_id)
    if not all_events:
        raise HTTPException(
            status_code=404,
            detail=f"No trace found for scan {scan_id}",
        )

    # Paginate
    paginated = all_events[offset:offset + limit]
    event_responses = [
        TraceEventResponse(
            ts=e.get("ts", ""),
            trace_id=e.get("trace_id", scan_id),
            event=e.get("event", ""),
            turn=e.get("turn"),
            scan_id=e.get("scan_id"),
            session_id=e.get("session_id"),
            data=e.get("data", {}),
        )
        for e in paginated
    ]
    return TraceListResponse(
        scan_id=scan_id,
        events=event_responses,
        count=len(event_responses),
        total_in_file=len(all_events),
    )


@router.get("/{scan_id}/download")
async def download_trace(scan_id: str):
    """Download the raw JSONL trace file.

    Returns the file as application/octet-stream attachment.
    """
    path = trace_path(scan_id)
    if not path.is_file():
        raise HTTPException(
            status_code=404,
            detail=f"Trace file not found: {path.name}",
        )
    return FileResponse(
        path=str(path),
        media_type="application/octet-stream",
        filename=path.name,
    )


@router.get("/{scan_id}/stats", response_model=TraceStatsResponse)
async def get_trace_stats_endpoint(scan_id: str) -> TraceStatsResponse:
    """Get summary statistics for a trace.

    Returns: event_count, turn_count, first/last event timestamp, file path,
    file size in bytes.
    """
    stats = get_trace_stats(scan_id)
    if stats["event_count"] == 0:
        raise HTTPException(
            status_code=404,
            detail=f"No trace found for scan {scan_id}",
        )
    return TraceStatsResponse(**stats)


@router.get("", response_model=TraceListAllResponse)
async def list_all_traces(
    limit: int = Query(50, ge=1, le=500),
) -> TraceListAllResponse:
    """List all trace files in the trace directory.

    Returns file info (scan_id, path, size, event_count) for each trace.
    Sorted by modification time (newest first).
    """
    trace_dir = settings.traces_dir
    if not trace_dir.is_dir():
        return TraceListAllResponse(traces=[], count=0, trace_dir=str(trace_dir))

    trace_files = sorted(
        trace_dir.glob("*.jsonl"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )[:limit]

    traces: list[TraceFileInfoResponse] = []
    for path in trace_files:
        scan_id = path.stem  # filename without .jsonl
        # Count events (read file, count lines)
        try:
            event_count = sum(1 for _ in path.open(encoding="utf-8"))
        except Exception:
            event_count = 0
        traces.append(TraceFileInfoResponse(
            scan_id=scan_id,
            file_path=str(path),
            file_size_bytes=path.stat().st_size,
            event_count=event_count,
        ))

    return TraceListAllResponse(
        traces=traces,
        count=len(traces),
        trace_dir=str(trace_dir),
    )


__all__ = ["router"]