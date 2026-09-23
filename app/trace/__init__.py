"""
VAPT-AI Replay Trace module.

Ported from EVVO Sentinel shield_engine/replay_trace.py.

Provides:
  - append_trace_event: write one JSONL line to data/traces/scan_<id>.jsonl
  - read_trace: read all events from a trace file
  - get_trace_stats: summary stats for a trace
  - ReplayTraceRecorder: high-level recorder + SQL row writer

Trace format (one JSONL line per event):
    {
        "ts": "2026-09-21T08:00:00+00:00",
        "trace_id": "scan_abc123",
        "event": "turn_start" | "action_selected" | "tool_executed" | ...,
        "turn": 5,
        "scan_id": "scan_abc123",
        "session_id": "session_xyz",
        "data": { ... event-specific payload, long values truncated ... }
    }

ReplayTrace SQL row (written at scan end by ReplayTraceRecorder.finalize):
    - trace_file_path: path to JSONL file
    - turn_count, total_reward, start/end_timestamp
    - kg_edges_updated_json: list of {edge_id, before, after, success}
    - final_epsilon, final_q_values_json (from RL policy)
"""
from __future__ import annotations

from app.trace.replay import (
    append_trace_event,
    read_trace,
    get_trace_stats,
    trace_path,
    ReplayTraceRecorder,
    _MAX_FIELD_CHARS,
)

__all__ = [
    "append_trace_event",
    "read_trace",
    "get_trace_stats",
    "trace_path",
    "ReplayTraceRecorder",
    "_MAX_FIELD_CHARS",
]