"""
VAPT-AI Replayable pentest trace store.

Python port of EVVO Sentinel shield_engine/replay_trace.py (96 LOC).

JSONL traces bridge the gap between per-scan execution and KG/RL learning:
  - the KG stores distilled knowledge (cross-scan);
  - the RL store stores (state, action, reward, next_state) tuples;
  - traces store replayable turn-by-turn evidence and attribution
    (full observation text, tool output, KG edge updates, reward breakdown).

Storage:
  - JSONL file at data/traces/scan_<id>.jsonl (one line per event)
  - ReplayTrace SQL row at scan end (metadata: turn_count, total_reward,
    trace_file_path, final_epsilon, final_q_values_json)

Thread safety: per-trace_id threading.Lock prevents concurrent write
corruption (multiple agents may write to the same scan trace).
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ── Config ───────────────────────────────────────────────────────────────
_MAX_FIELD_CHARS: int = int(os.getenv("VAPT_AI_TRACE_MAX_FIELD_CHARS", "4000"))

# Per-trace locks (prevents concurrent write corruption)
_locks: dict[str, threading.Lock] = {}
_locks_guard = threading.Lock()


def _now() -> str:
    """Current UTC timestamp in ISO 8601."""
    return datetime.now(timezone.utc).isoformat()


def _lock_for(trace_id: str) -> threading.Lock:
    """Get or create a threading.Lock for a trace_id."""
    with _locks_guard:
        if trace_id not in _locks:
            _locks[trace_id] = threading.Lock()
        return _locks[trace_id]


def _safe_value(value: Any) -> Any:
    """Recursively truncate long strings in trace data.

    Long values (> _MAX_FIELD_CHARS) are replaced with a dict containing
    a SHA-256 hash + preview. This keeps trace files manageable while
    preserving auditability (hash matches original).
    """
    if isinstance(value, str):
        if len(value) > _MAX_FIELD_CHARS:
            digest = hashlib.sha256(value.encode("utf-8", errors="replace")).hexdigest()
            return {
                "truncated": True,
                "sha256": digest,
                "chars": len(value),
                "preview": value[:_MAX_FIELD_CHARS],
            }
        return value
    if isinstance(value, dict):
        return {str(k): _safe_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_value(v) for v in value]
    return value


def trace_path(trace_id: str, trace_dir: Path | None = None) -> Path:
    """Get the JSONL file path for a trace_id.

    Args:
        trace_id: typically the scan_id (e.g. "scan_abc123")
        trace_dir: optional override for trace directory. If None, uses
            settings.traces_dir.

    Returns:
        Path to <trace_dir>/<sanitized_trace_id>.jsonl
    """
    if trace_dir is None:
        from app.core.config import settings
        trace_dir = settings.traces_dir
    # Sanitize: keep alnum + -_, cap at 160 chars
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in trace_id)[:160]
    return trace_dir / f"{safe}.jsonl"


def append_trace_event(
    trace_id: str,
    event_type: str,
    *,
    turn: int | None = None,
    scan_id: str | None = None,
    session_id: str | None = None,
    data: dict[str, Any] | None = None,
    trace_dir: Path | None = None,
) -> dict[str, Any]:
    """Append one event to a trace JSONL file.

    Args:
        trace_id: trace identifier (typically scan_id)
        event_type: event type (e.g. "turn_start", "action_selected",
            "tool_executed", "verifier_result", "turn_end", "scan_complete")
        turn: turn number within scan (optional)
        scan_id: scan ID for cross-reference (optional, defaults to trace_id)
        session_id: session ID for multi-agent attribution (optional)
        data: event payload dict (will be _safe_value'd to truncate long strings)
        trace_dir: optional override for trace directory

    Returns:
        The event dict that was written (with ts timestamp).
    """
    if trace_dir is None:
        from app.core.config import settings
        trace_dir = settings.traces_dir

    trace_dir.mkdir(parents=True, exist_ok=True)
    event = {
        "ts": _now(),
        "trace_id": trace_id,
        "event": event_type,
        "turn": turn,
        "scan_id": scan_id or trace_id,
        "session_id": session_id,
        "data": _safe_value(data or {}),
    }
    path = trace_path(trace_id, trace_dir)
    line = json.dumps(event, ensure_ascii=False, default=str)
    with _lock_for(trace_id):
        with path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    return event


def read_trace(
    trace_id: str,
    *,
    limit: int | None = None,
    trace_dir: Path | None = None,
) -> list[dict[str, Any]]:
    """Read events from a trace JSONL file.

    Args:
        trace_id: trace identifier
        limit: optional max events to return (most recent first N)
        trace_dir: optional override for trace directory

    Returns:
        list of event dicts in chronological order (oldest first).
        Empty list if trace file doesn't exist.
    """
    path = trace_path(trace_id, trace_dir)
    if not path.exists():
        return []
    events: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError as exc:
                logger.warning("trace %s: skipping malformed line: %s", trace_id, exc)
            if limit is not None and len(events) >= limit:
                break
    return events


def get_trace_stats(trace_id: str, trace_dir: Path | None = None) -> dict[str, Any]:
    """Get summary statistics for a trace.

    Returns:
        dict with: trace_id, event_count, turn_count, first_event_ts,
        last_event_ts, file_path, file_size_bytes
    """
    path = trace_path(trace_id, trace_dir)
    if not path.exists():
        return {
            "trace_id": trace_id,
            "event_count": 0,
            "turn_count": 0,
            "first_event_ts": None,
            "last_event_ts": None,
            "file_path": str(path),
            "file_size_bytes": 0,
        }
    events = read_trace(trace_id, trace_dir=trace_dir)
    turns = [e.get("turn") for e in events if e.get("turn") is not None]
    return {
        "trace_id": trace_id,
        "event_count": len(events),
        "turn_count": max(turns) + 1 if turns else 0,
        "first_event_ts": events[0]["ts"] if events else None,
        "last_event_ts": events[-1]["ts"] if events else None,
        "file_path": str(path),
        "file_size_bytes": path.stat().st_size,
    }


# ── ReplayTraceRecorder ──────────────────────────────────────────────────

class ReplayTraceRecorder:
    """High-level recorder that wraps append_trace_event + writes ReplayTrace
    SQL row at scan end.

    Usage (in scan_pipeline):
        recorder = ReplayTraceRecorder(scan_id)
        recorder.event("scan_start", data={"target": target})
        # ... each turn:
        recorder.event("turn_start", turn=t, data={...})
        recorder.event("action_selected", turn=t, data={...})
        recorder.event("tool_executed", turn=t, data={...})
        recorder.event("turn_end", turn=t, data={...})
        # ... at scan end:
        await recorder.finalize(session, policy=rl_policy, kg_edges_updated=[...])
    """

    def __init__(
        self,
        scan_id: str,
        session_id: str | None = None,
        trace_dir: Path | None = None,
    ) -> None:
        self.scan_id = scan_id
        self.session_id = session_id or scan_id
        self._trace_dir = trace_dir  # if None, uses settings.traces_dir
        self._turn_count = 0
        self._total_reward = 0.0
        self._start_ts = _now()
        self._kg_edges_updated: list[dict[str, Any]] = []

    def event(
        self,
        event_type: str,
        *,
        turn: int | None = None,
        data: dict[str, Any] | None = None,
        reward: float | None = None,
    ) -> dict[str, Any]:
        """Record one trace event. Updates internal counters."""
        if turn is not None and turn >= self._turn_count:
            self._turn_count = turn + 1
        if reward is not None:
            self._total_reward += float(reward)
        return append_trace_event(
            trace_id=self.scan_id,
            event_type=event_type,
            turn=turn,
            scan_id=self.scan_id,
            session_id=self.session_id,
            data=data,
            trace_dir=self._trace_dir,
        )

    def record_kg_edge_update(
        self,
        edge_id: str,
        before_probability: float,
        after_probability: float,
        success: bool,
    ) -> None:
        """Track KG edge outcome update for ReplayTrace SQL row."""
        self._kg_edges_updated.append({
            "edge_id": edge_id,
            "before": round(before_probability, 4),
            "after": round(after_probability, 4),
            "success": success,
        })

    async def finalize(
        self,
        session: Any,  # AsyncSession
        *,
        policy: Any = None,  # PentestPolicy (optional, for epsilon + q_values)
        scan_status: str = "completed",
    ) -> str | None:
        """Write ReplayTrace SQL row + scan_complete trace event.

        Args:
            session: SQLAlchemy AsyncSession
            policy: optional PentestPolicy (for final_epsilon + final_q_values_json)
            scan_status: scan completion status

        Returns:
            ReplayTrace row ID (UUID string) on success, None on failure.
        """
        # Write scan_complete event
        self.event("scan_complete", turn=self._turn_count - 1, data={
            "scan_status": scan_status,
            "total_reward": round(self._total_reward, 4),
            "turn_count": self._turn_count,
            "kg_edges_updated_count": len(self._kg_edges_updated),
        })

        # Build final metadata
        final_epsilon = None
        final_q_values = None
        if policy is not None:
            try:
                final_epsilon = float(policy.epsilon)
                # Sample Q-values from a zero state (representative snapshot)
                import numpy as np
                from app.rl import STATE_DIM
                sample_state = np.zeros(STATE_DIM, dtype=np.float32)
                q_vals = policy.q.q_values(sample_state)
                final_q_values = [round(float(v), 4) for v in q_vals.tolist()]
            except Exception as exc:
                logger.warning("Failed to capture final ε/q_values: %s", exc)

        # INSERT ReplayTrace SQL row
        try:
            from app.db.models.replay import ReplayTrace
            from sqlalchemy import select
            end_ts = _now()
            trace_file = trace_path(self.scan_id, self._trace_dir)

            # Check if row already exists (idempotent — update if so)
            stmt = select(ReplayTrace).where(ReplayTrace.scan_id == self.scan_id).limit(1)
            result = await session.execute(stmt)
            existing = result.scalars().first()

            if existing is None:
                row = ReplayTrace(
                    scan_id=self.scan_id,
                    trace_file_path=str(trace_file),
                    turn_count=self._turn_count,
                    total_reward=round(self._total_reward, 4),
                    start_timestamp=self._start_ts,
                    end_timestamp=end_ts,
                    kg_edges_updated_json=self._kg_edges_updated,
                    final_epsilon=final_epsilon,
                    final_q_values_json=final_q_values,
                    agent_mode=None,  # set by caller if available
                    scan_status=scan_status,
                )
                session.add(row)
                await session.flush()
                replay_id = str(row.id)
            else:
                # Update existing row
                existing.trace_file_path = str(trace_file)
                existing.turn_count = self._turn_count
                existing.total_reward = round(self._total_reward, 4)
                existing.end_timestamp = end_ts
                existing.kg_edges_updated_json = self._kg_edges_updated
                existing.final_epsilon = final_epsilon
                existing.final_q_values_json = final_q_values
                existing.scan_status = scan_status
                replay_id = str(existing.id)

            logger.info(
                "ReplayTrace finalized | scan=%s | turns=%d | reward=%.2f | kg_edges=%d | ε=%s",
                self.scan_id, self._turn_count, self._total_reward,
                len(self._kg_edges_updated), final_epsilon,
            )
            return replay_id
        except Exception as exc:
            logger.warning("ReplayTrace SQL insert failed: %s", exc)
            return None


__all__ = [
    # Functions
    "trace_path",
    "append_trace_event",
    "read_trace",
    "get_trace_stats",
    # Class
    "ReplayTraceRecorder",
    # Constants
    "_MAX_FIELD_CHARS",
]