"""
VAPT-AI Tool Output Artifacts — retrievable full tool output per scan.

Why this exists
---------------
The agent loop must bound its context: `_prune_old_tool_results()` truncates
old tool results to 1_500 chars and `MAX_LLM_OUTPUT_CHARS` (8_000) truncates
each fresh result before it reaches the LLM. Both notes told the model
"full output preserved in scan artifacts / spill file. Reference it if
needed." — but there was NO agent-facing tool that could read a past
execution. The note promised a capability that did not exist, so a finding
whose evidence sat in the truncated tail of a sqlmap run could not be
re-checked without re-running the tool.

This module stores each execution's output (up to the executor's 50 KB cap)
keyed by execution_id, so the `fetch_tool_output` tool can return any slice
of it on demand.

Design (mirrors `tool_call_cache.py`)
------------------------------------
    - PER-SCAN, IN-MEMORY only. Keyed by scan_id so scans are isolated.
    - FIFO eviction at MAX_ARTIFACTS_PER_SCAN so a 100-turn scan cannot
      grow without bound. Evicted artifacts are often still on disk in
      `data/evidence_spills/`, and `fetch_tool_output` falls back to the
      spill file when the in-memory copy is gone.
    - Stores the SAME body the LLM saw (pre-truncation), not the DB row:
      no DB round-trip, no ORM in the hot path.
    - Cleared by `clear_scan_artifacts(scan_id)` at scan end.

Usage
-----
    from app.agents.tool_artifacts import record_artifact, get_artifact

    # tool_bridge.execute_tool_call(), after the body is built:
    record_artifact(
        scan_id, execution.id, tool_name, body,
        spill_path=(execution.result or {}).get("spill_path"),
        command=cmd_str,
    )

    # react_agent's fetch_tool_output handler:
    art = get_artifact(scan_id, execution_id)
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# Cap on retained artifacts per scan (FIFO eviction beyond this).
MAX_ARTIFACTS_PER_SCAN = 40

# Hard cap on bytes retained per artifact. The executor already caps output
# at 50 KB, so this is a safety net for `to_llm_dict()` callers that bypass it.
MAX_ARTIFACT_CHARS = 200_000

# ── Per-scan state ────────────────────────────────────────────────────
# scan_id -> {execution_id: artifact dict}
_ARTIFACTS: dict[str, dict[str, dict[str, Any]]] = {}
# scan_id -> {tool_name: execution_id}  (most recent execution per tool)
_LATEST_BY_TOOL: dict[str, dict[str, str]] = {}


def _get_scan_store(scan_id: str) -> dict[str, dict[str, Any]]:
    store = _ARTIFACTS.get(scan_id)
    if store is None:
        store = {}
        _ARTIFACTS[scan_id] = store
    return store


def record_artifact(
    scan_id: str,
    execution_id: str,
    tool_name: str,
    body: str,
    *,
    spill_path: str | None = None,
    command: str = "",
) -> None:
    """Store one execution's full output for later retrieval.

    Args:
        scan_id: scan scope (artifacts are per-scan).
        execution_id: ExecutionService id — the key the agent sees in
            the `[execution_id] ...` footer of every tool result.
        tool_name: tool that produced the output.
        body: the full output string (pre-LLM truncation).
        spill_path: optional path to the on-disk spill file, used as a
            fallback when the in-memory copy has been evicted.
        command: the command line, so a retrieved artifact is auditable.
    """
    if not scan_id or not execution_id:
        return

    store = _get_scan_store(scan_id)

    # FIFO eviction — keep the newest MAX_ARTIFACTS_PER_SCAN entries.
    if execution_id not in store and len(store) >= MAX_ARTIFACTS_PER_SCAN:
        oldest = min(store.items(), key=lambda kv: kv[1].get("created", 0.0))
        store.pop(oldest[0], None)
        logger.debug(
            "tool_artifact_evicted | scan=%s | execution=%s (kept %d)",
            scan_id, oldest[0], len(store),
        )

    text = body or ""
    clipped = False
    if len(text) > MAX_ARTIFACT_CHARS:
        text = text[:MAX_ARTIFACT_CHARS]
        clipped = True

    store[execution_id] = {
        "execution_id": execution_id,
        "tool": tool_name,
        "command": command,
        "body": text,
        "total_chars": len(body or ""),
        "clip_note": (
            f"\n... [artifact itself capped at {MAX_ARTIFACT_CHARS} chars]"
            if clipped else ""
        ),
        "spill_path": spill_path,
        "created": time.time(),
    }
    _LATEST_BY_TOOL.setdefault(scan_id, {})[tool_name] = execution_id

    logger.debug(
        "tool_artifact_recorded | scan=%s | tool=%s | execution=%s | chars=%d | spill=%s",
        scan_id, tool_name, execution_id, len(text), bool(spill_path),
    )


def get_artifact(scan_id: str, execution_id: str) -> dict[str, Any] | None:
    """Look up one artifact by execution_id (None when unknown/evicted)."""
    if not scan_id or not execution_id:
        return None
    return _get_scan_store(scan_id).get(execution_id)


def find_latest(scan_id: str, tool_name: str) -> dict[str, Any] | None:
    """Return the most recent artifact for `tool_name` in this scan."""
    if not scan_id or not tool_name:
        return None
    execution_id = _LATEST_BY_TOOL.get(scan_id, {}).get(tool_name)
    if not execution_id:
        # Fall back to a reverse scan — covers calls recorded without
        # going through the latest-by-tool index (e.g. older artifacts).
        store = _get_scan_store(scan_id)
        matches = [a for a in store.values() if a.get("tool") == tool_name]
        if not matches:
            return None
        return max(matches, key=lambda a: a.get("created", 0.0))
    return _get_scan_store(scan_id).get(execution_id)


def list_artifacts(scan_id: str) -> list[dict[str, Any]]:
    """Summaries (no bodies) of the artifacts retained for this scan."""
    store = _get_scan_store(scan_id)
    ordered = sorted(store.values(), key=lambda a: a.get("created", 0.0))
    return [
        {
            "execution_id": a["execution_id"],
            "tool": a.get("tool", ""),
            "chars": a.get("total_chars", len(a.get("body") or "")),
        }
        for a in ordered
    ]


def read_spill(spill_path: str | None) -> str | None:
    """Read a spill file from disk (best-effort; None on any failure)."""
    if not spill_path:
        return None
    try:
        path = Path(spill_path)
        if not path.is_file():
            return None
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        logger.debug("spill_read_failed | path=%s | err=%s", spill_path, exc)
        return None


def clear_scan_artifacts(scan_id: str) -> None:
    """Drop all artifacts for a scan (called when the scan finishes)."""
    dropped = _ARTIFACTS.pop(scan_id, None)
    _LATEST_BY_TOOL.pop(scan_id, None)
    if dropped:
        logger.info(
            "tool_artifacts_cleared | scan=%s | count=%d", scan_id, len(dropped),
        )


__all__ = [
    "MAX_ARTIFACTS_PER_SCAN",
    "MAX_ARTIFACT_CHARS",
    "record_artifact",
    "get_artifact",
    "find_latest",
    "list_artifacts",
    "read_spill",
    "clear_scan_artifacts",
]
