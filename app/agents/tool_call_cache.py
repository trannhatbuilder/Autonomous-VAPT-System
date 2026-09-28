"""
VAPT-AI Tool Call Cache — Phase F5 (CyberStrikeAI-aligned).

Reduces wasted iterations in the agent loop by caching:

1. **Tool availability** — if a binary returned "Binary not found" or
   "[Errno 13] Permission denied" once during a scan, subsequent calls
   with the same tool name return the cached error immediately — no
   subprocess spawn, no LLM token wasted waiting for the same failure.

2. **Tool call results** — if the LLM calls the same tool with the
   same arguments twice in the same scan (a common pattern when the
   LLM forgets it already tried something), the second call returns
   the cached result from the first call — no re-execution of an
   expensive nmap/katana/feroxbuster scan.

3. **Scope-violation results** — if a tool was blocked by the scope
   guard once, future calls with the same target pattern return the
   cached block notice (the scope doesn't change mid-scan).

Design choices (mirror CyberStrikeAI pattern):
    - Cache is PER-SCAN (not global). Each scan has its own cache
      isolated in a singleton dict keyed by scan_id.
    - Cache is IN-MEMORY only (no DB persistence — would defeat the
      purpose of fast lookup). Cleared when scan completes/aborts.
    - Cache HITS produce a different log line than MISSES so we can
      verify the cache is working in journalctl.
    - Cache is OPT-IN by signature: only tools with stable args
      (target + a few key params) are cached. Tools with random
      parameters (e.g. `additional_args` with timestamps) are not
      cached to avoid false hits.

CyberStrikeAI's analogous logic:
    - `eino_pending_orphaned` event force-completes unclosed tool
      calls at run end (internal/handler/eino_run_completion_handler.go:61)
    - `seenToolCallSigs` idempotency map in agent.go:1162-1181 —
      prevents the same tool_call from being persisted twice
    - Tool result guard with spill-to-disk (we don't need that here
      since we cache by signature, not by content size)

Usage:
    from app.agents.tool_call_cache import (
        get_cached_unavailable,
        mark_unavailable,
        get_cached_result,
        cache_result,
        is_scope_blocked,
        mark_scope_blocked,
        clear_scan_cache,
    )

    # In agent loop, before calling execute_tool_call:
    cached_err = get_cached_unavailable(scan_id, tool_name)
    if cached_err:
        return cached_err  # skip subprocess spawn

    cached = get_cached_result(scan_id, tool_name, tool_args)
    if cached is not None:
        return cached  # skip subprocess spawn

    # ... execute tool ...

    if "[error] Binary not found" in output or "[Errno 13]" in output:
        mark_unavailable(scan_id, tool_name, output)
    elif "SCOPE VIOLATION" in output:
        mark_scope_blocked(scan_id, tool_name, tool_args, output)
    else:
        cache_result(scan_id, tool_name, tool_args, output)
"""
from __future__ import annotations

import hashlib
import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


# ── Per-scan cache state ────────────────────────────────────────────────
# Keyed by scan_id so each scan has its own isolated cache.
# Each scan's cache has 3 dicts:
#   - unavailable: tool_name -> cached error string (binary missing/blocked)
#   - results: signature -> cached output string (full tool result)
#   - scope_blocked: signature -> cached block message (scope violation)

_SCAN_CACHES: dict[str, dict[str, dict[str, str]]] = {}


def _get_scan_cache(scan_id: str) -> dict[str, dict[str, str]]:
    """Get or create the per-scan cache dict."""
    if scan_id not in _SCAN_CACHES:
        _SCAN_CACHES[scan_id] = {
            "unavailable": {},      # tool_name -> cached error string
            "results": {},         # signature -> cached output string
            "scope_blocked": {},   # signature -> cached block message
        }
    return _SCAN_CACHES[scan_id]


def clear_scan_cache(scan_id: str) -> None:
    """Drop the cache for a scan. Called when scan completes/aborts."""
    if scan_id in _SCAN_CACHES:
        cache = _SCAN_CACHES[scan_id]
        logger.info(
            "Tool call cache cleared | scan=%s | unavailable=%d | results=%d | scope_blocked=%d",
            scan_id, len(cache["unavailable"]), len(cache["results"]), len(cache["scope_blocked"]),
        )
        del _SCAN_CACHES[scan_id]


# ── Tool availability cache ──────────────────────────────────────────────


def get_cached_unavailable(scan_id: str, tool_name: str) -> str | None:
    """Check if a tool was previously marked unavailable for this scan.

    Returns the cached error string if the tool was marked unavailable,
    or None if the tool is considered available (or hasn't been tried yet).
    """
    cache = _get_scan_cache(scan_id)
    cached = cache["unavailable"].get(tool_name)
    if cached is not None:
        logger.info(
            "TOOL_AVAILABILITY_CACHE_HIT | scan=%s | tool=%s | returning cached error",
            scan_id, tool_name,
        )
    return cached


def mark_unavailable(scan_id: str, tool_name: str, error_output: str) -> None:
    """Mark a tool as unavailable for this scan.

    Subsequent calls to get_cached_unavailable(scan_id, tool_name) will
    return the cached error_output — skipping the subprocess spawn
    that would produce the same error.

    Called when the tool output contains:
        - "Binary not found"
        - "[Errno 13] Permission denied"
        - Other OS-level binary permission errors
    """
    cache = _get_scan_cache(scan_id)
    cache["unavailable"][tool_name] = error_output
    logger.warning(
        "TOOL_MARKED_UNAVAILABLE | scan=%s | tool=%s | future calls will skip subprocess",
        scan_id, tool_name,
    )


# ── Tool call result cache ──────────────────────────────────────────────


def _compute_call_signature(tool_name: str, tool_args: dict[str, Any]) -> str:
    """Compute a stable signature for a tool call.

    Format: sha256(tool_name + sorted args JSON)[:16]

    Used as the cache key so that two calls with the same tool_name +
    same args produce the same signature → cache hit.

    Args that are NOT included in the signature (because they vary
    randomly between calls):
        - additional_args with timestamps
        - random seeds
        - session IDs

    For now we include ALL args — the LLM is unlikely to pass timestamps
    or random seeds to security tools.
    """
    # Sort keys for deterministic JSON
    args_json = json.dumps(tool_args, sort_keys=True, default=str, ensure_ascii=False)
    raw = f"{tool_name}|{args_json}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def get_cached_result(
    scan_id: str, tool_name: str, tool_args: dict[str, Any]
) -> str | None:
    """Check if the same tool call was already executed in this scan.

    Returns the cached output string if found, or None if no cached
    result exists for this (tool_name, tool_args) combination.
    """
    cache = _get_scan_cache(scan_id)
    signature = _compute_call_signature(tool_name, tool_args)
    cached = cache["results"].get(signature)
    if cached is not None:
        logger.info(
            "TOOL_RESULT_CACHE_HIT | scan=%s | tool=%s | sig=%s | returning cached result",
            scan_id, tool_name, signature,
        )
    return cached


def cache_result(
    scan_id: str, tool_name: str, tool_args: dict[str, Any], output: str
) -> None:
    """Cache the output of a successful tool call.

    Future calls with the same (tool_name, tool_args) will return this
    cached output instead of re-executing the subprocess.

    Called after a successful tool execution (not for errors — those
    go to mark_unavailable or mark_scope_blocked instead).
    """
    cache = _get_scan_cache(scan_id)
    signature = _compute_call_signature(tool_name, tool_args)
    cache["results"][signature] = output
    logger.debug(
        "TOOL_RESULT_CACHED | scan=%s | tool=%s | sig=%s | output_len=%d",
        scan_id, tool_name, signature, len(output),
    )


# ── Scope-blocked cache ──────────────────────────────────────────────────


def is_scope_blocked(
    scan_id: str, tool_name: str, tool_args: dict[str, Any]
) -> str | None:
    """Check if a (tool, args) combo was previously blocked by the scope guard.

    Returns the cached block message if blocked before, or None.
    """
    cache = _get_scan_cache(scan_id)
    signature = _compute_call_signature(tool_name, tool_args)
    cached = cache["scope_blocked"].get(signature)
    if cached is not None:
        logger.info(
            "SCOPE_BLOCK_CACHE_HIT | scan=%s | tool=%s | sig=%s | returning cached block",
            scan_id, tool_name, signature,
        )
    return cached


def mark_scope_blocked(
    scan_id: str, tool_name: str, tool_args: dict[str, Any], block_message: str
) -> None:
    """Mark a (tool, args) combo as scope-blocked for this scan.

    Subsequent calls with the same args will return the cached block
    message without spawning a subprocess (which would also be blocked).
    """
    cache = _get_scan_cache(scan_id)
    signature = _compute_call_signature(tool_name, tool_args)
    cache["scope_blocked"][signature] = block_message
    logger.info(
        "SCOPE_BLOCK_CACHED | scan=%s | tool=%s | sig=%s",
        scan_id, tool_name, signature,
    )


__all__ = [
    "get_cached_unavailable",
    "mark_unavailable",
    "get_cached_result",
    "cache_result",
    "is_scope_blocked",
    "mark_scope_blocked",
    "clear_scan_cache",
]