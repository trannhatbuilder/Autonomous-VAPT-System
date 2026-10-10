"""
VAPT-AI Scan Worker — Celery-based scan execution.

Architecture:
    FastAPI → Celery Queue → N Worker Processes → app.pentest.scan_pipeline.run_scan_pipeline()

    Each scan is 1 Celery task. Workers run independently.
    Scale workers: `celery -A workers worker --concurrency=4`

Usage:
    celery -A workers worker --loglevel=info --concurrency=4
"""

from __future__ import annotations

import json
import os
import sys

# Ensure project root is on sys.path so Celery workers can import `app.*`
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
from datetime import datetime, timezone

# Celery setup
from celery import Celery
from celery.signals import worker_process_init, worker_process_shutdown

# Redis connection
REDIS_URL = os.getenv("EVVO_REDIS_URL", "redis://localhost:6379/0")

# Create Celery app
app = Celery(
    "evvo_scans",
    broker=REDIS_URL,
    backend=REDIS_URL,
)

# Celery configuration — optimized for long-running LLM scans
app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=3600,            # 1 hour max per scan
    task_soft_time_limit=3300,       # 55 minutes soft limit
    worker_prefetch_multiplier=1,    # One task per worker at a time (long tasks!)
    task_acks_late=True,             # Acknowledge after completion (not before)
    task_reject_on_worker_lost=True, # Re-queue if worker crashes
    result_expires=86400,            # Results expire after 24 hours
    worker_max_tasks_per_child=50,   # Restart worker after 50 tasks (memory leak prevention)
    broker_connection_retry_on_startup=True,
)

import redis as redis_lib

_redis_client = None

def get_redis():
    """Get Redis client (lazy initialization)."""
    global _redis_client
    if _redis_client is None:
        _redis_client = redis_lib.from_url(REDIS_URL, decode_responses=True)
    return _redis_client


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ============================================================
# Redis-based Scan Event System
# ============================================================

def append_event_redis(scan_id: str, event_type: str, line: str, progress: int = None, payload: dict = None) -> None:
    """Append event to scan job in Redis (used by Celery workers)."""
    r = get_redis()
    event_key = f"scan:{scan_id}:events"
    event = {
        "type": event_type,
        "line": line,
        "progress": progress,
        "payload": payload,
        "timestamp": utc_now(),
    }
    r.rpush(event_key, json.dumps(event))
    r.expire(event_key, 86400)  # 24 hour expiry


def set_scan_state(scan_id: str, state: dict) -> None:
    """Set scan state in Redis."""
    r = get_redis()
    state_key = f"scan:{scan_id}:state"
    r.set(state_key, json.dumps(state, default=str), ex=86400)


def get_scan_state(scan_id: str) -> dict | None:
    """Get scan state from Redis."""
    r = get_redis()
    state_key = f"scan:{scan_id}:state"
    data = r.get(state_key)
    if data:
        return json.loads(data)
    return None


# ============================================================
# Checkpoint System
# ============================================================

def save_checkpoint(scan_id: str, phase: str, progress: int, data: dict = None) -> None:
    """Save scan checkpoint to Redis for resume support."""
    r = get_redis()
    checkpoint_key = f"scan:{scan_id}:checkpoint"
    checkpoint = {
        "phase": phase,
        "progress": progress,
        "data": data or {},
        "timestamp": utc_now(),
    }
    r.set(checkpoint_key, json.dumps(checkpoint), ex=604800)  # 7 day expiry


def get_checkpoint(scan_id: str) -> dict | None:
    """Get scan checkpoint from Redis."""
    r = get_redis()
    checkpoint_key = f"scan:{scan_id}:checkpoint"
    data = r.get(checkpoint_key)
    if data:
        return json.loads(data)
    return None


def clear_checkpoint(scan_id: str) -> None:
    """Clear scan checkpoint after completion."""
    r = get_redis()
    r.delete(f"scan:{scan_id}:checkpoint")



def is_scan_cancelled(scan_id: str) -> bool:
    try:
        state = get_scan_state(scan_id) or {}
        return state.get("status") == "cancelled"
    except Exception:
        return False

def _raise_if_cancelled(scan_id: str) -> None:
    if is_scan_cancelled(scan_id):
        raise RuntimeError("Scan cancelled by user")

# ============================================================
# Celery Tasks
# ============================================================

@app.task(name="scan.cancel")
def cancel_scan_task(scan_id: str):
    """Cancel a running scan."""
    set_scan_state(scan_id, {"status": "cancelled", "cancelled_at": utc_now()})
    append_event_redis(scan_id, "status", "[STATUS]: Scan cancelled by user.", None, None)
    clear_checkpoint(scan_id)
    return {"scan_id": scan_id, "status": "cancelled"}


@app.task(name="scan.cleanup")
def cleanup_old_scans(max_age_hours: int = 24):
    """Cleanup old scan data from Redis."""
    r = get_redis()
    scan_keys = r.keys("scan:*:state")
    cleaned = 0
    for key in scan_keys:
        data = r.get(key)
        if data:
            state = json.loads(data)
            if state.get("status") in ["completed", "failed", "cancelled"]:
                r.delete(key)
                cleaned += 1
    return {"cleaned": cleaned}


@app.task(bind=True, name="scan.run_vapt", max_retries=2, default_retry_delay=30)
def run_vapt_scan_task(
    self,
    target: str,
    user_prompt: str = "",
    mode: str = "auto",
    transfer_targets: list[str] | None = None,
    scan_id: str | None = None,
    user_id: str | None = None,
    findings_override: list[dict] | None = None,
):
    """Execute the VAPT-AI v3.2 end-to-end scan pipeline (W13).

    Args:
        target: Target URL or IP (e.g. "http://localhost:8080/")
        user_prompt: Natural-language scan goal
        mode: Orchestration mode — "auto" | "supervisor" | "deep" | "plan_execute"
        transfer_targets: Optional override for the agent transfer sequence.
            Defaults to the 6-phase kill-chain (see scan_pipeline.DEFAULT_TRANSFER_TARGETS).
        scan_id: Optional scan ID (auto-generated if None)
        user_id: Optional user UUID (string form) for Scan.user_id FK
        findings_override: Optional list of finding dicts (for testing).
            Each dict must match PipelineFinding fields. When provided,
            the orchestrator is bypassed and findings are fed directly
            into the auditor + persistence + report stages.

    Returns:
        Dict matching ScanPipelineResult.to_dict() — includes scan_id,
        status, findings_total, findings_by_severity, report_pdf_path,
        report_sarif_path, duration_seconds.
    """
    import asyncio
    import uuid as uuid_mod
    from app.pentest.scan_pipeline import (
        run_scan_pipeline,
        PipelineFinding,
    )

    # Convert dict findings → PipelineFinding dataclasses
    pipeline_findings: list[PipelineFinding] | None = None
    if findings_override:
        pipeline_findings = [
            PipelineFinding(**f) for f in findings_override
        ]

    # Convert user_id string → UUID
    user_uuid = uuid_mod.UUID(user_id) if user_id else None

    try:
        result = asyncio.run(run_scan_pipeline(
            target=target,
            user_prompt=user_prompt,
            mode=mode,
            transfer_targets=transfer_targets,
            scan_id=scan_id,
            user_id=user_uuid,
            findings_override=pipeline_findings,
        ))
        return result.to_dict()

    except Exception as exc:
        # Append error event to Redis for SSE consumers
        try:
            sid = scan_id or "unknown"
            append_event_redis(sid, "error",
                f"[STATUS]: VAPT pipeline failed: {exc}",
                None, {"error": str(exc)})
            set_scan_state(sid, {
                "status": "failed",
                "completed_at": utc_now(),
                "error": str(exc),
            })
        except Exception:
            pass

        if self.request.retries < self.max_retries:
            raise self.retry(exc=exc, countdown=30 * (2 ** self.request.retries))
        raise


# ============================================================
# Worker Signals
# ============================================================

@worker_process_init.connect
def on_worker_init(**kwargs):
    """Initialize worker resources (DB pool, etc.)."""
    # Ensure project root is on sys.path for forked worker processes
    if _PROJECT_ROOT not in sys.path:
        sys.path.insert(0, _PROJECT_ROOT)
    print(f"[Celery Worker] Initialized — PID: {os.getpid()} (sys.path includes {_PROJECT_ROOT})")


@worker_process_shutdown.connect
def on_worker_shutdown(**kwargs):
    """Cleanup worker resources."""
    global _redis_client
    if _redis_client:
        _redis_client.close()
        _redis_client = None
    print("[Celery Worker] Shutdown complete")


# ============================================================
# Main entry point
# ============================================================

if __name__ == "__main__":
    app.start()