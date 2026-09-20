"""
EVVO Scan Worker - Celery-based scan execution with concurrent scaling.

Architecture:
    FastAPI → Celery Queue → N Worker Processes → run_llm_scan()
    
    Each scan is 1 Celery task. Workers run independently.
    Scale workers: `celery -A workers.scan_tasks worker --concurrency=4`

Usage:
    celery -A workers.scan_tasks worker --loglevel=info --concurrency=4
"""

from __future__ import annotations

import json
import os
import sys
import time

# Ensure project root is on sys.path so Celery workers can import `routes`, `models`, etc.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
from datetime import datetime, timezone
from typing import Any

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
# Core Scan Pipeline — runs inside Celery worker
# ============================================================

def _execute_scan_pipeline(scan_id: str, target_url: str,
                            scan_mode: str = "blackbox",
                            greybox_username: str | None = None,
                            greybox_password: str | None = None) -> dict[str, Any]:
    """
    Execute the real LLM-powered scan pipeline.
    
    This runs inside a Celery worker process — completely isolated from the
    FastAPI event loop. Multiple scans run in parallel across workers.
    
    Pipeline:
        Phase 1: HTTP Reconnaissance (httpx)           ~5-15s
        Phase 1b: Automated vulnerability tests         ~10-30s
        Phase 2: LLM Security Analysis (GPT/Claude)    ~30-90s
        Phase 3: LLM Report Generation (GPT/Claude)    ~20-60s
        Total: ~1-5 minutes per scan
    """
    import requests as _unused  # noqa: F401 — ensure requests available
    from routes.deps.scanner import run_llm_scan
    from routes.deps.scan_helpers import append_event, build_report_preview
    from routes.deps.scan_state import (
        update_runtime_scan_job, get_runtime_scan_job,
    )
    from routes.deps.database import db_connection
    from routes.deps.utils import utc_now as _utc_now_dt
    from routes.deps.security import audit_logger, decrypt_greybox_credentials

    # Update state to running
    started_at = _utc_now_dt()
    update_runtime_scan_job(scan_id, status="running", started_at=started_at, progress=0)
    
    # Also update Redis state for cross-process visibility
    set_scan_state(scan_id, {
        "status": "running",
        "started_at": utc_now(),
        "scan_mode": scan_mode,
    })
    
    save_checkpoint(scan_id, "starting", 5)

    # ── Decrypt greybox credentials if present ──
    if scan_mode == "greybox" and greybox_username is None:
        try:
            from routes.deps.database import db_readonly_connection
            with db_readonly_connection() as conn:
                cursor = conn.cursor()
                try:
                    cursor.execute("SELECT credentials_encrypted FROM scans WHERE id = %s", (scan_id,))
                    row = cursor.fetchone()
                finally:
                    cursor.close()
            if row and row[0]:
                greybox_username, greybox_password = decrypt_greybox_credentials(row[0])
                audit_logger.info(f"CELERY_GREYBOX_CREDS_LOADED | scan={scan_id}")
        except Exception as e:
            audit_logger.warning(f"CELERY_GREYBOX_CREDS_FAILED | scan={scan_id} | {e}")

    save_checkpoint(scan_id, "scanning", 10)
    _raise_if_cancelled(scan_id)

    # ── Run the real LLM scan pipeline ──
    start_time = time.time()
    try:
        result = run_llm_scan(
            target_url, "full", scan_id,
            greybox_username=greybox_username,
            greybox_password=greybox_password,
        )
    except Exception as exc:
        # Scan engine failed
        completed_at = _utc_now_dt()
        error_msg = str(exc)
        update_runtime_scan_job(scan_id, status="failed", completed_at=completed_at, error=error_msg)
        set_scan_state(scan_id, {"status": "failed", "error": error_msg, "completed_at": utc_now()})
        append_event(scan_id, "error", f"[STATUS]: Scan failed: {exc}", None, {"error": error_msg})
        
        # Update DB record
        try:
            from routes.deps.scan_helpers import update_scan_record
            update_scan_record(scan_id, status="failed", completed_at=completed_at, error=error_msg)
        except Exception:
            pass
        
        raise

    _raise_if_cancelled(scan_id)

    # ── Process results ──
    preview = build_report_preview(result)
    completed_at = _utc_now_dt()
    execution_ms = int((time.time() - start_time) * 1000)
    findings_count = len(result.get("vulnerabilities", []))

    # Update in-memory state (for SSE streaming)
    update_runtime_scan_job(
        scan_id, status="completed", completed_at=completed_at,
        result=result, preview=preview, progress=100,
    )
    
    # Update Redis state
    set_scan_state(scan_id, {
        "status": "completed",
        "completed_at": utc_now(),
        "execution_ms": execution_ms,
        "findings_count": findings_count,
        "risk_level": result.get("summary", {}).get("risk_level", "Unknown"),
    })

    # Update DB record
    try:
        from routes.deps.scan_helpers import update_scan_record
        update_scan_record(
            scan_id, status="completed", completed_at=completed_at,
            progress=100,
            result_json=json.dumps(result, ensure_ascii=False),
            preview_json=json.dumps(preview, ensure_ascii=False),
        )
    except Exception as exc:
        audit_logger.warning(f"CELERY_DB_UPDATE_FAILED | scan={scan_id} | {exc}")

    # ── Persist findings to findings table (Findings DB v2) ──
    try:
        from models.findings import bulk_create_findings
        from routes.deps.database import db_connection
        # Get engagement_id from scans table
        engagement_id = None
        with db_connection() as conn:
            c = conn.cursor()
            try:
                c.execute("SELECT engagement_id FROM scans WHERE id = %s", (scan_id,))
                row = c.fetchone()
                if row:
                    engagement_id = row[0]
            finally:
                c.close()
        vulnerabilities = result.get("vulnerabilities", [])
        if vulnerabilities:
            inserted = bulk_create_findings(scan_id, engagement_id, vulnerabilities)
            audit_logger.info(f"CELERY_FINDINGS_PERSISTED | scan={scan_id} | count={inserted}")
    except Exception as exc:
        audit_logger.warning(f"CELERY_FINDINGS_PERSIST_FAILED | scan={scan_id} | {exc}")

    # ── Record metrics ──
    try:
        from shield_engine.metrics import metrics as shield_metrics
        shield_metrics.record_scan(execution_ms, findings_count)
    except Exception:
        pass

    try:
        from routes.deps.scan_helpers import update_scan_record
        from routes.deps.scan_state import get_runtime_scan_job
        from routes.deps.database import db_connection
        from routes.deps.config import get_model_runtime_config
        # Persist to scan_metrics table
        with db_connection() as conn:
            c = conn.cursor()
            try:
                model_config = get_model_runtime_config()
                c.execute(
                    "INSERT INTO scan_metrics (scan_id, model_version, execution_time_ms, findings_count) VALUES (%s,%s,%s,%s)",
                    (scan_id, model_config.get("model", "unknown"), execution_ms, findings_count),
                )
                conn.commit()
            finally:
                c.close()
    except Exception as exc:
        audit_logger.warning(f"CELERY_METRICS_PERSIST_FAILED | scan={scan_id} | {exc}")

    # ── Send email notification ──
    try:
        job = get_runtime_scan_job(scan_id) or {}
        user_id = job.get("user_id")
        if user_id:
            from routes.deps.scan_helpers import get_user_by_id, get_user_settings
            user = get_user_by_id(user_id)
            if user:
                settings = get_user_settings(str(user_id))
                if settings.get("email_notifications", False):
                    from workers.email_service import EmailService
                    EmailService().send_scan_complete(user["email"], result)
    except Exception as exc:
        audit_logger.warning(f"CELERY_EMAIL_FAILED | scan={scan_id} | {exc}")

    clear_checkpoint(scan_id)
    audit_logger.info(f"CELERY_SCAN_COMPLETE | scan={scan_id} | time={execution_ms}ms | findings={findings_count}")

    return {
        "scan_id": scan_id,
        "status": "completed",
        "execution_ms": execution_ms,
        "findings_count": findings_count,
        "result": result,
    }


def _execute_public_scan_pipeline(scan_id: str, target_url: str,
                                    user_id: str | None = None) -> dict[str, Any]:
    """
    Execute public pre-AI phase (target validation only, caps at 25%).
    Runs inside Celery worker.
    """
    from routes.deps.scanner import build_pre_ai_placeholder_result
    from routes.deps.scan_state import (
        update_runtime_scan_job,
    )
    from routes.deps.utils import utc_now as _utc_now_dt
    from routes.deps.security import audit_logger
    from shield_engine.metrics import metrics as shield_metrics

    started_at = _utc_now_dt()
    update_runtime_scan_job(scan_id, status="running", started_at=started_at)
    set_scan_state(scan_id, {"status": "running", "started_at": utc_now()})

    start_time = time.time()
    try:
        append_event_redis(scan_id, "status", f"🔍 Validating target for AI pentest: {target_url}", 5)
        result = build_pre_ai_placeholder_result(target_url)
    except Exception as exc:
        completed_at = _utc_now_dt()
        error_msg = str(exc)
        update_runtime_scan_job(scan_id, status="failed", completed_at=completed_at, error=error_msg)
        set_scan_state(scan_id, {"status": "failed", "error": error_msg})
        raise

    preview = {
        "risk_level": "Scanning",
        "risk_score": 0,
        "total_vulnerabilities": 0,
        "total_findings": 0,
        "severity_distribution": result.get("severity_distribution", {}),
        "scan_type": "ai_pentest_pending",
        "target_url": target_url,
        "generated_at": utc_now(),
        "narrative": result.get("summary", {}).get("narrative", ""),
    }

    completed_at = _utc_now_dt()
    update_runtime_scan_job(
        scan_id, status="paused_at_25", completed_at=completed_at,
        result=result, preview=preview, progress=25,
    )
    set_scan_state(scan_id, {"status": "paused_at_25", "completed_at": utc_now()})

    # Update DB if user is authenticated
    if user_id:
        try:
            from routes.deps.scan_helpers import update_scan_record
            update_scan_record(
                scan_id, status="paused_at_25",
                completed_at=completed_at, progress=25,
                result_json=json.dumps(result, ensure_ascii=False),
                preview_json=json.dumps(preview, ensure_ascii=False),
            )
        except Exception:
            pass

    append_event_redis(scan_id, "nuclei_done", "✅ Target validation complete — Continuing with AI pentest.", 25, {"result": result})

    execution_ms = int((time.time() - start_time) * 1000)
    findings_count = len(result.get("vulnerabilities", []))
    shield_metrics.record_scan(execution_ms, findings_count)
    audit_logger.info(f"CELERY_PUBLIC_SCAN_COMPLETE | scan={scan_id} | time={execution_ms}ms | findings={findings_count}")

    return {"scan_id": scan_id, "status": "paused_at_25", "execution_ms": execution_ms}


# ============================================================
# Celery Tasks
# ============================================================

@app.task(bind=True, name="scan.run_llm", max_retries=3, default_retry_delay=60)
def run_scan_task(self, scan_id: str, target_url: str, user_id: str = None,
                   scan_mode: str = "blackbox",
                   greybox_username: str = None, greybox_password: str = None):
    """
    Execute full LLM-powered VAPT scan with retry and checkpoint support.
    
    Each task = 1 scan = 1 independent session.
    Models receive fresh tasks — no session overlap.
    
    Concurrency is controlled by --concurrency flag:
        celery -A workers.scan_tasks worker --concurrency=8
    """
    # Check for existing checkpoint (resume support)
    checkpoint = get_checkpoint(scan_id)
    if checkpoint:
        self.update_state(
            state="PROGRESS",
            meta={
                "phase": checkpoint["phase"],
                "progress": checkpoint["progress"],
                "message": f"Resuming from checkpoint: {checkpoint['phase']}",
            },
        )

    try:
        result = _execute_scan_pipeline(
            scan_id, target_url, scan_mode,
            greybox_username, greybox_password,
        )
        return result

    except Exception as exc:
        if self.request.retries < self.max_retries:
            save_checkpoint(scan_id, "retry_pending", self.request.retries * 25, {"error": str(exc)})
            raise self.retry(exc=exc, countdown=60 * (2 ** self.request.retries))

        # Max retries exceeded
        append_event_redis(scan_id, "error",
            f"[STATUS]: Scan failed after {self.max_retries} retries: {exc}",
            None, {"error": str(exc)})
        set_scan_state(scan_id, {
            "status": "failed",
            "completed_at": utc_now(),
            "error": str(exc),
            "retry_count": self.max_retries,
        })
        raise


@app.task(name="scan.run_public")
def run_public_scan_task(scan_id: str, target_url: str, user_id: str = None):
    """
    Execute public/basic Nuclei scan.
    Caps at 25% — full report requires login.
    """
    return _execute_public_scan_pipeline(scan_id, target_url, user_id)


@app.task(name="scan.retry", bind=True)
def retry_scan_task(self, scan_id: str, target_url: str):
    """Manually retry a failed scan."""
    clear_checkpoint(scan_id)
    set_scan_state(scan_id, {"status": "queued", "retry_initiated": utc_now()})
    run_scan_task.delay(scan_id, target_url)
    return {"scan_id": scan_id, "status": "queued"}


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


# ============================================================
# VAPT-AI v3.2 Pipeline Task (W13-S2)
# ============================================================
#
# This is the W13 entry point for the VAPT-AI end-to-end scan pipeline
# (app.pentest.scan_pipeline.run_scan_pipeline). It runs the full
# kill-chain: consent_check → blackboard init → orchestrator (LangGraph
# supervisor with 6-phase transfer_targets) → auditor → persist →
# report (PDF + SARIF) → cleanup.
#
# Unlike the EVVO legacy tasks above (run_scan_task, run_public_scan_task)
# which depend on routes.deps.* + shield_engine + Postgres, this task
# only depends on the VAPT-AI v3.2 stack: app.db.* (SQLite async) +
# app.orchestration.* + app.harness.* + app.evidence.* + app.report.*
#
# The task is sync (Celery requirement) but wraps an asyncio.run() call
# to the async pipeline. Cancellation, retry, and checkpoint behavior
# match the EVVO task pattern for consistency.

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