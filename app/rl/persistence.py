"""
SQL-backed RL checkpoint persistence for VAPT-AI.

This module bridges the in-memory DoubleQLearner/PentestPolicy with the
vapt_rl_checkpoints SQL table (defined in app/db/models/rl.py at W1-D).

Two-tier persistence:
  1. .npz cache at settings.rl_dir / "q_weights.npz" (fast-load, hot-path)
  2. SQL vapt_rl_checkpoints (durable, cross-restart)

Strategy:
  - Startup: try .npz cache first (fast), fall back to SQL if cache missing
  - After every scan: save_checkpoint() writes to BOTH .npz + SQL
  - Hash verification (SHA-256 of weights_blob) catches corruption

Storage format for weights_blob:
  - np.save(io.BytesIO(), weights_dict) → raw numpy bytes
  - zlib.compress(raw_bytes, level=6) → compressed blob
  - Stored as LargeBinary in vapt_rl_checkpoints.weights_blob

Storage format for q_meta_json:
  {
    "step": 1234,
    "epsilon": 0.15,
    "temperature": 0.5,
    "exploration_mode": "epsilon_greedy",
    "episodes": 42,
    "lr": 0.001,
    "gamma": 0.95,
    "tau": 0.005,
    "state_dim": 337,
    "action_dim": 7,
    "hidden_dim": 64,
    "architecture": "dueling_double_q",
    "total_params": 43784,
    "total_reward": 187.5,
    "scan_count": 42
  }
"""
from __future__ import annotations

import hashlib
import io
import json
import logging
import zlib
from datetime import datetime, UTC
from typing import Any

import numpy as np

from app.core.config import settings
from app.rl.q_learner import DoubleQLearner
from app.rl.policy import PentestPolicy

logger = logging.getLogger(__name__)


# ── Public API ───────────────────────────────────────────────────────────

async def save_checkpoint(
    q_learner: DoubleQLearner,
    policy: PentestPolicy | None = None,
    scan_id: str | None = None,
    total_reward: float = 0.0,
    session: Any = None,  # AsyncSession | None
    is_initial: bool = False,
) -> str | None:
    """Persist the current Q-network weights + policy state to SQL + .npz cache.

    Args:
        q_learner: DoubleQLearner instance (online network weights are saved)
        policy: PentestPolicy instance (epsilon, temperature, mode saved in meta).
            If None, only Q-learner state is saved.
        scan_id: optional FK to vapt_scans.id (None for seed checkpoint)
        total_reward: cumulative reward since last checkpoint
        session: optional SQLAlchemy AsyncSession. If None, opens fresh.
        is_initial: True for seed checkpoint (no scan, just initial weights)

    Returns:
        checkpoint_id (UUID string) on success, None on failure.
    """
    # Serialize weights → compressed blob
    weights_dict = q_learner.online.get_weights()
    blob, weights_hash = _serialize_weights(weights_dict)

    # Build metadata
    meta = _build_meta(q_learner, policy, scan_id, total_reward, is_initial)

    # Always update .npz cache (fast-load for next process restart)
    q_learner.save_cache()

    # INSERT into vapt_rl_checkpoints
    try:
        from app.db.models.rl import RLCheckpoint
        from sqlalchemy import select
    except ImportError as exc:
        logger.warning("RLCheckpoint import failed — SQL save skipped: %s", exc)
        return None

    async def _do_insert(sess: Any) -> str | None:
        row = RLCheckpoint(
            scan_id=scan_id,
            weights_blob=blob,
            q_meta_json=meta,
            is_initial=is_initial,
            weights_hash=weights_hash,
            adaptive_encoder_config=None,  # EVVO legacy field, kept None for W16
        )
        sess.add(row)
        await sess.flush()
        return str(row.id)

    if session is not None:
        checkpoint_id = await _do_insert(session)
        # Caller manages commit
        logger.info(
            "RL checkpoint saved (caller-session) | scan=%s | hash=%s... | step=%d",
            scan_id, weights_hash[:12], meta.get("step", 0),
        )
        return checkpoint_id

    # No session provided — open a fresh one
    try:
        from app.db.session import async_session
        async with async_session() as sess:
            checkpoint_id = await _do_insert(sess)
            await sess.commit()
        logger.info(
            "RL checkpoint saved | scan=%s | hash=%s... | step=%d | checkpoint_id=%s",
            scan_id, weights_hash[:12], meta.get("step", 0), checkpoint_id,
        )
        return checkpoint_id
    except Exception as exc:
        logger.warning("RL checkpoint SQL save failed: %s", exc)
        return None


async def load_latest_checkpoint(
    q_learner: DoubleQLearner,
    policy: PentestPolicy | None = None,
    session: Any = None,  # AsyncSession | None
) -> dict[str, Any] | None:
    """Load the latest checkpoint from SQL (or .npz cache as fallback).

    Restoration order:
      1. Try .npz cache (fast-load) — handled by DoubleQLearner._load_from_cache()
         at construction time. This function does NOT re-load .npz.
      2. Try SQL: SELECT latest vapt_rl_checkpoints ORDER BY created_at DESC LIMIT 1
      3. If SQL checkpoint is newer than cache (or cache missing), restore from SQL
      4. Verify SHA-256 hash before applying weights

    Args:
        q_learner: DoubleQLearner instance to restore weights into
        policy: optional PentestPolicy to restore ε/temperature/mode into
        session: optional AsyncSession. If None, opens fresh.

    Returns:
        checkpoint metadata dict on success, None if no checkpoint found
        or hash verification failed.
    """
    try:
        from app.db.models.rl import RLCheckpoint
        from sqlalchemy import select
    except ImportError as exc:
        logger.warning("RLCheckpoint import failed — SQL load skipped: %s", exc)
        return None

    async def _do_select(sess: Any) -> RLCheckpoint | None:
        stmt = (
            select(RLCheckpoint)
            .order_by(RLCheckpoint.created_at.desc())
            .limit(1)
        )
        result = await sess.execute(stmt)
        return result.scalars().first()

    if session is not None:
        row = await _do_select(session)
    else:
        try:
            from app.db.session import async_session
            async with async_session() as sess:
                row = await _do_select(sess)
        except Exception as exc:
            logger.warning("RL checkpoint SQL load failed: %s", exc)
            return None

    if row is None:
        logger.info("No RL checkpoint found in DB — starting fresh")
        return None

    # Verify hash
    weights_dict = _deserialize_weights(row.weights_blob)
    actual_hash = _hash_weights(weights_dict)
    if actual_hash != row.weights_hash:
        logger.error(
            "RL checkpoint hash mismatch | expected=%s | actual=%s | scan_id=%s — refusing to load",
            row.weights_hash, actual_hash, row.scan_id,
        )
        return None

    # Apply weights to Q-learner
    try:
        q_learner.online.set_weights(weights_dict)
        q_learner.target.copy_from(q_learner.online)
    except Exception as exc:
        logger.error("Failed to apply checkpoint weights to Q-learner: %s", exc)
        return None

    # Restore Q-learner scalar state from meta
    meta = row.q_meta_json or {}
    q_learner._step = int(meta.get("step", 0))
    q_learner.lr = float(meta.get("lr", q_learner.lr))
    q_learner.gamma = float(meta.get("gamma", q_learner.gamma))
    q_learner.tau = float(meta.get("tau", q_learner.tau))

    # Restore policy state
    if policy is not None:
        policy.epsilon = float(meta.get("epsilon", policy.epsilon))
        policy.temperature = float(meta.get("temperature", policy.temperature))
        policy._episode = int(meta.get("episodes", policy._episode))

    logger.info(
        "RL checkpoint loaded | scan_id=%s | step=%d | ε=%.4f | T=%.4f | hash=%s...",
        row.scan_id, q_learner._step, policy.epsilon if policy else 0.0,
        policy.temperature if policy else 0.0, row.weights_hash[:12],
    )

    # Also update .npz cache so next process restart loads from cache (fast)
    try:
        q_learner.save_cache()
    except Exception as exc:
        logger.warning("Failed to refresh .npz cache after SQL load: %s", exc)

    return meta


async def list_checkpoints(
    limit: int = 10,
    session: Any = None,
) -> list[dict[str, Any]]:
    """List recent RL checkpoints (newest first).

    Returns:
        list of dicts with: id, scan_id, created_at, weights_hash, is_initial,
        step, epsilon, episodes (from q_meta_json)
    """
    try:
        from app.db.models.rl import RLCheckpoint
        from sqlalchemy import select
    except ImportError:
        return []

    async def _do_select(sess: Any) -> list[RLCheckpoint]:
        stmt = (
            select(RLCheckpoint)
            .order_by(RLCheckpoint.created_at.desc())
            .limit(limit)
        )
        result = await sess.execute(stmt)
        return list(result.scalars().all())

    if session is not None:
        rows = await _do_select(session)
    else:
        try:
            from app.db.session import async_session
            async with async_session() as sess:
                rows = await _do_select(sess)
        except Exception as exc:
            logger.warning("list_checkpoints failed: %s", exc)
            return []

    out: list[dict[str, Any]] = []
    for row in rows:
        meta = row.q_meta_json or {}
        out.append({
            "id": str(row.id),
            "scan_id": row.scan_id,
            "created_at": row.created_at.isoformat() if row.created_at else None,
            "weights_hash": row.weights_hash,
            "is_initial": bool(row.is_initial),
            "step": int(meta.get("step", 0)),
            "epsilon": float(meta.get("epsilon", 0.0)),
            "episodes": int(meta.get("episodes", 0)),
            "architecture": meta.get("architecture", "dueling_double_q"),
            "total_params": int(meta.get("total_params", 0)),
        })
    return out


# ── Internal helpers ─────────────────────────────────────────────────────

def _serialize_weights(weights_dict: dict[str, np.ndarray]) -> tuple[bytes, str]:
    """Serialize weights dict → zlib-compressed bytes + SHA-256 hash.

    Format: zlib.compress(np.save(BytesIO, weights_dict), level=6)
    """
    buf = io.BytesIO()
    np.savez(buf, **weights_dict)
    raw = buf.getvalue()
    compressed = zlib.compress(raw, level=6)

    # Hash over the raw (uncompressed) weights for deterministic verification
    weights_hash = hashlib.sha256(raw).hexdigest()
    return compressed, weights_hash


def _deserialize_weights(blob: bytes) -> dict[str, np.ndarray]:
    """Decompress + load weights dict from SQL blob."""
    raw = zlib.decompress(blob)
    buf = io.BytesIO(raw)
    data = np.load(buf, allow_pickle=False)
    return dict(data)


def _hash_weights(weights_dict: dict[str, np.ndarray]) -> str:
    """Compute SHA-256 hash of weights dict (deterministic, raw bytes)."""
    buf = io.BytesIO()
    np.savez(buf, **weights_dict)
    return hashlib.sha256(buf.getvalue()).hexdigest()


def _build_meta(
    q_learner: DoubleQLearner,
    policy: PentestPolicy | None,
    scan_id: str | None,
    total_reward: float,
    is_initial: bool,
) -> dict[str, Any]:
    """Build q_meta_json dict for SQL checkpoint row."""
    meta: dict[str, Any] = {
        "step": int(q_learner._step),
        "lr": float(q_learner.lr),
        "gamma": float(q_learner.gamma),
        "tau": float(q_learner.tau),
        "state_dim": int(q_learner.state_dim),
        "action_dim": int(q_learner.action_dim),
        "hidden_dim": int(q_learner.hidden_dim),
        "architecture": "dueling_double_q",
        "total_params": int(q_learner._count_params()),
        "scan_id": scan_id,
        "total_reward": float(total_reward),
        "is_initial": bool(is_initial),
        "saved_at": datetime.now(UTC).isoformat(),
    }
    if policy is not None:
        meta["epsilon"] = float(policy.epsilon)
        meta["temperature"] = float(policy.temperature)
        meta["exploration_mode"] = str(policy.exploration_mode)
        meta["episodes"] = int(policy._episode)
        meta["epsilon_start"] = float(policy.epsilon_start)
        meta["epsilon_end"] = float(policy.epsilon_end)
        meta["epsilon_decay"] = float(policy.epsilon_decay)
    return meta