"""
RL training loop orchestration for VAPT-AI.

Provides high-level helpers that the Harness Bridge (W18) will call:

  - train_one_step(policy, store, batch_size)
        Sample a PER batch from in-memory SumTree, train Q-network,
        update priorities. Returns (loss, td_errors).

  - train_from_scan(scan_id, policy, store, batch_size, session)
        Pull ALL transitions of a single scan from Postgres for offline
        training. Useful for re-training on past scans without rerunning
        them. Returns (loss, td_errors).

  - train_multi_step(policy, store, n_steps, batch_size)
        Run N consecutive train_one_step() calls. Useful at end of scan
        to consolidate learning. Returns list of (loss, td_errors).

Logging: every step logs batch_size, loss, ε, β — useful for convergence
plots in the RL Stats Panel (W21).
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

from app.rl.policy import PentestPolicy
from app.rl.experience_store import ExperienceStore

logger = logging.getLogger(__name__)


def train_one_step(
    policy: PentestPolicy,
    store: ExperienceStore,
    batch_size: int = 64,
) -> tuple[float, list[float]]:
    """One PER-sampled gradient update.

    Flow:
      1. store.sample(batch_size) → (batch, is_weights, tree_indices)
      2. policy.q.train_batch(batch) → (loss, td_errors)
      3. store.update_priorities(tree_indices, td_errors)

    Args:
        policy: PentestPolicy (holds the Q-learner)
        store: ExperienceStore (PER SumTree)
        batch_size: PER batch size (default 64 per master plan §12 W16)

    Returns:
        (loss, td_errors) — loss is mean(td_error²). Empty batch → (0.0, []).
    """
    if len(store) == 0:
        logger.debug("train_one_step skipped — store empty")
        return 0.0, []

    loss, td_errors = policy.train_from_memory(store, batch_size=batch_size)
    return loss, td_errors


def train_multi_step(
    policy: PentestPolicy,
    store: ExperienceStore,
    n_steps: int = 1,
    batch_size: int = 64,
) -> list[tuple[float, list[float]]]:
    """Run N consecutive train_one_step() calls.

    Useful at end of scan to consolidate learning before checkpoint save.

    Args:
        n_steps: number of training steps (default 1)
        batch_size: PER batch size

    Returns:
        list of (loss, td_errors) per step. Empty steps skipped.
    """
    results: list[tuple[float, list[float]]] = []
    for i in range(n_steps):
        if len(store) == 0:
            logger.debug("train_multi_step stopped at step %d — store empty", i)
            break
        loss, td_errors = train_one_step(policy, store, batch_size=batch_size)
        results.append((loss, td_errors))
        logger.debug(
            "train_multi_step | step=%d/%d | loss=%.4f | batch_td_mean=%.4f",
            i + 1, n_steps, loss,
            float(np.mean(td_errors)) if td_errors else 0.0,
        )
    return results


async def train_from_scan(
    scan_id: str,
    policy: PentestPolicy,
    store: ExperienceStore,
    batch_size: int = 256,
    session: Any = None,
) -> tuple[float, list[float]]:
    """Offline training: pull all transitions of a scan from Postgres.

    Uses store.sample_from_db(batch_size, scan_id) which queries
    vapt_rl_transitions for that scan's transitions (ordered by priority DESC).

    Args:
        scan_id: scan to re-train on
        policy: PentestPolicy
        store: ExperienceStore
        batch_size: DB batch size (default 256 — larger than PER batch)
        session: optional AsyncSession

    Returns:
        (loss, td_errors) — empty DB result → (0.0, []).
    """
    loss, td_errors = await policy.train_from_db(
        store, batch_size=batch_size, scan_id=scan_id,
    )
    if not td_errors:
        logger.info("train_from_scan | scan=%s | no transitions found", scan_id)
    else:
        logger.info(
            "train_from_scan | scan=%s | batch=%d | loss=%.4f | td_mean=%.4f",
            scan_id, len(td_errors), loss, float(np.mean(td_errors)),
        )
    return loss, td_errors


async def train_from_recent_scans(
    scan_ids: list[str],
    policy: PentestPolicy,
    store: ExperienceStore,
    batch_size: int = 256,
    session: Any = None,
) -> list[tuple[str, float, list[float]]]:
    """Offline training across multiple scans.

    Args:
        scan_ids: list of scan IDs to re-train on (in order)
        policy/store: RL components
        batch_size: per-scan DB batch size

    Returns:
        list of (scan_id, loss, td_errors) per scan
    """
    results: list[tuple[str, float, list[float]]] = []
    for scan_id in scan_ids:
        loss, td_errors = await train_from_scan(
            scan_id=scan_id, policy=policy, store=store,
            batch_size=batch_size, session=session,
        )
        results.append((scan_id, loss, td_errors))
    return results


def get_training_stats(policy: PentestPolicy, store: ExperienceStore) -> dict[str, Any]:
    """Aggregate stats for /api/rl/stats endpoint.

    Returns:
        dict with policy stats + store stats + derived metrics
    """
    stats: dict[str, Any] = {
        "policy": policy.get_stats(),
        "store": store.get_stats(),
    }
    # Derived
    stats["buffer_utilization"] = (
        stats["store"]["size"] / stats["store"]["capacity"]
        if stats["store"]["capacity"] > 0 else 0.0
    )
    stats["beta_progress"] = (
        (stats["store"]["beta"] - stats["store"]["beta_start"])
        / (stats["store"]["beta_end"] - stats["store"]["beta_start"])
    )
    stats["epsilon_progress"] = (
        1.0 - (stats["policy"]["epsilon"] - stats["policy"]["epsilon_end"])
        / (stats["policy"]["epsilon_start"] - stats["policy"]["epsilon_end"])
        if stats["policy"]["epsilon_start"] != stats["policy"]["epsilon_end"] else 0.0
    )
    return stats