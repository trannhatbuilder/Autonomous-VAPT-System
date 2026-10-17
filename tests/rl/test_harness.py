"""
Unit tests for app/rl/harness.py (Phase 1 — RL Harness Bridge).

These tests exercise the harness in isolation — no real LLM, no real
tool execution, no real database. The PentestPolicy + ExperienceStore
are real (numpy Q-network + in-memory SumTree), so we get end-to-end
coverage of:
    - state encoding
    - action selection (with epsilon + curiosity)
    - guidance text rendering (advisory framing)
    - experience recording (PER push)
    - background training worker (async, _q_lock)
    - end_scan lifecycle (worker cancel, epsilon decay, checkpoint save)
    - action-followed metric
"""
from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

# Ensure project root is on sys.path (so `app` package is importable when
# this module is collected without a package-level conftest.py).
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Ensure the harness is enabled for these tests regardless of host env.
os.environ["VAPT_AI_RL_HARNESS_ENABLED"] = "true"
# Speed up the worker polling cadence for tests.
os.environ.setdefault("VAPT_AI_RL_TRAIN_INTERVAL_TURNS", "2")
os.environ.setdefault("VAPT_AI_RL_MIN_BUFFER_FOR_TRAIN", "4")
os.environ.setdefault("VAPT_AI_RL_TRAIN_BATCH_SIZE", "4")

from app.rl.harness import (  # noqa: E402
    HarnessBridge,
    TurnRecommendation,
    get_harness,
    is_rl_harness_enabled,
    release_harness,
    TRAIN_INTERVAL_TURNS,
)
from app.rl._session_builder import build_session  # noqa: E402


# ─── fixtures ─────────────────────────────────────────────────────────


def _make_session(
    *,
    turn: int = 1,
    findings: list[dict[str, Any]] | None = None,
    commands_run: int = 0,
    tokens_used: int = 0,
    last_command: str = "",
    last_vector: str = "",
    last_turn_success: bool = False,
) -> dict[str, Any]:
    """Build a minimal session dict that StateEncoder can consume."""
    return build_session(
        scan_id="test-scan",
        target="https://example.com",
        turn=turn,
        commands_run=commands_run,
        tokens_used=tokens_used,
        token_budget=10_000_000,
        last_command=last_command,
        last_vector=last_vector,
        last_turn_success=last_turn_success,
        findings=findings or [],
    )


@pytest.fixture
async def harness():
    """Fresh HarnessBridge per test. Auto-cleans at teardown."""
    h = HarnessBridge(scan_id="test-scan", target="https://example.com")
    try:
        yield h
    finally:
        # Best-effort cleanup — don't leak the train worker.
        try:
            await asyncio.wait_for(h.end_scan(episode_reward=0.0), timeout=3.0)
        except asyncio.TimeoutError:
            pass
        release_harness("test-scan")


# ─── 1. select_action returns a valid recommendation ──────────────────


async def test_select_action_returns_valid_recommendation(harness):
    session = _make_session(turn=1)
    rec = await harness.select_action(session, turn=1)

    assert isinstance(rec, TurnRecommendation)
    assert 0 <= rec.action_index < 7  # ACTION_DIM = 7
    assert rec.action_name in {
        "execute_command_recon",
        "execute_command_exploit",
        "execute_command_fuzz",
        "select_skill",
        "consult_kg",
        "report_vulnerability",
        "pentest_complete",
    }
    assert len(rec.q_values) == 7
    assert 0.0 <= rec.epsilon <= 1.0
    assert isinstance(rec.exploratory, bool)
    assert isinstance(rec.guidance_text, str)
    assert len(rec.guidance_text) > 0


# ─── 2. guidance text is advisory-framed ──────────────────────────────


def test_format_guidance_is_advisory_worded():
    text = HarnessBridge.format_guidance(
        action_name="execute_command_recon",
        q_max=0.5,
        epsilon=0.3,
        exploratory=False,
    )
    # Must contain advisory markers — never directive.
    assert "advisory" in text.lower()
    assert "your judgment wins" in text.lower()
    assert "suggested action" in text.lower()
    # Must NOT contain directive verbs.
    assert "required action" not in text.lower()
    assert "you must" not in text.lower()


def test_format_guidance_flags_low_confidence():
    """When q_max < 0.10, the hint should mention 'still learning'."""
    text = HarnessBridge.format_guidance(
        action_name="select_skill",
        q_max=0.05,
        epsilon=0.9,
        exploratory=True,
    )
    assert "still learning" in text.lower()
    assert "feel free to override" in text.lower()


def test_format_guidance_greedy_pick_when_not_exploratory():
    text = HarnessBridge.format_guidance(
        action_name="report_vulnerability",
        q_max=1.2,
        epsilon=0.05,
        exploratory=False,
    )
    assert "greedy pick" in text.lower()


# ─── 3. record_experience pushes to the PER store ─────────────────────


async def test_record_experience_pushes_to_per_store(harness):
    """select_action stashes the pre-turn state internally; record_experience
    pairs it with the post-turn state and pushes (s, a, r, s') into PER.

    With the internal-stash fix, the FIRST record_experience call DOES
    push a transition (because select_action already stashed _prev_state_vec).
    """
    # Turn 1: select_action stashes state → record_experience pushes 1 transition.
    session_1 = _make_session(turn=1)
    await harness.select_action(session_1, turn=1)
    await harness.record_experience(
        session_1, turn=1, action_idx=0, reward=0.5, done=False,
    )
    assert harness.buffer_size == 1

    # Turn 2: select_action re-stashes → record_experience pushes another.
    session_2 = _make_session(turn=2, commands_run=1, last_command="nmap")
    await harness.select_action(session_2, turn=2)
    await harness.record_experience(
        session_2, turn=2, action_idx=0, reward=1.0, done=False,
    )
    assert harness.buffer_size == 2


# ─── 4. action-followed metric ────────────────────────────────────────


async def test_record_action_followed_increments_counter(harness):
    session = _make_session(turn=1)
    await harness.select_action(session, turn=1)

    harness.record_action_followed(followed=True)
    harness.record_action_followed(followed=True)
    harness.record_action_followed(followed=False)

    stats = harness.get_stats()
    assert stats["followed_count"] == 2


# ─── 5. end_scan cancels the train worker + decays epsilon ────────────


async def test_end_scan_cancels_worker_and_persists(harness):
    # Push enough transitions to trigger at least one train step.
    sessions = [_make_session(turn=i, commands_run=i) for i in range(1, 8)]
    epsilon_before = harness.epsilon

    for i, s in enumerate(sessions, start=1):
        await harness.select_action(s, turn=i)
        await harness.record_experience(
            s, turn=i, action_idx=0, reward=float(i), done=False,
        )

    # Wait briefly so the worker can drain pending trains.
    await asyncio.sleep(0.3)

    await harness.end_scan(episode_reward=21.0)

    # Worker should be cancelled — calling end_scan again is a no-op.
    await harness.end_scan(episode_reward=21.0)  # should not raise

    # Epsilon should have decayed (end_episode applies the decay schedule).
    epsilon_after = harness.epsilon
    assert epsilon_after <= epsilon_before


# ─── 6. _q_lock prevents concurrent read/write ────────────────────────


async def test_q_lock_serializes_select_and_train():
    """
    select_action and _train_one_batch both acquire _q_lock. If they
    didn't, a numpy weight update could land mid-forward-pass and the
    Q-values would be garbage. This test verifies the Q-network read in
    select_action happens while the lock is held.
    """
    h = HarnessBridge(scan_id="lock-test", target="https://example.com")
    try:
        lock_held_during_q_read: list[bool] = []

        original_policy_select = h._policy.select_action

        def spy_policy_select(state_vec):
            # Reached from inside `async with self._q_lock:` in
            # HarnessBridge.select_action — so the lock MUST be held
            # here. Sampling the flag at call entry (outside the critical
            # section) would be racy and prove nothing.
            lock_held_during_q_read.append(h._q_lock.locked())
            return original_policy_select(state_vec)

        h._policy.select_action = spy_policy_select  # type: ignore

        session = _make_session(turn=1)
        # Run two selects concurrently — both must read under the lock.
        await asyncio.gather(
            h.select_action(session, 1),
            h.select_action(session, 1),
        )

        assert lock_held_during_q_read, (
            "policy.select_action was never invoked by select_action."
        )
        assert all(lock_held_during_q_read), (
            "Q-network read ran outside _q_lock — concurrent weight "
            "updates from the train worker could corrupt Q-values."
        )
    finally:
        await asyncio.wait_for(h.end_scan(episode_reward=0.0), timeout=3.0)
        release_harness("lock-test")


# ─── 7. is_rl_harness_enabled respects env var ────────────────────────


def test_is_rl_harness_enabled_env_toggle(monkeypatch):
    monkeypatch.setenv("VAPT_AI_RL_HARNESS_ENABLED", "false")
    assert is_rl_harness_enabled() is False

    monkeypatch.setenv("VAPT_AI_RL_HARNESS_ENABLED", "true")
    assert is_rl_harness_enabled() is True

    monkeypatch.setenv("VAPT_AI_RL_HARNESS_ENABLED", "1")
    assert is_rl_harness_enabled() is True

    monkeypatch.delenv("VAPT_AI_RL_HARNESS_ENABLED", raising=False)
    # Default when unset.
    assert is_rl_harness_enabled() is True


# ─── 8. get_harness / release_harness registry ────────────────────────


async def test_get_harness_returns_same_instance_per_scan():
    h1 = get_harness("scan-A", "https://a.example")
    h2 = get_harness("scan-A", "https://a.example")
    h3 = get_harness("scan-B", "https://b.example")
    assert h1 is h2  # same scan_id -> same instance
    assert h1 is not h3  # different scan_id -> different instance

    release_harness("scan-A")
    release_harness("scan-B")
    # Cleanup the workers.
    await asyncio.wait_for(h1.end_scan(episode_reward=0.0), timeout=3.0)
    await asyncio.wait_for(h3.end_scan(episode_reward=0.0), timeout=3.0)


# ─── 9. training worker trains when buffer crosses threshold ──────────


async def test_train_worker_drains_queue(monkeypatch):
    """
    When record_experience signals the worker AND the buffer is above
    MIN_BUFFER_FOR_TRAIN, the worker should run train_from_memory and
    append to _train_log.
    """
    # Force small thresholds so training kicks in quickly.
    monkeypatch.setenv("VAPT_AI_RL_TRAIN_INTERVAL_TURNS", "1")
    monkeypatch.setenv("VAPT_AI_RL_MIN_BUFFER_FOR_TRAIN", "2")
    monkeypatch.setenv("VAPT_AI_RL_TRAIN_BATCH_SIZE", "2")

    # Re-import to pick up new env values.
    import importlib
    import app.rl.harness as harness_mod
    importlib.reload(harness_mod)

    h = harness_mod.HarnessBridge(
        scan_id="train-test", target="https://example.com",
    )
    try:
        # Push 4 transitions (turns 1..4 — first turn seeds, so 3 real).
        for i in range(1, 5):
            s = _make_session(turn=i, commands_run=i)
            await h.select_action(s, turn=i)
            await h.record_experience(
                s, turn=i, action_idx=0, reward=float(i), done=False,
            )

        # Wait for the worker to drain.
        await asyncio.sleep(0.5)

        stats = h.get_stats()
        # At least one train step should have run.
        assert stats["train_steps"] >= 1, (
            f"Expected >=1 train step, got {stats['train_steps']}. "
            f"Buffer={stats['buffer_size']}."
        )
        assert stats["train_log"][0]["loss"] >= 0.0
    finally:
        await asyncio.wait_for(h.end_scan(episode_reward=10.0), timeout=3.0)
        harness_mod.release_harness("train-test")


# ─── 10. get_stats returns expected fields ────────────────────────────


async def test_get_stats_returns_expected_fields(harness):
    session = _make_session(turn=1)
    await harness.select_action(session, turn=1)
    harness.record_action_followed(followed=True)

    stats = harness.get_stats()
    expected_keys = {
        "scan_id", "turn_count", "episode_reward", "epsilon",
        "followed_count", "follow_rate", "exploratory_count",
        "buffer_size", "train_steps", "train_log",
        "checkpoint_loaded", "checkpoint_saved",
    }
    assert expected_keys.issubset(stats.keys())
    assert stats["scan_id"] == "test-scan"
    assert stats["followed_count"] == 1
    assert stats["follow_rate"] == 1.0  # 1 follow out of 1 turn