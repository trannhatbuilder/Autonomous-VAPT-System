"""
Tests for app.rl.experience_store — PER SumTree + SQL persistence.

Covers:
  - SumTree: add, update, sample, total, get
  - SumTree: capacity rounds to next power of 2
  - ExperienceStore: add (async), sample, update_priorities
  - PER: alpha=0.6, beta=0.4→1.0 annealing
  - Empty store edge case
  - IS weights computation
  - Stats reporting
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

os.environ.setdefault("VAPT_AI_POSTGRES_DB", "postgresql://vapt:vapt@localhost:5432/vapt_ai_test")
os.environ.setdefault("VAPT_AI_ENVIRONMENT", "dev")

import numpy as np
import pytest
import asyncio

from app.rl.experience_store import (
    ExperienceStore, SumTree,
    MAX_MEMORY_BUFFER, PER_ALPHA, PER_EPSILON,
    PER_BETA_START, PER_BETA_END, PER_BETA_ANNEAL_STEPS,
    _next_pow2,
)


# ── Section 1: _next_pow2 ────────────────────────────────────────────────

class TestNextPow2:
    def test_basic(self):
        assert _next_pow2(1) == 1
        assert _next_pow2(2) == 2
        assert _next_pow2(3) == 4
        assert _next_pow2(100) == 128
        assert _next_pow2(50000) == 65536

    def test_zero(self):
        assert _next_pow2(0) == 1


# ── Section 2: SumTree ───────────────────────────────────────────────────

class TestSumTree:
    def test_capacity_rounded_to_pow2(self):
        """Capacity 100 → 128, 50000 → 65536."""
        tree = SumTree(100)
        assert tree.capacity == 128

    def test_add_increments_size(self):
        tree = SumTree(100)
        for i in range(5):
            tree.add(priority=float(i + 1), data={"turn": i})
        assert tree.size == 5

    def test_total_after_adds(self):
        tree = SumTree(100)
        for i in range(5):
            tree.add(priority=float(i + 1), data={"turn": i})
        # 1+2+3+4+5 = 15
        assert tree.total() == 15.0

    def test_update_changes_total(self):
        tree = SumTree(100)
        for i in range(5):
            tree.add(priority=float(i + 1), data={"turn": i})
        # Update leaf 0 (priority 1.0 → 100.0)
        leaf_idx = tree.capacity  # first leaf
        tree.update(leaf_idx, 100.0)
        # New total: 15 - 1 + 100 = 114
        assert tree.total() == 114.0

    def test_sample_returns_correct_count(self):
        tree = SumTree(100)
        for i in range(10):
            tree.add(priority=float(i + 1), data={"turn": i})
        indices, priorities, samples = tree.sample(5)
        assert len(indices) == 5
        assert len(priorities) == 5
        assert len(samples) == 5

    def test_get_finds_leaf(self):
        tree = SumTree(100)
        tree.add(priority=10.0, data={"turn": 0})
        idx, prio, data = tree.get(5.0)  # cumulative priority 5.0 < 10.0
        assert data == {"turn": 0}
        assert prio == 10.0

    def test_max_priority_tracked(self):
        tree = SumTree(100)
        tree.add(priority=1.0, data="a")
        tree.add(priority=5.0, data="b")
        tree.add(priority=3.0, data="c")
        assert tree._max_priority == 5.0

    def test_empty_tree_sample(self):
        tree = SumTree(100)
        indices, priorities, samples = tree.sample(5)
        # Empty tree → total=0, sample returns empty
        assert indices == []
        assert samples == []


# ── Section 3: ExperienceStore (no DB) ───────────────────────────────────

class TestExperienceStore:
    def test_initial_state(self):
        store = ExperienceStore(max_buffer=1000)
        assert len(store) == 0
        assert store._beta == PER_BETA_START

    def test_add_increments_size(self):
        """Test add() via asyncio.run (DB will fail gracefully, SumTree works)."""
        async def run():
            store = ExperienceStore(max_buffer=1000)
            state = np.random.randn(337).astype(np.float32)
            next_state = np.random.randn(337).astype(np.float32)
            for i in range(5):
                await store.add(
                    scan_id="scan_test", turn=i,
                    state=state, action=i % 7, reward=float(i),
                    next_state=next_state, done=(i == 4),
                )
            assert len(store) == 5
        asyncio.run(run())

    def test_sample_returns_batch_with_is_weights(self):
        async def run():
            store = ExperienceStore(max_buffer=1000)
            state = np.random.randn(337).astype(np.float32)
            next_state = np.random.randn(337).astype(np.float32)
            for i in range(20):
                await store.add(
                    scan_id="scan_test", turn=i,
                    state=state, action=i % 7, reward=float(i),
                    next_state=next_state, done=(i == 19),
                )
            batch, is_weights, indices = store.sample(8)
            assert len(batch) == 8
            assert len(is_weights) == 8
            assert len(indices) == 8
            # Each batch item has is_weight key
            assert "is_weight" in batch[0]
            # State is numpy array
            assert isinstance(batch[0]["state"], np.ndarray)
            assert batch[0]["state"].shape == (337,)
        asyncio.run(run())

    def test_empty_store_sample(self):
        store = ExperienceStore(max_buffer=100)
        batch, weights, indices = store.sample(10)
        assert batch == []
        assert weights == []
        assert indices == []

    def test_update_priorities_changes_total(self):
        async def run():
            store = ExperienceStore(max_buffer=1000)
            state = np.random.randn(337).astype(np.float32)
            next_state = np.random.randn(337).astype(np.float32)
            for i in range(5):
                await store.add(
                    scan_id="scan_test", turn=i,
                    state=state, action=i, reward=float(i),
                    next_state=next_state, done=False,
                )
            total_before = store.tree.total()
            _, _, indices = store.sample(5)
            store.update_priorities(indices, [10.0, 5.0, 1.0, 0.5, 0.1])
            total_after = store.tree.total()
            assert total_after != total_before
        asyncio.run(run())

    def test_beta_annealing(self):
        """Beta should increase from 0.4 → 1.0 with each sample call."""
        async def run():
            store = ExperienceStore(max_buffer=1000)
            state = np.random.randn(337).astype(np.float32)
            for i in range(10):
                await store.add(
                    scan_id="s", turn=i, state=state, action=0,
                    reward=1.0, next_state=state, done=False,
                )
            beta_before = store._beta
            store.sample(5)
            beta_after = store._beta
            assert beta_after > beta_before
        asyncio.run(run())

    def test_get_stats(self):
        store = ExperienceStore(max_buffer=1000)
        stats = store.get_stats()
        assert "size" in stats
        assert "capacity" in stats
        assert "total_priority" in stats
        assert "beta" in stats
        assert "alpha" in stats
        assert stats["alpha"] == PER_ALPHA
        # max_buffer is the user-facing config (1000); tree.capacity is next_pow2(1024)
        assert stats["capacity"] == 1000


# ── Section 4: PER constants ─────────────────────────────────────────────

class TestPERConstants:
    def test_alpha_value(self):
        assert PER_ALPHA == 0.6

    def test_epsilon_value(self):
        assert PER_EPSILON == 1e-6

    def test_beta_range(self):
        assert PER_BETA_START == 0.4
        assert PER_BETA_END == 1.0
        assert PER_BETA_START < PER_BETA_END

    def test_beta_anneal_steps(self):
        assert PER_BETA_ANNEAL_STEPS == 100_000

    def test_max_memory_buffer(self):
        assert MAX_MEMORY_BUFFER == 50_000