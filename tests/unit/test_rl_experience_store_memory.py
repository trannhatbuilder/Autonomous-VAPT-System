"""EX1 — the SumTree must not store Python-float lists.

The bug: `add()` did `state.astype(np.float32).tolist()`, which converts a
337-float32 vector into 337 Python `float` objects (~24 B each) plus a pointer
slot (8 B) per element. Measured on this build:

    CURRENT (tolist)      22,431 B/transition  →  1,069.6 MB @ 50k
                                                 →  1,401.9 MB @ real capacity
    PROPOSED (float32)     3,671 B/transition  →    175.0 MB @ 50k
                                                 →    229.4 MB @ real capacity

The SumTree rounds `max_buffer=50_000` up to 65,536 leaves (`_next_pow2`), so
the true worst case was 1.4 GB, not the ~1.0 GB in the report.

These guards assert the *outcome* (bytes per transition), not the presence of a
particular call. A future "optimisation" that reintroduces a per-element Python
object — a `.tolist()`, a `tuple()`, a `list(...)` — fails on the measurement.

They also pin the two properties the two-line fix as originally written would
have broken:

  1. **Snapshot, not alias.** `np.asarray(x, dtype=np.float32)` returns `x`
     itself when it is already float32, so the buffer would share the caller's
     scratch array. Reusing that array would rewrite recorded history.
  2. **Read-only.** `sample()` returns the stored object (no copy) through a
     shallow `dict()`, so an in-place write by any consumer would corrupt the
     replay buffer silently. Now it raises.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import numpy as np
import pytest

from app.rl.experience_store import (
    MAX_MEMORY_BUFFER,
    ExperienceStore,
    _next_pow2,
    _snapshot_vector,
)
from app.rl.state_encoder import STATE_DIM

REPO = Path(__file__).resolve().parents[2]
STORE_SRC = (REPO / "app" / "rl" / "experience_store.py").read_text(encoding="utf-8")

# Measured with the deep-size walker in `_deep_bytes` below, at STATE_DIM=337.
# Generous upper bound: any per-element Python object blows past this by 6x.
MAX_BYTES_PER_TRANSITION = 6_000


def _deep_bytes(obj: object, _seen: set[int] | None = None) -> int:
    """Transitive size of a Python object graph, in bytes."""
    if _seen is None:
        _seen = set()
    if id(obj) in _seen:
        return 0
    _seen.add(id(obj))
    total = sys.getsizeof(obj)
    if isinstance(obj, (list, tuple)):
        total += sum(_deep_bytes(v, _seen) for v in obj)
    elif isinstance(obj, dict):
        total += sum(_deep_bytes(k, _seen) + _deep_bytes(v, _seen) for k, v in obj.items())
    return total


async def _no_persist(*_args: object, **_kwargs: object) -> None:
    """Skip the DB write: these tests are about the in-memory buffer only."""
    return None


async def _fill(store: ExperienceStore, n: int) -> None:
    store._persist_transition = _no_persist  # type: ignore[method-assign]
    rng = np.random.default_rng(0)
    for i in range(n):
        await store.add(
            scan_id="scan_mem",
            turn=i,
            state=rng.random(STATE_DIM, dtype=np.float32),
            action=i % 7,
            reward=0.1,
            next_state=rng.random(STATE_DIM, dtype=np.float32),
            done=False,
        )


# ─── the measurement itself ───────────────────────────────────────────


class TestBufferPayloadIsCompact:
    def test_transition_payload_stays_under_budget(self):
        """The class-level guard: bytes per stored transition."""
        store = ExperienceStore(max_buffer=1000)
        import asyncio

        asyncio.get_event_loop_policy().new_event_loop()
        asyncio.run(_fill(store, 200))

        stored = [d for d in store.tree.data if d is not None]
        assert len(stored) == 200
        per = _deep_bytes(stored) / len(stored)
        assert per < MAX_BYTES_PER_TRANSITION, (
            f"{per:.0f} B/transition — a Python-float list is back. Measured "
            f"22,431 B before the fix and 3,671 B after; budget is "
            f"{MAX_BYTES_PER_TRANSITION}."
        )

    def test_payload_is_at_least_6x_smaller_than_a_tolist_payload(self):
        """Direct A/B, so the budget above cannot drift into meaninglessness."""
        store = ExperienceStore(max_buffer=1000)
        import asyncio

        asyncio.run(_fill(store, 50))
        stored = [d for d in store.tree.data if d is not None][0]
        actual = _deep_bytes(stored)

        legacy = dict(stored)
        legacy["state"] = np.asarray(stored["state"]).tolist()
        legacy["next_state"] = np.asarray(stored["next_state"]).tolist()
        legacy_bytes = _deep_bytes(legacy)

        assert legacy_bytes / actual >= 6.0, (
            f"array payload {actual} B vs list payload {legacy_bytes} B"
        )

    def test_full_buffer_projection_is_under_a_gigabyte(self):
        """At the *effective* capacity (next power of two), not max_buffer."""
        store = ExperienceStore(max_buffer=1000)
        import asyncio

        asyncio.run(_fill(store, 100))
        stored = [d for d in store.tree.data if d is not None]
        per = _deep_bytes(stored) / len(stored)

        capacity = _next_pow2(MAX_MEMORY_BUFFER)
        projected_mb = per * capacity / 2**20
        assert projected_mb < 400, (
            f"projected {projected_mb:.0f} MB at {capacity} transitions "
            "(was 1,402 MB with lists)"
        )


# ─── the payload shape ────────────────────────────────────────────────


class TestStoredShape:
    @pytest.mark.asyncio
    async def test_stored_vectors_are_float32_ndarrays(self):
        store = ExperienceStore(max_buffer=100)
        await _fill(store, 3)
        for d in store.tree.data[:3]:
            for key in ("state", "next_state"):
                assert isinstance(d[key], np.ndarray), f"{key} is {type(d[key])}"
                assert d[key].dtype == np.float32
                assert d[key].shape == (STATE_DIM,)

    @pytest.mark.asyncio
    async def test_no_python_float_list_survives_in_the_tree(self):
        store = ExperienceStore(max_buffer=100)
        await _fill(store, 3)
        for d in store.tree.data[:3]:
            assert not isinstance(d["state"], list)
            assert not isinstance(d["next_state"], list)

    def test_add_does_not_call_tolist_on_the_stored_payload(self):
        """Source guard: the offending expression must not come back.

        `_insert_transition_row` legitimately calls `.tolist()` for the JSON
        column — that is transient. What must never return is a `.tolist()`
        inside the dict that goes into `tree.add()`.
        """
        tree = ast.parse(STORE_SRC)
        offender = None
        for node in ast.walk(tree):
            # Find the dict literal whose keys include "next_state".
            if not isinstance(node, ast.Dict):
                continue
            keys = [
                k.value for k in node.keys
                if isinstance(k, ast.Constant) and isinstance(k.value, str)
            ]
            if "next_state" not in keys:
                continue
            for value in node.values:
                if any(
                    isinstance(n, ast.Call)
                    and isinstance(n.func, ast.Attribute)
                    and n.func.attr == "tolist"
                    for n in ast.walk(value)
                ):
                    offender = node.lineno
        assert offender is None, (
            f"transition dict at experience_store.py:{offender} still calls "
            ".tolist() — 22,431 B/transition instead of 3,671 B"
        )


# ─── snapshot semantics (the part the 2-line fix got wrong) ───────────


class TestSnapshotNotAlias:
    def test_asarray_would_have_aliasled_so_we_copy(self):
        """Documents the trap: np.asarray is NOT a snapshot."""
        src = np.zeros(STATE_DIM, dtype=np.float32)
        assert np.asarray(src, dtype=np.float32) is src, (
            "numpy changed asarray semantics; the rationale in "
            "_snapshot_vector needs re-checking"
        )

    def test_snapshot_does_not_share_memory_with_the_source(self):
        src = np.zeros(STATE_DIM, dtype=np.float32)
        snap = _snapshot_vector(src)
        assert not np.shares_memory(snap, src)
        assert snap is not src

    def test_snapshot_is_read_only(self):
        snap = _snapshot_vector(np.zeros(STATE_DIM, dtype=np.float32))
        assert snap.flags.writeable is False
        with pytest.raises(ValueError):
            snap[0] = 1.0

    def test_snapshot_forces_float32_from_float64(self):
        snap = _snapshot_vector(np.zeros(STATE_DIM, dtype=np.float64))
        assert snap.dtype == np.float32
        assert snap.flags.writeable is False

    def test_snapshot_accepts_a_list(self):
        snap = _snapshot_vector([0.0] * STATE_DIM)
        assert snap.dtype == np.float32
        assert snap.shape == (STATE_DIM,)

    @pytest.mark.asyncio
    async def test_reusing_a_scratch_buffer_does_not_rewrite_history(self):
        """The behavioural version of the alias bug.

        `_prev_state_vec` is reassigned (not mutated) in the harness today,
        so this is latent — but `add()` must own its copy anyway. Same shape
        as the existing `test_add_increments_size`, which reuses one array.
        """
        store = ExperienceStore(max_buffer=100)
        store._persist_transition = _no_persist  # type: ignore[method-assign]

        scratch = np.ones(STATE_DIM, dtype=np.float32)
        for i in range(3):
            await store.add(
                scan_id="s", turn=i, state=scratch, action=0, reward=0.0,
                next_state=scratch, done=False,
            )
            scratch += 1.0  # caller recycles its buffer

        values = [float(d["state"][0]) for d in store.tree.data[:3]]
        assert values == [1.0, 2.0, 3.0], (
            f"stored transitions alias the caller's buffer: {values}"
        )


# ─── sampling hands out read-only arrays, no copies ───────────────────


class TestSampleHandsOutFrozenArrays:
    @pytest.mark.asyncio
    async def test_sampled_states_are_read_only_and_not_copies(self):
        store = ExperienceStore(max_buffer=100)
        await _fill(store, 20)
        batch, _w, _i = store.sample(4)
        assert batch
        for item in batch:
            assert item["state"].flags.writeable is False
            assert item["next_state"].flags.writeable is False
        # Same object as the tree holds — no 1.3 KB copy per sample.
        tree_ids = {id(d["state"]) for d in store.tree.data if d is not None}
        assert all(id(item["state"]) in tree_ids for item in batch)

    @pytest.mark.asyncio
    async def test_in_place_write_by_a_consumer_fails_loudly(self):
        store = ExperienceStore(max_buffer=100)
        await _fill(store, 10)
        batch, _w, _i = store.sample(2)
        with pytest.raises(ValueError):
            batch[0]["state"] *= 2.0

    @pytest.mark.asyncio
    async def test_legacy_list_payloads_are_normalised_too(self):
        """Rows from an older build / hand-built test data are still lists."""
        store = ExperienceStore(max_buffer=10)
        store.tree.add(
            1.0,
            {
                "state": [0.0] * STATE_DIM,
                "next_state": [1.0] * STATE_DIM,
                "action": 0, "reward": 0.0, "done": False,
                "scan_id": "legacy", "turn": 0,
            },
        )
        batch, _w, _i = store.sample(1)
        assert isinstance(batch[0]["state"], np.ndarray)
        assert batch[0]["state"].dtype == np.float32
        assert batch[0]["state"].flags.writeable is False

    @pytest.mark.asyncio
    async def test_training_still_reads_the_frozen_arrays(self):
        """The whole point: read-only must not break the learn path."""
        from app.rl.policy import PentestPolicy

        store = ExperienceStore(max_buffer=100)
        await _fill(store, 32)
        policy = PentestPolicy()
        loss, td_errors = policy.train_from_memory(store, batch_size=8)
        assert len(td_errors) == 8
        assert np.isfinite(loss)


# ─── the DB path must stay untouched ──────────────────────────────────


class TestDbPathUnaffected:
    @pytest.mark.asyncio
    async def test_persist_receives_the_original_arrays(self):
        captured = {}

        async def _capture(**kwargs):
            captured.update(kwargs)

        store = ExperienceStore(max_buffer=10)
        store._persist_transition = _capture  # type: ignore[method-assign]
        state = np.ones(STATE_DIM, dtype=np.float32)
        next_state = np.full(STATE_DIM, 2.0, dtype=np.float32)
        await store.add(
            scan_id="db", turn=1, state=state, action=2, reward=1.5,
            next_state=next_state, done=True,
        )
        assert np.shares_memory(captured["state"], state)
        assert np.shares_memory(captured["next_state"], next_state)

    def test_row_builder_still_converts_to_list_for_json(self):
        """`_insert_transition_row` owns its own `.tolist()`; do not delete it."""
        assert "state_vector=state.astype(np.float32).tolist()" in STORE_SRC
        assert "next_state_vector=next_state.astype(np.float32).tolist()" in STORE_SRC
