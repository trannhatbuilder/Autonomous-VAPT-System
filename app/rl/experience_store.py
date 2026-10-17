"""
Prioritized Experience Replay (PER) Store backed by PostgreSQL + in-memory buffer.

Python port of EVVO Sentinel shield_engine/rl/experience_store.py.

Adaptations for VAPT-AI:
  - DB layer: psycopg2 sync → async SQLAlchemy 2.0 (asyncpg)
  - DB table: rl_experiences (EVVO) → vapt_rl_transitions (VAPT-AI W1-D model)
  - add() is async (writes to DB) — sample()/update_priorities() stay sync
    (SumTree in-memory hot-path)
  - sample_from_db() is async (queries DB for offline training)

Implements Schaul et al. 2016 PER with sum-tree for O(log n) sampling.
Priorities are |TD-error| + ε, exponentiated by α (priority exponent).

PER parameters (per master plan §12 W16):
  - buffer_size: 50,000
  - alpha (priority exponent): 0.6  (0=uniform, 1=full prioritization)
  - beta (IS exponent): 0.4 → 1.0 annealed over 100,000 steps
  - epsilon: 1e-6 (non-zero priority guarantee)
"""
from __future__ import annotations

import json
import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ── PER constants ────────────────────────────────────────────────────────
MAX_MEMORY_BUFFER: int = 50_000  # in-memory SumTree capacity
PER_ALPHA: float = 0.6            # priority exponent
PER_EPSILON: float = 1e-6         # non-zero priority guarantee
PER_BETA_START: float = 0.4       # importance-sampling exponent start
PER_BETA_END: float = 1.0         # importance-sampling exponent end
PER_BETA_ANNEAL_STEPS: int = 100_000


def _next_pow2(n: int) -> int:
    """Round up to the next power of two.

    Defensive — keeps the implicit binary-heap layout balanced even when
    callers request odd capacities. EVVO audit P0-7 fix.
    """
    p = 1
    while p < n:
        p <<= 1
    return p


def _snapshot_vector(vec: Any) -> np.ndarray:
    """Freeze a state vector into a private, read-only float32 array.

    Why not `.tolist()` (what this used to do): a list of 337 Python floats
    costs ~10.8 KB against ~1.4 KB for the same data as a float32 array —
    each element is a 24-byte `float` object plus an 8-byte pointer slot, so
    it is 8x the payload before dict overhead. A full 50,000-transition
    buffer measured 22,431 B/transition = **1,069 MB** with lists, versus
    3,671 B/transition = **175 MB** with arrays. (And the SumTree rounds the
    capacity up to 65,536, so the real worst case was 1,402 MB, not 1,069.)
    Nothing raises when that memory is exhausted — the process just creeps
    toward the OOM killer while GC churn grows, which is why it survived
    review.

    Why a COPY and not `np.asarray(vec, dtype=np.float32)`:
      - `asarray` returns the caller's array *unchanged* when it is already
        float32, so the replay buffer would alias the caller's buffer. Any
        later in-place reuse of that scratch array would retroactively
        rewrite recorded history — silently, and only for the transitions
        that happen to share the buffer. `tolist()` was accidentally
        providing a snapshot; this keeps the snapshot and drops the cost.
      - Tests in this repo really do reuse one `state` array across many
        `add()` calls, so the aliasing path is one refactor away from real.

    Why read-only: `sample()` hands these arrays straight to `train_batch()`
    through a shallow `dict` copy. Marking them non-writeable turns
    "something normalised a state in place" from silent replay corruption
    into an immediate `ValueError` at the offending line.

    Copies are 1,348 B per vector (2 per transition, once per agent turn),
    i.e. noise next to the ~18.7 KB per transition this removes.
    """
    arr = np.array(vec, dtype=np.float32, copy=True)
    arr.flags.writeable = False
    return arr


# ── SumTree ──────────────────────────────────────────────────────────────
class SumTree:
    """
    Sum tree for efficient prioritized sampling in O(log n).

    Tree structure (1-indexed for cleaner parent/child arithmetic):
      - Leaves [capacity : 2*capacity] store priorities
      - Internal nodes [1 : capacity] store sum of children
      - Root (index 1) = total priority mass

    Sampling: generate random value s ∈ [0, total), traverse from root to
    leaf, descending left if s ≤ left child's sum, else right (subtracting
    left's sum). O(log n) per sample.
    """

    def __init__(self, capacity: int) -> None:
        # Round capacity up to next power of two so every leaf sits at the
        # same depth in the implicit binary-heap layout (EVVO audit P0-7).
        self.capacity = _next_pow2(max(1, capacity))
        self.tree = np.zeros(2 * self.capacity, dtype=np.float64)
        self.data: list[Any] = [None] * self.capacity
        self.write_idx = 0
        self.size = 0
        self._max_priority = 1.0  # tracked so add() can use real max

    def _propagate(self, idx: int, change: float) -> None:
        """Propagate priority change up the tree."""
        parent = idx // 2
        while parent >= 1:
            self.tree[parent] += change
            parent //= 2

    def _retrieve(self, idx: int, s: float) -> int:
        """Find leaf node for cumulative priority s."""
        left = 2 * idx
        right = 2 * idx + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        return self._retrieve(right, s - self.tree[left])

    def total(self) -> float:
        """Total priority mass in the tree (root value)."""
        return float(self.tree[1])

    def add(self, priority: float, data: Any) -> int:
        """Add data with given priority. Returns leaf index."""
        idx = self.write_idx + self.capacity
        self.data[self.write_idx] = data
        self.update(idx, priority)
        self.write_idx = (self.write_idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        return idx

    def update(self, idx: int, priority: float) -> None:
        """Update priority at leaf index. Propagates delta up the tree."""
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)
        if priority > self._max_priority:
            self._max_priority = float(priority)

    def get(self, s: float) -> tuple[int, float, Any]:
        """
        Sample one transition at cumulative priority s.

        Returns:
            (leaf_idx, priority, data)
        """
        idx = self._retrieve(1, s)
        data_idx = idx - self.capacity
        return idx, float(self.tree[idx]), self.data[data_idx]

    def sample(self, batch_size: int) -> tuple[list[int], list[float], list[Any]]:
        """Sample batch_size transitions with proportional priorities.

        Splits [0, total) into batch_size equal segments and samples one
        uniform random value from each segment — this gives stratified
        sampling (better than naive uniform for skewed priority distributions).
        """
        indices: list[int] = []
        priorities: list[float] = []
        samples: list[Any] = []

        total = self.total()
        if total <= 0:
            # Defensive: empty tree or all-zero priorities
            return [], [], []
        segment = total / batch_size

        for i in range(batch_size):
            low = segment * i
            high = segment * (i + 1)
            s = float(np.random.uniform(low, high))
            idx, prio, data = self.get(s)
            indices.append(idx)
            priorities.append(prio)
            samples.append(data)

        return indices, priorities, samples


# ── ExperienceStore ──────────────────────────────────────────────────────
class ExperienceStore:
    """
    Prioritized Experience Replay store.

    Two-tier storage:
      - In-memory SumTree: hot-path sampling during training (O(log n))
      - PostgreSQL vapt_rl_transitions: durable persistence for offline
        learning across restarts

    Async/sync split:
      - add()                → async (DB INSERT + SumTree update)
      - sample()             → sync (SumTree only — hot-path)
      - update_priorities()  → sync (SumTree only — hot-path)
      - sample_from_db()     → async (DB SELECT for offline training)
      - persist_priorities() → async (optional DB priority sync, batched)

    PER parameters (per master plan §12 W16):
      - alpha: 0.6
      - beta: 0.4 → 1.0 annealed over 100,000 steps
    """

    def __init__(self, max_buffer: int = MAX_MEMORY_BUFFER) -> None:
        self.max_buffer = max_buffer
        self.tree = SumTree(max_buffer)
        self._beta = PER_BETA_START
        self._beta_step = 0

    # ── write ─────────────────────────────────────────────────────────

    async def add(
        self,
        scan_id: str,
        turn: int,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        priority: float | None = None,
        action_name: str | None = None,
        reward_breakdown: dict[str, Any] | None = None,
        session: Any = None,  # AsyncSession | None
    ) -> None:
        """Add a transition with max priority (for new experiences).

        Args:
            scan_id: scan identifier (FK to vapt_scans.id)
            turn: turn number within scan
            state: 337-dim state vector (np.float32)
            action: action index 0-6
            reward: scalar reward
            next_state: 337-dim next state vector
            done: terminal flag
            priority: optional initial priority. If None, uses max priority
                in tree (ensures new experiences are sampled at least once).
            action_name: human-readable action name (e.g. "execute_command_recon")
            reward_breakdown: optional multi-signal reward breakdown dict
            session: optional SQLAlchemy AsyncSession. If provided, uses it
                for DB INSERT (caller manages commit). If None, opens a
                fresh async session internally.

        Side effects:
            - Updates in-memory SumTree (always)
            - INSERTs into vapt_rl_transitions (best-effort, never raises)

        The in-memory copy is a **private read-only snapshot** of `state` /
        `next_state` (`_snapshot_vector`), not a reference to the caller's
        array — the caller may recycle its buffer once this returns. The DB
        path is unaffected: `_persist_transition` receives the original
        arrays and `_insert_transition_row` builds JSON with its own
        `.tolist()`, which is transient rather than retained per buffer slot.
        """
        transition = {
            "scan_id": scan_id,
            "turn": int(turn),
            # float32 arrays, NOT `.tolist()` — see `_snapshot_vector` for the
            # measured 6.1x memory difference. `sample()` already returns
            # arrays (it converts legacy list payloads), so this is the shape
            # the training path has always consumed.
            "state": _snapshot_vector(state),
            "action": int(action),
            "reward": float(reward),
            "next_state": _snapshot_vector(next_state),
            "done": bool(done),
        }

        # PER priority: new transitions inherit the running max-priority so
        # they get sampled at least once before refinement.
        #
        # NOTE (monitoring, not a defect): `_max_priority` never decreases —
        # that is standard PER, and the invariant that every new transition is
        # sampled at least once depends on it. Combined with reward clipping at
        # [-100, +150], a single outlier transition can hold the maximum
        # permanently and be oversampled for the life of the buffer. If Q is
        # observed oscillating, cap priorities (e.g. at the 99th percentile)
        # rather than removing the max-inheritance rule.
        if priority is None:
            per_priority = float(self.tree._max_priority) or 1.0
        else:
            per_priority = (abs(priority) + PER_EPSILON) ** PER_ALPHA

        self.tree.add(per_priority, transition)

        # Persist to DB (best-effort — never raises)
        await self._persist_transition(
            scan_id=scan_id,
            turn=turn,
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
            per_priority=per_priority,
            action_name=action_name,
            reward_breakdown=reward_breakdown,
            session=session,
        )

    async def _persist_transition(
        self,
        scan_id: str,
        turn: int,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        per_priority: float,
        action_name: str | None,
        reward_breakdown: dict[str, Any] | None,
        session: Any,
    ) -> None:
        """INSERT transition into vapt_rl_transitions. Best-effort, never raises."""
        if session is not None:
            await self._insert_transition_row(
                session=session,
                scan_id=scan_id, turn=turn,
                state=state, action=action, reward=reward,
                next_state=next_state, done=done,
                per_priority=per_priority,
                action_name=action_name,
                reward_breakdown=reward_breakdown,
            )
            return

        # No session provided — open a fresh one
        try:
            from app.db.session import async_session
            async with async_session() as sess:
                await self._insert_transition_row(
                    session=sess,
                    scan_id=scan_id, turn=turn,
                    state=state, action=action, reward=reward,
                    next_state=next_state, done=done,
                    per_priority=per_priority,
                    action_name=action_name,
                    reward_breakdown=reward_breakdown,
                )
                await sess.commit()
        except Exception as exc:
            logger.warning("experience persist failed: %s", exc)

    async def _insert_transition_row(
        self,
        session: Any,
        scan_id: str,
        turn: int,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        per_priority: float,
        action_name: str | None,
        reward_breakdown: dict[str, Any] | None,
    ) -> None:
        """Actual INSERT using RLTransition SQLAlchemy model."""
        try:
            from app.db.models.rl import RLTransition
            row = RLTransition(
                scan_id=scan_id,
                turn=int(turn),
                state_vector=state.astype(np.float32).tolist(),
                action=int(action),
                reward=float(reward),
                next_state_vector=next_state.astype(np.float32).tolist(),
                done=bool(done),
                priority=float(per_priority),
                action_name=action_name,
                reward_breakdown=reward_breakdown,
            )
            session.add(row)
            await session.flush()
        except Exception as exc:
            logger.warning("RLTransition INSERT failed: %s", exc)

    def update_priorities(self, indices: list[int], td_errors: list[float]) -> None:
        """
        Update SumTree priorities after a training step (sync, hot-path).

        DB priority sync is deferred to persist_priorities() (async, batched)
        — the SumTree is the source of truth for sampling during a scan.
        """
        for idx, td_error in zip(indices, td_errors):
            priority = (abs(td_error) + PER_EPSILON) ** PER_ALPHA
            self.tree.update(idx, priority)

    async def persist_priorities(self, batch_size: int = 500) -> int:
        """
        Sync SumTree priorities back to DB (best-effort, batched).

        Returns:
            number of rows updated (0 if DB unavailable or empty tree).
        """
        if self.tree.size == 0:
            return 0
        # This is a no-op stub for W16 — full priority sync is W17+ scope
        # (requires storing tree_idx ↔ DB row_id mapping). The SumTree is
        # the source of truth during a scan; DB rows are for offline training
        # which uses sample_from_db() (uniform DB sample, no PER).
        return 0

    # ── read / sample ─────────────────────────────────────────────────

    def sample(self, batch_size: int = 64) -> tuple[list[dict[str, Any]], list[float], list[int]]:
        """
        Sample a prioritized batch with importance-sampling weights (sync).

        Returns:
            (batch, is_weights, tree_indices)
              - batch: list of transition dicts (with `is_weight` injected)
              - is_weights: importance-sampling weights for bias correction
              - tree_indices: indices for priority updates after training

        Every `state` / `next_state` in the returned batch is a **read-only
        float32 ndarray**, and it is the same object the tree holds — sample()
        does not copy 1.3 KB per sample on the training hot path. Consumers
        that need to modify one must copy it first; in-place writes raise
        instead of corrupting the replay buffer.
        """
        if self.tree.size == 0:
            return [], [], []

        n = min(batch_size, self.tree.size)
        indices, priorities, samples = self.tree.sample(n)

        # Compute importance-sampling weights: w_i = (N · P(i))^(-β)
        total = self.tree.total()
        if total <= 0:
            return [], [], []

        sampling_probs = np.array(priorities, dtype=np.float64) / total

        # Advance the anneal by the number of samples about to be handed out —
        # that is the number of gradient steps the caller will take, which is
        # the denominator `PER_BETA_ANNEAL_STEPS` is expressed in.
        #
        # `PER_BETA_ANNEAL_STEPS` used to be compared against a counter that
        # advanced by 1 per `sample()` call, making the anneal batch-size
        # dependent: at batch_size=64, β=1.0 needed 100,000 calls ≈ 6.4 million
        # gradient steps — 64× slower than the constant's name promises, so β
        # stayed near its 0.4 floor (weak correction) for the practical life of
        # the buffer.
        self._beta_step += n
        self._beta = min(
            PER_BETA_END,
            PER_BETA_START + (PER_BETA_END - PER_BETA_START)
            * self._beta_step / PER_BETA_ANNEAL_STEPS,
        )

        is_weights = np.power(self.tree.size * sampling_probs, -self._beta)
        max_w = is_weights.max() if is_weights.size > 0 else 1.0
        if max_w > 0:
            is_weights /= max_w  # normalize for stability

        batch: list[dict[str, Any]] = []
        kept_weights: list[float] = []
        kept_indices: list[int] = []
        for sample, w, idx in zip(samples, is_weights, indices):
            if sample is None:
                continue
            s = dict(sample)
            s["is_weight"] = float(w)
            # Legacy payloads (rows restored by an older build, or hand-built
            # test data) may still carry lists. Normalise them through the
            # same freezer so the whole batch obeys one contract: read-only
            # float32 arrays.
            if not isinstance(s.get("state"), np.ndarray):
                s["state"] = _snapshot_vector(s["state"])
            if not isinstance(s.get("next_state"), np.ndarray):
                s["next_state"] = _snapshot_vector(s["next_state"])
            batch.append(s)
            kept_weights.append(float(w))
            kept_indices.append(idx)

        # The three lists are positionally aligned by construction. They were
        # NOT before: `batch` skipped null samples while `is_weights` and
        # `indices` were returned whole, so one null leaf made
        # `update_priorities(indices, td_errors)` write every priority from
        # that position onward onto the WRONG leaf — silently, and for the
        # rest of the scan. A null leaf is reachable: `SumTree.get()` returns
        # whatever sits at the sampled leaf, and leaves that were never
        # written hold None.
        return batch, kept_weights, kept_indices

    async def sample_from_db(
        self,
        batch_size: int = 256,
        scan_id: str | None = None,
        session: Any = None,
    ) -> list[dict[str, Any]]:
        """Sample UNIFORMLY (or by scan_id) from Postgres for offline training.

        HOW uniformity is achieved: `ORDER BY random()`. This used to be
        `ORDER BY priority DESC`, which contradicted this docstring in two
        ways, so state both explicitly:

        1. It was *not* uniform. `LIMIT n` over a `priority DESC` ordering
           returns the same n rows on every call — no diversity at all, and
           `persist_priorities()` is a documented no-op stub (W17+ scope), so
           the stored `priority` never changes between runs. Repeated offline
           training therefore overfit to one fixed slice of the table.
        2. That slice was *not* the high-TD-error rows either. The DB
           `priority` column is written once at INSERT time from either
           `SumTree._max_priority` (a running in-memory maximum that every
           later row inherits) or `abs(rl_reward) * PER_PRIORITY_MULTIPLIER`
           for self-correction rejections. TD errors never reach it because
           `persist_priorities()` returns 0 without touching the DB. So
           `ORDER BY priority DESC` selected an arbitrary, priority-coincident
           subset — a bias with no meaning behind it.

        Because sampling is now truly uniform, `is_weight = 1.0` is the
        correct importance-sampling weight (uniform ⇒ no correction), and
        `PentestPolicy.train_from_db()` deliberately passes no `is_weights`,
        letting this per-sample value stand.

        Cost note: `ORDER BY random()` requires a full sort of the filtered
        set (the `priority` index cannot help). With `scan_id` that is a few
        hundred rows and fine. Unbounded (scan_id=None) over the 90-day
        retention window it degrades to O(N log N) — if that ever becomes
        hot, switch to `TABLESAMPLE SYSTEM` rather than reintroducing an
        ordering that carries no sampling meaning.

        Args:
            batch_size: max rows to return
            scan_id: optional filter (sample only from this scan's transitions)
            session: optional caller-owned AsyncSession. `training_loop.
                train_from_scan()` has always accepted a `session` argument and
                silently dropped it; it is forwarded here so the documented
                parameter actually scopes the read. A caller-owned session is
                used as-is (never committed/closed); without one we open and
                close our own.

        Returns:
            list of transition dicts with numpy state/next_state arrays.
            is_weight is set to 1.0 (uniform sampling — no IS correction).
        """
        try:
            from sqlalchemy import func, select
            from app.db.session import async_session
            from app.db.models.rl import RLTransition
        except ImportError as exc:
            logger.warning("RLTransition import failed: %s", exc)
            return []

        try:
            stmt = select(RLTransition)
            if scan_id is not None:
                stmt = stmt.where(RLTransition.scan_id == scan_id)
            # ORDER BY random() — uniform. NOT `priority DESC`: that returned
            # the same top-n rows every run (see the docstring for why the
            # ordering carried no meaning), which made offline training
            # both deterministic and biased.
            stmt = stmt.order_by(func.random()).limit(batch_size)

            async def _fetch(sess: Any):
                result = await sess.execute(stmt)
                return result.scalars().all()

            if session is not None:
                rows = await _fetch(session)
            else:
                async with async_session() as sess:
                    rows = await _fetch(sess)

            batch: list[dict[str, Any]] = []
            for row in rows:
                state_list = row.state_vector if isinstance(row.state_vector, list) else json.loads(row.state_vector)
                next_list = row.next_state_vector if isinstance(row.next_state_vector, list) else json.loads(row.next_state_vector)
                batch.append({
                    "scan_id": row.scan_id,
                    "turn": row.turn,
                    "state": np.array(state_list, dtype=np.float32),
                    "action": int(row.action),
                    "reward": float(row.reward),
                    "next_state": np.array(next_list, dtype=np.float32),
                    "done": bool(row.done),
                    "is_weight": 1.0,  # uniform (ORDER BY random) → no IS correction
                    "action_name": row.action_name,
                    "reward_breakdown": row.reward_breakdown,
                })
            return batch
        except Exception as exc:
            logger.warning("DB sample failed: %s", exc)
            return []

    # ── introspection ─────────────────────────────────────────────────

    def __len__(self) -> int:
        return self.tree.size

    @property
    def beta(self) -> float:
        """Current PER importance-sampling exponent β (0.4 → 1.0).

        Public so callers do not have to reach into `_beta` through
        `hasattr()` / `getattr()`. `PentestPolicy.train_from_memory()` logs
        it, and a private-attribute probe there was the only reason the
        attribute name was load-bearing across module boundaries.
        """
        return float(self._beta)

    @property
    def beta_step(self) -> int:
        """Gradient steps' worth of samples that have advanced the β anneal.

        Not the number of `sample()` calls: one call with `batch_size=64`
        advances this by 64 (or fewer if empty leaves were skipped), because
        `PER_BETA_ANNEAL_STEPS` is denominated in training steps.
        """
        return int(self._beta_step)

    def get_stats(self) -> dict[str, Any]:
        return {
            "size": self.tree.size,
            "capacity": self.max_buffer,
            "total_priority": self.tree.total(),
            "beta": round(self._beta, 4),
            "beta_step": self._beta_step,
            "alpha": PER_ALPHA,
            "beta_start": PER_BETA_START,
            "beta_end": PER_BETA_END,
            "beta_anneal_steps": PER_BETA_ANNEAL_STEPS,
            "max_priority": self.tree._max_priority,
        }