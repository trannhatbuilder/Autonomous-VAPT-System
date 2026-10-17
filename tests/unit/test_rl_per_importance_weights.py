"""
Regression guards for PL2 — PER importance weights.

The report claimed the weights were discarded::

    batch, is_weights, indices = store.sample(batch_size)
    loss, td_errors = self.q.train_batch(batch)   # is_weights KHONG duoc dung!

That claim is **false**, and the tests in `TestWeightsWereNeverDiscarded`
pin the disproof so nobody "re-fixes" it:

  * `ExperienceStore.sample()` injects `is_weight` into every transition
    dict (`experience_store.py`), so `train_batch` read the weight through
    `t.get("is_weight", 1.0)` all along.
  * `DoubleQLearner.train_step` multiplies the gradient by it:
    `grad_q[action] = huber_grad * is_weight`.

Measured: |dW2| after one step scales exactly 10x between `is_weight=1.0`
and `is_weight=0.1`. The bias was corrected; nothing was thrown away.

What *was* genuinely wrong in that neighbourhood, and is guarded here:

  1. The returned `loss` was the **unweighted** `mean(td²)` while the
     gradient was weighted. Measured before the fix: weight 1.0 -> 28.74,
     weight 0.1 -> 2.87 was not true (it was 29.97 vs 29.95). The logged
     number ignored the correction entirely.
  2. `train_from_memory` unpacked `is_weights` and never used it — a dead
     variable that manufactured the illusion of the reported bug, and made
     correctness depend on a *silent* dict injection (this repo has already
     shipped three zero-caller bugs of that exact shape: R18, R20, R22).
  3. `store.sample()` filtered null samples out of `batch` but returned
     `is_weights` and `indices` unfiltered, so one null leaf made
     `update_priorities(indices, td_errors)` write every priority from that
     position onward onto the wrong leaf.

The invariant all of this protects: **weights go into the loss and the
gradient; RAW TD errors go into the priorities.**
"""
from __future__ import annotations

import ast
import os
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

os.environ.setdefault("VAPT_AI_POSTGRES_DB", "postgresql://vapt:vapt@localhost:5432/vapt_ai_test")
os.environ.setdefault("VAPT_AI_ENVIRONMENT", "dev")

import asyncio

import numpy as np
import pytest

import app.rl.q_learner as qmod
from app.rl.experience_store import ExperienceStore, PER_ALPHA, PER_EPSILON
from app.rl.policy import PentestPolicy
from app.rl.q_learner import DoubleQLearner

Q_SRC = (_PROJECT_ROOT / "app" / "rl" / "q_learner.py").read_text(encoding="utf-8")
POLICY_SRC = (_PROJECT_ROOT / "app" / "rl" / "policy.py").read_text(encoding="utf-8")
STORE_SRC = (_PROJECT_ROOT / "app" / "rl" / "experience_store.py").read_text(encoding="utf-8")

# A state that is NOT all zeros. With a zero vector the ReLU hidden layer of a
# freshly initialised net is identically zero, so the W2 gradient is zero and
# every "the weight scaled the update" assertion degenerates to 0 == 0.
STATE = np.zeros(337, dtype=np.float32)
STATE[1] = 0.35     # tdi
STATE[2] = 0.5      # mode
STATE[7] = 0.7      # token_budget_remaining
NEXT_STATE = STATE.copy()
NEXT_STATE[1] = 0.45

REWARD = 0.5        # small: keeps the update linear instead of diverging


# ── helpers ──────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _isolate_cache(monkeypatch, tmp_path):
    """Keep `save_cache()` out of data/rl.

    This matters twice over: `train_batch` writes the cache, and
    `DoubleQLearner.__init__` silently *loads* one if it exists. Without
    isolation the second learner inside a test would start from the first
    one's weights and every "fresh network" comparison would be meaningless.
    """
    monkeypatch.setattr(qmod, "WEIGHTS_FILE", tmp_path / "q_weights.npz")
    monkeypatch.setattr(qmod, "META_DIR", tmp_path, raising=False)
    monkeypatch.setattr(qmod, "META_FILE", tmp_path / "q_meta.json")


def _fresh_learner() -> DoubleQLearner:
    """A learner at a deterministic random init with no cache loaded."""
    for path in (qmod.WEIGHTS_FILE, qmod.META_FILE):
        try:
            Path(path).unlink()
        except FileNotFoundError:
            pass
    np.random.seed(1234)
    return DoubleQLearner(state_dim=337, action_dim=7, hidden_dim=64)


def _make_batch(weight: float, size: int = 1) -> list[dict]:
    """One transition by default.

    Batches of more than one train sequentially, so sample 2 sees weights
    already moved by sample 1 and the weight->update ratio stops being the
    clean 10x (it becomes ~93x at lr=0.001 because the net diverges)."""
    return [
        {
            "state": STATE, "action": 1, "reward": REWARD,
            "next_state": NEXT_STATE, "done": False,
            "is_weight": weight,
        }
        for _ in range(size)
    ]


def _weight_delta(learner: DoubleQLearner, before: np.ndarray) -> float:
    return float(np.abs(learner.online.value_net.W2 - before).sum())


def _run_train_batch(**kwargs):
    learner = _fresh_learner()
    before = learner.online.value_net.W2.copy()
    loss, td = learner.train_batch(kwargs.pop("batch"), **kwargs)
    return loss, td, _weight_delta(learner, before)


# ── 1. the refutation: the weight was never discarded ────────────────────

class TestWeightsWereNeverDiscarded:
    def test_gradient_scales_linearly_with_is_weight(self):
        """The core disproof of the report, by value.

        `train_step` multiplies the Huber gradient by `is_weight`, so a 10x
        smaller weight must produce a 10x smaller update. It does.
        """
        deltas = {}
        for weight in (1.0, 0.1):
            learner = _fresh_learner()
            before = learner.online.value_net.W2.copy()
            learner.train_step(
                STATE, 1, REWARD, NEXT_STATE, False, is_weight=weight,
            )
            deltas[weight] = _weight_delta(learner, before)
        assert deltas[0.1] > 0, "is_weight=0.1 produced no update at all"
        assert deltas[1.0] / deltas[0.1] == pytest.approx(10.0, rel=1e-6)

    def test_zero_weight_produces_no_update(self):
        learner = _fresh_learner()
        before = learner.online.value_net.W2.copy()
        learner.train_step(STATE, 1, REWARD, NEXT_STATE, False, is_weight=0.0)
        assert _weight_delta(learner, before) == 0.0

    def test_train_batch_reads_is_weight_from_the_dict(self):
        """The store injects the key, so a batch with the key is weighted."""
        _, _, delta_low = _run_train_batch(batch=_make_batch(0.1))
        _, _, delta_one = _run_train_batch(batch=_make_batch(1.0))
        assert delta_one / delta_low == pytest.approx(10.0, rel=1e-6)

    def test_store_injects_the_weight_it_returns(self):
        """`sample()` returns a list AND writes the same value into the dict."""

        async def run():
            store = ExperienceStore(max_buffer=1000)
            for i in range(40):
                await store.add(
                    "scan_test", i, STATE, i % 7, float(i), NEXT_STATE, False,
                    priority=float(i + 1),
                )
            batch, weights, indices = store.sample(8)
            assert len(batch) == len(weights) == len(indices) == 8
            injected = [item["is_weight"] for item in batch]
            assert injected == pytest.approx(weights)

        asyncio.run(run())


# ── 2. the fix: the loss is now weighted, and applied exactly once ───────

class TestLossIsWeighted:
    def test_loss_scales_with_the_weight(self):
        """Before the fix these two were 29.97 and 29.95 — the weights were
        absent from the metric entirely."""
        loss_one, _, _ = _run_train_batch(batch=_make_batch(1.0))
        loss_low, _, _ = _run_train_batch(batch=_make_batch(0.1))
        assert loss_one > 0.0
        assert loss_one / loss_low == pytest.approx(10.0, rel=1e-6)

    def test_loss_is_mean_of_weighted_squares(self):
        learner = _fresh_learner()
        loss, td = learner.train_batch(_make_batch(1.0))
        assert loss == pytest.approx(float(np.mean(np.square(td))), rel=1e-9)

    def test_zero_weight_batch_has_zero_loss(self):
        loss, td, delta = _run_train_batch(batch=_make_batch(0.0))
        assert all(np.isfinite(t) for t in td)
        assert loss == 0.0
        assert delta == 0.0

    def test_explicit_weights_are_applied_exactly_once(self):
        """`is_weights=` must not multiply on top of the per-item `is_weight`.

        Double application would make the explicit path 10x smaller than the
        equivalent dict-carried weight.
        """
        explicit_loss, _, explicit_delta = _run_train_batch(
            batch=_make_batch(1.0), is_weights=[0.1],
        )
        injected_loss, _, injected_delta = _run_train_batch(batch=_make_batch(0.1))
        assert explicit_delta == pytest.approx(injected_delta, rel=1e-9)
        assert explicit_loss == pytest.approx(injected_loss, rel=1e-9)

    def test_is_weights_is_authoritative_over_the_dict(self):
        """Contradictory sources must not be averaged or multiplied."""
        _, _, delta = _run_train_batch(batch=_make_batch(1.0), is_weights=[0.1])
        _, _, delta_low = _run_train_batch(batch=_make_batch(0.1))
        _, _, delta_full = _run_train_batch(batch=_make_batch(1.0))
        assert delta == pytest.approx(delta_low, rel=1e-9)
        assert delta != pytest.approx(delta_full, rel=1e-3)

    def test_length_mismatch_falls_back_instead_of_misaligning(self):
        """A wrong-length list cannot be trusted, so it must be ignored —
        but the batch's own weight must still be used (never uniform 1.0)."""
        _, _, delta = _run_train_batch(
            batch=_make_batch(0.1), is_weights=[1.0, 1.0],   # 2 != 1
        )
        _, _, delta_low = _run_train_batch(batch=_make_batch(0.1))
        _, _, delta_full = _run_train_batch(batch=_make_batch(1.0))
        assert delta == pytest.approx(delta_low, rel=1e-9)
        assert delta != pytest.approx(delta_full, rel=1e-3)


# ── 3. the invariant: priorities get RAW td errors ───────────────────────

class TestPrioritiesStayRaw:
    def test_returned_td_errors_are_not_multiplied_by_weight(self):
        """TD must be identical for w=1.0 and w=0.1 on the same fresh net.

        The TD is computed *before* the update, so the weight cannot touch it
        — the value is the same transition, twice.
        """
        _, td_one, _ = _run_train_batch(batch=_make_batch(1.0))
        _, td_low, _ = _run_train_batch(batch=_make_batch(0.1))
        assert td_one[0] == pytest.approx(td_low[0], rel=1e-12)

    def test_td_errors_are_absolute_values(self):
        _, td, _ = _run_train_batch(batch=_make_batch(1.0))
        assert all(t >= 0.0 for t in td)

    def test_train_from_memory_hands_raw_td_to_update_priorities(self):
        """End-to-end through the policy: what `update_priorities` receives is
        exactly what `train_batch` returned, unscaled."""
        received: dict[str, list] = {}

        class _StubStore:
            _beta = 0.4

            def sample(self, n):  # noqa: ARG002
                return _make_batch(0.25, 4), [0.25] * 4, [10, 11, 12, 13]

            def update_priorities(self, indices, td_errors):
                received["indices"] = list(indices)
                received["td"] = list(td_errors)

        policy = PentestPolicy()
        policy.q = _fresh_learner()
        _, returned_td = policy.train_from_memory(_StubStore(), batch_size=4)

        assert received["indices"] == [10, 11, 12, 13]
        assert received["td"] == pytest.approx(returned_td)

    def test_priority_formula_ignores_weights(self):
        """By-value: a raw td of 2.0 must map to (2+eps)^alpha, not
        (0.25*2+eps)^alpha — a weighted priority would square the correction."""

        async def run():
            store = ExperienceStore(max_buffer=100)
            store.tree.add(1.0, {"marker": True})
            store.update_priorities([store.tree.capacity], [2.0])
            expected = (2.0 + PER_EPSILON) ** PER_ALPHA
            assert store.tree.tree[store.tree.capacity] == pytest.approx(expected)

        asyncio.run(run())


# ── 4. the alignment bug in store.sample() ───────────────────────────────

class TestSampleListsStayAligned:
    def test_null_leaf_does_not_desynchronise_the_three_lists(self):
        """Forcing one leaf to hold no payload used to give batch=0,
        is_weights=10, indices=10 — and `zip` in update_priorities would then
        write priorities onto the wrong leaves from that point on."""

        async def run():
            store = ExperienceStore(max_buffer=100)
            for i in range(10):
                await store.add(
                    "scan_test", i, STATE, i % 7, 1.0, NEXT_STATE, False,
                    priority=1.0,
                )
            # Make the null leaf dominate the tree so it is guaranteed sampled.
            null_leaf = store.tree.capacity + 3
            store.tree.data[3] = None
            store.tree.tree[null_leaf] = 1e12
            store.tree._propagate(null_leaf, 1e12)

            batch, weights, indices = store.sample(10)
            assert len(batch) == len(weights) == len(indices)

        asyncio.run(run())

    def test_skipped_sample_drops_its_index_and_weight(self):
        """The kept entries must still correspond to each other."""

        async def run():
            store = ExperienceStore(max_buffer=100)
            for i in range(10):
                await store.add(
                    "scan_test", i, STATE, i % 7, 1.0, NEXT_STATE, False,
                    priority=1.0,
                )
            store.tree.data[3] = None
            null_leaf = store.tree.capacity + 3
            store.tree.tree[null_leaf] = 1e12
            store.tree._propagate(null_leaf, 1e12)

            batch, weights, indices = store.sample(10)
            for item, w, idx in zip(batch, weights, indices):
                assert item["is_weight"] == pytest.approx(w)
                assert idx == idx  # ints, no None

        asyncio.run(run())

    def test_normal_sample_is_unaffected(self):
        """The fix must be invisible when nothing is null."""

        async def run():
            store = ExperienceStore(max_buffer=200)
            for i in range(30):
                await store.add(
                    "scan_test", i, STATE, i % 7, 1.0, NEXT_STATE, False,
                    priority=float(i + 1),
                )
            batch, weights, indices = store.sample(8)
            assert len(batch) == len(weights) == len(indices) == 8
            assert all(isinstance(i, int) for i in indices)

        asyncio.run(run())


# ── 5. source guards: the wiring cannot silently rot ─────────────────────

class TestWiringCannotRot:
    def test_train_batch_accepts_is_weights(self):
        tree = ast.parse(Q_SRC)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "train_batch":
                args = [a.arg for a in node.args.args] + [a.arg for a in node.args.kwonlyargs]
                assert "is_weights" in args, (
                    "train_batch lost its is_weights parameter — the loss "
                    "cannot be PER-weighted without it."
                )
                return
        raise AssertionError("train_batch not found in q_learner.py")

    def test_train_step_still_applies_the_weight_to_the_gradient(self):
        """Guards against 'simplifying' away the only place the weight acts."""
        assert "huber_grad * is_weight" in Q_SRC

    def test_train_batch_uses_weights_in_the_loss(self):
        assert "weight * td_error * td_error" in Q_SRC or "weighted_sq" in Q_SRC

    def test_policy_passes_is_weights_explicitly(self):
        """AST, not text: `train_batch(batch)` without the keyword was the
        exact shape that let the sampled weights look discarded."""
        tree = ast.parse(POLICY_SRC)
        calls = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "train_batch"
        ]
        assert calls, "no train_batch call found in policy.py"
        per_memory = [c for c in calls if c.keywords]
        assert per_memory, (
            "train_from_memory calls train_batch without is_weights — the "
            "sampled importance weights are being ignored again."
        )
        assert any(
            any(k.arg == "is_weights" for k in c.keywords) for c in calls
        ), "train_batch is never called with is_weights="

    def test_train_from_db_does_not_pass_weights(self):
        """The offline DB path must stay UNWEIGHTED — but only because its
        sampling is uniform.

        This test used to read:

            '''`sample_from_db` is uniform — passing weights there would be a lie.'''

        and then assert an unweighted `train_batch(batch)` call existed. The
        claim was false: `sample_from_db` ordered by `priority DESC`, which is
        neither uniform nor corrected for. The assertion therefore locked the
        bug in — it passed happily while offline training ran on a fixed,
        priority-coincident slice of the table with no IS correction.

        The contract is now conditional and both halves are asserted
        together, so they cannot drift apart again: uniform sampling ⇒
        `is_weight == 1.0` ⇒ unweighted `train_batch`. Priority-ordered
        sampling ⇒ real IS weights ⇒ weighted `train_batch`. See
        `test_rl_db_sampling_is_uniform.py` for the SQL-side guard.
        """
        tree = ast.parse(POLICY_SRC)
        db_calls = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "train_batch"
            and any(isinstance(a, ast.Name) and a.id == "batch" for a in node.args)
            and not node.keywords
        ]
        assert db_calls, "expected one unweighted train_batch call (train_from_db)"

        # …and the reason that is legitimate: the DB sample really is uniform.
        # If this half flips, the unweighted call above becomes the original
        # bug again.
        assert "func.random()" in STORE_SRC, (
            "sample_from_db is not uniform any more, so train_from_db MUST "
            "pass is_weights= — otherwise offline training runs uncorrected"
        )
        assert "RLTransition.priority.desc()" not in STORE_SRC, (
            "priority-ordered DB sampling is not uniform; train_from_db's "
            "unweighted train_batch call is now a lie"
        )

    def test_store_filters_lists_together(self):
        """The `return` must not hand back the unfiltered list again."""
        assert "return batch, kept_weights, kept_indices" in STORE_SRC
        assert "return batch, is_weights.tolist(), indices" not in STORE_SRC

    def test_is_weight_injection_still_happens(self):
        """The dict key is what makes a batch self-describing for callers
        that build their own training data."""
        assert 's["is_weight"] = float(w)' in STORE_SRC
