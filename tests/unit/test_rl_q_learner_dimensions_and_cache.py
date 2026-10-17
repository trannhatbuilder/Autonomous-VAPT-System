"""Guards for QL1–QL4 / EX3–EX4 (R33).

Five reports, three real defects, two notes. Each class gets one guard block:

QL1  `_load_from_cache()` validated only `value_W1.shape[1]` (state_dim). The
     other dimensions were installed unchecked by
     `TwoLayerMLP.set_weights()`, which copies blindly with no shape test:
       * action_dim 7→8 → 7 Q-values from an 8-action network, action 7
         unreachable, silently, forever.
       * hidden_dim 64→128 → does NOT crash (contrary to the report): the
         cached set is self-consistent, so the network runs 64-wide while
         `get_stats()` reports 128.
QL2  `ACTION_DIM = 7` was a duplicate of `len(ACTION_SPACE)` justified by a
     circular import that did not exist. Now derived in `app/rl/actions.py`.
QL3  `train_batch()` wrote the .npz on every batch (177 KiB, ~3.9 ms).
QL4  `train_step()` ran both MLP streams twice to recover `h1` for backprop.
EX3  β annealed per `sample()` call instead of per gradient step (64× slow).
EX4  `_max_priority` never decreases — correct by design; documented only.
"""

from __future__ import annotations

import ast
import inspect
import json
import textwrap

import numpy as np
import pytest

import app.rl.q_learner as qmod
from app.rl import actions as actions_mod
from app.rl import state_encoder
from app.rl.experience_store import (
    PER_BETA_ANNEAL_STEPS,
    PER_BETA_END,
    PER_BETA_START,
    ExperienceStore,
)
from app.rl.q_learner import (
    CACHE_SAVE_MIN_BATCHES,
    CACHE_SAVE_MIN_SECONDS,
    DoubleQLearner,
    DuelingNetwork,
)

STATE_DIM = state_encoder.STATE_DIM


# ── helpers ──────────────────────────────────────────────────────────────


@pytest.fixture()
def cache_paths(tmp_path, monkeypatch):
    weights = tmp_path / "q_weights.npz"
    meta = tmp_path / "q_meta.json"
    monkeypatch.setattr(qmod, "WEIGHTS_FILE", weights)
    monkeypatch.setattr(qmod, "META_FILE", meta)
    return weights, meta


def _write_cache(paths, *, state_dim=STATE_DIM, action_dim=7, hidden_dim=64,
                 step: int = 500, layout: str | None = None) -> None:
    weights, meta = paths
    net = DuelingNetwork(state_dim, action_dim, hidden_dim)
    np.savez(weights, **net.get_weights())
    # `save_cache()` stamps the state-vector layout; a fixture must too, or
    # the layout guard quarantines the cache before the shape guard can run
    # and every test here would pass for the wrong reason. `layout=None`
    # reproduces a pre-SE4 cache (same shapes, no stamp).
    meta.write_text(json.dumps({
        "step": step,
        "lr": 0.001,
        "layout": qmod.state_layout_id() if layout is None else layout,
    }))


def _batch(n: int = 2) -> list[dict]:
    return [
        {
            "state": np.random.randn(STATE_DIM).astype(np.float32),
            "action": 1,
            "reward": 0.5,
            "next_state": np.random.randn(STATE_DIM).astype(np.float32),
            "done": False,
            "is_weight": 1.0,
        }
        for _ in range(n)
    ]


# ── QL1: every dimension is validated, not just state_dim ────────────────


class TestCacheShapeGuard:
    def test_action_dim_mismatch_is_quarantined(self, cache_paths):
        """7-action network loading an 8-action cache must not adopt it.

        Before the fix this "succeeded": adv_W2 (8, 64) was copied in, forward
        returned 8 Q-values into a 7-action pipeline (or the mirror image),
        and the last action became unreachable without a single warning.
        """
        _write_cache(cache_paths, action_dim=8)
        learner = DoubleQLearner()
        weights, meta = cache_paths
        assert learner._step == 0, "step must reset with the quarantined cache"
        assert not weights.exists(), "stale cache was not quarantined"
        assert list(weights.parent.glob("q_weights.stale-shape.*.npz"))

    def test_hidden_dim_mismatch_is_quarantined(self, cache_paths):
        """The report claimed this crashes; it does not — it silently shrinks.

        A cached W1 of shape (64, 337) adopted by a 128-wide network produces
        a working forward pass with a 64-wide hidden layer. Nothing raises, so
        only an explicit shape guard can catch it.
        """
        _write_cache(cache_paths, hidden_dim=128)
        learner = DoubleQLearner()
        assert learner._step == 0
        assert not cache_paths[0].exists()

    def test_state_dim_mismatch_still_quarantined(self, cache_paths):
        """The original (EVVO P0-6) behaviour must survive the generalisation."""
        _write_cache(cache_paths, state_dim=128)
        learner = DoubleQLearner()
        assert learner._step == 0
        assert not cache_paths[0].exists()

    def test_arbitrary_future_dimension_is_covered(self, cache_paths):
        """The guard is derived from the live network, so a NEW dim is covered.

        This is the class fix: an `action_dim=9` learner must reject a 7-action
        cache without anyone editing `_load_from_cache` again.
        """
        _write_cache(cache_paths, action_dim=7)
        learner = DoubleQLearner(action_dim=9)
        assert learner._step == 0
        assert learner.online.advantage_net.W2.shape == (9, 64), (
            "network must keep its own freshly-initialised width"
        )

    def test_missing_weight_key_is_quarantined(self, cache_paths):
        """A partial cache would KeyError inside set_weights and be swallowed."""
        weights, meta = cache_paths
        full = DuelingNetwork(STATE_DIM, 7, 64).get_weights()
        full.pop("adv_b2")
        np.savez(weights, **full)
        meta.write_text(json.dumps({"step": 42}))

        learner = DoubleQLearner()
        assert learner._step == 0
        assert not weights.exists()

    def test_matching_cache_still_loads(self, cache_paths):
        """The guard must not reject a good cache — otherwise it proves nothing."""
        _write_cache(cache_paths, step=1234)
        learner = DoubleQLearner()
        assert learner._step == 1234
        assert cache_paths[0].exists(), "a valid cache must not be quarantined"

    def test_loaded_network_emits_action_dim_values(self, cache_paths):
        """The invariant the bug violated: |q_values| == ACTION_DIM."""
        _write_cache(cache_paths)
        learner = DoubleQLearner()
        q = learner.q_values(np.zeros(STATE_DIM, dtype=np.float32))
        assert q.shape == (qmod.ACTION_DIM,)

    def test_guard_reads_expected_shapes_from_the_network(self):
        """AST: shapes are derived, not restated as literals.

        A hardcoded expected-shape table would be the same defect class as the
        `ACTION_DIM = 7` literal — it can drift from the architecture it is
        supposed to protect.
        """
        src = inspect.getsource(DoubleQLearner._load_from_cache)
        assert "self.online.get_weights()" in src, (
            "expected shapes must come from the live network"
        )
        assert "value_W1" not in src, (
            "_load_from_cache special-cases value_W1 again instead of "
            "validating every weight"
        )


# ── QL2: dimensions are derived, not duplicated ──────────────────────────


class TestDerivedDimensions:
    def test_action_dim_is_len_of_action_space(self):
        assert actions_mod.ACTION_DIM == len(actions_mod.ACTION_SPACE)

    def test_single_action_space_object(self):
        """state_encoder must re-export the same list, not a copy."""
        assert state_encoder.ACTION_SPACE is actions_mod.ACTION_SPACE
        assert state_encoder.ACTION_DIM == actions_mod.ACTION_DIM

    def test_q_learner_derives_both_dimensions(self):
        assert qmod.ACTION_DIM == actions_mod.ACTION_DIM
        assert qmod.STATE_DIM == state_encoder.STATE_DIM

    def test_q_learner_does_not_assign_the_constants(self):
        """AST: `q_learner` may only import them.

        A literal assignment (`ACTION_DIM: int = 7`) is how the duplicate came
        to exist, and the comment justifying it ("circular import") was false:
        `state_encoder` imports nothing from `app.rl`.
        """
        tree = ast.parse(inspect.getsource(qmod))
        assigned = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for t in targets:
                    if isinstance(t, ast.Name):
                        assigned.add(t.id)
        assert "ACTION_DIM" not in assigned, "ACTION_DIM is re-hardcoded in q_learner"
        assert "STATE_DIM" not in assigned, "STATE_DIM is re-hardcoded in q_learner"

    def test_state_encoder_does_not_assign_action_space(self):
        tree = ast.parse(inspect.getsource(state_encoder))
        assigned = {
            t.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            for t in node.targets
            if isinstance(t, ast.Name)
        }
        assert "ACTION_SPACE" not in assigned, (
            "state_encoder defines a second ACTION_SPACE; there must be one"
        )

    def test_encoder_and_network_agree_on_sizes(self):
        """End-to-end: the network the policy builds matches the encoder."""
        learner = DoubleQLearner()
        assert learner.action_dim == len(state_encoder.ACTION_SPACE)
        assert learner.state_dim == state_encoder.STATE_DIM
        enc = np.zeros(state_encoder.STATE_DIM, dtype=np.float32)
        assert learner.q_values(enc).shape == (len(state_encoder.ACTION_SPACE),)


# ── QL3: cache writes are throttled, explicit saves are not ──────────────


class TestCacheWriteThrottle:
    def test_first_batch_still_writes(self, cache_paths, monkeypatch):
        """Durability floor: a process that trains once and exits keeps it."""
        calls: list[int] = []
        orig = DoubleQLearner.save_cache
        monkeypatch.setattr(
            DoubleQLearner, "save_cache",
            lambda self: (calls.append(1), orig(self))[1],
        )
        learner = DoubleQLearner()
        learner.train_batch(_batch())
        assert len(calls) == 1
        assert cache_paths[0].exists()

    def test_burst_of_batches_does_not_write_each_time(
        self, cache_paths, monkeypatch,
    ):
        calls: list[int] = []
        orig = DoubleQLearner.save_cache
        monkeypatch.setattr(
            DoubleQLearner, "save_cache",
            lambda self: (calls.append(1), orig(self))[1],
        )
        learner = DoubleQLearner()
        for _ in range(CACHE_SAVE_MIN_BATCHES + 4):
            learner.train_batch(_batch())
        # 1 unconditional + 1 at the N-batch boundary; not N+5.
        assert len(calls) == 2, f"expected throttle, got {len(calls)} writes"
        assert len(calls) < CACHE_SAVE_MIN_BATCHES

    def test_explicit_save_cache_is_never_throttled(self, cache_paths, monkeypatch):
        """The episode-end anchor must write on demand, always."""
        calls: list[int] = []
        orig = DoubleQLearner.save_cache
        monkeypatch.setattr(
            DoubleQLearner, "save_cache",
            lambda self: (calls.append(1), orig(self))[1],
        )
        learner = DoubleQLearner()
        learner._save_cache_throttled()          # consumes the "first" write
        calls.clear()
        learner.save_cache()
        assert len(calls) == 1, "explicit save_cache() must bypass the throttle"

    def test_nonfinite_bypasses_the_throttle(self, cache_paths):
        """A poisoned network must be reported now, not 25 batches later."""
        learner = DoubleQLearner()
        learner._save_cache_throttled()  # first write, resets the counters
        assert not learner._nonfinite_weight_keys()
        learner.online.value_net.W1[0, 0] = np.nan
        assert learner._nonfinite_weight_keys()
        learner._save_cache_throttled()
        assert learner._batches_since_cache_save == 0, (
            "the non-finite path must have attempted a save (and been refused)"
        )

    def test_throttle_constants_are_sane(self):
        assert CACHE_SAVE_MIN_BATCHES >= 2
        assert CACHE_SAVE_MIN_SECONDS > 0

    def test_train_batch_uses_the_throttle_not_save_cache(self):
        src = inspect.getsource(DoubleQLearner.train_batch)
        assert "_save_cache_throttled()" in src
        assert "self.save_cache()" not in src


# ── QL4: hidden activations are reused, math is unchanged ────────────────


class TestHiddenActivationReuse:
    def test_train_step_does_not_refoward_the_streams(self):
        src = inspect.getsource(DoubleQLearner.train_step)
        assert "forward_detailed" in src
        assert "value_net.forward(" not in src, (
            "train_step re-runs the value stream instead of reusing h1"
        )
        assert "advantage_net.forward(" not in src, (
            "train_step re-runs the advantage stream instead of reusing h1"
        )

    def test_forward_and_forward_detailed_agree(self):
        """The refactor must not change the numbers."""
        net = DuelingNetwork(STATE_DIM, 7, 64)
        x = np.random.randn(STATE_DIM).astype(np.float32)
        q1, v1, a1 = net.forward(x)
        q2, v2, a2, h1_v, h1_a = net.forward_detailed(x)
        assert np.array_equal(q1, q2)
        assert v1 == v2
        assert np.array_equal(a1, a2)
        assert h1_v.shape == (64,) and h1_a.shape == (64,)

    def test_gradient_uses_pre_update_weights(self):
        """TD error must be computed against the weights before the step.

        If a future refactor reuses activations captured *after* the SGD
        update, the reported |TD| silently changes meaning and PER priorities
        inherit the error. Recompute it independently here.
        """
        learner = DoubleQLearner()
        s = np.random.randn(STATE_DIM).astype(np.float32)
        sn = np.random.randn(STATE_DIM).astype(np.float32)

        q_before = learner.q_values(s)
        expected_target = learner._compute_target(0.5, sn, False)
        expected_td = abs(float(q_before[1]) - expected_target)

        reported = learner.train_step(s, 1, 0.5, sn, False)
        assert reported == pytest.approx(expected_td, rel=1e-6, abs=1e-6)

    def test_train_step_actually_updates_weights(self):
        """Guard against an over-eager refactor that drops the update."""
        learner = DoubleQLearner()
        before = learner.online.value_net.W1.copy()
        s = np.random.randn(STATE_DIM).astype(np.float32)
        learner.train_step(s, 1, 1.0, s.copy(), False)
        assert not np.array_equal(before, learner.online.value_net.W1)
        assert learner.train_steps == 1


# ── EX3: beta anneals per gradient step ──────────────────────────────────


class TestBetaCountsGradientSteps:
    async def _store_with(self, n: int) -> ExperienceStore:
        store = ExperienceStore(max_buffer=1000)
        for i in range(n):
            await store.add(
                scan_id="s", turn=i,
                state=np.zeros(STATE_DIM, dtype=np.float32),
                action=0, reward=1.0,
                next_state=np.zeros(STATE_DIM, dtype=np.float32), done=False,
            )
        return store

    async def test_one_call_advances_by_batch_size(self):
        store = await self._store_with(10)
        store.sample(5)
        assert store.beta_step == 5, (
            "beta_step must count gradient steps, not sample() calls"
        )

    async def test_two_calls_accumulate(self):
        store = await self._store_with(60)
        store.sample(50)
        store.sample(50)
        assert store.beta_step == 100

    async def test_beta_matches_the_formula(self):
        store = await self._store_with(10)
        store.sample(5)
        expect = PER_BETA_START + (PER_BETA_END - PER_BETA_START) * 5 / PER_BETA_ANNEAL_STEPS
        assert store.beta == pytest.approx(expect)

    async def test_empty_store_does_not_advance(self):
        store = ExperienceStore()
        assert store.sample(8) == ([], [], [])
        assert store.beta_step == 0

    def test_no_plus_one_counter_left(self):
        src = inspect.getsource(ExperienceStore.sample)
        assert "_beta_step += 1" not in src, (
            "per-call annealing is back: β will take batch_size× longer to "
            "reach 1.0 than PER_BETA_ANNEAL_STEPS implies"
        )

    def test_anneal_step_delta_is_the_sample_count(self):
        src = inspect.getsource(ExperienceStore.sample)
        assert "_beta_step += n" in src

    def test_beta_step_docstring_states_the_unit(self):
        doc = textwrap.dedent(ExperienceStore.beta_step.__doc__ or "").lower()
        assert "gradient step" in doc


# ── EX4: max-priority monotonicity is intentional and documented ─────────


class TestMaxPriorityIsMonotonic:
    async def test_new_transitions_inherit_the_running_max(self):
        store = ExperienceStore(max_buffer=100)
        state = np.zeros(STATE_DIM, dtype=np.float32)
        await store.add(scan_id="s", turn=0, state=state, action=0, reward=1.0,
                        next_state=state, done=False, priority=10.0)
        high = store.tree._max_priority
        await store.add(scan_id="s", turn=1, state=state, action=0, reward=0.0,
                        next_state=state, done=False)  # no priority supplied
        assert store.tree._max_priority == pytest.approx(high), (
            "an unprioritised transition must inherit the running maximum, not "
            "lower it"
        )

    def test_monitoring_note_is_present(self):
        """EX4 is documentation-only, so the documentation is the deliverable."""
        src = inspect.getsource(ExperienceStore.add)
        assert "never decreases" in src
        assert "percentile" in src, (
            "the note must name the mitigation (priority capping) so a future "
            "reader does not reach for the wrong fix"
        )
