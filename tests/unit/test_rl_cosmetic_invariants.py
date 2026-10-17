"""PL8 — cosmetic/invariant guards.

Five items, four of which are documentation-or-dead-code, and one of which
turned out to be a live cross-module coupling:

  1. `import math` unused — the report placed it "inline in the middle of
     policy.py, which computes with `**`". `policy.py` has no `import math` at
     all; the unused one is in `app/evidence/cvss.py` (top-of-module, after the
     docstring — not mid-file). Verified with `grep -c math` → 1 hit, the
     import itself.

  2. The decay comment said `≈ 0.98566`. Actual: `0.05 ** (1/200)` =
     0.9851329607687279. 0.98566 ** 200 = 0.0556, so a reader trusting the
     comment would think ε never reaches its documented ε_end of 0.05. The
     CONSTANT was always computed by formula, so behaviour was correct — only
     the annotation lied. Three places carried the number (module comment, the
     constant's trailing comment, `end_episode` docstring) plus one test
     comment.

  3. `store._beta if hasattr(store, "_beta")` — `PentestPolicy` reaching into
     `ExperienceStore`'s privates purely to log. Now a public `beta` property.

  4. `compute_bonus()` mutates the visit table; two selection methods per state
     per turn double-count. Documented as a caller contract (behaviour
     unchanged — see below).

  5. `train_from_memory()` blocks the event loop. Measured 31.7 ms at the
     default batch of 64, not the "vài ms" the report assumed.
"""

from __future__ import annotations

import ast
import math
from pathlib import Path

import numpy as np
import pytest

from app.rl.experience_store import (
    PER_BETA_END,
    PER_BETA_START,
    ExperienceStore,
)
from app.rl.policy import (
    DEFAULT_EPSILON_DECAY,
    DEFAULT_EPSILON_DECAY_SCANS,
    DEFAULT_EPSILON_END,
    DEFAULT_EPSILON_START,
    PentestPolicy,
)
from app.rl.state_encoder import STATE_DIM

REPO = Path(__file__).resolve().parents[2]
POLICY_SRC = (REPO / "app" / "rl" / "policy.py").read_text(encoding="utf-8")


# ─── 1. no unused `import math` ───────────────────────────────────────


class TestNoUnusedMathImport:
    @staticmethod
    def _unused_math_imports() -> list[str]:
        """Files that `import math` but never reference `math.` or `math`."""
        offenders = []
        for path in (REPO / "app").rglob("*.py"):
            src = path.read_text(encoding="utf-8")
            try:
                tree = ast.parse(src)
            except SyntaxError:  # pragma: no cover
                continue
            imports_math = any(
                isinstance(n, ast.Import) and any(a.name == "math" for a in n.names)
                for n in ast.walk(tree)
            )
            if not imports_math:
                continue
            uses = [
                n for n in ast.walk(tree)
                if isinstance(n, ast.Name) and n.id == "math"
            ]
            if not uses:
                offenders.append(str(path.relative_to(REPO)))
        return offenders

    def test_no_module_imports_math_without_using_it(self):
        assert not self._unused_math_imports(), (
            "unused `import math`: "
            f"{self._unused_math_imports()} — the calculation uses `**`"
        )

    def test_cvss_no_longer_imports_math(self):
        src = (REPO / "app" / "evidence" / "cvss.py").read_text(encoding="utf-8")
        assert "import math" not in src

    def test_policy_never_imported_math(self):
        """The report's premise — pin it so nobody 'restores' the import."""
        tree = ast.parse(POLICY_SRC)
        assert not any(
            isinstance(n, ast.Import) and any(a.name == "math" for a in n.names)
            for n in ast.walk(tree)
        )


# ─── 2. the decay annotation matches the constant ─────────────────────


class TestEpsilonDecayAnnotation:
    def test_constant_reaches_epsilon_end_at_200_scans(self):
        assert DEFAULT_EPSILON_DECAY ** DEFAULT_EPSILON_DECAY_SCANS == pytest.approx(
            DEFAULT_EPSILON_END, rel=1e-12
        )

    def test_constant_is_the_reported_value(self):
        """0.05^(1/200) = 0.98513296… — the comment must say 0.98513."""
        assert DEFAULT_EPSILON_DECAY == pytest.approx(0.98513, abs=5e-6)
        assert DEFAULT_EPSILON_DECAY == pytest.approx(
            (DEFAULT_EPSILON_END / DEFAULT_EPSILON_START) ** (1.0 / DEFAULT_EPSILON_DECAY_SCANS)
        )

    def test_the_documented_wrong_value_is_actually_wrong(self):
        """Documents *why* 0.98566 was wrong, so it cannot come back."""
        assert 1.0 * 0.98566 ** 200 == pytest.approx(0.0556, abs=1e-3)
        assert abs(0.98566 ** 200 - DEFAULT_EPSILON_END) > 1e-3

    def test_no_source_comment_still_says_98566(self):
        """The wrong number may appear ONLY where it is labelled wrong.

        `policy.py` deliberately quotes it in the "this used to read …" note so
        the mistake is documented rather than silently erased. Any other
        occurrence — a restored constant or a fresh copy-pasted comment — is a
        regression.
        """
        allowed_markers = ("used to read", "wrong", "actually wrong")
        self_path = Path(__file__).resolve()
        offenders = []
        for path in list((REPO / "app").rglob("*.py")) + list(
            (REPO / "tests").rglob("*.py")
        ):
            if path.resolve() == self_path:
                continue
            for lineno, line in enumerate(
                path.read_text(encoding="utf-8").splitlines(), start=1
            ):
                if "98566" not in line:
                    continue
                if any(m in line.lower() for m in allowed_markers):
                    continue
                offenders.append(f"{path.relative_to(REPO)}:{lineno}")
        assert not offenders, (
            "0.98566 is the wrong decay factor (0.05^(1/200) = 0.98513) and is "
            f"asserted as truth at: {offenders}"
        )

    def test_epsilon_at_100_scans_matches_the_documented_factor(self):
        """Behavioural lock: the comment on the test is now the real number."""
        policy = PentestPolicy()
        for _ in range(100):
            policy.end_episode(episode_reward=0.0)
        assert policy.epsilon == pytest.approx(0.98513 ** 100, abs=1e-3)
        assert policy.epsilon == pytest.approx(0.224, abs=0.01)


# ─── 3. beta is public ────────────────────────────────────────────────


class TestBetaIsPublic:
    def test_store_exposes_beta_property(self):
        store = ExperienceStore()
        assert store.beta == pytest.approx(PER_BETA_START)
        assert isinstance(store.beta, float)

    def test_beta_property_tracks_the_private_field(self):
        store = ExperienceStore()
        store._beta = 0.7
        assert store.beta == pytest.approx(0.7)

    def test_beta_step_property(self):
        store = ExperienceStore()
        assert store.beta_step == 0
        store._beta_step = 12
        assert store.beta_step == 12

    def test_annotation_is_load_bearing(self):
        """The properties must be typed, not `Any`-passthroughs."""
        ann = ExperienceStore.beta.fget.__annotations__.get("return")
        assert ann in ("float", float)

    def test_policy_does_not_touch_store_privates(self):
        assert "store._beta" not in POLICY_SRC
        assert 'hasattr(store, "_beta")' not in POLICY_SRC

    @staticmethod
    def _stub_store(beta=None):
        """A non-empty fake batch, optionally exposing a `beta` property."""
        rng = np.random.default_rng(0)
        batch = [
            {
                "state": rng.normal(size=STATE_DIM).astype(np.float32),
                "action": i % 7,
                "reward": 0.1,
                "next_state": rng.normal(size=STATE_DIM).astype(np.float32),
                "done": False,
            }
            for i in range(8)
        ]
        weights = [1.0] * 8
        indices = list(range(8))

        class _StubStore:
            def sample(self, n):
                return batch[:n], weights[:n], indices[:n]

            def update_priorities(self, idx, td):
                return None

        store = _StubStore()
        if beta is not None:
            # `_beta` is a property here, not an attribute — assigning would
            # raise, which is exactly what the guard below must prove.
            type(store).beta = property(lambda self: beta)
            assert isinstance(type(store).beta, property)
        return store

    def test_log_reports_the_stores_beta(self, caplog):
        import logging

        policy = PentestPolicy()
        with caplog.at_level(logging.INFO, logger="app.rl.policy"):
            policy.train_from_memory(self._stub_store(beta=0.65), batch_size=4)
        msgs = [r.getMessage() for r in caplog.records if "train_step" in r.getMessage()]
        assert msgs, "train_from_memory logged nothing for a non-empty batch"
        assert "β=0.650" in msgs[0], msgs[0]

    def test_log_falls_back_to_one_for_a_store_without_beta(self, caplog):
        """No PER in the double → report 1.0 (no correction), and no crash."""
        import logging

        policy = PentestPolicy()
        with caplog.at_level(logging.INFO, logger="app.rl.policy"):
            policy.train_from_memory(self._stub_store(), batch_size=4)
        msgs = [r.getMessage() for r in caplog.records if "train_step" in r.getMessage()]
        assert msgs
        assert "β=1.000" in msgs[0], msgs[0]

    def test_empty_batch_returns_zero_without_logging(self):
        class _Empty:
            def sample(self, n):
                return [], [], []

        policy = PentestPolicy()
        assert policy.train_from_memory(_Empty(), batch_size=4) == (0.0, [])


# ─── 4. the visit-count side effect is documented, not hidden ─────────


class TestSelectionIsSingleUsePerState:
    def test_compute_bonus_is_not_pure(self):
        """Two calls with the SAME state return different values."""
        policy = PentestPolicy()
        state = np.zeros(STATE_DIM, dtype=np.float32)
        first = policy.curiosity.compute_bonus(state)
        second = policy.curiosity.compute_bonus(state)
        assert second < first, (
            "compute_bonus looked pure — if the visit increment was removed, "
            "the caller contract documented on select_action() is obsolete"
        )

    def test_two_selections_double_count_the_visit(self):
        """This is the defect the docstring warns about. Pin the numbers."""
        one = PentestPolicy()
        state = np.zeros(STATE_DIM, dtype=np.float32)
        one.select_action(state)
        single = one.curiosity._visit_counts.copy()

        two = PentestPolicy()
        assert two.curiosity is not None
        two.select_action(state)
        two.select_action_boltzmann(state)  # caller error: second selection
        doubled = two.curiosity._visit_counts

        assert len(single) == 1 and len(doubled) == 1
        key = next(iter(single))
        assert doubled[key] == single[key] + 1

    def test_double_selection_advances_the_decay_counter_twice(self):
        policy = PentestPolicy()
        state = np.zeros(STATE_DIM, dtype=np.float32)
        before = policy.curiosity._decay_counter
        policy.select_action(state)
        policy.select_action_boltzmann(state)
        assert policy.curiosity._decay_counter == before + 2

    def test_documented_on_both_entry_points(self):
        tree = ast.parse(POLICY_SRC)
        for name in ("select_action", "select_action_boltzmann", "compute_bonus"):
            fn = next(
                n for n in ast.walk(tree)
                if isinstance(n, ast.FunctionDef) and n.name == name
            )
            doc = (ast.get_docstring(fn) or "").lower()
            assert "side effect" in doc or "mutat" in doc, (
                f"{name}() has a mutating side effect on the curiosity table "
                "but its docstring no longer says so"
            )

    def test_single_selection_per_turn_is_the_documented_contract(self):
        tree = ast.parse(POLICY_SRC)
        fn = next(
            n for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == "select_action"
        )
        doc = (ast.get_docstring(fn) or "").lower()
        assert "one" in doc and "per state" in doc

    def test_boltzmann_docstring_names_the_real_stats_key(self):
        """`get_stats()` reports `steps`, not `turn_count`.

        The docstring said `turn_count` when it was first written. A docstring
        that names a key which does not exist is the same defect class as the
        `0.98566` comment, so pin the key name against the real dict.
        """
        assert "steps" in PentestPolicy().get_stats()
        assert "turn_count" not in PentestPolicy().get_stats()

        tree = ast.parse(POLICY_SRC)
        fn = next(
            n for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == "select_action_boltzmann"
        )
        doc = ast.get_docstring(fn) or ""
        assert 'get_stats()["steps"]' in doc
        assert "turn_count" not in doc

    def test_only_select_action_increments_step(self):
        """Boltzmann is a read-only alternative entry point w.r.t. `_step`."""
        policy = PentestPolicy()
        state = np.zeros(STATE_DIM, dtype=np.float32)
        before = policy.get_stats()["steps"]
        policy.select_action_boltzmann(state)
        assert policy.get_stats()["steps"] == before
        policy.select_action(state)
        assert policy.get_stats()["steps"] == before + 1


# ─── 5. the sync-blocking contract ────────────────────────────────────


class TestTrainFromMemoryIsBlocking:
    def test_docstring_records_the_measurement(self):
        tree = ast.parse(POLICY_SRC)
        fn = next(
            n for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == "train_from_memory"
        )
        doc = ast.get_docstring(fn) or ""
        assert "BLOCKING" in doc
        assert "asyncio.to_thread" in doc, (
            "the escape hatch for a larger batch is not documented — the next "
            "person will re-measure instead of reading the number"
        )
        assert "31" in doc, "the measured ms for batch=64 is gone"

    def test_it_is_a_plain_def_not_async(self):
        tree = ast.parse(POLICY_SRC)
        assert not any(
            isinstance(n, ast.AsyncFunctionDef) and n.name == "train_from_memory"
            for n in ast.walk(tree)
        ), "train_from_memory became async — its call sites and docstring disagree"

    def test_harness_holds_the_lock_around_the_blocking_call(self):
        """If this lock is dropped, the blocking section stops excluding
        select_action() and the Q-network is read mid-update."""
        src = (REPO / "app" / "rl" / "harness.py").read_text(encoding="utf-8")
        idx = src.index("train_from_memory(")
        window = src[max(0, idx - 400): idx]
        assert "async with self._q_lock" in window, (
            "train_from_memory() is no longer called under _q_lock"
        )

    def test_cost_is_linear_in_batch_size(self):
        """Locks the shape of the measurement quoted in the docstring."""
        from app.rl.q_learner import DoubleQLearner

        q = DoubleQLearner()
        rng = np.random.default_rng(0)

        def batch(n):
            return [
                {
                    "state": rng.normal(size=STATE_DIM).astype(np.float32),
                    "action": int(rng.integers(0, 7)),
                    "reward": 0.1,
                    "next_state": rng.normal(size=STATE_DIM).astype(np.float32),
                    "done": False,
                }
                for _ in range(n)
            ]

        import time

        small, large = batch(64), batch(256)
        q.train_batch(small)  # warm up
        t0 = time.perf_counter()
        q.train_batch(small)
        t_small = time.perf_counter() - t0
        t0 = time.perf_counter()
        q.train_batch(large)
        t_large = time.perf_counter() - t0

        # 4x the samples should be ~4x the time (very loose: 2x..8x) — this is
        # a shape check, not a benchmark, so it stays green on slow CI.
        ratio = t_large / max(t_small, 1e-9)
        assert 2.0 < ratio < 8.0, f"train_batch is no longer linear: {ratio:.2f}x"
        # And the absolute figure the docstring quotes must be the right order
        # of magnitude.
        assert t_small < 0.5, f"batch=64 took {t_small * 1000:.0f} ms"
