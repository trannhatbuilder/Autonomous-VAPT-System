"""
Unit tests — RL "shadow mode": do not inject guidance from an untrained policy.

The bug: `react_agent` built `rl_guidance_msg` UNCONDITIONALLY. Half of all
turns therefore carried the advisory text of a policy that picks uniformly at
random. Verified on a real `PentestPolicy`:

    epsilon after weight load : 1.0
    exploratory               : 200/200 calls
    effective_epsilon         : {1.0}
    hints injectable under "not exploratory" alone : 0/200

So the proposed gate (`not rec.exploratory`) is necessary but NOT sufficient:
with `train_steps == 0` the network is still random-initialised, so an
"exploitation" argmax is arbitrary too — it merely doesn't *look* random.
The gate must be exploratory AND trained.
"""
from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from app.agents.react_agent import _should_inject_rl_guidance
from app.rl.harness import TurnRecommendation
from app.rl.policy import PentestPolicy

REPO_ROOT = Path(__file__).resolve().parents[2]
AGENT_FILE = REPO_ROOT / "app" / "agents" / "react_agent.py"


def _rec(*, exploratory: bool, epsilon: float = 0.5) -> TurnRecommendation:
    return TurnRecommendation(
        action_index=0,
        action_name="execute_command_recon",
        q_values=[0.0, 0.0],
        q_max=0.0,
        epsilon=epsilon,
        exploratory=exploratory,
        curiosity_bonus=0.0,
        guidance_text="## RL Guidance\nExplore...",
    )


class TestGate:
    def test_no_recommendation(self):
        assert _should_inject_rl_guidance(None, train_steps=50) == (
            False, "no_recommendation",
        )

    def test_exploratory_pick_is_suppressed_even_when_trained(self):
        assert _should_inject_rl_guidance(_rec(exploratory=True), train_steps=500) == (
            False, "exploratory",
        )

    def test_untrained_policy_is_suppressed_even_when_greedy(self):
        """The extra condition the proposal was missing."""
        assert _should_inject_rl_guidance(_rec(exploratory=False), train_steps=0) == (
            False, "untrained",
        )

    def test_trained_greedy_pick_is_injected(self):
        assert _should_inject_rl_guidance(_rec(exploratory=False), train_steps=1) == (
            True, "trained",
        )

    def test_reason_is_a_stable_token(self):
        """Reasons are logged/metricised — keep them machine-readable."""
        for rec, steps in ((None, 0), (_rec(exploratory=True), 9), (_rec(exploratory=False), 0)):
            _, reason = _should_inject_rl_guidance(rec, train_steps=steps)
            assert reason.islower()
            assert " " not in reason


class TestUntrainedPolicyIsRandom:
    """Evidence for why the gate exists — pins the underlying behaviour."""

    def test_fresh_policy_always_explores(self):
        policy = PentestPolicy()
        assert policy.epsilon == 1.0
        state = np.random.RandomState(0).randn(337).astype(np.float64)
        metas = [policy.select_action(state)[2] for _ in range(100)]
        assert all(m["exploratory"] for m in metas)
        assert {m["effective_epsilon"] for m in metas} == {1.0}

    def test_curiosity_bonus_cannot_lower_effective_epsilon(self):
        """`min(1.0, eps + bonus)` means a bonus never buys exploitation."""
        policy = PentestPolicy()
        state = np.random.RandomState(1).randn(337).astype(np.float64)
        meta = policy.select_action(state)[2]
        assert meta["effective_epsilon"] >= meta["epsilon"]

    def test_every_such_turn_would_have_injected_before(self):
        policy = PentestPolicy()
        state = np.random.RandomState(2).randn(337).astype(np.float64)
        metas = [policy.select_action(state)[2] for _ in range(30)]
        recs = [_rec(exploratory=m["exploratory"]) for m in metas]
        injected = [r for r in recs if _should_inject_rl_guidance(r, train_steps=0)[0]]
        assert injected == []


class TestHarnessExposesTrainSteps:
    def test_train_steps_mirrors_the_learner_counter(self):
        """The .npz cache carries `step`, so a fresh instance may already be >0."""
        from app.rl.q_learner import DoubleQLearner

        learner = DoubleQLearner()
        assert learner.train_steps == learner._step
        assert isinstance(learner.train_steps, int)
        assert learner.train_steps >= 0

    def test_untrained_learner_reports_zero(self, monkeypatch, tmp_path):
        """Force the no-cache path: a genuinely untrained net reports 0."""
        from app.rl import q_learner

        monkeypatch.setattr(q_learner, "WEIGHTS_FILE", tmp_path / "absent.npz")
        monkeypatch.setattr(q_learner, "META_FILE", tmp_path / "absent.json")
        assert q_learner.DoubleQLearner().train_steps == 0


class TestCallSiteUsesTheGate:
    def test_guidance_is_not_built_unconditionally(self):
        """Regression: the literal `rl_guidance_msg = {...}` must be guarded."""
        src = AGENT_FILE.read_text(encoding="utf-8")
        tree = ast.parse(src)
        guarded = False
        for node in ast.walk(tree):
            if isinstance(node, ast.If):
                names = {
                    n.id for n in ast.walk(node.test) if isinstance(n, ast.Name)
                }
                if "inject_guidance" in names:
                    if any(
                        isinstance(n, ast.Name) and n.id == "rl_guidance_msg"
                        for n in ast.walk(node)
                        if isinstance(n, ast.Assign)
                        for n in n.targets
                        if isinstance(n, ast.Name)
                    ):
                        guarded = True
        assert guarded, "rl_guidance_msg must only be assigned inside the gate"

    def test_gate_is_called_with_train_steps(self):
        src = AGENT_FILE.read_text(encoding="utf-8")
        assert "_should_inject_rl_guidance(" in src
        assert "harness.q_train_steps" in src

    def test_follow_reward_is_none_when_suppressed(self):
        """No advice in the prompt → following is coincidence, not signal."""
        src = AGENT_FILE.read_text(encoding="utf-8")
        assert "if rl_guidance_injected:" in src
        assert "followed: bool | None" in src

    def test_suppressed_hint_frees_the_rotation_slot(self):
        """rl_msg=None must let TDA/EGATS take the turn, not burn it."""
        from app.agents.react_agent import _select_advisory

        tda = {"role": "system", "content": "TDA"}
        egats = {"role": "system", "content": "EGATS"}
        name, msg = _select_advisory(1, rl_msg=None, tda_msg=tda, egats_msg=egats)
        assert (name, msg) == ("tda", tda)


class TestCacheLoadIsReal:
    """`_load_checkpoint_safe` used to call a nonexistent q.load_cache()."""

    def test_public_load_cache_exists_and_reports_false_without_file(
        self, monkeypatch, tmp_path,
    ):
        from app.rl import q_learner

        monkeypatch.setattr(q_learner, "WEIGHTS_FILE", tmp_path / "absent.npz")
        monkeypatch.setattr(q_learner, "META_FILE", tmp_path / "absent.json")
        assert q_learner.DoubleQLearner().load_cache() is False

    def test_public_load_cache_returns_true_when_file_present(self, tmp_path):
        """R27: build the cache here instead of reading the repo's data/rl.

        The previous version skipped when `data/rl/q_weights.npz` was absent,
        i.e. it asserted against whatever a developer's last scan happened to
        leave behind — and it silently stopped running once `tests/conftest.py`
        began isolating the RL artifact dir. A test must not depend on machine
        state.
        """
        from app.rl import q_learner

        learner = q_learner.DoubleQLearner()
        learner.save_cache()
        assert q_learner.WEIGHTS_FILE.exists(), "cache file was not written"

        fresh = q_learner.DoubleQLearner()
        assert fresh.load_cache() is True
