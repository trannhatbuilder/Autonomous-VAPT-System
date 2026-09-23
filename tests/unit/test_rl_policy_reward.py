"""
Tests for app.rl.policy + app.rl.reward.

Covers:
  - PentestPolicy: ε-greedy, Boltzmann, Curiosity modes
  - ε decay schedule: 1.0 → 0.05 over 200 scans
  - Adaptive ε decay (high/low reward)
  - CuriosityModule: bonus computation, state hashing
  - resolve_action: action → command translation
  - D26 reward shaping: 10 scenarios
  - Multi-signal weights (rule/human/replay)
  - d26_event() constructor + invalid type rejection
  - Reward clipping
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

from app.rl.policy import (
    PentestPolicy, CuriosityModule,
    DEFAULT_EPSILON_START, DEFAULT_EPSILON_END, DEFAULT_EPSILON_DECAY,
    DEFAULT_EPSILON_DECAY_SCANS,
    DEFAULT_TEMPERATURE_START, DEFAULT_TEMPERATURE_END,
)
from app.rl.reward import (
    compute_reward, d26_event,
    R_EXPLOIT_TIER1, R_EXPLOIT_TIER2, R_VERIFIED_FINDING, R_RECON_SUCCESS,
    R_TIMEOUT, R_FP_REJECTED, R_SCOPE_VIOLATION, R_USER_ABORTED,
    R_DESTRUCTIVE_NO_HITL, R_SCAN_COMPLETE,
    R_NOVEL_FINDING, R_MIN_CLIP, R_MAX_CLIP,
    WEIGHT_RULE_VERIFIER, WEIGHT_HUMAN_LABEL, WEIGHT_REPLAY_REPRODUCED,
)


# ── Section 1: ε-greedy ──────────────────────────────────────────────────

class TestEpsilonGreedy:
    def test_initial_epsilon_1_0(self):
        """Per master plan §12 W16: ε_start = 1.0."""
        policy = PentestPolicy(exploration_mode="epsilon_greedy")
        assert policy.epsilon == 1.0

    def test_epsilon_end_0_05(self):
        policy = PentestPolicy(exploration_mode="epsilon_greedy")
        assert policy.epsilon_end == 0.05

    def test_epsilon_decay_over_200_scans(self):
        """ε should decay from 1.0 → ~0.05 over 200 scans."""
        np.random.seed(42)
        policy = PentestPolicy()
        for _ in range(200):
            policy.end_episode(episode_reward=0.0)
        assert abs(policy.epsilon - 0.05) < 1e-3

    def test_epsilon_at_100_scans(self):
        """At scan 100, ε should be roughly halfway (log scale)."""
        np.random.seed(42)
        policy = PentestPolicy()
        for _ in range(100):
            policy.end_episode(episode_reward=0.0)
        # ε(100) ≈ 1.0 * 0.98566^100 ≈ 0.224
        assert 0.15 < policy.epsilon < 0.30

    def test_select_action_returns_3_tuple(self):
        policy = PentestPolicy()
        state = np.random.randn(337).astype(np.float32)
        action_idx, action_name, meta = policy.select_action(state)
        assert isinstance(action_idx, int)
        assert 0 <= action_idx < 7
        assert isinstance(action_name, str)
        assert isinstance(meta, dict)
        assert "q_values" in meta
        assert "epsilon" in meta

    def test_force_greedy_returns_argmax(self):
        """ε=0 → always returns argmax(Q). Disable curiosity so bonus doesn't push ε > 0."""
        policy = PentestPolicy(use_curiosity=False)
        policy.epsilon = 0.0
        state = np.random.randn(337).astype(np.float32)
        action_idx, _, meta = policy.select_action(state)
        assert meta["exploratory"] is False
        assert action_idx == meta["best_action"]


# ── Section 2: Adaptive ε decay ──────────────────────────────────────────

class TestAdaptiveEpsilonDecay:
    def test_high_reward_faster_decay(self):
        """avg_reward > 5 → faster decay (decay^1.2)."""
        np.random.seed(42)
        policy_fast = PentestPolicy()
        for _ in range(50):
            policy_fast.end_episode(episode_reward=10.0)

        np.random.seed(42)
        policy_normal = PentestPolicy()
        for _ in range(50):
            policy_normal.end_episode(episode_reward=0.0)

        assert policy_fast.epsilon < policy_normal.epsilon

    def test_low_reward_slower_decay(self):
        """avg_reward < -5 → slower decay (decay^0.8)."""
        np.random.seed(42)
        policy_slow = PentestPolicy()
        for _ in range(50):
            policy_slow.end_episode(episode_reward=-10.0)

        np.random.seed(42)
        policy_normal = PentestPolicy()
        for _ in range(50):
            policy_normal.end_episode(episode_reward=0.0)

        assert policy_slow.epsilon > policy_normal.epsilon


# ── Section 3: Boltzmann ─────────────────────────────────────────────────

class TestBoltzmann:
    def test_boltzmann_mode(self):
        policy = PentestPolicy(exploration_mode="boltzmann")
        assert policy.exploration_mode == "boltzmann"
        state = np.random.randn(337).astype(np.float32)
        action_idx, _, meta = policy.select_action(state)
        assert meta["mode"] == "boltzmann"
        assert "probs" in meta
        assert abs(sum(meta["probs"]) - 1.0) < 1e-5  # softmax normalizes

    def test_temperature_decay(self):
        """T should decay toward 0.1 over episodes."""
        policy = PentestPolicy(exploration_mode="boltzmann")
        for _ in range(100):
            policy.end_episode(episode_reward=0.0)
        assert policy.temperature < policy.temperature_start
        assert policy.temperature >= policy.temperature_end


# ── Section 4: Curiosity ─────────────────────────────────────────────────

class TestCuriosity:
    def test_curiosity_bonus_for_novel_state(self):
        """Novel state → high bonus (count=0 → bonus=0.5)."""
        c = CuriosityModule(bonus_scale=0.5)
        state = np.random.randn(337).astype(np.float32)
        bonus = c.compute_bonus(state)
        # First visit: count=0, bonus = 0.5 / sqrt(0+1) = 0.5
        assert 0.4 < bonus < 0.6

    def test_curiosity_bonus_decreases_with_visits(self):
        """Same state visited multiple times → bonus decreases."""
        c = CuriosityModule(bonus_scale=0.5)
        state = np.random.randn(337).astype(np.float32)
        bonuses = [c.compute_bonus(state) for _ in range(5)]
        # Bonus should generally decrease
        assert bonuses[-1] < bonuses[0]

    def test_curiosity_mode_2x_bonus_scale(self):
        """'curiosity' mode doubles the bonus_scale (0.5 → 1.0)."""
        policy = PentestPolicy(exploration_mode="curiosity")
        assert policy.curiosity._bonus_scale == 1.0  # 0.5 * 2

    def test_curiosity_stats(self):
        c = CuriosityModule()
        state = np.random.randn(337).astype(np.float32)
        c.compute_bonus(state)
        c.compute_bonus(state)
        stats = c.get_stats()
        assert "unique_states_visited" in stats
        assert "total_visits" in stats
        assert stats["total_visits"] >= 2


# ── Section 5: resolve_action ────────────────────────────────────────────

class TestResolveAction:
    def test_recon_action(self):
        policy = PentestPolicy()
        policy.epsilon = 0.0  # deterministic
        session = {"target_url": "http://example.com", "_turn": 5}
        result = policy.resolve_action("execute_command_recon", session)
        assert result["type"] == "execute_command"
        assert "http://example.com" in result["command"]

    def test_exploit_action(self):
        policy = PentestPolicy()
        session = {
            "target_url": "http://example.com",
            "recon_signals": {"technologies": ["wordpress"]},
        }
        result = policy.resolve_action("execute_command_exploit", session)
        assert result["type"] == "execute_command"
        assert "nuclei" in result["command"]

    def test_pentest_complete(self):
        policy = PentestPolicy()
        session = {"target_url": "http://example.com"}
        result = policy.resolve_action("pentest_complete", session)
        assert result["type"] == "pentest_complete"

    def test_consult_kg(self):
        policy = PentestPolicy()
        session = {
            "target_url": "http://example.com",
            "recon_signals": {"technologies": ["nginx", "php"]},
        }
        result = policy.resolve_action("consult_kg", session)
        assert result["type"] == "consult_kg"
        assert "nginx" in result["tech_stack"]


# ── Section 6: D26 Reward Scenarios ──────────────────────────────────────

class TestD26Reward:
    """Verify all 10 D26 reward scenarios from master plan §12 W16."""

    def _base_session(self):
        return {"findings": []}

    def test_scenario_1_exploit_tier1(self):
        """+5 confirmed exploit Tier-1 (shell obtained via C2Session)."""
        session = self._base_session()
        r, b = compute_reward(
            session=session, previous_findings_count=0, previous_verified_count=0,
            command_output="meterpreter session 1 opened",
            tokens_used_delta=1000, turn=5,
            turn_events=[d26_event("exploit_tier1", c2_session_id="sess_001")],
        )
        assert b["d26_signals"]["exploit_tier1"] == 5.0
        # Total reward should be >= 5 (plus efficiency bonus)
        assert r >= 5.0

    def test_scenario_2_exploit_tier2(self):
        """+3 confirmed exploit Tier-2 (partial PoC)."""
        session = self._base_session()
        r, b = compute_reward(
            session=session, previous_findings_count=0, previous_verified_count=0,
            command_output="sqlmap: dumped 5 tables",
            tokens_used_delta=800, turn=6,
            turn_events=[d26_event("exploit_tier2", tool="sqlmap")],
        )
        assert b["d26_signals"]["exploit_tier2"] == 3.0

    def test_scenario_3_recon_success(self):
        """+0.5 successful recon (asset discovered)."""
        session = self._base_session()
        r, b = compute_reward(
            session=session, previous_findings_count=0, previous_verified_count=0,
            command_output="whatweb: nginx/1.18 php/7.4",
            tokens_used_delta=200, turn=2,
            turn_events=[d26_event("recon_success", asset="target.com")],
        )
        assert b["d26_signals"]["recon_success"] == 0.5

    def test_scenario_4_timeout(self):
        """-1 timeout / no useful output."""
        session = self._base_session()
        r, b = compute_reward(
            session=session, previous_findings_count=0, previous_verified_count=0,
            command_output="", tokens_used_delta=100, turn=8,
            turn_events=[d26_event("timeout", tool="nuclei")],
        )
        assert b["d26_signals"]["timeout"] == -1.0

    def test_scenario_5_fp_rejected(self):
        """-2 verifier-rejected FP."""
        session = self._base_session()
        r, b = compute_reward(
            session=session, previous_findings_count=1, previous_verified_count=0,
            command_output="XSS reflected",
            tokens_used_delta=500, turn=4,
            turn_events=[d26_event("fp_rejected", finding_id="f1")],
        )
        assert b["d26_signals"]["fp_rejected"] == -2.0

    def test_scenario_6_scope_violation(self):
        """-3 scope violation (blocked)."""
        session = self._base_session()
        r, b = compute_reward(
            session=session, previous_findings_count=0, previous_verified_count=0,
            command_output="", tokens_used_delta=50, turn=3,
            turn_events=[d26_event("scope_violation", target="8.8.8.8")],
        )
        assert b["d26_signals"]["scope_violation"] == -3.0

    def test_scenario_7_user_aborted(self):
        """-5 user_aborted (HITL denied)."""
        session = self._base_session()
        r, b = compute_reward(
            session=session, previous_findings_count=0, previous_verified_count=0,
            command_output="", tokens_used_delta=0, turn=10,
            turn_events=[d26_event("user_aborted", approval_id="app_001")],
        )
        assert b["d26_signals"]["user_aborted"] == -5.0

    def test_scenario_8_destructive_no_hitl(self):
        """-10 destructive op without HITL (failsafe)."""
        session = self._base_session()
        r, b = compute_reward(
            session=session, previous_findings_count=0, previous_verified_count=0,
            command_output="", tokens_used_delta=0, turn=12,
            turn_events=[d26_event("destructive_no_hitl", tool="metasploit")],
        )
        assert b["d26_signals"]["destructive_no_hitl"] == -10.0

    def test_scenario_9_scan_complete(self):
        """+10 scan completed successfully."""
        session = {"findings": [
            {"name": "SQLi", "location": "/login", "severity": "high",
             "verified": True, "false_positive": False},
        ]}
        r, b = compute_reward(
            session=session, previous_findings_count=1, previous_verified_count=1,
            command_output="scan complete",
            tokens_used_delta=200, turn=30, is_done=True,
        )
        assert b["completion_bonus"] == 10.0

    def test_scenario_10_verified_finding(self):
        """+1 confirmed finding (verifier accepts)."""
        session = {"findings": [
            {"name": "XSS", "location": "/search", "severity": "medium",
             "verified": True, "false_positive": False},
        ]}
        r, b = compute_reward(
            session=session, previous_findings_count=0, previous_verified_count=0,
            command_output="XSS reflected", tokens_used_delta=500, turn=5,
        )
        # First verified finding → R_NOVEL_FINDING (20.0) * WEIGHT_RULE (1.0) = 20.0
        # (Finding is "novel" because it's the first in the session)
        assert b["finding_bonus"] >= 1.0


# ── Section 7: Multi-signal weights ──────────────────────────────────────

class TestMultiSignalWeights:
    def test_rule_signal_weight(self):
        """Rule-verified finding → WEIGHT_RULE_VERIFIER (1.0)."""
        session = {"findings": [
            {"name": "XSS", "location": "/", "severity": "medium",
             "verified": True, "false_positive": False},  # no replay/human
        ]}
        r, b = compute_reward(
            session=session, previous_findings_count=0, previous_verified_count=0,
            command_output="x", tokens_used_delta=0, turn=1,
        )
        # First finding is novel → R_NOVEL_FINDING (20) * WEIGHT_RULE (1.0) = 20
        assert b["finding_bonus"] == 20.0

    def test_replay_signal_weight(self):
        """replay_reproduced=True → WEIGHT_REPLAY_REPRODUCED (5.0)."""
        session = {"findings": [
            {"name": "SQLi", "location": "/", "severity": "high",
             "verified": True, "false_positive": False,
             "replay_reproduced": True},
        ]}
        r, b = compute_reward(
            session=session, previous_findings_count=0, previous_verified_count=0,
            command_output="x", tokens_used_delta=0, turn=1,
        )
        # Novel * WEIGHT_REPLAY = 20 * 5 = 100
        assert b["finding_bonus"] == 100.0

    def test_human_signal_weight(self):
        """human_verified=True → WEIGHT_HUMAN_LABEL (3.0)."""
        session = {"findings": [
            {"name": "XSS", "location": "/", "severity": "medium",
             "verified": True, "false_positive": False,
             "human_verified": True},
        ]}
        r, b = compute_reward(
            session=session, previous_findings_count=0, previous_verified_count=0,
            command_output="x", tokens_used_delta=0, turn=1,
        )
        # Novel * WEIGHT_HUMAN = 20 * 3 = 60
        assert b["finding_bonus"] == 60.0


# ── Section 8: d26_event + clipping ──────────────────────────────────────

class TestD26EventAndClipping:
    def test_d26_event_valid(self):
        evt = d26_event("exploit_tier1", c2_session_id="s1")
        assert evt["type"] == "exploit_tier1"
        assert evt["c2_session_id"] == "s1"

    def test_d26_event_invalid_type_raises(self):
        with pytest.raises(ValueError):
            d26_event("invalid_type")

    def test_reward_clip_upper(self):
        """Reward clips at R_MAX_CLIP (150)."""
        # Many tier1 events → reward > 150 → clipped
        session = {"findings": [
            {"name": "X", "location": "/", "severity": "high",
             "verified": True, "false_positive": False,
             "replay_reproduced": True},
        ]}
        events = [d26_event("exploit_tier1") for _ in range(20)]
        r, b = compute_reward(
            session=session, previous_findings_count=0, previous_verified_count=0,
            command_output="x", tokens_used_delta=0, turn=1,
            turn_events=events,
        )
        assert r <= R_MAX_CLIP

    def test_reward_clip_lower(self):
        """Reward clips at R_MIN_CLIP (-100)."""
        session = {"findings": []}
        events = [d26_event("destructive_no_hitl") for _ in range(20)]
        r, b = compute_reward(
            session=session, previous_findings_count=0, previous_verified_count=0,
            command_output="", tokens_used_delta=0, turn=1,
            turn_events=events,
        )
        assert r >= R_MIN_CLIP

    def test_reward_breakdown_structure(self):
        """compute_reward returns (reward, breakdown_dict)."""
        session = {"findings": []}
        r, b = compute_reward(
            session=session, previous_findings_count=0, previous_verified_count=0,
            command_output="x", tokens_used_delta=0, turn=1,
        )
        assert isinstance(r, float)
        assert isinstance(b, dict)
        assert "total" in b
        assert "pre_clip" in b
        assert "d26_signals" in b
        assert "finding_bonus" in b