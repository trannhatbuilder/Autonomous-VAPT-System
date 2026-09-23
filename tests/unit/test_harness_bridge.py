"""
Tests for app.harness.bridge — HarnessBridge control tower.
"""
from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

os.environ.setdefault("VAPT_AI_POSTGRES_DB", "postgresql://vapt:vapt@localhost:5432/vapt_ai_test")
os.environ.setdefault("VAPT_AI_ENVIRONMENT", "dev")

import pytest

from app.harness.bridge import HarnessBridge, get_harness_bridge, clear_harness_bridge, get_all_bridge_stats
from app.rl import PentestPolicy, ExperienceStore
from app.kg.graph import KnowledgeGraph
from app.kg.types import NodeType, EdgeType
from app.harness.types import VerificationResult, VerificationStatus


@pytest.fixture
def kg_with_path():
    """KG with a Technology → AttackVector → Finding path."""
    kg = KnowledgeGraph()
    nginx_id = kg.add_node(NodeType.TECHNOLOGY, "tag:nginx",
                            metadata={"cvss_base_score": 5.0})
    sqli_id = kg.add_node(NodeType.ATTACK_VECTOR, "SQL Injection",
                           metadata={"cvss_base_score": 9.8})
    finding_id = kg.add_node(NodeType.FINDING, "sqli_login")
    kg.add_edge(nginx_id, sqli_id, EdgeType.HAS_VULN)
    kg.add_edge(sqli_id, finding_id, EdgeType.PRODUCES_FINDING)
    return kg


@pytest.fixture
def bridge(kg_with_path, tmp_path):
    """Fresh HarnessBridge with provided KG + fresh policy/store."""
    policy = PentestPolicy()
    store = ExperienceStore(max_buffer=1000)
    # Patch trace dir to tmp_path
    import app.trace.replay as trace_mod
    original_trace_dir = trace_mod.trace_path
    return HarnessBridge(
        scan_id="scan_test_001",
        target_url="http://example.com",
        policy=policy,
        store=store,
        kg=kg_with_path,
    )


class TestHarnessBridgeInit:
    def test_instantiation(self, kg_with_path):
        bridge = HarnessBridge(
            scan_id="scan_abc", target_url="http://t.com",
            policy=PentestPolicy(), store=ExperienceStore(),
            kg=kg_with_path,
        )
        assert bridge.scan_id == "scan_abc"
        assert bridge.target_url == "http://t.com"
        assert bridge._turn_count == 0
        assert bridge._episode_reward == 0.0

    def test_lazy_init_components(self):
        """Without explicit policy/store/kg, bridge lazy-loads from singletons."""
        bridge = HarnessBridge(scan_id="scan_lazy", target_url="http://t.com")
        # _policy, _store, _kg should be None initially
        assert bridge._policy is None
        assert bridge._store is None
        assert bridge._kg is None


class TestSelectAction:
    def test_returns_valid_action(self, bridge):
        """select_action returns action_index in [0, 7) + state_vec."""
        session = {
            "_turn": 0, "_commands_run": 0,
            "_tokens_used": 1000, "_token_budget": 128000,
            "_last_command": "nuclei",
            "recon_signals": {"technologies": ["nginx"]},
            "findings": [],
        }
        result = bridge.select_action(session, turn=0)
        assert 0 <= result["action_index"] < 7
        assert isinstance(result["action_name"], str)
        assert result["state_vec"].shape == (337,)

    def test_state_vec_stored_in_session(self, bridge):
        """select_action stores _rl_state_vec + _rl_action_idx in session."""
        session = {"_turn": 0, "findings": []}
        bridge.select_action(session, turn=0)
        assert "_rl_state_vec" in session
        assert "_rl_action_idx" in session

    def test_kg_consult_with_matching_tech(self, bridge):
        """When session tech_stack matches KG, KG consult returns paths."""
        session = {
            "_turn": 0, "findings": [],
            "recon_signals": {"technologies": ["nginx"]},
        }
        result = bridge.select_action(session, turn=0)
        # KG has "tag:nginx" node → should find it
        assert result["kg_consult_result"] is not None
        assert len(result["egats_paths"]) > 0

    def test_kg_consult_no_matching_tech(self, bridge):
        """When tech_stack doesn't match KG, kg_consult_result is None."""
        session = {
            "_turn": 0, "findings": [],
            "recon_signals": {"technologies": ["apache"]},  # not in KG
        }
        result = bridge.select_action(session, turn=0)
        assert result["kg_consult_result"] is None


class TestRecordExperience:
    def test_records_to_store(self, bridge):
        """record_experience adds (s, a, r, s') to RL store."""
        session = {
            "_turn": 0, "_commands_run": 0,
            "_tokens_used": 1000, "_token_budget": 128000,
            "_last_command": "nuclei",
            "recon_signals": {"technologies": ["nginx"]},
            "findings": [],
        }
        bridge.select_action(session, turn=0)
        session_after = dict(session)
        session_after["_commands_run"] = 1
        asyncio.run(bridge.record_experience(
            session=session_after, turn=0, action_idx=0,
            reward=1.0, done=False, observation="ok",
        ))
        assert bridge._episode_reward == 1.0
        assert len(bridge._store) == 1  # SumTree has 1 entry

    def test_no_previous_state_skips(self, bridge):
        """Without _rl_state_vec in session, record_experience skips."""
        session = {"_turn": 0, "findings": []}  # no _rl_state_vec
        asyncio.run(bridge.record_experience(
            session=session, turn=0, action_idx=0,
            reward=1.0, done=False,
        ))
        assert bridge._episode_reward == 0.0  # skipped


class TestTrainStep:
    def test_insufficient_experiences(self, bridge):
        """train_step returns trained=False when store has < batch_size/2."""
        result = bridge.train_step(batch_size=64)
        assert result["trained"] is False
        assert result["reason"] == "insufficient_experiences"


class TestProcessRejection:
    def test_returns_negative_reward(self, bridge):
        """process_rejection returns rl_reward < 0."""
        finding = {
            "name": "SQLi", "severity": "high",
            "location": "http://x.com/login", "vuln_type": "SQLi",
        }
        verif = VerificationResult(
            status=VerificationStatus.FALSE_POSITIVE, confidence=0.9,
            method="curl_reprobe", evidence="Not found",
        )
        result = bridge.process_rejection(finding, verif, session={"_turn": 0})
        assert result["rl_reward"] < 0
        assert "corrective_message" in result


class TestEndEpisode:
    def test_epsilon_decays(self, bridge):
        """end_episode decays epsilon."""
        eps_before = bridge.policy.epsilon
        bridge._episode_reward = 5.0
        bridge.end_episode()
        eps_after = bridge.policy.epsilon
        assert eps_after < eps_before

    def test_resets_counters(self, bridge):
        """end_episode resets per-session counters."""
        bridge._episode_reward = 10.0
        bridge._turn_count = 5
        bridge.end_episode()
        assert bridge._episode_reward == 0.0
        assert bridge._turn_count == 0


class TestGetStats:
    def test_stats_contains_scan_id(self, bridge):
        stats = bridge.get_stats()
        assert stats["scan_id"] == "scan_test_001"
        assert "turn_count" in stats
        assert "episode_reward" in stats


class TestRegistry:
    def test_get_harness_bridge_returns_same_instance(self):
        b1 = get_harness_bridge("scan_reg_001", "http://t.com")
        b2 = get_harness_bridge("scan_reg_001")
        assert b1 is b2
        clear_harness_bridge("scan_reg_001")

    def test_clear_harness_bridge(self):
        get_harness_bridge("scan_reg_002", "http://t.com")
        clear_harness_bridge("scan_reg_002")
        b3 = get_harness_bridge("scan_reg_002")
        # New instance after clear
        assert b3.scan_id == "scan_reg_002"
        clear_harness_bridge("scan_reg_002")

    def test_get_all_bridge_stats(self):
        get_harness_bridge("scan_reg_003", "http://t.com")
        stats = get_all_bridge_stats()
        assert "scan_reg_003" in stats
        clear_harness_bridge("scan_reg_003")