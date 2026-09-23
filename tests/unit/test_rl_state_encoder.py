"""
Tests for app.rl.state_encoder — 337-dim state vector encoder.

Covers:
  - State vector shape + dtype (337, float32)
  - Section layout (8 scalars + 256 tech + 32 vec + 32 tool + 9 flags)
  - Hash bucketing deterministic (SHA-256, not PYTHONHASHSEED)
  - Multi-hot tech stack encoding
  - One-hot last-vector / last-tool encoding
  - Flag section correctness
  - from_session dict adapter
  - Padding/truncation defense
  - Empty session edge case
  - Action space size (7) + names
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure project root is on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Set test env BEFORE importing app modules
os.environ.setdefault("VAPT_AI_POSTGRES_DB", "postgresql://vapt:vapt@localhost:5432/vapt_ai_test")
os.environ.setdefault("VAPT_AI_ENVIRONMENT", "dev")

import numpy as np
import pytest

from app.rl.state_encoder import (
    StateEncoder, PentestState, StateEncodingConfig, DEFAULT_STATE_CONFIG,
    STATE_DIM, ACTION_DIM, ACTION_SPACE,
    TECH_STACK_VOCAB, VECTOR_VOCAB, TOOL_VOCAB,
    _hash_bucket, _one_hot, encode_session,
)


# ── Section 1: Constants ─────────────────────────────────────────────────

class TestConstants:
    def test_state_dim_337(self):
        """STATE_DIM must be exactly 337 (8+256+32+32+9)."""
        assert STATE_DIM == 337

    def test_action_dim_7(self):
        assert ACTION_DIM == 7

    def test_action_space_names(self):
        """All 7 actions match master plan §12 W16 action space."""
        assert ACTION_SPACE == [
            "execute_command_recon",
            "execute_command_exploit",
            "execute_command_fuzz",
            "select_skill",
            "consult_kg",
            "report_vulnerability",
            "pentest_complete",
        ]

    def test_vocab_sizes(self):
        assert TECH_STACK_VOCAB == 256
        assert VECTOR_VOCAB == 32
        assert TOOL_VOCAB == 32


# ── Section 2: Helpers ───────────────────────────────────────────────────

class TestHelpers:
    def test_hash_bucket_deterministic(self):
        """SHA-256 hash must be reproducible across runs (not PYTHONHASHSEED)."""
        h1 = _hash_bucket("nginx", 256)
        h2 = _hash_bucket("nginx", 256)
        assert h1 == h2
        assert 0 <= h1 < 256

    def test_hash_bucket_different_inputs(self):
        """Different strings should (usually) hash to different buckets."""
        h1 = _hash_bucket("nginx", 256)
        h2 = _hash_bucket("apache", 256)
        assert h1 != h2  # not guaranteed but extremely likely with SHA-256

    def test_hash_bucket_range(self):
        for s in ["", "a", "nginx", "php", "wordpress"]:
            h = _hash_bucket(s, 256)
            assert 0 <= h < 256

    def test_one_hot_in_range(self):
        v = _one_hot(5, 32)
        assert v.shape == (32,)
        assert v.dtype == np.float32
        assert v[5] == 1.0
        assert v.sum() == 1.0

    def test_one_hot_out_of_range(self):
        """Out-of-range index → zero vector (no signal)."""
        v_neg = _one_hot(-1, 32)
        v_big = _one_hot(100, 32)
        assert v_neg.sum() == 0.0
        assert v_big.sum() == 0.0


# ── Section 3: PentestState.to_vector ────────────────────────────────────

class TestPentestStateVector:
    def _make_state(self, **overrides):
        defaults = dict(
            turn=10, tdi=0.5, mode="llm",
            findings_count=5, verified_findings=3, fp_findings=1,
            commands_run=20, tokens_used=10000, token_budget_remaining=0.5,
            tech_stack=("nginx", "php", "mysql"),
            last_vector="SQLi", last_tool="recon",
            last_success=True, kg_available=True,
            planner_paths_live=10, planner_paths_pruned=5,
            recon_login=True, recon_api=False, recon_upload=True,
            recon_graphql=False, recon_websocket=True,
        )
        defaults.update(overrides)
        return PentestState(**defaults)

    def test_vector_shape_and_dtype(self):
        state = self._make_state()
        vec = state.to_vector()
        assert vec.shape == (337,)
        assert vec.dtype == np.float32

    def test_scalar_section_8_dims(self):
        """Section 1: 8 scalar normalized features."""
        state = self._make_state(turn=50, tdi=0.3, mode="broaden")
        vec = state.to_vector()
        scalars = vec[:8]
        assert scalars.shape == (8,)
        # All scalars in [0, 1]
        assert (scalars >= 0).all() and (scalars <= 1).all()
        # Turn normalized
        assert abs(scalars[0] - 0.5) < 1e-6  # 50/100
        # TDI
        assert abs(scalars[1] - 0.3) < 1e-6
        # Mode broaden=1.0
        assert scalars[2] == 1.0

    def test_tech_stack_multihot(self):
        """Section 2: 256-dim tech stack multi-hot — 3 techs → 3 set bits."""
        state = self._make_state(tech_stack=("nginx", "php", "mysql"))
        vec = state.to_vector()
        tech_vec = vec[8:8 + 256]
        assert int(tech_vec.sum()) == 3
        # Each set bit is 1.0
        assert set(tech_vec[tech_vec > 0].tolist()) == {1.0}

    def test_last_vector_onehot(self):
        """Section 3: 32-dim one-hot — exactly 1 set bit."""
        state = self._make_state(last_vector="SQLi")
        vec = state.to_vector()
        vec_section = vec[8 + 256:8 + 256 + 32]
        assert int(vec_section.sum()) == 1

    def test_last_tool_onehot(self):
        """Section 4: 32-dim one-hot — exactly 1 set bit."""
        state = self._make_state(last_tool="recon")
        vec = state.to_vector()
        tool_section = vec[8 + 256 + 32:8 + 256 + 32 + 32]
        assert int(tool_section.sum()) == 1

    def test_flags_section_9_dims(self):
        """Section 5: 9 flags."""
        state = self._make_state()
        vec = state.to_vector()
        flags = vec[-9:]
        assert flags.shape == (9,)
        # First flag is last_success (True → 1.0)
        assert flags[0] == 1.0
        # Second flag is kg_available (True → 1.0)
        assert flags[1] == 1.0
        # recon_login (True → 1.0), recon_api (False → 0.0)
        assert flags[4] == 1.0  # recon_login
        assert flags[5] == 0.0  # recon_api

    def test_reproducibility(self):
        """Same state → same vector (deterministic)."""
        state = self._make_state()
        v1 = state.to_vector()
        v2 = state.to_vector()
        assert np.array_equal(v1, v2)


# ── Section 4: StateEncoder.from_session ─────────────────────────────────

class TestStateEncoderFromSession:
    def test_full_session(self):
        session = {
            "_turn": 5, "_commands_run": 12,
            "_tokens_used": 5000, "_token_budget": 128000,
            "_last_command": "nuclei -u http://target.com",
            "_last_vector": "XSS",
            "_last_turn_success": True,
            "_kg_consulted": True,
            "tdi_log": [{"tdi": 0.3, "mode": "exploit"}],
            "recon_signals": {
                "technologies": ["nginx", "php"],
                "has_login": True, "has_api": False, "has_upload": True,
                "has_graphql": False, "has_websocket": True,
            },
            "findings": [
                {"name": "SQLi", "location": "/login", "severity": "high",
                 "verified": True, "false_positive": False},
                {"name": "XSS", "location": "/search", "severity": "medium",
                 "verified": False, "false_positive": True},
            ],
        }
        state = StateEncoder.from_session(session)
        assert state.turn == 5
        assert state.tdi == 0.3
        assert state.mode == "exploit"
        assert state.findings_count == 2
        assert state.verified_findings == 1
        assert state.fp_findings == 1
        assert state.commands_run == 12
        assert state.last_success is True
        assert state.kg_available is True
        assert state.recon_login is True
        assert state.recon_api is False
        assert state.last_tool == "recon"  # nuclei → recon

    def test_empty_session(self):
        """Empty session dict → all defaults, no crash."""
        state = StateEncoder.from_session({})
        assert state.turn == 0
        assert state.tdi == 0.5  # default
        assert state.mode == "llm"  # default
        assert state.findings_count == 0
        assert state.token_budget_remaining >= 0.0

    def test_encode_session_one_liner(self):
        session = {"_turn": 3}
        v1 = encode_session(session)
        v2 = StateEncoder.from_session(session).to_vector()
        assert np.array_equal(v1, v2)

    def test_infer_tool_from_command(self):
        """_infer_tool_from_command maps commands → tool categories."""
        assert StateEncoder._infer_tool_from_command("nuclei -u http://x") == "recon"
        assert StateEncoder._infer_tool_from_command("nmap -sV 10.0.0.1") == "recon"
        assert StateEncoder._infer_tool_from_command("sqlmap -u http://x --batch") == "exploit"
        assert StateEncoder._infer_tool_from_command("msfconsole -q") == "exploit"
        assert StateEncoder._infer_tool_from_command("ffuf -u http://x/FUZZ") == "fuzz"
        assert StateEncoder._infer_tool_from_command("gobuster dir -u http://x") == "fuzz"
        assert StateEncoder._infer_tool_from_command("") == "recon"  # default