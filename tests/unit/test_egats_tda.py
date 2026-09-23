"""
Tests for app.harness.egats + app.harness.tda.

Covers:
  - TDA: compute_tdi 3 modes (broaden/exploit/llm)
  - TDA: dimension calculators (horizon, evidence, context, success)
  - TDA: evidence_score_from_label
  - EGATS: seed + search with mock simulate_fn
  - EGATS: UCB formula (unattempted=+inf, exploration, penalty, specificity)
  - EGATS: pruning (TDI > 0.6 after k_min_prune)
  - EGATS: backprop updates KG edges
  - EGATS: mode_for mapping
"""
from __future__ import annotations

import math
import os
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

os.environ.setdefault("VAPT_AI_POSTGRES_DB", "postgresql://vapt:vapt@localhost:5432/vapt_ai_test")
os.environ.setdefault("VAPT_AI_ENVIRONMENT", "dev")

import pytest

from app.harness.tda import (
    compute_tdi, Mode, TDIResult,
    horizon_normalised, evidence_confidence, context_load, historical_success,
    evidence_score_from_label,
    TDI_BROADEN, TDI_EXPLOIT,
    VERIFIED_EXPLOIT, CONFIRMED_VULN, PLAUSIBLE_HYPOTHESIS, SPECULATIVE_HYPOTHESIS,
    W_H, W_E, W_C, W_S, DEFAULT_MAX_HORIZON,
)
from app.harness.egats import (
    EGATS, SimulationOutcome, EGATSResult,
    DEFAULT_UCB_C, DEFAULT_LAMBDA, DEFAULT_K_MIN_PRUNE,
    DEFAULT_MU_SPECIFICITY, DEFAULT_EMA_ALPHA,
)
from app.kg.graph import KnowledgeGraph
from app.kg.types import NodeType, EdgeType


# ── Section 1: TDA constants ─────────────────────────────────────────────

class TestTDAConstants:
    def test_weights_sum(self):
        """Paper weights: H=0.30, E=0.30, C=0.20, S=0.20 (sum=1.0)."""
        assert W_H == 0.30
        assert W_E == 0.30
        assert W_C == 0.20
        assert W_S == 0.20
        assert abs((W_H + W_E + W_C + W_S) - 1.0) < 1e-6

    def test_thresholds(self):
        assert TDI_BROADEN == 0.60
        assert TDI_EXPLOIT == 0.30
        assert TDI_BROADEN > TDI_EXPLOIT

    def test_evidence_levels_ordered(self):
        assert VERIFIED_EXPLOIT > CONFIRMED_VULN > PLAUSIBLE_HYPOTHESIS > SPECULATIVE_HYPOTHESIS


# ── Section 2: TDA dimension calculators ─────────────────────────────────

class TestTDADimensions:
    def test_horizon_normalised(self):
        assert horizon_normalised(0, 20) == 0.0
        assert horizon_normalised(10, 20) == 0.5
        assert horizon_normalised(20, 20) == 1.0
        assert horizon_normalised(25, 20) == 1.0  # clipped
        assert horizon_normalised(10, 0) == 0.0   # zero max_horizon

    def test_evidence_confidence(self):
        assert evidence_confidence([]) == SPECULATIVE_HYPOTHESIS
        assert abs(evidence_confidence([0.8, 0.6]) - 0.7) < 1e-6
        assert evidence_confidence([1.5, -0.5]) == 0.5  # clipped to [0,1] then averaged

    def test_context_load(self):
        assert context_load(0, 128000) == 0.0
        assert context_load(64000, 128000) == 0.5
        assert context_load(200000, 128000) == 1.0  # clipped
        assert context_load(100, 0) == 0.0  # zero context_window

    def test_historical_success_laplace(self):
        """Laplace smoothing: (s+α)/(n+α+β) with α=β=1."""
        assert abs(historical_success(0, 0) - 0.5) < 1e-6  # 1/2
        assert abs(historical_success(8, 10) - 0.75) < 1e-6  # 9/12
        assert abs(historical_success(0, 10) - (1/12)) < 1e-6  # 1/12


# ── Section 3: TDA compute_tdi + 3 modes ─────────────────────────────────

class TestTDAModes:
    def test_mode_broaden(self):
        """High difficulty → BROADEN (TDI > 0.6)."""
        r = compute_tdi(
            remaining_steps=18, path_evidence=[],
            tokens_used=120000, context_window=128000,
            successes=0, attempts=5,
        )
        assert r.tdi > TDI_BROADEN
        assert r.mode == Mode.BROADEN

    def test_mode_exploit(self):
        """Low difficulty → EXPLOIT (TDI < 0.3)."""
        r = compute_tdi(
            remaining_steps=2, path_evidence=[1.0, 0.9, 0.8],
            tokens_used=5000, context_window=128000,
            successes=8, attempts=10,
        )
        assert r.tdi < TDI_EXPLOIT
        assert r.mode == Mode.EXPLOIT

    def test_mode_llm_pick(self):
        """Middle difficulty → LLM_PICK (0.3 ≤ TDI ≤ 0.6)."""
        r = compute_tdi(
            remaining_steps=10, path_evidence=[0.5, 0.6],
            tokens_used=64000, context_window=128000,
            successes=3, attempts=7,
        )
        assert TDI_EXPLOIT <= r.tdi <= TDI_BROADEN
        assert r.mode == Mode.LLM_PICK

    def test_tdi_result_as_dict(self):
        r = compute_tdi(
            remaining_steps=5, path_evidence=[0.7],
            tokens_used=30000, context_window=128000,
            successes=4, attempts=6,
        )
        d = r.as_dict()
        assert "tdi" in d and "mode" in d
        assert d["mode"] in ("broaden", "llm", "exploit")


# ── Section 4: evidence_score_from_label ─────────────────────────────────

class TestEvidenceLabel:
    @pytest.mark.parametrize("label,expected", [
        ("verified", VERIFIED_EXPLOIT),
        ("exploit", VERIFIED_EXPLOIT),
        ("confirmed", CONFIRMED_VULN),
        ("vuln", CONFIRMED_VULN),
        ("plausible", PLAUSIBLE_HYPOTHESIS),
        ("hypothesis", PLAUSIBLE_HYPOTHESIS),
        ("speculative", SPECULATIVE_HYPOTHESIS),
        ("guess", SPECULATIVE_HYPOTHESIS),
        (None, SPECULATIVE_HYPOTHESIS),
        ("", SPECULATIVE_HYPOTHESIS),
        ("unknown", SPECULATIVE_HYPOTHESIS),  # fallback
    ])
    def test_label_mapping(self, label, expected):
        assert evidence_score_from_label(label) == expected

    def test_case_insensitive(self):
        assert evidence_score_from_label("VERIFIED") == VERIFIED_EXPLOIT
        assert evidence_score_from_label("Confirmed") == CONFIRMED_VULN


# ── Section 5: EGATS constants + dataclasses ─────────────────────────────

class TestEGATSConstants:
    def test_ucb_c_sqrt2(self):
        assert abs(DEFAULT_UCB_C - math.sqrt(2)) < 1e-6

    def test_defaults(self):
        assert DEFAULT_LAMBDA == 0.5
        assert DEFAULT_K_MIN_PRUNE == 3
        assert DEFAULT_MU_SPECIFICITY == 0.2
        assert DEFAULT_EMA_ALPHA == 0.3

    def test_simulation_outcome(self):
        o = SimulationOutcome(success=True, tdi=0.3, note="ok")
        assert o.success is True
        assert o.tdi == 0.3


# ── Section 6: EGATS seed + search ───────────────────────────────────────

class TestEGATSSeedSearch:
    @pytest.fixture
    def kg_with_paths(self):
        kg = KnowledgeGraph()
        nginx_id = kg.add_node(NodeType.TECHNOLOGY, "nginx 1.18.0",
                                metadata={"cvss_base_score": 5.0})
        sqli_id = kg.add_node(NodeType.ATTACK_VECTOR, "SQL Injection",
                              metadata={"cvss_base_score": 9.8})
        finding_id = kg.add_node(NodeType.FINDING, "sqli_login")
        kg.add_edge(nginx_id, sqli_id, EdgeType.HAS_VULN)
        kg.add_edge(sqli_id, finding_id, EdgeType.PRODUCES_FINDING)
        # Second path
        xss_id = kg.add_node(NodeType.ATTACK_VECTOR, "XSS",
                              metadata={"cvss_base_score": 6.1})
        finding_xss = kg.add_node(NodeType.FINDING, "xss_search")
        kg.add_edge(nginx_id, xss_id, EdgeType.HAS_VULN)
        kg.add_edge(xss_id, finding_xss, EdgeType.PRODUCES_FINDING)
        return kg, nginx_id

    def test_seed_returns_count(self, kg_with_paths):
        kg, nginx_id = kg_with_paths
        egats = EGATS(kg)
        seeded = egats.seed([nginx_id], max_depth=4, top_k=5)
        assert seeded > 0
        assert len(egats.stats()) == seeded

    def test_search_returns_result(self, kg_with_paths):
        kg, nginx_id = kg_with_paths
        egats = EGATS(kg)
        egats.seed([nginx_id])

        def mock_sim(path_nodes):
            return SimulationOutcome(success=True, tdi=0.3, note="ok")

        result = egats.search(mock_sim, max_iterations=3)
        assert isinstance(result, EGATSResult)
        assert result.iterations > 0
        assert result.best_path is not None

    def test_search_empty_kg(self):
        kg = KnowledgeGraph()
        egats = EGATS(kg)
        # No seed → no paths → search returns 0 iterations
        result = egats.search(lambda p: SimulationOutcome(True, 0.5), max_iterations=3)
        assert result.iterations == 0
        assert result.best_path is None


# ── Section 7: EGATS UCB formula ─────────────────────────────────────────

class TestEGATSUCB:
    def test_unattempted_is_inf(self):
        """Unattempted paths get +inf so they're always explored first."""
        from app.harness.egats import _PathStat
        kg = KnowledgeGraph()
        egats = EGATS(kg)
        stat = _PathStat(nodes=["a"], edges=[], base_score=0.5, attempts=0)
        assert egats._ucb(stat, total_n=10) == math.inf

    def test_ucb_exploration_term(self):
        """UCB should include c*sqrt(ln(N)/n) exploration term."""
        from app.harness.egats import _PathStat
        kg = KnowledgeGraph()
        egats = EGATS(kg, ucb_c=1.0, lambda_penalty=0.0, mu_specificity=0.0)
        stat = _PathStat(nodes=["a"], edges=[], base_score=0.5, attempts=3, tdi_ema=0.5)
        ucb = egats._ucb(stat, total_n=10)
        # phi=0.5 + 1*sqrt(ln(10)/3) - 0 + 0 = 0.5 + sqrt(0.767) ≈ 0.5 + 0.876 = 1.376
        expected = 0.5 + math.sqrt(math.log(10) / 3)
        assert abs(ucb - expected) < 1e-4

    def test_ucb_tdi_penalty(self):
        """Higher TDI → lower UCB (penalised)."""
        from app.harness.egats import _PathStat
        kg = KnowledgeGraph()
        egats = EGATS(kg, lambda_penalty=0.5)
        stat_low_tdi = _PathStat(nodes=["a"], edges=[], base_score=0.5, attempts=3, tdi_ema=0.2)
        stat_high_tdi = _PathStat(nodes=["b"], edges=[], base_score=0.5, attempts=3, tdi_ema=0.8)
        ucb_low = egats._ucb(stat_low_tdi, total_n=10)
        ucb_high = egats._ucb(stat_high_tdi, total_n=10)
        assert ucb_low > ucb_high  # lower TDI → higher UCB

    def test_ucb_specificity_boost(self):
        """Higher match_specificity → higher UCB (boost)."""
        from app.harness.egats import _PathStat
        kg = KnowledgeGraph()
        egats = EGATS(kg, mu_specificity=0.5)
        stat_low_spec = _PathStat(nodes=["a"], edges=[], base_score=0.5, attempts=3,
                                  tdi_ema=0.5, match_specificity=0.0)
        stat_high_spec = _PathStat(nodes=["b"], edges=[], base_score=0.5, attempts=3,
                                   tdi_ema=0.5, match_specificity=1.0)
        ucb_low = egats._ucb(stat_low_spec, total_n=10)
        ucb_high = egats._ucb(stat_high_spec, total_n=10)
        assert ucb_high > ucb_low


# ── Section 8: EGATS pruning ─────────────────────────────────────────────

class TestEGATSPruning:
    def test_pruning_high_tdi(self):
        """Paths with TDI > 0.6 after k_min_prune attempts should be pruned."""
        kg = KnowledgeGraph()
        nginx_id = kg.add_node(NodeType.TECHNOLOGY, "nginx")
        sqli_id = kg.add_node(NodeType.ATTACK_VECTOR, "SQLi")
        finding_id = kg.add_node(NodeType.FINDING, "sqli")
        kg.add_edge(nginx_id, sqli_id, EdgeType.HAS_VULN)
        kg.add_edge(sqli_id, finding_id, EdgeType.PRODUCES_FINDING)

        egats = EGATS(kg, k_min_prune=2)
        egats.seed([nginx_id])

        def always_hard(path_nodes):
            return SimulationOutcome(success=False, tdi=0.9, note="too hard")

        egats.search(always_hard, max_iterations=5)
        pruned = [s for s in egats.stats() if s.pruned]
        assert len(pruned) > 0

    def test_no_pruning_low_tdi(self):
        """Paths with TDI < 0.6 should NOT be pruned."""
        kg = KnowledgeGraph()
        nginx_id = kg.add_node(NodeType.TECHNOLOGY, "nginx")
        sqli_id = kg.add_node(NodeType.ATTACK_VECTOR, "SQLi")
        finding_id = kg.add_node(NodeType.FINDING, "sqli")
        kg.add_edge(nginx_id, sqli_id, EdgeType.HAS_VULN)
        kg.add_edge(sqli_id, finding_id, EdgeType.PRODUCES_FINDING)

        egats = EGATS(kg, k_min_prune=2)
        egats.seed([nginx_id])

        def always_easy(path_nodes):
            return SimulationOutcome(success=True, tdi=0.2, note="easy")

        egats.search(always_easy, max_iterations=5)
        pruned = [s for s in egats.stats() if s.pruned]
        assert len(pruned) == 0


# ── Section 9: EGATS backprop updates KG edges ───────────────────────────

class TestEGATSBackprop:
    def test_backprop_updates_kg_edges(self):
        """EGATS backprop should call kg.update_edge_outcome() on path edges."""
        kg = KnowledgeGraph()
        nginx_id = kg.add_node(NodeType.TECHNOLOGY, "nginx")
        sqli_id = kg.add_node(NodeType.ATTACK_VECTOR, "SQLi")
        finding_id = kg.add_node(NodeType.FINDING, "sqli")
        kg.add_edge(nginx_id, sqli_id, EdgeType.HAS_VULN)
        kg.add_edge(sqli_id, finding_id, EdgeType.PRODUCES_FINDING)

        before_attempts = kg.get_edge(sqli_id, finding_id).total_attempts

        egats = EGATS(kg)
        egats.seed([nginx_id])
        egats.search(lambda p: SimulationOutcome(True, 0.3), max_iterations=3)

        after_attempts = kg.get_edge(sqli_id, finding_id).total_attempts
        assert after_attempts > before_attempts  # backprop wrote to KG


# ── Section 10: EGATS mode_for ───────────────────────────────────────────

class TestEGATSModeFor:
    def test_mode_for_exploit(self):
        from app.harness.egats import _PathStat
        kg = KnowledgeGraph()
        egats = EGATS(kg)
        stat = _PathStat(nodes=["a"], edges=[], base_score=0.5, tdi_ema=0.2)
        assert egats.mode_for(stat) == Mode.EXPLOIT

    def test_mode_for_llm(self):
        from app.harness.egats import _PathStat
        kg = KnowledgeGraph()
        egats = EGATS(kg)
        stat = _PathStat(nodes=["a"], edges=[], base_score=0.5, tdi_ema=0.4)
        assert egats.mode_for(stat) == Mode.LLM_PICK

    def test_mode_for_broaden(self):
        from app.harness.egats import _PathStat
        kg = KnowledgeGraph()
        egats = EGATS(kg)
        stat = _PathStat(nodes=["a"], edges=[], base_score=0.5, tdi_ema=0.7)
        assert egats.mode_for(stat) == Mode.BROADEN

    def test_mode_for_no_tdi(self):
        """No tdi_ema → default 0.5 → LLM_PICK."""
        from app.harness.egats import _PathStat
        kg = KnowledgeGraph()
        egats = EGATS(kg)
        stat = _PathStat(nodes=["a"], edges=[], base_score=0.5, tdi_ema=None)
        assert egats.mode_for(stat) == Mode.LLM_PICK