"""
Tests for app.kg.graph — KnowledgeGraph (NetworkX + SQL bridge).

Covers:
  - add_node / get_node / find_node (idempotent)
  - add_edge (valid + invalid type combinations)
  - get_edge / update_edge_outcome (Laplace probability)
  - get_attack_paths (BFS path scoring)
  - consult / record_consult_hit / record_consult_miss
  - fork_for_scan / merge_scan (per-scan snapshot flow)
  - stats / search / clear
  - 6 node types + 5 edge types coverage
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

import pytest

from app.kg.graph import KnowledgeGraph, get_kg, reset_kg_singleton
from app.kg.types import NodeType, EdgeType, make_node_id


# ── Test fixture ─────────────────────────────────────────────────────────

@pytest.fixture
def kg():
    """Fresh KnowledgeGraph for each test."""
    return KnowledgeGraph()


def _build_sample_kg(kg: KnowledgeGraph) -> dict[str, str]:
    """Build a sample KG with all 6 node types + 5 edge types. Returns node IDs."""
    ids = {}
    ids["nginx"] = kg.add_node(NodeType.TECHNOLOGY, "nginx 1.18.0",
                                metadata={"cvss_base_score": 5.0})
    ids["sqli"] = kg.add_node(NodeType.ATTACK_VECTOR, "SQL Injection",
                              metadata={"wstg_test_id": "WSTG-INPV-05", "cvss_base_score": 9.8})
    ids["finding"] = kg.add_node(NodeType.FINDING, "sqli_login_page")
    ids["rec"] = kg.add_node(NodeType.RECOMMENDATION, "Use prepared statements")
    ids["ver"] = kg.add_node(NodeType.VERIFICATION, "sqlmap --batch --dump")
    ids["outcome"] = kg.add_node(NodeType.OUTCOME, "Database dumped")

    # 5 edges (one per type)
    kg.add_edge(ids["nginx"], ids["sqli"], EdgeType.HAS_VULN)
    kg.add_edge(ids["sqli"], ids["finding"], EdgeType.PRODUCES_FINDING)
    kg.add_edge(ids["finding"], ids["rec"], EdgeType.RECOMMENDS)
    kg.add_edge(ids["finding"], ids["ver"], EdgeType.VERIFIED_BY)
    kg.add_edge(ids["finding"], ids["outcome"], EdgeType.RESULTED_IN)
    return ids


# ── Section 1: Node ops ──────────────────────────────────────────────────

class TestNodeOps:
    def test_add_node_returns_id(self, kg):
        node_id = kg.add_node(NodeType.TECHNOLOGY, "nginx 1.18.0")
        assert node_id.startswith("kg_")
        assert len(node_id) == 19  # "kg_" + 16 hex chars

    def test_add_node_idempotent(self, kg):
        id1 = kg.add_node(NodeType.TECHNOLOGY, "nginx 1.18.0", metadata={"a": 1})
        id2 = kg.add_node(NodeType.TECHNOLOGY, "nginx 1.18.0", metadata={"b": 2})
        assert id1 == id2  # same ID
        # Metadata merged
        node = kg.get_node(id1)
        assert node.metadata.get("a") == 1
        assert node.metadata.get("b") == 2

    def test_get_node_not_found(self, kg):
        assert kg.get_node("nonexistent") is None

    def test_find_node(self, kg):
        node_id = kg.add_node(NodeType.ATTACK_VECTOR, "SQL Injection")
        found = kg.find_node(NodeType.ATTACK_VECTOR, "SQL Injection")
        assert found == node_id
        not_found = kg.find_node(NodeType.ATTACK_VECTOR, "XSS")
        assert not_found is None

    def test_6_node_types(self, kg):
        """All 6 NodeType values are addable."""
        for nt in NodeType:
            node_id = kg.add_node(nt, f"test_{nt.value}")
            assert node_id is not None
            node = kg.get_node(node_id)
            assert node.node_type == nt


# ── Section 2: Edge ops ──────────────────────────────────────────────────

class TestEdgeOps:
    def test_add_edge_valid(self, kg):
        ids = _build_sample_kg(kg)
        assert kg.graph.number_of_edges() == 5

    def test_add_edge_invalid_type_mismatch(self, kg):
        """Tech→Finding has_vuln should fail (must be Tech→AttackVector)."""
        tech_id = kg.add_node(NodeType.TECHNOLOGY, "nginx")
        finding_id = kg.add_node(NodeType.FINDING, "sqli")
        ok = kg.add_edge(tech_id, finding_id, EdgeType.HAS_VULN)
        assert ok is False

    def test_add_edge_missing_endpoint(self, kg):
        ok = kg.add_edge("nonexistent_src", "nonexistent_tgt", EdgeType.HAS_VULN)
        assert ok is False

    def test_get_edge(self, kg):
        ids = _build_sample_kg(kg)
        edge = kg.get_edge(ids["nginx"], ids["sqli"])
        assert edge is not None
        assert edge.edge_type == EdgeType.HAS_VULN

    def test_update_edge_outcome(self, kg):
        ids = _build_sample_kg(kg)
        # 2 successes + 1 failure
        kg.update_edge_outcome(ids["sqli"], ids["finding"], success=True, scan_id="s1")
        kg.update_edge_outcome(ids["sqli"], ids["finding"], success=True, scan_id="s2")
        kg.update_edge_outcome(ids["sqli"], ids["finding"], success=False, scan_id="s3")
        edge = kg.get_edge(ids["sqli"], ids["finding"])
        assert edge.success_count == 2
        assert edge.total_attempts == 3
        # Laplace: (2+1)/(3+2) = 3/5 = 0.6
        assert abs(edge.probability - 0.6) < 1e-6

    def test_update_edge_outcome_not_found(self, kg):
        ok = kg.update_edge_outcome("nope1", "nope2", success=True)
        assert ok is False

    def test_5_edge_types(self, kg):
        """All 5 EdgeType values can be added (with correct node types)."""
        ids = _build_sample_kg(kg)
        edge_counts = kg.stats()["edges_by_type"]
        assert set(edge_counts.keys()) == {et.value for et in EdgeType}


# ── Section 3: Probability Laplace smoothing ─────────────────────────────

class TestProbabilityLaplace:
    def test_fresh_edge_0_5(self, kg):
        """Fresh edge (0/0) → 0.5 uncertain prior."""
        ids = _build_sample_kg(kg)
        edge = kg.get_edge(ids["nginx"], ids["sqli"])
        assert abs(edge.probability - 0.5) < 1e-6

    def test_all_success_0_75(self, kg):
        """8/8 → (8+1)/(8+2) = 0.9 — not 1.0 (Laplace prior)."""
        ids = _build_sample_kg(kg)
        for _ in range(8):
            kg.update_edge_outcome(ids["sqli"], ids["finding"], success=True)
        edge = kg.get_edge(ids["sqli"], ids["finding"])
        assert abs(edge.probability - 0.9) < 1e-6

    def test_all_failure_low(self, kg):
        """0/10 → (0+1)/(10+2) = 1/12 ≈ 0.083 — not 0.0 (Laplace prior)."""
        ids = _build_sample_kg(kg)
        for _ in range(10):
            kg.update_edge_outcome(ids["sqli"], ids["finding"], success=False)
        edge = kg.get_edge(ids["sqli"], ids["finding"])
        assert 0.05 < edge.probability < 0.15


# ── Section 4: Path queries ──────────────────────────────────────────────

class TestPathQueries:
    def test_get_attack_paths(self, kg):
        ids = _build_sample_kg(kg)
        paths = kg.get_attack_paths([ids["nginx"]], max_depth=4, top_k=5)
        assert len(paths) > 0
        # Path should start at nginx and end at Finding
        for p in paths:
            assert p.nodes[0] == ids["nginx"]
            last_node_type = kg.graph.nodes[p.nodes[-1]].get("node_type")
            assert last_node_type == NodeType.FINDING.value

    def test_get_attack_paths_empty_source(self, kg):
        paths = kg.get_attack_paths([], max_depth=4)
        assert paths == []

    def test_get_attack_paths_nonexistent_source(self, kg):
        paths = kg.get_attack_paths(["nonexistent"], max_depth=4)
        assert paths == []

    def test_consult_returns_consult_id(self, kg):
        ids = _build_sample_kg(kg)
        result = kg.consult([ids["nginx"]])
        assert result.consult_id.startswith("consult_")
        assert len(result.consult_id) > 10

    def test_record_consult_hit(self, kg):
        ids = _build_sample_kg(kg)
        result = kg.consult([ids["nginx"]])
        kg.record_consult_hit(result.consult_id, finding_id="finding_abc")
        # Internal consult log should have hit recorded
        assert result.consult_id in kg._consult_log
        assert kg._consult_log[result.consult_id]["hit"]["finding_id"] == "finding_abc"


# ── Section 5: Snapshot flow ─────────────────────────────────────────────

class TestSnapshotFlow:
    def test_fork_for_scan(self, kg):
        ids = _build_sample_kg(kg)
        fork = kg.fork_for_scan("scan_test_001")
        assert fork._is_fork is True
        assert fork._fork_scan_id == "scan_test_001"
        # Same nodes + edges as main KG
        assert fork.graph.number_of_nodes() == kg.graph.number_of_nodes()
        assert fork.graph.number_of_edges() == kg.graph.number_of_edges()

    def test_fork_isolated_from_main(self, kg):
        """Writes to fork should NOT affect main KG."""
        ids = _build_sample_kg(kg)
        main_attempts_before = kg.get_edge(ids["sqli"], ids["finding"]).total_attempts

        fork = kg.fork_for_scan("scan_test_001")
        fork.update_edge_outcome(ids["sqli"], ids["finding"], success=True, scan_id="scan_test_001")

        # Main KG unchanged
        main_attempts_after = kg.get_edge(ids["sqli"], ids["finding"]).total_attempts
        assert main_attempts_after == main_attempts_before

        # Fork has the update
        fork_edge = fork.get_edge(ids["sqli"], ids["finding"])
        assert fork_edge.total_attempts > main_attempts_before

    def test_merge_scan(self, kg):
        ids = _build_sample_kg(kg)
        main_before = kg.get_edge(ids["sqli"], ids["finding"]).total_attempts

        fork = kg.fork_for_scan("scan_test_001")
        fork.update_edge_outcome(ids["sqli"], ids["finding"], success=True, scan_id="scan_test_001")
        fork.update_edge_outcome(ids["sqli"], ids["finding"], success=False, scan_id="scan_test_001")

        result = kg.merge_scan("scan_test_001", fork)
        assert result["edges_updated"] >= 1

        # Main KG now has updated counts
        main_after = kg.get_edge(ids["sqli"], ids["finding"])
        assert main_after.total_attempts == main_before + 2


# ── Section 6: Introspection ─────────────────────────────────────────────

class TestIntrospection:
    def test_stats(self, kg):
        _build_sample_kg(kg)
        stats = kg.stats()
        assert stats["total_nodes"] == 6
        assert stats["total_edges"] == 5
        assert "Technology" in stats["nodes_by_type"]
        assert "has_vuln" in stats["edges_by_type"]
        assert stats["is_fork"] is False

    def test_search(self, kg):
        ids = _build_sample_kg(kg)
        results = kg.search("sql", limit=10)
        # Should find SQL Injection + sqli_login_page + sqlmap verification
        assert len(results) >= 2
        for r in results:
            assert "sql" in r["name"].lower()

    def test_clear(self, kg):
        _build_sample_kg(kg)
        assert kg.graph.number_of_nodes() > 0
        kg.clear()
        assert kg.graph.number_of_nodes() == 0
        assert kg.graph.number_of_edges() == 0


# ── Section 7: Singleton ─────────────────────────────────────────────────

class TestSingleton:
    def test_get_kg_returns_same_instance(self):
        reset_kg_singleton()
        kg1 = get_kg()
        kg2 = get_kg()
        assert kg1 is kg2

    def test_reset_kg_singleton(self):
        reset_kg_singleton()
        kg1 = get_kg()
        kg1.add_node(NodeType.TECHNOLOGY, "test")
        reset_kg_singleton()
        kg2 = get_kg()
        assert kg2.graph.number_of_nodes() == 0  # fresh after reset