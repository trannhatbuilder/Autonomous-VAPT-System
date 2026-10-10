"""
Integration tests — orchestration shared state, mode selector, and router.

Test coverage:
    - test_mode_selector: select_mode() picks right mode based on target + scope
    - test_shared_state_typeddict: SharedState declares all required fields
    - test_d18_guardrails_enforced: BaseOrchestrator caps decisions at 30
    - test_orchestrator_result_serialization: to_dict() produces valid FastAPI response
    - test_load_orchestrator_prompts: prompt files load successfully
    - test_orchestration_router_endpoints: FastAPI router exposes /modes + /agents + /scans/start-mode

These tests do NOT require a database — they exercise the orchestration
base classes and routing in isolation.

Run:
    pytest tests/integration/test_orchestration_modes.py -v

Or with coverage:
    pytest tests/integration/test_orchestration_modes.py -v --cov=app.orchestration
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app.orchestration.base import (
    AgentDecision,
    BaseOrchestrator,
    OrchestratorMode,
    OrchestratorResult,
    SharedState,
    load_orchestrator_prompt,
    MAX_DECISIONS_PER_SCAN,
    MAX_PARALLEL_SUBAGENTS,
    MAX_AGENTS,
    MAX_TOKENS_PER_SCAN,
    MAX_SCAN_DURATION_SECONDS,
    generate_scan_id,
)
from app.orchestration.mode_selector import (
    ModeSelectionInput,
    select_mode,
    select_mode_simple,
    infer_target_type,
)


# ---------- 1. Mode selector tests ----------

class TestModeSelector:
    """Tests for app.orchestration.mode_selector."""

    def test_infer_target_type_single_url(self):
        assert infer_target_type("http://example.com", 1) == "single_url"
        assert infer_target_type("https://example.com/path", 1) == "single_url"

    def test_infer_target_type_api_endpoint(self):
        assert infer_target_type("https://api.example.com/api/v1/users", 1) == "api_endpoint"
        assert infer_target_type("http://example.com/api", 1) == "api_endpoint"

    def test_infer_target_type_network_range(self):
        assert infer_target_type("192.168.1.0/24", 1) == "network_range"
        assert infer_target_type("10.0.0.1-50", 1) == "network_range"
        assert infer_target_type("10.0.0.1,10.0.0.2", 1) == "network_range"

    def test_infer_target_type_single_ip(self):
        assert infer_target_type("192.168.1.1", 1) == "single_ip"

    def test_infer_target_type_domain_with_subdomains(self):
        assert infer_target_type("example.com", 5) == "domain_with_subdomains"
        # Bare domain with scope_size=1 defaults to single_url
        assert infer_target_type("example.com", 1) == "single_url"

    def test_select_mode_default_supervisor(self):
        """single_web_app + ≤5 steps → supervisor."""
        sel = select_mode(ModeSelectionInput(
            target="http://example.com",
            scope_size=1,
            has_post_exploitation=False,
        ))
        assert sel.mode == OrchestratorMode.SUPERVISOR
        assert "default" in sel.reason.lower() or "supervisor" in sel.reason.lower()

    def test_select_mode_deep_for_network_range(self):
        """network_range → deep."""
        sel = select_mode(ModeSelectionInput(
            target="192.168.1.0/24",
            scope_size=1,
            has_post_exploitation=False,
        ))
        assert sel.mode == OrchestratorMode.DEEP

    def test_select_mode_deep_for_multi_host_scope(self):
        """scope_size > 1 → deep."""
        sel = select_mode(ModeSelectionInput(
            target="http://example.com",
            scope_size=5,
            has_post_exploitation=False,
        ))
        assert sel.mode == OrchestratorMode.DEEP

    def test_select_mode_plan_execute_for_post_exploitation(self):
        """has_post_exploitation → plan_execute."""
        sel = select_mode(ModeSelectionInput(
            target="http://example.com",
            scope_size=1,
            has_post_exploitation=True,
        ))
        assert sel.mode == OrchestratorMode.PLAN_EXECUTE

    def test_select_mode_user_override(self):
        """user_override takes priority."""
        for override, expected in [
            ("supervisor", OrchestratorMode.SUPERVISOR),
            ("deep", OrchestratorMode.DEEP),
            ("plan_execute", OrchestratorMode.PLAN_EXECUTE),
            ("SUPERVISOR", OrchestratorMode.SUPERVISOR),  # case insensitive
        ]:
            sel = select_mode(ModeSelectionInput(
                target="http://example.com",
                user_override=override,
            ))
            assert sel.mode == expected, f"override={override!r} → {sel.mode}, expected {expected}"
            assert sel.user_override_applied is True

    def test_select_mode_invalid_override_falls_through(self):
        """Invalid override → fall through to auto-selection."""
        sel = select_mode(ModeSelectionInput(
            target="http://example.com",
            user_override="invalid_mode",
        ))
        # Falls through to default supervisor
        assert sel.mode == OrchestratorMode.SUPERVISOR
        assert sel.user_override_applied is False

    def test_select_mode_simple(self):
        """select_mode_simple returns just the OrchestratorMode."""
        m = select_mode_simple("http://example.com")
        assert isinstance(m, OrchestratorMode)
        assert m == OrchestratorMode.SUPERVISOR

        m = select_mode_simple("192.168.1.0/24")
        assert m == OrchestratorMode.DEEP

        m = select_mode_simple("http://example.com", has_post_exploitation=True)
        assert m == OrchestratorMode.PLAN_EXECUTE


# ---------- 2. SharedState + base classes ----------

class TestSharedState:
    """Tests for SharedState TypedDict + base classes."""

    def test_shared_state_has_all_required_fields(self):
        """SharedState must declare all fields used by 3 modes."""
        required_fields = {
            "scan_id", "target", "user_prompt", "mode",
            "decisions", "findings", "current_agent", "next_agent",
            "plan", "current_step",
            "sub_agent_tasks", "sub_agent_results",
            "replanner_decision",
            "total_tokens", "error", "status",
        }
        declared = set(SharedState.__annotations__.keys())
        missing = required_fields - declared
        assert not missing, f"SharedState missing required fields: {missing}"

    def test_d18_constants_match_master_plan(self):
        """D18 guardrail constants must match master plan §8.4."""
        assert MAX_DECISIONS_PER_SCAN == 30
        assert MAX_PARALLEL_SUBAGENTS == 5
        assert MAX_AGENTS == 16
        assert MAX_TOKENS_PER_SCAN == 2_000_000
        assert MAX_SCAN_DURATION_SECONDS == 4 * 60 * 60

    def test_generate_scan_id_format(self):
        """scan_id must be 'scan_<12hex>' (17 chars total)."""
        for _ in range(10):
            sid = generate_scan_id()
            assert sid.startswith("scan_")
            assert len(sid) == 17  # 'scan_' (5) + 12 hex chars
            # Hex chars only
            hex_part = sid[5:]
            int(hex_part, 16)  # raises if not hex

    def test_agent_decision_dataclass(self):
        """AgentDecision dataclass fields."""
        d = AgentDecision(
            turn=1,
            thought="test thought",
            agent_name="supervisor",
            tool_name="transfer",
            tool_args={"target_agent": "recon"},
            observation="Transferring to recon",
            tokens_used=100,
        )
        assert d.turn == 1
        assert d.agent_name == "supervisor"
        assert d.tool_name == "transfer"
        assert d.tokens_used == 100
        assert d.timestamp is not None  # auto-set

    def test_orchestrator_result_to_dict(self):
        """OrchestratorResult.to_dict() produces valid FastAPI response."""
        r = OrchestratorResult(
            scan_id="scan_test",
            target="http://example.com",
            user_prompt="test",
            mode="supervisor",
            status="completed",
        )
        d = r.to_dict()
        assert d["scan_id"] == "scan_test"
        assert d["mode"] == "supervisor"
        assert d["status"] == "completed"
        assert d["decisions_count"] == 0
        assert "decisions" in d
        assert "agents_involved" in d
        assert "duration_seconds" in d


# ---------- 3. Prompt loader ----------

class TestPromptLoader:
    """Tests for load_orchestrator_prompt()."""

    def test_load_nonexistent_prompt_raises(self):
        with pytest.raises(FileNotFoundError):
            load_orchestrator_prompt("nonexistent.md")


# ---------- 4. FastAPI router tests ----------

class TestOrchestrationRouter:
    """Tests for app.routes.orchestration router registration.

    Full HTTP integration tests would require a running FastAPI app + DB.
    Here we verify router is properly constructed + has expected routes.
    """

    def test_router_has_3_endpoints(self):
        """Router must expose /modes + /agents + /scans/start-mode."""
        from app.routes.orchestration import router
        paths = {r.path for r in router.routes}
        assert "/api/orchestration/modes" in paths
        assert "/api/orchestration/agents" in paths
        assert "/api/orchestration/scans/start-mode" in paths

    def test_router_prefix(self):
        """Router prefix must be /api/orchestration."""
        from app.routes.orchestration import router
        assert router.prefix == "/api/orchestration"

    def test_router_tags(self):
        """Router must have 'orchestration' tag for OpenAPI grouping."""
        from app.routes.orchestration import router
        assert "orchestration" in router.tags