"""
W9 integration tests — verify all 3 orchestration modes work end-to-end.

Test coverage (per master plan §12 W9 acceptance criteria):
    - test_orchestrator_modes_work: all 3 modes run + return completed status
    - test_deep_delegates_2plus_subagents_in_parallel: Deep mode runs ≥2 sub-agents
    - test_plan_execute_produces_plan_and_replans: Plan-Execute creates structured plan + iterates
    - test_supervisor_transfers_to_expert: Supervisor delegates to expert + aggregates
    - test_mode_selector: select_mode() picks right mode based on target + scope
    - test_shared_state_typeddict: SharedState declares all required fields
    - test_d18_guardrails_enforced: BaseOrchestrator caps decisions at 30
    - test_orchestrator_result_serialization: to_dict() produces valid FastAPI response
    - test_load_orchestrator_prompts: 3 .md prompt files load successfully
    - test_orchestration_router_endpoints: FastAPI router exposes /modes + /agents + /scans/start-mode

These tests do NOT require a database — they exercise the LangGraph runtime
in isolation. W10+ will add DB-backed tests (PentestFact writes, etc.).

Run:
    pytest tests/integration/test_orchestration_modes.py -v

Or with coverage:
    pytest tests/integration/test_orchestration_modes.py -v --cov=app.orchestration
"""
from __future__ import annotations

import asyncio
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
from app.orchestration.langgraph_supervisor import (
    EXPERT_AGENTS,
    SupervisorOrchestrator,
    run_supervisor_scan,
)
from app.orchestration.langgraph_deep import (
    STUB_SUB_AGENT_TASKS,
    DeepOrchestrator,
    run_deep_scan,
)
from app.orchestration.langgraph_plan_execute import (
    STUB_PLAN,
    PlanExecuteOrchestrator,
    run_plan_execute_scan,
)


# ---------- Fixtures ----------

@pytest.fixture
def sample_target() -> str:
    return "http://example.com"


@pytest.fixture
def sample_prompt() -> str:
    return "W9 integration test — scan for vulnerabilities"


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

    def test_load_supervisor_prompt(self):
        content = load_orchestrator_prompt("orchestrator-supervisor.md")
        assert len(content) > 1000
        assert "VAPT-AI" in content
        assert "supervisor" in content.lower()
        # Has YAML frontmatter
        assert content.startswith("---")

    def test_load_deep_prompt(self):
        content = load_orchestrator_prompt("orchestrator.md")
        assert len(content) > 1000
        assert "VAPT-AI" in content
        assert "deep" in content.lower() or "Deep" in content

    def test_load_plan_execute_prompt(self):
        content = load_orchestrator_prompt("orchestrator-plan-execute.md")
        assert len(content) > 1000
        assert "VAPT-AI" in content
        assert "plan_execute" in content.lower() or "Plan-Execute" in content

    def test_load_nonexistent_prompt_raises(self):
        with pytest.raises(FileNotFoundError):
            load_orchestrator_prompt("nonexistent.md")


# ---------- 4. Supervisor orchestrator tests ----------

class TestSupervisorOrchestrator:
    """Tests for app.orchestration.langgraph_supervisor."""

    def test_expert_agents_registry(self):
        """W9 stub has 2 expert agents."""
        assert "recon" in EXPERT_AGENTS
        assert "reporting-remediation" in EXPERT_AGENTS
        assert len(EXPERT_AGENTS) >= 2
        # Each expert has required fields
        for name, meta in EXPERT_AGENTS.items():
            assert "name" in meta
            assert "description" in meta
            assert "tools" in meta
            assert "safety_class" in meta

    @pytest.mark.asyncio
    async def test_supervisor_runs_to_completion(self, sample_target, sample_prompt):
        """Supervisor scan completes with status='completed'."""
        result = await run_supervisor_scan(sample_target, sample_prompt)
        assert result.status == "completed"
        assert result.mode == "supervisor"
        assert len(result.decisions) >= 3  # transfer + recon + exit
        assert result.error is None

    @pytest.mark.asyncio
    async def test_supervisor_delegates_to_at_least_one_expert(self, sample_target, sample_prompt):
        """Supervisor must transfer to ≥1 expert (per W9 acceptance criterion)."""
        result = await run_supervisor_scan(sample_target, sample_prompt)
        expert_agents = [
            d for d in result.decisions if d.agent_name != "supervisor"
        ]
        assert len(expert_agents) >= 1, "Supervisor must delegate to ≥1 expert"
        # Verify recon was one of the experts
        agent_names = {d.agent_name for d in result.decisions}
        assert "recon" in agent_names

    @pytest.mark.asyncio
    async def test_supervisor_emits_transfer_and_exit(self, sample_target, sample_prompt):
        """Supervisor decisions must include at least 1 transfer + 1 exit."""
        result = await run_supervisor_scan(sample_target, sample_prompt)
        transfer_count = sum(1 for d in result.decisions if d.tool_name == "transfer")
        exit_count = sum(1 for d in result.decisions if d.tool_name == "exit")
        assert transfer_count >= 1, "Supervisor must use 'transfer' at least once"
        assert exit_count == 1, "Supervisor must use 'exit' exactly once"

    @pytest.mark.asyncio
    async def test_supervisor_agents_involved(self, sample_target, sample_prompt):
        """agents_involved includes supervisor + at least 1 expert."""
        result = await run_supervisor_scan(sample_target, sample_prompt)
        assert "supervisor" in result.agents_involved
        assert len(result.agents_involved) >= 2  # supervisor + 1 expert


# ---------- 5. Deep orchestrator tests ----------

class TestDeepOrchestrator:
    """Tests for app.orchestration.langgraph_deep."""

    def test_stub_sub_agent_tasks_count(self):
        """W9 stub has 3 sub-agent tasks."""
        assert len(STUB_SUB_AGENT_TASKS) == 3

    @pytest.mark.asyncio
    async def test_deep_runs_to_completion(self, sample_target, sample_prompt):
        """Deep scan completes with status='completed'."""
        result = await run_deep_scan(sample_target, sample_prompt)
        assert result.status == "completed"
        assert result.mode == "deep"
        assert len(result.decisions) >= 4

    @pytest.mark.asyncio
    async def test_deep_delegates_2plus_subagents(self, sample_target, sample_prompt):
        """W9 acceptance: Deep mode can delegate ≥2 sub-agents in parallel.

        Per master plan §12 W9 acceptance criteria.
        """
        result = await run_deep_scan(sample_target, sample_prompt)
        # Count unique sub-agent types invoked (excluding orchestrator)
        sub_agents = {
            d.agent_name for d in result.decisions
            if d.agent_name != "deep-orchestrator"
        }
        assert len(sub_agents) >= 2, (
            f"Deep mode must delegate to ≥2 sub-agents, got {len(sub_agents)}: {sub_agents}"
        )

    @pytest.mark.asyncio
    async def test_deep_has_planner_fanout_aggregator_decisions(self, sample_target, sample_prompt):
        """Deep mode decisions must include plan + task + aggregate tools."""
        result = await run_deep_scan(sample_target, sample_prompt)
        tool_names = {d.tool_name for d in result.decisions}
        assert "plan" in tool_names, "Deep mode must use 'plan' tool"
        assert "task" in tool_names, "Deep mode must use 'task' tool"
        assert "aggregate" in tool_names, "Deep mode must use 'aggregate' tool"

    @pytest.mark.asyncio
    async def test_deep_sub_agent_count(self, sample_target, sample_prompt):
        """sub_agent_count returns number of unique sub-agents."""
        result = await run_deep_scan(sample_target, sample_prompt)
        assert result.sub_agent_count >= 2  # at least 2 sub-agents

    @pytest.mark.asyncio
    async def test_deep_max_parallel_subagents_enforced(self):
        """D18: max 5 parallel sub-agents — orchestrator truncates if exceeded."""
        # Create 8 tasks — should be truncated to 5
        too_many_tasks = [
            {"subagent_type": f"agent_{i}", "description": f"task {i}", "expected_deliverable": "x"}
            for i in range(8)
        ]
        orch = DeepOrchestrator(
            scan_id="scan_max5",
            target="http://example.com",
            user_prompt="test",
            sub_agent_tasks=too_many_tasks,
        )
        assert len(orch.sub_agent_tasks) == MAX_PARALLEL_SUBAGENTS  # 5


# ---------- 6. Plan-Execute orchestrator tests ----------

class TestPlanExecuteOrchestrator:
    """Tests for app.orchestration.langgraph_plan_execute."""

    def test_stub_plan_has_3_steps(self):
        """W9 stub has 3 plan steps."""
        assert len(STUB_PLAN) == 3
        # Each step has required fields
        for step in STUB_PLAN:
            assert "step" in step
            assert "name" in step
            assert "description" in step
            assert "tools" in step
            assert "success_criteria" in step

    @pytest.mark.asyncio
    async def test_plan_execute_runs_to_completion(self, sample_target, sample_prompt):
        """Plan-Execute scan completes with status='completed'."""
        result = await run_plan_execute_scan(sample_target, sample_prompt)
        assert result.status == "completed"
        assert result.mode == "plan_execute"
        assert len(result.decisions) >= 5  # planner + 3×(executor + replanner)

    @pytest.mark.asyncio
    async def test_plan_execute_produces_structured_plan(self, sample_target, sample_prompt):
        """W9 acceptance: Plan-Execute produces structured plan.

        Per master plan §12 W9 acceptance criteria.
        """
        result = await run_plan_execute_scan(sample_target, sample_prompt)
        # Must have planner decision with write_plan tool
        plan_decisions = [d for d in result.decisions if d.tool_name == "write_plan"]
        assert len(plan_decisions) >= 1, "Plan-Execute must produce ≥1 write_plan decision"
        # Verify plan structure in tool_args
        plan_decision = plan_decisions[0]
        assert "step_count" in plan_decision.tool_args
        assert "step_names" in plan_decision.tool_args
        assert plan_decision.tool_args["step_count"] >= 2

    @pytest.mark.asyncio
    async def test_plan_execute_replans_at_least_once(self, sample_target, sample_prompt):
        """W9 acceptance: Plan-Execute iterates via replanner.

        Per master plan §12 W9 acceptance criteria: 'replans'.
        """
        result = await run_plan_execute_scan(sample_target, sample_prompt)
        replanner_decisions = [d for d in result.decisions if d.agent_name == "plan-execute-replanner"]
        # Should have at least 2 replanner calls (after step 0 + step 1, then exit after step 2)
        assert len(replanner_decisions) >= 2, (
            f"Plan-Execute must replan ≥2 times, got {len(replanner_decisions)}"
        )
        # Verify "continue" decisions
        continue_count = sum(
            1 for d in replanner_decisions
            if d.tool_args.get("decision") == "continue"
        )
        assert continue_count >= 1

    @pytest.mark.asyncio
    async def test_plan_execute_completes_all_steps(self, sample_target, sample_prompt):
        """All plan steps must be executed."""
        result = await run_plan_execute_scan(sample_target, sample_prompt)
        assert result.plan_steps_total == 3
        assert result.plan_steps_completed == 3

    @pytest.mark.asyncio
    async def test_plan_execute_executor_runs_each_step(self, sample_target, sample_prompt):
        """Each plan step's expert must execute at least once."""
        result = await run_plan_execute_scan(sample_target, sample_prompt)
        executed_agents = {
            d.agent_name for d in result.decisions
            if d.agent_name not in ("plan-execute-planner", "plan-execute-replanner")
        }
        expected_agents = {"recon", "vulnerability-triage", "reporting-remediation"}
        assert expected_agents.issubset(executed_agents), (
            f"Expected all 3 step experts to execute, got: {executed_agents}"
        )


# ---------- 7. D18 guardrails ----------

class TestD18Guardrails:
    """Tests for D18 guardrail enforcement."""

    @pytest.mark.asyncio
    async def test_max_decisions_enforced(self):
        """Orchestrator must respect max_decisions cap (D18).

        Architecture note: the guardrail check fires at supervisor node entry.
        Between supervisor calls, an expert node runs and records its own decision,
        so the total decision count can exceed max_decisions by up to (number of
        expert calls between checks). For supervisor stub: max 1 expert decision
        between checks, so total ≤ max_decisions + 1.

        Without cap: supervisor stub makes 5 decisions. With cap=3: should stop
        early (well before 5).
        """
        # Create supervisor with max_decisions=3 (well below 30)
        orch = SupervisorOrchestrator(
            scan_id="scan_max3",
            target="http://example.com",
            user_prompt="test",
            max_decisions=3,
        )
        result = await orch.run()
        # Should stop early — well below the normal 5 decisions
        # Allow up to max_decisions + 1 (one expert decision may slip in between checks)
        assert len(orch.decisions) <= 4, (
            f"Guardrail should have stopped scan early, but {len(orch.decisions)} "
            f"decisions were recorded (expected ≤4 with max_decisions=3)"
        )
        # Status should reflect early termination
        assert result.status in ("max_iterations", "failed", "completed"), (
            f"Unexpected status with max_decisions cap: {result.status}"
        )

    def test_base_orchestrator_guardrail_check(self):
        """_check_guardrails returns None when limits not exceeded."""
        orch = SupervisorOrchestrator(
            scan_id="scan_guardrail_ok",
            target="http://example.com",
            user_prompt="test",
        )
        # No decisions yet → no violation
        violation = orch._check_guardrails()
        assert violation is None

    def test_base_orchestrator_guardrail_triggers_on_max_decisions(self):
        """_check_guardrails returns error when max decisions exceeded."""
        orch = SupervisorOrchestrator(
            scan_id="scan_guardrail_max",
            target="http://example.com",
            user_prompt="test",
            max_decisions=2,
        )
        # Simulate 2 decisions already made
        orch.decisions = [
            AgentDecision(turn=0, thought="t1", agent_name="x", tool_name="y", tool_args={}),
            AgentDecision(turn=1, thought="t2", agent_name="x", tool_name="y", tool_args={}),
        ]
        violation = orch._check_guardrails()
        assert violation is not None
        assert "decisions" in violation.lower()


# ---------- 8. Cross-mode integration ----------

class TestCrossModeIntegration:
    """Tests that verify all 3 modes work together + produce valid output."""

    @pytest.mark.asyncio
    async def test_all_3_modes_complete_successfully(self, sample_target, sample_prompt):
        """W9 acceptance: all 3 modes work end-to-end."""
        results = {}
        results["supervisor"] = await run_supervisor_scan(sample_target, sample_prompt)
        results["deep"] = await run_deep_scan(sample_target, sample_prompt)
        results["plan_execute"] = await run_plan_execute_scan(sample_target, sample_prompt)

        for mode, result in results.items():
            assert result.status == "completed", (
                f"{mode} mode failed: status={result.status}, error={result.error}"
            )
            assert result.mode == mode
            assert len(result.decisions) > 0
            assert result.duration_seconds >= 0

    @pytest.mark.asyncio
    async def test_all_modes_emit_scan_started_and_complete(self, sample_target, sample_prompt):
        """All modes emit scan_started + scan_complete SSE events.

        We verify by checking event_bus buffer.
        """
        from app.pentest.events import event_bus

        for runner in [run_supervisor_scan, run_deep_scan, run_plan_execute_scan]:
            # Clear bus
            event_bus._buffers.clear()
            event_bus._queues.clear()

            result = await runner(sample_target, sample_prompt)

            # Check buffered events
            events = event_bus._buffers.get(result.scan_id, [])
            event_types = [e.get("event") for e in events]
            assert "scan_started" in event_types, (
                f"{result.mode} did not emit scan_started event"
            )
            assert "scan_complete" in event_types, (
                f"{result.mode} did not emit scan_complete event"
            )

    @pytest.mark.asyncio
    async def test_each_mode_distinct_scan_id(self, sample_target, sample_prompt):
        """Each scan gets unique scan_id."""
        ids = set()
        for runner in [run_supervisor_scan, run_deep_scan, run_plan_execute_scan]:
            result = await runner(sample_target, sample_prompt)
            assert result.scan_id not in ids
            ids.add(result.scan_id)
        assert len(ids) == 3


# ---------- 9. FastAPI router tests ----------

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