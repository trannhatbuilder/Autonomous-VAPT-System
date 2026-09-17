"""
W10 unit tests — verify all 16 agents + registry + BaseAgent + tool allowlist.

Test coverage (per master plan §12 W10 acceptance criteria):
    - Each agent invoked independently (no orchestrator needed)
    - Each agent has correct tool allowlist (read_only vs destructive)
    - Max 30 decisions per scan enforced (D18)
    - 16 agents total (3 orchestrators + 13 sub-agents)
    - Tool allowlist enforcement in SubprocessExecutor

Test classes:
    TestAgentRegistry           (10 tests) — registry metadata + lookups
    TestBaseAgent               (8 tests)  — base class + factory + D18
    TestSpecialistAgents        (13 tests) — one per sub-agent
    TestToolAllowlistEnforcement (5 tests) — SubprocessExecutor integration
    TestOrchestrationRouter     (5 tests)  — new W10-S6 endpoints

Run:
    pytest tests/unit/test_agents.py -v
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app.agents.registry import (
    AgentMetadata,
    agent_registry,
    get_agent,
    is_destructive,
    is_tool_allowed,
    list_agents,
    list_orchestrators,
    list_sub_agents,
    load_prompt,
)
from app.agents.base import (
    AgentDecision,
    AgentRunResult,
    BaseAgent,
    create_agent,
    MAX_DECISIONS_PER_AGENT,
    MAX_TOKENS_PER_SCAN,
    MAX_SCAN_DURATION_SECONDS,
)


# ---------- Fixtures ----------

@pytest.fixture
def sample_target() -> str:
    return "http://example.com"


@pytest.fixture
def sample_task() -> str:
    return "Test task for W10"


# ---------- 1. Agent Registry ----------

class TestAgentRegistry:
    """Tests for app.agents.registry."""

    def test_total_count_16(self):
        """W10-S2: registry loads 16 agents (3 orchestrators + 13 sub-agents)."""
        assert agent_registry.total_count == 16
        assert agent_registry.orchestrator_count == 3
        assert agent_registry.sub_agent_count == 13

    def test_orchestrators_marked_correctly(self):
        """3 orchestrators have is_orchestrator=True + orchestration_mode set."""
        orchs = list_orchestrators()
        assert len(orchs) == 3
        modes = {o.orchestration_mode for o in orchs}
        assert modes == {"deep", "plan_execute", "supervisor"}

    def test_safety_class_distribution(self):
        """W10: 6 read_only + 7 destructive + 3 advisory = 16 agents."""
        read_only = agent_registry.list_read_only()
        destructive = agent_registry.list_destructive()
        advisory = agent_registry.list_by_safety_class("advisory")

        assert len(read_only) == 6
        assert len(destructive) == 7
        assert len(advisory) == 3
        assert len(read_only) + len(destructive) + len(advisory) == 16

    def test_known_agents_present(self):
        """All 13 sub-agents from W10 plan are registered."""
        expected_sub_agents = {
            "engagement-planning",
            "intel-collection",
            "recon",
            "attack-surface-enumeration",
            "vulnerability-triage",
            "penetration",
            "privilege-escalation",
            "lateral-movement",
            "persistence-maintenance",
            "impact-exfiltration",
            "opsec-evasion",
            "cleanup-rollback",
            "reporting-remediation",
        }
        actual = {a.name for a in list_sub_agents()}
        missing = expected_sub_agents - actual
        extra = actual - expected_sub_agents
        assert not missing, f"Missing agents: {missing}"
        assert not extra, f"Unexpected agents: {extra}"

    def test_agent_metadata_fields(self):
        """AgentMetadata has all required fields."""
        meta = get_agent("recon")
        assert meta is not None
        assert meta.name == "recon"
        assert meta.display_name  # non-empty
        assert meta.description  # non-empty
        assert meta.safety_class in ("read_only", "destructive", "advisory")
        assert isinstance(meta.tool_allowlist, tuple)
        assert meta.max_iterations <= 30  # D18 cap
        assert meta.prompt_file.endswith(".md")
        assert isinstance(meta.is_orchestrator, bool)

    def test_tool_allowlist_correctness(self):
        """Each agent has expected tool allowlist."""
        cases = [
            ("recon", ("nmap", "httpx", "whatweb", "masscan", "rustscan")),
            ("penetration", ("sqlmap", "metasploit", "mimikatz")),
            ("vulnerability-triage", ("nuclei", "nikto", "dalfox", "wpscan", "fscan")),
            ("lateral-movement", ("netexec", "impacket", "responder")),
            ("privilege-escalation", ("linpeas", "winpeas", "mimikatz")),
            ("engagement-planning", ()),  # no tools
            ("reporting-remediation", ()),  # no tools
        ]
        for agent_name, expected_tools in cases:
            meta = get_agent(agent_name)
            assert meta is not None, f"Agent {agent_name} not found"
            assert meta.tool_allowlist == expected_tools, (
                f"{agent_name}: expected {expected_tools}, got {meta.tool_allowlist}"
            )

    def test_is_tool_allowed_matrix(self):
        """Tool allowlist validation across agents."""
        cases = [
            # (agent_name, tool_name, expected_allowed)
            ("recon", "nmap", True),
            ("recon", "sqlmap", False),
            ("recon", "metasploit", False),
            ("penetration", "sqlmap", True),
            ("penetration", "nmap", False),
            ("vulnerability-triage", "nuclei", True),
            ("vulnerability-triage", "metasploit", False),
            ("engagement-planning", "nmap", False),
            ("cleanup-rollback", "nmap", False),
        ]
        for agent_name, tool_name, expected in cases:
            actual = is_tool_allowed(agent_name, tool_name)
            assert actual == expected, (
                f"is_tool_allowed({agent_name!r}, {tool_name!r}) = {actual}, expected {expected}"
            )

    def test_is_destructive_flag(self):
        """Destructive agents correctly flagged for HITL requirement."""
        destructive_agents = {"penetration", "privilege-escalation", "lateral-movement",
                              "persistence-maintenance", "impact-exfiltration",
                              "opsec-evasion", "cleanup-rollback"}
        for agent_name in destructive_agents:
            assert is_destructive(agent_name), f"{agent_name} should be destructive"
        # Spot-check read-only agents
        for agent_name in ("recon", "vulnerability-triage", "reporting-remediation"):
            assert not is_destructive(agent_name), f"{agent_name} should NOT be destructive"

    def test_load_prompt_returns_content(self):
        """load_prompt returns non-empty .md content."""
        for name in ("recon", "penetration", "orchestrator-supervisor"):
            content = load_prompt(name)
            assert len(content) > 1000
            assert "VAPT-AI" in content

    def test_unknown_agent_returns_none(self):
        """get_agent returns None for unknown agent (no exception)."""
        assert get_agent("nonexistent-agent") is None

    def test_to_dict_serialization(self):
        """AgentMetadata.to_dict() produces valid JSON-serializable dict."""
        meta = get_agent("recon")
        d = meta.to_dict()
        assert d["name"] == "recon"
        assert isinstance(d["tool_allowlist"], list)  # tuple → list for JSON
        assert d["safety_class"] == "read_only"
        assert d["is_orchestrator"] is False


# ---------- 2. BaseAgent + factory ----------

class TestBaseAgent:
    """Tests for app.agents.base."""

    def test_d18_constants_match_master_plan(self):
        """D18 constants match master plan §8.4."""
        assert MAX_DECISIONS_PER_AGENT == 30
        assert MAX_TOKENS_PER_SCAN == 2_000_000
        assert MAX_SCAN_DURATION_SECONDS == 4 * 60 * 60

    def test_agent_decision_dataclass(self):
        """AgentDecision dataclass fields."""
        d = AgentDecision(
            turn=1, agent_name="recon", thought="test",
            tool_name="nmap", tool_args={"target": "x"},
        )
        assert d.turn == 1
        assert d.agent_name == "recon"
        assert d.tokens_used == 0  # default
        assert d.timestamp is not None

    def test_agent_run_result_to_dict(self):
        """AgentRunResult.to_dict() produces valid dict."""
        r = AgentRunResult(
            agent_name="recon",
            scan_id="scan_test",
            target="http://example.com",
            task_description="test task",
            status="completed",
        )
        d = r.to_dict()
        assert d["agent_name"] == "recon"
        assert d["status"] == "completed"
        assert d["decisions_count"] == 0

    def test_create_agent_factory_for_sub_agent(self, sample_target, sample_task):
        """create_agent() returns correct subclass instance."""
        agent = create_agent(
            agent_name="recon",
            scan_id="scan_factory_test",
            target=sample_target,
            task_description=sample_task,
        )
        assert isinstance(agent, BaseAgent)
        assert agent.name == "recon"
        assert agent.safety_class == "read_only"
        assert "nmap" in agent.tool_allowlist

    def test_create_agent_factory_for_orchestrator_fails(self, sample_target):
        """create_agent() with orchestrator name raises (orchestrators can't be invoked directly).

        Actually the factory doesn't check — but the resulting agent will be a BaseAgent
        without a proper specialist class, so we expect ImportError.
        """
        with pytest.raises((ImportError, KeyError)):
            create_agent(
                agent_name="orchestrator-supervisor",
                scan_id="scan_test_orch",
                target=sample_target,
            )

    def test_create_agent_factory_unknown_agent_raises(self, sample_target):
        """create_agent() with unknown agent raises KeyError."""
        with pytest.raises(KeyError):
            create_agent(
                agent_name="nonexistent",
                scan_id="scan_test_unknown",
                target=sample_target,
            )

    def test_base_agent_max_iterations_capped(self, sample_target):
        """max_iterations is capped at D18 limit (30)."""
        agent = create_agent(
            agent_name="recon",
            scan_id="scan_max_iter",
            target=sample_target,
            max_iterations=100,  # exceeds D18 cap
        )
        assert agent.max_iterations == MAX_DECISIONS_PER_AGENT

    def test_base_agent_system_prompt_loaded(self, sample_target):
        """system_prompt is loaded from .md file at __init__."""
        agent = create_agent(
            agent_name="penetration",
            scan_id="scan_prompt_test",
            target=sample_target,
        )
        assert len(agent.system_prompt) > 1000
        assert "VAPT-AI" in agent.system_prompt
        assert "Penetration Specialist" in agent.system_prompt


# ---------- 3. Specialist agents (one test per sub-agent) ----------

class TestSpecialistAgents:
    """Each of the 13 sub-agents can be invoked independently."""

    SUB_AGENT_NAMES = [
        "engagement-planning",
        "intel-collection",
        "recon",
        "attack-surface-enumeration",
        "vulnerability-triage",
        "penetration",
        "privilege-escalation",
        "lateral-movement",
        "persistence-maintenance",
        "impact-exfiltration",
        "opsec-evasion",
        "cleanup-rollback",
        "reporting-remediation",
    ]

    @pytest.mark.parametrize("agent_name", SUB_AGENT_NAMES)
    @pytest.mark.asyncio
    async def test_agent_runs_to_completion(self, agent_name, sample_target):
        """Each sub-agent runs + returns status='completed'."""
        agent = create_agent(
            agent_name=agent_name,
            scan_id=f"scan_test_{agent_name.replace('-', '_')}",
            target=sample_target,
            task_description=f"W10 test: {agent_name}",
        )
        result = await agent.run()
        assert result.status == "completed", (
            f"{agent_name} failed: status={result.status} error={result.error}"
        )
        assert len(result.decisions) == 1  # W10 stub: 1 decision per run
        assert result.agent_name == agent_name
        assert result.error is None

    @pytest.mark.parametrize("agent_name", SUB_AGENT_NAMES)
    @pytest.mark.asyncio
    async def test_agent_decision_has_correct_metadata(self, agent_name, sample_target):
        """Each agent's decision has correct agent_name + tool from allowlist."""
        agent = create_agent(
            agent_name=agent_name,
            scan_id=f"scan_meta_{agent_name.replace('-', '_')}",
            target=sample_target,
        )
        result = await agent.run()
        d = result.decisions[0]
        assert d.agent_name == agent_name
        # Tool_name (if any) must be in agent's allowlist
        if d.tool_name is not None:
            assert d.tool_name in agent.tool_allowlist, (
                f"{agent_name} returned tool {d.tool_name!r} not in its allowlist "
                f"{agent.tool_allowlist}"
            )

    @pytest.mark.asyncio
    async def test_destructive_agent_metadata_set(self, sample_target):
        """Destructive agents have is_destructive=True."""
        for agent_name in ("penetration", "privilege-escalation", "lateral-movement"):
            agent = create_agent(
                agent_name=agent_name,
                scan_id=f"scan_destr_{agent_name}",
                target=sample_target,
            )
            assert agent.is_destructive is True
            assert agent.safety_class == "destructive"

    @pytest.mark.asyncio
    async def test_read_only_agent_metadata_set(self, sample_target):
        """Read-only agents have is_destructive=False."""
        for agent_name in ("recon", "vulnerability-triage", "reporting-remediation"):
            agent = create_agent(
                agent_name=agent_name,
                scan_id=f"scan_ro_{agent_name}",
                target=sample_target,
            )
            assert agent.is_destructive is False
            assert agent.safety_class == "read_only"


# ---------- 4. Tool allowlist enforcement in SubprocessExecutor ----------

class TestToolAllowlistEnforcement:
    """Tests for W10-S7: tool allowlist enforcement in SubprocessExecutor."""

    @pytest.mark.asyncio
    async def test_agent_name_none_skips_allowlist_check(self):
        """When agent_name=None, allowlist check is skipped (W9 backward compat)."""
        from app.sandbox.executor import SubprocessExecutor
        executor = SubprocessExecutor()
        # Use a fake command — will fail due to scope guard, but NOT due to allowlist
        result = await executor.execute(
            command=["nonexistent-binary", "--version"],
            target="127.0.0.1",  # triggers SSRF guard, not allowlist
            timeout=2,
            agent_name=None,
        )
        # Should NOT have TOOL_ALLOWLIST_VIOLATION
        assert not (result.error and "TOOL_ALLOWLIST_VIOLATION" in str(result.error))

    @pytest.mark.asyncio
    async def test_allowed_tool_passes_allowlist_check(self):
        """When agent_name='recon' + binary='nmap' (in allowlist), no violation."""
        from app.sandbox.executor import SubprocessExecutor
        executor = SubprocessExecutor()
        result = await executor.execute(
            command=["nmap", "-V"],  # version flag
            target="127.0.0.1",
            timeout=2,
            agent_name="recon",
        )
        # Should NOT have TOOL_ALLOWLIST_VIOLATION (might fail for other reasons like scope)
        assert not (result.error and "TOOL_ALLOWLIST_VIOLATION" in str(result.error))

    @pytest.mark.asyncio
    async def test_disallowed_tool_triggers_violation(self):
        """When agent_name='recon' + binary='sqlmap' (NOT in allowlist), violation."""
        from app.sandbox.executor import SubprocessExecutor
        executor = SubprocessExecutor()
        result = await executor.execute(
            command=["sqlmap", "--version"],
            target="127.0.0.1",
            timeout=2,
            agent_name="recon",  # sqlmap NOT in recon's allowlist
        )
        assert result.success is False
        assert result.error is not None
        assert "TOOL_ALLOWLIST_VIOLATION" in str(result.error)
        assert "sqlmap" in str(result.error)
        assert "recon" in str(result.error)

    @pytest.mark.asyncio
    async def test_unknown_agent_triggers_violation(self):
        """When agent_name is unknown, validate_tool_call returns (False, 'agent not found')."""
        from app.sandbox.executor import SubprocessExecutor
        executor = SubprocessExecutor()
        result = await executor.execute(
            command=["nmap", "-V"],
            target="127.0.0.1",
            timeout=2,
            agent_name="nonexistent-agent",
        )
        assert result.success is False
        assert "TOOL_ALLOWLIST_VIOLATION" in str(result.error)
        assert "agent not found" in str(result.error)

    @pytest.mark.asyncio
    async def test_destructive_agent_tool_still_passes_allowlist(self):
        """Destructive agents can use their allowed tools (HITL gate is separate)."""
        from app.sandbox.executor import SubprocessExecutor
        executor = SubprocessExecutor()
        # penetration agent can use sqlmap (in allowlist)
        # HITL gate is a separate layer (W12) — W10 only enforces allowlist
        result = await executor.execute(
            command=["sqlmap", "--version"],
            target="127.0.0.1",
            timeout=2,
            agent_name="penetration",
        )
        # Should NOT have TOOL_ALLOWLIST_VIOLATION (sqlmap IS in penetration's allowlist)
        assert not (result.error and "TOOL_ALLOWLIST_VIOLATION" in str(result.error))


# ---------- 5. Orchestration router (W10-S6 new endpoints) ----------

class TestOrchestrationRouter:
    """Tests for new W10-S6 endpoints in app.routes.orchestration."""

    def test_router_has_5_endpoints(self):
        """Router now exposes 5 endpoints (W9 had 3, W10 added 2)."""
        from app.routes.orchestration import router
        paths = {r.path for r in router.routes}
        assert "/api/orchestration/modes" in paths
        assert "/api/orchestration/agents" in paths
        assert "/api/orchestration/agents/{name}" in paths  # W10-S6 NEW
        assert "/api/orchestration/agents/{name}/invoke" in paths  # W10-S6 NEW
        assert "/api/orchestration/scans/start-mode" in paths

    def test_router_prefix(self):
        """Router prefix is /api/orchestration."""
        from app.routes.orchestration import router
        assert router.prefix == "/api/orchestration"

    def test_invoke_request_model_validates_target(self):
        """InvokeAgentRequest requires target field."""
        from app.routes.orchestration import InvokeAgentRequest
        from pydantic import ValidationError

        # Missing target should fail
        with pytest.raises(ValidationError):
            InvokeAgentRequest()

        # With target should pass
        req = InvokeAgentRequest(target="http://example.com")
        assert req.target == "http://example.com"
        assert req.task_description == ""
        assert req.scan_id is None
        assert req.max_iterations is None

    def test_invoke_request_max_iterations_capped(self):
        """InvokeAgentRequest rejects max_iterations > 30 (D18 cap)."""
        from app.routes.orchestration import InvokeAgentRequest
        from pydantic import ValidationError

        # 30 is OK (D18 cap)
        req = InvokeAgentRequest(target="http://example.com", max_iterations=30)
        assert req.max_iterations == 30

        # 31 is rejected
        with pytest.raises(ValidationError):
            InvokeAgentRequest(target="http://example.com", max_iterations=31)

    def test_start_mode_scan_request_accepts_transfer_targets(self):
        """StartModeScanRequest accepts optional transfer_targets (W10-S5)."""
        from app.routes.orchestration import StartModeScanRequest

        req = StartModeScanRequest(
            target="http://example.com",
            mode="supervisor",
            transfer_targets=["recon", "vulnerability-triage", "reporting-remediation"],
        )
        assert req.transfer_targets == ["recon", "vulnerability-triage", "reporting-remediation"]

        # Default is None
        req2 = StartModeScanRequest(target="http://example.com")
        assert req2.transfer_targets is None