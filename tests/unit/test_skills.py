"""
W11 unit tests — verify all 24 skills + loader + agent_mapping + auto-load.

Test coverage (per master plan §12 W11 acceptance criteria):
    - All 24 skills loadable
    - Skill → agent mapping correct
    - Relevant skills auto-loaded per agent role (BaseAgent.loaded_skills)
    - Skill content visible via API (4 new endpoints)

Test classes:
    TestSkillManifest          (8 tests)  — SkillManifest dataclass + parsing
    TestSkillLoader            (10 tests) — SkillLoader singleton + lazy-load + cache
    TestAgentSkillMapping      (8 tests)  — skill → agent mapping table
    TestBaseAgentSkills        (8 tests)  — BaseAgent auto-load + properties
    TestSkillsRouter           (5 tests)  — 4 new endpoints + 9 total
    TestSkillsContent          (5 tests)  — spot-check skill body content

Run:
    pytest tests/unit/test_skills.py -v
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app.skills import (
    Skill,
    SkillLoader,
    SkillManifest,
    skill_loader,
    get_skills_for_agent,
    get_agents_for_skill,
    get_mapping_stats,
    reload_maps,
)
from app.skills.base import parse_skill_md
from app.agents.base import create_agent, BaseAgent


# ---------- Fixtures ----------

@pytest.fixture
def sample_target() -> str:
    return "http://example.com"


@pytest.fixture(autouse=True)
def _reload_skills():
    """Reload skills package + maps before each test for isolation."""
    reload_maps()
    yield
    reload_maps()


# ---------- 1. SkillManifest ----------

class TestSkillManifest:
    """Tests for app.skills.base.SkillManifest + parse_skill_md."""

    def test_total_skills_loaded(self):
        """W11-S1: 24 skills loaded from app/skills/."""
        assert skill_loader.total_count == 24

    def test_parse_skill_md_returns_manifest_and_body(self):
        """parse_skill_md returns (SkillManifest, body) tuple."""
        content = """\
---
name: test-skill
description: Test skill for parsing
license: mit
compatibility: ">=3.0"
metadata:
  version: "1.0.0"
  tags: ["test", "demo"]
  triggers: ["test trigger"]
allowed-tools: "nmap,nuclei"
mapped-agents: ["recon"]
---

# Test Skill

This is a test body.
"""
        manifest, body = parse_skill_md(content)
        assert manifest.name == "test-skill"
        assert manifest.description == "Test skill for parsing"
        assert manifest.license == "mit"
        assert manifest.compatibility == ">=3.0"
        assert manifest.version == "1.0.0"
        assert manifest.tags == ["test", "demo"]
        assert manifest.triggers == ["test trigger"]
        assert manifest.allowed_tools == "nmap,nuclei"
        assert manifest.allowed_tools_list == ["nmap", "nuclei"]
        assert manifest.mapped_agents == ("recon",)
        assert "Test Skill" in body

    def test_parse_skill_md_missing_frontmatter_raises(self):
        """Missing frontmatter raises ValueError."""
        with pytest.raises(ValueError, match="front matter"):
            parse_skill_md("# No frontmatter\n\nBody only")

    def test_parse_skill_md_empty_raises(self):
        """Empty content raises ValueError."""
        with pytest.raises(ValueError, match="empty"):
            parse_skill_md("")

    def test_parse_skill_md_missing_name_raises(self):
        """Missing name field raises ValueError."""
        content = """\
---
description: Has description but no name
---

Body
"""
        with pytest.raises(ValueError, match="name"):
            parse_skill_md(content)

    def test_skill_manifest_to_dict(self):
        """to_dict() produces valid JSON-serializable dict."""
        m = skill_loader.get_manifest("web-attack-methods")
        assert m is not None
        d = m.to_dict()
        assert d["name"] == "web-attack-methods"
        assert isinstance(d["tags"], list)
        assert isinstance(d["triggers"], list)
        assert isinstance(d["mapped_agents"], list)
        assert "version" in d

    def test_skill_manifest_allowed_tools_list(self):
        """allowed_tools_list parses comma-separated string."""
        m = skill_loader.get_manifest("web-attack-methods")
        assert m is not None
        assert isinstance(m.allowed_tools_list, list)
        # Should contain at least one tool
        assert len(m.allowed_tools_list) > 0

    def test_skill_class_to_dict(self):
        """Skill.to_dict() includes body when include_body=True."""
        skill = skill_loader.load_skill("web-attack-methods")
        d_no_body = skill.to_dict(include_body=False)
        assert "body" not in d_no_body
        assert d_no_body["body_length"] > 0

        d_with_body = skill.to_dict(include_body=True)
        assert "body" in d_with_body
        assert len(d_with_body["body"]) > 100


# ---------- 2. SkillLoader ----------

class TestSkillLoader:
    """Tests for app.skills.loader.SkillLoader."""

    def test_total_count_24(self):
        """Loader reports 24 skills."""
        assert skill_loader.total_count == 24

    def test_list_skills_returns_24_manifests(self):
        """list_skills() returns 24 SkillManifest instances."""
        skills = skill_loader.list_skills()
        assert len(skills) == 24
        # All should be SkillManifest instances
        for s in skills:
            assert isinstance(s, SkillManifest)
            assert s.name  # non-empty
            assert s.description  # non-empty

    def test_list_skill_names_sorted(self):
        """list_skill_names() returns sorted list."""
        names = skill_loader.list_skill_names()
        assert len(names) == 24
        assert names == sorted(names)

    def test_get_manifest_known_skill(self):
        """get_manifest returns manifest for known skill."""
        m = skill_loader.get_manifest("web-attack-methods")
        assert m is not None
        assert m.name == "web-attack-methods"

    def test_get_manifest_unknown_returns_none(self):
        """get_manifest returns None for unknown skill."""
        assert skill_loader.get_manifest("nonexistent-skill") is None

    def test_require_manifest_raises_for_unknown(self):
        """require_manifest raises KeyError for unknown skill."""
        with pytest.raises(KeyError):
            skill_loader.require_manifest("nonexistent-skill")

    def test_load_skill_returns_skill_with_body(self):
        """load_skill returns Skill instance with body loaded."""
        skill = skill_loader.load_skill("web-attack-methods")
        assert isinstance(skill, Skill)
        assert skill.name == "web-attack-methods"
        assert skill.has_body
        assert len(skill.body) > 500

    def test_load_skill_body_caches(self):
        """load_skill_body caches body for repeated calls."""
        body1 = skill_loader.load_skill_body("attack-surface-recon")
        body2 = skill_loader.load_skill_body("attack-surface-recon")
        assert body1 == body2

    def test_search_by_tag(self):
        """search_by_tag returns matching skills."""
        web_skills = skill_loader.search_by_tag("web")
        assert len(web_skills) > 0
        for s in web_skills:
            assert "web" in [t.lower() for t in s.tags]

    def test_search_by_trigger(self):
        """search_by_trigger matches keyword in triggers."""
        sqli_skills = skill_loader.search_by_trigger("sql injection")
        assert len(sqli_skills) > 0
        # web-attack-methods should be among the matches
        names = [s.name for s in sqli_skills]
        assert "web-attack-methods" in names


# ---------- 3. Agent → Skill Mapping ----------

class TestAgentSkillMapping:
    """Tests for app.skills.agent_mapping."""

    def test_all_13_sub_agents_have_skills(self):
        """W11-S4: All 13 sub-agents have at least 1 mapped skill."""
        from app.agents.registry import list_sub_agents
        sub_agents = list_sub_agents()
        assert len(sub_agents) == 13

        for agent in sub_agents:
            skills = get_skills_for_agent(agent.name)
            assert len(skills) >= 1, (
                f"Agent {agent.name!r} has no mapped skills"
            )

    def test_orchestrators_have_no_skills(self):
        """W11-S4: Orchestrators (advisory) have no auto-loaded skills."""
        from app.agents.registry import list_orchestrators
        for orch in list_orchestrators():
            skills = get_skills_for_agent(orch.name)
            assert skills == [], (
                f"Orchestrator {orch.name!r} should have no mapped skills, got: {skills}"
            )

    def test_get_skills_for_agent_recon(self):
        """recon agent has expected skills."""
        recon_skills = get_skills_for_agent("recon")
        assert "attack-surface-recon" in recon_skills
        assert "network-recon-methodology" in recon_skills
        assert "web-fingerprinting" in recon_skills

    def test_get_skills_for_agent_penetration(self):
        """penetration agent has expected skills (should have several)."""
        pen_skills = get_skills_for_agent("penetration")
        assert "web-attack-methods" in pen_skills
        assert "post-exploitation" in pen_skills
        assert "metasploit-integration" in pen_skills
        # Should have at least 4 skills (it's a complex role)
        assert len(pen_skills) >= 4

    def test_get_agents_for_skill_web_attack(self):
        """web-attack-methods skill maps to penetration + vulnerability-triage."""
        agents = get_agents_for_skill("web-attack-methods")
        assert "penetration" in agents
        assert "vulnerability-triage" in agents
        assert len(agents) >= 2

    def test_get_agents_for_skill_unknown_returns_empty(self):
        """Unknown skill returns empty list (no exception)."""
        assert get_agents_for_skill("nonexistent-skill") == []

    def test_mapping_stats(self):
        """get_mapping_stats returns expected stats."""
        stats = get_mapping_stats()
        assert stats["total_skills"] == 24
        assert stats["total_agents_with_skills"] == 13
        assert stats["avg_skills_per_agent"] > 0
        assert stats["max_skills_per_agent"] >= 4  # penetration has 8
        assert stats["agent_with_most_skills"] == "penetration"
        assert stats["skills_without_agents"] == []  # all skills mapped to ≥1 agent

    def test_each_skill_has_at_least_one_agent(self):
        """Every skill is mapped to at least 1 agent (no orphan skills)."""
        for skill_manifest in skill_loader.list_skills():
            agents = get_agents_for_skill(skill_manifest.name)
            assert len(agents) >= 1, (
                f"Skill {skill_manifest.name!r} has no mapped agents"
            )


# ---------- 4. BaseAgent skill integration ----------

class TestBaseAgentSkills:
    """Tests for W11-S5: BaseAgent auto-loads skills."""

    @pytest.mark.asyncio
    async def test_agent_loaded_skills_property(self, sample_target):
        """agent.loaded_skills returns list of mapped skill names."""
        agent = create_agent(
            agent_name="recon",
            scan_id="scan_test_skills",
            target=sample_target,
            task_description="Test",
        )
        assert isinstance(agent.loaded_skills, list)
        assert "attack-surface-recon" in agent.loaded_skills
        assert "network-recon-methodology" in agent.loaded_skills

    @pytest.mark.asyncio
    async def test_agent_get_loaded_skill_manifests(self, sample_target):
        """get_loaded_skill_manifests returns SkillManifest objects."""
        agent = create_agent(
            agent_name="penetration",
            scan_id="scan_test_manifests",
            target=sample_target,
        )
        manifests = agent.get_loaded_skill_manifests()
        assert len(manifests) == len(agent.loaded_skills)
        for m in manifests:
            assert isinstance(m, SkillManifest)
            assert m.name in agent.loaded_skills

    @pytest.mark.asyncio
    async def test_agent_load_skill_body(self, sample_target):
        """load_skill_body returns Markdown body."""
        agent = create_agent(
            agent_name="recon",
            scan_id="scan_test_body",
            target=sample_target,
        )
        body = agent.load_skill_body("attack-surface-recon")
        assert isinstance(body, str)
        assert len(body) > 100
        assert "## Overview" in body or "## Methodology" in body

    @pytest.mark.asyncio
    async def test_agent_get_skill_context_for_llm(self, sample_target):
        """get_skill_context_for_llm returns formatted string."""
        agent = create_agent(
            agent_name="penetration",
            scan_id="scan_test_llm_ctx",
            target=sample_target,
        )
        ctx = agent.get_skill_context_for_llm()
        assert "Loaded skills:" in ctx
        # Each skill should be listed with its description
        for skill_name in agent.loaded_skills:
            assert skill_name in ctx

    @pytest.mark.asyncio
    async def test_agent_with_no_skills(self, sample_target):
        """Agents with no mapped skills return empty list + '(no skills loaded)'. """
        # Orchestrator-supervisor has no skills (advisory)
        # But create_agent raises for orchestrator — use a sub-agent with 0 skills
        # All 13 sub-agents have skills, so test with a hypothetical scenario:
        # override _loaded_skill_names
        agent = create_agent(
            agent_name="recon",
            scan_id="scan_no_skills",
            target=sample_target,
        )
        agent._loaded_skill_names = []  # simulate no skills
        ctx = agent.get_skill_context_for_llm()
        assert "no skills" in ctx.lower()

    @pytest.mark.asyncio
    async def test_agent_run_result_includes_loaded_skills(self, sample_target):
        """AgentRunResult.to_dict() includes loaded_skills field."""
        agent = create_agent(
            agent_name="recon",
            scan_id="scan_result_skills",
            target=sample_target,
        )
        result = await agent.run()
        d = result.to_dict()
        assert "loaded_skills" in d
        assert isinstance(d["loaded_skills"], list)
        assert "attack-surface-recon" in d["loaded_skills"]

    @pytest.mark.asyncio
    async def test_agent_metadata_to_dict_includes_skills(self, sample_target):
        """AgentMetadata.to_dict() includes skills field (W11-S7)."""
        agent = create_agent(
            agent_name="penetration",
            scan_id="scan_meta_skills",
            target=sample_target,
        )
        d = agent.metadata.to_dict()
        assert "skills" in d
        assert "web-attack-methods" in d["skills"]
        assert "metasploit-integration" in d["skills"]

    @pytest.mark.asyncio
    async def test_agent_load_skill_body_unknown_raises(self, sample_target):
        """load_skill_body raises KeyError for unknown skill."""
        agent = create_agent(
            agent_name="recon",
            scan_id="scan_unknown_skill",
            target=sample_target,
        )
        with pytest.raises(KeyError):
            agent.load_skill_body("nonexistent-skill")


# ---------- 5. Orchestration router (skills endpoints) ----------

class TestSkillsRouter:
    """Tests for 4 new W11-S6 skill endpoints."""

    def test_router_has_9_endpoints(self):
        """Router now exposes 9 endpoints (W9: 3 + W10: 2 + W11: 4)."""
        from app.routes.orchestration import router
        paths = {r.path for r in router.routes}
        # W9 endpoints
        assert "/api/orchestration/modes" in paths
        assert "/api/orchestration/agents" in paths
        assert "/api/orchestration/scans/start-mode" in paths
        # W10 endpoints
        assert "/api/orchestration/agents/{name}" in paths
        assert "/api/orchestration/agents/{name}/invoke" in paths
        # W11-S6 endpoints
        assert "/api/orchestration/skills" in paths
        assert "/api/orchestration/skills/{name}" in paths
        assert "/api/orchestration/agents/{name}/skills" in paths
        assert "/api/orchestration/skills/{name}/agents" in paths

    def test_router_total_count(self):
        """Router has 12 routes total (W9: 3 + W10: 2 + W11: 4 + W12: 3)."""
        from app.routes.orchestration import router
        # W11 had 9, W12 added 3 more (harness endpoints) → 12 total
        assert len(router.routes) >= 9  # at least W9+W10+W11
        assert len(router.routes) == 12  # exactly 12 after W12

    def test_router_skills_endpoints_use_get(self):
        """All 4 new skills endpoints use GET method."""
        from app.routes.orchestration import router
        skill_paths = {
            "/api/orchestration/skills",
            "/api/orchestration/skills/{name}",
            "/api/orchestration/agents/{name}/skills",
            "/api/orchestration/skills/{name}/agents",
        }
        for route in router.routes:
            if route.path in skill_paths:
                assert "GET" in route.methods, (
                    f"Route {route.path} should support GET, got {route.methods}"
                )

    def test_router_prefix_unchanged(self):
        """Router prefix is still /api/orchestration."""
        from app.routes.orchestration import router
        assert router.prefix == "/api/orchestration"

    def test_router_tags_unchanged(self):
        """Router tags still include 'orchestration'."""
        from app.routes.orchestration import router
        assert "orchestration" in router.tags


# ---------- 6. Skill content spot-checks ----------

class TestSkillsContent:
    """Spot-check that key skills have expected content."""

    def test_web_attack_methods_has_owasp_content(self):
        """web-attack-methods skill references OWASP WSTG."""
        skill = skill_loader.load_skill("web-attack-methods")
        assert "OWASP" in skill.body or "WSTG" in skill.body
        assert "SQLi" in skill.body or "SQL Injection" in skill.body
        assert "XSS" in skill.body

    def test_metasploit_integration_has_msfrpcd_content(self):
        """metasploit-integration skill references msfrpcd."""
        skill = skill_loader.load_skill("metasploit-integration")
        assert "msfrpcd" in skill.body.lower() or "msfrpcd" in skill.manifest.description.lower()
        assert "metasploit_client" in skill.body or "MetasploitClient" in skill.body

    def test_all_skills_have_5_sections(self):
        """All 24 skills have the 5-section template (Overview, When to Use, Methodology, Tool Recipes, Output Standards)."""
        for name in skill_loader.list_skill_names():
            skill = skill_loader.load_skill(name)
            for section in ["## Overview", "## When to Use", "## Methodology",
                             "## Tool Recipes", "## Output Standards"]:
                assert section in skill.body, (
                    f"Skill {name!r} missing section: {section}"
                )

    def test_all_skills_have_vapt_ai_note(self):
        """All skills have VAPT-AI Specific Notes section."""
        for name in skill_loader.list_skill_names():
            skill = skill_loader.load_skill(name)
            assert "## VAPT-AI Specific Notes" in skill.body, (
                f"Skill {name!r} missing VAPT-AI Specific Notes section"
            )

    def test_destructive_skills_mention_hitl(self):
        """Destructive skills mention HITL approval requirement."""
        destructive_skill_names = [
            "post-exploitation", "metasploit-integration",
            "persistence-techniques", "impact-proof-methodology",
        ]
        for name in destructive_skill_names:
            skill = skill_loader.load_skill(name)
            assert "HITL" in skill.body, (
                f"Skill {name!r} should mention HITL approval"
            )