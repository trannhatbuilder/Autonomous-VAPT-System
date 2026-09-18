"""
VAPT-AI Skills Package — W11.

24 skill packages following the agentskills.io specification (same as
CyberStrikeAI internal/skillpackage). Each skill is a directory
app/skills/<name>/SKILL.md with YAML frontmatter + Markdown body.

Public API:
    - SkillManifest: parsed YAML frontmatter (port from CyberStrikeAI types.go)
    - Skill: full skill instance (manifest + body)
    - SkillLoader: singleton loader with lazy-load + cache
    - AGENT_SKILL_MAP: skill → agent mapping table
    - get_skills_for_agent(agent_name): list skills mapped to an agent
    - get_agents_for_skill(skill_name): list agents that should load a skill

Usage:
    from app.skills import (
        SkillLoader, SkillManifest, Skill,
        get_skills_for_agent, get_agents_for_skill,
    )

    # List all skills (summary metadata only — fast, no body load)
    skills = SkillLoader.list_skills()
    for s in skills:
        print(f"{s.name}: {s.description}")

    # Get single skill with full body
    skill = SkillLoader.load_skill("web-attack-methods")
    print(skill.body[:200])

    # Find skills mapped to an agent
    agent_skills = get_skills_for_agent("recon")
    # Returns: ["attack-surface-recon", "network-recon-methodology", "web-fingerprinting"]

Architecture:
    ┌───────────────────────────────────────────────────────┐
    │                  SkillLoader (singleton)              │
    │                                                       │
    │   Skills dir: app/skills/<name>/SKILL.md              │
    │                                                       │
    │   Lazy load:                                          │
    │     1. List skill dirs (24 subdirs with SKILL.md)     │
    │     2. Parse YAML frontmatter (cache parsed manifest) │
    │     3. Load body on demand (load_skill_body)          │
    │                                                       │
    │   Cache:                                              │
    │     - _manifests_cache: dict[name, SkillManifest]     │
    │     - _bodies_cache: dict[name, str] (lazy)           │
    └───────────────────────────────────────────────────────┘
                            │
                            ▼
    ┌───────────────────────────────────────────────────────┐
    │            agent_mapping.py (W11-S4)                  │
    │                                                       │
    │   AGENT_SKILL_MAP: explicit mapping table             │
    │     e.g. "recon" → ["attack-surface-recon",           │
    │                      "network-recon-methodology",     │
    │                      "web-fingerprinting"]            │
    │                                                       │
    │   Reverse map (skill → agents) auto-derived.          │
    └───────────────────────────────────────────────────────┘
                            │
                            ▼
    ┌───────────────────────────────────────────────────────┐
    │       BaseAgent.__init__() (W11-S5 wiring)            │
    │                                                       │
    │   After loading metadata from agent_registry:         │
    │     self.loaded_skills = get_skills_for_agent(name)   │
    │                                                       │
    │   Agent can access skill content via:                 │
    │     for skill_name in self.loaded_skills:             │
    │         skill = SkillLoader.load_skill(skill_name)    │
    └───────────────────────────────────────────────────────┘
"""
from __future__ import annotations

from app.skills.base import Skill, SkillManifest
from app.skills.loader import SkillLoader, skill_loader
from app.skills.agent_mapping import (
    AGENT_SKILL_MAP,
    SKILL_AGENT_MAP,
    get_skills_for_agent,
    get_agents_for_skill,
    get_skills_for_agent_with_manifests,
    get_mapping_stats,
    reload_maps,
)

__all__ = [
    # Base classes
    "Skill",
    "SkillManifest",
    # Loader
    "SkillLoader",
    "skill_loader",  # singleton instance
    # Mapping
    "AGENT_SKILL_MAP",
    "SKILL_AGENT_MAP",
    "get_skills_for_agent",
    "get_agents_for_skill",
    "get_skills_for_agent_with_manifests",
    "get_mapping_stats",
    "reload_maps",
]