"""
VAPT-AI Skill → Agent Mapping — W11-S4.

Explicit mapping table between agents and skills. Auto-derived from
the `mapped-agents` field in each skill's SKILL.md YAML frontmatter
(W11-S1 generator writes this field based on the `agents` list per skill).

Usage:
    from app.skills.agent_mapping import (
        get_skills_for_agent,
        get_agents_for_skill,
        AGENT_SKILL_MAP,
        SKILL_AGENT_MAP,
    )

    # Forward lookup: agent → skills
    recon_skills = get_skills_for_agent("recon")
    # Returns: ["attack-surface-recon", "network-recon-methodology", "web-fingerprinting"]

    # Reverse lookup: skill → agents
    web_attack_agents = get_agents_for_skill("web-attack-methods")
    # Returns: ["penetration", "vulnerability-triage"]

Architecture:
    - AGENT_SKILL_MAP: dict[agent_name, list[skill_name]] (forward map)
    - SKILL_AGENT_MAP: dict[skill_name, list[agent_name]] (reverse map, auto-derived)
    - Both maps loaded lazily from SkillLoader on first access

Mapping rules:
    - Each skill can be mapped to 1+ agents
    - Each agent can have 0+ skills (some agents like engagement-planning
      might have only 1 skill, others like penetration might have 4-5)
    - Mapping is explicit (in SKILL.md frontmatter), not auto-inferred
      from tags — this makes behavior predictable
    - Skills without mapped-agents field are listed but not auto-loaded
      by any agent
"""
from __future__ import annotations

import logging
from typing import Any

from app.skills.loader import skill_loader, SkillManifest

logger = logging.getLogger(__name__)


# ---------- Forward map (agent → skills) ----------

def _build_agent_skill_map() -> dict[str, list[str]]:
    """Build the agent → skills map from skill_registry.

    Iterates all skills, reads `mapped_agents` from each skill's manifest,
    and builds a forward map.

    Returns:
        Dict mapping agent_name → list of skill_names that should auto-load.
    """
    forward_map: dict[str, list[str]] = {}

    for skill_manifest in skill_loader.list_skills():
        for agent_name in skill_manifest.mapped_agents:
            forward_map.setdefault(agent_name, []).append(skill_manifest.name)

    # Sort each agent's skill list for deterministic order
    for agent_name in forward_map:
        forward_map[agent_name].sort()

    return forward_map


# ---------- Reverse map (skill → agents) ----------

def _build_skill_agent_map() -> dict[str, list[str]]:
    """Build the skill → agents map from skill_registry.

    Returns:
        Dict mapping skill_name → list of agent_names that should load it.
    """
    reverse_map: dict[str, list[str]] = {}

    for skill_manifest in skill_loader.list_skills():
        reverse_map[skill_manifest.name] = list(skill_manifest.mapped_agents)

    return reverse_map


# ---------- Cached maps (lazy-loaded) ----------

# Module-level singletons — populated on first access
_AGENT_SKILL_MAP_CACHE: dict[str, list[str]] | None = None
_SKILL_AGENT_MAP_CACHE: dict[str, list[str]] | None = None


def _get_agent_skill_map() -> dict[str, list[str]]:
    """Get (or build) the agent → skills map."""
    global _AGENT_SKILL_MAP_CACHE
    if _AGENT_SKILL_MAP_CACHE is None:
        _AGENT_SKILL_MAP_CACHE = _build_agent_skill_map()
        logger.info(
            "Built agent → skills map: %d agents with at least 1 skill",
            len(_AGENT_SKILL_MAP_CACHE),
        )
    return _AGENT_SKILL_MAP_CACHE


def _get_skill_agent_map() -> dict[str, list[str]]:
    """Get (or build) the skill → agents map."""
    global _SKILL_AGENT_MAP_CACHE
    if _SKILL_AGENT_MAP_CACHE is None:
        _SKILL_AGENT_MAP_CACHE = _build_skill_agent_map()
    return _SKILL_AGENT_MAP_CACHE


# ---------- Public properties (cached maps) ----------

def AGENT_SKILL_MAP() -> dict[str, list[str]]:
    """Get the agent → skills map (cached).

    Returns a copy to prevent external mutation of the cache.
    """
    return dict(_get_agent_skill_map())


def SKILL_AGENT_MAP() -> dict[str, list[str]]:
    """Get the skill → agents map (cached).

    Returns a copy to prevent external mutation of the cache.
    """
    return dict(_get_skill_agent_map())


# ---------- Public lookup functions ----------

def get_skills_for_agent(agent_name: str) -> list[str]:
    """Get list of skill names that should auto-load for an agent.

    Args:
        agent_name: Agent name (e.g. "recon", "penetration")

    Returns:
        Sorted list of skill names. Empty list if agent has no mapped skills
        or agent doesn't exist in any skill's mapped-agents field.
    """
    return list(_get_agent_skill_map().get(agent_name, []))


def get_agents_for_skill(skill_name: str) -> list[str]:
    """Get list of agent names that should auto-load a skill.

    Args:
        skill_name: Skill name (e.g. "web-attack-methods")

    Returns:
        List of agent names. Empty list if skill has no mapped agents
        or skill doesn't exist.
    """
    return list(_get_skill_agent_map().get(skill_name, []))


def get_skills_for_agent_with_manifests(agent_name: str) -> list[SkillManifest]:
    """Get list of SkillManifest objects for an agent (loaded skill metadata).

    Convenience function — combines get_skills_for_agent + skill_loader.get_manifest.

    Args:
        agent_name: Agent name.

    Returns:
        List of SkillManifest objects (in same order as get_skills_for_agent).
    """
    skill_names = get_skills_for_agent(agent_name)
    manifests: list[SkillManifest] = []
    for name in skill_names:
        m = skill_loader.get_manifest(name)
        if m is not None:
            manifests.append(m)
    return manifests


# ---------- Cache management ----------

def reload_maps() -> None:
    """Force reload of both maps on next access (useful for tests)."""
    global _AGENT_SKILL_MAP_CACHE, _SKILL_AGENT_MAP_CACHE
    _AGENT_SKILL_MAP_CACHE = None
    _SKILL_AGENT_MAP_CACHE = None
    # Also reload the underlying skill_loader manifests
    skill_loader.reload()


# ---------- Stats ----------

def get_mapping_stats() -> dict[str, Any]:
    """Get statistics about skill → agent mapping.

    Returns:
        Dict with stats: total_skills, total_agents_with_skills,
        avg_skills_per_agent, max_skills_per_agent, agent_with_most_skills.
    """
    agent_map = _get_agent_skill_map()
    skill_map = _get_skill_agent_map()

    total_skills = skill_loader.total_count
    total_agents_with_skills = len(agent_map)

    skill_counts = [len(skills) for skills in agent_map.values()]
    avg_skills = sum(skill_counts) / len(skill_counts) if skill_counts else 0
    max_skills = max(skill_counts) if skill_counts else 0
    agent_with_most = max(agent_map.items(), key=lambda x: len(x[1]))[0] if agent_map else None

    skills_without_agents = [
        name for name, agents in skill_map.items() if not agents
    ]

    return {
        "total_skills": total_skills,
        "total_agents_with_skills": total_agents_with_skills,
        "avg_skills_per_agent": round(avg_skills, 2),
        "max_skills_per_agent": max_skills,
        "agent_with_most_skills": agent_with_most,
        "skills_without_agents": skills_without_agents,
    }