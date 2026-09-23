"""
VAPT-AI Knowledge Graph seeder.

Populates the KG from 3 sources at app startup (idempotent — safe to
run on every boot):

    1. 24 skill packages (app/skills/<name>/SKILL.md)
       → Technology + AttackVector nodes
       → has_vuln edges (Technology → AttackVector)
       Initial probability: 0.5 (uncertain prior)

    2. OWASP WSTG v4.2 catalog (data/wstg_catalog.yaml — 119 IDs)
       → AttackVector nodes (with wstg_test_id metadata)
       Initial probability: 0.5

    3. MITRE ATT&CK v15.1 catalog (data/attack_mapping.yaml — 44 techniques)
       → AttackVector nodes (with mitre_attack_technique metadata)
       Initial probability: 0.5

After seeding, the KG has:
    - ~24 skill-derived AttackVector nodes
    - 119 WSTG-derived AttackVector nodes
    - 44 ATT&CK-derived AttackVector nodes
    - Total: ~180+ AttackVector nodes ready for path queries
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from app.kg.graph import KnowledgeGraph, SEED_PROBABILITY
from app.kg.types import EdgeType, NodeType

logger = logging.getLogger(__name__)


@dataclass
class SeedResult:
    """Result of a seeding operation — counts for logging/debugging."""

    source: str
    nodes_inserted: int = 0
    nodes_updated: int = 0
    nodes_skipped: int = 0
    edges_inserted: int = 0
    edges_updated: int = 0
    errors: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "nodes_inserted": self.nodes_inserted,
            "nodes_updated": self.nodes_updated,
            "nodes_skipped": self.nodes_skipped,
            "edges_inserted": self.edges_inserted,
            "edges_updated": self.edges_updated,
            "errors": list(self.errors),
        }


# ── Seeders ──────────────────────────────────────────────────────────────

def seed_from_skills(kg: KnowledgeGraph, skills_dir: Path | None = None) -> SeedResult:
    """Seed KG from skill packages.

    Each SKILL.md frontmatter has `name`, `description`, `tags`. We create:
        - AttackVector node per skill (name = skill name, metadata = tags)
        - Technology nodes for each tag (e.g. "web", "sqli", "xss")
        - has_vuln edge: Technology → AttackVector

    Args:
        kg: KnowledgeGraph to seed into
        skills_dir: optional path to skills/ directory. If None, uses
            app/skills/ relative to this file.

    Returns:
        SeedResult with counts.
    """
    import frontmatter  # python-frontmatter

    result = SeedResult(source="skills")

    if skills_dir is None:
        skills_dir = Path(__file__).resolve().parent.parent / "skills"
    if not skills_dir.is_dir():
        result.errors.append(f"skills dir not found: {skills_dir}")
        return result

    skill_count = 0
    for skill_md in skills_dir.glob("*/SKILL.md"):
        try:
            post = frontmatter.load(skill_md)
            skill_name = post.get("name", skill_md.parent.name)
            description = post.get("description", "")
            tags = post.get("metadata", {}).get("tags", []) or []

            # AttackVector node for the skill
            av_id = kg.add_node(
                NodeType.ATTACK_VECTOR,
                f"skill:{skill_name}",
                metadata={
                    "source": "skill",
                    "skill_name": skill_name,
                    "description": description,
                    "tags": tags,
                    "initial_probability": SEED_PROBABILITY,
                },
            )
            # Was it inserted or updated?
            if kg.graph.nodes[av_id].get("created_at") == kg.graph.nodes[av_id].get("updated_at"):
                result.nodes_inserted += 1
            else:
                result.nodes_updated += 1

            # Technology nodes per tag + has_vuln edges
            for tag in tags:
                tag = tag.strip().lower()
                if not tag:
                    continue
                tech_id = kg.add_node(
                    NodeType.TECHNOLOGY,
                    f"tag:{tag}",
                    metadata={"source": "skill_tag", "tag": tag},
                )
                if kg.graph.nodes[tech_id].get("created_at") == kg.graph.nodes[tech_id].get("updated_at"):
                    result.nodes_inserted += 1
                else:
                    result.nodes_updated += 1

                # has_vuln: Technology → AttackVector
                if kg.add_edge(tech_id, av_id, EdgeType.HAS_VULN):
                    result.edges_inserted += 1

            skill_count += 1
        except Exception as exc:
            result.errors.append(f"{skill_md.name}: {exc}")
            result.nodes_skipped += 1

    logger.info(
        "KG seeded from skills | skills=%d nodes_inserted=%d edges_inserted=%d errors=%d",
        skill_count, result.nodes_inserted, result.edges_inserted, len(result.errors),
    )
    return result


def seed_from_wstg_catalog(kg: KnowledgeGraph, yaml_path: Path | None = None) -> SeedResult:
    """Seed KG from OWASP WSTG v4.2 catalog YAML.

    Creates AttackVector nodes for each WSTG test ID with metadata:
        wstg_test_id, name, category, description, related_cwe, related_mitre_attack

    Args:
        kg: KnowledgeGraph to seed into
        yaml_path: optional path to wstg_catalog.yaml. If None, uses
            data/wstg_catalog.yaml relative to project root.

    Returns:
        SeedResult with counts.
    """
    result = SeedResult(source="wstg_catalog")

    if yaml_path is None:
        yaml_path = Path(__file__).resolve().parent.parent.parent / "data" / "wstg_catalog.yaml"
    if not yaml_path.is_file():
        result.errors.append(f"WSTG catalog not found: {yaml_path}")
        return result

    try:
        with open(yaml_path, encoding="utf-8") as f:
            catalog = yaml.safe_load(f)
    except Exception as exc:
        result.errors.append(f"YAML parse error: {exc}")
        return result

    tests = catalog.get("tests") or []
    for test in tests:
        wstg_id = test.get("wstg_id")
        if not wstg_id:
            result.nodes_skipped += 1
            continue

        name = test.get("name", wstg_id)
        category = test.get("category", "")
        description = test.get("description", "")
        related_cwe = test.get("related_cwe", []) or []
        related_mitre = test.get("related_mitre_attack", []) or []

        av_id = kg.add_node(
            NodeType.ATTACK_VECTOR,
            f"wstg:{wstg_id}",
            metadata={
                "source": "wstg_catalog",
                "wstg_test_id": wstg_id,
                "name": name,
                "category": category,
                "description": description,
                "related_cwe": related_cwe,
                "related_mitre_attack": related_mitre,
                "initial_probability": SEED_PROBABILITY,
            },
        )
        if kg.graph.nodes[av_id].get("created_at") == kg.graph.nodes[av_id].get("updated_at"):
            result.nodes_inserted += 1
        else:
            result.nodes_updated += 1

    logger.info(
        "KG seeded from WSTG catalog | tests=%d nodes_inserted=%d nodes_updated=%d",
        len(tests), result.nodes_inserted, result.nodes_updated,
    )
    return result


def seed_from_attack_catalog(kg: KnowledgeGraph, yaml_path: Path | None = None) -> SeedResult:
    """Seed KG from MITRE ATT&CK catalog YAML.

    Creates AttackVector nodes for each ATT&CK technique with metadata:
        mitre_attack_technique, name, tactic, description, detection, mitigation

    Args:
        kg: KnowledgeGraph to seed into
        yaml_path: optional path to attack_mapping.yaml.

    Returns:
        SeedResult with counts.
    """
    result = SeedResult(source="attack_catalog")

    if yaml_path is None:
        yaml_path = Path(__file__).resolve().parent.parent.parent / "data" / "attack_mapping.yaml"
    if not yaml_path.is_file():
        result.errors.append(f"ATT&CK catalog not found: {yaml_path}")
        return result

    try:
        with open(yaml_path, encoding="utf-8") as f:
            catalog = yaml.safe_load(f)
    except Exception as exc:
        result.errors.append(f"YAML parse error: {exc}")
        return result

    techniques = catalog.get("techniques") or []
    for tech in techniques:
        tech_id = tech.get("technique_id")
        if not tech_id:
            result.nodes_skipped += 1
            continue

        name = tech.get("name", tech_id)
        tactic = tech.get("tactic", "")
        description = tech.get("description", "")
        detection = tech.get("detection", "")
        mitigation = tech.get("mitigation", "")
        related_wstg = tech.get("related_wstg_ids", []) or []
        related_cwe = tech.get("related_cwe", []) or []

        av_id = kg.add_node(
            NodeType.ATTACK_VECTOR,
            f"attack:{tech_id}",
            metadata={
                "source": "attack_catalog",
                "mitre_attack_technique": tech_id,
                "name": name,
                "tactic": tactic,
                "description": description,
                "detection": detection,
                "mitigation": mitigation,
                "related_wstg_ids": related_wstg,
                "related_cwe": related_cwe,
                "initial_probability": SEED_PROBABILITY,
            },
        )
        if kg.graph.nodes[av_id].get("created_at") == kg.graph.nodes[av_id].get("updated_at"):
            result.nodes_inserted += 1
        else:
            result.nodes_updated += 1

    logger.info(
        "KG seeded from ATT&CK catalog | techniques=%d nodes_inserted=%d nodes_updated=%d",
        len(techniques), result.nodes_inserted, result.nodes_updated,
    )
    return result


def seed_all(kg: KnowledgeGraph) -> dict[str, Any]:
    """Run all 3 seeders (skills + WSTG + ATT&CK). Idempotent.

    Args:
        kg: KnowledgeGraph to seed into

    Returns:
        dict with combined results from all 3 seeders.
    """
    skills_result = seed_from_skills(kg)
    wstg_result = seed_from_wstg_catalog(kg)
    attack_result = seed_from_attack_catalog(kg)

    return {
        "skills": skills_result.as_dict(),
        "wstg": wstg_result.as_dict(),
        "attack": attack_result.as_dict(),
        "total_nodes_inserted": (
            skills_result.nodes_inserted
            + wstg_result.nodes_inserted
            + attack_result.nodes_inserted
        ),
        "total_edges_inserted": (
            skills_result.edges_inserted
            + wstg_result.edges_inserted
            + attack_result.edges_inserted
        ),
        "total_errors": (
            len(skills_result.errors)
            + len(wstg_result.errors)
            + len(attack_result.errors)
        ),
    }


__all__ = [
    "SeedResult",
    "seed_from_skills",
    "seed_from_wstg_catalog",
    "seed_from_attack_catalog",
    "seed_all",
]