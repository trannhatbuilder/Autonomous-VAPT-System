"""
VAPT-AI Skill Loader — W11-S3.

Singleton SkillLoader that lazily loads skill manifests from app/skills/.
Caches parsed frontmatter for fast repeated lookups. Body is loaded on
demand (progressive disclosure).

Features:
    - Lazy load: skill dirs enumerated on first access, manifests parsed on demand
    - Cache: _manifests_cache dict (manifest by name), _bodies_cache dict (body by name)
    - Thread-safe: read-only after first load (frozen dataclass + cached dict)
    - Reloadable: reload() method for tests

Usage:
    from app.skills import SkillLoader, skill_loader

    # List all skills (summary only — fast)
    summaries = skill_loader.list_skills()
    for s in summaries:
        print(f"{s.name}: {s.description}")

    # Get single skill with full body (lazy-loaded)
    skill = skill_loader.load_skill("web-attack-methods")
    print(skill.body[:200])

    # Get just the manifest (faster — no body load)
    manifest = skill_loader.get_manifest("web-attack-methods")

    # Search skills by tag
    web_skills = skill_loader.search_by_tag("web")

    # Search skills by trigger keyword
    matched = skill_loader.search_by_trigger("sql injection")
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from app.skills.base import Skill, SkillManifest, parse_skill_md

logger = logging.getLogger(__name__)


# ---------- Constants ----------

SKILLS_DIR = Path(__file__).resolve().parent
SKILL_MD_FILENAME = "SKILL.md"


# ---------- SkillLoader ----------

class SkillLoader:
    """Singleton loader for skill packages.

    Lazy-loads SkillManifest from app/skills/<name>/SKILL.md files.
    Caches parsed manifests for fast repeated lookups. Body is loaded
    on demand (progressive disclosure — large bodies only loaded when
    actually needed).

    Thread-safety: read-only after first load (frozen dataclass + cached dict).
    """

    def __init__(self, skills_dir: Path | None = None):
        self.skills_dir: Path = skills_dir or SKILLS_DIR
        self._manifests_cache: dict[str, SkillManifest] | None = None
        self._bodies_cache: dict[str, str] = {}  # body cache (lazy-loaded)

    # ---------- Lazy load ----------

    @property
    def manifests(self) -> dict[str, SkillManifest]:
        """Lazy-loaded manifests dict. Parses YAML frontmatter on first access."""
        if self._manifests_cache is None:
            self._manifests_cache = self._load_all_manifests()
        return self._manifests_cache

    def _load_all_manifests(self) -> dict[str, SkillManifest]:
        """Parse all SKILL.md files in skills_dir.

        Returns:
            Dict mapping skill name → SkillManifest.

        Raises:
            FileNotFoundError: if skills_dir doesn't exist.
        """
        if not self.skills_dir.is_dir():
            raise FileNotFoundError(
                f"Skills directory not found: {self.skills_dir}"
            )

        result: dict[str, SkillManifest] = {}

        for skill_dir in sorted(self.skills_dir.iterdir()):
            if not skill_dir.is_dir():
                continue
            if skill_dir.name.startswith("."):
                continue

            skill_md_path = skill_dir / SKILL_MD_FILENAME
            if not skill_md_path.is_file():
                continue

            try:
                content = skill_md_path.read_text(encoding="utf-8")
                manifest, _body = parse_skill_md(content)
                # Sanity check: manifest.name should match directory name
                if manifest.name != skill_dir.name:
                    logger.warning(
                        "Skill name mismatch: dir=%r manifest.name=%r — using manifest name",
                        skill_dir.name, manifest.name,
                    )
                result[manifest.name] = manifest
            except Exception as e:
                logger.error("Failed to parse %s: %s", skill_md_path, e)
                continue

        logger.info(
            "Loaded %d skill manifests from %s",
            len(result), self.skills_dir,
        )
        return result

    def reload(self) -> None:
        """Force reload on next access (useful for tests)."""
        self._manifests_cache = None
        self._bodies_cache.clear()

    # ---------- Lookup ----------

    def get_manifest(self, name: str) -> SkillManifest | None:
        """Lookup skill manifest by name (no body load). Returns None if not found."""
        return self.manifests.get(name)

    def require_manifest(self, name: str) -> SkillManifest:
        """Lookup manifest. Raises KeyError if not found."""
        m = self.get_manifest(name)
        if m is None:
            raise KeyError(
                f"Unknown skill: {name!r}. Available: {sorted(self.manifests.keys())}"
            )
        return m

    def load_skill(self, name: str) -> Skill:
        """Load full skill (manifest + body). Body is cached.

        Args:
            name: Skill name (e.g. "web-attack-methods")

        Returns:
            Skill instance with body loaded.

        Raises:
            KeyError: if skill not found.
        """
        manifest = self.require_manifest(name)

        # Lazy-load body (cached)
        if name not in self._bodies_cache:
            skill_md_path = self.skills_dir / name / SKILL_MD_FILENAME
            content = skill_md_path.read_text(encoding="utf-8")
            _, body = parse_skill_md(content)
            self._bodies_cache[name] = body

        return Skill(
            name=name,
            manifest=manifest,
            body=self._bodies_cache[name],
            dir_path=self.skills_dir / name,
            skill_md_path=self.skills_dir / name / SKILL_MD_FILENAME,
        )

    def load_skill_body(self, name: str) -> str:
        """Load just the body (Markdown content). Cached.

        Args:
            name: Skill name.

        Returns:
            Body as string.

        Raises:
            KeyError: if skill not found.
        """
        # Verify skill exists
        self.require_manifest(name)
        if name not in self._bodies_cache:
            skill = self.load_skill(name)
            return skill.body
        return self._bodies_cache[name]

    # ---------- Listing ----------

    def list_skills(self) -> list[SkillManifest]:
        """List all skill manifests (no body load — fast)."""
        return list(self.manifests.values())

    def list_skill_names(self) -> list[str]:
        """List all skill names (sorted)."""
        return sorted(self.manifests.keys())

    # ---------- Search ----------

    def search_by_tag(self, tag: str) -> list[SkillManifest]:
        """Find skills with a specific tag.

        Args:
            tag: Tag to search for (case-insensitive exact match).

        Returns:
            List of matching SkillManifest instances.
        """
        tag_lower = tag.lower()
        return [
            m for m in self.manifests.values()
            if any(t.lower() == tag_lower for t in m.tags)
        ]

    def search_by_trigger(self, keyword: str) -> list[SkillManifest]:
        """Find skills whose triggers contain the keyword (case-insensitive substring).

        Args:
            keyword: Keyword to search for in triggers.

        Returns:
            List of matching SkillManifest instances.
        """
        kw_lower = keyword.lower()
        return [
            m for m in self.manifests.values()
            if any(kw_lower in t.lower() for t in m.triggers)
        ]

    def search_by_name_substring(self, query: str) -> list[SkillManifest]:
        """Find skills whose name contains the query (case-insensitive)."""
        q_lower = query.lower()
        return [
            m for m in self.manifests.values()
            if q_lower in m.name.lower()
        ]

    def search_by_description_substring(self, query: str) -> list[SkillManifest]:
        """Find skills whose description contains the query (case-insensitive)."""
        q_lower = query.lower()
        return [
            m for m in self.manifests.values()
            if q_lower in m.description.lower()
        ]

    # ---------- Counting ----------

    @property
    def total_count(self) -> int:
        """Total number of skills loaded."""
        return len(self.manifests)

    # ---------- Serialization ----------

    def to_dict(self) -> dict[str, Any]:
        """Serialize entire loader state (for debugging / API response)."""
        return {
            "skills_dir": str(self.skills_dir),
            "total_count": self.total_count,
            "skills": [m.to_dict() for m in self.list_skills()],
        }


# ---------- Singleton instance ----------

skill_loader = SkillLoader()


# ---------- Module-level convenience functions ----------

def get_manifest(name: str) -> SkillManifest | None:
    """Module-level shortcut: skill_loader.get_manifest(name)."""
    return skill_loader.get_manifest(name)


def load_skill(name: str) -> Skill:
    """Module-level shortcut: skill_loader.load_skill(name)."""
    return skill_loader.load_skill(name)


def list_skills() -> list[SkillManifest]:
    """Module-level shortcut: skill_loader.list_skills()."""
    return skill_loader.list_skills()


def list_skill_names() -> list[str]:
    """Module-level shortcut: skill_loader.list_skill_names()."""
    return skill_loader.list_skill_names()