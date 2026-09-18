"""
VAPT-AI Skill Base Classes — W11-S2.

Defines the SkillManifest dataclass (port from CyberStrikeAI internal/skillpackage/types.go)
and Skill class (manifest + body). Follows the agentskills.io specification.

Schema:
    SkillManifest fields:
        name:           str (required)
        description:    str (required)
        license:        str (optional, default "apache-2.0")
        compatibility:  str (optional, e.g. ">=3.2")
        metadata:       dict (optional)
            version:    str (e.g. "1.0.0")
            tags:       list[str]
            triggers:   list[str]
        allowed_tools:  str (optional, comma-separated tool list)
        mapped_agents:  list[str] (parsed from frontmatter, not standard agentskills.io)

    Skill fields:
        name:           str (same as manifest.name)
        manifest:       SkillManifest (parsed YAML frontmatter)
        body:           str (Markdown content after frontmatter)
        dir_path:       Path (absolute path to skill directory)
        skill_md_path:  Path (absolute path to SKILL.md)
"""
from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ---------- SkillManifest dataclass ----------

@dataclass(frozen=True)
class SkillManifest:
    """Parsed YAML frontmatter of a SKILL.md file.

    Port of CyberStrikeAI internal/skillpackage/types.go SkillManifest.
    Frozen for hashability + thread-safety after load.

    Attributes:
        name:           Skill name (matches directory name)
        description:    Short description (1-2 sentences)
        license:        License (default "apache-2.0")
        compatibility:  Version compatibility string (e.g. ">=3.2")
        metadata:       Optional metadata map (version, tags, triggers)
        allowed_tools:  Comma-separated tool allowlist (string, not list —
                        matches CyberStrikeAI SkillManifest.AllowedTools which is string)
        mapped_agents:  List of agent names that should auto-load this skill.
                        Parsed from frontmatter `mapped-agents` field (VAPT-AI extension,
                        not in standard agentskills.io spec).
    """
    name: str
    description: str
    license: str = "apache-2.0"
    compatibility: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    allowed_tools: str = ""
    mapped_agents: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict (for FastAPI response)."""
        d = asdict(self)
        d["mapped_agents"] = list(self.mapped_agents)  # tuple → list for JSON
        # Flatten metadata for easier API consumption
        if self.metadata:
            d["version"] = self.metadata.get("version", "")
            d["tags"] = list(self.metadata.get("tags", []))
            d["triggers"] = list(self.metadata.get("triggers", []))
        else:
            d["version"] = ""
            d["tags"] = []
            d["triggers"] = []
        return d

    @property
    def tags(self) -> list[str]:
        """Convenience accessor for metadata.tags."""
        return list(self.metadata.get("tags", []))

    @property
    def triggers(self) -> list[str]:
        """Convenience accessor for metadata.triggers."""
        return list(self.metadata.get("triggers", []))

    @property
    def version(self) -> str:
        """Convenience accessor for metadata.version."""
        return str(self.metadata.get("version", ""))

    @property
    def allowed_tools_list(self) -> list[str]:
        """Parse allowed_tools (comma-separated string) → list."""
        if not self.allowed_tools:
            return []
        return [t.strip() for t in self.allowed_tools.split(",") if t.strip()]


# ---------- Skill class ----------

@dataclass(frozen=True)
class Skill:
    """Full skill instance — manifest + body + paths.

    Frozen for hashability. Body is loaded lazily by SkillLoader (not at
    Skill construction time) to support progressive disclosure.

    Attributes:
        name:           Skill name (same as manifest.name)
        manifest:       Parsed YAML frontmatter (SkillManifest)
        body:           Markdown body (loaded on demand, empty until loaded)
        dir_path:       Absolute path to skill directory
        skill_md_path:  Absolute path to SKILL.md file
    """
    name: str
    manifest: SkillManifest
    body: str  # empty until SkillLoader.load_skill_body() is called
    dir_path: Path
    skill_md_path: Path

    def to_dict(self, include_body: bool = False) -> dict[str, Any]:
        """Serialize to dict.

        Args:
            include_body: If True, include full Markdown body (large).
                          Default False — only metadata (for summary listings).
        """
        d = self.manifest.to_dict()
        d["dir_path"] = str(self.dir_path)
        d["skill_md_path"] = str(self.skill_md_path)
        d["body_length"] = len(self.body)
        if include_body:
            d["body"] = self.body
        return d

    @property
    def has_body(self) -> bool:
        """True if body has been loaded (non-empty)."""
        return bool(self.body)


# ---------- Parsing helpers ----------

def parse_skill_md(content: str) -> tuple[SkillManifest, str]:
    """Parse SKILL.md content into (SkillManifest, body).

    Port of CyberStrikeAI internal/skillpackage/frontmatter.go ParseSkillMD.

    Args:
        content: Full SKILL.md file content (with YAML frontmatter + body).

    Returns:
        (SkillManifest, body) tuple.

    Raises:
        ValueError: if frontmatter is missing or malformed.
    """
    # Strip BOM if present
    text = content.lstrip("\ufeff")
    if not text.strip():
        raise ValueError("SKILL.md is empty")

    lines = text.split("\n")
    if len(lines) < 2 or lines[0].strip() != "---":
        raise ValueError(
            "SKILL.md must start with YAML front matter (---) per Agent Skills standard"
        )

    # Find closing ---
    fm_lines: list[str] = []
    i = 1
    while i < len(lines):
        if lines[i].strip() == "---":
            break
        fm_lines.append(lines[i])
        i += 1

    if i >= len(lines):
        raise ValueError("SKILL.md: front matter must end with a line containing only ---")

    body = "\n".join(lines[i + 1:]).strip()
    fm_text = "\n".join(fm_lines)

    # Parse YAML (use yaml module for robustness)
    import yaml
    try:
        fm_data = yaml.safe_load(fm_text) or {}
    except yaml.YAMLError as e:
        raise ValueError(f"SKILL.md front matter YAML parse error: {e}") from e

    # Build SkillManifest
    name = str(fm_data.get("name", "")).strip()
    description = str(fm_data.get("description", "")).strip()
    if not name or not description:
        raise ValueError(
            f"SKILL.md frontmatter must have 'name' and 'description' (got name={name!r})"
        )

    license_str = str(fm_data.get("license", "apache-2.0")).strip()
    compatibility = str(fm_data.get("compatibility", "")).strip()
    metadata = dict(fm_data.get("metadata", {}) or {})
    allowed_tools = str(fm_data.get("allowed-tools", "")).strip()

    # Parse mapped-agents (VAPT-AI extension)
    mapped_agents_raw = fm_data.get("mapped-agents", []) or []
    if isinstance(mapped_agents_raw, str):
        # Comma-separated string
        mapped_agents = tuple(
            a.strip() for a in mapped_agents_raw.split(",") if a.strip()
        )
    elif isinstance(mapped_agents_raw, list):
        mapped_agents = tuple(str(a).strip() for a in mapped_agents_raw if a)
    else:
        mapped_agents = tuple()

    manifest = SkillManifest(
        name=name,
        description=description,
        license=license_str,
        compatibility=compatibility,
        metadata=metadata,
        allowed_tools=allowed_tools,
        mapped_agents=mapped_agents,
    )

    return manifest, body