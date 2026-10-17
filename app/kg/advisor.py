"""
VAPT-AI KG Advisor — markdown rendering + tech-stack extraction layer
on top of KnowledgeGraph.

The existing KnowledgeGraph class (app/kg/graph.py) speaks in node IDs
and ConsultResult dataclasses. The agent loop and the LLM need:
  - A way to turn recon tool output into tech-stack node IDs.
  - A markdown brief the LLM can read.
  - A structured plan for later phases (EGATS, TDA).

This module bridges that gap. It is the ONLY KG surface the agent loop
should touch — direct KnowledgeGraph calls stay internal to the advisor.

Advisory framing
----------------
All rendered text uses advisory framing (consistent with RL Phase 1):
  - "Past hits on similar stacks: ..."
  - "Advisory — your judgment wins."
  - Never "Recommended Path: ... — follow this."

The LLM is free to disagree with the KG. The KG's job is to surface
prior-attack intel, not to dictate the next action.
"""
from __future__ import annotations

import logging
from typing import Any

from app.kg._recon_parser import parse_recon_output
from app.kg.graph import KnowledgeGraph, get_kg
from app.kg.types import (
    ConsultResult,
    EdgeType,
    KGPath,
    NodeType,
)

logger = logging.getLogger(__name__)


# ─── tech-stack extraction ────────────────────────────────────────────


def extract_tech_stack(
    *,
    technologies: list[str] | None = None,
    recon_signals: dict[str, Any] | None = None,
    last_command: str = "",
    last_tool_output: str = "",
    target_url: str = "",
) -> list[str]:
    """Build a canonical tech-stack token list from any signals available.

    Combines 3 sources (later sources win on dedup):
        1. `technologies` — explicit list (e.g. from a recon module).
        2. `recon_signals["technologies"]` — structured recon output.
        3. Parsing `last_tool_output` if `last_command` is a recon tool
           (nmap / whatweb / httpx).

    Returns a deduplicated list of canonical tech names (lowercase,
    alias-resolved). Empty list when no signals are available.
    """
    tokens: list[str] = []

    # Source 1: explicit list
    if technologies:
        tokens.extend(technologies)

    # Source 2: recon_signals dict
    if recon_signals and isinstance(recon_signals, dict):
        sig_techs = recon_signals.get("technologies") or []
        if isinstance(sig_techs, list):
            tokens.extend(sig_techs)

    # Source 3: parse last recon tool output
    if last_command and last_tool_output:
        parsed = parse_recon_output(last_command, last_tool_output)
        if parsed:
            tokens.extend(parsed)

    # Dedup preserving order. We don't normalize here — the parsers
    # already normalize, and any explicit `technologies` entries are
    # assumed canonical by the caller.
    seen: set[str] = set()
    out: list[str] = []
    for t in tokens:
        if not t:
            continue
        t_lower = t.lower().strip()
        if t_lower in seen:
            continue
        seen.add(t_lower)
        out.append(t_lower)
    return out


def _tech_names_to_node_ids(
    kg: KnowledgeGraph, tech_names: list[str]
) -> list[str]:
    """Map canonical tech names to KG Technology node IDs.

    For each tech name, try in order:
        1. Exact match: find_node(TECHNOLOGY, name)
        2. Tag prefix: find_node(TECHNOLOGY, "tag:" + name)
        3. Create new: add_node(TECHNOLOGY, name) if not found

    The seeder creates Technology nodes from skill tags (e.g. "tag:web",
    "tag:sqli"), not from actual technologies (e.g. "php", "nginx").
    When whatweb detects "php", there's no matching node — so we create
    one on-the-fly. This lets EGATS seed paths immediately, and after
    a few scans the KG will learn edges from tech → attack vectors via
    the consult hit/miss feedback loop.
    """
    node_ids: list[str] = []
    for name in tech_names:
        if not name:
            continue
        nid = None
        # 1. Exact match
        try:
            nid = kg.find_node(NodeType.TECHNOLOGY, name)
        except Exception:
            nid = None
        # 2. Tag prefix match (seeder creates "tag:web", "tag:sqli"...)
        if not nid:
            try:
                nid = kg.find_node(NodeType.TECHNOLOGY, f"tag:{name}")
            except Exception:
                nid = None
        # 3. Create new tech node on-the-fly
        if not nid:
            try:
                nid = kg.add_node(
                    NodeType.TECHNOLOGY, name,
                    metadata={"source": "recon_parser", "auto_created": True},
                )
                logger.info(
                    "kg_tech_node_created | name=%s | node_id=%s",
                    name, nid,
                )
            except Exception as exc:
                logger.debug(
                    "kg_tech_node_create_failed | name=%s | err=%s",
                    name, exc,
                )
                continue
        if nid:
            node_ids.append(nid)
    return node_ids


# ─── consult_text — markdown brief for LLM injection ──────────────────


def consult_text(
    kg: KnowledgeGraph,
    tech_stack: list[str],
    *,
    focus: str | None = None,
    top_k: int = 5,
) -> tuple[str, str]:
    """Consult the KG and render a markdown advisory brief for the LLM.

    Args:
        kg: KnowledgeGraph instance (use get_kg() for the singleton).
        tech_stack: canonical tech names (e.g. ["php", "nginx", "mysql"]).
        focus: optional attack-vector focus (e.g. "sqli", "xss") to
            narrow the consult. None = general consult.
        top_k: max number of paths to include in the brief.

    Returns:
        (consult_id, markdown_text)
        - consult_id: UUID the agent quotes back when reporting a finding
          so the KG can credit the recommendation. "" on failure.
        - markdown_text: the advisory brief. "" on failure or when the
          KG has no paths for this stack.
    """
    if not tech_stack:
        return "", ""

    node_ids = _tech_names_to_node_ids(kg, tech_stack)
    if not node_ids:
        logger.debug(
            "kg_consult_no_matching_nodes | tech_stack=%s", tech_stack,
        )
        return "", ""

    try:
        result = kg.consult(node_ids, top_k=top_k)
    except Exception as exc:
        logger.warning("kg_consult_failed | err=%s", exc)
        return "", ""

    if not result.paths:
        return result.consult_id, ""

    return result.consult_id, render_advisory_brief(result, tech_stack, focus)


def render_advisory_brief(
    consult_result: ConsultResult,
    tech_stack: list[str] | None = None,
    focus: str | None = None,
) -> str:
    """Render a ConsultResult as an advisory markdown brief.

    Framing rules (advisory only — Phase 2):
        - Header: "## KG Hints (advisory — your judgment wins)"
        - "Past hits on similar stacks" (descriptive, not prescriptive)
        - Always include "(advisory — not directives)"
        - When paths is empty, return "" (caller should not inject).
    """
    if not consult_result or not consult_result.paths:
        return ""

    lines: list[str] = [
        "## KG Hints (advisory — not directives)",
        "",
    ]

    if tech_stack:
        lines.append(f"**Detected stack:** {', '.join(tech_stack)}")
        lines.append("")

    if focus:
        lines.append(f"**Focus:** {focus}")
        lines.append("")

    lines.append(
        "Past hits on similar stacks (from prior scans). Use these as "
        "one signal among many — trust your reasoning over the KG when "
        "they disagree."
    )
    lines.append("")

    for i, path in enumerate(consult_result.paths[:5], 1):
        # Render the path as an arrow chain. We use node NAMES not IDs
        # so the LLM sees human-readable labels.
        chain_parts: list[str] = []
        for nid in path.nodes:
            try:
                node = consult_result.notes and None  # placeholder
            except Exception:
                node = None
            chain_parts.append(nid)  # fallback to ID if name lookup fails

        # KGPath.nodes are node IDs. We can't resolve them to names here
        # without a kg reference — the caller (consult_text) has the kg
        # but the rendering is decoupled. For Phase 2, we render IDs.
        # Phase 3 can pass a name resolver if needed.
        chain = " → ".join(chain_parts) if chain_parts else "(empty path)"

        confidence = path.score  # 0..1 — severity * probability product
        specificity = path.match_specificity  # 0..1 — inventory match

        specificity_tag = (
            "exact match" if specificity >= 0.75
            else "partial match" if specificity >= 0.5
            else "generic"
        )

        lines.append(
            f"{i}. [{specificity_tag} · score={confidence:.2f}] {chain}"
        )

    if consult_result.notes:
        lines.append("")
        lines.append("**Notes:**")
        for note in consult_result.notes:
            lines.append(f"- {note}")

    lines.append("")
    lines.append(
        f"_Pass `kg_consult_id=\"{consult_result.consult_id}\"` in "
        f"`record_vulnerability` when you confirm a finding that "
        f"follows any of the above paths — the KG will credit the "
        f"recommendation and learn from the outcome._"
    )

    return "\n".join(lines)


# ─── consult_plan — structured plan for RL/EGATS consumption ──────────


def consult_plan(
    kg: KnowledgeGraph,
    tech_stack: list[str],
    *,
    inventory: list[dict[str, Any]] | None = None,
    focus: str | None = None,
    top_k: int = 5,
) -> dict[str, Any]:
    """Consult the KG and return a structured plan (machine-readable).

    Used by later phases (EGATS UCB path selection, TDA difficulty
    scoring). Phase 2 only uses consult_text() for the LLM brief — this
    function is provided so Phase 3+ can read from the same consult
    call without re-querying the KG.
    """
    if not tech_stack:
        return {"consult_id": "", "paths": [], "tech_stack": []}

    node_ids = _tech_names_to_node_ids(kg, tech_stack)
    if not node_ids:
        return {"consult_id": "", "paths": [], "tech_stack": tech_stack}

    try:
        result = kg.consult(node_ids, top_k=top_k)
    except Exception as exc:
        logger.warning("kg_consult_plan_failed | err=%s", exc)
        return {"consult_id": "", "paths": [], "tech_stack": tech_stack}

    return {
        "consult_id": result.consult_id,
        "paths": [p.as_dict() for p in result.paths],
        "tech_stack": tech_stack,
        "inventory_summary": result.inventory_summary,
        "notes": list(result.notes),
    }


# ─── attach helper for HarnessBridge ──────────────────────────────────


def get_kg_safe() -> KnowledgeGraph | None:
    """Get the KG singleton, returning None on failure.

    Used by HarnessBridge.attach_kg() so RL doesn't crash when the KG
    is unavailable (e.g. during unit tests without a DB).
    """
    try:
        return get_kg()
    except Exception as exc:
        logger.debug("kg_unavailable | err=%s", exc)
        return None


__all__ = [
    "extract_tech_stack",
    "consult_text",
    "consult_plan",
    "render_advisory_brief",
    "get_kg_safe",
]