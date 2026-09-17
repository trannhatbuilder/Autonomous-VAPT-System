"""
VAPT-AI Orchestration Mode Selector — W9-S6.

Decides which of the 3 orchestration modes to use for a given scan.

Per master plan §12 W9 task 6:
    single_web_app + ≤5 steps       → supervisor (default)
    network_range + subnets > 1     → deep
    full_kill_chain or has_post_exploitation → plan_execute
    default                          → supervisor

Selection inputs:
    - target_type:  "single_url" | "network_range" | "domain_with_subdomains" | "api_endpoint"
    - scope_size:   number of in-scope hosts/URLs (1 = single, >1 = range)
    - has_post_exploitation: bool — user wants full kill-chain (exploit → privesc → lateral)
    - user_override: explicit mode string from user (highest priority)

Decision matrix (priority order):
    1. user_override (if set + valid) → use that mode
    2. has_post_exploitation=True    → plan_execute
    3. scope_size > 1                → deep (parallel sub-agents)
    4. target_type="network_range"   → deep
    5. default                        → supervisor

Usage:
    from app.orchestration.mode_selector import select_mode, ModeSelectionInput

    selection = select_mode(ModeSelectionInput(
        target="http://example.com",
        target_type="single_url",
        scope_size=1,
        has_post_exploitation=False,
    ))
    print(selection.mode)  # OrchestratorMode.SUPERVISOR
    print(selection.reason)  # "Default mode for single-target scan"
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

from app.orchestration.base import OrchestratorMode

logger = logging.getLogger(__name__)


# ---------- Input/output dataclasses ----------

@dataclass
class ModeSelectionInput:
    """Input to select_mode(). All fields optional — defaults to supervisor.

    Attributes:
        target: Target URL or IP (used for inference when target_type is None)
        target_type: Explicit target type hint
        scope_size: Number of in-scope hosts (1 = single target, >1 = range)
        has_post_exploitation: Whether user wants full kill-chain
        user_override: Explicit mode string from user (highest priority)
                       — accepts "deep" / "plan_execute" / "supervisor"
    """
    target: str = ""
    target_type: str | None = None
    scope_size: int = 1
    has_post_exploitation: bool = False
    user_override: str | None = None


@dataclass
class ModeSelection:
    """Result of select_mode()."""
    mode: OrchestratorMode
    reason: str
    inferred_target_type: str | None = None
    user_override_applied: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode.value,
            "reason": self.reason,
            "inferred_target_type": self.inferred_target_type,
            "user_override_applied": self.user_override_applied,
        }


# ---------- Constants ----------

# CIDR regex (simple — matches a.b.c.d/nn)
_CIDR_RE = re.compile(
    r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}/\d{1,2}$"
)

# IP range regex (a.b.c.d-e.f.g.h or a.b.c.d/nn or comma-separated)
_IP_RANGE_RE = re.compile(
    r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}[-/,]"
)


# ---------- Target type inference ----------

def infer_target_type(target: str, scope_size: int = 1) -> str:
    """Infer target_type from the target string + scope_size.

    Returns one of:
        "single_url"               — http(s)://example.com[/path]
        "single_ip"                — 192.168.1.1
        "network_range"            — 192.168.1.0/24 or 192.168.1.1-50
        "domain_with_subdomains"   — example.com with scope_size > 1
        "api_endpoint"             — http(s)://example.com/api/*

    For ambiguous cases (e.g., bare domain), defaults to "single_url".
    """
    target = target.strip()
    if not target:
        return "single_url"

    # CIDR or IP range → network_range
    if _CIDR_RE.match(target) or _IP_RANGE_RE.match(target):
        return "network_range"

    # URL parse
    if target.startswith(("http://", "https://")):
        try:
            parsed = urlparse(target)
            path = parsed.path or ""
            if path.startswith("/api/") or path == "/api":
                return "api_endpoint"
        except Exception:
            pass
        # If scope_size > 1, user probably provided a domain + subdomains list
        if scope_size > 1:
            return "domain_with_subdomains"
        return "single_url"

    # Bare IP
    if re.match(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$", target):
        return "single_ip"

    # Bare domain — assume single_url
    if scope_size > 1:
        return "domain_with_subdomains"
    return "single_url"


# ---------- Mode selector ----------

def select_mode(inp: ModeSelectionInput) -> ModeSelection:
    """Decide which orchestration mode to use.

    Decision priority (highest first):
        1. user_override (if valid)
        2. has_post_exploitation=True → plan_execute
        3. target_type="network_range" OR scope_size > 1 → deep
        4. default → supervisor

    Args:
        inp: ModeSelectionInput with target + scope info.

    Returns:
        ModeSelection with chosen mode + human-readable reason.
    """
    # Step 1: User override (highest priority)
    if inp.user_override:
        override_str = inp.user_override.strip().lower()
        try:
            mode = OrchestratorMode(override_str)
            return ModeSelection(
                mode=mode,
                reason=f"User override: explicit mode='{override_str}' requested",
                inferred_target_type=inp.target_type or infer_target_type(inp.target, inp.scope_size),
                user_override_applied=True,
            )
        except ValueError:
            logger.warning(
                "Invalid user_override=%r — falling through to auto-selection. "
                "Valid values: deep, plan_execute, supervisor.",
                inp.user_override,
            )

    # Step 2: Infer target_type if not provided
    target_type = inp.target_type or infer_target_type(inp.target, inp.scope_size)

    # Step 3: has_post_exploitation → plan_execute
    if inp.has_post_exploitation:
        return ModeSelection(
            mode=OrchestratorMode.PLAN_EXECUTE,
            reason=(
                "full_kill_chain or has_post_exploitation requested → Plan-Execute "
                "(planner → executor → replanner loop for full exploitation chain)"
            ),
            inferred_target_type=target_type,
        )

    # Step 4: network_range or multi-host scope → deep
    if target_type == "network_range" or inp.scope_size > 1:
        return ModeSelection(
            mode=OrchestratorMode.DEEP,
            reason=(
                f"target_type={target_type!r} scope_size={inp.scope_size} → Deep "
                f"(parallel sub-agents for multi-host scan)"
            ),
            inferred_target_type=target_type,
        )

    # Step 5: Default → supervisor
    return ModeSelection(
        mode=OrchestratorMode.SUPERVISOR,
        reason=(
            f"Default mode for target_type={target_type!r} scope_size={inp.scope_size} "
            f"→ Supervisor (transfer mechanism, best for single-target web app)"
        ),
        inferred_target_type=target_type,
    )


# ---------- Convenience function ----------

def select_mode_simple(
    target: str,
    scope_size: int = 1,
    has_post_exploitation: bool = False,
    user_override: str | None = None,
) -> OrchestratorMode:
    """Simplified selector — returns just the OrchestratorMode enum.

    For full selection metadata (reason, inferred type), use select_mode().
    """
    selection = select_mode(ModeSelectionInput(
        target=target,
        scope_size=scope_size,
        has_post_exploitation=has_post_exploitation,
        user_override=user_override,
    ))
    return selection.mode