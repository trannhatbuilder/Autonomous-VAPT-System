"""
Reward shaping for VAPT-AI pentest RL.

Python port of EVVO Sentinel shield_engine/rl/reward.py, adapted to implement
master plan D26 reward shaping (10 scenarios) while preserving EVVO's
multi-signal architecture (rule / human / replay weights).

D26 Reward Shaping (master plan §12 W16):
    +5   confirmed exploit Tier-1 (shell obtained via C2Session)
    +3   confirmed exploit Tier-2 (partial PoC — e.g. SQLi data dump)
    +1   confirmed finding (verifier accepts)
    +0.5 successful recon (asset discovered)
    -1   timeout / no useful output
    -2   verifier-rejected FP
    -3   scope violation (blocked by scope guard)
    -5   user_aborted (HITL denied)
    -10  destructive op without HITL (should never happen — failsafe)
    +10  scan completed successfully (terminal bonus)

Multi-signal weights (preserved from EVVO — break self-confirm loop):
    WEIGHT_RULE_VERIFIER     = 1.0  (weakest — agent-coupled)
    WEIGHT_HUMAN_LABEL       = 3.0
    WEIGHT_REPLAY_REPRODUCED = 5.0  (strongest — external signal)

The final reward is:
    R = D26_base_signal × multi_signal_weight + R_explore + R_efficiency
    clipped to [-100, 150]
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# ── D26 Reward constants (master plan §12 W16) ───────────────────────────
# These are the BASE rewards. Multi-signal weights apply on top.
R_EXPLOIT_TIER1: float = 5.0       # shell obtained via C2Session
R_EXPLOIT_TIER2: float = 3.0       # partial PoC (e.g. SQLi data dump)
R_VERIFIED_FINDING: float = 1.0    # verifier-accepted finding
R_RECON_SUCCESS: float = 0.5       # asset discovered
R_TIMEOUT: float = -1.0            # no useful output / timeout
R_FP_REJECTED: float = -2.0        # verifier-rejected false positive
R_SCOPE_VIOLATION: float = -3.0    # scope guard blocked
R_USER_ABORTED: float = -5.0       # HITL denied
R_DESTRUCTIVE_NO_HITL: float = -10.0  # destructive op without HITL (failsafe)
R_SCAN_COMPLETE: float = 10.0      # terminal bonus

# EVVO multi-signal weights (preserved for FP penalty + finding bonus)
R_NOVEL_FINDING: float = 20.0      # CVE/vector not seen before in KG
R_REDUNDANT_FINDING: float = 1.0   # finding already known, still useful
R_WASTED_TURN: float = -0.5        # empty/error output
R_EFFICIENT: float = 0.2           # small bonus per turn for progress
R_TOKEN_PENALTY_PER_1K: float = -0.005
R_FOLLOWED_VERIFIED: float = 0.5   # RL guidance followed + verified
R_FOLLOWED_NO_FINDING: float = -0.1  # RL guidance followed, no finding

WEIGHT_RULE_VERIFIER: float = 1.0
WEIGHT_HUMAN_LABEL: float = 3.0
WEIGHT_REPLAY_REPRODUCED: float = 5.0

# Reward clip bounds (wider than EVVO because D26 +10/-10 + multi-signal
# 5x can produce large magnitudes — clipping prevents Q-network instability).
R_MIN_CLIP: float = -100.0
R_MAX_CLIP: float = 150.0


def compute_reward(
    session: dict[str, Any],
    previous_findings_count: int,
    previous_verified_count: int,
    command_output: str,
    tokens_used_delta: int,
    turn: int,
    is_done: bool = False,
    action_followed: bool | None = None,
    turn_events: list[dict[str, Any]] | None = None,
) -> tuple[float, dict[str, Any]]:
    """
    Compute reward for a single turn transition per D26 reward shaping.

    Args:
        session: in-memory session dict (with `findings`, `recon_signals`, etc.)
        previous_findings_count: len(findings) before this turn
        previous_verified_count: verified findings before this turn
        command_output: raw stdout from the executed command
        tokens_used_delta: tokens consumed this turn
        turn: current turn number
        is_done: whether the scan completed this turn
        action_followed: did the agent follow RL guidance this turn?
        turn_events: list of event dicts from this turn. Each event has a
            `type` field that triggers a D26 reward:
              - "exploit_tier1"   → +5  (shell obtained)
              - "exploit_tier2"   → +3  (partial PoC)
              - "recon_success"   → +0.5 (asset discovered)
              - "timeout"         → -1
              - "fp_rejected"     → -2
              - "scope_violation" → -3
              - "user_aborted"    → -5
              - "destructive_no_hitl" → -10

    Returns:
        (reward, breakdown) — both the scalar reward and a dict breakdown
        for logging / debugging / replay trace.
    """
    findings = session.get("findings") or []
    current_count = len(findings)
    current_verified = sum(
        1 for f in findings
        if f.get("verified") and not f.get("false_positive")
    )
    current_fp = sum(1 for f in findings if f.get("false_positive"))

    reward = 0.0
    breakdown: dict[str, Any] = {
        "d26_signals": {},
        "finding_bonus": 0.0,
        "fp_penalty": 0.0,
        "wasted_turn": 0.0,
        "token_penalty": 0.0,
        "efficiency_bonus": 0.0,
        "completion_bonus": 0.0,
        "rl_follow_bonus": 0.0,
        "total": 0.0,
    }

    # ── 1. D26 event-based signals (turn_events) ──────────────────────
    if turn_events:
        for event in turn_events:
            evt_type = event.get("type")
            if evt_type == "exploit_tier1":
                reward += R_EXPLOIT_TIER1
                breakdown["d26_signals"]["exploit_tier1"] = R_EXPLOIT_TIER1
            elif evt_type == "exploit_tier2":
                reward += R_EXPLOIT_TIER2
                breakdown["d26_signals"]["exploit_tier2"] = R_EXPLOIT_TIER2
            elif evt_type == "recon_success":
                reward += R_RECON_SUCCESS
                breakdown["d26_signals"]["recon_success"] = R_RECON_SUCCESS
            elif evt_type == "timeout":
                reward += R_TIMEOUT
                breakdown["d26_signals"]["timeout"] = R_TIMEOUT
            elif evt_type == "fp_rejected":
                reward += R_FP_REJECTED
                breakdown["d26_signals"]["fp_rejected"] = R_FP_REJECTED
            elif evt_type == "scope_violation":
                reward += R_SCOPE_VIOLATION
                breakdown["d26_signals"]["scope_violation"] = R_SCOPE_VIOLATION
            elif evt_type == "user_aborted":
                reward += R_USER_ABORTED
                breakdown["d26_signals"]["user_aborted"] = R_USER_ABORTED
            elif evt_type == "destructive_no_hitl":
                reward += R_DESTRUCTIVE_NO_HITL
                breakdown["d26_signals"]["destructive_no_hitl"] = R_DESTRUCTIVE_NO_HITL

    # ── 2. Finding bonus (multi-signal) ───────────────────────────────
    # Score each newly-verified finding individually for novelty.
    if current_verified > previous_verified_count:
        prev_credited_ids = session.setdefault("_credited_verified_ids", set())
        new_bonus = 0.0
        new_count = 0
        for f in findings:
            if not f.get("verified") or f.get("false_positive"):
                continue
            fid = _finding_identity(f)
            if fid in prev_credited_ids:
                continue
            prev_credited_ids.add(fid)
            new_count += 1
            novel = _is_novel_finding(session, f)
            # D26 base = R_VERIFIED_FINDING (+1); novel bumps to R_NOVEL_FINDING (+20)
            base = R_NOVEL_FINDING if novel else R_VERIFIED_FINDING
            new_bonus += base * _signal_weight(f)
        reward += new_bonus
        breakdown["finding_bonus"] = new_bonus
        if new_count:
            logger.debug("reward: +%.1f for %d verified finding(s)", new_bonus, new_count)

    # ── 3. False positive penalty (multi-signal) ──────────────────────
    prev_fp = session.get("_prev_fp_count", 0)
    if current_fp > prev_fp:
        new_fps = current_fp - prev_fp
        fp_penalty = 0.0
        for f in findings[-new_fps:]:
            mult = WEIGHT_HUMAN_LABEL if f.get("false_positive_at") else 1.0
            fp_penalty += R_FP_REJECTED * mult
        reward += fp_penalty
        breakdown["fp_penalty"] = fp_penalty
        session["_prev_fp_count"] = current_fp

    # ── 4. Wasted turn penalty (empty/error output) ───────────────────
    output_len = len((command_output or "").strip())
    if output_len < 50 and not is_done and not turn_events:
        reward += R_WASTED_TURN
        breakdown["wasted_turn"] = R_WASTED_TURN

    # ── 5. Token efficiency penalty (per 1k tokens) ───────────────────
    token_pen = (tokens_used_delta / 1000.0) * R_TOKEN_PENALTY_PER_1K
    reward += token_pen
    breakdown["token_penalty"] = token_pen

    # ── 6. Small survival bonus (encourages scan progress) ────────────
    reward += R_EFFICIENT
    breakdown["efficiency_bonus"] = R_EFFICIENT

    # ── 7. Completion bonus (D26 +10) ─────────────────────────────────
    if is_done and current_verified > 0:
        reward += R_SCAN_COMPLETE
        breakdown["completion_bonus"] = R_SCAN_COMPLETE

    # ── 8. RL guidance follow bonus/penalty ───────────────────────────
    if action_followed is True:
        if current_verified > previous_verified_count:
            reward += R_FOLLOWED_VERIFIED
            breakdown["rl_follow_bonus"] = R_FOLLOWED_VERIFIED
        else:
            reward += R_FOLLOWED_NO_FINDING
            breakdown["rl_follow_bonus"] = R_FOLLOWED_NO_FINDING

    # ── 9. Clip ───────────────────────────────────────────────────────
    clipped = float(max(R_MIN_CLIP, min(R_MAX_CLIP, reward)))
    breakdown["total"] = clipped
    breakdown["pre_clip"] = float(reward)
    breakdown["clip_bounds"] = [R_MIN_CLIP, R_MAX_CLIP]

    return clipped, breakdown


# ── Multi-signal helpers (preserved from EVVO) ───────────────────────────
def _signal_weight(finding: dict[str, Any]) -> float:
    """Pick the strongest ground-truth signal weight present on a finding.

    Priority (highest first):
      1. replay_reproduced=True   → WEIGHT_REPLAY_REPRODUCED (5x)
      2. human_verified=True      → WEIGHT_HUMAN_LABEL (3x)
      3. verified=True (rule)     → WEIGHT_RULE_VERIFIER (1x)

    Replay-reproduced = offline replay of the agent's verification command
    produced a matching observation. Human-verified = user clicked /verify
    in the UI. Rule-verified = weakest (coupled to agent-supplied evidence).
    """
    if finding.get("replay_reproduced"):
        return WEIGHT_REPLAY_REPRODUCED
    if finding.get("human_verified"):
        return WEIGHT_HUMAN_LABEL
    return WEIGHT_RULE_VERIFIER


def _finding_identity(finding: dict[str, Any]) -> str:
    """Stable identity for credit-tracking: name + location + severity."""
    return (
        f"{finding.get('name', '')}|"
        f"{finding.get('location', '')}|"
        f"{finding.get('severity', '')}"
    )


def _is_novel_finding(session: dict[str, Any], finding: dict[str, Any]) -> bool:
    """Heuristic: a finding is novel if its name+location combo hasn't been
    seen in this session.
    """
    if not finding:
        return False
    key = f"{finding.get('name', '')}:{finding.get('location', '')}"
    seen = session.get("_seen_finding_keys", set())
    if key in seen:
        return False
    seen.add(key)
    session["_seen_finding_keys"] = seen
    return True


# ── Convenience: D26 event constructor ───────────────────────────────────
def d26_event(event_type: str, **extra: Any) -> dict[str, Any]:
    """Construct a D26 reward event for compute_reward(turn_events=[...]).

    Args:
        event_type: one of:
            exploit_tier1, exploit_tier2, recon_success, timeout,
            fp_rejected, scope_violation, user_aborted, destructive_no_hitl
        **extra: additional metadata (e.g. tool_name, target, c2_session_id)

    Returns:
        event dict suitable for passing in turn_events list.
    """
    valid_types = {
        "exploit_tier1", "exploit_tier2", "recon_success", "timeout",
        "fp_rejected", "scope_violation", "user_aborted", "destructive_no_hitl",
    }
    if event_type not in valid_types:
        raise ValueError(
            f"Invalid D26 event type: {event_type!r}. Valid: {sorted(valid_types)}"
        )
    return {"type": event_type, **extra}