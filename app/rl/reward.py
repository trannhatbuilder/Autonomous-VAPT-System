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

The final reward is a SUM of independent terms — not one base signal scaled
by a weight, and NOT anything to do with exploration:

    R = Σ D26_event_base(turn_events)                     # §1
      + Σ finding_base × signal_weight                    # §2
      + Σ R_FP_REJECTED × signal_weight                   # §3
      + R_WASTED_TURN                                     # §4
      + (tokens_used_delta / 1000) × R_TOKEN_PENALTY_PER_1K  # §5
      + R_EFFICIENT                                       # §6
      + R_SCAN_COMPLETE                                   # §7
      + R_FOLLOWED_VERIFIED | R_FOLLOWED_NO_FINDING       # §8
    clipped to [R_MIN_CLIP, R_MAX_CLIP]

`R_explore` is deliberately absent because it does not exist. Intrinsic
curiosity lives in `app/rl/policy.py::CuriosityModule` and acts on ε (resp.
temperature) at action-selection time; it never touches this function. An
earlier version of this docstring advertised `+ R_explore + R_efficiency`,
which sent readers looking for a reward term nobody wrote.
"""
from __future__ import annotations

import logging
from typing import Any

from app.evidence.finding_state import (
    count_verified_findings,
    finding_identity,
    finding_novelty_key,
    is_verified_finding,
)

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
R_EFFICIENT: float = 0.2           # productive turn (NOT a per-turn survival stipend)
R_TOKEN_PENALTY_PER_1K: float = -0.005

# A turn "produced output" once its stripped command output reaches this
# many characters. Shared by the wasted-turn penalty (§4) and the
# efficiency bonus (§6) so the two can never disagree — they used to be a
# literal 50 and an unconditional bonus, which is how a turn could be
# simultaneously "too empty to count" and "efficient enough to pay".
USEFUL_OUTPUT_MIN_CHARS: int = 50
R_FOLLOWED_VERIFIED: float = 0.5   # RL guidance followed + verified
R_FOLLOWED_NO_FINDING: float = -0.1  # RL guidance followed, no finding

WEIGHT_RULE_VERIFIER: float = 1.0
WEIGHT_HUMAN_LABEL: float = 3.0
WEIGHT_REPLAY_REPRODUCED: float = 5.0

# Reward clip bounds (wider than EVVO because D26 +10/-10 + multi-signal
# 5x can produce large magnitudes — clipping prevents Q-network instability).
R_MIN_CLIP: float = -100.0
R_MAX_CLIP: float = 150.0

# Single source of truth for event pricing. §1 and `d26_event()` both read
# this table, so the two lists can no longer drift: previously the 8 types
# were spelled out TWICE (an if/elif chain in §1 and a `valid_types` set in
# the constructor), which made it possible to accept an event in `d26_event()`
# and then drop it without a trace inside `compute_reward()`.
D26_EVENT_REWARDS: dict[str, float] = {
    "exploit_tier1": R_EXPLOIT_TIER1,
    "exploit_tier2": R_EXPLOIT_TIER2,
    "recon_success": R_RECON_SUCCESS,
    "timeout": R_TIMEOUT,
    "fp_rejected": R_FP_REJECTED,
    "scope_violation": R_SCOPE_VIOLATION,
    "user_aborted": R_USER_ABORTED,
    "destructive_no_hitl": R_DESTRUCTIVE_NO_HITL,
}

# ── Turn-to-turn credit bookkeeping ──────────────────────────────────────
# `compute_reward()` keeps three pieces of state describing what has ALREADY
# been paid out:
#
#   _credited_verified_ids : finding identities already paid the bonus
#   _seen_finding_keys     : novelty heuristic memory
#   _penalised_fp_ids      : finding identities already penalised as FPs
#
# These belong to the SCAN, not to one turn. The agent loop calls
# `build_session()` every turn and that returns a FRESH dict, so storing
# them on the session resets them constantly. The consequences are not
# cosmetic: a previously-credited finding gets paid again on any turn where
# the gate re-opens, the SAME false positive is penalised on every remaining
# turn, and `_is_novel_finding()` reports everything as novel (so every
# finding earns the +20 novelty tier instead of +1).
#
# Callers whose session is rebuilt per turn therefore own the state and pass
# it in explicitly:
#     credit = new_credit_state()          # once, at the top of the scan
#     reward, breakdown = compute_reward(..., credit_state=credit)
REWARD_CREDIT_KEYS: tuple[str, ...] = (
    "_credited_verified_ids",
    "_seen_finding_keys",
    "_penalised_fp_ids",
)


def new_credit_state() -> dict[str, Any]:
    """Fresh scan-lifetime reward bookkeeping container.

    Returned fully populated (three empty sets) so that a caller can dump
    it or assert on its shape without special-casing first use.
    """
    return {
        "_credited_verified_ids": set(),
        "_seen_finding_keys": set(),
        "_penalised_fp_ids": set(),
    }


def compute_reward(
    session: dict[str, Any],
    previous_verified_count: int,
    command_output: str,
    tokens_used_delta: int,
    is_done: bool = False,
    action_followed: bool | None = None,
    turn_events: list[dict[str, Any]] | None = None,
    credit_state: dict[str, Any] | None = None,
) -> tuple[float, dict[str, Any]]:
    """
    Compute reward for a single turn transition per D26 reward shaping.

    Args:
        session: in-memory session dict (with `findings`, `recon_signals`, etc.)
        previous_verified_count: verified findings before this turn. Verified
            COUNT, not a findings count: §2 pays for confirmed discoveries and
            §8 asks "did this turn confirm anything new?", so a raw findings
            count can never answer either question (it cannot tell a confirmed
            SQLi from a rejected FP) and previously sat unused in the signature.
        command_output: raw stdout from the executed command
        tokens_used_delta: tokens consumed this turn
        is_done: whether the scan completed this turn
        action_followed: did the agent follow RL guidance this turn?
        turn_events: list of event dicts from this turn. `type` must be a key
            of `D26_EVENT_REWARDS`; anything else is logged at DEBUG level and
            does not pay (see §1 — dropping unknown types silently made a
            malformed turn indistinguishable from a positive one).
        credit_state: scan-lifetime payout memory (see `new_credit_state`).
            MUST be passed by any caller that rebuilds `session` every turn
            — `build_session()` returns a fresh dict, so state kept on the
            session is lost and old findings get paid again. When omitted
            (single-shot callers, tests), credit state lives on `session`,
            which is correct as long as one session == one whole episode.

    Returns:
        (reward, breakdown) — both the scalar reward and a dict breakdown
        for logging / debugging / replay trace.
    """
    # Payout memory: prefer the caller-owned scan-lifetime dict. Falling
    # back to `session` keeps single-shot callers working, but a caller
    # that rebuilds the session each turn and forgets this argument
    # silently re-pays credited findings — hence the AST guard in
    # tests/unit/test_rl_reward_baseline.py.
    credit = credit_state if credit_state is not None else session

    findings = session.get("findings") or []
    # One definition of "verified", shared with the agent loop's
    # `prev_verified_count` and the `last_turn_success` feature. An inline
    # `f.get("verified") and not f.get("false_positive")` here was the
    # second copy of the rule and the loop's was the third.
    current_verified = count_verified_findings(findings)

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
    # Driven by `D26_EVENT_REWARDS`, the same table `d26_event()` validates
    # against. The old if/elif chain had no `else`, so a typo'd or future
    # event type vanished without a trace — and because §4/§6 gate on
    # `not turn_events`, that turn still looked "eventful" while paying
    # nothing, which is worse than either paying or ignoring it cleanly.
    if turn_events:
        for event in turn_events:
            evt_type = event.get("type") if isinstance(event, dict) else None
            base = D26_EVENT_REWARDS.get(evt_type)
            if base is None:
                logger.debug(
                    "reward: unknown D26 event type %r dropped "
                    "(valid: %s)", evt_type, sorted(D26_EVENT_REWARDS),
                )
                continue
            reward += base
            breakdown["d26_signals"][evt_type] = base

    # ── 2. Finding bonus (multi-signal) ───────────────────────────────
    # Score each newly-verified finding individually for novelty.
    if current_verified > previous_verified_count:
        prev_credited_ids = credit.setdefault("_credited_verified_ids", set())
        new_bonus = 0.0
        new_count = 0
        for f in findings:
            # Same predicate as the gate above, not a second inline copy:
            # the previous `f.get("verified")` filter skipped findings that
            # carried only the canonical `verification` level, so the turn
            # passed the gate (count went up) and then paid +0 with no log.
            if not is_verified_finding(f):
                continue
            fid = _finding_identity(f)
            if fid in prev_credited_ids:
                continue
            prev_credited_ids.add(fid)
            new_count += 1
            novel = _is_novel_finding(credit, f)
            # D26 base = R_VERIFIED_FINDING (+1); novel bumps to R_NOVEL_FINDING (+20)
            base = R_NOVEL_FINDING if novel else R_VERIFIED_FINDING
            new_bonus += base * _signal_weight(f)
        reward += new_bonus
        breakdown["finding_bonus"] = new_bonus
        if new_count:
            logger.debug("reward: +%.1f for %d verified finding(s)", new_bonus, new_count)

    # ── 3. False positive penalty (multi-signal) ──────────────────────
    # Attribution is by IDENTITY, never by list position. The previous
    # version sliced `findings[-new_fps:]`, which assumes the newly
    # flagged FPs are the LAST entries — false whenever a human marks an
    # older finding (the normal case: the label arrives after the scan)
    # or the Phase-5 auditor rejects something recorded turns ago. The
    # slice then charged the penalty to whichever findings happened to
    # sit at the tail, so the 3x human-label multiplier landed on the
    # wrong record and was lost (-2.0 instead of -6.0 in the repro).
    #
    # Selecting on `false_positive` (NOT on `false_positive_at`) is also
    # deliberate: `false_positive_at` is the human-label TIMESTAMP, used
    # only to pick the multiplier. Rule-based rejections (scan_pipeline
    # writes false_positive=True with no timestamp) would be invisible to
    # a timestamp filter — a "fix" that filters on it silently restores
    # a permanently-zero penalty.
    #
    # Identity-keyed (not count-keyed) also removes a stateful hole: the
    # old `_prev_fp_count` was assigned INSIDE the `current_fp > prev_fp`
    # branch, so once a label was removed the counter stuck above the live
    # count and every later FP became free.
    penalised_fps = credit.setdefault("_penalised_fp_ids", set())
    fp_penalty = 0.0
    for f in findings:
        if not f.get("false_positive"):
            continue
        fid = _finding_identity(f)
        if fid in penalised_fps:
            continue
        penalised_fps.add(fid)
        mult = WEIGHT_HUMAN_LABEL if f.get("false_positive_at") else 1.0
        fp_penalty += R_FP_REJECTED * mult
    if fp_penalty:
        reward += fp_penalty
        breakdown["fp_penalty"] = fp_penalty

    # ── 4. Wasted turn penalty (empty/error output) ───────────────────
    # `USEFUL_OUTPUT_MIN_CHARS` is shared with §6 below: one definition of
    # "this turn produced something", used by both halves.
    output_len = len((command_output or "").strip())
    has_useful_output = output_len >= USEFUL_OUTPUT_MIN_CHARS
    # A turn is PRODUCTIVE only if it produced output AND nothing marked it
    # as a failure. `turn_events` currently carries negatives only (timeout /
    # scope_violation), and the positives are already priced by §1 — so an
    # eventful turn must never also collect the flat stipend.
    productive_turn = has_useful_output and not turn_events and not is_done
    if not has_useful_output and not is_done and not turn_events:
        reward += R_WASTED_TURN
        breakdown["wasted_turn"] = R_WASTED_TURN

    # ── 5. Token efficiency penalty (per 1k tokens) ───────────────────
    token_pen = (tokens_used_delta / 1000.0) * R_TOKEN_PENALTY_PER_1K
    reward += token_pen
    breakdown["token_penalty"] = token_pen

    # ── 6. Productive-turn bonus (NOT a survival stipend) ─────────────
    # This used to be unconditional `reward += R_EFFICIENT`. Two problems:
    #
    #   * An empty turn NETTED -0.3 (-0.5 + 0.2), so the flat bonus was
    #     cancelling 40% of the very penalty it travelled with. A 100-turn
    #     stall with long-but-worthless output collected +20 — exactly one
    #     novel finding — while the agent learned nothing.
    #   * It paid on turns that were explicitly flagged as failures.
    #
    # Gating on `productive_turn` restores the intended contrast: an empty
    # turn scores -0.5 vs +0.2 for a productive one (a 0.7 spread instead
    # of 0.5), and a stalled turn no longer draws a salary. The terminal
    # turn is excluded too: it carries `command_output=""` by construction
    # and is rewarded by §7 instead.
    if productive_turn:
        reward += R_EFFICIENT
        breakdown["efficiency_bonus"] = R_EFFICIENT

    # ── 7. Completion bonus (D26 +10) ─────────────────────────────────
    if is_done and current_verified > 0:
        reward += R_SCAN_COMPLETE
        breakdown["completion_bonus"] = R_SCAN_COMPLETE

    # ── 8. RL guidance follow bonus/penalty ───────────────────────────
    # ADVISORY signal, not an environment outcome: this pays for "the LLM did
    # what the policy suggested", which is only evidence of agreement between
    # two components of the same system. It is kept because the magnitudes are
    # deliberately small (±0.5 / -0.1) next to §1–§7, and because the positive
    # branch is gated on a NEW verified finding — it can only fire on a turn
    # where §2 already paid. The flat -0.1 is the only per-turn component here
    # and caps out at -10 per 100 turns, i.e. it can undo at most one terminal
    # bonus. When reading a reward trace: +0.5 here means "advisor pleased",
    # NOT "target compromised".
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
    """Stable identity for credit-tracking: title + location + severity.

    Delegates to `app.evidence.finding_state` — this used to read `name`
    while the ReAct loop wrote `title`, so every finding of a given
    severity shared the identity `"||high"` and only the first was paid.
    Do not re-implement the alias handling here.
    """
    return finding_identity(finding)


def _is_novel_finding(credit: dict[str, Any], finding: dict[str, Any]) -> bool:
    """Heuristic: a finding is novel if its title+location combo hasn't been
    seen in this episode.

    Takes the credit dict rather than the session on purpose: novelty
    memory has to be as long-lived as payout memory, otherwise everything
    looks novel on every turn and each finding earns +20 instead of +1.
    """
    if not finding:
        return False
    key = finding_novelty_key(finding)
    seen = credit.get("_seen_finding_keys", set())
    if key in seen:
        return False
    seen.add(key)
    credit["_seen_finding_keys"] = seen
    return True


# ── Convenience: D26 event constructor ───────────────────────────────────
def d26_event(event_type: str, **extra: Any) -> dict[str, Any]:
    """Construct a D26 reward event for compute_reward(turn_events=[...]).

    Args:
        event_type: a key of `D26_EVENT_REWARDS`:
            exploit_tier1, exploit_tier2, recon_success, timeout,
            fp_rejected, scope_violation, user_aborted, destructive_no_hitl
        **extra: additional metadata (e.g. tool_name, target, c2_session_id)

    Returns:
        event dict suitable for passing in turn_events list.

    Raises:
        ValueError: the type has no reward entry. Validation reads the same
            table §1 prices from, so an accepted event is always paid.
    """
    if event_type not in D26_EVENT_REWARDS:
        raise ValueError(
            f"Invalid D26 event type: {event_type!r}. "
            f"Valid: {sorted(D26_EVENT_REWARDS)}"
        )
    return {"type": event_type, **extra}