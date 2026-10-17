"""Canonical in-memory finding dict — one producer, one key set.

WHY THIS MODULE EXISTS
----------------------
Three places independently defined what a "finding" is:

  1. ``app/agents/react_agent.py`` builds the dict after a successful
     ``record_vulnerability`` call,
  2. ``app/rl/tda.py`` scores that dict on the evidence ladder,
  3. ``app/rl/reward.py`` derives payout identity and novelty from it.

Each review round found a new drift between those three, and every drift
was silent — a missing key makes ``dict.get()`` return ``""``, not raise:

  * the agent stored ``title``, but ``reward._finding_identity`` read
    ``name`` → every finding of a given severity collapsed to the SAME
    identity (``"||high"``), so only the first one was ever paid;
  * ``_is_novel_finding`` read ``name``/``location`` → every novelty key
    collapsed to ``":"``, so every finding after the first fell from the
    +20 novelty tier to +1;
  * ``findings.append`` had ``location`` available in ``tool_args`` (the
    tool schema declares it) but did not store it.

So: ``finding_to_state()`` is the ONLY place the dict is constructed, and
``finding_title()`` / ``finding_location()`` are the ONLY place field
aliases are resolved. Consumers must not re-implement either.

CONTRACT
--------
``FINDING_STATE_KEYS`` is the complete key set of a finding produced by
``finding_to_state()``. Tests assert that every key the TDA ladder, the
state encoder and the reward shaper read is in that set — so adding a
consumer-side read forces a producer-side decision.
"""
from __future__ import annotations

from typing import Any

from app.evidence.verifier import (
    VERIFICATION_CONFIRMED,
    VERIFICATION_PLAUSIBLE,
    VERIFICATION_REPLAY_VERIFIED,
    parse_verification,
    verification_rank,
)

# ── Field aliases ────────────────────────────────────────────────────────
# Ordered by preference. `title` is the tool-schema name (and what the
# report/UI shows); `name` is the legacy/reward-side spelling. Both must
# resolve, because findings also arrive from replay traces and tests.
TITLE_KEYS: tuple[str, ...] = ("title", "name")

# `location` is NOT in the tool schema's `required` list, so the LLM often
# omits it; `target` is required. `url`/`endpoint` are legacy spellings
# that other call sites used. Falling back matters: an empty location
# collapses every finding of a severity into one payout identity.
LOCATION_KEYS: tuple[str, ...] = ("location", "url", "endpoint", "target")

# Every key emitted by finding_to_state(). Keep in sync — the contract
# test in tests/unit/test_finding_state.py enforces it.
FINDING_STATE_KEYS: tuple[str, ...] = (
    "title",
    "location",
    "severity",
    "vuln_type",
    "verification",
    "verification_reason",
    "verified",
    "replay_reproduced",
    "human_verified",
    "false_positive",
    "false_positive_at",
)


def _first_nonempty_str(finding: dict[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = finding.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


# ── Tolerant accessors (the only place aliases are resolved) ─────────────
def finding_title(finding: dict[str, Any]) -> str:
    """Human title of a finding, tolerating `name` as a legacy alias."""
    if not isinstance(finding, dict):
        return ""
    return _first_nonempty_str(finding, TITLE_KEYS)


def finding_location(finding: dict[str, Any]) -> str:
    """Endpoint/URL of a finding, tolerating legacy spellings."""
    if not isinstance(finding, dict):
        return ""
    return _first_nonempty_str(finding, LOCATION_KEYS)


def finding_severity(finding: dict[str, Any]) -> str:
    """Lower-cased severity. Normalised so `High` and `high` are one rung."""
    if not isinstance(finding, dict):
        return ""
    value = finding.get("severity")
    return value.strip().lower() if isinstance(value, str) else ""


def finding_identity(finding: dict[str, Any]) -> str:
    """Stable payout identity: title + location + severity.

    Used by `app.rl.reward` to make sure a finding is paid at most once.
    Must stay specific enough to keep two DIFFERENT findings apart — an
    identity that collapses makes the reward silently skip every finding
    after the first of its severity.
    """
    return "|".join(
        (finding_title(finding), finding_location(finding), finding_severity(finding))
    )


def finding_novelty_key(finding: dict[str, Any]) -> str:
    """Key for the "have we seen this before?" heuristic.

    Deliberately coarser than `finding_identity` (no severity): the same
    issue re-reported at a different severity is the SAME issue, so it
    should not earn the novelty tier twice.
    """
    return f"{finding_title(finding)}:{finding_location(finding)}"


# ── The producer ─────────────────────────────────────────────────────────
def is_verified_finding(finding: dict[str, Any]) -> bool:
    """True when the VERIFIER accepted this finding and nobody refuted it.

    This is THE one definition of "verified" for RL. `reward.py` (§2/§7/§8),
    the ReAct loop's `prev_verified_count`, the `last_turn_success` feature,
    the TDA `successes` count and the state encoder's `verified_findings`
    all have to agree: they ask the same question — "did this turn confirm
    something?" — and when they disagree the RL state contradicts its own
    reward (a turn that paid +20 logging zero verified findings, or vice
    versa). Nothing may re-implement this rule inline.

    Resolution order — CANONICAL FIELD FIRST, legacy booleans as fallback:

      0. ``false_positive`` truthy -> False
         A refutation outranks every positive signal, including a stale
         ``verified: True`` left behind before the Phase-5 auditor ran.
      1. ``verification`` is a known level -> rank >= rank("confirmed")
         ``verification`` is the canonical provenance field written by
         ``app/evidence/verifier.py``. When present it WINS over the legacy
         booleans — otherwise ``verification="plausible"`` plus a stale
         ``verified: True`` would re-promote a finding the verifier only
         rated plausible.
      2. ``replay_reproduced`` / ``human_verified`` truthy -> True
      3. ``verified`` truthy -> True (legacy spelling; still accepted
         because findings also arrive from replay traces)
      4. otherwise -> False

    The rank thresholds are imported from the verifier, so this predicate
    and ``tda.evidence_score_from_finding() >= CONFIRMED_VULN`` are the same
    rule in two tables — an equivalence pinned by
    tests/unit/test_evidence_single_source_of_truth.py.

    Do NOT "simplify" this back to ``bool(finding.get("verified"))``. That
    version silently counted zero findings for every producer that writes
    only the canonical ``verification`` field, which turns the state
    encoder's fifth scalar and the reward's finding bonus into dead
    features with no exception anywhere.
    """
    if not isinstance(finding, dict):
        return False
    if finding.get("false_positive"):
        return False
    # `verification_rank` returns 0 for None/unknown, which is exactly the
    # "fall through to the legacy flags" signal we want here: a malformed
    # level must not demote a finding that also carries a real boolean.
    rank = verification_rank(finding.get("verification"))
    if rank:
        return rank >= verification_rank(VERIFICATION_CONFIRMED)
    if finding.get("replay_reproduced") or finding.get("human_verified"):
        return True
    return bool(finding.get("verified"))


def count_verified_findings(findings: list[dict[str, Any]]) -> int:
    """Number of verifier-accepted, non-refuted findings.

    Counts VERIFIED findings, never findings. `len(findings)` also counts
    records the verifier only rated `plausible`, which pay nothing in
    `reward.py` §2 — using it as a success signal makes the state vector
    claim progress on a turn that earned no reward.
    """
    return sum(1 for f in findings if is_verified_finding(f))


def finding_to_state(
    tool_args: dict[str, Any] | None,
    result_json: dict[str, Any] | None = None,
    *,
    default_verification: str = VERIFICATION_PLAUSIBLE,
) -> dict[str, Any]:
    """Build the canonical in-memory finding from a `record_vulnerability` call.

    Args:
        tool_args: the arguments the LLM passed to `record_vulnerability`
            (title, severity, vuln_type, target, location, ...).
        result_json: the parsed JSON that `tool_bridge._record_vulnerability`
            returned. Only its VERDICT fields are trusted.
        default_verification: level to assume when the result carried no
            verdict at all. Defaults to `plausible` — a finding must never
            self-promote because the verdict was missing.

    Returns:
        A dict whose keys are exactly `FINDING_STATE_KEYS`.

    Provenance rules (do not "simplify" these):

      * `verification` comes from `result_json`, never from `tool_args`:
        the LLM must not be able to grade its own finding.
      * `verified` / `replay_reproduced` are DERIVED from `verification`.
        `result_json["verified"]` is the DB row state and is always False
        at insert time (Phase 5 flips it), so copying it would score every
        finding as unverified. `replay_reproduced` is not in the bridge's
        response at all — reading it directly would silently disable the
        5x replay weight in `reward._signal_weight`.
      * `false_positive` is False here by construction: the bridge only
        records findings that its verifier accepted. The Phase-5 auditor
        sets the DB column afterwards, which is why `false_positive_at`
        exists as an explicit (normally None) key.
    """
    args: dict[str, Any] = tool_args or {}
    result: dict[str, Any] = result_json or {}

    verification = parse_verification(
        result.get("verification"), default=default_verification
    )

    return {
        "title": finding_title(args),
        "location": finding_location(args),
        "severity": finding_severity(args),
        "vuln_type": _first_nonempty_str(args, ("vuln_type", "type")),
        "verification": verification,
        "verification_reason": str(result.get("verification_reason") or ""),
        # Derived from the verifier verdict — see the docstring.
        "verified": verification
        in (VERIFICATION_CONFIRMED, VERIFICATION_REPLAY_VERIFIED),
        "replay_reproduced": verification == VERIFICATION_REPLAY_VERIFIED,
        "human_verified": result.get("human_verified") is True,
        # Not set at record time; the post-scan auditor owns this.
        "false_positive": False,
        "false_positive_at": result.get("false_positive_at"),
    }


__all__ = [
    "FINDING_STATE_KEYS",
    "TITLE_KEYS",
    "LOCATION_KEYS",
    "finding_identity",
    "finding_location",
    "finding_novelty_key",
    "finding_severity",
    "finding_title",
    "finding_to_state",
    "is_verified_finding",
    "count_verified_findings",
]
