"""
VAPT-AI RL — Task Difficulty Assessment (TDA).

Implements the 4-dimensional Task Difficulty Index (TDI) used to gate
the agent's strategy each turn. The TDI is a scalar in [0, 1] that
measures how hard the current scan state is, computed from 4 dimensions:

    H — Horizon Estimation:   remaining steps / actual scan budget
    E — Evidence Confidence:  mean confidence over the current path
    C — Context Saturation:   prompt tokens of the LATEST request / model window
    S — Historical Success:   Laplace-smoothed success rate on this branch

TDI formula
-----------
    TDI = w_H * H + w_E * (1 - E) + w_C * C + w_S * (1 - S)

Default weights:
    w_H = w_E = 0.30
    w_C = w_S = 0.20

Mode mapping
------------
    TDI > 0.60  -> BROADEN   (expand reconnaissance — task is hard)
    TDI <= 0.60 -> LLM_PICK  (let the LLM decide)

EXPLOIT is an override-only state (see "Critical override" below), not
a formula threshold. The old `TDI < 0.30 -> EXPLOIT` rule was removed
because it was mathematically unreachable: with no confirmed finding,
E <= PLAUSIBLE_HYPOTHESIS (0.5) and S -> 0 as attempts grow, so

    TDI >= 0.30 * (1 - 0.5) + 0.20 * (1 - 0) = 0.35 > 0.30

and once a finding exists the override fires before the raw TDI is
consulted. Keeping that branch would only mislead log readers into
thinking EXPLOIT was ever selected by the formula.

Context dimension (C)
---------------------
C answers "how full is the model's context window RIGHT NOW?", which is
the quantity that actually degrades reasoning and forces pruning. It
is a SATURATION measure, not a spend measure:

    C = prompt_tokens(last request) / model_context_window

Two earlier definitions were wrong:

  1. `total_tokens_so_far / TOKEN_HARD_LIMIT` measured the SCAN BUDGET
     (money), not the model's context. It rose monotonically with the
     number of turns even when every prompt was tiny, so the inverse
     incentive appeared: a long, expensive scan pushed C up, pushed TDI
     up past 0.60, and forced BROADEN — broadening recon exactly when
     the scan was already burning the most budget.
  2. Dividing by `TOKEN_HARD_LIMIT` (10M) instead of the model's real
     window (e.g. 128K) also compressed C to ~0.002 at 20K tokens, i.e.
     the dimension was effectively disabled.

With history pruning active (`_prune_old_tool_results`), C stays low
and nearly flat — a small but CORRECT signal, instead of a large
misleading one. `context_load()` is kept only for callers that still
pass a cumulative budget; `compute_tdi()` prefers `context_utilisation`.

Horizon normalisation
---------------------
`max_horizon` MUST be the scan's actual turn budget (`max_turns`, e.g.
100). A fixed horizon smaller than the scan (the old default of 20)
clamped H to 1.0 for 80% of a 100-turn scan, effectively deleting the
horizon dimension from the signal. `resolve_max_horizon()` enforces
this and never lets the denominator fall below the real budget.

Evidence scoring
----------------
    VERIFIED_EXPLOIT       = 1.0   (PoC reproduced, shell obtained)
    CONFIRMED_VULN         = 0.8   (verifier accepted)
    PLAUSIBLE_HYPOTHESIS   = 0.5   (scanner match, unverified)
    SPECULATIVE_HYPOTHESIS = 0.3   (no evidence yet — default for empty path)

Critical override
-----------------
When a finding has already been CONFIRMED (evidence >= CONFIRMED_VULN),
the agent should deep-dive on the confirmed vector rather than broaden
reconnaissance. The TDI formula naturally stays high in this state
(because S only falls as attempts accumulate), so we override the mode
to EXPLOIT regardless of the raw TDI. With the default override enabled
this is the ONLY way to reach Mode.EXPLOIT. It is the #1 bug fix from
the reference design — without it, the agent loops in broaden_recon
after confirming a SQLi instead of deepening the finding.

The gate is EVIDENCE-based, not count-based: `should_force_exploit()`
requires at least one finding scoring >= CONFIRMED_VULN (0.8). A merely
plausible finding (0.5 — e.g. recorded from a scanner log without a
reproduced result) does NOT lock the agent into a vector. A count-based
gate (`len(findings) > 0`) let a single hallucinated finding tunnel the
whole scan onto a vulnerability that may not exist.

The override keeps the RAW dimension values (tdi/h/e/c/s). It only sets
mode=EXPLOIT and overridden=True, so downstream consumers (RL state
encoder) still see a real difficulty signal rather than a constant.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

# Canonical verification level names. Imported (not re-typed) so the
# evidence ladder here and the verifier in app/evidence/verifier.py can
# never drift apart on spelling.
from app.evidence.verifier import (
    VERIFICATION_CONFIRMED,
    VERIFICATION_PLAUSIBLE,
    VERIFICATION_REPLAY_VERIFIED,
)


# ── evidence scoring ──────────────────────────────────────────────────
VERIFIED_EXPLOIT: float = 1.0
CONFIRMED_VULN: float = 0.8
PLAUSIBLE_HYPOTHESIS: float = 0.5
SPECULATIVE_HYPOTHESIS: float = 0.3

# ── TDI weights (paper defaults) ──────────────────────────────────────
W_H: float = 0.30
W_E: float = 0.30
W_C: float = 0.20
W_S: float = 0.20

# Invariant: the weights must sum to 1.0, otherwise TDI's range is no
# longer [0, 1] and the TDI_BROADEN threshold stops meaning what it says.
MODEL_WEIGHTS_SUM: float = W_H + W_E + W_C + W_S

# Enforced at IMPORT time, not merely asserted in a test. If someone
# retunes a weight and forgets the others, every TDI log line keeps
# printing a plausible-looking 0.xxx while TDI is really in [0, 1.2] —
# and a scan runs ~100 turns on a threshold that no longer maps to the
# same difficulty. Fail loudly at load instead of 40 minutes into a scan.
if abs(MODEL_WEIGHTS_SUM - 1.0) > 1e-9:
    raise RuntimeError(
        f"TDI weights must sum to 1.0 (got {MODEL_WEIGHTS_SUM}) — "
        "TDI_BROADEN threshold loses meaning otherwise"
    )

# ── mode thresholds ───────────────────────────────────────────────────
TDI_BROADEN: float = 0.60
# No TDI_EXPLOIT threshold: EXPLOIT is override-only. See the module
# docstring ("Mode mapping") for the proof that a <0.30 threshold is
# unreachable without a confirmed finding.

# ── Laplace smoothing prior for S ─────────────────────────────────────
# Treat each branch as having 1 prior success and 1 prior failure so
# fresh branches start at S=0.5 instead of 0.0 or 1.0.
LAPLACE_ALPHA: float = 1.0
LAPLACE_BETA: float = 1.0


class Mode(str, Enum):
    """TDA strategy mode — drives the hint rendered to the LLM.

    EXPLOIT is produced only by the critical override in HarnessBridge;
    compute_tdi() itself returns BROADEN or LLM_PICK.
    """

    BROADEN = "broaden"   # TDI > 0.60 — expand recon
    LLM_PICK = "llm"      # TDI <= 0.60 — defer to model
    EXPLOIT = "exploit"   # override-only — confirmed finding exists


@dataclass
class TDIResult:
    """One turn's TDI computation result.

    The `overridden` flag is True when the critical override fired
    (a confirmed finding exists -> forced EXPLOIT). The override does
    NOT rewrite the raw dimensions: `tdi`, `h`, `e`, `c`, `s` always
    hold the formula output. Consumers that need the forced mode read
    `mode`, and can distinguish a formula BROADEN/LLM_PICK from an
    override by checking `overridden`.

    `override_evidence` is the best evidence score among the findings
    that triggered the override (None when no override fired).
    """

    tdi: float
    h: float          # normalised horizon in [0, 1]
    e: float          # evidence confidence in [0.3, 1.0]
    c: float          # context load in [0, 1]
    s: float          # historical success rate in [0, 1]
    mode: Mode
    overridden: bool = False  # True when critical override fired
    override_evidence: float | None = None  # evidence score that fired it

    def as_dict(self) -> dict[str, float | str | bool | None]:
        return {
            "tdi": round(self.tdi, 4),
            "h": round(self.h, 4),
            "e": round(self.e, 4),
            "c": round(self.c, 4),
            "s": round(self.s, 4),
            "mode": self.mode.value,
            "overridden": self.overridden,
            "override_evidence": (
                round(self.override_evidence, 4)
                if self.override_evidence is not None else None
            ),
        }


# ── dimension calculators ─────────────────────────────────────────────


def horizon_normalised(remaining_steps: int, max_horizon: int) -> float:
    """H_hat in [0, 1]. Bigger = further from done = harder.

    `max_horizon` must be the scan's real turn budget. A non-positive
    value is a caller bug: silently returning 0.0 used to make a
    misconfigured scan look "easy", so raise instead.
    """
    if max_horizon <= 0:
        raise ValueError(
            "max_horizon must be the scan's positive turn budget, "
            f"got {max_horizon}"
        )
    return max(0.0, min(1.0, remaining_steps / max_horizon))


def evidence_confidence(path_scores: list[float]) -> float:
    """Mean confidence over path. Empty path -> 0.3 (speculative)."""
    if not path_scores:
        return SPECULATIVE_HYPOTHESIS
    clipped = [max(0.0, min(1.0, s)) for s in path_scores]
    return sum(clipped) / len(clipped)


def context_saturation(prompt_tokens: int, model_context_window: int) -> float:
    """C — saturation of the LATEST request, clipped to [0, 1].

    Args:
        prompt_tokens: prompt tokens the provider reported for the most
            recent completion (what is actually in the model's context
            window on that call), NOT the cumulative scan total.
        model_context_window: the model's real context window
            (e.g. 128_000 for most current chat models).

    Returns:
        prompt_tokens / model_context_window in [0, 1]. Returns 0.0 when
        the window is unknown (<= 0) so a misconfigured model cannot
        fake a saturated context.
    """
    if model_context_window <= 0:
        return 0.0
    if prompt_tokens <= 0:
        return 0.0
    return max(0.0, min(1.0, prompt_tokens / model_context_window))


def context_load(tokens_used: int, context_window: int) -> float:
    """DEPRECATED cumulative-budget variant of C, clipped to [0, 1].

    This is a SPEND measure (tokens burnt so far / scan budget), not a
    context-saturation measure — see the module docstring. It is kept
    for backwards compatibility with callers that only track a running
    total. New code must use `context_saturation()` so C reflects the
    model's context window and not the scan's wallet.
    """
    if context_window <= 0:
        return 0.0
    return max(0.0, min(1.0, tokens_used / context_window))


def historical_success(successes: int, attempts: int) -> float:
    """Laplace-smoothed success rate, clamped to [0, 1].

    Semantics: the caller decides the SCOPE. Both arguments must be
    counted in the same unit over the same scope.

    - Scan-level (what HarnessBridge currently passes): `successes` =
      confirmed findings, `attempts` = tool calls executed.
    - Branch-level (paper definition): `successes`/`attempts` = the
      EGATS PathStat counters for the branch under evaluation. Pass
      those from egats_select() if per-branch S is desired.

    Do NOT mix scopes (e.g. branch successes with scan attempts) — the
    ratio would be meaningless.

    Clamping matters because one tool call can yield several findings
    (successes > attempts), which previously produced S > 1 and made
    (1 - S) negative — artificially lowering the TDI.
    """
    successes = max(0, successes)
    attempts = max(0, attempts)
    # A success cannot outnumber attempts in this unit space.
    successes = min(successes, attempts)
    num = successes + LAPLACE_ALPHA
    den = attempts + LAPLACE_ALPHA + LAPLACE_BETA
    if den <= 0:
        return 0.5
    return max(0.0, min(1.0, num / den))


# ── TDI aggregator ────────────────────────────────────────────────────


def resolve_max_horizon(max_turns: int, configured: int | None = None) -> int:
    """Resolve the horizon denominator for a scan.

    Args:
        max_turns: the scan's actual turn budget (e.g. 100).
        configured: optional operator override (VAPT_AI_TDA_MAX_HORIZON).
            0 or None means "auto" (use max_turns).

    Returns:
        The denominator to pass to compute_tdi(). Never smaller than
        max_turns, so H cannot saturate merely because the configured
        horizon is shorter than the scan.

    Raises:
        ValueError: if max_turns is not positive.
    """
    if max_turns <= 0:
        raise ValueError(f"max_turns must be positive, got {max_turns}")
    if configured is None or configured <= 0:
        return max_turns
    return max(max_turns, configured)


def compute_tdi(
    *,
    remaining_steps: int,
    path_evidence: list[float],
    successes: int,
    attempts: int,
    max_horizon: int,
    context_utilisation: float | None = None,
    tokens_used: int | None = None,
    context_window: int | None = None,
    w_h: float = W_H,
    w_e: float = W_E,
    w_c: float = W_C,
    w_s: float = W_S,
) -> TDIResult:
    """Compute the Task Difficulty Index for the current turn.

    Args:
        remaining_steps: turns remaining in the scan (max_turns - current_turn).
        path_evidence: list of per-finding evidence scores in [0, 1].
        successes: number of verified (non-false-positive) findings so far.
        attempts: number of tool calls executed so far.
        max_horizon: the scan's actual turn budget (max_turns). Required —
            use resolve_max_horizon() when an operator override exists.
        context_utilisation: C — saturation of the latest request in
            [0, 1] (see `context_saturation`). This is the PREFERRED
            source and what HarnessBridge passes.
        tokens_used, context_window: legacy cumulative-budget inputs,
            used only when `context_utilisation` is None. Kept so older
            callers/tests keep working; they measure spend, not context.
        w_h, w_e, w_c, w_s: TDI weights. Default to paper values.

    Raises:
        ValueError: if no context signal is supplied at all (neither
            `context_utilisation` nor the legacy pair). Silently using
            C = 0 would hide a wiring bug and make every scan look easy.

    Returns:
        TDIResult with the raw TDI, the 4 dimension values, and the mode.
        The mode is BROADEN or LLM_PICK; EXPLOIT is override-only and
        the caller (HarnessBridge.tda_compute) applies that override when
        findings exist.
    """
    h = horizon_normalised(remaining_steps, max_horizon)
    e = evidence_confidence(path_evidence)
    if context_utilisation is not None:
        c = max(0.0, min(1.0, context_utilisation))
    elif tokens_used is not None and context_window is not None:
        c = context_load(tokens_used, context_window)
    else:
        raise ValueError(
            "compute_tdi needs a context signal: pass context_utilisation "
            "(preferred) or the legacy tokens_used + context_window pair"
        )
    s = historical_success(successes, attempts)
    tdi = w_h * h + w_e * (1.0 - e) + w_c * c + w_s * (1.0 - s)
    tdi = max(0.0, min(1.0, tdi))
    mode = Mode.BROADEN if tdi > TDI_BROADEN else Mode.LLM_PICK
    return TDIResult(tdi=tdi, h=h, e=e, c=c, s=s, mode=mode)


# ── helper: classify raw evidence label -> numeric score ──────────────
#
# This map is the SINGLE source of truth for "label string -> ladder rung",
# shared by `evidence_score_from_label()` and `evidence_score_from_finding()`
# so the two can never disagree about the same word.
#
# Two bugs lived here:
#   1. `replay_verified` — the CANONICAL level name from
#      app/evidence/verifier.py (and the one `evidence_score_from_finding`
#      already honoured explicitly) — was MISSING, so
#      `evidence_score_from_label("replay_verified")` returned 0.3
#      (SPECULATIVE). The strongest possible evidence scored BELOW "we have
#      no evidence at all". Latent, but exported as public API.
#   2. `"verified"` mapped to VERIFIED_EXPLOIT (1.0) while
#      `evidence_score_from_finding({"verified": True})` returned
#      CONFIRMED_VULN (0.8) — the same word, two rungs. `"verified"` is the
#      legacy boolean and means "a verifier accepted it", i.e. 0.8.
#
# The canonical names are imported from the verifier itself so a new level
# cannot be spelled one way there and another way here. Coverage of
# VERIFICATION_LEVELS is asserted by tests/unit/test_evidence_label_map.py.
_EVIDENCE_LABEL_MAP: dict[str, float] = {
    VERIFICATION_REPLAY_VERIFIED: VERIFIED_EXPLOIT,  # canonical: replay reproduced
    "replay": VERIFIED_EXPLOIT,                      # shorthand
    "exploit": VERIFIED_EXPLOIT,                     # PoC reproduced / shell obtained
    "verified": CONFIRMED_VULN,                      # legacy boolean == verifier-accepted
    VERIFICATION_CONFIRMED: CONFIRMED_VULN,          # canonical: tool verdict present
    "vuln": CONFIRMED_VULN,                          # shorthand
    VERIFICATION_PLAUSIBLE: PLAUSIBLE_HYPOTHESIS,    # canonical: no outcome assertion
    "hypothesis": PLAUSIBLE_HYPOTHESIS,              # shorthand
    "speculative": SPECULATIVE_HYPOTHESIS,           # below plausible; no evidence
    "guess": SPECULATIVE_HYPOTHESIS,                 # shorthand
}


def evidence_score_from_label(label: str | None) -> float:
    """Map a human-readable evidence label to a numeric score.

    Accepts both the canonical verifier levels (`plausible` / `confirmed` /
    `replay_verified`) and the legacy shorthand (`verified`, `vuln`,
    `hypothesis`, ...). Unknown or empty labels default to
    SPECULATIVE_HYPOTHESIS (0.3) — never to a confirming rung.
    """
    if not label or not isinstance(label, str):
        return SPECULATIVE_HYPOTHESIS
    key = label.strip().lower()
    return _EVIDENCE_LABEL_MAP.get(key, SPECULATIVE_HYPOTHESIS)


def evidence_score_from_finding(finding: dict) -> float:
    """Derive an evidence score from a finding dict.

    Priority (highest first):
        0. finding["false_positive"] is True   -> SPECULATIVE (0.3)
           (a finding the verifier already rejected is a REFUTATION, so
           it ranks at the FLOOR of the ladder — it both can never
           confirm and must not outrank an unexamined hypothesis)
        1. finding["verification"] == "replay_verified" -> VERIFIED_EXPLOIT (1.0)
        2. finding["verification"] == "confirmed"       -> CONFIRMED_VULN (0.8)
        3. finding["verification"] == "plausible"       -> PLAUSIBLE_HYPOTHESIS (0.5)
           (`verification` is the CANONICAL provenance field written by
           tool_bridge._record_vulnerability via app/evidence/verifier.py;
           when present it wins over the legacy booleans below — otherwise
           a stale `verified: True` could re-promote a plausible finding)
        4. finding["replay_reproduced"]/["human_verified"] is True  -> 1.0
        5. finding["verified"] is True                              -> 0.8
        6. finding exists but not verified                          -> 0.5

    Historical note: the ReAct loop hardcoded `{"verified": True}` for
    every recorded finding, so rungs 1-2 were unreachable and this ladder
    degraded to `len(findings) > 0`. Do not reintroduce a default True.
    """
    if not isinstance(finding, dict):
        return SPECULATIVE_HYPOTHESIS
    # Explicit rejection wins over any other evidence flag — contradictory
    # records must never trigger the EXPLOIT override.
    #
    # It also ranks at the FLOOR, not at PLAUSIBLE. Returning 0.5 meant a
    # finding the auditor had already DISPROVED scored exactly the same as
    # `verification="plausible"` — one nobody has checked yet — and pulled
    # the mean E UP (E is a mean, and 0.5 > SPECULATIVE), so recording junk
    # *lowered* perceived difficulty. A refutation is strictly the weakest
    # evidence available, so it belongs next to "speculative".
    if finding.get("false_positive"):
        return SPECULATIVE_HYPOTHESIS
    level = finding.get("verification")
    if isinstance(level, str):
        # One lookup shared with evidence_score_from_label() so the label
        # and the dict paths can never score the same word differently.
        # Unknown strings fall through to the legacy boolean rungs below
        # (a malformed level must not silently promote a finding).
        known = _EVIDENCE_LABEL_MAP.get(level.strip().lower())
        if known is not None:
            return known
    if finding.get("replay_reproduced") or finding.get("human_verified"):
        return VERIFIED_EXPLOIT
    if finding.get("verified"):
        return CONFIRMED_VULN
    return PLAUSIBLE_HYPOTHESIS


def max_evidence_score(findings: list[dict]) -> float:
    """Best evidence score among `findings` (0.0 when the list is empty)."""
    if not findings:
        return 0.0
    return max(evidence_score_from_finding(f) for f in findings)


def should_force_exploit(
    findings: list[dict], threshold: float = CONFIRMED_VULN,
) -> bool:
    """Decide whether the EXPLOIT override may fire.

    Gate on EVIDENCE, not on the number of findings. A single
    unverified/plausible finding (score 0.5 — e.g. a scanner log with no
    reproduced result) must not tunnel the agent onto that vector; only
    a finding at or above CONFIRMED_VULN (0.8, i.e. verifier-accepted or
    replay/human-verified) may lock the scan's direction.

    Args:
        findings: finding dicts (see evidence_score_from_finding).
        threshold: minimum evidence score that justifies the override.

    Returns:
        True when at least one finding scores >= threshold.
    """
    return any(
        evidence_score_from_finding(f) >= threshold for f in findings
    )


# ── TDA hint rendering ────────────────────────────────────────────────


def render_tda_hint(tdi_result: TDIResult) -> str:
    """Render the TDI result as an advisory system message for the LLM.

    Framing rules (advisory — consistent with RL Phase 1 + KG Phase 2):
        - Header always includes "(advisory — your judgment wins)".
        - "Phase:" — descriptive, not "You must:".
        - When overridden=True, explicitly say so + explain why.
    """
    mode = tdi_result.mode

    if tdi_result.overridden:
        return (
            "## TDA Phase Guidance (advisory — overridden)\n\n"
            f"**Phase:** EXPLOIT (forced — confirmed finding exists)\n"
            f"**Difficulty Index:** {tdi_result.tdi:.3f} (raw formula value; "
            f"override fired on evidence "
            f"{tdi_result.override_evidence if tdi_result.override_evidence is not None else 'n/a'})\n"
            f"**Horizon:** {tdi_result.h:.2f}  **Evidence:** {tdi_result.e:.2f}  "
            f"**Context:** {tdi_result.c:.2f}  **Success rate:** {tdi_result.s:.2f}\n\n"
            "A confirmed finding already exists. Deep-dive on the "
            "confirmed vector BEFORE expanding reconnaissance. Report "
            "additional findings on the same vector immediately.\n\n"
            "Advisory — your judgment wins, but skipping a confirmed "
            "vector to broaden recon is almost always wrong."
        )

    if mode == Mode.BROADEN:
        return (
            "## TDA Phase Guidance (advisory — your judgment wins)\n\n"
            f"**Phase:** BROADEN\n"
            f"**Difficulty Index:** {tdi_result.tdi:.3f} (high — task is hard)\n"
            f"**Horizon:** {tdi_result.h:.2f}  **Evidence:** {tdi_result.e:.2f}  "
            f"**Context:** {tdi_result.c:.2f}  **Success rate:** {tdi_result.s:.2f}\n\n"
            "Task difficulty is high. Consider:\n"
            "- Expand reconnaissance (new subdomains, alternative endpoints)\n"
            "- Review JavaScript bundles for hidden API endpoints\n"
            "- Switch to a different attack vector class\n\n"
            "Advisory — your judgment wins."
        )

    if mode == Mode.EXPLOIT:
        return (
            "## TDA Phase Guidance (advisory — your judgment wins)\n\n"
            f"**Phase:** EXPLOIT\n"
            f"**Difficulty Index:** {tdi_result.tdi:.3f} (low — task is tractable)\n"
            f"**Horizon:** {tdi_result.h:.2f}  **Evidence:** {tdi_result.e:.2f}  "
            f"**Context:** {tdi_result.c:.2f}  **Success rate:** {tdi_result.s:.2f}\n\n"
            "Task difficulty is low. Deep-dive on the current vector. "
            "Attempt exploitation with confidence. Report findings "
            "immediately upon confirmation.\n\n"
            "Advisory — your judgment wins."
        )

    # LLM_PICK
    return (
        "## TDA Phase Guidance (advisory — your judgment wins)\n\n"
        f"**Phase:** ASSESS\n"
        f"**Difficulty Index:** {tdi_result.tdi:.3f} (moderate)\n"
        f"**Horizon:** {tdi_result.h:.2f}  **Evidence:** {tdi_result.e:.2f}  "
        f"**Context:** {tdi_result.c:.2f}  **Success rate:** {tdi_result.s:.2f}\n\n"
        "Task difficulty is moderate. Use your judgment to select the "
        "next action — continue the current strategy or pivot based on "
        "findings.\n\n"
        "Advisory — your judgment wins."
    )


__all__ = [
    # Constants
    "VERIFIED_EXPLOIT", "CONFIRMED_VULN", "PLAUSIBLE_HYPOTHESIS",
    "SPECULATIVE_HYPOTHESIS",
    "W_H", "W_E", "W_C", "W_S",
    "TDI_BROADEN",
    "LAPLACE_ALPHA", "LAPLACE_BETA",
    # Types
    "Mode", "TDIResult",
    # Functions
    "horizon_normalised", "resolve_max_horizon",
    "evidence_confidence", "context_saturation", "context_load",
    "historical_success", "compute_tdi",
    "evidence_score_from_label", "evidence_score_from_finding",
    "max_evidence_score", "should_force_exploit",
    "render_tda_hint",
]