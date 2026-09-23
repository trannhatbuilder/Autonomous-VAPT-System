"""
VAPT-AI Task Difficulty Assessment (TDA).

Python port of EVVO Sentinel shield_engine/tda.py.

Implements the 4-dimensional difficulty index from Excalibur
(arxiv 2602.17622). Used by EGATS (W17-S7) to penalise high-difficulty
branches in MCTS, and by the Harness Bridge (W18) to gate skill execution.

TDI formula
-----------
    TDI = w_H * H_hat + w_E * (1 - E) + w_C * C + w_S * (1 - S)

Dimensions
----------
    H — Horizon Estimation:   remaining steps, normalised per branch
    E — Evidence Confidence:  mean confidence over the current path
    C — Context Load:         fraction of context window consumed
    S — Historical Success:   Laplace-smoothed success rate on this branch

Default weights (paper; robust within ±3% across 256 configs):
    w_H = w_E = 0.30
    w_C = w_S = 0.20

Mode thresholds
---------------
    TDI > 0.60          → broaden reconnaissance (task is hard)
    TDI < 0.30          → depth-first exploitation (task is tractable)
    0.30 <= TDI <= 0.60 → let the LLM decide
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

# ── evidence scoring (from the paper) ─────────────────────────────
VERIFIED_EXPLOIT: float = 1.0
CONFIRMED_VULN: float = 0.8
PLAUSIBLE_HYPOTHESIS: float = 0.5
SPECULATIVE_HYPOTHESIS: float = 0.3

# ── TDI weights (paper defaults) ──────────────────────────────────
W_H: float = 0.30
W_E: float = 0.30
W_C: float = 0.20
W_S: float = 0.20

# ── mode thresholds ───────────────────────────────────────────────
TDI_BROADEN: float = 0.60   # TDI > 0.60 → broaden recon
TDI_EXPLOIT: float = 0.30   # TDI < 0.30 → exploit depth-first

# ── horizon normalisation ─────────────────────────────────────────
# Paper reports MAE=4.2 steps on absolute values; rank is what matters.
# We normalise by an assumed max horizon so H_hat ∈ [0, 1].
DEFAULT_MAX_HORIZON: int = 20

# ── Laplace smoothing prior for S ─────────────────────────────────
# Treat each branch as having 1 prior success and 1 prior failure so
# fresh branches start at S=0.5 instead of 0.0 or 1.0.
LAPLACE_ALPHA: float = 1.0
LAPLACE_BETA: float = 1.0


class Mode(str, Enum):
    """TDA mode — what the agent should do given current difficulty."""

    BROADEN = "broaden"   # TDI > 0.60 — expand reconnaissance
    LLM_PICK = "llm"      # middle band — defer to LLM
    EXPLOIT = "exploit"   # TDI < 0.30 — go deep on current path


@dataclass
class TDIResult:
    """Result of compute_tdi() — difficulty index + per-dimension breakdown."""

    tdi: float
    h: float          # normalised horizon (0..1, higher = further from done)
    e: float          # evidence confidence (0..1, higher = more confident)
    c: float          # context load (0..1, higher = more tokens used)
    s: float          # historical success rate (0..1, Laplace-smoothed)
    mode: Mode

    def as_dict(self) -> dict[str, float | str]:
        return {
            "tdi": round(self.tdi, 4),
            "h": round(self.h, 4),
            "e": round(self.e, 4),
            "c": round(self.c, 4),
            "s": round(self.s, 4),
            "mode": self.mode.value,
        }


# ── dimension calculators ─────────────────────────────────────────

def horizon_normalised(remaining_steps: int, max_horizon: int = DEFAULT_MAX_HORIZON) -> float:
    """H_hat ∈ [0, 1]. Bigger = further from done = harder.

    Args:
        remaining_steps: estimated steps to reach goal (e.g. planner depth)
        max_horizon: normalisation cap (default 20 per paper)

    Returns:
        normalised horizon in [0, 1]
    """
    if max_horizon <= 0:
        return 0.0
    return max(0.0, min(1.0, remaining_steps / max_horizon))


def evidence_confidence(path_scores: list[float]) -> float:
    """Mean confidence over path. Empty path → 0.3 (speculative).

    Args:
        path_scores: list of evidence scores along the path (each in [0, 1])

    Returns:
        mean confidence in [0, 1]
    """
    if not path_scores:
        return SPECULATIVE_HYPOTHESIS
    clipped = [max(0.0, min(1.0, s)) for s in path_scores]
    return sum(clipped) / len(clipped)


def context_load(tokens_used: int, context_window: int) -> float:
    """Fraction of context window consumed, clipped to [0, 1].

    Args:
        tokens_used: total tokens consumed so far in the scan
        context_window: max context window for the LLM (e.g. 128_000)

    Returns:
        context load in [0, 1]
    """
    if context_window <= 0:
        return 0.0
    return max(0.0, min(1.0, tokens_used / context_window))


def historical_success(successes: int, attempts: int) -> float:
    """Laplace-smoothed success rate. Fresh branches → 0.5.

    Laplace smoothing: (successes + α) / (attempts + α + β)
    With α=β=1: fresh branch (0/0) → 1/2 = 0.5 (neutral prior)

    Args:
        successes: number of successful attempts on this branch
        attempts: total attempts on this branch

    Returns:
        smoothed success rate in [0, 1]
    """
    num = successes + LAPLACE_ALPHA
    den = attempts + LAPLACE_ALPHA + LAPLACE_BETA
    return num / den if den > 0 else 0.5


# ── TDI aggregator ────────────────────────────────────────────────

def compute_tdi(
    *,
    remaining_steps: int,
    path_evidence: list[float],
    tokens_used: int,
    context_window: int,
    successes: int,
    attempts: int,
    max_horizon: int = DEFAULT_MAX_HORIZON,
    w_h: float = W_H,
    w_e: float = W_E,
    w_c: float = W_C,
    w_s: float = W_S,
) -> TDIResult:
    """Compute Task Difficulty Index from 4 dimensions.

    TDI = w_H * H_hat + w_E * (1 - E) + w_C * C + w_S * (1 - S)

    All dimensions are in [0, 1]. Higher TDI = harder task.
    Note that E and S are inverted (1 - E, 1 - S) because low evidence
    confidence and low historical success both increase difficulty.

    Args:
        remaining_steps: estimated steps to goal
        path_evidence: list of evidence scores along current path
        tokens_used: total tokens consumed in scan
        context_window: LLM context window size
        successes: successful attempts on this branch
        attempts: total attempts on this branch
        max_horizon: normalisation cap for H
        w_h, w_e, w_c, w_s: dimension weights (paper defaults if omitted)

    Returns:
        TDIResult with tdi, h, e, c, s, mode
    """
    h = horizon_normalised(remaining_steps, max_horizon)
    e = evidence_confidence(path_evidence)
    c = context_load(tokens_used, context_window)
    s = historical_success(successes, attempts)

    tdi = w_h * h + w_e * (1.0 - e) + w_c * c + w_s * (1.0 - s)
    tdi = max(0.0, min(1.0, tdi))

    if tdi > TDI_BROADEN:
        mode = Mode.BROADEN
    elif tdi < TDI_EXPLOIT:
        mode = Mode.EXPLOIT
    else:
        mode = Mode.LLM_PICK

    return TDIResult(tdi=tdi, h=h, e=e, c=c, s=s, mode=mode)


# ── helper: classify raw evidence label → numeric score ───────────

_EVIDENCE_LABEL_MAP: dict[str, float] = {
    "verified": VERIFIED_EXPLOIT,
    "exploit": VERIFIED_EXPLOIT,
    "confirmed": CONFIRMED_VULN,
    "vuln": CONFIRMED_VULN,
    "plausible": PLAUSIBLE_HYPOTHESIS,
    "hypothesis": PLAUSIBLE_HYPOTHESIS,
    "speculative": SPECULATIVE_HYPOTHESIS,
    "guess": SPECULATIVE_HYPOTHESIS,
}


def evidence_score_from_label(label: str | None) -> float:
    """Map a human-readable evidence label to a numeric score.

    Args:
        label: one of verified/exploit/confirmed/vuln/plausible/hypothesis/
               speculative/guess (case-insensitive). None → speculative.

    Returns:
        evidence score in [0.3, 1.0]
    """
    if not label:
        return SPECULATIVE_HYPOTHESIS
    key = label.strip().lower()
    return _EVIDENCE_LABEL_MAP.get(key, SPECULATIVE_HYPOTHESIS)


__all__ = [
    # Constants
    "VERIFIED_EXPLOIT", "CONFIRMED_VULN", "PLAUSIBLE_HYPOTHESIS",
    "SPECULATIVE_HYPOTHESIS",
    "W_H", "W_E", "W_C", "W_S",
    "TDI_BROADEN", "TDI_EXPLOIT",
    "DEFAULT_MAX_HORIZON",
    "LAPLACE_ALPHA", "LAPLACE_BETA",
    # Types
    "Mode", "TDIResult",
    # Functions
    "horizon_normalised", "evidence_confidence", "context_load",
    "historical_success", "compute_tdi", "evidence_score_from_label",
]