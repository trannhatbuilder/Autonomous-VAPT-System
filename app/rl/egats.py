"""
VAPT-AI RL — Evidence-Guided Attack Tree Search (EGATS).

MCTS-style bandit over attack paths exposed by the Knowledge Graph,
with UCB selection penalised by TDA difficulty.

Selection score
---------------
    UCB(p) = phi(p) + c * sqrt(ln(N) / n_p) - lambda * delta(p) + mu * specificity(p)

    phi(p)        = path's static score from KG (severity * probability product)
    c             = exploration constant (default sqrt(2))
    N             = total attempts across all alive paths
    n_p           = attempts on path p
    lambda        = TDI penalty weight (default 0.5)
    delta(p)      = TDI EMA for path p (penalises high-difficulty paths)
    mu            = specificity boost weight (default 0.2)
    specificity   = 0..1 — how well path matches target inventory

Special case: unattempted paths (n_p = 0) get UCB = +inf so they are
always explored at least once.

Pruning
-------
Any path attempted at least `k_min_prune` times whose rolling TDI EMA
stays above `TDI_BROADEN` (0.60) is permanently dropped from selection.
Pruned paths remain in `stats()` for debugging but are never selected
again. This addresses the "premature commitment" failure mode where the
planner keeps retrying a path that consistently produces high-difficulty
states.

Integration
-----------
Unlike the offline `search(simulate_fn)` pattern, this EGATS is designed
for LIVE integration with the agent loop:

    1. seed_from_kg(kg, source_node_ids)  — called once after first KG consult
    2. select()                           — called each turn, returns best path
    3. record_outcome(path_key, success, tdi)  — called after each finding

The agent loop owns the "simulation" (running tools + observing results);
EGATS only owns path selection + statistics tracking. This keeps the
planner unit-testable without spinning up an LLM.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

from app.rl.tda import TDI_BROADEN

logger = logging.getLogger(__name__)


# ── defaults ──────────────────────────────────────────────────────────
DEFAULT_UCB_C: float = math.sqrt(2.0)
DEFAULT_LAMBDA: float = 0.5
DEFAULT_K_MIN_PRUNE: int = 3
DEFAULT_EMA_ALPHA: float = 0.3
DEFAULT_MU_SPECIFICITY: float = 0.2


def _normalise_edges(raw: Any) -> list[tuple[str, str, str]]:
    """Normalise a KG edge collection to ``(source, target, edge_type)``.

    Accepts either already-tuple edges (KGPath.edges) or dict edges
    (KGPath.as_dict()["edges"]) and tolerates malformed entries by
    skipping them.
    """
    if not raw:
        return []
    out: list[tuple[str, str, str]] = []
    for e in raw:
        if isinstance(e, (tuple, list)) and len(e) >= 3:
            out.append((str(e[0]), str(e[1]), str(e[2])))
        elif isinstance(e, dict):
            src = e.get("source", e.get("source_id"))
            dst = e.get("target", e.get("target_id"))
            etype = e.get("edge_type", e.get("type", ""))
            if src is not None and dst is not None:
                out.append((str(src), str(dst), str(etype)))
        else:
            logger.debug("egats_bad_edge_skipped | edge=%r", e)
    return out


@dataclass
class PathStat:
    """Per-path statistics tracked by EGATS.

    One instance per unique attack path (identified by `nodes` tuple).
    Mutated by `record_outcome()` — not thread-safe, but EGATS is
    single-threaded (one instance per scan, called from the agent loop).
    """

    nodes: list[str]                                # node IDs in path order
    edges: list[tuple[str, str, str]] = field(default_factory=list)
    base_score: float = 0.0                         # phi(p) from KG
    attempts: int = 0
    successes: int = 0
    tdi_ema: float | None = None                    # EMA of TDI outcomes
    pruned: bool = False
    # 0..1 — how well path matches target inventory.
    # 1.0 = exact component+version match
    # 0.75 = exact product+version match
    # 0.5 = component family
    # 0.25 = product family
    # 0.0 = legacy path (no inventory context)
    match_specificity: float = 0.0
    history: list[dict[str, Any]] = field(default_factory=list)

    def key(self) -> tuple[str, ...]:
        """Stable hashable identity for the path."""
        return tuple(self.nodes)

    def as_dict(self) -> dict[str, Any]:
        return {
            "nodes": list(self.nodes),
            "edges": list(self.edges),
            "base_score": round(self.base_score, 4),
            "attempts": self.attempts,
            "successes": self.successes,
            "tdi_ema": round(self.tdi_ema, 4) if self.tdi_ema is not None else None,
            "pruned": self.pruned,
            "match_specificity": round(self.match_specificity, 4),
            "confidence": round(self.successes / max(self.attempts, 1), 4),
        }


@dataclass
class EGATSResult:
    """Snapshot returned by EGATS.select() — what the agent loop sees.

    `path` is None when no alive paths remain (all pruned or never
    seeded). The agent loop should skip EGATS hint injection in that
    case.
    """

    path: PathStat | None
    ucb_score: float                      # UCB value of selected path (+inf for unattempted)
    confidence: float                     # successes / max(attempts, 1)
    total_paths: int                      # total seeded (including pruned)
    alive_paths: int                      # non-pruned
    pruned_paths: int                     # pruned count


class EGATS:
    """Evidence-Guided Attack Tree Search planner.

    One instance per scan. Holds no global state — all state lives in
    `_stats` (dict of PathStat keyed by node tuple).
    """

    def __init__(
        self,
        *,
        ucb_c: float = DEFAULT_UCB_C,
        lambda_penalty: float = DEFAULT_LAMBDA,
        k_min_prune: int = DEFAULT_K_MIN_PRUNE,
        ema_alpha: float = DEFAULT_EMA_ALPHA,
        mu_specificity: float = DEFAULT_MU_SPECIFICITY,
    ) -> None:
        self.ucb_c = ucb_c
        self.lambda_penalty = lambda_penalty
        self.k_min_prune = k_min_prune
        self.ema_alpha = ema_alpha
        self.mu_specificity = mu_specificity
        self._stats: dict[tuple[str, ...], PathStat] = {}

    # ── candidate seeding ─────────────────────────────────────────

    def seed_from_kg(
        self,
        kg: Any,
        source_node_ids: list[str],
        *,
        max_depth: int = 4,
        top_k: int = 10,
        target_kind: str = "Finding",
    ) -> int:
        """Pull initial candidate paths from the KG.

        Args:
            kg: KnowledgeGraph instance.
            source_node_ids: KG node IDs to start paths from (typically
                Technology nodes for the detected tech stack).
            max_depth: max path length.
            top_k: max paths to seed.
            target_kind: target node type (default "Finding").

        Returns:
            Number of NEW paths seeded (duplicates are skipped).
        """
        if not source_node_ids:
            return 0
        try:
            paths = kg.get_attack_paths(
                source_node_ids,
                max_depth=max_depth,
                top_k=top_k,
                target_kind=target_kind,
            )
        except Exception as exc:
            logger.warning("egats_seed_failed | err=%s", exc)
            return 0

        seeded = 0
        for p in paths:
            # KGPath is a dataclass with .nodes / .edges / .score / .match_specificity.
            # It also has .as_dict() which returns edges as list of dicts.
            # We normalize both forms into list of (source, target, edge_type) tuples.
            if hasattr(p, "nodes"):
                # KGPath dataclass instance.
                nodes = list(p.nodes)
                edges = _normalise_edges(getattr(p, "edges", []) or [])
                base_score = float(getattr(p, "score", 0.0))
                specificity = float(getattr(p, "match_specificity", 0.0) or 0.0)
            elif isinstance(p, dict):
                # Dict form (e.g. from KGPath.as_dict()).
                nodes = list(p.get("nodes", []))
                edges = _normalise_edges(p.get("edges", []))
                base_score = float(p.get("score", 0.0))
                specificity = float(p.get("match_specificity", 0.0) or 0.0)
            else:
                # Unknown path shape — skip.
                continue

            if not nodes:
                continue

            stat = PathStat(
                nodes=nodes,
                edges=edges,
                base_score=base_score,
                match_specificity=specificity,
            )
            key = stat.key()
            if key not in self._stats:
                self._stats[key] = stat
                seeded += 1

        logger.info(
            "egats_seeded | new_paths=%d | total_paths=%d",
            seeded, len(self._stats),
        )
        return seeded

    # ── selection ─────────────────────────────────────────────────

    def _ucb(self, stat: PathStat, total_n: int) -> float:
        """Compute UCB score for a path.

        Unattempted paths get +inf so they're always explored once.
        """
        if stat.attempts == 0:
            return math.inf
        explore = self.ucb_c * math.sqrt(
            math.log(max(total_n, 1)) / stat.attempts
        )
        delta = stat.tdi_ema if stat.tdi_ema is not None else 0.5
        spec_bonus = self.mu_specificity * stat.match_specificity
        return stat.base_score + explore - self.lambda_penalty * delta + spec_bonus

    def _alive(self) -> list[PathStat]:
        return [s for s in self._stats.values() if not s.pruned]

    def select(self) -> EGATSResult:
        """Select the highest-UCB alive path.

        Returns EGATSResult with path=None when no alive paths remain.
        """
        alive = self._alive()
        total_paths = len(self._stats)
        pruned_paths = total_paths - len(alive)

        if not alive:
            return EGATSResult(
                path=None,
                ucb_score=0.0,
                confidence=0.0,
                total_paths=total_paths,
                alive_paths=0,
                pruned_paths=pruned_paths,
            )

        total_n = sum(s.attempts for s in alive)
        best = max(alive, key=lambda s: self._ucb(s, total_n))
        ucb_score = self._ucb(best, total_n)
        confidence = best.successes / max(best.attempts, 1)

        return EGATSResult(
            path=best,
            ucb_score=ucb_score,
            confidence=confidence,
            total_paths=total_paths,
            alive_paths=len(alive),
            pruned_paths=pruned_paths,
        )

    # ── backprop + pruning ────────────────────────────────────────

    def record_outcome(
        self,
        path_key: tuple[str, ...],
        success: bool,
        tdi: float,
    ) -> None:
        """Backprop an outcome onto a path + maybe prune.

        Args:
            path_key: the path's node tuple (from PathStat.key()).
            success: True if the path led to a confirmed finding.
            tdi: the TDI value at the time of the outcome (used to
                update the path's TDI EMA — high TDI = path is hard).

        Side effects:
            - Increments attempts/successes on the path.
            - Updates tdi_ema via exponential moving average.
            - Appends to history.
            - Calls _maybe_prune() — may set pruned=True.
        """
        stat = self._stats.get(path_key)
        if stat is None:
            logger.debug(
                "egats_record_unknown_path | key=%s | skipping", path_key,
            )
            return

        stat.attempts += 1
        if success:
            stat.successes += 1

        # TDI EMA update.
        if stat.tdi_ema is None:
            stat.tdi_ema = float(tdi)
        else:
            stat.tdi_ema = (
                (1.0 - self.ema_alpha) * stat.tdi_ema
                + self.ema_alpha * float(tdi)
            )

        stat.history.append({
            "success": success,
            "tdi": float(tdi),
        })

        self._maybe_prune(stat)

        logger.info(
            "egats_outcome | path=%s | success=%s | tdi=%.3f | "
            "attempts=%d | successes=%d | tdi_ema=%.3f | pruned=%s",
            "→".join(stat.nodes), success, tdi,
            stat.attempts, stat.successes,
            stat.tdi_ema if stat.tdi_ema is not None else 0.5,
            stat.pruned,
        )

    def _maybe_prune(self, stat: PathStat) -> None:
        """Prune paths that consistently produce high-difficulty states.

        Condition: attempts >= k_min_prune AND tdi_ema > TDI_BROADEN.
        Pruned paths stay in _stats (for debugging) but are excluded
        from _alive() and therefore never selected again.
        """
        if stat.pruned or stat.attempts < self.k_min_prune:
            return
        if stat.tdi_ema is not None and stat.tdi_ema > TDI_BROADEN:
            stat.pruned = True
            logger.info(
                "egats_pruned | path=%s | attempts=%d | tdi_ema=%.3f",
                "→".join(stat.nodes), stat.attempts, stat.tdi_ema,
            )

    # ── inspection helpers ────────────────────────────────────────

    def stats(self) -> list[PathStat]:
        """Return all PathStat objects (including pruned)."""
        return list(self._stats.values())

    def to_dict(self) -> dict[str, Any]:
        """Serializable snapshot — the ONLY way planner state crosses a boundary.

        `EGATS` is a live, mutable object. Anything that stores or serializes
        state (the RL session dict, an episode dump, an SSE payload, the
        `scan_record.rl_stats` JSON column) raises
        `TypeError: Object of type EGATS is not JSON serializable` the moment
        the planner itself is passed through. Consumers therefore take this
        dict, never `self`.

        Contains only ints, floats, strings, lists and dicts, so
        `json.dumps()` always succeeds. Keys `alive` / `pruned` are what
        `StateEncoder.from_session()` reads for the two planner dimensions.
        """
        all_stats = list(self._stats.values())
        alive = [s for s in all_stats if not s.pruned]
        return {
            "total_paths": len(all_stats),
            "alive": len(alive),
            "pruned": len(all_stats) - len(alive),
            "total_attempts": sum(s.attempts for s in all_stats),
            "total_successes": sum(s.successes for s in all_stats),
            "paths": [s.as_dict() for s in all_stats],
        }

    def get_stats_summary(self) -> dict[str, Any]:
        """Summary for the rl_stats.egats sub-dict."""
        all_stats = list(self._stats.values())
        alive = [s for s in all_stats if not s.pruned]
        pruned = [s for s in all_stats if s.pruned]
        # Find the path with the most successes (best track record).
        best = max(all_stats, key=lambda s: (s.successes, -s.attempts)) if all_stats else None
        return {
            "enabled": True,
            "seeded": len(all_stats),
            "alive": len(alive),
            "pruned": len(pruned),
            "total_attempts": sum(s.attempts for s in all_stats),
            "total_successes": sum(s.successes for s in all_stats),
            "best_path": best.as_dict() if best else None,
            "paths": [s.as_dict() for s in all_stats[:10]],  # top 10 for debug
        }


# ── hint rendering ────────────────────────────────────────────────────


def render_egats_hint(result: EGATSResult) -> str:
    """Render the EGATS result as an advisory system message.

    Returns empty string when result.path is None (no alive paths).
    Framing is advisory — consistent with RL Phase 1 + KG Phase 2 +
    TDA Phase 3.
    """
    if result.path is None:
        return ""

    p = result.path
    chain = " → ".join(p.nodes)
    confidence = p.successes / max(p.attempts, 1)

    if p.attempts == 0:
        track_record = "unattempted — exploring first"
    elif confidence >= 0.5:
        track_record = f"{p.successes}/{p.attempts} successes — exploiting"
    else:
        track_record = f"{p.successes}/{p.attempts} successes — still exploring"

    if p.match_specificity >= 0.75:
        specificity_tag = "exact match"
    elif p.match_specificity >= 0.5:
        specificity_tag = "partial match"
    else:
        specificity_tag = "generic"

    ucb_str = f"{result.ucb_score:.3f}" if result.ucb_score != math.inf else "inf (unattempted)"

    return (
        "## EGATS Path Suggestion (advisory — your judgment wins)\n\n"
        f"**Recommended path:** {chain}\n"
        f"**UCB score:** {ucb_str}\n"
        f"**Track record:** {track_record}\n"
        f"**Specificity:** {specificity_tag}\n\n"
        "This path has the highest UCB score among alive attack paths. "
        "Consider testing it next — but your reasoning wins. If your "
        "reconnaissance suggests a different vector, follow your reasoning.\n\n"
        "Advisory — your judgment wins."
    )


__all__ = [
    # Constants
    "DEFAULT_UCB_C", "DEFAULT_LAMBDA", "DEFAULT_K_MIN_PRUNE",
    "DEFAULT_EMA_ALPHA", "DEFAULT_MU_SPECIFICITY",
    # Types
    "PathStat", "EGATSResult",
    # Class
    "EGATS",
    # Functions
    "render_egats_hint",
]