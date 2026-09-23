"""
VAPT-AI Evidence-Guided Attack Tree Search (EGATS).

Python port of EVVO Sentinel shield_engine/egats.py.

MCTS-style bandit over attack paths exposed by KnowledgeGraph, with
UCB selection penalised by TDA difficulty (Excalibur, arxiv 2602.17622).

Selection score
---------------
    UCB(p) = phi(p) + c * sqrt(ln(N) / n_p) - lambda * delta(p) + mu * specificity(p)

      phi(p)         = path's static score from KG (severity * historical wins)
      c              = exploration constant (default sqrt(2))
      N              = total simulations across all paths
      n_p            = simulations spent on path p
      lambda         = TDI penalty weight
      delta(p)       = current TDI estimate for path p (EMA)
      mu             = specificity boost weight (small, e.g. 0.2)
      specificity(p) = 0..1 — how well path matches target inventory

Pruning
-------
Any path attempted at least `k_min_prune` times whose rolling TDI
average stays above `TDI_BROADEN` (0.60) is permanently dropped —
addresses the "premature commitment" Type-B failure category.

Integration
-----------
EGATS is callback-driven: the caller supplies a `simulate_fn(path)`
that runs the actual pentest action and returns (success, tdi). This
keeps the planner unit-testable without spinning up an LLM.

Backprop
--------
After each simulation the outcome is written to every edge along
the path via `kg.update_edge_outcome(...)`, so the next call to
`kg.get_attack_paths(...)` reflects what was learned.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Callable

from app.harness.tda import TDI_BROADEN, TDI_EXPLOIT, Mode

logger = logging.getLogger(__name__)

# ── Defaults (from EVVO) ─────────────────────────────────────────────────
DEFAULT_UCB_C: float = math.sqrt(2.0)
DEFAULT_LAMBDA: float = 0.5
DEFAULT_K_MIN_PRUNE: int = 3
DEFAULT_MAX_ITERATIONS: int = 10
# Boost applied to the UCB score for paths anchored on specific product/
# component versions from the current target's inventory. Small so it nudges
# ordering without dominating the score — empirical cold-start safety.
DEFAULT_MU_SPECIFICITY: float = 0.2
DEFAULT_EMA_ALPHA: float = 0.3


# ── Dataclasses ──────────────────────────────────────────────────────────

@dataclass
class SimulationOutcome:
    """Result of one simulation, returned by the caller's callback."""

    success: bool
    tdi: float
    note: str = ""


SimulateFn = Callable[[list[str]], SimulationOutcome]


@dataclass
class _PathStat:
    """Internal per-path statistics — one per candidate path."""

    nodes: list[str]
    edges: list[tuple[str, str, str]]
    base_score: float          # phi(p) from KG
    attempts: int = 0
    successes: int = 0
    tdi_ema: float | None = None
    pruned: bool = False
    history: list[SimulationOutcome] = field(default_factory=list)
    # 0..1, carries KG.get_paths_for_inventory specificity: 1.0 exact
    # component_version match, 0.75 product_version, 0.5 component family,
    # 0.25 product family. 0.0 = legacy path (no inventory context).
    match_specificity: float = 0.0

    def key(self) -> tuple[str, ...]:
        return tuple(self.nodes)


@dataclass
class EGATSResult:
    """Result of EGATS.search() — best path + all explored paths."""

    best_path: _PathStat | None
    explored: list[_PathStat]
    iterations: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "iterations": self.iterations,
            "best_path": None if self.best_path is None else {
                "nodes": self.best_path.nodes,
                "score": round(self.best_path.base_score, 4),
                "attempts": self.best_path.attempts,
                "successes": self.best_path.successes,
                "tdi_ema": (
                    None if self.best_path.tdi_ema is None
                    else round(self.best_path.tdi_ema, 4)
                ),
            },
            "explored": [
                {
                    "nodes": p.nodes,
                    "attempts": p.attempts,
                    "successes": p.successes,
                    "pruned": p.pruned,
                }
                for p in self.explored
            ],
        }


# ── EGATS planner ────────────────────────────────────────────────────────

class EGATS:
    """Evidence-Guided Attack Tree Search planner.

    Holds no global state — one instance per pentest session.

    Usage:
        egats = EGATS(kg)
        egats.seed(source_node_ids=[...])
        result = egats.search(simulate_fn=my_simulate_callback)
        if result.best_path:
            # execute result.best_path.nodes
    """

    def __init__(
        self,
        kg: Any,  # KnowledgeGraph
        *,
        ucb_c: float = DEFAULT_UCB_C,
        lambda_penalty: float = DEFAULT_LAMBDA,
        k_min_prune: int = DEFAULT_K_MIN_PRUNE,
        ema_alpha: float = DEFAULT_EMA_ALPHA,
        mu_specificity: float = DEFAULT_MU_SPECIFICITY,
    ) -> None:
        self.kg = kg
        self.ucb_c = ucb_c
        self.lambda_penalty = lambda_penalty
        self.k_min_prune = k_min_prune
        self.ema_alpha = ema_alpha
        self.mu_specificity = mu_specificity
        self._stats: dict[tuple[str, ...], _PathStat] = {}

    # ── candidate seeding ─────────────────────────────────────────

    def seed(
        self,
        source_node_ids: list[str],
        *,
        max_depth: int = 4,
        top_k: int = 10,
        target_kind: Any = None,  # NodeType.FINDING
    ) -> int:
        """Pull initial candidate paths from the KG. Returns count seeded.

        Args:
            source_node_ids: starting node IDs (usually Technology nodes)
            max_depth: max path length
            top_k: max paths to seed
            target_kind: target NodeType (default Finding)

        Returns:
            number of new paths seeded
        """
        # Lazy import to avoid circular dependency
        from app.kg.types import NodeType
        if target_kind is None:
            target_kind = NodeType.FINDING

        paths = self.kg.get_attack_paths(
            source_node_ids,
            max_depth=max_depth,
            top_k=top_k,
            target_kind=target_kind,
        )
        seeded = 0
        for p in paths:
            stat = _PathStat(
                nodes=list(p.nodes),
                edges=list(p.edges),
                base_score=float(p.score),
                match_specificity=float(p.match_specificity),
            )
            key = stat.key()
            if key not in self._stats:
                self._stats[key] = stat
                seeded += 1
        return seeded

    def seed_from_inventory(
        self,
        inventory: list[Any],
        *,
        max_depth: int = 4,
        top_k: int = 10,
        target_kind: Any = None,
    ) -> int:
        """Seed candidate paths from the target's inventory.

        Args:
            inventory: list of inventory items (Technology nodes)
            max_depth, top_k, target_kind: same as seed()

        Returns:
            number of new paths seeded (0 if KG has no inventory data)
        """
        from app.kg.types import NodeType
        if target_kind is None:
            target_kind = NodeType.FINDING

        if not inventory:
            return 0
        # Convert inventory items to source node IDs
        source_ids: list[str] = []
        for item in inventory:
            if isinstance(item, str):
                source_ids.append(item)
            elif hasattr(item, "id"):
                source_ids.append(str(item.id))
            elif isinstance(item, dict) and "id" in item:
                source_ids.append(str(item["id"]))
        if not source_ids:
            return 0
        return self.seed(source_ids, max_depth=max_depth, top_k=top_k, target_kind=target_kind)

    # ── selection ─────────────────────────────────────────────────

    def _ucb(self, stat: _PathStat, total_n: int) -> float:
        """UCB score for a path. Unattempted paths get +inf so they're
        always explored once.
        """
        if stat.attempts == 0:
            return math.inf
        explore = self.ucb_c * math.sqrt(math.log(max(total_n, 1)) / stat.attempts)
        delta = stat.tdi_ema if stat.tdi_ema is not None else 0.5
        # Boost paths that match the current target's inventory precisely.
        spec_bonus = self.mu_specificity * stat.match_specificity
        return stat.base_score + explore - self.lambda_penalty * delta + spec_bonus

    def _alive(self) -> list[_PathStat]:
        """All non-pruned paths."""
        return [s for s in self._stats.values() if not s.pruned]

    def _select(self) -> _PathStat | None:
        """Select the path with the highest UCB score."""
        alive = self._alive()
        if not alive:
            return None
        total_n = sum(s.attempts for s in alive)
        return max(alive, key=lambda s: self._ucb(s, total_n))

    # ── backprop + pruning ────────────────────────────────────────

    def _backprop(self, stat: _PathStat, outcome: SimulationOutcome) -> None:
        """Update path statistics + persist outcome to KG edges."""
        stat.attempts += 1
        if outcome.success:
            stat.successes += 1
        # EMA update for TDI
        if stat.tdi_ema is None:
            stat.tdi_ema = outcome.tdi
        else:
            stat.tdi_ema = (
                (1.0 - self.ema_alpha) * stat.tdi_ema + self.ema_alpha * outcome.tdi
            )
        stat.history.append(outcome)

        # Persist outcome to KG so future scans inherit the lesson.
        for src, dst, _rel in stat.edges:
            try:
                self.kg.update_edge_outcome(
                    src, dst, success=outcome.success, scan_id=None,
                )
            except Exception as exc:
                logger.debug("update_edge_outcome failed for (%s,%s): %s", src, dst, exc)

    def _maybe_prune(self, stat: _PathStat) -> None:
        """Prune paths whose TDI EMA stays above TDI_BROADEN after k_min_prune attempts."""
        if stat.pruned or stat.attempts < self.k_min_prune:
            return
        if stat.tdi_ema is not None and stat.tdi_ema > TDI_BROADEN:
            stat.pruned = True
            logger.info(
                "EGATS pruned path %s after %d attempts (tdi_ema=%.3f)",
                "→".join(stat.nodes), stat.attempts, stat.tdi_ema,
            )

    # ── main loop ─────────────────────────────────────────────────

    def search(
        self,
        simulate_fn: SimulateFn,
        *,
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
    ) -> EGATSResult:
        """Run up to `max_iterations` of select→simulate→backprop→prune.

        Stops early when:
            - No alive path remains
            - One path has reached >= k_min_prune attempts with at least
              one success (early exploitation signal)

        Args:
            simulate_fn: callback that takes a path (list of node IDs) and
                returns a SimulationOutcome (success, tdi, note).
            max_iterations: max search iterations.

        Returns:
            EGATSResult with best_path + all explored paths.
        """
        i = 0
        for _ in range(max_iterations):
            stat = self._select()
            if stat is None:
                break
            i += 1
            outcome = simulate_fn(stat.nodes)
            self._backprop(stat, outcome)
            self._maybe_prune(stat)
            if (
                stat.attempts >= self.k_min_prune
                and stat.successes >= 1
                and not stat.pruned
            ):
                # Found a reliable path — exploit signal, stop searching.
                break

        explored = sorted(
            self._stats.values(),
            key=lambda s: (s.successes, -s.attempts),
            reverse=True,
        )
        best = next((s for s in explored if s.successes > 0 and not s.pruned), None)
        if best is None:
            best = next((s for s in explored if not s.pruned), None)
        return EGATSResult(best_path=best, explored=explored, iterations=i)

    # ── inspection helpers ────────────────────────────────────────

    def stats(self) -> list[_PathStat]:
        """Return all path statistics (for debugging / UI)."""
        return list(self._stats.values())

    def mode_for(self, stat: _PathStat) -> Mode:
        """Map a path's current TDI EMA back to a TDA mode.

        Args:
            stat: _PathStat instance

        Returns:
            Mode.BROADEN if TDI > 0.60, Mode.EXPLOIT if TDI < 0.30,
            Mode.LLM_PICK otherwise.
        """
        delta = stat.tdi_ema if stat.tdi_ema is not None else 0.5
        if delta > TDI_BROADEN:
            return Mode.BROADEN
        if delta < TDI_EXPLOIT:
            return Mode.EXPLOIT
        return Mode.LLM_PICK


__all__ = [
    # Constants
    "DEFAULT_UCB_C", "DEFAULT_LAMBDA", "DEFAULT_K_MIN_PRUNE",
    "DEFAULT_MAX_ITERATIONS", "DEFAULT_MU_SPECIFICITY", "DEFAULT_EMA_ALPHA",
    # Dataclasses
    "SimulationOutcome", "EGATSResult",
    # Main class
    "EGATS",
    # Type alias
    "SimulateFn",
]