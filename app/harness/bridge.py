"""
VAPT-AI Harness Bridge — unified control tower connecting RL + TDA + EGATS +
KG + Verifier + Self-Correction + Replay Trace.

Python port of EVVO Sentinel shield_engine/harness_bridge.py (selective port).

Per-scan instance (NOT global singleton — each scan creates its own bridge
to hold per-session state: _turn_count, _episode_reward, _findings_before,
_last_actions).

Deferred to Phase 2+ (per master plan W18 + user directive to only port
features that exist in CyberStrikeAI or EVVO):
  - Model Router (EVVO opt-in feature)
  - Command Gater (VAPT-AI uses app/sandbox/scope_guard.py instead)
  - Hierarchical Memory (VAPT-AI uses PentestFact blackboard instead)
  - Adaptive State Encoder (VAPT-AI uses fixed 337-dim encoder)

Main loop integration (called by scan_pipeline each turn):
    bridge = HarnessBridge(scan_id, target)
    # each turn:
    action_result = bridge.select_action(session, turn)
    # ... agent executes action ...
    bridge.record_experience(session, turn, action_idx, reward, done)
    # periodically:
    bridge.train_step(batch_size=64)
    # when verifier rejects:
    bridge.process_rejection(finding, verification_result, session)
    # at scan end:
    bridge.end_episode()
    await bridge.recorder.finalize(session, policy=bridge.policy)
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class HarnessBridge:
    """Unified bridge connecting RL + TDA + EGATS + KG + Verifier + Self-Correction.

    Maintains per-session state and provides a clean API for the agent loop.
    One instance per scan (not a global singleton).
    """

    def __init__(
        self,
        scan_id: str,
        target_url: str = "",
        *,
        policy: Any = None,        # PentestPolicy (optional — lazy-load if None)
        store: Any = None,         # ExperienceStore (optional)
        kg: Any = None,            # KnowledgeGraph (optional)
        verifier: Any = None,      # VulnerabilityVerifier (optional)
    ) -> None:
        self.scan_id = scan_id
        self.target_url = target_url

        # Lazy-init components (use provided or load from singletons)
        self._policy = policy
        self._store = store
        self._kg = kg
        self._verifier = verifier
        self._self_correction: Any = None
        self._egats: Any = None
        self._recorder: Any = None

        # Session tracking
        self._turn_count = 0
        self._episode_reward = 0.0
        self._findings_before = 0
        self._verified_before = 0
        self._last_actions: dict[str, dict[str, Any]] = {}

        logger.info(
            "HarnessBridge initialized | scan=%s | target=%s",
            scan_id, target_url[:80],
        )

    # ── lazy init ─────────────────────────────────────────────────

    def _get_policy(self) -> Any:
        """Get RL PentestPolicy — use provided, or app/main.py singleton."""
        if self._policy is None:
            # Import here to avoid circular dependency at module load
            import app.main as main_mod
            if hasattr(main_mod, "_rl_policy") and main_mod._rl_policy is not None:
                self._policy = main_mod._rl_policy
            else:
                from app.rl import PentestPolicy
                self._policy = PentestPolicy()
                logger.warning(
                    "RL policy not bootstrapped in main.py — using fresh instance "
                    "(ε will not decay across scans)",
                )
        return self._policy

    def _get_store(self) -> Any:
        """Get RL ExperienceStore — use provided, or app/main.py singleton."""
        if self._store is None:
            import app.main as main_mod
            if hasattr(main_mod, "_rl_store") and main_mod._rl_store is not None:
                self._store = main_mod._rl_store
            else:
                from app.rl import ExperienceStore
                self._store = ExperienceStore()
        return self._store

    def _get_kg(self) -> Any:
        """Get KnowledgeGraph singleton."""
        if self._kg is None:
            from app.kg import get_kg
            self._kg = get_kg()
        return self._kg

    def _get_verifier(self) -> Any:
        """Get VulnerabilityVerifier singleton."""
        if self._verifier is None:
            from app.harness.verifier import get_verifier
            self._verifier = get_verifier()
        return self._verifier

    def _get_self_correction(self) -> Any:
        """Get SelfCorrectionEngine singleton."""
        if self._self_correction is None:
            from app.harness.self_correction import get_self_correction_engine
            self._self_correction = get_self_correction_engine()
        return self._self_correction

    def _get_egats(self) -> Any:
        """Get EGATS instance for this scan (per-scan, not singleton)."""
        if self._egats is None:
            from app.harness.egats import EGATS
            from app.core.config import settings
            self._egats = EGATS(
                self._get_kg(),
                ucb_c=settings.kg_ucb_c,
                lambda_penalty=settings.kg_lambda_penalty,
                k_min_prune=settings.kg_k_min_prune,
                ema_alpha=settings.kg_ema_alpha,
                mu_specificity=settings.kg_mu_specificity,
            )
        return self._egats

    def _get_recorder(self) -> Any:
        """Get ReplayTraceRecorder for this scan."""
        if self._recorder is None:
            from app.trace import ReplayTraceRecorder
            self._recorder = ReplayTraceRecorder(scan_id=self.scan_id, session_id=self.scan_id)
        return self._recorder

    @property
    def recorder(self) -> Any:
        """Public access to the ReplayTraceRecorder."""
        return self._get_recorder()

    @property
    def policy(self) -> Any:
        """Public access to the RL PentestPolicy."""
        return self._get_policy()

    # ── RL action selection ───────────────────────────────────────

    def select_action(self, session: dict[str, Any], turn: int) -> dict[str, Any]:
        """Select next action via RL policy + EGATS path ranking.

        Args:
            session: live session dict (will be mutated with _turn, _rl_state_vec,
                _rl_action_idx, last_recommendation)
            turn: current turn number

        Returns:
            dict with: action_index, action_name, meta, state, state_vec,
            kg_consult_result (optional), egats_paths (optional)
        """
        from app.rl import StateEncoder

        session["_turn"] = turn
        state = StateEncoder.from_session(session)
        state_vec = state.to_vector()

        # EGATS: consult KG for attack paths (if KG has data for this target)
        kg_consult_result = None
        egats_paths: list = []
        try:
            kg = self._get_kg()
            if kg.graph.number_of_nodes() > 0:
                # Find Technology nodes matching the target's tech stack
                recon = session.get("recon_signals") or {}
                tech_stack = recon.get("technologies") or []
                source_ids: list[str] = []
                for tech in tech_stack[:5]:  # cap at 5 sources
                    node_id = kg.find_node(
                        # Try both "tag:..." and direct tech name
                        __import__("app.kg.types", fromlist=["NodeType"]).NodeType.TECHNOLOGY,
                        f"tag:{tech.lower()}",
                    )
                    if node_id:
                        source_ids.append(node_id)
                if source_ids:
                    consult = kg.consult(source_ids, max_depth=4, top_k=5)
                    kg_consult_result = consult.as_dict()
                    egats_paths = consult.paths
                    session["last_kg_consult"] = kg_consult_result
        except Exception as exc:
            logger.debug("KG consult skipped: %s", exc)

        # RL policy: select action from 337-dim state
        policy = self._get_policy()
        action_idx, action_name, meta = policy.select_action(state_vec)

        session["_rl_state_vec"] = state_vec
        session["_rl_action_idx"] = action_idx

        result = {
            "action_index": action_idx,
            "action_name": action_name,
            "meta": meta,
            "state": state,
            "state_vec": state_vec,
            "kg_consult_result": kg_consult_result,
            "egats_paths": [p.as_dict() for p in egats_paths],
        }

        self._last_actions[self.scan_id] = {
            "action_index": action_idx,
            "action_name": action_name,
            "meta": meta,
            "turn": turn,
        }

        # Record to replay trace
        try:
            self._get_recorder().event(
                "action_selected",
                turn=turn,
                data={
                    "action_index": action_idx,
                    "action_name": action_name,
                    "epsilon": meta.get("epsilon"),
                    "exploratory": meta.get("exploratory"),
                    "q_values": meta.get("q_values", [])[:7],  # cap for trace size
                    "kg_paths_count": len(egats_paths),
                },
            )
        except Exception as exc:
            logger.debug("trace action_selected failed: %s", exc)

        return result

    # ── experience recording ──────────────────────────────────────

    async def record_experience(
        self,
        session: dict[str, Any],
        turn: int,
        action_idx: int,
        reward: float,
        done: bool,
        *,
        observation: str = "",
        action_name: str | None = None,
    ) -> None:
        """Record (s, a, r, s', done) experience to RL store + replay trace.

        Args:
            session: live session dict
            turn: turn number
            action_idx: action taken
            reward: reward received
            done: terminal flag
            observation: tool output / observation text (for trace)
            action_name: human-readable action name (for trace)
        """
        store = self._get_store()

        prev_state_vec = session.get("_rl_state_vec")
        if prev_state_vec is None:
            logger.debug("No previous state recorded, skipping experience")
            return

        from app.rl import StateEncoder
        current_state = StateEncoder.from_session(session)
        current_state_vec = current_state.to_vector()

        # Add to RL experience store (async — writes to SQL)
        await store.add(
            scan_id=self.scan_id,
            turn=turn,
            state=prev_state_vec,
            action=action_idx,
            reward=reward,
            next_state=current_state_vec,
            done=done,
            action_name=action_name,
        )

        self._episode_reward += reward
        self._turn_count = turn

        # Record to replay trace
        try:
            self._get_recorder().event(
                "turn_end",
                turn=turn,
                data={
                    "action_index": action_idx,
                    "action_name": action_name,
                    "reward": round(reward, 4),
                    "done": done,
                    "observation_preview": observation[:500] if observation else "",
                    "episode_reward_so_far": round(self._episode_reward, 4),
                },
                reward=reward,
            )
        except Exception as exc:
            logger.debug("trace turn_end failed: %s", exc)

        logger.debug(
            "Experience recorded | scan=%s | turn=%d | action=%d | reward=%.2f | episode_total=%.2f",
            self.scan_id, turn, action_idx, reward, self._episode_reward,
        )

    # ── training ──────────────────────────────────────────────────

    def train_step(self, batch_size: int = 64) -> dict[str, Any]:
        """Run one RL training step (PER-sampled batch).

        Returns:
            dict with: trained (bool), loss, td_errors (first 5), epsilon,
            buffer_size. If insufficient experiences, returns {trained: False}.
        """
        policy = self._get_policy()
        store = self._get_store()

        if len(store) < batch_size // 2:
            return {"trained": False, "reason": "insufficient_experiences", "buffer_size": len(store)}

        from app.rl import train_one_step
        loss, td_errors = train_one_step(policy, store, batch_size=batch_size)

        return {
            "trained": True,
            "loss": round(loss, 6),
            "td_errors": [round(e, 4) for e in td_errors[:5]],
            "epsilon": round(policy.epsilon, 4),
            "buffer_size": len(store),
        }

    def end_episode(self) -> None:
        """End the scan episode — decay epsilon + reset per-session counters.

        Call this at scan completion (in pipeline phase 'complete').
        """
        policy = self._get_policy()
        policy.end_episode(self._episode_reward)

        logger.info(
            "Episode ended | scan=%s | reward=%.2f | turns=%d | epsilon=%.4f",
            self.scan_id, self._episode_reward, self._turn_count, policy.epsilon,
        )

        self._episode_reward = 0.0
        self._turn_count = 0
        self._findings_before = 0
        self._verified_before = 0

    # ── self-correction ───────────────────────────────────────────

    def process_rejection(
        self,
        finding: dict[str, Any],
        verification_result: Any,
        session: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Process a verifier rejection — triggers self-correction + negative RL reward.

        Args:
            finding: finding dict (agent output format)
            verification_result: VerificationResult from verifier
            session: optional live session dict (for RL state vector)

        Returns:
            dict from SelfCorrectionEngine.process_rejection():
              corrective_message, rl_reward, pattern_blacklisted,
              rejection_analysis, pattern_rejection_count
        """
        from app.harness.self_correction import process_verification_rejection

        result = process_verification_rejection(
            finding=finding,
            verification_result=verification_result,
            session_id=self.scan_id,
            session=session,
        )

        # Record rejection to RL store with negative reward
        if session and "_rl_state_vec" in session:
            try:
                store = self._get_store()
                prev_state = session["_rl_state_vec"]
                action_idx = session.get("_rl_action_idx", 5)  # default: report_vulnerability
                from app.rl import StateEncoder
                current_state = StateEncoder.from_session(session)

                # async add — but process_rejection is sync, so we schedule it
                # The caller (scan_pipeline) is async and should await this.
                # For now, we add to in-memory SumTree only (DB write deferred).
                import asyncio
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # We're in an async context — create a task
                    asyncio.create_task(store.add(
                        scan_id=self.scan_id,
                        turn=session.get("_turn", 0),
                        state=prev_state,
                        action=action_idx,
                        reward=result["rl_reward"],
                        next_state=current_state.to_vector(),
                        done=False,
                        priority=abs(result["rl_reward"]) * 2,
                        action_name="report_vulnerability_rejected",
                    ))
                else:
                    # No event loop — sync fallback (in-memory only)
                    store.tree.add(
                        priority=abs(result["rl_reward"]) * 2,
                        data={
                            "scan_id": self.scan_id,
                            "turn": session.get("_turn", 0),
                            "state": prev_state.tolist(),
                            "action": action_idx,
                            "reward": result["rl_reward"],
                            "next_state": current_state.to_vector().tolist(),
                            "done": False,
                        },
                    )
            except Exception as exc:
                logger.debug("rejection_experience_failed: %s", exc)

        # Record to replay trace
        try:
            self._get_recorder().event(
                "verifier_rejection",
                turn=session.get("_turn") if session else None,
                data={
                    "finding_name": finding.get("name", ""),
                    "rl_reward": round(result["rl_reward"], 4),
                    "rejection_type": result["rejection_analysis"].rejection_type,
                    "pattern_blacklisted": result["pattern_blacklisted"],
                    "pattern_rejection_count": result["pattern_rejection_count"],
                },
                reward=result["rl_reward"],
            )
        except Exception as exc:
            logger.debug("trace verifier_rejection failed: %s", exc)

        return result

    # ── persistence ───────────────────────────────────────────────

    def get_last_rl_action(self) -> dict[str, Any] | None:
        """Get the last RL action for this scan (for debugging / UI)."""
        return self._last_actions.get(self.scan_id)

    def get_stats(self) -> dict[str, Any]:
        """Get bridge statistics for debugging / UI."""
        stats: dict[str, Any] = {
            "scan_id": self.scan_id,
            "target_url": self.target_url,
            "turn_count": self._turn_count,
            "episode_reward": round(self._episode_reward, 2),
        }
        if self._policy:
            stats["policy"] = self._policy.get_stats()
        if self._store:
            stats["experience_store"] = self._store.get_stats()
        return stats


# ── Module-level registry (per-scan instances) ───────────────────────────

_bridges: dict[str, "HarnessBridge"] = {}


def get_harness_bridge(scan_id: str, target_url: str = "") -> HarnessBridge:
    """Get or create a HarnessBridge for a scan_id.

    One bridge per scan — holds per-session state. Bridges are cleaned up
    via clear_harness_bridge() at scan end.
    """
    if scan_id not in _bridges:
        _bridges[scan_id] = HarnessBridge(scan_id=scan_id, target_url=target_url)
    return _bridges[scan_id]


def clear_harness_bridge(scan_id: str) -> None:
    """Remove a bridge from the registry (call at scan end)."""
    _bridges.pop(scan_id, None)


def get_all_bridge_stats() -> dict[str, dict[str, Any]]:
    """Get stats for all active bridges (for /api/harness/stats endpoint)."""
    return {sid: b.get_stats() for sid, b in _bridges.items()}


__all__ = [
    "HarnessBridge",
    "get_harness_bridge",
    "clear_harness_bridge",
    "get_all_bridge_stats",
]