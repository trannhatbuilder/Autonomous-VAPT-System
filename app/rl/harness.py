"""
VAPT-AI RL — Harness Bridge.

Single entry point connecting the ReAct agent loop to the RL policy,
experience store, and training pipeline. One instance per scan.

Per-turn contract:
    harness = get_harness(scan_id, target)        # at scan start

    # 1. BEFORE the LLM call — encode state, query policy, format hint
    rec = harness.select_action(session_dict, turn)
    sys_msg = {"role": "system", "content": rec.guidance_text}
    turn_messages = base_messages + [sys_msg]
    response = await chat_completion(..., messages=turn_messages, ...)

    # 2. AFTER the tool result lands — compute reward, push transition
    await harness.record_experience(
        session_next, turn, rec.action_index, reward, done=False,
        reward_breakdown=breakdown,
    )
    harness.record_action_followed(followed=True_or_False)

    # 3. At scan end — cancel train worker, decay epsilon, save checkpoint
    await harness.end_scan(episode_reward=harness.get_stats()["episode_reward"])
    release_harness(scan_id)

Threading model
---------------
The Q-network is plain numpy (no torch / tf). Numpy arrays are NOT
thread-safe across readers + a writer. To prevent select_action() from
reading a half-updated weight matrix while the training worker is
mid-backprop, all Q-network access goes through an asyncio.Lock:

    select_action()   sync    acquires _q_lock for ~1 ms (forward pass)
    _train_worker()   async   acquires _q_lock for ~50 ms per batch

select_action is therefore blocked for at most ~50 ms every TRAIN_INTERVAL_TURNS
turns — acceptable for pentest scans where each LLM round-trip is 1-5 s.

Wired surfaces
--------------
Everything the bridge originally scoped for "later phases" is now live;
see the individual methods rather than relying on this list, but in
short:
    - KG (Knowledge Graph): `kg` property + `kg_consult()` + `kg_merge_fork()`
    - TDA (Task Difficulty Index): `tdi_log` populated per turn
    - EGATS (UCB attack-path planner): `egats_select()` /
      `egats_mark_hint_injected()` / `egats_record_outcome()`
    - Self-correction on verifier rejection: `process_rejection()`

Unwired surfaces are the ones that legacy code paths still fake — do not
assume a feature exists just because the state dict has a key for it.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from app.rl.experience_store import ExperienceStore
from app.rl.policy import PentestPolicy
from app.rl.state_encoder import ACTION_SPACE, StateEncoder

logger = logging.getLogger(__name__)


# ─── tunables (env-overridable) ───────────────────────────────────────
def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    try:
        return int(raw) if raw is not None else default
    except ValueError:
        return default


TRAIN_INTERVAL_TURNS: int = _env_int("VAPT_AI_RL_TRAIN_INTERVAL_TURNS", 5)
TRAIN_BATCH_SIZE: int = _env_int("VAPT_AI_RL_TRAIN_BATCH_SIZE", 64)
MIN_BUFFER_FOR_TRAIN: int = _env_int("VAPT_AI_RL_MIN_BUFFER_FOR_TRAIN", 32)
TRAIN_GRACE_PERIOD_S: float = 0.25  # poll cadence inside the worker


# ─── KG integration toggle (Phase 2) ─────────────────────────────────
def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.lower() in ("1", "true", "yes", "on")


# KG is ON by default. Set VAPT_AI_KG_ENABLED=false to disable KG
# integration entirely (the harness will skip consult + outcome recording).
KG_ENABLED_DEFAULT: bool = _env_bool("VAPT_AI_KG_ENABLED", True)
KG_TOP_K_DEFAULT: int = _env_int("VAPT_AI_KG_TOP_K", 5)


# ─── TDA integration toggle (Phase 3) ────────────────────────────────
# TDA is ON by default. Set VAPT_AI_TDA_ENABLED=false to disable the
# per-turn TDI computation and LLM hint injection.
TDA_ENABLED_DEFAULT: bool = _env_bool("VAPT_AI_TDA_ENABLED", True)
# Critical override: when findings exist, force EXPLOIT mode regardless
# of the raw TDI. This is the #1 bug fix from the reference design —
# without it, the agent loops in broaden_recon after confirming a finding.
TDA_FORCE_EXPLOIT_ON_FINDING: bool = _env_bool(
    "VAPT_AI_TDA_FORCE_EXPLOIT_ON_FINDING", True,
)
# Horizon normalisation. 0 = auto: use the scan's actual turn budget
# (max_turns). A fixed value smaller than max_turns would saturate H at
# 1.0 for most of the scan (bug: 20 vs a 100-turn loop). An explicit
# value larger than max_turns is honoured but can never shrink below it.
TDA_MAX_HORIZON_DEFAULT: int = _env_int("VAPT_AI_TDA_MAX_HORIZON", 0)
# Model context window used as the C denominator (context SATURATION,
# not scan spend). 128K matches the default `max_total_tokens` of
# app.core.channels.ChannelConfig; the agent loop overrides this with
# the live LLM channel's value when it is available.
MODEL_CONTEXT_WINDOW_DEFAULT: int = _env_int(
    "VAPT_AI_MODEL_CONTEXT_WINDOW", 128_000,
)


# ─── EGATS integration toggle (Phase 4) ──────────────────────────────
# EGATS is ON by default. Set VAPT_AI_EGATS_ENABLED=false to disable
# the UCB1 attack-path planner (the harness will skip seed/select/
# record_outcome — no path hint injected).
EGATS_ENABLED_DEFAULT: bool = _env_bool("VAPT_AI_EGATS_ENABLED", True)


# ─── Self-Correction integration toggle (Phase 5) ────────────────────
# Self-Correction is ON by default. Set VAPT_AI_SC_ENABLED=false to
# disable rejection processing (no corrective message, no escalating
# RL reward, no pattern tracking).
SC_ENABLED_DEFAULT: bool = _env_bool("VAPT_AI_SC_ENABLED", True)


# ─── Per-scan KG Fork toggle (Phase 6) ───────────────────────────────
# OFF by default — writes go straight to the main KG (Phase 2-5 behavior).
# When true, each scan forks the KG into a deepcopy; the fork is merged
# back into the main KG only on scan status="completed".
KG_ISOLATION_DEFAULT: bool = _env_bool("VAPT_AI_KG_ISOLATION", False)


# ─── action -> human-readable hint ────────────────────────────────────
# Maps the 7 discrete RL actions to the tool families the agent actually
# sees. Used by format_guidance() to render an actionable hint rather
# than a bare enum string the LLM cannot act on.
_ACTION_HINT: dict[str, str] = {
    "execute_command_recon": (
        "reconnaissance tools (nmap, httpx, whatweb, subfinder, amass, masscan, "
        "rustscan, gau, waybackurls)"
    ),
    "execute_command_exploit": (
        "exploitation tools (sqlmap, metasploit, hydra, impacket, netexec, "
        "mimikatz, john, hashcat, linpeas, winpeas) — only if you have "
        "confirmed evidence from a prior phase"
    ),
    "execute_command_fuzz": (
        "fuzzing & enumeration tools (ffuf, gobuster, feroxbuster, katana, "
        "nikto, nuclei, dalfox, wpscan)"
    ),
    "select_skill": (
        "load an additional skill playbook relevant to the current tech stack"
    ),
    "consult_kg": (
        "consult the knowledge graph for prior-attack intel on this stack"
    ),
    "report_vulnerability": (
        "record a confirmed finding via record_vulnerability — only if you "
        "have evidence in hand"
    ),
    "pentest_complete": (
        "wrap up — call exit with a summary of findings"
    ),
}


@dataclass
class TurnRecommendation:
    """Snapshot of what the RL policy said for one turn.

    The agent loop reads `guidance_text` and appends it as a separate
    system message before calling the LLM. Everything else is metadata
    for logging / SSE / metrics.
    """

    action_index: int                  # 0..ACTION_DIM-1
    action_name: str                   # e.g. "execute_command_recon"
    q_values: list[float]              # raw Q(s, a) for all actions
    q_max: float                       # max(q_values) — confidence proxy
    epsilon: float                     # current exploration rate
    exploratory: bool                  # True if action was a random pick
    curiosity_bonus: float             # intrinsic-curiosity contribution
    guidance_text: str                 # pre-rendered advisory system message
    meta: dict[str, Any] = field(default_factory=dict)


class HarnessBridge:
    """Per-scan RL bridge.

    Lifecycle:
        __init__       load checkpoint, spawn train worker
        select_action  called once per turn (sync)
        record_*       called once per turn (async)
        end_scan       cancel worker, decay epsilon, save checkpoint
    """

    def __init__(self, scan_id: str, target: str) -> None:
        self.scan_id = scan_id
        self.target = target

        # ── core RL components ────────────────────────────────────
        self._policy = PentestPolicy()
        self._store = ExperienceStore()

        # ── concurrency primitives ────────────────────────────────
        # _q_lock  : guards Q-network reads + writes (numpy not thread-safe)
        # _train_cv: signals worker to wake up early (e.g. on end_scan)
        self._q_lock = asyncio.Lock()
        self._train_cv = asyncio.Condition()
        self._stop_event = asyncio.Event()
        self._pending_trains = 0
        self._turns_since_train = 0

        # ── per-episode bookkeeping ───────────────────────────────
        self._episode_reward = 0.0
        self._turn_count = 0
        self._followed_count = 0
        self._exploratory_count = 0
        self._train_log: list[dict[str, Any]] = []

        # ── RL transition pairing ─────────────────────────────────
        # The agent loop builds a fresh session dict each turn (via
        # build_session), so we CANNOT stash _rl_state_vec in the session
        # dict and expect it to survive to the next turn. Instead, we
        # track the previous turn's state vector + action internally.
        #
        # select_action() stashes the current state vector here.
        # record_experience() reads it as the "prev_state" half of the
        # (s, a, r, s') transition, then stashes the post-turn state
        # vector for the NEXT turn's pairing.
        self._prev_state_vec: np.ndarray | None = None
        self._prev_action_idx: int | None = None

        # ── last action (so record_action_followed knows what to credit) ──
        self._last_action_name: str | None = None

        # ── KG integration state (Phase 2 + Phase 6) ─────────────────
        # Lazy-attached: the KG singleton is only fetched when the agent
        # loop calls attach_kg() (typically right after harness init).
        # This keeps unit tests fast (no DB needed) and lets us disable
        # KG by simply never calling attach_kg().
        #
        # Phase 6 — Per-scan KG Fork:
        # When VAPT_AI_KG_ISOLATION=true, attach_kg() forks the main KG
        # into self._kg (a deepcopy). All consult/hit/miss/edge_outcome
        # writes go to the fork. kg_merge_fork() merges the fork back
        # into the main KG ONLY on scan status="completed" — failed/
        # aborted scans don't corrupt the global KG.
        self._kg: Any = None  # KnowledgeGraph | None (fork when isolating)
        self._kg_main: Any = None  # main KG singleton (merge target)
        self._kg_enabled: bool = KG_ENABLED_DEFAULT
        self._kg_isolation_enabled: bool = KG_ISOLATION_DEFAULT
        self._kg_fork_active: bool = False  # True when self._kg is a fork
        self._last_consult_id: str | None = None
        self._tech_stack: list[str] = []
        self._kg_consult_count: int = 0
        self._kg_hit_count: int = 0
        self._kg_miss_count: int = 0

        # ── TDA integration state (Phase 3) ──────────────────────────
        # Per-turn TDI history. The last entry is the "current" TDI;
        # StateEncoder reads it to populate the 337-dim state vector.
        # The agent loop reads it to inject the TDA hint into the LLM
        # prompt. Empty list until tda_compute() is called for the
        # first time.
        self._tdi_log: list[dict[str, Any]] = []
        self._tda_enabled: bool = TDA_ENABLED_DEFAULT
        self._tda_override_count: int = 0
        self._tda_max_horizon: int = TDA_MAX_HORIZON_DEFAULT

        # ── EGATS integration state (Phase 4) ────────────────────────
        # Lazy-created: the EGATS planner is only instantiated when
        # egats_seed() is called (typically after the first KG consult
        # in Phase 2). Stays None when EGATS is disabled or when no KG
        # paths are available.
        self._egats: Any = None  # EGATS | None
        self._egats_enabled: bool = EGATS_ENABLED_DEFAULT
        self._egats_seeded: bool = False
        # Path keys are tracked in TWO slots on purpose:
        #   _egats_selected_path_key — what UCB picked this turn (display
        #       only; the round-robin injects the EGATS hint on ~1 turn in 3).
        #   _egats_injected_path_key — the path whose hint was ACTUALLY
        #       placed in the prompt this turn. ONLY this key is eligible
        #       for outcome backprop; crediting the selected-but-unsent
        #       path made the planner learn from hints the LLM never saw.
        self._egats_selected_path_key: tuple[str, ...] | None = None
        self._egats_injected_path_key: tuple[str, ...] | None = None

        # ── Self-Correction integration state (Phase 5) ──────────────
        # Lazy-created: the SelfCorrectionEngine is only instantiated
        # when process_rejection() is called for the first time. Stays
        # None when Self-Correction is disabled or no rejection has
        # occurred yet.
        self._sc: Any = None  # SelfCorrectionEngine | None
        self._sc_enabled: bool = SC_ENABLED_DEFAULT

        # ── load checkpoint (best-effort — fresh policy on failure) ──
        self._load_checkpoint_safe()

        # ── spawn background training worker ──────────────────────
        self._train_worker_task = asyncio.create_task(self._train_worker())
        logger.info(
            "rl_harness_init | scan=%s | target=%s | epsilon=%.4f | "
            "buffer=%d | train_interval=%d",
            scan_id, target, self._policy.epsilon, len(self._store),
            TRAIN_INTERVAL_TURNS,
        )

    # ──────────────────────────────────────────────────────────────────
    # Public: per-turn entry points
    # ──────────────────────────────────────────────────────────────────

    async def select_action(
        self, session: dict[str, Any], turn: int
    ) -> TurnRecommendation:
        """Encode state, query policy, return recommendation + guidance.

        Acquires _q_lock briefly (Q forward pass ~1 ms).
        """
        session["_turn"] = turn
        state = StateEncoder.from_session(session)
        state_vec = state.to_vector()

        # Stash the current state vector internally so record_experience()
        # can pair it with the post-turn state to form (s, a, r, s').
        # We use an instance attribute (not the session dict) because the
        # agent loop builds a fresh session dict each turn.
        self._prev_state_vec = state_vec

        # Curiosity + policy forward under lock (Q read).
        async with self._q_lock:
            action_idx, action_name, meta = self._policy.select_action(state_vec)

        q_values = meta.get("q_values", [])
        q_max = float(max(q_values)) if q_values else 0.0
        epsilon = float(meta.get("epsilon", self._policy.epsilon))
        exploratory = bool(meta.get("exploratory", False))
        curiosity_bonus = float(meta.get("curiosity_bonus", 0.0))

        # Stash for record_action_followed() — the agent tells us after the
        # LLM picks a tool whether that tool matched our suggested category.
        self._last_action_name = action_name

        guidance = self.format_guidance(
            action_name=action_name,
            q_max=q_max,
            epsilon=epsilon,
            exploratory=exploratory,
        )

        if exploratory:
            self._exploratory_count += 1

        logger.info(
            "rl_select | scan=%s | turn=%d | action=%s | q_max=%.4f | "
            "epsilon=%.4f | exploratory=%s",
            self.scan_id, turn, action_name, q_max, epsilon, exploratory,
        )

        return TurnRecommendation(
            action_index=action_idx,
            action_name=action_name,
            q_values=q_values,
            q_max=q_max,
            epsilon=epsilon,
            exploratory=exploratory,
            curiosity_bonus=curiosity_bonus,
            guidance_text=guidance,
            meta=meta,
        )

    @staticmethod
    def format_guidance(
        *,
        action_name: str,
        q_max: float,
        epsilon: float,
        exploratory: bool,
    ) -> str:
        """Render the advisory system message for the LLM.

        Framing rules (Phase 1 — Advisory authority):
            - Always labeled "(advisory — your judgment wins)" in the header.
            - "Suggested action:" — never "Required action:".
            - "Hint:" — never "Do:".
            - When exploratory=True, explicitly tell the LLM to feel free to
              override.
            - When q_max < 0.10, flag the policy as "still learning" so the
              LLM knows the hint is low-confidence.
        """
        hint = _ACTION_HINT.get(action_name, action_name)

        # Confidence framing.
        if q_max < 0.10:
            confidence_line = (
                f"**Confidence:** q_max={q_max:.3f} (low — policy still "
                f"learning; treat this as a weak prior)"
            )
        else:
            confidence_line = f"**Confidence:** q_max={q_max:.3f}"

        # Exploration framing.
        if exploratory:
            exploration_line = (
                f"**Exploration:** epsilon={epsilon:.3f} (exploratory pick — "
                f"feel free to override)"
            )
        else:
            exploration_line = (
                f"**Exploration:** epsilon={epsilon:.3f} (greedy pick — "
                f"policy thinks this is the best action right now)"
            )

        return (
            "## RL Turn Guidance (advisory — your judgment wins)\n\n"
            f"**Suggested action:** {action_name}\n"
            f"{confidence_line}\n"
            f"{exploration_line}\n\n"
            f"**Hint:** Prioritize {hint}.\n\n"
            "This is one signal among many. If your reasoning disagrees, "
            "follow your reasoning — the policy learns from the outcome "
            "either way via the reward signal."
        )

    async def record_experience(
        self,
        session: dict[str, Any],
        turn: int,
        action_idx: int,
        reward: float,
        done: bool,
        reward_breakdown: dict[str, Any] | None = None,
        action_name: str | None = None,
    ) -> None:
        """Push (s, a, r, s', done) into the PER store and signal the worker.

        Async — may touch the database (ExperienceStore.add is async).

        State pairing: select_action() stashed the pre-turn state vector
        in self._prev_state_vec. We use that as the "s" half of the
        transition, encode the post-turn session as "s'", and push the
        full (s, a, r, s', done) tuple into the PER store. Then we
        update _prev_state_vec to s' so the NEXT turn can pair it.

        `action_idx` MUST be the action the agent actually executed this
        turn, not the action the policy suggested — see the callsite in
        react_agent (off-policy label mismatch: `train_step` puts the
        gradient on `grad_q[action]`, so a suggested-but-not-executed
        action trains the wrong head with another action's reward).

        `action_name` defaults to the last suggested name for backwards
        compatibility; callers that know the executed action should pass
        it explicitly so the stored label matches `action_idx`.
        """
        if self._prev_state_vec is None:
            # select_action was never called this turn (or harness was
            # reset). Nothing to pair — skip silently.
            logger.debug(
                "rl_record_skipped | scan=%s | turn=%d | reason=no_prev_state",
                self.scan_id, turn,
            )
            return

        prev_state_vec = self._prev_state_vec
        next_state_vec = StateEncoder.from_session(session).to_vector()

        try:
            await self._store.add(
                scan_id=self.scan_id,
                turn=turn,
                state=prev_state_vec,
                action=action_idx,
                reward=float(reward),
                next_state=next_state_vec,
                done=bool(done),
                action_name=action_name or self._last_action_name,
                reward_breakdown=reward_breakdown,
            )
        except Exception as exc:
            logger.warning(
                "rl_record_failed | scan=%s | turn=%d | err=%s",
                self.scan_id, turn, exc,
            )
            return

        self._episode_reward += float(reward)
        self._turn_count = turn

        # Stash post-state for the next turn's pairing.
        self._prev_state_vec = next_state_vec
        self._prev_action_idx = action_idx

        # Signal training worker.
        self._turns_since_train += 1
        if self._turns_since_train >= TRAIN_INTERVAL_TURNS:
            self._turns_since_train = 0
            async with self._train_cv:
                self._pending_trains += 1
                self._train_cv.notify_all()

        logger.info(
            "rl_reward | scan=%s | turn=%d | reward=%.4f | "
            "episode_reward=%.4f | buffer=%d",
            self.scan_id, turn, reward, self._episode_reward, len(self._store),
        )

    def record_action_followed(self, followed: bool) -> None:
        """Agent calls this after the LLM picks a tool this turn.

        `followed` is True if the LLM's tool choice mapped to the RL
        suggestion's category, False otherwise. Used for the follow-rate
        metric — high follow-rate with rising reward suggests the policy
        is giving useful hints.
        """
        if followed:
            self._followed_count += 1
        logger.debug(
            "rl_action_followed | scan=%s | followed=%s | "
            "follow_rate=%.3f",
            self.scan_id, followed,
            self._followed_count / max(self._turn_count, 1),
        )

    # ──────────────────────────────────────────────────────────────────
    # Public: KG integration (Phase 2)
    # ──────────────────────────────────────────────────────────────────

    def attach_kg(self, kg: Any = None) -> None:
        """Attach a KnowledgeGraph instance (defaults to the singleton).

        Lazy attachment: the harness does NOT touch the KG in __init__
        so unit tests can run without a DB. The agent loop calls this
        once after creating the harness, before the first turn.

        If VAPT_AI_KG_ENABLED=false, this is a no-op (KG stays None).

        Phase 6 — Per-scan KG Fork:
        When VAPT_AI_KG_ISOLATION=true, this method fetches the main KG
        singleton, then forks it into a per-scan deepcopy. All subsequent
        consult/hit/miss writes go to the fork. The fork is merged back
        into the main KG only when kg_merge_fork() is called with
        status="completed" (typically from the exit-tool success path).
        """
        if not self._kg_enabled:
            logger.info("kg_disabled | scan=%s", self.scan_id)
            return

        # Fetch the main KG (either passed in or via singleton).
        if kg is not None:
            main_kg = kg
        else:
            try:
                from app.kg.advisor import get_kg_safe
                main_kg = get_kg_safe()
            except Exception as exc:
                logger.warning(
                    "kg_attach_failed | scan=%s | err=%s | continuing without KG",
                    self.scan_id, exc,
                )
                self._kg = None
                self._kg_main = None
                return

        if main_kg is None:
            self._kg = None
            self._kg_main = None
            return

        self._kg_main = main_kg

        # Phase 6: fork if isolation is enabled.
        if self._kg_isolation_enabled and hasattr(main_kg, "fork_for_scan"):
            try:
                self._kg = main_kg.fork_for_scan(self.scan_id)
                self._kg_fork_active = True
                logger.info(
                    "kg_forked | scan=%s | fork_nodes=%s | fork_edges=%s",
                    self.scan_id,
                    getattr(self._kg, "stats", lambda: {})().get("nodes", "?"),
                    getattr(self._kg, "stats", lambda: {})().get("edges", "?"),
                )
            except Exception as exc:
                logger.warning(
                    "kg_fork_failed | scan=%s | err=%s | falling back to main KG",
                    self.scan_id, exc,
                )
                self._kg = main_kg  # fall back to main KG (no isolation)
                self._kg_fork_active = False
        else:
            # No isolation — use main KG directly (Phase 2-5 behavior).
            self._kg = main_kg
            self._kg_fork_active = False

        if self._kg is not None:
            logger.info(
                "kg_attached | scan=%s | nodes=%s | edges=%s | forked=%s",
                self.scan_id,
                getattr(self._kg, "stats", lambda: {})().get("nodes", "?"),
                getattr(self._kg, "stats", lambda: {})().get("edges", "?"),
                self._kg_fork_active,
            )

    def kg_merge_fork(self, *, scan_status: str) -> dict[str, Any] | None:
        """Merge the per-scan KG fork back into the main KG.

        Phase 6 — only meaningful when VAPT_AI_KG_ISOLATION=true and a
        fork was created. Called by the agent loop at scan end.

        Merge policy:
            - status="completed"  → merge fork into main KG (learning
              signal is preserved for future scans)
            - any other status    → discard fork (aborted/error/max_iters
              scans don't corrupt the global KG)

        Args:
            scan_status: the final scan status from run_react_scan's
                return dict ("completed" / "aborted" / "error" /
                "max_iterations" / "token_budget_exceeded").

        Returns:
            Merge stats dict (nodes_added, edges_added, edges_updated)
            when merged, None when fork not active or discarded.
        """
        if not self._kg_fork_active or self._kg is None or self._kg_main is None:
            return None

        if scan_status != "completed":
            logger.info(
                "kg_fork_discarded | scan=%s | status=%s | fork_not_merged",
                self.scan_id, scan_status,
            )
            self._kg_fork_active = False
            self._kg = None
            return None

        try:
            if not hasattr(self._kg_main, "merge_scan"):
                logger.warning(
                    "kg_merge_unavailable | scan=%s | main KG has no merge_scan()",
                    self.scan_id,
                )
                self._kg_fork_active = False
                return None
            merge_stats = self._kg_main.merge_scan(self.scan_id, self._kg)
            logger.info(
                "kg_fork_merged | scan=%s | stats=%s",
                self.scan_id, merge_stats,
            )
            self._kg_fork_active = False
            return merge_stats
        except Exception as exc:
            logger.warning(
                "kg_merge_failed | scan=%s | err=%s",
                self.scan_id, exc,
            )
            self._kg_fork_active = False
            return None

    def set_tech_stack(self, tech_stack: list[str]) -> None:
        """Record the detected tech stack for this scan.

        Called by the agent loop after parsing the first recon tool
        output. Subsequent kg_consult() calls use this stack unless
        overridden.
        """
        self._tech_stack = list(tech_stack or [])
        logger.info(
            "kg_tech_stack_set | scan=%s | stack=%s",
            self.scan_id, self._tech_stack,
        )

    def recon_signals(self) -> dict[str, Any]:
        """Tech-stack signals for `build_session(recon_signals=...)`.

        Plain data, like `egats_snapshot()`: a fresh dict holding a copy of
        the detected stack, so a session can never alias `_tech_stack`.

        This accessor exists because there was none. `set_tech_stack()` was
        called on the first recon tool output and `kg_consult()` used the
        stack internally, but nothing ever handed it to the state builder —
        so `recon_signals` was `{}` on every turn of every scan and the
        256-dim technology multi-hot block of the 337-dim vector was
        permanently zero. Silence, not a crash; nothing failed.

        Returns `{}` (not `{"technologies": []}`) while the stack is still
        unknown, matching the builder's documented default.
        """
        if not self._tech_stack:
            return {}
        return {"technologies": list(self._tech_stack)}

    def kg_consult(
        self,
        tech_stack: list[str] | None = None,
        focus: str | None = None,
        top_k: int = KG_TOP_K_DEFAULT,
    ) -> tuple[str, str]:
        """Consult the KG and return (consult_id, advisory_markdown).

        Args:
            tech_stack: override the stored tech stack for this consult.
                None = use the stack set via set_tech_stack().
            focus: optional attack-vector focus (e.g. "sqli") to narrow
                the consult. Used when the LLM calls consult_kg tool.
            top_k: max paths to include in the brief.

        Returns:
            (consult_id, markdown_text)
            - consult_id: "" when KG is unavailable or no paths found.
            - markdown_text: "" when KG is unavailable or no paths found.

        Side effects:
            - Stores consult_id in self._last_consult_id for outcome
              pairing (kg_record_finding_outcome reads it).
            - Increments self._kg_consult_count.
        """
        if self._kg is None:
            return "", ""

        stack = tech_stack if tech_stack is not None else self._tech_stack
        if not stack:
            logger.debug(
                "kg_consult_skipped | scan=%s | reason=no_tech_stack",
                self.scan_id,
            )
            return "", ""

        try:
            from app.kg.advisor import consult_text
            consult_id, markdown = consult_text(
                self._kg, stack, focus=focus, top_k=top_k,
            )
        except Exception as exc:
            logger.warning(
                "kg_consult_failed | scan=%s | err=%s",
                self.scan_id, exc,
            )
            return "", ""

        if consult_id:
            self._last_consult_id = consult_id
            self._kg_consult_count += 1
            logger.info(
                "kg_consult | scan=%s | consult_id=%s | paths_brief_len=%d | "
                "focus=%s",
                self.scan_id, consult_id, len(markdown), focus or "(none)",
            )
        return consult_id, markdown

    def kg_record_finding_outcome(
        self, finding_id: str, success: bool, reason: str = "",
    ) -> None:
        """Record whether the last KG consult led to a confirmed finding.

        Called by the agent loop after record_vulnerability returns:
            - success=True (status="recorded") → record_consult_hit
            - success=False (status="rejected") → record_consult_miss

        If there is no active consult_id, this is a no-op (the finding
        was reported without a KG consult, e.g. on turn 1 before any
        recon ran).
        """
        if self._kg is None or not self._last_consult_id:
            return
        try:
            if success:
                self._kg.record_consult_hit(
                    self._last_consult_id, finding_id or "",
                )
                self._kg_hit_count += 1
                logger.info(
                    "kg_hit | scan=%s | consult_id=%s | finding_id=%s",
                    self.scan_id, self._last_consult_id, finding_id,
                )
            else:
                self._kg.record_consult_miss(
                    self._last_consult_id, reason or "rejected",
                )
                self._kg_miss_count += 1
                logger.info(
                    "kg_miss | scan=%s | consult_id=%s | reason=%s",
                    self.scan_id, self._last_consult_id, reason,
                )
        except Exception as exc:
            logger.warning(
                "kg_record_outcome_failed | scan=%s | err=%s",
                self.scan_id, exc,
            )

    def kg_get_stats(self) -> dict[str, Any]:
        """Snapshot of KG activity for the rl_stats dict."""
        return {
            "kg_attached": self._kg is not None,
            "kg_enabled": self._kg_enabled,
            "tech_stack": list(self._tech_stack),
            "consult_count": self._kg_consult_count,
            "hit_count": self._kg_hit_count,
            "miss_count": self._kg_miss_count,
            "last_consult_id": self._last_consult_id,
            # Phase 6: fork status
            "isolation_enabled": self._kg_isolation_enabled,
            "fork_active": self._kg_fork_active,
        }

    @property
    def kg(self) -> Any:
        """Public read access to the attached KnowledgeGraph (or None).

        The graph is either the main singleton or, when
        VAPT_AI_KG_ISOLATION is on, this scan's fork. Read-only by
        convention: callers that want to WRITE must go through the
        kg_* methods so the fork/merge bookkeeping stays correct.

        Added because react_agent.py was reaching into `harness._kg`
        from outside the class — a private attribute access that broke
        silently if the internal name ever changed.
        """
        return self._kg

    # ──────────────────────────────────────────────────────────────────
    # Public: TDA integration (Phase 3)
    # ──────────────────────────────────────────────────────────────────

    def tda_compute(
        self,
        *,
        turn: int,
        max_turns: int,
        findings: list[dict[str, Any]],
        successes: int,
        attempts: int,
        prompt_tokens: int = 0,
        model_context_window: int = MODEL_CONTEXT_WINDOW_DEFAULT,
        tokens_used: int = 0,
        path_evidence: list[float] | None = None,
        advisory_tokens: int = 0,
    ) -> Any:
        """Compute TDI for this turn + apply the evidence-gated override.

        Args:
            turn: 1-indexed current turn number.
            max_turns: scan's max_iterations. This is also the H
                normalisation budget unless VAPT_AI_TDA_MAX_HORIZON
                explicitly sets a larger value.
            findings: list of finding dicts (used for the override
                decision + evidence extraction when path_evidence is None).
            successes: count of verified, non-false-positive findings.
            attempts: count of tool calls executed.
            prompt_tokens: prompt tokens the provider reported for the
                MOST RECENT completion — the C numerator. This measures
                how full the model's context window is right now, which
                is what actually forces pruning and degrades reasoning.
                Do NOT pass the cumulative scan total here: that is a
                spend measure and made C (and therefore TDI) rise with
                scan length, forcing BROADEN exactly when the scan was
                already the most expensive it had ever been.
            model_context_window: the model's real context window
                (128K default). Dividing by the 10M scan budget instead
                compressed C to ~0.002 and effectively disabled it.
            tokens_used: cumulative LLM tokens so far (billing counter,
                logged only — not a TDI input any more).
            path_evidence: optional explicit list of evidence scores.
                When None, derives from `findings` via
                evidence_score_from_finding().
            advisory_tokens: advisory tokens injected by the harness's
                OWN advisory messages (RL/TDA/EGATS/KG/SC hints) into the
                SAME request that produced `prompt_tokens` — i.e. the
                PREVIOUS turn's per-turn advisory load. These are
                subtracted from `prompt_tokens` before computing C, so the
                TDA hint cannot inflate its own difficulty signal and
                re-trigger BROADEN (self-reinforcing loop). This is a
                best-effort estimate (chars/4) — see the agent loop.
                NEVER pass a cumulative scan total here: it grows every
                turn while one request's prompt tokens stay bounded by
                history pruning, so C would read 0 for the rest of the
                scan.

        Returns:
            TDIResult (with .overridden=True when the override fired).
            None when TDA is disabled. On override, the raw formula
            dimensions are preserved and only `mode`/`overridden`/
            `override_evidence` change.

        Side effects:
            - Appends a dict to self._tdi_log (consumed by StateEncoder
              via build_session's tdi_log param).
            - Increments self._tda_override_count when override fires.
        """
        if not self._tda_enabled:
            return None

        from app.rl.tda import (
            compute_tdi,
            context_saturation,
            evidence_score_from_finding,
            max_evidence_score,
            resolve_max_horizon,
            should_force_exploit,
            Mode,
            TDIResult,
        )

        # Derive path_evidence from findings when not explicitly passed.
        if path_evidence is None:
            path_evidence = [evidence_score_from_finding(f) for f in findings]

        # H must be normalised by the scan's REAL budget. The legacy
        # fixed default of 20 clamped H to 1.0 for 80% of a 100-turn
        # scan, removing the horizon dimension from the TDI signal.
        max_horizon = resolve_max_horizon(max_turns, self._tda_max_horizon)
        remaining_steps = max(0, max_turns - turn)

        # C must measure the agent's real consumption, not the harness's
        # own advisory overhead — otherwise a hint that fires because
        # difficulty is high also pushes difficulty higher.
        content_tokens = max(0, prompt_tokens - max(0, advisory_tokens))
        utilisation = context_saturation(content_tokens, model_context_window)

        result = compute_tdi(
            remaining_steps=remaining_steps,
            path_evidence=path_evidence,
            context_utilisation=utilisation,
            successes=successes,
            attempts=attempts,
            max_horizon=max_horizon,
        )

        # ── Critical override: CONFIRMED finding -> force EXPLOIT ───
        # This is the ONLY path to Mode.EXPLOIT. Gated on EVIDENCE, not
        # on findings_count: a single plausible/hallucinated finding
        # (0.5) must not tunnel the scan onto a vector that may not
        # exist. Only a verifier-accepted (0.8) or reproduced (1.0)
        # finding may lock the direction.
        # The raw TDI/h/e/c/s are preserved so the RL state encoder
        # keeps a real difficulty signal instead of a constant 0.20.
        evidence = max_evidence_score(findings)
        if (
            TDA_FORCE_EXPLOIT_ON_FINDING
            and result.mode != Mode.EXPLOIT
            and should_force_exploit(findings)
        ):
            logger.info(
                "tda_override | scan=%s | turn=%d | raw_mode=%s | "
                "raw_tdi=%.3f | findings=%d | evidence=%.2f -> forced EXPLOIT",
                self.scan_id, turn, result.mode.value, result.tdi,
                len(findings), evidence,
            )
            result = TDIResult(
                tdi=result.tdi,   # raw formula value — never fabricated
                h=result.h,
                e=result.e,
                c=result.c,
                s=result.s,
                mode=Mode.EXPLOIT,
                overridden=True,
                override_evidence=evidence,
            )
            self._tda_override_count += 1

        # Push to log (consumed by StateEncoder + agent loop).
        self._tdi_log.append({
            "turn": turn,
            "tdi": result.tdi,
            "mode": result.mode.value,
            "h": result.h,
            "e": result.e,
            "c": result.c,
            "s": result.s,
            "overridden": result.overridden,
            "override_evidence": result.override_evidence,
        })

        logger.info(
            "tda_compute | scan=%s | turn=%d | tdi=%.3f | mode=%s | "
            "h=%.2f e=%.2f c=%.2f s=%.2f | prompt=%d content=%d window=%d | "
            "spent=%d | overridden=%s",
            self.scan_id, turn, result.tdi, result.mode.value,
            result.h, result.e, result.c, result.s,
            prompt_tokens, content_tokens, model_context_window,
            tokens_used, result.overridden,
        )
        return result

    @staticmethod
    def tda_render_hint(tdi_result: Any) -> str:
        """Render the TDI result as an advisory system message.

        Delegates to app.rl.tda.render_tda_hint — kept as a method so
        the agent loop calls harness.tda_render_hint(rec) without
        importing the tda module directly.
        """
        from app.rl.tda import render_tda_hint
        return render_tda_hint(tdi_result)

    def tda_get_stats(self) -> dict[str, Any]:
        """Snapshot of TDA activity for the rl_stats.tda sub-dict."""
        current = self._tdi_log[-1] if self._tdi_log else None
        return {
            "tda_enabled": self._tda_enabled,
            "current_mode": current["mode"] if current else None,
            "current_tdi": current["tdi"] if current else None,
            "current_overridden": current["overridden"] if current else False,
            "override_count": self._tda_override_count,
            "mode_history": [e["mode"] for e in self._tdi_log[-20:]],
            "tdi_history": [round(e["tdi"], 4) for e in self._tdi_log[-20:]],
            "turns_computed": len(self._tdi_log),
        }

    @property
    def tdi_log(self) -> list[dict[str, Any]]:
        """Read-only access to the per-turn TDI history."""
        return list(self._tdi_log)

    # ──────────────────────────────────────────────────────────────────
    # Public: EGATS integration (Phase 4)
    # ──────────────────────────────────────────────────────────────────

    def egats_seed(
        self,
        kg: Any,
        source_node_ids: list[str],
        *,
        max_depth: int = 4,
        top_k: int = 10,
    ) -> int:
        """Seed EGATS with attack paths from the KG.

        Called by the agent loop after the first KG consult (Phase 2)
        produces a tech stack. Instantiates the EGATS planner if needed,
        then pulls attack paths from the KG.

        Args:
            kg: KnowledgeGraph instance (use self._kg).
            source_node_ids: KG Technology node IDs for the detected stack.
            max_depth: max path length (default 4).
            top_k: max paths to seed (default 10).

        Returns:
            Number of NEW paths seeded. 0 when EGATS is disabled or KG
            is unavailable.
        """
        if not self._egats_enabled:
            return 0
        if kg is None or not source_node_ids:
            return 0

        # Lazy-create the EGATS planner.
        if self._egats is None:
            from app.rl.egats import EGATS
            self._egats = EGATS()

        seeded = self._egats.seed_from_kg(
            kg, source_node_ids, max_depth=max_depth, top_k=top_k,
        )
        if seeded > 0:
            self._egats_seeded = True
            logger.info(
                "egats_seed | scan=%s | new_paths=%d | total_paths=%d",
                self.scan_id, seeded, len(self._egats.stats()),
            )
        return seeded

    def egats_select(self) -> Any:
        """Select the highest-UCB alive path for THIS turn.

        Returns EGATSResult (with path=None when no alive paths or
        EGATS not seeded). Stashes the selected path's key in
        _egats_selected_path_key for display/metrics and CLEARS
        _egats_injected_path_key, marking the start of a new turn: an
        outcome may only be credited to a hint injected in the SAME turn.

        Returns None when EGATS is disabled or not seeded.
        """
        # New turn boundary — a hint injected on a previous turn must not
        # be credited with this turn's outcome.
        self._egats_injected_path_key = None

        if not self._egats_enabled or self._egats is None or not self._egats_seeded:
            return None

        result = self._egats.select()
        if result.path is not None:
            self._egats_selected_path_key = result.path.key()
            logger.info(
                "egats_select | scan=%s | path=%s | ucb=%.3f | "
                "attempts=%d | successes=%d | alive=%d | pruned=%d",
                self.scan_id,
                "→".join(result.path.nodes),
                result.ucb_score,
                result.path.attempts,
                result.path.successes,
                result.alive_paths,
                result.pruned_paths,
            )
        else:
            self._egats_selected_path_key = None
            logger.debug(
                "egats_select_no_alive | scan=%s | total=%d | pruned=%d",
                self.scan_id, result.total_paths, result.pruned_paths,
            )
        return result

    def egats_mark_hint_injected(self) -> bool:
        """Mark the selected path as ACTUALLY shown to the LLM this turn.

        Called by the agent loop at the injection site, i.e. only after
        `_select_advisory` picked the EGATS message out of the round-robin.
        `egats_select()` runs every turn but the rotation injects EGATS on
        roughly one turn in three, so selecting alone says nothing about
        what the model saw.

        Returns True when a path was armed for outcome pairing.
        """
        if self._egats_selected_path_key is None:
            return False
        self._egats_injected_path_key = self._egats_selected_path_key
        logger.info(
            "egats_hint_injected | scan=%s | path=%s",
            self.scan_id,
            "→".join(self._egats_injected_path_key),
        )
        return True

    @staticmethod
    def egats_render_hint(result: Any) -> str:
        """Render the EGATS result as an advisory system message.

        Delegates to app.rl.egats.render_egats_hint. Returns empty
        string when result.path is None.
        """
        from app.rl.egats import render_egats_hint
        return render_egats_hint(result)

    def egats_record_outcome(self, success: bool, tdi: float) -> None:
        """Backprop an outcome onto the last INJECTED path + maybe prune.

        Called by the agent loop after record_vulnerability:
            - success=True (status="recorded") → path led to a finding
            - success=False (status="rejected") → path was a dead end

        Only the path whose hint was actually injected this turn is
        credited (see egats_mark_hint_injected). When no hint was
        injected the outcome is dropped — a path the LLM never saw must
        not collect attempts/successes, otherwise the UCB scores and the
        pruning decisions are driven by round-robin bookkeeping rather
        than by evidence.

        The armed key is CONSUMED: one injection yields at most one
        backprop, so a turn with several record_vulnerability calls
        counts as a single attempt.

        Uses the current TDI (from tdi_log) to update the path's TDI EMA.
        Calls _maybe_prune() — may set pruned=True if the path has been
        attempted >= k_min_prune times with tdi_ema > TDI_BROADEN.

        No-op when EGATS is disabled, not seeded, or no hint was
        injected this turn.
        """
        if (
            not self._egats_enabled
            or self._egats is None
            or self._egats_injected_path_key is None
        ):
            logger.debug(
                "egats_outcome_skipped | scan=%s | success=%s | "
                "reason=hint_not_injected",
                self.scan_id, success,
            )
            return

        path_key = self._egats_injected_path_key
        self._egats_injected_path_key = None
        try:
            self._egats.record_outcome(
                path_key, success=success, tdi=float(tdi),
            )
        except Exception as exc:
            logger.warning(
                "egats_record_failed | scan=%s | err=%s",
                self.scan_id, exc,
            )

    def egats_get_stats(self) -> dict[str, Any]:
        """Snapshot of EGATS activity for the rl_stats.egats sub-dict."""
        if not self._egats_enabled:
            return {"enabled": False}
        if self._egats is None:
            return {
                "enabled": True,
                "seeded": False,
                "note": "EGATS not seeded (no KG paths available)",
            }
        summary = self._egats.get_stats_summary()
        summary["enabled"] = True
        summary["seeded"] = self._egats_seeded
        return summary

    def egats_snapshot(self) -> dict[str, Any]:
        """Plain-data planner snapshot for `build_session(egats_planner=...)`.

        The session dict is plain data that every turn is rebuilt and that
        must survive `json.dumps`, so the agent loop must NOT pass the live
        `EGATS` object — it passes this dict instead. Returns `{}` when
        EGATS is disabled or not yet seeded; the encoder then reads
        0 live / 0 pruned, which is the truthful value (no paths exist).
        """
        if not self._egats_enabled or self._egats is None:
            return {}
        return self._egats.to_dict()

    # ──────────────────────────────────────────────────────────────────
    # Public: Self-Correction integration
    # ──────────────────────────────────────────────────────────────────

    async def process_rejection(
        self,
        finding: dict[str, Any],
        reason: str,
    ) -> dict[str, Any] | None:
        """Process a rejected finding — generate corrective feedback + RL reward.

        Called by the agent loop when record_vulnerability returns
        status="rejected". The harness:
            1. Lazy-creates a SelfCorrectionEngine.
            2. Calls engine.process_rejection(finding, reason) which:
               - Computes a pattern signature from (vuln_type, endpoint)
               - Increments the in-memory pattern count
               - Derives root_cause + corrective_action + suggested_verification
               - Builds a 3-tier corrective message (tier 1/2/3)
               - Computes an escalating RL reward (-5/-7/-10)
            3. Pushes the rejection as a high-priority experience into
               the PER store (priority = |reward| * 2) so the policy
               learns to avoid the rejected pattern.
            4. Stashes the corrective message for next-turn LLM injection
               (the agent loop calls sc_consume_corrective_message() at
               the start of the next turn).

        Args:
            finding: the rejected finding dict (with vuln_type, location,
                title, severity, etc.).
            reason: rejection reason from record_vulnerability output.

        Returns:
            dict from SelfCorrectionEngine.process_rejection(), or None
            when Self-Correction is disabled.
        """
        if not self._sc_enabled:
            return None

        # Lazy-create the engine.
        if self._sc is None:
            from app.rl.self_correction import SelfCorrectionEngine
            self._sc = SelfCorrectionEngine()

        result = self._sc.process_rejection(
            finding=finding, reason=reason, session_id=self.scan_id,
        )

        # Push the rejection as a high-priority RL experience so the
        # policy learns to avoid the rejected pattern. We use the
        # prev_state_vec stashed by select_action() + the action_idx
        # from the last select_action call.
        rl_reward = result.get("rl_reward", -5.0)
        per_priority = result.get("per_priority", abs(rl_reward) * 2.0)
        if self._prev_state_vec is not None:
            try:
                # Build a "current state" for the next_state half of the
                # transition. We reuse the prev_state since the rejection
                # didn't change the scan state (no new tool ran).
                next_state_vec = self._prev_state_vec
                # The rejection penalises the action that led to this state,
                # so label the transition with the EXECUTED action index and
                # derive the name from it — a hardcoded "report_vulnerability"
                # next to an arbitrary index is the same off-policy label
                # mismatch the agent loop had (see record_experience).
                fallback_idx = ACTION_SPACE.index("report_vulnerability")
                action_idx = (
                    self._prev_action_idx
                    if self._prev_action_idx is not None
                    else fallback_idx
                )
                action_name = (
                    ACTION_SPACE[action_idx]
                    if 0 <= action_idx < len(ACTION_SPACE)
                    else "report_vulnerability"
                )
                # AWAIT directly — this coroutine is already on the event
                # loop, so `get_event_loop()` + `create_task()` bought
                # nothing and lost two things: the store write could be
                # dropped when no loop was running (`if loop.is_running()`
                # with no else), and the task was never retained, so it
                # could be garbage-collected mid-flight or silently die
                # when end_scan() cancelled the worker. Awaiting also means
                # the caller can rely on the transition being in the store
                # before the rejection is reported back to the LLM.
                await self._store.add(
                    scan_id=self.scan_id,
                    turn=self._turn_count,
                    state=self._prev_state_vec,
                    action=action_idx,
                    reward=rl_reward,
                    next_state=next_state_vec,
                    done=False,
                    priority=per_priority,
                    action_name=action_name,
                    reward_breakdown={
                        "type": "rejection",
                        "pattern_signature": result.get("pattern_signature"),
                        "pattern_count": result.get("pattern_count"),
                        "tier": result.get("tier"),
                    },
                )
                logger.info(
                    "sc_rl_experience | scan=%s | reward=%.1f | priority=%.1f | "
                    "sig=%s | count=%d",
                    self.scan_id, rl_reward, per_priority,
                    result.get("pattern_signature"), result.get("pattern_count"),
                )
            except Exception as exc:
                logger.warning(
                    "sc_rl_experience_failed | scan=%s | err=%s",
                    self.scan_id, exc,
                )

        return result

    def sc_consume_corrective_message(self) -> str | None:
        """Pop the pending corrective message (if any).

        Called by the agent loop at the start of each turn. Returns the
        corrective message from the last rejection and clears it, so it
        is injected exactly once (on the turn immediately after the
        rejection). Returns None when there is no pending message.
        """
        if not self._sc_enabled or self._sc is None:
            return None
        return self._sc.get_pending_corrective_message()

    def sc_get_stats(self) -> dict[str, Any]:
        """Snapshot of Self-Correction activity for the rl_stats.sc sub-dict."""
        if not self._sc_enabled:
            return {"enabled": False}
        if self._sc is None:
            return {
                "enabled": True,
                "total_rejections": 0,
                "unique_patterns": 0,
                "blacklisted_patterns": 0,
                "pending_message": False,
            }
        return self._sc.get_stats()

    # ──────────────────────────────────────────────────────────────────
    # Public: scan lifecycle
    # ──────────────────────────────────────────────────────────────────

    async def end_scan(self, episode_reward: float) -> None:
        """Cancel train worker, decay epsilon, persist checkpoint.

        Idempotent — safe to call multiple times (subsequent calls no-op).
        """
        if self._stop_event.is_set():
            return

        # 1. Signal worker to stop and wait for graceful exit.
        self._stop_event.set()
        async with self._train_cv:
            self._train_cv.notify_all()
        try:
            await asyncio.wait_for(self._train_worker_task, timeout=5.0)
        except asyncio.TimeoutError:
            self._train_worker_task.cancel()
            try:
                await self._train_worker_task
            except asyncio.CancelledError:
                pass
            logger.warning(
                "rl_train_worker_force_cancel | scan=%s", self.scan_id,
            )

        # 2. Decay epsilon + record episode reward (drives adaptive decay).
        self._policy.end_episode(episode_reward)

        # 3. Persist checkpoint.
        await self._save_checkpoint_safe(episode_reward)

        logger.info(
            "rl_end | scan=%s | episode_reward=%.4f | epsilon_end=%.4f | "
            "turns=%d | followed=%d | exploratory=%d | buffer=%d",
            self.scan_id, episode_reward, self._policy.epsilon,
            self._turn_count, self._followed_count,
            self._exploratory_count, len(self._store),
        )

    def get_stats(self) -> dict[str, Any]:
        """Snapshot for SSE event + scan_record.rl_stats JSON column."""
        policy_stats = self._policy.get_stats()
        return {
            "scan_id": self.scan_id,
            "turn_count": self._turn_count,
            "episode_reward": round(self._episode_reward, 4),
            "epsilon": round(self._policy.epsilon, 4),
            "temperature": round(self._policy.temperature, 4),
            # `episodes` = completed scans the policy has learned from, a
            # cumulative counter carried across processes by policy_state.json
            # (see _restore_policy_state). Distinct from `train_steps` below,
            # which counts *batches in this process* — a gate must not treat
            # the two as interchangeable.
            "episodes": int(policy_stats.get("episodes", 0)),
            "followed_count": self._followed_count,
            "follow_rate": round(
                self._followed_count / max(self._turn_count, 1), 4
            ),
            "exploratory_count": self._exploratory_count,
            "buffer_size": len(self._store),
            "train_steps": len(self._train_log),
            # Cumulative gradient steps, restored from the .npz cache — NOT
            # `len(_train_log)`. Exposed so callers don't have to reach
            # through `harness.q_train_steps` (and confuse the two).
            "q_train_steps": int(self._policy.q.train_steps),
            "train_log": list(self._train_log[-10:]),
            "checkpoint_loaded": getattr(self, "_ckpt_loaded", False),
            "checkpoint_saved": getattr(self, "_ckpt_saved", False),
            # Phase 2: KG activity
            "kg": self.kg_get_stats(),
            # Phase 3: TDA activity
            "tda": self.tda_get_stats(),
            # Phase 4: EGATS activity
            "egats": self.egats_get_stats(),
            # Phase 5: Self-Correction activity
            "sc": self.sc_get_stats(),
        }

    @property
    def epsilon(self) -> float:
        return self._policy.epsilon

    @property
    def buffer_size(self) -> int:
        return len(self._store)

    @property
    def q_train_steps(self) -> int:
        """How many gradient steps the Q-network has performed.

        0 = still random-initialised. The agent loop uses this to keep RL
        guidance out of the prompt until the policy has learned something:
        with no training every Q-value is noise, so an "exploitation" pick
        is no more meaningful than a random one. Mirrors the learner's own
        counter, which is persisted in the .npz cache meta.
        """
        return int(self._policy.q.train_steps)

    # ──────────────────────────────────────────────────────────────────
    # Private: background training worker
    # ──────────────────────────────────────────────────────────────────

    async def _train_worker(self) -> None:
        """Drain training queue every TRAIN_INTERVAL_TURNS turns.

        Acquires _q_lock for ~50 ms per batch (numpy backprop). Cancels
        on end_scan().
        """
        try:
            while not self._stop_event.is_set():
                async with self._train_cv:
                    try:
                        await asyncio.wait_for(
                            self._train_cv.wait_for(
                                lambda: self._pending_trains > 0
                                or self._stop_event.is_set()
                            ),
                            timeout=TRAIN_GRACE_PERIOD_S,
                        )
                    except asyncio.TimeoutError:
                        continue

                if self._stop_event.is_set() and self._pending_trains == 0:
                    break

                if len(self._store) < MIN_BUFFER_FOR_TRAIN:
                    self._pending_trains = max(0, self._pending_trains - 1)
                    continue

                stats = await self._train_one_batch()
                if stats is not None:
                    self._train_log.append(stats)
                    self._pending_trains = max(0, self._pending_trains - 1)

        except asyncio.CancelledError:
            logger.debug(
                "rl_train_worker_cancelled | scan=%s", self.scan_id,
            )
            raise

    async def _train_one_batch(self) -> dict[str, Any] | None:
        """Run one train_from_memory under _q_lock. Returns stats dict."""
        try:
            async with self._q_lock:
                loss, td_errors = self._policy.train_from_memory(
                    self._store, batch_size=TRAIN_BATCH_SIZE
                )
        except Exception as exc:
            logger.warning(
                "rl_train_failed | scan=%s | err=%s", self.scan_id, exc,
            )
            return None

        stats = {
            "loss": round(float(loss), 6),
            "td_err_max": round(float(max(td_errors) if td_errors else 0.0), 4),
            "buffer_size": len(self._store),
            "epsilon": round(self._policy.epsilon, 4),
        }
        logger.info(
            "rl_train | scan=%s | loss=%.6f | td_err_max=%.4f | "
            "buffer=%d | epsilon=%.4f",
            self.scan_id, stats["loss"], stats["td_err_max"],
            stats["buffer_size"], stats["epsilon"],
        )
        return stats

    # ──────────────────────────────────────────────────────────────────
    # Private: checkpoint load / save
    # ──────────────────────────────────────────────────────────────────

    def _policy_state_path(self) -> Path:
        """Where ε / T / episode counter are persisted between scans.

        Derived from `q_learner.WEIGHTS_FILE` (not a module constant of our
        own) so that any test or deployment that redirects the Q cache also
        redirects this file — the two must never disagree about which scan
        history they belong to.
        """
        from app.rl import q_learner as _qmod

        return Path(_qmod.WEIGHTS_FILE).parent / "policy_state.json"

    def _save_policy_state(self) -> None:
        """Persist the exploration schedule. Best-effort, never raises."""
        path = self._policy_state_path()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = self._policy.state_dict()
            payload["scan_id"] = self.scan_id
            tmp = path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(payload), encoding="utf-8")
            # Atomic replace: a crash mid-write must not leave a truncated
            # file that the next scan would fail to parse (and then silently
            # fall back to ε_start, i.e. exactly the bug this file fixes).
            tmp.replace(path)
            logger.debug(
                "rl_policy_state_saved | scan=%s | epsilon=%.4f | episodes=%d",
                self.scan_id, payload["epsilon"], payload["episode"],
            )
        except Exception as exc:
            logger.warning(
                "rl_policy_state_save_failed | scan=%s | path=%s | err=%s",
                self.scan_id, path, exc,
            )

    def _restore_policy_state(self) -> bool:
        """Restore ε / T / episode counter from disk. True if applied."""
        path = self._policy_state_path()
        try:
            if not path.exists():
                return False
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning(
                "rl_policy_state_load_failed | scan=%s | path=%s | err=%s "
                "(falling back to policy defaults)",
                self.scan_id, path, exc,
            )
            return False
        applied = self._policy.load_state_dict(payload)
        if not applied:
            logger.warning(
                "rl_policy_state_ignored | scan=%s | path=%s "
                "(payload rejected — policy defaults kept)",
                self.scan_id, path,
            )
        return applied

    def _load_checkpoint_safe(self) -> None:
        """Load latest checkpoint. Best-effort — fresh policy on failure.

        We cannot await here (called from __init__ which is sync), so we
        schedule the async load and let the first select_action block on
        _q_lock until load finishes. In practice the load completes in
        <100 ms from .npz cache, well before turn 1's LLM round-trip.
        """
        self._ckpt_loaded = False
        try:
            # Fast .npz cache load (sync). The learner already tries this in
            # __init__; calling it again is a no-op re-read that keeps this
            # method honest about what is on disk.
            loaded = self._policy.q.load_cache()
            self._ckpt_loaded = bool(loaded)
            # Restore the exploration schedule. `q.load_cache()` carries
            # weights + train_steps ONLY — ε / T / episode counter live in
            # the Q-learner's world not the policy's, so they need their
            # own sidecar. Skipping this (the pre-R27 behaviour) reset ε to
            # 1.0 on every scan, which pinned the gate in
            # `_should_inject_rl_guidance` to `exploratory` for 100% of
            # turns of every scan — the RL advisory channel was dead code
            # in production. See HarnessBridge._policy_state_path.
            restored_eps = self._restore_policy_state()
            logger.info(
                "rl_checkpoint_loaded | scan=%s | weights=%s | train_steps=%d "
                "| epsilon=%.4f (%s) | episodes=%d",
                self.scan_id, "restored" if loaded else "none",
                self._policy.q.train_steps, self._policy.epsilon,
                "restored" if restored_eps else "default",
                self._policy.get_stats().get("episodes", 0),
            )
        except Exception as exc:
            logger.debug(
                "rl_checkpoint_load_skipped | scan=%s | err=%s",
                self.scan_id, exc,
            )

    async def _save_checkpoint_safe(self, episode_reward: float) -> None:
        """Persist checkpoint to .npz cache + SQL table.

        Called from end_scan() — awaits the async save_checkpoint from
        app.rl.persistence instead of fire-and-forgetting it. The old
        version scheduled `_save_sql()` with `loop.create_task()` and kept
        no reference: `end_scan()` then cancelled the worker and returned,
        so the task could be collected (or die at loop shutdown) and the
        SQL row was silently missing. `except RuntimeError: pass` also
        swallowed the no-loop case with no log at all.
        """
        self._ckpt_saved = False
        try:
            self._policy.q.save_cache()
            self._ckpt_saved = True
        except Exception as exc:
            logger.warning(
                "rl_checkpoint_save_failed | scan=%s | err=%s",
                self.scan_id, exc,
            )

        # `save_cache()` writes the Q side only. Persist the exploration
        # schedule next to it or the next scan starts the ε decay over
        # from 1.0 (see the R27 note in policy.PentestPolicy.state_dict).
        self._save_policy_state()

        # The .npz cache above is the source of truth for the next scan;
        # the SQL row is best-effort but must be awaited or logged.
        try:
            from app.rl.persistence import save_checkpoint
            await save_checkpoint(
                q_learner=self._policy.q,
                policy=self._policy,
                scan_id=self.scan_id,
                total_reward=episode_reward,
            )
            logger.info(
                "rl_checkpoint_saved_sql | scan=%s | epsilon=%.4f",
                self.scan_id, self._policy.epsilon,
            )
        except Exception as exc:
            logger.warning(
                "rl_checkpoint_sql_save_failed | scan=%s | err=%s",
                self.scan_id, exc,
            )


# ─── per-scan registry ────────────────────────────────────────────────
# get_harness / release_harness let the agent loop grab the bridge by
# scan_id without threading it through every function call. One instance
# per scan, enforced by this dict.
_harnesses: dict[str, HarnessBridge] = {}


def get_harness(scan_id: str, target: str) -> HarnessBridge:
    """Get or create the per-scan HarnessBridge."""
    if scan_id not in _harnesses:
        _harnesses[scan_id] = HarnessBridge(scan_id=scan_id, target=target)
    return _harnesses[scan_id]


def release_harness(scan_id: str) -> None:
    """Drop the per-scan HarnessBridge from the registry.

    Called by the agent loop after end_scan(). The instance itself may
    still be referenced by an in-flight training task — that's fine, the
    task holds its own reference and will GC naturally.
    """
    _harnesses.pop(scan_id, None)


def is_rl_harness_enabled() -> bool:
    """Read the VAPT_AI_RL_HARNESS_ENABLED env flag."""
    return os.getenv("VAPT_AI_RL_HARNESS_ENABLED", "true").lower() in (
        "1", "true", "yes",
    )


__all__ = [
    "HarnessBridge",
    "TurnRecommendation",
    "get_harness",
    "release_harness",
    "is_rl_harness_enabled",
    "TRAIN_INTERVAL_TURNS",
    "TRAIN_BATCH_SIZE",
    "MIN_BUFFER_FOR_TRAIN",
    # Phase 2: KG integration
    "KG_ENABLED_DEFAULT",
    "KG_TOP_K_DEFAULT",
    # Phase 3: TDA integration
    "TDA_ENABLED_DEFAULT",
    "TDA_FORCE_EXPLOIT_ON_FINDING",
    "TDA_MAX_HORIZON_DEFAULT",
    "MODEL_CONTEXT_WINDOW_DEFAULT",
    # Phase 4: EGATS integration
    "EGATS_ENABLED_DEFAULT",
    # Phase 5: Self-Correction integration
    "SC_ENABLED_DEFAULT",
    # Phase 6: Per-scan KG Fork
    "KG_ISOLATION_DEFAULT",
]