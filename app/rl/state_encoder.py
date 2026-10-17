"""
POMDP State Encoder — converts a live pentest session into a numeric feature vector.

Python port of EVVO Sentinel shield_engine/rl/state_encoder.py.

The session dict format is produced by the Harness Bridge (W18 will implement
app/harness/bridge.py). For W16, the encoder accepts the same dict shape as
EVVO so the existing RL math works unchanged.

State vector layout (337 dims, float32):
    Section 1: 8 scalars (normalized 0..1)
        - turn / max_turns
        - tdi (Task Difficulty Index)
        - mode (broaden=1.0, llm=0.5, exploit=0.0)
        - findings_count / max_findings
        - verified_findings / max_verified_findings
        - fp_findings / max_fp_findings
        - commands_run / max_commands
        - token_budget_remaining

    Section 2: 256-dim tech stack multi-hot (hash-bucketed — the stack is an
        OPEN set: `technologies` is whatever the recon parser found, so a
        vocabulary cannot be enumerated in advance)
    Section 3: 32-dim last-vector one-hot (DIRECT index into VECTOR_LABELS)
    Section 4: 32-dim last-tool one-hot (DIRECT index into TOOL_LABELS)
    Section 5: 9-dim flags = 7 booleans + 2 planner ratios
        - last_success                 (bool)
        - kg_available                 (bool)
        - planner_paths_live / max     (ratio)
        - planner_paths_pruned / max   (ratio)
        - recon_login, recon_api, recon_upload, recon_graphql, recon_websocket

Sections 3 and 4 are CLOSED sets derived from `ACTION_SPACE`, so they are
indexed directly rather than hashed — see `_label_index()` for why hashing a
closed set was a defect, not a style choice.

Pure numpy, no torch dependency.
"""
from __future__ import annotations

import hashlib
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

# Action space lives in its own dependency-free module so `q_learner` can
# import it without importing the encoder (see app/rl/actions.py for why the
# old "circular import" justification was fictional). Re-exported here because
# this has been the de-facto public path for `ACTION_SPACE` for several rounds
# of callers (`react_agent`, `policy`, `harness`, tests).
from app.rl.actions import ACTION_DIM, ACTION_SPACE, tool_category  # noqa: F401
from app.evidence.finding_state import count_verified_findings

logger = logging.getLogger(__name__)


# ── Feature dimensions ───────────────────────────────────────────────────
# EVVO audit P0-6: previously STATE_DIM = 128 while the encoder produced
# 8 + 256 + 32 + 32 + 9 = 337 floats and silently truncated to 128, dropping
# the whole tool/vector/flag tail and most of the tech bucket. STATE_DIM now
# matches the actual encoder output so every feature reaches the Q-network.


@dataclass(frozen=True)
class StateEncodingConfig:
    """Caps for normalizing scalar features to [0, 1].

    These are *display* caps — values above the cap are clipped to 1.0,
    not discarded. This keeps the Q-network input bounded without losing
    ordinal information.
    """

    max_turns: float = 100.0
    max_findings: float = 50.0
    max_verified_findings: float = 50.0
    max_fp_findings: float = 20.0
    max_commands: float = 200.0
    max_planner_paths: float = 50.0
    tech_stack_vocab: int = 256
    vector_vocab: int = 32
    tool_vocab: int = 32


DEFAULT_STATE_CONFIG = StateEncodingConfig()

# Fixed-size vocabularies for categorical hashing
TECH_STACK_VOCAB = DEFAULT_STATE_CONFIG.tech_stack_vocab  # 256
VECTOR_VOCAB = DEFAULT_STATE_CONFIG.vector_vocab          # 32
TOOL_VOCAB = DEFAULT_STATE_CONFIG.tool_vocab              # 32

_SCALAR_DIMS = 8        # see PentestState.to_vector section 1
# Section 5 is 7 booleans (last_success, kg_available, recon_login/api/upload/
# graphql/websocket) plus 2 NORMALISED PLANNER RATIOS (live, pruned). The
# comment here used to read "(8 booleans + planner_paths_pruned)", which is
# neither the count nor the composition: it invited the next reader to "fix"
# the block by moving a flag into the ratios or vice versa.
_FLAG_DIMS = 9          # 7 booleans + 2 planner ratios
STATE_DIM = (
    _SCALAR_DIMS + TECH_STACK_VOCAB + VECTOR_VOCAB + TOOL_VOCAB + _FLAG_DIMS
)  # 8 + 256 + 32 + 32 + 9 = 337

# Bumped whenever the MEANING of any dimension changes without its size
# changing. `q_learner._load_from_cache()` compares this against the value
# stamped into `q_meta.json` and quarantines a cache that was trained under a
# different layout — shapes alone cannot detect a re-mapping, and adopting
# such a cache gives a network that runs fine on dimensions that no longer
# mean what it learned.
#
# 1 — Sections 3/4 were SHA-256 hash buckets.
# 2 — Sections 3/4 are direct indices into VECTOR_LABELS / TOOL_LABELS.
STATE_LAYOUT_VERSION: int = 2

# Action space — see app/rl/actions.py. `ACTION_SPACE` / `ACTION_DIM` are
# imported above and re-exported; do not redefine them here.

# ── Canonical `last_vector` vocabulary — the ONLY definition ──────────────
# `react_agent._infer_vector_from_tools()` produces
# `action.replace("execute_command_", "")`, so the vocabulary is a pure
# function of ACTION_SPACE. Deriving it means the two cannot drift: adding
# an action automatically legalizes its vector label, and removing one
# automatically retires it.
#
# The previous contract lived only in `_session_builder`'s docstring, which
# listed {"recon", "exploit", "fuzz", "report", "complete", ""} — already
# wrong for three of the six tokens the producer actually emitted
# (`consult_kg`, `report_vulnerability`, `pentest_complete`). A docstring is
# not a validator, so nothing caught it.
#
# "" is the designated "no vector" label (turn 1, content-only turns, and
# anything the builder refuses).
_ACTION_PREFIX = "execute_command_"

VECTOR_LABELS: tuple[str, ...] = tuple(
    action.replace(_ACTION_PREFIX, "") for action in ACTION_SPACE
) + ("",)

# ── Canonical `last_tool` vocabulary — the ONLY definition ────────────────
# `tool_category()` in `app/rl/actions.py` returns exactly the
# `execute_command_*` subset of ACTION_SPACE ("recon" / "exploit" / "fuzz"),
# or "" when the last action produced no shell tool at all (turn 1, or a
# turn that only called `record_vulnerability` / `exit` / `consult_kg`).
# Deriving it here means the encoder's Section 4 cannot accept a label that
# `tool_category()` would never produce, and vice versa.
TOOL_LABELS: tuple[str, ...] = tuple(
    action.replace(_ACTION_PREFIX, "")
    for action in ACTION_SPACE
    if action.startswith(_ACTION_PREFIX)
)


def state_layout_id() -> str:
    """Fingerprint of the state vector's LAYOUT (sizes + label orderings).

    Stamped into `q_meta.json` by `q_learner.save_cache()` and compared on
    load. `STATE_DIM` alone is not an identity: SE4 changed Sections 3 and 4
    from hash buckets to direct indices without changing a single array shape,
    so a shape-only guard would have adopted the old weights and run them on
    dimensions that no longer encode the same thing.
    """
    return (
        f"v{STATE_LAYOUT_VERSION}"
        f"|{_SCALAR_DIMS}+{TECH_STACK_VOCAB}+{VECTOR_VOCAB}+{TOOL_VOCAB}+{_FLAG_DIMS}"
        f"|vec:{'/'.join(VECTOR_LABELS)}"
        f"|tool:{'/'.join(TOOL_LABELS)}"
    )


# ── Helpers ──────────────────────────────────────────────────────────────
def _one_hot(index: int, size: int) -> np.ndarray:
    """One-hot encode `index` into a vector of length `size`.

    Out-of-range indices (including -1 for "unknown") produce a zero vector,
    which the Q-network treats as "no signal" — this is intentional.
    """
    v = np.zeros(size, dtype=np.float32)
    if 0 <= index < size:
        v[index] = 1.0
    return v


def _hash_bucket(text: str, size: int) -> int:
    """Deterministic SHA-256 hash into [0, size).

    Using SHA-256 (not Python's built-in hash()) because Python's hash is
    salted per-process by default (PYTHONHASHSEED), which would make state
    encodings non-reproducible across runs — fatal for offline replay.

    ONLY for OPEN sets. Section 2 (tech stack) needs it because the vocabulary
    is whatever the recon parser found. Sections 3 and 4 are closed sets
    derived from ACTION_SPACE and must NOT use it — see `_label_index()`.
    """
    h = hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()
    return int(h, 16) % size


def _label_index(label: Any, labels: tuple[str, ...]) -> int:
    """Position of a CLOSED-SET label, or -1 (→ zero vector) when unknown.

    Why not `_hash_bucket` for these two sections: with a closed set the
    vocabulary is already an ordered tuple, so hashing it is pure indirection
    and buys two defects.

      1. Aliasing risk. Eight labels into 32 buckets collide with probability
         ~61% under a random hash (birthday bound). The current labels happen
         to be collision-free, which is luck, not design — and a collision is
         permanent because the hash is deterministic.
      2. Typo → invented feature. `_hash_bucket` maps EVERY string into
         [0, 32), so an unknown label does NOT produce a zero vector; it
         produces a private bucket and masquerades as a legitimate attack
         vector. That had to be papered over in `_session_builder`, one module
         away from the producer, and only for `last_vector`.

    Direct indexing makes both impossible: a known label gets its own stable
    dimension, and anything else gets the documented "no signal" zero vector.
    Case and surrounding whitespace are folded — `"Recon"` is the same vector.
    """
    if not isinstance(label, str):
        return -1
    try:
        return labels.index(label.strip().lower())
    except ValueError:
        return -1


def _unit_interval(value: Any, default: float = 0.0) -> float:
    """Coerce to a finite float in [0, 1], falling back to `default`.

    A plain `min(max(x, 0.0), 1.0)` is not enough: NaN propagates through both
    comparisons unchanged, so a corrupt TDI entry would inject NaN into the
    Q-network input and poison every forward pass derived from it. Non-finite
    values are treated as "no information" instead.
    """
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(number):
        return default
    return float(min(max(number, 0.0), 1.0))


def _planner_count(snapshot: Any, key: str) -> int:
    """Non-negative int from an `EGATS.to_dict()` snapshot, 0 when absent.

    Deliberately narrow: only a type/format error in the value falls back to
    0. The old code wrapped a whole `planner.stats()` introspection in
    `except Exception: pass`, which turned every contract violation into a
    silent 0/0 and made the two planner dimensions permanently dead.
    """
    if not isinstance(snapshot, Mapping):
        return 0
    try:
        return max(0, int(snapshot.get(key, 0) or 0))
    except (TypeError, ValueError):
        return 0


# ── PentestState ─────────────────────────────────────────────────────────
@dataclass(frozen=True)
class PentestState:
    """Structured state extracted from a session.

    Frozen so it can be safely cached, hashed, and reused across the
    Q-network forward pass, the CuriosityModule, and the replay trace.
    """

    turn: int
    tdi: float
    mode: str  # "broaden" | "llm" | "exploit"
    findings_count: int
    verified_findings: int
    fp_findings: int
    commands_run: int
    tokens_used: int
    token_budget_remaining: float  # 0..1
    tech_stack: tuple[str, ...]
    last_vector: str
    last_tool: str
    last_success: bool  # did previous turn yield a verified finding?
    kg_available: bool
    planner_paths_live: int
    planner_paths_pruned: int
    recon_login: bool
    recon_api: bool
    recon_upload: bool
    recon_graphql: bool
    recon_websocket: bool

    def to_vector(self) -> np.ndarray:
        """Flatten to a fixed-size float32 vector of shape (STATE_DIM,).

        Sections (see module docstring):
            1. 8 scalars (normalized 0..1)
            2. 256-dim tech stack multi-hot
            3. 32-dim last-vector one-hot
            4. 32-dim last-tool one-hot
            5. 9-dim flags
        """
        parts: list[np.ndarray] = []
        cfg = DEFAULT_STATE_CONFIG

        # 1. Scalar normalized features — 8 dims
        scalars = np.array([
            min(self.turn / cfg.max_turns, 1.0),
            # Clamped and NaN-guarded: the TDI ladder produces [0, 1] today,
            # but a single corrupt `tdi_log` entry (or an explicit
            # `tdi_log_entry=` override from a caller) would otherwise put a
            # network input outside the range every other scalar respects.
            _unit_interval(self.tdi, default=0.5),
            1.0 if self.mode == "broaden" else (0.5 if self.mode == "llm" else 0.0),
            min(self.findings_count / cfg.max_findings, 1.0),
            min(self.verified_findings / cfg.max_verified_findings, 1.0),
            min(self.fp_findings / cfg.max_fp_findings, 1.0),
            min(self.commands_run / cfg.max_commands, 1.0),
            _unit_interval(self.token_budget_remaining),
        ], dtype=np.float32)
        parts.append(scalars)

        # 2. Tech stack multi-hot — TECH_STACK_VOCAB dims
        tech_vec = np.zeros(TECH_STACK_VOCAB, dtype=np.float32)
        for t in self.tech_stack:
            idx = _hash_bucket(t, TECH_STACK_VOCAB)
            tech_vec[idx] = 1.0
        parts.append(tech_vec)

        # 3. Last vector one-hot — VECTOR_VOCAB dims
        # DIRECT index into the closed vocabulary. This used to be
        # `_hash_bucket(self.last_vector, VECTOR_VOCAB)`, which had two
        # defects the surrounding comment only documented rather than fixed:
        # a hash collision would make two different vectors share one feature
        # permanently, and an unknown label got a private bucket of its own
        # instead of the intended "no signal" zero vector — so a typo invented
        # a feature. `_session_builder._normalize_vector()` was added to mask
        # the second symptom one layer away from the producer; it is now
        # redundant, and this line is correct on its own.
        parts.append(_one_hot(_label_index(self.last_vector, VECTOR_LABELS), VECTOR_VOCAB))

        # 4. Last tool one-hot — TOOL_VOCAB dims
        # Same closed-set treatment. `last_tool` is "recon"/"exploit"/"fuzz",
        # or "" when the last action ran no shell tool at all — "" is the
        # zero vector here, because "no tool bucket" is the absence of a
        # signal rather than a third category (Section 3 already carries the
        # exact action for those turns).
        parts.append(_one_hot(_label_index(self.last_tool, TOOL_LABELS), TOOL_VOCAB))

        # 5. Boolean flags + planner-pruned ratio — 9 dims
        flags = np.array([
            float(self.last_success),
            float(self.kg_available),
            min(self.planner_paths_live / cfg.max_planner_paths, 1.0),
            min(self.planner_paths_pruned / cfg.max_planner_paths, 1.0),
            float(self.recon_login),
            float(self.recon_api),
            float(self.recon_upload),
            float(self.recon_graphql),
            float(self.recon_websocket),
        ], dtype=np.float32)
        parts.append(flags)

        # Concatenate and pad/truncate to STATE_DIM (defensive — guards
        # against config drift between encoder and Q-network).
        vec = np.concatenate(parts)
        if vec.shape[0] < STATE_DIM:
            vec = np.pad(vec, (0, STATE_DIM - vec.shape[0]), mode="constant")
        elif vec.shape[0] > STATE_DIM:
            vec = vec[:STATE_DIM]
        return vec.astype(np.float32)


# ── StateEncoder ─────────────────────────────────────────────────────────
class StateEncoder:
    """Builds PentestState from a live session dict.

    The session dict is the in-memory representation maintained by the
    Harness Bridge (W18). It is NOT a SQLAlchemy model — it's a plain dict
    that accumulates state across turns within a single scan.

    Required keys (all optional — missing keys default to "empty"):
        _turn: int                     — current turn number
        _commands_run: int             — total commands executed
        _tokens_used: int              — LLM tokens consumed so far
        _token_budget: int             — total token budget for the scan
        _last_command: str             — last shell command OR tool name run
        _last_tool: str                — the tool that actually ran on the last
                                         turn. OPTIONAL and takes precedence over
                                         `_last_command`. Accepts a tool name, a
                                         command line, or an action name; the
                                         bucket comes from
                                         `app.rl.actions.tool_category()`, so an
                                         action with no shell bucket (e.g.
                                         `record_vulnerability`) encodes as
                                         "no signal" instead of "recon".
                                         `_session_builder.build_session()`
                                         emits it as the `last_tool=` argument.
        _last_vector: str              — last attack vector tried
        _last_turn_success: bool       — did last turn yield a verified finding?
        _kg_consulted: bool            — was KG queried in that same turn?
        tdi_log: list[dict]            — TDA history, last entry wins
        recon_signals: dict            — tech stack + recon flags
        findings: list[dict]           — findings list
        egats_planner: dict|None       — EGATS.to_dict() snapshot (plain data,
                                         NOT the live planner: the session dict
                                         must survive json.dumps). Only
                                         `alive` and `pruned` are read.
    """

    @staticmethod
    def from_session(
        session: dict[str, Any],
        tdi_log_entry: dict[str, Any] | None = None,
    ) -> PentestState:
        """Extract PentestState from the in-memory session dict.

        Args:
            session: live session dict (see class docstring for keys).
            tdi_log_entry: optional explicit TDI override (used by harness
                bridge to inject the current turn's TDI before the encoder
                runs). Takes precedence over session["tdi_log"].

        Returns:
            PentestState — frozen, hashable, can be cached.
        """
        recon = session.get("recon_signals") or {}
        findings = session.get("findings") or []
        # ONE definition of "verified" — see app.evidence.finding_state.
        # This block used to count `f.get("verified") and not
        # f.get("false_positive")` inline, i.e. the LEGACY boolean only.
        # Every producer that writes just the canonical
        # `verification: "confirmed"` then scored 0 here forever: the fifth
        # scalar of Section 1 (and any caller deriving last_turn_success
        # from the same count) went dead with no exception and no log line.
        verified = count_verified_findings(findings)
        fp = sum(1 for f in findings if f.get("false_positive"))
        planner = session.get("egats_planner")

        # Planner path counts — read from the PLAIN-DATA snapshot built by
        # `EGATS.to_dict()`. This used to introspect a live planner via
        # `planner.stats()` inside a bare `except Exception: pass`; that
        # swallowed every contract violation and silently returned 0/0, so a
        # caller passing anything the loop could not walk pinned both planner
        # dimensions to zero forever. Counts are now read directly, and a
        # non-mapping is treated as "no planner" without hiding errors.
        live_paths = _planner_count(planner, "alive")
        pruned_paths = _planner_count(planner, "pruned")

        # TDI (Task Difficulty Index)
        tdi = 0.5
        mode = "llm"
        if tdi_log_entry:
            tdi = tdi_log_entry.get("tdi", 0.5)
            mode = tdi_log_entry.get("mode", "llm")
        elif session.get("tdi_log"):
            last = session["tdi_log"][-1]
            tdi = last.get("tdi", 0.5)
            mode = last.get("mode", "llm")

        # Token budget remaining (0..1)
        tokens_used = session.get("_tokens_used", 0)
        token_budget = session.get("_token_budget", 128_000)
        if token_budget <= 0:
            token_rem = 0.0
        else:
            token_rem = max(0.0, 1.0 - tokens_used / token_budget)

        # Last action hints.
        #
        # `_last_tool` is the EXPLICIT signal and it wins. Reading it is not a
        # nicety: `_last_command` is only ever a guess at "which tool ran",
        # and when the last turn ran no shell tool at all (turn 1, or a turn
        # that only called `record_vulnerability` / `exit` / `consult_kg`) the
        # old code either reused the previous turn's command or defaulted to
        # "recon" — a state feature claiming reconnaissance happened on a turn
        # that recorded a finding.
        #
        # Resolution, in order:
        #   1. session["_last_tool"], when non-empty → `tool_category()` of it,
        #      with NO fallback. An explicit value that maps to "" MEANS ""
        #      ("no shell bucket"); falling back would let the stale
        #      `_last_command` override the caller, which is the same
        #      "heuristic wins over the explicit signal" bug in reverse.
        #   2. otherwise `tool_category(_last_command)`.
        #   3. otherwise "" — no signal, encoded as a zero vector.
        last_cmd = session.get("_last_command", "") or ""
        explicit_tool = session.get("_last_tool")
        if isinstance(explicit_tool, str) and explicit_tool.strip():
            last_tool = tool_category(explicit_tool)
        else:
            last_tool = StateEncoder._infer_tool_from_command(last_cmd)

        last_vector = session.get("_last_vector", "") or ""
        last_success = bool(session.get("_last_turn_success", False))

        return PentestState(
            turn=int(session.get("_turn", 0)),
            tdi=round(float(tdi), 4),
            mode=str(mode),
            findings_count=len(findings),
            verified_findings=verified,
            fp_findings=fp,
            commands_run=int(session.get("_commands_run", 0)),
            tokens_used=int(tokens_used),
            token_budget_remaining=round(float(token_rem), 4),
            tech_stack=tuple(recon.get("technologies") or []),
            last_vector=last_vector,
            last_tool=last_tool,
            last_success=last_success,
            kg_available=bool(session.get("_kg_consulted")),
            planner_paths_live=live_paths,
            planner_paths_pruned=pruned_paths,
            recon_login=bool(recon.get("has_login")),
            recon_api=bool(recon.get("has_api")),
            recon_upload=bool(recon.get("has_upload")),
            recon_graphql=bool(recon.get("has_graphql")),
            recon_websocket=bool(recon.get("has_websocket")),
        )

    @staticmethod
    def _infer_tool_from_command(cmd: str) -> str:
        """Tool bucket for the state's `last_tool` feature.

        DERIVED from `app.rl.actions.TOOL_TO_ACTION` — the same table the agent
        loop uses for credit assignment. It used to be a hand-written substring
        heuristic in this file, and it disagreed with that table on five tools:

            nuclei, nikto, dalfox, wpscan, fscan -> "recon" here,
                                                    execute_command_fuzz there

        so the state vector told the Q-network "the last action was recon"
        while the reward for that same action was credited to the fuzz head.
        The state contradicting its own reward is the SE1 failure class again;
        a second table is what made it possible.

        Returns "" when the value names no shell bucket (unknown binary, or a
        non-command action such as `record_vulnerability`). It used to return
        "recon" as a catch-all default, which made every unclassifiable turn —
        including a turn that only recorded a finding — actively assert that
        reconnaissance had run. "" is encoded as the documented "no signal"
        zero vector, which is the honest answer, and it is also what turn 1
        produces.

        Prefer `session["_last_tool"]` when the caller knows the tool directly;
        `from_session` consults it first and only falls back to this.
        """
        return tool_category(cmd)


# ── Convenience ──────────────────────────────────────────────────────────
def encode_session(
    session: dict[str, Any],
    tdi_log_entry: dict[str, Any] | None = None,
) -> np.ndarray:
    """One-liner: session dict → 337-dim state vector.

    Equivalent to:
        StateEncoder.from_session(session, tdi_log_entry).to_vector()
    """
    return StateEncoder.from_session(session, tdi_log_entry).to_vector()