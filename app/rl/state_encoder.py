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

    Section 2: 256-dim tech stack multi-hot (hash-bucketed)
    Section 3: 32-dim last-vector one-hot (hash-bucketed)
    Section 4: 32-dim last-tool one-hot (hash-bucketed)
    Section 5: 9-dim flags
        - last_success
        - kg_available
        - planner_paths_live / max
        - planner_paths_pruned / max
        - recon_login, recon_api, recon_upload, recon_graphql, recon_websocket

Pure numpy, no torch dependency.
"""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

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
_FLAG_DIMS = 9          # see PentestState.to_vector section 5 (8 booleans + planner_paths_pruned)
STATE_DIM = (
    _SCALAR_DIMS + TECH_STACK_VOCAB + VECTOR_VOCAB + TOOL_VOCAB + _FLAG_DIMS
)  # 8 + 256 + 32 + 32 + 9 = 337

# Action space — must match DoubleQLearner output size (app.rl.q_learner)
ACTION_SPACE: list[str] = [
    "execute_command_recon",
    "execute_command_exploit",
    "execute_command_fuzz",
    "select_skill",
    "consult_kg",
    "report_vulnerability",
    "pentest_complete",
]
ACTION_DIM = len(ACTION_SPACE)  # 7


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
    """
    h = hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()
    return int(h, 16) % size


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
            float(self.tdi),
            1.0 if self.mode == "broaden" else (0.5 if self.mode == "llm" else 0.0),
            min(self.findings_count / cfg.max_findings, 1.0),
            min(self.verified_findings / cfg.max_verified_findings, 1.0),
            min(self.fp_findings / cfg.max_fp_findings, 1.0),
            min(self.commands_run / cfg.max_commands, 1.0),
            float(self.token_budget_remaining),
        ], dtype=np.float32)
        parts.append(scalars)

        # 2. Tech stack multi-hot — TECH_STACK_VOCAB dims
        tech_vec = np.zeros(TECH_STACK_VOCAB, dtype=np.float32)
        for t in self.tech_stack:
            idx = _hash_bucket(t, TECH_STACK_VOCAB)
            tech_vec[idx] = 1.0
        parts.append(tech_vec)

        # 3. Last vector one-hot — VECTOR_VOCAB dims
        parts.append(_one_hot(_hash_bucket(self.last_vector, VECTOR_VOCAB), VECTOR_VOCAB))

        # 4. Last tool one-hot — TOOL_VOCAB dims
        parts.append(_one_hot(_hash_bucket(self.last_tool, TOOL_VOCAB), TOOL_VOCAB))

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
        _last_command: str             — last shell command executed
        _last_vector: str              — last attack vector tried
        _last_turn_success: bool       — did last turn yield a verified finding?
        _kg_consulted: bool            — was KG queried this turn?
        tdi_log: list[dict]            — TDA history, last entry wins
        recon_signals: dict            — tech stack + recon flags
        findings: list[dict]           — findings list
        egats_planner: object|None     — planner with .stats() method
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
        verified = sum(
            1 for f in findings
            if f.get("verified") and not f.get("false_positive")
        )
        fp = sum(1 for f in findings if f.get("false_positive"))
        planner = session.get("egats_planner")

        # Planner stats — defensive: planner may be None (W17 will wire KG/EGATS)
        live_paths = 0
        pruned_paths = 0
        if planner is not None:
            try:
                for stat in planner.stats():
                    if stat.pruned:
                        pruned_paths += 1
                    else:
                        live_paths += 1
            except Exception:
                pass

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

        # Last action hints — derive last_tool from last_command
        last_cmd = session.get("_last_command", "") or ""
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
        """Heuristic mapping from shell command → RL tool category.

        Used as a fallback when no explicit _last_tool is set on the session.
        The 3 categories match the first 3 entries of ACTION_SPACE:
            - "recon"   → execute_command_recon
            - "exploit" → execute_command_exploit
            - "fuzz"    → execute_command_fuzz
        """
        cmd_lower = cmd.lower()
        if any(t in cmd_lower for t in ("nuclei", "nikto", "nmap", "whatweb", "httpx", "subfinder")):
            return "recon"
        if any(t in cmd_lower for t in ("sqlmap", "metasploit", "msfconsole", "exploit")):
            return "exploit"
        if any(t in cmd_lower for t in ("ffuf", "gobuster", "feroxbuster", "arjun", "paramspider", "katana")):
            return "fuzz"
        return "recon"  # default


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