"""
VAPT-AI RL — session dict builder for StateEncoder.

The ReAct loop in app/agents/react_agent.py maintains scan state as local
variables (turn counter, token total, findings list, last tool called, ...).
StateEncoder.from_session() expects a flat dict with specific keys. This
helper bridges the two: it packs the loop's locals into the dict shape the
encoder consumes, so the agent file stays readable and the encoder stays
decoupled from the agent's internal layout.

Feature status (this file) — keep every row HONEST. This list has already
drifted once in EACH direction: `tdi_log` was still documented as a future
stub after it had gone live, and `recon_signals` was documented as
"populated in Phase 2" while no caller passed it at all — so 256 of the 337
dimensions were zero on every turn of every scan until it was wired.
    - recon_signals: {}       -> WIRED. Callers pass
                                 HarnessBridge.recon_signals(); {} means the
                                 tech stack has not been parsed yet this scan
    - tdi_log: []             -> ACTIVE (Phase 3). HarnessBridge.tda_compute()
                                 appends per turn; the encoder reads the LAST
                                 entry, and the list is empty until the first
                                 compute of the scan
    - egats_planner: None     -> ACTIVE (Phase 4). A PLAIN dict from
                                 EGATS.to_dict(), never a live EGATS instance
    - _kg_consulted: False    -> per-turn; True only for a turn that ran a KG query

The encoder tolerates every empty default (it falls back to neutral feature
values), so an UNWIRED field still yields a valid 337-dim vector — it simply
carries no signal, and silently. That is why each row above states whether
the field is actually wired, not which phase it belongs to.

The returned dict is a SNAPSHOT, and the builder is STATELESS. Both halves
matter:

  * Snapshot — nothing in the result aliases a caller-owned object. A
    `list(findings)` copy is NOT enough: it shares the inner finding dicts,
    so a verifier/auditor that flipped a field after the call would rewrite
    the state a consumer already encoded. Every nested structure is detached
    (`_snapshot_value`) two levels down — session → list → dict → scalar,
    which is the full depth of the payload (see `FINDING_STATE_KEYS`:
    all-scalar values, so this is a complete copy, not a best effort).

  * Stateless — do NOT park state you intend to keep across turns in here.
    A fresh dict is constructed on every turn, so anything written into the
    result is discarded with it. This is exactly how the reward shaper's
    scan-lifetime bookkeeping got re-initialised every turn: `_seen_finding_keys`,
    `_credited_verified_ids` and `_penalised_fp_ids` must be owned by the
    CALLER (see `app.rl.reward.new_credit_state()` and the `credit_state=`
    parameter), never read back from a session. Symptoms when this is
    violated are subtle: a credited finding is paid again whenever the gate
    re-opens, the same false positive is penalised on every remaining turn,
    and every finding looks novel (+20 instead of +1).
"""
from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any, TypedDict

from app.rl.state_encoder import VECTOR_LABELS

logger = logging.getLogger(__name__)

# Unknown labels already reported. `build_session()` runs once per turn, so
# per-call de-duplication would log the same typo ~100 times per scan.
_WARNED_VECTORS: set[str] = set()

# Non-mapping planner types already reported (same warn-once reasoning).
_WARNED_PLANNERS: set[str] = set()

# Counter fields that failed int coercion, already reported (one per field).
_WARNED_COUNTS: set[str] = set()


def _as_count(value: Any, field: str) -> int:
    """Coerce a counter to `int` at the boundary without raising.

    This builder is where the agent loop's locals cross into plain data, so
    it is also the last place that can refuse to blow up. Every caller wraps
    `build_session()` in `try/except Exception: log.warning(...)`, which means
    an `int()` TypeError does not fail loudly — it silently drops the RL step
    for that turn (no state, no reward, no experience). A feature that dies
    quietly is worse than one that reports a bad number.

    So the policy matches `_normalize_vector` (R19) and `_normalize_planner`
    (R20): degrade to a neutral value and warn ONCE per field. `int("5")`
    and `int(3.9)` still behave exactly as before; only genuinely
    uncoercible input (`None`, `"abc"`, `nan`, `inf`, a `dict`) changes,
    from a swallowed crash into 0 plus one log line.

    `OverflowError` is listed explicitly because `int(float("inf"))` raises
    it, not `ValueError` — a hand-written `except (TypeError, ValueError)`
    would let infinity through as a crash.
    """
    if isinstance(value, bool):  # before int: bool IS an int subclass
        return int(value)
    if isinstance(value, int):
        return value
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        if field not in _WARNED_COUNTS:
            _WARNED_COUNTS.add(field)
            logger.warning(
                "build_session: %s=%r is not an integer — using 0",
                field, value,
            )
        return 0


def _snapshot_value(value: Any, depth: int = 2) -> Any:
    """Detach a nested value from the caller's objects, two levels deep.

    `list(findings)` copies the LIST but shares every finding dict inside it,
    so a verifier that mutated `verified` / `false_positive` after this call
    would silently rewrite a session another consumer had already encoded.
    The same aliasing applied to `recon_signals` (passed straight through) and
    to the entries of `tdi_log`.

    Depth 2 is a full copy of this payload, not an approximation: session →
    list → dict → scalar. `finding_to_state()` emits only str/bool/None
    values (`FINDING_STATE_KEYS`), so there is nothing below the dict to
    share. `copy.deepcopy` would buy nothing and costs a memo walk per turn.

    The `depth` floor also makes the function safe on self-referential input:
    at depth 0 the value is returned as-is instead of recursing forever.
    """
    if depth <= 0:
        return value
    if isinstance(value, Mapping):
        return {k: _snapshot_value(v, depth - 1) for k, v in value.items()}
    if isinstance(value, list):
        return [_snapshot_value(v, depth - 1) for v in value]
    return value


class EgatsPlannerSnapshot(TypedDict, total=False):
    """Plain-data form of an `EGATS` planner, as built by `EGATS.to_dict()`.

    A `TypedDict`, not `Any`: the session dict is rebuilt every turn, is read
    by `StateEncoder.from_session()`, and is expected to survive `json.dumps`
    (episode dumps, SSE payloads, the `rl_stats` JSON column). A live planner
    object satisfies none of that — it mutates, and serializing it raises
    `TypeError`.

    Only `alive` and `pruned` are read by the encoder (the two planner dims of
    the 337-dim vector); the rest is diagnostic context for logs.
    """

    total_paths: int
    alive: int
    pruned: int
    total_attempts: int
    total_successes: int
    paths: list[dict[str, Any]]


def _normalize_planner(value: Any) -> dict[str, Any] | None:
    """Admit only a plain mapping (or None) as `egats_planner`.

    `Any | None` was the old annotation and it was a lie with a bite: the
    field is forwarded verbatim into the session dict, so a caller passing the
    live `EGATS` instance produced a session dict that `json.dumps` cannot
    serialize (`TypeError: Object of type EGATS is not JSON serializable`).
    The builder is a serialization boundary and must not accept "Anything".

    The correct call is `planner.to_dict()` — callers never pass the planner.
    A non-mapping is dropped with a one-time warning rather than raising:
    same policy as `_normalize_vector()`, a mapping smell in one caller must
    not abort a scan that is already mid-episode. Dropping it yields the
    encoder's neutral 0/0, which is at least truthful.
    """
    if value is None:
        return None
    if isinstance(value, Mapping):
        # One level deeper than the session's own payload: the snapshot is
        # itself a dict, so its `paths` list holds path dicts (dict → list →
        # dict → scalar). Depth 2 here would share the path dicts.
        return _snapshot_value(value, depth=3)
    name = type(value).__name__
    if name not in _WARNED_PLANNERS:
        _WARNED_PLANNERS.add(name)
        logger.warning(
            "build_session: egats_planner must be a serializable mapping "
            "(pass EGATS.to_dict()); got %s — dropped",
            name,
        )
    return None


def _normalize_vector(value: Any) -> str:
    """Force `last_vector` into `VECTOR_LABELS`, admitting nothing else.

    `StateEncoder` one-hot encodes this field with `_hash_bucket()`, which
    accepts ANY string — an unknown label does not degrade to "no signal",
    it occupies a bucket of its own and reads as a distinct attack vector.
    Silent by construction: no exception, no warning, just a feature that
    means nothing.

    Deliberately does NOT raise. A bad label is a mapping smell in the
    caller, not a reason to abort a running scan mid-episode — the policy
    gets the "unknown" bucket and a warning names the offender. Warn-once
    per distinct value so the log stays readable.

    Case and surrounding whitespace are folded because they carry no
    meaning here: `"Recon"` and `" recon "` are the same vector, and
    treating them as three buckets would teach the policy nothing.
    """
    text = str(value or "").strip().lower()
    if text in VECTOR_LABELS:
        return text
    if text and text not in _WARNED_VECTORS:
        _WARNED_VECTORS.add(text)
        logger.warning(
            "build_session: unknown last_vector %r mapped to '' "
            "(valid: %s)",
            text, sorted(VECTOR_LABELS),
        )
    return ""


def build_session(
    *,
    scan_id: str,
    target: str,
    turn: int,
    commands_run: int,
    tokens_used: int,
    token_budget: int,
    last_command: str,
    last_vector: str,
    last_turn_success: bool,
    findings: list[dict[str, Any]],
    last_tool: str = "",
    kg_consulted: bool = False,
    tdi_log: list[dict[str, Any]] | None = None,
    recon_signals: dict[str, Any] | None = None,
    egats_planner: EgatsPlannerSnapshot | None = None,
) -> dict[str, Any]:
    """Assemble the in-memory session dict consumed by StateEncoder.

    Args:
        scan_id: current scan identifier (for logging only — encoder ignores).
        target: target URL or IP (for logging only — encoder ignores).
        turn: 1-indexed ReAct turn number.
        commands_run: cumulative tool calls so far this scan.
        tokens_used: cumulative LLM tokens consumed so far this scan.
        token_budget: hard token budget for the scan (used to compute the
            "fraction remaining" feature).
        last_command: name of the most-recently dispatched tool
            (e.g. "nmap", "ffuf"). Empty string on turn 1.
        last_tool: the tool that ACTUALLY ran on the previous turn, stated
            explicitly. Emitted as `_last_tool`, which
            `StateEncoder.from_session()` consults BEFORE `_last_command`.
            Accepts a tool name ("nuclei"), a command line ("sqlmap -u ..."),
            or an action name ("record_vulnerability"); the bucket is derived
            by `app.rl.actions.tool_category()`, so it lands in the same
            vocabulary the credit table uses.
            Why this exists: `last_command` cannot express "the last turn ran
            no shell tool at all", so a turn that only called
            `record_vulnerability` / `exit` / `consult_kg` used to encode as
            "recon". Leaving this empty keeps the old derivation
            (`tool_category(last_command)`), which now returns "" for those
            turns instead of guessing.
            Default "" means "not supplied" — it is NOT the same as an
            explicit "" from `from_session`'s point of view, and no caller
            should rely on that distinction: pass the tool name when you have
            it.
        last_vector: short attack-vector label derived from last_command.
            Must be a member of `state_encoder.VECTOR_LABELS` (the seven
            ACTION_SPACE labels, `execute_command_` prefix stripped, plus
            ""); anything else is folded to "" with a one-time warning.
            Do not re-list the vocabulary here — it is derived, and a
            hand-maintained copy is exactly how this field drifted from
            three live tokens (`consult_kg`, `report_vulnerability`,
            `pentest_complete`) before.
        last_turn_success: True if the previous turn raised the count of
            VERIFIED findings, i.e. `count_verified_findings(findings) >
            prev_verified_count` (see `app.evidence.finding_state`).
            Deliberately NOT "the findings count went up": a record the
            verifier only rated `plausible` also raises `len(findings)`,
            pays 0 in `reward.compute_reward` §2, and would make the state
            vector claim progress on a turn that earned nothing. The
            encoder/decoder pair treats this as the "last action worked"
            feature, so it must agree with the reward's definition of
            "worked".
        findings: list of finding dicts built by
            `finding_to_state()` — one entry per `record_vulnerability`
            call whose result had `status == "recorded"`. Rejected calls
            are NOT in this list (`status` lives on the tool result, not on
            the dict; `FINDING_STATE_KEYS` has no `status` key), so a
            consumer must not filter on it. The encoder counts verified vs.
            false-positive.
        kg_consulted: True if the turn that just finished ran a KG query.
            PER-TURN, not a latch. This builder is stateless and returns a
            fresh dict, so "has the scan ever consulted the KG" cannot be
            represented here at all — a caller that wants that for logging
            must keep its own scan-lifetime flag. It matters: the encoder
            turns this into ONE dimension of the 9-dim flag block, and a
            latched value is constant from turn 2 onward, i.e. a dead
            feature that teaches the policy nothing. The caller resets it
            per turn for the same reason `last_turn_success` is per-turn.
        tdi_log: per-turn TDI history. ACTIVE (Phase 3): HarnessBridge.tda_compute()
            appends one entry per turn; the encoder reads the last. Empty until
            the first compute of the scan, which is honest, not a stub.
        recon_signals: tech-stack + recon flags, from
            `HarnessBridge.recon_signals()`. `{}` means the tech stack has not
            been parsed yet (it is set once, on the first recon tool output).
            Do NOT drop this argument when adding a call site: the encoder
            turns `technologies` into the 256-dim multi-hot block, and an
            omitted argument leaves 256 of 337 dims permanently zero — which
            is exactly what happened before this was wired.
        egats_planner: plain-data planner snapshot — the return value of
            `EGATS.to_dict()`, or None. NOT the live `EGATS` object: this
            builder is the serialization boundary for the session dict, and
            a live planner makes the dict raise `TypeError` under
            `json.dumps`. Non-mapping values are dropped with a one-time
            warning. The encoder reads only `alive` and `pruned` from it.

    Returns:
        Flat dict with the keys StateEncoder.from_session() reads. Every
        nested value is a detached copy — mutating `findings`, `recon_signals`
        or `tdi_log` after the call cannot alter the returned session. The
        dict carries no scanned-lifetime state: it is rebuilt every turn, so
        persistence belongs to the caller (see the module docstring).
    """
    return {
        # ── scalar state (8 dims after normalization) ──────────────
        "_turn": _as_count(turn, "turn"),
        "_commands_run": _as_count(commands_run, "commands_run"),
        "_tokens_used": _as_count(tokens_used, "tokens_used"),
        # NOT clamped to >= 1. `state_encoder.from_session()` already handles
        # `token_budget <= 0` explicitly (`token_rem = 0.0`), so a floor of 1
        # would not prevent a division — it would turn "budget unknown" into
        # "100% of the budget remains" (1.0), a false positive signal in the
        # reward/state. Zero stays zero and reads as "no budget information".
        "_token_budget": _as_count(token_budget, "token_budget"),
        "_last_command": str(last_command or ""),
        "_last_tool": str(last_tool or ""),
        "_last_vector": _normalize_vector(last_vector),
        "_last_turn_success": bool(last_turn_success),
        "_kg_consulted": bool(kg_consulted),
        # ── structured state (multi-hot / one-hot inputs) ──────────
        "recon_signals": _snapshot_value(recon_signals or {}),
        "findings": _snapshot_value(findings),
        # ── Phase 3+ stubs (encoder tolerates empty/None) ──────────
        "tdi_log": _snapshot_value(tdi_log or []),
        "egats_planner": _normalize_planner(egats_planner),
        # ── opaque metadata (encoder ignores, useful for logs) ─────
        "scan_id": str(scan_id),
        "target": str(target),
    }