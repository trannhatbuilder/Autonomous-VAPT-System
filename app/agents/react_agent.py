"""
VAPT-AI Real ReAct Agent — LLM + Tools + SSE events.

This is the core scan engine: a single-agent ReAct loop that plans,
calls real security tools, observes their output, and records verified
findings. Each iteration emits SSE events (scan_progress) so the
frontend chat shows real-time AI thinking + tool calls + results.

Architecture:
    1. Build system prompt (structured 5-phase pentest workflow).
    2. Build tool schemas (32 security tools + record_vulnerability
       + fetch_tool_output + exit).
    3. Preflight: 1-shot LLM ping to verify the API key works
       (fail fast instead of cryptic failure mid-loop).
    4. Loop:
       a. (Phase 1) RL HarnessBridge.select_action() — encode state,
          query policy, format advisory hint.
       a2. (Phase 3) RL HarnessBridge.tda_compute() — compute Task
           Difficulty Index for this turn, render mode hint.
       b. Call LLM via chat_completion with messages + tools.
       c. If LLM returns tool_calls:
          - Emit SSE: scan_progress (thought + tool_name).
          - Execute each tool via SubprocessExecutor.
          - Hard-block tools that already failed N times (retry guard).
          - Phase tracking: MONOTONIC (1→2→3→4→5, never backwards).
            Phase 3 skip detection (flexible): if LLM jumps from Phase 1/2
            to Phase 4, check for parameter evidence (URL with query string,
            or data/cookie/form_data in tool_args). Skip is ALLOWED with
            evidence; warning emitted only when skipping without evidence.
          - Advisory round-robin: inject ONLY ONE advisory message per turn
            (cycling RL → TDA → EGATS). SC corrective message always
            injected (high priority). KG brief injected once. This prevents
            attention dilution from 5 simultaneous advisory messages.
          - Emit SSE: scan_progress (observation).
          - Append tool results to messages.
          - (Phase 1) RL HarnessBridge.record_experience() — compute
            reward, push transition into PER buffer, signal training.
          - (Phase 2) After first recon tool: parse output → extract
            tech stack → KG consult → stash brief for next-turn injection.
          - (Phase 2) After record_vulnerability: record consult hit/miss
            on the KG so edge probabilities update.
          - Prune old tool results (context-window management).
          - Continue loop.
       d. If LLM returns content only (no tool_calls): LLM is done
          thinking — append assistant message and continue.
       e. If LLM calls "exit" tool: scan complete — return summary.
       f. Max iterations reached: return what we have.
    5. (Phase 1) RL HarnessBridge.end_scan() — cancel training worker,
       decay epsilon, persist checkpoint.

Stop conditions and guardrails:
    - MAX_AGENT_TURNS = 100 (D18 guardrail).
    - Token budget tracking: soft warning + hard cancel (ENFORCED in loop).
    - Phase tracking is MONOTONIC (1 -> 2 -> 3 -> 4 -> 5, never backwards).

Per-tool retry guard:
    A tool that fails MAX_TOOL_FAILURES times in one scan is hard-blocked.
    INVOCATION failures (bad flag, unknown option, parse error) cannot be
    fixed by retrying with different VALUES — the LLM doesn't know the
    correct flag, and each retry burns an iteration + subprocess spawn.

Context pruning:
    Old tool results are truncated in the message history. The FULL output
    is preserved by tool_bridge (50KB cap + disk spill), so evidence is
    never lost — only the LLM's working context is bounded.

RL Harness (Phase 1):
    Advisory-only — the policy suggests an action category each turn, the
    LLM keeps full judgment. Toggle with VAPT_AI_RL_HARNESS_ENABLED.

KG Integration (Phase 2):
    After the first recon tool produces output, the agent parses it for
    tech-stack tokens, consults the Knowledge Graph for prior-attack
    intel, and injects an advisory brief as a system message on the next
    turn. After each record_vulnerability call, the harness records a
    consult hit (finding accepted) or miss (finding rejected) so the KG
    edge probabilities update. The LLM can also call the `consult_kg`
    tool manually mid-scan with a focus_vector. Toggle with
    VAPT_AI_KG_ENABLED.

TDA Integration (Phase 3):
    Each turn, the harness computes a Task Difficulty Index (TDI) from 4
    dimensions: horizon (remaining steps normalised by the scan's actual
    max_turns), evidence confidence, context load, and historical success
    rate. The TDI maps to BROADEN (>0.60, expand recon) or LLM_PICK
    (<=0.60, let LLM decide). EXPLOIT is override-only and gated on
    EVIDENCE: when a finding scores >= CONFIRMED_VULN (verifier-accepted,
    replay-verified, or human-verified), the harness forces EXPLOIT
    regardless of the raw TDI — a merely plausible finding does not lock
    the scan's direction. The raw TDI is preserved through the override
    so the RL state vector keeps a real difficulty signal. The TDI is
    also fed into the RL state vector via the tdi_log slot in
    build_session(). Toggle with VAPT_AI_TDA_ENABLED.

EGATS Integration (Phase 4):
    After the first KG consult produces a tech stack, the harness seeds
    an EGATS (Evidence-Guided Attack Tree Search) planner with attack
    paths from the KG. Each turn, EGATS selects the highest-UCB alive
    path and injects it as an advisory system message. After each
    record_vulnerability call, the harness records the outcome (success
    or fail) on the selected path, updates its TDI EMA, and prunes paths
    that consistently produce high-difficulty states. Toggle with
    VAPT_AI_EGATS_ENABLED.

Self-Correction Integration (Phase 5):
    When record_vulnerability returns status="rejected", the harness
    processes the rejection: computes a pattern signature from
    (vuln_type, normalized_endpoint), tracks in-memory rejection counts,
    renders a 3-tier corrective message (corrective → warning → blacklist),
    and pushes an escalating RL reward (-5/-7/-10) into the PER store
    with high priority. The corrective message is injected as a system
    message on the next turn so the LLM learns from the rejection.
    Toggle with VAPT_AI_SC_ENABLED.

Per-Scan KG Fork (Phase 6):
    When VAPT_AI_KG_ISOLATION=true, the harness forks the KG into a
    per-scan deepcopy at attach time. All consult/hit/miss/edge_outcome
    writes go to the fork. The fork is merged back into the main KG ONLY
    when the scan status="completed" — failed/aborted/max_iterations
    scans discard the fork to avoid corrupting the global KG with partial
    learning signals. Toggle with VAPT_AI_KG_ISOLATION (default false).
"""
from __future__ import annotations

import json
import logging
import os
import re
from collections import defaultdict
from typing import Any

from app.agents.llm_client import chat_completion
from app.agents.tool_bridge import build_tool_schemas, execute_tool_call, create_executor
from app.rl.harness import (
    HarnessBridge,
    MODEL_CONTEXT_WINDOW_DEFAULT,
    TurnRecommendation,
    get_harness,
    is_rl_harness_enabled,
    release_harness,
)
from app.rl._session_builder import build_session
from app.rl.reward import compute_reward, d26_event, new_credit_state
from app.evidence.finding_state import (
    count_verified_findings,
    finding_to_state,
)
from app.rl.state_encoder import ACTION_SPACE
from app.kg.advisor import extract_tech_stack

logger = logging.getLogger(__name__)


# ── RL action category mapping ───────────────────────────────────────
# The table itself lives in `app/rl/actions.py` so the state encoder can read
# the SAME mapping for its `last_tool` feature. It used to be defined here,
# with a second hand-written mapping in `state_encoder._infer_tool_from_command`
# that disagreed on nuclei/nikto/dalfox/wpscan/fscan — so the state vector
# reported "recon" for an action credited to the fuzz head.
#
# Imported and re-exported under the historical private name: tests and
# `_did_follow_rl_action()` read `_TOOL_TO_ACTION`, and a rename would churn
# them without changing behaviour.
from app.rl.actions import TOOL_TO_ACTION as _TOOL_TO_ACTION  # noqa: E402

# NOTE: there is deliberately no `curl` entry. `curl` is allowlisted by the
# sandbox but is not an LLM tool, and `resolve_action()` no longer emits it
# (the exploit branch used to build a `curl` GraphQL probe, which neither
# round-tripped nor survived credit assignment). An entry here would be dead
# configuration that makes the table look more complete than it is.


# ── Recon tool set ────────────────────────────────────────────────────
# Tools whose output we parse for tech-stack tokens. The first time one
# of these runs in a scan, we parse its output, extract the tech stack,
# and consult the KG. Subsequent recon calls don't re-trigger the
# auto-consult (the LLM can still call consult_kg manually if needed).
_RECON_TOOLS: set[str] = {
    "nmap", "whatweb", "httpx", "masscan", "rustscan",
    "subfinder", "amass", "gau", "waybackurls",
}


def _did_follow_rl_action(called_tools: list[str], rl_action: str) -> bool:
    """True if ANY called tool this turn maps to the RL-suggested action.

    called_tools: list of tool names the LLM actually called this turn.
    rl_action:    the action_name returned by HarnessBridge.select_action().
    """
    if not called_tools or not rl_action:
        return False
    return any(_TOOL_TO_ACTION.get(t, "") == rl_action for t in called_tools)


def _should_inject_rl_guidance(rec: "TurnRecommendation | None", *, train_steps: int) -> tuple[bool, str]:
    """Shadow mode: only show RL guidance when the policy is worth listening to.

    The originating measurement was: ε starts at 1.0 every scan, so
    `effective_epsilon` is 1.0 and EVERY action is a random pick — 200/200
    `exploratory` over a fresh policy. Injecting that text 1 turn in 3 was a
    pure noise generator: it cost context and actively misdirected the agent.

    (That comment used to quote `min(1.0, epsilon + curiosity_bonus)` as the
    reason ε stayed pinned. Since the curiosity bonus is now SCALED by the
    remaining ε budget, the bonus reaches exactly 0.0 at `epsilon_end`, so the
    pinning argument only holds early in a run. The gate does not depend on
    the old argument: it is a per-turn test of the action actually returned.)

    Two conditions, both cheap:

      exploratory   — this particular action was NOT the greedy choice. A
                      single non-greedy pick must never be shown as a
                      recommendation. `exploratory` is measured per mode:
                      ε-greedy reports "the random branch was taken",
                      Boltzmann reports "the sample differed from argmax"
                      (see `PentestPolicy._select_boltzmann` — a hardcoded
                      True there disabled this gate for every turn).
      train_steps   — the Q-network has performed at least one gradient
                      step. Without this, `exploratory=False` is still
                      meaningless: an argmax over randomly-initialised
                      weights is arbitrary, it just doesn't *look* random.

    Guidance therefore fades in on its own as the policy trains; nothing
    has to be flipped by hand. SSE telemetry is emitted regardless — see
    the caller — so the action is still observable while suppressed.

    Returns:
        (inject, reason) — reason is a stable token for logging/metrics.
    """
    if rec is None:
        return False, "no_recommendation"
    if rec.exploratory:
        return False, "exploratory"
    if train_steps <= 0:
        return False, "untrained"
    return True, "trained"


def _resolve_executed_action(called_tools: list[str]) -> tuple[int | None, str | None]:
    """Map the tools the LLM ACTUALLY ran to an RL action index + name.

    Why this exists (credit-assignment bug): the loop used to record
    `action_idx=rl_action_rec.action_index` — the action the POLICY
    suggested — together with the reward of whatever the LLM actually did.
    `DoubleQLearner.train_step` puts the gradient only on
    `grad_q[action]`, so whenever the LLM ignored the suggestion the
    update taught Q(s, a_suggested) the outcome of a DIFFERENT action.
    With a low follow-rate that is a systematic, not random, wrong label.

    Picking the LAST mapped tool is deliberate: the reward's
    `command_output` is taken from the last tool message of the turn, and
    the next state is observed after it — so the last action is the one
    that immediately produced the reward being credited.

    Returns:
        (index, name) — (None, None) when no called tool maps onto
        ACTION_SPACE (e.g. a content-only turn, or `fetch_tool_output`,
        which is a context utility and not an action in the policy's
        space). The caller must then SKIP the transition rather than
        invent a label.
    """
    if not called_tools:
        return None, None
    for tool in reversed(called_tools):
        action = _TOOL_TO_ACTION.get(tool, "")
        if not action:
            continue
        try:
            return ACTION_SPACE.index(action), action
        except ValueError:  # mapping drifted from ACTION_SPACE
            logger.warning(
                "rl_action_not_in_space | tool=%s | action=%s", tool, action,
            )
    return None, None


def _infer_vector_from_tools(tools: list[str]) -> str:
    """Derive last_vector (recon/exploit/fuzz/report/complete) for state.

    The encoder uses this as a one-hot feature. Returns "" if no tools
    were called (turn 1 or content-only assistant turns).
    """
    if not tools:
        return ""
    action = _TOOL_TO_ACTION.get(tools[-1], "")
    if not action:
        return ""
    return action.replace("execute_command_", "")


# ── D26 turn events (reward.py §1) ───────────────────────────────────
# `compute_reward(..., turn_events=[...])` is the ONLY path that pays
# R_TIMEOUT (-1.0) and R_SCOPE_VIOLATION (-3.0). Nothing in the agent loop
# ever built those events, so both branches were dead: a timed-out tool call
# scored exactly like a successful no-op (and BETTER than an empty result —
# the "[error] TIMEOUT: ..." text is longer than the wasted-turn threshold,
# so §4 never fired and the +0.2 efficiency bonus was paid instead).
#
# These markers are what `tool_bridge.execute_tool_call` actually puts in the
# tool message (`output_parts`). Keep them in sync with that function — a
# rewrite of those strings silently re-kills the signals.
_TIMEOUT_MARKERS: tuple[str, ...] = (
    "[error] TIMEOUT:",      # ExecutionStatus.HARD_TIMEOUT (hard kill)
    "[error] Timeout after",  # SubprocessExecutor soft timeout
)
# Long messages that still mean "nothing came back". Both are exclusive
# to a zero-yield execution: tool_bridge emits `[empty_result]` only when
# `output_parts` is empty, and `[error] Tool execution failed:` only when
# the execution status is FAILED (no stdout at all). They are far longer
# than the 50-char "produced output" threshold, so without this they
# collected the productive-turn bonus — a stall drawing a salary.
# D26 groups these with `timeout` ("-1 timeout / no useful output").
_NO_USEFUL_OUTPUT_MARKERS: tuple[str, ...] = (
    "[empty_result]",
    "[error] Tool execution failed:",
)
_SCOPE_VIOLATION_MARKERS: tuple[str, ...] = (
    "SCOPE VIOLATION:",       # tool_bridge COMPLETED-with-scope_violation
    "[error] Scope violation:",
)


def _d26_events_from_tool_outputs(
    tool_messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Synthesize D26 turn events from the tool messages of ONE turn.

    Only signals that are (a) observable in the agent loop and (b) NOT
    already rewarded elsewhere are emitted. Deliberately excluded:

      * `exploit_tier1` / `exploit_tier2` — no data source. Tier 1 needs a
        C2 session (`app.c2` does not exist in this tree); the sqlmap
        `parse_hints` regexes are declarative metadata that NOTHING reads
        (see app/tools/dnsenum.yaml). Emitting either on top of the
        +20/+1 finding bonus would also double-pay the same event.
      * `fp_rejected` — reward.py §3 already derives it from the
        penalised-identity set (`_penalised_fp_ids`), which is idempotent
        per finding.
      * `recon_success` — recon volume is already captured by
        `findings` / `_commands_run` in the state, and paying +0.5 per
        recon turn would reward tool spam.
      * `user_aborted` / `destructive_no_hitl` — both are HITL outcomes;
        the HITL gate returns its message as the tool output so the agent
        can reason about it, and those are policy outcomes, not RL signal.
    """
    events: list[dict[str, Any]] = []
    for msg in tool_messages:
        if not isinstance(msg, dict) or msg.get("role") != "tool":
            continue
        content = msg.get("content")
        if not isinstance(content, str) or not content:
            continue
        if any(marker in content for marker in _TIMEOUT_MARKERS):
            events.append(d26_event("timeout"))
        elif any(marker in content for marker in _NO_USEFUL_OUTPUT_MARKERS):
            # Zero-yield execution with a verbose explanation attached.
            events.append(d26_event("timeout"))
        elif any(marker in content for marker in _SCOPE_VIOLATION_MARKERS):
            events.append(d26_event("scope_violation"))
    return events


# ── Agent stop conditions ────────────────────────────────────────────
MAX_AGENT_TURNS = 100
# Earlier exit-gate logic (min turns / min commands) was removed because
# it prevented scans from completing legitimately when they had fewer
# than 20 turns / 25 commands. Kept here as 0 to preserve the constants
# for any caller that still reads them.
MIN_AGENT_TURNS = 0
MIN_COMMANDS_FOR_COMPLETE = 0

# ── Premature-exit SOFT gate ─────────────────────────────────────────
# With the HARD gate disabled (MIN_* above are 0), the only thing stopping
# an `exit` at iteration 7 is prompt discipline — which the model does not
# always follow. A hard block would re-introduce the original bug (scans
# that legitimately need < 25 commands could never finish), so the gate is
# SOFT: the first `exit` below these thresholds is challenged with a
# counter-question, and the SECOND `exit` is always honoured. No hard
# block, no infinite loop, one extra turn of cost at most.
EXIT_SOFT_GATE_MIN_COMMANDS = 10  # tool calls below this look premature
EXIT_SOFT_GATE_MIN_TURNS = 8      # and so does finishing before this turn

# Backward-compat alias (old code may reference MAX_ITERATIONS)
MAX_ITERATIONS = MAX_AGENT_TURNS  # D18 guardrail: max 100 turns per scan

# ── Retry guard ──────────────────────────────────────────────────────
# A tool that failed MAX_TOOL_FAILURES times in one scan is hard-blocked.
# Key insight: INVOCATION failures (bad flag, unknown option, parse error)
# cannot be fixed by retrying with different VALUES — the LLM doesn't know
# the correct flag, and each retry burns an iteration + subprocess spawn.
MAX_TOOL_FAILURES = 2

# Substrings (lowercased) that mark a tool output as a failure. Kept
# conservative — broad markers like "error" alone would false-positive
# on nikto/nuclei findings text.
#
# Every marker beyond the original set was verified against the REAL binary
# on this host, because two of the failure modes that matter most were
# invisible to the old list:
#   whatweb  -> `unrecognized option \`--output-format'`  exit 0 (!), 149 B
#   sqlmap   -> `sqlmap: error: missing a mandatory option (-d, -u, ...)` exit 2
# NOTE: whatweb exits 0 on a rejected option, so exit-code checks alone can
# never catch it — only the output text can.
_TOOL_FAIL_MARKERS: tuple[str, ...] = (
    "binary not found",
    "tool execution error",
    "error executing",
    "no such option",
    "unrecognized option",          # GNU getopt — whatweb: unrecognized option `--output-format'
    "unrecognized arguments",       # argparse — "unrecognized arguments: --x"
    "missing a mandatory option",   # sqlmap 1.10.4 VERIFIED: "missing a mandatory option (-d, -u, ...)"
    "missing mandatory option",     # same complaint, other sqlmap builds
    "unknown flag",
    "flag provided but not defined",
    "incorrect usage",
    "usage:",                       # argparse/getopt usage banner — see _WEAK_TOOL_FAIL_MARKERS
    "parse error",
    "invalid value",        # gobuster: invalid value "15" for flag -to
    "tool '",
)

# Markers that are ALSO plausible in normal output, so they only count when
# the whole output is short. `usage:` appears in any crawled documentation
# page; a real arg-parser complaint is a few hundred bytes, a crawl result
# is not. Without this split `usage:` would let a katana crawl of a docs
# site block katana as "broken".
_WEAK_TOOL_FAIL_MARKERS: tuple[str, ...] = ("usage:",)
_WEAK_MARKER_MAX_CHARS = 4_000

# ── Token budget (enforced) ──────────────────────────────────────────
# Raw token counts (the $ equivalent depends on the model's pricing).
# Defaults sized for ~$5 soft / ~$20 hard on a mid-tier model — tune per
# deployment via env or per llm_config in future.
TOKEN_SOFT_LIMIT = 2_500_000   # soft warning
TOKEN_HARD_LIMIT = 10_000_000  # hard cancel

# ── Token BUDGET fed to the RL state encoder ─────────────────────────
# NOT the same thing as the limits above. Those are runaway guards: they
# exist to abort a pathological scan and sit an order of magnitude above
# what a normal scan spends. Wiring one of them into
# `build_session(token_budget=...)` made the encoder's
# `token_budget_remaining` feature read ~0.998 for an entire scan — i.e. a
# CONSTANT input, dead in the same way the old TDI C dimension was dead
# (it used the 10M limit as the context-window denominator).
# This is the volume a scan is expected to consume, so the feature moves
# from 1.0 towards 0.0 as the budget is used up. Calibrate per deployment
# with VAPT_AI_SCAN_TOKEN_BUDGET.
#
# The env override is VALIDATED rather than trusted. It used to be a bare
# `int(os.getenv(...))`, which accepted `0` and `-5` happily — and because
# `StateEncoder.from_session()` maps `token_budget <= 0` to 0.0, a single bad
# env var pinned `token_budget_remaining` to 0.0 on EVERY turn: the exact
# dead-constant input this constant exists to prevent, reintroduced through
# the configuration door. The same is true of any value at or above
# TOKEN_SOFT_LIMIT, which is by definition the runaway threshold rather than
# expected spend (that is the original bug, back again). Both are rejected
# with a warning and the default is used, so the feature degrades to
# "correctly calibrated" instead of "silently constant".
_DEFAULT_SCAN_TOKEN_BUDGET = 500_000


def _load_scan_token_budget() -> int:
    """Read `VAPT_AI_SCAN_TOKEN_BUDGET`, falling back to the default.

    Valid range is `0 < value < TOKEN_SOFT_LIMIT`. Anything else — a
    non-numeric string, zero, a negative number, or a value so large the
    resulting fraction barely moves — is logged and replaced by the default
    rather than silently disabling the feature.

    Not raising: this runs at import time, and an unparseable tuning knob
    should not stop the whole server from starting.
    """
    raw = os.getenv("VAPT_AI_SCAN_TOKEN_BUDGET", str(_DEFAULT_SCAN_TOKEN_BUDGET))
    try:
        value = int(raw)
    except (TypeError, ValueError):
        logger.warning(
            "VAPT_AI_SCAN_TOKEN_BUDGET=%r is not an integer — using %d",
            raw, _DEFAULT_SCAN_TOKEN_BUDGET,
        )
        return _DEFAULT_SCAN_TOKEN_BUDGET
    if value <= 0 or value >= TOKEN_SOFT_LIMIT:
        logger.warning(
            "VAPT_AI_SCAN_TOKEN_BUDGET=%d is outside (0, %d) — the RL token "
            "feature would be a constant. Using %d instead.",
            value, TOKEN_SOFT_LIMIT, _DEFAULT_SCAN_TOKEN_BUDGET,
        )
        return _DEFAULT_SCAN_TOKEN_BUDGET
    return value


SCAN_TOKEN_BUDGET = _load_scan_token_budget()

# ── Context management ───────────────────────────────────────────────
# Old tool results are truncated in the message history. The FULL output
# is preserved by tool_bridge: capped at 50KB, spilled to disk beyond
# that, and stored by `tool_artifacts` under its execution_id. The note
# below therefore names the tool that can actually read it back
# (`fetch_tool_output`) — an earlier version pointed at "scan artifacts /
# spill file", which the agent had no way to open.
MAX_TOOL_RESULT_CHARS = 15_000   # keep recent results up to this size
KEEP_RECENT_RESULTS = 6          # number of most-recent results kept full
PRUNE_TO_CHARS = 1_500           # old results truncated to this + pointer

_TOOL_RESULT_PRUNED_NOTE = (
    "\n... [truncated {n} chars to save context — call "
    "fetch_tool_output(execution_id='{execution_id}') to read the full "
    "output if you need it.]"
)
_TOOL_RESULT_PRUNED_NOTE_NO_ID = (
    "\n... [truncated {n} chars to save context — the full output is in "
    "the scan artifacts. Re-run the tool if you need the detail again.]"
)
# Execution ids appear as a footer appended by tool_bridge.execute_tool_call:
#     "\n[execution_id] <uuid>"
_EXECUTION_ID_RE = re.compile(r"\[execution_id\]\s*([A-Za-z0-9_\-]+)")


def _prune_old_tool_results(messages: list[dict[str, Any]]) -> int:
    """Truncate OLD tool results in the message history (in place).

    Keeps the most recent KEEP_RECENT_RESULTS tool messages intact and
    truncates older ones to PRUNE_TO_CHARS. Returns number pruned.

    The truncation notice carries the message's execution_id when there is
    one, so the LLM can page the full output back with `fetch_tool_output`
    instead of re-running an expensive tool.

    Called once per iteration after all tool results are appended.
    """
    tool_indices = [i for i, m in enumerate(messages) if m.get("role") == "tool"]
    if len(tool_indices) <= KEEP_RECENT_RESULTS:
        return 0

    pruned = 0
    for idx in tool_indices[:-KEEP_RECENT_RESULTS]:
        msg = messages[idx]
        content = msg.get("content") or ""
        if len(content) > PRUNE_TO_CHARS * 2:
            removed = len(content) - PRUNE_TO_CHARS
            match = _EXECUTION_ID_RE.search(content)
            if match:
                note = _TOOL_RESULT_PRUNED_NOTE.format(
                    n=removed, execution_id=match.group(1),
                )
            else:
                note = _TOOL_RESULT_PRUNED_NOTE_NO_ID.format(n=removed)
            msg["content"] = content[:PRUNE_TO_CHARS] + note
            pruned += 1
    return pruned


def _calc_progress(iteration: int, max_iterations: int) -> int:
    """Progress inside the loop — 20%..90%, never exceeds 95."""
    frac = iteration / max(max_iterations, 1)
    return min(95, 20 + int(frac * 70))


# ── Advisory round-robin (1 message per turn, with fall-through) ─────
# Order matters only for the rotation: RL → TDA → EGATS.
ADVISORY_ROTATION: tuple[str, ...] = ("rl", "tda", "egats")


def _select_advisory(
    iteration: int,
    *,
    rl_msg: dict[str, Any] | None,
    tda_msg: dict[str, Any] | None,
    egats_msg: dict[str, Any] | None,
) -> tuple[str | None, dict[str, Any] | None]:
    """Pick exactly ONE advisory message for this turn.

    The rotation slot is `(iteration - 1) % 3`, but if that advisor has
    nothing to say we FALL THROUGH to the next one in the rotation. The
    earlier `if/elif/elif` chain silently returned nothing instead, so a
    third of all turns carried no intelligence at all — most visibly
    before EGATS is seeded from the KG (egats_select() → None), i.e.
    exactly the early phase where guidance matters most.

    The rotation index still advances by one per turn, so fall-through
    cannot let a single advisor monopolise the slot.

    Returns:
        (picked_name, message) — (None, None) when no advisor has content.
    """
    messages = {"rl": rl_msg, "tda": tda_msg, "egats": egats_msg}
    slot = (iteration - 1) % len(ADVISORY_ROTATION)
    for offset in range(len(ADVISORY_ROTATION)):
        name = ADVISORY_ROTATION[(slot + offset) % len(ADVISORY_ROTATION)]
        msg = messages.get(name)
        if msg:
            return name, msg
    return None, None


def _exit_needs_challenge(
    *,
    exit_challenged: bool,
    findings_count: int,
    commands_run: int,
    iteration: int,
) -> bool:
    """True when an `exit` call should be questioned once before honouring it.

    Soft gate, not a hard block: with MIN_AGENT_TURNS / MIN_COMMANDS_FOR_
    COMPLETE at 0 there was NO guard against an exit at turn 7, and the old
    hard gate broke legitimate short scans. So we challenge the FIRST exit
    that arrives with too little work and no findings, and require a second
    explicit exit to finish. `exit_challenged` makes this happen at most
    once, so it can never loop.
    """
    if exit_challenged:
        return False
    if findings_count > 0:
        return False
    return (
        commands_run < EXIT_SOFT_GATE_MIN_COMMANDS
        and iteration < EXIT_SOFT_GATE_MIN_TURNS
    )


# ── P1: preflight LLM check ──────────────────────────────────────────

async def _preflight_llm_check(llm_config: dict[str, Any]) -> tuple[bool, str]:
    """P1: verify the OpenAI API key works before entering the ReAct loop.

    Sends a 1-token "ping" message to the configured model. If the call
    succeeds (any 2xx response), the API key + base URL + model name are
    all valid. If it fails, we return a human-readable error instead of
    crashing mid-loop after spending tokens on the system prompt.

    Returns:
        (ok, error_message) — error_message is "" when ok=True.
    """
    try:
        response = await chat_completion(
            llm_config=llm_config,
            messages=[
                {"role": "user", "content": "ping"},
            ],
            tools=None,  # no tools — just a minimal call
            temperature=0.0,
        )
        # A 2xx response proves the key/base_url/model are valid. Reasoning
        # models may leave `content` empty (their answer sits in
        # `reasoning_content`), so accept reasoning output and usage too.
        content = (response.get("content") or "").strip()
        reasoning = (response.get("reasoning_content") or "").strip()
        if content or reasoning or response.get("tool_calls") or response.get("usage"):
            return True, ""
        return False, "LLM returned empty response — check provider config"
    except Exception as exc:
        msg = str(exc)
        # Common error patterns → friendlier messages
        if "401" in msg or "Invalid API key" in msg or "Incorrect API key" in msg:
            return False, (
                f"OpenAI API key invalid or unauthorized. "
                f"Go to Settings → AI Channels and re-enter the key. "
                f"Underlying error: {msg}"
            )
        if "404" in msg or "model_not_found" in msg or "does not exist" in msg:
            return False, (
                f"Model {llm_config.get('model', '?')} not found on this "
                f"provider. Check the model name in Settings. "
                f"Underlying error: {msg}"
            )
        if "Connection" in msg or "timeout" in msg.lower() or "connect" in msg.lower():
            base_url = llm_config.get("base_url", "default")
            return False, (
                f"Cannot reach LLM endpoint ({base_url}). "
                f"Check network + base_url setting. "
                f"Underlying error: {msg}"
            )
        return False, f"LLM preflight failed: {msg}"


# ── P1: install hint for missing binaries ────────────────────────────

# Maps binary names → install instructions for the LLM observation message.
# When SubprocessExecutor.execute() returns error="Binary not found: X",
# we replace the raw error with this friendly hint so the LLM knows the tool
# isn't installed on the host and can switch to a different available tool
# (or stop attempting this tool family).
BINARY_INSTALL_HINTS: dict[str, str] = {
    "nmap":         "nmap is not installed on the VAPT-AI host. Install with: `apt install nmap` (Debian/Ubuntu).",
    "nuclei":       "nuclei is not installed. Install with: `go install github.com/projectdiscovery/nuclei/v3/cmd/nuclei@latest` and add $GOPATH/bin to PATH.",
    "sqlmap":       "sqlmap is not installed. Install with: `apt install sqlmap` or `git clone --depth 1 https://github.com/sqlmapproject/sqlmap.git /opt/sqlmap`.",
    "ffuf":         "ffuf is not installed. Install with: `go install github.com/ffuf/ffuf/v2@latest`.",
    "gobuster":     "gobuster is not installed. Install with: `apt install gobuster`.",
    "feroxbuster":  "feroxbuster is not installed. Install from: https://github.com/epi052/feroxbuster/releases.",
    "httpx":        "httpx (projectdiscovery) is not installed. Install with: `go install github.com/projectdiscovery/httpx/cmd/httpx@latest`.",
    "whatweb":      "whatweb is not installed. Install with: `apt install whatweb`.",
    "nikto":        "nikto is not installed. Install with: `apt install nikto` or `git clone https://github.com/sullo/nikto.git`.",
    "dalfox":       "dalfox is not installed. Install with: `go install github.com/hahwul/dalfox@latest`.",
    "wpscan":       "wpscan is not installed. Install from: https://github.com/wpscanner/wpscan#installers (gem install wpscan).",
    "subfinder":    "subfinder is not installed. Install with: `go install github.com/projectdiscovery/subfinder/v2/cmd/subfinder@latest`.",
    "katana":       "katana is not installed. Install with: `go install github.com/projectdiscovery/katana/cmd/katana@latest`.",
    "amass":        "amass is not installed. Install with: `go install -v github.com/owasp-amass/amass/v4/...@master`.",
    "masscan":      "masscan is not installed. Install with: `apt install masscan` (needs root for raw sockets).",
    "rustscan":     "rustscan is not installed. Install with: `cargo install rustscan` or use the docker image `rustscan/rustscan:latest`.",
    "gau":          "gau is not installed. Install with: `go install github.com/lc/gau/v2/cmd/gau@latest`.",
    "waybackurls":  "waybackurls is not installed. Install with: `go install github.com/tomnomnom/waybackurls@latest`.",
    "hydra":        "hydra is not installed. Install with: `apt install hydra`.",
    "hashcat":      "hashcat is not installed. Install with: `apt install hashcat`.",
    "john":         "john (John the Ripper) is not installed. Install with: `apt install john`.",
    "fierce":       "fierce is not installed. Install with: `apt install fierce` or `gem install fierce`.",
    "dnsenum":      "dnsenum is not installed. Install with: `apt install dnsenum`.",
    "impacket-smbexec":       "impacket is not installed. Install with: `pipx install impacket` (or `pip install impacket`).",
    "impacket-psexec":         "impacket is not installed. Install with: `pipx install impacket`.",
    "impacket-wmiexec":        "impacket is not installed. Install with: `pipx install impacket`.",
    "impacket-secretsdump":    "impacket is not installed. Install with: `pipx install impacket`.",
    "impacket-GetUserSPNs":   "impacket is not installed. Install with: `pipx install impacket`.",
    "netexec":      "netexec (a.k.a. nxc) is not installed. Install with: `pipx install netexec` or `pip install netexec`.",
    "responder":    "responder is not installed. Install with: `apt install responder` or `git clone https://github.com/lgandx/Responder.git`.",
    "linpeas":      "linpeas is not installed. Download from: https://github.com/peass-ng/PEASS-ng/releases/latest (LinPEAS).",
    "winpeas.exe":  "winpeas.exe is not installed. Download from: https://github.com/peass-ng/PEASS-ng/releases/latest (WinPEAS).",
    "mimikatz.exe": "mimikatz.exe is not installed. Download from: https://github.com/gentilkiwi/mimikatz/releases.",
}


def _format_binary_not_found_hint(error_or_output: str) -> str:
    """If input contains 'Binary not found: X', return output with install hint.

    Used in two flows:
        1. Exception path: error = "Binary not found: nmap" (raised)
        2. Tool-result path: output = "...\\n[error] Binary not found: nmap"
           (returned by execute_tool_call embedding SubprocessExecutor's
           ToolResult.error string)

    For flow 2, preserves the original output text + appends the install
    hint as a clear "HINT" block so the LLM understands both what happened
    and what to do next.

    If no "Binary not found" marker, returns the input unchanged.
    """
    if "Binary not found" not in error_or_output:
        return error_or_output

    # Extract binary name from "Binary not found: <name>"
    marker_idx = error_or_output.find("Binary not found:")
    after_marker = error_or_output[marker_idx + len("Binary not found:"):]
    # Take first whitespace-delimited token (could have leading space)
    try:
        binary = after_marker.strip().split()[0].rstrip(",.;:")
    except Exception:
        return error_or_output

    hint = BINARY_INSTALL_HINTS.get(binary)
    if not hint:
        hint = (
            f"binary '{binary}' is not installed on the VAPT-AI host. "
            f"Ask the operator to install it, or try a different available tool. "
            f"Do NOT retry the same tool."
        )

    return (
        f"{error_or_output}\n\n"
        f"───── INSTALL HINT ─────\n"
        f"{hint}\n"
        f"───── END HINT ─────\n"
    )


# ── Failure classification for the retry guard ───────────────────────

def _looks_like_failure(tool_output: str) -> bool:
    """True if the tool output indicates the call itself failed.

    Conservative: only matches invocation/runtime failure markers, never
    plain words like 'error' that appear inside normal scan findings.

    The window is 2000 chars, not 800. sqlmap prints a ~440-byte ASCII
    banner and, once arguments ARE supplied, its legal disclaimer pushes the
    actual error line past 800 chars — the failure then classified as a
    SUCCESS, which reset `tool_fail_counts` (line ~1565). Net effect: a
    phantom sqlmap run not only escaped the block, it refreshed the tool's
    credit and could loop forever.
    """
    head = tool_output[:2000].lower()
    for marker in _TOOL_FAIL_MARKERS:
        if marker not in head:
            continue
        # Weak markers are only trustworthy in a SHORT output (see the
        # _WEAK_TOOL_FAIL_MARKERS comment).
        if marker in _WEAK_TOOL_FAIL_MARKERS and len(tool_output) > _WEAK_MARKER_MAX_CHARS:
            continue
        return True
    return False


# ── System prompt — structured pentest workflow (5 phases) ───────────

SYSTEM_PROMPT = """You are VAPT-AI — an autonomous pentest agent. You have pre-authorized access to the target. Do NOT ask for permission.

You have access to ALL tools. Use them directly — there is no supervisor, no specialist transfer, no HITL gate.

## CRITICAL: no shell, no arbitrary commands

There is NO shell/execute tool. You CANNOT run `curl`, `echo`, `cat`, or any arbitrary command. If you need raw HTTP requests, use httpx / whatweb / katana / nuclei — they cover the same use cases. NEVER call tools that are not in your Available Tools list — they do not exist and calling them wastes iterations.

## MANDATORY: Follow this 5-phase workflow IN ORDER

Do NOT skip phases. Do NOT jump ahead. Complete each phase before moving to the next.

IMPORTANT — recording findings: call `record_vulnerability` IMMEDIATELY when you confirm a vulnerability with evidence, in whichever phase you confirmed it. Do NOT batch recordings to the end. Phase 4 is only for vulnerabilities that need ACTIVE exploitation to confirm (SQLi, XSS, command injection...). Facts verifiable by observation (exposed panel, missing header, info disclosure) are recorded as soon as seen.

### Phase 1: Reconnaissance (gather info about target)
Goal: Understand the target — what ports are open, what services run, what tech stack is used.

Steps:
1. Run `nmap` against the target host (use -Pn — most targets block ping; skip if target is a single URL on a known port)
2. Run `httpx` to probe HTTP/HTTPS services (note the [cdn] field — it tells you if the target is behind Cloudflare/Akamai/etc.)
3. Run `whatweb` to identify tech stack (PHP, ASP.NET, nginx, Apache, etc.)
4. Run `subfinder` and/or `amass` if target is a domain (find subdomains)

Output expected: list of open ports, services, tech stack, subdomains, CDN/WAF presence.
DO NOT move to Phase 2 until you know what the target is running.

### Phase 2: Scanning & Enumeration (map the attack surface)
Goal: Discover endpoints, directories, hidden files, and configurations.

Steps:
1. Run `katana` (depth 2, max_time 2m) to crawl the site and discover pages/parameters
2. Run `ffuf` or `feroxbuster` with a wordlist to find hidden directories/files (/admin, /backup.zip, /.env, /setup.php, /phpmyadmin/)
3. Run `nikto` for web server misconfiguration checks
4. Record interesting endpoints found (login pages, admin panels, API endpoints, config files)

Output expected: list of discovered URLs, directories, files, parameters.
DO NOT move to Phase 3 until you have mapped the attack surface.

### Phase 3: Vulnerability Analysis / Triage (identify potential vulns)
Goal: From the mapped surface, identify what MIGHT be vulnerable. Prioritize by severity.

**Phase 3 is REQUIRED by default** — scanners (nuclei, nikto, dalfox) detect
issues you can't see from URLs alone: missing headers, info disclosure, exposed
panels, outdated components, CVEs. Running a scanner first confirms the
vulnerability class and justifies your Phase 4 tool selection.

**When you CAN skip Phase 3 (proceed directly to Phase 4):**
- You have already identified a SPECIFIC injectable parameter during Phase 2
  (e.g. `/page.php?id=1` with a visible query parameter).
- You have form data or cookie-based evidence (authenticated endpoint).
- The tool_args include `data`, `cookie`, or `form_data` fields.

**When you MUST run Phase 3 first (do NOT skip):**
- You only have bare URLs (no query parameters).
- You haven't enumerated the attack surface yet.
- You're guessing the vulnerability class (SQLi vs XSS vs LFI).

Steps (run AT LEAST ONE scanner before Phase 4, unless you have parameter evidence):
1. Run `nuclei` with severity `high,critical,medium` against discovered URLs.
   This is the PRIMARY scanner — it covers CVEs, misconfig, exposed panels,
   default logins, and many vulnerability classes in one pass.
2. Run `nikto` for web server misconfiguration checks (complements nuclei).
3. Run `dalfox` on parameters found in Phase 2 if you suspect XSS.
4. Probe key endpoints with `httpx` to check for exposed panels and
   information disclosure (phpMyAdmin, phpinfo(), error_log, server-status, .env).
5. Use `httpx`/`whatweb` to check for outdated components (old jQuery, PHP version, etc.)
6. List potential vulnerabilities with severity ranking (critical → low).

Output expected: list of potential vulnerabilities with severity.
**DO NOT skip Phase 3 unless you have specific parameter evidence.** If you call
sqlmap/hydra/metasploit on a bare URL without running a scanner first, you are
doing blind exploitation and will likely waste turns.

### Phase 4: Exploitation (prove the vulnerability is real)
Goal: For each potential vulnerability from Phase 3 that needs active exploitation, run an exploit to PROVE it is real. A scanner warning is NOT enough — you need a PoC (command + server response).

Steps:
- SQL Injection: Run `sqlmap` on suspected injectable parameters (batch mode). The sqlmap output IS the PoC.
- Parameter discovery: test interesting parameters found in Phase 2 with sqlmap/dalfox as appropriate.
- Missing Security Headers: tool output showing the response headers IS the PoC.
- Exposed Panel: probe output showing 200 OK on /phpmyadmin/ IS the PoC.
- Misconfiguration: `nikto` or `nuclei` output showing the misconfig IS the PoC.

For EACH confirmed vulnerability, IMMEDIATELY call `record_vulnerability` with:
- `evidence`: the EXACT tool output (command + server response) that proves the vuln
- `tool`: the NAME of the tool that produced the evidence (`sqlmap`, `dalfox`, `nuclei`, `nmap`, `httpx`, ...). This is REQUIRED — it is shown as "Tool used" in the report. Do NOT write `agent`.
- `command`: the EXACT command you ran, with REAL values already substituted
- `vuln_type`: sqli, xss, lfi, rce, information-disclosure, security_misconfiguration, etc.
- `severity`: critical, high, medium, low, info
- `cvss_score`: estimate based on impact (critical=9-10, high=7-8.9, medium=4-6.9, low=0.1-3.9)
- `cwe_id`: if known (e.g. CWE-89 for SQLi, CWE-79 for XSS)
- `wstg_test_id`: if known (e.g. WSTG-INPV-05 for SQLi)
- `description`: what + where + why exploitable
- `remediation`: how to fix (specific, not generic)

DO NOT call record_vulnerability without evidence. The call will be REJECTED.

### Phase 5: Reporting & Remediation (wrap up)
Goal: Summarize all findings and exit.

Steps:
1. Review all recorded vulnerabilities
2. Call `exit` with a summary:
   - Total findings count
   - List of findings with severity
   - Overall assessment of the target's security posture
   - SCAN LIMITATIONS: if the target is behind a WAF/CDN/challenge page, state explicitly what could NOT be tested and why the results are limited. A scan that saw only a CDN edge is NOT evidence that the origin is secure.

## Evidence Requirement (CRITICAL)

record_vulnerability REQUIRES the `evidence` parameter containing raw tool output.
PoC = command you ran + server's response. Without evidence, the call is REJECTED.

Example — Exposed phpMyAdmin:
```json
{
  "title": "Exposed phpMyAdmin Panel",
  "vuln_type": "security_misconfiguration",
  "severity": "high",
  "cvss_score": 7.5,
  "cwe_id": "CWE-16",
  "wstg_test_id": "WSTG-CONF-04",
  "target": "https://target.com",
  "location": "/phpMyAdmin/",
  "evidence": "$ httpx -u https://target.com/phpMyAdmin/\\nhttps://target.com/phpMyAdmin/ [200] [phpMyAdmin] [MySQL,PHP,jQuery]",
  "description": "phpMyAdmin is publicly accessible without IP restriction at /phpMyAdmin/. An attacker can attempt brute-force or exploit known phpMyAdmin CVEs against this instance.",
  "remediation": "Restrict phpMyAdmin access to trusted IPs (allowlist) or place it behind VPN. Keep phpMyAdmin updated to the latest version."
}
```

Rules
ALWAYS follow the 5-phase workflow in order. Do NOT skip phases.
**Phase 3 is REQUIRED by default** — run at least one scanner (nuclei, nikto, or dalfox) before Phase 4 exploitation. **Exception:** you MAY skip Phase 3 if you have specific parameter evidence (a URL with query parameters like `/page.php?id=1`, or form data/cookie fields in tool_args). Without evidence, Phase 3 prevents blind exploitation and helps you select the correct tool (sqlmap for SQLi, dalfox for XSS, etc.).
ALWAYS run a tool first, capture its output, THEN call record_vulnerability with that output as evidence.
NEVER write placeholders in `evidence` or `command`. The fields must contain the EXACT command with REAL values — `https://target.com/?q=x`, not `{{URL}}?q=x`, not `<target>`, not `$URL`. A placeholder makes the PoC non-reproducible and the finding looks fabricated.
ALWAYS set the `tool` parameter to the real tool name (`sqlmap`, `dalfox`, `nuclei`, `nmap`, `httpx`...). NEVER set it to `agent`, `curl`, or the exploit phase name — the report displays it as "Tool used".
metasploit: use `auxiliary/scanner` modules FIRST (version checks, non-destructive probes). Use `exploit/` modules ONLY for vulnerabilities you already confirmed with evidence (sqlmap output, nuclei match). NEVER guess exploits against unconfirmed targets.
Do NOT fabricate findings — only report what tools actually detected.
Call record_vulnerability for EACH confirmed vulnerability (1 call per finding), IMMEDIATELY when confirmed.
When done with all 5 phases, call exit with a summary of all findings.
If a tool fails, try a different tool — do NOT call the same failed tool more than twice. The system will hard-block tools that keep failing.
If a tool returns EMPTY output, do NOT conclude the target is down or secure. Empty output usually means: (a) WAF/challenge page, (b) your IP is rate-limited, (c) wrong flags. Try different flags or a different tool ONCE, then note the limitation in your final report.
If the target is behind a CDN/WAF (check httpx [cdn] field), ALL port-scan and web-scan results reflect the CDN edge, NOT the origin server. State this limitation in the final summary.

Available Tools
Recon: nmap, httpx, whatweb, masscan, rustscan, subfinder, amass, gau, waybackurls
Enumerate: ffuf, gobuster, feroxbuster, katana, gau, waybackurls
Vuln Scan: nuclei, nikto, dalfox, wpscan, fscan
Exploit: sqlmap, metasploit, hydra
Utility: record_vulnerability, fetch_tool_output, exit, consult_kg

`consult_kg` queries the Knowledge Graph for prior-attack intel on the current target's tech stack. It returns an advisory brief listing attack paths that worked on similar stacks in past scans. Call it after you have a basic tech-stack fingerprint (post-recon) or before switching attack-vector classes (e.g. moving from recon to exploitation). Pass `focus_vector` (e.g. "sqli", "xss", "lfi") to narrow the consult. The brief is advisory — your reasoning wins.

`fetch_tool_output` re-reads the FULL output of an earlier tool run when a result was cut off (`[truncated N chars]` / `[middle truncated ...]`). Pass the `execution_id` from that result's `[execution_id] ...` footer, or `tool_name` for that tool's newest output. Use it instead of re-running an expensive scan when you need the tail of a long result (e.g. the sqlmap verdict) as evidence.

(That is the COMPLETE list. There is no execute/shell/curl tool. Do not attempt to call anything else.)

Actually RUN the tools — do NOT just describe what you would do. Each tool call produces real output that becomes evidence for findings.
"""


async def run_react_scan(
    target: str,
    user_prompt: str,
    llm_config: dict[str, Any],
    scan_id: str,
    max_iterations: int = 100,
    executor: Any = None,
    rl_enabled: bool | None = None,
) -> dict[str, Any]:
    """Run a single-agent ReAct scan loop (one LLM + all tools).

    No LangGraph supervisor, no specialist transfers, no HITL gate.
    One LLM + all tools + record_vulnerability + exit.

    Args:
        target: target URL or IP (e.g. "http://example.com").
        user_prompt: user's natural-language scan request.
        llm_config: LLM config from DB (provider, api_key, model, ...).
        scan_id: scan ID for SSE events + DB attribution.
        max_iterations: max ReAct iterations (default 100).
        executor: SubprocessExecutor instance (created if None).
        rl_enabled: toggle the RL Harness Bridge for this scan.
            None -> read from VAPT_AI_RL_HARNESS_ENABLED env (default true).

    Returns:
        dict with: status, findings_count, iterations, total_tokens,
        final_summary, error, rl_stats (None when RL is disabled).
    """
    from app.pentest.events import emit_scan_progress
    from app.pentest.events import emit_phase_change

    # ── RL harness setup (Phase 1) ──────────────────────────────────
    # Initialize the harness variable early so _finalize_rl() can close
    # over it. The actual harness instance is created AFTER the preflight
    # check passes — no point spinning up the train worker if the LLM
    # key is bad.
    if rl_enabled is None:
        rl_enabled = is_rl_harness_enabled()
    harness: HarnessBridge | None = None

    # R27: one done=True transition per episode, no matter which of the six
    # return paths fires. Set by `_record_terminal_transition()`.
    _terminal_recorded = False

    async def _finalize_rl(scan_status: str = "unknown") -> dict[str, Any] | None:
        """Cleanup RL harness at scan end. Idempotent.

        Called from every return path. When the harness was never
        initialized (preflight failure or RL disabled), returns None.
        Otherwise: cancels the train worker, decays epsilon, persists
        the checkpoint, merges the KG fork (Phase 6) when status is
        "completed", and returns the final stats dict.

        Args:
            scan_status: the final scan status — "completed", "aborted",
                "error", "max_iterations", "token_budget_exceeded", or
                "unknown" (default when status not yet determined).
                Phase 6 uses this to decide whether to merge the KG fork.
        """
        if harness is None:
            return None
        try:
            episode_reward = harness.get_stats().get("episode_reward", 0.0)
            await harness.end_scan(episode_reward=episode_reward)
            # Phase 6: merge KG fork only on successful scan completion.
            # Failed/aborted scans discard the fork to avoid corrupting
            # the global KG with partial/inconsistent learning signals.
            try:
                harness.kg_merge_fork(scan_status=scan_status)
            except Exception as exc:
                logger.warning(
                    "kg_merge_fork_failed | scan=%s | err=%s",
                    scan_id, exc,
                )
            return harness.get_stats()
        except Exception as exc:
            logger.warning("rl_end_failed | scan=%s | err=%s", scan_id, exc)
            return None
        finally:
            release_harness(scan_id)

    async def _record_terminal_transition(reason: str) -> None:
        """Push the FINAL (s, a, r, s', done=True) transition.

        Without this the store contains only `done=False` rows: every
        episode ended by `exit` / turn exhaustion left the Q-target
        bootstrapping off `max Q(s')` of a state that never existed, and
        `episode_reward` was missing the closing turn entirely.

        The action is `pentest_complete` because ending the episode is
        exactly what this turn did (`exit` maps to it in
        `_TOOL_TO_ACTION`). Must run BEFORE `_finalize_rl()` — `end_scan()`
        stops the training worker.

        Best-effort: any failure is logged and swallowed, since this is
        bookkeeping on a return path.
        """
        nonlocal _terminal_recorded
        if harness is None:
            return
        if _terminal_recorded:
            # R27: several return paths can now reach here (abort, budget,
            # loop error, exit, exhaustion). Recording twice would push two
            # done=True rows and double the closing reward into
            # `episode_reward`, which `end_episode()` then uses for the
            # reward-adaptive ε decay branch.
            logger.debug(
                "rl_terminal_transition_skipped | scan=%s | reason=%s "
                "(already recorded)",
                scan_id, reason,
            )
            return
        _terminal_recorded = True
        try:
            terminal_session = build_session(
                scan_id=scan_id,
                target=target,
                turn=iterations + 1,
                commands_run=commands_run,
                tokens_used=total_tokens,
                token_budget=SCAN_TOKEN_BUDGET,
                last_command=last_command,
                last_tool=last_tool,
                last_vector=last_vector,
                last_turn_success=last_turn_success,
                findings=findings,
                kg_consulted=last_turn_kg_consulted,
                tdi_log=harness.tdi_log,
                # Detected tech stack -> the 256-dim multi-hot block. Omitted
                # here it stays {} and 256 of 337 dims are zero forever.
                recon_signals=harness.recon_signals(),
                # Plain-data EGATS snapshot — NOT the live planner. The
                # session dict is plain data and must survive json.dumps;
                # passing `harness._egats` made it unserializable.
                egats_planner=harness.egats_snapshot(),
            )
            # is_done=True: no wasted-turn penalty and the completion bonus
            # applies (reward.py §4/§7).
            reward, breakdown = compute_reward(
                session=terminal_session,
                previous_verified_count=prev_verified_count,
                command_output="",
                tokens_used_delta=0,
                is_done=True,
                action_followed=None,
                credit_state=rl_credit_state,
            )
            breakdown["terminal"] = reason
            await harness.record_experience(
                session=terminal_session,
                turn=iterations,
                action_idx=ACTION_SPACE.index("pentest_complete"),
                action_name="pentest_complete",
                reward=reward,
                done=True,
                reward_breakdown=breakdown,
            )
            # NOTE: `prev_verified_count` is NOT advanced here. Both call
            # sites return immediately after this helper, so no further
            # reward is computed in this episode — advancing it would be
            # dead code (and would need `nonlocal`).
            logger.info(
                "rl_terminal_transition | scan=%s | turn=%d | reason=%s | "
                "reward=%.4f",
                scan_id, iterations, reason, reward,
            )
        except Exception as exc:
            logger.warning(
                "rl_terminal_transition_failed | scan=%s | reason=%s | err=%s",
                scan_id, reason, exc,
            )

    # ── P1: preflight LLM check ──────────────────────────────────
    await emit_scan_progress(scan_id, thought="Checking LLM config (OpenAI API key, model, base_url)...",
                             agent_name="orchestrator", progress=5)

    ok, err = await _preflight_llm_check(llm_config)
    if not ok:
        logger.error("LLM preflight failed | scan=%s | error=%s", scan_id, err)
        # R27: no terminal transition here on purpose. `harness` is still
        # None (created at the loop head) and no turn was ever recorded, so
        # there is no open episode to close — `_record_terminal_transition`
        # would return immediately. The three in-loop paths that DO leave an
        # episode open are handled below.
        await emit_scan_progress(scan_id, thought=f"❌ LLM preflight failed: {err}",
                                 agent_name="orchestrator", progress=5)
        return {
            "status": "error",
            "findings_count": 0,
            "iterations": 0,
            "total_tokens": 0,
            "final_summary": f"LLM preflight failed: {err}",
            "error": err,
            "rl_stats": await _finalize_rl(scan_status="error"),
        }

    logger.info("LLM preflight OK | scan=%s | model=%s",
                scan_id, llm_config.get("model"))
    await emit_scan_progress(scan_id, thought="✅ LLM config OK. Starting ReAct loop...",
                             agent_name="orchestrator", progress=10)

    # Build tool schemas — ALL tools, no allowlist
    tool_schemas = build_tool_schemas()

    # Create executor for this scan (if not passed in)
    if executor is None:
        executor = create_executor(target, scan_id)

    # Build initial messages
    user_message = f"Target: {target}\n\nRequest: {user_prompt}\n\nStart scanning this target."
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

    total_tokens = 0
    # Prompt tokens reported by the provider for the MOST RECENT completion.
    # This — not `total_tokens` — is what the TDA C dimension measures: it is
    # how full the model's context window is right now. `total_tokens` stays
    # the billing/scan-budget counter (TOKEN_SOFT_LIMIT / TOKEN_HARD_LIMIT).
    last_prompt_tokens = 0
    # Model context window (C denominator). Read from the live LLM channel
    # config (`max_total_tokens`, default 128K) so C is not scaled against
    # the 10M scan budget — that made C ~0.002 and effectively disabled it.
    try:
        model_context_window = int(llm_config.get("max_total_tokens") or 0) or MODEL_CONTEXT_WINDOW_DEFAULT
    except (TypeError, ValueError):
        model_context_window = MODEL_CONTEXT_WINDOW_DEFAULT
    findings_count = 0
    iterations = 0
    final_summary = ""
    budget_warned = False
    # Set once the premature-exit gate has challenged an `exit` call. The
    # second consecutive exit is always honoured (never hard-block).
    exit_challenged = False

    # ── Per-tool failure tracking (retry guard) ───────────────────
    tool_fail_counts: dict[str, int] = defaultdict(int)

    # ── RL harness: initialize now that preflight passed ──────────
    if rl_enabled:
        try:
            harness = get_harness(scan_id, target)
            await emit_scan_progress(
                scan_id,
                thought=f"RL harness active | epsilon={harness.epsilon:.3f} | buffer={harness.buffer_size}",
                agent_name="rl",
                progress=12,
            )
        except Exception as exc:
            logger.warning(
                "rl_harness_init_failed | scan=%s | err=%s | continuing without RL",
                scan_id, exc,
            )
            harness = None

    # ── KG attachment (Phase 2) ───────────────────────────────────
    # attach_kg() is a no-op when VAPT_AI_KG_ENABLED=false or when the
    # KG singleton can't be fetched (e.g. DB unavailable). The harness
    # gracefully degrades to "no KG" mode in either case.
    if harness is not None:
        try:
            harness.attach_kg()
        except Exception as exc:
            logger.warning(
                "kg_attach_failed_outer | scan=%s | err=%s | continuing without KG",
                scan_id, exc,
            )

    # ── Per-turn state locals for RL encoding ─────────────────────
    # These are updated each iteration and fed to build_session() so
    # StateEncoder can produce the 337-dim state vector.
    commands_run = 0
    last_command = ""
    # SE3: the tool that produced `last_command`, carried separately. Echoing
    # the tool NAME into `last_command` used to be the only way to say "this
    # ran", which conflated two things the encoder needs apart: `last_command`
    # is a shell string whose head is often `sudo`/`env`/`bash -c`, so
    # re-deriving the bucket from it silently fell back to "recon" (or, after
    # the R35 fix, to no signal at all). `""` means "no tool has run yet" and
    # is deliberately NOT the same as "unknown tool".
    last_tool = ""
    last_vector = ""
    last_turn_success = False
    # PER-TURN, not a latch. Read at build_session(kg_consulted=...) below,
    # which is why it is assigned LATE in the iteration: the rl_session built
    # at the top of turn N therefore carries turn N-1's value, exactly like
    # `last_turn_success`. A latched (sticky-True) flag would turn the
    # encoder's `kg_available` dimension into a constant from turn 2 onward.
    last_turn_kg_consulted = False
    findings: list[dict[str, Any]] = []
    prev_tokens = 0
    prev_verified_count = 0
    # Scan-lifetime reward bookkeeping (`_credited_verified_ids`,
    # `_seen_finding_keys`, `_penalised_fp_ids`). `build_session()` returns a
    # fresh dict every turn, so this state CANNOT live on the session — it
    # would reset each turn, re-crediting old findings and re-penalising
    # the same false positive forever. Owned here, passed as `credit_state=`
    # to every compute_reward() call. Never rebound, only mutated.
    rl_credit_state: dict[str, Any] = new_credit_state()
    # Cumulative tokens injected by the harness's own advisory messages
    # (RL / TDA / EGATS / KG brief / SC corrective). Estimated at
    # chars/4. Telemetry / rl_stats only — NOT a TDI input (see below).
    advisory_tokens_injected = 0
    # Advisory tokens present in the LAST request we sent. THIS is what
    # gets subtracted from `last_prompt_tokens`, because both describe the
    # same request. Subtracting the cumulative `advisory_tokens_injected`
    # instead is a unit mismatch: it grows every turn while a single
    # request's prompt tokens stay bounded by history pruning, so
    # `prompt_tokens - advisory_tokens` hits 0 after a few dozen turns and
    # C reads 0 for the rest of the scan (the opposite of the intent).
    advisory_tokens_last_turn = 0
    rl_action_rec: TurnRecommendation | None = None

    # ── Per-turn state locals for KG integration (Phase 2) ────────
    # kg_brief: the markdown brief from the last KG consult. Injected
    #          as a system message on the NEXT turn (after the recon
    #          that produced it has landed its tool result).
    # kg_auto_consulted: True after the first recon-tool output has
    #          been parsed and the KG consulted. Prevents re-consulting
    #          on every subsequent recon call.
    # kg_brief_injected: True after kg_brief has been injected once.
    #          We don't re-inject the same brief on every turn (token
    #          cost). The LLM can call consult_kg manually to refresh.
    kg_brief: str = ""
    kg_auto_consulted: bool = False
    kg_brief_injected: bool = False

    # ── Phase tracking: map tool name → pentest phase ─────────────
    # record_vulnerability is NOT mapped — it fires in ANY phase (the
    # agent records as soon as evidence exists, which is correct
    # behavior). Mapping it to 5 caused Phase 3→5→4 jumps in the UI.
    PHASE_TOOLS = {
        # Phase 1: Reconnaissance
        "nmap": 1, "httpx": 1, "whatweb": 1, "masscan": 1, "rustscan": 1,
        "subfinder": 1, "amass": 1, "dnsenum": 1, "fierce": 1,
        "theharvester": 1, "gau": 1, "waybackurls": 1,
        # Phase 2: Scanning & Enumeration
        "ffuf": 2, "gobuster": 2, "feroxbuster": 2, "katana": 2,
        "nikto": 2,
        # Phase 3: Vulnerability Analysis
        "nuclei": 3, "dalfox": 3, "wpscan": 3, "fscan": 3,
        # Phase 4: Exploitation
        "sqlmap": 4, "metasploit": 4, "hydra": 4,
        # Phase 5: Reporting (exit only — record_vulnerability is phase-agnostic)
        "exit": 5,
    }
    PHASE_NAMES = {
        1: "recon",
        2: "enumeration",
        3: "vulnerability-analysis",
        4: "exploitation",
        5: "reporting",
    }
    PHASE_PROGRESS = {1: 15, 2: 35, 3: 55, 4: 75, 5: 90}
    current_phase = 1  # set below via emit

    # Emit Phase 1 start
    await emit_phase_change(
        scan_id, phase="recon", progress=15,
        message="Phase 1: Reconnaissance — gathering target info",
        agent_name="orchestrator",
    )

    logger.info("ReAct scan started | scan=%s | target=%s | max_iter=%d | rl=%s",
                scan_id, target, max_iterations, rl_enabled)

    # ── ReAct loop ──────────────────────────────────────────────
    for iteration in range(max_iterations):
        iterations = iteration + 1
        loop_progress = _calc_progress(iterations, max_iterations)

        # ── Abort check ──────────────────────────────────────────
        # Honor the panic button BEFORE the next LLM call — without this,
        # aborting a scan only takes effect after max_iterations.
        from app.pentest.scan_registry import scan_registry
        if await scan_registry.is_aborted(scan_id):
            logger.warning(
                "ReAct scan aborting — scan_registry.abort_event is set | scan=%s | iter=%d",
                scan_id, iterations,
            )
            # R27: the panic button also ends the episode. Without a done=True
            # row the Q-target keeps bootstrapping off a state that never
            # existed, and `episode_reward` misses the closing turn.
            await _record_terminal_transition("aborted")
            return {
                "status": "aborted",
                "findings_count": findings_count,
                "iterations": iterations,
                "total_tokens": total_tokens,
                "final_summary": f"Scan aborted at iteration {iterations} (panic button).",
                "error": "user_panic_button",
                "rl_stats": await _finalize_rl(scan_status="aborted"),
            }

        try:
            # Call LLM
            await emit_scan_progress(scan_id, thought=f"Thinking... (iteration {iterations}/{max_iterations})",
                                     agent_name="orchestrator", progress=loop_progress)

            # ── RL: encode state + select action + format advisory hint ──
            rl_guidance_msg: dict[str, Any] | None = None
            # True only when the guidance text was actually shown to the
            # LLM this turn. Gates the "followed RL advice" reward: with no
            # advice in the prompt, following is coincidence and must not
            # earn a bonus (nor miss one).
            rl_guidance_injected = False
            rl_action_rec = None
            if harness is not None:
                try:
                    rl_session = build_session(
                        scan_id=scan_id,
                        target=target,
                        turn=iterations,
                        commands_run=commands_run,
                        tokens_used=total_tokens,
                        token_budget=SCAN_TOKEN_BUDGET,
                        last_command=last_command,
                        last_tool=last_tool,
                        last_vector=last_vector,
                        last_turn_success=last_turn_success,
                        findings=findings,
                        # Per-turn KG signal. Here it still holds the
                        # PREVIOUS turn's value (assigned late, like
                        # last_turn_success), which is what the encoder's
                        # "last action" features describe.
                        kg_consulted=last_turn_kg_consulted,
                        # Phase 3: feed TDI history into the RL state
                        # vector. The encoder reads tdi_log[-1] to
                        # populate the TDI feature (dim 1 of the 8
                        # scalars in the 337-dim vector).
                        tdi_log=harness.tdi_log if harness else [],
                        # Tech stack for the 256-dim multi-hot block. Without
                        # this the session's `recon_signals` was always {} and
                        # the entire tech-stack feature was dead.
                        recon_signals=harness.recon_signals() if harness else {},
                        # Phase 4: planner path counts (2 of the 9 flag dims).
                        # Plain data from EGATS.to_dict(); the live planner
                        # object would make the session dict unserializable.
                        egats_planner=harness.egats_snapshot() if harness else {},
                    )
                    rl_action_rec = await harness.select_action(rl_session, iterations)
                    # Shadow mode — suppress guidance while the policy is
                    # still random (see _should_inject_rl_guidance).
                    inject_guidance, suppression_reason = _should_inject_rl_guidance(
                        rl_action_rec, train_steps=harness.q_train_steps,
                    )
                    if inject_guidance:
                        rl_guidance_msg = {
                            "role": "system",
                            "content": rl_action_rec.guidance_text,
                        }
                        rl_guidance_injected = True
                    else:
                        logger.debug(
                            "rl_guidance_suppressed | scan=%s | turn=%d | "
                            "reason=%s | action=%s | epsilon=%.3f | train_steps=%d",
                            scan_id, iterations, suppression_reason,
                            rl_action_rec.action_name, rl_action_rec.epsilon,
                            harness.q_train_steps,
                        )
                    await emit_scan_progress(
                        scan_id,
                        thought=f"RL: {rl_action_rec.action_name} "
                                f"(epsilon={rl_action_rec.epsilon:.3f}, "
                                f"exploratory={rl_action_rec.exploratory})"
                                + ("" if inject_guidance
                                   else f" [guidance suppressed: {suppression_reason}]"),
                        agent_name="rl",
                        progress=loop_progress,
                    )
                except Exception as exc:
                    logger.warning(
                        "rl_select_failed | scan=%s | turn=%d | err=%s",
                        scan_id, iterations, exc,
                    )
                    rl_action_rec = None
                    rl_guidance_msg = None

            # ── TDA: compute TDI for this turn + render mode hint ────
            # Phase 3 — runs AFTER select_action (so the RL state vector
            # for THIS turn doesn't include TDI yet — that's OK, the TDI
            # feeds into the NEXT turn's state via tdi_log). The hint is
            # injected as a system message alongside RL guidance + KG brief.
            # `advisory_tokens_injected` keeps the harness's own hint text
            # out of the context-load dimension (C) — otherwise a hint
            # emitted because difficulty is high would raise C, raise the
            # TDI, and re-trigger BROADEN (self-reinforcing loop).
            tda_hint_msg: dict[str, Any] | None = None
            if harness is not None:
                try:
                    tdi_result = harness.tda_compute(
                        turn=iterations,
                        max_turns=max_iterations,
                        findings=findings,
                        # C = context SATURATION of the latest request, not
                        # cumulative spend. `tokens_used` is kept for the
                        # rl_stats log only.
                        prompt_tokens=last_prompt_tokens,
                        model_context_window=model_context_window,
                        tokens_used=total_tokens,
                        # S must use the SAME evidence definition as the
                        # EXPLOIT gate, otherwise a replay-verified finding
                        # locks the mode while S still reports 'no success'.
                        # It is literally the same call:
                        # `evidence_score_from_finding(f) >= CONFIRMED_VULN`
                        # inline here was a FOURTH definition of "verified"
                        # and the only evidence-based one — the encoder and
                        # reward used the legacy boolean, so an
                        # action could be rewarded for a finding the TDI
                        # ladder did not see (or the reverse).
                        successes=count_verified_findings(findings),
                        attempts=commands_run,
                        # Must be the PREVIOUS turn's per-turn advisory load
                        # (the request that produced `last_prompt_tokens`).
                        advisory_tokens=advisory_tokens_last_turn,
                    )
                    if tdi_result is not None:
                        tda_hint_msg = {
                            "role": "system",
                            "content": harness.tda_render_hint(tdi_result),
                        }
                        await emit_scan_progress(
                            scan_id,
                            thought=f"TDA: {tdi_result.mode.value} "
                                    f"(TDI={tdi_result.tdi:.3f}"
                                    f"{', overridden' if tdi_result.overridden else ''})",
                            agent_name="tda",
                            progress=loop_progress,
                        )
                except Exception as exc:
                    logger.warning(
                        "tda_compute_failed | scan=%s | turn=%d | err=%s",
                        scan_id, iterations, exc,
                    )
                    tda_hint_msg = None

            # ── EGATS: select highest-UCB path + render hint ────────
            # Phase 4 — runs AFTER TDA compute (so EGATS can use the
            # current TDI for UCB penalty). The hint is injected as a
            # separate system message (advisory — your judgment wins).
            #
            # NOTE: selection happens EVERY turn but the round-robin below
            # only injects the EGATS message ~1 turn in 3. Selection alone
            # therefore says nothing about what the LLM saw, so the SSE
            # announcement and the outcome pairing both wait for the
            # rotation to actually pick EGATS (see the injection block).
            egats_hint_msg: dict[str, Any] | None = None
            egats_result: Any = None
            if harness is not None:
                try:
                    egats_result = harness.egats_select()
                    if egats_result is not None and egats_result.path is not None:
                        hint_text = harness.egats_render_hint(egats_result)
                        if hint_text:
                            egats_hint_msg = {
                                "role": "system",
                                "content": hint_text,
                            }
                except Exception as exc:
                    logger.warning(
                        "egats_select_failed | scan=%s | turn=%d | err=%s",
                        scan_id, iterations, exc,
                    )
                    egats_hint_msg = None
                    egats_result = None

            # ── Self-Correction: consume pending corrective message ──
            # Phase 5 — if the previous turn had a rejected finding,
            # harness.process_rejection() stashed a corrective message.
            # Pop it here and inject as a system message (advisory).
            # The message is consumed exactly once (FIFO — if multiple
            # rejections happened, only the latest is injected; the
            # pattern count still escalates in the engine).
            sc_corrective_msg: dict[str, Any] | None = None
            if harness is not None:
                try:
                    corrective_text = harness.sc_consume_corrective_message()
                    if corrective_text:
                        sc_corrective_msg = {
                            "role": "system",
                            "content": corrective_text,
                        }
                        await emit_scan_progress(
                            scan_id,
                            thought=f"Self-Correction: injecting corrective feedback ({len(corrective_text)} chars)",
                            agent_name="sc",
                            progress=loop_progress,
                        )
                        logger.info(
                            "sc_corrective_injected | scan=%s | turn=%d | msg_len=%d",
                            scan_id, iterations, len(corrective_text),
                        )
                except Exception as exc:
                    logger.warning(
                        "sc_consume_failed | scan=%s | turn=%d | err=%s",
                        scan_id, iterations, exc,
                    )

            # ── Round-robin advisory injection (1 msg/turn) ────────────
            # QUALITY FIX: Injecting all 5 advisory messages (RL + TDA +
            # EGATS + KG + SC) every turn causes attention dilution — the
            # LLM loses focus on actual vuln detection and starts missing
            # findings (regression observed: 5 findings → 3, SQLi missed).
            #
            # Round-robin: each turn, inject ONLY ONE advisory (cycling
            # through RL → TDA → EGATS; see the rotation block below).
            # SC corrective message is ALWAYS injected (it's high-priority
            # feedback on a rejection — too important to delay). KG brief
            # is injected once (first time only), not repeated.
            #
            # This keeps intelligence layer active while preserving LLM
            # attention budget for actual scan reasoning.
            extra_msgs: list[dict[str, Any]] = []

            # SC corrective message — always inject (high priority).
            if sc_corrective_msg:
                extra_msgs.append(sc_corrective_msg)

            # Round-robin: pick ONE advisory this turn.
            # Rotation: RL → TDA → EGATS, repeating. KG is NOT part of the
            # rotation — its brief is injected once, the first time it
            # becomes available (see below). If the scheduled advisor has
            # nothing to say, `_select_advisory` falls through to the next
            # one instead of burning the turn's slot (the old if/elif/elif
            # chain injected nothing, which starved the early turns before
            # EGATS was seeded).
            picked_advisory, advisory_msg = _select_advisory(
                iterations,
                rl_msg=rl_guidance_msg,
                tda_msg=tda_hint_msg,
                egats_msg=egats_hint_msg,
            )
            if advisory_msg is not None:
                extra_msgs.append(advisory_msg)
                logger.debug(
                    "advisory_round_robin | scan=%s | turn=%d | picked=%s",
                    scan_id, iterations, picked_advisory,
                )
                # Identity check against the dict `_select_advisory`
                # returned — no string coupling to ADVISORY_ROTATION.
                if (
                    harness is not None
                    and egats_hint_msg is not None
                    and advisory_msg is egats_hint_msg
                ):
                    # The LLM is about to see this path: arm it as the ONLY
                    # eligible outcome target for this turn. Crediting the
                    # merely-selected path let EGATS learn from hints the
                    # model never received (round-robin injects 1 turn in 3).
                    try:
                        harness.egats_mark_hint_injected()
                    except Exception as exc:
                        logger.debug(
                            "egats_mark_hint_injected_failed | scan=%s | "
                            "turn=%d | err=%s",
                            scan_id, iterations, exc,
                        )
                    if egats_result is not None and egats_result.path is not None:
                        await emit_scan_progress(
                            scan_id,
                            thought=(
                                f"EGATS: path {' → '.join(egats_result.path.nodes[-2:])} "
                                f"(UCB={egats_result.ucb_score:.3f}, "
                                f"{egats_result.path.successes}/{egats_result.path.attempts})"
                            ),
                            agent_name="egats",
                            progress=loop_progress,
                        )
            else:
                logger.debug(
                    "advisory_round_robin | scan=%s | turn=%d | no advisory "
                    "available (rl=%s tda=%s egats=%s)",
                    scan_id, iterations,
                    bool(rl_guidance_msg), bool(tda_hint_msg),
                    bool(egats_hint_msg),
                )

            # KG brief: inject once after the first recon produces it.
            # We mark kg_brief_injected=True so we don't keep re-injecting
            # the same brief on every subsequent turn (token cost). The
            # LLM can call consult_kg manually to refresh.
            if kg_brief and not kg_brief_injected:
                extra_msgs.append({"role": "system", "content": kg_brief})
                kg_brief_injected = True
                logger.info(
                    "kg_brief_injected | scan=%s | turn=%d | brief_len=%d",
                    scan_id, iterations, len(kg_brief),
                )
                await emit_scan_progress(
                    scan_id,
                    thought=f"KG brief injected ({len(kg_brief)} chars)",
                    agent_name="kg",
                    progress=loop_progress,
                )
            turn_messages = messages + extra_msgs

            # Record the advisory overhead injected THIS turn. It is
            # excluded from the next turn's TDI context numerator (see
            # tda_compute(advisory_tokens=...)). chars/4 is an estimate —
            # no tokenizer is invoked just for telemetry.
            turn_advisory_tokens = sum(
                len(m.get("content") or "") for m in extra_msgs
            ) // 4
            advisory_tokens_injected += turn_advisory_tokens
            # Per-turn value pairs with `last_prompt_tokens` below: both
            # describe the SAME request, so the next turn's C subtraction
            # cancels exactly the harness's own contribution.
            advisory_tokens_last_turn = turn_advisory_tokens

            response = await chat_completion(
                llm_config=llm_config,
                messages=turn_messages,
                tools=tool_schemas,
            )

            # Defensive usage parsing (usage may be None/missing)
            _usage = response.get("usage") or {}
            total_tokens += _usage.get("total_tokens", 0) or 0
            # Feed the NEXT turn's C dimension. Providers that omit usage
            # (some local/proxy gateways) get a chars/4 estimate over what
            # we actually sent, so C degrades gracefully instead of
            # silently reporting an empty context.
            _reported_prompt = _usage.get("prompt_tokens") or 0
            if _reported_prompt:
                last_prompt_tokens = int(_reported_prompt)
            else:
                last_prompt_tokens = sum(
                    len(m.get("content") or "") for m in turn_messages
                ) // 4

            # ── Token budget enforcement ─────────────────────────
            if total_tokens > TOKEN_SOFT_LIMIT and not budget_warned:
                budget_warned = True
                await emit_scan_progress(
                    scan_id,
                    thought=f"⚠️ Token budget warning: {total_tokens:,} tokens used (soft limit {TOKEN_SOFT_LIMIT:,}). Wrap up efficiently.",
                    agent_name="orchestrator", progress=loop_progress,
                )
            if total_tokens > TOKEN_HARD_LIMIT:
                logger.warning("Token hard limit exceeded | scan=%s | tokens=%d", scan_id, total_tokens)
                await emit_scan_progress(
                    scan_id,
                    thought=f"🛑 Token budget exceeded ({total_tokens:,}). Scan cancelled.",
                    agent_name="orchestrator", progress=95,
                )
                # R27: budget exhaustion ends the episode too — same reason.
                await _record_terminal_transition("token_budget_exceeded")
                return {
                    "status": "error",
                    "findings_count": findings_count,
                    "iterations": iterations,
                    "total_tokens": total_tokens,
                    "final_summary": f"Token budget exceeded ({total_tokens:,} tokens). Scan cancelled.",
                    "error": "token_budget_exceeded",
                    "rl_stats": await _finalize_rl(scan_status="token_budget_exceeded"),
                }

            # Append assistant message to history
            assistant_msg: dict[str, Any] = {"role": "assistant", "content": response["content"]}
            if response["tool_calls"]:
                assistant_msg["tool_calls"] = [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": tc["arguments"],
                        },
                    }
                    for tc in response["tool_calls"]
                ]
            messages.append(assistant_msg)

            # If LLM is thinking (content but no tool_calls), continue to next iteration
            if not response["tool_calls"]:
                if response["content"]:
                    logger.info("LLM thinking (no tool calls) | scan=%s | iter=%d | content=%s",
                                scan_id, iterations, response["content"][:100])
                    await emit_scan_progress(scan_id, thought=response["content"][:200],
                                             agent_name="orchestrator", progress=loop_progress)
                continue

            # ── Execute tool calls ───────────────────────────────
            # Marker: tool messages appended from here on belong to THIS
            # turn. The RL block below slices them back out to synthesize
            # D26 turn events — scanning all of `messages` instead would
            # re-fire a timeout several turns after it happened.
            turn_tool_msgs_start = len(messages)
            # New turn's tool dispatch: clear the per-turn KG signal so a
            # consult from an earlier turn cannot leak forward (an accidental
            # latch). Deliberately reset HERE and not at the top of the loop:
            # `rl_session` is built before this line, so it still sees the
            # value of the turn that last dispatched tools — the same
            # "last RECORDED turn" convention as `prev_verified_count`.
            last_turn_kg_consulted = False
            for tc in response["tool_calls"]:
                tool_name = tc["name"]

                # ── Surface malformed JSON args to the LLM ────────
                try:
                    tool_args = json.loads(tc["arguments"]) if tc["arguments"] else {}
                except json.JSONDecodeError as e:
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": (
                            f"ERROR: your arguments for '{tool_name}' were not valid JSON ({e}). "
                            f"Raw arguments received: {str(tc['arguments'])[:200]}. "
                            f"Retry with correctly formatted JSON arguments."
                        ),
                    })
                    await emit_scan_progress(
                        scan_id,
                        thought=f"⚠️ Malformed arguments for {tool_name} — asking LLM to retry with valid JSON",
                        agent_name="orchestrator", tool_name=tool_name,
                        progress=loop_progress,
                    )
                    continue

                # ── Hard-block tools that keep failing ────────────
                if (tool_name not in ("record_vulnerability", "exit")
                        and tool_fail_counts[tool_name] >= MAX_TOOL_FAILURES):
                    blocked_msg = (
                        f"🚫 Tool '{tool_name}' is BLOCKED — it already failed "
                        f"{tool_fail_counts[tool_name]} times in this scan (limit {MAX_TOOL_FAILURES}). "
                        f"Retrying will be rejected. Switch to a different tool or move to the next step."
                    )
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": blocked_msg,
                    })
                    await emit_scan_progress(
                        scan_id,
                        thought=f"🚫 Blocked repeat-failed tool: {tool_name}",
                        agent_name="orchestrator", tool_name=tool_name,
                        progress=loop_progress,
                    )
                    continue

                # ── Phase tracking: MONOTONIC — only increase ──────
                # Once we enter Phase N, we never go back to Phase < N.
                # Prevents "Phase 3 → Phase 1" jumps when the LLM calls
                # httpx (a Phase 1 tool) during Phase 3 to verify endpoints.
                tool_phase = PHASE_TOOLS.get(tool_name, 0)
                if tool_phase > current_phase:
                    # ── Phase 3 skip detection (flexible) ─────────────
                    # If LLM tries to jump from Phase 1/2 directly to Phase 4
                    # (skipping Phase 3 vuln scanning), check whether the LLM
                    # has already gathered parameter evidence.
                    #
                    # Phase 3 skip is ALLOWED when:
                    #   - The current tool call targets a URL with a query
                    #     parameter (e.g. /page.php?id=1) — this means the
                    #     LLM has identified a specific injectable parameter
                    #     during Phase 2 enumeration.
                    #   - OR the tool_args contain a "cookie" or "data" field
                    #     (indicating the LLM is doing authenticated/form-based
                    #     exploitation with a specific target in mind).
                    #
                    # Phase 3 skip is NOT allowed (emit warning) when:
                    #   - The LLM calls sqlmap/hydra/metasploit on a bare URL
                    #     without query parameters — this is blind exploitation.
                    #   - current_phase < 2 (haven't even enumerated yet).
                    #
                    # This is advisory only — we never hard-block.
                    if (
                        tool_phase >= 4
                        and current_phase < 3
                        and tool_name not in ("record_vulnerability", "exit")
                    ):
                        # Check if LLM has parameter evidence.
                        target_url = str(tool_args.get("target") or tool_args.get("url") or "")
                        has_param_evidence = (
                            "?" in target_url
                            and "=" in target_url.split("?", 1)[1]
                        ) or (
                            "data" in tool_args
                            or "cookie" in tool_args
                            or "form_data" in tool_args
                        )

                        if has_param_evidence:
                            # Legitimate skip — LLM has a specific parameter to test.
                            logger.info(
                                "phase_skip_with_evidence | scan=%s | from=%d to=%d | "
                                "tool=%s | target=%s | param_evidence=True",
                                scan_id, current_phase, tool_phase, tool_name,
                                target_url[:80],
                            )
                        else:
                            # Suspicious skip — no parameter evidence, likely blind exploitation.
                            skipped = list(range(current_phase + 1, tool_phase))
                            logger.warning(
                                "phase_skip_no_evidence | scan=%s | from=%d to=%d | "
                                "skipped=%s | tool=%s | target=%s | "
                                "LLM is skipping Phase 3 without parameter evidence — "
                                "blind exploitation likely to fail",
                                scan_id, current_phase, tool_phase, skipped,
                                tool_name, target_url[:80],
                            )
                            await emit_scan_progress(
                                scan_id,
                                thought=(
                                    f"⚠️ Phase skip warning: {tool_name} is Phase {tool_phase} "
                                    f"but Phase 3 (vuln scanning) was not run AND no parameter "
                                    f"evidence detected (target has no query string). "
                                    f"Consider running nuclei/nikto/dalfox first to identify "
                                    f"the correct vulnerability class. If you have specific "
                                    f"parameter evidence, proceed — otherwise this is blind "
                                    f"exploitation and may waste turns."
                                ),
                                agent_name="orchestrator",
                                progress=loop_progress,
                            )

                    current_phase = tool_phase
                    phase_name = PHASE_NAMES.get(tool_phase, "unknown")
                    phase_progress = PHASE_PROGRESS.get(tool_phase, 50)
                    await emit_phase_change(
                        scan_id, phase=phase_name, progress=phase_progress,
                        message=f"Phase {tool_phase}: {phase_name.replace('-', ' ').title()}",
                        agent_name="orchestrator",
                    )
                    logger.info("Phase change | scan=%s | phase=%d (%s) | tool=%s",
                                scan_id, tool_phase, phase_name, tool_name)

                # Emit SSE: tool_call
                await emit_scan_progress(scan_id, thought=f"🔧 Running: {tool_name}({json.dumps(tool_args, ensure_ascii=False)[:100]})",
                                         agent_name="orchestrator", tool_name=tool_name,
                                         progress=loop_progress)

                logger.info("Tool call | scan=%s | iter=%d | tool=%s | args=%s",
                            scan_id, iterations, tool_name,
                            json.dumps(tool_args, ensure_ascii=False)[:80])

                # Check for exit tool
                if tool_name == "exit":
                    # ── Premature-exit SOFT gate ────────────────────────
                    # MIN_AGENT_TURNS / MIN_COMMANDS_FOR_COMPLETE are 0, so
                    # previously NOTHING stopped an `exit` at iteration 7:
                    # the only guard was prompt discipline. Re-enabling the
                    # old HARD gate would reintroduce the bug it was removed
                    # for (legitimate short scans could never finish), so
                    # this gate is SOFT — challenge once, then comply.
                    #
                    # Triggers only when the scan is BOTH short on turns and
                    # short on commands AND has recorded no finding at all.
                    # The second `exit` is always honoured, so the worst case
                    # costs one extra turn and cannot loop.
                    if _exit_needs_challenge(
                        exit_challenged=exit_challenged,
                        findings_count=len(findings),
                        commands_run=commands_run,
                        iteration=iterations,
                    ):
                        exit_challenged = True
                        challenge = (
                            f"EXIT NOT ACCEPTED YET — this looks premature. "
                            f"You have run {commands_run} tool commands over "
                            f"{iterations} turns and recorded 0 findings.\n"
                            f"Checklist before finishing:\n"
                            f"  1. Ports/services identified (nmap)?\n"
                            f"  2. Web stack fingerprinted (httpx/whatweb)?\n"
                            f"  3. Hidden paths brute-forced (ffuf/feroxbuster)?\n"
                            f"  4. Vulnerability scan run (nuclei/nikto/dalfox)?\n"
                            f"  5. Every candidate recorded with record_vulnerability?\n"
                            f"If a step genuinely does not apply, say so in your "
                            f"summary and call `exit` AGAIN — the second exit is "
                            f"always accepted. Otherwise continue testing."
                        )
                        logger.info(
                            "exit_soft_gate_challenged | scan=%s | iter=%d | "
                            "commands=%d/%d | turns=%d/%d | findings=0",
                            scan_id, iterations, commands_run,
                            EXIT_SOFT_GATE_MIN_COMMANDS, iterations,
                            EXIT_SOFT_GATE_MIN_TURNS,
                        )
                        await emit_scan_progress(
                            scan_id,
                            thought=(
                                f"⚠️ Exit challenged — only {commands_run} commands "
                                f"and 0 findings so far. Asking the agent to confirm."
                            ),
                            agent_name="orchestrator", tool_name="exit",
                            progress=loop_progress,
                        )
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tc["id"],
                            "content": challenge,
                        })
                        continue

                    final_summary = tool_args.get("summary", "Scan complete.")
                    logger.info("Scan exit | scan=%s | iter=%d | summary=%s",
                                scan_id, iterations, final_summary[:100])
                    await emit_scan_progress(scan_id, thought=f"✅ Done: {final_summary[:150]}",
                                             agent_name="orchestrator", progress=95)

                    # Terminal transition — the episode ends HERE. Recorded
                    # before _finalize_rl() stops the training worker, and
                    # before the return so the DB count below can't disturb it.
                    await _record_terminal_transition("exit")

                    # Count findings from DB (source of truth — replaces the
                    # in-loop counter which could drift from rejections)
                    try:
                        from app.db.session import async_session
                        from app.db.models.pentest import Finding
                        from sqlalchemy import select, func
                        async with async_session() as session:
                            count = await session.scalar(
                                select(func.count(Finding.id)).where(Finding.scan_id == scan_id)
                            ) or 0
                            findings_count = count
                    except Exception:
                        pass

                    return {
                        "status": "completed",
                        "findings_count": findings_count,
                        "iterations": iterations,
                        "total_tokens": total_tokens,
                        "final_summary": final_summary,
                        "error": None,
                        "rl_stats": await _finalize_rl(scan_status="completed"),
                    }

                # ── consult_kg tool (Phase 2) ───────────────────────
                # The LLM can manually query the KG mid-scan with a focus
                # vector. Returns the advisory brief as the tool result so
                # the LLM reads it as part of the conversation.
                if tool_name == "consult_kg":
                    focus_vector = tool_args.get("focus_vector") or tool_args.get("focus")
                    logger.info(
                        "consult_kg tool | scan=%s | iter=%d | focus=%s",
                        scan_id, iterations, focus_vector or "(none)",
                    )
                    if harness is None:
                        tool_output = (
                            "KG not available — RL harness is disabled. "
                            "Continue scanning without KG hints."
                        )
                    else:
                        try:
                            consult_id, brief = harness.kg_consult(focus=focus_vector)
                            # The KG was queried in THIS turn — that is what
                            # build_session(kg_consulted=...) reports. An
                            # empty brief still counts: the query ran, and
                            # the policy needs "was KG consulted" to differ
                            # from "was a tool called at all".
                            last_turn_kg_consulted = True
                            if brief:
                                tool_output = brief
                                # Mark this brief for injection on the next
                                # turn too (so the LLM sees it as a system
                                # message, not just a tool result).
                                kg_brief = brief
                                kg_brief_injected = False  # re-inject on next turn
                                await emit_scan_progress(
                                    scan_id,
                                    thought=f"KG consulted (focus={focus_vector or 'none'}) — {len(brief)} chars brief",
                                    agent_name="kg",
                                    progress=loop_progress,
                                )
                            else:
                                tool_output = (
                                    "KG consulted but returned no paths for "
                                    f"focus='{focus_vector or '(none)'}'. The KG "
                                    "may not have data for this stack yet. "
                                    "Continue with your own reconnaissance."
                                )
                        except Exception as exc:
                            tool_output = f"KG consult failed: {exc}"
                            logger.warning(
                                "consult_kg_failed | scan=%s | err=%s",
                                scan_id, exc,
                            )

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": tool_output,
                    })
                    continue

                # Execute the tool
                try:
                    tool_output = await execute_tool_call(
                        tool_name=tool_name,
                        tool_args=tool_args,
                        target=target,
                        scan_id=scan_id,
                        executor=executor,
                    )

                    # Detect "Binary not found" inside the returned
                    # tool_output → inject install hint so the LLM can recover.
                    if "Binary not found" in tool_output:
                        tool_output = _format_binary_not_found_hint(tool_output)

                    if tool_name == "record_vulnerability":
                        # Only count SUCCESSFULLY recorded findings —
                        # _record_vulnerability may return status "rejected"
                        # (e.g. missing evidence); those are NOT findings.
                        counted = False
                        finding_id_str = ""
                        result_json: dict[str, Any] = {}
                        try:
                            result_json = json.loads(tool_output)
                            counted = result_json.get("status") == "recorded"
                            finding_id_str = str(result_json.get("finding_id") or result_json.get("id") or "")
                        except (json.JSONDecodeError, AttributeError):
                            counted = False
                        if counted:
                            findings_count += 1
                            # ── RL: track the finding for state encoding ──
                            # Shape comes from app.evidence.finding_state —
                            # the single producer for the in-memory finding
                            # dict shared with the TDA ladder and the
                            # reward shaper. Do not hand-build it here:
                            # this block used to omit `location` (and write
                            # `title` while the reward side read `name`),
                            # collapsing every finding of a severity onto
                            # one payout identity.
                            finding_state = finding_to_state(
                                tool_args, result_json
                            )
                            findings.append(finding_state)
                            await emit_scan_progress(
                                scan_id,
                                thought=f"Finding recorded: {finding_state['title'][:60]} "
                                        f"[verification={finding_state['verification']}]",
                                agent_name="evidence",
                                progress=loop_progress,
                            )
                            # ── KG: record consult hit (Phase 2) ──
                            # The finding was confirmed — credit the last
                            # KG consult (if any) so the KG edge probability
                            # goes up for this attack path.
                            if harness is not None:
                                try:
                                    harness.kg_record_finding_outcome(
                                        finding_id=finding_id_str or f"scan-{scan_id}-#{findings_count}",
                                        success=True,
                                    )
                                except Exception as exc:
                                    logger.debug(
                                        "kg_record_hit_failed | scan=%s | err=%s",
                                        scan_id, exc,
                                    )
                                # ── Phase 4: EGATS record outcome (success) ──
                                # The selected path led to a confirmed
                                # finding — backprop success + update TDI EMA.
                                try:
                                    current_tdi = (
                                        harness.tdi_log[-1]["tdi"]
                                        if harness.tdi_log else 0.5
                                    )
                                    harness.egats_record_outcome(
                                        success=True, tdi=current_tdi,
                                    )
                                except Exception as exc:
                                    logger.debug(
                                        "egats_record_hit_failed | scan=%s | err=%s",
                                        scan_id, exc,
                                    )
                            await emit_scan_progress(
                                scan_id, thought=f"📋 Recorded vulnerability #{findings_count}",
                                agent_name="orchestrator", tool_name="record_vulnerability",
                                progress=loop_progress)
                        else:
                            # ── KG: record consult miss (Phase 2) ──
                            # The finding was rejected — the KG consult
                            # (if any) led to a dead end. Record the miss
                            # so the KG edge probability goes down.
                            if harness is not None:
                                try:
                                    harness.kg_record_finding_outcome(
                                        finding_id="",
                                        success=False,
                                        reason="record_vulnerability rejected (missing/invalid evidence)",
                                    )
                                except Exception as exc:
                                    logger.debug(
                                        "kg_record_miss_failed | scan=%s | err=%s",
                                        scan_id, exc,
                                    )
                                # ── Phase 4: EGATS record outcome (fail) ──
                                # The selected path was a dead end —
                                # backprop failure + update TDI EMA.
                                try:
                                    current_tdi = (
                                        harness.tdi_log[-1]["tdi"]
                                        if harness.tdi_log else 0.5
                                    )
                                    harness.egats_record_outcome(
                                        success=False, tdi=current_tdi,
                                    )
                                except Exception as exc:
                                    logger.debug(
                                        "egats_record_miss_failed | scan=%s | err=%s",
                                        scan_id, exc,
                                    )
                                # ── Phase 5: Self-Correction on rejection ──
                                # Process the rejection: compute pattern
                                # signature, increment pattern count,
                                # render 3-tier corrective message, push
                                # escalating RL reward to PER. The
                                # corrective message will be consumed
                                # + injected on the NEXT turn.
                                try:
                                    rejection_reason = "rejected"
                                    try:
                                        result_json = json.loads(tool_output)
                                        rejection_reason = str(
                                            result_json.get("reason") or "rejected"
                                        )
                                    except (json.JSONDecodeError, AttributeError):
                                        pass
                                    sc_result = await harness.process_rejection(
                                        finding=tool_args,
                                        reason=rejection_reason,
                                    )
                                    if sc_result is not None:
                                        await emit_scan_progress(
                                            scan_id,
                                            thought=(
                                                f"Self-Correction: rejection #{sc_result['pattern_count']} "
                                                f"for pattern {sc_result['pattern_signature'][:8]}... "
                                                f"(tier {sc_result['tier']}, reward {sc_result['rl_reward']:.1f})"
                                            ),
                                            agent_name="sc",
                                            progress=loop_progress,
                                        )
                                except Exception as exc:
                                    logger.debug(
                                        "sc_process_rejection_failed | scan=%s | err=%s",
                                        scan_id, exc,
                                    )
                            await emit_scan_progress(
                                scan_id, thought=f"⚠️ record_vulnerability rejected (missing/invalid evidence)",
                                agent_name="orchestrator", tool_name="record_vulnerability",
                                progress=loop_progress)
                    else:
                        # Emit observation (truncated for SSE)
                        obs_preview = tool_output[:200].replace("\n", " ")
                        await emit_scan_progress(scan_id, thought=f"📤 Result {tool_name}: {obs_preview}",
                                                 agent_name="orchestrator", tool_name=tool_name,
                                                 progress=loop_progress)

                        # Track failures for the retry guard
                        if _looks_like_failure(tool_output):
                            tool_fail_counts[tool_name] += 1
                            if tool_fail_counts[tool_name] >= MAX_TOOL_FAILURES:
                                # Append the block notice to THIS result too,
                                # so the LLM sees it in the same context.
                                tool_output += (
                                    f"\n\n🚫 NOTE: '{tool_name}' has now failed "
                                    f"{tool_fail_counts[tool_name]} times. It will be "
                                    f"BLOCKED on the next attempt. Switch tools."
                                )
                        else:
                            tool_fail_counts[tool_name] = 0  # success resets

                        # ── KG: auto-consult after first recon (Phase 2) ──
                        # After the first recon tool produces output, parse
                        # it for tech-stack tokens, consult the KG, and
                        # stash the brief for injection on the next turn.
                        # Only fires once per scan (kg_auto_consulted guard).
                        if (
                            harness is not None
                            and not kg_auto_consulted
                            and tool_name in _RECON_TOOLS
                            and tool_output
                            and not _looks_like_failure(tool_output)
                        ):
                            try:
                                tech_stack = extract_tech_stack(
                                    last_command=tool_name,
                                    last_tool_output=tool_output,
                                )
                                if tech_stack:
                                    harness.set_tech_stack(tech_stack)
                                    _consult_id, brief = harness.kg_consult()
                                    # Auto-consult is a KG query too — miss it
                                    # here and the feature only reports the
                                    # LLM's manual calls.
                                    last_turn_kg_consulted = True
                                    if brief:
                                        kg_brief = brief
                                        kg_brief_injected = False
                                        logger.info(
                                            "kg_auto_consult | scan=%s | turn=%d | "
                                            "stack=%s | brief_len=%d",
                                            scan_id, iterations,
                                            tech_stack, len(brief),
                                        )
                                        await emit_scan_progress(
                                            scan_id,
                                            thought=f"KG auto-consulted (stack={', '.join(tech_stack)}) — {len(brief)} chars brief ready",
                                            agent_name="kg",
                                            progress=loop_progress,
                                        )
                                        # ── Phase 4: seed EGATS from the tech stack ──
                                        # EGATS pulls attack paths from the KG
                                        # using the Technology node IDs for the
                                        # detected stack. Each turn, egats_select()
                                        # will return the highest-UCB path.
                                        try:
                                            from app.kg.types import NodeType
                                            # Public accessor — see
                                            # HarnessBridge.kg. Reaching into
                                            # `harness._kg` from here coupled
                                            # the agent loop to a private
                                            # attribute name.
                                            kg_instance = harness.kg
                                            if kg_instance is not None:
                                                node_ids = []
                                                for tech in tech_stack:
                                                    nid = kg_instance.find_node(
                                                        NodeType.TECHNOLOGY, tech,
                                                    )
                                                    if nid:
                                                        node_ids.append(nid)
                                                if node_ids:
                                                    seeded = harness.egats_seed(
                                                        kg_instance, node_ids,
                                                    )
                                                    if seeded > 0:
                                                        await emit_scan_progress(
                                                            scan_id,
                                                            thought=f"EGATS seeded {seeded} attack paths from KG",
                                                            agent_name="egats",
                                                            progress=loop_progress,
                                                        )
                                        except Exception as exc:
                                            logger.warning(
                                                "egats_seed_failed | scan=%s | turn=%d | err=%s",
                                                scan_id, iterations, exc,
                                            )
                                    else:
                                        logger.info(
                                            "kg_auto_consult_empty | scan=%s | turn=%d | "
                                            "stack=%s | no paths found",
                                            scan_id, iterations, tech_stack,
                                        )
                            except Exception as exc:
                                logger.warning(
                                    "kg_auto_consult_failed | scan=%s | turn=%d | err=%s",
                                    scan_id, iterations, exc,
                                )
                            finally:
                                # Mark as consulted even on failure — we
                                # don't want to retry the auto-consult on
                                # every subsequent recon call.
                                kg_auto_consulted = True

                except Exception as exc:
                    tool_output = f"Error executing {tool_name}: {exc}"
                    err_str = str(exc)
                    if "Binary not found" in err_str or "FileNotFoundError" in err_str:
                        tool_output = _format_binary_not_found_hint(err_str)
                    tool_fail_counts[tool_name] += 1
                    logger.error("Tool execution error | scan=%s | tool=%s | error=%s",
                                 scan_id, tool_name, exc)
                    await emit_scan_progress(scan_id, thought=f"❌ Error {tool_name}: {str(exc)[:100]}",
                                             agent_name="orchestrator", tool_name=tool_name,
                                             progress=loop_progress)

                # Append tool result to messages
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": tool_output,
                })

            # ── RL: record experience + compute reward + signal training ──
            # Fires after ALL tool calls for this turn are dispatched and
            # their results appended to messages. Records ONE (s, a, r, s')
            # transition per turn, where `a` is the action the LLM ACTUALLY
            # executed (derived from the tools it called) — NOT the action
            # the policy suggested. "followed" separately tracks whether
            # those happened to match (follow-rate metric).
            if harness is not None and rl_action_rec is not None:
                try:
                    called_tools_this_turn: list[str] = [
                        tc["name"] for tc in (response.get("tool_calls") or [])
                    ]
                    # Only measure/reward following when the advice was in
                    # the prompt. `None` tells compute_reward to skip the
                    # follow branch entirely (bool | None).
                    if rl_guidance_injected:
                        followed: bool | None = _did_follow_rl_action(
                            called_tools_this_turn, rl_action_rec.action_name,
                        )
                        harness.record_action_followed(followed)
                    else:
                        followed = None

                    # Update local state for the post-turn session build.
                    commands_run += len(called_tools_this_turn)
                    if called_tools_this_turn:
                        last_command = called_tools_this_turn[-1]
                        # SE3: same source as `last_command`, kept as its own
                        # local so the encoder never has to re-parse the
                        # command string to recover the tool.
                        last_tool = called_tools_this_turn[-1]
                        last_vector = _infer_vector_from_tools(called_tools_this_turn)
                    last_turn_success = (
                        count_verified_findings(findings) > prev_verified_count
                    )
                    tokens_delta = max(0, total_tokens - prev_tokens)
                    prev_tokens = total_tokens
                    # DELIBERATELY NOT advanced here. `prev_verified_count`
                    # means "the verified count at the last RECORDED
                    # transition", and compute_reward() below needs the value
                    # from BEFORE this turn. Overwriting it here made
                    # `current_verified > previous_verified_count` always
                    # False, so the +20/+1 finding bonus never fired:
                    # recording a confirmed SQLi and doing nothing scored
                    # identically (both +0.195). It is advanced after the
                    # transition is actually recorded.

                    # Build the post-turn session (the "next state" for RL).
                    rl_session_next = build_session(
                        scan_id=scan_id,
                        target=target,
                        turn=iterations + 1,
                        commands_run=commands_run,
                        tokens_used=total_tokens,
                        token_budget=SCAN_TOKEN_BUDGET,
                        last_command=last_command,
                        last_tool=last_tool,
                        last_vector=last_vector,
                        last_turn_success=last_turn_success,
                        findings=findings,
                        # This turn's KG signal (the transition's s').
                        kg_consulted=last_turn_kg_consulted,
                        # Phase 3: feed TDI history into the RL state.
                        tdi_log=harness.tdi_log if harness else [],
                        # Tech stack for s' — must mirror the pre-turn call or
                        # the transition compares two different encodings.
                        recon_signals=harness.recon_signals() if harness else {},
                        # Phase 4: planner path counts for s' (plain data).
                        egats_planner=harness.egats_snapshot() if harness else {},
                    )

                    # Last tool output — passed to compute_reward so it can
                    # detect "no useful output" / "timeout" patterns.
                    last_tool_output = ""
                    for m in reversed(messages):
                        if m.get("role") == "tool":
                            last_tool_output = m.get("content") or ""
                            break

                    # D26 events for this turn (reward.py §1). Empty list is
                    # the normal case and leaves §4 (wasted turn) armed.
                    turn_events = _d26_events_from_tool_outputs(
                        messages[turn_tool_msgs_start:]
                    )

                    # CREDIT ASSIGNMENT: record the action the LLM actually
                    # executed, NOT the one the policy suggested. The reward
                    # below is the outcome of the executed action, and
                    # train_step() only updates grad_q[action] — recording
                    # the suggested index trains the wrong head whenever the
                    # LLM ignores the hint (the common case today).
                    executed_idx, executed_action = _resolve_executed_action(
                        called_tools_this_turn
                    )
                    if executed_idx is None:
                        # No representable action this turn (content-only
                        # turn, or only utility tools such as
                        # fetch_tool_output). There is no (s, a, r, s')
                        # tuple to record — inventing `a` would poison the
                        # Q-table exactly as the old code did.
                        logger.debug(
                            "rl_record_skipped | scan=%s | turn=%d | "
                            "reason=no_executed_action | tools=%s",
                            scan_id, iterations, called_tools_this_turn,
                        )
                    else:
                        reward, reward_breakdown = compute_reward(
                            session=rl_session_next,
                            previous_verified_count=prev_verified_count,
                            command_output=last_tool_output,
                            tokens_used_delta=tokens_delta,
                            is_done=False,
                            action_followed=followed,
                            credit_state=rl_credit_state,
                            turn_events=turn_events,
                        )

                        await harness.record_experience(
                            session=rl_session_next,
                            turn=iterations,
                            action_idx=executed_idx,
                            action_name=executed_action,
                            reward=reward,
                            done=False,
                            reward_breakdown=reward_breakdown,
                        )

                        # Baseline advances only AFTER a transition was
                        # recorded against it — see the note at the top of
                        # this block.
                        prev_verified_count = count_verified_findings(findings)
                except Exception as exc:
                    logger.warning(
                        "rl_record_failed | scan=%s | turn=%d | err=%s",
                        scan_id, iterations, exc,
                    )

            # Prune old tool results after this iteration's results
            # are appended — bounds the message history across 100 turns.
            pruned = _prune_old_tool_results(messages)
            if pruned:
                logger.debug("Context pruning | scan=%s | pruned=%d old tool results",
                             scan_id, pruned)

        except Exception as exc:
            logger.error("ReAct loop error | scan=%s | iter=%d | error=%s",
                         scan_id, iterations, exc)
            # R27: a crash ends the episode too. Guarded by
            # `_terminal_recorded`, so this is a no-op if the failure
            # happened after `exit`/`max_iterations` already closed it.
            await _record_terminal_transition("error")
            return {
                "status": "error",
                "findings_count": findings_count,
                "iterations": iterations,
                "total_tokens": total_tokens,
                "final_summary": f"Scan failed at iteration {iterations}: {exc}",
                "error": str(exc),
                "rl_stats": await _finalize_rl(scan_status="error"),
            }

    # Max iterations reached
    logger.warning("Max iterations reached | scan=%s | iter=%d", scan_id, iterations)
    await emit_scan_progress(scan_id, thought=f"⚠️ Reached limit of {max_iterations} iterations. Ending scan.",
                             agent_name="orchestrator", progress=90)

    # Turn exhaustion also ends the episode — same terminal transition as
    # `exit`, so the buffer is never left without a done=True row.
    await _record_terminal_transition("max_iterations")

    return {
        "status": "max_iterations",
        "findings_count": findings_count,
        "iterations": iterations,
        "total_tokens": total_tokens,
        "final_summary": f"Scan reached max iterations ({max_iterations}). Partial results may be available.",
        "error": None,
        "rl_stats": await _finalize_rl(scan_status="max_iterations"),
    }


__all__ = [
    "run_react_scan",
    "SYSTEM_PROMPT",
    "MAX_ITERATIONS",
]