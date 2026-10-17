"""Canonical action space — the single definition both RL layers derive from.

Why this module exists
----------------------
`ACTION_SPACE` used to live in `state_encoder.py` while `q_learner.py` hard-coded
`ACTION_DIM = 7` next to this comment:

    # Hard-coded here to avoid circular import: state_encoder imports from
    # q_learner for ACTION_DIM, and q_learner needs STATE_DIM at
    # class-instantiation time.

Both halves of that sentence were false: `state_encoder` imported nothing from
`q_learner` (nor anything else from `app`), so there was no cycle to avoid. The
comment described a constraint that did not exist, and the duplicate constant it
justified was a live desync hazard — adding an eighth action to `ACTION_SPACE`
would have left the Q-network with seven outputs.

Keeping the vocabulary in its own dependency-free module makes the derivation
unconditional: there is exactly one list, and `ACTION_DIM` is its length. Any
consumer that needs the size imports it; none of them can forget to.

Action index ↔ name mapping is positional, so **never reorder this list** —
cached Q-weights and logged transitions are indexed by position. Append only.
"""
from __future__ import annotations

# Order is part of the wire format (Q-network output index, DB `action` column,
# SARIF/log labels). Append new actions; do not insert or reorder.
ACTION_SPACE: list[str] = [
    "execute_command_recon",
    "execute_command_exploit",
    "execute_command_fuzz",
    "select_skill",
    "consult_kg",
    "report_vulnerability",
    "pentest_complete",
]

ACTION_DIM: int = len(ACTION_SPACE)  # 7 today — derived, never hard-coded


# ── Tool → action mapping — THE table ─────────────────────────────────
# Which RL action a callable tool credits. This used to live only in
# `app/agents/react_agent.py`, where it was the credit-assignment table, while
# `app/rl/state_encoder.py` maintained a SECOND, hand-written mapping for its
# `last_tool` feature. The two disagreed:
#
#     tool      encoder said   credit said              bucket
#     nuclei    "recon"        execute_command_fuzz     MISMATCH
#     nikto     "recon"        execute_command_fuzz     MISMATCH
#     dalfox    "recon"        execute_command_fuzz     MISMATCH
#     wpscan    "recon"        execute_command_fuzz     MISMATCH
#     fscan     "recon"        execute_command_fuzz     MISMATCH
#
# Consequence: the state vector told the Q-network that the last action was
# reconnaissance while the reward for that same action was credited to the
# fuzz head — the state contradicting its own reward, the same failure class
# as the `verified` count in `app/evidence/finding_state.py`.
#
# It lives here (dependency-free, no `app` imports) so BOTH the encoder and
# the agent loop can read one table: `state_encoder` cannot import
# `react_agent` (react_agent imports state_encoder), and `react_agent` can
# import this without any cycle.
#
# Scope note: `nuclei` is deliberately in the FUZZ bucket. It is a scanner
# that DETECTS templates; it does not exploit anything. Moving it to exploit
# (as two separate reports have proposed) would (a) contradict the credit
# table and (b) make every scan look like an exploitation attempt, starving
# the fuzz head of its only signal.
TOOL_TO_ACTION: dict[str, str] = {
    # Phase 1: Reconnaissance
    "nmap": "execute_command_recon",
    "httpx": "execute_command_recon",
    "whatweb": "execute_command_recon",
    "masscan": "execute_command_recon",
    "rustscan": "execute_command_recon",
    "subfinder": "execute_command_recon",
    "amass": "execute_command_recon",
    "dnsenum": "execute_command_recon",
    "fierce": "execute_command_recon",
    "theharvester": "execute_command_recon",
    "gau": "execute_command_recon",
    "waybackurls": "execute_command_recon",
    # Phase 2: Scanning & Enumeration (treated as fuzz in the action space)
    "ffuf": "execute_command_fuzz",
    "gobuster": "execute_command_fuzz",
    "feroxbuster": "execute_command_fuzz",
    "katana": "execute_command_fuzz",
    "nikto": "execute_command_fuzz",
    # Phase 3: Vulnerability Analysis (also fuzz in the action space)
    "nuclei": "execute_command_fuzz",
    "dalfox": "execute_command_fuzz",
    "wpscan": "execute_command_fuzz",
    "fscan": "execute_command_fuzz",
    # Phase 4: Exploitation
    "sqlmap": "execute_command_exploit",
    "metasploit": "execute_command_exploit",
    "hydra": "execute_command_exploit",
    # Phase 4b: Post-exploitation & credential access
    #
    # These eight are real tools (app/tools/*.yaml). `_resolve_executed_action()`
    # returns (None, None) for an unmapped tool and the caller then SKIPS the
    # transition, so a scan that spent its whole run in mimikatz/netexec/john
    # taught the policy nothing at all.
    "impacket": "execute_command_exploit",
    "mimikatz": "execute_command_exploit",
    "netexec": "execute_command_exploit",
    "responder": "execute_command_exploit",
    "john": "execute_command_exploit",
    "hashcat": "execute_command_exploit",
    "linpeas": "execute_command_exploit",
    "winpeas": "execute_command_exploit",
    # Utility
    "record_vulnerability": "report_vulnerability",
    "exit": "pentest_complete",
    # `consult_kg` is a DISPATCH TYPE, not an LLM tool. It stays here because
    # `resolve_action()` returns `{"type": "consult_kg"}` and the round-trip
    # guard maps a non-command dispatch back through this table.
    "consult_kg": "consult_kg",
}

_COMMAND_PREFIX = "execute_command_"

# Binary → tool-name aliases, for the DISPLAY-only `tool_category()` lookup
# below. `msfconsole` is what Metasploit's executable is actually called, and
# `last_command` has historically been fed command lines as well as tool
# names; without this, a real command line would resolve to "unknown" and the
# state feature would lose the only exploit signal it can get from that form.
#
# Deliberately NOT part of `TOOL_TO_ACTION`: aliases must never create credit
# buckets. The LLM calls tools by their schema name (`metasploit`), and the
# tests in `test_rl_action_mapping_consistency.py` assert that every credit key
# is a real tool — adding aliases there would make them look callable.
_TOOL_ALIASES: dict[str, str] = {
    "msfconsole": "metasploit",
    "msf": "metasploit",
}


def tool_category(tool_name_or_command: str) -> str:
    """`"recon"` / `"exploit"` / `"fuzz"` for a tool, `""` when unknown.

    This is the DERIVED view used by the state encoder's `last_tool` feature
    (Section 4 of the 337-dim vector). It exists so the encoder cannot hold a
    second opinion about which bucket a tool belongs to: the answer comes from
    `TOOL_TO_ACTION`, the same table credit assignment uses.

    Accepts a bare tool name (`"nuclei"`) or a command line (`"nuclei -u
    http://x"`, `"/usr/bin/nuclei -t cves/"`) — the first token's basename is
    looked up. An unknown tool returns `""`, deliberately NOT a default
    bucket: a silent default is how `nuclei` ended up misfiled as recon, and
    an unmapped tool has no credit bucket either (`_resolve_executed_action()`
    skips the transition), so inventing one here would reintroduce the same
    disagreement in miniature.

    `report_vulnerability` / `pentest_complete` / `consult_kg` are actions, not
    command buckets, so they also return `""` — they never produce shell
    reconnaissance and must not be encoded as if they did.
    """
    if not isinstance(tool_name_or_command, str):
        return ""
    text = tool_name_or_command.strip()
    if not text:
        return ""
    head = text.split()[0].rsplit("/", 1)[-1].lower()
    head = _TOOL_ALIASES.get(head, head)
    action = TOOL_TO_ACTION.get(head, "")
    if action.startswith(_COMMAND_PREFIX):
        return action[len(_COMMAND_PREFIX):]
    return ""


__all__ = [
    "ACTION_DIM",
    "ACTION_SPACE",
    "TOOL_TO_ACTION",
    "tool_category",
]
