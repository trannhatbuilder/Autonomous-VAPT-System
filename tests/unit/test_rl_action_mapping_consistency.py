"""PL7 — the three action-mapping tables must agree.

There are THREE places that describe which tool belongs to which RL action,
and nothing was checking that they agree:

  1. `app.rl.harness._ACTION_HINT`        — what we TELL the LLM to call
  2. `app.agents.react_agent._TOOL_TO_ACTION` — how we CREDIT what it called
  3. `app.rl.policy.PentestPolicy.resolve_action` — what we would EXECUTE

A disagreement in any pair is a silent bug, not a crash:

  hint ≠ credit    the LLM follows the advice and the reward lands in the
                   wrong bucket (`train_step` puts the gradient on
                   `grad_q[action]`, so this is a systematic wrong label).
  execute ≠ credit the policy suggests action A, the prescribed command is
                   credited as action B, and Q(A) never learns.
  tool unmapped    `_resolve_executed_action()` returns (None, None) and the
                   caller SKIPS the transition — no learning signal at all.

Measured before this round (`/tmp/pl7_audit.py`):

    resolve_action(execute_command_exploit) -> `nuclei`
        -> _TOOL_TO_ACTION["nuclei"] == "execute_command_fuzz"   MISMATCH
    real tools with NO credit bucket:
        hashcat, impacket, john, linpeas, mimikatz, netexec,
        responder, winpeas                                       (8 tools)
    _ACTION_HINT tool mentions disagreeing with _TOOL_TO_ACTION:  none

The report's proposed table would have made things worse: it asked for
`nuclei` -> exploit (which would make every nuclei *scan* look like an
exploitation attempt and starve the fuzz bucket) and for `dirb` / `wfuzz`
credit entries (neither is an installed tool — `app/tools/*.yaml` has neither,
so the entries could never fire). The actual defect was the opposite
direction: `resolve_action` was emitting a fuzz tool for the exploit action.
"""

from __future__ import annotations

import ast
import re
import shlex
from pathlib import Path

import pytest

from app.agents.react_agent import _TOOL_TO_ACTION, _resolve_executed_action
from app.rl.harness import _ACTION_HINT
from app.rl.policy import PentestPolicy
from app.rl.state_encoder import ACTION_SPACE

REPO = Path(__file__).resolve().parents[2]
TOOLS_DIR = REPO / "app" / "tools"
POLICY_SRC = (REPO / "app" / "rl" / "policy.py").read_text(encoding="utf-8")

# Built-ins added by `tool_bridge.build_tool_schemas()` — not YAML tools, but
# the LLM can call them, so they need a credit bucket.
BUILTIN_TOOLS = {"record_vulnerability", "fetch_tool_output", "exit"}

# Tools the LLM can call that deliberately produce NO transition.
#   fetch_tool_output — context utility: reads back a previous output, does
#                       not act on the target, so there is nothing to credit.
NO_CREDIT_BY_DESIGN = {"fetch_tool_output"}

# `_TOOL_TO_ACTION` keys that are not LLM tools. Both are `resolve_action()`
# DISPATCH TYPES: the returned dict's `type` is looked up in this table by the
# round-trip guard, and never arrives as a tool-call name.
DISPATCH_TYPES = {"consult_kg"}

# ACTION_SPACE members with no callable tool in the agent loop, so the hint can
# never be acted on directly. Asserted explicitly so the gap stays visible and
# cannot grow unnoticed.
#   select_skill — no tool by that name exists; `SkillLoader` is only used by
#                  `app/routes/orchestration.py`, never by the ReAct loop, so
#                  the hint costs the LLM a turn every time it complies.
#   consult_kg   — no tool by that name either; the KG consult happens
#                  AUTOMATICALLY after the first `_RECON_TOOLS` run, which is
#                  why the hint is harmless but not directly actionable.
KNOWN_NO_DISPATCH = {"consult_kg", "select_skill"}


def _real_tools() -> set[str]:
    names: set[str] = set()
    for path in TOOLS_DIR.glob("*.yaml"):
        m = re.search(r"^name:\s*(\S+)", path.read_text(encoding="utf-8"), re.M)
        if m:
            names.add(m.group(1).strip())
    return names | BUILTIN_TOOLS


def _hint_tools(hint: str) -> list[str]:
    """Tool names listed in parentheses inside a hint string."""
    names: list[str] = []
    for group in re.findall(r"\(([^)]*)\)", hint):
        names += [tok.strip() for tok in group.split(",") if tok.strip()]
    return names


# ─── 1. execute ↔ credit round-trip ───────────────────────────────────


class TestResolveActionRoundTrips:
    """Every command resolve_action emits must credit the action it came from."""

    @pytest.mark.parametrize("action", ACTION_SPACE)
    def test_command_first_token_credits_the_same_action(self, action):
        policy = PentestPolicy()
        session = {
            "target_url": "http://example.com",
            "_turn": 1,
            "recon_signals": {"technologies": []},
        }
        out = policy.resolve_action(action, session)
        if out["type"] != "execute_command":
            # Non-command dispatch: `report_vulnerability` / `pentest_complete`
            # / `select_skill` identify themselves by `type`, `consult_kg` is
            # looked up in the credit table. Either way the dispatch type must
            # name the SAME action, or the caller records the wrong one.
            assert out["type"] == action or _TOOL_TO_ACTION.get(out["type"]) == action, (
                f"{action}: dispatch type {out['type']!r} does not identify the "
                "action it came from"
            )
            assert out["rl_action"] == action
            return

        binary = shlex.split(out["command"])[0]
        assert binary in _real_tools(), (
            f"{action}: resolve_action emitted {binary!r}, which is not a tool "
            "the executor can run (not in app/tools/*.yaml, not a built-in)."
        )
        assert _TOOL_TO_ACTION.get(binary) == action, (
            f"{action}: resolve_action emits {binary!r}, which "
            f"_TOOL_TO_ACTION credits as "
            f"{_TOOL_TO_ACTION.get(binary)!r}. The agent loop would credit the "
            "reward to a different action than the policy chose."
        )

    @pytest.mark.parametrize(
        "tech", [[], ["wordpress"], ["graphql"], ["nginx", "php"]]
    )
    def test_exploit_branch_never_emits_a_scan_tool(self, tech):
        """Tech-aware branches must stay inside the exploit bucket."""
        policy = PentestPolicy()
        session = {"target_url": "http://example.com", "recon_signals": {"technologies": tech}}
        out = policy.resolve_action("execute_command_exploit", session)
        binary = shlex.split(out["command"])[0]
        assert _TOOL_TO_ACTION[binary] == "execute_command_exploit"

    def test_recon_branch_covers_both_turns(self):
        policy = PentestPolicy()
        for turn in (1, 10):
            out = policy.resolve_action(
                "execute_command_recon",
                {"target_url": "http://example.com", "_turn": turn},
            )
            binary = shlex.split(out["command"])[0]
            assert _TOOL_TO_ACTION[binary] == "execute_command_recon"


# ─── 2. unknown action is loud, not silent ────────────────────────────


class TestUnknownActionIsNotSilent:
    def test_unknown_action_warns(self, caplog):
        import logging

        policy = PentestPolicy()
        with caplog.at_level(logging.WARNING, logger="app.rl.policy"):
            out = policy.resolve_action(
                "execute_command_typo", {"target_url": "http://example.com"}
            )
        assert any("rl_resolve_action_unknown" in r.message for r in caplog.records), (
            "an unknown action was substituted with no log line — the caller "
            "gets a plausible command that does the wrong thing"
        )
        assert out["fallback"] is True
        assert out["requested_action"] == "execute_command_typo"
        # `rl_action` must report what was CHOSEN, not what was rejected.
        assert out["rl_action"] == "execute_command_recon"

    def test_substituted_action_still_round_trips(self):
        policy = PentestPolicy()
        out = policy.resolve_action("nope", {"target_url": "http://example.com"})
        binary = shlex.split(out["command"])[0]
        assert _TOOL_TO_ACTION[binary] == out["rl_action"]

    def test_known_action_is_not_marked_fallback(self):
        policy = PentestPolicy()
        for action in ACTION_SPACE:
            out = policy.resolve_action(action, {"target_url": "http://e.com"})
            assert "fallback" not in out, action
            assert "requested_action" not in out, action

    def test_no_silent_curl_fallback_remains(self):
        """The old `curl -sI` default was unmapped and unlogged."""
        assert 'f"curl -sI' not in POLICY_SRC

    def test_unknown_action_logs_exactly_once(self, caplog):
        import logging

        policy = PentestPolicy()
        with caplog.at_level(logging.WARNING, logger="app.rl.policy"):
            policy.resolve_action("bogus", {"target_url": "http://e.com"})
        hits = [r for r in caplog.records if "rl_resolve_action_unknown" in r.message]
        assert len(hits) == 1


# ─── 3. the credit table covers every callable tool ───────────────────


class TestCreditTableCoversEveryTool:
    def test_every_real_tool_has_a_credit_bucket(self):
        missing = _real_tools() - set(_TOOL_TO_ACTION) - NO_CREDIT_BY_DESIGN
        assert not missing, (
            f"these tools can be called but have no credit bucket, so every "
            f"turn that uses them SKIPS the transition: {sorted(missing)}"
        )

    def test_post_exploitation_tools_are_in_the_exploit_bucket(self):
        for tool in (
            "impacket", "mimikatz", "netexec", "responder",
            "john", "hashcat", "linpeas", "winpeas",
        ):
            assert _TOOL_TO_ACTION[tool] == "execute_command_exploit", tool

    def test_every_credit_key_is_a_tool_or_a_documented_dispatch_type(self):
        stray = set(_TOOL_TO_ACTION) - _real_tools() - DISPATCH_TYPES
        assert not stray, (
            f"credit-table keys that are neither a callable tool nor a known "
            f"dispatch type: {sorted(stray)}"
        )

    def test_fetch_tool_output_is_deliberately_uncredited(self):
        assert _resolve_executed_action(["fetch_tool_output"]) == (None, None)

    def test_no_credit_entry_is_unreachable_dead_config(self):
        """Dead entries are how these tables rot — keep the set explicit."""
        real = _real_tools()
        dead = {k for k in _TOOL_TO_ACTION if k not in real} - DISPATCH_TYPES
        assert not dead, (
            f"{sorted(dead)} are in _TOOL_TO_ACTION but no such tool exists — "
            "either remove them or document them as a dispatch type"
        )


# ─── 4. hint ↔ credit agreement ───────────────────────────────────────


class TestHintAgreesWithCredit:
    def test_every_hint_tool_is_in_the_same_bucket(self):
        real = _real_tools()
        bad = []
        for action, hint in _ACTION_HINT.items():
            for tool in _hint_tools(hint):
                if tool not in real:
                    continue  # prose, not a tool name
                if _TOOL_TO_ACTION.get(tool) != action:
                    bad.append((action, tool, _TOOL_TO_ACTION.get(tool)))
        assert not bad, (
            f"the hint tells the LLM to call a tool that is credited "
            f"elsewhere (action, tool, actual bucket): {bad}"
        )

    def test_nuclei_stays_in_the_fuzz_bucket(self):
        """The report proposed moving it to exploit — that would make every
        nuclei scan look like an exploitation attempt."""
        assert _TOOL_TO_ACTION["nuclei"] == "execute_command_fuzz"
        assert "nuclei" in _ACTION_HINT["execute_command_fuzz"].lower()
        assert "nuclei" not in _ACTION_HINT["execute_command_exploit"].lower()

    def test_exploit_hint_lists_the_post_exploitation_tools(self):
        hint = _ACTION_HINT["execute_command_exploit"].lower()
        for tool in ("sqlmap", "metasploit", "hydra", "mimikatz", "impacket"):
            assert tool in hint, f"{tool} credits exploit but is not hinted"

    def test_dirb_and_wfuzz_are_not_fake_entries(self):
        """The report asked for both. Neither is an installed tool."""
        real = _real_tools()
        for tool in ("dirb", "wfuzz"):
            assert tool not in real
            assert tool not in _TOOL_TO_ACTION


# ─── 5. the two "terminal" actions really are dispatchable ────────────


class TestTerminalActionsAreDispatchable:
    """B10 pattern: a hint that names an action the caller cannot handle is a
    no-op that costs the LLM a turn every time it complies."""

    def test_report_vulnerability_maps_to_a_real_tool(self):
        from app.agents.tool_bridge import build_tool_schemas

        schemas = build_tool_schemas()
        names = {s["function"]["name"] for s in schemas}
        assert "record_vulnerability" in names
        assert _TOOL_TO_ACTION["record_vulnerability"] == "report_vulnerability"

    def test_pentest_complete_maps_to_a_real_tool(self):
        from app.agents.tool_bridge import build_tool_schemas

        schemas = build_tool_schemas()
        names = {s["function"]["name"] for s in schemas}
        assert "exit" in names
        assert _TOOL_TO_ACTION["exit"] == "pentest_complete"

    def test_only_documented_actions_lack_dispatch(self):
        """Keeps the known gap from growing silently."""
        from app.agents.tool_bridge import build_tool_schemas

        names = {s["function"]["name"] for s in build_tool_schemas()}
        no_dispatch = {
            action for action in ACTION_SPACE
            if not any(
                _TOOL_TO_ACTION.get(tool) == action for tool in names
            )
        }
        assert no_dispatch == KNOWN_NO_DISPATCH, (
            f"actions with no callable tool changed: {sorted(no_dispatch)}. "
            "If a new one appeared, its hint is now a no-op."
        )


# ─── 6. the executor boundary the report flagged ──────────────────────


class TestExecutorBoundary:
    """`resolve_action` returns a command STRING. It must never reach a shell."""

    def test_no_shell_true_in_application_code(self):
        offenders = []
        for path in list((REPO / "app").rglob("*.py")) + list(
            (REPO / "workers").rglob("*.py")
        ):
            src = path.read_text(encoding="utf-8")
            if "shell=True" in src:
                offenders.append(str(path.relative_to(REPO)))
        assert not offenders, (
            f"shell=True is a re-entry point for the command-injection class "
            f"already fixed in the sqlmap wrapper: {offenders}"
        )

    def test_executor_splits_and_passes_a_list(self):
        src = (REPO / "app" / "sandbox" / "executor.py").read_text(encoding="utf-8")
        assert "shlex.split(command)" in src, (
            "the executor no longer splits string commands — a string reaching "
            "create_subprocess_exec would be treated as a single binary name, "
            "or fall back to a shell"
        )

    def test_generated_commands_survive_shlex_split(self):
        """A quoting bug in resolve_action would make the command unrunnable."""
        policy = PentestPolicy()
        for action in ACTION_SPACE:
            out = policy.resolve_action(
                action,
                {
                    "target_url": "http://example.com/a b?x=1&y=2",
                    "_turn": 1,
                    "recon_signals": {"technologies": ["wordpress", "graphql"]},
                },
            )
            if out["type"] != "execute_command":
                continue
            tokens = shlex.split(out["command"])
            assert tokens, action
            # The URL must come through as ONE token, not shattered on the
            # '&' — that is what shlex.quote() in resolve_action is for.
            assert any("example.com" in t for t in tokens), (
                f"{action}: target was mangled by quoting: {tokens}"
            )

    def test_no_command_uses_a_shell_metacharacter_that_needs_one(self):
        """`;`, `|`, `&&`, `$()` in a generated command imply shell=True."""
        policy = PentestPolicy()
        forbidden = ("&&", "||", ";", "$(", "`")
        for action in ACTION_SPACE:
            out = policy.resolve_action(
                action, {"target_url": "http://example.com", "_turn": 1}
            )
            if out["type"] != "execute_command":
                continue
            for token in forbidden:
                assert token not in out["command"], (
                    f"{action}: command needs a shell ({token!r}): "
                    f"{out['command']}"
                )


# ─── 7. AST guard: the docstring contract stays true ──────────────────


class TestResolveActionSourceGuard:
    def test_resolve_action_checks_action_space_before_dispatch(self):
        tree = ast.parse(POLICY_SRC)
        fn = next(
            n for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == "resolve_action"
        )
        src = ast.get_source_segment(POLICY_SRC, fn) or ""
        assert "not in ACTION_SPACE" in src, (
            "resolve_action lost its unknown-action guard"
        )
        assert src.index("not in ACTION_SPACE") < src.index("execute_command_recon"), (
            "the unknown-action guard runs after dispatch has begun"
        )

    def test_docstring_states_the_round_trip_contract(self):
        tree = ast.parse(POLICY_SRC)
        fn = next(
            n for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == "resolve_action"
        )
        doc = ast.get_docstring(fn) or ""
        assert "_TOOL_TO_ACTION" in doc and "round-trip" in doc.lower(), (
            "the invariant that keeps credit assignment correct is no longer "
            "documented where someone editing this function will see it"
        )
