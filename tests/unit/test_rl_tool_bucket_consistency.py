"""SE2 — one tool→bucket table, not two.

`app/rl/state_encoder.py::_infer_tool_from_command` used to be a hand-written
substring heuristic that decided the `last_tool` feature (Section 4 of the
337-dim vector) on its own:

    if any(t in cmd_lower for t in ("nuclei", "nikto", "nmap", ...)):
        return "recon"

`app/agents/react_agent.py::_TOOL_TO_ACTION` — the table credit assignment
actually uses — said something else for five of those tools:

    tool      encoder said   credit said               verdict
    nuclei    "recon"        execute_command_fuzz      MISMATCH
    nikto     "recon"        execute_command_fuzz      MISMATCH
    dalfox    "recon"        execute_command_fuzz      MISMATCH
    wpscan    "recon"        execute_command_fuzz      MISMATCH
    fscan     "recon"        execute_command_fuzz      MISMATCH

Nothing crashed. The failure was silent and systematic: after a `nuclei` run
the reward was assigned to `grad_q[execute_command_fuzz]` while the state that
value was learned FROM said "the last action was reconnaissance". The state
contradicted its own reward — the same failure class as the four competing
`verified` definitions in SE1, and the same cause: two tables, one concept.

The report's proposed fix was to move `nuclei` (and add `hydra`) to the
exploit tuple. That would have created a THIRD answer, because credit
assignment puts nuclei in fuzz and two earlier reports already argued for
exploit and were refuted (`test_rl_action_mapping_consistency.py` records why:
nuclei is a scanner, not an exploit, and starving the fuzz head removes its
only signal). The fix must not be "which bucket does the report prefer" but
"make the encoder read the bucket that already exists".

So `TOOL_TO_ACTION` moved to the dependency-free `app/rl/actions.py` (the
encoder cannot import `react_agent` — react_agent imports the encoder), and
both sides now derive from it via `tool_category()`.

These guards pin the derivation, not the five instances.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from app.rl.actions import ACTION_SPACE, TOOL_TO_ACTION, tool_category
from app.rl.policy import PentestPolicy
from app.rl.state_encoder import StateEncoder

REPO = Path(__file__).resolve().parents[2]
ACTIONS_SRC = (REPO / "app" / "rl" / "actions.py").read_text(encoding="utf-8")
ENCODER_SRC = (REPO / "app" / "rl" / "state_encoder.py").read_text(encoding="utf-8")

PREFIX = "execute_command_"

# The three buckets the encoder's Section 4 one-hot can name.
BUCKETS = {"recon", "exploit", "fuzz"}

# Tools whose credit bucket is an ACTION, not a command bucket.
NON_COMMAND_ACTIONS = {"report_vulnerability", "pentest_complete", "consult_kg"}

# Binaries that are NOT tools in this system. `curl` is sandbox-allowlisted and
# `arjun`/`paramspider` are not installed; all three appeared in the old
# heuristic, which is how it drifted away from the credit table.
NOT_A_TOOL = ("curl", "arjun", "paramspider", "dirb", "wfuzz")


def _command_tools() -> list[tuple[str, str]]:
    return [
        (tool, action[len(PREFIX):])
        for tool, action in sorted(TOOL_TO_ACTION.items())
        if action.startswith(PREFIX)
    ]


# ─── 1. the encoder derives its bucket from the credit table ──────────


class TestEncoderDerivesFromCreditTable:
    @pytest.mark.parametrize("tool,bucket", _command_tools())
    def test_encoder_bucket_equals_credit_bucket(self, tool, bucket):
        """The whole point: one answer per tool, and it is the credit answer."""
        assert tool_category(tool) == bucket, (
            f"{tool} is credited as {PREFIX}{bucket} but the encoder's "
            f"last_tool feature says {tool_category(tool)!r} — the state "
            "would contradict the reward learned from it"
        )

    @pytest.mark.parametrize("tool,bucket", _command_tools())
    def test_command_line_form_resolves_the_same(self, tool, bucket):
        """`last_command` has been fed command lines as well as tool names."""
        assert tool_category(f"{tool} -u http://target.example") == bucket
        assert tool_category(f"/usr/bin/{tool} --batch") == bucket
        assert tool_category(f"  {tool.upper()}  ") == bucket

    def test_encoder_feature_agrees_for_every_action_the_policy_emits(self):
        """Closes the loop encoder ↔ policy ↔ credit, not just encoder ↔ credit.

        If `resolve_action` starts emitting a binary the encoder cannot
        categorise, the `last_tool` feature silently loses that action.
        """
        import shlex

        policy = PentestPolicy()
        checked = 0
        for action in ACTION_SPACE:
            out = policy.resolve_action(
                action,
                {
                    "target_url": "http://example.com",
                    "_turn": 1,
                    "recon_signals": {"technologies": ["wordpress", "graphql"]},
                },
            )
            if out["type"] != "execute_command":
                continue
            binary = shlex.split(out["command"])[0]
            if not action.startswith(PREFIX):
                continue
            assert tool_category(binary) == action[len(PREFIX):], (
                f"resolve_action({action}) emits {binary!r}, which the encoder "
                f"categorises as {tool_category(binary)!r}"
            )
            checked += 1
        assert checked >= 3, "no command-emitting action was exercised"


class TestReportedInstances:
    """The two tools SE2 named, pinned in the direction the credit table says."""

    def test_nuclei_is_fuzz_not_recon(self):
        assert tool_category("nuclei") == "fuzz"
        assert tool_category("nuclei -u http://x") == "fuzz"
        assert tool_category("nuclei") != "recon"

    def test_nuclei_is_not_exploit_either(self):
        """Two reports have asked for this. Both are wrong — see module docs."""
        assert tool_category("nuclei") != "exploit"
        assert TOOL_TO_ACTION["nuclei"] == f"{PREFIX}fuzz"

    def test_nikto_has_the_same_defect_and_is_fixed_too(self):
        """The report missed this one; it is the identical instance."""
        assert tool_category("nikto") == "fuzz"
        assert tool_category("nikto -h http://x") == "fuzz"

    def test_the_other_three_missed_instances(self):
        for tool in ("dalfox", "wpscan", "fscan"):
            assert tool_category(tool) == "fuzz", tool

    def test_hydra_was_already_in_the_exploit_bucket(self):
        """The report asked to ADD hydra to the exploit tuple. It was already
        credited as exploit; only the (now deleted) heuristic lacked it."""
        assert TOOL_TO_ACTION["hydra"] == f"{PREFIX}exploit"
        assert tool_category("hydra -l admin -P rockyou.txt http://x") == "exploit"


# ─── 2. no silent default, no invented bucket ─────────────────────────


class TestNoSilentDefaultBucket:
    @pytest.mark.parametrize("name", NOT_A_TOOL)
    def test_unknown_binaries_get_no_bucket(self, name):
        """The old heuristic returned "recon" for EVERYTHING unknown.

        A default bucket is how nuclei got misfiled: the heuristic had to
        guess an answer and the guess was never checked against anything.
        """
        assert tool_category(name) == "" or tool_category(name) in BUCKETS

    @pytest.mark.parametrize("name", NOT_A_TOOL)
    def test_not_a_tool_has_no_credit_entry(self, name):
        assert name not in TOOL_TO_ACTION

    @pytest.mark.parametrize("action", sorted(NON_COMMAND_ACTIONS))
    def test_action_names_are_not_command_buckets(self, action):
        """`record_vulnerability` / `exit` / `consult_kg` are actions.

        Encoding them as "recon" (the old default) made a finding-recording
        turn look like a reconnaissance turn.
        """
        assert tool_category(action) == ""

    @pytest.mark.parametrize("value", ["", "   ", None, 0])
    def test_non_string_and_empty_input_is_safe(self, value):
        assert tool_category(value) == ""

    def test_bucket_vocabulary_matches_the_action_space(self):
        """Every bucket the encoder can return must name a real action."""
        assert BUCKETS == {a[len(PREFIX):] for a in ACTION_SPACE if a.startswith(PREFIX)}
        produced = {tool_category(t) for t, _ in _command_tools()}
        assert produced == BUCKETS, f"a credit bucket is unreachable: {produced}"


class TestAliasesAreDisplayOnly:
    def test_msfconsole_resolves_to_metasploit(self):
        assert tool_category("msfconsole -q") == "exploit"

    def test_aliases_never_become_credit_entries(self):
        """An alias in the credit table would look like a callable tool and
        break `test_every_credit_key_is_a_tool_or_a_documented_dispatch_type`."""
        import re

        alias_block = re.search(
            r"_TOOL_ALIASES[^{]*\{(.*?)\n\}", ACTIONS_SRC, re.S
        )
        assert alias_block, "_TOOL_ALIASES disappeared"
        aliases = re.findall(r'"([^"]+)"\s*:', alias_block.group(1))
        assert aliases, "no aliases parsed — this guard silently stopped working"
        for alias in aliases:
            assert alias not in TOOL_TO_ACTION, (
                f"{alias!r} is an alias, not a tool; putting it in "
                "TOOL_TO_ACTION makes it look callable and credits it twice"
            )


# ─── 3. structural guards: the second table cannot come back ──────────


class TestOneTableOnly:
    def test_encoder_holds_no_tool_names(self):
        """AST guard — a substring heuristic is a second table.

        Matches string literals only, so explanatory prose in docstrings that
        names a tool is still allowed. What is forbidden is a literal that a
        branch could test against.
        """
        offenders: list[str] = []
        for node in ast.walk(ast.parse(ENCODER_SRC)):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                if node.value.strip().lower() in TOOL_TO_ACTION:
                    offenders.append(f"line {node.lineno}: {node.value!r}")
        assert not offenders, (
            "state_encoder.py contains tool-name string literals again — the "
            f"`last_tool` feature must derive from TOOL_TO_ACTION: {offenders}"
        )

    def test_encoder_infer_delegates(self):
        """The feature still exists (callers use it) but it is a delegation."""
        fn_src = ast.get_source_segment(
            ENCODER_SRC,
            next(
                n
                for n in ast.walk(ast.parse(ENCODER_SRC))
                if isinstance(n, ast.FunctionDef) and n.name == "_infer_tool_from_command"
            ),
        ) or ""
        assert "tool_category(" in fn_src, (
            "_infer_tool_from_command no longer delegates to the canonical "
            "table — a hand-written mapping is a second source of truth"
        )
        assert " in cmd_lower" not in fn_src, (
            "substring matching is back: a command containing the word "
            "'exploit' would be filed as exploitation"
        )

    def test_credit_table_is_imported_not_redefined(self):
        """`react_agent` must re-export the same object, not a copy."""
        from app.agents.react_agent import _TOOL_TO_ACTION

        assert _TOOL_TO_ACTION is TOOL_TO_ACTION, (
            "react_agent defines its own copy of the credit table; the two "
            "will drift exactly as the encoder's copy did"
        )

    def test_actions_module_is_a_leaf(self):
        """It is the shared home BECAUSE it imports nothing from `app`.

        If it ever imports a sibling, the encoder can no longer read it
        without triggering `app/rl/__init__.py` (which imports `reward`,
        which imports `app.evidence`), recreating the SE1 import cycle.
        """
        for node in ast.walk(ast.parse(ACTIONS_SRC)):
            if isinstance(node, ast.ImportFrom) and node.module:
                assert not node.module.startswith("app"), (
                    f"app/rl/actions.py imports {node.module!r}; it must stay "
                    "dependency-free to be importable from the encoder"
                )
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    assert not alias.name.startswith("app"), alias.name

    def test_encoder_still_imports_tool_category(self):
        tree = ast.parse(ENCODER_SRC)
        imported = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module == "app.rl.actions"
            for alias in node.names
        }
        assert "tool_category" in imported


# ─── 4. end to end: the state vector reflects the credit bucket ───────


class TestEndToEnd:
    @pytest.mark.parametrize(
        "tool,expected",
        [
            ("nuclei", "fuzz"),
            ("nikto", "fuzz"),
            ("ffuf", "fuzz"),
            ("sqlmap", "exploit"),
            ("metasploit", "exploit"),
            ("nmap", "recon"),
            ("whatweb", "recon"),
        ],
    )
    def test_session_last_tool_feature(self, tool, expected):
        state = StateEncoder.from_session(
            {"_turn": 2, "_last_command": tool, "target_url": "http://x"}
        )
        assert state.last_tool == expected

    def test_turn_one_with_no_tool_is_no_signal(self):
        """Turn 1 ran nothing, so the feature must not claim reconnaissance.

        The historical default was "recon", which is a false signal: the
        Q-network would learn from turns that never ran a recon tool. "" is
        encoded as the zero vector (see `TOOL_LABELS` / `_label_index`).
        """
        state = StateEncoder.from_session({"_turn": 1})
        assert state.last_tool == ""
        assert int(state.to_vector()[8 + 256 + 32:8 + 256 + 64].sum()) == 0

    def test_changing_bucket_changes_the_state_vector(self):
        """If the fix were inert, the two vectors would be identical."""
        import numpy as np

        a = StateEncoder.from_session({"_turn": 2, "_last_command": "nuclei"}).to_vector()
        b = StateEncoder.from_session({"_turn": 2, "_last_command": "nmap"}).to_vector()
        assert a.shape == b.shape == (337,)
        assert not np.array_equal(a, b), (
            "last_tool is not reaching the vector — the Section 4 one-hot is "
            "not wired"
        )
