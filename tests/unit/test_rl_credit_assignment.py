"""
Unit tests — RL credit assignment must use the EXECUTED action (B4).

The bug: the loop pushed

    action_idx=rl_action_rec.action_index   # the policy's SUGGESTION
    reward=<outcome of whatever the LLM did>

into the PER store. `DoubleQLearner.train_step()` only updates
`grad_q[action]`, so every turn where the LLM ignored the hint taught
`Q(s, a_suggested)` the outcome of a DIFFERENT action. With a low
follow-rate (100% while the policy is untrained — see
test_rl_shadow_mode.py) that is a *systematic* wrong label, not noise.

The fix records the action derived from the tools actually called, and
SKIPS the transition when no called tool maps onto ACTION_SPACE (there is
no honest label to record).
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

from app.agents.react_agent import (
    _TOOL_TO_ACTION,
    _infer_vector_from_tools,
    _resolve_executed_action,
)
from app.rl.state_encoder import ACTION_SPACE

REPO_ROOT = Path(__file__).resolve().parents[2]
AGENT_FILE = REPO_ROOT / "app" / "agents" / "react_agent.py"
HARNESS_FILE = REPO_ROOT / "app" / "rl" / "harness.py"


class TestResolveExecutedAction:
    def test_single_tool(self):
        assert _resolve_executed_action(["nuclei"]) == (
            ACTION_SPACE.index("execute_command_fuzz"),
            "execute_command_fuzz",
        )

    def test_no_tools_is_not_an_action(self):
        """Content-only turn → no transition. Never invent a label."""
        assert _resolve_executed_action([]) == (None, None)

    def test_utility_tool_only_is_not_an_action(self):
        """`fetch_tool_output` is a context utility, not in ACTION_SPACE."""
        assert "fetch_tool_output" not in ACTION_SPACE
        assert _resolve_executed_action(["fetch_tool_output"]) == (None, None)

    def test_skips_unmapped_tools_to_find_a_real_one(self):
        assert _resolve_executed_action(
            ["fetch_tool_output", "record_vulnerability"]
        ) == (ACTION_SPACE.index("report_vulnerability"), "report_vulnerability")

    def test_last_mapped_tool_wins(self):
        """Reward's command_output comes from the LAST tool — credit it."""
        assert _resolve_executed_action(["nmap", "sqlmap"]) == (
            ACTION_SPACE.index("execute_command_exploit"),
            "execute_command_exploit",
        )

    def test_last_mapped_tool_wins_with_trailing_utility(self):
        assert _resolve_executed_action(["sqlmap", "fetch_tool_output"]) == (
            ACTION_SPACE.index("execute_command_exploit"),
            "execute_command_exploit",
        )

    def test_unknown_tool_names_are_ignored(self):
        assert _resolve_executed_action(["not_a_real_tool"]) == (None, None)

    @pytest.mark.parametrize("tool,action", sorted(_TOOL_TO_ACTION.items()))
    def test_every_mapped_action_is_in_the_action_space(self, tool, action):
        """`_TOOL_TO_ACTION` must not drift from ACTION_SPACE."""
        assert action in ACTION_SPACE, f"{tool} -> {action} is not in ACTION_SPACE"

    def test_index_matches_action_space_order(self):
        for tool, action in _TOOL_TO_ACTION.items():
            idx, name = _resolve_executed_action([tool])
            assert ACTION_SPACE[idx] == action == name

    def test_agrees_with_infer_vector_from_tools(self):
        """`_infer_vector_from_tools` uses the same last-tool rule."""
        tools = ["nmap", "dalfox"]
        idx, name = _resolve_executed_action(tools)
        assert _infer_vector_from_tools(tools) == name.replace("execute_command_", "")


class TestCallSiteRecordsExecutedAction:
    @staticmethod
    def _all_record_calls() -> list[dict[str, str]]:
        """Every harness.record_experience(...) call, as {kwarg: source}."""
        tree = ast.parse(AGENT_FILE.read_text(encoding="utf-8"))
        out: list[dict[str, str]] = []
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "record_experience"
            ):
                out.append({
                    kw.arg: ast.unparse(kw.value)
                    for kw in node.keywords
                    if kw.arg
                })
        return out

    def _record_call_kwargs(self) -> dict[str, str]:
        """Kwargs of the PER-TURN record_experience call.

        There are two call sites since the terminal-transition fix: the
        per-turn one (labelled with the executed action) and the one in
        `_record_terminal_transition()` (labelled `pentest_complete`,
        `done=True`). Select by kwarg so the B4 guards keep checking the
        per-turn site instead of asserting a call count.
        """
        calls = self._all_record_calls()
        turn_calls = [kw for kw in calls if kw.get("action_idx") == "executed_idx"]
        assert len(turn_calls) == 1, (
            f"expected 1 per-turn record_experience call, got {len(turn_calls)} "
            f"(all calls: {calls})"
        )
        return turn_calls[0]

    def test_action_idx_is_the_executed_action(self):
        kw = self._record_call_kwargs()
        assert kw["action_idx"] == "executed_idx"

    def test_suggested_action_index_is_gone(self):
        """The regression itself."""
        for kw in self._all_record_calls():
            assert kw["action_idx"] != "rl_action_rec.action_index"

    def test_action_name_is_the_executed_action(self):
        kw = self._record_call_kwargs()
        assert kw["action_name"] == "executed_action"

    def test_transition_is_skipped_without_an_executed_action(self):
        src = AGENT_FILE.read_text(encoding="utf-8")
        assert "if executed_idx is None:" in src
        assert "reason=no_executed_action" in src

    def test_resolver_is_called_before_recording(self):
        src = AGENT_FILE.read_text(encoding="utf-8")
        assert "_resolve_executed_action(" in src
        # Anchor on the per-turn call, not `await harness.record_experience(`
        # — the terminal-transition helper contains that string too and is
        # defined earlier in the file.
        assert src.index("_resolve_executed_action(") < src.index(
            "action_idx=executed_idx"
        )


class TestHarnessAcceptsExplicitActionName:
    def test_record_experience_has_action_name_param(self):
        import inspect

        from app.rl.harness import HarnessBridge

        params = inspect.signature(HarnessBridge.record_experience).parameters
        assert "action_name" in params
        assert params["action_name"].default is None

    def test_action_name_defaults_to_last_suggestion(self):
        src = HARNESS_FILE.read_text(encoding="utf-8")
        assert "action_name=action_name or self._last_action_name" in src

    def test_rejection_transition_labels_index_consistently(self):
        """The self-correction path had a hardcoded name + arbitrary index."""
        src = HARNESS_FILE.read_text(encoding="utf-8")
        assert 'action_name="report_vulnerability",' not in src
        assert "ACTION_SPACE[action_idx]" in src


class TestTrainingPutsGradientOnTheRecordedAction:
    """Why the label matters: only one action head receives the gradient."""

    def test_grad_is_nonzero_only_at_the_recorded_action(self):
        import numpy as np

        from app.rl.q_learner import DoubleQLearner

        learner = DoubleQLearner()
        state = np.zeros(learner.state_dim, dtype=np.float32)
        before = learner.online.forward(state)[0].copy()

        learner.train_step(
            state=state,
            action=ACTION_SPACE.index("execute_command_recon"),
            reward=1.0,
            next_state=state,
            done=True,
        )
        after = learner.online.forward(state)[0]
        assert not np.allclose(before, after), "the recorded action's head must move"

    def test_wrong_index_puts_the_dominant_update_on_the_wrong_head(self):
        """Documents the mechanism behind the bug, not the fix.

        This is a duelling head (Q = V + A - mean(A)), so the shared value
        stream leaks a small update into every action column. What matters
        is that the *dominant* update — the advantage term, which is the
        only part that carries policy information — lands on whichever index
        was recorded. Labelling the transition with the suggested action
        therefore teaches `Q(s, a_suggested)` about `a_executed`'s outcome.
        """
        import numpy as np

        from app.rl.q_learner import DoubleQLearner

        suggested = ACTION_SPACE.index("pentest_complete")
        executed = ACTION_SPACE.index("execute_command_recon")

        learner = DoubleQLearner()
        state = np.zeros(learner.state_dim, dtype=np.float32)
        before = learner.online.forward(state)[0].copy()
        learner.train_step(
            state=state, action=suggested, reward=1.0, next_state=state, done=True,
        )
        deltas = np.abs(learner.online.forward(state)[0] - before)

        assert int(np.argmax(deltas)) == suggested
        assert deltas[suggested] > 1.5 * deltas[executed], (
            "recorded head must absorb the advantage gradient; the others may "
            "only move via the shared value stream"
        )
