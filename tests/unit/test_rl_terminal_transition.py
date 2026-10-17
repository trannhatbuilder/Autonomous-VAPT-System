"""Guards for B8 — the missing terminal (done=True) transition.

Before this fix the PER store only ever received `done=False` rows: the
`exit` branch returned straight into `_finalize_rl()` and the
`max_iterations` branch returned without recording anything, so
Q-learning bootstrapped off `max Q(s')` of a state that never existed and
`episode_reward` lost the closing turn's reward entirely.

These tests are deliberately split between AST checks (call sites must
keep passing a *variable* `done`) and runtime checks (the terminal
transition must actually land in the store as done=True with the
`pentest_complete` action).
"""
from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

APP_DIR = Path(__file__).resolve().parents[2] / "app"
AGENT_FILE = APP_DIR / "agents" / "react_agent.py"
HARNESS_FILE = APP_DIR / "rl" / "harness.py"


def _agent_tree() -> ast.Module:
    return ast.parse(AGENT_FILE.read_text(encoding="utf-8"))


def _record_calls() -> list[ast.Call]:
    return [
        node for node in ast.walk(_agent_tree())
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "record_experience"
    ]


def _kwargs(call: ast.Call) -> dict[str, str]:
    return {kw.arg: ast.unparse(kw.value) for kw in call.keywords if kw.arg}


# ─── AST: call-site shape ────────────────────────────────────────────


class TestTerminalCallSite:
    def _terminal(self) -> dict[str, str]:
        calls = [kw for kw in map(_kwargs, _record_calls()) if kw.get("done") == "True"]
        assert len(calls) == 1, (
            f"expected exactly 1 done=True record_experience call, got {len(calls)}"
        )
        return calls[0]

    def test_a_done_true_transition_exists(self):
        kw = self._terminal()
        assert kw["done"] == "True"

    def test_terminal_action_is_pentest_complete(self):
        kw = self._terminal()
        assert "pentest_complete" in kw["action_idx"]
        assert kw["action_name"] == "'pentest_complete'"

    def test_terminal_transition_is_done_aware(self):
        """The terminal reward must be computed with is_done=True.

        `reward.py` gates both the wasted-turn penalty and the completion
        bonus on `is_done`, so `is_done=False` there would make the
        closing turn's reward identical to a mid-scan turn.
        """
        fn = next(
            node for node in ast.walk(_agent_tree())
            if isinstance(node, ast.AsyncFunctionDef)
            and node.name == "_record_terminal_transition"
        )
        body = ast.unparse(fn)
        assert "is_done=True" in body
        assert "done=True" in body

    def test_terminal_transition_helper_exists(self):
        assert "async def _record_terminal_transition" in AGENT_FILE.read_text(
            encoding="utf-8"
        )

    def test_exit_path_records_terminal_transition(self):
        src = AGENT_FILE.read_text(encoding="utf-8")
        assert 'await _record_terminal_transition("exit")' in src

    def test_max_iterations_path_records_terminal_transition(self):
        src = AGENT_FILE.read_text(encoding="utf-8")
        assert 'await _record_terminal_transition("max_iterations")' in src

    def test_terminal_transition_precedes_finalize(self):
        """`_finalize_rl` stops the train worker, so it must run last."""
        src = AGENT_FILE.read_text(encoding="utf-8")
        exit_idx = src.index('await _record_terminal_transition("exit")')
        finalize_idx = src.index('rl_stats": await _finalize_rl(scan_status="completed")')
        assert exit_idx < finalize_idx

    def test_per_turn_call_is_still_done_false(self):
        kw = [k for k in map(_kwargs, _record_calls()) if k.get("action_idx") == "executed_idx"]
        assert len(kw) == 1
        assert kw[0]["done"] == "False"


# ─── runtime: the transition actually lands ──────────────────────────


class _FakeStore:
    def __init__(self) -> None:
        self.rows: list[dict] = []

    async def add(self, **kw) -> None:
        self.rows.append(kw)

    def __len__(self) -> int:
        return len(self.rows)


class TestRecordExperienceDoneFlag:
    """`record_experience` must forward the done flag it is given."""

    @pytest.mark.asyncio
    async def test_done_true_is_forwarded_to_store(self):
        from app.rl.harness import HarnessBridge
        from app.rl._session_builder import build_session

        bridge = HarnessBridge(scan_id="t-done", target="example.com")
        store = _FakeStore()
        bridge._store = store
        # `select_action` is what stashes _prev_state_vec; record_experience
        # refuses to write without it.
        session = build_session(
            scan_id="t-done", target="example.com", turn=1,
            commands_run=1, tokens_used=1000, token_budget=500_000,
            last_command="", last_vector="", last_turn_success=True,
            findings=[],
        )
        bridge._prev_state_vec = np.zeros(337, dtype=np.float32)

        await bridge.record_experience(
            session=session, turn=5, action_idx=6,
            action_name="pentest_complete", reward=1.0, done=True,
        )

        assert len(store.rows) == 1
        assert store.rows[0]["done"] is True
        assert store.rows[0]["action"] == 6
        assert store.rows[0]["action_name"] == "pentest_complete"

        if bridge._train_worker_task is not None:
            bridge._train_worker_task.cancel()

    @pytest.mark.asyncio
    async def test_episode_reward_includes_terminal_reward(self):
        from app.rl.harness import HarnessBridge
        from app.rl._session_builder import build_session

        bridge = HarnessBridge(scan_id="t-ep", target="example.com")
        bridge._store = _FakeStore()
        bridge._prev_state_vec = np.zeros(337, dtype=np.float32)
        session = build_session(
            scan_id="t-ep", target="example.com", turn=1,
            commands_run=1, tokens_used=1000, token_budget=500_000,
            last_command="", last_vector="", last_turn_success=True,
            findings=[],
        )

        await bridge.record_experience(
            session=session, turn=1, action_idx=0,
            action_name="execute_command_recon", reward=2.5, done=False,
        )
        await bridge.record_experience(
            session=session, turn=2, action_idx=6,
            action_name="pentest_complete", reward=0.75, done=True,
        )

        assert bridge.get_stats()["episode_reward"] == pytest.approx(3.25)

        if bridge._train_worker_task is not None:
            bridge._train_worker_task.cancel()
