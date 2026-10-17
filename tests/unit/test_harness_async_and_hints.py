"""Guards for B10 (stale "no-op" hint text) and B11 (fire-and-forget async).

B10
    `_ACTION_HINT["consult_kg"]` told the LLM the KG consult was
    "(Phase-2 feature — currently a no-op)" long after `consult_kg` was
    actually wired — the agent was being actively talked out of using a
    working tool.

B11
    `process_rejection()` and `_save_checkpoint_safe()` scheduled
    coroutines with `asyncio.get_event_loop().create_task(...)`, kept no
    reference to the returned Task, and swallowed the no-loop case with
    `except RuntimeError: pass` / a bare `if loop.is_running():` with no
    `else`. The write could be dropped, garbage-collected, or killed at
    loop shutdown with nothing in the logs.
"""
from __future__ import annotations

import ast
import inspect
from pathlib import Path

import numpy as np
import pytest

APP_DIR = Path(__file__).resolve().parents[2] / "app"
HARNESS_FILE = APP_DIR / "rl" / "harness.py"


def _src() -> str:
    return HARNESS_FILE.read_text(encoding="utf-8")


def _tree() -> ast.Module:
    return ast.parse(_src())


# ─── B10 ─────────────────────────────────────────────────────────────


class TestConsultKgHintIsNotDeprecated:
    def test_hint_does_not_claim_no_op(self):
        from app.rl.harness import _ACTION_HINT

        hint = _ACTION_HINT["consult_kg"]
        assert "no-op" not in hint.lower()
        assert "phase-2" not in hint.lower()

    def test_hint_still_describes_the_tool(self):
        from app.rl.harness import _ACTION_HINT

        assert "knowledge graph" in _ACTION_HINT["consult_kg"].lower()

    def test_every_action_has_a_hint(self):
        from app.rl.state_encoder import ACTION_SPACE
        from app.rl.harness import _ACTION_HINT

        assert set(_ACTION_HINT) == set(ACTION_SPACE)

    def test_no_hint_marks_itself_unimplemented(self):
        from app.rl.harness import _ACTION_HINT

        for action, hint in _ACTION_HINT.items():
            assert "currently a no-op" not in hint, action
            assert "not implemented" not in hint.lower(), action

    def test_module_docstring_drops_phase1_scope_claim(self):
        src = _src()
        assert "Phase-1 scope" not in src
        assert "tdi_log stays []" not in src
        assert "egats_planner stays None" not in src


# ─── B11: process_rejection ──────────────────────────────────────────


class TestProcessRejectionIsAwaited:
    def test_is_a_coroutine_function(self):
        from app.rl.harness import HarnessBridge

        assert inspect.iscoroutinefunction(HarnessBridge.process_rejection)

    def test_no_get_event_loop_call_in_harness(self):
        """`asyncio.get_event_loop()` is deprecated from sync context in 3.12
        and, worse, was used here to schedule work from inside a coroutine
        that could simply await it. Checks CALLS, not comments."""
        offenders = [
            ast.unparse(node)
            for node in ast.walk(_tree())
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in {"get_event_loop", "get_event_loop_policy"}
        ]
        assert offenders == []

    def test_no_fire_and_forget_create_task(self):
        """Any `create_task` must keep a reference on `self`.

        The training worker is scheduled as
        `self._train_worker_task = asyncio.create_task(...)` — that is
        fine. A bare `loop.create_task(...)` / `asyncio.create_task(...)`
        whose result is discarded is not: the event loop only holds a weak
        reference, so the task can be garbage-collected mid-flight.
        """
        tree = _tree()
        call_count = sum(
            1 for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and ast.unparse(node.func).endswith("create_task")
        )
        stored = [
            ast.unparse(node.targets[0])
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Call)
            and ast.unparse(node.value.func).endswith("create_task")
        ]
        assert call_count == len(stored), (
            f"{call_count - len(stored)} create_task result(s) discarded; "
            "store the Task on self so it cannot be GC'd"
        )
        for target in stored:
            assert target.startswith("self."), f"task stored on {target}"

    @pytest.mark.asyncio
    async def test_rejection_write_is_in_the_store_after_await(self):
        """Awaiting means the transition exists before we return."""
        from app.rl.harness import HarnessBridge

        bridge = HarnessBridge(scan_id="t-sc", target="example.com")
        rows: list[dict] = []

        class _Store:
            async def add(self, **kw):
                rows.append(kw)

            def __len__(self):
                return len(rows)

        bridge._store = _Store()
        bridge._sc_enabled = True
        if not hasattr(bridge, "_sc") or bridge._sc is None:
            from app.rl.self_correction import SelfCorrectionEngine
            bridge._sc = SelfCorrectionEngine()

        bridge._prev_state_vec = np.zeros(337, dtype=np.float32)
        bridge._prev_action_idx = 5
        bridge._turn_count = 3

        result = await bridge.process_rejection(
            finding={"vuln_type": "xss", "location": "http://example.com/x"},
            reason="no proof of concept",
        )

        assert result is not None
        assert rows, "rejection experience must be awaited into the store"
        assert rows[0]["reward"] <= 0
        assert rows[0]["priority"] > 0
        assert rows[0]["reward_breakdown"]["type"] == "rejection"

        if bridge._train_worker_task is not None:
            bridge._train_worker_task.cancel()

    @pytest.mark.asyncio
    async def test_agent_loop_awaits_process_rejection(self):
        agent_src = (
            APP_DIR / "agents" / "react_agent.py"
        ).read_text(encoding="utf-8")
        assert "await harness.process_rejection(" in agent_src
        assert "= harness.process_rejection(" not in agent_src


# ─── B11: checkpoint save ────────────────────────────────────────────


class TestCheckpointSaveIsAwaited:
    def test_save_checkpoint_safe_is_async(self):
        from app.rl.harness import HarnessBridge

        assert inspect.iscoroutinefunction(HarnessBridge._save_checkpoint_safe)

    def test_end_scan_awaits_the_checkpoint(self):
        src = _src()
        assert "await self._save_checkpoint_safe(episode_reward)" in src

    def test_sql_failure_is_logged_not_swallowed(self):
        src = _src()
        # The runtime-exception path around save_checkpoint must log.
        assert "rl_checkpoint_sql_save_failed" in src

    @pytest.mark.asyncio
    async def test_npz_saved_even_without_a_database(self):
        from app.rl.harness import HarnessBridge

        bridge = HarnessBridge(scan_id="t-ckpt", target="example.com")
        await bridge._save_checkpoint_safe(1.0)
        # SQL will fail (no DB in unit tests) but must not raise.
        assert bridge._ckpt_saved is True
