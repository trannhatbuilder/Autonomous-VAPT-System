"""
Unit tests — EGATS outcome attribution must follow the INJECTED hint (B5).

The bug: `egats_select()` runs every turn and used to stash
`_egats_last_path_key` unconditionally, while the advisory round-robin
injects the EGATS message on roughly one turn in three (and the RL shadow
mode can shift the slot again). So on ~2/3 of turns the path credited by
`egats_record_outcome()` was a path whose hint was never placed in the
prompt — UCB attempts/successes, the TDI EMA and the pruning decisions
were all driven by round-robin bookkeeping instead of by evidence.

The fix separates two slots:
  `_egats_selected_path_key` — what UCB picked (display / metrics)
  `_egats_injected_path_key` — what the LLM actually saw (eligible for
                               backprop, cleared at every turn boundary by
                               `egats_select()` and consumed on use)
"""
from __future__ import annotations

import ast
import asyncio
import contextlib
from pathlib import Path

import pytest
import pytest_asyncio

from app.agents.react_agent import _select_advisory
from app.rl.harness import HarnessBridge

REPO_ROOT = Path(__file__).resolve().parents[2]
AGENT_FILE = REPO_ROOT / "app" / "agents" / "react_agent.py"
HARNESS_FILE = REPO_ROOT / "app" / "rl" / "harness.py"

PATH_A = ["tech:nginx", "cve:2021-1234"]
PATH_B = ["tech:nginx", "cve:2022-9999"]


class _FakeKG:
    """Minimal stand-in for KnowledgeGraph.get_attack_paths()."""

    def __init__(self, paths):
        self._paths = paths

    def get_attack_paths(self, source_node_ids, *, max_depth=4, top_k=10, target_kind="Finding"):
        # Dict form accepted by EGATS.seed_from_kg (a bare list is skipped
        # because it has no `.nodes` and is not a dict).
        return [
            {"nodes": list(p), "edges": [], "score": 0.5, "match_specificity": 0.0}
            for p in self._paths
        ]


@pytest_asyncio.fixture
async def bridge(monkeypatch):
    # HarnessBridge spawns a training worker in __init__ -> needs a loop.
    monkeypatch.setenv("VAPT_AI_TDA", "1")
    instance = HarnessBridge(scan_id="scan_egats_attr", target="http://t.test")
    try:
        instance.egats_seed(_FakeKG([PATH_A, PATH_B]), ["tech:nginx"])
        assert instance._egats_seeded is True
        yield instance
    finally:
        task = getattr(instance, "_train_worker_task", None)
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task


def _attempts(bridge: HarnessBridge, nodes: list[str]) -> int:
    return bridge._egats._stats[tuple(nodes)].attempts


def _successes(bridge: HarnessBridge, nodes: list[str]) -> int:
    return bridge._egats._stats[tuple(nodes)].successes


class TestAttribution:
    def test_seed_created_paths(self, bridge):
        assert len(bridge._egats.stats()) == 2

    def test_selected_path_is_recorded_for_display(self, bridge):
        result = bridge.egats_select()
        assert result.path is not None
        assert bridge._egats_selected_path_key == result.path.key()

    def test_outcome_without_injection_is_dropped(self, bridge):
        """The B5 regression: selecting is not the same as injecting."""
        bridge.egats_select()
        bridge.egats_record_outcome(success=True, tdi=0.4)

        assert _attempts(bridge, PATH_A) == 0
        assert _successes(bridge, PATH_A) == 0
        assert _attempts(bridge, PATH_B) == 0

    def test_outcome_after_injection_is_credited(self, bridge):
        bridge.egats_select()
        assert bridge.egats_mark_hint_injected() is True
        bridge.egats_record_outcome(success=True, tdi=0.4)

        assert _attempts(bridge, PATH_A) == 1
        assert _successes(bridge, PATH_A) == 1

    def test_failure_is_credited_as_attempt_without_success(self, bridge):
        bridge.egats_select()
        bridge.egats_mark_hint_injected()
        bridge.egats_record_outcome(success=False, tdi=0.7)

        assert _attempts(bridge, PATH_A) == 1
        assert _successes(bridge, PATH_A) == 0

    def test_credit_cannot_leak_across_turns(self, bridge):
        """Turn 1 injects; turn 2 selects but does not inject."""
        bridge.egats_select()
        bridge.egats_mark_hint_injected()

        bridge.egats_select()               # new turn boundary
        bridge.egats_record_outcome(success=True, tdi=0.4)

        assert _attempts(bridge, PATH_A) == 0

    def test_one_injection_counts_as_one_attempt(self, bridge):
        """A turn with several record_vulnerability calls is one attempt."""
        bridge.egats_select()
        bridge.egats_mark_hint_injected()
        bridge.egats_record_outcome(success=True, tdi=0.4)
        bridge.egats_record_outcome(success=True, tdi=0.4)

        assert _attempts(bridge, PATH_A) == 1

    async def test_mark_without_selection_returns_false(self, monkeypatch):
        """Unseeded EGATS → select() is None → nothing can be armed."""
        from app.rl.harness import HarnessBridge as HB

        instance = HB(scan_id="scan_egats_unseeded", target="http://t.test")
        try:
            assert instance.egats_select() is None
            assert instance.egats_mark_hint_injected() is False
            # And the outcome is a no-op rather than a crash.
            instance.egats_record_outcome(success=True, tdi=0.5)
        finally:
            task = getattr(instance, "_train_worker_task", None)
            if task is not None:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await task


class TestRoundRobinSimulation:
    """End-to-end through `_select_advisory`: credit only injected turns."""

    def test_only_one_turn_in_three_can_be_credited(self, bridge):
        """6 turns, RL suppressed (shadow mode) → EGATS takes 2 of them."""
        egats_msg = {"role": "system", "content": "EGATS hint"}
        tda_msg = {"role": "system", "content": "TDA hint"}
        injected = 0

        for turn in range(1, 7):
            bridge.egats_select()           # runs EVERY turn
            picked, _msg = _select_advisory(
                turn, rl_msg=None, tda_msg=tda_msg, egats_msg=egats_msg,
            )
            if picked == "egats":
                injected += 1
                bridge.egats_mark_hint_injected()
            bridge.egats_record_outcome(success=True, tdi=0.4)

        # Old code: 6 attempts (one per turn). New code: only the injected
        # ones. Total across paths, because a credited path stops being
        # +inf-UCB and the planner rotates to the next unattempted one.
        assert injected == 2
        assert _attempts(bridge, PATH_A) + _attempts(bridge, PATH_B) == injected

    def test_before_the_fix_every_turn_would_have_counted(self, bridge):
        """Pins the magnitude of the bug: attempts == turns, not injections."""
        egats_msg = {"role": "system", "content": "EGATS hint"}
        tda_msg = {"role": "system", "content": "TDA hint"}

        turns = 0
        for turn in range(1, 7):
            bridge.egats_select()
            turns += 1
            picked, _ = _select_advisory(
                turn, rl_msg=None, tda_msg=tda_msg, egats_msg=egats_msg,
            )
            # Deliberately do NOT call egats_mark_hint_injected()
            bridge.egats_record_outcome(success=True, tdi=0.4)

        assert turns == 6
        assert _attempts(bridge, PATH_A) == 0, (
            "without an injected hint nothing may be backpropagated"
        )


class TestCallSiteWiring:
    def test_react_agent_marks_injection(self):
        src = AGENT_FILE.read_text(encoding="utf-8")
        assert "harness.egats_mark_hint_injected()" in src

    def test_mark_is_guarded_by_identity_not_string(self):
        src = AGENT_FILE.read_text(encoding="utf-8")
        assert "advisory_msg is egats_hint_msg" in src
        assert 'picked_advisory == "egats"' not in src

    def test_mark_runs_inside_the_injected_branch(self):
        """AST: the call sits in `if advisory_msg is not None:`."""
        tree = ast.parse(AGENT_FILE.read_text(encoding="utf-8"))
        found = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.If):
                continue
            if "advisory_msg is not None" not in ast.unparse(node.test):
                continue
            for sub in ast.walk(node):
                if (
                    isinstance(sub, ast.Call)
                    and isinstance(sub.func, ast.Attribute)
                    and sub.func.attr == "egats_mark_hint_injected"
                ):
                    found.append(sub)
        assert len(found) == 1, "egats_mark_hint_injected must be gated by injection"

    def test_sse_announcement_is_gated_too(self):
        """The UI must not claim a path the model never saw."""
        tree = ast.parse(AGENT_FILE.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.If):
                continue
            if "advisory_msg is not None" not in ast.unparse(node.test):
                continue
            body = ast.unparse(node)
            assert "egats_mark_hint_injected" in body
            assert "EGATS: path" in body

    def test_harness_backprops_the_injected_key(self):
        """`_egats_selected_path_key` must never reach record_outcome."""
        src = HARNESS_FILE.read_text(encoding="utf-8")
        tree = ast.parse(src)

        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.FunctionDef)
                and node.name == "egats_record_outcome"
            ):
                continue
            body = ast.unparse(node)
            assert "_egats_injected_path_key" in body
            assert "_egats_selected_path_key" not in body
            return
        pytest.fail("egats_record_outcome not found")

    def test_turn_boundary_clears_the_injected_key(self, bridge):
        bridge.egats_select()
        bridge.egats_mark_hint_injected()
        assert bridge._egats_injected_path_key is not None
        bridge.egats_select()
        assert bridge._egats_injected_path_key is None
