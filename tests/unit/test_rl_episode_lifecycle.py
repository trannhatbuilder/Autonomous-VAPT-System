"""R27 / PL4 — episode lifecycle guards.

The report claimed `end_episode()` was never called ("P0 blocker"). It was
wrong: `grep -rn "end_episode" app/` returns exactly three hits — a docstring,
the definition, and the call at `harness.py:1393`, reached from all six
return paths via `_finalize_rl()` -> `harness.end_scan()`.

What the report *did* miss is worse than what it claimed:

1. `end_episode()` computed an ε decay that was then thrown away, because the
   per-scan `HarnessBridge` starts from a fresh `PentestPolicy()` and
   `_save_checkpoint_safe()` wrote only Q weights. Measured across three
   consecutive scans: ε at turn 1 was 1.000000 three times, and ε after
   `end_scan()` was 0.970487 three times — the W16 schedule never advanced.
   Consequence: `_should_inject_rl_guidance()` returned
   `(False, "exploratory")` on 200/200 turns of a fresh scan, i.e. the RL
   advisory channel was dead code in production.

2. `_record_terminal_transition()` ran on only 2 of the 6 return paths, so
   `error` / `aborted` / `token_budget_exceeded` episodes closed without a
   `done=True` row.

These tests lock in both fixes. The behavioural ones measure ε *by value*
across scan boundaries — a source-level `assert "state_dict" in src` cannot
detect a value that is written but never read back.
"""

from __future__ import annotations

import ast
import asyncio
import json
import tempfile
from pathlib import Path

import pytest

from app.rl.harness import HarnessBridge
from app.rl.policy import DEFAULT_EPSILON_END, PentestPolicy

REPO = Path(__file__).resolve().parents[2]
HARNESS_SRC = (REPO / "app" / "rl" / "harness.py").read_text(encoding="utf-8")
POLICY_SRC = (REPO / "app" / "rl" / "policy.py").read_text(encoding="utf-8")
AGENT_SRC = (REPO / "app" / "agents" / "react_agent.py").read_text(encoding="utf-8")


# ─── isolation ────────────────────────────────────────────────────────


@pytest.fixture
def rl_dir(tmp_path, monkeypatch):
    """Redirect the .npz cache AND the new policy_state.json into tmp.

    The state path is derived from `q_learner.WEIGHTS_FILE` precisely so
    that this single patch isolates both — if that derivation is broken the
    tests below start writing into the repo's data/rl and leak into each
    other, which is itself a signal.
    """
    import app.rl.q_learner as qmod

    monkeypatch.setattr(qmod, "WEIGHTS_FILE", tmp_path / "q_weights.npz")
    monkeypatch.setattr(qmod, "META_FILE", tmp_path / "q_meta.json")
    monkeypatch.setattr(qmod, "META_DIR", tmp_path, raising=False)
    return tmp_path


# ─── 1. the premise, locked: end_episode IS called ────────────────────


class TestEndEpisodeIsWired:
    def test_end_scan_calls_end_episode(self):
        """AST, not substring: a comment mentioning the name must not pass."""
        tree = ast.parse(HARNESS_SRC)
        calls = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                calls.add(node.func.attr)
        assert "end_episode" in calls, (
            "HarnessBridge.end_scan() no longer calls policy.end_episode() — "
            "the ε schedule is dead."
        )

    def test_end_episode_is_the_only_definition(self):
        assert POLICY_SRC.count("def end_episode") == 1


# ─── 2. ε / T / episode counter survive the scan boundary ─────────────


class TestExplorationSchedulePersists:
    def test_policy_state_dict_round_trips(self):
        """By value, not by source inspection."""
        p = PentestPolicy()
        p.end_episode(1.0)
        saved_eps, saved_T, saved_ep = p.epsilon, p.temperature, p.get_stats()["episodes"]

        q = PentestPolicy()
        assert q.epsilon == 1.0, "fresh policy must start at epsilon_start"
        assert q.load_state_dict(p.state_dict()) is True
        assert q.epsilon == pytest.approx(saved_eps)
        assert q.temperature == pytest.approx(saved_T)
        assert q.get_stats()["episodes"] == saved_ep == 1

    def test_load_state_dict_clamps_to_configured_range(self):
        """A hand-edited file must not push ε outside the schedule."""
        p = PentestPolicy(epsilon_end=0.1, epsilon_start=0.9)
        p.load_state_dict(
            {"epsilon": 5.0, "temperature": 99.0, "episode": 3,
             "episode_rewards": []}
        )
        assert p.epsilon == pytest.approx(0.9)
        assert p.temperature <= p.temperature_start

        p.load_state_dict(
            {"epsilon": -1.0, "temperature": -1.0, "episode": 3,
             "episode_rewards": []}
        )
        assert p.epsilon == pytest.approx(0.1)
        assert p.temperature >= p.temperature_end

    @pytest.mark.parametrize(
        "bad",
        [
            {},
            {"epsilon": "nan-not-a-float"},
            {"epsilon": 0.5, "temperature": 0.5, "episode": "x"},
            {"epsilon": float("nan"), "temperature": 0.5, "episode": 1},
            {"epsilon": float("inf"), "temperature": 0.5, "episode": 1},
        ],
    )
    def test_load_state_dict_rejects_bad_payloads(self, bad):
        p = PentestPolicy()
        before = (p.epsilon, p.temperature, p.get_stats()["episodes"])
        assert p.load_state_dict(bad) is False, (
            "a malformed state file was accepted — a half-restored policy is "
            "worse than a fresh one."
        )
        assert (p.epsilon, p.temperature, p.get_stats()["episodes"]) == before

    def test_episode_rewards_deque_is_restored(self):
        """Needed for the reward-adaptive ε branch in end_episode()."""
        p = PentestPolicy()
        for r in [10.0] * 5:
            p.end_episode(r)
        q = PentestPolicy()
        q.load_state_dict(p.state_dict())
        assert list(q._episode_rewards) == [10.0] * 5

    @pytest.mark.asyncio
    async def test_second_harness_sees_decayed_epsilon(self, rl_dir):
        """THE regression test: three scans, ε must be a chain, not a constant.

        Pre-fix measurement was [0.970487, 0.970487, 0.970487] because every
        scan began at 1.0. The assertion is on the *trend*, so it does not
        hard-code the decay constant.
        """
        first_turns: list[float] = []
        for i in range(3):
            h = HarnessBridge(scan_id=f"chain-{i}", target="https://example.com")
            first_turns.append(h.epsilon)
            await h.end_scan(episode_reward=1.0)

        assert first_turns[0] == pytest.approx(1.0), (
            "scan 1 must still start at epsilon_start — nothing to restore yet"
        )
        assert first_turns == sorted(first_turns, reverse=True), (
            f"ε did not decay across scans: {first_turns}. A constant here "
            "means the schedule is being recomputed and discarded each scan."
        )
        assert first_turns[2] < first_turns[1] < first_turns[0], (
            f"ε chain is not strictly decreasing: {first_turns}"
        )

    @pytest.mark.asyncio
    async def test_episode_counter_accumulates_across_scans(self, rl_dir):
        hackathon = []
        for i in range(3):
            h = HarnessBridge(scan_id=f"ep-{i}", target="https://example.com")
            hackathon.append(h.get_stats()["episodes"])
            await h.end_scan(episode_reward=1.0)
        assert hackathon == [0, 1, 2], (
            f"episodes counter did not accumulate: {hackathon} — the proposed "
            "maturity gate `stats['episodes'] >= N` would never open."
        )

    @pytest.mark.asyncio
    async def test_state_file_is_written_and_parseable(self, rl_dir):
        h = HarnessBridge(scan_id="statefile", target="https://example.com")
        await h.end_scan(episode_reward=2.0)
        path = rl_dir / "policy_state.json"
        assert path.exists(), "policy_state.json was not written"
        payload = json.loads(path.read_text(encoding="utf-8"))
        for key in ("epsilon", "temperature", "episode", "episode_rewards"):
            assert key in payload, f"policy_state.json lost `{key}`"
        assert payload["episode"] == 1

    def test_state_path_derives_from_weights_file(self):
        """Isolation depends on this; hard-coding DATA_DIR would leak."""
        tree = ast.parse(HARNESS_SRC)
        fn = next(
            n for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == "_policy_state_path"
        )
        src = ast.get_source_segment(HARNESS_SRC, fn) or ""
        assert "WEIGHTS_FILE" in src, (
            "_policy_state_path() no longer derives from q_learner.WEIGHTS_FILE "
            "— tests that redirect the Q cache will leak state into data/rl."
        )

    @pytest.mark.asyncio
    async def test_corrupt_state_file_falls_back_to_defaults(self, rl_dir):
        (rl_dir / "policy_state.json").write_text("{not json", encoding="utf-8")
        h = HarnessBridge(scan_id="corrupt", target="https://example.com")
        assert h.epsilon == pytest.approx(1.0), (
            "a corrupt state file must degrade to the policy default, not raise"
        )

    @pytest.mark.asyncio
    async def test_missing_state_file_falls_back_to_defaults(self, rl_dir):
        h = HarnessBridge(scan_id="nostate", target="https://example.com")
        assert h.epsilon == pytest.approx(1.0)


# ─── 3. the guidance gate actually opens ──────────────────────────────


class TestGuidanceGateOpens:
    @pytest.mark.asyncio
    async def test_gate_reads_only_keys_get_stats_exposes(self, rl_dir):
        """The report's gate used `stats["episodes"]`, which did not exist."""
        h = HarnessBridge(scan_id="keys", target="https://example.com")
        try:
            stats = h.get_stats()
            for key in ("episodes", "train_steps", "q_train_steps",
                        "buffer_size", "turn_count", "epsilon", "temperature"):
                assert key in stats, (
                    f"get_stats() lost `{key}` — a maturity gate keyed on it "
                    "would raise KeyError in production, not in tests."
                )
        finally:
            await h.end_scan(episode_reward=0.0)

    @pytest.mark.asyncio
    async def test_train_steps_and_q_train_steps_are_distinct(self, rl_dir):
        """`train_steps` is per-process; `q_train_steps` is restored from disk.

        Conflating them is how `q.train_steps > 1000` becomes a vacuous gate:
        the on-disk cache already reports 617 in a cold process.
        """
        h = HarnessBridge(scan_id="distinct", target="https://example.com")
        try:
            stats = h.get_stats()
            assert stats["train_steps"] == 0
            assert stats["q_train_steps"] == h.q_train_steps
        finally:
            await h.end_scan(episode_reward=0.0)

    @pytest.mark.asyncio
    async def test_epsilon_below_one_yields_non_exploratory_picks(self, rl_dir):
        """ε=0.05 must produce greedy picks, i.e. `exploratory=False` turns.

        Pre-fix this was impossible: every scan started at ε=1.0 so every
        turn was exploratory and the gate rejected 100% of them.
        """
        (rl_dir / "policy_state.json").write_text(
            json.dumps({"epsilon": 0.05, "temperature": 0.1, "episode": 200,
                        "episode_rewards": [1.0] * 10, "scan_id": "prev"}),
            encoding="utf-8",
        )
        h = HarnessBridge(scan_id="gate", target="https://example.com")
        assert h.epsilon == pytest.approx(0.05)

        recs = [await h.select_action({}, 1) for _ in range(120)]
        non_exploratory = [r for r in recs if not r.exploratory]
        assert len(non_exploratory) > 60, (
            f"only {len(non_exploratory)}/120 picks were greedy at ε=0.05 — "
            "ε is not reaching the policy's select path."
        )
        h._stop_event.set()


# ─── 4. one done=True transition per episode ──────────────────────────


def _terminal_transition_calls() -> list[ast.Call]:
    tree = ast.parse(AGENT_SRC)
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id == "_record_terminal_transition":
                out.append(node)
    return out


class TestTerminalTransitions:
    def test_all_in_loop_exit_paths_record_a_terminal_transition(self):
        reasons = set()
        for call in _terminal_transition_calls():
            if call.args and isinstance(call.args[0], ast.Constant):
                reasons.add(call.args[0].value)
        expected = {"exit", "max_iterations", "aborted",
                    "token_budget_exceeded", "error"}
        assert expected.issubset(reasons), (
            f"missing terminal transition on return path(s): "
            f"{sorted(expected - reasons)} — those episodes close without a "
            "done=True row and the Q-target bootstraps off a phantom state."
        )

    def test_terminal_transition_precedes_finalize_rl(self):
        """Every in-loop `_record_terminal_transition(...)` must come before
        its `_finalize_rl(...)`: `end_scan()` stops the training worker.

        The preflight-error path (the first `_finalize_rl`) is exempt on
        purpose: `harness` is still None and no turn was ever recorded, so
        there is no open episode to close. That exemption is asserted
        explicitly rather than silently skipped.
        """
        tree = ast.parse(AGENT_SRC)
        rec_lines = sorted(
            n.lineno for n in ast.walk(tree)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
            and n.func.id == "_record_terminal_transition"
        )
        fin_lines = sorted(
            n.lineno for n in ast.walk(tree)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
            and n.func.id == "_finalize_rl"
        )
        assert len(rec_lines) == 5, (
            f"expected 5 terminal-transition call sites, found {len(rec_lines)}"
        )
        assert len(fin_lines) == 6, (
            f"expected 6 _finalize_rl call sites, found {len(fin_lines)}"
        )

        first_fin = fin_lines[0]
        assert not any(r < first_fin for r in rec_lines), (
            "the preflight path must stay transition-free (no episode open)"
        )
        window = AGENT_SRC.splitlines()[max(0, first_fin - 16):first_fin]
        assert any("no terminal transition here on purpose" in ln for ln in window), (
            "the preflight exemption lost its explanatory comment — the next "
            "reader will 'fix' it and add an unreachable call."
        )

        # Each later finalize must have a transition between it and the
        # previous finalize.
        for prev, cur in zip(fin_lines, fin_lines[1:]):
            assert any(prev < r < cur for r in rec_lines), (
                f"_finalize_rl at line {cur} has no "
                f"_record_terminal_transition between line {prev} and {cur} — "
                "that episode closes without a done=True row."
            )

    def test_transition_handler_is_idempotent(self):
        """Recording twice would double the terminal reward into episode_reward,
        which `end_episode()` feeds to the reward-adaptive ε branch."""
        tree = ast.parse(AGENT_SRC)
        fn = next(
            n for n in ast.walk(tree)
            if isinstance(n, ast.AsyncFunctionDef)
            and n.name == "_record_terminal_transition"
        )
        src = ast.get_source_segment(AGENT_SRC, fn) or ""
        assert "_terminal_recorded" in src, (
            "_record_terminal_transition() lost its idempotency guard"
        )
        assert "nonlocal _terminal_recorded" in src

    def test_terminal_paths_do_not_reference_unbound_loop_vars(self):
        """The preflight-error path returns before `iterations`/`total_tokens`
        are bound (line ~1176+); adding a transition call there would raise
        NameError at runtime, which `except Exception` would hide."""
        tree = ast.parse(AGENT_SRC)
        rec_lines = {
            n.lineno for n in ast.walk(tree)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
            and n.func.id == "_record_terminal_transition"
        }
        # Find first assignment of each loop variable.
        first_assign: dict[str, int] = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for tgt in node.targets:
                    if isinstance(tgt, ast.Name) and tgt.id in (
                        "iterations", "total_tokens", "commands_run",
                        "findings", "prev_verified_count",
                    ):
                        first_assign.setdefault(tgt.id, node.lineno)
        assert first_assign, "loop variables not found — parser drift"
        latest_init = max(first_assign.values())
        assert all(line > latest_init for line in rec_lines), (
            f"a terminal transition at line "
            f"{min(l for l in rec_lines if l <= latest_init)} fires before "
            f"line {latest_init}, where loop variables are still unbound"
        )


# ─── 5. the epsilon note is no longer a lie ───────────────────────────


class TestDocsMatchBehaviour:
    def test_stale_not_persisted_note_is_gone(self):
        assert "epsilon=%.4f (fresh — not persisted)" not in HARNESS_SRC, (
            "harness.py still claims epsilon is not persisted — it is now, "
            "and the log line would mislead the next reader."
        )

    def test_state_dict_documents_why(self):
        assert "state_dict" in POLICY_SRC
        tree = ast.parse(POLICY_SRC)
        fn = next(
            n for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == "state_dict"
        )
        doc = ast.get_docstring(fn) or ""
        assert "scan" in doc.lower(), (
            "state_dict() must explain that the schedule is defined in scans, "
            "otherwise someone will 'simplify' it away again."
        )
