"""Guards for B9 — the RL token-pressure feature was fed a dead denominator.

`state_encoder.from_session()` computes

    token_rem = 1 - tokens_used / token_budget

and `react_agent` passed `token_budget=TOKEN_HARD_LIMIT` (10,000,000).
A scan spends 17k–500k tokens, so `token_rem` sat at 0.95–0.998 for its
whole life — a constant input, i.e. the same class of bug as the old TDI
C dimension (which used the 10M limit as a context-window denominator).

The fix separates the two ideas that were conflated:
  * TOKEN_SOFT_LIMIT / TOKEN_HARD_LIMIT — runaway guards that abort a scan.
  * SCAN_TOKEN_BUDGET — the volume a scan is expected to spend, used as the
    RL feature's denominator so the feature actually moves.
"""
from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

APP_DIR = Path(__file__).resolve().parents[2] / "app"
AGENT_FILE = APP_DIR / "agents" / "react_agent.py"
ENCODER_FILE = APP_DIR / "rl" / "state_encoder.py"


def _agent_src() -> str:
    return AGENT_FILE.read_text(encoding="utf-8")


def _build_session_calls() -> list[ast.Call]:
    """Every production `build_session(...)` call in the agent loop."""
    tree = ast.parse(_agent_src())
    return [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "build_session"
    ]


class TestBudgetIsNotTheHardLimit:
    def test_build_session_never_gets_the_hard_limit(self):
        """The regression: no build_session(..., token_budget=TOKEN_HARD_LIMIT)."""
        src = _agent_src()
        assert "token_budget=TOKEN_HARD_LIMIT" not in src

    def test_build_session_gets_the_scan_budget(self):
        """AST, not `src.count(...)`: a comment can satisfy a text count.

        The old assertion was `src.count("token_budget=SCAN_TOKEN_BUDGET") == 3`,
        which passes if a commented-out call mentions the pair and fails if
        the real call is reformatted across lines. Parse the calls instead.
        """
        calls = _build_session_calls()
        assert len(calls) == 3, (
            f"expected 3 production call sites, got {len(calls)}"
        )
        for call in calls:
            passed = {kw.arg: kw.value for kw in call.keywords}
            assert "token_budget" in passed, (
                f"build_session at line {call.lineno} omits token_budget"
            )
            assert ast.unparse(passed["token_budget"]) == "SCAN_TOKEN_BUDGET", (
                f"line {call.lineno} passes "
                f"{ast.unparse(passed['token_budget'])} — the encoder's "
                f"fraction only moves for the SCAN budget, never a limit"
            )

    def test_scan_budget_is_well_below_the_limits(self):
        from app.agents import react_agent as ra

        assert 0 < ra.SCAN_TOKEN_BUDGET < ra.TOKEN_SOFT_LIMIT < ra.TOKEN_HARD_LIMIT

    def test_scan_budget_is_env_tunable(self, monkeypatch):
        import importlib

        from app.agents import react_agent as ra

        monkeypatch.setenv("VAPT_AI_SCAN_TOKEN_BUDGET", "123456")
        reloaded = importlib.reload(ra)
        try:
            assert reloaded.SCAN_TOKEN_BUDGET == 123_456
        finally:
            monkeypatch.delenv("VAPT_AI_SCAN_TOKEN_BUDGET", raising=False)
            importlib.reload(ra)

    def test_limits_are_untouched(self):
        from app.agents import react_agent as ra

        assert ra.TOKEN_SOFT_LIMIT == 2_500_000
        assert ra.TOKEN_HARD_LIMIT == 10_000_000


class TestEnvOverrideCannotDisableTheFeature:
    """The env door must not be able to reintroduce the dead-constant bug.

    `SCAN_TOKEN_BUDGET = int(os.getenv(...))` accepted 0 and -5 happily. The
    encoder maps `token_budget <= 0` to 0.0, so one bad env var pinned the
    feature to 0.0 on every turn — a constant input, which is precisely what
    SCAN_TOKEN_BUDGET was introduced to eliminate.
    """

    @staticmethod
    def _reload_with(monkeypatch, raw: str):
        import importlib

        from app.agents import react_agent as ra

        monkeypatch.setenv("VAPT_AI_SCAN_TOKEN_BUDGET", raw)
        try:
            return importlib.reload(ra).SCAN_TOKEN_BUDGET
        finally:
            monkeypatch.delenv("VAPT_AI_SCAN_TOKEN_BUDGET", raising=False)
            importlib.reload(ra)

    @pytest.mark.parametrize("raw", ["0", "-5", "-1"])
    def test_non_positive_values_fall_back_to_the_default(self, monkeypatch, raw):
        from app.agents.react_agent import _DEFAULT_SCAN_TOKEN_BUDGET

        assert self._reload_with(monkeypatch, raw) == _DEFAULT_SCAN_TOKEN_BUDGET

    @pytest.mark.parametrize("raw", ["2500000", "10000000", "999999999"])
    def test_values_beyond_the_soft_limit_fall_back(self, monkeypatch, raw):
        """That is the ORIGINAL bug: a denominator no scan can reach."""
        from app.agents.react_agent import _DEFAULT_SCAN_TOKEN_BUDGET

        assert self._reload_with(monkeypatch, raw) == _DEFAULT_SCAN_TOKEN_BUDGET

    def test_garbage_falls_back_instead_of_crashing_import(self, monkeypatch):
        """Import-time `int()` used to raise here and take the server with it."""
        from app.agents.react_agent import _DEFAULT_SCAN_TOKEN_BUDGET

        assert self._reload_with(monkeypatch, "abc") == _DEFAULT_SCAN_TOKEN_BUDGET

    def test_a_zero_budget_would_have_been_a_constant(self):
        """Why the guard is worth having, demonstrated by value."""
        from app.rl._session_builder import build_session
        from app.rl.state_encoder import StateEncoder

        def rem(tokens_used: int, budget: int) -> float:
            return StateEncoder.from_session(
                build_session(
                    scan_id="s", target="example.com", turn=1, commands_run=1,
                    tokens_used=tokens_used, token_budget=budget,
                    last_command="", last_vector="", last_turn_success=True,
                    findings=[],
                )
            ).token_budget_remaining

        assert {rem(t, 0) for t in (0, 1_000, 400_000)} == {0.0}
        from app.agents.react_agent import SCAN_TOKEN_BUDGET

        assert len({rem(t, SCAN_TOKEN_BUDGET) for t in (0, 1_000, 400_000)}) == 3

    def test_valid_override_still_wins(self, monkeypatch):
        assert self._reload_with(monkeypatch, "123456") == 123_456


class TestFeatureActuallyMoves:
    """The whole point: token_rem must vary across realistic token totals."""

    @staticmethod
    def _token_rem(tokens_used: int, budget: int) -> float:
        from app.rl._session_builder import build_session
        from app.rl.state_encoder import StateEncoder

        session = build_session(
            scan_id="s", target="example.com", turn=1, commands_run=1,
            tokens_used=tokens_used, token_budget=budget,
            last_command="", last_vector="", last_turn_success=True,
            findings=[],
        )
        state = StateEncoder.from_session(session)
        return state.token_budget_remaining

    def test_varies_across_a_real_scan(self):
        from app.agents.react_agent import SCAN_TOKEN_BUDGET

        early = self._token_rem(20_000, SCAN_TOKEN_BUDGET)
        mid = self._token_rem(250_000, SCAN_TOKEN_BUDGET)
        late = self._token_rem(450_000, SCAN_TOKEN_BUDGET)

        assert early > mid > late
        # Not a constant input: the feature must span a usable range.
        assert early - late > 0.5

    def test_old_denominator_was_flat(self):
        """Proof the bug was real — with 10M the feature barely moves."""
        old = [self._token_rem(t, 10_000_000) for t in (17_000, 250_000, 500_000)]
        assert max(old) - min(old) < 0.05

    def test_bounds_respected(self):
        from app.agents.react_agent import SCAN_TOKEN_BUDGET

        assert self._token_rem(0, SCAN_TOKEN_BUDGET) == 1.0
        assert self._token_rem(SCAN_TOKEN_BUDGET * 2, SCAN_TOKEN_BUDGET) == 0.0


class TestEncoderMathUnchanged:
    def test_from_session_still_clamps(self):
        src = ENCODER_FILE.read_text(encoding="utf-8")
        assert "token_budget = session.get(\"_token_budget\"" in src
        assert "max(0.0, 1.0 - tokens_used / token_budget)" in src

    def test_feature_index_is_real(self):
        from app.rl._session_builder import build_session
        from app.rl.state_encoder import StateEncoder

        def _vec(tokens_used: int):
            return StateEncoder.from_session(
                build_session(
                    scan_id="s", target="example.com", turn=1, commands_run=1,
                    tokens_used=tokens_used, token_budget=500_000,
                    last_command="", last_vector="", last_turn_success=True,
                    findings=[],
                )
            ).to_vector()

        assert isinstance(_vec(0), np.ndarray)
        assert _vec(0).shape == _vec(500_000).shape
        assert not np.array_equal(_vec(0), _vec(500_000))
