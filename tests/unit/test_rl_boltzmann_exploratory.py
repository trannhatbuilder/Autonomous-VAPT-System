"""
Regression guards for PL3 — the Boltzmann `exploratory` flag.

The report's premise here is **correct**: `_select_boltzmann` hardcoded

    "exploratory": True,

for every action, so the field carried no information at all. Measured
before the fix on a real `PentestPolicy(exploration_mode="boltzmann")`:

    T=1.0  → picked argmax 70.0% of 2000 turns | reported exploratory 100%
    T=0.1  → picked argmax 100.0% of 2000 turns | reported exploratory 100%

The second row is the damning one: at `temperature_end` the softmax is
effectively deterministic — 2000/2000 picks equal argmax — and the policy
still claimed to be exploring every single turn. That permanently disabled
the B3 shadow-mode gate (`_should_inject_rl_guidance` returns `False,
"exploratory"` before it ever looks at `train_steps`), and inflated
`HarnessBridge._exploratory_count`, which is exported as
`exploratory_count` in `/rl/stats`.

Same class, found next door: the Boltzmann meta had **no `epsilon` key**,
so `HarnessBridge.select_action()`'s
`float(meta.get("epsilon", self._policy.epsilon))` fallback fired and every
Boltzmann run reported the ε-greedy schedule — a knob this mode never
reads — as the exploration rate, in the guidance text, the `rl_select` log
and `TurnRecommendation.epsilon`.

One part of the report is REFUTED. It claimed the hardcoded flag also
meant `egats_mark_hint_injected()` "was never called" and that EGATS
attribution was therefore skewed. It is not: the EGATS call site is gated
on `advisory_msg is egats_hint_msg`, never on `exploratory`. Measured, the
effect runs the other way — with RL guidance suppressed, `_select_advisory`
falls through, so EGATS keeps its designed 1-in-3 share (100/300) and TDA
takes the slack (200/300). No starved attribution.
"""
from __future__ import annotations

import ast
import asyncio
import os
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

os.environ.setdefault("VAPT_AI_RL_HARNESS_ENABLED", "true")
os.environ.setdefault("VAPT_AI_POSTGRES_DB", "postgresql://vapt:vapt@localhost:5432/vapt_ai_test")

import numpy as np
import pytest

from app.agents.react_agent import _select_advisory, _should_inject_rl_guidance
from app.rl._session_builder import build_session
from app.rl.policy import PentestPolicy

POLICY_SRC = (_PROJECT_ROOT / "app" / "rl" / "policy.py").read_text(encoding="utf-8")
AGENT_SRC = (_PROJECT_ROOT / "app" / "agents" / "react_agent.py").read_text(encoding="utf-8")
HARNESS_SRC = (_PROJECT_ROOT / "app" / "rl" / "harness.py").read_text(encoding="utf-8")

_STATE = np.random.RandomState(7).randn(337).astype(np.float32)


def _method_src(name: str) -> str:
    """Source of a single method, via AST (survives reformatting/comments)."""
    tree = ast.parse(POLICY_SRC)
    lines = POLICY_SRC.splitlines()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return "\n".join(lines[node.lineno - 1: node.end_lineno])
    raise AssertionError(f"{name} not found in policy.py")


def _boltzmann_policy(temperature: float, *, seed: int = 7) -> PentestPolicy:
    np.random.seed(seed)
    policy = PentestPolicy(exploration_mode="boltzmann", use_curiosity=False)
    policy.temperature = temperature
    return policy


# ── 1. exploratory is a measurement ─────────────────────────────────────

class TestExploratoryIsMeasured:
    def test_exploratory_is_exactly_action_differs_from_best(self):
        """The invariant, over 500 draws at three temperatures."""
        for temperature in (1.0, 0.5, 0.1):
            policy = _boltzmann_policy(temperature)
            for _ in range(500):
                action, _, meta = policy.select_action(_STATE)
                assert meta["exploratory"] is (action != meta["best_action"])

    def test_best_action_is_the_argmax_of_q_values(self):
        policy = _boltzmann_policy(0.4)
        for _ in range(50):
            _, _, meta = policy.select_action(_STATE)
            assert meta["best_action"] == int(np.argmax(meta["q_values"]))

    def test_exploratory_is_a_real_bool(self):
        _, _, meta = _boltzmann_policy(0.4).select_action(_STATE)
        assert type(meta["exploratory"]) is bool

    def test_at_temperature_end_the_policy_is_size_zero_exploratory(self):
        """The old code reported 100% here — 2000/2000 picks were argmax."""
        policy = _boltzmann_policy(0.1)
        assert policy.temperature == policy.temperature_end
        metas = [policy.select_action(_STATE)[2] for _ in range(200)]
        assert sum(m["exploratory"] for m in metas) == 0

    def test_high_temperature_still_explores(self):
        """The fix must not swing to 'never exploratory' either."""
        policy = _boltzmann_policy(1.0)
        metas = [policy.select_action(_STATE)[2] for _ in range(400)]
        rate = sum(m["exploratory"] for m in metas) / len(metas)
        assert 0.05 < rate < 0.60, f"exploratory rate {rate:.2%} looks wrong for T=1.0"

    def test_exploratory_rate_tracks_temperature(self):
        """Hotter ⇒ more disagreement with argmax, monotonically."""
        rates = []
        for temperature in (1.0, 0.5, 0.1):
            policy = _boltzmann_policy(temperature)
            metas = [policy.select_action(_STATE)[2] for _ in range(400)]
            rates.append(sum(m["exploratory"] for m in metas) / len(metas))
        assert rates[0] > rates[1] > rates[2]


# ── 2. the missing `epsilon` key and the silent fallback ────────────────

class TestEpsilonIsReported:
    def test_boltzmann_meta_has_epsilon(self):
        _, _, meta = _boltzmann_policy(0.4).select_action(_STATE)
        assert "epsilon" in meta, (
            "Boltzmann meta lost its `epsilon` key — HarnessBridge will "
            "silently fall back to the ε-greedy schedule, which this mode "
            "never reads."
        )

    def test_epsilon_is_one_minus_probability_of_argmax(self):
        for temperature in (1.0, 0.5, 0.1):
            _, _, meta = _boltzmann_policy(temperature).select_action(_STATE)
            expected = 1.0 - meta["probs"][meta["best_action"]]
            assert meta["epsilon"] == pytest.approx(expected, abs=5e-5)
            assert 0.0 <= meta["epsilon"] <= 1.0

    def test_meta_schema_matches_epsilon_greedy(self):
        """Both modes must emit the keys the harness reads unconditionally."""
        _, _, greedy = PentestPolicy(use_curiosity=False).select_action(_STATE)
        _, _, boltz = _boltzmann_policy(0.4).select_action(_STATE)
        required = {"q_values", "selected_action", "best_action", "epsilon",
                    "exploratory", "curiosity_bonus", "mode"}
        assert required <= set(greedy)
        assert required <= set(boltz)

    def test_harness_takes_epsilon_from_meta_not_from_the_policy(self):
        """End-to-end: a sentinel ε on the policy must not leak into the rec."""
        from app.rl.harness import HarnessBridge, release_harness

        async def run():
            harness = HarnessBridge(scan_id="pl3-boltzmann", target="https://example.com")
            try:
                policy = _boltzmann_policy(0.1)
                policy.epsilon = 0.0123           # sentinel: the ε-greedy knob
                harness._policy = policy
                session = build_session(
                    scan_id="pl3-boltzmann",
                    target="https://example.com",
                    turn=1,
                    commands_run=0,
                    tokens_used=0,
                    token_budget=10_000_000,
                    last_command="",
                    last_vector="",
                    last_turn_success=False,
                    findings=[],
                )
                rec = await harness.select_action(session, turn=1)
                return rec
            finally:
                try:
                    await asyncio.wait_for(harness.end_scan(episode_reward=0.0), timeout=3.0)
                except Exception:  # noqa: BLE001 — teardown must never mask the assertion
                    pass
                release_harness("pl3-boltzmann")

        rec = asyncio.run(run())
        assert rec.epsilon == pytest.approx(rec.meta["epsilon"])
        assert rec.epsilon != pytest.approx(0.0123)


# ── 3. the B3 gate is unblocked ─────────────────────────────────────────

class TestShadowModeGateUnblocked:
    def test_deterministic_boltzmann_pick_is_injectable(self):
        """At T_end every pick is greedy, so the gate must now allow them."""
        policy = _boltzmann_policy(0.1)
        meta = policy.select_action(_STATE)[2]
        rec = _rec_from_meta(meta, epsilon=meta["epsilon"])
        assert _should_inject_rl_guidance(rec, train_steps=1) == (True, "trained")

    def test_exploratory_boltzmann_pick_is_still_suppressed(self):
        """Find a genuinely non-greedy draw at T=1.0 and check it is hidden."""
        policy = _boltzmann_policy(1.0)
        for _ in range(200):
            meta = policy.select_action(_STATE)[2]
            if meta["exploratory"]:
                rec = _rec_from_meta(meta, epsilon=meta["epsilon"])
                assert _should_inject_rl_guidance(rec, train_steps=500) == (
                    False, "exploratory",
                )
                return
        raise AssertionError("no exploratory draw in 200 attempts at T=1.0")


def _rec_from_meta(meta: dict, *, epsilon: float):
    from app.rl.harness import TurnRecommendation

    return TurnRecommendation(
        action_index=meta["selected_action"],
        action_name="execute_command_recon",
        q_values=meta["q_values"],
        q_max=float(max(meta["q_values"])),
        epsilon=epsilon,
        exploratory=meta["exploratory"],
        curiosity_bonus=meta["curiosity_bonus"],
        guidance_text="## RL Guidance",
        meta=meta,
    )


# ── 4. the refuted half of the report: EGATS is not starved ─────────────

class TestEgatsClaimIsRefuted:
    @staticmethod
    def _share(rl, tda, egats, turns: int = 300) -> dict[str, int]:
        counts: dict[str, int] = {"rl": 0, "tda": 0, "egats": 0, "none": 0}
        for iteration in range(1, turns + 1):
            name, _ = _select_advisory(
                iteration, rl_msg=rl, tda_msg=tda, egats_msg=egats,
            )
            counts[name or "none"] += 1
        return counts

    def test_suppressing_rl_does_not_starve_egats(self):
        """It used to be claimed that this made egats_mark_hint_injected()
        unreachable. Suppression only changes the fall-through order."""
        msg = {"role": "system", "content": "x"}
        baseline = self._share(msg, msg, msg)
        suppressed = self._share(None, msg, msg)
        assert baseline["egats"] == 100
        assert suppressed["egats"] == 100, (
            "EGATS share changed when RL was suppressed — the claim that the "
            "rotation starves EGATS is false."
        )
        assert suppressed["tda"] == 200   # the slack goes here, not to a void

    def test_egats_marking_cannot_be_gated_by_exploratory(self):
        """`egats_mark_hint_injected()` takes no exploratory argument, so no
        value of that flag can make it "never called"."""
        tree = ast.parse(HARNESS_SRC)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "egats_mark_hint_injected":
                args = [a.arg for a in node.args.args]
                assert args == ["self"], args
                return
        raise AssertionError("egats_mark_hint_injected not found")

    def test_rl_suppression_never_yields_an_empty_advisory_turn(self):
        """With TDA present there is always something to inject."""
        msg = {"role": "system", "content": "x"}
        assert self._share(None, msg, msg)["none"] == 0


# ── 5. source guards ────────────────────────────────────────────────────

class TestSourceCannotRegress:
    def test_exploratory_is_not_hardcoded(self):
        src = _method_src("_select_boltzmann")
        assert '"exploratory": True' not in src
        assert "action != best_a" in src

    def test_best_action_is_reported(self):
        src = _method_src("_select_boltzmann")
        assert '"best_action": best_a' in src

    def test_epsilon_key_is_reported(self):
        src = _method_src("_select_boltzmann")
        assert '"epsilon"' in src
        assert "probs[best_a]" in src

    def test_harness_fallback_still_exists_and_is_why_the_key_matters(self):
        """If someone removes the fallback, this guard should be revisited —
        not silently kept."""
        assert 'meta.get("epsilon", self._policy.epsilon)' in HARNESS_SRC

    def test_react_agent_no_longer_cites_the_pre_R24_formula(self):
        """The gate's rationale asserted that `epsilon + curiosity_bonus` was
        "pinned at 1.0". The bonus has been schedule-scaled since R24, so that
        is only true early in a run and cannot be the standing justification."""
        tree = ast.parse(AGENT_SRC)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "_should_inject_rl_guidance":
                doc = ast.get_docstring(node) or ""
                assert "pinned at 1.0" not in doc, (
                    "the shadow-mode docstring still claims ε is pinned at 1.0 "
                    "for the whole run — false since the bonus became "
                    "schedule-scaled in R24."
                )
                assert "SCALED" in doc, "the corrected rationale is missing"
                return
        raise AssertionError("_should_inject_rl_guidance not found")

    def test_egats_call_site_is_not_gated_on_exploratory(self):
        """Source-level: the statement that arms EGATS must not read the flag."""
        lines = AGENT_SRC.splitlines()
        call_idx = next(
            i for i, line in enumerate(lines) if "harness.egats_mark_hint_injected()" in line
        )
        window = "\n".join(lines[call_idx - 14: call_idx + 1])
        assert "exploratory" not in window
