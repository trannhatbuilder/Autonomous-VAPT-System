"""Round 10 (R1) — the finding bonus must actually fire.

Three bugs, one family: **reward bookkeeping that does not survive the
call boundary.**

R1. `react_agent` advanced `prev_findings_count` / `prev_verified_count`
    *before* calling `compute_reward()`, so inside `compute_reward()`
    `current_verified > previous_verified_count` was **always False** and
    `finding_bonus` never fired at all. Recording a confirmed SQLi scored
    identically to doing nothing (+0.195 either way, vs +20.195 with the
    fix). `last_turn_success` on the line above compared *before* the
    overwrite — proving the intent was "prev = before this turn" and the
    update was simply in the wrong place.

R2. `build_session()` returns a **fresh dict every turn**, but
    `compute_reward()` stored its payout memory (`_credited_verified_ids`,
    `_seen_finding_keys`, `_penalised_fp_ids`) on the session. So once R1
    was fixed, an already-paid finding would be paid again on the next
    turn the gate re-opened (+20 -> +40), and the same false positive
    would be penalised on every remaining turn of the scan.

The fix is structural: that state belongs to the SCAN, so the caller owns
it (`new_credit_state()`) and passes it as `credit_state=`. Keeping it on
the session only works when one session == one whole episode.

R3. `_is_novel_finding()` used the same per-turn set, so everything looked
    novel — every finding would earn the +20 novelty tier rather than +1.

R2/R3 were unreachable while R1 was broken (the gate never opened), which
is exactly why they had to be fixed in the same change: fixing R1 alone
would have turned a dead branch into a double-paying one.
"""

from __future__ import annotations

import ast
import logging
import re
from pathlib import Path

import pytest

from app.rl._session_builder import build_session
from app.rl.reward import (
    D26_EVENT_REWARDS,
    REWARD_CREDIT_KEYS,
    R_EFFICIENT,
    R_FOLLOWED_NO_FINDING,
    R_FOLLOWED_VERIFIED,
    R_FP_REJECTED,
    R_NOVEL_FINDING,
    R_SCOPE_VIOLATION,
    R_TIMEOUT,
    R_VERIFIED_FINDING,
    R_WASTED_TURN,
    USEFUL_OUTPUT_MIN_CHARS,
    WEIGHT_HUMAN_LABEL,
    compute_reward,
    d26_event,
    new_credit_state,
)

AGENT_FILE = Path(__file__).resolve().parents[2] / "app" / "agents" / "react_agent.py"


def _finding(name: str, *, verified: bool = True) -> dict:
    """A COHERENT finding: the canonical level agrees with the legacy flag.

    SE1 made `is_verified_finding()` canonical-first, so
    `verification="confirmed"` + `verified=False` is a CONFIRMED finding and
    is paid — the level wins over the stale boolean, exactly as
    `tda.evidence_score_from_finding()` has always resolved it. The old
    fixture hardcoded `verification": "confirmed"` for both arms, which
    described a state the pipeline cannot produce: `finding_to_state()`
    derives `verified` FROM `verification`. Keep the two in sync here so the
    fixture keeps testing what its name claims.
    """
    return {
        "name": name,
        "location": f"/{name}",
        "severity": "high",
        "verification": "confirmed" if verified else "plausible",
        "verified": verified,
        "false_positive": False,
    }


def _session(findings: list[dict]) -> dict:
    return build_session(
        scan_id="scan-r1",
        target="http://target",
        turn=1,
        commands_run=3,
        tokens_used=1000,
        token_budget=500_000,
        last_command="sqlmap",
        last_vector="exploit",
        last_turn_success=True,
        findings=findings,
    )


def _reward(
    session: dict,
    prev_findings: int,
    prev_verified: int,
    credit_state: dict | None = None,
) -> tuple[float, dict]:
    # `prev_findings` is NOT forwarded: compute_reward() no longer takes a
    # findings count (see the §2 note in reward.py — a count cannot tell a
    # confirmed server from a rejected FP, so §2/§8 key off the verified
    # count plus identity sets). It stays in the signature so every call
    # site still spells out the scenario it claims to test, and is read by
    # the sanity check below rather than being dead weight.
    assert prev_findings >= prev_verified, (
        f"scenario is incoherent: {prev_verified} verified findings out of "
        f"only {prev_findings} findings"
    )
    return compute_reward(
        session=session,
        previous_verified_count=prev_verified,
        # Long enough to dodge the wasted-turn penalty; keep the other
        # signals out of the way so we can read finding_bonus cleanly.
        command_output="x" * 200,
        tokens_used_delta=0,
        is_done=False,
        action_followed=None,
        credit_state=credit_state,
    )


# ── the invariant R1 violated ─────────────────────────────────────────


class TestFindingBonusFiresOnNewVerifiedFinding:
    def test_first_verified_finding_is_paid(self):
        _, breakdown = _reward(_session([_finding("sqli")]), 0, 0)
        # Novel in this session -> R_NOVEL_FINDING (20) x rule weight (1)
        assert breakdown["finding_bonus"] > 0.0

    def test_payout_at_least_the_base_verified_reward(self):
        """Even the non-novel tier must be worth R_VERIFIED_FINDING."""
        _, breakdown = _reward(_session([_finding("sqli")]), 0, 0)
        assert breakdown["finding_bonus"] >= R_VERIFIED_FINDING

    def test_no_payout_when_nothing_was_added(self):
        _, breakdown = _reward(_session([_finding("sqli")]), 1, 1)
        assert breakdown["finding_bonus"] == 0.0

    def test_finding_versus_idle_turn_are_now_distinguishable(self):
        """The actual symptom: +0.195 vs +0.195 before the fix."""
        found, _ = _reward(_session([_finding("sqli")]), 0, 0)
        idle, _ = _reward(_session([]), 0, 0)
        assert found > idle + 5.0

    def test_second_finding_is_paid_on_its_own_turn(self):
        _, breakdown = _reward(
            _session([_finding("sqli"), _finding("xss")]), 1, 1
        )
        assert breakdown["finding_bonus"] > 0.0

    def test_unverified_finding_is_not_paid(self):
        _, breakdown = _reward(_session([_finding("sqli", verified=False)]), 0, 0)
        assert breakdown["finding_bonus"] == 0.0


class TestCreditStateSurvivesFreshSessions:
    """R2/R3: payout memory must outlive a single `session` dict."""

    def test_credited_id_survives_a_fresh_session(self):
        credit = new_credit_state()
        a = _finding("sqli")

        _, first = _reward(_session([a]), 0, 0, credit_state=credit)
        assert first["finding_bonus"] > 0.0
        assert credit["_credited_verified_ids"], "nothing was recorded as paid"

        # Next turn: a NEW finding appears, so the gate re-opens. `a` must
        # not be paid again.
        _, second = _reward(
            _session([a, _finding("xss")]), 1, 1, credit_state=credit
        )
        assert second["finding_bonus"] == pytest.approx(first["finding_bonus"]), (
            "the previously-credited finding was paid a second time — the "
            "credit state is not surviving build_session()"
        )

    def test_without_credit_state_the_double_payment_reproduces(self):
        """Pins the R2 bug, so the guard cannot be removed silently."""
        a = _finding("sqli")
        _, first = _reward(_session([a]), 0, 0)
        # No credit_state= — exactly the old behaviour.
        _, second = _reward(_session([a, _finding("xss")]), 1, 1)
        assert second["finding_bonus"] == pytest.approx(2 * first["finding_bonus"])

    def test_novelty_memory_survives(self):
        """R3: an already-seen key must not earn the +20 novelty tier again.

        Two findings with the same name+location but different severity
        have DIFFERENT identities (so both get paid) yet the SAME novelty
        key, so the second must fall back to R_VERIFIED_FINDING.
        """
        credit = new_credit_state()
        high = _finding("sqli")
        medium = {**_finding("sqli"), "severity": "medium"}

        _, first = _reward(_session([high]), 0, 0, credit_state=credit)
        assert first["finding_bonus"] == pytest.approx(R_NOVEL_FINDING)

        _, second = _reward(
            _session([high, medium]), 1, 1, credit_state=credit
        )
        assert second["finding_bonus"] == pytest.approx(R_VERIFIED_FINDING), (
            "novelty memory reset, so a repeat name+location earned the "
            "novelty tier instead of the redundant tier"
        )


class TestFalsePositiveIsPenalisedOnce:
    def _fp(self) -> dict:
        return {
            "name": "bogus",
            "location": "/bogus",
            "severity": "low",
            "verification": "plausible",
            "verified": False,
            "false_positive": True,
        }

    def test_penalty_fires_then_stops(self):
        credit = new_credit_state()
        penalties = []
        for _ in range(3):
            _, breakdown = _reward(
                _session([self._fp()]), 0, 0, credit_state=credit
            )
            penalties.append(breakdown["fp_penalty"])
        assert penalties[0] == pytest.approx(R_FP_REJECTED), "first sighting must be penalised"
        assert penalties[1] == 0.0, "the same FP was penalised again"
        assert penalties[2] == 0.0
        assert credit["_penalised_fp_ids"] == {"bogus|/bogus|low"}

    def test_without_credit_state_the_penalty_repeats(self):
        penalties = [
            _reward(_session([self._fp()]), 0, 0)[1]["fp_penalty"]
            for _ in range(3)
        ]
        assert penalties == pytest.approx([R_FP_REJECTED] * 3)


class TestFalsePositiveAttribution:
    """Round 14 (R5) — the FP penalty must be attributed by IDENTITY.

    `findings[-new_fps:]` assumed the newly-flagged false positives were the
    LAST entries. They are not: a human marks an old finding from the UI, and
    the Phase-5 auditor rejects something recorded turns earlier. The slice
    then charged the penalty to whatever sat at the tail, so
    `WEIGHT_HUMAN_LABEL` (3x) landed on a finding that was never rejected.

    Measured on the pre-fix code, `[fp(with timestamp), ok, ok]`:
    `fp_penalty == -2.0` instead of `-6.0`.

    Selecting on `false_positive_at` (as the report suggested) would be a
    worse "fix": that field is the human-label timestamp, not the FP flag,
    and rule-based rejections (`scan_pipeline` writes `false_positive=True`
    with no timestamp) would be invisible — a permanently-zero penalty.
    """

    @staticmethod
    def _fp(name: str = "bogus", *, at: str | None = None) -> dict:
        return {
            "title": name,
            "location": f"/{name}",
            "severity": "low",
            "verification": "plausible",
            "verified": False,
            "false_positive": True,
            "false_positive_at": at,
        }

    def test_old_finding_flagged_fp_still_gets_the_human_multiplier(self):
        """The R5 symptom: the FP is FIRST in the list."""
        findings = [
            self._fp("xss", at="2026-10-17T10:00:00Z"),
            _finding("sqli"),
            _finding("headers"),
        ]
        _, breakdown = _reward(
            _session(findings), 3, 0, credit_state=new_credit_state()
        )
        assert breakdown["fp_penalty"] == pytest.approx(
            R_FP_REJECTED * WEIGHT_HUMAN_LABEL
        ), (
            "the tail slice charged the penalty to a finding that was not "
            "rejected, losing the 3x human-label weight"
        )

    def test_fp_in_the_middle_of_the_list_is_penalised(self):
        findings = [_finding("sqli"), self._fp("xss"), _finding("headers")]
        _, breakdown = _reward(
            _session(findings), 3, 0, credit_state=new_credit_state()
        )
        assert breakdown["fp_penalty"] == pytest.approx(R_FP_REJECTED)

    def test_rule_rejected_fp_without_timestamp_is_still_penalised(self):
        """Refutes the proposed `false_positive_at` filter.

        This is exactly what `scan_pipeline` writes for a finding with no
        evidence: `false_positive=True`, `false_positive_at=None`.
        """
        _, breakdown = _reward(
            _session([self._fp("no-evidence")]), 1, 0,
            credit_state=new_credit_state(),
        )
        assert breakdown["fp_penalty"] == pytest.approx(R_FP_REJECTED), (
            "filtering on false_positive_at would skip every rule-based "
            "rejection and restore a permanently-zero penalty"
        )

    def test_two_distinct_fps_are_both_penalised(self):
        findings = [self._fp("a"), _finding("ok"), self._fp("b")]
        _, breakdown = _reward(
            _session(findings), 3, 0, credit_state=new_credit_state()
        )
        assert breakdown["fp_penalty"] == pytest.approx(2 * R_FP_REJECTED)

    def test_each_fp_is_charged_exactly_once_across_turns(self):
        credit = new_credit_state()
        findings = [self._fp("a"), self._fp("b"), _finding("ok")]
        penalties = [
            _reward(_session(findings), 3, 0, credit_state=credit)[1]["fp_penalty"]
            for _ in range(3)
        ]
        assert penalties[0] == pytest.approx(2 * R_FP_REJECTED)
        assert penalties[1:] == [0.0, 0.0]

    def test_removing_a_label_does_not_make_later_fps_free(self):
        """The stale-count hole the identity set removes.

        `_prev_fp_count` was assigned INSIDE `if current_fp > prev_fp`, so
        after a label was withdrawn the counter stayed above the live count
        and `current_fp > prev_fp` could never be true again — every later
        false positive was free.
        """
        credit = new_credit_state()
        flagged = self._fp("bogus")
        cleared = {**flagged, "false_positive": False}

        _reward(_session([flagged]), 1, 0, credit_state=credit)
        _reward(_session([cleared]), 1, 0, credit_state=credit)  # label withdrawn
        _, breakdown = _reward(
            _session([self._fp("fresh")]), 1, 0, credit_state=credit
        )
        assert breakdown["fp_penalty"] == pytest.approx(R_FP_REJECTED), (
            "a false positive flagged after a label was withdrawn went "
            "unpenalised — the counter is stuck above the live count"
        )

    def test_no_fps_means_no_penalty(self):
        _, breakdown = _reward(
            _session([_finding("sqli")]), 1, 1, credit_state=new_credit_state()
        )
        assert breakdown["fp_penalty"] == 0.0


# ── Round 14 (R6): the flat per-turn bonus is a stipend, not a signal ───
#
# `reward += R_EFFICIENT` was unconditional. Two consequences:
#   * an empty turn NETTED -0.3 (-0.5 + 0.2) — the stipend cancelled 40%
#     of the wasted-turn penalty it travelled with;
#   * 100 turns of long-but-worthless output collected +20, exactly one
#     novel finding, so stalling paid as well as succeeding.
#
# Gating on a PRODUCTIVE turn fixes both and restores the contrast the
# constants advertise (0.7 spread, not 0.5).

REWARD_FILE = Path(__file__).resolve().parents[2] / "app" / "rl" / "reward.py"


def _reward_file() -> str:
    return REWARD_FILE.read_text(encoding="utf-8")


def _turn(output: str, *, is_done: bool = False, messages: list[dict] | None = None):
    """compute_reward() for one turn, events synthesized like production does."""
    from app.agents.react_agent import _d26_events_from_tool_outputs

    return compute_reward(
        session=_session([]),
        previous_verified_count=0,
        command_output=output,
        tokens_used_delta=0,
        is_done=is_done,
        action_followed=None,
        credit_state=new_credit_state(),
        turn_events=_d26_events_from_tool_outputs(messages or []),
    )


class TestProductiveTurnBonus:
    def test_empty_turn_gets_no_bonus(self):
        total, breakdown = _turn("")
        assert breakdown["efficiency_bonus"] == 0.0
        assert total == pytest.approx(R_WASTED_TURN), (
            "an empty turn must cost the full wasted-turn penalty — the "
            "stipend used to refund 40% of it"
        )

    def test_short_output_gets_no_bonus(self):
        total, breakdown = _turn("x" * (USEFUL_OUTPUT_MIN_CHARS - 1))
        assert breakdown["efficiency_bonus"] == 0.0
        assert total == pytest.approx(R_WASTED_TURN)

    def test_productive_turn_gets_the_bonus(self):
        total, breakdown = _turn("80/tcp open http nginx/1.18.0\n" * 3)
        assert breakdown["efficiency_bonus"] == pytest.approx(R_EFFICIENT)
        assert breakdown["wasted_turn"] == 0.0
        assert total == pytest.approx(R_EFFICIENT)

    def test_contrast_is_the_advertised_07(self):
        """The report's goal: empty -0.5 vs productive +0.2."""
        empty, _ = _turn("")
        productive, _ = _turn("80/tcp open http nginx/1.18.0\n" * 3)
        assert productive - empty == pytest.approx(0.7), (
            "the wasted-turn penalty is still being damped by a flat bonus"
        )

    def test_timeout_turn_gets_no_bonus(self):
        """A stall's message is LONG, so a length check alone won't stop it."""
        msg = (
            "[error] TIMEOUT: 'nmap' was killed after 300s without producing "
            "a result (stall #1).\nDO NOT retry 'nmap' with the same settings."
        )
        total, breakdown = _turn(msg, messages=[{"role": "tool", "content": msg}])
        assert breakdown["efficiency_bonus"] == 0.0
        assert total == pytest.approx(R_TIMEOUT), (
            "a timed-out turn still collected the productive-turn bonus — "
            "the one-liner `if output_len >= 50` does not stop this, because "
            "every stall message tool_bridge writes is verbose"
        )

    def test_scope_violation_turn_gets_no_bonus(self):
        msg = "SCOPE VIOLATION: target not in declared scope. Tool blocked."
        total, breakdown = _turn(msg, messages=[{"role": "tool", "content": msg}])
        assert breakdown["efficiency_bonus"] == 0.0
        assert total == pytest.approx(R_SCOPE_VIOLATION)

    def test_terminal_turn_gets_no_bonus(self):
        _, breakdown = _turn("", is_done=True)
        assert breakdown["efficiency_bonus"] == 0.0, (
            "the terminal transition passes command_output='' by design and "
            "is rewarded by the completion bonus instead"
        )

    def test_a_hundred_stalled_turns_no_longer_out_earn_a_novel_finding(self):
        msg = (
            "[empty_result] stdout: 0 bytes, stderr: 0 bytes, exit_code: 0. "
            "The tool produced nothing — possible causes: the target is "
            "behind a WAF/challenge page."
        )
        per_turn, breakdown = _turn(msg, messages=[{"role": "tool", "content": msg}])
        assert breakdown["efficiency_bonus"] == 0.0
        assert per_turn == pytest.approx(R_TIMEOUT)
        assert per_turn * 100 < 0, "stalling 100 turns must not be profitable"
        assert per_turn * 100 < R_NOVEL_FINDING, (
            "100 stalled turns now score better than one novel finding"
        )


class TestEfficiencyBonusCannotBecomeUnconditional:
    """Source guard: the stipend must stay behind a condition."""

    def _efficient_augassigns(self) -> tuple[ast.Module, list[ast.AugAssign]]:
        tree = ast.parse(_reward_file())
        nodes = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.AugAssign)
            and isinstance(node.value, ast.Name)
            and node.value.id == "R_EFFICIENT"
        ]
        return tree, nodes

    def test_bonus_is_inside_a_conditional(self):
        # One parse, one tree: nodes from a second parse are not the same
        # objects and the parent walk would never find them.
        tree, nodes = self._efficient_augassigns()
        assert len(nodes) == 1, f"expected exactly one R_EFFICIENT payout, got {len(nodes)}"

        parents: dict[ast.AST, ast.AST] = {}
        for node in ast.walk(tree):
            for child in ast.iter_child_nodes(node):
                parents[child] = node

        node: ast.AST = nodes[0]
        while node in parents:
            node = parents[node]
            if isinstance(node, ast.If):
                return
        raise AssertionError(
            "`reward += R_EFFICIENT` is unconditional — a stalled turn will "
            "again draw a stipend (+0.2/turn = one novel finding per 100 turns)"
        )

    def test_wasted_turn_and_bonus_share_one_threshold(self):
        """Both halves must read `USEFUL_OUTPUT_MIN_CHARS`, never a literal."""
        source = _reward_file()
        assert "USEFUL_OUTPUT_MIN_CHARS" in source
        assert "output_len < 50" not in source, (
            "the 50-char threshold was inlined again — the two halves can "
            "drift and a turn can be both 'too empty' and 'efficient'"
        )

    def test_no_useful_output_markers_are_wired(self):
        """Zero-yield executions carry long explanations, not short output."""
        agent_source = AGENT_FILE.read_text(encoding="utf-8")
        bridge_source = TOOL_BRIDGE_FILE.read_text(encoding="utf-8")
        for marker in ("[empty_result]", "[error] Tool execution failed:"):
            assert marker in bridge_source, (
                f"{marker} no longer appears in tool_bridge.py — "
                "_NO_USEFUL_OUTPUT_MARKERS matches this exact text"
            )
            assert marker in agent_source

    def test_empty_result_is_reported_as_a_timeout_event(self):
        from app.agents.react_agent import _d26_events_from_tool_outputs

        events = _d26_events_from_tool_outputs([{
            "role": "tool",
            "content": "[empty_result] stdout: 0 bytes, stderr: 0 bytes, "
                       "exit_code: 0. The tool produced nothing.",
        }])
        assert [e["type"] for e in events] == ["timeout"]

    def test_failed_execution_is_reported_as_a_timeout_event(self):
        from app.agents.react_agent import _d26_events_from_tool_outputs

        events = _d26_events_from_tool_outputs([{
            "role": "tool",
            "content": "[error] Tool execution failed: Binary not found: dalfox",
        }])
        assert [e["type"] for e in events] == ["timeout"]


# ── Round 15 (R7): the reward API's own surface ────────────────────────
#
# Three defects the report named, all of them about what compute_reward()
# *claims* vs what it *does*:
#
#   R7a  `previous_findings_count` was accepted, documented, passed by two
#        production call sites and ~19 tests — and never read. `turn` was in
#        exactly the same state and nobody had noticed.
#   R7b  The module docstring advertised `R_explore`, a term that does not
#        exist anywhere in the reward function (curiosity lives in policy.py
#        and acts on ε).
#   R7c  §1's if/elif chain had no `else`, so an unrecognised event type was
#        dropped without a trace — and because §4/§6 gate on `not
#        turn_events`, that turn silently stopped qualifying for the
#        productive-turn bonus too. The 8 valid types were also spelled out
#        twice, so `d26_event()` could accept a type §1 would never pay.


class TestRewardFunctionHasNoDeadParameters:
    """R7a — a parameter that is never read is a lie in the signature."""

    def _compute_reward_def(self) -> ast.FunctionDef:
        tree = ast.parse(_reward_file())
        return next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "compute_reward"
        )

    def test_every_parameter_is_read_in_the_body(self):
        fn = self._compute_reward_def()
        params = [a.arg for a in fn.args.args] + [a.arg for a in fn.args.kwonlyargs]
        loaded = {
            node.id
            for node in ast.walk(fn)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
        }
        dead = [p for p in params if p not in loaded]
        assert dead == [], (
            f"compute_reward() accepts {dead} but never reads them. A dead "
            "parameter reads as 'this is part of the decision' to every "
            "caller — `previous_findings_count` sat here for 10 rounds and "
            "`turn` was never noticed. Delete it or use it."
        )

    def test_no_caller_passes_the_removed_arguments(self):
        root = Path(__file__).resolve().parents[2]
        targets = list((root / "app").rglob("*.py")) + list(
            (root / "tests" / "unit").rglob("*.py")
        )
        offenders: list[str] = []
        for path in targets:
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"))
            except (SyntaxError, UnicodeDecodeError):
                continue
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                func = node.func
                name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
                if name != "compute_reward":
                    continue
                for kw in node.keywords:
                    if kw.arg in ("previous_findings_count", "turn"):
                        offenders.append(f"{path.name}:{node.lineno} {kw.arg}=")
        assert offenders == [], (
            f"compute_reward() no longer takes these arguments: {offenders}"
        )


class TestDocstringFormulaMatchesTheCode:
    """R7b — the formula is documentation, and documentation can lie."""

    def _module_docstring(self) -> str:
        return ast.get_docstring(ast.parse(_reward_file())) or ""

    def _formula_block(self) -> str:
        lines = self._module_docstring().splitlines()
        start = next(
            (i for i, line in enumerate(lines) if line.strip().startswith("R = ")),
            None,
        )
        assert start is not None, "the reward formula disappeared from the docstring"
        block: list[str] = []
        for line in lines[start:]:
            block.append(line)
            if "clipped to" in line:
                break
        return "\n".join(block)

    def test_formula_does_not_advertise_exploration(self):
        block = self._formula_block()
        assert "R_explore" not in block, (
            "the docstring formula still claims an exploration term; curiosity "
            "is implemented in app/rl/policy.py and acts on ε, never on the "
            "reward (this is the R7b defect)"
        )

    def test_every_term_named_in_the_formula_exists(self):
        import app.rl.reward as reward_module

        names = set(re.findall(r"\bR_[A-Z0-9_]+\b", self._formula_block()))
        assert names, "the formula names no reward constants at all"
        missing = sorted(n for n in names if not hasattr(reward_module, n))
        assert missing == [], (
            f"the formula advertises {missing}, which do not exist in "
            "app/rl/reward.py"
        )

    def test_formula_is_a_sum_of_sections(self):
        block = self._formula_block()
        assert "R_WASTED_TURN" in block and "R_EFFICIENT" in block
        assert "R_SCAN_COMPLETE" in block
        assert "D26_EVENT_REWARDS" in _reward_file()


class TestD26EventsAreNeverDroppedSilently:
    """R7c — one table, and an unknown type leaves a trace."""

    def test_table_and_payer_cannot_drift(self):
        for evt_type, base in D26_EVENT_REWARDS.items():
            _, breakdown = compute_reward(
                session={"findings": []},
                previous_verified_count=0,
                command_output="x" * 200,
                tokens_used_delta=0,
                turn_events=[d26_event(evt_type)],
            )
            assert breakdown["d26_signals"][evt_type] == base, (
                f"{evt_type} is in D26_EVENT_REWARDS but §1 did not pay it"
            )

    def test_no_hardcoded_type_literals_left_in_the_payer(self):
        source = _reward_file()
        assert 'evt_type == "' not in source, (
            "§1 is back to an if/elif chain — the type list is duplicated "
            "again and can drift from D26_EVENT_REWARDS"
        )

    def test_unknown_event_type_is_logged(self, caplog):
        with caplog.at_level(logging.DEBUG, logger="app.rl.reward"):
            _, breakdown = compute_reward(
                session={"findings": []},
                previous_verified_count=0,
                command_output="x" * 200,
                tokens_used_delta=0,
                turn_events=[{"type": "shell_obtained_lol"}],
            )
        assert breakdown["d26_signals"] == {}
        messages = [r.getMessage() for r in caplog.records]
        assert any("unknown D26 event type" in m for m in messages), (
            f"an unknown event type vanished with no trace: {messages}"
        )

    def test_unknown_event_still_never_earns_the_productive_bonus(self):
        """Garbage is not evidence of productivity (§6 requires no events)."""
        _, breakdown = compute_reward(
            session={"findings": []},
            previous_verified_count=0,
            command_output="x" * 200,
            tokens_used_delta=0,
            turn_events=[{"type": "shell_obtained_lol"}],
        )
        assert breakdown["efficiency_bonus"] == 0.0
        assert breakdown["wasted_turn"] == 0.0

    def test_constructor_rejects_anything_outside_the_table(self):
        with pytest.raises(ValueError, match="Invalid D26 event type"):
            d26_event("shell_obtained_lol")


class TestFollowBonusIsAdvisoryOnly:
    """R7d — documented as an advisor-pleasing signal, and it stays small."""

    def _call(self, *, followed: bool, verified_before: int, verified_now: int):
        findings = [
            {"name": f"f{i}", "location": f"/{i}", "severity": "high",
             "verified": True, "false_positive": False}
            for i in range(verified_now)
        ]
        return compute_reward(
            session={"findings": findings},
            previous_verified_count=verified_before,
            command_output="x" * 200,
            tokens_used_delta=0,
            action_followed=followed,
            credit_state=new_credit_state(),
        )[1]

    def test_followed_without_a_finding_costs_a_little(self):
        assert self._call(
            followed=True, verified_before=1, verified_now=1
        )["rl_follow_bonus"] == pytest.approx(R_FOLLOWED_NO_FINDING)

    def test_followed_with_a_finding_pays_a_little(self):
        assert self._call(
            followed=True, verified_before=0, verified_now=1
        )["rl_follow_bonus"] == pytest.approx(R_FOLLOWED_VERIFIED)

    def test_bonus_is_tiny_next_to_a_verified_finding(self):
        assert abs(R_FOLLOWED_VERIFIED) < R_VERIFIED_FINDING, (
            "the advisor-pleasing bonus must stay smaller than the weakest "
            "real finding reward, or it becomes worth farming"
        )

    def test_it_is_documented_as_advisory(self):
        source = _reward_file()
        assert "ADVISORY" in source, (
            "§8 pays for agreement with the policy, not for an environment "
            "outcome; that has to be stated where the term is computed"
        )


class TestNewCreditState:
    def test_keys_match_the_documented_contract(self):
        assert set(REWARD_CREDIT_KEYS) == set(new_credit_state())

    def test_is_fresh_per_call(self):
        """A module-level singleton would leak state between two scans."""
        a, b = new_credit_state(), new_credit_state()
        assert a is not b
        assert a["_credited_verified_ids"] is not b["_credited_verified_ids"]
        assert a["_seen_finding_keys"] is not b["_seen_finding_keys"]

    def test_containers_are_mutable(self):
        state = new_credit_state()
        state["_credited_verified_ids"].add("x")  # must not raise
        state["_seen_finding_keys"].add("y")
        assert state["_penalised_fp_ids"] == set()

    def test_does_not_borrow_encoder_keys(self):
        """`_turn`, `_tokens_used`, ... also start with `_`; they must not
        end up in the reward bookkeeping dict."""
        assert not set(new_credit_state()) & {"_turn", "_tokens_used", "_last_command"}

    def test_credit_state_defaults_to_none(self):
        """Single-shot callers (and existing tests) must keep working."""
        import inspect

        sig = inspect.signature(compute_reward)
        assert sig.parameters["credit_state"].default is None


# ── source guard: the ordering bug must not come back ─────────────────


class TestReActAgentOrdering:
    def _record_block(self) -> ast.If:
        """Find the `else:` branch that records the per-turn transition."""
        tree = ast.parse(AGENT_FILE.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not (
                isinstance(func, ast.Attribute)
                and func.attr == "record_experience"
            ):
                continue
            for kw in node.keywords:
                if kw.arg == "reward_breakdown" and isinstance(kw.value, ast.Name):
                    if kw.value.id == "reward_breakdown":
                        return node  # the per-turn site, not the terminal one
        raise AssertionError("could not find the per-turn record_experience call")

    def test_baseline_update_comes_after_record_experience(self):
        tree = ast.parse(AGENT_FILE.read_text(encoding="utf-8"))
        record = self._record_block()
        record_lineno = record.lineno

        updates = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in (
                    "prev_findings_count",
                    "prev_verified_count",
                ):
                    if isinstance(node.value, ast.Constant) and node.value.value == 0:
                        continue  # the initialisation at the top of the scan
                    updates.append((target.id, node.lineno))

        assert updates, (
            "no baseline advance found at all — prev_findings_count / "
            "prev_verified_count would stay 0 and every transition would "
            "re-credit every finding"
        )
        for name, lineno in updates:
            assert lineno > record_lineno, (
                f"`{name}` is advanced at line {lineno}, BEFORE the "
                f"transition is recorded at line {record_lineno}. "
                "compute_reward() then receives prev == current, so "
                "`current_verified > previous_verified_count` is always "
                "False and the finding bonus never fires."
            )

    def test_every_compute_reward_call_passes_credit_state(self):
        """`credit_state=None` falls back to the session — correct only if
        the session survives the whole episode, which it does not here."""
        tree = ast.parse(AGENT_FILE.read_text(encoding="utf-8"))
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "compute_reward"
        ]

        assert len(calls) == 2, f"expected 2 compute_reward sites, got {len(calls)}"
        for call in calls:
            kwargs = {kw.arg for kw in call.keywords}
            assert "credit_state" in kwargs, (
                f"the compute_reward() at line {call.lineno} does not pass "
                "credit_state= — build_session() hands it a fresh dict, so "
                "already-credited findings would be paid again and the same "
                "false positive penalised on every turn"
            )


# ── Round 12 (R4): D26 turn_events wiring ────────────────────────────
#
# `compute_reward(..., turn_events=...)` was implemented, unit-tested and
# re-exported from `app.rl` — but NO production caller ever passed it. All
# eight event types were dead, which meant R_TIMEOUT (-1.0) and
# R_SCOPE_VIOLATION (-3.0) never fired. A timed-out tool call therefore
# scored like a successful one, and *better* than an empty result: the
# "[error] TIMEOUT: ..." text is longer than §4's 50-char threshold, so the
# wasted-turn penalty was skipped and the +0.2 efficiency bonus was paid.
#
# The guards below lock the wiring at three layers: the call site, the
# marker strings the helper depends on, and the helper's own behaviour.

TOOL_BRIDGE_FILE = (
    Path(__file__).resolve().parents[2] / "app" / "agents" / "tool_bridge.py"
)


def _per_turn_compute_reward() -> ast.Call:
    """The per-turn site (is_done=False), not the terminal one."""
    tree = ast.parse(AGENT_FILE.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "compute_reward"
        ):
            continue
        for kw in node.keywords:
            if (
                kw.arg == "is_done"
                and isinstance(kw.value, ast.Constant)
                and kw.value.value is False
            ):
                return node
    raise AssertionError("could not find the per-turn compute_reward call")


class TestTurnEventsAreWired:
    def test_per_turn_call_passes_turn_events(self):
        call = _per_turn_compute_reward()
        kwargs = {kw.arg for kw in call.keywords}
        assert "turn_events" in kwargs, (
            f"the per-turn compute_reward() at line {call.lineno} does not "
            "pass turn_events= — reward.py §1 is then unreachable and "
            "R_TIMEOUT / R_SCOPE_VIOLATION are dead again"
        )

    def test_events_are_built_before_the_reward_call(self):
        tree = ast.parse(AGENT_FILE.read_text(encoding="utf-8"))
        call = _per_turn_compute_reward()
        builds = [
            node.lineno
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            and any(
                isinstance(t, ast.Name) and t.id == "turn_events"
                for t in node.targets
            )
        ]
        assert builds, "nothing assigns `turn_events` in the agent loop"
        assert min(builds) < call.lineno, (
            f"`turn_events` is built at line {min(builds)}, after the "
            f"compute_reward() at line {call.lineno} that consumes it"
        )

    def test_markers_still_exist_in_tool_bridge(self):
        """The helper keys off literal substrings emitted by tool_bridge.

        If someone rewords those messages the signals die silently — this
        is the cross-file half of the guard.
        """
        source = TOOL_BRIDGE_FILE.read_text(encoding="utf-8")
        agent_source = AGENT_FILE.read_text(encoding="utf-8")
        for marker in ('"[error] TIMEOUT:', '"SCOPE VIOLATION:'):
            assert marker in source, (
                f"{marker} no longer appears in tool_bridge.py — the D26 "
                "event synthesis matches on this exact text, so reword it "
                "there AND in react_agent._TIMEOUT_MARKERS / "
                "_SCOPE_VIOLATION_MARKERS"
            )
        assert "[error] TIMEOUT:" in agent_source
        assert "SCOPE VIOLATION:" in agent_source


class TestTurnEventSynthesis:
    """Behaviour of `_d26_events_from_tool_outputs`."""

    @staticmethod
    def _events(messages: list[dict]) -> list[dict]:
        from app.agents.react_agent import _d26_events_from_tool_outputs

        return _d26_events_from_tool_outputs(messages)

    def test_hard_timeout_yields_timeout_event(self):
        events = self._events([{
            "role": "tool",
            "content": "[error] TIMEOUT: 'nmap' was killed after 300s "
                       "without producing a result (stall #1).",
        }])
        assert [e["type"] for e in events] == ["timeout"]

    def test_soft_timeout_yields_timeout_event(self):
        events = self._events([{
            "role": "tool",
            "content": "[error] Timeout after 300s",
        }])
        assert [e["type"] for e in events] == ["timeout"]

    def test_scope_violation_yields_scope_event(self):
        events = self._events([{
            "role": "tool",
            "content": "SCOPE VIOLATION: target not in declared scope. Tool blocked.",
        }])
        assert [e["type"] for e in events] == ["scope_violation"]

    def test_normal_output_yields_nothing(self):
        events = self._events([{
            "role": "tool",
            "content": "80/tcp open  http  nginx 1.18.0\n443/tcp open  ssl",
        }])
        assert events == []

    def test_non_tool_and_non_string_messages_are_ignored(self):
        events = self._events([
            {"role": "assistant", "content": "[error] TIMEOUT: nope"},
            {"role": "tool", "content": None},
            {"role": "tool", "content": ["[error] TIMEOUT:"]},
        ])
        assert events == []

    def test_both_events_in_one_turn_are_reported(self):
        events = self._events([
            {"role": "tool", "content": "[error] TIMEOUT: killed"},
            {"role": "tool", "content": "SCOPE VIOLATION: nope"},
        ])
        assert [e["type"] for e in events] == ["timeout", "scope_violation"]

    def test_emitted_types_are_accepted_by_d26_event(self):
        """Anything the helper can emit must be a valid D26 type."""
        from app.rl.reward import d26_event

        for event_type in ("timeout", "scope_violation"):
            assert d26_event(event_type)["type"] == event_type


class TestTurnEventsChangeTheReward:
    """End-to-end: the synthesised events must move the scalar reward."""

    @staticmethod
    def _turn(messages: list[dict]) -> dict:
        from app.agents.react_agent import _d26_events_from_tool_outputs

        _, breakdown = compute_reward(
            session=_session([]),
            previous_verified_count=0,
            command_output="[error] TIMEOUT: 'nmap' was killed after 300s "
                           "without producing a result (stall #1).",
            tokens_used_delta=0,
            is_done=False,
            action_followed=None,
            credit_state=new_credit_state(),
            turn_events=_d26_events_from_tool_outputs(messages),
        )
        return breakdown

    def test_timeout_is_penalised(self):
        breakdown = self._turn([{"role": "tool", "content": "[error] TIMEOUT: killed"}])
        assert breakdown["d26_signals"]["timeout"] == R_TIMEOUT

    def test_timeout_turn_is_not_also_charged_a_wasted_turn(self):
        """§4 is gated on `not turn_events` — no double penalty."""
        breakdown = self._turn([{"role": "tool", "content": "[error] TIMEOUT: killed"}])
        assert breakdown["wasted_turn"] == 0.0

    def test_scope_violation_is_penalised(self):
        breakdown = self._turn([
            {"role": "tool", "content": "SCOPE VIOLATION: out of scope"},
        ])
        assert breakdown["d26_signals"]["scope_violation"] == R_SCOPE_VIOLATION

    def test_timeout_scores_worse_than_a_clean_success(self):
        timed_out = self._turn([{"role": "tool", "content": "[error] TIMEOUT: killed"}])
        clean = self._turn([{"role": "tool", "content": "80/tcp open http"}])
        assert timed_out["total"] < clean["total"], (
            "a timed-out tool call still scores at least as well as a "
            "successful one — the R4 symptom"
        )


# ══════════════════════════════════════════════════════════════════════════
# Round 17 (SB1) — `last_turn_success` must mean "a finding was VERIFIED"
# ══════════════════════════════════════════════════════════════════════════
# The builder documented two different contracts for the same boolean:
#
#   "True if the previous turn produced a NEW verified finding
#    (i.e. findings_count went up)"          ← app/rl/_session_builder.py
#
# The two halves are NOT equivalent. `len(findings)` grows for any record
# the verifier accepted, including one it only rated `plausible`; §2 of
# `compute_reward` pays 0 for that same finding. The producer implemented
# the `findings_count` half, so the state vector reported progress on
# turns that earned nothing — the state contradicting its own reward.
#
# The consumer had it right all along (`state_encoder.py`: "did last turn
# yield a verified finding?"), which is what made this a producer bug
# rather than a documentation-only typo.
BUILDER_FILE = (
    Path(__file__).resolve().parents[2] / "app" / "rl" / "_session_builder.py"
)
STATE_ENCODER_FILE = (
    Path(__file__).resolve().parents[2] / "app" / "rl" / "state_encoder.py"
)


def _builder_param_doc() -> str:
    """The `build_session()` function docstring (not the module docstring).

    Splitting on `\"\"\"` is not enough: the module docstring comes first and
    the parameter SIGNATURE also contains `last_turn_success:`, so naive
    text matching picks the wrong block and the guard then asserts against
    code instead of prose. Use the AST.
    """
    tree = ast.parse(BUILDER_FILE.read_text())
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "build_session":
            doc = ast.get_docstring(node)
            assert doc, "build_session() lost its docstring"
            return doc
    raise AssertionError("build_session() not found in _session_builder.py")


def _plausible_finding(name: str = "hdr") -> dict:
    """A RECORDED finding the verifier rated plausible — pays nothing."""
    from app.evidence.finding_state import finding_to_state

    return finding_to_state(
        {"title": name, "location": "/", "severity": "low", "vuln_type": "headers"},
        {"status": "recorded", "verification": "plausible"},
    )


def _confirmed_finding(name: str = "sqli") -> dict:
    from app.evidence.finding_state import finding_to_state

    return finding_to_state(
        {"title": name, "location": "/", "severity": "high", "vuln_type": "sqli"},
        {"status": "recorded", "verification": "confirmed"},
    )


class TestLastTurnSuccessMeansVerified:
    def test_single_definition_of_verified_is_shared(self):
        """`reward.py` must not keep its own copy of the verified rule."""
        src = REWARD_FILE.read_text()
        assert "count_verified_findings(" in src, (
            "reward.py re-implements the verified test inline; the loop, the "
            "reward and `last_turn_success` must share one definition"
        )

    def test_agent_loop_does_not_use_the_findings_count(self):
        """The producer must compare VERIFIED counts, not raw record counts."""
        src = AGENT_FILE.read_text()
        assert "findings_count > prev_findings_count" not in src, (
            "last_turn_success is back to `findings_count > prev_findings_count` "
            "— a plausible-only record then counts as success while §2 pays 0"
        )
        assert "count_verified_findings(findings) > prev_verified_count" in src

    def test_prev_findings_count_is_gone(self):
        """It had no reader left; a dead baseline is how the drift started."""
        src = AGENT_FILE.read_text()
        assert "prev_findings_count" not in src

    def test_builder_docstring_names_one_contract(self):
        doc = _builder_param_doc()
        assert "findings_count went up" not in doc, (
            "_session_builder.py still documents the non-equivalent "
            "`findings_count went up` clause"
        )
        assert "prev_verified_count" in doc

    def test_builder_docstring_does_not_claim_a_status_key(self):
        """`FINDING_STATE_KEYS` has no `status` — filtering on it selects 0."""
        doc = _builder_param_doc()
        assert 'status == "recorded" or' not in doc
        assert "status" not in doc or "no `status` key" in doc, (
            "the `findings` docstring mentions `status` without telling the "
            "reader it is absent from the dict — that omission is what made "
            "a reviewer propose filtering on it"
        )

    def test_encoder_and_producer_agree(self):
        """The consumer's docstring was already correct — keep it that way."""
        src = STATE_ENCODER_FILE.read_text()
        assert "did last turn yield a verified finding?" in src

    def test_plausible_record_is_not_a_success(self):
        from app.evidence.finding_state import count_verified_findings

        findings = [_plausible_finding()]
        assert count_verified_findings(findings) == 0, (
            "a plausible-only record counted as verified — this is exactly "
            "the turn that earns 0 from §2"
        )
        assert len(findings) == 1, "precondition: the record IS in the list"

    def test_confirmed_record_is_a_success(self):
        from app.evidence.finding_state import count_verified_findings

        assert count_verified_findings([_confirmed_finding()]) == 1

    def test_status_filter_would_select_nothing(self):
        """The proposed `f.get("status") == "recorded"` filter is a no-op."""
        finding = _confirmed_finding()
        assert "status" not in finding
        assert sum(1 for f in [finding] if f.get("status") == "recorded") == 0

    def test_state_and_reward_agree_on_the_plausible_turn(self):
        """`last_turn_success` False <=> §2 pays 0, on the same turn."""
        from app.evidence.finding_state import count_verified_findings

        findings = [_plausible_finding()]
        session = _session(findings)
        _, breakdown = compute_reward(
            session=session,
            previous_verified_count=0,
            command_output="x" * 200,
            tokens_used_delta=0,
            credit_state=new_credit_state(),
        )
        assert breakdown["finding_bonus"] == 0.0
        assert (count_verified_findings(findings) > 0) is False, (
            "the state vector would claim progress on a turn the reward "
            "scored as no-progress"
        )
