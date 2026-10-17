"""Guards for SE1 — one definition of "verified", canonical field first.

The bug: every producer had migrated to the canonical `verification` field
(verifier levels `plausible` / `confirmed` / `replay_verified`), but the
state encoder still counted the LEGACY boolean:

    verified = sum(1 for f in findings
                   if f.get("verified") and not f.get("false_positive"))

A finding carrying only `verification: "confirmed"` scored 0 there forever:
Section 1's fifth scalar went dead, and any caller deriving
`last_turn_success` from the same count lost the success signal — silently,
with no exception and no log line.

The fix is not a seventh copy of the rule. `is_verified_finding()` in
`app/evidence/finding_state.py` is the ONE definition, and it now resolves
the canonical field first. Everything that counts "did this turn confirm
something?" calls it: the encoder, the reward shaper, the ReAct loop and
the TDI `successes` input.
"""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

import pytest

from app.evidence.finding_state import (
    FINDING_STATE_KEYS,
    count_verified_findings,
    finding_to_state,
    is_verified_finding,
)
from app.evidence.verifier import (
    VERIFICATION_CONFIRMED,
    VERIFICATION_LEVELS,
    VERIFICATION_PLAUSIBLE,
    VERIFICATION_REPLAY_VERIFIED,
    verification_rank,
)
from app.rl._session_builder import build_session
from app.rl.reward import compute_reward, new_credit_state
from app.rl.state_encoder import StateEncoder
from app.rl.tda import CONFIRMED_VULN, evidence_score_from_finding

ROOT = Path(__file__).resolve().parents[2]
APP = ROOT / "app"
FINDING_STATE_FILE = APP / "evidence" / "finding_state.py"
STATE_ENCODER_FILE = APP / "rl" / "state_encoder.py"
REWARD_FILE = APP / "rl" / "reward.py"
AGENT_FILE = APP / "agents" / "react_agent.py"

# Consumers that must never re-implement the rule. `finding_state.py` is
# excluded because `is_verified_finding()` IS the implementation.
CONSUMERS = (STATE_ENCODER_FILE, REWARD_FILE, AGENT_FILE)


def _finding(**fields) -> dict:
    return dict(fields)


# ── the predicate ────────────────────────────────────────────────────────


class TestCanonicalFieldWins:
    def test_confirmed_level_counts(self):
        """The exact regression: canonical-only findings used to score 0."""
        assert is_verified_finding(_finding(verification=VERIFICATION_CONFIRMED))

    def test_replay_verified_counts(self):
        assert is_verified_finding(_finding(verification=VERIFICATION_REPLAY_VERIFIED))

    def test_plausible_does_not_count(self):
        assert not is_verified_finding(_finding(verification=VERIFICATION_PLAUSIBLE))

    def test_level_comparison_is_case_and_space_tolerant(self):
        assert is_verified_finding(_finding(verification="  CONFIRMED  "))

    def test_canonical_beats_a_stale_legacy_true(self):
        """`plausible` + a leftover `verified: True` is still NOT verified.

        The producer used to hardcode `{"verified": True}` on every record.
        If the canonical field says plausible, that legacy flag is stale and
        must not re-promote the finding — otherwise recording junk inflates
        the state vector that the reward refuses to pay for.
        """
        finding = _finding(verification=VERIFICATION_PLAUSIBLE, verified=True)
        assert not is_verified_finding(finding)

    def test_canonical_confirmed_with_legacy_false_counts(self):
        """A producer that writes only the canonical level may leave the
        derived boolean unset/False; the level is authoritative."""
        finding = _finding(verification=VERIFICATION_CONFIRMED, verified=False)
        assert is_verified_finding(finding)

    def test_false_positive_outranks_every_positive_signal(self):
        for fields in (
            {"verification": VERIFICATION_CONFIRMED},
            {"verification": VERIFICATION_REPLAY_VERIFIED},
            {"verified": True},
            {"replay_reproduced": True},
            {"human_verified": True},
        ):
            assert not is_verified_finding(_finding(false_positive=True, **fields)), (
                f"a refuted finding counted as verified: {fields}"
            )

    def test_legacy_only_findings_still_work(self):
        """Replay traces and older tests carry no `verification` key."""
        assert is_verified_finding(_finding(verified=True))
        assert not is_verified_finding(_finding(verified=False))
        assert not is_verified_finding({})

    def test_unknown_level_falls_through_to_the_legacy_flags(self):
        """Neither a promotion nor a demotion for a malformed level."""
        assert is_verified_finding(_finding(verification="bogus", verified=True))
        assert not is_verified_finding(_finding(verification="bogus"))

    def test_non_dict_is_not_verified(self):
        for value in (None, "confirmed", [], 1):
            assert not is_verified_finding(value)  # type: ignore[arg-type]


# ── the two implementations must stay equivalent ─────────────────────────


MATRIX = [
    {},
    {"verified": True},
    {"verified": False},
    {"verification": VERIFICATION_PLAUSIBLE},
    {"verification": VERIFICATION_CONFIRMED},
    {"verification": VERIFICATION_REPLAY_VERIFIED},
    {"verification": "CONFIRMED"},
    {"verification": "bogus"},
    {"verification": "bogus", "verified": True},
    {"verification": VERIFICATION_PLAUSIBLE, "verified": True},
    {"verification": VERIFICATION_CONFIRMED, "verified": False},
    {"verification": VERIFICATION_CONFIRMED, "false_positive": True},
    {"replay_reproduced": True},
    {"human_verified": True},
    {"verification": VERIFICATION_REPLAY_VERIFIED, "false_positive": True},
]


class TestPredicateEqualsEvidenceLadder:
    """`is_verified_finding(f)` <=> `evidence_score_from_finding(f) >= 0.8`.

    `finding_state.py` cannot import `app.rl.tda` (that would make the
    evidence layer depend on RL, and `app/rl/__init__` imports `reward`,
    which imports `finding_state` — a real cycle). So the two tables are
    kept separate on purpose; this class pins their agreement so they
    cannot drift.
    """

    @pytest.mark.parametrize("finding", MATRIX, ids=lambda f: json.dumps(f))
    def test_equivalence(self, finding):
        assert is_verified_finding(finding) is (
            evidence_score_from_finding(finding) >= CONFIRMED_VULN
        )

    @pytest.mark.parametrize("level", VERIFICATION_LEVELS)
    def test_producer_output_agrees_with_the_predicate(self, level):
        """`finding_to_state` derives `verified` from `verification`; both
        roads must lead to the same answer for every level the verifier can
        emit."""
        finding = finding_to_state(
            {"title": "t", "location": "/", "severity": "high"},
            {"status": "recorded", "verification": level},
        )
        expected = level in (VERIFICATION_CONFIRMED, VERIFICATION_REPLAY_VERIFIED)
        assert is_verified_finding(finding) is expected

    def test_later_refutation_flips_both_views(self):
        """The Phase-5 auditor sets `false_positive` after the fact."""
        finding = finding_to_state(
            {"title": "sqli", "location": "/login", "severity": "high"},
            {"status": "recorded", "verification": VERIFICATION_CONFIRMED},
        )
        assert is_verified_finding(finding)
        finding["false_positive"] = True
        assert not is_verified_finding(finding)
        assert evidence_score_from_finding(finding) < CONFIRMED_VULN


# ── all consumers delegate ───────────────────────────────────────────────


def _get_calls(path: Path, keys: set[str]) -> list[tuple[str, int]]:
    """`x.get("<key>")` literals in a module, as (key, lineno).

    AST, not text: every consumer documents the old inline expression in a
    comment, and a grep-based guard would flag its own explanation.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: list[tuple[str, int]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (
            isinstance(func, ast.Attribute)
            and func.attr == "get"
            and isinstance(func.value, ast.Name)
        ):
            continue
        for arg in node.args:
            if isinstance(arg, ast.Constant) and arg.value in keys:
                found.append((str(arg.value), node.lineno))
    return found


def _calls_named(path: Path, name: str) -> list[int]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == name
    ]


class TestConsumersDelegate:
    @pytest.mark.parametrize("path", CONSUMERS, ids=lambda p: p.name)
    def test_no_legacy_verified_read(self, path: Path):
        assert _get_calls(path, {"verified"}) == [], (
            f"{path.name} reads the legacy `verified` boolean again instead "
            "of calling is_verified_finding()/count_verified_findings() — "
            "canonical-only findings will be silently dropped"
        )

    def test_encoder_counts_through_the_helper(self):
        assert _calls_named(STATE_ENCODER_FILE, "count_verified_findings"), (
            "state_encoder.py no longer calls count_verified_findings()"
        )

    def test_reward_uses_the_predicate_per_finding(self):
        assert _calls_named(REWARD_FILE, "is_verified_finding"), (
            "reward.py's per-finding novelty loop filtered on a different "
            "rule than the gate above it"
        )

    def test_agent_tdi_successes_uses_the_helper(self):
        """The TDI `successes=` argument was the fourth definition."""
        tree = ast.parse(AGENT_FILE.read_text(encoding="utf-8"))
        found: list[bool] = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            for kw in node.keywords:
                if kw.arg != "successes":
                    continue
                value = kw.value
                found.append(
                    isinstance(value, ast.Call)
                    and isinstance(value.func, ast.Name)
                    and value.func.id == "count_verified_findings"
                )
        assert found, "react_agent.py no longer passes `successes=` to tda_compute"
        assert all(found), (
            "`successes=` is computed inline — it must be "
            "count_verified_findings(findings), the same predicate the "
            "encoder and reward use"
        )

    def test_agent_no_longer_carries_its_own_evidence_ladder_call(self):
        # AST on the import statements, not a text scan: the explanatory
        # comment above the fixed call site quotes the old expression, and
        # a substring guard would flag its own documentation.
        tree = ast.parse(AGENT_FILE.read_text(encoding="utf-8"))
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[-1] for alias in node.names)
        assert "evidence_score_from_finding" not in imported, (
            "react_agent.py imports the TDA ladder again for a count; the "
            "count belongs in count_verified_findings()"
        )
        assert "CONFIRMED_VULN" not in imported

    def test_predicate_is_the_only_legacy_read_in_finding_state(self):
        """Inside `finding_state.py` the legacy read is allowed — but only
        in the one function that defines the rule."""
        tree = ast.parse(FINDING_STATE_FILE.read_text(encoding="utf-8"))
        hits: list[str] = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef):
                continue
            for sub in ast.walk(node):
                if not isinstance(sub, ast.Call):
                    continue
                func = sub.func
                if not (
                    isinstance(func, ast.Attribute)
                    and func.attr == "get"
                    and isinstance(func.value, ast.Name)
                ):
                    continue
                if any(
                    isinstance(a, ast.Constant) and a.value == "verified"
                    for a in sub.args
                ):
                    hits.append(node.name)
        assert hits == ["is_verified_finding"], (
            f"legacy `verified` is read in {sorted(set(hits))} — the rule "
            "must live in is_verified_finding() only"
        )


# ── behavioural: encoder and reward see the canonical finding ────────────


def _session(findings: list[dict]) -> dict:
    return build_session(
        scan_id="scan-se1",
        target="http://target",
        turn=1,
        commands_run=3,
        tokens_used=1000,
        token_budget=500_000,
        last_command="sqlmap",
        last_vector="exploit",
        last_turn_success=False,
        findings=findings,
    )


class TestEndToEnd:
    def test_encoder_counts_a_canonical_only_finding(self):
        """The reported failure: `verified_findings` stuck at 0 forever."""
        findings = [_finding(verification=VERIFICATION_CONFIRMED)]
        state = StateEncoder.from_session({"findings": findings})
        assert state.verified_findings == 1, (
            "a finding with only `verification: confirmed` was not counted"
        )
        vec = state.to_vector()
        assert vec.shape == (337,)

    def test_encoder_counts_replay_verified(self):
        findings = [_finding(verification=VERIFICATION_REPLAY_VERIFIED)]
        assert StateEncoder.from_session({"findings": findings}).verified_findings == 1

    def test_encoder_does_not_count_plausible(self):
        findings = [_finding(verification=VERIFICATION_PLAUSIBLE)]
        assert StateEncoder.from_session({"findings": findings}).verified_findings == 0

    def test_encoder_and_reward_agree_on_the_canonical_turn(self):
        findings = [_finding(
            title="sqli", location="/login", severity="high",
            verification=VERIFICATION_CONFIRMED, false_positive=False,
        )]
        session = _session(findings)
        _, breakdown = compute_reward(
            session=session,
            previous_verified_count=0,
            command_output="x" * 200,
            tokens_used_delta=0,
            credit_state=new_credit_state(),
        )
        encoder_count = StateEncoder.from_session(session).verified_findings
        assert encoder_count == 1
        assert breakdown["finding_bonus"] > 0, (
            "the encoder sees progress but the reward pays nothing — state "
            "and reward are reading different definitions again"
        )

    def test_encoder_and_reward_agree_on_a_plausible_turn(self):
        findings = [_finding(
            title="hdr", location="/", severity="low",
            verification=VERIFICATION_PLAUSIBLE, false_positive=False,
        )]
        session = _session(findings)
        _, breakdown = compute_reward(
            session=session,
            previous_verified_count=0,
            command_output="x" * 200,
            tokens_used_delta=0,
            credit_state=new_credit_state(),
        )
        assert StateEncoder.from_session(session).verified_findings == 0
        assert breakdown["finding_bonus"] == 0.0

    def test_count_is_the_predicate_sum(self):
        findings = [f for f in MATRIX]
        assert count_verified_findings(findings) == sum(
            1 for f in findings if is_verified_finding(f)
        )


# ── the refuted half: `verification == "false_positive"` ─────────────────


class TestFalsePositiveIsNotAVerificationLevel:
    """SE1 proposed also reading `verification == "false_positive"`.

    There is no such level. The verifier emits exactly three, and the
    canonical flag for a refuted finding is the boolean
    `false_positive` (in `FINDING_STATE_KEYS`). Adding the branch would be
    dead code that also implies a vocabulary the verifier does not have.
    """

    def test_verification_levels_are_the_three_rungs(self):
        assert set(VERIFICATION_LEVELS) == {
            VERIFICATION_PLAUSIBLE,
            VERIFICATION_CONFIRMED,
            VERIFICATION_REPLAY_VERIFIED,
        }
        assert "false_positive" not in VERIFICATION_LEVELS

    def test_false_positive_has_no_rank(self):
        assert verification_rank("false_positive") == 0

    def test_false_positive_is_in_the_finding_contract(self):
        assert "false_positive" in FINDING_STATE_KEYS

    def test_encoder_counts_the_canonical_boolean(self):
        findings = [
            _finding(verification=VERIFICATION_CONFIRMED, false_positive=True),
            _finding(verification=VERIFICATION_PLAUSIBLE, false_positive=False),
        ]
        state = StateEncoder.from_session({"findings": findings})
        assert state.fp_findings == 1
        assert state.verified_findings == 0, (
            "a refuted finding must not be counted as verified"
        )


# ── layering: evidence must not import RL ────────────────────────────────


class TestLayering:
    def test_finding_state_does_not_import_rl(self):
        """Reusing `evidence_score_from_finding` here would be tempting but
        wrong: `app/rl/__init__` imports `reward`, which imports
        `finding_state` — the cycle would only appear under a cold import.

        The equivalence test above is the correct way to keep the two
        tables aligned.
        """
        tree = ast.parse(FINDING_STATE_FILE.read_text(encoding="utf-8"))
        imported: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                imported.append(node.module)
            elif isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
        assert not [m for m in imported if m == "app.rl" or m.startswith("app.rl.")], (
            f"app.evidence.finding_state imports RL modules: {imported}"
        )

    def test_cold_import_of_finding_state_works(self):
        """Subprocess, so the parent process's import state cannot mask a
        cycle introduced by a future edit."""
        result = subprocess.run(
            [sys.executable, "-c",
             "import app.evidence.finding_state; "
             "import app.rl.state_encoder; "
             "print('ok')"],
            cwd=str(ROOT), capture_output=True, text=True,
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "ok"
