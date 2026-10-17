"""Round 9 (Claim A) — guards for the TDA evidence-label map.

Two bugs lived in ``app/rl/tda.py::_EVIDENCE_LABEL_MAP``:

1. ``"replay_verified"`` — the CANONICAL, strongest level from
   ``app/evidence/verifier.py``, and the one ``evidence_score_from_finding()``
   already honoured explicitly — was absent from the label map. So
   ``evidence_score_from_label("replay_verified")`` returned **0.3**
   (SPECULATIVE_HYPOTHESIS): the best evidence we can possibly have scored
   *below* "we have no evidence at all" (0.5 for a phantom finding).

2. ``"verified"`` mapped to VERIFIED_EXPLOIT (1.0) while
   ``evidence_score_from_finding({"verified": True})`` returned
   CONFIRMED_VULN (0.8) — the same word, two different rungs, in a module
   whose whole job is to produce one number per finding.

These tests pin the invariant that justifies the module: **one label, one
rung**, and the ladder must never score a stronger label lower than a
weaker one.

Note on Claim B (same round): the claim that ``react_agent`` hardcodes
``{"verified": True}`` instead of reading the verifier verdict is covered
by ``TestReActAgentReadsVerifierVerdict`` at the bottom of this file — the
pre-existing guard in ``test_evidence_verification.py::_recorded_finding``
is a *mirror* of the agent's dict, not an assertion on the agent's source,
so it could not catch a regression back to a hardcoded ``True``.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from app.evidence.verifier import (
    VERIFICATION_CONFIRMED,
    VERIFICATION_LEVELS,
    VERIFICATION_PLAUSIBLE,
    VERIFICATION_REPLAY_VERIFIED,
)
from app.rl.tda import (
    CONFIRMED_VULN,
    PLAUSIBLE_HYPOTHESIS,
    SPECULATIVE_HYPOTHESIS,
    VERIFIED_EXPLOIT,
    _EVIDENCE_LABEL_MAP,
    evidence_score_from_finding,
    evidence_score_from_label,
    max_evidence_score,
    should_force_exploit,
)

AGENT_FILE = Path(__file__).resolve().parents[2] / "app" / "agents" / "react_agent.py"
FINDING_STATE_FILE = (
    Path(__file__).resolve().parents[2] / "app" / "evidence" / "finding_state.py"
)

# The rung each canonical verifier level must land on. This is the
# contract between app/evidence/verifier.py and app/rl/tda.py.
EXPECTED_RUNG = {
    VERIFICATION_PLAUSIBLE: PLAUSIBLE_HYPOTHESIS,
    VERIFICATION_CONFIRMED: CONFIRMED_VULN,
    VERIFICATION_REPLAY_VERIFIED: VERIFIED_EXPLOIT,
}


class TestLabelMapCoversCanonicalLevels:
    """The drift guard: adding a level to the verifier must not be silent."""

    @pytest.mark.parametrize("level", VERIFICATION_LEVELS)
    def test_every_verifier_level_is_in_the_map(self, level: str):
        assert level in _EVIDENCE_LABEL_MAP, (
            f"{level!r} is a canonical verification level in "
            f"app/evidence/verifier.py but is missing from "
            f"tda._EVIDENCE_LABEL_MAP — evidence_score_from_label() would "
            f"silently return SPECULATIVE_HYPOTHESIS ({SPECULATIVE_HYPOTHESIS})."
        )

    @pytest.mark.parametrize("level,expected", sorted(EXPECTED_RUNG.items()))
    def test_canonical_levels_land_on_their_rung(self, level: str, expected: float):
        assert evidence_score_from_label(level) == expected

    def test_replay_verified_is_the_top_rung(self):
        """The exact bug from Claim A: it used to return 0.3."""
        assert evidence_score_from_label("replay_verified") == VERIFIED_EXPLOIT
        assert evidence_score_from_label("replay_verified") == 1.0

    def test_replay_verified_outranks_a_phantom_finding(self):
        """0.3 vs 0.5 was the absurdity: confirmed PoC < no evidence."""
        assert evidence_score_from_label(
            VERIFICATION_REPLAY_VERIFIED
        ) > evidence_score_from_finding({})


class TestLabelAndFindingAgree:
    """`evidence_score_from_label(x) == evidence_score_from_finding({"verification": x})`.

    These two functions are exported together and describe the same ladder.
    Divergence is how bug 1 survived: the finding path honoured
    `replay_verified`, the label path did not.
    """

    @pytest.mark.parametrize("level", VERIFICATION_LEVELS)
    def test_same_word_same_rung(self, level: str):
        assert evidence_score_from_label(level) == evidence_score_from_finding(
            {"verification": level}
        )

    @pytest.mark.parametrize("label", sorted(_EVIDENCE_LABEL_MAP))
    def test_every_map_entry_round_trips_through_a_finding(self, label: str):
        assert evidence_score_from_label(label) == evidence_score_from_finding(
            {"verification": label}
        )

    def test_legacy_verified_boolean_matches_the_word_verified(self):
        """Bug 2: `"verified"` was 1.0 in the map, 0.8 in the dict path."""
        assert evidence_score_from_finding({"verified": True}) == CONFIRMED_VULN
        assert evidence_score_from_label("verified") == CONFIRMED_VULN

    def test_verified_flag_and_replay_flag_still_differ(self):
        # The 0.8 / 1.0 split is intentional: "a verifier accepted it" is
        # not the same as "a payload replayed and reproduced".
        assert evidence_score_from_finding({"verified": True}) == CONFIRMED_VULN
        assert evidence_score_from_finding({"replay_reproduced": True}) == VERIFIED_EXPLOIT
        assert CONFIRMED_VULN < VERIFIED_EXPLOIT


class TestLadderIsMonotonic:
    """A stronger label must never score below a weaker one."""

    ORDERED_LABELS = [
        "guess",
        "speculative",
        "hypothesis",
        "plausible",
        "vuln",
        "confirmed",
        "verified",
        "exploit",
        "replay",
        "replay_verified",
    ]

    def test_scores_are_non_decreasing(self):
        scores = [evidence_score_from_label(lbl) for lbl in self.ORDERED_LABELS]
        assert scores == sorted(scores), (
            "evidence ladder is not monotonic: "
            + ", ".join(f"{lbl}={score}" for lbl, score in zip(self.ORDERED_LABELS, scores))
        )

    def test_every_rung_is_reachable_from_the_label_path(self):
        reachable = {evidence_score_from_label(lbl) for lbl in self.ORDERED_LABELS}
        assert reachable == {
            SPECULATIVE_HYPOTHESIS,
            PLAUSIBLE_HYPOTHESIS,
            CONFIRMED_VULN,
            VERIFIED_EXPLOIT,
        }


class TestUnknownLabelsStaySpeculative:
    @pytest.mark.parametrize("label", [None, "", "   ", "totally-made-up", "vErIfIeD_2"])
    def test_unknown_or_empty_falls_to_the_bottom(self, label):
        assert evidence_score_from_label(label) == SPECULATIVE_HYPOTHESIS

    def test_label_lookup_is_case_and_whitespace_insensitive(self):
        assert evidence_score_from_label("  Replay_Verified  ") == VERIFIED_EXPLOIT
        assert evidence_score_from_label("CONFIRMED") == CONFIRMED_VULN

    def test_non_string_input_is_tolerated(self):
        assert evidence_score_from_label(123) == SPECULATIVE_HYPOTHESIS  # type: ignore[arg-type]


class TestOverrideStillGatedOnTheTopRungs:
    """The label fix must not let a weak label through the EXPLOIT gate."""

    def test_plausible_does_not_force(self):
        assert should_force_exploit([{"verification": VERIFICATION_PLAUSIBLE}]) is False

    def test_confirmed_forces(self):
        assert should_force_exploit([{"verification": VERIFICATION_CONFIRMED}]) is True

    def test_replay_verified_forces(self):
        assert should_force_exploit([{"verification": VERIFICATION_REPLAY_VERIFIED}]) is True

    def test_empty_list_does_not_force(self):
        assert should_force_exploit([]) is False
        assert max_evidence_score([]) == 0.0

    def test_false_positive_beats_a_recorded_level(self):
        # Contradictory records must never reach the gate, and a refutation
        # ranks at the floor — it cannot tie with an unexamined hypothesis.
        assert (
            evidence_score_from_finding(
                {"verification": VERIFICATION_REPLAY_VERIFIED, "false_positive": True}
            )
            == SPECULATIVE_HYPOTHESIS
        )


# ---------------------------------------------------------------------------
# Claim B guard — the ReAct loop must read the verifier verdict
# ---------------------------------------------------------------------------


def _parse_agent() -> ast.Module:
    return ast.parse(AGENT_FILE.read_text(encoding="utf-8"))


def _parse_finding_state() -> ast.Module:
    """The producer module — where the finding dict is now built."""
    return ast.parse(FINDING_STATE_FILE.read_text(encoding="utf-8"))


def _finding_dict_literals(tree: ast.AST) -> list[ast.Dict]:
    """Dict literals returned by `finding_to_state()` (`return { ... }`)."""
    found: list[ast.Dict] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Return) and isinstance(node.value, ast.Dict):
            found.append(node.value)
    return found


def _record_vulnerability_branch(tree: ast.Module) -> ast.If:
    """Locate `if tool_name == "record_vulnerability":` in the ReAct loop."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        test = node.test
        if not isinstance(test, ast.Compare):
            continue
        left = test.left
        if not (isinstance(left, ast.Name) and left.id == "tool_name"):
            continue
        if len(test.comparators) != 1:
            continue
        comparator = test.comparators[0]
        if isinstance(comparator, ast.Constant) and comparator.value == "record_vulnerability":
            return node
    raise AssertionError(
        f'could not find `if tool_name == "record_vulnerability":` in {AGENT_FILE}'
    )


def _findings_append_dicts(node: ast.AST) -> list[ast.Dict]:
    """All dict literals passed to `findings.append({...})` inside `node`."""
    found: list[ast.Dict] = []
    for child in ast.walk(node):
        if not isinstance(child, ast.Call):
            continue
        func = child.func
        if not isinstance(func, ast.Attribute) or func.attr != "append":
            continue
        if not (isinstance(func.value, ast.Name) and func.value.id == "findings"):
            continue
        for arg in child.args:
            if isinstance(arg, ast.Dict):
                found.append(arg)
    return found


def _dict_keys(node: ast.Dict) -> list[str]:
    return [k.value for k in node.keys if isinstance(k, ast.Constant)]


def _dict_value(node: ast.Dict, key: str) -> ast.expr | None:
    for k, v in zip(node.keys, node.values):
        if isinstance(k, ast.Constant) and k.value == key:
            return v
    return None


class TestReActAgentReadsVerifierVerdict:
    """Claim B: is the evidence ladder actually wired to the verifier?

    The answer must come from the agent's SOURCE, not from a hand-written
    fixture that happens to look like the agent's dict.

    Round 11 moved the dict construction into
    `app/evidence/finding_state.finding_to_state()` so the ReAct loop, the
    TDA ladder and the reward shaper cannot drift apart. The invariants
    below survive the move — they just now have to be checked where the
    dict is actually built, plus one link proving the agent uses it.
    """

    def test_the_branch_exists(self):
        branch = _record_vulnerability_branch(_parse_agent())
        builds = [
            n
            for n in ast.walk(branch)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name)
            and n.func.id == "finding_to_state"
        ]
        assert builds, (
            "the record_vulnerability branch no longer builds its finding "
            "via finding_to_state() -- the key set would drift from the "
            "TDA ladder / reward shaper again"
        )

    def test_verification_comes_from_parse_verification(self):
        calls = [
            n
            for n in ast.walk(_parse_finding_state())
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name)
            and n.func.id == "parse_verification"
        ]
        assert calls, (
            "finding_to_state no longer calls parse_verification(...) — the "
            "finding's verification level must come from tool_bridge's "
            "verifier verdict, never from tool_args or a literal."
        )

    def test_verification_is_parsed_from_the_tool_result(self):
        """The source of the level must be `result_json`, not `tool_args`."""
        parsed_from_result = False
        for node in ast.walk(_parse_finding_state()):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "parse_verification"
            ):
                continue
            for arg in list(node.args) + [kw.value for kw in node.keywords]:
                if isinstance(arg, ast.Call) and isinstance(arg.func, ast.Attribute):
                    if arg.func.attr == "get" and isinstance(arg.func.value, ast.Name):
                        if arg.func.value.id in ("result_json", "result"):
                            parsed_from_result = True
        assert parsed_from_result, (
            "parse_verification(...) is not fed from `result.get(...)` — "
            "the verdict must come from the tool_bridge response payload."
        )

    def test_verification_is_never_read_from_tool_args(self):
        """Neither the agent branch nor the producer may grade the finding
        from the LLM's own claim about itself."""
        offenders: list[int] = []
        for tree in (_parse_agent(), _parse_finding_state()):
            for node in ast.walk(tree):
                if not (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "get"
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id in ("tool_args", "args")
                ):
                    continue
                keys = [
                    a.value for a in node.args
                    if isinstance(a, ast.Constant) and isinstance(a.value, str)
                ]
                if "verification" in keys or "verified" in keys:
                    offenders.append(node.lineno)
        assert not offenders, (
            f"the verification verdict is read from tool_args/args at lines "
            f"{offenders} — that is the LLM's own claim about itself."
        )

    def test_no_hardcoded_verified_true_in_the_finding_dict(self):
        """The original bug: `"verified": True` for every recorded finding."""
        dicts = _finding_dict_literals(_parse_finding_state())
        assert dicts, "finding_to_state() no longer returns a dict literal"
        for dict_node in dicts:
            keys = _dict_keys(dict_node)
            assert "verification" in keys, (
                "the findings dict must carry the canonical `verification` "
                f"level (got keys: {keys})"
            )
            verified = _dict_value(dict_node, "verified")
            assert not (isinstance(verified, ast.Constant) and verified.value is True), (
                'the findings dict hardcodes `"verified": True` — every '
                "finding would score CONFIRMED_VULN (0.8) regardless of "
                "evidence, collapsing the 0.3/0.5/0.8/1.0 ladder."
            )

    def test_verified_flag_is_derived_from_the_level(self):
        derived = False
        for dict_node in _finding_dict_literals(_parse_finding_state()):
            value = _dict_value(dict_node, "verified")
            if isinstance(value, (ast.Compare, ast.BoolOp)):
                derived = True
        assert derived, (
            '`"verified"` must be derived from the verification level '
            "(e.g. `verification in (VERIFICATION_CONFIRMED, "
            "VERIFICATION_REPLAY_VERIFIED)`), not supplied as a literal."
        )

    def test_agent_imports_the_verifier_levels(self):
        """The producer derives the ladder booleans from the verifier's
        canonical level constants, not from re-typed strings."""
        source = FINDING_STATE_FILE.read_text(encoding="utf-8")
        assert "VERIFICATION_REPLAY_VERIFIED" in source
        assert "VERIFICATION_CONFIRMED" in source
