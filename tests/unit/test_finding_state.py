"""Round 11 (R3) — one producer, one key set, no more silent drift.

R3: `reward._finding_identity` read `name`/`location` while the ReAct loop
wrote `title` and never stored `location` at all. `dict.get()` returns
`""` rather than raising, so:

  * every finding of a given severity produced the identity `"||high"`,
    and only the FIRST one of each severity was ever paid;
  * `_is_novel_finding` produced the key `":"` for everything, so every
    finding after the first dropped from the +20 novelty tier to +1;
  * `(the location was sitting in tool_args the whole time)`.

The fix is not "add the missing key" — that just resets the clock. It is
`app/evidence/finding_state.py`: one producer (`finding_to_state`) and
one place that resolves field aliases. The tests below pin BOTH halves:
the tolerant accessors, and a contract that any new consumer-side read
must be backed by the producer.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from app.evidence.finding_state import (
    FINDING_STATE_KEYS,
    LOCATION_KEYS,
    TITLE_KEYS,
    finding_identity,
    finding_location,
    finding_novelty_key,
    finding_severity,
    finding_title,
    finding_to_state,
)
from app.evidence.verifier import (
    VERIFICATION_CONFIRMED,
    VERIFICATION_PLAUSIBLE,
    VERIFICATION_REPLAY_VERIFIED,
)

APP = Path(__file__).resolve().parents[2] / "app"
AGENT_FILE = APP / "agents" / "react_agent.py"


def _agent_args(**overrides) -> dict:
    """The shape the LLM actually sends to record_vulnerability."""
    args = {
        "title": "SQL Injection in /login",
        "severity": "high",
        "vuln_type": "sqli",
        "target": "http://target",
        "location": "/login?id=1",
        "evidence": "sqlmap: parameter 'id' is vulnerable",
        "description": "...",
    }
    args.update(overrides)
    return args


def _result(**overrides) -> dict:
    """The shape tool_bridge._record_vulnerability actually returns."""
    result = {
        "status": "recorded",
        "finding_id": "abc",
        "verification": VERIFICATION_CONFIRMED,
        "verification_reason": "sqlmap verdict present",
        # Always False at insert time — Phase 5 flips the DB row.
        "verified": False,
        "poc_status": "successful",
    }
    result.update(overrides)
    return result


# ── the identity collapse ──────────────────────────────────────────────


class TestIdentityIsSpecific:
    def test_agent_shape_produces_distinct_identities(self):
        """The exact bug: three high-severity findings, one identity."""
        a = finding_to_state(_agent_args(title="SQLi in /login", location="/login"), _result())
        b = finding_to_state(_agent_args(title="XSS in /search", location="/search"), _result())
        c = finding_to_state(_agent_args(title="Missing HSTS", location="/"), _result())

        identities = {finding_identity(x) for x in (a, b, c)}
        assert len(identities) == 3, f"identities collapsed to {identities}"

    def test_identity_is_not_just_the_severity(self):
        f = finding_to_state(_agent_args(), _result())
        assert finding_identity(f) != f"||{f['severity']}"
        assert "SQL Injection in /login" in finding_identity(f)

    def test_same_finding_twice_shares_an_identity(self):
        """Re-recording after a rejection must not be paid a second time."""
        first = finding_to_state(_agent_args(), _result())
        second = finding_to_state(_agent_args(), _result())
        assert finding_identity(first) == finding_identity(second)

    def test_same_title_different_location_is_different(self):
        a = finding_to_state(_agent_args(location="/a"), _result())
        b = finding_to_state(_agent_args(location="/b"), _result())
        assert finding_identity(a) != finding_identity(b)

    def test_severity_is_normalised(self):
        upper = finding_to_state(_agent_args(severity="HIGH"), _result())
        lower = finding_to_state(_agent_args(severity="high"), _result())
        assert finding_severity(upper) == "high"
        assert finding_identity(upper) == finding_identity(lower)


class TestNoveltyKeyIsSpecific:
    def test_agent_shape_produces_distinct_novelty_keys(self):
        credit: set[str] = set()
        keys = []
        for title, loc in (
            ("SQLi in /login", "/login"),
            ("XSS in /search", "/search"),
            ("Missing HSTS", "/"),
        ):
            f = finding_to_state(_agent_args(title=title, location=loc), _result())
            keys.append(finding_novelty_key(f))
        assert len(set(keys)) == 3
        assert keys != [":"] * 3

    def test_repeat_is_not_novelty_again(self):
        a = finding_to_state(_agent_args(), _result())
        b = finding_to_state(_agent_args(severity="medium"), _result())
        # Same title+location, different severity -> same novelty key.
        assert finding_novelty_key(a) == finding_novelty_key(b)


# ── tolerant accessors ─────────────────────────────────────────────────


class TestAliasResolution:
    def test_title_prefers_title_then_name(self):
        assert finding_title({"title": "T", "name": "N"}) == "T"
        assert finding_title({"name": "N"}) == "N"

    def test_blank_title_falls_through_to_name(self):
        """A present-but-empty key must not shadow a populated alias."""
        assert finding_title({"title": "", "name": "N"}) == "N"
        assert finding_title({"title": "   ", "name": "N"}) == "N"

    def test_location_falls_back_to_target(self):
        """`location` is optional in the tool schema; `target` is required."""
        assert finding_location({"target": "http://t"}) == "http://t"
        assert finding_location({"endpoint": "/e"}) == "/e"
        assert finding_location({"url": "u"}) == "u"

    def test_location_prefers_the_specific_value(self):
        assert finding_location({"location": "/x", "target": "http://t"}) == "/x"

    def test_missing_everything_yields_empty(self):
        assert finding_title({}) == ""
        assert finding_location({}) == ""
        assert finding_severity({}) == ""

    def test_non_dict_is_tolerated(self):
        assert finding_title(None) == ""  # type: ignore[arg-type]
        assert finding_location("nope") == ""  # type: ignore[arg-type]
        assert finding_identity(None) == "||"  # type: ignore[arg-type]

    def test_agent_args_without_location_uses_target(self):
        """The LLM omits the optional `location` — identity must still be
        specific enough for `target` to keep findings apart."""
        args = _agent_args(location="")
        a = finding_to_state(args, _result())
        b = finding_to_state({**args, "target": "http://other"}, _result())
        assert finding_location(a) == "http://target"
        assert finding_identity(a) != finding_identity(b)


# ── the producer contract ──────────────────────────────────────────────


class TestFindingToState:
    def test_emits_exactly_the_documented_keys(self):
        state = finding_to_state(_agent_args(), _result())
        assert tuple(state) == FINDING_STATE_KEYS

    def test_keys_are_all_strings(self):
        assert all(isinstance(k, str) for k in FINDING_STATE_KEYS)

    def test_verification_comes_from_the_result(self):
        state = finding_to_state(_agent_args(), _result())
        assert state["verification"] == VERIFICATION_CONFIRMED

    def test_missing_verdict_defaults_to_plausible(self):
        """A missing verdict must never self-promote to confirmed."""
        state = finding_to_state(_agent_args(), {"status": "recorded"})
        assert state["verification"] == VERIFICATION_PLAUSIBLE
        assert state["verified"] is False

    def test_unknown_verdict_defaults_to_plausible(self):
        state = finding_to_state(_agent_args(), _result(verification="totally-fine"))
        assert state["verification"] == VERIFICATION_PLAUSIBLE

    def test_does_not_trust_the_llm_verdict(self):
        """`verification` in tool_args is the LLM grading itself."""
        args = _agent_args(verification=VERIFICATION_REPLAY_VERIFIED)
        state = finding_to_state(args, {"status": "recorded"})
        assert state["verification"] == VERIFICATION_PLAUSIBLE

    def test_does_not_copy_result_verified(self):
        """`result["verified"]` is the DB row state (False until Phase 5)."""
        state = finding_to_state(_agent_args(), _result(verified=False))
        assert state["verified"] is True, (
            "the boolean must be DERIVED from the verdict; copying "
            "result['verified'] scores every finding as unverified"
        )

    def test_replay_level_sets_replay_reproduced(self):
        """The bridge's response has no `replay_reproduced` field, so it
        has to be derived — otherwise the 5x replay weight never fires."""
        state = finding_to_state(
            _agent_args(), _result(verification=VERIFICATION_REPLAY_VERIFIED)
        )
        assert state["replay_reproduced"] is True
        assert state["verified"] is True

    def test_confirmed_level_is_not_replay_reproduced(self):
        state = finding_to_state(_agent_args(), _result())
        assert state["replay_reproduced"] is False

    def test_plausible_level_is_unverified(self):
        state = finding_to_state(
            _agent_args(), _result(verification=VERIFICATION_PLAUSIBLE)
        )
        assert state["verified"] is False
        assert state["replay_reproduced"] is False

    def test_recorded_finding_is_not_a_false_positive(self):
        """The bridge only records findings its verifier accepted."""
        state = finding_to_state(_agent_args(false_positive=True), _result(false_positive=True))
        assert state["false_positive"] is False
        assert state["false_positive_at"] is None

    def test_location_is_preserved_for_identity(self):
        state = finding_to_state(_agent_args(), _result())
        assert state["location"] == "/login?id=1"

    def test_none_args_are_tolerated(self):
        state = finding_to_state(None, None)
        assert state["verification"] == VERIFICATION_PLAUSIBLE
        assert state["verified"] is False
        assert state["title"] == ""

    def test_severity_stored_lowercase(self):
        state = finding_to_state(_agent_args(severity="Critical"), _result())
        assert state["severity"] == "critical"


# ── the contract that stops the NEXT drift ─────────────────────────────

CONSUMERS = (
    APP / "rl" / "reward.py",
    APP / "rl" / "tda.py",
    APP / "rl" / "state_encoder.py",
    APP / "agents" / "react_agent.py",
)

# Findings also arrive from replay traces / tests using the legacy
# spellings, which the accessors above resolve. Those are allowed.
ALIAS_KEYS = set(TITLE_KEYS) | set(LOCATION_KEYS)


def _consumed_keys(path: Path) -> dict[str, int]:
    """Every `f.get("k")` / `finding.get("k")` literal key in a module."""
    found: dict[str, int] = {}
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (
            isinstance(func, ast.Attribute)
            and func.attr == "get"
            and isinstance(func.value, ast.Name)
            and func.value.id in ("f", "finding")
        ):
            continue
        for arg in node.args:
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                found.setdefault(arg.value, node.lineno)
    return found


class TestConsumerContract:
    """Any key a consumer reads must be produced by `finding_to_state`."""

    @pytest.mark.parametrize("path", CONSUMERS, ids=lambda p: p.name)
    def test_consumed_keys_are_produced(self, path: Path):
        consumed = _consumed_keys(path)
        if not consumed:
            # A consumer with zero direct `f.get("...")` reads must delegate
            # to the shared accessors, otherwise `assert consumed` would be
            # vacuous and the drift this test exists to catch would go
            # unchecked. react_agent.py reads finding fields only through
            # `count_verified_findings()` / `finding_to_state()`; that is the
            # *good* outcome — one definition of "verified" shared with
            # reward.py instead of the third inline copy it used to have.
            src = path.read_text(encoding="utf-8")
            assert (
                "count_verified_findings(" in src
                or "is_verified_finding(" in src
                or "finding_title(" in src
                or "finding_location(" in src
                or "finding_severity(" in src
            ), (
                f"{path.name} reads no finding keys and calls no shared "
                f"accessor — the finding fields it needs are being resolved "
                f"some other way"
            )
            return
        unknown = {
            key: line
            for key, line in consumed.items()
            if key not in set(FINDING_STATE_KEYS) and key not in ALIAS_KEYS
        }
        assert not unknown, (
            f"{path.name} reads finding keys that finding_to_state() does "
            f"not emit: {unknown}. A missing key silently reads as '' — add "
            "it to app/evidence/finding_state.py instead of defaulting here."
        )

    def test_contract_covers_the_economy_bearing_fields(self):
        """Sanity: the contract test would catch the R3 field drift."""
        assert {"title", "location", "severity", "verification"} <= set(FINDING_STATE_KEYS)
        assert "name" not in FINDING_STATE_KEYS or "name" in TITLE_KEYS


class TestProducerIsTheOnlyConstructingSite:
    def test_react_agent_calls_finding_to_state(self):
        tree = ast.parse(AGENT_FILE.read_text(encoding="utf-8"))
        calls = [
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name)
            and n.func.id == "finding_to_state"
        ]
        assert calls, "react_agent no longer builds findings via finding_to_state()"

    def test_react_agent_no_longer_hand_builds_the_finding_dict(self):
        """A `findings.append({...})` literal reintroduces the drift class."""
        tree = ast.parse(AGENT_FILE.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not (
                isinstance(func, ast.Attribute)
                and func.attr == "append"
                and isinstance(func.value, ast.Name)
                and func.value.id == "findings"
            ):
                continue
            for arg in node.args:
                assert not isinstance(arg, ast.Dict), (
                    f"findings.append({{...}}) literal at line {node.lineno} — "
                    "use finding_to_state() so the key set stays in sync "
                    "with the TDA ladder and the reward shaper"
                )

    def test_react_agent_does_not_reimplement_identity(self):
        source = AGENT_FILE.read_text(encoding="utf-8")
        assert "finding_to_state" in source
