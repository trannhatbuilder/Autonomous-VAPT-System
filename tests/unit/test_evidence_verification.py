"""Regression tests for app.evidence.verifier + the evidence ladder.

Bug (found after the TDA evidence gate was introduced):

    app/agents/react_agent.py appended

        {"title": ..., "severity": ..., "verified": True, "false_positive": False}

    for EVERY finding the tool bridge reported as `status == "recorded"`.
    `tool_bridge._record_vulnerability` only rejects findings with *no*
    evidence string, so "evidence is non-empty" was silently promoted to
    "verified". Consequences:

      1. `evidence_score_from_finding()` returned CONFIRMED_VULN (0.8) for
         all of them — the 0.5 / 0.8 / 1.0 ladder collapsed to two rungs.
      2. `replay_reproduced` / `human_verified` were never set, so
         VERIFIED_EXPLOIT (1.0) was unreachable inside the loop.
      3. The TDA EXPLOIT override — advertised as evidence-gated — was in
         practice `len(findings) > 0` with extra steps.

The fix: `_record_vulnerability` runs a deterministic rule verifier over
the attached evidence, returns `{"verification": "plausible"|"confirmed"|
"replay_verified"}`, and the agent appends THAT level instead of a
hardcoded True.
"""
from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pytest

from app.evidence.verifier import (
    VERIFICATION_CONFIRMED,
    VERIFICATION_LEVELS,
    VERIFICATION_PLAUSIBLE,
    VERIFICATION_REPLAY_VERIFIED,
    assess_verification,
    classify_vuln,
    parse_verification,
    verification_rank,
)
from app.rl.tda import (
    CONFIRMED_VULN,
    PLAUSIBLE_HYPOTHESIS,
    SPECULATIVE_HYPOTHESIS,
    VERIFIED_EXPLOIT,
    evidence_score_from_finding,
    max_evidence_score,
    should_force_exploit,
)


# ── verbatim tool output samples (trimmed, never paraphrased) ──────

SQLMAP_CONFIRMED = (
    "sqlmap identified the following injection point(s) with a total of 42 HTTP(s) requests:\n"
    "---\n"
    "Parameter: id (GET)\n"
    "    Type: boolean-based blind\n"
    "    Title: AND boolean-based blind - WHERE or HAVING clause\n"
    "    Payload: id=1 AND 5310=5310\n"
    "---\n"
    "back-end DBMS: MySQL >= 5.0\n"
)

SQLMAP_BANNER_ONLY = (
    "        ___\n"
    "       __H__\n"
    " ___ ___[(]_____ ___ ___\n"
    "|_ -| . [(]     | .'| . |\n"
    "|___|_  [.]_|_|_|__,|  _|\n"
    "      |_|V...       |_|   https://sqlmap.org\n"
)

DALFOX_CONFIRMED = (
    "[POC][V][GET] http://target/search?q=abc%3Cscript%3Ealert(1)%3C%2Fscript%3E\n"
    "[V] Triggered XSS on parameter q\n"
)

NUCLEI_CONFIRMED = (
    "[cve-2020-14882] [http] [critical] https://target:7001/console/css/%252e%252e%252fconsole.portal\n"
)

ENV_LEAK_CONFIRMED = (
    "HTTP/2 200\n"
    "content-type: text/plain\n"
    "\n"
    "APP_NAME=Laravel\n"
    "APP_KEY=base64:Zm9vYmFyYmF6cXV1eA==\n"
    "DB_PASSWORD=s3cr3t-prod-password\n"
)

PASSWD_CONFIRMED = "root:x:0:0:root:/root:/bin/bash\ndaemon:x:1:1:daemon:/usr/sbin:/usr/sbin/nologin\n"

HEADER_MISC_NOTE = (
    "curl -sI https://target/\n"
    "HTTP/2 200\n"
    "server: nginx\n"
    "content-type: text/html\n"
    "X-Frame-Options header is absent from the response.\n"
)


class TestClassifyVuln:
    @pytest.mark.parametrize(
        "vuln_type,expected",
        [
            ("sqli", "sqli"),
            ("sql-injection", "sqli"),
            ("SQL Injection", "sqli"),
            ("blind-sqli", "sqli"),
            ("xss", "xss"),
            ("stored-xss", "xss"),
            ("lfi", "lfi"),
            ("path-traversal", "lfi"),
            ("rce", "rce"),
            ("command-injection", "rce"),
            ("information-disclosure", "exposure"),
            ("sensitive-file-exposure", "exposure"),
            ("default-credentials", "default_creds"),
            ("ssrf", "ssrf"),
            ("open-redirect", "open_redirect"),
        ],
    )
    def test_classes(self, vuln_type, expected):
        assert classify_vuln(vuln_type) == expected

    def test_unknown_class(self):
        assert classify_vuln("weird-vuln-class") == ""
        assert classify_vuln("") == ""
        assert classify_vuln(None) == ""


class TestAssessVerification:
    def test_no_evidence_is_plausible(self):
        level, reason = assess_verification(vuln_type="sqli", evidence="")
        assert level == VERIFICATION_PLAUSIBLE
        assert "no evidence" in reason

    def test_short_evidence_is_plausible(self):
        level, _ = assess_verification(vuln_type="sqli", evidence="vulnerable")
        assert level == VERIFICATION_PLAUSIBLE

    def test_sqlmap_verdict_confirms(self):
        level, reason = assess_verification(vuln_type="sqli", evidence=SQLMAP_CONFIRMED)
        assert level == VERIFICATION_CONFIRMED
        assert "sqli" in reason

    def test_sqlmap_banner_only_is_plausible(self):
        """Banner present, no verdict. This is exactly the phantom-sqlmap case."""
        level, reason = assess_verification(vuln_type="sqli", evidence=SQLMAP_BANNER_ONLY)
        assert level == VERIFICATION_PLAUSIBLE
        assert "no sqli verdict" in reason

    def test_dalfox_verdict_confirms(self):
        level, _ = assess_verification(vuln_type="xss", evidence=DALFOX_CONFIRMED)
        assert level == VERIFICATION_CONFIRMED

    def test_env_leak_confirms(self):
        level, _ = assess_verification(
            vuln_type="sensitive-file-exposure", evidence=ENV_LEAK_CONFIRMED
        )
        assert level == VERIFICATION_CONFIRMED

    def test_passwd_read_confirms(self):
        level, _ = assess_verification(vuln_type="lfi", evidence=PASSWD_CONFIRMED)
        assert level == VERIFICATION_CONFIRMED

    def test_nuclei_template_match_confirms_generically(self):
        """Class-agnostic tool verdict — nuclei matched a template."""
        level, _ = assess_verification(vuln_type="cve-2020-14882", evidence=NUCLEI_CONFIRMED)
        assert level == VERIFICATION_CONFIRMED

    def test_missing_header_note_is_plausible(self):
        """Evidence exists, but no tool asserted anything exploitable."""
        level, _ = assess_verification(
            vuln_type="security_misconfiguration", evidence=HEADER_MISC_NOTE
        )
        assert level == VERIFICATION_PLAUSIBLE

    def test_wrong_class_is_plausible(self):
        """A .env dump must not confirm a claimed SQLi."""
        level, _ = assess_verification(vuln_type="sqli", evidence=ENV_LEAK_CONFIRMED)
        assert level == VERIFICATION_PLAUSIBLE

    def test_replay_flag_wins(self):
        level, reason = assess_verification(
            vuln_type="sqli", evidence=SQLMAP_BANNER_ONLY, replay_reproduced=True
        )
        assert level == VERIFICATION_REPLAY_VERIFIED
        assert "replay" in reason

    def test_human_flag_wins(self):
        level, _ = assess_verification(
            vuln_type="sqli", evidence="short", human_verified=True
        )
        assert level == VERIFICATION_REPLAY_VERIFIED


class TestParseVerification:
    def test_known_levels(self):
        for level in VERIFICATION_LEVELS:
            assert parse_verification(level) == level
            assert parse_verification(level.upper()) == level

    def test_unknown_never_promotes(self):
        assert parse_verification("CONFIRMED!!") == VERIFICATION_PLAUSIBLE
        assert parse_verification(None) == VERIFICATION_PLAUSIBLE
        assert parse_verification(3) == VERIFICATION_PLAUSIBLE

    def test_ranks(self):
        assert verification_rank(VERIFICATION_PLAUSIBLE) == 1
        assert verification_rank(VERIFICATION_CONFIRMED) == 2
        assert verification_rank(VERIFICATION_REPLAY_VERIFIED) == 3
        assert verification_rank("nonsense") == 0


# ── the ladder consumed by the TDA gate ───────────────────────────


class TestVerificationLadder:
    def test_plausible_scores_half(self):
        f = {"verification": "plausible"}
        assert evidence_score_from_finding(f) == PLAUSIBLE_HYPOTHESIS

    def test_confirmed_scores_confirmed(self):
        f = {"verification": "confirmed"}
        assert evidence_score_from_finding(f) == CONFIRMED_VULN

    def test_replay_verified_scores_full(self):
        f = {"verification": "replay_verified"}
        assert evidence_score_from_finding(f) == VERIFIED_EXPLOIT

    def test_speculative_scores_speculative(self):
        assert evidence_score_from_finding({"verification": "speculative"}) == SPECULATIVE_HYPOTHESIS

    def test_verification_overrides_legacy_verified_true(self):
        """Canonical provenance wins — a stale `verified: True` can't promote."""
        f = {"verification": "plausible", "verified": True}
        assert evidence_score_from_finding(f) == PLAUSIBLE_HYPOTHESIS
        assert should_force_exploit([f]) is False

    def test_false_positive_still_wins_over_verification(self):
        f = {"verification": "confirmed", "false_positive": True}
        assert evidence_score_from_finding(f) == SPECULATIVE_HYPOTHESIS

    def test_false_positive_does_not_tie_with_unexamined_plausible(self):
        """A refutation must rank BELOW a finding nobody has checked.

        This used to return PLAUSIBLE_HYPOTHESIS (0.5), i.e. a finding the
        auditor had already disproved was worth exactly as much evidence as
        one with no verdict at all — and, being a mean, it pulled E (and
        therefore perceived difficulty) the wrong way.
        """
        refuted = {"verification": "confirmed", "false_positive": True}
        unexamined = {"verification": "plausible"}
        assert evidence_score_from_finding(refuted) < evidence_score_from_finding(unexamined)

    def test_legacy_booleans_still_work(self):
        assert evidence_score_from_finding({"verified": True}) == CONFIRMED_VULN
        assert evidence_score_from_finding({"replay_reproduced": True}) == VERIFIED_EXPLOIT
        assert evidence_score_from_finding({"human_verified": True}) == VERIFIED_EXPLOIT


# ── the actual regression: does the override fire on record? ──────


class TestOverrideDoesNotFireOnPlausibleRecord:
    """A finding that was recorded but not independently verified.

    This is the bug the user reported: 5 findings recorded, override
    fires immediately after finding #1, scan tunnels onto one vector.
    """

    def _recorded_finding(self, vuln_type: str, evidence: str) -> dict:
        """Mirror exactly what react_agent now appends."""
        level, _ = assess_verification(vuln_type=vuln_type, evidence=evidence)
        return {
            "title": f"{vuln_type} on target",
            "severity": "high",
            "vuln_type": vuln_type,
            "verification": level,
            "verified": level in (VERIFICATION_CONFIRMED, VERIFICATION_REPLAY_VERIFIED),
            "replay_reproduced": level == VERIFICATION_REPLAY_VERIFIED,
            "human_verified": False,
            "false_positive": False,
        }

    def test_plausible_record_does_not_force(self):
        f = self._recorded_finding("sqli", SQLMAP_BANNER_ONLY)
        assert f["verification"] == VERIFICATION_PLAUSIBLE
        assert f["verified"] is False
        assert should_force_exploit([f]) is False

    def test_successes_counter_ignores_plausible(self):
        findings = [
            self._recorded_finding("sqli", SQLMAP_BANNER_ONLY),
            self._recorded_finding("security_misconfiguration", HEADER_MISC_NOTE),
        ]
        successes = sum(
            1 for f in findings if evidence_score_from_finding(f) >= CONFIRMED_VULN
        )
        assert successes == 0
        assert should_force_exploit(findings) is False

    def test_confirmed_record_forces(self):
        f = self._recorded_finding("sqli", SQLMAP_CONFIRMED)
        assert f["verification"] == VERIFICATION_CONFIRMED
        assert should_force_exploit([f]) is True
        assert evidence_score_from_finding(f) == CONFIRMED_VULN

    def test_mixed_scan_forces_on_the_confirmed_one(self):
        findings = [
            self._recorded_finding("xss", "curl -s 'http://t/?q=1' -> 200 OK, no payload echo"),
            self._recorded_finding("sqli", SQLMAP_CONFIRMED),
        ]
        assert should_force_exploit(findings) is True
        assert max_evidence_score(findings) == CONFIRMED_VULN

    def test_replay_verified_reaches_top_rung(self):
        level, _ = assess_verification(
            vuln_type="sqli", evidence=SQLMAP_CONFIRMED, replay_reproduced=True
        )
        assert level == VERIFICATION_REPLAY_VERIFIED
        assert evidence_score_from_finding({"verification": level}) == VERIFIED_EXPLOIT
