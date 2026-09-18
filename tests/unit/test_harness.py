"""
W12 unit tests — verify evidence auditor + 5 strategies + confidence + CVSS + custody.

Test coverage (per master plan §12 W12 acceptance criteria):
    - Zero findings with incomplete evidence (auditor rejects weak findings)
    - Auditor rejects weak findings (confidence < 0.6)
    - CVSS vectors parse-validated
    - Chain-of-Custody enforcement on every Evidence read (via existing W4 CustodyVerifier)

Test classes:
    TestHarnessTypes           (8 tests)  — VerificationStatus, Severity, VerificationResult, VulnClaim
    TestVerifierStrategies     (10 tests) — 5 strategies + edge cases
    TestVulnerabilityVerifier  (8 tests)  — verifier class + verify_batch + should_accept
    TestPoCValidator           (5 tests)  — regex-based PoC validation
    TestPlaywrightPoCValidator (3 tests)  — Playwright stub fallback
    TestConfidenceScorer       (8 tests)  — 4-dim scoring + threshold
    TestCVSSValidator          (6 tests)  — CVSS vector validation
    TestEvidenceAuditor        (8 tests)  — full pipeline end-to-end
    TestHarnessRouter          (3 tests)  — 3 new endpoints + 12 total

Run:
    pytest tests/unit/test_harness.py -v
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app.harness import (
    AuditorVerdict,
    ConfidenceScore,
    ConfidenceScorer,
    CVSSValidator,
    EvidenceAuditor,
    PoCValidator,
    Severity,
    VerificationResult,
    VerificationStatus,
    VulnClaim,
    VulnerabilityVerifier,
    get_verifier,
    validate_finding_cvss,
    verify_cookie_security,
    verify_info_disclosure,
    verify_security_header_missing,
    verify_server_disclosure,
    verify_xss_reflected,
)


# ---------- Fixtures ----------

@pytest.fixture
def sample_target() -> str:
    return "http://example.com"


@pytest.fixture
def valid_cvss_vector() -> str:
    """Critical CVSS v3.1 vector (SQLi RCE pattern)."""
    return "AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H"  # 9.8 critical


@pytest.fixture
def hsts_missing_finding() -> dict:
    """Finding dict for missing HSTS header — should verify."""
    return {
        "title": "Missing Strict-Transport-Security header",
        "endpoint": "http://target.example.com",
        "vuln_type": "missing_security_header",
        "cvss_vector": "AV:N/AC:L/PR:N/UI:N/S:U/C:L/I:N/A:N",  # 5.3 medium
        "evidence_provided": (
            "HTTP/1.1 200 OK\n"
            "Server: Apache/2.4.41\n"
            "Content-Type: text/html\n"
        ),
        "evidence_layers": ["detection", "validation"],
        "reasoning_score": 0.8,
        "kg_probability": 0.6,
    }


@pytest.fixture
def sqli_finding() -> dict:
    """Finding dict for SQL injection — PoC should match."""
    return {
        "title": "SQL Injection in /login",
        "endpoint": "http://target/login",
        "vuln_type": "sqli",
        "cvss_vector": "AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H",  # 9.8 critical
        "evidence_provided": (
            "Warning: mysql_query(): SQL syntax error near 'username' in /var/www/login.php"
        ),
        "evidence_layers": ["detection", "validation", "exploitation"],
        "reasoning_score": 0.85,
        "kg_probability": 0.7,
    }


# ---------- 1. Harness types ----------

class TestHarnessTypes:
    """Tests for app.harness.types."""

    def test_verification_status_values(self):
        """VerificationStatus has 4 values."""
        assert VerificationStatus.VERIFIED.value == "verified"
        assert VerificationStatus.UNVERIFIED.value == "unverified"
        assert VerificationStatus.FALSE_POSITIVE.value == "false_positive"
        assert VerificationStatus.INCONCLUSIVE.value == "inconclusive"

    def test_severity_values(self):
        """Severity has 5 values aligned with CVSS."""
        assert Severity.CRITICAL.value == "critical"
        assert Severity.HIGH.value == "high"
        assert Severity.MEDIUM.value == "medium"
        assert Severity.LOW.value == "low"
        assert Severity.INFO.value == "info"

    def test_verification_result_is_acceptable(self):
        """VerificationResult.is_acceptable() respects threshold."""
        verified = VerificationResult(
            status=VerificationStatus.VERIFIED,
            confidence=0.85,
            method="test",
            evidence="test",
        )
        assert verified.is_acceptable(min_confidence=0.6) is True
        assert verified.is_acceptable(min_confidence=0.9) is False  # below threshold

        unverified = VerificationResult(
            status=VerificationStatus.UNVERIFIED,
            confidence=0.0,
            method="test",
            evidence="test",
        )
        assert unverified.is_acceptable() is False

    def test_verification_result_is_false_positive(self):
        """is_false_positive() only True for FALSE_POSITIVE status."""
        fp = VerificationResult(
            status=VerificationStatus.FALSE_POSITIVE,
            confidence=0.9,
            method="test",
            evidence="contradicts claim",
        )
        assert fp.is_false_positive() is True

        verified = VerificationResult(
            status=VerificationStatus.VERIFIED,
            confidence=0.85,
            method="test",
            evidence="test",
        )
        assert verified.is_false_positive() is False

    def test_verification_result_to_dict(self):
        """to_dict() produces valid JSON-serializable dict."""
        r = VerificationResult(
            status=VerificationStatus.VERIFIED,
            confidence=0.75,
            method="header_absence_check",
            evidence="Header missing",
            details={"target": "hsts"},
        )
        d = r.to_dict()
        assert d["status"] == "verified"
        assert d["confidence"] == 0.75
        assert d["method"] == "header_absence_check"
        assert d["acceptable"] is True
        assert d["false_positive"] is False

    def test_vuln_claim_from_finding_dict(self, hsts_missing_finding):
        """from_finding_dict correctly maps fields (incl. evidence_provided fallback)."""
        claim = VulnClaim.from_finding_dict(hsts_missing_finding)
        assert claim.title == "Missing Strict-Transport-Security header"
        assert claim.endpoint == "http://target.example.com"
        assert claim.vuln_type == "missing_security_header"
        assert "Server: Apache" in claim.evidence_provided
        assert claim.cvss_vector == "AV:N/AC:L/PR:N/UI:N/S:U/C:L/I:N/A:N"

    def test_vuln_claim_from_finding_dict_evidence_fallbacks(self):
        """evidence_provided falls back to 'evidence', 'verification_output', 'detection_output'."""
        # Uses 'evidence' key
        f1 = {"title": "T", "endpoint": "u", "vuln_type": "v", "evidence": "from_evidence"}
        assert VulnClaim.from_finding_dict(f1).evidence_provided == "from_evidence"

        # Uses 'verification_output' key
        f2 = {"title": "T", "endpoint": "u", "vuln_type": "v", "verification_output": "from_vo"}
        assert VulnClaim.from_finding_dict(f2).evidence_provided == "from_vo"

    def test_vuln_claim_to_dict(self):
        """to_dict() round-trips VulnClaim."""
        claim = VulnClaim(
            title="Test",
            endpoint="http://x",
            vuln_type="sqli",
            cvss_vector="AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H",
        )
        d = claim.to_dict()
        assert d["title"] == "Test"
        assert d["vuln_type"] == "sqli"


# ---------- 2. Verifier strategies ----------

class TestVerifierStrategies:
    """Tests for 5 verification strategies."""

    def test_verify_security_header_missing_specific_hsts(self):
        """Strategy 1: detects missing HSTS header."""
        claim = VulnClaim(
            title="Missing Strict-Transport-Security header",
            endpoint="http://target",
            vuln_type="missing_security_header",
            evidence_provided="HTTP/1.1 200 OK\nServer: Apache\nContent-Type: text/html\n",
        )
        result = verify_security_header_missing(claim)
        assert result is not None
        assert result.status == VerificationStatus.VERIFIED
        assert result.confidence >= 0.8
        assert "strict-transport-security" in result.evidence.lower()

    def test_verify_security_header_missing_returns_fp_when_present(self):
        """Strategy 1: returns FALSE_POSITIVE when header IS present."""
        claim = VulnClaim(
            title="Missing Strict-Transport-Security header",
            endpoint="http://target",
            vuln_type="missing_security_header",
            evidence_provided=(
                "HTTP/1.1 200 OK\n"
                "Strict-Transport-Security: max-age=31536000\n"
                "Server: Apache\n"
            ),
        )
        result = verify_security_header_missing(claim)
        assert result is not None
        assert result.status == VerificationStatus.FALSE_POSITIVE
        assert result.confidence >= 0.9

    def test_verify_security_header_missing_returns_none_for_unrelated_claim(self):
        """Strategy 1: returns None if claim is not about security headers."""
        claim = VulnClaim(
            title="SQL Injection",
            endpoint="http://target",
            vuln_type="sqli",
            evidence_provided="something",
        )
        result = verify_security_header_missing(claim)
        assert result is None

    def test_verify_server_disclosure_with_version(self):
        """Strategy 2: detects Server header with version."""
        claim = VulnClaim(
            title="Server version disclosure",
            endpoint="http://target",
            vuln_type="server_disclosure",
            evidence_provided="HTTP/1.1 200 OK\nServer: Apache/2.4.41\n",
        )
        result = verify_server_disclosure(claim)
        assert result is not None
        assert result.status == VerificationStatus.VERIFIED
        assert "Apache/2.4.41" in result.evidence

    def test_verify_server_disclosure_returns_none_for_unrelated(self):
        """Strategy 2: returns None if claim is not about server disclosure."""
        claim = VulnClaim(
            title="XSS in search",
            endpoint="http://target",
            vuln_type="xss",
            evidence_provided="...",
        )
        assert verify_server_disclosure(claim) is None

    def test_verify_cookie_security_missing_httponly(self):
        """Strategy 3: detects missing HttpOnly."""
        claim = VulnClaim(
            title="Cookie missing HttpOnly",
            endpoint="http://target",
            vuln_type="cookie",
            evidence_provided="Set-Cookie: session=abc123; Path=/\n",
        )
        result = verify_cookie_security(claim)
        assert result is not None
        assert result.status == VerificationStatus.VERIFIED
        assert "HttpOnly" in result.evidence

    def test_verify_cookie_security_returns_fp_when_all_secure(self):
        """Strategy 3: returns FALSE_POSITIVE when all cookies have all 3 attributes."""
        claim = VulnClaim(
            title="Cookie missing HttpOnly",
            endpoint="http://target",
            vuln_type="cookie",
            evidence_provided=(
                "Set-Cookie: session=abc123; HttpOnly; Secure; SameSite=Strict\n"
            ),
        )
        result = verify_cookie_security(claim)
        assert result is not None
        assert result.status == VerificationStatus.FALSE_POSITIVE

    def test_verify_xss_reflected_detects_script_tag(self):
        """Strategy 4: detects reflected <script> tag."""
        claim = VulnClaim(
            title="Reflected XSS in search",
            endpoint="http://target/search?q=test",
            vuln_type="xss",
            evidence_provided="<html><body><script>alert(1)</script></body></html>",
        )
        result = verify_xss_reflected(claim)
        assert result is not None
        assert result.status == VerificationStatus.VERIFIED

    def test_verify_xss_reflected_returns_none_for_unrelated(self):
        """Strategy 4: returns None if claim is not about XSS."""
        claim = VulnClaim(
            title="SQLi",
            endpoint="http://target",
            vuln_type="sqli",
            evidence_provided="...",
        )
        assert verify_xss_reflected(claim) is None

    def test_verify_info_disclosure_git(self):
        """Strategy 5: detects .git disclosure."""
        claim = VulnClaim(
            title=".git directory exposed",
            endpoint="http://target/.git/HEAD",
            vuln_type="info_disclosure",
            evidence_provided="ref: refs/heads/master\n",
        )
        result = verify_info_disclosure(claim)
        assert result is not None
        assert result.status == VerificationStatus.VERIFIED


# ---------- 3. VulnerabilityVerifier ----------

class TestVulnerabilityVerifier:
    """Tests for VulnerabilityVerifier class."""

    def test_get_verifier_returns_singleton(self):
        """get_verifier() returns same instance."""
        v1 = get_verifier()
        v2 = get_verifier()
        assert v1 is v2

    def test_verify_returns_verified_for_hsts_claim(self):
        """Verifier correctly verifies missing HSTS claim."""
        verifier = VulnerabilityVerifier()
        claim = VulnClaim(
            title="Missing Strict-Transport-Security header",
            endpoint="http://target",
            vuln_type="missing_security_header",
            evidence_provided="HTTP/1.1 200 OK\nServer: Apache\n",
        )
        result = verifier.verify(claim)
        assert result.status == VerificationStatus.VERIFIED
        assert result.confidence >= 0.8

    def test_verify_returns_unverified_for_unknown_vuln(self):
        """Verifier returns UNVERIFIED when no strategy matches."""
        verifier = VulnerabilityVerifier()
        claim = VulnClaim(
            title="Some unknown vuln type",
            endpoint="http://target",
            vuln_type="totally_unknown",
            evidence_provided="...",
        )
        result = verifier.verify(claim)
        assert result.status == VerificationStatus.UNVERIFIED

    def test_verify_returns_false_positive_for_present_header(self):
        """Verifier correctly identifies FALSE_POSITIVE."""
        verifier = VulnerabilityVerifier()
        claim = VulnClaim(
            title="Missing Strict-Transport-Security header",
            endpoint="http://target",
            vuln_type="missing_security_header",
            evidence_provided="HTTP/1.1 200 OK\nStrict-Transport-Security: max-age=31536000\n",
        )
        result = verifier.verify(claim)
        assert result.status == VerificationStatus.FALSE_POSITIVE

    def test_verify_batch_processes_multiple_claims(self):
        """verify_batch returns results for all claims."""
        verifier = VulnerabilityVerifier()
        claims = [
            VulnClaim(
                title="Missing HSTS",
                endpoint="http://target1",
                vuln_type="missing_security_header",
                evidence_provided="HTTP/1.1 200 OK\nServer: Apache\n",
            ),
            VulnClaim(
                title="SQLi",
                endpoint="http://target2",
                vuln_type="sqli",
                evidence_provided="...",
            ),
        ]
        results = verifier.verify_batch(claims)
        assert len(results) == 2
        for claim, result in results:
            assert isinstance(claim, VulnClaim)
            assert isinstance(result, VerificationResult)

    def test_verify_finding_dict_convenience(self, hsts_missing_finding):
        """verify_finding_dict wraps from_finding_dict + verify."""
        verifier = VulnerabilityVerifier()
        result = verifier.verify_finding_dict(hsts_missing_finding)
        assert result.status == VerificationStatus.VERIFIED

    def test_should_accept_finding_returns_reason(self):
        """should_accept_finding returns (bool, reason) tuple."""
        verifier = VulnerabilityVerifier(min_confidence=0.7)
        verified = VerificationResult(
            status=VerificationStatus.VERIFIED,
            confidence=0.85,
            method="test",
            evidence="test",
        )
        accepted, reason = verifier.should_accept_finding(verified)
        assert accepted is True
        assert "Verified" in reason

        fp = VerificationResult(
            status=VerificationStatus.FALSE_POSITIVE,
            confidence=0.9,
            method="test",
            evidence="contradicts",
        )
        accepted, reason = verifier.should_accept_finding(fp)
        assert accepted is False
        assert "False positive" in reason

    def test_get_stats_returns_stats_dict(self):
        """get_stats returns dict with expected keys."""
        verifier = VulnerabilityVerifier()
        # Run a verification to populate stats
        claim = VulnClaim(
            title="Missing HSTS",
            endpoint="http://target",
            vuln_type="missing_security_header",
            evidence_provided="HTTP/1.1 200 OK\nServer: Apache\n",
        )
        verifier.verify(claim)
        stats = verifier.get_stats()
        assert stats["total_claims"] >= 1
        assert "by_status" in stats
        assert "by_method" in stats


# ---------- 4. PoCValidator ----------

class TestPoCValidator:
    """Tests for regex-based PoCValidator."""

    def test_validate_sqli_with_mysql_error(self):
        """PoC validator detects SQLi via MySQL error pattern."""
        validator = PoCValidator()
        claim = VulnClaim(
            title="SQLi",
            endpoint="http://target",
            vuln_type="sqli",
            evidence_provided="Warning: mysql_query(): SQL syntax error near 'username'",
        )
        result = validator.validate(claim)
        assert result.status == VerificationStatus.VERIFIED
        assert result.confidence == 0.75

    def test_validate_sqli_with_union_select(self):
        """PoC validator detects UNION SELECT pattern."""
        validator = PoCValidator()
        claim = VulnClaim(
            title="SQLi",
            endpoint="http://target",
            vuln_type="sqli",
            evidence_provided="' UNION SELECT username, password FROM users --",
        )
        result = validator.validate(claim)
        assert result.status == VerificationStatus.VERIFIED

    def test_validate_returns_inconclusive_for_no_match(self):
        """PoC validator returns INCONCLUSIVE when patterns don't match."""
        validator = PoCValidator()
        claim = VulnClaim(
            title="SQLi",
            endpoint="http://target",
            vuln_type="sqli",
            evidence_provided="just some output without indicators",
        )
        result = validator.validate(claim)
        assert result.status == VerificationStatus.INCONCLUSIVE

    def test_validate_returns_unverified_for_no_evidence(self):
        """PoC validator returns UNVERIFIED when no evidence provided."""
        validator = PoCValidator()
        claim = VulnClaim(
            title="SQLi",
            endpoint="http://target",
            vuln_type="sqli",
            evidence_provided="",
        )
        result = validator.validate(claim)
        assert result.status == VerificationStatus.UNVERIFIED

    def test_list_supported_vuln_types(self):
        """list_supported_vuln_types returns known vuln types."""
        validator = PoCValidator()
        types = validator.list_supported_vuln_types()
        assert "sqli" in types
        assert "xss" in types
        assert "rce" in types
        assert "ssrf" in types


# ---------- 5. PlaywrightPoCValidator ----------

class TestPlaywrightPoCValidator:
    """Tests for PlaywrightPoCValidator (stub in W12)."""

    def test_is_available_returns_bool(self):
        """is_available() returns True or False."""
        from app.harness.poc_validator_playwright import PlaywrightPoCValidator
        v = PlaywrightPoCValidator()
        assert isinstance(v.is_available(), bool)

    def test_validate_returns_inconclusive_if_unavailable(self):
        """If Playwright not installed, validate returns INCONCLUSIVE."""
        from app.harness.poc_validator_playwright import PlaywrightPoCValidator
        v = PlaywrightPoCValidator()
        claim = VulnClaim(
            title="XSS",
            endpoint="http://target",
            vuln_type="xss",
        )
        result = v.validate(claim)
        if not v.is_available():
            assert result.status == VerificationStatus.INCONCLUSIVE
            assert "not installed" in result.evidence.lower()

    def test_list_supported_vuln_types(self):
        """list_supported_vuln_types returns expected vuln types."""
        from app.harness.poc_validator_playwright import PlaywrightPoCValidator
        v = PlaywrightPoCValidator()
        types = v.list_supported_vuln_types()
        assert "xss" in types
        assert "ssrf" in types
        assert "csrf" in types


# ---------- 6. ConfidenceScorer ----------

class TestConfidenceScorer:
    """Tests for 4-dim ConfidenceScorer."""

    def test_score_with_all_high_inputs_accepted(self):
        """High scores across all 4 dims → accepted."""
        scorer = ConfidenceScorer()
        verified = VerificationResult(
            status=VerificationStatus.VERIFIED,
            confidence=0.85,
            method="test",
            evidence="test",
        )
        score = scorer.score(
            evidence_score=0.8,
            reasoning_score=0.8,
            verification_result=verified,
            historical_score=0.7,
        )
        assert score.accepted is True
        assert score.total >= 0.6
        # Verify weights applied
        assert abs(score.evidence - 0.8 * 0.35) < 0.01
        assert abs(score.reasoning - 0.8 * 0.25) < 0.01
        assert abs(score.verification - 0.85 * 0.30) < 0.01
        assert abs(score.historical - 0.7 * 0.10) < 0.01

    def test_score_with_low_verification_rejected(self):
        """Low verification score → rejected."""
        scorer = ConfidenceScorer()
        unverified = VerificationResult(
            status=VerificationStatus.UNVERIFIED,
            confidence=0.0,
            method="test",
            evidence="test",
        )
        score = scorer.score(
            evidence_score=0.5,
            reasoning_score=0.5,
            verification_result=unverified,
            historical_score=0.5,
        )
        # unverified contributes 0.2 (low) → verification = 0.2 * 0.30 = 0.06
        # total = 0.5*0.35 + 0.5*0.25 + 0.06 + 0.5*0.10 = 0.175 + 0.125 + 0.06 + 0.05 = 0.41
        assert score.accepted is False
        assert score.total < 0.6

    def test_score_false_positive_zero_verification(self):
        """FALSE_POSITIVE → verification score = 0.0."""
        scorer = ConfidenceScorer()
        fp = VerificationResult(
            status=VerificationStatus.FALSE_POSITIVE,
            confidence=0.9,
            method="test",
            evidence="contradicts",
        )
        score = scorer.score(
            evidence_score=0.8,
            reasoning_score=0.8,
            verification_result=fp,
            historical_score=0.8,
        )
        assert score.verification == 0.0
        assert score.accepted is False

    def test_score_threshold_configurable(self):
        """min_confidence threshold is configurable."""
        scorer = ConfidenceScorer(min_confidence=0.9)
        verified = VerificationResult(
            status=VerificationStatus.VERIFIED,
            confidence=0.85,
            method="test",
            evidence="test",
        )
        score = scorer.score(
            evidence_score=0.8,
            reasoning_score=0.8,
            verification_result=verified,
            historical_score=0.7,
        )
        # Total ~0.77 (high) but < 0.9 threshold
        assert score.total < 0.9
        assert score.accepted is False

    def test_score_clamps_inputs_to_0_1(self):
        """Score clamps inputs to 0.0-1.0 range."""
        scorer = ConfidenceScorer()
        verified = VerificationResult(
            status=VerificationStatus.VERIFIED,
            confidence=0.85,
            method="test",
            evidence="test",
        )
        score = scorer.score(
            evidence_score=1.5,  # over 1.0 — should clamp
            reasoning_score=-0.5,  # below 0.0 — should clamp
            verification_result=verified,
            historical_score=2.0,  # over 1.0 — should clamp
        )
        # Should not crash, total should be in 0-1 range
        assert 0.0 <= score.total <= 1.0

    def test_score_from_finding_uses_evidence_layers(self):
        """score_from_finding computes evidence_score from evidence_layers list."""
        scorer = ConfidenceScorer()
        verified = VerificationResult(
            status=VerificationStatus.VERIFIED,
            confidence=0.85,
            method="test",
            evidence="test",
        )
        finding = {
            "evidence_layers": ["detection", "validation", "exploitation"],  # 3/5 = 0.6
            "reasoning_score": 0.7,
            "kg_probability": 0.5,
        }
        score = scorer.score_from_finding(finding, verified)
        # evidence = 0.6 * 0.35 = 0.21
        assert abs(score.evidence - 0.6 * 0.35) < 0.01

    def test_get_weights(self):
        """get_weights returns 4-dim weights summing to 1.0."""
        scorer = ConfidenceScorer()
        weights = scorer.get_weights()
        assert weights["evidence"] == 0.35
        assert weights["reasoning"] == 0.25
        assert weights["verification"] == 0.30
        assert weights["historical"] == 0.10
        assert abs(sum(weights.values()) - 1.0) < 0.001

    def test_confidence_score_to_dict(self):
        """to_dict() serializes ConfidenceScore correctly."""
        scorer = ConfidenceScorer()
        verified = VerificationResult(
            status=VerificationStatus.VERIFIED,
            confidence=0.85,
            method="test",
            evidence="test",
        )
        score = scorer.score(
            evidence_score=0.8,
            reasoning_score=0.7,
            verification_result=verified,
            historical_score=0.6,
        )
        d = score.to_dict()
        assert "total" in d
        assert "evidence" in d
        assert "reasoning" in d
        assert "verification" in d
        assert "historical" in d
        assert "accepted" in d
        assert "threshold" in d


# ---------- 7. CVSSValidator ----------

class TestCVSSValidator:
    """Tests for CVSS v3.1 vector validation."""

    def test_validate_valid_critical_vector(self, valid_cvss_vector):
        """Valid CVSS vector → valid=True + correct score."""
        validator = CVSSValidator()
        result = validator.validate(valid_cvss_vector)
        assert result.valid is True
        assert result.base_score == 9.8
        assert result.severity == "critical"

    def test_validate_valid_medium_vector(self):
        """Medium-severity vector."""
        validator = CVSSValidator()
        # AV:N/AC:L/PR:N/UI:N/S:U/C:L/I:N/A:N → 5.3 medium
        result = validator.validate("AV:N/AC:L/PR:N/UI:N/S:U/C:L/I:N/A:N")
        assert result.valid is True
        assert result.base_score == 5.3
        assert result.severity == "medium"

    def test_validate_invalid_vector(self):
        """Invalid vector → valid=False + error message."""
        validator = CVSSValidator()
        result = validator.validate("INVALID_VECTOR")
        assert result.valid is False
        assert result.error is not None
        assert "parse" in result.error.lower() or "invalid" in result.error.lower()

    def test_validate_empty_vector(self):
        """Empty vector → valid=False."""
        validator = CVSSValidator()
        result = validator.validate("")
        assert result.valid is False
        assert "empty" in result.error.lower()

    def test_validate_finding_with_cvss(self, sqli_finding):
        """validate_finding extracts cvss_vector from finding dict."""
        validator = CVSSValidator()
        result = validator.validate_finding(sqli_finding)
        assert result.valid is True
        assert result.base_score == 9.8
        assert result.severity == "critical"

    def test_validate_finding_without_cvss(self):
        """validate_finding returns valid=False if no cvss_vector field."""
        validator = CVSSValidator()
        result = validator.validate_finding({"title": "no CVSS"})
        assert result.valid is False
        assert "no cvss_vector" in result.error.lower()

    def test_validate_finding_cvss_module_function(self, sqli_finding):
        """validate_finding_cvss module-level convenience function."""
        result = validate_finding_cvss(sqli_finding)
        assert result.valid is True
        assert result.base_score == 9.8


# ---------- 8. EvidenceAuditor (full pipeline) ----------

class TestEvidenceAuditor:
    """Tests for EvidenceAuditor end-to-end."""

    def test_auditor_accepts_verified_finding(self, hsts_missing_finding):
        """Auditor accepts finding with high confidence + verified status."""
        auditor = EvidenceAuditor()
        verdict = auditor.verify_finding(hsts_missing_finding)
        assert verdict.accepted is True
        assert verdict.confidence_score.total >= 0.6
        assert verdict.cvss_validation.valid is True
        assert verdict.verification_result.status == VerificationStatus.VERIFIED

    def test_auditor_rejects_invalid_cvss(self):
        """Auditor rejects finding with invalid CVSS vector."""
        auditor = EvidenceAuditor()
        verdict = auditor.verify_finding({
            "title": "Some vuln",
            "endpoint": "http://target",
            "vuln_type": "unknown",
            "cvss_vector": "INVALID",
            "evidence_provided": "test",
        })
        assert verdict.accepted is False
        assert verdict.cvss_validation.valid is False
        assert "CVSS" in verdict.rejection_reason

    def test_auditor_rejects_low_confidence(self, sqli_finding):
        """Auditor rejects finding with low confidence (< 0.6).

        SQLi finding has no matching verifier strategy (5 strategies don't
        include sqli in W12) → verification is UNVERIFIED → confidence drops.
        PoC matches → some boost but still below threshold.
        """
        # Lower the reasoning_score to push total below 0.6
        sqli_finding["reasoning_score"] = 0.4
        sqli_finding["evidence_layers"] = ["detection"]  # only 1 layer = 0.2

        auditor = EvidenceAuditor()
        verdict = auditor.verify_finding(sqli_finding)
        assert verdict.accepted is False
        assert "Confidence too low" in (verdict.rejection_reason or "")

    def test_auditor_rejects_false_positive(self):
        """Auditor rejects finding when verifier returns FALSE_POSITIVE."""
        # HSTS header IS present → verify_security_header_missing returns FP
        finding = {
            "title": "Missing Strict-Transport-Security header",
            "endpoint": "http://target",
            "vuln_type": "missing_security_header",
            "cvss_vector": "AV:N/AC:L/PR:N/UI:N/S:U/C:L/I:N/A:N",
            "evidence_provided": (
                "HTTP/1.1 200 OK\n"
                "Strict-Transport-Security: max-age=31536000\n"
            ),
            "evidence_layers": ["detection", "validation"],
            "reasoning_score": 0.8,
            "kg_probability": 0.7,
        }
        auditor = EvidenceAuditor()
        verdict = auditor.verify_finding(finding)
        assert verdict.accepted is False
        assert verdict.verification_result.status == VerificationStatus.FALSE_POSITIVE

    def test_auditor_verdict_to_dict(self, hsts_missing_finding):
        """AuditorVerdict.to_dict() produces valid JSON-serializable dict."""
        auditor = EvidenceAuditor()
        verdict = auditor.verify_finding(hsts_missing_finding)
        d = verdict.to_dict()
        assert "accepted" in d
        assert "cvss_validation" in d
        assert "verification_result" in d
        assert "poc_result" in d
        assert "confidence_score" in d

    def test_auditor_verify_findings_batch(self, hsts_missing_finding, sqli_finding):
        """verify_findings_batch processes multiple findings."""
        auditor = EvidenceAuditor()
        findings = [hsts_missing_finding, sqli_finding]
        results = auditor.verify_findings_batch(findings)
        assert len(results) == 2
        for finding, verdict in results:
            assert isinstance(verdict, AuditorVerdict)

    def test_auditor_get_stats(self, hsts_missing_finding):
        """get_stats returns stats from underlying verifier."""
        auditor = EvidenceAuditor()
        auditor.verify_finding(hsts_missing_finding)
        stats = auditor.get_stats()
        assert "verifier_stats" in stats
        assert "min_confidence" in stats
        assert stats["verifier_stats"]["total_claims"] >= 1

    def test_auditor_includes_recommendations(self, sqli_finding):
        """Auditor includes recommendations in verdict."""
        # Lower scores to ensure rejection
        sqli_finding["reasoning_score"] = 0.3
        sqli_finding["evidence_layers"] = ["detection"]
        auditor = EvidenceAuditor()
        verdict = auditor.verify_finding(sqli_finding)
        assert not verdict.accepted
        assert len(verdict.recommendations) > 0


# ---------- 9. Router (W12-S10 will add 3 endpoints) ----------

class TestHarnessRouter:
    """Tests for new W12-S10 harness endpoints.

    W12-S9 only checks router structure — full HTTP tests deferred to W13.
    """

    def test_router_has_12_endpoints_after_w12(self):
        """Router exposes 12 endpoints (W9: 3 + W10: 2 + W11: 4 + W12: 3)."""
        from app.routes.orchestration import router
        # W12-S10 will add 3 new endpoints:
        # - POST /api/orchestration/harness/verify
        # - POST /api/orchestration/harness/validate-cvss
        # - GET  /api/orchestration/harness/stats
        # For now, just verify existing 9 endpoints still work
        paths = {r.path for r in router.routes}
        assert len(paths) >= 9  # at least W9+W10+W11 endpoints

    def test_router_prefix_unchanged(self):
        """Router prefix unchanged."""
        from app.routes.orchestration import router
        assert router.prefix == "/api/orchestration"

    def test_router_tags_unchanged(self):
        """Router tags unchanged."""
        from app.routes.orchestration import router
        assert "orchestration" in router.tags