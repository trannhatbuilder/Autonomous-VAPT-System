"""
Tests for app.harness.self_correction — RejectionAnalyzer + SelfCorrectionEngine.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

os.environ.setdefault("VAPT_AI_POSTGRES_DB", "postgresql://vapt:vapt@localhost:5432/vapt_ai_test")
os.environ.setdefault("VAPT_AI_ENVIRONMENT", "dev")

import pytest

from app.harness.self_correction import (
    SelfCorrectionEngine, RejectionAnalyzer, RejectionAnalysis,
    get_self_correction_engine, reset_self_correction_singleton,
    process_verification_rejection,
)
from app.harness.types import VerificationResult, VerificationStatus, VulnClaim


@pytest.fixture
def claim():
    return VulnClaim(
        title="SQL Injection in /login",
        endpoint="https://target.com/login?id=123",
        vuln_type="SQL Injection",
        payload="1' OR 1=1 --",
        evidence_provided="sqlmap: vulnerable",
        severity="high",
    )


@pytest.fixture
def fp_result():
    return VerificationResult(
        status=VerificationStatus.FALSE_POSITIVE,
        confidence=0.9,
        method="curl_reprobe",
        evidence="Pattern not found in response — false positive",
    )


class TestRejectionAnalyzer:
    def test_analyze_false_positive(self, claim, fp_result):
        analyzer = RejectionAnalyzer()
        analysis = analyzer.analyze(claim, fp_result)
        assert analysis.rejection_type == "false_positive"
        assert "not present" in analysis.root_cause
        assert analysis.confidence_penalty == 0.3
        assert len(analysis.pattern_signature) == 16

    def test_analyze_inconclusive(self, claim):
        result = VerificationResult(
            status=VerificationStatus.INCONCLUSIVE,
            confidence=0.5,
            method="header_check",
            evidence="Network error prevented verification",
        )
        analyzer = RejectionAnalyzer()
        analysis = analyzer.analyze(claim, result)
        assert analysis.rejection_type == "inconclusive"
        assert analysis.confidence_penalty == 0.2

    def test_analyze_unverified_no_payload(self):
        """Unverified with no payload → specific root cause."""
        claim = VulnClaim(
            title="Missing HSTS",
            endpoint="https://target.com/",
            vuln_type="Header",
            payload="",  # no payload
            evidence_provided="N/A",
        )
        result = VerificationResult(
            status=VerificationStatus.UNVERIFIED,
            confidence=0.3,
            method="header_check",
            evidence="Could not verify",
        )
        analyzer = RejectionAnalyzer()
        analysis = analyzer.analyze(claim, result)
        assert analysis.rejection_type == "unverified"
        assert "payload" in analysis.root_cause.lower()

    def test_pattern_signature_deterministic(self, claim, fp_result):
        """Same (vuln_type, endpoint) → same pattern_signature."""
        analyzer = RejectionAnalyzer()
        a1 = analyzer.analyze(claim, fp_result)
        a2 = analyzer.analyze(claim, fp_result)
        assert a1.pattern_signature == a2.pattern_signature

    def test_different_endpoints_different_signatures(self):
        """Different endpoints → different pattern signatures."""
        analyzer = RejectionAnalyzer()
        c1 = VulnClaim(title="SQLi", endpoint="/login?id=1", vuln_type="SQLi")
        c2 = VulnClaim(title="SQLi", endpoint="/search?q=1", vuln_type="SQLi")
        r = VerificationResult(VerificationStatus.FALSE_POSITIVE, 0.9, "m", "e")
        a1 = analyzer.analyze(c1, r)
        a2 = analyzer.analyze(c2, r)
        assert a1.pattern_signature != a2.pattern_signature

    def test_endpoint_normalization(self):
        """Numeric IDs are normalized to {id} for pattern matching."""
        analyzer = RejectionAnalyzer()
        c1 = VulnClaim(title="SQLi", endpoint="/users/123/posts", vuln_type="SQLi")
        c2 = VulnClaim(title="SQLi", endpoint="/users/456/posts", vuln_type="SQLi")
        r = VerificationResult(VerificationStatus.FALSE_POSITIVE, 0.9, "m", "e")
        a1 = analyzer.analyze(c1, r)
        a2 = analyzer.analyze(c2, r)
        # Normalized to /users/{id}/posts → same signature
        assert a1.pattern_signature == a2.pattern_signature


class TestSelfCorrectionEngine:
    def test_process_rejection_returns_negative_reward(self, claim, fp_result):
        engine = SelfCorrectionEngine()
        result = engine.process_rejection(claim, fp_result, session_id="s1")
        assert result["rl_reward"] < 0
        assert result["pattern_blacklisted"] is False  # first rejection
        assert result["pattern_rejection_count"] == 1

    def test_escalating_penalty(self, claim, fp_result):
        """Repeated rejections of same pattern → more negative reward."""
        engine = SelfCorrectionEngine()
        rewards = []
        for _ in range(4):
            r = engine.process_rejection(claim, fp_result, session_id="s1")
            rewards.append(r["rl_reward"])
        # Rewards should get more negative
        assert rewards[-1] < rewards[0]

    def test_blacklist_after_3_rejections(self, claim, fp_result):
        """After 3+ rejections of same pattern → blacklisted."""
        engine = SelfCorrectionEngine()
        for _ in range(3):
            engine.process_rejection(claim, fp_result, session_id="s1")
        result = engine.process_rejection(claim, fp_result, session_id="s1")
        assert result["pattern_blacklisted"] is True
        assert "BLACKLISTED" in result["corrective_message"]

    def test_different_patterns_tracked_separately(self, claim, fp_result):
        """Different vuln types have separate rejection counts."""
        engine = SelfCorrectionEngine()
        engine.process_rejection(claim, fp_result, session_id="s1")
        claim2 = VulnClaim(title="XSS", endpoint="/search", vuln_type="XSS",
                          payload="<script>", evidence_provided="reflected")
        result2 = engine.process_rejection(claim2, fp_result, session_id="s1")
        assert result2["pattern_rejection_count"] == 1  # fresh pattern

    def test_corrective_message_contains_root_cause(self, claim, fp_result):
        engine = SelfCorrectionEngine()
        result = engine.process_rejection(claim, fp_result, session_id="s1")
        assert result["corrective_message"]
        assert "CORRECTION" in result["corrective_message"]

    def test_engine_stats(self, claim, fp_result):
        engine = SelfCorrectionEngine()
        for _ in range(3):
            engine.process_rejection(claim, fp_result, session_id="s1")
        stats = engine.get_stats()
        assert stats["total_patterns_tracked"] == 1
        assert stats["total_rejections"] == 3
        assert stats["blacklisted_patterns"] == 1

    def test_singleton(self):
        """get_self_correction_engine returns same instance."""
        reset_self_correction_singleton()
        e1 = get_self_correction_engine()
        e2 = get_self_correction_engine()
        assert e1 is e2


class TestProcessVerificationRejection:
    def test_convenience_function(self):
        """process_verification_rejection converts finding dict → VulnClaim."""
        finding = {
            "name": "SQL Injection",
            "severity": "high",
            "location": "http://target.com/login?id=1",
            "vuln_type": "SQLi",
        }
        result = VerificationResult(
            status=VerificationStatus.FALSE_POSITIVE,
            confidence=0.9,
            method="curl_reprobe",
            evidence="Not found",
        )
        out = process_verification_rejection(finding, result, session_id="s1")
        assert out["rl_reward"] < 0
        assert "rejection_analysis" in out