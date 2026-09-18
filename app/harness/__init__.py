"""
VAPT-AI Harness Package — W12.

Evidence auditor + FP filtering + CVSS validator + chain-of-custody enforcement.

Per master plan §12 W12:
    - 5 verification strategies (port from EVVO vulnerability_verifier.py)
    - Playwright-based PoC validator (port from EVVO poc_validator_playwright.py)
    - Cross-tool validation (nuclei → sqlmap confirm)
    - 4-dim confidence scoring (Evidence 0.35 + Reasoning 0.25 + Verification 0.30 + Historical 0.10)
    - FP filtering (findings below 0.6 → rejected)
    - CVSS v3.1 vector parse-validation (reuse app/evidence/cvss.py from W2)
    - Chain-of-Custody enforcement on every Evidence read (reuse app/evidence/custody.py from W4)

Public API:
    - VerificationStatus: enum (verified, unverified, false_positive, inconclusive)
    - Severity: enum (critical, high, medium, low, info)
    - VerificationResult: dataclass (status, confidence, method, evidence)
    - VulnClaim: dataclass (title, endpoint, vuln_type, evidence, ...)
    - VulnerabilityVerifier: multi-strategy verifier class
    - 5 strategies: verify_security_header_missing, verify_server_disclosure,
      verify_cookie_security, verify_xss_reflected, verify_info_disclosure
    - PoCValidator: regex-based fallback validator
    - PlaywrightPoCValidator: Playwright-based validator (stub if playwright not installed)
    - ConfidenceScorer: 4-dim confidence scoring
    - CVSSValidator: wraps app/evidence/cvss.py for finding-level validation
    - EvidenceAuditor: orchestrator class combining all layers

Architecture:
    ┌────────────────────────────────────────────────────────────────┐
    │                  EvidenceAuditor (orchestrator)                │
    │                       app/harness/auditor.py                   │
    │                                                                │
    │   verify_finding(finding_dict) → AuditorVerdict                │
    │                                                                │
    │   1. Validate CVSS vector (CVSSValidator)                      │
    │   2. Run 5 verification strategies (VulnerabilityVerifier)     │
    │   3. Run PoC validator (PoCValidator or PlaywrightPoCValidator)│
    │   4. Cross-tool validation (if applicable)                     │
    │   5. Compute 4-dim confidence score (ConfidenceScorer)         │
    │   6. Enforce chain-of-custody (CustodyVerifier)                │
    │   7. Return verdict (accept/reject/inconclusive)               │
    └────────────────────────────────────────────────────────────────┘
                              │
       ┌──────────────────────┼──────────────────────────┐
       ▼                      ▼                          ▼
  ┌─────────────┐    ┌─────────────────┐    ┌──────────────────┐
  │ Verifier    │    │ PoC Validator   │    │ ConfidenceScorer │
  │5 strategies │    │ (regex + PW)    │    │ (4-dim)          │
  └─────────────┘    └─────────────────┘    └──────────────────┘
"""
from __future__ import annotations

from app.harness.types import (
    Severity,
    VerificationResult,
    VerificationStatus,
    VulnClaim,
)
from app.harness.verifier_strategies import (
    verify_cookie_security,
    verify_info_disclosure,
    verify_security_header_missing,
    verify_server_disclosure,
    verify_xss_reflected,
)
from app.harness.verifier import VulnerabilityVerifier, get_verifier
from app.harness.poc_validator import PoCValidator
from app.harness.confidence import ConfidenceScorer, ConfidenceScore
from app.harness.cvss_validator import CVSSValidator, validate_finding_cvss
from app.harness.auditor import EvidenceAuditor, AuditorVerdict

__all__ = [
    # Types
    "VerificationStatus",
    "Severity",
    "VerificationResult",
    "VulnClaim",
    # Strategies
    "verify_security_header_missing",
    "verify_server_disclosure",
    "verify_cookie_security",
    "verify_xss_reflected",
    "verify_info_disclosure",
    # Verifier
    "VulnerabilityVerifier",
    "get_verifier",
    # PoC validators
    "PoCValidator",
    # Confidence
    "ConfidenceScorer",
    "ConfidenceScore",
    # CVSS
    "CVSSValidator",
    "validate_finding_cvss",
    # Auditor
    "EvidenceAuditor",
    "AuditorVerdict",
]