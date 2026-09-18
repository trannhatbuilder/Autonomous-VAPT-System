"""
VAPT-AI CVSS Validator — W12-S7.

Wraps app/evidence/cvss.py (W2) for finding-level validation.
Per master plan §12 W12: "CVSS v3.1 vector parse-validation: every finding
must have valid CVSS vector string."

Usage:
    from app.harness.cvss_validator import CVSSValidator, validate_finding_cvss

    validator = CVSSValidator()
    result = validator.validate("AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H")
    if result.valid:
        print(result.base_score)  # 9.8
        print(result.severity)    # "critical"
    else:
        print(result.error)       # parse error message
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Reuse W2's cvss.py (already in app/evidence/cvss.py)
from app.evidence.cvss import (
    CVSSMetrics,
    calculate_base_score,
    calculate_cvss,
    get_severity,
    parse_cvss_vector,
)


# ---------- CVSSValidationResult dataclass ----------

@dataclass(frozen=True)
class CVSSValidationResult:
    """Result of validating a CVSS v3.1 vector string.

    Attributes:
        valid: True if vector parsed successfully
        vector: The input vector string
        base_score: Calculated base score (0.0-10.0) if valid
        severity: Severity level (critical/high/medium/low/none) if valid
        metrics: Parsed CVSSMetrics if valid
        error: Error message if invalid (None if valid)
    """
    valid: bool
    vector: str
    base_score: float | None = None
    severity: str | None = None
    metrics: CVSSMetrics | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "vector": self.vector,
            "base_score": self.base_score,
            "severity": self.severity,
            "error": self.error,
        }


# ---------- CVSSValidator class ----------

class CVSSValidator:
    """CVSS v3.1 vector validator.

    Wraps app/evidence/cvss.py (W2) with finding-level validation.
    """

    def validate(self, vector: str) -> CVSSValidationResult:
        """Validate a CVSS v3.1 vector string.

        Args:
            vector: CVSS v3.1 vector string (e.g. "AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H")

        Returns:
            CVSSValidationResult with valid=True/False + parsed metrics if valid.
        """
        if not vector or not isinstance(vector, str):
            return CVSSValidationResult(
                valid=False,
                vector=str(vector) if vector else "",
                error="CVSS vector is empty or not a string",
            )

        vector = vector.strip()

        # Try to parse the vector
        try:
            metrics = parse_cvss_vector(vector)
        except (ValueError, KeyError) as e:
            return CVSSValidationResult(
                valid=False,
                vector=vector,
                error=f"CVSS vector parse error: {e}",
            )
        except Exception as e:
            return CVSSValidationResult(
                valid=False,
                vector=vector,
                error=f"Unexpected error parsing CVSS vector: {e}",
            )

        # Calculate base score
        try:
            base_score = calculate_base_score(metrics)
        except Exception as e:
            return CVSSValidationResult(
                valid=False,
                vector=vector,
                error=f"CVSS base score calculation failed: {e}",
            )

        # Determine severity
        severity = get_severity(base_score)

        return CVSSValidationResult(
            valid=True,
            vector=vector,
            base_score=base_score,
            severity=severity,
            metrics=metrics,
        )

    def validate_finding(self, finding: dict[str, Any]) -> CVSSValidationResult:
        """Validate CVSS vector in a finding dict.

        Args:
            finding: Finding dict with 'cvss_vector' field.

        Returns:
            CVSSValidationResult. If finding has no cvss_vector, returns
            valid=False with error.
        """
        vector = finding.get("cvss_vector", "")
        if not vector:
            return CVSSValidationResult(
                valid=False,
                vector="",
                error="Finding has no cvss_vector field",
            )
        return self.validate(vector)

    def get_severity_label(self, score: float) -> str:
        """Get severity label from CVSS score (0.0-10.0).

        Convenience wrapper around app.evidence.cvss.get_severity.

        Args:
            score: CVSS base score (0.0-10.0)

        Returns:
            Severity label: "none" / "low" / "medium" / "high" / "critical"
        """
        return get_severity(score)


# ---------- Module-level convenience functions ----------

def validate_finding_cvss(finding: dict[str, Any]) -> CVSSValidationResult:
    """Convenience: validate CVSS vector in a finding dict.

    Args:
        finding: Finding dict with 'cvss_vector' field.

    Returns:
        CVSSValidationResult.
    """
    return CVSSValidator().validate_finding(finding)


def validate_cvss_vector(vector: str) -> CVSSValidationResult:
    """Convenience: validate a CVSS v3.1 vector string.

    Args:
        vector: CVSS v3.1 vector string.

    Returns:
        CVSSValidationResult.
    """
    return CVSSValidator().validate(vector)


# ---------- Singleton instance ----------

_cvss_validator_singleton: CVSSValidator | None = None


def get_cvss_validator() -> CVSSValidator:
    """Get singleton CVSSValidator instance."""
    global _cvss_validator_singleton
    if _cvss_validator_singleton is None:
        _cvss_validator_singleton = CVSSValidator()
    return _cvss_validator_singleton