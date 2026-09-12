"""
VAPT-AI CVSS v3.1 calculator — D22.

Parses CVSS v3.1 vector strings + computes base score using the official
FIRST.org specification.

CVSS v3.1 vector format:
    AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H

Metrics:
    AV  Attack Vector          N(etwork) / A(djacent) / L(ocal) / P(hysical)
    AC  Attack Complexity      L(ow) / H(igh)
    PR  Privileges Required    N(one) / L(ow) / H(igh)
    UI  User Interaction       N(one) / R(equired)
    S   Scope                  U(nchanged) / C(hanged)
    C   Confidentiality        H(igh) / L(ow) / N(one)
    I   Integrity             H(igh) / L(ow) / N(one)
    A   Availability          H(igh) / L(ow) / N(one)

Base score formula (FIRST.org CVSS v3.1):
    ISS = 1 - [(1-C)*(1-I)*(1-A)]
    Impact = S==U ? 6.42*ISS : 7.52*(ISS-0.029) - 3.25*(ISS-0.02)^15
    Exploitability = 8.22 * AV * AC * PR * UI
    BaseScore (S==U) = roundup(min(Impact+Exploitability, 10))
    BaseScore (S==C) = roundup(min(1.08*(Impact+Exploitability), 10))

Severity rating:
    0.0       → None
    0.1-3.9   → Low
    4.0-6.9   → Medium
    7.0-8.9   → High
    9.0-10.0  → Critical

Usage:
    from app.evidence.cvss import parse_cvss_vector, calculate_base_score, get_severity

    vector = "AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H"
    metrics = parse_cvss_vector(vector)
    score = calculate_base_score(metrics)  # 9.8
    severity = get_severity(score)  # "critical"
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

# ---------- Metric value tables (CVSS v3.1 spec) ----------

# Attack Vector (AV)
AV_VALUES = {"N": 0.85, "A": 0.62, "L": 0.55, "P": 0.2}
AV_NAMES = {"N": "Network", "A": "Adjacent", "L": "Local", "P": "Physical"}

# Attack Complexity (AC)
AC_VALUES = {"L": 0.77, "H": 0.44}
AC_NAMES = {"L": "Low", "H": "High"}

# Privileges Required (PR) — depends on Scope
PR_VALUES_UNCHANGED = {"N": 0.85, "L": 0.62, "H": 0.27}
PR_VALUES_CHANGED = {"N": 0.85, "L": 0.68, "H": 0.5}
PR_NAMES = {"N": "None", "L": "Low", "H": "High"}

# User Interaction (UI)
UI_VALUES = {"N": 0.85, "R": 0.62}
UI_NAMES = {"N": "None", "R": "Required"}

# Scope (S) — not numeric, but affects formula
S_VALUES = {"U": "Unchanged", "C": "Changed"}

# Confidentiality / Integrity / Availability (CIA)
CIA_VALUES = {"H": 0.56, "L": 0.22, "N": 0.0}
CIA_NAMES = {"H": "High", "L": "Low", "N": "None"}


@dataclass
class CVSSMetrics:
    """Parsed CVSS v3.1 metrics."""
    av: str  # Attack Vector
    ac: str  # Attack Complexity
    pr: str  # Privileges Required
    ui: str  # User Interaction
    s: str   # Scope
    c: str   # Confidentiality
    i: str   # Integrity
    a: str   # Availability

    def to_dict(self) -> dict[str, str]:
        return {
            "AV": self.av,
            "AC": self.ac,
            "PR": self.pr,
            "UI": self.ui,
            "S": self.s,
            "C": self.c,
            "I": self.i,
            "A": self.a,
        }


# ---------- Vector parsing ----------

def parse_cvss_vector(vector: str) -> CVSSMetrics:
    """Parse a CVSS v3.1 vector string into CVSSMetrics.

    Accepts formats:
        "AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H"
        "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H"

    Raises ValueError on invalid format or unknown metric values.
    """
    if not vector or not isinstance(vector, str):
        raise ValueError("Vector must be a non-empty string")

    # Strip CVSS:3.1/ prefix if present
    v = vector.strip()
    if v.startswith("CVSS:3."):
        v = v.split("/", 1)[1] if "/" in v else v

    # Parse metric:value pairs
    pairs = {}
    for part in v.split("/"):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise ValueError(f"Invalid metric format: '{part}'. Expected 'METRIC:VALUE'")
        metric, value = part.split(":", 1)
        metric = metric.strip().upper()
        value = value.strip().upper()
        pairs[metric] = value

    # Validate required metrics
    required = ["AV", "AC", "PR", "UI", "S", "C", "I", "A"]
    missing = [m for m in required if m not in pairs]
    if missing:
        raise ValueError(f"Missing required metrics: {missing}")

    # Validate metric values
    validators = {
        "AV": AV_VALUES,
        "AC": AC_VALUES,
        "PR": PR_VALUES_UNCHANGED,  # keys same for both
        "UI": UI_VALUES,
        "S": S_VALUES,
        "C": CIA_VALUES,
        "I": CIA_VALUES,
        "A": CIA_VALUES,
    }
    for metric, valid_values in validators.items():
        if pairs[metric] not in valid_values:
            raise ValueError(
                f"Invalid value for {metric}: '{pairs[metric]}'. "
                f"Must be one of {list(valid_values.keys())}"
            )

    return CVSSMetrics(
        av=pairs["AV"],
        ac=pairs["AC"],
        pr=pairs["PR"],
        ui=pairs["UI"],
        s=pairs["S"],
        c=pairs["C"],
        i=pairs["I"],
        a=pairs["A"],
    )


# ---------- Base score calculation ----------

def _roundup(value: float) -> float:
    """CVSS v3.1 roundup function — round up to 1 decimal place.

    Per spec: roundup(x) = ceil(x * 10) / 10
    But with floating point fix: if x*10 is integer, return x, else ceil.
    """
    int_input = round(value * 100000)
    if int_input % 10000 == 0:
        return int_input / 100000.0
    else:
        return (int_input // 10000 + 1) / 10.0


def calculate_base_score(metrics: CVSSMetrics) -> float:
    """Calculate CVSS v3.1 base score (0.0-10.0).

    Uses the official FIRST.org formula.
    """
    # Confidentiality/Integrity/Availability impact values
    c_val = CIA_VALUES[metrics.c]
    i_val = CIA_VALUES[metrics.i]
    a_val = CIA_VALUES[metrics.a]

    # ISS (Impact Sub-Score)
    iss = 1 - ((1 - c_val) * (1 - i_val) * (1 - a_val))

    # Impact — depends on Scope
    if metrics.s == "U":
        impact = 6.42 * iss
    else:  # S == "C"
        impact = 7.52 * (iss - 0.029) - 3.25 * (iss - 0.02) ** 15

    # Exploitability
    av_val = AV_VALUES[metrics.av]
    ac_val = AC_VALUES[metrics.ac]
    pr_val = PR_VALUES_CHANGED[metrics.pr] if metrics.s == "C" else PR_VALUES_UNCHANGED[metrics.pr]
    ui_val = UI_VALUES[metrics.ui]
    exploitability = 8.22 * av_val * ac_val * pr_val * ui_val

    # Base score
    if impact <= 0:
        return 0.0

    if metrics.s == "U":
        base_score = _roundup(min(impact + exploitability, 10))
    else:  # S == "C"
        base_score = _roundup(min(1.08 * (impact + exploitability), 10))

    return base_score


# ---------- Severity ----------

def get_severity(score: float) -> str:
    """Get severity rating from base score.

    Returns: "none" / "low" / "medium" / "high" / "critical"
    """
    if score == 0.0:
        return "none"
    elif score <= 3.9:
        return "low"
    elif score <= 6.9:
        return "medium"
    elif score <= 8.9:
        return "high"
    else:  # 9.0-10.0
        return "critical"


# ---------- Convenience ----------

def calculate_cvss(vector: str) -> dict[str, Any]:
    """One-shot: parse vector → compute score + severity.

    Returns:
        {
            "vector": "AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H",
            "metrics": {AV: N, AC: L, ...},
            "base_score": 9.8,
            "severity": "critical",
        }
    """
    metrics = parse_cvss_vector(vector)
    score = calculate_base_score(metrics)
    return {
        "vector": vector,
        "metrics": metrics.to_dict(),
        "base_score": score,
        "severity": get_severity(score),
    }


if __name__ == "__main__":
    # Quick test with known vectors
    test_vectors = [
        ("AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H", 9.8, "critical"),   # Classic RCE
        ("AV:N/AC:L/PR:N/UI:N/S:U/C:N/I:N/A:H", 7.5, "high"),        # DoS
        ("AV:N/AC:H/PR:H/UI:N/S:U/C:L/I:L/A:L", 4.1, "medium"),      # Medium severity (verified NVD)
        ("AV:P/AC:L/PR:N/UI:N/S:U/C:N/I:N/A:N", 0.0, "none"),        # No impact
        ("AV:N/AC:L/PR:N/UI:N/S:C/C:H/I:H/A:H", 10.0, "critical"),   # Max score (scope changed)
    ]

    print("=== CVSS v3.1 Calculator Test ===\n")
    for vector, expected_score, expected_severity in test_vectors:
        result = calculate_cvss(vector)
        score_ok = abs(result["base_score"] - expected_score) < 0.05
        sev_ok = result["severity"] == expected_severity
        status = "OK" if score_ok and sev_ok else "FAIL"
        print(f"{status} {vector}")
        print(f"   score={result['base_score']} (expected {expected_score}) severity={result['severity']} (expected {expected_severity})")
