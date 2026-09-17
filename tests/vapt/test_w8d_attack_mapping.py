"""
W8-D tests — verify MITRE ATT&CK catalog YAML structure + completeness.

Tests:
1. YAML loads + parses without error
2. Catalog has expected top-level keys (version, source, last_updated, license, scope, tactics, techniques)
3. Version is "15.1" or higher (Enterprise ATT&CK v15.x)
4. At least 30 techniques present (target: 30+)
5. All 13 ATT&CK Enterprise tactics represented
6. Every technique has required fields (technique_id, name, tactic, description, detection, mitigation, related_wstg_ids, related_cwe, example_uses, applicable_to_web_mvp)
7. Technique IDs match format T<NNNN> or T<NNNN>.<NNN>
8. Technique IDs are unique
9. At least 70% of techniques marked applicable_to_web_mvp=True
10. All techniques have at least 1 CWE mapping (or empty list if network-only)
11. Critical techniques present (T1190 Exploit Public-Facing App, T1059 Command Execution, T1110 Brute Force, T1552 Unsecured Credentials)
12. Each technique has at least 1 example_use (practical pentest scenario)
13. Each technique has detection text (for report)
14. Each technique has mitigation text (for report remediation)
15. Technique IDs valid MITRE ATT&CK format
16. Tactic in technique matches one of 13 official ATT&CK tactics (with multi-tactic comma-separated allowed)
17. WSTG cross-references valid (every WSTG ID exists in wstg_catalog.yaml)
18. T1190 has correct related_wstg_ids (includes WSTG-INPV-05 SQLi)
19. T1059 has CWE-78 (Command Injection)
20. T1110 has CWE-307 (Rate Limit)
21. T1552 has CWE-798 (Hardcoded Credentials)

Run:
    pytest tests/vapt/test_w8d_attack_mapping.py -v
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ATTACK_CATALOG_PATH = PROJECT_ROOT / "data" / "attack_mapping.yaml"
WSTG_CATALOG_PATH = PROJECT_ROOT / "data" / "wstg_catalog.yaml"

# Expected 13 ATT&CK Enterprise tactics
EXPECTED_TACTIC_CODES = {
    "Reconnaissance", "Initial Access", "Execution", "Persistence",
    "Privilege Escalation", "Defense Evasion", "Credential Access",
    "Discovery", "Lateral Movement", "Collection", "Command and Control",
    "Exfiltration", "Impact",
}

# Technique ID format: T<NNNN> or T<NNNN>.<NNN>
TECHNIQUE_ID_REGEX = re.compile(r"^T\d{4}(\.\d{3})?$")

# Required fields in each technique entry
REQUIRED_FIELDS = {
    "technique_id", "name", "tactic", "description",
    "detection", "mitigation",
    "related_wstg_ids", "related_cwe", "example_uses",
    "applicable_to_web_mvp",
}

# Critical techniques that MUST be present
CRITICAL_TECHNIQUES = [
    "T1190",  # Exploit Public-Facing Application
    "T1059",  # Command and Scripting Interpreter
    "T1078",  # Valid Accounts
    "T1110",  # Brute Force
    "T1552",  # Unsecured Credentials
    "T1505",  # Server Software Component (Web Shell)
    "T1046",  # Network Service Discovery (SSRF mapping)
    "T1592",  # Gather Victim Host Information
]


# ---------- Fixtures ----------

@pytest.fixture(scope="module")
def catalog() -> dict:
    """Load ATT&CK catalog YAML once for all tests in this module."""
    assert ATTACK_CATALOG_PATH.exists(), f"ATT&CK catalog not found at {ATTACK_CATALOG_PATH}"
    with ATTACK_CATALOG_PATH.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="module")
def wstg_catalog() -> dict:
    """Load WSTG catalog to verify cross-references."""
    if not WSTG_CATALOG_PATH.exists():
        return None
    with WSTG_CATALOG_PATH.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------- Test 1: YAML loads ----------

def test_yaml_loads(catalog):
    """Catalog YAML parses without error."""
    assert catalog is not None
    assert isinstance(catalog, dict)


# ---------- Test 2: Top-level keys ----------

def test_top_level_keys(catalog):
    """Catalog has expected top-level keys."""
    expected_keys = {"version", "source", "last_updated", "license", "scope", "tactics", "techniques"}
    actual_keys = set(catalog.keys())
    missing = expected_keys - actual_keys
    assert not missing, f"Missing top-level keys: {missing}"


# ---------- Test 3: Version is 15.x ----------

def test_version(catalog):
    """Catalog version matches Enterprise ATT&CK v15.x or higher."""
    v = catalog["version"]
    assert v.startswith("15.") or v.startswith("16."), f"Expected version 15.x or 16.x, got {v}"


# ---------- Test 4: At least 30 techniques ----------

def test_at_least_30_techniques(catalog):
    """Catalog contains at least 30 techniques (target: 30+)."""
    techniques = catalog["techniques"]
    assert len(techniques) >= 30, f"Expected >= 30 techniques, got {len(techniques)}"


# ---------- Test 5: All 13 ATT&CK tactics represented ----------

def test_all_13_tactics_present(catalog):
    """All 13 ATT&CK Enterprise tactics are represented (techniques may span multiple tactics)."""
    # Build set of all tactic atoms across techniques (split by ", ")
    all_tactics = set()
    for t in catalog["techniques"]:
        for tactic in t["tactic"].split(", "):
            all_tactics.add(tactic.strip())

    missing = EXPECTED_TACTIC_CODES - all_tactics
    assert not missing, f"Missing tactics: {missing}"


# ---------- Test 6: All techniques have required fields ----------

def test_all_techniques_have_required_fields(catalog):
    """Every technique entry has all 10 required fields."""
    for i, t in enumerate(catalog["techniques"]):
        missing = REQUIRED_FIELDS - set(t.keys())
        assert not missing, f"Technique #{i} ({t.get('technique_id', '?')}) missing fields: {missing}"


# ---------- Test 7: Technique ID format ----------

def test_technique_id_format(catalog):
    """All technique IDs match MITRE ATT&CK format T<NNNN> or T<NNNN>.<NNN>."""
    bad = [t["technique_id"] for t in catalog["techniques"] if not TECHNIQUE_ID_REGEX.match(t["technique_id"])]
    assert not bad, f"Malformed technique IDs: {bad}"


# ---------- Test 8: Technique IDs unique ----------

def test_technique_ids_unique(catalog):
    """All technique IDs are unique (no duplicates)."""
    ids = [t["technique_id"] for t in catalog["techniques"]]
    duplicates = {x for x in ids if ids.count(x) > 1}
    assert not duplicates, f"Duplicate technique IDs: {duplicates}"


# ---------- Test 9: Web MVP coverage ----------

def test_web_mvp_coverage(catalog):
    """At least 70% of techniques marked applicable_to_web_mvp=True."""
    total = len(catalog["techniques"])
    web_mvp = sum(1 for t in catalog["techniques"] if t["applicable_to_web_mvp"])
    ratio = web_mvp / total
    assert ratio >= 0.70, f"Only {web_mvp}/{total} ({ratio:.0%}) marked web_mvp=True (need >=70%)"


# ---------- Test 10: CWE mappings (empty OK for network-only) ----------

def test_cwe_mappings_present(catalog):
    """related_cwe field is a list (can be empty for network-only techniques)."""
    for t in catalog["techniques"]:
        assert isinstance(t["related_cwe"], list), \
            f"{t['technique_id']}: related_cwe must be a list, got {type(t['related_cwe'])}"


# ---------- Test 11: Critical techniques present ----------

def test_critical_techniques_present(catalog):
    """Critical ATT&CK techniques are present in catalog."""
    all_ids = {t["technique_id"] for t in catalog["techniques"]}
    missing = [tid for tid in CRITICAL_TECHNIQUES if tid not in all_ids]
    assert not missing, f"Missing critical techniques: {missing}"


# ---------- Test 12: Each technique has >=1 example_use ----------

def test_each_technique_has_examples(catalog):
    """Every technique has at least 1 practical example_use."""
    no_examples = [
        t["technique_id"] for t in catalog["techniques"]
        if not t.get("example_uses")
    ]
    assert not no_examples, f"Techniques missing example_uses: {no_examples}"


# ---------- Test 13: Each technique has detection text ----------

def test_each_technique_has_detection(catalog):
    """Every technique has detection text (for report's Detection section)."""
    no_detect = [
        t["technique_id"] for t in catalog["techniques"]
        if not t.get("detection") or len(t["detection"]) < 20
    ]
    assert not no_detect, f"Techniques with weak/missing detection: {no_detect}"


# ---------- Test 14: Each technique has mitigation text ----------

def test_each_technique_has_mitigation(catalog):
    """Every technique has mitigation text (for report's Remediation section)."""
    no_mitig = [
        t["technique_id"] for t in catalog["techniques"]
        if not t.get("mitigation") or len(t["mitigation"]) < 20
    ]
    assert not no_mitig, f"Techniques with weak/missing mitigation: {no_mitig}"


# ---------- Test 15: WSTG cross-references valid ----------

def test_wstg_cross_references_valid(catalog, wstg_catalog):
    """Every related_wstg_ids reference exists in WSTG catalog (if WSTG catalog loaded)."""
    if wstg_catalog is None:
        pytest.skip("WSTG catalog not found — skipping cross-reference test")

    wstg_ids = {t["wstg_id"] for t in wstg_catalog["tests"]}

    bad_refs = []
    for tech in catalog["techniques"]:
        for wstg_ref in tech.get("related_wstg_ids", []):
            if wstg_ref not in wstg_ids:
                bad_refs.append((tech["technique_id"], wstg_ref))

    assert not bad_refs, f"Invalid WSTG references in techniques: {bad_refs[:5]}"


# ---------- Test 16: Tactic in technique matches ATT&CK official ----------

def test_tactic_matches_official(catalog):
    """Each technique's tactic (split by ', ') is in the 13 official ATT&CK tactics."""
    bad_tactics = []
    for t in catalog["techniques"]:
        for tactic in t["tactic"].split(", "):
            if tactic.strip() not in EXPECTED_TACTIC_CODES:
                bad_tactics.append((t["technique_id"], tactic))

    assert not bad_tactics, f"Invalid tactics: {bad_tactics}"


# ---------- Test 17: T1190 maps to SQLi ----------

def test_t1190_maps_to_sqli(catalog):
    """T1190 (Exploit Public-Facing App) references WSTG-INPV-05 (SQLi)."""
    t1190 = next(t for t in catalog["techniques"] if t["technique_id"] == "T1190")
    assert "WSTG-INPV-05" in t1190["related_wstg_ids"]
    assert "CWE-89" in t1190["related_cwe"]


# ---------- Test 18: T1059 has CWE-78 ----------

def test_t1059_has_cwe_78(catalog):
    """T1059 (Command and Scripting Interpreter) has CWE-78 (Command Injection)."""
    t1059 = next(t for t in catalog["techniques"] if t["technique_id"] == "T1059")
    assert "CWE-78" in t1059["related_cwe"]


# ---------- Test 19: T1110 has CWE-307 ----------

def test_t1110_has_cwe_307(catalog):
    """T1110 (Brute Force) has CWE-307 (Improper Restriction of Excessive Auth Attempts)."""
    t1110 = next(t for t in catalog["techniques"] if t["technique_id"] == "T1110")
    assert "CWE-307" in t1110["related_cwe"]


# ---------- Test 20: T1552 has CWE-798 ----------

def test_t1552_has_cwe_798(catalog):
    """T1552 (Unsecured Credentials) has CWE-798 (Use of Hardcoded Credentials)."""
    t1552 = next(t for t in catalog["techniques"] if t["technique_id"] == "T1552")
    assert "CWE-798" in t1552["related_cwe"]


# ---------- Test 21: T1046 maps to SSRF ----------

def test_t1046_maps_to_ssrf(catalog):
    """T1046 (Network Service Discovery) references WSTG-INPV-20 (SSRF)."""
    t1046 = next(t for t in catalog["techniques"] if t["technique_id"] == "T1046")
    assert "WSTG-INPV-20" in t1046["related_wstg_ids"]


# ---------- Test 22: Catalog stats summary ----------

def test_catalog_stats_summary(catalog):
    """Print catalog stats (informational test — always passes)."""
    total = len(catalog["techniques"])
    web_mvp = sum(1 for t in catalog["techniques"] if t["applicable_to_web_mvp"])
    by_tactic = {}
    for t in catalog["techniques"]:
        for tactic in t["tactic"].split(", "):
            by_tactic[tactic] = by_tactic.get(tactic, 0) + 1

    print(f"\n--- ATT&CK Catalog Stats ---")
    print(f"Total techniques: {total}")
    print(f"Web MVP applicable: {web_mvp} ({web_mvp/total:.0%})")
    print(f"Tactics represented: {len(by_tactic)}")
    for tactic in sorted(by_tactic.keys()):
        print(f"  - {tactic}: {by_tactic[tactic]}")
    print(f"---------------------------\n")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "--tb=short"])
