"""
W8-C tests — verify WSTG v4.2 catalog YAML structure + completeness.

Tests:
1. YAML loads + parses without error
2. Catalog has expected top-level keys (version, source, last_updated, license, categories, tests)
3. Version is "4.2" (matches OWASP WSTG v4.2)
4. At least 80 test IDs present (target: 80+, actual: 119)
5. All 12 OWASP WSTG categories present
6. Every test entry has required fields (wstg_id, name, category, description, related_mitre_attack, related_cwe, applicable_to_web_mvp)
7. WSTG IDs match format WSTG-<CAT>-<NN> (4-letter code + 2-digit number)
8. WSTG IDs are unique (no duplicates)
9. Categories in tests match categories in catalog header
10. At least 80% of tests marked applicable_to_web_mvp=True
11. All tests have at least 1 MITRE ATT&CK mapping
12. All tests have at least 1 CWE mapping
13. Specific known tests present (SQLi, XSS, CSRF, SSRF, IDOR, RCE)
14. Description length >= 50 chars for all tests

Run:
    pytest tests/vapt/test_w8c_wstg_catalog.py -v

This test does NOT require DB — just YAML parsing.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

# Path to catalog YAML — relative to project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
WSTG_CATALOG_PATH = PROJECT_ROOT / "data" / "wstg_catalog.yaml"

# Expected 12 OWASP WSTG v4.2 categories
EXPECTED_CATEGORIES = {
    "Information Gathering",
    "Configuration and Deployment Management",
    "Identity Management",
    "Authentication",
    "Authorization",
    "Session Management",
    "Input Validation",
    "Error Handling",
    "Weak Cryptography",
    "Business Logic",
    "Client-side",
    "APIs",
}

# Expected WSTG ID format: WSTG-XXXX-NN where XXXX is a 3-5 letter code, NN is 2+ digits
WSTG_ID_REGEX = re.compile(r"^WSTG-[A-Z]{3,5}-\d{2,}$")

# Required fields in each test entry
REQUIRED_FIELDS = {
    "wstg_id", "name", "category", "description",
    "related_mitre_attack", "related_cwe", "applicable_to_web_mvp",
}

# Specific tests that MUST be present (critical WAPT-AI MVP coverage)
CRITICAL_TESTS = [
    "WSTG-INPV-01",   # Reflected XSS
    "WSTG-INPV-02",   # Stored XSS
    "WSTG-INPV-05",   # SQL Injection
    "WSTG-INPV-13",   # Command Injection (RCE)
    "WSTG-INPV-19",   # SSTI
    "WSTG-INPV-20",   # SSRF
    "WSTG-SESS-05",   # CSRF
    "WSTG-ATHZ-04",   # IDOR
    "WSTG-ATHN-04",   # Auth bypass
    "WSTG-ATHZ-01",   # Path Traversal
    "WSTG-CONF-07",   # HSTS
    "WSTG-CRYP-01",   # Weak TLS
]


# ---------- Fixtures ----------

@pytest.fixture(scope="module")
def catalog() -> dict:
    """Load WSTG catalog YAML once for all tests in this module."""
    assert WSTG_CATALOG_PATH.exists(), f"WSTG catalog not found at {WSTG_CATALOG_PATH}"
    with WSTG_CATALOG_PATH.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------- Test 1: YAML loads without error ----------

def test_yaml_loads(catalog):
    """Catalog YAML parses without error."""
    assert catalog is not None
    assert isinstance(catalog, dict)


# ---------- Test 2: Top-level keys ----------

def test_top_level_keys(catalog):
    """Catalog has expected top-level keys."""
    expected_keys = {"version", "source", "last_updated", "license", "categories", "tests"}
    actual_keys = set(catalog.keys())
    missing = expected_keys - actual_keys
    assert not missing, f"Missing top-level keys: {missing}"


# ---------- Test 3: Version is 4.2 ----------

def test_version_is_4_2(catalog):
    """Catalog version matches OWASP WSTG v4.2."""
    assert catalog["version"] == "4.2", f"Expected version '4.2', got '{catalog['version']}'"


# ---------- Test 4: At least 80 test IDs ----------

def test_at_least_80_tests(catalog):
    """Catalog contains at least 80 test IDs (target: 80+, actual: 119)."""
    tests = catalog["tests"]
    assert len(tests) >= 80, f"Expected >= 80 tests, got {len(tests)}"


# ---------- Test 5: All 12 categories present ----------

def test_all_12_categories_present(catalog):
    """All 12 OWASP WSTG v4.2 categories are represented."""
    actual_categories = {t["category"] for t in catalog["tests"]}
    missing = EXPECTED_CATEGORIES - actual_categories
    assert not missing, f"Missing categories: {missing}"


# ---------- Test 6: Every test has required fields ----------

def test_all_tests_have_required_fields(catalog):
    """Every test entry has all 7 required fields."""
    for i, t in enumerate(catalog["tests"]):
        missing = REQUIRED_FIELDS - set(t.keys())
        assert not missing, f"Test #{i} ({t.get('wstg_id', '?')}) missing fields: {missing}"


# ---------- Test 7: WSTG ID format ----------

def test_wstg_id_format(catalog):
    """All WSTG IDs match format WSTG-XXXX-NN."""
    bad = [t["wstg_id"] for t in catalog["tests"] if not WSTG_ID_REGEX.match(t["wstg_id"])]
    assert not bad, f"Malformed WSTG IDs: {bad[:5]}"


# ---------- Test 8: WSTG IDs unique ----------

def test_wstg_ids_unique(catalog):
    """All WSTG IDs are unique (no duplicates)."""
    ids = [t["wstg_id"] for t in catalog["tests"]]
    duplicates = {x for x in ids if ids.count(x) > 1}
    assert not duplicates, f"Duplicate WSTG IDs: {duplicates}"


# ---------- Test 9: Categories in tests match catalog header ----------

def test_categories_match_header(catalog):
    """Every test's category appears in the catalog's categories list."""
    header_categories = {c["name"] for c in catalog["categories"]}
    test_categories = {t["category"] for t in catalog["tests"]}
    orphaned = test_categories - header_categories
    assert not orphaned, f"Tests reference categories not in header: {orphaned}"


# ---------- Test 10: >=80% applicable to web MVP ----------

def test_web_mvp_coverage(catalog):
    """At least 80% of tests are marked applicable_to_web_mvp=True."""
    total = len(catalog["tests"])
    web_mvp = sum(1 for t in catalog["tests"] if t["applicable_to_web_mvp"])
    ratio = web_mvp / total
    assert ratio >= 0.80, f"Only {web_mvp}/{total} ({ratio:.0%}) marked web_mvp=True (need >=80%)"


# ---------- Test 11: All tests have >=1 MITRE ATT&CK mapping ----------

def test_all_tests_have_mitre_mapping(catalog):
    """Every test has at least 1 MITRE ATT&CK technique ID."""
    no_mitre = [
        t["wstg_id"] for t in catalog["tests"]
        if not t.get("related_mitre_attack")
    ]
    assert not no_mitre, f"Tests missing MITRE mapping: {no_mitre}"


# ---------- Test 12: All tests have >=1 CWE mapping ----------

def test_all_tests_have_cwe_mapping(catalog):
    """Every test has at least 1 CWE ID."""
    no_cwe = [
        t["wstg_id"] for t in catalog["tests"]
        if not t.get("related_cwe")
    ]
    assert not no_cwe, f"Tests missing CWE mapping: {no_cwe}"


# ---------- Test 13: Critical tests present ----------

def test_critical_tests_present(catalog):
    """Critical WAPT-AI MVP tests are present in catalog."""
    all_ids = {t["wstg_id"] for t in catalog["tests"]}
    missing = [tid for tid in CRITICAL_TESTS if tid not in all_ids]
    assert not missing, f"Missing critical tests: {missing}"


# ---------- Test 14: Description length >=50 chars ----------

def test_description_min_length(catalog):
    """All test descriptions are at least 50 characters."""
    short = [
        (t["wstg_id"], len(t["description"]))
        for t in catalog["tests"]
        if len(t["description"]) < 50
    ]
    assert not short, f"Descriptions too short: {short[:5]}"


# ---------- Test 15: MITRE technique IDs valid format ----------

def test_mitre_id_format(catalog):
    """All MITRE ATT&CK IDs match format T<NNNN> or T<NNNN>.<NNN>."""
    mitre_regex = re.compile(r"^T\d{4}(\.\d{3})?$")
    bad = []
    for t in catalog["tests"]:
        for m in t["related_mitre_attack"]:
            if not mitre_regex.match(m):
                bad.append((t["wstg_id"], m))
    assert not bad, f"Malformed MITRE IDs: {bad[:5]}"


# ---------- Test 16: CWE IDs valid format ----------

def test_cwe_id_format(catalog):
    """All CWE IDs match format CWE-<NNN>."""
    cwe_regex = re.compile(r"^CWE-\d+$")
    bad = []
    for t in catalog["tests"]:
        for c in t["related_cwe"]:
            if not cwe_regex.match(c):
                bad.append((t["wstg_id"], c))
    assert not bad, f"Malformed CWE IDs: {bad[:5]}"


# ---------- Test 17: SQLi test has correct mappings ----------

def test_sqli_mappings(catalog):
    """WSTG-INPV-05 (SQL Injection) has CWE-89 and T1190/T1078 mappings."""
    sqli = next(t for t in catalog["tests"] if t["wstg_id"] == "WSTG-INPV-05")
    assert "CWE-89" in sqli["related_cwe"]
    assert "T1190" in sqli["related_mitre_attack"]


# ---------- Test 18: XSS test has correct mappings ----------

def test_xss_mappings(catalog):
    """WSTG-INPV-01 (Reflected XSS) has CWE-79 mapping."""
    xss = next(t for t in catalog["tests"] if t["wstg_id"] == "WSTG-INPV-01")
    assert "CWE-79" in xss["related_cwe"]


# ---------- Test 19: SSRF test has correct mappings ----------

def test_ssrf_mappings(catalog):
    """WSTG-INPV-20 (SSRF) has CWE-918 mapping."""
    ssrf = next(t for t in catalog["tests"] if t["wstg_id"] == "WSTG-INPV-20")
    assert "CWE-918" in ssrf["related_cwe"]


# ---------- Test 20: Command injection has CWE-78 ----------

def test_cmd_injection_mappings(catalog):
    """WSTG-INPV-13 (Command Injection) has CWE-78 mapping."""
    cmdi = next(t for t in catalog["tests"] if t["wstg_id"] == "WSTG-INPV-13")
    assert "CWE-78" in cmdi["related_cwe"]


# ---------- Test 21: Category counts ----------

def test_category_counts(catalog):
    """Each category has at least 3 tests (sanity check on catalog breadth)."""
    by_cat = {}
    for t in catalog["tests"]:
        by_cat.setdefault(t["category"], 0)
        by_cat[t["category"]] += 1

    small_cats = {cat: n for cat, n in by_cat.items() if n < 3}
    assert not small_cats, f"Categories with <3 tests: {small_cats}"


# ---------- Test 22: Catalog stats summary (informational) ----------

def test_catalog_stats_summary(catalog):
    """Print catalog stats (informational test — always passes)."""
    total = len(catalog["tests"])
    web_mvp = sum(1 for t in catalog["tests"] if t["applicable_to_web_mvp"])
    by_cat = {}
    for t in catalog["tests"]:
        by_cat.setdefault(t["category"], 0)
        by_cat[t["category"]] += 1

    print(f"\n--- WSTG Catalog Stats ---")
    print(f"Total tests: {total}")
    print(f"Web MVP applicable: {web_mvp} ({web_mvp/total:.0%})")
    print(f"Categories: {len(by_cat)}")
    for cat in sorted(by_cat.keys()):
        print(f"  - {cat}: {by_cat[cat]}")
    print(f"-------------------------\n")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "--tb=short"])
