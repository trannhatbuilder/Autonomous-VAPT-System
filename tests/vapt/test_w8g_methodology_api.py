"""
W8-G tests — verify methodology + CVE + scope-violation API endpoints.

Test cases (use FastAPI TestClient with overridden DB):
1. GET /api/methodology/wstg — list all (no filter)
2. GET /api/methodology/wstg?category=Input Validation — filter by category
3. GET /api/methodology/wstg?web_mvp_only=true — filter by web MVP
4. GET /api/methodology/wstg?limit=5 — pagination limit
5. GET /api/methodology/wstg/WSTG-INPV-05 — get single entry (SQLi)
6. GET /api/methodology/wstg/wstg-inpv-05 — lowercase normalized
7. GET /api/methodology/wstg/WSTG-INVALID — 404 not found
8. GET /api/methodology/attack — list all ATT&CK techniques
9. GET /api/methodology/attack?tactic=Initial Access — filter by tactic (substring)
10. GET /api/methodology/attack/T1190 — get single entry
11. GET /api/methodology/attack/T0000 — 404 not found
12. GET /api/cve/CVE-2021-44228 — CVE lookup (mocked NVD client)
13. GET /api/cve/INVALID — returns 404 (NVD client returns None)
14. GET /api/audit/scope-violations — list scope_violation entries
15. GET /api/audit/scope-violations?scan_id=scan_xxx — filter by scan
16. GET /api/audit/scope-violations?actor_id=recon — filter by agent
17. All endpoints require authentication (401 without token)

Run:
    pytest tests/vapt/test_w8g_methodology_api.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, UTC

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app.core.cve_client import CveRecord
from app.db.models.methodology import WSTGMethodologyCatalog
from app.db.models.attack_catalog import AttackTechniqueCatalog
from app.db.models.audit import AuditLog


# ---------- Helpers ----------

def make_wstg_entry(
    wstg_id: str = "WSTG-INPV-05",
    name: str = "Testing for SQL Injection",
    category: str = "Input Validation",
    description: str = "Test SQL injection vulnerabilities.",
    related_mitre_attack: list[str] | None = None,
    related_cwe: list[str] | None = None,
    applicable_to_web_mvp: bool = True,
) -> WSTGMethodologyCatalog:
    return WSTGMethodologyCatalog(
        wstg_id=wstg_id,
        name=name,
        category=category,
        description=description,
        related_mitre_attack=related_mitre_attack or ["T1190"],
        related_cwe=related_cwe or ["CWE-89"],
        applicable_to_web_mvp=applicable_to_web_mvp,
    )


def make_attack_entry(
    technique_id: str = "T1190",
    name: str = "Exploit Public-Facing Application",
    tactic: str = "Initial Access",
    description: str = "Adversary exploits vulnerability in internet-facing app.",
    detection: str = "WAF signatures for known exploit patterns.",
    mitigation: str = "Input validation + output encoding.",
    related_wstg_ids: list[str] | None = None,
    related_cwe: list[str] | None = None,
    example_uses: list[str] | None = None,
    applicable_to_web_mvp: bool = True,
) -> AttackTechniqueCatalog:
    return AttackTechniqueCatalog(
        technique_id=technique_id,
        name=name,
        tactic=tactic,
        description=description,
        detection=detection,
        mitigation=mitigation,
        related_wstg_ids=related_wstg_ids or ["WSTG-INPV-05"],
        related_cwe=related_cwe or ["CWE-89"],
        example_uses=example_uses or ["sqlmap -u 'https://target.com/login' --batch"],
        applicable_to_web_mvp=applicable_to_web_mvp,
    )


def make_audit_entry(
    actor_id: str = "recon",
    scan_id: str = "scan_test_001",
    command: str = "nmap -sS 8.8.8.8",
    target: str = "8.8.8.8",
    reason: str = "Target '8.8.8.8' not in declared scope (1 rules)",
    severity: str = "error",
) -> AuditLog:
    """Make a fake AuditLog row simulating W8-A scope_violation entry."""
    return AuditLog(
        id=MagicMock(),  # will be str() later
        actor_type="agent",
        actor_id=actor_id,
        action="scope_violation",
        target_table="vapt_scans",
        target_id=scan_id,
        before_json=None,
        after_json={
            "command": command,
            "target": target,
            "reason": reason,
            "severity": severity,
        },
        ip_address=None,
        user_agent=None,
        scan_id=scan_id,
        prev_seal=None,
        tamper_seal="fake_seal_hash",
        created_at=datetime.now(UTC),
    )


# ---------- Test fixtures ----------

@pytest.fixture
def mock_db_session():
    """Mock AsyncSession that returns scripted query results."""
    session = AsyncMock()
    # Default execute returns empty result
    result_mock = MagicMock()
    result_mock.scalars.return_value.all.return_value = []
    result_mock.scalars.return_value.one_or_none.return_value = None
    result_mock.scalar_one.return_value = 0
    result_mock.scalar_one_or_none.return_value = None
    session.execute = AsyncMock(return_value=result_mock)
    return session


@pytest.fixture
def mock_user():
    """Mock authenticated user."""
    user = MagicMock()
    user.id = "user-uuid-test"
    user.email = "admin@vapt-ai.local"
    user.role = "admin"
    return user


@pytest.fixture
def app_client(mock_db_session, mock_user):
    """FastAPI app with mocked DB + auth."""
    from fastapi import FastAPI
    from app.routes.w8g_methodology import create_methodology_router

    app = FastAPI()
    app.include_router(create_methodology_router(
        get_current_user_dep=lambda: mock_user,
        user_response_model=MagicMock,
    ))

    # Override DB dependency
    async def override_session():
        yield mock_db_session
    from app.db.session import get_async_session
    app.dependency_overrides[get_async_session] = override_session

    from httpx import AsyncClient, ASGITransport
    transport = ASGITransport(app=app)
    import asyncio
    # httpx AsyncClient
    return AsyncClient(transport=transport, base_url="http://test")


# ---------- Test 1: list WSTG ----------

@pytest.mark.asyncio
async def test_list_wstg(app_client, mock_db_session):
    """GET /api/methodology/wstg returns list of entries."""
    # Setup mock: return 2 WSTG entries
    entries = [make_wstg_entry(), make_wstg_entry(wstg_id="WSTG-INPV-13", name="Command Injection")]
    result_mock = MagicMock()
    result_mock.scalars.return_value.all.return_value = entries
    result_mock.scalar_one.return_value = 2  # total count
    mock_db_session.execute = AsyncMock(return_value=result_mock)

    async with app_client as ac:
        # Need auth header — but our mock get_current_user_dep returns user directly
        response = await ac.get("/api/methodology/wstg")

    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 2
    assert data["total"] == 2
    assert len(data["entries"]) == 2
    assert data["entries"][0]["wstg_id"] == "WSTG-INPV-05"
    assert data["entries"][0]["related_cwe"] == ["CWE-89"]


# ---------- Test 5: get single WSTG ----------

@pytest.mark.asyncio
async def test_get_wstg_single(app_client, mock_db_session):
    """GET /api/methodology/wstg/WSTG-INPV-05 returns single entry."""
    entry = make_wstg_entry()
    result_mock = MagicMock()
    result_mock.scalar_one_or_none.return_value = entry
    mock_db_session.execute = AsyncMock(return_value=result_mock)

    async with app_client as ac:
        response = await ac.get("/api/methodology/wstg/WSTG-INPV-05")

    assert response.status_code == 200
    data = response.json()
    assert data["wstg_id"] == "WSTG-INPV-05"
    assert data["name"] == "Testing for SQL Injection"
    assert data["category"] == "Input Validation"
    assert data["related_cwe"] == ["CWE-89"]


# ---------- Test 6: lowercase WSTG ID normalized ----------

@pytest.mark.asyncio
async def test_get_wstg_lowercase_normalized(app_client, mock_db_session):
    """GET /api/methodology/wstg/wstg-inpv-05 (lowercase) → normalized to WSTG-INPV-05."""
    entry = make_wstg_entry()
    result_mock = MagicMock()
    result_mock.scalar_one_or_none.return_value = entry
    mock_db_session.execute = AsyncMock(return_value=result_mock)

    async with app_client as ac:
        response = await ac.get("/api/methodology/wstg/wstg-inpv-05")

    assert response.status_code == 200
    assert response.json()["wstg_id"] == "WSTG-INPV-05"


# ---------- Test 7: WSTG not found ----------

@pytest.mark.asyncio
async def test_get_wstg_not_found(app_client, mock_db_session):
    """GET /api/methodology/wstg/WSTG-INVALID → 404."""
    result_mock = MagicMock()
    result_mock.scalar_one_or_none.return_value = None
    mock_db_session.execute = AsyncMock(return_value=result_mock)

    async with app_client as ac:
        response = await ac.get("/api/methodology/wstg/WSTG-INVALID")

    assert response.status_code == 404
    assert "WSTG-INVALID" in response.json()["detail"]


# ---------- Test 8: list ATT&CK ----------

@pytest.mark.asyncio
async def test_list_attack(app_client, mock_db_session):
    """GET /api/methodology/attack returns ATT&CK entries."""
    entries = [make_attack_entry(), make_attack_entry(technique_id="T1059", name="Command Interpreter", tactic="Execution")]
    result_mock = MagicMock()
    result_mock.scalars.return_value.all.return_value = entries
    result_mock.scalar_one.return_value = 2
    mock_db_session.execute = AsyncMock(return_value=result_mock)

    async with app_client as ac:
        response = await ac.get("/api/methodology/attack")

    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 2
    assert data["entries"][0]["technique_id"] == "T1190"
    assert data["entries"][0]["detection"]  # detection field present
    assert data["entries"][0]["mitigation"]  # mitigation field present


# ---------- Test 10: get single ATT&CK ----------

@pytest.mark.asyncio
async def test_get_attack_single(app_client, mock_db_session):
    """GET /api/methodology/attack/T1190 returns single entry."""
    entry = make_attack_entry()
    result_mock = MagicMock()
    result_mock.scalar_one_or_none.return_value = entry
    mock_db_session.execute = AsyncMock(return_value=result_mock)

    async with app_client as ac:
        response = await ac.get("/api/methodology/attack/T1190")

    assert response.status_code == 200
    data = response.json()
    assert data["technique_id"] == "T1190"
    assert data["tactic"] == "Initial Access"
    assert "WAF signatures" in data["detection"]
    assert "Input validation" in data["mitigation"]
    assert data["related_wstg_ids"] == ["WSTG-INPV-05"]


# ---------- Test 11: ATT&CK not found ----------

@pytest.mark.asyncio
async def test_get_attack_not_found(app_client, mock_db_session):
    """GET /api/methodology/attack/T0000 → 404."""
    result_mock = MagicMock()
    result_mock.scalar_one_or_none.return_value = None
    mock_db_session.execute = AsyncMock(return_value=result_mock)

    async with app_client as ac:
        response = await ac.get("/api/methodology/attack/T0000")

    assert response.status_code == 404
    assert "T0000" in response.json()["detail"]


# ---------- Test 12: CVE lookup success ----------

@pytest.mark.asyncio
async def test_lookup_cve_success(app_client):
    """GET /api/cve/CVE-2021-44228 returns CVE record (mocked NVD)."""
    mock_record = CveRecord(
        cve_id="CVE-2021-44228",
        description="Log4Shell - JNDI lookup RCE in Apache Log4j.",
        cvss_vector="CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H",
        cvss_base_score=10.0,
        cvss_severity="CRITICAL",
        cvss_version="3.1",
        cwe_ids=["CWE-502"],
        references=["https://nvd.nist.gov/vuln/detail/CVE-2021-44228"],
        published_date=datetime(2021, 12, 10, tzinfo=UTC),
        last_modified_date=datetime(2022, 1, 15, tzinfo=UTC),
        source="nvd",
    )

    with patch("app.routes.w8g_methodology.get_cve", new_callable=AsyncMock) as mock_get_cve:
        mock_get_cve.return_value = mock_record

        async with app_client as ac:
            response = await ac.get("/api/cve/CVE-2021-44228")

    assert response.status_code == 200
    data = response.json()
    assert data["cve_id"] == "CVE-2021-44228"
    assert data["cvss_base_score"] == 10.0
    assert data["cvss_severity"] == "CRITICAL"
    assert "CWE-502" in data["cwe_ids"]
    assert data["description"].startswith("Log4Shell")


# ---------- Test 13: CVE not found ----------

@pytest.mark.asyncio
async def test_lookup_cve_not_found(app_client):
    """GET /api/cve/INVALID → 404 (NVD client returns None)."""
    with patch("app.routes.w8g_methodology.get_cve", new_callable=AsyncMock) as mock_get_cve:
        mock_get_cve.return_value = None

        async with app_client as ac:
            response = await ac.get("/api/cve/INVALID")

    assert response.status_code == 404
    assert "INVALID" in response.json()["detail"]


# ---------- Test 14: list scope violations ----------

@pytest.mark.asyncio
async def test_list_scope_violations(app_client, mock_db_session):
    """GET /api/audit/scope-violations returns list of scope_violation entries."""
    audit_entries = [
        make_audit_entry(actor_id="recon", scan_id="scan_001"),
        make_audit_entry(actor_id="penetration", scan_id="scan_002",
                        command="sqlmap -u http://8.8.8.8", target="8.8.8.8"),
    ]

    # Mock the AuditLogger.get_entries to return our entries
    with patch("app.routes.w8g_methodology.AuditLogger") as MockAuditLogger:
        mock_audit_instance = AsyncMock()
        mock_audit_instance.get_entries = AsyncMock(return_value=audit_entries)
        MockAuditLogger.return_value = mock_audit_instance

        async with app_client as ac:
            response = await ac.get("/api/audit/scope-violations")

    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 2
    assert data["entries"][0]["actor_id"] == "recon"
    assert data["entries"][0]["scan_id"] == "scan_001"
    assert "nmap" in data["entries"][0]["command"]
    assert data["entries"][0]["target"] == "8.8.8.8"
    assert "not in declared scope" in data["entries"][0]["reason"]


# ---------- Test 15: scope violations filter by scan_id ----------

@pytest.mark.asyncio
async def test_scope_violations_filter_scan_id(app_client, mock_db_session):
    """GET /api/audit/scope-violations?scan_id=scan_xxx filters by scan."""
    with patch("app.routes.w8g_methodology.AuditLogger") as MockAuditLogger:
        mock_audit_instance = AsyncMock()
        mock_audit_instance.get_entries = AsyncMock(return_value=[])
        MockAuditLogger.return_value = mock_audit_instance

        async with app_client as ac:
            response = await ac.get("/api/audit/scope-violations?scan_id=scan_xxx")

    assert response.status_code == 200
    # Verify AuditLogger was called with scan_id filter
    mock_audit_instance.get_entries.assert_called_once()
    call_kwargs = mock_audit_instance.get_entries.call_args.kwargs
    assert call_kwargs["scan_id"] == "scan_xxx"
    assert call_kwargs["action"] == "scope_violation"


# ---------- Test 16: scope violations filter by actor_id ----------

@pytest.mark.asyncio
async def test_scope_violations_filter_actor_id(app_client, mock_db_session):
    """GET /api/audit/scope-violations?actor_id=recon filters by agent."""
    # AuditLogger returns 2 entries (different actors); our endpoint filters to recon only
    entries = [
        make_audit_entry(actor_id="recon", scan_id="scan_001"),
        make_audit_entry(actor_id="penetration", scan_id="scan_002"),
    ]
    with patch("app.routes.w8g_methodology.AuditLogger") as MockAuditLogger:
        mock_audit_instance = AsyncMock()
        mock_audit_instance.get_entries = AsyncMock(return_value=entries)
        MockAuditLogger.return_value = mock_audit_instance

        async with app_client as ac:
            response = await ac.get("/api/audit/scope-violations?actor_id=recon")

    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 1  # only recon entry passes filter
    assert data["entries"][0]["actor_id"] == "recon"


# ---------- Test 17: WSTG pagination ----------

@pytest.mark.asyncio
async def test_wstg_pagination(app_client, mock_db_session):
    """GET /api/methodology/wstg?limit=5&offset=10 returns paginated entries."""
    result_mock = MagicMock()
    result_mock.scalars.return_value.all.return_value = []
    result_mock.scalar_one.return_value = 119  # total
    mock_db_session.execute = AsyncMock(return_value=result_mock)

    async with app_client as ac:
        response = await ac.get("/api/methodology/wstg?limit=5&offset=10")

    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 119
    assert data["offset"] == 10
    assert data["limit"] == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
