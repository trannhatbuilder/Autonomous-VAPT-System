"""
W8-B tests — verify NVD REST API v2 client behavior.

Test cases (all use mock httpx — no actual network calls):
1.  get_cve() with valid CVE ID → fetches from NVD, caches, returns CveRecord
2.  get_cve() with invalid format (e.g. "CVE-24-1") → returns None, no HTTP call
3.  get_cve() cache hit — second call within TTL → no HTTP call (returns cached)
4.  get_cve() cache miss — after TTL expiry → re-fetches from NVD
5.  get_cve() network error (httpx.ConnectError) → returns None, no exception
6.  get_cve() HTTP 403 (rate limited) → returns None, no exception
7.  get_cve() HTTP 404 (not found) → returns None
8.  get_cve() HTTP 500 → returns None, no exception
9.  get_cve() malformed JSON response → returns None, no exception
10. get_cve() CVE not in vulnerabilities array → returns None
11. get_cve() lower-case input "cve-2024-1234" → normalized to "CVE-2024-1234"
12. Rate limit — 6th request within 30s without API key → returns None (no HTTP)
13. Rate limit — 51st request within 30s with API key → returns None (no HTTP)
14. Disk cache — record persisted to data/cve_cache/CVE-*.json
15. Disk cache — expired entry (>24h) is re-fetched
16. API key header — when api_key is set, header "apiKey" is sent
17. close() — closes httpx client gracefully
18. Disabled client (enabled=False) → returns None, no HTTP call

Run:
    pytest tests/vapt/test_w8b_cve_client.py -v
"""
from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime, UTC
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

# Ensure project root on sys.path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app.core.cve_client import (
    NvdClient,
    CveRecord,
    NVD_API_BASE,
    RATE_LIMIT_MAX_REQUESTS_NO_KEY,
    RATE_LIMIT_MAX_REQUESTS_WITH_KEY,
    RATE_LIMIT_WINDOW_SECONDS,
)


# ---------- Mock NVD response ----------

MOCK_NVD_RESPONSE = {
    "resultsPerPage": 1,
    "startIndex": 0,
    "totalResults": 1,
    "format": "NVD_CVE",
    "version": "2.0",
    "vulnerabilities": [
        {
            "cve": {
                "id": "CVE-2024-1234",
                "sourceIdentifier": "nvd@nist.gov",
                "published": "2024-01-15T00:00:00.000",
                "lastModified": "2024-02-01T00:00:00.000",
                "descriptions": [
                    {"lang": "en", "value": "SQL injection vulnerability in example app."},
                    {"lang": "es", "value": "Vulnerabilidad de inyección SQL."},
                ],
                "metrics": {
                    "cvssMetricV31": [
                        {
                            "cvssData": {
                                "version": "3.1",
                                "vectorString": "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H",
                                "baseScore": 9.8,
                                "baseSeverity": "CRITICAL",
                            },
                            "source": "nvd@nist.gov",
                            "type": "Primary",
                        }
                    ]
                },
                "weaknesses": [
                    {
                        "source": "nvd@nist.gov",
                        "type": "Primary",
                        "description": [
                            {"lang": "en", "value": "CWE-89"}
                        ]
                    },
                    {
                        "source": "nvd@nist.gov",
                        "type": "Secondary",
                        "description": [
                            {"lang": "en", "value": "CWE-89"}
                        ]
                    }
                ],
                "references": [
                    {"url": "https://example.com/advisory", "source": "vendor"},
                    {"url": "https://nvd.nist.gov/vuln/detail/CVE-2024-1234", "source": "nvd"},
                ],
            }
        }
    ],
}

MOCK_NVD_RESPONSE_EMPTY = {
    "resultsPerPage": 0,
    "startIndex": 0,
    "totalResults": 0,
    "format": "NVD_CVE",
    "version": "2.0",
    "vulnerabilities": [],
}


# ---------- Fixtures ----------

@pytest.fixture
def cache_dir(tmp_path) -> Path:
    """Isolated cache dir per test."""
    d = tmp_path / "cve_cache"
    d.mkdir()
    return d


@pytest.fixture
def mock_httpx_response_ok() -> httpx.Response:
    """Mock 200 OK response from NVD."""
    return httpx.Response(
        status_code=200,
        json=MOCK_NVD_RESPONSE,
        request=httpx.Request("GET", NVD_API_BASE),
    )


@pytest.fixture
def mock_httpx_response_empty() -> httpx.Response:
    """Mock 200 OK with empty vulnerabilities (CVE not found)."""
    return httpx.Response(
        status_code=200,
        json=MOCK_NVD_RESPONSE_EMPTY,
        request=httpx.Request("GET", NVD_API_BASE),
    )


@pytest.fixture
def mock_httpx_response_403() -> httpx.Response:
    """Mock 403 rate-limited response."""
    return httpx.Response(
        status_code=403,
        text="Rate limit exceeded",
        request=httpx.Request("GET", NVD_API_BASE),
    )


@pytest.fixture
def mock_httpx_response_404() -> httpx.Response:
    """Mock 404 not found response."""
    return httpx.Response(
        status_code=404,
        text="Not Found",
        request=httpx.Request("GET", NVD_API_BASE),
    )


@pytest.fixture
def mock_httpx_response_500() -> httpx.Response:
    """Mock 500 server error response."""
    return httpx.Response(
        status_code=500,
        text="Internal Server Error",
        request=httpx.Request("GET", NVD_API_BASE),
    )


@pytest.fixture
def mock_httpx_response_malformed() -> httpx.Response:
    """Mock malformed JSON response."""
    return httpx.Response(
        status_code=200,
        text='{"vulnerabilities": [invalid json',
        request=httpx.Request("GET", NVD_API_BASE),
    )


@pytest.fixture
def client(cache_dir) -> NvdClient:
    """NvdClient with isolated cache_dir + no API key."""
    return NvdClient(
        cache_dir=cache_dir,
        cache_ttl_hours=24,
        api_key=None,
        enabled=True,
    )


@pytest.fixture
def client_with_key(cache_dir) -> NvdClient:
    """NvdClient with API key set (50 req/30s rate limit)."""
    return NvdClient(
        cache_dir=cache_dir,
        cache_ttl_hours=24,
        api_key="test-api-key-12345",
        enabled=True,
    )


@pytest.fixture
def client_disabled(cache_dir) -> NvdClient:
    """NvdClient with enabled=False."""
    return NvdClient(
        cache_dir=cache_dir,
        enabled=False,
    )


# ---------- Tests ----------

# Test 1: valid CVE → fetches + caches + returns record
@pytest.mark.asyncio
async def test_get_cve_valid(client, mock_httpx_response_ok):
    """Valid CVE ID → fetches from NVD, returns CveRecord with parsed fields."""
    with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = mock_httpx_response_ok

        record = await client.get_cve("CVE-2024-1234")

    assert record is not None
    assert record.cve_id == "CVE-2024-1234"
    assert "SQL injection vulnerability" in record.description
    assert record.cvss_vector == "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H"
    assert record.cvss_base_score == 9.8
    assert record.cvss_severity == "CRITICAL"
    assert record.cvss_version == "3.1"
    assert record.cwe_ids == ["CWE-89"]  # deduped
    assert len(record.references) == 2
    assert record.published_date is not None
    assert record.last_modified_date is not None
    assert record.source == "nvd"


# Test 2: invalid CVE format → returns None, no HTTP call
@pytest.mark.asyncio
async def test_invalid_cve_format(client):
    """Invalid format (no 4-digit year) → None, no HTTP call."""
    with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
        record = await client.get_cve("CVE-24-1")  # too few digits

    assert record is None
    mock_get.assert_not_called()


# Test 3: cache hit — second call within TTL → no HTTP call
@pytest.mark.asyncio
async def test_cache_hit_no_http(client, mock_httpx_response_ok):
    """Second call within TTL returns cached record, no HTTP call."""
    with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = mock_httpx_response_ok

        # First call — fetches from NVD
        record1 = await client.get_cve("CVE-2024-1234")
        # Second call — should be cache hit
        record2 = await client.get_cve("CVE-2024-1234")

    assert record1 is not None
    assert record2 is not None
    assert record2.cve_id == record1.cve_id
    # Only ONE HTTP call should have been made
    assert mock_get.call_count == 1


# Test 4: cache miss after TTL expiry → re-fetches
@pytest.mark.asyncio
async def test_cache_expiry_refetches(client, mock_httpx_response_ok, cache_dir):
    """After TTL expires, record is re-fetched from NVD."""
    # Manually write an expired cache file
    cache_file = cache_dir / "CVE-2024-1234.json"
    cache_payload = {
        "_cached_at": time.time() - (25 * 3600),  # 25h ago (expired)
        "_cve_id": "CVE-2024-1234",
        "record": CveRecord(cve_id="CVE-2024-1234", description="stale").to_dict(),
    }
    cache_file.write_text(json.dumps(cache_payload))

    with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = mock_httpx_response_ok

        record = await client.get_cve("CVE-2024-1234")

    assert record is not None
    # Description should be the fresh one (from NVD), not the stale cache
    assert "SQL injection vulnerability" in record.description
    # HTTP call WAS made
    assert mock_get.call_count == 1


# Test 5: network error → returns None, no exception
@pytest.mark.asyncio
async def test_network_error_returns_none(client):
    """httpx.ConnectError → returns None, no exception raised."""
    with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
        mock_get.side_effect = httpx.ConnectError("Connection refused")

        record = await client.get_cve("CVE-2024-1234")

    assert record is None


# Test 6: HTTP 403 (rate limited) → returns None
@pytest.mark.asyncio
async def test_http_403_returns_none(client, mock_httpx_response_403):
    """HTTP 403 → returns None, no exception."""
    with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = mock_httpx_response_403

        record = await client.get_cve("CVE-2024-1234")

    assert record is None


# Test 7: HTTP 404 (not found) → returns None
@pytest.mark.asyncio
async def test_http_404_returns_none(client, mock_httpx_response_404):
    """HTTP 404 → returns None."""
    with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = mock_httpx_response_404

        record = await client.get_cve("CVE-2024-1234")

    assert record is None


# Test 8: HTTP 500 → returns None
@pytest.mark.asyncio
async def test_http_500_returns_none(client, mock_httpx_response_500):
    """HTTP 500 → returns None, no exception (graceful fallback)."""
    with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = mock_httpx_response_500

        record = await client.get_cve("CVE-2024-1234")

    assert record is None


# Test 9: malformed JSON → returns None
@pytest.mark.asyncio
async def test_malformed_json_returns_none(client, mock_httpx_response_malformed):
    """Malformed JSON → returns None, no exception (graceful fallback)."""
    with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = mock_httpx_response_malformed

        record = await client.get_cve("CVE-2024-1234")

    assert record is None


# Test 10: empty vulnerabilities array → returns None
@pytest.mark.asyncio
async def test_empty_vulnerabilities_returns_none(client, mock_httpx_response_empty):
    """NVD returns 200 but vulnerabilities array is empty → None."""
    with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = mock_httpx_response_empty

        record = await client.get_cve("CVE-2024-9999")

    assert record is None


# Test 11: lower-case CVE ID normalized to upper-case
@pytest.mark.asyncio
async def test_lowercase_normalized(client, mock_httpx_response_ok):
    """Input 'cve-2024-1234' → normalized to 'CVE-2024-1234'."""
    with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = mock_httpx_response_ok

        record = await client.get_cve("cve-2024-1234")  # lowercase

    assert record is not None
    assert record.cve_id == "CVE-2024-1234"
    # Check the actual HTTP request used the normalized ID
    call_kwargs = mock_get.call_args.kwargs
    assert call_kwargs["params"]["cveId"] == "CVE-2024-1234"


# Test 12: rate limit (no API key) — 6th request within 30s → returns None
@pytest.mark.asyncio
async def test_rate_limit_no_key(client, mock_httpx_response_ok):
    """Without API key, 6th request in 30s is rejected (5 req/30s limit)."""
    with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = mock_httpx_response_ok

        # First 5 should succeed (different CVE IDs to avoid cache hits)
        results = []
        for i in range(RATE_LIMIT_MAX_REQUESTS_NO_KEY):
            r = await client.get_cve(f"CVE-2024-{1000 + i:04d}")
            results.append(r)

        # 6th request — rate limited → returns None
        record = await client.get_cve("CVE-2024-9999")

    # First 5 succeeded
    assert all(r is not None for r in results)
    # 6th failed (rate limited)
    assert record is None
    # Only 5 HTTP calls were made (6th was rejected before HTTP)
    assert mock_get.call_count == RATE_LIMIT_MAX_REQUESTS_NO_KEY


# Test 13: rate limit (with API key) — 51st request → returns None
@pytest.mark.asyncio
async def test_rate_limit_with_key(client_with_key, mock_httpx_response_ok):
    """With API key, 51st request in 30s is rejected (50 req/30s limit)."""
    with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = mock_httpx_response_ok

        # Make 50 requests (all should succeed)
        for i in range(RATE_LIMIT_MAX_REQUESTS_WITH_KEY):
            await client_with_key.get_cve(f"CVE-2024-{i:04d}")

        # 51st — rate limited
        record = await client_with_key.get_cve("CVE-2024-9999")

    assert record is None
    assert mock_get.call_count == RATE_LIMIT_MAX_REQUESTS_WITH_KEY


# Test 14: disk cache persistence
@pytest.mark.asyncio
async def test_disk_cache_persistence(client, mock_httpx_response_ok, cache_dir):
    """Record is persisted to data/cve_cache/CVE-*.json after fetch."""
    with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = mock_httpx_response_ok

        await client.get_cve("CVE-2024-1234")

    cache_file = cache_dir / "CVE-2024-1234.json"
    assert cache_file.exists()
    payload = json.loads(cache_file.read_text())
    assert payload["_cve_id"] == "CVE-2024-1234"
    assert payload["record"]["cve_id"] == "CVE-2024-1234"
    assert payload["record"]["cvss_base_score"] == 9.8
    assert "_cached_at" in payload


# Test 15: disk cache hit on new client instance (cross-process persistence)
@pytest.mark.asyncio
async def test_disk_cache_hit_new_instance(client, mock_httpx_response_ok, cache_dir):
    """Disk cache survives across NvdClient instances (cross-process)."""
    # First client fetches + caches to disk
    with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = mock_httpx_response_ok
        await client.get_cve("CVE-2024-1234")
        await client.close()

    # New client instance reads from disk cache (no HTTP call)
    client2 = NvdClient(cache_dir=cache_dir, cache_ttl_hours=24)
    with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get2:
        record = await client2.get_cve("CVE-2024-1234")

    assert record is not None
    assert record.cve_id == "CVE-2024-1234"
    # No HTTP call — disk cache hit
    mock_get2.assert_not_called()
    await client2.close()


# Test 16: API key header sent when api_key is set
@pytest.mark.asyncio
async def test_api_key_header_sent(client_with_key, mock_httpx_response_ok):
    """When api_key is set, 'apiKey' header is sent in request."""
    with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = mock_httpx_response_ok

        await client_with_key.get_cve("CVE-2024-1234")

    call_kwargs = mock_get.call_args.kwargs
    headers = call_kwargs["headers"]
    assert "apiKey" in headers
    assert headers["apiKey"] == "test-api-key-12345"


# Test 17: close() closes httpx client
@pytest.mark.asyncio
async def test_close_client(client, mock_httpx_response_ok):
    """close() closes the underlying httpx.AsyncClient."""
    with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = mock_httpx_response_ok
        await client.get_cve("CVE-2024-1234")  # triggers client creation

    assert client._http_client is not None

    await client.close()

    assert client._http_client is None


# Test 18: disabled client → returns None, no HTTP call
@pytest.mark.asyncio
async def test_disabled_client_returns_none(client_disabled):
    """enabled=False → returns None immediately, no HTTP call."""
    with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
        record = await client_disabled.get_cve("CVE-2024-1234")

    assert record is None
    mock_get.assert_not_called()


# Test 19: parser handles CVE with no CVSS metrics (newly published)
@pytest.mark.asyncio
async def test_parser_handles_no_cvss(client):
    """CVE without CVSS metrics (just published) → CveRecord with None CVSS fields."""
    response_no_cvss = {
        "vulnerabilities": [{
            "cve": {
                "id": "CVE-2024-9999",
                "descriptions": [{"lang": "en", "value": "Reserved CVE — no details yet."}],
                "metrics": {},
                "weaknesses": [],
                "references": [],
            }
        }]
    }
    mock_resp = httpx.Response(
        status_code=200, json=response_no_cvss,
        request=httpx.Request("GET", NVD_API_BASE),
    )
    with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = mock_resp

        record = await client.get_cve("CVE-2024-9999")

    assert record is not None
    assert record.cve_id == "CVE-2024-9999"
    assert record.description == "Reserved CVE — no details yet."
    assert record.cvss_vector is None
    assert record.cvss_base_score is None
    assert record.cvss_severity is None
    assert record.cwe_ids == []
    assert record.references == []


# Test 20: parser dedupes CWE IDs
@pytest.mark.asyncio
async def test_parser_dedupes_cwe(client):
    """Multiple identical CWE entries → deduped to single entry."""
    response_dups = {
        "vulnerabilities": [{
            "cve": {
                "id": "CVE-2024-5555",
                "descriptions": [{"lang": "en", "value": "Test"}],
                "metrics": {},
                "weaknesses": [
                    {"description": [{"lang": "en", "value": "CWE-79"}]},
                    {"description": [{"lang": "en", "value": "CWE-79"}]},
                    {"description": [{"lang": "en", "value": "CWE-89"}]},
                    {"description": [{"lang": "en", "value": "CWE-79"}]},
                ],
                "references": [],
            }
        }]
    }
    mock_resp = httpx.Response(
        status_code=200, json=response_dups,
        request=httpx.Request("GET", NVD_API_BASE),
    )
    with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = mock_resp
        record = await client.get_cve("CVE-2024-5555")

    assert record is not None
    # Should be deduped: [CWE-79, CWE-89] (not [CWE-79, CWE-79, CWE-89, CWE-79])
    assert record.cwe_ids == ["CWE-79", "CWE-89"]


# Test 21: parser skips "NVD-CWE-noinfo" entries
@pytest.mark.asyncio
async def test_parser_skips_noinfo_cwe(client):
    """'NVD-CWE-noinfo' weakness entries are skipped (not real CWE IDs)."""
    response_noinfo = {
        "vulnerabilities": [{
            "cve": {
                "id": "CVE-2024-6666",
                "descriptions": [{"lang": "en", "value": "Test"}],
                "metrics": {},
                "weaknesses": [
                    {"description": [{"lang": "en", "value": "NVD-CWE-noinfo"}]},
                    {"description": [{"lang": "en", "value": "CWE-22"}]},
                ],
                "references": [],
            }
        }]
    }
    mock_resp = httpx.Response(
        status_code=200, json=response_noinfo,
        request=httpx.Request("GET", NVD_API_BASE),
    )
    with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = mock_resp
        record = await client.get_cve("CVE-2024-6666")

    assert record is not None
    assert record.cwe_ids == ["CWE-22"]  # noinfo skipped


# Test 22: CveRecord serialization round-trip (to_dict + from_dict)
def test_cve_record_serialization_roundtrip():
    """CveRecord.to_dict() + from_dict() round-trips correctly."""
    original = CveRecord(
        cve_id="CVE-2024-1234",
        description="Test CVE",
        cvss_vector="CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H",
        cvss_base_score=9.8,
        cvss_severity="CRITICAL",
        cvss_version="3.1",
        cwe_ids=["CWE-89", "CWE-22"],
        references=["https://example.com"],
        published_date=datetime(2024, 1, 15, tzinfo=UTC),
        last_modified_date=datetime(2024, 2, 1, tzinfo=UTC),
        source="nvd",
    )
    d = original.to_dict()
    restored = CveRecord.from_dict(d)

    assert restored.cve_id == original.cve_id
    assert restored.description == original.description
    assert restored.cvss_vector == original.cvss_vector
    assert restored.cvss_base_score == original.cvss_base_score
    assert restored.cvss_severity == original.cvss_severity
    assert restored.cvss_version == original.cvss_version
    assert restored.cwe_ids == original.cwe_ids
    assert restored.references == original.references
    assert restored.published_date == original.published_date
    assert restored.last_modified_date == original.last_modified_date


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
