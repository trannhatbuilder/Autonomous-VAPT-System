"""
VAPT-AI CVE/NVD API Client — async httpx client for NVD REST API v2.

W8-B: Live CVE lookup with 24h disk cache + graceful fallback.

Architecture (CyberStrikeAI-style — caller injects, client stays testable):
    - Singleton NvdClient instance created at app startup.
    - get_cve(cve_id) → CveRecord | None.
    - Cache: in-memory dict + JSON file at data/cve_cache/<cve_id>.json.
    - TTL: 24 hours (configurable via VAPT_AI_NVD_CACHE_TTL_HOURS).
    - Rate limit handling: 5 req/30s without API key, 50 req/30s with API key.
    - Graceful fallback: on network error / rate limit / malformed JSON →
      return None + log warning. NEVER raise — CVE lookup is best-effort.

NVD REST API v2 docs:
    https://services.nvd.nist.gov/rest/json/cves/2.0?cveId=CVE-2024-1234
    Response shape:
    {
      "resultsPerPage": 1,
      "startIndex": 0,
      "totalResults": 1,
      "format": "NVD_CVE",
      "version": "2.0",
      "vulnerabilities": [
        {
          "cve": {
            "id": "CVE-2024-1234",
            "sourceIdentifier": "...",
            "published": "2024-01-15T00:00:00.000",
            "lastModified": "2024-02-01T00:00:00.000",
            "descriptions": [
              {"lang": "en", "value": "Description text..."}
            ],
            "metrics": {
              "cvssMetricV31": [
                {
                  "cvssData": {
                    "version": "3.1",
                    "vectorString": "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H",
                    "baseScore": 9.8,
                    "baseSeverity": "CRITICAL"
                  },
                  "source": "nvd@nist.gov",
                  "type": "Primary"
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
              }
            ],
            "references": [
              {"url": "https://example.com/advisory", "source": "..."}
            ]
          }
        }
      ]
    }

Usage:
    from app.core.cve_client import nvd_client, get_cve

    record = await get_cve("CVE-2024-1234")
    if record:
        print(f"{record.cve_id} — CVSS {record.cvss_base_score} ({record.cvss_severity})")
        print(f"  CWE: {record.cwe_ids}")
        print(f"  Desc: {record.description[:100]}")
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, UTC
from pathlib import Path
from typing import Any

import httpx

# Note: `settings` is imported lazily inside get_nvd_client() to keep this
# module testable in isolation (without DB / .env loaded).

logger = logging.getLogger(__name__)


# ---------- Constants ----------

NVD_API_BASE = "https://services.nvd.nist.gov/rest/json/cves/2.0"
NVD_API_TIMEOUT_SECONDS = 15  # NVD can be slow
DEFAULT_CACHE_TTL_HOURS = 24
DEFAULT_USER_AGENT = "VAPT-AI/3.2.1 (security-research; +https://vapt-ai.local)"

# Rate limit constants (NVD applies these to all callers per IP):
# Without API key: 5 requests per rolling 30-second window.
# With API key:    50 requests per rolling 30-second window.
RATE_LIMIT_WINDOW_SECONDS = 30
RATE_LIMIT_MAX_REQUESTS_NO_KEY = 5
RATE_LIMIT_MAX_REQUESTS_WITH_KEY = 50

# Regex to validate CVE ID format (CVE-YYYY-NNNN+ where NNNN is 4+ digits)
CVE_ID_REGEX = re.compile(r"^CVE-\d{4}-\d{4,}$", re.IGNORECASE)


# ---------- Data classes ----------

@dataclass
class CveRecord:
    """Normalized CVE record — extracted from NVD API response.

    All fields are optional except cve_id — NVD sometimes returns records
    with partial data (e.g. no CVSS yet for newly-published CVEs).
    """
    cve_id: str
    description: str = ""
    cvss_vector: str | None = None
    cvss_base_score: float | None = None
    cvss_severity: str | None = None  # LOW / MEDIUM / HIGH / CRITICAL
    cvss_version: str | None = None  # "3.1" / "3.0" / "2.0"
    cwe_ids: list[str] = field(default_factory=list)  # ["CWE-89", "CWE-22"]
    references: list[str] = field(default_factory=list)  # URLs
    published_date: datetime | None = None
    last_modified_date: datetime | None = None
    source: str = "nvd"  # nvd | mitre | cache_only

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict for JSON cache file."""
        d = asdict(self)
        # Convert datetime to ISO for JSON
        if self.published_date:
            d["published_date"] = self.published_date.isoformat()
        if self.last_modified_date:
            d["last_modified_date"] = self.last_modified_date.isoformat()
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "CveRecord":
        """Deserialize from dict (JSON cache file)."""
        published = d.get("published_date")
        last_mod = d.get("last_modified_date")
        return cls(
            cve_id=d["cve_id"],
            description=d.get("description", ""),
            cvss_vector=d.get("cvss_vector"),
            cvss_base_score=d.get("cvss_base_score"),
            cvss_severity=d.get("cvss_severity"),
            cvss_version=d.get("cvss_version"),
            cwe_ids=d.get("cwe_ids", []),
            references=d.get("references", []),
            published_date=datetime.fromisoformat(published) if published else None,
            last_modified_date=datetime.fromisoformat(last_mod) if last_mod else None,
            source=d.get("source", "cache_only"),
        )


# ---------- Cache entry ----------

@dataclass
class _CacheEntry:
    """Internal cache entry — record + cached_at timestamp."""
    record: CveRecord
    cached_at: float  # unix timestamp


# ---------- NVD Client ----------

class NvdClient:
    """Async NVD REST API v2 client with disk + in-memory cache.

    Features:
        - Async via httpx.AsyncClient (lazy init, reused across calls)
        - In-memory dict cache for current process
        - Disk cache at data/cve_cache/<cve_id>.json for cross-process persistence
        - TTL: configurable (default 24h) — expired entries are re-fetched
        - Rate limit: tracks last 30s of requests; if exceeded, returns None
        - Graceful fallback: NEVER raises — on any error, returns None + logs warning
        - Optional API key via VAPT_AI_NVD_API_KEY env var (raises rate limit to 50 req/30s)
    """

    def __init__(
        self,
        cache_dir: Path | None = None,
        cache_ttl_hours: int = DEFAULT_CACHE_TTL_HOURS,
        api_key: str | None = None,
        timeout_seconds: int = NVD_API_TIMEOUT_SECONDS,
        enabled: bool = True,
    ):
        if cache_dir is None:
            # Lazy import — only needed if cache_dir not explicitly passed
            from app.core.config import settings
            cache_dir = settings.data_dir / "cve_cache"
        self.cache_dir = cache_dir
        self.cache_ttl_seconds = cache_ttl_hours * 3600
        self.api_key = api_key  # if None, uses anonymous (5 req/30s)
        self.timeout_seconds = timeout_seconds
        self.enabled = enabled

        # In-memory cache
        self._memory_cache: dict[str, _CacheEntry] = {}
        self._memory_cache_lock = asyncio.Lock()

        # Rate limit tracking (rolling window)
        self._request_timestamps: list[float] = []
        self._rate_limit_lock = asyncio.Lock()

        # httpx client (lazy init — only created when first needed)
        self._http_client: httpx.AsyncClient | None = None

    # ---------- Public API ----------

    async def get_cve(self, cve_id: str) -> CveRecord | None:
        """Fetch CVE record by ID.

        Lookup order:
            1. Validate CVE ID format (return None if invalid)
            2. Check in-memory cache
            3. Check disk cache
            4. Fetch from NVD API (respecting rate limit)
            5. Cache result in both memory + disk
            6. Return CveRecord or None (not found / error)

        NEVER raises — returns None on any failure.
        """
        if not self.enabled:
            logger.debug("NVD client disabled — skipping lookup for %s", cve_id)
            return None

        # Normalize: uppercase
        cve_id = cve_id.upper().strip()

        # Validate format
        if not CVE_ID_REGEX.match(cve_id):
            logger.warning("Invalid CVE ID format: %s (expected CVE-YYYY-NNNN+)", cve_id)
            return None

        # 1. Check in-memory cache
        cached = await self._get_from_memory_cache(cve_id)
        if cached is not None:
            logger.debug("Cache HIT (memory): %s", cve_id)
            return cached

        # 2. Check disk cache
        cached = await self._get_from_disk_cache(cve_id)
        if cached is not None:
            logger.debug("Cache HIT (disk): %s", cve_id)
            await self._set_in_memory_cache(cve_id, cached)
            return cached

        # 3. Check rate limit before making API call
        if not await self._check_rate_limit():
            logger.warning("NVD rate limit reached — skipping lookup for %s", cve_id)
            return None

        # 4. Fetch from NVD API
        try:
            record = await self._fetch_from_nvd(cve_id)
        except Exception as e:
            # Graceful fallback — log + return None
            logger.warning("NVD fetch failed for %s: %s", cve_id, e)
            return None

        if record is None:
            logger.info("CVE not found in NVD: %s", cve_id)
            return None

        # 5. Cache result
        await self._set_in_memory_cache(cve_id, record)
        await self._set_in_disk_cache(cve_id, record)

        return record

    async def close(self) -> None:
        """Close httpx client. Call on app shutdown."""
        if self._http_client is not None:
            await self._http_client.aclose()
            self._http_client = None
            logger.debug("NVD httpx client closed")

    # ---------- In-memory cache ----------

    async def _get_from_memory_cache(self, cve_id: str) -> CveRecord | None:
        async with self._memory_cache_lock:
            entry = self._memory_cache.get(cve_id)
            if entry is None:
                return None
            # Check TTL
            age = time.time() - entry.cached_at
            if age > self.cache_ttl_seconds:
                # Expired — remove from cache
                del self._memory_cache[cve_id]
                return None
            # Return cached record
            return entry.record

    async def _set_in_memory_cache(self, cve_id: str, record: CveRecord) -> None:
        async with self._memory_cache_lock:
            self._memory_cache[cve_id] = _CacheEntry(
                record=record,
                cached_at=time.time(),
            )

    # ---------- Disk cache ----------

    async def _get_from_disk_cache(self, cve_id: str) -> CveRecord | None:
        """Read cache file from data/cve_cache/<cve_id>.json."""
        cache_file = self._cache_file_path(cve_id)
        if not cache_file.exists():
            return None
        try:
            data = json.loads(cache_file.read_text(encoding="utf-8"))
            cached_at = data.get("_cached_at", 0)
            age = time.time() - cached_at
            if age > self.cache_ttl_seconds:
                # Expired — delete file
                cache_file.unlink(missing_ok=True)
                return None
            return CveRecord.from_dict(data["record"])
        except Exception as e:
            logger.warning("Failed to read disk cache for %s: %s", cve_id, e)
            return None

    async def _set_in_disk_cache(self, cve_id: str, record: CveRecord) -> None:
        """Write cache file to data/cve_cache/<cve_id>.json."""
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            cache_file = self._cache_file_path(cve_id)
            payload = {
                "_cached_at": time.time(),
                "_cve_id": cve_id,
                "record": record.to_dict(),
            }
            cache_file.write_text(
                json.dumps(payload, indent=2, default=str),
                encoding="utf-8",
            )
        except Exception as e:
            # Disk write failures are non-fatal (in-memory cache still works)
            logger.warning("Failed to write disk cache for %s: %s", cve_id, e)

    def _cache_file_path(self, cve_id: str) -> Path:
        """Cache file path: data/cve_cache/CVE-2024-1234.json."""
        # Replace dots/dashes to avoid filesystem issues (keep format readable)
        safe_name = cve_id  # CVE IDs are already filesystem-safe
        return self.cache_dir / f"{safe_name}.json"

    # ---------- Rate limiting ----------

    async def _check_rate_limit(self) -> bool:
        """Check if we can make another API call within rate limit.

        Cleans up old timestamps (older than 30s) and returns True if
        we have budget for one more request.
        """
        max_requests = (
            RATE_LIMIT_MAX_REQUESTS_WITH_KEY if self.api_key
            else RATE_LIMIT_MAX_REQUESTS_NO_KEY
        )
        now = time.time()
        cutoff = now - RATE_LIMIT_WINDOW_SECONDS

        async with self._rate_limit_lock:
            # Filter out timestamps older than window
            self._request_timestamps = [
                ts for ts in self._request_timestamps if ts > cutoff
            ]
            if len(self._request_timestamps) >= max_requests:
                return False
            # Reserve a slot (will be filled when _fetch_from_nvd runs)
            self._request_timestamps.append(now)
            return True

    # ---------- HTTP fetch ----------

    async def _fetch_from_nvd(self, cve_id: str) -> CveRecord | None:
        """Fetch CVE from NVD REST API v2.

        Returns CveRecord or None (not found). Raises on network/HTTP errors.
        """
        client = await self._get_http_client()

        headers = {
            "User-Agent": DEFAULT_USER_AGENT,
            "Accept": "application/json",
        }
        params = {"cveId": cve_id}
        if self.api_key:
            headers["apiKey"] = self.api_key

        response = await client.get(
            NVD_API_BASE,
            params=params,
            headers=headers,
            timeout=self.timeout_seconds,
        )

        # Rate limit response (NVD returns 403 when over limit)
        if response.status_code == 403:
            logger.warning(
                "NVD returned 403 (rate limited) for %s — wait 30s before retry",
                cve_id,
            )
            return None

        # Not found
        if response.status_code == 404:
            return None

        # Other HTTP errors — raise (will be caught by caller)
        if response.status_code >= 400:
            raise httpx.HTTPStatusError(
                f"NVD API error: HTTP {response.status_code} {response.reason_phrase}",
                request=response.request,
                response=response,
            )

        # Parse JSON
        data = response.json()
        return self._parse_nvd_response(cve_id, data)

    async def _get_http_client(self) -> httpx.AsyncClient:
        """Lazy-init the httpx client."""
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(
                follow_redirects=True,
                # No default timeout — we set per-request in _fetch_from_nvd
            )
        return self._http_client

    # ---------- Response parser ----------

    @staticmethod
    def _parse_nvd_response(cve_id: str, data: dict[str, Any]) -> CveRecord | None:
        """Parse NVD REST API v2 response into CveRecord.

        Returns None if the CVE is not in the response (NVD returns 200
        with empty vulnerabilities list for unknown CVE IDs).
        """
        vulnerabilities = data.get("vulnerabilities", [])
        if not vulnerabilities:
            return None

        # Take the first (and usually only) CVE
        cve_data = vulnerabilities[0].get("cve", {})
        if not cve_data:
            return None

        # ID (must match requested ID)
        actual_id = cve_data.get("id", "").upper()
        if actual_id != cve_id:
            logger.warning(
                "NVD returned different CVE ID: requested=%s, got=%s",
                cve_id, actual_id,
            )

        # Description (English)
        descriptions = cve_data.get("descriptions", [])
        description = ""
        for desc in descriptions:
            if desc.get("lang") == "en":
                description = desc.get("value", "")
                break

        # CVSS metrics (prefer v3.1 > v3.0 > v2.0)
        metrics = cve_data.get("metrics", {})
        cvss_vector = None
        cvss_base_score = None
        cvss_severity = None
        cvss_version = None

        for metric_key in ("cvssMetricV31", "cvssMetricV30", "cvssMetricV2"):
            if metric_key in metrics and metrics[metric_key]:
                # Take the primary metric (or first if no primary)
                metric_list = metrics[metric_key]
                primary = next(
                    (m for m in metric_list if m.get("type") == "Primary"),
                    metric_list[0],
                )
                cvss_data = primary.get("cvssData", {})
                cvss_vector = cvss_data.get("vectorString")
                cvss_base_score = cvss_data.get("baseScore")
                cvss_severity = cvss_data.get("baseSeverity")
                cvss_version = cvss_data.get("version")
                break

        # Weaknesses (CWE IDs)
        cwe_ids: list[str] = []
        for weakness in cve_data.get("weaknesses", []):
            for desc in weakness.get("description", []):
                if desc.get("lang") == "en":
                    cwe_value = desc.get("value", "")
                    # Sometimes NVD returns "NVD-CWE-noinfo" — skip those
                    if cwe_value.startswith("CWE-"):
                        cwe_ids.append(cwe_value)
        # Dedupe while preserving order
        seen: set[str] = set()
        cwe_ids = [c for c in cwe_ids if not (c in seen or seen.add(c))]

        # References (URLs only)
        references: list[str] = []
        for ref in cve_data.get("references", []):
            url = ref.get("url")
            if url:
                references.append(url)

        # Dates
        published_date = _parse_nvd_date(cve_data.get("published"))
        last_modified_date = _parse_nvd_date(cve_data.get("lastModified"))

        return CveRecord(
            cve_id=actual_id or cve_id,
            description=description,
            cvss_vector=cvss_vector,
            cvss_base_score=cvss_base_score,
            cvss_severity=cvss_severity,
            cvss_version=cvss_version,
            cwe_ids=cwe_ids,
            references=references,
            published_date=published_date,
            last_modified_date=last_modified_date,
            source="nvd",
        )


def _parse_nvd_date(date_str: str | None) -> datetime | None:
    """Parse NVD date format (ISO 8601 with milliseconds + Z)."""
    if not date_str:
        return None
    try:
        # NVD format: "2024-01-15T00:00:00.000"
        # (no timezone — NVD dates are UTC implicit)
        return datetime.fromisoformat(date_str.replace("Z", "+00:00"))
    except (ValueError, TypeError) as e:
        logger.debug("Failed to parse NVD date %r: %s", date_str, e)
        return None


# ---------- Module-level singleton ----------

# Lazy-init via property on Settings — created on first access
_nvd_client: NvdClient | None = None


def get_nvd_client() -> NvdClient:
    """Get the singleton NvdClient instance.

    Reads config from settings on first call. Subsequent calls return the
    same instance (in-memory cache persists across requests).
    """
    global _nvd_client
    if _nvd_client is None:
        # Lazy import — keeps module testable in isolation
        from app.core.config import settings

        api_key_raw = (
            settings.nvd_api_key.get_secret_value()
            if hasattr(settings, "nvd_api_key") and settings.nvd_api_key
            else None
        )
        _nvd_client = NvdClient(
            cache_dir=settings.data_dir / "cve_cache",
            cache_ttl_hours=getattr(settings, "nvd_cache_ttl_hours", DEFAULT_CACHE_TTL_HOURS),
            api_key=api_key_raw,
            timeout_seconds=getattr(settings, "nvd_timeout_seconds", NVD_API_TIMEOUT_SECONDS),
            enabled=getattr(settings, "nvd_enabled", True),
        )
        logger.info(
            "NvdClient initialized (cache_dir=%s, ttl=%dh, api_key=%s, enabled=%s)",
            _nvd_client.cache_dir,
            _nvd_client.cache_ttl_seconds // 3600,
            "yes" if _nvd_client.api_key else "no",
            _nvd_client.enabled,
        )
    return _nvd_client


async def get_cve(cve_id: str) -> CveRecord | None:
    """Convenience function — uses singleton NvdClient."""
    return await get_nvd_client().get_cve(cve_id)


async def close_nvd_client() -> None:
    """Close singleton NvdClient on app shutdown."""
    global _nvd_client
    if _nvd_client is not None:
        await _nvd_client.close()
        _nvd_client = None


# Convenience module-level instance (lazy via get_nvd_client)
nvd_client = get_nvd_client  # function — call as nvd_client() or await get_cve(...)
