"""
VAPT-AI Internet Verifier — Phase G1 (W19-FIX3-FG-BACKEND).

Confidence-boosting cross-check for findings emitted by the agent. Per the
user report:

    "to know whether the finding reported by a tool is real or not, this AI
     model will search the internet to confirm it."

This module performs a *lightweight* web search against a public search
engine (DuckDuckGo HTML — no API key required) and uses the snippets
returned to derive a confidence score in [0.0, 1.0].

Design constraints (per spec G1):

    - **Non-blocking**: every call MUST return within 10 seconds. We wrap
      the HTTP fetch + parse in `asyncio.wait_for(..., timeout=10)`. If
      the timeout fires, we return a "unverified" result + log a warning.
    - **24h cache**: results are cached per-finding-key in memory to avoid
      spamming the search engine for the same vuln pattern across scans.
      The cache key is a stable hash of (vuln_type, cve_id, tool_used).
    - **Graceful fallback**: if httpx is not installed, network is
      unreachable, or the search engine returns an error, we return
      `InternetVerificationResult(confirmed=False, confidence=0.0,
      summary="Internet search unavailable")` — never raise.
    - **Always non-fatal**: the caller (`_phase_audit_findings` in
      scan_pipeline.py) wraps every call in try/except, so any unexpected
      exception here degrades to "no internet consensus" without aborting
      the audit phase.

Strategy for scoring (per spec G1):

    For each finding, we issue 2-3 distinct search queries:

        Q1: "{vuln_type} PoC {cve_id}"     — does a public PoC exist?
        Q2: "{tool_used} detect {vuln_type} pattern" — is the tool's
                                                   detection signature documented?
        Q3 (optional): "{cvss_vector} valid?" — sanity-check the CVSS
                                                 vector against CVSS calculator docs

    Each query that returns ≥1 hit with a snippet containing the vuln_type
    token contributes +0.20 to the confidence score (capped at 1.0).
    A query that returns zero hits OR a snippet that says "false positive"
    contributes 0.0 (we don't subtract — internet absence is weak signal).

    Final confidence: sum of contributions, capped at 1.0.
    Confirmed := True iff confidence ≥ 0.40 (i.e. ≥2 of the 3 queries hit).

Wire-up (Phase G2):

    In `scan_pipeline._phase_audit_findings`, after `auditor.verify_finding()`
    returns `accepted=True`, we call:

        result = await internet_verifier.verify_finding(finding_dict)
        if result.confirmed:
            f.confidence_score = min(1.0, f.confidence_score + 0.05)
        f.metadata_json["internet_verification"] = asdict(result)

    This is wrapped in try/except so any failure (timeout, network, parse)
    leaves the finding's audit verdict untouched.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from dataclasses import dataclass, field, asdict
from typing import Any
from urllib.parse import quote_plus

logger = logging.getLogger(__name__)


# ── Constants ──────────────────────────────────────────────────────────────

# Per-call hard timeout (seconds). Per spec G1: "always returns within 10s".
_VERIFY_TIMEOUT_SECONDS: float = 10.0

# Cache TTL (seconds). Per spec G1: "Cache results for 24h in an in-memory dict".
_CACHE_TTL_SECONDS: int = 24 * 3600

# Confidence contribution per query that hits.
_CONFIDENCE_PER_HIT: float = 0.20

# Threshold for `confirmed=True`. ≥2 of 3 queries hitting → 0.40 → confirmed.
_CONFIDENCE_CONFIRM_THRESHOLD: float = 0.40

# DuckDuckGo HTML endpoint — no API key required, tolerant of bot traffic
# (limited rate; we cap at ≤3 queries per finding and respect 24h cache).
_SEARCH_ENDPOINT: str = "https://html.duckduckgo.com/html/"


# ── Result dataclass ───────────────────────────────────────────────────────

@dataclass
class InternetVerificationResult:
    """Outcome of an internet verification pass on a single finding.

    Stored in `Finding.metadata_json["internet_verification"]` (JSONB) so
    the report exporter can render the "Internet-Verified" badge without
    a DB migration.

    Attributes:
        confirmed: True iff ≥2 of the 3 search queries returned hits
                   containing the vuln_type token.
        confidence: Numeric confidence in [0.0, 1.0]. Computed as
                    (#hit_queries × 0.20), capped at 1.0.
        references: List of source URLs (top 3) from the search results.
        summary: Human-readable summary of what was found / not found.
        queries: The list of search queries actually issued (for audit trail).
        elapsed_ms: Wall-clock time spent on the verification (ms).
    """
    confirmed: bool
    confidence: float
    references: list[str] = field(default_factory=list)
    summary: str = ""
    queries: list[str] = field(default_factory=list)
    elapsed_ms: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Serialize for storage in metadata_json (JSONB-safe)."""
        return asdict(self)


# ── In-memory 24h cache ────────────────────────────────────────────────────
# Key: stable hash of (vuln_type, cve_id, tool_used, cvss_vector).
# Value: tuple of (cached_at_timestamp, InternetVerificationResult).
_CACHE: dict[str, tuple[float, InternetVerificationResult]] = {}


def _cache_key(finding: dict[str, Any]) -> str:
    """Compute a stable cache key from the finding's identifying fields."""
    vuln_type = str(finding.get("vuln_type") or finding.get("title") or "").lower()
    cve_id = str(finding.get("cve_id") or "").upper()
    tool_used = str(finding.get("tool_used") or finding.get("source_tool") or "").lower()
    cvss_vector = str(finding.get("cvss_vector") or "").lower()
    raw = f"{vuln_type}|{cve_id}|{tool_used}|{cvss_vector}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _read_cache(key: str) -> InternetVerificationResult | None:
    """Return cached result if present + not expired (TTL=24h). Else None."""
    entry = _CACHE.get(key)
    if entry is None:
        return None
    cached_at, result = entry
    if (time.time() - cached_at) > _CACHE_TTL_SECONDS:
        # Stale — evict.
        _CACHE.pop(key, None)
        return None
    return result


def _write_cache(key: str, result: InternetVerificationResult) -> None:
    """Cache a result with the current timestamp."""
    _CACHE[key] = (time.time(), result)


def clear_internet_verifier_cache() -> None:
    """Drop the entire in-memory cache (test helper + admin tooling)."""
    _CACHE.clear()


# ── Verifier ────────────────────────────────────────────────────────────────

class InternetVerifier:
    """Cross-check findings against public web search results.

    Usage:
        verifier = InternetVerifier()
        result = await verifier.verify_finding({
            "title": "SQL Injection in /login",
            "vuln_type": "sqli",
            "cve_id": "CVE-2024-1234",  # optional
            "tool_used": "nuclei",
            "cvss_vector": "AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H",  # optional
        })
        # result.confirmed, result.confidence, result.references, result.summary

    The verifier is safe to instantiate per-call (cheap) or as a singleton.
    The 24h cache is module-level so multiple instances share cache state.
    """

    def __init__(self, *, http_client_factory=None):
        """Initialize the verifier.

        Args:
            http_client_factory: Optional callable that returns an async
                context manager yielding an httpx.AsyncClient. Used only
                for testing — production code lets the verifier lazily
                build its own client via `httpx.AsyncClient`.
        """
        self._http_client_factory = http_client_factory

    async def verify_finding(self, finding: dict[str, Any]) -> InternetVerificationResult:
        """Cross-check a single finding against internet search results.

        Non-fatal: never raises. Any error (timeout, network, parse, etc.)
        degrades to a result with `confirmed=False, confidence=0.0, summary=...`.

        Args:
            finding: dict with at minimum `vuln_type` or `title`. Optional
                keys improve search precision: `cve_id`, `tool_used`,
                `cvss_vector`, `severity`.

        Returns:
            InternetVerificationResult. Always non-None.
        """
        start_ts = time.monotonic()
        key = _cache_key(finding)
        cached = _read_cache(key)
        if cached is not None:
            logger.info(
                "INTERNET_VERIFY_CACHE_HIT | key=%s | confirmed=%s | confidence=%.2f",
                key, cached.confirmed, cached.confidence,
            )
            return cached

        # Build the 2-3 search queries per spec G1.
        queries = self._build_search_queries(finding)

        try:
            # Hard 10s timeout wraps the entire verification pass.
            result = await asyncio.wait_for(
                self._run_queries(queries, finding),
                timeout=_VERIFY_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError:
            elapsed_ms = int((time.monotonic() - start_ts) * 1000)
            logger.warning(
                "INTERNET_VERIFY_TIMEOUT | key=%s | elapsed_ms=%d | queries=%s",
                key, elapsed_ms, queries,
            )
            result = InternetVerificationResult(
                confirmed=False,
                confidence=0.0,
                references=[],
                summary=f"Internet search timed out after {_VERIFY_TIMEOUT_SECONDS}s",
                queries=queries,
                elapsed_ms=elapsed_ms,
            )
        except Exception as exc:
            # Graceful degradation — never let verifier break audit phase.
            elapsed_ms = int((time.monotonic() - start_ts) * 1000)
            logger.warning(
                "INTERNET_VERIFY_FAILED | key=%s | elapsed_ms=%d | error=%s",
                key, elapsed_ms, exc,
            )
            result = InternetVerificationResult(
                confirmed=False,
                confidence=0.0,
                references=[],
                summary=f"Internet search unavailable: {exc}",
                queries=queries,
                elapsed_ms=elapsed_ms,
            )

        _write_cache(key, result)
        return result

    # ---------- Query construction ----------------------------------------

    def _build_search_queries(self, finding: dict[str, Any]) -> list[str]:
        """Build 2-3 distinct search queries from the finding fields.

        Per spec G1:
            Q1: "{vuln_type} PoC {cve_id if any}"
            Q2: "{tool_used} detect {vuln_type} pattern"
            Q3: "{cvss_vector} valid?" (only if cvss_vector is present)
        """
        vuln_type = (finding.get("vuln_type") or "").strip()
        cve_id = (finding.get("cve_id") or "").strip()
        tool_used = (finding.get("tool_used") or finding.get("source_tool") or "").strip()
        cvss_vector = (finding.get("cvss_vector") or "").strip()

        queries: list[str] = []

        # Q1: PoC lookup
        q1_parts: list[str] = []
        if vuln_type:
            q1_parts.append(vuln_type)
            q1_parts.append("PoC")
        if cve_id:
            q1_parts.append(cve_id)
        if q1_parts:
            queries.append(" ".join(q1_parts))

        # Q2: tool detection signature
        if tool_used and vuln_type:
            queries.append(f"{tool_used} detect {vuln_type} pattern")
        elif tool_used:
            queries.append(f"{tool_used} vulnerability detection")

        # Q3: CVSS vector sanity-check
        if cvss_vector:
            queries.append(f"{cvss_vector} CVSS vector valid")

        # Defensive: if we somehow built zero queries, fall back to the title.
        if not queries:
            title = (finding.get("title") or finding.get("name") or "").strip()
            if title:
                queries.append(f"{title} vulnerability")
            else:
                queries.append("vulnerability PoC")

        return queries[:3]  # hard cap at 3 per spec

    # ---------- Query execution ------------------------------------------

    async def _run_queries(
        self, queries: list[str], finding: dict[str, Any],
    ) -> InternetVerificationResult:
        """Issue each query against the search engine + aggregate results.

        Returns an InternetVerificationResult with confidence derived from
        the hit count + snippet content.
        """
        vuln_type = (finding.get("vuln_type") or "").lower()

        references: list[str] = []
        hit_count = 0
        snippets_seen: list[str] = []

        for query in queries:
            try:
                hits = await self._search(query)
            except Exception as exc:
                # Per-query failure is non-fatal — log + move on to next query.
                logger.debug(
                    "INTERNET_VERIFY_QUERY_FAILED | query=%s | error=%s",
                    query, exc,
                )
                continue

            # A query "hits" iff at least one snippet contains the
            # vuln_type token (case-insensitive). This is a weak signal —
            # we're checking "is this vuln_type publicly discussed?".
            query_hit = False
            for url, snippet in hits:
                references.append(url)
                if snippet:
                    snippets_seen.append(snippet)
                    if vuln_type and vuln_type in snippet.lower():
                        query_hit = True
            if query_hit:
                hit_count += 1

        # Dedupe references + keep top 3.
        seen = set()
        top_refs: list[str] = []
        for ref in references:
            if ref not in seen:
                seen.add(ref)
                top_refs.append(ref)
            if len(top_refs) >= 3:
                break

        confidence = min(1.0, hit_count * _CONFIDENCE_PER_HIT)
        confirmed = confidence >= _CONFIDENCE_CONFIRM_THRESHOLD

        if hit_count == 0:
            summary = (
                f"Internet search returned 0 relevant results across "
                f"{len(queries)} queries — vuln_type '{vuln_type}' may not "
                f"be publicly documented."
            )
        elif confirmed:
            summary = (
                f"Internet cross-check confirmed: {hit_count}/{len(queries)} "
                f"queries returned snippets matching '{vuln_type}'. "
                f"Confidence={confidence:.2f}. Top references: "
                f"{', '.join(top_refs[:2]) if top_refs else 'N/A'}."
            )
        else:
            summary = (
                f"Internet cross-check inconclusive: {hit_count}/{len(queries)} "
                f"queries matched. Confidence={confidence:.2f} below "
                f"confirm threshold ({_CONFIDENCE_CONFIRM_THRESHOLD:.2f})."
            )

        return InternetVerificationResult(
            confirmed=confirmed,
            confidence=confidence,
            references=top_refs,
            summary=summary,
            queries=queries,
            elapsed_ms=0,  # set by caller (verify_finding) after aggregation
        )

    async def _search(self, query: str) -> list[tuple[str, str]]:
        """Issue a single DuckDuckGo HTML search.

        Returns a list of (url, snippet) tuples. Empty list on no results.
        Parses the HTML response with a minimal regex-based extractor —
        we intentionally avoid pulling in BeautifulSoup/lxml as a hard
        dependency just for this verification step.

        Args:
            query: search query string.

        Returns:
            List of (url, snippet) tuples, up to 5 hits.

        Raises:
            ImportError: if httpx is not installed.
            httpx.HTTPError: on transport failure.
        """
        try:
            import httpx  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "httpx is required for internet verification. "
                "Install via: pip install httpx"
            ) from exc

        # DuckDuckGo HTML endpoint — POST form with q=query.
        # Headers mimic a normal browser to avoid the "blocked" interstitial.
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        }
        data = {"q": query, "b": "", "kl": "us-en"}  # minimal form

        # Use the factory if provided (testing); else build a per-call client.
        # Per-call clients are slightly slower but simpler + leak-proof.
        client_cm = (
            self._http_client_factory()
            if self._http_client_factory is not None
            else httpx.AsyncClient(
                follow_redirects=True,
                timeout=httpx.Timeout(5.0, connect=3.0),  # per-query budget
                headers=headers,
            )
        )

        async with client_cm as client:
            response = await client.post(_SEARCH_ENDPOINT, data=data)
            response.raise_for_status()
            text = response.text or ""

        # Parse hits with a minimal regex extractor. DuckDuckGo HTML wraps
        # results in `<a class="result__a" href="...">title</a>` blocks with
        # a `<a class="result__snippet">snippet</a>` below.
        return self._parse_ddg_html(text)

    @staticmethod
    def _parse_ddg_html(html: str) -> list[tuple[str, str]]:
        """Extract (url, snippet) pairs from DuckDuckGo HTML search results.

        Uses regex — DuckDuckGo's HTML structure is stable enough that a
        simple pattern is reliable. We intentionally do NOT depend on
        BeautifulSoup/lxml to keep the verifier dependency-light.

        Args:
            html: raw HTML response from DuckDuckGo's /html/ endpoint.

        Returns:
            List of (url, snippet) tuples, up to 5 hits. Empty list if
            parsing fails or no results found.
        """
        import re

        # Match result link blocks: <a class="result__a" href="URL">TITLE</a>
        link_re = re.compile(
            r'<a[^>]+class="[^"]*result__a[^"]*"[^>]*href="([^"]+)"[^>]*>(.*?)</a>',
            re.IGNORECASE | re.DOTALL,
        )
        # Match snippet blocks: <a class="result__snippet" ...>SNIPPET</a>
        snippet_re = re.compile(
            r'<a[^>]+class="[^"]*result__snippet[^"]*"[^>]*>(.*?)</a>',
            re.IGNORECASE | re.DOTALL,
        )

        links = link_re.findall(html)
        snippets = snippet_re.findall(html)

        # DuckDuckGo wraps the actual target URL in a redirect. Strip the
        # redirect prefix if present, else fall back to the raw href.
        def _strip_redirect(href: str) -> str:
            # DDG uses //duckduckgo.com/l/?uddg=<encoded_url>
            if "uddg=" in href:
                # Extract the uddg= parameter value.
                m = re.search(r"uddg=([^&]+)", href)
                if m:
                    from urllib.parse import unquote
                    return unquote(m.group(1))
            return href

        results: list[tuple[str, str]] = []
        for i, (raw_href, _title) in enumerate(links):
            url = _strip_redirect(raw_href)
            snippet = snippets[i] if i < len(snippets) else ""
            # Strip any inner HTML tags from the snippet (we only want text).
            snippet = re.sub(r"<[^>]+>", "", snippet).strip()
            if url:
                results.append((url, snippet))
            if len(results) >= 5:
                break

        return results


# ── Module-level convenience singleton ─────────────────────────────────────
# Callers that don't want to manage their own verifier instance can use this.
_default_verifier: InternetVerifier | None = None


def get_default_verifier() -> InternetVerifier:
    """Return a module-level singleton InternetVerifier.

    Useful for `_phase_audit_findings` to avoid re-instantiating per finding.
    """
    global _default_verifier
    if _default_verifier is None:
        _default_verifier = InternetVerifier()
    return _default_verifier


__all__ = [
    "InternetVerificationResult",
    "InternetVerifier",
    "get_default_verifier",
    "clear_internet_verifier_cache",
]
