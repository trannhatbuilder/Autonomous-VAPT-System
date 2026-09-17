"""
W8-B config patch — add NVD client settings to app/core/config.py.

Apply these additions to your local app/core/config.py (insert after
the "Metasploit RPC" section, around line 130):

    # ---------- NVD / CVE (W8-B) ----------
    nvd_enabled: bool = Field(default=True, validation_alias="VAPT_AI_NVD_ENABLED")
    nvd_api_key: SecretStr | None = Field(
        default=None,
        validation_alias="VAPT_AI_NVD_API_KEY",
        description="Optional NVD API key (raises rate limit from 5 to 50 req/30s). "
                    "Request one at https://nvd.nist.gov/developers/request-an-api-key",
    )
    nvd_cache_ttl_hours: int = Field(
        default=24,
        validation_alias="VAPT_AI_NVD_CACHE_TTL_HOURS",
        description="Cache TTL in hours. NVD updates daily, so 24h is reasonable.",
    )
    nvd_timeout_seconds: int = Field(
        default=15,
        validation_alias="VAPT_AI_NVD_TIMEOUT_SECONDS",
        description="Per-request HTTP timeout (NVD can be slow).",
    )

And add these to .env.example (after the MCP section):

    # ---------- NVD / CVE (W8-B) ----------
    # Enable CVE lookup via NVD REST API v2 (https://nvd.nist.gov/developers)
    VAPT_AI_NVD_ENABLED=true
    # Optional API key — raises rate limit from 5 to 50 req/30s
    # Request one at https://nvd.nist.gov/developers/request-an-api-key
    # VAPT_AI_NVD_API_KEY=your-nvd-api-key-here
    # Cache TTL (hours) — NVD updates daily, 24h is reasonable default
    VAPT_AI_NVD_CACHE_TTL_HOURS=24
    # Per-request HTTP timeout (seconds)
    VAPT_AI_NVD_TIMEOUT_SECONDS=15
"""
from __future__ import annotations

# This file is informational — user copies the snippets above into their
# local config.py and .env.example files.
#
# The snippets have been verified to:
# 1. Use the canonical VAPT_AI_* env var prefix
# 2. Use SecretStr for the API key (consistent with other secrets)
# 3. Provide sensible defaults (NVD works without API key for low volume)
# 4. Document rate limit implications in the description field
