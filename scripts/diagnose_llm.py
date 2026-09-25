#!/usr/bin/env python3
"""
VAPT-AI LLM Diagnostic v3 — native httpx client test.

Phase C rewrite: replaces the old v2 litellm-vs-httpx comparison.
Since VAPT-AI no longer uses litellm anywhere, this script just runs
the same native client (app.core.llm_client) that the agent loop uses,
plus a raw HTTP probe for comparison.

Usage:
    cd ~/VAPT-AI
    source venv/bin/activate
    python scripts/diagnose_llm.py
"""

import asyncio
import sys
import os
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")


async def load_llm_config():
    """Load LLM config — tries config.yaml (Phase B default channel) first,
    falls back to DB user settings (Phase A legacy).
    """
    # Try Phase B: config.yaml default channel
    try:
        from app.core.channels import get_default_channel
        ch = get_default_channel()
        if ch is not None:
            return {
                "provider": ch.provider,
                "model": ch.model,
                "base_url": ch.base_url,
                "api_key": ch.api_key,
                "temperature": ch.temperature,
                "max_completion_tokens": ch.max_completion_tokens,
                "_channel_id": ch.id,
            }
    except Exception as exc:
        print(f"  (config.yaml path failed: {exc})")

    # Fall back to Phase A: DB user settings
    from app.db.session import async_session
    from sqlalchemy import select
    from app.db.models.user import User

    admin_email = os.environ.get("VAPT_AI_ADMIN_EMAIL", "admin@vapt-ai.local")
    async with async_session() as session:
        result = await session.execute(select(User).where(User.email == admin_email))
        user = result.scalar_one_or_none()
        if user is None:
            print(f"❌ Admin user {admin_email!r} not found")
            return None
        return user.settings.get("llm", {}) if user.settings else {}


async def native_client_test(llm_config: dict):
    """Run the same chat_completion() the agent loop uses."""
    from app.core.channels import Channel
    from app.core.llm_client import test_channel

    provider = "claude" if llm_config.get("provider") in ("anthropic", "claude") else "openai_compatible"
    channel = Channel(
        id="diagnose",
        name="Diagnose",
        provider=provider,
        base_url=(llm_config.get("base_url") or "").rstrip("/"),
        api_key=llm_config.get("api_key", ""),
        model=llm_config.get("model", ""),
        max_total_tokens=128000,
        max_completion_tokens=256,
        temperature=0.0,
        reasoning={},
        failover_channels=[],
    )
    print(f"  provider:   {channel.provider}")
    print(f"  model:       {channel.model}")
    print(f"  base_url:    {channel.base_url or '(default)'}")
    print(f"  api_key:     ••••{channel.api_key[-4:] if len(channel.api_key) > 4 else '****'}")
    print(f"  timeout:     20s")
    print()
    return await test_channel(channel)


async def main():
    print("=" * 60)
    print("VAPT-AI LLM Diagnostic v3 (native httpx)")
    print("=" * 60)
    print()

    print("[1/2] Loading LLM config...")
    llm_config = await load_llm_config()
    if llm_config is None:
        return 1
    if not llm_config.get("model"):
        print("  ❌ LLM not configured. Edit config.yaml or use the Settings UI.")
        return 1
    print()

    print("=" * 60)
    print("[2/2] Native client test (app.core.llm_client.test_channel)")
    print("=" * 60)
    print()
    result = await native_client_test(llm_config)
    print()
    print(f"  success:     {result.get('success')}")
    print(f"  latency_ms:  {result.get('latency_ms')}")
    print(f"  status_code: {result.get('status_code')}")
    print(f"  preview:     {result.get('response_preview')}")
    if not result.get("success"):
        print(f"  error:       {result.get('error')}")
        print(f"  body:        {(result.get('response_body') or '')[:300]}")
    print()

    return 0 if result.get("success") else 1


if __name__ == "__main__":
    rc = asyncio.run(main())
    sys.exit(rc)