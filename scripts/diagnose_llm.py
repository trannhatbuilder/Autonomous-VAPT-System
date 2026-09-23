#!/usr/bin/env python3
"""
VAPT-AI LLM Diagnostic v2 — raw HTTP test (bypass litellm).

Cải tiến từ v1: ngoài litellm call, script này còn gọi HTTP trực tiếp để xem
response thật từ NVIDIA/OpenAI/etc. Nếu litellm hang nhưng raw HTTP trả
response → litellm có bug. Nếu cả 2 hang/timeout → network thật sự slow.

Usage:
    cd ~/VAPT-AI
    source venv/bin/activate
    python scripts/diagnose_llm_v2.py
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


async def load_user_llm_config():
    """Load LLM config from DB for admin user."""
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


async def raw_http_test(llm_config: dict):
    """Call NVIDIA/OpenAI directly with httpx — bypasses litellm.

    Shows the REAL HTTP response (status code, body, timing) so we know
    exactly what the endpoint is saying.
    """
    import httpx

    base_url = llm_config.get("base_url", "https://api.openai.com/v1/").rstrip("/")
    api_key = llm_config.get("api_key", "")
    model = llm_config.get("model", "gpt-4o-mini")

    # Build chat/completions URL
    chat_url = f"{base_url}/chat/completions"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "ping"}],
        "max_tokens": 5,
        "temperature": 0.0,
    }

    print(f"  POST {chat_url}")
    print(f"  Authorization: Bearer {api_key[:4]}...{api_key[-4:]}")
    print(f"  Body: {payload}")
    print(f"  Timeout: 60s (NVIDIA NIM can be slow on first call)")
    print()

    start = time.time()
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(chat_url, headers=headers, json=payload)
        elapsed = time.time() - start

        print(f"  ✅ Got HTTP response in {elapsed:.2f}s")
        print(f"  Status: {response.status_code} {response.reason_phrase}")
        print(f"  Response headers:")
        for k, v in list(response.headers.items())[:8]:
            print(f"    {k}: {v[:200]}")
        print()
        print(f"  Response body (first 1000 chars):")
        body = response.text
        print(f"  {body[:1000]}")
        print()
        return response.status_code, body, elapsed

    except httpx.ConnectTimeout:
        elapsed = time.time() - start
        print(f"  ❌ ConnectTimeout after {elapsed:.2f}s")
        print(f"     Cannot establish TCP connection to {base_url}")
        return None, None, elapsed
    except httpx.ReadTimeout:
        elapsed = time.time() - start
        print(f"  ❌ ReadTimeout after {elapsed:.2f}s")
        print(f"     TCP connected, but no response from server.")
        print(f"     Likely causes:")
        print(f"       - NVIDIA NIM is cold-starting the model (try again in 30s)")
        print(f"       - Model name is invalid → server silently hangs")
        print(f"       - Network firewall is dropping the response")
        return None, None, elapsed
    except httpx.ConnectError as exc:
        elapsed = time.time() - start
        print(f"  ❌ ConnectError after {elapsed:.2f}s: {exc}")
        return None, None, elapsed
    except Exception as exc:
        elapsed = time.time() - start
        print(f"  ❌ Error after {elapsed:.2f}s: {type(exc).__name__}: {exc}")
        return None, None, elapsed


async def litellm_test(llm_config: dict):
    """Run litellm test (60s timeout)."""
    import litellm
    litellm.suppress_debug_info = True

    provider = llm_config.get("provider", "openai")
    model = llm_config.get("model", "gpt-4o-mini")
    base_url = llm_config.get("base_url", "")
    api_key = llm_config.get("api_key", "")

    # For openai-compatible providers with custom base_url, use model directly
    if provider in ("glm", "minimax") or (provider == "openai" and base_url):
        litellm_model = model
    else:
        provider_prefix_map = {
            "openai": "openai",
            "anthropic": "anthropic",
            "deepseek": "deepseek",
            "groq": "groq",
            "google": "gemini",
            "ollama": "ollama",
        }
        prefix = provider_prefix_map.get(provider, "openai")
        litellm_model = f"{prefix}/{model}"

    params = {
        "model": litellm_model,
        "messages": [{"role": "user", "content": "ping"}],
        "temperature": 0.0,
        "max_tokens": 5,
        "api_key": api_key,
    }
    if base_url:
        params["api_base"] = base_url.rstrip("/")

    print(f"  model:     {litellm_model}")
    print(f"  api_base:  {base_url or '(default)'}")
    print(f"  timeout:   60s")
    print()

    start = time.time()
    try:
        response = await asyncio.wait_for(
            litellm.acompletion(**params),
            timeout=60.0,
        )
        elapsed = time.time() - start
        choice = response.choices[0]
        content = (choice.message.content or "").strip()
        tokens = response.usage.total_tokens if response.usage else 0
        print(f"  ✅ Response in {elapsed:.2f}s")
        print(f"  content: {content!r}")
        print(f"  tokens:  {tokens}")
        return True
    except asyncio.TimeoutError:
        elapsed = time.time() - start
        print(f"  ❌ TIMEOUT after {elapsed:.2f}s")
        return False
    except Exception as exc:
        elapsed = time.time() - start
        print(f"  ❌ Error after {elapsed:.2f}s: {type(exc).__name__}: {exc}")
        return False


async def main():
    print("=" * 60)
    print("VAPT-AI LLM Diagnostic v2")
    print("=" * 60)
    print()

    # Step 1: Load LLM config
    print("[1/3] Loading LLM config from DB...")
    llm_config = await load_user_llm_config()
    if llm_config is None:
        return 1
    if not llm_config.get("provider"):
        print("  ❌ LLM not configured. Go to Settings → Save first.")
        return 1

    print(f"  provider:    {llm_config.get('provider')}")
    print(f"  model:       {llm_config.get('model')}")
    print(f"  base_url:    {llm_config.get('base_url', '(default)')}")
    api_key = llm_config.get("api_key", "")
    masked = f"••••{api_key[-4:]}" if len(api_key) > 4 else "****"
    print(f"  api_key:     {masked}")
    print()

    # Step 2: Raw HTTP test (bypass litellm)
    print("=" * 60)
    print("[2/3] Raw HTTP test (bypass litellm)")
    print("=" * 60)
    print()
    print("Goal: see exactly what NVIDIA/OpenAI responds (status code + body).")
    print("If this fails too → real network/endpoint issue.")
    print("If this succeeds → litellm has a bug with your config.")
    print()
    http_status, http_body, http_elapsed = await raw_http_test(llm_config)
    print()

    # Step 3: litellm test (for comparison)
    print("=" * 60)
    print("[3/3] litellm test (same params, via litellm.acompletion)")
    print("=" * 60)
    print()
    litellm_ok = await litellm_test(llm_config)
    print()

    # Diagnosis
    print("=" * 60)
    print("Diagnosis")
    print("=" * 60)
    print()

    if http_status == 200:
        print("✅ Raw HTTP test succeeded. NVIDIA endpoint + API key + model name all OK.")
        if not litellm_ok:
            print("❌ litellm test failed. litellm has a bug with your config.")
            print("   Workaround: bypass litellm — call OpenAI/NVIDIA directly with httpx.")
            print("   Contact me if you want me to write a custom chat_completion() that")
            print("   uses httpx instead of litellm.")
        else:
            print("✅ litellm also OK. LLM is fully working.")
            print("→ If scan still hangs, the issue is in the orchestrator code, not LLM.")
    elif http_status == 401:
        print("❌ HTTP 401 — API key is invalid for NVIDIA endpoint.")
        print("   Fix: Get a valid NVIDIA NIM API key from https://build.nvidia.com/")
        print("   → UI Settings → paste new key → Save → retry")
    elif http_status == 404:
        print(f"❌ HTTP 404 — endpoint not found OR model name is wrong.")
        print(f"   Model name: {llm_config.get('model')!r}")
        print(f"   Verify model name at: https://build.nvidia.com/explore/discover")
        print(f"   Common NVIDIA NIM model names:")
        print(f"     - google/gemma-7b")
        print(f"     - google/gemma-2-9b-it")
        print(f"     - google/gemma-2-27b-it")
        print(f"     - meta/llama-3.1-8b-instruct")
        print(f"     - meta/llama-3.1-70b-instruct")
        print(f"     - mistralai/mistral-7b-instruct")
        print(f"     - nvidia/llama-3.1-nemotron-70b-instruct")
    elif http_status == 400:
        print(f"❌ HTTP 400 — Bad request. NVIDIA rejected the payload.")
        if http_body:
            print(f"   Response: {http_body[:500]}")
        print(f"   Common cause: model name has wrong format (use lowercase + dash, no spaces)")
    elif http_status is None:
        print(f"❌ No HTTP response (timeout or connect error).")
        print(f"   Likely causes:")
        print(f"   1. NVIDIA NIM is cold-starting the model — try again in 30-60s")
        print(f"   2. Model name doesn't exist — NVIDIA silently drops the request")
        print(f"   3. Network firewall blocking the response (TCP connected but data doesn't arrive)")
        print(f"   ")
        print(f"   Next steps:")
        print(f"   1. Wait 60s + retry this script")
        print(f"   2. Try a different model name (e.g., 'meta/llama-3.1-8b-instruct')")
        print(f"   3. Test with curl directly:")
        print(f"      curl -X POST {llm_config.get('base_url', '')}chat/completions \\")
        print(f"        -H 'Authorization: Bearer <your-key>' \\")
        print(f"        -H 'Content-Type: application/json' \\")
        print(f"        -d '{{\"model\":\"{llm_config.get('model')}\",\"messages\":[{{\"role\":\"user\",\"content\":\"ping\"}}],\"max_tokens\":5}}'")
    else:
        print(f"❌ HTTP {http_status} — unexpected status.")
        print(f"   Response body: {http_body[:500] if http_body else '(empty)'}")

    return 0 if http_status == 200 else 1


if __name__ == "__main__":
    rc = asyncio.run(main())
    sys.exit(rc)