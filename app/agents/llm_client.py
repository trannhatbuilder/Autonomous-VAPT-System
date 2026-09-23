"""
VAPT-AI LLM Client — multi-provider LLM wrapper using litellm.

Reads user's LLM config from DB (vapt_users.settings.llm) and provides
a unified async chat_completion() interface for the ReAct agent loop.

Supports: openai, anthropic, glm (z.ai), minimax, deepseek, groq, google, ollama
via litellm's provider routing.
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


async def get_user_llm_config(session: Any, user_id: str) -> dict[str, Any]:
    """Load LLM config from user's settings in DB.

    Returns dict with: provider, base_url, api_key, model,
    max_total_tokens, max_completion_tokens, temperature
    """
    import uuid as uuid_mod
    from sqlalchemy import select
    from app.db.models.user import User

    result = await session.execute(
        select(User).where(User.id == uuid_mod.UUID(user_id))
    )
    db_user = result.scalar_one_or_none()
    if db_user is None:
        raise ValueError(f"User {user_id} not found")

    settings = db_user.settings or {}
    llm = settings.get("llm", {})
    if not llm.get("provider"):
        raise ValueError(
            "LLM not configured. Go to Settings → configure LLM provider, "
            "API key, and model name."
        )
    return llm


def build_litellm_params(
    llm_config: dict[str, Any],
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    temperature: float | None = None,
) -> dict[str, Any]:
    """Build litellm.acompletion() kwargs from user LLM config.

    litellm uses model prefix routing:
        openai/    → OpenAI / OpenAI-compatible
        anthropic/ → Claude
        glm/       → GLM (z.ai) — litellm maps to openai-compatible
        minimax/   → MiniMax
        deepseek/  → DeepSeek
        groq/      → Groq
        gemini/    → Google Gemini
        ollama/    → Ollama (local)
    """
    provider = llm_config.get("provider", "openai")
    model = llm_config.get("model", "gpt-4o-mini")
    api_key = llm_config.get("api_key", "")
    base_url = llm_config.get("base_url", "")
    max_tokens = llm_config.get("max_completion_tokens", 4096)
    temp = temperature if temperature is not None else llm_config.get("temperature", 0.7)

    # Build model string with provider prefix
    # litellm convention: "provider/model_name"
    provider_prefix_map = {
        "openai": "openai",
        "anthropic": "anthropic",
        "glm": "openai",         # GLM uses OpenAI-compatible API
        "minimax": "openai",     # MiniMax uses OpenAI-compatible API
        "deepseek": "deepseek",
        "groq": "groq",
        "google": "gemini",
        "ollama": "ollama",
    }
    prefix = provider_prefix_map.get(provider, "openai")

    # For openai-compatible providers (glm, minimax), don't add prefix
    # — just use model name directly with custom base_url
    if provider in ("glm", "minimax") or (provider == "openai" and base_url):
        litellm_model = model  # litellm auto-detects via base_url
    elif prefix == "openai":
        litellm_model = f"{prefix}/{model}"
    else:
        litellm_model = f"{prefix}/{model}"

    params: dict[str, Any] = {
        "model": litellm_model,
        "messages": messages,
        "temperature": temp,
        "max_tokens": max_tokens,
    }

    # API key
    if api_key:
        params["api_key"] = api_key

    # Base URL for OpenAI-compatible providers (GLM, MiniMax, custom OpenAI)
    if base_url:
        params["api_base"] = base_url.rstrip("/")

    # Tools (function calling)
    if tools:
        params["tools"] = tools
        params["tool_choice"] = "auto"

    return params


async def chat_completion(
    llm_config: dict[str, Any],
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    temperature: float | None = None,
) -> dict[str, Any]:
    """Call LLM via litellm.acompletion().

    Args:
        llm_config: user LLM config from DB (provider, api_key, model, ...)
        messages: chat messages list
        tools: optional tool schemas for function calling
        temperature: override temperature

    Returns:
        dict with:
            content: str — LLM text response
            tool_calls: list[dict] — tool calls if any
            finish_reason: str — "stop" | "tool_calls" | "length"
            usage: dict — token counts
    """
    import litellm

    # Suppress litellm's verbose logging
    litellm.suppress_debug_info = True

    params = build_litellm_params(llm_config, messages, tools, temperature)

    logger.info(
        "LLM call | model=%s | messages=%d | tools=%d | temp=%.1f",
        params.get("model", "?"),
        len(messages),
        len(tools) if tools else 0,
        params.get("temperature", 0.7),
    )

    try:
        response = await litellm.acompletion(**params)
        choice = response.choices[0]
        message = choice.message

        result: dict[str, Any] = {
            "content": message.content or "",
            "tool_calls": [],
            "finish_reason": choice.finish_reason or "stop",
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                "total_tokens": response.usage.total_tokens if response.usage else 0,
            },
        }

        # Extract tool calls if present
        if message.tool_calls:
            for tc in message.tool_calls:
                result["tool_calls"].append({
                    "id": tc.id,
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                })

        logger.info(
            "LLM response | finish=%s | content_len=%d | tool_calls=%d | tokens=%d",
            result["finish_reason"],
            len(result["content"]),
            len(result["tool_calls"]),
            result["usage"]["total_tokens"],
        )
        return result

    except Exception as exc:
        logger.error("LLM call failed: %s", exc)
        raise


__all__ = [
    "get_user_llm_config",
    "build_litellm_params",
    "chat_completion",
]