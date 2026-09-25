"""
VAPT-AI LLM Client shim — Phase B/C.

Backward-compatible shim for the existing agent loop
(app/agents/react_agent.py, app/agents/base.py, app/orchestration/langgraph_supervisor.py).

Loads the **default channel** from `config.yaml` (via app.core.channels)
and delegates to app.core.llm_client.chat_completion (native httpx).

Existing call sites stay unchanged:
    llm_config = await get_user_llm_config(session, str(user_id))
    response = await chat_completion(llm_config, messages, tools, temperature)

The `llm_config` dict now contains keys:
    provider, base_url, api_key, model, max_total_tokens,
    max_completion_tokens, temperature, channel_id, reasoning
"""
from __future__ import annotations

import logging
from typing import Any

from app.core.channels import Channel, get_default_channel
from app.core.llm_client import (
    LLMError,
    chat_completion as _native_chat,
    get_failover_channels,
)

logger = logging.getLogger(__name__)


# ---------- Channel → config dict (for backward compat) ----------

def _channel_to_config(channel: Channel) -> dict[str, Any]:
    """Convert a Channel object into the dict shape expected by the agent loop."""
    return {
        "channel_id": channel.id,
        "provider": channel.provider,
        "base_url": channel.base_url,
        "api_key": channel.api_key,
        "model": channel.model,
        "max_total_tokens": channel.max_total_tokens,
        "max_completion_tokens": channel.max_completion_tokens,
        "temperature": channel.temperature,
        "reasoning": channel.reasoning or {},
        "failover_channels": list(channel.failover_channels or []),
    }


async def get_user_llm_config(session: Any, user_id: str) -> dict[str, Any]:
    """Return LLM config as a dict (same shape as Phase A).

    Phase B: ignores `user_id` (config.yaml is shared, single-tenant).
    Loads the **default channel**. Raises ValueError if none configured.
    """
    channel = get_default_channel()
    if channel is None:
        raise ValueError(
            "LLM not configured. Open Settings → Channels and configure at least "
            "one channel with provider, base_url, api_key, and model. "
            "Config file: config.yaml (project root)."
        )
    return _channel_to_config(channel)


def get_channel_config(channel_id: str | None = None) -> dict[str, Any]:
    """Synchronous helper — load a specific channel by id (or default).

    Useful for non-async contexts (e.g. module-level setup).
    """
    from app.core.channels import get_channel, get_default_channel
    if channel_id:
        ch = get_channel(channel_id)
        if ch is None:
            raise ValueError(f"Channel {channel_id!r} not found in config.yaml")
        return _channel_to_config(ch)
    ch = get_default_channel()
    if ch is None:
        raise ValueError("No LLM channel configured. Edit config.yaml.")
    return _channel_to_config(ch)


def build_chat_params(
    llm_config: dict[str, Any],
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    temperature: float | None = None,
) -> dict[str, Any]:
    """Build a params dict for the native client.

    Renamed from the deprecated `build_litellm_params` (no litellm anymore).
    """
    return {
        "messages": messages,
        "tools": tools,
        "temperature": temperature if temperature is not None else llm_config.get("temperature", 0.7),
    }


async def chat_completion(
    llm_config: dict[str, Any],
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    temperature: float | None = None,
) -> dict[str, Any]:
    """Call LLM via native httpx client.

    Phase C: supports failover — if the primary channel returns a retryable
    error (429/5xx/timeout/conn-error), tries each channel in
    `llm_config["failover_channels"]` in order.

    Args:
        llm_config: dict from get_user_llm_config() — must have provider,
                    base_url, api_key, model, max_completion_tokens, etc.
                    Optional: failover_channels: list[str] (channel IDs).
        messages: chat messages list (OpenAI format)
        tools: optional tool schemas (OpenAI function calling format)
        temperature: override temperature

    Returns:
        dict with: content, tool_calls, finish_reason, usage,
        channel_used (id of channel that succeeded),
        attempts (list of {channel, status, error}).
    """
    # Build primary Channel from config dict
    channel = Channel(
        id=llm_config.get("channel_id", "default"),
        name=llm_config.get("channel_id", "default"),
        provider=llm_config.get("provider", "openai_compatible"),
        base_url=llm_config.get("base_url", ""),
        api_key=llm_config.get("api_key", ""),
        model=llm_config.get("model", ""),
        max_total_tokens=int(llm_config.get("max_total_tokens", 128000)),
        max_completion_tokens=int(llm_config.get("max_completion_tokens", 4096)),
        temperature=float(llm_config.get("temperature", 0.7)),
        reasoning=llm_config.get("reasoning") or {},
        failover_channels=list(llm_config.get("failover_channels") or []),
    )
    # Resolve failover Channels from IDs (silently skip missing)
    failovers = get_failover_channels(channel)
    return await _native_chat(channel, messages, tools, temperature, failovers)


__all__ = [
    "get_user_llm_config",
    "get_channel_config",
    "build_chat_params",
    "chat_completion",
    "LLMError",
]