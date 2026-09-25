"""
VAPT-AI native LLM client — Phase C (reasoning auto + failover + better errors).

Two providers only, no litellm:
  - openai_compatible  → POST {base_url}/chat/completions
  - claude             → POST {base_url}/v1/messages

Public API:
  - chat_completion(channel, messages, tools, temperature, failover_channels) -> dict
  - test_channel(channel) -> dict (success / latency / error / response_body)

The returned dict mimics the old litellm wrapper shape:
  {
    "content": str,
    "reasoning_content": str,       # chain-of-thought for reasoning models
    "tool_calls": list[dict],
    "finish_reason": str,
    "usage": {"prompt_tokens", "completion_tokens", "total_tokens"},
    "channel_used": str,            # which channel actually succeeded
    "attempts": list[dict],         # every attempt's status / error
  }
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any

import httpx

from app.core.channels import Channel, get_channel

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 180.0
# Phase E: increased from 60s → 180s for chat_completion (scan pipeline).
# Scan supervisor call uses full system prompt + 2 tools (transfer, exit).
# Specialist agent call uses full system prompt + ~10-30 tool schemas.
# Free-tier 70B models on OpenRouter can take 60-120s+ to process these.
# Test calls still use TEST_TIMEOUT=20s (separate) so channel test UI stays fast.
TEST_TIMEOUT = 20.0

# Token budget for the "Hi" connection probe.
#
# WHY NOT 5: reasoning models (DeepSeek `deepseek-flash`/`deepseek-reasoner`,
# OpenAI o-series, Claude with extended thinking) emit their chain of thought
# in a SEPARATE field (`reasoning_content` for OpenAI-compatible, a `thinking`
# content block for Anthropic). With `max_tokens=5` the whole budget is spent
# on reasoning, the visible `content` comes back empty with
# `finish_reason="length"`, and the UI wrongly reported
# "Empty response — check base_url path". 256 leaves room for the reasoning
# preamble plus a short greeting, so a healthy endpoint is recognised as OK.
TEST_MAX_TOKENS = 256

# HTTP status codes that are retryable on a different channel.
RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}

# Reasoning model family detectors (substring match on lowercased model name).
_OPENAI_REASONING_HINTS = ("o3", "o4", "gpt-5", "gpt-6", "o1-", "o1 ", "openai/o3", "openai/o4")
_CLAUDE_REASONING_HINTS = ("claude-3-7", "claude-3.7", "claude-4", "claude-opus-4", "claude-sonnet-4")
_DEEPSEEK_REASONING_HINTS = ("deepseek-r", "deepseek-reasoner", "deepseek-r1")
_QWEN_REASONING_HINTS = ("qwq", "qwen3-max", "qwen-reasoning")


# ============================================================
# Public entry point
# ============================================================

async def chat_completion(
    channel: Channel,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    temperature: float | None = None,
    failover_channels: list[Channel] | None = None,
) -> dict[str, Any]:
    """Call LLM, optionally failing over to backup channels on retryable errors.

    Args:
        channel: primary Channel to try first.
        messages: chat messages — OpenAI format [{role, content}, ...]
        tools: optional OpenAI tool schemas for function calling.
        temperature: override channel.temperature.
        failover_channels: list of backup Channel objects. Tried in order
            if the primary returns 429/5xx/timeout/conn-error.

    Returns:
        dict with: content, tool_calls, finish_reason, usage,
        channel_used (id of the channel that succeeded), attempts (list of {channel, status, error}).
    """
    attempts: list[dict[str, Any]] = []
    chain: list[Channel] = [channel, *(failover_channels or [])]

    last_error: LLMError | None = None
    for idx, ch in enumerate(chain):
        attempt: dict[str, Any] = {"channel": ch.id, "status": None, "error": None}
        try:
            result = await _call_one_channel(ch, messages, tools, temperature)
            result["channel_used"] = ch.id
            attempt["status"] = "ok"
            attempts.append(attempt)
            result["attempts"] = attempts
            if idx > 0:
                logger.info("LLM failover succeeded on channel %r after %d attempts", ch.id, idx + 1)
            return result
        except LLMError as exc:
            attempt["status"] = "error"
            attempt["error"] = str(exc)
            attempt["status_code"] = exc.status_code
            attempts.append(attempt)
            last_error = exc
            # Decide if retryable
            is_retryable = (
                exc.status_code in RETRYABLE_STATUS
                or exc.status_code is None   # network/timeout/conn-error
            )
            if not is_retryable or idx == len(chain) - 1:
                # 4xx (auth/not-found) OR ran out of failovers → bubble up
                exc.attempts = attempts
                raise
            logger.warning(
                "LLM call failed on channel %r (status=%s) — trying failover %d/%d",
                ch.id, exc.status_code, idx + 1, len(chain),
            )
            continue
    # Should never reach here
    assert last_error is not None
    last_error.attempts = attempts
    raise last_error


async def test_channel(channel: Channel) -> dict[str, Any]:
    """Send a 1-token 'Hi' ping and return success/latency + detailed error.

    Returns:
        {
          "success": bool,
          "model": str,
          "latency_ms": int,
          "error": str | None,
          "status_code": int | None,
          "response_preview": str | null,   # success: first 80 chars of model response
          "response_body": str | null,     # failure: first 500 chars of upstream response
        }
    """
    start = time.monotonic()
    logger.info(
        "test_channel START | channel=%s | provider=%s | model=%s | url=%s | timeout=%ss",
        channel.id, channel.provider, channel.model, channel.base_url, TEST_TIMEOUT,
    )
    try:
        response = await asyncio.wait_for(
            _call_one_channel(
                channel,
                messages=[{"role": "user", "content": "Hi"}],
                tools=None,
                temperature=0.0,
                override_max_tokens=TEST_MAX_TOKENS,
            ),
            timeout=TEST_TIMEOUT,
        )
        latency_ms = int((time.monotonic() - start) * 1000)
        content = (response.get("content") or "").strip()
        reasoning = (response.get("reasoning_content") or "").strip()
        if not content and not reasoning and not response.get("tool_calls"):
            logger.warning(
                "test_channel EMPTY RESPONSE | channel=%s | latency=%dms",
                channel.id, latency_ms,
            )
            return {
                "success": False,
                "model": channel.model,
                "latency_ms": latency_ms,
                "error": "Empty response — check base_url path (model not available on this endpoint?)",
                "status_code": None,
                "response_preview": None,
                "response_body": None,
            }
        # Reasoning models may answer entirely inside `reasoning_content` /
        # `thinking` when the probe budget is small. That still proves the key,
        # base_url and model are valid, so treat it as a successful connection.
        if content:
            preview = content[:80]
        else:
            preview = f"(reasoning) {reasoning[:80]}".strip()
        logger.info(
            "test_channel SUCCESS | channel=%s | latency=%dms | content=%r",
            channel.id, latency_ms, preview,
        )
        return {
            "success": True,
            "model": channel.model,
            "latency_ms": latency_ms,
            "error": None,
            "status_code": 200,
            "response_preview": preview,
            "response_body": None,
        }
    except asyncio.TimeoutError:
        latency_ms = int((time.monotonic() - start) * 1000)
        logger.warning(
            "test_channel TIMEOUT | channel=%s | latency=%dms (>%ss) | url=%s",
            channel.id, latency_ms, TEST_TIMEOUT, channel.base_url,
        )
        return {
            "success": False,
            "model": channel.model,
            "latency_ms": latency_ms,
            "error": (
                f"Timeout after {TEST_TIMEOUT}s — {channel.provider} endpoint at {channel.base_url} "
                f"did not respond. Common causes: (1) free-tier model is queued behind paid requests "
                f"(OpenRouter :free models can take 30-60s+); (2) network issue; (3) wrong base_url. "
                f"Tip: try a different model, or test directly with curl to bypass Next.js proxy."
            ),
            "status_code": None,
            "response_preview": None,
            "response_body": None,
        }
    except LLMError as exc:
        latency_ms = int((time.monotonic() - start) * 1000)
        logger.error(
            "test_channel LLMError | channel=%s | latency=%dms | status=%s | error=%s",
            channel.id, latency_ms, exc.status_code, str(exc)[:300],
        )
        return {
            "success": False,
            "model": channel.model,
            "latency_ms": latency_ms,
            "error": _friendly_error(exc, channel),
            "status_code": exc.status_code,
            "response_preview": None,
            "response_body": (exc.response_body or "")[:500] if exc.response_body else None,
        }
    except Exception as exc:
        latency_ms = int((time.monotonic() - start) * 1000)
        logger.exception(
            "test_channel UNEXPECTED ERROR | channel=%s | latency=%dms | error_type=%s",
            channel.id, latency_ms, type(exc).__name__,
        )
        return {
            "success": False,
            "model": channel.model,
            "latency_ms": latency_ms,
            "error": _friendly_error(exc, channel),
            "status_code": None,
            "response_preview": None,
            "response_body": None,
        }


def get_failover_channels(channel: Channel) -> list[Channel]:
    """Resolve a channel's failover_channels list to Channel objects.

    Silently skips IDs that don't exist in config.yaml.
    """
    out: list[Channel] = []
    for cid in (channel.failover_channels or []):
        ch = get_channel(cid)
        if ch is not None and ch.id != channel.id:
            out.append(ch)
    return out


# ============================================================
# Internal: call one channel (no failover)
# ============================================================

async def _call_one_channel(
    channel: Channel,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    temperature: float | None,
    override_max_tokens: int | None = None,
) -> dict[str, Any]:
    """Call exactly one channel — no failover. Raises LLMError on failure."""
    if channel.provider == "claude":
        return await _call_claude(channel, messages, tools, temperature, override_max_tokens)
    return await _call_openai_compat(channel, messages, tools, temperature, override_max_tokens)


# ============================================================
# OpenAI-compatible
# ============================================================

async def _call_openai_compat(
    channel: Channel,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    temperature: float | None,
    override_max_tokens: int | None = None,
) -> dict[str, Any]:
    """Call OpenAI-compatible /chat/completions endpoint."""
    base_url = (channel.base_url or "https://api.openai.com/v1").rstrip("/")
    url = f"{base_url}/chat/completions"

    payload: dict[str, Any] = {
        "model": channel.model,
        "messages": messages,
        "max_tokens": override_max_tokens or channel.max_completion_tokens,
        "temperature": channel.temperature if temperature is None else temperature,
    }

    # Resolve reasoning fields (auto-detect when mode == "auto")
    reasoning = _resolve_reasoning(channel)
    if reasoning.get("reasoning_effort"):
        payload["reasoning_effort"] = reasoning["reasoning_effort"]
    if reasoning.get("deepseek_thinking"):
        payload["thinking"] = {"type": "enabled"}

    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"

        if "openrouter.ai" in base_url:
            payload["provider"] = {"require_parameters": True}

    headers = {
        "Authorization": f"Bearer {channel.api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    logger.info(
        "LLM call | provider=openai_compat | model=%s | url=%s | messages=%d | tools=%d | temp=%s",
        channel.model, url, len(messages), len(tools) if tools else 0,
        payload.get("temperature"),
    )

    try:
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.post(url, json=payload, headers=headers)
        return _parse_openai_response(resp, channel)
    except httpx.TimeoutException as exc:
        raise LLMError(f"Request timeout: {exc}", status_code=None) from exc
    except httpx.ConnectError as exc:
        raise LLMError(f"Cannot connect to {url}: {exc}", status_code=None) from exc
    except httpx.HTTPError as exc:
        raise LLMError(f"HTTP error: {exc}", status_code=None) from exc


def _parse_openai_response(resp: httpx.Response, channel: Channel) -> dict[str, Any]:
    """Parse OpenAI-compatible response. Raises LLMError with body on API error."""
    if resp.status_code >= 400:
        body_text = resp.text or ""
        try:
            body_json = resp.json()
            err_msg = (
                body_json.get("error", {}).get("message")
                or body_json.get("message")
                or json.dumps(body_json)[:500]
            )
        except Exception:
            err_msg = body_text[:500] if body_text else f"HTTP {resp.status_code}"
        raise LLMError(
            f"API returned HTTP {resp.status_code}: {err_msg}",
            status_code=resp.status_code,
            response_body=body_text,
        )

    try:
        data = resp.json()
    except Exception as exc:
        raise LLMError(
            f"Invalid JSON response: {exc}",
            status_code=resp.status_code,
            response_body=resp.text[:500] if resp.text else "",
        ) from exc

    choices = data.get("choices") or []
    if not choices:
        raise LLMError(
            f"API response missing 'choices' — check base_url path (got keys: {list(data.keys())})",
            status_code=resp.status_code,
            response_body=json.dumps(data)[:500],
        )
    choice = choices[0]
    message = choice.get("message") or {}

    result = {
        "content": message.get("content") or "",
        # Reasoning models (DeepSeek R1/flash, OpenAI o-series via gateways)
        # put the chain of thought here instead of in `content`.
        "reasoning_content": message.get("reasoning_content") or message.get("reasoning") or "",
        "tool_calls": [],
        "finish_reason": choice.get("finish_reason") or "stop",
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
    }

    raw_tcs = message.get("tool_calls") or []
    for tc in raw_tcs:
        fn = tc.get("function") or {}
        result["tool_calls"].append({
            "id": tc.get("id", ""),
            "name": fn.get("name", ""),
            "arguments": fn.get("arguments", "{}"),
        })

    usage = data.get("usage") or {}
    if usage:
        result["usage"] = {
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
        }

    logger.info(
        "LLM response | finish=%s | content_len=%d | tool_calls=%d | tokens=%d",
        result["finish_reason"],
        len(result["content"]),
        len(result["tool_calls"]),
        result["usage"]["total_tokens"],
    )
    return result


# ============================================================
# Claude (Anthropic native)
# ============================================================

async def _call_claude(
    channel: Channel,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    temperature: float | None,
    override_max_tokens: int | None = None,
) -> dict[str, Any]:
    """Call Anthropic Messages API."""
    base_url = (channel.base_url or "https://api.anthropic.com").rstrip("/")
    url = f"{base_url}/v1/messages"

    system_parts = [m["content"] for m in messages if m.get("role") == "system" and m.get("content")]
    chat_messages = [m for m in messages if m.get("role") != "system"]
    converted = []
    for m in chat_messages:
        role = m.get("role")
        content = m.get("content")
        if role in ("user", "assistant") and content:
            converted.append({"role": role, "content": str(content) if not isinstance(content, str) else content})

    payload: dict[str, Any] = {
        "model": channel.model,
        "messages": converted,
        "max_tokens": override_max_tokens or channel.max_completion_tokens,
        "temperature": channel.temperature if temperature is None else temperature,
    }
    if system_parts:
        payload["system"] = "\n\n".join(system_parts)

    if tools:
        claude_tools = []
        for t in tools:
            fn = t.get("function") or t
            claude_tools.append({
                "name": fn.get("name", ""),
                "description": fn.get("description", ""),
                "input_schema": fn.get("parameters") or {"type": "object", "properties": {}},
            })
        payload["tools"] = claude_tools
        payload["tool_choice"] = {"type": "auto"}

    # Reasoning — Claude 3.7+ extended thinking
    reasoning = _resolve_reasoning(channel)
    if reasoning.get("claude_thinking"):
        payload["thinking"] = {"type": "enabled", "budget_tokens": reasoning.get("claude_budget", 10000)}

    headers = {
        "x-api-key": channel.api_key,
        "anthropic-version": "2023-06-01",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    logger.info(
        "LLM call | provider=claude | model=%s | url=%s | messages=%d | tools=%d",
        channel.model, url, len(messages), len(tools) if tools else 0,
    )

    try:
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.post(url, json=payload, headers=headers)
        return _parse_claude_response(resp)
    except httpx.TimeoutException as exc:
        raise LLMError(f"Request timeout: {exc}", status_code=None) from exc
    except httpx.ConnectError as exc:
        raise LLMError(f"Cannot connect to {url}: {exc}", status_code=None) from exc
    except httpx.HTTPError as exc:
        raise LLMError(f"HTTP error: {exc}", status_code=None) from exc


def _parse_claude_response(resp: httpx.Response) -> dict[str, Any]:
    """Parse Anthropic Messages API response → unified shape."""
    if resp.status_code >= 400:
        body_text = resp.text or ""
        try:
            body_json = resp.json()
            err_msg = (
                body_json.get("error", {}).get("message")
                or body_json.get("message")
                or json.dumps(body_json)[:500]
            )
        except Exception:
            err_msg = body_text[:500] if body_text else f"HTTP {resp.status_code}"
        raise LLMError(
            f"Claude API HTTP {resp.status_code}: {err_msg}",
            status_code=resp.status_code,
            response_body=body_text,
        )

    try:
        data = resp.json()
    except Exception as exc:
        raise LLMError(
            f"Invalid JSON: {exc}",
            status_code=resp.status_code,
            response_body=resp.text[:500] if resp.text else "",
        ) from exc

    content_blocks = data.get("content") or []
    text_parts = []
    reasoning_parts = []
    tool_calls = []
    for block in content_blocks:
        btype = block.get("type")
        if btype == "text":
            text_parts.append(block.get("text", ""))
        elif btype == "thinking":
            # Extended-thinking / reasoning models (Claude 3.7+, DeepSeek via
            # the Anthropic-compatible endpoint) return reasoning here. Keep it
            # so a reasoning-only probe is not mistaken for an empty response.
            reasoning_parts.append(block.get("thinking") or block.get("text") or "")
        elif btype == "tool_use":
            tool_calls.append({
                "id": block.get("id", ""),
                "name": block.get("name", ""),
                "arguments": json.dumps(block.get("input") or {}),
            })

    usage = data.get("usage") or {}

    return {
        "content": "".join(text_parts),
        "reasoning_content": "".join(reasoning_parts),
        "tool_calls": tool_calls,
        "finish_reason": "tool_calls" if tool_calls else (data.get("stop_reason") or "stop"),
        "usage": {
            "prompt_tokens": usage.get("input_tokens", 0),
            "completion_tokens": usage.get("output_tokens", 0),
            "total_tokens": usage.get("input_tokens", 0) + usage.get("output_tokens", 0),
        },
    }


# ============================================================
# Reasoning auto-detection
# ============================================================

def _resolve_reasoning(channel: Channel) -> dict[str, Any]:
    """Resolve effective reasoning fields based on mode + model name.

    Returns dict possibly containing:
      - reasoning_effort (OpenAI o-series / GPT-5)
      - deepseek_thinking (bool)
      - claude_thinking (bool)
      - claude_budget (int)
    """
    raw = channel.reasoning or {}
    mode = str(raw.get("mode", "auto")).lower()
    effort = str(raw.get("effort", "medium")).lower()
    budget = int(raw.get("budget_tokens", 10000))

    if mode == "off":
        return {}

    model_lower = (channel.model or "").lower()

    # If mode is "on" (force) — apply based on provider
    if mode == "on":
        if channel.provider == "claude":
            return {
                "claude_thinking": True,
                "claude_budget": budget,
            }
        if "deepseek" in model_lower or "r1" in model_lower:
            return {"deepseek_thinking": True}
        if any(h in model_lower for h in _OPENAI_REASONING_HINTS):
            return {"reasoning_effort": effort or "medium"}
        # Generic "on" with OpenAI-compat → just send reasoning_effort (no-op for non-reasoning models)
        return {"reasoning_effort": effort or "medium"}

    # mode == "auto" — detect by model name
    if any(h in model_lower for h in _OPENAI_REASONING_HINTS):
        return {"reasoning_effort": effort or "medium"}
    if any(h in model_lower for h in _DEEPSEEK_REASONING_HINTS):
        return {"deepseek_thinking": True}
    if any(h in model_lower for h in _QWEN_REASONING_HINTS):
        # Qwen3 reasoning models — send via OpenAI-compat extra_body
        return {"reasoning_effort": effort or "medium"}
    if channel.provider == "claude" and any(h in model_lower for h in _CLAUDE_REASONING_HINTS):
        return {
            "claude_thinking": True,
            "claude_budget": budget,
        }
    return {}


# ============================================================
# Error helpers
# ============================================================

class LLMError(Exception):
    """Raised when LLM call fails — carries HTTP status_code + body."""

    def __init__(
        self,
        message: str,
        status_code: int | None = None,
        response_body: str | None = None,
        attempts: list[dict[str, Any]] | None = None,
    ):
        super().__init__(message)
        self.status_code = status_code
        self.response_body = response_body
        self.attempts = attempts or []


def _friendly_error(exc: Exception, channel: Channel) -> str:
    """Convert exception to user-friendly message — includes status code + hint."""
    if isinstance(exc, LLMError):
        msg = str(exc)
        status = exc.status_code
        if status == 401:
            return f"401 Unauthorized — API key invalid OR not authorized for this model. Underlying: {msg[:300]}"
        if status == 403:
            return f"403 Forbidden — key may not have access to {channel.model!r}. Try a different model or upgrade plan. Underlying: {msg[:300]}"
        if status == 404:
            return f"404 Not Found — model {channel.model!r} not available at {channel.base_url}. Check model name spelling + path. Underlying: {msg[:300]}"
        if status == 429:
            return f"429 Rate limited — too many requests OR free-tier quota exhausted. Wait or switch to failover channel. Underlying: {msg[:300]}"
        if status in (500, 502, 503, 504):
            return f"{status} Upstream error — provider is having issues. Try again or failover. Underlying: {msg[:300]}"
        if status is None:
            if "timeout" in msg.lower() or "timed out" in msg.lower():
                return f"Network timeout — cannot reach {channel.base_url} within {DEFAULT_TIMEOUT}s. Check network/proxy/DNS. Underlying: {msg[:300]}"
            if "connect" in msg.lower() or "connection" in msg.lower():
                return f"Connection failed — cannot reach {channel.base_url}. Check URL + network. Underlying: {msg[:300]}"
        return msg[:500]
    # Generic
    msg = str(exc)
    if "timeout" in msg.lower():
        return f"Timeout after {DEFAULT_TIMEOUT}s — cannot reach {channel.provider} endpoint at {channel.base_url}. Underlying: {msg[:200]}"
    return msg[:500]


__all__ = [
    "chat_completion",
    "test_channel",
    "get_failover_channels",
    "LLMError",
    "RETRYABLE_STATUS",
    "TEST_MAX_TOKENS",
]