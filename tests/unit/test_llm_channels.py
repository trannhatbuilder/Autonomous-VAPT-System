"""Regression tests for the AI-channel connection probe (Settings → AI Channels).

Root cause being guarded here
-----------------------------
Reasoning models (DeepSeek ``deepseek-flash``/``deepseek-reasoner``, OpenAI
o-series, Claude with extended thinking) emit their chain of thought in a
SEPARATE field -- ``reasoning_content`` for OpenAI-compatible APIs, a
``thinking`` content block for Anthropic's Messages API.  The old probe asked
for only ``max_tokens=5``, so the entire budget was consumed by reasoning, the
visible ``content`` came back empty with ``finish_reason="length"``, and the GUI
reported "Empty response -- check base_url path" even though the API key,
base URL and model were perfectly valid.

These tests pin: (a) reasoning output is parsed, (b) a reasoning-only 2xx
response is a SUCCESS, and (c) the probe asks for enough tokens.

Run:
    pytest tests/unit/test_llm_channels.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app.core import llm_client
from app.core.channels import Channel
from app.core.llm_client import (
    TEST_MAX_TOKENS,
    _parse_claude_response,
    _parse_openai_response,
)
# Aliased so pytest does not collect the imported helper as a test itself.
from app.core.llm_client import test_channel as probe_channel


def _response(payload: dict) -> httpx.Response:
    return httpx.Response(
        200,
        json=payload,
        request=httpx.Request("POST", "https://example.invalid/v1/messages"),
    )


def _channel(**over) -> Channel:
    base = dict(
        id="deepseek",
        name="DeepSeek",
        provider="openai_compatible",
        base_url="https://api.deepseek.com/v1",
        api_key="sk-test",
        model="deepseek-flash",
        max_completion_tokens=4096,
        temperature=0.7,
        reasoning={"mode": "on", "effort": "medium"},
    )
    base.update(over)
    return Channel(**base)


class TestReasoningParsing:
    def test_claude_thinking_block_is_captured(self):
        out = _parse_claude_response(_response({
            "content": [{"type": "thinking", "thinking": "let me think", "signature": "s"}],
            "stop_reason": "max_tokens",
            "usage": {"input_tokens": 31, "output_tokens": 5},
        }))
        assert out["content"] == ""
        assert out["reasoning_content"] == "let me think"

    def test_claude_keeps_text_and_tool_calls(self):
        out = _parse_claude_response(_response({
            "content": [
                {"type": "thinking", "thinking": "plan"},
                {"type": "text", "text": "running nmap"},
                {"type": "tool_use", "id": "tu1", "name": "run_nmap", "input": {"target": "x"}},
            ],
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 10, "output_tokens": 20},
        }))
        assert out["content"] == "running nmap"
        assert out["reasoning_content"] == "plan"
        assert out["finish_reason"] == "tool_calls"
        assert out["tool_calls"][0]["name"] == "run_nmap"

    def test_openai_reasoning_content_is_captured(self):
        out = _parse_openai_response(
            _response({
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "", "reasoning_content": "thinking hard"},
                    "finish_reason": "length",
                }],
                "usage": {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10},
            }),
            _channel(),
        )
        assert out["content"] == ""
        assert out["reasoning_content"] == "thinking hard"


class TestConnectionProbe:
    def test_probe_budget_is_above_a_reasoning_preamble(self):
        assert TEST_MAX_TOKENS >= 64

    async def test_reasoning_only_response_counts_as_success(self, monkeypatch):
        """The exact regression: 200 OK + reasoning-only must NOT say 'Empty response'."""

        async def fake_call(channel, messages, tools, temperature, override_max_tokens=None):
            assert override_max_tokens == TEST_MAX_TOKENS, "probe must use TEST_MAX_TOKENS"
            return {
                "content": "",
                "reasoning_content": "The user said hi, I should greet them.",
                "tool_calls": [],
                "finish_reason": "length",
                "usage": {"prompt_tokens": 4, "completion_tokens": 5, "total_tokens": 9},
            }

        monkeypatch.setattr(llm_client, "_call_one_channel", fake_call)
        result = await probe_channel(_channel())

        assert result["success"] is True
        assert result["error"] is None
        assert "reasoning" in (result["response_preview"] or "")

    async def test_truly_empty_response_still_fails(self, monkeypatch):
        async def fake_call(channel, messages, tools, temperature, override_max_tokens=None):
            return {
                "content": "",
                "reasoning_content": "",
                "tool_calls": [],
                "finish_reason": "stop",
                "usage": {},
            }

        monkeypatch.setattr(llm_client, "_call_one_channel", fake_call)
        result = await probe_channel(_channel())

        assert result["success"] is False
        assert "Empty response" in (result["error"] or "")
