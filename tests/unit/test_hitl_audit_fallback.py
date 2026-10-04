"""Regression tests: HITL audit agent must use the configured AI channel.

Bug: `AuditAgent._build_audit_channel()` resolved API keys ONLY from
provider-specific env vars (OPENAI_API_KEY, ANTHROPIC_API_KEY, ...). When the
operator configured their LLM via **Settings → AI Channels** (config.yaml) and
left those env vars empty:

  - `_is_llm_configured()` returned False
  - every destructive tool review fell back to `reject`
    (`decided_by="audit_agent_fallback"`)
  - even harmless calls like `metasploit command:"version"` were rejected,
    which blocked the entire penetration phase.

The audit agent now falls back to config.yaml's default channel.

Run:
    pytest tests/unit/test_hitl_audit_fallback.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import app.core.channels as channels_mod
from app.core.channels import Channel
from app.hitl.audit_agent import AuditAgent

PROVIDER_ENV_VARS = (
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GLM_API_KEY",
    "DEEPSEEK_API_KEY",
    "GROQ_API_KEY",
    "GEMINI_API_KEY",
    "MINIMAX_API_KEY",
)


def _fake_channel() -> Channel:
    return Channel(
        id="deepseek",
        name="DeepSeek",
        provider="openai_compatible",
        base_url="https://api.deepseek.com/v1",
        api_key="sk-configured-in-yaml",
        model="deepseek-flash",
        max_completion_tokens=4096,
        temperature=0.7,
    )


@pytest.fixture
def no_provider_env(monkeypatch):
    for var in PROVIDER_ENV_VARS:
        monkeypatch.delenv(var, raising=False)


class TestAuditChannelFallback:
    def test_uses_config_channel_when_no_env_key(self, no_provider_env, monkeypatch):
        monkeypatch.setattr(channels_mod, "get_default_channel", _fake_channel)

        agent = AuditAgent(model="gpt-4o-mini")  # provider key removed above

        assert agent._is_llm_configured() is True, (
            "audit agent should be considered configured via config.yaml"
        )
        channel = agent._build_audit_channel()
        assert channel is not None
        assert channel.api_key == "sk-configured-in-yaml"
        assert channel.model == "deepseek-flash"      # channel's own model
        assert channel.provider == "openai_compatible"
        assert channel.base_url == "https://api.deepseek.com/v1"

    def test_no_config_channel_still_unconfigured(self, no_provider_env, monkeypatch):
        monkeypatch.setattr(channels_mod, "get_default_channel", lambda: None)

        agent = AuditAgent(model="gpt-4o-mini")

        assert agent._is_llm_configured() is False
        assert agent._build_audit_channel() is None

    def test_config_channel_without_key_is_unusable(self, no_provider_env, monkeypatch):
        blank = _fake_channel()
        blank.api_key = ""
        monkeypatch.setattr(channels_mod, "get_default_channel", lambda: blank)

        agent = AuditAgent(model="gpt-4o-mini")

        assert agent._is_llm_configured() is False

    def test_explicit_env_key_still_wins(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-env-openai")
        # A config channel exists but must NOT be used when the env key is set.
        monkeypatch.setattr(channels_mod, "get_default_channel", _fake_channel)

        agent = AuditAgent(model="gpt-4o-mini")
        channel = agent._build_audit_channel()

        assert channel is not None
        assert channel.api_key == "sk-env-openai"
        assert channel.base_url == "https://api.openai.com/v1"
        assert channel.model == "gpt-4o-mini"

    def test_audit_channel_disables_extended_thinking(self, no_provider_env, monkeypatch):
        monkeypatch.setattr(channels_mod, "get_default_channel", _fake_channel)
        channel = AuditAgent(model="gpt-4o-mini")._build_audit_channel()
        assert channel is not None
        # The audit decision must be fast/cheap — no chain-of-thought budget.
        assert channel.reasoning == {}
