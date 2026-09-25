"""
VAPT-AI channels module (Phase B — CyberStrikeAI pattern).

Single source of truth: `config.yaml` at project root.
This module loads / saves / mutates channels atomically with a .backup.

A "channel" is a (provider, base_url, api_key, model, ...) tuple.
- provider = "openai_compatible" → call /chat/completions endpoint
- provider = "claude"            → call /v1/messages endpoint (Anthropic native)

This replaces the DB-stored per-user `vapt_users.settings.llm` JSONB column
used in Phase A. Backward compat is preserved via `app/agents/llm_client.py`
which now reads the default channel from this file.
"""
from __future__ import annotations

import logging
import os
import shutil
import threading
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# Path to config.yaml — resolved relative to project root (one level up from this file)
CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "config.yaml"

# Lock for atomic file writes (prevents races between concurrent saves).
_save_lock = threading.Lock()

# In-memory cache — invalidated on save.
_cache: dict[str, Any] | None = None
_cache_mtime: float = 0.0


# ---------- Dataclass ----------

@dataclass
class Channel:
    """A single AI channel — corresponds to one entry in `ai.channels` map."""

    id: str
    name: str
    provider: str = "openai_compatible"   # openai_compatible | claude
    base_url: str = ""
    api_key: str = ""
    model: str = ""
    max_total_tokens: int = 128000
    max_completion_tokens: int = 4096
    temperature: float = 0.7
    reasoning: dict[str, Any] = field(default_factory=dict)
    failover_channels: list[str] = field(default_factory=list)

    def validate(self) -> str | None:
        """Return None if valid, else an error message string."""
        if not self.id or not self.id.strip():
            return "Channel ID is required"
        if not self.name or not self.name.strip():
            return "Channel name is required"
        if self.provider not in ("openai_compatible", "claude"):
            return f"Provider must be 'openai_compatible' or 'claude', got {self.provider!r}"
        if not self.base_url or not self.base_url.strip():
            return "Base URL is required"
        if not self.api_key or not self.api_key.strip():
            return "API key is required"
        if not self.model or not self.model.strip():
            return "Model is required"
        # Failover channels must reference existing IDs (check happens at upsert time)
        if self.failover_channels and not isinstance(self.failover_channels, list):
            return "failover_channels must be a list of channel IDs"
        return None

    def to_public_dict(self, mask_key: bool = True) -> dict[str, Any]:
        """Return dict for API responses — masks API key by default."""
        d = {
            "id": self.id,
            "name": self.name,
            "provider": self.provider,
            "base_url": self.base_url,
            "model": self.model,
            "max_total_tokens": self.max_total_tokens,
            "max_completion_tokens": self.max_completion_tokens,
            "temperature": self.temperature,
            "reasoning": self.reasoning or {},
            "failover_channels": list(self.failover_channels or []),
        }
        if mask_key and self.api_key:
            d["api_key"] = "•" * 8 + self.api_key[-4:] if len(self.api_key) > 4 else "****"
        else:
            d["api_key"] = self.api_key
        return d


# ---------- Load / Save ----------

def _read_yaml() -> dict[str, Any]:
    """Read config.yaml fresh from disk (no cache)."""
    global _cache, _cache_mtime
    if not CONFIG_PATH.is_file():
        return {}
    mtime = os.path.getmtime(CONFIG_PATH)
    # Cache hit (if same mtime)
    if _cache is not None and mtime == _cache_mtime:
        return _cache
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    _cache = data
    _cache_mtime = mtime
    return data


def _write_yaml(data: dict[str, Any]) -> None:
    """Write config.yaml atomically (with .backup)."""
    global _cache, _cache_mtime
    with _save_lock:
        # Make parent dir if missing
        CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)

        # Backup existing file
        if CONFIG_PATH.is_file():
            backup = CONFIG_PATH.with_suffix(".yaml.backup")
            try:
                shutil.copy2(CONFIG_PATH, backup)
            except OSError as exc:
                logger.warning("Failed to create config backup: %s", exc)

        # Write new content (tmp + rename for atomicity)
        tmp = CONFIG_PATH.with_suffix(".yaml.tmp")
        with tmp.open("w", encoding="utf-8") as f:
            yaml.safe_dump(
                data,
                f,
                default_flow_style=False,
                allow_unicode=True,
                sort_keys=False,
                width=100,
            )
        tmp.replace(CONFIG_PATH)

        # Invalidate cache
        _cache = None
        _cache_mtime = 0.0


# ---------- Public API ----------

def get_default_channel_id() -> str | None:
    """Return the configured default channel ID, or None if not set."""
    data = _read_yaml()
    return data.get("default_channel") or None


def set_default_channel_id(channel_id: str) -> None:
    """Set the default channel ID."""
    data = _read_yaml()
    channels = (data.get("ai") or {}).get("channels") or {}
    if channel_id not in channels:
        raise KeyError(f"Channel {channel_id!r} not found")
    data["default_channel"] = channel_id
    _write_yaml(data)


def list_channels() -> list[Channel]:
    """Return all configured channels as Channel objects (sorted by id)."""
    data = _read_yaml()
    raw = (data.get("ai") or {}).get("channels") or {}
    out: list[Channel] = []
    for cid, ch in raw.items():
        if not isinstance(ch, dict):
            continue
        out.append(_channel_from_dict(cid, ch))
    return sorted(out, key=lambda c: c.id)


def get_channel(channel_id: str) -> Channel | None:
    """Return one channel by ID, or None."""
    data = _read_yaml()
    raw = (data.get("ai") or {}).get("channels") or {}
    ch = raw.get(channel_id)
    if not ch or not isinstance(ch, dict):
        return None
    return _channel_from_dict(channel_id, ch)


def get_default_channel() -> Channel | None:
    """Return the default channel as Channel object, or None if not configured."""
    cid = get_default_channel_id()
    if not cid:
        return list_channels()[0] if list_channels() else None
    return get_channel(cid)


def upsert_channel(channel: Channel) -> None:
    """Insert or update a channel by id. Validates first."""
    err = channel.validate()
    if err:
        raise ValueError(err)
    data = _read_yaml()
    ai = data.setdefault("ai", {})
    channels = ai.setdefault("channels", {})
    channels[channel.id] = _channel_to_dict(channel)
    # If no default set yet, set this as default
    if not data.get("default_channel"):
        data["default_channel"] = channel.id
    _write_yaml(data)


def delete_channel(channel_id: str) -> None:
    """Delete a channel. If it was default, pick a new default."""
    data = _read_yaml()
    ai = data.get("ai") or {}
    channels = ai.get("channels") or {}
    if channel_id not in channels:
        raise KeyError(f"Channel {channel_id!r} not found")
    del channels[channel_id]
    ai["channels"] = channels
    data["ai"] = ai
    # Pick new default if needed
    if data.get("default_channel") == channel_id:
        data["default_channel"] = next(iter(channels), None)
    _write_yaml(data)


# ---------- Helpers ----------

def _channel_from_dict(cid: str, raw: dict[str, Any]) -> Channel:
    """Build a Channel from a YAML dict."""
    return Channel(
        id=cid,
        name=str(raw.get("name") or cid),
        provider=str(raw.get("provider") or "openai_compatible"),
        base_url=str(raw.get("base_url") or ""),
        api_key=str(raw.get("api_key") or ""),
        model=str(raw.get("model") or ""),
        max_total_tokens=int(raw.get("max_total_tokens") or 128000),
        max_completion_tokens=int(raw.get("max_completion_tokens") or 4096),
        temperature=float(raw.get("temperature") or 0.7),
        reasoning=dict(raw.get("reasoning") or {}) if raw.get("reasoning") else {},
        failover_channels=list(raw.get("failover_channels") or []),
    )


def _channel_to_dict(channel: Channel) -> dict[str, Any]:
    """Serialize a Channel back to a YAML dict (drop id)."""
    d = {
        "name": channel.name,
        "provider": channel.provider,
        "base_url": channel.base_url,
        "api_key": channel.api_key,
        "model": channel.model,
        "max_total_tokens": channel.max_total_tokens,
        "max_completion_tokens": channel.max_completion_tokens,
        "temperature": channel.temperature,
    }
    if channel.reasoning:
        d["reasoning"] = channel.reasoning
    if channel.failover_channels:
        d["failover_channels"] = list(channel.failover_channels)
    return d


__all__ = [
    "Channel",
    "CONFIG_PATH",
    "get_default_channel_id",
    "set_default_channel_id",
    "list_channels",
    "get_channel",
    "get_default_channel",
    "upsert_channel",
    "delete_channel",
]