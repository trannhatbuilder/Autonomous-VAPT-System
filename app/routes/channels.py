"""
VAPT-AI channels router (Phase B — CyberStrikeAI pattern).

Endpoints:
  GET    /api/channels                  — list all channels (api_key masked)
  GET    /api/channels/{channel_id}     — get one channel (api_key masked)
  POST   /api/channels                  — create new channel (full payload)
  PUT    /api/channels/{channel_id}     — update existing channel
  DELETE /api/channels/{channel_id}     — delete channel
  POST   /api/channels/{channel_id}/test — test connection (sends 'Hi' ping)
  POST   /api/channels/test             — test inline (no save, no auth) — for curl
  POST   /api/channels/default/{channel_id} — set default channel
  GET    /api/channels/default           — get current default channel id
"""
from __future__ import annotations

import logging
import traceback
from typing import Any

from fastapi import APIRouter, Body, Depends, HTTPException
from pydantic import BaseModel, Field

from app.core.channels import (
    Channel,
    CONFIG_PATH,
    delete_channel,
    get_channel,
    get_default_channel_id,
    list_channels,
    set_default_channel_id,
    upsert_channel,
)
from app.core.llm_client import TEST_MAX_TOKENS, test_channel

logger = logging.getLogger(__name__)


# ---------- Pydantic models ----------

class ChannelRequest(BaseModel):
    """Schema for create/update channel requests."""
    id: str = Field(..., description="Channel ID (e.g. 'openai', 'claude', 'glm')")
    name: str
    provider: str = Field(..., description="openai_compatible | claude")
    base_url: str
    api_key: str = Field(..., description="API key (plain). If starts with •, preserve existing key on update.")
    model: str
    max_total_tokens: int = 128000
    max_completion_tokens: int = 4096
    temperature: float = 0.7
    reasoning: dict[str, Any] = Field(default_factory=dict)
    failover_channels: list[str] = Field(
        default_factory=list,
        description="List of channel IDs to try as fallback on retryable errors (429/5xx/timeout).",
    )


class TestRequest(BaseModel):
    """For inline test (without saving) — POST /api/channels/test."""
    id: str = "test"
    name: str = "test"
    provider: str
    base_url: str
    api_key: str
    model: str
    max_total_tokens: int = 128000
    # Reasoning models burn tokens on their chain of thought before emitting any
    # visible text, so the probe needs headroom (see llm_client.TEST_MAX_TOKENS).
    max_completion_tokens: int = TEST_MAX_TOKENS
    temperature: float = 0.0
    reasoning: dict[str, Any] = Field(default_factory=dict)
    failover_channels: list[str] = Field(default_factory=list)


def create_channels_router(get_current_user) -> APIRouter:
    """Factory: creates a channels router with auth dependency injected."""
    router = APIRouter(prefix="/api/channels", tags=["channels"])

    # ---------- GET /api/channels ----------

    @router.get("")
    async def list_all_channels(
        user=Depends(get_current_user),
    ) -> dict[str, Any]:
        """List all configured channels (api_key masked)."""
        channels = list_channels()
        default_id = get_default_channel_id()
        return {
            "channels_count": len(channels),
            "default_channel": default_id,
            "config_path": str(CONFIG_PATH),
            "channels": [c.to_public_dict(mask_key=True) for c in channels],
        }

    # ---------- GET /api/channels/default ----------

    @router.get("/default")
    async def get_default(
        user=Depends(get_current_user),
    ) -> dict[str, Any]:
        """Return the default channel ID."""
        return {"default_channel": get_default_channel_id()}

    # ---------- POST /api/channels/default/{id} ----------

    @router.post("/default/{channel_id}")
    async def set_default(
        channel_id: str,
        user=Depends(get_current_user),
    ) -> dict[str, Any]:
        """Set the default channel."""
        try:
            set_default_channel_id(channel_id)
        except KeyError:
            raise HTTPException(status_code=404, detail=f"Channel {channel_id!r} not found")
        return {"status": "ok", "default_channel": channel_id}

    # ---------- GET /api/channels/diag (no auth, no LLM call) ----------
    # Phase D: simple connectivity check — confirms FastAPI + channels module are alive
    # IMPORTANT: must be registered BEFORE /{channel_id} route, otherwise FastAPI
    # matches "diag" as a channel_id and returns 404.

    @router.get("/diag")
    async def diag() -> dict[str, Any]:
        """Diagnostic endpoint — no auth, no LLM call. Confirms FastAPI is up."""
        try:
            channels = list_channels()
            default_id = get_default_channel_id()
            return {
                "ok": True,
                "fastapi_alive": True,
                "config_path": str(CONFIG_PATH),
                "config_exists": CONFIG_PATH.is_file(),
                "channels_count": len(channels),
                "default_channel": default_id,
                "channels_summary": [
                    {
                        "id": c.id,
                        "provider": c.provider,
                        "model": c.model,
                        "base_url": c.base_url,
                        "has_api_key": bool(c.api_key),
                        "key_masked": c.to_public_dict()["api_key"],
                        "failover_channels": list(c.failover_channels or []),
                    }
                    for c in channels
                ],
            }
        except Exception as exc:
            logger.exception("diag UNHANDLED EXCEPTION")
            return {
                "ok": False,
                "error": f"{type(exc).__name__}: {str(exc)[:300]}",
                "traceback": traceback.format_exc()[:1500],
            }

    # ---------- GET /api/channels/{id} ----------

    @router.get("/{channel_id}")
    async def get_one(
        channel_id: str,
        user=Depends(get_current_user),
    ) -> dict[str, Any]:
        ch = get_channel(channel_id)
        if ch is None:
            raise HTTPException(status_code=404, detail="Channel not found")
        return ch.to_public_dict(mask_key=True)

    # ---------- POST /api/channels ----------

    @router.post("")
    async def create_channel(
        req: ChannelRequest = Body(...),
        user=Depends(get_current_user),
    ) -> dict[str, Any]:
        # Validate uniqueness
        if get_channel(req.id) is not None:
            raise HTTPException(status_code=409, detail=f"Channel {req.id!r} already exists. Use PUT to update.")
        ch = Channel(
            id=req.id,
            name=req.name,
            provider=req.provider,
            base_url=req.base_url,
            api_key=req.api_key,
            model=req.model,
            max_total_tokens=req.max_total_tokens,
            max_completion_tokens=req.max_completion_tokens,
            temperature=req.temperature,
            reasoning=req.reasoning or {},
            failover_channels=req.failover_channels or [],
        )
        try:
            upsert_channel(ch)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        return {"status": "ok", "channel": ch.to_public_dict(mask_key=True)}

    # ---------- PUT /api/channels/{id} ----------

    @router.put("/{channel_id}")
    async def update_channel(
        channel_id: str,
        req: ChannelRequest = Body(...),
        user=Depends(get_current_user),
    ) -> dict[str, Any]:
        existing = get_channel(channel_id)
        if existing is None:
            raise HTTPException(status_code=404, detail=f"Channel {channel_id!r} not found")
        # Preserve existing api_key if request sends a masked one (••••...)
        api_key = req.api_key
        if api_key.startswith("•"):
            api_key = existing.api_key
        # Allow ID change? No — ID is the URL key. If they want to rename ID, must delete+create.
        ch = Channel(
            id=channel_id,
            name=req.name,
            provider=req.provider,
            base_url=req.base_url,
            api_key=api_key,
            model=req.model,
            max_total_tokens=req.max_total_tokens,
            max_completion_tokens=req.max_completion_tokens,
            temperature=req.temperature,
            reasoning=req.reasoning or {},
            failover_channels=req.failover_channels or [],
        )
        try:
            upsert_channel(ch)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        return {"status": "ok", "channel": ch.to_public_dict(mask_key=True)}

    # ---------- DELETE /api/channels/{id} ----------

    @router.delete("/{channel_id}")
    async def remove_channel(
        channel_id: str,
        user=Depends(get_current_user),
    ) -> dict[str, Any]:
        try:
            delete_channel(channel_id)
        except KeyError:
            raise HTTPException(status_code=404, detail=f"Channel {channel_id!r} not found")
        return {"status": "ok", "deleted": channel_id}

    # ---------- POST /api/channels/{id}/test ----------

    @router.post("/{channel_id}/test")
    async def test_one(
        channel_id: str,
        user=Depends(get_current_user),
    ) -> dict[str, Any]:
        """Test an already-saved channel (the saved key is used)."""
        try:
            ch = get_channel(channel_id)
            if ch is None:
                raise HTTPException(status_code=404, detail=f"Channel {channel_id!r} not found")
            if not ch.api_key or ch.api_key.startswith("•"):
                return {
                    "success": False,
                    "error": "Cannot test — saved api_key is masked. Re-enter the key first.",
                    "channel_id": channel_id,
                    "response_body": None,
                }
            return await test_channel(ch)
        except HTTPException:
            raise
        except Exception as exc:
            logger.exception(
                "test_one UNHANDLED EXCEPTION | channel_id=%s | error_type=%s",
                channel_id, type(exc).__name__,
            )
            return {
                "success": False,
                "error": f"Unhandled error: {type(exc).__name__}: {str(exc)[:300]}",
                "channel_id": channel_id,
                "traceback": traceback.format_exc()[:1500],
                "response_body": None,
            }

    # ---------- POST /api/channels/test (inline, no save) ----------

    @router.post("/test")
    async def test_inline(
        req: TestRequest = Body(...),
        user=Depends(get_current_user),
    ) -> dict[str, Any]:
        """Test a channel config inline (without saving). Used by the UI
        'Test Connection' button before Save."""
        try:
            if req.api_key.startswith("•"):
                return {
                    "success": False,
                    "error": "Cannot test with masked API key. Enter the actual key.",
                    "response_body": None,
                }
            ch = Channel(
                id=req.id or "inline-test",
                name=req.name or "Inline test",
                provider=req.provider,
                base_url=req.base_url,
                api_key=req.api_key,
                model=req.model,
                max_total_tokens=req.max_total_tokens,
                # Never let a tiny probe budget mask a healthy reasoning model.
                max_completion_tokens=max(req.max_completion_tokens or 0, TEST_MAX_TOKENS),
                temperature=0.0,
                reasoning=req.reasoning or {},
                failover_channels=[],  # inline test doesn't do failover
            )
            return await test_channel(ch)
        except Exception as exc:
            logger.exception(
                "test_inline UNHANDLED EXCEPTION | model=%s | error_type=%s",
                getattr(req, "model", "?"), type(exc).__name__,
            )
            return {
                "success": False,
                "error": f"Unhandled error: {type(exc).__name__}: {str(exc)[:300]}",
                "traceback": traceback.format_exc()[:1500],
                "response_body": None,
            }

    return router


__all__ = ["create_channels_router"]