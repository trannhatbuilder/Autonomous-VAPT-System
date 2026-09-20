"""
VAPT-AI C2 Unified Shell Interface (W14-S6).

D28 decision: MSF meterpreter sessions + sqlmap --os-shell webshells +
custom Python beacons ALL sync into the same c2_sessions table
(distinguished by beacon_type field).

This module provides bridge functions that take sessions from external
tools (Metasploit RPC, sqlmap webshell) and create C2Session rows in the
unified table.

Reference: CyberStrikeAI does NOT have this — VAPT-AI adds the unified
table per master plan §3.2 (5 key differences, #3: Unified C2).
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime, UTC
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models.c2 import C2Session
from app.c2.types import BeaconType, ListenerType, SessionStatus

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sync functions
# ---------------------------------------------------------------------------

async def sync_msf_session(
    session: AsyncSession,
    scan_id: str,
    msf_session: dict[str, Any],
    listener_id: str | None = None,
) -> C2Session:
    """Sync a Metasploit session (meterpreter or shell) into the unified table.

    Called by the Metasploit integration (app.exploit.session_manager) when
    a new MSF session is established. Creates a C2Session row with
    beacon_type="msf_meterpreter" or "msf_shell".

    Args:
        session: Async SQLAlchemy session
        scan_id: Scan ID
        msf_session: Dict from MSF RPC session.list with fields:
            - id (int, MSF internal ID)
            - type (str: "meterpreter", "shell", etc.)
            - tunnel_peer (str: IP:port)
            - via_exploit (str)
            - via_payload (str)
            - info (str: "User: user @ host")
            - platform (str)
            - arch (str)
        listener_id: Optional listener UUID

    Returns:
        C2Session ORM instance (flushed, not committed)
    """
    msf_session_id = msf_session.get("id")
    msf_type = msf_session.get("type", "shell")
    tunnel_peer = msf_session.get("tunnel_peer", "unknown")
    info_str = msf_session.get("info", "")  # e.g. "User: user @ host"

    # Parse user + host from info string
    user = None
    host = None
    if "@" in info_str:
        try:
            parts = info_str.split("@")
            user = parts[0].replace("User:", "").strip()
            host = parts[1].strip()
        except Exception:
            pass

    # Determine beacon_type
    if "meterpreter" in msf_type.lower():
        beacon_type = BeaconType.MSF_METERPRETER
    else:
        beacon_type = BeaconType.MSF_SHELL

    # Check for existing session by MSF session_id (stored in metadata)
    msf_meta_id = str(msf_session_id)
    existing_result = await session.execute(
        select(C2Session)
        .where(C2Session.scan_id == scan_id)
        .where(C2Session.metadata_json["msf_session_id"].as_string() == msf_meta_id)
        .limit(1)
    )
    existing = existing_result.scalar_one_or_none()

    now = datetime.now(UTC)
    is_first_seen = existing is None

    if existing is not None:
        # Update heartbeat
        existing.last_seen_at = now
        existing.status = SessionStatus.ACTIVE.value
        await session.flush()
        c2_session = existing
    else:
        c2_session = C2Session(
            scan_id=scan_id,
            beacon_type=beacon_type.value,
            listener_type=ListenerType.HTTP_BEACON.value,  # MSF session established over RPC, not via our listener
            listener_id=listener_id,
            remote_address=tunnel_peer,
            hostname=host,
            username=user,
            os=msf_session.get("platform", "").lower() or None,
            arch=msf_session.get("arch", "").lower() or None,
            checkin_at=now,
            last_seen_at=now,
            status=SessionStatus.ACTIVE.value,
            metadata_json={
                "msf_session_id": str(msf_session_id),
                "msf_session_type": msf_type,
                "via_exploit": msf_session.get("via_exploit"),
                "via_payload": msf_session.get("via_payload"),
                "msf_info": info_str,
                "tunnel_peer": tunnel_peer,
                "first_seen_at": now.isoformat(),
                "last_checkin": now.isoformat(),
                "sync_source": "metasploit_rpc",
            },
        )
        session.add(c2_session)
        await session.flush()

    # Emit session_online SSE event
    try:
        from app.pentest.events import event_bus
        await event_bus.publish(scan_id, {
            "event": "session_online",
            "scan_id": scan_id,
            "session_id": str(c2_session.id),
            "beacon_type": c2_session.beacon_type,
            "msf_session_id": str(msf_session_id),
            "hostname": c2_session.hostname,
            "is_first_seen": is_first_seen,
            "timestamp": now.isoformat(),
        })
    except Exception as e:
        logger.warning("Failed to emit MSF session_online SSE: %s", e)

    logger.info(
        "MSF session synced: scan=%s session=%s msf_id=%s beacon=%s",
        scan_id, c2_session.id, msf_session_id, c2_session.beacon_type,
    )
    return c2_session


async def sync_sqlmap_webshell(
    session: AsyncSession,
    scan_id: str,
    webshell_url: str,
    webshell_type: str = "php",
    dbms: str | None = None,
    os: str | None = None,
    listener_id: str | None = None,
) -> C2Session:
    """Sync a sqlmap --os-shell webshell into the unified table.

    Called by the sqlmap tool wrapper (app.tools.sqlmap) when --os-shell
    successfully establishes a webshell. Creates a C2Session row with
    beacon_type="sqlmap_webshell".

    Args:
        session: Async SQLAlchemy session
        scan_id: Scan ID
        webshell_url: URL of the uploaded webshell (e.g. http://target/tmp/shell.php)
        webshell_type: Type of webshell (php, asp, jsp, etc.)
        dbms: Database management system if known (mysql, postgres, etc.)
        os: Target OS if detected by sqlmap
        listener_id: Optional listener UUID

    Returns:
        C2Session ORM instance (flushed, not committed)
    """
    # Check for existing session by webshell_url
    existing_result = await session.execute(
        select(C2Session)
        .where(C2Session.scan_id == scan_id)
        .where(C2Session.metadata_json["webshell_url"].as_string() == webshell_url)
        .limit(1)
    )
    existing = existing_result.scalar_one_or_none()

    now = datetime.now(UTC)
    is_first_seen = existing is None

    if existing is not None:
        existing.last_seen_at = now
        existing.status = SessionStatus.ACTIVE.value
        await session.flush()
        c2_session = existing
    else:
        c2_session = C2Session(
            scan_id=scan_id,
            beacon_type=BeaconType.SQLMAP_WEBSHELL.value,
            listener_type=ListenerType.HTTP_BEACON.value,  # Webshell accessed over HTTP
            listener_id=listener_id,
            remote_address=webshell_url,  # Reuse remote_address for webshell URL
            hostname=None,  # sqlmap doesn't always report hostname
            username=None,
            os=os.lower() if os else None,
            arch=None,
            checkin_at=now,
            last_seen_at=now,
            status=SessionStatus.ACTIVE.value,
            metadata_json={
                "webshell_url": webshell_url,
                "webshell_type": webshell_type,
                "dbms": dbms,
                "sync_source": "sqlmap_os_shell",
                "first_seen_at": now.isoformat(),
                "last_checkin": now.isoformat(),
            },
        )
        session.add(c2_session)
        await session.flush()

    # Emit session_online SSE event
    try:
        from app.pentest.events import event_bus
        await event_bus.publish(scan_id, {
            "event": "session_online",
            "scan_id": scan_id,
            "session_id": str(c2_session.id),
            "beacon_type": c2_session.beacon_type,
            "webshell_url": webshell_url,
            "webshell_type": webshell_type,
            "is_first_seen": is_first_seen,
            "timestamp": now.isoformat(),
        })
    except Exception as e:
        logger.warning("Failed to emit sqlmap session_online SSE: %s", e)

    logger.info(
        "sqlmap webshell synced: scan=%s session=%s url=%s",
        scan_id, c2_session.id, webshell_url,
    )
    return c2_session


# ---------------------------------------------------------------------------
# Unified session listing
# ---------------------------------------------------------------------------

async def list_unified_sessions(
    session: AsyncSession,
    scan_id: str,
    beacon_type: str | None = None,
    status: str | None = None,
) -> list[C2Session]:
    """List all C2 sessions for a scan — unified across all beacon types.

    D28 unified table: this is the single entry point for the UI dashboard.
    Filters by beacon_type + status if provided.

    Args:
        session: Async SQLAlchemy session
        scan_id: Scan ID
        beacon_type: Optional filter (python_beacon, msf_meterpreter, etc.)
        status: Optional filter (active, dead, etc.)

    Returns:
        List of C2Session ORM instances, ordered by last_seen_at DESC.
    """
    stmt = (
        select(C2Session)
        .where(C2Session.scan_id == scan_id)
        .order_by(C2Session.last_seen_at.desc())
    )
    if beacon_type:
        stmt = stmt.where(C2Session.beacon_type == beacon_type)
    if status:
        stmt = stmt.where(C2Session.status == status)

    result = await session.execute(stmt)
    return list(result.scalars().all())


async def get_session_stats(
    session: AsyncSession,
    scan_id: str,
) -> dict[str, Any]:
    """Get aggregate stats for C2 sessions in a scan.

    Returns a dict with:
        - total: total session count
        - by_beacon_type: {python_beacon: N, msf_meterpreter: N, ...}
        - by_status: {active: N, dead: N, ...}
    """
    sessions = await list_unified_sessions(session, scan_id)

    by_beacon: dict[str, int] = {}
    by_status: dict[str, int] = {}
    for s in sessions:
        bt = s.beacon_type or "unknown"
        by_beacon[bt] = by_beacon.get(bt, 0) + 1
        st = s.status or "unknown"
        by_status[st] = by_status.get(st, 0) + 1

    return {
        "scan_id": scan_id,
        "total": len(sessions),
        "by_beacon_type": by_beacon,
        "by_status": by_status,
    }