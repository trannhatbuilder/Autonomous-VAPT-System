"""
VAPT-AI C2 TCP Beacon Listener (W15-S2).

Python port of CyberStrikeAI internal/c2/listener_tcp.go.

TCP framing protocol (port from CyberStrikeAI tcp_beacon_server.go):
    Magic: b"CSB1" (4 bytes) — distinguishes beacon traffic from port scanners
    Length: uint32 big-endian (4 bytes) — payload length
    Payload: AES-GCM encrypted bytes (raw, not base64 — TCP is binary)

Dual-mode (per CyberStrikeAI allow_legacy_shell config):
    Encrypted mode (default): beacon sends magic bytes + AES-GCM frames
    Legacy raw shell (allow_legacy_shell=True): accepts bash -i >& /dev/tcp
        reverse shells — NO AUTH, NO ENCRYPTION. Lab testing only.

Per-session write mutex via asyncio.Lock (serializes task dispatch).

Task dispatch: server pushes tasks to beacon over the same TCP connection
(every 500ms, manager.pop_tasks_for_bacon() is polled for queued tasks).

Reference: CyberStrikeAI internal/c2/listener_tcp.go (Apache 2.0).
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import struct
import time
import uuid
from datetime import datetime, UTC
from typing import Any

from app.c2.types import (
    ListenerType, ImplantCheckInRequest, ImplantCheckInResponse,
    TaskResultRequest, ListenerConfig,
)
from app.c2.crypto import decrypt_aes_gcm, encrypt_aes_gcm, CryptoError
from app.c2.manager import C2Manager, InvalidInputError

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants — match CyberStrikeAI tcp_beacon_server.go
# ---------------------------------------------------------------------------

TCP_BEACON_MAGIC = b"CSB1"           # 4-byte magic for encrypted beacon protocol
TCP_PEEK_TIMEOUT = 5.0                # seconds to wait for magic bytes
TCP_READ_TIMEOUT = 60.0               # idle read timeout (refreshes heartbeat)
TCP_DISPATCH_INTERVAL = 0.5          # task dispatch poll interval (500ms)
TCP_MAX_FRAME_BYTES = 16 * 1024 * 1024  # 16 MB max frame size


# ---------------------------------------------------------------------------
# TCP Beacon Listener
# ---------------------------------------------------------------------------

class TCPBeaconListener:
    """TCP reverse beacon listener (port from CyberStrikeAI listener_tcp.go).

    Accepts inbound TCP connections from beacons. Two modes:
        1. Encrypted beacon (default): magic bytes CSB1 + AES-GCM frames
        2. Legacy raw shell (allow_legacy_shell=True): interactive shell
           via bash /dev/tcp or nc — no auth, lab use only

    Lifecycle:
        listener = TCPBeaconListener(manager, config, key, token, listener_id, ...)
        await listener.start()
        ...
        await listener.stop()
    """

    def __init__(
        self,
        manager: C2Manager,
        config: ListenerConfig,
        encryption_key_b64: str,
        implant_token_b64: str,
        listener_id: str,
        bind_host: str = "127.0.0.1",
        bind_port: int = 8444,
    ):
        self.manager = manager
        self.config = config
        self.config.apply_defaults()
        self.encryption_key_b64 = encryption_key_b64
        self.implant_token_b64 = implant_token_b64
        self.listener_id = listener_id
        self.bind_host = bind_host
        self.bind_port = bind_port

        self._server: asyncio.base_events.Server | None = None
        self._conns: dict[str, _TCPConn] = {}    # session_id → connection state
        self._lock = asyncio.Lock()
        self._stopped = False
        self._dispatch_task: asyncio.Task | None = None

    @property
    def type(self) -> str:
        return ListenerType.TCP_REVERSE.value

    # ---------- Lifecycle ----------

    async def start(self) -> None:
        """Start the TCP server (async, non-blocking)."""
        self._server = await asyncio.start_server(
            self._handle_conn,
            host=self.bind_host,
            port=self.bind_port,
        )

        # Start task dispatcher loop (pushes queued tasks to active connections)
        self._dispatch_task = asyncio.create_task(self._task_dispatcher_loop())

        await self.manager.register_running_listener(self.listener_id, self)

        logger.info(
            "C2 TCP listener started: id=%s bind=%s:%d",
            self.listener_id, self.bind_host, self.bind_port,
        )

    async def stop(self) -> None:
        """Gracefully stop the listener + close all active connections."""
        if self._stopped:
            return
        self._stopped = True

        if self._dispatch_task is not None:
            self._dispatch_task.cancel()
            try:
                await self._dispatch_task
            except asyncio.CancelledError:
                pass

        # Close all active connections
        async with self._lock:
            conns = list(self._conns.values())
            self._conns.clear()
        for tc in conns:
            try:
                tc.writer.close()
                await tc.writer.wait_closed()
            except Exception:
                pass

        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()

        await self.manager.unregister_running_listener(self.listener_id)
        logger.info("C2 TCP listener stopped: id=%s", self.listener_id)

    # ---------- Connection handling ----------

    async def _handle_conn(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        """Handle a single TCP connection.

        Reads magic bytes (CSB1) → if matches, encrypted beacon path.
        Otherwise, if allow_legacy_shell → raw shell path. Else → reject.
        """
        remote = writer.get_extra_info("peername", default=("unknown", 0))
        remote_str = f"{remote[0]}:{remote[1]}"

        try:
            # Peek magic bytes (with timeout)
            try:
                prefix = await asyncio.wait_for(
                    reader.readexactly(4), timeout=TCP_PEEK_TIMEOUT,
                )
            except (asyncio.IncompleteReadError, asyncio.TimeoutError):
                # Not enough bytes for magic — fall through to legacy shell check
                prefix = b""

            if prefix == TCP_BEACON_MAGIC:
                await self._handle_encrypted_beacon(reader, writer, remote_str)
                return

            # Not encrypted beacon — check if legacy shell allowed
            if not self.config.allow_legacy_shell:
                logger.debug(
                    "tcp_reverse rejected unencrypted connection: remote=%s",
                    remote_str,
                )
                return

            # Legacy raw shell mode
            # Put back the prefix bytes (already consumed) — for legacy mode
            # we treat the entire stream as interactive shell
            await self._handle_legacy_shell(reader, writer, remote_str, prefix)

        except (ConnectionResetError, BrokenPipeError):
            pass
        except Exception as e:
            logger.warning("TCP handle_conn error: remote=%s %s", remote_str, e)
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass

    async def _handle_encrypted_beacon(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        remote: str,
    ) -> None:
        """Handle encrypted TCP beacon session.

        Protocol (port from CyberStrikeAI tcp_beacon_server.go):
            1. Beacon sends magic CSB1 (already consumed in _handle_conn)
            2. Beacon sends AES-GCM encrypted ImplantCheckInRequest
               Frame: [4-byte length BE][encrypted payload]
            3. Server decrypts → manager.ingest_checkin()
            4. Server responds with encrypted ImplantCheckInResponse (framed)
            5. Loop: beacon polls tasks → server pushes via _task_dispatcher_loop
        """
        host = remote.split(":")[0] if ":" in remote else remote

        # Read check-in frame
        try:
            checkin_bytes = await self._read_frame(reader)
        except (asyncio.IncompleteReadError, asyncio.TimeoutError, CryptoError) as e:
            logger.warning("TCP beacon checkin read failed: %s", e)
            return

        try:
            plaintext = decrypt_aes_gcm(self.encryption_key_b64, checkin_bytes.decode("ascii"))
        except (CryptoError, UnicodeDecodeError) as e:
            logger.warning("TCP beacon checkin decrypt failed: %s", e)
            return

        import json
        try:
            req_data = json.loads(plaintext)
            req = ImplantCheckInRequest.from_dict(req_data)
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning("TCP beacon checkin parse failed: %s", e)
            return

        # Populate remote IP
        if not req.internal_ip:
            req.internal_ip = host

        # Persist via manager
        try:
            session = await self.manager.ingest_checkin(
                listener_id=self.listener_id,
                listener_type=self.type,
                req=req,
                encryption_key_b64=self.encryption_key_b64,
                implant_token_b64=self.implant_token_b64,
            )
        except InvalidInputError as e:
            logger.warning("TCP beacon checkin invalid: %s", e)
            return

        # Send check-in response (encrypted + framed)
        response = ImplantCheckInResponse(
            session_id=str(session.id),
            next_sleep=req.sleep_seconds or self.config.default_sleep,
            next_jitter=req.jitter_percent,
            has_tasks=False,  # tasks pushed async via dispatcher
            server_time=int(datetime.now(UTC).timestamp()),
        )
        response_json = json.dumps(response.to_dict()).encode("utf-8")
        encrypted = encrypt_aes_gcm(self.encryption_key_b64, response_json)
        await self._write_frame(writer, encrypted.encode("ascii"))

        # Register connection in active sessions
        tc = _TCPConn(session_id=str(session.id), reader=reader, writer=writer)
        async with self._lock:
            # Close any existing connection for this session
            old = self._conns.get(str(session.id))
            if old is not None:
                try:
                    old.writer.close()
                    await old.writer.wait_closed()
                except Exception:
                    pass
            self._conns[str(session.id)] = tc

        logger.info(
            "TCP beacon connected: scan=%s session=%s remote=%s",
            self.manager.scan_id, session.id, remote,
        )

        try:
            # Main loop — keep connection alive + read beacon results
            while not self._stopped:
                try:
                    frame_bytes = await asyncio.wait_for(
                        self._read_frame(reader), timeout=TCP_READ_TIMEOUT,
                    )
                except asyncio.TimeoutError:
                    # Idle — refresh heartbeat
                    await self._touch_session(str(session.id))
                    continue
                except asyncio.IncompleteReadError:
                    # Beacon disconnected
                    break

                # Decrypt + parse task result
                try:
                    result_plain = decrypt_aes_gcm(
                        self.encryption_key_b64, frame_bytes.decode("ascii"),
                    )
                    result_data = json.loads(result_plain)
                    result = TaskResultRequest.from_dict(result_data)
                    await self.manager.submit_task_result(result.task_id, result)
                except (CryptoError, json.JSONDecodeError, KeyError) as e:
                    logger.warning("TCP beacon result parse failed: %s", e)

        finally:
            async with self._lock:
                self._conns.pop(str(session.id), None)
            try:
                await self.manager.mark_session_dead(str(session.id))
            except Exception:
                pass

    async def _handle_legacy_shell(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        remote: str,
        prefix: bytes,
    ) -> None:
        """Handle legacy raw shell connection (allow_legacy_shell=True).

        Generates a stable implant_uuid from listener_id + remote IP so
        reconnects from the same source reuse the same session.
        """
        host = remote.split(":")[0] if ":" in remote else remote
        uuid_seed = f"{self.listener_id}|{host}"
        hash_bytes = hashlib.sha256(uuid_seed.encode("utf-8")).digest()
        implant_uuid = hash_bytes[:8].hex()

        checkin = ImplantCheckInRequest(
            implant_uuid=implant_uuid,
            hostname=f"tcp_{host}",
            username="unknown",
            os="unknown",
            arch="unknown",
            internal_ip=host,
            sleep_seconds=0,  # interactive — no sleep
            jitter_percent=0,
            metadata={"transport": "tcp_reverse_legacy", "remote": remote},
        )

        try:
            session = await self.manager.ingest_checkin(
                listener_id=self.listener_id,
                listener_type=self.type,
                req=checkin,
                encryption_key_b64=self.encryption_key_b64,
                implant_token_b64=self.implant_token_b64,
            )
        except InvalidInputError as e:
            logger.warning("TCP legacy shell checkin failed: %s", e)
            return

        tc = _TCPConn(session_id=str(session.id), reader=reader, writer=writer)
        async with self._lock:
            self._conns[str(session.id)] = tc

        logger.info(
            "TCP legacy shell connected: scan=%s session=%s remote=%s",
            self.manager.scan_id, session.id, remote,
        )

        try:
            # Main loop — read unsolicited output + refresh heartbeat
            while not self._stopped:
                try:
                    data = await asyncio.wait_for(
                        reader.read(4096), timeout=TCP_READ_TIMEOUT,
                    )
                except asyncio.TimeoutError:
                    await self._touch_session(str(session.id))
                    continue
                except asyncio.IncompleteReadError:
                    break

                if not data:
                    break

                # Refresh heartbeat on any data
                await self._touch_session(str(session.id))

                # Emit unsolicited output event (for UI debugging)
                try:
                    from app.pentest.events import event_bus
                    await event_bus.publish(self.manager.scan_id, {
                        "event": "task_complete",
                        "scan_id": self.manager.scan_id,
                        "session_id": str(session.id),
                        "status": "unsolicited_output",
                        "output": data.decode("utf-8", errors="replace")[:500],
                        "timestamp": datetime.now(UTC).isoformat(),
                    })
                except Exception:
                    pass

        finally:
            async with self._lock:
                self._conns.pop(str(session.id), None)
            try:
                await self.manager.mark_session_dead(str(session.id))
            except Exception:
                pass

    # ---------- Task dispatcher ----------

    async def _task_dispatcher_loop(self) -> None:
        """Background task: poll manager for queued tasks + push to beacons.

        Runs every TCP_DISPATCH_INTERVAL (500ms). For each active connection,
        pops up to 5 tasks and sends them (encrypted + framed).
        """
        while not self._stopped:
            try:
                async with self._lock:
                    snapshot = list(self._conns.values())

                for tc in snapshot:
                    try:
                        tasks = await self.manager.pop_tasks_for_beacon(
                            tc.session_id, limit=5,
                        )
                    except Exception as e:
                        logger.warning("Task pop failed: %s", e)
                        continue

                    for task in tasks:
                        # Send task asynchronously (don't block dispatcher)
                        asyncio.create_task(self._send_task(tc, task))
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning("TCP dispatch loop error: %s", e)

            await asyncio.sleep(TCP_DISPATCH_INTERVAL)

    async def _send_task(self, tc: "_TCPConn", task: Any) -> None:
        """Send a single task to a TCP beacon (encrypted + framed)."""
        import json
        envelope = {
            "task_id": str(task.id),
            "type": task.command,
            "payload": task.args_json or {},
            "level": task.level,
        }
        try:
            plaintext = json.dumps(envelope).encode("utf-8")
            encrypted = encrypt_aes_gcm(self.encryption_key_b64, plaintext)
            async with tc.write_lock:
                tc.writer.write(self._frame(encrypted.encode("ascii")))
                await tc.writer.drain()
        except (ConnectionResetError, BrokenPipeError):
            # Connection died mid-send — mark session dead
            async with self._lock:
                self._conns.pop(tc.session_id, None)
            try:
                await self.manager.mark_session_dead(tc.session_id)
            except Exception:
                pass
        except Exception as e:
            logger.warning("TCP task send failed: session=%s %s", tc.session_id, e)

    # ---------- Frame helpers ----------

    @staticmethod
    def _frame(payload: bytes) -> bytes:
        """Frame a payload: [4-byte length BE][payload]."""
        return struct.pack(">I", len(payload)) + payload

    async def _write_frame(self, writer: asyncio.StreamWriter, payload: bytes) -> None:
        """Write a framed payload to a writer."""
        writer.write(self._frame(payload))
        await writer.drain()

    async def _read_frame(self, reader: asyncio.StreamReader) -> bytes:
        """Read a framed payload from a reader.

        Returns the payload bytes (without the 4-byte length prefix).
        """
        length_bytes = await reader.readexactly(4)
        length = struct.unpack(">I", length_bytes)[0]
        if length > TCP_MAX_FRAME_BYTES:
            raise ValueError(f"Frame too large: {length} bytes (max {TCP_MAX_FRAME_BYTES})")
        return await reader.readexactly(length)

    async def _touch_session(self, session_id: str) -> None:
        """Refresh session last_seen_at (heartbeat)."""
        try:
            from app.db.models.c2 import C2Session
            from sqlalchemy import update
            from app.c2.types import SessionStatus
            stmt = (
                update(C2Session)
                .where(C2Session.id == uuid.UUID(session_id))
                .values(last_seen_at=datetime.now(UTC), status=SessionStatus.ACTIVE.value)
            )
            await self.manager.session.execute(stmt)
            await self.manager.session.flush()
        except Exception as e:
            logger.debug("Touch session failed: %s", e)


# ---------------------------------------------------------------------------
# Per-connection state
# ---------------------------------------------------------------------------

class _TCPConn:
    """Per-TCP-connection state (port from CyberStrikeAI tcpReverseConn)."""

    def __init__(
        self,
        session_id: str,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ):
        self.session_id = session_id
        self.reader = reader
        self.writer = writer
        self.write_lock = asyncio.Lock()  # serializes task writes


# ---------------------------------------------------------------------------
# Convenience: start TCP listener as a background asyncio task
# ---------------------------------------------------------------------------

async def start_tcp_listener(
    manager: C2Manager,
    bind_host: str = "127.0.0.1",
    bind_port: int = 8444,
    config: ListenerConfig | None = None,
    listener_name: str = "tcp_listener",
) -> tuple[TCPBeaconListener, dict[str, Any]]:
    """Create + start a TCP beacon listener.

    Returns (listener_instance, listener_metadata).
    """
    from app.c2.manager import CreateListenerInput

    config = config or ListenerConfig()
    config.apply_defaults()

    inp_meta = await manager.create_listener(
        inp=CreateListenerInput(
            name=listener_name,
            scan_id=manager.scan_id,
            listener_type=ListenerType.TCP_REVERSE.value,
            bind_host=bind_host,
            bind_port=bind_port,
            config=config,
        )
    )

    listener = TCPBeaconListener(
        manager=manager,
        config=config,
        encryption_key_b64=inp_meta["encryption_key_b64"],
        implant_token_b64=inp_meta["implant_token_b64"],
        listener_id=inp_meta["listener_id"],
        bind_host=inp_meta["bind_host"],
        bind_port=inp_meta["bind_port"],
    )

    await listener.start()
    return listener, inp_meta