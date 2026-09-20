"""
VAPT-AI C2 WebSocket Beacon Listener (W15-S3).

Python port of CyberStrikeAI internal/c2/listener_websocket.go.

WebSocket frame protocol (port from CyberStrikeAI listener_websocket.go):
    client → server: {"type": "checkin"|"result", "data": <encrypted_payload>}
    server → client: {"type": "task"|"sleep", "data": <encrypted_payload>}

All `data` fields are AES-GCM encrypted base64 strings (same envelope as
HTTP listener — app.c2.crypto).

Advantages over HTTP beacon (per CyberStrikeAI):
    - Long-lived connection (no polling, low latency)
    - New tasks pushed instantly to beacon
    - Suitable for interactive sessions (streaming output)

Authentication: X-Implant-Token header (or Authorization: Bearer) at WS
upgrade time. Unauthenticated upgrades → close immediately.

Reference: CyberStrikeAI internal/c2/listener_websocket.go (Apache 2.0).
"""
from __future__ import annotations

import asyncio
import json
import logging
import uuid
from datetime import datetime, UTC
from typing import Any

from aiohttp import web, WSMsgType

# WebSocketResponse is accessed as web.WebSocketResponse in aiohttp
WebSocketResponse = web.WebSocketResponse

from app.c2.types import (
    ListenerType, ImplantCheckInRequest, ImplantCheckInResponse,
    TaskResultRequest, ListenerConfig,
)
from app.c2.crypto import decrypt_aes_gcm, encrypt_aes_gcm, CryptoError
from app.c2.manager import C2Manager, InvalidInputError

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WS_DEFAULT_PATH = "/ws"
WS_DISPATCH_INTERVAL = 0.5  # 500ms task push poll
WS_MSG_MAX_SIZE = 16 * 1024 * 1024  # 16 MB


# ---------------------------------------------------------------------------
# WebSocket Beacon Listener
# ---------------------------------------------------------------------------

class WebSocketBeaconListener:
    """WebSocket beacon listener (port from CyberStrikeAI listener_websocket.go).

    Spawns an aiohttp.web server with a single /ws endpoint. Beacon upgrades
    to WebSocket, sends checkin frame, then receives tasks + sends results
    over the same connection.

    Lifecycle:
        listener = WebSocketBeaconListener(manager, config, key, token, ...)
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
        bind_port: int = 8445,
        ws_path: str | None = None,
        use_tls: bool = False,
        tls_cert_path: str | None = None,
        tls_key_path: str | None = None,
    ):
        self.manager = manager
        self.config = config
        self.config.apply_defaults()
        self.encryption_key_b64 = encryption_key_b64
        self.implant_token_b64 = implant_token_b64
        self.listener_id = listener_id
        self.bind_host = bind_host
        self.bind_port = bind_port
        self.ws_path = ws_path or WS_DEFAULT_PATH
        self.use_tls = use_tls
        self.tls_cert_path = tls_cert_path
        self.tls_key_path = tls_key_path

        self._runner: web.AppRunner | None = None
        self._site: web.TCPSite | None = None
        self._conns: dict[str, _WSConn] = {}    # session_id → connection state
        self._lock = asyncio.Lock()
        self._stopped = False
        self._dispatch_task: asyncio.Task | None = None

    @property
    def type(self) -> str:
        return ListenerType.WEBSOCKET.value

    # ---------- Lifecycle ----------

    async def start(self) -> None:
        """Start the WebSocket server."""
        app = web.Application(client_max_size=WS_MSG_MAX_SIZE)
        app.router.add_get(self.ws_path, self._handle_ws_upgrade)
        # Disguise fallback for non-WS requests
        app.router.add_route("*", "/{tail:.*}", self._disguised_reject)

        self._runner = web.AppRunner(app)
        await self._runner.setup()

        ssl_context = None
        if self.use_tls:
            ssl_context = self._build_ssl_context()

        self._site = web.TCPSite(
            self._runner,
            self.bind_host,
            self.bind_port,
            ssl_context=ssl_context,
        )
        await self._site.start()

        # Start task dispatcher loop
        self._dispatch_task = asyncio.create_task(self._task_dispatcher_loop())

        await self.manager.register_running_listener(self.listener_id, self)

        logger.info(
            "C2 WebSocket listener started: id=%s bind=%s:%d ws_path=%s tls=%s",
            self.listener_id, self.bind_host, self.bind_port, self.ws_path, self.use_tls,
        )

    async def stop(self) -> None:
        """Gracefully stop the listener + close all WS connections."""
        if self._stopped:
            return
        self._stopped = True

        if self._dispatch_task is not None:
            self._dispatch_task.cancel()
            try:
                await self._dispatch_task
            except asyncio.CancelledError:
                pass

        # Close all active WS connections
        async with self._lock:
            conns = list(self._conns.values())
            self._conns.clear()
        for wc in conns:
            try:
                await wc.ws.close()
            except Exception:
                pass

        if self._site is not None:
            await self._site.stop()
        if self._runner is not None:
            await self._runner.cleanup()

        await self.manager.unregister_running_listener(self.listener_id)
        logger.info("C2 WebSocket listener stopped: id=%s", self.listener_id)

    def _build_ssl_context(self):
        """Build SSLContext for WSS (same logic as HTTP listener)."""
        import ssl
        from cryptography import x509
        from cryptography.x509.oid import NameOID
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import ec
        from datetime import timedelta

        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)

        if self.tls_cert_path and self.tls_key_path:
            ctx.load_cert_chain(self.tls_cert_path, self.tls_key_path)
            return ctx

        if not self.config.tls_auto_self_sign:
            raise RuntimeError("TLS requested but no cert + tls_auto_self_sign=False")

        key = ec.generate_private_key(ec.SECP256R1())
        subject = issuer = x509.Name([
            x509.NameAttribute(NameOID.COMMON_NAME, "localhost"),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "VAPT-AI"),
        ])
        cert = (
            x509.CertificateBuilder()
            .subject_name(subject)
            .issuer_name(issuer)
            .public_key(key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(datetime.now(UTC))
            .not_valid_after(datetime.now(UTC) + timedelta(days=365))
            .add_extension(
                x509.SubjectAlternativeName([x509.DNSName("localhost")]),
                critical=False,
            )
            .sign(key, hashes.SHA256())
        )
        cert_pem = cert.public_bytes(serialization.Encoding.PEM).decode("ascii")
        key_pem = key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        ).decode("ascii")
        ctx.load_cert_chain(cert_pem, key_pem)
        return ctx

    # ---------- Auth ----------

    def _check_implant_token(self, request: web.Request) -> bool:
        """Validate X-Implant-Token or Authorization: Bearer header."""
        import hmac
        # Try X-Implant-Token first (CyberStrikeAI convention)
        token = request.headers.get("X-Implant-Token", "")
        if token and hmac.compare_digest(token, self.implant_token_b64):
            return True
        # Fallback: Authorization: Bearer
        auth = request.headers.get("Authorization", "")
        if auth.startswith("Bearer "):
            bearer_token = auth[7:].strip()
            if hmac.compare_digest(bearer_token, self.implant_token_b64):
                return True
        return False

    async def _disguised_reject(self, request: web.Request) -> web.Response:
        """404 for non-WS requests."""
        return web.Response(
            status=404,
            text="<html><body><h1>Not Found</h1></body></html>",
            content_type="text/html",
        )

    # ---------- WebSocket handler ----------

    async def _handle_ws_upgrade(self, request: web.Request) -> WebSocketResponse:
        """Handle WebSocket upgrade request.

        Validates implant token → upgrades to WS → reads checkin frame →
        registers session → enters main loop (read results + push tasks).
        """
        if not self._check_implant_token(request):
            return web.Response(status=403, text="forbidden")

        ws = web.WebSocketResponse(
            max_msg_size=WS_MSG_MAX_SIZE,
            heartbeat=30,  # server-side ping every 30s
        )
        await ws.prepare(request)

        remote = request.remote or "unknown"

        try:
            # Read first message — must be checkin
            first_msg = await ws.receive(timeout=10.0)
            if first_msg.type != WSMsgType.TEXT:
                await ws.close(code=4001, message="expected text checkin")
                return ws

            try:
                frame = json.loads(first_msg.data)
                if frame.get("type") != "checkin":
                    await ws.close(code=4002, message="expected checkin type")
                    return ws

                # Decrypt checkin payload
                enc_b64 = frame.get("data", "")
                plaintext = decrypt_aes_gcm(self.encryption_key_b64, enc_b64)
                req_data = json.loads(plaintext)
                req = ImplantCheckInRequest.from_dict(req_data)
            except (json.JSONDecodeError, CryptoError, KeyError) as e:
                logger.warning("WS checkin parse failed: %s", e)
                await ws.close(code=4003, message=f"checkin parse failed: {e}")
                return ws

            # Populate remote IP
            if not req.internal_ip:
                req.internal_ip = remote

            # Persist session
            try:
                session = await self.manager.ingest_checkin(
                    listener_id=self.listener_id,
                    listener_type=self.type,
                    req=req,
                    encryption_key_b64=self.encryption_key_b64,
                    implant_token_b64=self.implant_token_b64,
                )
            except InvalidInputError as e:
                logger.warning("WS checkin invalid: %s", e)
                await ws.close(code=4004, message=f"checkin invalid: {e}")
                return ws

            # Send checkin response
            response = ImplantCheckInResponse(
                session_id=str(session.id),
                next_sleep=req.sleep_seconds or self.config.default_sleep,
                next_jitter=req.jitter_percent,
                has_tasks=False,
                server_time=int(datetime.now(UTC).timestamp()),
            )
            response_enc = encrypt_aes_gcm(
                self.encryption_key_b64,
                json.dumps(response.to_dict()).encode("utf-8"),
            )
            await ws.send_str(json.dumps({"type": "checkin_ack", "data": response_enc}))

            # Register connection
            wc = _WSConn(session_id=str(session.id), ws=ws)
            async with self._lock:
                old = self._conns.get(str(session.id))
                if old is not None:
                    try:
                        await old.ws.close()
                    except Exception:
                        pass
                self._conns[str(session.id)] = wc

            logger.info(
                "WS beacon connected: scan=%s session=%s remote=%s",
                self.manager.scan_id, session.id, remote,
            )

            # Main loop — read result frames
            async for msg in ws:
                if msg.type == WSMsgType.TEXT:
                    await self._handle_ws_message(wc, msg.data)
                elif msg.type == WSMsgType.ERROR:
                    logger.warning("WS error: %s", ws.exception())
                    break
                elif msg.type == WSMsgType.CLOSE:
                    break

        except asyncio.TimeoutError:
            logger.warning("WS checkin timeout from %s", remote)
        except Exception as e:
            logger.warning("WS handler error: %s", e)
        finally:
            async with self._lock:
                self._conns.pop(wc.session_id, None) if 'wc' in locals() else None
            try:
                if 'session' in locals():
                    await self.manager.mark_session_dead(str(session.id))
            except Exception:
                pass

        return ws

    async def _handle_ws_message(self, wc: "_WSConn", raw: str) -> None:
        """Handle a single WS message from beacon."""
        try:
            frame = json.loads(raw)
            msg_type = frame.get("type")
            enc_data = frame.get("data", "")

            if msg_type == "result":
                # Decrypt + parse task result
                try:
                    plaintext = decrypt_aes_gcm(self.encryption_key_b64, enc_data)
                    result_data = json.loads(plaintext)
                    result = TaskResultRequest.from_dict(result_data)
                    await self.manager.submit_task_result(result.task_id, result)
                except (CryptoError, json.JSONDecodeError, KeyError) as e:
                    logger.warning("WS result parse failed: %s", e)

            elif msg_type == "heartbeat":
                # Just refresh last_seen
                await self._touch_session(wc.session_id)

            else:
                logger.debug("Unknown WS message type: %s", msg_type)

        except json.JSONDecodeError as e:
            logger.warning("WS message JSON parse failed: %s", e)

    # ---------- Task dispatcher ----------

    async def _task_dispatcher_loop(self) -> None:
        """Background task: poll manager for queued tasks + push to WS beacons."""
        while not self._stopped:
            try:
                async with self._lock:
                    snapshot = list(self._conns.values())

                for wc in snapshot:
                    try:
                        tasks = await self.manager.pop_tasks_for_beacon(
                            wc.session_id, limit=5,
                        )
                    except Exception as e:
                        logger.warning("WS task pop failed: %s", e)
                        continue

                    for task in tasks:
                        asyncio.create_task(self._push_task(wc, task))
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning("WS dispatch loop error: %s", e)

            await asyncio.sleep(WS_DISPATCH_INTERVAL)

    async def _push_task(self, wc: "_WSConn", task: Any) -> None:
        """Push a task to a WS beacon."""
        envelope = {
            "task_id": str(task.id),
            "type": task.command,
            "payload": task.args_json or {},
            "level": task.level,
        }
        try:
            plaintext = json.dumps(envelope).encode("utf-8")
            encrypted = encrypt_aes_gcm(self.encryption_key_b64, plaintext)
            async with wc.write_lock:
                await wc.ws.send_str(
                    json.dumps({"type": "task", "data": encrypted})
                )
        except Exception as e:
            logger.warning("WS task push failed: session=%s %s", wc.session_id, e)

    async def _touch_session(self, session_id: str) -> None:
        """Refresh session last_seen_at."""
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
            logger.debug("WS touch session failed: %s", e)


# ---------------------------------------------------------------------------
# Per-connection state
# ---------------------------------------------------------------------------

class _WSConn:
    """Per-WebSocket-connection state (port from CyberStrikeAI wsConn)."""

    def __init__(self, session_id: str, ws: WebSocketResponse):
        self.session_id = session_id
        self.ws = ws
        self.write_lock = asyncio.Lock()  # serialize writes


# ---------------------------------------------------------------------------
# Convenience: start WS listener
# ---------------------------------------------------------------------------

async def start_websocket_listener(
    manager: C2Manager,
    bind_host: str = "127.0.0.1",
    bind_port: int = 8445,
    use_tls: bool = False,
    ws_path: str | None = None,
    config: ListenerConfig | None = None,
    listener_name: str = "ws_listener",
) -> tuple[WebSocketBeaconListener, dict[str, Any]]:
    """Create + start a WebSocket beacon listener."""
    from app.c2.manager import CreateListenerInput

    config = config or ListenerConfig()
    config.apply_defaults()

    inp_meta = await manager.create_listener(
        inp=CreateListenerInput(
            name=listener_name,
            scan_id=manager.scan_id,
            listener_type=ListenerType.WEBSOCKET.value,
            bind_host=bind_host,
            bind_port=bind_port,
            config=config,
        )
    )

    listener = WebSocketBeaconListener(
        manager=manager,
        config=config,
        encryption_key_b64=inp_meta["encryption_key_b64"],
        implant_token_b64=inp_meta["implant_token_b64"],
        listener_id=inp_meta["listener_id"],
        bind_host=inp_meta["bind_host"],
        bind_port=inp_meta["bind_port"],
        ws_path=ws_path or WS_DEFAULT_PATH,
        use_tls=use_tls,
        tls_cert_path=config.tls_cert_path,
        tls_key_path=config.tls_key_path,
    )

    await listener.start()
    return listener, inp_meta