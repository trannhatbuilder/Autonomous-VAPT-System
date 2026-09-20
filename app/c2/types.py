"""
VAPT-AI C2 Types — enums + dataclasses (W14-S3).

Python port of CyberStrikeAI internal/c2/types.go.

Maps 1:1 to the C2 DB schema defined in app/db/models/c2.py (W1) and
shared with the HTTP listener (W14-S5) and beacon protocol (W14-S7).

Reference: CyberStrikeAI internal/c2/types.go (Apache 2.0).
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


# ---------------------------------------------------------------------------
# Enums — string values match DB columns (no implicit conversion)
# ---------------------------------------------------------------------------

class ListenerType(str, Enum):
    """C2 listener transport type. Matches c2_sessions.listener_type column."""
    TCP_REVERSE = "tcp_reverse"
    HTTP_BEACON = "http"
    HTTPS_BEACON = "https"
    WEBSOCKET = "websocket"

    @classmethod
    def all(cls) -> list["ListenerType"]:
        return [cls.TCP_REVERSE, cls.HTTP_BEACON, cls.HTTPS_BEACON, cls.WEBSOCKET]

    @classmethod
    def is_valid(cls, value: str) -> bool:
        try:
            cls(value.lower().strip())
            return True
        except ValueError:
            return False


class SessionStatus(str, Enum):
    """C2 session lifecycle status. Matches c2_sessions.status column."""
    ACTIVE = "active"
    SLEEPING = "sleeping"
    DEAD = "dead"
    KILLED = "killed"


class TaskStatus(str, Enum):
    """C2 task lifecycle status. Matches c2_tasks.status column."""
    QUEUED = "queued"
    SENT = "sent"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskType(str, Enum):
    """Task type — negotiated between server + beacon (W14-S7).

    Matches CyberStrikeAI types.go TaskType constants.
    L1-L5 classification per master plan §5 (HITL gate for L3+).
    """
    # General tasks (L1-L2: read-only or low-impact)
    EXEC = "exec"                  # Execute arbitrary command (shell -c)
    SHELL = "shell"                # Interactive command (preserve cwd)
    PWD = "pwd"                    # Current directory (L1 read-only)
    CD = "cd"                      # Change directory (L1 read-only)
    LS = "ls"                      # List directory (L1 read-only)
    PS = "ps"                      # List processes (L1 read-only)
    KILL_PROC = "kill_proc"        # Kill process (L3 destructive)
    UPLOAD = "upload"              # Push file to target (L3 file write)
    DOWNLOAD = "download"          # Pull file from target (L2 file read)
    SCREENSHOT = "screenshot"      # Screenshot (L2 read-only)
    SLEEP = "sleep"                # Adjust heartbeat (L1)
    EXIT = "exit"                  # Beacon exit (L2)
    SELF_DELETE = "self_delete"    # Exit + delete binary (L5 destructive)

    # Advanced tasks (L4-L5)
    PORT_FWD = "port_fwd"
    SOCKS_START = "socks_start"
    SOCKS_STOP = "socks_stop"
    LOAD_ASSEMBLY = "load_assembly"
    PERSIST = "persist"            # Persistence mechanism (L5)

    @classmethod
    def all(cls) -> list["TaskType"]:
        return list(cls)

    @classmethod
    def is_dangerous(cls, task_type: "TaskType | str") -> bool:
        """Check if a task type requires HITL gate (L3+ per master plan D18).

        Ported from CyberStrikeAI IsDangerousTaskType().
        """
        if isinstance(task_type, str):
            try:
                task_type = cls(task_type)
            except ValueError:
                return False
        return task_type in {
            cls.KILL_PROC, cls.UPLOAD, cls.SELF_DELETE,
            cls.PORT_FWD, cls.SOCKS_START, cls.LOAD_ASSEMBLY, cls.PERSIST,
        }

    @classmethod
    def hitl_level(cls, task_type: "TaskType | str") -> int:
        """Return HITL level (1-5) for a task type.

        L1 = read-only, L2 = low-impact, L3 = destructive, L4 = advanced,
        L5 = persistence/destructive+cleanup.
        """
        if isinstance(task_type, str):
            try:
                task_type = cls(task_type)
            except ValueError:
                return 5  # Unknown → most restrictive
        levels = {
            cls.PWD: 1, cls.CD: 1, cls.LS: 1, cls.PS: 1, cls.SLEEP: 1,
            cls.DOWNLOAD: 2, cls.SCREENSHOT: 2, cls.EXIT: 2, cls.EXEC: 2,
            cls.SHELL: 2,
            cls.KILL_PROC: 3, cls.UPLOAD: 3,
            cls.PORT_FWD: 4, cls.SOCKS_START: 4, cls.SOCKS_STOP: 4,
            cls.LOAD_ASSEMBLY: 4,
            cls.SELF_DELETE: 5, cls.PERSIST: 5,
        }
        return levels.get(task_type, 5)


class BeaconType(str, Enum):
    """D28 unified C2: beacon source type. Matches c2_sessions.beacon_type column."""
    PYTHON_BEACON = "python_beacon"
    MSF_METERPRETER = "msf_meterpreter"
    MSF_SHELL = "msf_shell"
    SQLMAP_WEBSHELL = "sqlmap_webshell"
    CUSTOM = "custom"


# ---------------------------------------------------------------------------
# ListenerConfig — decoded c2_listeners.config_json (port from types.go)
# ---------------------------------------------------------------------------

@dataclass
class ListenerConfig:
    """Listener runtime configuration.

    Stored as JSONB in c2_sessions.metadata_json (or c2_listeners.config_json
    once W15 adds a separate listeners table).
    """
    # HTTP/HTTPS beacon endpoint paths
    beacon_check_in_path: str = "/checkin"
    beacon_tasks_path: str = "/tasks"
    beacon_result_path: str = "/result"
    beacon_upload_path: str = "/upload"
    beacon_file_path: str = "/file/"

    # HTTPS-specific
    tls_cert_path: str | None = None
    tls_key_path: str | None = None
    tls_auto_self_sign: bool = True

    # Beacon defaults (initial values, beacon can override at check-in)
    default_sleep: int = 5         # seconds
    default_jitter: int = 0        # 0-100 percent

    # OPSEC: optional command deny regex (block dangerous commands at server)
    command_deny_regex: list[str] = field(default_factory=list)

    # Max concurrent tasks per session (0 = unlimited)
    max_concurrent_tasks: int = 0

    # Callback host (implant-side host, separate from bind_host for NAT/ECS)
    callback_host: str | None = None

    # Allow legacy unencrypted shell on tcp_reverse (default false)
    allow_legacy_shell: bool = False

    def apply_defaults(self) -> None:
        """Fill missing fields with default values (mutates self)."""
        if not self.beacon_check_in_path.strip():
            self.beacon_check_in_path = "/checkin"
        if not self.beacon_tasks_path.strip():
            self.beacon_tasks_path = "/tasks"
        if not self.beacon_result_path.strip():
            self.beacon_result_path = "/result"
        if not self.beacon_upload_path.strip():
            self.beacon_upload_path = "/upload"
        if not self.beacon_file_path.strip():
            self.beacon_file_path = "/file/"
        if self.default_sleep <= 0:
            self.default_sleep = 5
        if self.default_jitter < 0:
            self.default_jitter = 0
        if self.default_jitter > 100:
            self.default_jitter = 100

    def to_dict(self) -> dict[str, Any]:
        return {
            "beacon_check_in_path": self.beacon_check_in_path,
            "beacon_tasks_path": self.beacon_tasks_path,
            "beacon_result_path": self.beacon_result_path,
            "beacon_upload_path": self.beacon_upload_path,
            "beacon_file_path": self.beacon_file_path,
            "tls_cert_path": self.tls_cert_path,
            "tls_key_path": self.tls_key_path,
            "tls_auto_self_sign": self.tls_auto_self_sign,
            "default_sleep": self.default_sleep,
            "default_jitter": self.default_jitter,
            "command_deny_regex": self.command_deny_regex,
            "max_concurrent_tasks": self.max_concurrent_tasks,
            "callback_host": self.callback_host,
            "allow_legacy_shell": self.allow_legacy_shell,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ListenerConfig":
        cfg = cls()
        if "beacon_check_in_path" in d:
            cfg.beacon_check_in_path = d["beacon_check_in_path"]
        if "beacon_tasks_path" in d:
            cfg.beacon_tasks_path = d["beacon_tasks_path"]
        if "beacon_result_path" in d:
            cfg.beacon_result_path = d["beacon_result_path"]
        if "beacon_upload_path" in d:
            cfg.beacon_upload_path = d["beacon_upload_path"]
        if "beacon_file_path" in d:
            cfg.beacon_file_path = d["beacon_file_path"]
        if "tls_cert_path" in d:
            cfg.tls_cert_path = d["tls_cert_path"]
        if "tls_key_path" in d:
            cfg.tls_key_path = d["tls_key_path"]
        if "tls_auto_self_sign" in d:
            cfg.tls_auto_self_sign = bool(d["tls_auto_self_sign"])
        if "default_sleep" in d:
            cfg.default_sleep = int(d["default_sleep"])
        if "default_jitter" in d:
            cfg.default_jitter = int(d["default_jitter"])
        if "command_deny_regex" in d:
            cfg.command_deny_regex = list(d["command_deny_regex"])
        if "max_concurrent_tasks" in d:
            cfg.max_concurrent_tasks = int(d["max_concurrent_tasks"])
        if "callback_host" in d:
            cfg.callback_host = d["callback_host"]
        if "allow_legacy_shell" in d:
            cfg.allow_legacy_shell = bool(d["allow_legacy_shell"])
        return cfg


# ---------------------------------------------------------------------------
# Beacon protocol messages (port from types.go)
# ---------------------------------------------------------------------------

@dataclass
class ImplantCheckInRequest:
    """Beacon → server check-in request body (after AES-GCM decrypt).

    Matches CyberStrikeAI ImplantCheckInRequest struct.
    """
    implant_uuid: str
    hostname: str
    username: str
    os: str
    arch: str
    pid: int
    process_name: str
    is_admin: bool
    internal_ip: str
    user_agent: str = ""
    sleep_seconds: int = 5
    jitter_percent: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "uuid": self.implant_uuid,
            "hostname": self.hostname,
            "username": self.username,
            "os": self.os,
            "arch": self.arch,
            "pid": self.pid,
            "process_name": self.process_name,
            "is_admin": self.is_admin,
            "internal_ip": self.internal_ip,
            "user_agent": self.user_agent,
            "sleep_seconds": self.sleep_seconds,
            "jitter_percent": self.jitter_percent,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ImplantCheckInRequest":
        return cls(
            implant_uuid=d.get("uuid", ""),
            hostname=d.get("hostname", ""),
            username=d.get("username", ""),
            os=d.get("os", ""),
            arch=d.get("arch", ""),
            pid=int(d.get("pid", 0)),
            process_name=d.get("process_name", ""),
            is_admin=bool(d.get("is_admin", False)),
            internal_ip=d.get("internal_ip", ""),
            user_agent=d.get("user_agent", ""),
            sleep_seconds=int(d.get("sleep_seconds", 5)),
            jitter_percent=int(d.get("jitter_percent", 0)),
            metadata=d.get("metadata", {}),
        )


@dataclass
class ImplantCheckInResponse:
    """Server → beacon check-in response body (before AES-GCM encrypt)."""
    session_id: str                # server-side session UUID (str form)
    next_sleep: int = 5
    next_jitter: int = 0
    has_tasks: bool = False
    server_time: int = 0           # Unix epoch seconds

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "next_sleep": self.next_sleep,
            "next_jitter": self.next_jitter,
            "has_tasks": self.has_tasks,
            "server_time": self.server_time,
        }


@dataclass
class TaskPollResponse:
    """Server → beacon task poll response (list of pending tasks)."""
    tasks: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"tasks": self.tasks}


@dataclass
class TaskResultRequest:
    """Beacon → server task result submission (after AES-GCM decrypt)."""
    task_id: str
    result_text: str = ""
    exit_code: int = 0
    error: str = ""
    result_blob_path: str | None = None
    duration_ms: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "result_text": self.result_text,
            "exit_code": self.exit_code,
            "error": self.error,
            "result_blob_path": self.result_blob_path,
            "duration_ms": self.duration_ms,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TaskResultRequest":
        return cls(
            task_id=d.get("task_id", ""),
            result_text=d.get("result_text", ""),
            exit_code=int(d.get("exit_code", 0)),
            error=d.get("error", ""),
            result_blob_path=d.get("result_blob_path"),
            duration_ms=int(d.get("duration_ms", 0)),
        )