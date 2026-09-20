"""
VAPT-AI C2 Subsystem (W14).

Python port of CyberStrikeAI internal/c2/ (Apache 2.0).

Modules:
    crypto          — AES-256-GCM envelope (port from CyberStrikeAI crypto.go)
    types           — Enums + dataclasses (ListenerType, SessionStatus,
                      TaskStatus, TaskType, BeaconType, ListenerConfig,
                      ImplantCheckInRequest/Response, TaskResultRequest)
    manager         — C2Manager (singleton-per-scan orchestrator) with
                      EventBus hooks for SSE emission
    listener_http   — HTTPBeaconListener (aiohttp, AES-GCM beacon protocol)
    sync            — D28 unified shell interface (MSF + sqlmap → C2Session)
    beacon_sample   — Sample Python beacon (reference for beacon protocol)

Reference: CyberStrikeAI internal/c2/ (Apache 2.0). D28 unified C2 per
master plan §3.2 difference #3.
"""
from app.c2.crypto import (
    generate_aes_key,
    generate_implant_token,
    encrypt_aes_gcm,
    decrypt_aes_gcm,
    encrypt_aes_gcm_with_aad,
    decrypt_aes_gcm_with_aad,
    CryptoError,
    KeyDecodeError,
    CiphertextError,
)
from app.c2.types import (
    ListenerType,
    SessionStatus,
    TaskStatus,
    TaskType,
    BeaconType,
    ListenerConfig,
    ImplantCheckInRequest,
    ImplantCheckInResponse,
    TaskResultRequest,
    TaskPollResponse,
)
from app.c2.manager import (
    C2Manager,
    C2Hooks,
    CreateListenerInput,
    EnqueueTaskInput,
    InvalidInputError,
    SessionNotFoundError,
    SessionInactiveError,
    ListenerNotFoundError,
    C2Error,
    create_c2_manager,
)
from app.c2.listener_http import (
    HTTPBeaconListener,
    start_http_listener,
)
from app.c2.listener_tcp import (
    TCPBeaconListener,
    start_tcp_listener,
)
from app.c2.listener_websocket import (
    WebSocketBeaconListener,
    start_websocket_listener,
)
from app.c2.payload_builder import (
    PayloadBuilder,
    PayloadBuilderInput,
    BuildResult,
    OnelinerKind,
    build_payload,
)
from app.c2.sync import (
    sync_msf_session,
    sync_sqlmap_webshell,
    list_unified_sessions,
    get_session_stats,
)

__all__ = [
    # Crypto
    "generate_aes_key", "generate_implant_token",
    "encrypt_aes_gcm", "decrypt_aes_gcm",
    "encrypt_aes_gcm_with_aad", "decrypt_aes_gcm_with_aad",
    "CryptoError", "KeyDecodeError", "CiphertextError",
    # Types
    "ListenerType", "SessionStatus", "TaskStatus", "TaskType", "BeaconType",
    "ListenerConfig",
    "ImplantCheckInRequest", "ImplantCheckInResponse",
    "TaskResultRequest", "TaskPollResponse",
    # Manager
    "C2Manager", "C2Hooks", "CreateListenerInput", "EnqueueTaskInput",
    "InvalidInputError", "SessionNotFoundError", "SessionInactiveError",
    "ListenerNotFoundError", "C2Error", "create_c2_manager",
    # Listeners
    "HTTPBeaconListener", "start_http_listener",
    "TCPBeaconListener", "start_tcp_listener",
    "WebSocketBeaconListener", "start_websocket_listener",
    # Payload builder
    "PayloadBuilder", "PayloadBuilderInput", "BuildResult",
    "OnelinerKind", "build_payload",
    # Sync
    "sync_msf_session", "sync_sqlmap_webshell",
    "list_unified_sessions", "get_session_stats",
]