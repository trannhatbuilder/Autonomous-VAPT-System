"""
VAPT-AI C2 Cryptography — AES-256-GCM envelope (W14-S2).

Python port of CyberStrikeAI internal/c2/crypto.go.

Protocol (per CyberStrikeAI design):
    Format: base64( nonce(12) || ciphertext+tag(16) )
    - GCM AEAD tag (16 bytes) gives integrity + confidentiality
    - 96-bit nonce from os.urandom — collision probability < 2^-32 per 4B msgs
    - Per-listener 32-byte AES-256 key generated at listener creation
    - Beacon receives key at deploy time (hardcoded into beacon script)

Two variants:
    1. Plain AES-GCM        — used for task payloads, beacon checkins
    2. AES-GCM-with-AAD     — used when binding ciphertext to a context
       (e.g. session_id) to prevent cross-session replay attacks

Reference: CyberStrikeAI internal/c2/crypto.go (Apache 2.0).
"""
from __future__ import annotations

import base64
import os
from typing import Union

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

# Constants — match CyberStrikeAI crypto.go
AES_KEY_BYTES = 32          # AES-256
NONCE_BYTES = 12            # GCM standard nonce size (96-bit)
GCM_TAG_BYTES = 16          # GCM authentication tag
MIN_CIPHERTEXT_BYTES = NONCE_BYTES + GCM_TAG_BYTES

BytesLike = Union[bytes, bytearray, memoryview]


class CryptoError(Exception):
    """Base error for C2 crypto operations."""


class KeyDecodeError(CryptoError):
    """Raised when an AES key is not valid base64 or wrong length."""


class CiphertextError(CryptoError):
    """Raised when ciphertext is too short, malformed, or fails AEAD verification."""


# ---------------------------------------------------------------------------
# Key + token generation
# ---------------------------------------------------------------------------

def generate_aes_key() -> str:
    """Generate a random 32-byte AES-256 key, base64-encoded.

    Used at listener creation time. The key is shared with the beacon
    via deploy-time injection (env var / hardcoded into payload).

    Returns:
        Base64 (standard) string of 32 random bytes.
    """
    key = os.urandom(AES_KEY_BYTES)
    return base64.b64encode(key).decode("ascii")


def generate_implant_token() -> str:
    """Generate a 32-byte implant (beacon) auth token, base64url-encoded.

    Used as the `Authorization: Bearer <token>` header value for beacon
    HTTP requests. CyberStrikeAI uses RawURLEncoding (no padding) for
    compactness in HTTP headers.

    Returns:
        Base64url (no padding) string of 32 random bytes.
    """
    token = os.urandom(AES_KEY_BYTES)
    return base64.urlsafe_b64encode(token).rstrip(b"=").decode("ascii")


# ---------------------------------------------------------------------------
# Plain AES-GCM (no AAD)
# ---------------------------------------------------------------------------

def encrypt_aes_gcm(key_b64: str, plaintext: bytes) -> str:
    """Encrypt plaintext with AES-256-GCM, return base64(nonce || ciphertext+tag).

    Args:
        key_b64: Base64-encoded 32-byte AES-256 key
        plaintext: Plaintext bytes to encrypt

    Returns:
        Base64 string of (nonce[12] || ciphertext+tag[N+16])

    Raises:
        KeyDecodeError: if key is not valid base64 or wrong length
    """
    key = _decode_key(key_b64)
    nonce = os.urandom(NONCE_BYTES)
    aesgcm = AESGCM(key)
    ct = aesgcm.encrypt(nonce, bytes(plaintext), None)
    out = nonce + ct
    return base64.b64encode(out).decode("ascii")


def decrypt_aes_gcm(key_b64: str, enc_b64: str) -> bytes:
    """Decrypt base64(nonce || ciphertext+tag) with AES-256-GCM.

    Args:
        key_b64: Base64-encoded 32-byte AES-256 key
        enc_b64: Base64-encoded ciphertext envelope

    Returns:
        Plaintext bytes

    Raises:
        KeyDecodeError: if key is malformed
        CiphertextError: if ciphertext is too short, base64-invalid, or
                        fails AEAD verification (key mismatch / tampered)
    """
    key = _decode_key(key_b64)
    try:
        raw = base64.b64decode(enc_b64, validate=True)
    except (ValueError, base64.binascii.Error) as e:
        raise CiphertextError("ciphertext base64 invalid") from e

    if len(raw) < MIN_CIPHERTEXT_BYTES:
        raise CiphertextError(
            f"ciphertext too short: {len(raw)} bytes "
            f"(need ≥ {MIN_CIPHERTEXT_BYTES})"
        )

    nonce = raw[:NONCE_BYTES]
    ct = raw[NONCE_BYTES:]
    aesgcm = AESGCM(key)
    try:
        return aesgcm.decrypt(nonce, ct, None)
    except InvalidTag as e:
        raise CiphertextError(
            "aead open failed (key mismatch or tampered)"
        ) from e


# ---------------------------------------------------------------------------
# AES-GCM with AAD (Additional Authenticated Data)
# ---------------------------------------------------------------------------

def encrypt_aes_gcm_with_aad(
    key_b64: str, plaintext: bytes, aad: bytes,
) -> str:
    """Encrypt with AAD bound to context (e.g. session_id).

    Prevents cross-session replay: ciphertext from session A cannot be
    fed to session B because the AAD (session_id) won't match.

    Args:
        key_b64: Base64-encoded 32-byte AES-256 key
        plaintext: Plaintext bytes
        aad: Additional Authenticated Data (bound to ciphertext, not encrypted)

    Returns:
        Base64 string of (nonce[12] || ciphertext+tag[N+16])
    """
    key = _decode_key(key_b64)
    nonce = os.urandom(NONCE_BYTES)
    aesgcm = AESGCM(key)
    ct = aesgcm.encrypt(nonce, bytes(plaintext), bytes(aad))
    out = nonce + ct
    return base64.b64encode(out).decode("ascii")


def decrypt_aes_gcm_with_aad(
    key_b64: str, enc_b64: str, aad: bytes,
) -> bytes:
    """Decrypt with AAD verification.

    Args:
        key_b64: Base64-encoded 32-byte AES-256 key
        enc_b64: Base64-encoded ciphertext envelope
        aad: Additional Authenticated Data (must match what was used at encrypt time)

    Returns:
        Plaintext bytes

    Raises:
        KeyDecodeError: if key is malformed
        CiphertextError: if ciphertext is too short, base64-invalid, or
                        fails AEAD verification (key/tamper/AAD mismatch)
    """
    key = _decode_key(key_b64)
    try:
        raw = base64.b64decode(enc_b64, validate=True)
    except (ValueError, base64.binascii.Error) as e:
        raise CiphertextError("ciphertext base64 invalid") from e

    if len(raw) < MIN_CIPHERTEXT_BYTES:
        raise CiphertextError(
            f"ciphertext too short: {len(raw)} bytes "
            f"(need ≥ {MIN_CIPHERTEXT_BYTES})"
        )

    nonce = raw[:NONCE_BYTES]
    ct = raw[NONCE_BYTES:]
    aesgcm = AESGCM(key)
    try:
        return aesgcm.decrypt(nonce, ct, bytes(aad))
    except InvalidTag as e:
        raise CiphertextError(
            "aead open failed (key mismatch, tampered, or AAD mismatch)"
        ) from e


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _decode_key(key_b64: str) -> bytes:
    """Decode + validate a base64-encoded AES-256 key.

    Args:
        key_b64: Base64-encoded 32-byte key

    Returns:
        32-byte raw key

    Raises:
        KeyDecodeError: if base64 invalid or length != 32 bytes
    """
    try:
        key = base64.b64decode(key_b64, validate=True)
    except (ValueError, base64.binascii.Error) as e:
        raise KeyDecodeError("key base64 invalid") from e
    if len(key) != AES_KEY_BYTES:
        raise KeyDecodeError(
            f"key must be {AES_KEY_BYTES} bytes (AES-256), got {len(key)}"
        )
    return key