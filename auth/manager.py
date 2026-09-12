"""
VAPT-AI JWT authentication — single-user.

Hardcoded admin:
    email:    admin@vapt-ai.local
    password: from VAPT_AI_ADMIN_PASSWORD env var (.env)

Flow:
    1. POST /api/auth/login {email, password}
       → validates credentials, checks lockout, issues access + refresh tokens
       → returns {access_token, refresh_token, token_type, expires_in, user}

    2. GET /api/auth/me (Authorization: Bearer <access_token>)
       → returns current user info

    3. POST /api/auth/refresh {refresh_token}
       → rotates refresh token (old revoked, new issued)
       → returns new {access_token, refresh_token}

    4. POST /api/auth/logout (Authorization: Bearer <access_token>)
       → revokes refresh token chain

Rate limiting + lockout:
    - Max 5 failed login attempts per 15 minutes (configurable via settings)
    - After 5 failures, account locked for 15 minutes
    - Successful login resets failed_attempts counter

Token rotation (fixes EVVO refresh-token bug):
    - Each refresh token is single-use
    - On refresh, old token revoked + new one issued
    - Replaced_by_id tracks rotation chain for audit
    - 7-day TTL on refresh tokens (configurable)

Storage:
    - Users in vapt_users table (UUID PK, bcrypt password hash)
    - Refresh tokens in vapt_refresh_tokens table (SHA-256 hash, not raw token)
"""
from __future__ import annotations

import hashlib
import secrets
import uuid
from datetime import datetime, timedelta, UTC
from typing import Any

from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.db.models.user import RefreshToken, User


# ---------- Password hashing ----------

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(plain: str) -> str:
    """Hash a password using bcrypt."""
    return pwd_context.hash(plain)


def verify_password(plain: str, hashed: str) -> bool:
    """Verify a password against a bcrypt hash."""
    try:
        return pwd_context.verify(plain, hashed)
    except Exception:
        return False


# ---------- Token hashing (SHA-256 — for storage) ----------

def hash_token(token: str) -> str:
    """SHA-256 hash a refresh token for storage (never store raw token)."""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


# ---------- JWT token creation ----------

def create_access_token(user_id: uuid.UUID, email: str) -> tuple[str, datetime]:
    """Create a short-lived JWT access token.

    Returns (token, expires_at).
    """
    now = datetime.now(UTC)
    expires_at = now + timedelta(minutes=settings.jwt_access_token_ttl_minutes)
    payload: dict[str, Any] = {
        "sub": str(user_id),
        "email": email,
        "type": "access",
        "iat": int(now.timestamp()),
        "exp": int(expires_at.timestamp()),
    }
    token = jwt.encode(payload, settings.jwt_secret.get_secret_value(), algorithm=settings.jwt_algorithm)
    return token, expires_at


def create_refresh_token() -> tuple[str, datetime]:
    """Create a long-lived refresh token (opaque random string, not JWT).

    Returns (token, expires_at).
    """
    now = datetime.now(UTC)
    expires_at = now + timedelta(days=settings.jwt_refresh_token_ttl_days)
    # 48 bytes of entropy → 64 URL-safe chars
    token = secrets.token_urlsafe(48)
    return token, expires_at


# ---------- JWT token verification ----------

class AuthError(Exception):
    """Base auth error."""

    def __init__(self, message: str, status_code: int = 401):
        self.message = message
        self.status_code = status_code
        super().__init__(message)


class InvalidCredentialsError(AuthError):
    """Email/password mismatch."""


class AccountLockedError(AuthError):
    """Account locked due to too many failed attempts."""

    def __init__(self, locked_until: datetime):
        remaining = int((locked_until - datetime.now(UTC)).total_seconds() / 60)
        super().__init__(
            f"Account locked. Try again in {remaining} minute(s).",
            status_code=423,
        )
        self.locked_until = locked_until


class InvalidTokenError(AuthError):
    """Invalid or expired token."""


class TokenRevokedError(AuthError):
    """Refresh token has been revoked."""


def decode_access_token(token: str) -> dict[str, Any]:
    """Decode + verify a JWT access token. Raises InvalidTokenError on failure."""
    try:
        payload = jwt.decode(
            token,
            settings.jwt_secret.get_secret_value(),
            algorithms=[settings.jwt_algorithm],
        )
        if payload.get("type") != "access":
            raise InvalidTokenError("Invalid token type")
        return payload
    except JWTError as e:
        raise InvalidTokenError(f"Invalid or expired token: {e}") from e


# ---------- Auth manager ----------

class AuthManager:
    """Single-user auth manager.

    All methods are async — designed for use with AsyncSession from
    app.db.session.get_async_session.
    """

    def __init__(self, session: AsyncSession):
        self.session = session

    # ---------- User bootstrap ----------

    async def bootstrap_admin_user(self) -> User:
        """Create the admin user if not exists.

        Called on app startup. Uses hardcoded email + password from .env.
        Idempotent — if user exists, returns it unchanged.
        """
        admin_email = "admin@vapt-ai.local"
        admin_password = settings.admin_password.get_secret_value()

        # Check if admin user exists
        result = await self.session.execute(
            select(User).where(User.email == admin_email)
        )
        user = result.scalar_one_or_none()

        if user is not None:
            return user

        # Create admin user
        user = User(
            email=admin_email,
            password_hash=hash_password(admin_password),
            role="admin",
            display_name="VAPT-AI Admin",
            is_active=True,
            failed_login_attempts=0,
        )
        self.session.add(user)
        await self.session.flush()  # get user.id without commit
        return user

    # ---------- Login ----------

    async def login(self, email: str, password: str, ip: str | None = None) -> dict[str, Any]:
        """Login with email + password.

        Returns {access_token, refresh_token, token_type, expires_in, user}.
        Raises InvalidCredentialsError or AccountLockedError.
        """
        # 1. Find user by email
        result = await self.session.execute(
            select(User).where(User.email == email)
        )
        user = result.scalar_one_or_none()

        # 2. Check account lockout
        if user and user.locked_until and user.locked_until > datetime.now(UTC):
            raise AccountLockedError(user.locked_until)

        # 3. Verify password
        if not user or not verify_password(password, user.password_hash):
            # Increment failed attempts (if user exists)
            if user:
                user.failed_login_attempts += 1
                if user.failed_login_attempts >= settings.auth_max_failed_attempts:
                    user.locked_until = datetime.now(UTC) + timedelta(
                        minutes=settings.auth_lockout_minutes
                    )
                await self.session.flush()
            raise InvalidCredentialsError("Invalid email or password")

        # 4. Check account is active
        if not user.is_active:
            raise InvalidCredentialsError("Account is disabled")

        # 5. Reset failed attempts on successful login
        user.failed_login_attempts = 0
        user.locked_until = None
        user.last_login_at = datetime.now(UTC)
        await self.session.flush()

        # 6. Issue tokens
        access_token, access_expires = create_access_token(user.id, user.email)
        refresh_token, refresh_expires = create_refresh_token()

        # 7. Store refresh token hash
        rt_row = RefreshToken(
            user_id=user.id,
            token_hash=hash_token(refresh_token),
            expires_at=refresh_expires,
            issued_from_ip=ip,
        )
        self.session.add(rt_row)
        await self.session.flush()

        return {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "token_type": "bearer",
            "expires_in": settings.jwt_access_token_ttl_minutes * 60,
            "user": {
                "id": str(user.id),
                "email": user.email,
                "role": user.role,
                "display_name": user.display_name,
            },
        }

    # ---------- Refresh ----------

    async def refresh(self, refresh_token: str, ip: str | None = None) -> dict[str, Any]:
        """Rotate refresh token. Returns new {access_token, refresh_token, ...}.

        Old refresh token is revoked + replaced_by_id set to new token's id.
        Raises TokenRevokedError or InvalidTokenError.
        """
        token_hash = hash_token(refresh_token)

        # 1. Find refresh token by hash
        result = await self.session.execute(
            select(RefreshToken).where(RefreshToken.token_hash == token_hash)
        )
        rt = result.scalar_one_or_none()

        if rt is None:
            raise InvalidTokenError("Invalid refresh token")

        if rt.revoked:
            raise TokenRevokedError("Refresh token has been revoked")

        if rt.expires_at < datetime.now(UTC):
            # Mark as revoked (cleanup)
            rt.revoked = True
            rt.revoked_at = datetime.now(UTC)
            rt.revoked_reason = "expired"
            await self.session.flush()
            raise InvalidTokenError("Refresh token has expired")

        # 2. Get user
        result = await self.session.execute(select(User).where(User.id == rt.user_id))
        user = result.scalar_one_or_none()
        if user is None or not user.is_active:
            raise InvalidTokenError("User not found or disabled")

        # 3. Revoke old refresh token
        rt.revoked = True
        rt.revoked_at = datetime.now(UTC)
        rt.revoked_reason = "rotated"

        # 4. Issue new tokens
        new_access_token, access_expires = create_access_token(user.id, user.email)
        new_refresh_token, new_refresh_expires = create_refresh_token()

        new_rt = RefreshToken(
            user_id=user.id,
            token_hash=hash_token(new_refresh_token),
            expires_at=new_refresh_expires,
            issued_from_ip=ip,
        )
        self.session.add(new_rt)
        await self.session.flush()

        # 5. Link old → new (rotation chain)
        rt.replaced_by_id = new_rt.id
        await self.session.flush()

        return {
            "access_token": new_access_token,
            "refresh_token": new_refresh_token,
            "token_type": "bearer",
            "expires_in": settings.jwt_access_token_ttl_minutes * 60,
            "user": {
                "id": str(user.id),
                "email": user.email,
                "role": user.role,
                "display_name": user.display_name,
            },
        }

    # ---------- Logout ----------

    async def logout(self, refresh_token: str) -> None:
        """Revoke a refresh token (logout)."""
        token_hash = hash_token(refresh_token)

        result = await self.session.execute(
            select(RefreshToken).where(RefreshToken.token_hash == token_hash)
        )
        rt = result.scalar_one_or_none()

        if rt is None:
            return  # silent — don't leak whether token existed

        if not rt.revoked:
            rt.revoked = True
            rt.revoked_at = datetime.now(UTC)
            rt.revoked_reason = "logout"
            await self.session.flush()

    # ---------- Get current user ----------

    async def get_user_by_id(self, user_id: uuid.UUID) -> User | None:
        """Fetch user by ID. Used by get_current_user dependency."""
        result = await self.session.execute(select(User).where(User.id == user_id))
        return result.scalar_one_or_none()

    async def get_user_by_email(self, email: str) -> User | None:
        """Fetch user by email."""
        result = await self.session.execute(select(User).where(User.email == email))
        return result.scalar_one_or_none()
