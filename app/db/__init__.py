"""
VAPT-AI database layer.

Public API:
    from app.db.base import Base, UUIDPrimaryKey, StringPrimaryKey, TimestampMixin, utcnow
    from app.db.session import get_async_session, async_session, get_async_engine, dispose_async_engine
    from app.db.sync_session import sync_session, get_sync_engine, dispose_sync_engine

Models (W1-C, W1-D will add):
    from app.db.models import ...  # noqa: F401  (import to register with Base.metadata)
"""
from app.db.base import (
    Base,
    NAMING_CONVENTION,
    StringPrimaryKey,
    TimestampMixin,
    UUIDPrimaryKey,
    utcnow,
)
from app.db.session import (
    async_session,
    async_session_factory,
    dispose_async_engine,
    get_async_engine,
    get_async_session,
    get_async_session_factory,
)
from app.db.sync_session import (
    dispose_sync_engine,
    get_sync_engine,
    get_sync_session_factory,
    sync_session,
    sync_session_factory,
)

__all__ = [
    # Base
    "Base",
    "NAMING_CONVENTION",
    "UUIDPrimaryKey",
    "StringPrimaryKey",
    "TimestampMixin",
    "utcnow",
    # Async
    "async_session",
    "async_session_factory",
    "get_async_engine",
    "get_async_session",
    "get_async_session_factory",
    "dispose_async_engine",
    # Sync
    "sync_session",
    "sync_session_factory",
    "get_sync_engine",
    "get_sync_session_factory",
    "dispose_sync_engine",
]