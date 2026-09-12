"""
VAPT-AI sync database session factory.

Uses psycopg2 driver (postgresql+psycopg2://) — required by Alembic (which
runs synchronously) and by legacy EVVO code that hasn't been migrated to async.

DO NOT use this in FastAPI route handlers — use app.db.session.get_async_session
instead. Sync DB calls in async routes block the event loop.

Usage (sync scripts):
    from app.db.sync_session import sync_session_factory

    with sync_session_factory() as session:
        ... # use session
        session.commit()
"""
from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Iterator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import settings

logger = logging.getLogger(__name__)


# ---------- Engine ----------

def _build_sync_engine():
    """Build the sync engine with psycopg2 driver."""
    dsn = settings.sync_postgres_dsn
    engine = create_engine(
        dsn,
        echo=settings.debug and settings.environment == "dev",
        pool_size=settings.db_pool_min_conn,
        max_overflow=settings.db_pool_max_conn - settings.db_pool_min_conn,
        pool_timeout=settings.db_pool_timeout,
        pool_recycle=3600,
        pool_pre_ping=True,
        future=True,
    )
    logger.info("sync engine created: %s (pool %d-%d)",
                dsn.replace(settings.effective_postgres_dsn.split("@")[0].split("://")[-1], "***")
                   if "@" in dsn else dsn,
                settings.db_pool_min_conn, settings.db_pool_max_conn)
    return engine


_sync_engine = None


def get_sync_engine():
    """Return the singleton sync engine (creates on first call)."""
    global _sync_engine
    if _sync_engine is None:
        _sync_engine = _build_sync_engine()
    return _sync_engine


# ---------- Session factory ----------

def _build_sync_session_factory() -> sessionmaker[Session]:
    return sessionmaker(
        bind=get_sync_engine(),
        class_=Session,
        expire_on_commit=False,
        autoflush=False,
        autocommit=False,
    )


_sync_session_factory: sessionmaker[Session] | None = None


def get_sync_session_factory() -> sessionmaker[Session]:
    """Return the singleton sync session factory (creates on first call)."""
    global _sync_session_factory
    if _sync_session_factory is None:
        _sync_session_factory = _build_sync_session_factory()
    return _sync_session_factory


# Alias
sync_session_factory = get_sync_session_factory


# ---------- Context manager ----------

@contextmanager
def sync_session() -> Iterator[Session]:
    """Context manager for sync scripts.

    Usage:
        with sync_session() as session:
            session.add(obj)
            session.commit()
    """
    factory = get_sync_session_factory()
    session = factory()
    try:
        yield session
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


# ---------- Lifecycle ----------

def dispose_sync_engine() -> None:
    """Dispose the sync engine — call on app shutdown."""
    global _sync_engine, _sync_session_factory
    if _sync_engine is not None:
        _sync_engine.dispose()
        logger.info("sync engine disposed")
    _sync_engine = None
    _sync_session_factory = None