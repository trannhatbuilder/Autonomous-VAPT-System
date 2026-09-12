"""
VAPT-AI async database session factory.

Uses asyncpg driver (postgresql+asyncpg://) with SQLAlchemy 2.0 async ORM.
This is the canonical DB access layer for all NEW VAPT-AI code (W1-C onward).

Legacy EVVO code (routes/, models/, workers/, shield_engine/) still uses
the sync psycopg2 pool in db_pool.py — that code is being migrated incrementally.

Usage in FastAPI routes:
    from app.db.session import get_async_session, AsyncSession

    @router.get("/items")
    async def list_items(session: AsyncSession = Depends(get_async_session)):
        result = await session.execute(select(Item))
        return result.scalars().all()

Usage in scripts / workers:
    from app.db.session import async_session_factory

    async with async_session_factory() as session:
        ... # use session
        await session.commit()

Connection pool:
    asyncpg uses its own internal pool. We configure it via create_async_engine
    pool_size / max_overflow / pool_timeout / pool_recycle.
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.core.config import settings

logger = logging.getLogger(__name__)


# ---------- Engine ----------

def _build_async_engine() -> AsyncEngine:
    """Build the async engine with pool settings from config."""
    dsn = settings.async_postgres_dsn
    engine = create_async_engine(
        dsn,
        echo=settings.debug and settings.environment == "dev",
        pool_size=settings.db_pool_min_conn,
        max_overflow=settings.db_pool_max_conn - settings.db_pool_min_conn,
        pool_timeout=settings.db_pool_timeout,
        pool_recycle=3600,  # recycle connections every hour (avoid stale conns)
        pool_pre_ping=True,  # check connection is alive before checkout
        future=True,
    )
    logger.info("async engine created: %s (pool %d-%d)",
                dsn.replace(settings.effective_postgres_dsn.split("@")[0].split("://")[-1], "***")
                   if "@" in dsn else dsn,
                settings.db_pool_min_conn, settings.db_pool_max_conn)
    return engine


# Singleton engine — created on first access
_async_engine: AsyncEngine | None = None


def get_async_engine() -> AsyncEngine:
    """Return the singleton async engine (creates on first call)."""
    global _async_engine
    if _async_engine is None:
        _async_engine = _build_async_engine()
    return _async_engine


# ---------- Session factory ----------

def _build_session_factory() -> async_sessionmaker[AsyncSession]:
    """Build the async session factory bound to the singleton engine."""
    return async_sessionmaker(
        bind=get_async_engine(),
        class_=AsyncSession,
        expire_on_commit=False,  # don't expire objects after commit (avoid lazy-load N+1)
        autoflush=False,         # explicit flush() — avoids surprises
        autocommit=False,
    )


_async_session_factory: async_sessionmaker[AsyncSession] | None = None


def get_async_session_factory() -> async_sessionmaker[AsyncSession]:
    """Return the singleton async session factory (creates on first call)."""
    global _async_session_factory
    if _async_session_factory is None:
        _async_session_factory = _build_session_factory()
    return _async_session_factory


# Alias for backward naming compat
async_session_factory = get_async_session_factory


# ---------- FastAPI dependency ----------

async def get_async_session() -> AsyncIterator[AsyncSession]:
    """FastAPI dependency: yields an AsyncSession, auto-closes on exit.

    Commits on success, rolls back on exception.

    Usage:
        @router.get("/items")
        async def list_items(session: AsyncSession = Depends(get_async_session)):
            ...
    """
    factory = get_async_session_factory()
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


# ---------- Script helper ----------

@asynccontextmanager
async def async_session() -> AsyncIterator[AsyncSession]:
    """Context manager for use outside FastAPI (Celery workers, scripts).

    Usage:
        async with async_session() as session:
            session.add(obj)
            await session.commit()
    """
    factory = get_async_session_factory()
    async with factory() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise


# ---------- Lifecycle ----------

async def dispose_async_engine() -> None:
    """Dispose the async engine — call on app shutdown."""
    global _async_engine, _async_session_factory
    if _async_engine is not None:
        await _async_engine.dispose()
        logger.info("async engine disposed")
    _async_engine = None
    _async_session_factory = None