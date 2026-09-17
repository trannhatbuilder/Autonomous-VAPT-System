"""
VAPT-AI test configuration — for tests under tests/vapt/.

Sets up:
    - Test DB: vapt_ai_test (PostgreSQL, separate from production vapt_ai)
    - Test environment variables (override .env)
    - pytest-asyncio config (auto mode)
    - DB fixtures: clean DB before each test, dispose engine after

Test DB setup (run once manually — see docs/POSTGRESQL_SETUP.md):
    sudo -u postgres psql -c "CREATE DATABASE vapt_ai_test OWNER vapt;"
    sudo -u postgres psql -d vapt_ai_test -c "CREATE EXTENSION IF NOT EXISTS vector;"
    sudo -u postgres psql -d vapt_ai_test -c "CREATE EXTENSION IF NOT EXISTS pgcrypto;"
    sudo -u postgres psql -d vapt_ai_test -c "GRANT ALL ON SCHEMA public TO vapt;"
    cd ~/VAPT-AI && DATABASE_URL=postgresql://vapt:vapt%40NhatVKU@localhost:5432/vapt_ai_test alembic upgrade head
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure project root is on sys.path (so `app` package is importable)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ---------- Test environment (override .env) ----------
# Use test DB (vapt_ai_test) — never touch production vapt_ai
# IMPORTANT: tests/conftest.py (EVVO legacy) sets ENVIRONMENT=test via setdefault.
# pydantic-settings is case-insensitive + has no env_prefix, so "ENVIRONMENT=test"
# leaks into our Settings (Literal["dev","staging","prod"]) and breaks collection.
# We must explicitly clear it before setting VAPT_AI_ENVIRONMENT.
os.environ.pop("ENVIRONMENT", None)
os.environ["VAPT_AI_ENVIRONMENT"] = "dev"  # must be dev/staging/prod per Settings literal
os.environ["VAPT_AI_DEBUG"] = "false"
os.environ["VAPT_AI_LOG_LEVEL"] = "WARNING"  # quiet logs during tests

# Test DB URL (URL-encoded password vapt@NhatVKU → vapt%40NhatVKU)
_TEST_DB_URL = "postgresql://vapt:vapt%40NhatVKU@localhost:5432/vapt_ai_test"
os.environ["VAPT_AI_POSTGRES_DB"] = _TEST_DB_URL
os.environ["EVVO_POSTGRES_DB"] = _TEST_DB_URL  # legacy alias
os.environ["DATABASE_URL"] = _TEST_DB_URL  # for alembic env.py fallback

# Test auth secrets (deterministic — not real secrets, OK for tests)
os.environ["VAPT_AI_JWT_SECRET"] = "test-jwt-secret-48-bytes-deterministic-for-tests-only"
os.environ["VAPT_AI_ENCRYPTION_KEY"] = "test-fernet-key-deterministic-for-tests-only"
os.environ["VAPT_AI_ADMIN_EMAIL"] = "admin@vapt-ai.local"
os.environ["VAPT_AI_ADMIN_PASSWORD"] = "test-admin-password-123"

# Disable MCP + external integrations during tests
os.environ["VAPT_AI_MCP_ENABLED"] = "false"
os.environ["VAPT_AI_MCP_METASPLOIT_ENABLED"] = "false"

# Redis (test DB 15 — separate from production DB 0)
os.environ["VAPT_AI_REDIS_URL"] = "redis://localhost:6379/15"

# ---------- pytest-asyncio config ----------
import pytest


def pytest_collection_modifyitems(config, items):
    """Auto-mark async tests with asyncio marker."""
    import inspect
    for item in items:
        if "asyncio" in item.keywords:
            continue
        if hasattr(item, "function") and inspect.iscoroutinefunction(item.function):
            item.add_marker(pytest.mark.asyncio)


# ---------- DB fixtures ----------

@pytest.fixture
async def db_engine():
    """Create a fresh async engine for tests."""
    from app.db.session import get_async_engine, dispose_async_engine
    engine = get_async_engine()
    yield engine
    await dispose_async_engine()


@pytest.fixture
async def db_session(db_engine):
    """Provide a clean AsyncSession for each test.

    Tables are TRUNCATED (not dropped) before each test for speed.
    """
    from sqlalchemy import text
    from app.db.session import async_session_factory
    # Import models package to register them with Base.metadata
    import app.db.models  # noqa: F401

    # Truncate all vapt_ tables
    async with db_engine.begin() as conn:
        result = await conn.execute(text(
            "SELECT tablename FROM pg_tables WHERE schemaname='public' AND tablename LIKE 'vapt_%'"
        ))
        table_names = [row[0] for row in result]
        if table_names:
            await conn.execute(text("SET session_replication_role = 'replica'"))
            for table in table_names:
                await conn.execute(text(f"TRUNCATE TABLE {table} RESTART IDENTITY CASCADE"))
            await conn.execute(text("SET session_replication_role = 'origin'"))

    async with async_session_factory() as session:
        yield session
        await session.rollback()


@pytest.fixture
async def admin_user(db_session):
    """Create the admin user (admin@vapt-ai.local) for tests."""
    from app.auth.manager import AuthManager
    auth = AuthManager(db_session)
    user = await auth.bootstrap_admin_user()
    await db_session.commit()
    return user


@pytest.fixture
async def admin_token(admin_user, db_session):
    """Login as admin + return {access_token, refresh_token, user}."""
    from app.auth.manager import AuthManager
    auth = AuthManager(db_session)
    result = await auth.login(
        email="admin@vapt-ai.local",
        password="test-admin-password-123",
        ip="127.0.0.1",
    )
    await db_session.commit()
    return result


@pytest.fixture
async def client(db_session):
    """FastAPI TestClient with DB session overridden."""
    from httpx import AsyncClient, ASGITransport
    from app.main import create_app
    from app.db.session import get_async_session

    app = create_app()

    async def override_get_session():
        yield db_session

    app.dependency_overrides[get_async_session] = override_get_session

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac

    app.dependency_overrides.clear()
