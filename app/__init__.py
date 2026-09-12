"""
VAPT-AI v3.2 — Top-level Python package.

This package is being built incrementally during W1. The legacy EVVO modules
(`routes/`, `models/`, `workers/`, `shield_engine/`, `mcp_servers/`) remain
at the project root and are being migrated into `app/` over time.

Current layout (W1):
    app/
    ├── __init__.py          # this file
    ├── core/
    │   └── config.py        # Pydantic Settings (loaded from .env + config.yaml)
    ├── db/
    │   ├── __init__.py
    │   ├── base.py          # DeclarativeBase + mixins (UUID, Timestamp)
    │   ├── session.py       # async session factory (asyncpg + SQLAlchemy 2.0)
    │   ├── sync_session.py  # sync session factory (psycopg2 — for Alembic + scripts)
    │   └── models/          # SQLAlchemy 2.0 declarative models (W1-C, W1-D)
    ├── auth/
    │   └── manager.py       # JWT single-user auth (W1-F)
    └── mcp/
        └── server.py        # MCP server skeleton (W1-G)

W1-A acceptance: `uvicorn app.main:app` works (W1-F will add `app/main.py`).
"""

__version__ = "3.2.1"
__project__ = "VAPT-AI"