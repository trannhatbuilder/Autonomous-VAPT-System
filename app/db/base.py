"""
VAPT-AI SQLAlchemy 2.0 declarative base + shared mixins.

All VAPT-AI models (in app/db/models/) MUST inherit from `Base` defined here.
This ensures consistent naming conventions, primary key types, and timestamps.

Legacy EVVO models (in /models/ at project root) use raw SQL via psycopg2
and are NOT registered against this Base. They will be migrated incrementally
in later weeks. For W1, only NEW tables (added in W1-C, W1-D) inherit from Base.

W1-E will register Base.metadata in alembic/env.py so `alembic revision
--autogenerate` detects the new models.
"""
from __future__ import annotations

import uuid
from datetime import datetime, UTC
from typing import Any

from sqlalchemy import DateTime, MetaData, String, func
from sqlalchemy.orm import DeclarativeBase, Mapped, declared_attr, mapped_column
from sqlalchemy.dialects.postgresql import UUID


# Naming convention for constraints — required for Alembic autogenerate
# to produce consistently-named constraints (so `--autogenerate` is diffable).
NAMING_CONVENTION: dict[str, str] = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


# Construct MetaData with naming convention FIRST — DeclarativeBase will
# pick this up via the `metadata` class argument below.
_metadata = MetaData(naming_convention=NAMING_CONVENTION)


class Base(DeclarativeBase):
    """Root declarative base for all VAPT-AI SQLAlchemy 2.0 models.

    Subclasses MUST define `__tablename__`. Use the `UUIDPrimaryKey` and
    `TimestampMixin` mixins for consistent PK + audit columns.
    """

    metadata = _metadata

    # Type annotation for IDE / mypy — subclasses override this.
    __tablename__: str

    def __repr__(self) -> str:
        """Default repr — show table name + PK for debugging."""
        pk_cols = list(self.__table__.primary_key.columns)
        if pk_cols:
            pk_name = pk_cols[0].name
            pk_val = getattr(self, pk_name, None)
            return f"<{self.__class__.__name__} {pk_name}={pk_val!r}>"
        return f"<{self.__class__.__name__}>"

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict — useful for JSON responses + audit logs."""
        out: dict[str, Any] = {}
        for col in self.__table__.columns:
            val = getattr(self, col.name, None)
            # Convert datetime → ISO 8601 string for JSON safety
            if isinstance(val, datetime):
                val = val.isoformat()
            # Convert UUID → string
            elif isinstance(val, uuid.UUID):
                val = str(val)
            out[col.name] = val
        return out


# ---------- Mixins ----------

class UUIDPrimaryKey:
    """Mixin: UUID primary key (PostgreSQL native uuid type).

    Use for tables where the PK is a UUID (most VAPT-AI tables).
    Avoids SERIAL integer PKs (which leak row counts + are harder to shard).
    """

    @declared_attr
    @classmethod
    def id(cls) -> Mapped[uuid.UUID]:
        return mapped_column(
            UUID(as_uuid=True),
            primary_key=True,
            default=uuid.uuid4,
            server_default=func.gen_random_uuid(),
        )


class StringPrimaryKey:
    """Mixin: String primary key (for tables that use natural keys).

    Use sparingly — e.g. scans.id (which is a short slug like 'scan_abc123').
    Most tables should use UUIDPrimaryKey instead.
    """

    @declared_attr
    @classmethod
    def id(cls) -> Mapped[str]:
        return mapped_column(String(64), primary_key=True)


class TimestampMixin:
    """Mixin: created_at + updated_at columns with auto-fill.

    `created_at` is set once on INSERT (server_default=now()).
    `updated_at` is set on INSERT AND refreshed on UPDATE (onupdate=now()).

    Both are timezone-aware UTC timestamps (TIMESTAMPTZ in PostgreSQL).
    """

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )


# ---------- Helpers ----------

def utcnow() -> datetime:
    """Return current UTC time as timezone-aware datetime.

    Use this instead of datetime.utcnow() (deprecated in Python 3.12).
    """
    return datetime.now(UTC)