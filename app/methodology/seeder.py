"""
VAPT-AI Methodology Seeder — loads catalog YAMLs into PostgreSQL at app startup.

W8-E: Idempotent loader for:
    1. data/wstg_catalog.yaml        → vapt_wstg_methodology_catalog   (119 tests, W8-C)
    2. data/attack_mapping.yaml      → vapt_attack_technique_catalog    (44 techniques, W8-D)

Idempotent strategy:
    - For each catalog entry, check if (wstg_id | technique_id) already exists.
    - If exists: UPDATE non-key fields (name, description, related_mitre_attack, etc.)
      → catalog YAML is the source of truth; DB syncs to it.
    - If not exists: INSERT new row.
    - This makes the seeder safe to run on every startup without manual cleanup.

Error handling:
    - Missing YAML file → log warning, skip (don't crash app)
    - Malformed YAML → log error, skip (don't crash app)
    - DB error → log error, raise (caller decides whether to crash)
    - Empty catalog file → log warning, skip

Usage (called from app/main.py lifespan):
    from app.methodology.seeder import seed_all_catalogs
    async with async_session() as session:
        result = await seed_all_catalogs(session)
        await session.commit()
        # result: {"wstg": {"inserted": N, "updated": M}, "attack": {...}}

Manual run (for testing):
    python3 -m app.methodology.seeder
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models.methodology import WSTGMethodologyCatalog
from app.db.models.attack_catalog import AttackTechniqueCatalog

logger = logging.getLogger(__name__)


# ---------- Constants ----------
# These are computed lazily via _get_wstg_path() / _get_attack_path() so that
# tests can override via yaml_path argument without needing settings to load.

def _get_wstg_path() -> Path:
    """Get WSTG catalog path from settings.data_dir (lazy import)."""
    try:
        from app.core.config import settings
        return settings.data_dir / "wstg_catalog.yaml"
    except ImportError:
        # Fallback for tests where settings is unavailable
        return Path("data/wstg_catalog.yaml")


def _get_attack_path() -> Path:
    """Get ATT&CK catalog path from settings.data_dir (lazy import)."""
    try:
        from app.core.config import settings
        return settings.data_dir / "attack_mapping.yaml"
    except ImportError:
        return Path("data/attack_mapping.yaml")


WSTG_CATALOG_PATH = Path("data/wstg_catalog.yaml")  # default; overridden by settings
ATTACK_CATALOG_PATH = Path("data/attack_mapping.yaml")


# ---------- Result dataclass ----------

class SeedResult:
    """Result of seeding one catalog."""

    def __init__(self, catalog_name: str):
        self.catalog_name = catalog_name
        self.inserted = 0
        self.updated = 0
        self.skipped = 0  # entries skipped due to errors
        self.errors: list[str] = []

    @property
    def total(self) -> int:
        return self.inserted + self.updated + self.skipped

    def to_dict(self) -> dict[str, Any]:
        return {
            "catalog": self.catalog_name,
            "inserted": self.inserted,
            "updated": self.updated,
            "skipped": self.skipped,
            "total": self.total,
            "errors": self.errors,
        }


# ---------- WSTG seeder ----------

async def seed_wstg_catalog(session: AsyncSession, yaml_path: Path | None = None) -> SeedResult:
    """Seed vapt_wstg_methodology_catalog from data/wstg_catalog.yaml.

    Idempotent: existing rows are UPDATEd, new rows are INSERTed.

    Args:
        session: Async DB session
        yaml_path: Override path (default: settings.data_dir / "wstg_catalog.yaml")

    Returns:
        SeedResult with inserted/updated/skipped counts
    """
    result = SeedResult("wstg")
    path = yaml_path or _get_wstg_path()

    if not path.exists():
        msg = f"WSTG catalog YAML not found at {path} — skipping seed"
        logger.warning(msg)
        result.errors.append(msg)
        return result

    # Load YAML
    try:
        with path.open(encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        msg = f"Failed to parse WSTG YAML: {e}"
        logger.error(msg)
        result.errors.append(msg)
        return result

    tests = data.get("tests", [])
    if not tests:
        msg = f"WSTG catalog is empty (no 'tests' key) at {path}"
        logger.warning(msg)
        result.errors.append(msg)
        return result

    logger.info("Seeding WSTG catalog: %d entries from %s", len(tests), path)

    for entry in tests:
        try:
            await _upsert_wstg_entry(session, entry, result)
        except Exception as e:
            result.skipped += 1
            err_msg = f"Failed to seed WSTG entry {entry.get('wstg_id', '?')}: {e}"
            result.errors.append(err_msg)
            logger.error(err_msg)

    logger.info(
        "WSTG catalog seeded: inserted=%d updated=%d skipped=%d",
        result.inserted, result.updated, result.skipped,
    )
    return result


async def _upsert_wstg_entry(session: AsyncSession, entry: dict[str, Any], result: SeedResult) -> None:
    """Insert or update a single WSTG entry (idempotent)."""
    wstg_id = entry["wstg_id"]

    # Check if exists
    stmt = select(WSTGMethodologyCatalog).where(WSTGMethodologyCatalog.wstg_id == wstg_id)
    existing = (await session.execute(stmt)).scalar_one_or_none()

    if existing is None:
        # INSERT
        session.add(
            WSTGMethodologyCatalog(
                wstg_id=wstg_id,
                name=entry["name"],
                category=entry["category"],
                description=entry.get("description", ""),
                related_mitre_attack=entry.get("related_mitre_attack", []),
                related_cwe=entry.get("related_cwe", []),
                applicable_to_web_mvp=entry.get("applicable_to_web_mvp", True),
            )
        )
        result.inserted += 1
    else:
        # UPDATE (catalog YAML is source of truth)
        existing.name = entry["name"]
        existing.category = entry["category"]
        existing.description = entry.get("description", "")
        existing.related_mitre_attack = entry.get("related_mitre_attack", [])
        existing.related_cwe = entry.get("related_cwe", [])
        existing.applicable_to_web_mvp = entry.get("applicable_to_web_mvp", True)
        result.updated += 1


# ---------- ATT&CK seeder ----------

async def seed_attack_catalog(session: AsyncSession, yaml_path: Path | None = None) -> SeedResult:
    """Seed vapt_attack_technique_catalog from data/attack_mapping.yaml.

    Idempotent: existing rows are UPDATEd, new rows are INSERTed.

    Args:
        session: Async DB session
        yaml_path: Override path (default: settings.data_dir / "attack_mapping.yaml")

    Returns:
        SeedResult with inserted/updated/skipped counts
    """
    result = SeedResult("attack")
    path = yaml_path or _get_attack_path()

    if not path.exists():
        msg = f"ATT&CK catalog YAML not found at {path} — skipping seed"
        logger.warning(msg)
        result.errors.append(msg)
        return result

    # Load YAML
    try:
        with path.open(encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        msg = f"Failed to parse ATT&CK YAML: {e}"
        logger.error(msg)
        result.errors.append(msg)
        return result

    techniques = data.get("techniques", [])
    if not techniques:
        msg = f"ATT&CK catalog is empty (no 'techniques' key) at {path}"
        logger.warning(msg)
        result.errors.append(msg)
        return result

    logger.info("Seeding ATT&CK catalog: %d entries from %s", len(techniques), path)

    for entry in techniques:
        try:
            await _upsert_attack_entry(session, entry, result)
        except Exception as e:
            result.skipped += 1
            err_msg = f"Failed to seed ATT&CK entry {entry.get('technique_id', '?')}: {e}"
            result.errors.append(err_msg)
            logger.error(err_msg)

    logger.info(
        "ATT&CK catalog seeded: inserted=%d updated=%d skipped=%d",
        result.inserted, result.updated, result.skipped,
    )
    return result


async def _upsert_attack_entry(session: AsyncSession, entry: dict[str, Any], result: SeedResult) -> None:
    """Insert or update a single ATT&CK entry (idempotent)."""
    technique_id = entry["technique_id"]

    # Check if exists
    stmt = select(AttackTechniqueCatalog).where(
        AttackTechniqueCatalog.technique_id == technique_id
    )
    existing = (await session.execute(stmt)).scalar_one_or_none()

    if existing is None:
        # INSERT
        session.add(
            AttackTechniqueCatalog(
                technique_id=technique_id,
                name=entry["name"],
                tactic=entry["tactic"],
                description=entry.get("description", ""),
                detection=entry.get("detection", ""),
                mitigation=entry.get("mitigation", ""),
                related_wstg_ids=entry.get("related_wstg_ids", []),
                related_cwe=entry.get("related_cwe", []),
                example_uses=entry.get("example_uses", []),
                applicable_to_web_mvp=entry.get("applicable_to_web_mvp", True),
            )
        )
        result.inserted += 1
    else:
        # UPDATE (catalog YAML is source of truth)
        existing.name = entry["name"]
        existing.tactic = entry["tactic"]
        existing.description = entry.get("description", "")
        existing.detection = entry.get("detection", "")
        existing.mitigation = entry.get("mitigation", "")
        existing.related_wstg_ids = entry.get("related_wstg_ids", [])
        existing.related_cwe = entry.get("related_cwe", [])
        existing.example_uses = entry.get("example_uses", [])
        existing.applicable_to_web_mvp = entry.get("applicable_to_web_mvp", True)
        result.updated += 1


# ---------- Combined seeder (called from app/main.py lifespan) ----------

async def seed_all_catalogs(session: AsyncSession) -> dict[str, dict[str, Any]]:
    """Seed both WSTG + ATT&CK catalogs.

    Safe to call on every app startup — idempotent.

    Args:
        session: Async DB session (caller commits)

    Returns:
        {"wstg": {...}, "attack": {...}} — see SeedResult.to_dict()
    """
    logger.info("Seeding methodology catalogs (WSTG + ATT&CK)")
    wstg_result = await seed_wstg_catalog(session)
    attack_result = await seed_attack_catalog(session)
    return {
        "wstg": wstg_result.to_dict(),
        "attack": attack_result.to_dict(),
    }


# ---------- Manual run entry point ----------

async def _main() -> None:
    """Manual run for testing — uses the async session factory."""
    import asyncio
    from app.db.session import async_session

    async with async_session() as session:
        result = await seed_all_catalogs(session)
        await session.commit()
        print("Seed result:")
        import json
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    import asyncio
    asyncio.run(_main())
