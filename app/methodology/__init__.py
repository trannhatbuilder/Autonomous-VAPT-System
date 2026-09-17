"""VAPT-AI methodology package — catalog seeders for WSTG + ATT&CK.

W8-E: Loads OWASP WSTG v4.2 catalog + MITRE ATT&CK v15.1 curated catalog
into PostgreSQL at app startup (idempotent — safe to run on every boot).

Public API:
    seed_all_catalogs(session) — seed both catalogs (used in app/main.py lifespan)
    seed_wstg_catalog(session) — seed only WSTG
    seed_attack_catalog(session) — seed only ATT&CK
"""
from app.methodology.seeder import (
    SeedResult,
    seed_all_catalogs,
    seed_attack_catalog,
    seed_wstg_catalog,
)

__all__ = [
    "SeedResult",
    "seed_all_catalogs",
    "seed_attack_catalog",
    "seed_wstg_catalog",
]