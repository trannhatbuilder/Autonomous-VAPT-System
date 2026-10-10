"""
VAPT-AI Worker Package — Celery background tasks.

Exports ONLY the Celery app so systemd services can use:
    celery -A workers worker -l info -Q scans,retests,retention
    celery -A workers beat -l info

The Celery app is defined in scan_tasks.py. This __init__.py re-exports it
as `celery_app` (canonical VAPT-AI name) so both `celery -A workers worker`
and `celery -A workers.scan_tasks worker` work.

W2-D fix: __init__.py must not import heavyweight/optional submodules — if
any failed to import (missing dep, circular import) Celery could not start.
It exports ONLY celery_app. Import tasks/utilities from their submodules
directly when needed, e.g. `from workers.scan_tasks import run_vapt_scan_task`.
"""

# Import the Celery app from scan_tasks.py + re-export under canonical name
from .scan_tasks import app as celery_app

# Also export as `app` for backward compat (EVVO tests use `from workers.scan_tasks import app`)
app = celery_app

__all__ = [
    # Celery app (canonical VAPT-AI name — used by systemd services)
    "celery_app",
    # Celery app (backward compat with EVVO — used by `celery -A workers.scan_tasks`)
    "app",
]
