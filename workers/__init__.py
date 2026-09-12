"""
VAPT-AI Worker Package — Celery background tasks.

Exports ONLY the Celery app so systemd services can use:
    celery -A workers worker -l info -Q scans,retests,retention
    celery -A workers beat -l info

The Celery app is defined in scan_tasks.py. This __init__.py re-exports it
as `celery_app` (canonical VAPT-AI name) so both `celery -A workers worker`
and `celery -A workers.scan_tasks worker` work.

W2-D fix: previously __init__.py imported EVVO legacy utilities
(RedisRateLimiter, HealthChecker, etc.) which have complex dependency
chains. If any of them failed to import (missing dep, circular import),
Celery could not start. Now we export ONLY celery_app — utilities can
be imported directly from their submodules when needed.

To import EVVO legacy utilities, use:
    from workers.redis_rate_limiter import RedisRateLimiter
    from workers.health import HealthChecker
    from workers.scan_tasks import run_scan_task
(Not: from workers import RedisRateLimiter — that no longer works)
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
