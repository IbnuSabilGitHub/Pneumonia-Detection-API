"""Compatibility layer for advanced rate limiting.

This module now re-exports the refactored implementation from
`app.core.rate_limiting` while keeping the previous import path
stable. Please migrate imports to `app.core.rate_limiting`.
"""

import logging
from typing import Any, Dict, Optional

from .rate_limiting.api import create_advanced_rate_limiter  # noqa: F401

# Re-export public API from the new package
from .rate_limiting.core import AdvancedRateLimiter  # noqa: F401
from .storage_factory import StorageType  # re-export for callers using old path

logger = logging.getLogger(__name__)

# Global instance (kept for backward compatibility)
advanced_rate_limiter: Optional[AdvancedRateLimiter] = None


def get_rate_limiter() -> Optional[AdvancedRateLimiter]:
    """Get the current rate limiter instance.

    Deprecated: Prefer importing from `app.core.rate_limiting`.
    """
    return advanced_rate_limiter


def set_rate_limiter(limiter: AdvancedRateLimiter) -> None:
    """Set the global rate limiter instance.

    Deprecated: Prefer importing from `app.core.rate_limiting`.
    """
    global advanced_rate_limiter
    if limiter is not None:
        logger.debug(
            "advanced_rate_limiting: using compatibility wrapper; consider migrating imports to app.core.rate_limiting.*"
        )
    advanced_rate_limiter = limiter
