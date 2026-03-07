"""
Compatibility layer for advanced rate limiting.

This module provides backward compatibility by re-exporting functions
from the rate_limiting package and dependencies module.
"""

from typing import Optional

from .dependencies import get_dependencies
from .rate_limiting import AdvancedRateLimiter
from .rate_limiting.api import create_advanced_rate_limiter

__all__ = [
    "AdvancedRateLimiter",
    "create_advanced_rate_limiter",
    "get_rate_limiter",
    "set_rate_limiter",
]


def get_rate_limiter() -> Optional[AdvancedRateLimiter]:
    """
    Get the global rate limiter instance.

    Returns:
        Optional[AdvancedRateLimiter]: The rate limiter instance if initialized,
            None otherwise.
    """
    return get_dependencies().rate_limiter


def set_rate_limiter(rate_limiter: AdvancedRateLimiter) -> None:
    """
    Set the global rate limiter instance.

    Args:
        rate_limiter: The rate limiter instance to set globally.
    """
    get_dependencies().rate_limiter = rate_limiter
