"""
Backward compatibility shim for advanced rate limiting.
The advanced rate limiting functionality has been refactored.
This module provides compatibility for existing code.
"""

from ..core.logger import get_logger

logger = get_logger(__name__)


def get_rate_limiter():
    """
    Get the advanced rate limiter instance.
    
    This is a backward compatibility function.
    The actual rate limiting is now handled by the middleware
    and user rate limiting systems.
    
    Returns:
        A mock rate limiter object or None
    """
    logger.debug("get_rate_limiter() called - using compatibility shim")
    return None
