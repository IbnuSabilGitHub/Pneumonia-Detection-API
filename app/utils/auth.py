"""
Authentication and authorization utilities for admin endpoints.
"""

import secrets
from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader

from ..core.logger import get_logger
from ..core.settings import settings

logger = get_logger(__name__)

# API Key header scheme
api_key_header = APIKeyHeader(name="X-Admin-API-Key", auto_error=False)


async def verify_admin_api_key(api_key: str = Security(api_key_header)) -> str:
    """
    Verify admin API key for protected endpoints.

    Args:
        api_key: API key from request header

    Returns:
        str: Validated API key

    Raises:
        HTTPException: If API key is invalid or missing
    """
    # Get admin API key from environment
    admin_api_key = getattr(settings, "admin_api_key", None)

    # If no admin key configured, disable admin endpoints
    if not admin_api_key:
        logger.warning("Admin API key not configured, blocking admin endpoint access")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error": "Admin endpoints not configured",
                "message": "Contact administrator to enable admin features",
            },
        )

    # Check if API key provided
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "error": "Missing API key",
                "message": "Admin endpoints require X-Admin-API-Key header",
                "required_header": "X-Admin-API-Key",
            },
        )

    # Verify API key using constant-time comparison
    if not secrets.compare_digest(api_key, admin_api_key):
        logger.warning(f"Invalid admin API key attempt from request")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={
                "error": "Invalid API key",
                "message": "The provided API key is not valid",
            },
        )

    logger.info("Admin API key validated successfully")
    return api_key
