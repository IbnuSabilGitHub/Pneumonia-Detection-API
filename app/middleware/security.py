"""
Custom middleware for security and logging.
"""
import time
from fastapi import Request, HTTPException, status
from fastapi.responses import JSONResponse

from ..core.logger import get_logger
from ..utils.security import get_client_ip, rate_limiter
from ..utils.exceptions import RateLimitError

logger = get_logger(__name__)


async def rate_limit_middleware(request: Request, call_next):
    """
    Rate limiting middleware.
    
    Args:
        request: FastAPI request object
        call_next: Next middleware/endpoint function
        
    Returns:
        Response or raises HTTPException if rate limited
    """
    # Skip rate limiting for health check and docs
    if request.url.path in ["/", "/docs", "/redoc", "/openapi.json"]:
        return await call_next(request)
    
    client_ip = get_client_ip(request)
    
    if not rate_limiter.is_allowed(client_ip):
        logger.warning(f"Rate limit exceeded for IP: {client_ip}")
        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content={
                "detail": "Rate limit exceeded. Please try again later.",
                "error_code": "RATE_LIMIT_EXCEEDED"
            }
        )
    
    response = await call_next(request)
    return response


async def logging_middleware(request: Request, call_next):
    """
    Request logging middleware.
    
    Args:
        request: FastAPI request object
        call_next: Next middleware/endpoint function
        
    Returns:
        Response with logged request details
    """
    start_time = time.time()
    client_ip = get_client_ip(request)
    
    # Log request
    logger.info(
        f"Request: {request.method} {request.url.path} "
        f"from {client_ip}"
    )
    
    response = await call_next(request)
    
    # Log response
    process_time = time.time() - start_time
    logger.info(
        f"Response: {response.status_code} "
        f"for {request.method} {request.url.path} "
        f"({process_time:.3f}s)"
    )
    
    return response


async def error_handling_middleware(request: Request, call_next):
    """
    Global error handling middleware.
    
    Args:
        request: FastAPI request object
        call_next: Next middleware/endpoint function
        
    Returns:
        Response or error response
    """
    try:
        response = await call_next(request)
        return response
    except HTTPException:
        # Re-raise HTTP exceptions (they're handled by FastAPI)
        raise
    except Exception as e:
        # Log unexpected errors
        logger.error(f"Unexpected error: {e}", exc_info=True)
        
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "detail": "Internal server error",
                "error_code": "INTERNAL_ERROR"
            }
        )
