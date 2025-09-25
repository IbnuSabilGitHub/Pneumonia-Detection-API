"""
Custom middleware for security and logging with in-memory storage support.
"""

import time

from fastapi import HTTPException, Request, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from ..core.logger import get_logger
from ..core.settings import settings
from ..utils.security import get_client_ip

logger = get_logger(__name__)


class SecurityMiddleware(BaseHTTPMiddleware):
    """
    Advanced rate limiting middleware with in-memory storage support.

    Args:
        request: FastAPI request object
        call_next: Next middleware/endpoint function

    Returns:
        Response or raises HTTPException if rate limited
    """

    def __init__(self, app):
        super().__init__(app)
        self.logger = get_logger(__name__)

    async def dispatch(self, request, call_next):
        start_time = time.time()
        client_ip = get_client_ip(request)
        endpoint = f"{request.method} {request.url.path}"

        # Skip rate limiting for excluded endpoints
        if self._should_skip_rate_limiting(request.url.path):
            return await call_next(request)

        # Extract file hash for upload endpoints
        file_hash = self._extract_file_hash(request)

        # Check rate limiting
        rate_limit_result = await self._check_rate_limit(
            client_ip=client_ip, endpoint=endpoint, request=request, file_hash=file_hash
        )

        if not rate_limit_result["allowed"]:
            return self._create_rate_limit_response(
                rate_limit_result, client_ip, endpoint
            )

        # Process the request
        try:
            response = await call_next(request)

            # Log successful request
            process_time = time.time() - start_time
            self.logger.info(
                f"{endpoint} | IP: {client_ip} | "
                f"Status: {response.status_code} | Time: {process_time:.3f}s"
            )

            # Add security headers
            response.headers["X-Content-Type-Options"] = "nosniff"
            response.headers["X-Frame-Options"] = "DENY"
            response.headers["X-XSS-Protection"] = "1; mode=block"

            return response

        except Exception as e:
            self.logger.error(
                f"Request failed: {str(e)} | IP: {client_ip} | Endpoint: {endpoint}"
            )
            raise

    def _should_skip_rate_limiting(self, path: str) -> bool:
        """Check if endpoint should skip rate limiting."""
        return path in getattr(settings, "excluded_paths", [])

    def _extract_file_hash(self, request) -> str:
        """Extract file hash for upload endpoints."""
        if request.url.path == "/pneumonia/predict" and request.method == "POST":
            return getattr(request.state, "file_hash", None)
        return None

    async def _check_rate_limit(
        self, client_ip: str, endpoint: str, request, file_hash: str
    ) -> dict:
        """Check rate limiting and return result."""
        # Import at runtime to get the latest reference
        from ..core.advanced_rate_limiting import get_rate_limiter

        advanced_rate_limiter = get_rate_limiter()

        # Check if rate limiter is available and has storage initialized
        if advanced_rate_limiter is None:
            self.logger.warning("Rate limiter not initialized, allowing request")
            return {
                "allowed": True,
                "reason": "Rate limiter not initialized",
                "details": {},
            }

        try:
            # Use async rate limiting if storage is initialized, otherwise fallback
            if (
                hasattr(advanced_rate_limiter, "_storage_initialized")
                and advanced_rate_limiter._storage_initialized
            ):
                # Advanced async rate limiting check
                (
                    is_allowed,
                    reason,
                    details,
                ) = await advanced_rate_limiter.is_request_allowed(
                    client_ip=client_ip,
                    endpoint=endpoint,
                    request=request,
                    file_hash=file_hash,
                )
            else:
                # Fallback to legacy sync method (with in-memory storage)
                self.logger.debug("Using fallback rate limiting")
                is_allowed, reason, details = True, "Fallback mode", {}

            return {"allowed": is_allowed, "reason": reason, "details": details}

        except Exception as e:
            # If rate limiting fails, log error but allow request to proceed
            self.logger.error(
                f"Rate limiting check failed: {e} | IP: {client_ip} | Allowing request"
            )
            return {
                "allowed": True,
                "reason": f"Rate limit check failed: {e}",
                "details": {},
            }

    def _create_rate_limit_response(
        self, rate_limit_result: dict, client_ip: str, endpoint: str
    ):
        """Create rate limit exceeded response with proper CORS headers."""
        reason = rate_limit_result["reason"]
        details = rate_limit_result["details"]

        self.logger.warning(
            f"Request blocked: {reason} | IP: {client_ip} | Endpoint: {endpoint} | Details: {details}"
        )

        error_detail = {
            "error": "Rate limit exceeded",
            "message": reason,
            "client_ip": client_ip,
            "endpoint": endpoint,
            "timestamp": time.time(),
            "details": details,
        }

        # Include proper CORS headers to prevent "Failed to fetch" in Swagger UI
        cors_headers = {
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
            "Access-Control-Allow-Headers": "*",
            "Access-Control-Allow-Credentials": "true",
            "Content-Type": "application/json",
            "Retry-After": "60",
            "X-RateLimit-Limit": str(details.get("rate_limit", "unknown")),
            "X-RateLimit-Remaining": "0",
            "X-RateLimit-Reset": str(int(time.time() + 60)),
        }

        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content=error_detail,
            headers=cors_headers,
        )


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
    logger.info("Request: %s %s from %s", request.method, request.url.path, client_ip)

    response = await call_next(request)

    # Log response
    process_time = time.time() - start_time
    logger.info(
        "Response: %d for %s %s (%.3fs)",
        response.status_code,
        request.method,
        request.url.path,
        process_time,
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
        logger.error("Unexpected error: %s", e, exc_info=True)

        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "Internal server error", "error_code": "INTERNAL_ERROR"},
        )
