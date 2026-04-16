"""
Custom middleware for security and logging with in-memory storage support.
Supports both IP-based rate limiting (legacy) and JWT user-based rate limiting.
"""

import os
import time
from typing import Optional

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
        # Allow enabling extra diagnostic headers without changing code
        self.enable_debug_headers = os.getenv(
            "RATE_LIMIT_DEBUG_HEADERS", "false"
        ).lower() in ("1", "true", "yes")
        # Trust proxy chain? (X-Forwarded-For first hop vs last)
        self.trust_proxy = os.getenv("TRUST_PROXY", "true").lower() in (
            "1",
            "true",
            "yes",
        )

    async def dispatch(self, request, call_next):
        start_time = time.time()
        client_ip = get_client_ip(request)
        # If multiple IPs in X-Forwarded-For and trust_proxy disabled, take last (direct peer) for stricter behavior
        xff = request.headers.get("x-forwarded-for")
        if xff and not self.trust_proxy:
            try:
                client_ip = xff.split(',')[-1].strip()
            except Exception:
                pass
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

            # Add rate limiting headers for successful requests
            await self._add_rate_limit_headers(response, client_ip, request)

            # Optional debug headers (never enable in public prod by default)
            if self.enable_debug_headers:
                try:
                    from ..core.advanced_rate_limiting import get_rate_limiter

                    rl = get_rate_limiter()
                    if rl and rl.attack_detector:
                        response.headers["X-Attack-Score"] = str(
                            round(rl.attack_detector.attack_score, 3)
                        )
                        atk_type = (
                            getattr(rl.attack_detector, "last_attack_type", "")
                            or "none"
                        )
                        response.headers["X-Last-Attack-Type"] = atk_type
                except Exception:
                    pass

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
        """Check rate limiting and return result.
        
        Uses user-based rate limiting if JWT auth is enabled,
        otherwise falls back to IP-based rate limiting.
        """
        # Try user-based rate limiting first if JWT auth is enabled
        if settings.jwt_auth_enabled and settings.user_rate_limiting_enabled:
            result = await self._check_user_rate_limit(request)
            if result:
                return result
        
        # Fall back to IP-based rate limiting
        return await self._check_ip_rate_limit(client_ip, endpoint, request, file_hash)

    def _extract_user_id_from_request(self, request: Request) -> Optional[str]:
        """Extract user_id from JWT token in Authorization header."""
        try:
            auth_header = request.headers.get("Authorization", "")
            if not auth_header.startswith("Bearer "):
                return None
            
            token = auth_header[7:]  # Remove "Bearer " prefix
            
            # Import JWT utilities
            from ..utils.jwt_auth import verify_jwt_token
            
            payload = verify_jwt_token(token)
            if payload:
                return payload.user_id
            return None
        except Exception as e:
            self.logger.debug(f"Failed to extract user_id from JWT: {e}")
            return None

    async def _check_user_rate_limit(self, request: Request) -> Optional[dict]:
        """Check user-based rate limiting using JWT identity.
        
        Returns:
            dict with rate limit result if user authenticated, None otherwise.
        """
        user_id = self._extract_user_id_from_request(request)
        if not user_id:
            # No valid JWT, skip user rate limiting (will fall back to IP-based)
            return None
        
        try:
            from ..core.user_rate_limiting import get_user_rate_limiter
            
            user_rate_limiter = get_user_rate_limiter()
            if not user_rate_limiter:
                self.logger.debug("User rate limiter not initialized")
                return None
            
            # Check user rate limit
            rate_info = await user_rate_limiter.check_rate_limit(user_id)
            
            if rate_info.allowed:
                # Store rate info in request state for later header addition
                request.state.user_rate_info = rate_info
                return {
                    "allowed": True,
                    "reason": "User rate limit OK",
                    "details": {
                        "user_id": user_id[:8] + "...",
                        "requests_made": rate_info.requests_made,
                        "requests_limit": rate_info.requests_limit,
                        "requests_remaining": rate_info.requests_remaining,
                    },
                }
            else:
                return {
                    "allowed": False,
                    "reason": f"User rate limit exceeded ({rate_info.requests_made}/{rate_info.requests_limit})",
                    "details": {
                        "user_id": user_id[:8] + "...",
                        "requests_made": rate_info.requests_made,
                        "requests_limit": rate_info.requests_limit,
                        "window_size": rate_info.window_size,
                        "retry_after": rate_info.retry_after,
                        "rate_info": rate_info,
                    },
                }
        
        except Exception as e:
            self.logger.error(f"User rate limit check failed: {e}")
            return None  # Fall back to IP-based

    async def _check_ip_rate_limit(
        self, client_ip: str, endpoint: str, request, file_hash: str
    ) -> dict:
        """Check IP-based rate limiting (legacy method)."""
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

            # Ensure details is always a dictionary
            if not isinstance(details, dict):
                details = {"original_details": details}

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
            "details": {
                k: v for k, v in details.items() 
                if k != "rate_info"  # Don't include full rate_info object
            },
        }

        # Check if this is a user-based rate limit response
        if "rate_info" in details:
            rate_info = details["rate_info"]
            window_size = rate_info.window_size
            retry_after = rate_info.retry_after
            limit_info = f"{rate_info.requests_limit} per {window_size}s"
            remaining = 0
            
            cors_headers = {
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
                "Access-Control-Allow-Headers": "*",
                "Access-Control-Allow-Credentials": "true",
                "Content-Type": "application/json",
                "Retry-After": str(retry_after),
                "X-RateLimit-Limit": limit_info,
                "X-RateLimit-Remaining": str(remaining),
                "X-RateLimit-Reset": str(int(rate_info.reset_time)),
                "X-RateLimit-Window": str(window_size),
                "X-RateLimit-Type": "user",
            }
            
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content=error_detail,
                headers=cors_headers,
            )

        # Extract rate limiting information from details (handle different formats)
        if isinstance(details, dict):
            # Handle nested dictionary format
            if "ip" in details and isinstance(details["ip"], dict):
                ip_count = details["ip"].get("requests_in_window", 0)
            else:
                ip_count = details.get("requests_in_window", 0)

            fingerprint_count = details.get("fingerprint_count", 0)
            max_requests = details.get("max_requests_per_ip", 100)
            max_fingerprint = details.get("max_fingerprint_requests", 50)
            window_size = details.get("window_size", 300)
        else:
            # Fallback for non-dictionary details
            ip_count = 0
            fingerprint_count = 0
            max_requests = 100
            max_fingerprint = 50
            window_size = 300

        # Determine which limit was exceeded
        if "IP rate limit" in reason:
            remaining = max(0, max_requests - ip_count)
            limit_info = f"{max_requests} per {window_size}s"
        elif "Fingerprint" in reason:
            remaining = max(0, max_fingerprint - fingerprint_count)
            limit_info = f"{max_fingerprint} per {window_size}s"
        else:
            remaining = 0
            limit_info = f"{max_requests} per {window_size}s"

        # Include proper CORS headers to prevent "Failed to fetch" in Swagger UI
        cors_headers = {
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
            "Access-Control-Allow-Headers": "*",
            "Access-Control-Allow-Credentials": "true",
            "Content-Type": "application/json",
            "Retry-After": str(window_size),
            "X-RateLimit-Limit": limit_info,
            "X-RateLimit-Remaining": str(remaining),
            "X-RateLimit-Reset": str(int(time.time() + window_size)),
            "X-RateLimit-Window": str(window_size),
        }

        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content=error_detail,
            headers=cors_headers,
        )

    async def _add_rate_limit_headers(self, response, client_ip: str, request) -> None:
        """Add rate limiting headers to successful responses."""
        try:
            # Check for user rate limit info first (JWT auth)
            if hasattr(request.state, "user_rate_info"):
                rate_info = request.state.user_rate_info
                response.headers["X-RateLimit-Limit"] = f"{rate_info.requests_limit} per {rate_info.window_size}s"
                response.headers["X-RateLimit-Remaining"] = str(rate_info.requests_remaining)
                response.headers["X-RateLimit-Reset"] = str(int(rate_info.reset_time))
                response.headers["X-RateLimit-Window"] = str(rate_info.window_size)
                response.headers["X-RateLimit-Type"] = "user"
                return

            # Fall back to IP-based rate limit headers
            # Import at runtime to get the latest reference
            from ..core.advanced_rate_limiting import get_rate_limiter

            advanced_rate_limiter = get_rate_limiter()

            if advanced_rate_limiter is None:
                return

            # Get rate limiting configuration
            config = advanced_rate_limiter.rate_limiting_config
            max_requests = config.get("max_requests_per_ip", 100)
            max_fingerprint = config.get("max_fingerprint_requests", 50)
            window_size = config.get("window_size", 300)

            # Get current usage if possible
            if (
                hasattr(advanced_rate_limiter, "rate_limit_manager")
                and advanced_rate_limiter.rate_limit_manager
            ):
                try:
                    ip_count = await advanced_rate_limiter.rate_limit_manager.get_ip_request_count(
                        client_ip
                    )
                    fingerprint = advanced_rate_limiter.fingerprint_manager.create_request_fingerprint(
                        request
                    )
                    fingerprint_count = await advanced_rate_limiter.rate_limit_manager.get_fingerprint_request_count(
                        fingerprint
                    )

                    # Use the more restrictive limit
                    ip_remaining = max(0, max_requests - ip_count)
                    fp_remaining = max(0, max_fingerprint - fingerprint_count)
                    remaining = min(ip_remaining, fp_remaining)

                    response.headers[
                        "X-RateLimit-Limit"
                    ] = f"{max_requests} per {window_size}s"
                    response.headers["X-RateLimit-Remaining"] = str(remaining)
                    response.headers["X-RateLimit-Reset"] = str(
                        int(time.time() + window_size)
                    )
                    response.headers["X-RateLimit-Window"] = str(window_size)

                except Exception as e:
                    # If we can't get counts, just add basic headers
                    response.headers[
                        "X-RateLimit-Limit"
                    ] = f"{max_requests} per {window_size}s"
                    response.headers["X-RateLimit-Window"] = str(window_size)
            else:
                # Fallback headers
                response.headers[
                    "X-RateLimit-Limit"
                ] = f"{max_requests} per {window_size}s"
                response.headers["X-RateLimit-Window"] = str(window_size)

        except Exception as e:
            # Don't break the response if header addition fails
            self.logger.debug(f"Failed to add rate limit headers: {e}")


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
    # Debug specific paths
    if "/pneumonia/model/info" in request.url.path:
        print(f"[DEBUG] Processing model/info request", flush=True)
    
    try:
        response = await call_next(request)
        return response
    except HTTPException as e:
        # Re-raise HTTP exceptions (they're handled by FastAPI)  
        if "/pneumonia/model/info" in request.url.path:
            print(f"[DEBUG] HTTPException: {e}", flush=True)
        raise
    except Exception as e:
        # Log unexpected errors with debug info
        import sys
        import traceback
        error_msg = f"Unexpected error in {request.url.path}: {type(e).__name__}: {e}"
        print(f"[DEBUG] {error_msg}", file=sys.stderr, flush=True)
        traceback.print_exc(file=sys.stderr)
        logger.error("Unexpected error: %s", e, exc_info=True)

        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "Internal server error", "error_code": "INTERNAL_ERROR"},
        )
