"""
Middleware factory for centralized middleware configuration and setup.

This module provides a factory pattern for setting up all application middleware
in the correct order with proper configuration. It handles security middleware,
CORS, trusted hosts, rate limiting, and custom HTTP middleware.

The middleware is applied in reverse order (last added executes first), so the
factory ensures proper ordering for security and functionality.
"""

from typing import Dict, List

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from ..core.logger import get_logger
from ..core.settings import settings
from ..middleware.security import (
    SecurityMiddleware,
    error_handling_middleware,
    logging_middleware,
)

logger = get_logger(__name__)


class MiddlewareFactory:
    """
    Factory class for setting up all application middleware components.

    This factory provides centralized configuration and ensures proper ordering
    of all middleware components. It handles the complexity of middleware
    ordering (reverse application order) and provides individual setup methods
    for different middleware categories.
    """

    @staticmethod
    def setup_security_middleware(app: FastAPI) -> None:
        """
        Setup security-related middleware for request protection.

        Args:
            app (FastAPI): FastAPI application instance to configure

        """
        logger.info("🔒 Setting up security middleware...")

        # Add security middleware (applied first)
        app.add_middleware(SecurityMiddleware)
        logger.debug("✅ Security middleware added")

    @staticmethod
    def setup_cors_middleware(app: FastAPI) -> None:
        """
        Setup Cross-Origin Resource Sharing (CORS) middleware.

        Args:
            app (FastAPI): FastAPI application instance to configure
        """
        logger.info("🌐 Setting up CORS middleware...")

        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origins,
            allow_credentials=True,
            allow_methods=["GET", "POST"],
            allow_headers=["*"],
        )
        logger.debug("✅ CORS middleware added with origins: %s", settings.cors_origins)

    @staticmethod
    def setup_trusted_host_middleware(app: FastAPI) -> None:
        """
        Setup trusted host middleware for host validation.

        Args:
            app (FastAPI): FastAPI application instance to configure
        """
        logger.info("🛡️ Setting up trusted host middleware...")

        app.add_middleware(TrustedHostMiddleware, allowed_hosts=settings.trusted_hosts)
        logger.debug(
            f"✅ Trusted host middleware added with hosts: {settings.trusted_hosts}"
        )

    @staticmethod
    def setup_custom_middleware(app: FastAPI) -> None:
        """
        Setup custom HTTP middleware for logging and error handling.

        Args:
            app (FastAPI): FastAPI application instance to configure

        Side Effects:
            - Adds error_handling_middleware for global exception handling
            - Adds logging_middleware for request/response logging
            - Logs custom middleware setup progress

        """
        logger.info("⚙️ Setting up custom middleware...")

        # Note: Middleware is applied in reverse order
        # So we add them in reverse order of execution
        app.middleware("http")(error_handling_middleware)
        app.middleware("http")(logging_middleware)

        logger.debug("✅ Custom middleware added (error handling, logging)")

    @staticmethod
    def setup_rate_limiting(app: FastAPI) -> None:
        """
        Setup SlowAPI rate limiting for backward compatibility.
        DISABLED: Using Advanced Rate Limiting instead.

        Args:
            app (FastAPI): FastAPI application instance to configure

        Side Effects:
            - Creates and attaches Limiter instance to app.state
            - Adds RateLimitExceeded exception handler
            - Logs rate limiting setup progress

        """
        logger.info("⏱️ SlowAPI rate limiting disabled - using Advanced Rate Limiting")

        # DISABLED: SlowAPI rate limiting - using Advanced Rate Limiting instead
        # Create limiter instance
        # limiter = Limiter(key_func=get_remote_address)
        # app.state.limiter = limiter

        # Add exception handler for rate limit exceeded
        # app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

        logger.debug("✅ Advanced rate limiting will be used instead of SlowAPI")

    @staticmethod
    def setup_all_middleware(app: FastAPI) -> None:
        """
        Setup all middleware components in the correct execution order.

        This method orchestrates the setup of all middleware components,
        ensuring they are added in the proper order for optimal security
        and functionality. The order accounts for FastAPI's reverse
        middleware execution pattern.

        Args:
            app (FastAPI): FastAPI application instance to configure

        Side Effects:
            - Calls all individual middleware setup methods
            - Logs overall middleware setup progress
            - Ensures proper middleware execution order

        Execution Order (first to last in request processing):
            1. Custom HTTP Middleware (error handling, logging)
            2. Security Middleware (rate limiting, attack detection)
            3. CORS Middleware (cross-origin policy)
            4. Trusted Host Middleware (host validation)
            5. SlowAPI Rate Limiting (compatibility layer)
        """
        logger.info("🔧 Setting up all middleware...")

        # Order matters! Middleware is applied in reverse order
        # So the last added middleware executes first

        # 1. Setup rate limiting (for SlowAPI compatibility)
        MiddlewareFactory.setup_rate_limiting(app)

        # 2. Setup trusted host middleware
        MiddlewareFactory.setup_trusted_host_middleware(app)

        # 3. Setup CORS middleware
        MiddlewareFactory.setup_cors_middleware(app)

        # 4. Setup security middleware
        MiddlewareFactory.setup_security_middleware(app)

        # 5. Setup custom middleware (applied last, executes first)
        MiddlewareFactory.setup_custom_middleware(app)

        logger.info("✅ All middleware configured successfully")

    @staticmethod
    def get_middleware_info() -> Dict[str, any]:
        """
        Get comprehensive information about configured middleware.

        Provides detailed information about the middleware stack configuration,
        including execution order, security features, and configuration details.

        Returns:
            Dict[str, any]: Comprehensive middleware information containing:
                - middleware_order (List[str]): Middleware execution order
                - security_features (List[str]): Security capabilities
                - cors_origins (List[str]): Allowed CORS origins
                - trusted_hosts (List[str]): Trusted host whitelist
                - rate_limiting (str): Rate limiting system description
        """
        return {
            "middleware_order": [
                "Custom HTTP Middleware (error_handling, logging)",
                "Security Middleware (rate limiting, attack detection)",
                "CORS Middleware",
                "Trusted Host Middleware",
                "SlowAPI Rate Limiting",
            ],
            "security_features": [
                "Advanced rate limiting with IP switching detection",
                "Request fingerprinting",
                "Behavioral analysis",
                "Global attack scoring",
                "File validation and duplicate detection",
            ],
            "cors_origins": settings.cors_origins,
            "trusted_hosts": settings.trusted_hosts,
            "rate_limiting": "Hybrid (SlowAPI + Advanced Rate Limiter)",
        }
