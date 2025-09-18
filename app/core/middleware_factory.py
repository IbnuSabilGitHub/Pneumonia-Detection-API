"""
Middleware factory for centralized middleware configuration.
"""

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
    Factory class for setting up all application middleware.

    Provides centralized configuration and proper ordering of middleware.
    """

    @staticmethod
    def setup_security_middleware(app: FastAPI) -> None:
        """
        Setup security-related middleware.

        Args:
            app: FastAPI application instance
        """
        logger.info("🔒 Setting up security middleware...")

        # Add security middleware (applied first)
        app.add_middleware(SecurityMiddleware)
        logger.debug("✅ Security middleware added")

    @staticmethod
    def setup_cors_middleware(app: FastAPI) -> None:
        """
        Setup CORS middleware.

        Args:
            app: FastAPI application instance
        """
        logger.info("🌐 Setting up CORS middleware...")

        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origins,
            allow_credentials=True,
            allow_methods=["GET", "POST"],
            allow_headers=["*"],
        )
        logger.debug(f"✅ CORS middleware added with origins: {settings.cors_origins}")

    @staticmethod
    def setup_trusted_host_middleware(app: FastAPI) -> None:
        """
        Setup trusted host middleware.

        Args:
            app: FastAPI application instance
        """
        logger.info("🛡️ Setting up trusted host middleware...")

        app.add_middleware(TrustedHostMiddleware, allowed_hosts=settings.trusted_hosts)
        logger.debug(
            f"✅ Trusted host middleware added with hosts: {settings.trusted_hosts}"
        )

    @staticmethod
    def setup_custom_middleware(app: FastAPI) -> None:
        """
        Setup custom HTTP middleware.

        Args:
            app: FastAPI application instance
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
        Setup SlowAPI rate limiting for compatibility.

        Args:
            app: FastAPI application instance
        """
        logger.info("⏱️ Setting up SlowAPI rate limiting...")

        # Create limiter instance
        limiter = Limiter(key_func=get_remote_address)
        app.state.limiter = limiter

        # Add exception handler for rate limit exceeded
        app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

        logger.debug("✅ SlowAPI rate limiting configured")

    @staticmethod
    def setup_all_middleware(app: FastAPI) -> None:
        """
        Setup all middleware in the correct order.

        Args:
            app: FastAPI application instance
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
    def get_middleware_info() -> dict:
        """
        Get information about configured middleware.

        Returns:
            Dict containing middleware configuration info
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
