"""
Startup manager for handling service initialization and lifecycle management.
"""

import asyncio
from typing import Any, Dict, List, Optional, Union

from ..services.prediction import PneumoniaPredictionService
from .user_rate_limiting import create_user_rate_limiter, UserRateLimiter
from .logger import get_logger

logger = get_logger(__name__)


class StartupManager:
    """
    Manages the startup and initialization of application services.

    This class provides a modular way to initialize services with proper
    error handling and reporting.
    """

    def __init__(self) -> None:
        """Initialize the startup manager with empty service containers."""
        self.services: Dict[str, Any] = {}
        self.startup_errors: List[str] = []
        self.warnings: List[str] = []

    async def initialize_prediction_service(self) -> bool:
        """
        Initialize prediction service with comprehensive error handling.

        Returns:
            bool: True if initialization successful, False if failed.
                Failures are logged and added to startup_errors.

        """
        try:
            logger.info("🤖 Initializing prediction service...")
            service = PneumoniaPredictionService()
            service.load_model()

            self.services["prediction"] = service
            logger.info("✅ Prediction service initialized successfully")
            return True

        except Exception as e:
            error_msg = f"Prediction service initialization failed: {e}"
            logger.error("❌ %s", error_msg)
            self.startup_errors.append(error_msg)
            return False

    # initialize_rate_limiter removed - using user_rate_limiter only

    async def initialize_user_rate_limiter(
        self, 
        user_rate_config: Dict[str, Any]
    ) -> bool:
        """
        Initialize user-based rate limiter for JWT authentication.

        Args:
            user_rate_config (Dict[str, Any]): User rate limiting configuration containing:
                - enabled: Whether user rate limiting is enabled
                - max_requests: Max requests per user per window
                - window_size: Window size in seconds
                - use_supabase: Whether to use Supabase storage
                - supabase_url: Supabase project URL
                - supabase_key: Supabase API key

        Returns:
            bool: True if initialization successful, False if failed.
        """
        if not user_rate_config.get("enabled", True):
            logger.info("⏭️ User rate limiting disabled by configuration")
            return True

        try:
            logger.info("🛡️ Initializing user rate limiter...")

            user_limiter = await create_user_rate_limiter(
                max_requests=user_rate_config.get("max_requests", 100),
                window_size=user_rate_config.get("window_size", 3600),
                supabase_url=user_rate_config.get("supabase_url"),
                supabase_key=user_rate_config.get("supabase_key"),
                use_supabase=user_rate_config.get("use_supabase", True)
            )

            status = user_limiter.get_status()
            storage_type = status.get("storage", {}).get("backend_type", "unknown")
            
            logger.info(
                "✅ User rate limiter initialized: %s/%ds per user, storage: %s",
                user_rate_config.get("max_requests", 100),
                user_rate_config.get("window_size", 3600),
                storage_type
            )

            self.services["user_rate_limiter"] = user_limiter
            return True

        except Exception as e:
            error_msg = f"User rate limiter initialization failed: {e}"
            logger.error("❌ %s", error_msg)
            self.startup_errors.append(error_msg)
            return False

    async def run_health_checks(self) -> Dict[str, Any]:
        """
        Run health checks on initialized services.

        Returns:
            Dict[str, Any]: Health status report containing:
                - healthy (bool): Overall health status
                - services (Dict): Individual service health details
                - prediction: Model loading and availability status
                - rate_limiter: Storage connectivity and functionality
        """
        health_status = {}

        # Check prediction service
        if "prediction" in self.services:
            try:
                service = self.services["prediction"]
                # Simple health check - verify model is loaded
                health_status["prediction_service"] = (
                    hasattr(service, "model") and service.model is not None
                )
            except Exception:
                health_status["prediction_service"] = False
        else:
            health_status["prediction_service"] = False

        # Check user rate limiter
        if "user_rate_limiter" in self.services:
            try:
                user_limiter = self.services["user_rate_limiter"]
                status = user_limiter.get_status()
                health_status["user_rate_limiter"] = status.get("initialized", False)
            except Exception:
                health_status["user_rate_limiter"] = False
        else:
            health_status["user_rate_limiter"] = False

        return health_status

    async def startup(
        self, 
        user_rate_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Run all startup tasks with concurrent initialization.

        Args:
            user_rate_config (Optional[Dict[str, Any]]): User rate limiting config
                for JWT-based rate limiting. If None, user rate limiting is disabled.

        Returns:
            Dict[str, Any]: Comprehensive startup report containing:
                - services (Dict[str, Any]): Dictionary of initialized services
                - errors (List[str]): List of error messages from failed initializations
                - warnings (List[str]): List of warning messages from fallback scenarios
                - success_count (int): Number of successfully initialized services
                - total_services (int): Total number of services attempted
                - health_status (Dict[str, Any]): Results from post-startup health checks

        """
        logger.info("🚀 Starting application services...")

        # Initialize services concurrently for better startup time
        tasks = [
            self.initialize_prediction_service(),
        ]
        
        # Add user rate limiter if config provided
        if user_rate_config and user_rate_config.get("enabled", False):
            tasks.append(self.initialize_user_rate_limiter(user_rate_config))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Count successful initializations
        success_count = sum(1 for r in results if r is True)
        total_services = len(tasks)

        # Run health checks
        health_status = await self.run_health_checks()

        # Log startup summary
        if self.startup_errors:
            logger.error("❌ Startup completed with %d errors", len(self.startup_errors))
            for error in self.startup_errors:
                logger.error("   • %s", error)

        if self.warnings:
            logger.warning("⚠️ Startup completed with %d warnings", len(self.warnings))
            for warning in self.warnings:
                logger.warning("   • %s", warning)

        if success_count == total_services:
            logger.info("✅ All %d services initialized successfully", total_services)
        else:
            logger.warning(
                "⚠️ %d/%d services initialized successfully",
                success_count,
                total_services,
            )

        return {
            "services": self.services,
            "errors": self.startup_errors,
            "warnings": self.warnings,
            "success_count": success_count,
            "total_services": total_services,
            "health_status": health_status,
        }

    async def shutdown(self) -> None:
        """
        Gracefully shutdown all services.
        """
        logger.info("🛑 Shutting down services...")

        shutdown_errors = []

        # Shutdown rate limiter
        if "rate_limiter" in self.services:
            try:
                limiter = self.services["rate_limiter"]
                if limiter.storage and hasattr(limiter.storage, "disconnect"):
                    await limiter.storage.disconnect()
                    logger.info("✅ Rate limiter storage disconnected")
                else:
                    logger.info("✅ Rate limiter cleanup completed (memory storage)")
            except Exception as e:
                error_msg = f"Rate limiter shutdown error: {e}"
                logger.error("❌ %s", error_msg)
                shutdown_errors.append(error_msg)

        # Shutdown user rate limiter
        if "user_rate_limiter" in self.services:
            try:
                user_limiter = self.services["user_rate_limiter"]
                await user_limiter.shutdown()
                logger.info("✅ User rate limiter shut down")
            except Exception as e:
                error_msg = f"User rate limiter shutdown error: {e}"
                logger.error("❌ %s", error_msg)
                shutdown_errors.append(error_msg)

        # Shutdown prediction service
        if "prediction" in self.services:
            try:
                service = self.services["prediction"]
                if hasattr(service, "cleanup"):
                    service.cleanup()
                    logger.info("✅ Prediction service cleaned up")
                else:
                    logger.info(
                        "✅ Prediction service cleanup completed (no cleanup needed)"
                    )
            except Exception as e:
                error_msg = f"Prediction service shutdown error: {e}"
                logger.error("❌ %s", error_msg)
                shutdown_errors.append(error_msg)

        if shutdown_errors:
            logger.error("❌ Shutdown completed with %d errors", len(shutdown_errors))
        else:
            logger.info("✅ All services shut down successfully")

        # Clear services
        self.services.clear()
