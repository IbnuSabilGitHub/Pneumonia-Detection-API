"""
Startup manager for handling service initialization.
"""

import asyncio
from typing import Any, Dict, List, Optional

from ..services.prediction import PneumoniaPredictionService
from .advanced_rate_limiting import create_advanced_rate_limiter
from .logger import get_logger
from .storage_factory import StorageType

logger = get_logger(__name__)


class StartupManager:
    """
    Manages the startup and initialization of application services.

    This class provides a modular way to initialize services with proper
    error handling and reporting.
    """

    def __init__(self):
        self.services = {}
        self.startup_errors = []
        self.warnings = []

    async def initialize_prediction_service(self) -> bool:
        """
        Initialize prediction service with error handling.

        Returns:
            bool: True if successful, False otherwise
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

    async def initialize_rate_limiter(self, storage_config: Dict[str, Any]) -> bool:
        """
        Initialize rate limiter with fallback to memory storage.

        Args:
            storage_config: Storage configuration dictionary

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            logger.info("🛡️ Initializing rate limiter...")

            # Try with configured storage type first
            storage_type = (
                StorageType.MEMORY
            )  # Default to memory for single-instance deployments

            rate_limiter = await create_advanced_rate_limiter(
                storage_type=storage_type, storage_config=storage_config
            )

            # Test storage connection
            if rate_limiter.storage:
                storage_info = await rate_limiter.storage.get_info()
                logger.info(
                    "✅ Rate limiter initialized with %s storage",
                    storage_info.get('backend_type', 'unknown'),
                )
            else:
                logger.warning("⚠️ Rate limiter initialized without storage backend")
                self.warnings.append("Rate limiter running without persistent storage")

            self.services["rate_limiter"] = rate_limiter
            return True

        except Exception as e:
            # Try fallback to memory storage
            try:
                logger.warning(
                    "Primary rate limiter failed, attempting fallback: %s", e
                )
                fallback_limiter = await create_advanced_rate_limiter(
                    storage_type=StorageType.MEMORY, storage_config={"max_size": 1000}
                )

                self.services["rate_limiter"] = fallback_limiter
                self.warnings.append(
                    f"Using fallback memory storage for rate limiter: {e}"
                )
                logger.info("✅ Rate limiter initialized with fallback memory storage")
                return True

            except Exception as fallback_error:
                error_msg = (
                    f"Rate limiter initialization failed completely: {fallback_error}"
                )
                logger.error("❌ %s", error_msg)
                self.startup_errors.append(error_msg)
                return False

    async def run_health_checks(self) -> Dict[str, bool]:
        """
        Run health checks on initialized services.

        Returns:
            Dict[str, bool]: Health status for each service
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

        # Check rate limiter
        if "rate_limiter" in self.services:
            try:
                limiter = self.services["rate_limiter"]
                # Check if storage is accessible
                if limiter.storage:
                    health_status["rate_limiter"] = await limiter.storage.ping()
                else:
                    health_status[
                        "rate_limiter"
                    ] = True  # Memory storage always available
            except Exception:
                health_status["rate_limiter"] = False
        else:
            health_status["rate_limiter"] = False

        return health_status

    async def startup(self, storage_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run all startup tasks with concurrent initialization.

        Args:
            storage_config: Storage configuration dictionary

        Returns:
            Dict containing services, errors, warnings, and health status
        """
        logger.info("🚀 Starting application services...")

        # Initialize services concurrently for better startup time
        tasks = [
            self.initialize_prediction_service(),
            self.initialize_rate_limiter(storage_config),
        ]

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
