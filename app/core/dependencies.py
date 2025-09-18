"""
Dependency injection container for application services.
"""

from functools import lru_cache
from typing import Optional

from ..services.prediction import PneumoniaPredictionService
from .advanced_rate_limiting import AdvancedRateLimiter
from .logger import get_logger

logger = get_logger(__name__)


class AppDependencies:
    """
    Central dependency container for application services.

    This replaces global variables with a proper dependency injection pattern.
    """

    def __init__(self):
        self._prediction_service: Optional[PneumoniaPredictionService] = None
        self._rate_limiter: Optional[AdvancedRateLimiter] = None
        self._initialized = False

    @property
    def prediction_service(self) -> Optional[PneumoniaPredictionService]:
        """Get the prediction service instance."""
        return self._prediction_service

    @prediction_service.setter
    def prediction_service(self, service: PneumoniaPredictionService):
        """Set the prediction service instance."""
        self._prediction_service = service
        logger.debug("Prediction service injected into dependencies")

    @property
    def rate_limiter(self) -> Optional[AdvancedRateLimiter]:
        """Get the rate limiter instance."""
        return self._rate_limiter

    @rate_limiter.setter
    def rate_limiter(self, limiter: AdvancedRateLimiter):
        """Set the rate limiter instance."""
        self._rate_limiter = limiter
        logger.debug("Rate limiter injected into dependencies")

    @property
    def is_initialized(self) -> bool:
        """Check if dependencies are initialized."""
        return self._initialized

    def mark_initialized(self):
        """Mark dependencies as initialized."""
        self._initialized = True
        logger.info("Dependencies marked as initialized")

    def get_service_status(self) -> dict:
        """Get status of all services."""
        return {
            "prediction_service": self._prediction_service is not None,
            "rate_limiter": self._rate_limiter is not None,
            "initialized": self._initialized,
        }


@lru_cache()
def get_dependencies() -> AppDependencies:
    """
    Get the singleton dependency container.

    Uses lru_cache to ensure single instance across the application.
    """
    return AppDependencies()


def get_prediction_service() -> Optional[PneumoniaPredictionService]:
    """Get prediction service from dependencies."""
    return get_dependencies().prediction_service


def get_rate_limiter() -> Optional[AdvancedRateLimiter]:
    """Get rate limiter from dependencies."""
    return get_dependencies().rate_limiter
