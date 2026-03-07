"""
Dependency injection container for application services.
"""

from functools import lru_cache
from typing import Dict, Optional

from ..services.prediction import PneumoniaPredictionService
from .rate_limiting import AdvancedRateLimiter
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
        """Get the prediction service instance

        Returns:
            Optional[PneumoniaPredictionService]: The prediction service if set, else None.
        """
        return self._prediction_service

    @prediction_service.setter
    def prediction_service(self, service: PneumoniaPredictionService):
        """Set the prediction service instance

        Args:
            service (PneumoniaPredictionService): The prediction service instance to set.
        """
        self._prediction_service = service
        logger.debug("Prediction service injected into dependencies")

    @property
    def rate_limiter(self) -> Optional[AdvancedRateLimiter]:
        """Get the rate limiter instance

        Returns:
            Optional[AdvancedRateLimiter]: The rate limiter if set, else None.

        """
        return self._rate_limiter

    @rate_limiter.setter
    def rate_limiter(self, limiter: AdvancedRateLimiter):
        """Set the rate limiter instance

        Args:
            limiter (AdvancedRateLimiter): The rate limiter instance to set.
        """
        self._rate_limiter = limiter
        logger.debug("Rate limiter injected into dependencies")

    @property
    def is_initialized(self) -> bool:
        """Check if dependencies are initialized

        Returns:
            bool: True if initialized, else False.
        """
        return self._initialized

    def mark_initialized(self) -> None:
        """Mark dependencies as initialized"""
        self._initialized = True
        logger.info("Dependencies marked as initialized")

    def get_service_status(self) -> Dict[str, bool]:
        """Get status of all services.

        Returns:
            Dict[str, bool]: Dictionary containing service availability status.
                Keys: 'prediction_service', 'rate_limiter', 'initialized'
                Values: Boolean indicating if service is available/initialized
        """
        return {
            "prediction_service": self._prediction_service is not None,
            "rate_limiter": self._rate_limiter is not None,
            "initialized": self._initialized,
        }


@lru_cache()
def get_dependencies() -> AppDependencies:
    """
    Get the singleton dependency container.

    Returns:
        AppDependencies: The singleton instance of the dependency container.
    """
    return AppDependencies()


def get_prediction_service() -> Optional[PneumoniaPredictionService]:
    """Get prediction service from dependencies.

    Returns:
        Optional[PneumoniaPredictionService]: The prediction service instance
            if it has been injected, None otherwise.
    """
    return get_dependencies().prediction_service


def get_rate_limiter() -> Optional[AdvancedRateLimiter]:
    """Get rate limiter from dependencies.

    Returns:
        Optional[AdvancedRateLimiter]: The rate limiter instance
            if it has been injected, None otherwise.
    """
    return get_dependencies().rate_limiter
