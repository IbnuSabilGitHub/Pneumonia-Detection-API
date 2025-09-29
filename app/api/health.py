"""
Health check and monitoring endpoints.
"""

import time

from fastapi import APIRouter, Depends, HTTPException, status
from slowapi import Limiter
from slowapi.util import get_remote_address

from ..core.dependencies import get_prediction_service as get_prediction_service_dep
from ..core.logger import get_logger
from ..core.settings import settings
from ..docs.sections.health_metadata import HealthMetadata
from ..models.health_schemas import HealthErrorResponse, HealthResponse
from ..services.prediction import PneumoniaPredictionService

logger = get_logger(__name__)
router = APIRouter()
health_metadata = HealthMetadata.get_metadata()

# For backward compatibility with slowapi
limiter = Limiter(key_func=get_remote_address)

# Track service start time for uptime calculation
_start_time = time.time()


def get_prediction_service() -> PneumoniaPredictionService:
    """
    Dependency to get prediction service instance from dependency container.

    Uses the proper dependency injection system instead of manual injection.
    """
    return get_prediction_service_dep()


@router.get(
    "/health",
    response_model=HealthResponse,
    tags=["Health"],
    summary="Service Health Check (Alternative)",
    description="Alternative endpoint for health checking - same functionality as root endpoint",
)
async def health_check_alt(
    prediction_service: PneumoniaPredictionService = Depends(get_prediction_service),
):
    """Alternative health check endpoint."""
    return await health_check(prediction_service)


@router.get("/", response_model=HealthResponse, tags=["Health"], **health_metadata)
async def health_check(
    prediction_service: PneumoniaPredictionService = Depends(get_prediction_service),
):
    """
    **Comprehensive Health Check**

    Returns detailed health status including:
    - Service availability and operational state
    - AI model loading status and readiness
    - API version information
    - Service uptime since last restart

    This endpoint is designed for:
    - Load balancer health checks
    - Monitoring system integration
    - Service status verification
    - Troubleshooting and diagnostics

    **Returns:**
        HealthResponse: Complete health status information

    **Status Meanings:**
        - `healthy`: All systems operational
        - `partial`: Service running with limitations
        - `unhealthy`: Critical issues detected
    """
    try:
        uptime = time.time() - _start_time

        # Determine health status
        model_loaded = prediction_service.is_loaded() if prediction_service else False

        if prediction_service and model_loaded:
            status_val = "healthy"
        elif prediction_service and not model_loaded:
            status_val = "partial"  # Service exists but model not loaded
            logger.warning("Prediction service available but model not loaded")
        else:
            logger.error(
                "Prediction service not available - service initialization failed"
            )
            status_val = "unhealthy"  # Critical service not available

        return HealthResponse(
            status=status_val,
            model_loaded=model_loaded,
            version=settings.app_version,
            uptime=uptime,
        )
    except Exception as e:
        logger.error("Health check failed: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=HealthErrorResponse(
                detail=f"Health check failed: {str(e)}",
                error_code="HEALTH_CHECK_FAILED",
            ).model_dump(),
        )
