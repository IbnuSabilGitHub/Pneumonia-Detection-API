"""
Health check and monitoring endpoints.
"""
import time
from fastapi import APIRouter,  Depends
from slowapi import Limiter
from slowapi.util import get_remote_address

from ..models.schemas import HealthResponse
from ..services.prediction import PneumoniaPredictionService
from ..core.settings import settings
from ..core.logger import get_logger
from ..docs.health_metadata import HealthMetadata

logger = get_logger(__name__)
router = APIRouter()
health_metadata = HealthMetadata.get_metadata()

# For backward compatibility with slowapi
limiter = Limiter(key_func=get_remote_address)

# Track service start time for uptime calculation
_start_time = time.time()


def get_prediction_service() -> PneumoniaPredictionService:
    """Dependency to get prediction service instance."""
    # This will be injected by the main app
    return getattr(get_prediction_service, '_service', None)


@router.get(
    "/health", 
    response_model=HealthResponse, 
    tags=["Health"],
    summary="Service Health Check (Alternative)",
    description="Alternative endpoint for health checking - same functionality as root endpoint"
)

@router.get(
    "/",
    response_model=HealthResponse,
    tags=["Health"],
    **health_metadata
)

async def health_check(
    prediction_service: PneumoniaPredictionService = Depends(get_prediction_service)
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
    uptime = time.time() - _start_time
    
    # Determine health status
    model_loaded = prediction_service.is_loaded() if prediction_service else False
    
    if prediction_service and model_loaded:
        status = "healthy"
    elif prediction_service and not model_loaded:
        status = "partial"  # Service exists but model not loaded
    else:
        status = "partial"  # Service not available but app is running
    
    return HealthResponse(
        status=status,
        model_loaded=model_loaded,
        version=settings.app_version,
        uptime=uptime
    )
