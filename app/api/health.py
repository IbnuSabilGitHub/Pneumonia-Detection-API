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

logger = get_logger(__name__)
router = APIRouter()

# For backward compatibility with slowapi
limiter = Limiter(key_func=get_remote_address)

# Track service start time for uptime calculation
_start_time = time.time()


def get_prediction_service() -> PneumoniaPredictionService:
    """Dependency to get prediction service instance."""
    # This will be injected by the main app
    return getattr(get_prediction_service, '_service', None)


@router.get("/", response_model=HealthResponse, tags=["Health"])
async def health_check(
    prediction_service: PneumoniaPredictionService = Depends(get_prediction_service)
):
    """
    Health check endpoint.
    
    Returns:
        Service health status including model loading state and uptime
    """
    uptime = time.time() - _start_time
    
    return HealthResponse(
        status="healthy" if prediction_service and prediction_service.is_loaded() else "unhealthy",
        model_loaded=prediction_service.is_loaded() if prediction_service else False,
        version=settings.app_version,
        uptime=uptime
    )
