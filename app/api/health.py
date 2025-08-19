"""
Health check and monitoring endpoints.
"""
import time
from fastapi import APIRouter, Request, Depends
from slowapi import Limiter
from slowapi.util import get_remote_address

from ..models.schemas import HealthResponse, SecurityStatusResponse
from ..services.prediction import PneumoniaPredictionService
from ..utils.security import get_client_ip, rate_limiter, file_hash_cache
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


@router.get("/security/status", response_model=SecurityStatusResponse, tags=["Security"])
@limiter.limit("10/minute")
async def security_status(request: Request):
    """
    Get security status and rate limiting information.
    
    This endpoint is for educational/development purposes to show
    current security measures and rate limiting status.
    
    Returns:
        Security status including rate limiting and cache information
    """
    client_ip = get_client_ip(request)
    
    # Get current rate limit status
    requests_count = rate_limiter.get_request_count(client_ip)
    is_blocked = rate_limiter.is_blocked(client_ip)
    cache_size = file_hash_cache.get_cache_size()
    
    return SecurityStatusResponse(
        client_ip=client_ip,
        requests_in_last_minute=requests_count,
        rate_limit=f"{settings.rate_limit_requests} requests per {settings.rate_limit_window} seconds",
        is_blocked=is_blocked,
        cache_entries=cache_size,
        security_features=[
            f"Rate Limiting ({settings.rate_limit_requests}/min per IP)",
            f"File Size Validation ({settings.max_file_size / (1024 * 1024):.0f}MB max)",
            f"File Type Validation ({', '.join(settings.allowed_extensions)})",
            "Image Content Validation",
            f"Duplicate Detection ({settings.cache_duration}s cache)",
            "Request Logging with IP tracking",
            "Educational/Learning Mode (No Auth Required)"
        ]
    )
