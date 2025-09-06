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


@router.get(
    "/", 
    response_model=HealthResponse, 
    tags=["Health"],
    summary="🏥 Service Health Check",
    description="""
Health Check Endpoint

Provides comprehensive health status information about the Pneumonia Detection API service.

📊 Health Status Levels

• healthy: All systems operational, model loaded and ready
• partial: Service running but with limitations (e.g., model not loaded)  
• unhealthy: Critical issues detected

📋 Response Information

• Status: Current health state of the service
• Model Status: Whether AI models are loaded and ready
• Version: Current API version
• Uptime: Time since service start (in seconds)

🔍 Use Cases

• Load Balancer Health Checks: Monitor service availability
• Monitoring Systems: Track service uptime and status
• Troubleshooting: Verify service and model status
• Development: Quick service verification

⚡ Performance

• Response Time: < 100ms typical
• Rate Limiting: No rate limits applied
• Caching: Status computed in real-time
    """,
    response_description="Service health status with detailed information"
)
@router.get(
    "/health", 
    response_model=HealthResponse, 
    tags=["Health"],
    summary="🏥 Service Health Check (Alternative)",
    description="Alternative endpoint for health checking - same functionality as root endpoint"
)
async def health_check(
    prediction_service: PneumoniaPredictionService = Depends(get_prediction_service)
):
    """
    **🏥 Comprehensive Health Check**
    
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
