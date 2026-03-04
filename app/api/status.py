"""
Security Status Endpoint
"""

from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, status

from ..core.logger import get_logger
from ..core.settings import settings
from ..docs.sections.status_metadata import StatusMetadata
from ..models.security_schemes import SecurityErrorResponse, SecurityStatusResponse
from ..utils.jwt_auth import verify_admin_jwt_or_api_key

logger = get_logger(__name__)
router = APIRouter()

status_metadata = StatusMetadata.get_metadata()


def _map_security_data_to_model(security_data: dict) -> dict:
    """
    Map the security data from rate limiter to AdvancedProtection model format.

    Args:
        security_data: Raw security data from rate limiter

    Returns:
        dict: Mapped data matching AdvancedProtection model schema
    """
    # Extract values with proper defaults and type conversions
    global_attack_score = security_data.get("global_attack_score", 0.0)
    requests_per_minute = security_data.get("requests_per_minute", 0)

    # Handle recent_unique_ips (can be from multiple sources)
    recent_unique_ips = (
        security_data.get("recent_unique_ips", 0)
        or security_data.get("unique_ips_last_hour", 0)
        or 0
    )

    # Handle blocked_fingerprints (ensure it's an integer)
    blocked_fingerprints = security_data.get("blocked_fingerprints", 0)
    if isinstance(blocked_fingerprints, str):
        # Convert "unknown" to 0
        blocked_fingerprints = 0
    elif not isinstance(blocked_fingerprints, int):
        blocked_fingerprints = 0

    # Handle storage_type (map from storage_backend)
    storage_type = (
        security_data.get("storage_type")
        or security_data.get("storage_backend")
        or "memory"
    )

    # Optional fields
    avg_response_time_ms = security_data.get("avg_response_time_ms")
    total_requests_24h = security_data.get("total_requests_24h")

    return {
        "global_attack_score": float(global_attack_score),
        "requests_per_minute": int(requests_per_minute),
        "recent_unique_ips": int(recent_unique_ips),
        "blocked_fingerprints": int(blocked_fingerprints),
        "storage_type": str(storage_type),
        "avg_response_time_ms": avg_response_time_ms,
        "total_requests_24h": total_requests_24h,
    }


@router.get(
    "/status",
    tags=["Security"],
    response_model=SecurityStatusResponse,
    **status_metadata,
)
async def get_security_status(
    admin_id: str = Depends(verify_admin_jwt_or_api_key) if not settings.enable_public_status else None,
) -> SecurityStatusResponse:
    """
    **Advanced Security System Status**

    Provides comprehensive real-time status of the multi-layer security protection
    system including threat detection, rate limiting, and attack prevention measures.
    F
    **Security Layers Monitored:**
    - **Rate Limiting**: Request frequency controls per IP
    - **Attack Detection**: Sophisticated attack pattern recognition
    - **Behavioral Analysis**: User behavior anomaly detection
    - **Request Fingerprinting**: Unique request identification and tracking
    - **IP Switching Detection**: Rapid IP change pattern detection
    - **File Duplication Prevention**: Duplicate upload detection
    - **Global Threat Scoring**: Overall attack probability assessment

    **Status Categories:**
    - **active**: Security system fully operational
    - **not_initialized**: Security components not ready
    - **degraded**: Partial security functionality

    **Returns:**
        dict: Complete security status with:
        - Overall system health and operational status
        - List of active protection features
        - Current threat levels and attack scores
        - Storage backend status and performance
        - Timestamp for status validity

    **Use Cases:**
        - Real-time security monitoring
        - Threat level assessment
        - Security audit and compliance
        - System health verification
    """
    # Import at runtime to get the latest reference
    from ..core.advanced_rate_limiting import get_rate_limiter

    advanced_rate_limiter = get_rate_limiter()

    if advanced_rate_limiter is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=SecurityErrorResponse(
                detail="Rate limiter not initialized",
                error_code="RATE_LIMITER_NOT_INITIALIZED",
                service_status="not_initialized",
            ).model_dump(),
        )

    # Use async method if available, otherwise fallback
    try:
        if (
            hasattr(advanced_rate_limiter, "_storage_initialized")
            and advanced_rate_limiter._storage_initialized
        ):
            security_data = await advanced_rate_limiter.get_security_status_async()
        else:
            security_data = advanced_rate_limiter.get_security_status()

        # Map the security data to match AdvancedProtection model
        advanced_protection = _map_security_data_to_model(security_data)

    except Exception as e:
        logger.error("Failed to get security status: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=SecurityErrorResponse(
                detail=f"Failed to get security status: {str(e)}",
                error_code="SECURITY_STATUS_ERROR",
                service_status="error",
            ).model_dump(),
        )

    return SecurityStatusResponse(
        service="Pneumonia Detection API",
        security_status="active",
        timestamp=datetime.now().isoformat(),
        advanced_protection=advanced_protection,
        protection_features=[
            "Multi-layer Rate Limiting (In-Memory)",
            "IP Switching Attack Detection (In-Memory)",
            "Request Fingerprinting (In-Memory)",
            "Behavioral Analysis (In-Memory)",
            "Global Attack Scoring (In-Memory)",
            "Duplicate File Detection (In-Memory)",
            "Persistent Storage (In-Memory)",
            "Single-instance Optimized (In-Memory)",
        ],
    )
