"""
Security Statistics Endpoint
"""

from datetime import datetime

from fastapi import APIRouter, HTTPException, status

from ..core.logger import get_logger
from ..docs.sections.stat_metadata import StatMetadata
from ..models.security_schemes import SecurityStatsErrorResponse, SecurityStatsResponse

logger = get_logger(__name__)
router = APIRouter()
stat_metadata = StatMetadata.get_metadata()


@router.get(
    "/stats", tags=["Security"], response_model=SecurityStatsResponse, **stat_metadata
)
async def get_security_stats() -> SecurityStatsResponse:
    """
    **omprehensive Security Analytics Dashboard**

    Provides detailed statistical analysis and metrics of the security protection
    system including threat patterns, attack detection results, and system performance.

    **Metrics Categories:**

    1. **Threat Analysis**
       - Global attack probability scoring (0.0-1.0 scale)
       - Attack pattern recognition and classification
       - Threat level interpretation and recommendations

    2. **Traffic Analytics**
       - Real-time request rate monitoring
       - Unique IP address tracking and analysis
       - Request pattern and behavioral analysis

    3. **Protection Effectiveness**
       - Blocked request fingerprint statistics
       - Attack prevention success rates
       - False positive/negative analysis

    4. **System Performance**
       - Security system response times
       - Storage backend utilization
       - Processing efficiency metrics

    **Threat Level Interpretation:**
    - **LOW** (0.0-0.3): Normal operations, standard monitoring
    - **MEDIUM** (0.3-0.7): Elevated vigilance, possible threats
    - **HIGH** (0.7-1.0): Active attacks, enhanced protection mode

    **Returns:**
        dict: Comprehensive security analytics including:
        - Raw security metrics and measurements
        - Human-readable interpretation of threat levels
        - Actionable insights for security decisions
        - Performance benchmarks and system health

    **Use Cases:**
        - Security operations center (SOC) monitoring
        - Threat intelligence and pattern analysis
        - Performance optimization and tuning
        - Compliance and audit reporting
        - Incident response and investigation
    """
    # Import at runtime to get the latest reference
    from ..core.advanced_rate_limiting import get_rate_limiter

    advanced_rate_limiter = get_rate_limiter()

    if advanced_rate_limiter is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=SecurityStatsErrorResponse(
                detail="Rate limiter not initialized",
                error_code="RATE_LIMITER_NOT_INITIALIZED",
                component="rate_limiter",
                initialization_status="failed",
            ).model_dump(),
        )

    try:
        # Use async method if available
        if (
            hasattr(advanced_rate_limiter, "_storage_initialized")
            and advanced_rate_limiter._storage_initialized
        ):
            security_metrics = await advanced_rate_limiter.get_security_status_async()
        else:
            security_metrics = advanced_rate_limiter.get_security_status()

    except Exception as e:
        logger.error("Failed to get security stats: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=SecurityStatsErrorResponse(
                detail=f"Failed to get security stats: {str(e)}",
                error_code="SECURITY_STATS_RETRIEVAL_ERROR",
                component="security_analytics",
                operation="get_security_status",
            ).model_dump(),
        )

    # Generate threat level interpretation
    attack_score = security_metrics.get("global_attack_score", 0.0)
    threat_level = (
        "LOW" if attack_score < 0.3 else "MEDIUM" if attack_score < 0.7 else "HIGH"
    )

    # Build comprehensive interpretation
    interpretation = {
        "attack_score": {
            "value": attack_score,
            "level": threat_level,
            "description": "Global attack probability score (0.0-1.0)",
        },
        "request_rate": {
            "value": security_metrics.get("requests_per_minute", 0),
            "description": "Total requests in the last minute",
        },
        "unique_ips": {
            "value": security_metrics.get("recent_unique_ips", 0),
            "description": "Number of unique IP addresses in recent activity",
        },
        "blocked_count": {
            "value": security_metrics.get("blocked_fingerprints", 0),
            "description": "Number of currently blocked request fingerprints",
        },
    }

    # Optional analytics summary for enhanced insights
    analytics_summary = None
    if "total_requests_24h" in security_metrics:
        analytics_summary = {
            "daily_total": security_metrics.get("total_requests_24h", 0),
            "threat_level": threat_level,
            "storage_backend": security_metrics.get("storage_type", "unknown"),
        }

    # Determine storage type if available
    storage_type = (
        security_metrics.get("storage_backend")
        or security_metrics.get("storage_type")
        or "memory"
    )

    return SecurityStatsResponse(
        service="Pneumonia Detection API",
        security_metrics=security_metrics,
        timestamp=datetime.now().isoformat(),
        status="active",
        storage_type=storage_type,
        interpretation=interpretation,
        analytics_summary=analytics_summary,
    )
