"""
Security API endpoints.
"""
from datetime import datetime
from fastapi import APIRouter

from ..core.logger import get_logger
from ..core.advanced_rate_limiting import advanced_rate_limiter

logger = get_logger(__name__)
router = APIRouter()


@router.get("/status")
async def get_security_status():
    """
    Get enhanced security status with advanced protection info and Redis backend.
    
    Returns:
        Security status including active protections and statistics
    """
    if advanced_rate_limiter is None:
        return {
            "service": "Pneumonia Detection API",
            "security_status": "not_initialized",
            "timestamp": datetime.now().isoformat(),
            "error": "Rate limiter not initialized"
        }
    
    # Use async method if available, otherwise fallback
    try:
        if hasattr(advanced_rate_limiter, '_storage_initialized') and advanced_rate_limiter._storage_initialized:
            status = await advanced_rate_limiter.get_security_status_async()
        else:
            status = advanced_rate_limiter.get_security_status()
    except Exception as e:
        logger.error(f"Failed to get security status: {e}")
        status = {"error": str(e)}
    
    return {
        "service": "Pneumonia Detection API",
        "security_status": "active",
        "timestamp": datetime.now().isoformat(),
        "advanced_protection": status,
        "protection_features": [
            "Multi-layer Rate Limiting (Redis)",
            "IP Switching Attack Detection (Redis)", 
            "Request Fingerprinting (Redis)",
            "Behavioral Analysis (Redis)",
            "Global Attack Scoring (Redis)",
            "Duplicate File Detection (Redis)",
            "Persistent Storage (Redis)",
            "Multi-instance Support (Redis)"
        ]
    }


@router.get("/stats")
async def get_security_stats():
    """
    Get detailed security statistics with Redis backend support.
    
    Returns:
        Detailed security statistics and metrics
    """
    if advanced_rate_limiter is None:
        return {
            "error": "Rate limiter not initialized",
            "timestamp": datetime.now().isoformat()
        }
    
    try:
        # Use async method if available
        if hasattr(advanced_rate_limiter, '_storage_initialized') and advanced_rate_limiter._storage_initialized:
            status = await advanced_rate_limiter.get_security_status_async()
        else:
            status = advanced_rate_limiter.get_security_status()
    except Exception as e:
        logger.error(f"Failed to get security stats: {e}")
        return {
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }
    
    return {
        "security_metrics": status,
        "timestamp": datetime.now().isoformat(),
        "interpretation": {
            "attack_score": {
                "value": status["global_attack_score"],
                "level": (
                    "LOW" if status["global_attack_score"] < 0.3 else
                    "MEDIUM" if status["global_attack_score"] < 0.7 else
                    "HIGH"
                ),
                "description": "Global attack probability score (0.0-1.0)"
            },
            "request_rate": {
                "value": status["requests_per_minute"],
                "description": "Total requests in the last minute"
            },
            "unique_ips": {
                "value": status["recent_unique_ips"],
                "description": "Number of unique IP addresses in recent activity"
            },
            "blocked_count": {
                "value": status["blocked_fingerprints"],
                "description": "Number of currently blocked request fingerprints"
            }
        }
    }
