"""
Security API endpoints.
"""
from datetime import datetime
from fastapi import APIRouter

from ..core.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()


@router.get("/status")
async def get_security_status():
    """
    Get enhanced security status with advanced protection info and in-memory storage.
    
    Returns:
        Security status including active protections and statistics
    """
    # Import at runtime to get the latest reference
    from ..core.advanced_rate_limiting import get_rate_limiter
    advanced_rate_limiter = get_rate_limiter()
    
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
            "Multi-layer Rate Limiting (In-Memory)",
            "IP Switching Attack Detection (In-Memory)", 
            "Request Fingerprinting (In-Memory)",
            "Behavioral Analysis (In-Memory)",
            "Global Attack Scoring (In-Memory)",
            "Duplicate File Detection (In-Memory)",
            "Persistent Storage (In-Memory)",
            "Single-instance Optimized (In-Memory)"
        ]
    }


@router.get("/stats")
async def get_security_stats():
    """
    Get detailed security statistics with in-memory storage support.
    
    Returns:
        Detailed security statistics and metrics
    """
    # Import at runtime to get the latest reference
    from ..core.advanced_rate_limiting import get_rate_limiter
    advanced_rate_limiter = get_rate_limiter()
    
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
