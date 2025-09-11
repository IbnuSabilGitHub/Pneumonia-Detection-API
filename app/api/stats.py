"""
Security Statistics Endpoint
"""

from datetime import datetime
from fastapi import APIRouter
from ..docs.stat_metadata import StatMetadata

from ..core.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()
stat_metadata = StatMetadata.get_metadata()

@router.get("/stats",tags=["Security"],**stat_metadata)
async def get_security_stats():
    """
    **📊 Comprehensive Security Analytics Dashboard**
    
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