"""
Security Status Endpoint
"""
from datetime import datetime
from fastapi import APIRouter
from ..docs.status import StatusMetadata

from ..core.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()

status_metadata = StatusMetadata.get_metadata()


@router.get("/status", **status_metadata)
async def get_security_status():
    """
    **🛡️ Advanced Security System Status**
    
    Provides comprehensive real-time status of the multi-layer security protection
    system including threat detection, rate limiting, and attack prevention measures.
    
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