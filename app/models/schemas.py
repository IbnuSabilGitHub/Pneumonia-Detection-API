"""
Pydantic models for request/response validation.
"""
from pydantic import BaseModel, Field
from typing import Dict, Optional, List


class PredictionResponse(BaseModel):
    """
    **AI Pneumonia Detection Response**
    
    Comprehensive response model containing AI analysis results, confidence metrics,
    and medical recommendations for chest X-ray pneumonia detection.
    
    **Fields:**
    - **prediction**: Primary AI classification result
    - **confidence**: Numerical confidence level (0.0-1.0)
    - **probabilities**: Detailed class probability breakdown
    - **medical_recommendation**: Contextual medical guidance
    - **model_version**: AI model version identifier
    - **model_type**: Specific model architecture used
    - **disclaimer**: Important medical disclaimer text
    
    **Medical Disclaimer:**
    All predictions are for educational/research purposes only.
    Never use as substitute for professional medical diagnosis.
    """
    
    prediction: str = Field(
        ..., 
        description="AI classification result",
        example="NORMAL",
        pattern="^(NORMAL|PNEUMONIA)$"
    )
    confidence: float = Field(
        ..., 
        ge=0.0, 
        le=1.0, 
        description="AI confidence score (0.0=uncertain, 1.0=highly confident)",
        example=0.92
    )
    probabilities: Dict[str, float] = Field(
        ..., 
        description="Individual class probabilities breakdown",
        example={"NORMAL": 0.92, "PNEUMONIA": 0.08}
    )
    medical_recommendation: str = Field(
        ..., 
        description="Contextual medical guidance based on AI results",
        example="Normal results - maintain regular health checkups"
    )
    model_version: str = Field(
        default="v1.0", 
        description="AI model version used for analysis",
        example="v1.0"
    )
    model_type: str = Field(
        ...,
        description="Specific AI model architecture (standard or efficientnet_b0)",
        example="standard",
        pattern="^(standard|efficientnet_b0)$"
    )
    disclaimer: str = Field(
        default="This model is for educational purposes only. Consult a healthcare professional for medical advice.",
        description="Important medical disclaimer",
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "prediction": "NORMAL",
                "confidence": 0.92,
                "probabilities": {
                    "NORMAL": 0.92,
                    "PNEUMONIA": 0.08
                },
                "medical_recommendation": "Normal results - maintain regular health checkups",
                "model_version": "v1.0",
                "model_type": "standard",
                "disclaimer": "This model is for educational purposes only. Consult a healthcare professional for medical advice."
            }
        }


class HealthResponse(BaseModel):
    """
    **Service Health Status Response**
    
    Comprehensive health check response providing detailed information about
    API service status, model availability, and system uptime.
    
    **Health Status Levels:**
    - **healthy**: All systems operational, models loaded
    - **partial**: Service running with limitations
    - **unhealthy**: Critical issues detected
    
    **Fields:**
    - **status**: Overall service health indicator
    - **model_loaded**: AI model availability status
    - **version**: Current API version
    - **uptime**: Service runtime in seconds
    """
    
    status: str = Field(
        ..., 
        description="Overall service health status",
        example="healthy",
        pattern="^(healthy|partial|unhealthy)$"
    )
    model_loaded: bool = Field(
        ..., 
        description="AI model loading and readiness status",
        example=True
    )
    version: str = Field(
        ..., 
        description="Current API version identifier",
        example="1.0.0"
    )
    uptime: Optional[float] = Field(
        None, 
        description="Service uptime since last restart (seconds)",
        example=3600.5,
        ge=0.0
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "model_loaded": True,
                "version": "1.0.0",
                "uptime": 3600.5
            }
        }


class SecurityStatusResponse(BaseModel):
    """
    **Advanced Security System Status Response**
    
    Comprehensive security status response providing real-time information about
    the multi-layer security protection system including threat detection,
    rate limiting, and attack prevention measures.
    
    **Security Layers Monitored:**
    - Multi-layer Rate Limiting (In-Memory)
    - IP Switching Attack Detection (In-Memory)
    - Request Fingerprinting (In-Memory)
    - Behavioral Analysis (In-Memory)
    - Global Attack Scoring (In-Memory)
    - Duplicate File Detection (In-Memory)
    
    **Status Categories:**
    - **active**: Security system fully operational
    - **not_initialized**: Security components not ready
    - **degraded**: Partial security functionality
    """
    
    service: str = Field(
        ..., 
        description="API service name",
        example="Pneumonia Detection API"
    )
    security_status: str = Field(
        ..., 
        description="Overall security system status",
        example="active",
        pattern="^(active|not_initialized|degraded)$"
    )
    timestamp: str = Field(
        ..., 
        description="Status timestamp in ISO format",
        example="2025-09-10T10:30:00.000Z"
    )
    advanced_protection: Dict = Field(
        ..., 
        description="Detailed security metrics and protection data",
        example={
            "global_attack_score": 0.15,
            "requests_per_minute": 23,
            "recent_unique_ips": 8,
            "blocked_fingerprints": 2,
            "storage_type": "memory"
        }
    )
    protection_features: List[str] = Field(
        ..., 
        description="List of active security protection features",
        example=[
            "Multi-layer Rate Limiting (In-Memory)",
            "IP Switching Attack Detection (In-Memory)",
            "Request Fingerprinting (In-Memory)",
            "Behavioral Analysis (In-Memory)",
            "Global Attack Scoring (In-Memory)",
            "Duplicate File Detection (In-Memory)",
            "Persistent Storage (In-Memory)",
            "Single-instance Optimized (In-Memory)"
        ]
    )
    error: Optional[str] = Field(
        None, 
        description="Error message if security system has issues",
        example="Rate limiter not initialized"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "service": "Pneumonia Detection API",
                "security_status": "active",
                "timestamp": "2025-09-10T10:30:00.000Z",
                "advanced_protection": {
                    "global_attack_score": 0.15,
                    "requests_per_minute": 23,
                    "recent_unique_ips": 8,
                    "blocked_fingerprints": 2,
                    "storage_type": "memory",
                    "avg_response_time_ms": 85,
                    "total_requests_24h": 1847
                },
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
        }


class SecurityErrorResponse(BaseModel):
    """
    **Security System Error Response**
    
    Error response when the security system encounters issues
    or is not properly initialized.
    """
    
    service: str = Field(
        ..., 
        description="API service name",
        example="Pneumonia Detection API"
    )
    security_status: str = Field(
        ..., 
        description="Security system status indicating error state",
        example="not_initialized"
    )
    timestamp: str = Field(
        ..., 
        description="Error timestamp in ISO format",
        example="2025-09-10T10:30:00.000Z"
    )
    error: str = Field(
        ..., 
        description="Detailed error message",
        example="Rate limiter not initialized"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "service": "Pneumonia Detection API",
                "security_status": "not_initialized",
                "timestamp": "2025-09-10T10:30:00.000Z",
                "error": "Rate limiter not initialized"
            }
        }
        
class SecurityStatsResponse(BaseModel):
    """
    **� Advanced Security Analytics Response**
    
    Comprehensive security statistics response providing detailed analysis of 
    security events, threat patterns, and protection effectiveness over time.
    
    **Analytics Categories:**
    - **Threat Analysis**: Global attack probability scoring and interpretation
    - **Traffic Analytics**: Real-time request monitoring and IP tracking
    - **Protection Effectiveness**: Blocked requests and success rates
    - **System Performance**: Response times and resource utilization
    
    **Threat Level Interpretation:**
    - **LOW** (0.0-0.3): Normal operations, standard monitoring
    - **MEDIUM** (0.3-0.7): Elevated vigilance, possible threats  
    - **HIGH** (0.7-1.0): Active attacks, enhanced protection mode
    """
    
    security_metrics: Dict = Field(
        ..., 
        description="📈 Raw security metrics and measurements",
        example={
            "global_attack_score": 0.25,
            "requests_per_minute": 45,
            "recent_unique_ips": 12,
            "blocked_fingerprints": 3,
            "storage_type": "memory",
            "avg_response_time_ms": 85,
            "total_requests_24h": 2847
        }
    )
    
    timestamp: str = Field(
        ..., 
        description="⏰ Analytics timestamp in ISO format",
        example="2025-09-12T10:30:00.000Z"
    )
    
    interpretation: Dict = Field(
        ..., 
        description="🎯 Human-readable interpretation of security metrics",
        example={
            "attack_score": {
                "value": 0.25,
                "level": "LOW",
                "description": "Global attack probability score (0.0-1.0)"
            },
            "request_rate": {
                "value": 45,
                "description": "Total requests in the last minute"
            },
            "unique_ips": {
                "value": 12,
                "description": "Number of unique IP addresses in recent activity"
            },
            "blocked_count": {
                "value": 3,
                "description": "Number of currently blocked request fingerprints"
            }
        }
    )
    
    analytics_summary: Optional[Dict] = Field(
        None,
        description="📊 Optional summary analytics for extended periods",
        example={
            "hourly_trend": "increasing",
            "daily_average": 42.5,
            "threat_level_changes": 3
        }
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "security_metrics": {
                    "global_attack_score": 0.25,
                    "requests_per_minute": 45,
                    "recent_unique_ips": 12,
                    "blocked_fingerprints": 3,
                    "storage_type": "memory",
                    "avg_response_time_ms": 85,
                    "total_requests_24h": 2847
                },
                "timestamp": "2025-09-12T10:30:00.000Z",
                "interpretation": {
                    "attack_score": {
                        "value": 0.25,
                        "level": "LOW",
                        "description": "Global attack probability score (0.0-1.0)"
                    },
                    "request_rate": {
                        "value": 45,
                        "description": "Total requests in the last minute"
                    },
                    "unique_ips": {
                        "value": 12,
                        "description": "Number of unique IP addresses in recent activity"
                    },
                    "blocked_count": {
                        "value": 3,
                        "description": "Number of currently blocked request fingerprints"
                    }
                },
                "analytics_summary": {
                    "hourly_trend": "increasing",
                    "daily_average": 42.5,
                    "threat_level_changes": 3
                }
            }
        }


class SecurityStatsErrorResponse(BaseModel):
    """
    **Security Statistics Error Response**
    
    Error response when security statistics cannot be retrieved due to
    system issues or initialization problems.
    """
    
    error: str = Field(
        ..., 
        description="❌ Detailed error message",
        example="Rate limiter not initialized"
    )
    
    error_code: Optional[str] = Field(
        None,
        description="🏷️ Application-specific error code", 
        example="RATE_LIMITER_NOT_INITIALIZED"
    )
    
    timestamp: str = Field(
        ..., 
        description="⏰ Error timestamp in ISO format",
        example="2025-09-12T10:30:00.000Z"
    )
    
    details: Optional[Dict] = Field(
        None,
        description="📋 Additional error context and debugging information",
        example={
            "component": "rate_limiter",
            "initialization_status": "failed"
        }
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "error": "Rate limiter not initialized",
                "error_code": "RATE_LIMITER_NOT_INITIALIZED",
                "timestamp": "2025-09-12T10:30:00.000Z",
                "details": {
                    "component": "rate_limiter", 
                    "initialization_status": "failed"
                }
            }
        }


class RateLimitErrorResponse(BaseModel):
    """
    **Rate Limit Exceeded Response**
    
    Response returned when API rate limits are exceeded.
    Includes details about the limit and when to retry.
    """
    
    error: str = Field(
        ..., 
        description="Error message",
        example="Rate limit exceeded"
    )
    
    message: str = Field(
        ..., 
        description="Detailed reason for rate limiting",
        example="Too many requests from IP address"
    )
    
    client_ip: str = Field(
        ..., 
        description="Client IP address that exceeded the limit",
        example="192.168.1.1"
    )
    
    endpoint: str = Field(
        ..., 
        description="Endpoint that was rate limited",
        example="/pneumonia/predict"
    )
    
    timestamp: float = Field(
        ..., 
        description="Unix timestamp when rate limit was triggered",
        example=1694537400.123
    )
    
    details: Dict = Field(
        ..., 
        description="Additional rate limiting details",
        example={
            "rate_limit": "5 per minute",
            "retry_after": 60
        }
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "error": "Rate limit exceeded",
                "message": "Too many requests from IP address",
                "client_ip": "192.168.1.1",
                "endpoint": "/pneumonia/predict", 
                "timestamp": 1694537400.123,
                "details": {
                    "rate_limit": "5 per minute",
                    "retry_after": 60
                }
            }
        }


class SecurityStatResponse(BaseModel):
    """Response model for error cases."""
    
    detail: str = Field(..., description="Error message")
    error_code: Optional[str] = Field(None, description="Application-specific error code")
    timestamp: Optional[str] = Field(None, description="Error timestamp")
    
    class Config:
        json_schema_extra = {
            "example": {
                "detail": "File size exceeds limit of 10.0 MB",
                "error_code": "FILE_TOO_LARGE",
                "timestamp": "2025-08-19T10:30:00Z"
            }
        }
