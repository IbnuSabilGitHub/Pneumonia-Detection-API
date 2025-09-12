"""
Pydantic models for request/response validation.
"""
from pydantic import BaseModel, Field
from typing import Dict, Optional, List


class PredictionResponse(BaseModel):
    """
    **🔬 AI Pneumonia Detection Response**
    
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
        description="🎯 AI classification result",
        example="NORMAL",
        pattern="^(NORMAL|PNEUMONIA)$"
    )
    confidence: float = Field(
        ..., 
        ge=0.0, 
        le=1.0, 
        description="📊 AI confidence score (0.0=uncertain, 1.0=highly confident)",
        example=0.92
    )
    probabilities: Dict[str, float] = Field(
        ..., 
        description="📈 Individual class probabilities breakdown",
        example={"NORMAL": 0.92, "PNEUMONIA": 0.08}
    )
    medical_recommendation: str = Field(
        ..., 
        description="💡 Contextual medical guidance based on AI results",
        example="✅ Normal results - maintain regular health checkups"
    )
    model_version: str = Field(
        default="v1.0", 
        description="🤖 AI model version used for analysis",
        example="v1.0"
    )
    model_type: str = Field(
        ...,
        description="🧠 Specific AI model architecture (standard or efficientnet_b0)",
        example="standard",
        pattern="^(standard|efficientnet_b0)$"
    )
    disclaimer: str = Field(
        default="This model is for educational purposes only. Consult a healthcare professional for medical advice.",
        description="⚠️ Important medical disclaimer",
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
                "medical_recommendation": "✅ Normal results - maintain regular health checkups",
                "model_version": "v1.0",
                "model_type": "standard",
                "disclaimer": "This model is for educational purposes only. Consult a healthcare professional for medical advice."
            }
        }


class HealthResponse(BaseModel):
    """
    **🏥 Service Health Status Response**
    
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
        description="🚦 Overall service health status",
        example="healthy",
        pattern="^(healthy|partial|unhealthy)$"
    )
    model_loaded: bool = Field(
        ..., 
        description="🤖 AI model loading and readiness status",
        example=True
    )
    version: str = Field(
        ..., 
        description="📌 Current API version identifier",
        example="1.0.0"
    )
    uptime: Optional[float] = Field(
        None, 
        description="⏱️ Service uptime since last restart (seconds)",
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
    **🛡️ Advanced Security System Status Response**
    
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
        description="🏥 API service name",
        example="Pneumonia Detection API"
    )
    security_status: str = Field(
        ..., 
        description="🛡️ Overall security system status",
        example="active",
        pattern="^(active|not_initialized|degraded)$"
    )
    timestamp: str = Field(
        ..., 
        description="⏰ Status timestamp in ISO format",
        example="2025-09-10T10:30:00.000Z"
    )
    advanced_protection: Dict = Field(
        ..., 
        description="📊 Detailed security metrics and protection data",
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
        description="🔒 List of active security protection features",
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
        description="❌ Error message if security system has issues",
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
    **❌ Security System Error Response**
    
    Error response when the security system encounters issues
    or is not properly initialized.
    """
    
    service: str = Field(
        ..., 
        description="🏥 API service name",
        example="Pneumonia Detection API"
    )
    security_status: str = Field(
        ..., 
        description="🛡️ Security system status indicating error state",
        example="not_initialized"
    )
    timestamp: str = Field(
        ..., 
        description="⏰ Error timestamp in ISO format",
        example="2025-09-10T10:30:00.000Z"
    )
    error: str = Field(
        ..., 
        description="❌ Detailed error message",
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


class ErrorResponse(BaseModel):
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
