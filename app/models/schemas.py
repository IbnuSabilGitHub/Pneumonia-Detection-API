"""
Pydantic models for request/response validation.
"""
from pydantic import BaseModel, Field
from typing import Dict, Optional


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
    """Response model for security status endpoint."""
    
    client_ip: str = Field(..., description="Client IP address")
    requests_in_last_minute: int = Field(..., description="Number of requests in the last minute")
    rate_limit: str = Field(..., description="Rate limit configuration")
    is_blocked: bool = Field(..., description="Whether the IP is currently blocked")
    cache_entries: int = Field(..., description="Number of cached file hashes")
    security_features: list = Field(..., description="List of enabled security features")
    
    class Config:
        json_schema_extra = {
            "example": {
                "client_ip": "192.168.1.1",
                "requests_in_last_minute": 3,
                "rate_limit": "5 requests per minute",
                "is_blocked": False,
                "cache_entries": 15,
                "security_features": [
                    "Rate Limiting (5/min per IP)",
                    "File Size Validation (10MB max)",
                    "File Type Validation (JPG, JPEG, PNG)",
                    "Image Content Validation",
                    "Duplicate Detection (5min cache)"
                ]
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
