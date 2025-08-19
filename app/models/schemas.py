"""
Pydantic models for request/response validation.
"""
from pydantic import BaseModel, Field
from typing import Dict, Optional


class PredictionResponse(BaseModel):
    """Response model for pneumonia prediction."""
    
    prediction: str = Field(..., description="Predicted class (NORMAL or PNEUMONIA)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Prediction confidence score")
    probabilities: Dict[str, float] = Field(..., description="Class probabilities")
    medical_recommendation: str = Field(..., description="Medical recommendation based on prediction")
    model_version: str = Field(default="v1.0", description="Model version used for prediction")
    disclaimer: str = Field(
        default="This model is for educational purposes only. Consult a healthcare professional for medical advice.",
        description="Medical disclaimer"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "prediction": "NORMAL",
                "confidence": 0.87,
                "probabilities": {
                    "NORMAL": 0.87,
                    "PNEUMONIA": 0.13
                },
                "medical_recommendation": "✅ Normal results - maintain regular health checkups",
                "model_version": "v1.0",
                "disclaimer": "This model is for educational purposes only. Consult a healthcare professional for medical advice."
            }
        }


class HealthResponse(BaseModel):
    """Response model for health check endpoint."""
    
    status: str = Field(..., description="Service health status")
    model_loaded: bool = Field(..., description="Whether the ML model is loaded")
    version: str = Field(..., description="API version")
    uptime: Optional[float] = Field(None, description="Service uptime in seconds")
    
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
