"""
Pydantic models for prediction schemas with ReDoc compatibility.
"""

from pydantic import BaseModel, Field
from typing import  Optional, List
import time


class BaseErrorResponse(BaseModel):
    """Base error response with common fields for consistency."""
    
    detail: str = Field(
        ..., 
        description="Human-readable error message explaining what went wrong"
    )
    error_code: str = Field(
        ...,
        description="Machine-readable error code for programmatic handling"
    )
    timestamp: str = Field(
        default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime()),
        description="Error occurrence timestamp in ISO 8601 format"
    )


class Probabilities(BaseModel):
    """Individual class probabilities breakdown for pneumonia detection."""
    
    NORMAL: float = Field(
        ..., 
        description="Probability score for normal/healthy chest X-ray",
        ge=0.0, le=1.0,
        example=0.92
    )
    PNEUMONIA: float = Field(
        ..., 
        description="Probability score for pneumonia detection", 
        ge=0.0, le=1.0,
        example=0.08
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "NORMAL": 0.92,
                "PNEUMONIA": 0.08
            }
        }


class ModelInfo(BaseModel):
    """Information about the AI model used for prediction."""
    
    model_type: str = Field(
        ..., 
        description="Type of neural network architecture used",
        example="standard"
    )
    model_version: str = Field(
        ..., 
        description="Version identifier of the trained model",
        example="v1.0"
    )
    architecture: str = Field(
        ..., 
        description="Detailed model architecture description",
        example="CNN-based pneumonia detection model"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "model_type": "standard",
                "model_version": "v1.0",
                "architecture": "CNN-based pneumonia detection model"
            }
        }

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
    - **model_info**: Information about the AI model used
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
    probabilities: Probabilities = Field(
        ..., 
        description="Individual class probabilities breakdown"
    )
    medical_recommendation: str = Field(
        ..., 
        description="Contextual medical guidance based on AI results",
        example="Normal results - maintain regular health checkups"
    )
    model_info: ModelInfo = Field(
        ...,
        description="Information about the AI model used for this prediction"
    )
    disclaimer: str = Field(
        default="This model is for educational purposes only. Consult a healthcare professional for medical advice.",
        description="Important medical disclaimer and usage guidelines",
    )
    
    class Config:
        json_schema_extra = {
            "examples": [
                {
                    "summary": "Normal Chest X-ray",
                    "description": "Normal chest X-ray with high confidence",
                    "value": {
                        "prediction": "NORMAL",
                        "confidence": 0.92,
                        "probabilities": {
                            "NORMAL": 0.92,
                            "PNEUMONIA": 0.08
                        },
                        "medical_recommendation": "Normal results - maintain regular health checkups",
                        "model_info": {
                            "model_type": "standard",
                            "model_version": "v1.0",
                            "architecture": "CNN-based pneumonia detection model"
                        },
                        "disclaimer": "This model is for educational purposes only. Consult a healthcare professional for medical advice."
                    }
                },
                {
                    "summary": "Pneumonia Detection",
                    "description": "Pneumonia detected with high confidence",
                    "value": {
                        "prediction": "PNEUMONIA",
                        "confidence": 0.87,
                        "probabilities": {
                            "NORMAL": 0.13,
                            "PNEUMONIA": 0.87
                        },
                        "medical_recommendation": "Pneumonia detected - seek immediate medical attention",
                        "model_info": {
                            "model_type": "standard",
                            "model_version": "v1.0",
                            "architecture": "CNN-based pneumonia detection model"
                        },
                        "disclaimer": "This model is for educational purposes only. Consult a healthcare professional for medical advice."
                    }
                }
            ]
        }
        
        

class ValidationError(BaseModel):
    """Individual validation error information."""
    
    loc: List[str] = Field(
        ...,
        description="Location of the error in the request",
        example=["body", "file"]
    )
    msg: str = Field(
        ...,
        description="Validation error message",
        example="Field required"
    )
    type: str = Field(
        ...,
        description="Type of validation error",
        example="missing"
    )


class PredictionValidationErrorResponse(BaseModel):
    """
    **Prediction Request Validation Error Response**
    
    Response returned when the prediction request fails validation.
    This includes missing files, invalid content types, or malformed requests.
    """
    
    detail: List[ValidationError] = Field(
        ..., 
        description="List of validation errors with detailed information"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "detail": [
                    {
                        "loc": ["body", "file"],
                        "msg": "Field required",
                        "type": "missing"
                    }
                ]
            }
        }


class PredictionBadRequestResponse(BaseErrorResponse):
    """
    **Prediction Bad Request Error Response**
    
    Response returned when the prediction request has invalid parameters
    such as unsupported file types, invalid image content, or missing data.
    """
    
    class Config:
        json_schema_extra = {
            "example": {
                "detail": "Unsupported file type. Allowed: jpg, jpeg, png",
                "error_code": "INVALID_FILE_FORMAT",
                "timestamp": "2025-09-13T10:30:00.000Z"
            }
        }


class PredictionConflictResponse(BaseErrorResponse):
    """
    **Prediction Conflict Error Response**
    
    Response returned when a duplicate file is detected or there's
    a conflict with the current request state.
    """
    
    retry_after: Optional[int] = Field(
        None,
        description="Suggested wait time in seconds before retry",
        example=60
    )
    
    file_hash: Optional[str] = Field(
        None,
        description="Hash of the uploaded file (first 8 characters)",
        example="a1b2c3d4"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "detail": "Duplicate file detected. Please wait before uploading the same image again.",
                "error_code": "DUPLICATE_FILE",
                "retry_after": 60,
                "file_hash": "a1b2c3d4",
                "timestamp": "2025-09-13T10:30:00.000Z"
            }
        }


class PredictionPayloadTooLargeResponse(BaseErrorResponse):
    """
    **Prediction Payload Too Large Error Response**
    
    Response returned when the uploaded file exceeds the maximum
    allowed file size limit.
    """
    
    max_size_mb: float = Field(
        ...,
        description="Maximum allowed file size in megabytes",
        example=10.0
    )
    
    actual_size_mb: Optional[float] = Field(
        None,
        description="Actual uploaded file size in megabytes",
        example=15.2
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "detail": "File size exceeds limit of 10.0 MB",
                "error_code": "FILE_TOO_LARGE",
                "max_size_mb": 10.0,
                "actual_size_mb": 15.2,
                "timestamp": "2025-09-13T10:30:00.000Z"
            }
        }


class PredictionRateLimitResponse(BaseErrorResponse):
    """
    **Prediction Rate Limit Exceeded Response**
    
    Response returned when the client has exceeded the rate limit
    for prediction requests (5 per minute per IP).
    """
    
    limit: str = Field(
        ...,
        description="Applied rate limit description",
        example="5 per minute"
    )
    
    retry_after: int = Field(
        ...,
        description="Seconds to wait before next request",
        example=60
    )
    
    client_ip: str = Field(
        ...,
        description="Client IP address that exceeded the limit",
        example="192.168.1.1"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "detail": "Rate limit exceeded: 5 per minute",
                "error_code": "RATE_LIMIT_EXCEEDED",
                "limit": "5 per minute",
                "retry_after": 60,
                "client_ip": "192.168.1.1",
                "timestamp": "2025-09-13T10:30:00.000Z"
            }
        }


class PredictionServiceUnavailableResponse(BaseErrorResponse):
    """
    **Prediction Service Unavailable Error Response**
    
    Response returned when the AI prediction service is not available,
    model is not loaded, or there's a server-side issue.
    """
    
    service_status: Optional[str] = Field(
        None,
        description="Current service status",
        example="model_not_loaded"
    )
    
    retry_after: Optional[int] = Field(
        None,
        description="Suggested retry time in seconds",
        example=30
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "detail": "Prediction service is not available",
                "error_code": "SERVICE_UNAVAILABLE",
                "service_status": "not_initialized",
                "retry_after": 30,
                "timestamp": "2025-09-13T10:30:00.000Z"
            }
        }


class PredictionInternalServerErrorResponse(BaseErrorResponse):
    """
    **Prediction Internal Server Error Response**
    
    Response returned when an unexpected server-side error occurs
    during image processing or prediction.
    """
    
    error_id: Optional[str] = Field(
        None,
        description="Unique error identifier for support",
        example="err_abc123def456"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "detail": "Internal server error",
                "error_code": "INTERNAL_SERVER_ERROR",
                "error_id": "err_abc123def456",
                "timestamp": "2025-09-13T10:30:00.000Z"
            }
        }


# Generic Error Response for Prediction
PredictionErrorResponse = PredictionInternalServerErrorResponse