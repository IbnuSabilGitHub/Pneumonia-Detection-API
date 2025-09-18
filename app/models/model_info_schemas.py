"""
pydantic models for model info schemas
"""

from typing import List, Optional

from pydantic import BaseModel, Field

from .base import BaseErrorResponse


class ModelInfoResponse(BaseModel):
    """
    **AI Model Information Response**

    Comprehensive response containing detailed information about the
    currently loaded AI model including configuration, architecture,
    and performance characteristics.
    """

    loaded: bool = Field(..., description="Model loading status", example=True)

    model_path: Optional[str] = Field(
        None,
        description="Path to the loaded ONNX model file",
        example="models/pneumonia_model_efficientnet_b0.onnx",
    )

    input_name: Optional[str] = Field(
        None, description="Model input tensor name", example="input"
    )

    output_name: Optional[str] = Field(
        None, description="Model output tensor name", example="output"
    )

    mean: Optional[float] = Field(
        None,
        description="Normalization mean value used for preprocessing",
        example=0.480,
    )

    std: Optional[float] = Field(
        None,
        description="Normalization standard deviation for preprocessing",
        example=0.237,
    )

    target_size: Optional[List[int]] = Field(
        None,
        description="Expected input image dimensions [height, width]",
        example=[192, 192],
    )

    labels: Optional[List[str]] = Field(
        None, description="Model output class labels", example=["NORMAL", "PNEUMONIA"]
    )

    model_type: Optional[str] = Field(
        None, description="Model architecture type", example="efficientnet_b0"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "loaded": True,
                "model_path": "models/pneumonia_model_efficientnet_b0.onnx",
                "input_name": "input",
                "output_name": "output",
                "mean": 0.480,
                "std": 0.237,
                "target_size": [192, 192],
                "labels": ["NORMAL", "PNEUMONIA"],
                "model_type": "efficientnet_b0",
            }
        }


class ValidationErrorDetail(BaseModel):
    """Individual validation error information."""

    loc: List[str] = Field(
        ...,
        description="Location of the error in the request",
        example=["query", "model"],
    )
    msg: str = Field(
        ...,
        description="Validation error message",
        example="Invalid model type specified",
    )
    type: str = Field(
        ..., description="Type of validation error", example="value_error"
    )


class ModelInfoValidationErrorResponse(BaseModel):
    """
    **Model Info Validation Error Response**

    Response returned when the model info request fails validation.
    """

    detail: List[ValidationErrorDetail] = Field(
        ..., description="List of validation errors with detailed information"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "detail": [
                    {
                        "loc": ["query", "model"],
                        "msg": "Invalid model type specified",
                        "type": "value_error",
                    }
                ]
            }
        }


class ModelInfoNotFoundResponse(BaseErrorResponse):
    """
    **Model Info Not Found Error Response**

    Response returned when a specific model is not found.
    """

    available_models: Optional[List[str]] = Field(
        None,
        description="List of available model types",
        example=["standard", "efficientnet_b0"],
    )

    class Config:
        json_schema_extra = {
            "example": {
                "detail": "Model 'invalid_model' not found",
                "error_code": "MODEL_NOT_FOUND",
                "available_models": ["standard", "efficientnet_b0"],
                "timestamp": "2025-09-13T10:30:00.000Z",
            }
        }


class ModelInfoServiceUnavailableResponse(BaseErrorResponse):
    """
    **Model Info Service Unavailable Error Response**

    Response returned when the model information service is not available.
    """

    service_status: Optional[str] = Field(
        None, description="Current service status", example="not_initialized"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "detail": "Prediction service not available",
                "error_code": "SERVICE_UNAVAILABLE",
                "service_status": "not_initialized",
                "timestamp": "2025-09-13T10:30:00.000Z",
            }
        }


# Generic Error Response for Model Info
ModelInfoErrorResponse = ModelInfoServiceUnavailableResponse
