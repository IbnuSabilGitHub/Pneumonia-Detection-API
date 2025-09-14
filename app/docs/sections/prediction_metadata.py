from typing import Dict, Any
from ...models.error_codes import ErrorCode
from ...models.prediction_schemas import (
    PredictionResponse,
    PredictionValidationErrorResponse,
    PredictionBadRequestResponse, 
    PredictionConflictResponse,
    PredictionPayloadTooLargeResponse,
    PredictionRateLimitResponse,
    PredictionServiceUnavailableResponse,
    PredictionInternalServerErrorResponse
)
from ...docs.base_builder import build_response

class PredictionMetadata:
    """Metadata for documentations prediction penumonia endpoint."""
    
    @staticmethod
    def get_title() -> str:
        return "<h2>🤖 Pneumonia Detection from Chest X-ray</h2>"
    
    @staticmethod
    def get_description() -> str:
        """Get prediction endpoint description."""
        return """
<p>Advanced AI analysis of chest X-ray images to detect pneumonia with high accuracy and confidence scoring.</p>
"""

    @staticmethod
    def get_inpurt_requirements() -> str:
        return """
# <h3>📋 <strong>Input Requirements</strong></h3>

<ul>
<li><strong>File Format</strong>: JPG, JPEG, or PNG</li>
<li><strong>File Size</strong>: Maximum 10MB</li>
<li><strong>Image Type</strong>: Chest X-ray (frontal view recommended)</li>
<li><strong>Quality</strong>: Clear, well-exposed medical images</li>
</ul>
"""

    @staticmethod
    def get_ai_models_available() -> str:
        return """"
<h3>🤖 <strong>AI Models Available</strong></h3>

<ol>
<li><strong>Standard Model</strong> (<code>standard</code>)
<ul>
<li>Fast inference (~200ms)</li>
<li>Baseline CNN architecture</li>
<li>Good for high-volume processing</li>
</ul>
</li>
<li><strong>EfficientNet-B0</strong> (<code>efficientnet_b0</code>)
<ul>
<li>Higher accuracy (~300ms)</li>
<li>Advanced transfer learning</li>
<li>Recommended for critical analysis</li>
</ul>
</li>
</ol>
"""

    @staticmethod
    def get_response_details() -> str:
        return """

<h3>📊 <strong>Response Details</strong></h3>

<ul>
<li><strong>Prediction</strong>: NORMAL or PNEUMONIA classification</li>
<li><strong>Confidence</strong>: Score from 0.0 to 1.0 (higher = more confident)</li>
<li><strong>Probabilities</strong>: Individual class probabilities</li>
<li><strong>Medical Recommendation</strong>: Contextual guidance based on results</li>
<li><strong>Model Info</strong>: Version and type used for analysis</li>
</ul>
"""

    @staticmethod
    def get_security_validation() -> str:
        return """
<h3>🛡️ <strong>Security Validation</strong></h3>

<ul>
<li><strong>Rate Limiting</strong>: 5 requests per minute per IP</li>
<li><strong>File Validation</strong>: Automatic size and format checking</li>
<li><strong>Content Analysis</strong>: AI-powered image validation</li>
<li><strong>Duplicate Detection</strong>: Prevents repeated identical uploads</li>
<li><strong>Request Logging</strong>: Comprehensive audit trail</li>
</ul>
"""

    @staticmethod
    def get_important_medical_disclaimer() -> str:
        return """
<h3>⚠️ <strong>Important Medical Disclaimer</strong></h3>

<p><strong>This AI system is for educational and research purposes only.</strong><br>
Results should NEVER replace professional medical diagnosis.<br>
Always consult qualified healthcare professionals for medical decisions.</p>
"""

    @staticmethod
    def get_best_practices() -> str:
        return """
<h3>💡 <strong>Best Practices</strong></h3>

<ul>
<li>Use high-quality, clear chest X-ray images</li>
<li>Ensure proper image orientation (frontal view)</li>
<li>Consider using EfficientNet-B0 for critical analysis</li>
<li>Review confidence scores and recommendations</li>
<li>Always validate results with medical professionals</li>
</ul>
"""
    @staticmethod
    def example_usage() -> str:
        return """
<h3>🚀 <strong>Example Usage</strong></h3>

<pre><code>curl -X POST "http://localhost:8000/pneumonia/predict" \\
     -H "Content-Type: multipart/form-data" \\
     -F "file=@chest_xray.jpg"
</code></pre>
"""

    @staticmethod
    def get_response_description() -> str:
        return "Detailed pneumonia detection results with confidence scores"
    
    @staticmethod
    def get_200_response() -> Dict[str, Any]:
        return build_response(
        description= "Prediction successful",
        model= PredictionResponse,
        example= {
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
    )


    @staticmethod
    def get_400_response() -> Dict[str, Any]:
        return build_response(
            description="Invalid input (file format, size, or content)",
            model=PredictionBadRequestResponse,
            examples={
                "invalid_format": {
                    "summary": "Invalid file format",
                    "value": {
                        "detail": "Unsupported file type. Allowed: jpg, jpeg, png",
                        "error_code": ErrorCode.INVALID_FILE_FORMAT,
                        "timestamp": "2025-09-13T10:30:00.000Z",
                    },
                },
                "invalid_content": {
                    "summary": "Invalid image content",
                    "value": {
                        "detail": "Image does not appear to be a valid chest X-ray",
                        "error_code": ErrorCode.INVALID_IMAGE_CONTENT,
                        "timestamp": "2025-09-13T10:30:00.000Z",
                    },
                },
                "no_file_provided": {
                    "summary": "No file in request",
                    "value": {
                        "detail": "No file provided",
                        "error_code": ErrorCode.MISSING_FILE,
                        "timestamp": "2025-09-13T10:30:00.000Z",
                    },
                },
                "image_validation_failed": {
                    "summary": "Image validation error",
                    "value": {
                        "detail": "Image file is corrupted or unreadable",
                        "error_code": ErrorCode.IMAGE_VALIDATION_ERROR,
                        "timestamp": "2025-09-13T10:30:00.000Z",
                    },
                },
            },
        )
    
    @staticmethod
    def get_409_response() -> Dict[str, Any]:
        return build_response(
            description="Duplicate file detected",
            model=PredictionConflictResponse,
            example={
                "summary": "Duplicate upload",
                "value": {
                    "detail": "Duplicate file detected. Please wait before uploading the same image again.",
                    "error_code": ErrorCode.DUPLICATE_FILE,
                    "retry_after": 60,
                    "file_hash": "a1b2c3d4",
                    "timestamp": "2025-09-13T10:30:00.000Z",
                },
            }
        )
    
    @staticmethod
    def get_413_response() -> Dict[str, Any]:
        return build_response(
            description="File too large",
            model=PredictionPayloadTooLargeResponse,
            example={
                "summary": "Uploaded file exceeds limit",
                "value": {
                    "detail": "File size exceeds limit of 10.0 MB",
                    "error_code": ErrorCode.FILE_TOO_LARGE,
                    "max_size_mb": 10.0,
                    "actual_size_mb": 15.2,
                    "timestamp": "2025-09-13T10:30:00.000Z",
                },
            }
        )
        
    @staticmethod
    def get_422_response() -> Dict[str, Any]:
        return build_response(
            description="Validation Error",
            model=PredictionValidationErrorResponse,
            examples={
                "missing_file": {
                    "summary": "No file provided",
                    "value": {
                        "detail": [
                            {"loc": ["body", "file"], "msg": "Field required", "type": "missing"}
                        ]
                    },
                },
                "invalid_content_type": {
                    "summary": "Invalid Content-Type header",
                    "value": {
                        "detail": [
                            {"loc": ["body", "file"], "msg": "Expected multipart/form-data", "type": "value_error"}
                        ]
                    },
                },
            }
        )
        
    @staticmethod
    def get_429_response() -> Dict[str, Any]:
        return build_response(
            "Rate limit exceeded",
            model=PredictionRateLimitResponse,
            example={
                "summary": "Too many requests",
                "value": {
                    "detail": "Rate limit exceeded: 5 per minute",
                    "error_code": ErrorCode.RATE_LIMIT_EXCEEDED,
                    "limit": "5 per minute",
                    "retry_after": 60,
                    "client_ip": "192.168.1.1",
                    "timestamp": "2025-09-13T10:30:00.000Z",
                },
            },
        )
        
    @staticmethod
    def get_503_response() -> Dict[str, Any]:
        return build_response(
            description="Service unavailable (model not loaded)",
            model= PredictionServiceUnavailableResponse,
            examples= {
                "service_unavailable": {
                    "summary": "Prediction service not available",
                    "value": {
                        "detail": "Prediction service is not available",
                        "error_code": ErrorCode.SERVICE_UNAVAILABLE,
                        "service_status": "not_initialized",
                        "retry_after": 30,
                        "timestamp": "2025-09-13T10:30:00.000Z"
                    }
                },
                "model_not_loaded": {
                    "summary": "AI model not loaded",
                    "value": {
                        "detail": "AI model is not loaded or failed to initialize",
                        "error_code": ErrorCode.MODEL_NOT_LOADED,
                        "service_status": "model_error",
                        "retry_after": 60,
                        "timestamp": "2025-09-13T10:30:00.000Z"
                    }
                }
            }
        )
        
    @staticmethod
    def get_500_response() -> Dict[str, Any]:
        return build_response(
            description= "Internal server error",
            model= PredictionInternalServerErrorResponse,
            examples= {
                "generic_error": {
                    "summary": "Generic server error",
                    "value": {
                        "detail": "Internal server error",
                        "error_code": ErrorCode.INTERNAL_SERVER_ERROR,
                        "error_id": "err_abc123def456",
                        "timestamp": "2025-09-13T10:30:00.000Z"
                    }
                },
                "prediction_failed": {
                    "summary": "Prediction processing failed",
                    "value": {
                        "detail": "Failed to process image",
                        "error_code": ErrorCode.PREDICTION_FAILED,
                        "error_id": "err_pred789xyz",
                        "timestamp": "2025-09-13T10:30:00.000Z"
                    }
                }
            }
        )
        
    @classmethod
    def get_full_description(cls) -> str:
        """Get complete API description by combining all sections"""
        sections = [
            cls.get_title(),
            cls.get_description(),
            cls.get_inpurt_requirements(),
            cls.get_ai_models_available(),
            cls.get_response_details(),
            cls.get_security_validation(),
            cls.get_important_medical_disclaimer(),
            cls.get_best_practices(),
            cls.example_usage()
        ]
        return "".join(sections)
    
    @classmethod
    def get_responses(cls) -> Dict[int, Dict[str, Any]]:
        """get all response descriptions"""
        return {
            200: cls.get_200_response(),
            400: cls.get_400_response(),
            409: cls.get_409_response(),
            413: cls.get_413_response(),
            422: cls.get_422_response(),
            429: cls.get_429_response(),
            503: cls.get_503_response(),
            500: cls.get_500_response(),
        }
        
    @classmethod
    def get_metadata(cls) -> Dict[str, Any]:
        """Get complete metadata for FastAPI endpoint configuration."""
        return {
            "summary": "AI-Powered Pneumonia Detection",
            "description": cls.get_full_description(),
            "response_description": cls.get_response_description(),
            "responses": cls.get_responses()
        }
        
    
    