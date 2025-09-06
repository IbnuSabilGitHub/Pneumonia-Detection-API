"""
Pneumonia prediction API endpoints.
"""
import io
from fastapi import APIRouter, File, UploadFile, HTTPException, status, Request, Depends, Query
from slowapi import Limiter
from slowapi.util import get_remote_address

from ..models.schemas import PredictionResponse
from ..services.prediction import PneumoniaPredictionService
from ..utils.security import get_client_ip, calculate_file_hash, file_hash_cache
from ..utils.validation import (
    validate_file_extension, 
    validate_file_size, 
    validate_image_integrity,
    validate_image_content,
    get_image_stats
)
from ..utils.exceptions import (
    FileValidationError, 
    ImageValidationError, 
    PredictionError
)
from ..core.settings import settings
from ..core.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()

# For backward compatibility with slowapi
limiter = Limiter(key_func=get_remote_address)


def get_prediction_service(
    model: str = Query(
        "standard", 
        description="🤖 Choose AI model for prediction",
        enum=["standard", "efficientnet_b0"],
        example="standard"
    )
) -> PneumoniaPredictionService:
    """
    **🤖 AI Model Service Provider**
    
    Provides access to trained pneumonia detection models with automatic loading and validation.
    
    **Available Models:**
    - `standard`: Baseline CNN architecture (faster inference)
    - `efficientnet_b0`: Advanced transfer learning model (higher accuracy)
    
    **Model Features:**
    - Automatic model loading and caching
    - Model validation and health checks
    - Performance optimization
    - Error handling and fallback mechanisms
    """
    services = {
        "standard": PneumoniaPredictionService(
            model_path=settings.model_path,
            stats_path=settings.model_stats_path
        ),
        "efficientnet_b0": PneumoniaPredictionService(
            model_path=settings.model_path_efficientnet_b0,
            stats_path=settings.model_stats_path_efficientnet_b0
        )
    }
    service = services.get(model)
    if not service:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model '{model}' not found"
        )
    
    # Load the model if not already loaded
    if not service.is_loaded():
        try:
            service.load_model()
        except Exception as e:
            logger.error(f"Failed to load {model} model: {e}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Failed to load {model} model"
            )
    
    return service


@router.post(
    "/predict", 
    response_model=PredictionResponse, 
    tags=["Pneumonia Detection"],
    summary="🔬 AI-Powered Pneumonia Detection",
    description="""
<h2>🤖 Pneumonia Detection from Chest X-ray</h2>

<p>Advanced AI analysis of chest X-ray images to detect pneumonia with high accuracy and confidence scoring.</p>

<h3>📋 <strong>Input Requirements</strong></h3>

<ul>
<li><strong>File Format</strong>: JPG, JPEG, or PNG</li>
<li><strong>File Size</strong>: Maximum 10MB</li>
<li><strong>Image Type</strong>: Chest X-ray (frontal view recommended)</li>
<li><strong>Quality</strong>: Clear, well-exposed medical images</li>
</ul>

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

<h3>📊 <strong>Response Details</strong></h3>

<ul>
<li><strong>Prediction</strong>: NORMAL or PNEUMONIA classification</li>
<li><strong>Confidence</strong>: Score from 0.0 to 1.0 (higher = more confident)</li>
<li><strong>Probabilities</strong>: Individual class probabilities</li>
<li><strong>Medical Recommendation</strong>: Contextual guidance based on results</li>
<li><strong>Model Info</strong>: Version and type used for analysis</li>
</ul>

<h3>🛡️ <strong>Security &amp; Validation</strong></h3>

<ul>
<li><strong>Rate Limiting</strong>: 5 requests per minute per IP</li>
<li><strong>File Validation</strong>: Automatic size and format checking</li>
<li><strong>Content Analysis</strong>: AI-powered image validation</li>
<li><strong>Duplicate Detection</strong>: Prevents repeated identical uploads</li>
<li><strong>Request Logging</strong>: Comprehensive audit trail</li>
</ul>

<h3>⚠️ <strong>Important Medical Disclaimer</strong></h3>

<p><strong>This AI system is for educational and research purposes only.</strong><br>
Results should NEVER replace professional medical diagnosis.<br>
Always consult qualified healthcare professionals for medical decisions.</p>

<h3>💡 <strong>Best Practices</strong></h3>

<ul>
<li>Use high-quality, clear chest X-ray images</li>
<li>Ensure proper image orientation (frontal view)</li>
<li>Consider using EfficientNet-B0 for critical analysis</li>
<li>Review confidence scores and recommendations</li>
<li>Always validate results with medical professionals</li>
</ul>

<h3>🚀 <strong>Example Usage</strong></h3>

<pre><code>curl -X POST "http://localhost:8000/pneumonia/predict" \\
     -H "Content-Type: multipart/form-data" \\
     -F "file=@chest_xray.jpg"
</code></pre>
    """,
    response_description="Detailed pneumonia detection results with confidence scores",
    responses={
        200: {
            "description": "Prediction successful",
            "content": {
                "application/json": {
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
            }
        },
        400: {
            "description": "Invalid input (file format, size, or content)",
            "content": {
                "application/json": {
                    "examples": {
                        "invalid_format": {
                            "summary": "Invalid file format",
                            "value": {
                                "detail": "Unsupported file type. Allowed: jpg, jpeg, png",
                                "error_code": "INVALID_FILE_FORMAT"
                            }
                        },
                        "invalid_content": {
                            "summary": "Invalid image content",
                            "value": {
                                "detail": "Image does not appear to be a valid chest X-ray",
                                "error_code": "INVALID_IMAGE_CONTENT"
                            }
                        }
                    }
                }
            }
        },
        409: {
            "description": "Duplicate file detected",
            "content": {
                "application/json": {
                    "example": {
                        "detail": "Duplicate file detected. Please wait before uploading the same image again.",
                        "error_code": "DUPLICATE_FILE"
                    }
                }
            }
        },
        413: {
            "description": "File too large",
            "content": {
                "application/json": {
                    "example": {
                        "detail": "File size exceeds limit of 10.0 MB",
                        "error_code": "FILE_TOO_LARGE"
                    }
                }
            }
        },
        429: {
            "description": "Rate limit exceeded",
            "content": {
                "application/json": {
                    "example": {
                        "detail": "Rate limit exceeded: 5 per minute",
                        "error_code": "RATE_LIMIT_EXCEEDED"
                    }
                }
            }
        },
        503: {
            "description": "Service unavailable (model not loaded)",
            "content": {
                "application/json": {
                    "example": {
                        "detail": "Prediction service is not available",
                        "error_code": "SERVICE_UNAVAILABLE"
                    }
                }
            }
        }
    }
)
@limiter.limit("5/minute")  # SlowAPI rate limiting as backup
async def predict_pneumonia(
    request: Request,
    file: UploadFile = File(
        ..., 
        description="📸 Chest X-ray image file (JPG, JPEG, PNG - max 10MB)",
        example="chest_xray.jpg"
    ),
    prediction_service: PneumoniaPredictionService = Depends(get_prediction_service)
):
    """
    **🔬 Advanced AI Pneumonia Detection**
    
    Analyzes chest X-ray images using state-of-the-art deep learning models to detect pneumonia
    with high accuracy and provides detailed medical recommendations.
    
    **Process Flow:**
    1. 🔍 **File Validation**: Size, format, and integrity checks
    2. 🖼️ **Content Analysis**: AI-powered image validation
    3. 🤖 **AI Prediction**: Deep learning model inference
    4. 📊 **Result Analysis**: Confidence scoring and interpretation
    5. 💡 **Medical Guidance**: Contextual recommendations
    
    **Security Features:**
    - Rate limiting (5 requests/minute per IP)
    - Duplicate detection and prevention
    - Comprehensive request logging
    - Multi-layer input validation
    
    **Args:**
        request: FastAPI request object (for security tracking)
        file: Uploaded chest X-ray image file
        prediction_service: AI model service (auto-injected)

    **Returns:**
        PredictionResponse: Comprehensive analysis results with:
        - Classification (NORMAL/PNEUMONIA)
        - Confidence scores and probabilities
        - Medical recommendations
        - Model information and disclaimers

    **Raises:**
        HTTPException: For validation, processing, or service errors
        
    **Medical Disclaimer:**
        Results are for educational purposes only. Always consult
        healthcare professionals for medical diagnosis and treatment.
    """
    client_ip = get_client_ip(request)
    
    # Validate prediction service is available
    if not prediction_service or not prediction_service.is_loaded():
        logger.error("Prediction service not available")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Prediction service is not available"
        )
    
    # Validate file exists
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No file provided"
        )
    
    # Validate file extension
    if not validate_file_extension(file.filename):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type. Allowed: {', '.join(settings.allowed_extensions)}"
        )
    
    try:
        # Read file contents
        contents = await file.read()
        
        # Validate file size
        if not validate_file_size(contents):
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File size exceeds limit of {settings.max_file_size / (1024 * 1024):.1f} MB"
            )
        
        # Check for duplicate uploads
        file_hash = calculate_file_hash(contents)
        if file_hash_cache.is_duplicate(file_hash):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Duplicate file detected. Please wait before uploading the same image again."
            )
        
        # Validate image integrity and get PIL Image
        contents_io = io.BytesIO(contents)
        image = validate_image_integrity(contents_io)
        
        # Validate image content (basic X-ray checks)
        if not validate_image_content(image):
            logger.warning(f"Invalid image content detected: {file.filename} from {client_ip}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Image does not appear to be a valid chest X-ray"
            )
        
        # Get image statistics for logging
        image_stats = get_image_stats(image)
        
        # Make prediction
        result = prediction_service.predict(image)
        
        # Log successful prediction
        logger.info(
            f"Prediction successful - IP: {client_ip}, "
            f"File: {file.filename}, Hash: {file_hash[:8]}, "
            f"Size: {image_stats['size']}, "
            f"Result: {result['prediction']}, "
            f"model: {result['model_type']}, "
            f"Confidence: {result['confidence']:.3f}"
        )
        
        return PredictionResponse(**result)
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except FileValidationError as e:
        logger.error(f"File validation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except ImageValidationError as e:
        logger.error(f"Image validation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except PredictionError as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process image"
        )
    except Exception as e:
        logger.error(f"Unexpected error in prediction: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )
    finally:
        # Clean up
        if 'file' in locals():
            await file.close()


@router.get(
    "/model/info", 
    tags=["Model"],
    summary="📊 AI Model Information",
    description="""
<h2>🤖 Machine Learning Model Details</h2>

<p>Provides comprehensive information about the currently loaded AI model including
architecture details, performance metrics, and training statistics.</p>

<h3>📋 <strong>Information Included</strong></h3>

<ul>
<li><strong>Model Architecture</strong>: Network structure and layer details</li>
<li><strong>Training Metrics</strong>: Accuracy, precision, recall, F1-score</li>
<li><strong>Dataset Info</strong>: Training data characteristics</li>
<li><strong>Performance Stats</strong>: Inference time and optimization details</li>
<li><strong>Version Info</strong>: Model version and build information</li>
</ul>

<h3>🎯 <strong>Use Cases</strong></h3>

<ul>
<li><strong>Model Validation</strong>: Verify correct model loading</li>
<li><strong>Performance Analysis</strong>: Review model capabilities</li>
<li><strong>Integration Planning</strong>: Understand model characteristics</li>
<li><strong>Debugging</strong>: Troubleshoot model-related issues</li>
</ul>

<h3>📊 <strong>Model Comparison</strong></h3>

<p>Use the <code>model</code> query parameter to get information about different models:
- <strong>Standard</strong>: Faster inference, good baseline performance
- <strong>EfficientNet-B0</strong>: Higher accuracy, advanced architecture</p>

<h3>⚡ <strong>Performance</strong></h3>

<ul>
<li><strong>Response Time</strong>: &lt; 50ms typical</li>
<li><strong>Rate Limiting</strong>: No limits applied</li>
<li><strong>Caching</strong>: Model info cached for performance</li>
</ul>
    """,
    response_description="Detailed model information and statistics",
    responses={
        200: {
            "description": "Model information retrieved successfully",
            "content": {
                "application/json": {
                    "example": {
                        "model_type": "standard",
                        "model_version": "v1.0",
                        "architecture": "CNN",
                        "input_shape": [224, 224, 3],
                        "classes": ["NORMAL", "PNEUMONIA"],
                        "training_accuracy": 0.94,
                        "validation_accuracy": 0.91,
                        "inference_time_ms": 200,
                        "model_size_mb": 15.2
                    }
                }
            }
        },
        404: {
            "description": "Model not found",
            "content": {
                "application/json": {
                    "example": {
                        "detail": "Model 'invalid_model' not found",
                        "error_code": "MODEL_NOT_FOUND"
                    }
                }
            }
        },
        503: {
            "description": "Prediction service unavailable",
            "content": {
                "application/json": {
                    "example": {
                        "detail": "Prediction service not available",
                        "error_code": "SERVICE_UNAVAILABLE"
                    }
                }
            }
        }
    }
)
async def get_model_info(
    prediction_service: PneumoniaPredictionService = Depends(get_prediction_service)
):
    """
    **📊 Comprehensive Model Information**
    
    Retrieves detailed information about the currently loaded AI model including:
    - Model architecture and configuration
    - Training and validation performance metrics
    - Input/output specifications
    - Inference performance characteristics
    - Model versioning and build details
    
    **Information Categories:**
    - **Architecture**: Network structure and parameters
    - **Performance**: Accuracy, precision, recall metrics
    - **Configuration**: Input shapes, class definitions
    - **Runtime**: Inference timing and optimization
    
    **Returns:**
        dict: Complete model information and statistics
        
    **Use Cases:**
        - Model validation and verification
        - Performance analysis and benchmarking
        - Integration planning and debugging
        - System monitoring and diagnostics
    """
    if not prediction_service:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Prediction service not available"
        )
    
    return prediction_service.get_model_info()
