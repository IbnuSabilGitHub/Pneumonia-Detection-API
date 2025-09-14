"""
Pneumonia prediction API endpoints.
"""
import io
import time
from fastapi import APIRouter, File, UploadFile, HTTPException, status, Request, Depends
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.util import get_remote_address


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
from ..docs.sections.prediction_metadata import PredictionMetadata
from ..utils.get_prediction_service import get_prediction_service
from ..models.prediction_schemas import PredictionResponse

logger = get_logger(__name__)
router = APIRouter()
prediction_metadata = PredictionMetadata.get_metadata()

# For backward compatibility with slowapi
limiter = Limiter(key_func=get_remote_address)


@router.post(
    "/predict", 
    response_model=PredictionResponse,
    tags=["Pneumonia Detection"],
    **prediction_metadata,
)
@limiter.limit("5/minute")  # SlowAPI rate limiting as backup
async def predict_pneumonia(
    request: Request,
    file: UploadFile = File(
        ..., 
        description="Chest X-ray image file (JPG, JPEG, PNG - max 10MB)",
        example="chest_xray.jpg"
    ),
    prediction_service: PneumoniaPredictionService = Depends(get_prediction_service)
):
    """
    **Advanced AI Pneumonia Detection**
    
    Analyzes chest X-ray images using state-of-the-art deep learning models to detect pneumonia
    with high accuracy and provides detailed medical recommendations.
    
    **Process Flow:**
    1. **File Validation**: Size, format, and integrity checks
    2. **Content Analysis**: AI-powered image validation
    3. **AI Prediction**: Deep learning model inference
    4. **Result Analysis**: Confidence scoring and interpretation
    5. **Medical Guidance**: Contextual recommendations
    
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
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "detail": "Prediction service is not available",
                "error_code": "SERVICE_UNAVAILABLE",
                "service_status": "not_initialized",
                "retry_after": 30,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime())
            }
        )
    
    # Validate file exists
    if not file.filename:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "detail": "No file provided",
                "error_code": "MISSING_FILE",
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime())
            }
        )
    
    # Validate file extension
    if not validate_file_extension(file.filename):
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "detail": f"Unsupported file type. Allowed: {', '.join(settings.allowed_extensions)}",
                "error_code": "INVALID_FILE_FORMAT",
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime())
            }
        )
    
    try:
        # Read file contents
        contents = await file.read()
        
        # Validate file size
        if not validate_file_size(contents):
            file_size_mb = len(contents) / (1024 * 1024)
            max_size_mb = settings.max_file_size / (1024 * 1024)
            return JSONResponse(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                content={
                    "detail": f"File size exceeds limit of {max_size_mb:.1f} MB",
                    "error_code": "FILE_TOO_LARGE",
                    "max_size_mb": max_size_mb,
                    "actual_size_mb": round(file_size_mb, 2),
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime())
                }
            )
        
        # Check for duplicate uploads
        file_hash = calculate_file_hash(contents)
        if file_hash_cache.is_duplicate(file_hash):
            return JSONResponse(
                status_code=status.HTTP_409_CONFLICT,
                content={
                    "detail": "Duplicate file detected. Please wait before uploading the same image again.",
                    "error_code": "INVALID_MODEL",
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime())
                }
            )
        
        # Validate image integrity and get PIL Image
        contents_io = io.BytesIO(contents)
        image = validate_image_integrity(contents_io)
        
        # Validate image content (basic X-ray checks)
        if not validate_image_content(image):
            logger.warning(f"Invalid image content detected: {file.filename} from {client_ip}")
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={
                    "detail": "Image does not appear to be a valid chest X-ray",
                    "error_code": "INVALID_IMAGE_CONTENT",
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime())
                }
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
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "detail": str(e),
                "error_code": "FILE_VALIDATION_ERROR",
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime())
            }
        )
    except ImageValidationError as e:
        logger.error(f"Image validation error: {e}")
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "detail": str(e),
                "error_code": "IMAGE_VALIDATION_ERROR",
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime())
            }
        )
    except PredictionError as e:
        logger.error(f"Prediction error: {e}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "detail": "Failed to process image",
                "error_code": "PREDICTION_ERROR",
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime())
            }
        )
    except Exception as e:
        logger.error(f"Unexpected error in prediction: {e}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "detail": "Internal server error",
                "error_code": "INTERNAL_ERROR",
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime())
            }
        )
    finally:
        # Clean up
        if 'file' in locals():
            await file.close()