"""
Pneumonia prediction API endpoints.
"""
import io
from datetime import datetime
from fastapi import APIRouter, File, UploadFile, HTTPException, status, Request, Depends
from slowapi import Limiter
from slowapi.util import get_remote_address

from ..models.schemas import PredictionResponse, ErrorResponse
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


def get_prediction_service() -> PneumoniaPredictionService:
    """Dependency to get prediction service instance."""
    return getattr(get_prediction_service, '_service', None)


@router.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
@limiter.limit("5/minute")  # SlowAPI rate limiting as backup
async def predict_pneumonia(
    request: Request,
    file: UploadFile = File(..., description="Chest X-ray image (JPG, JPEG, PNG)"),
    prediction_service: PneumoniaPredictionService = Depends(get_prediction_service)
):
    """
    Predict pneumonia from chest X-ray image.

    This endpoint analyzes a chest X-ray image and provides a prediction
    of whether pneumonia is detected along with confidence scores and
    medical recommendations.

    Args:
        request: FastAPI request object (for IP tracking)
        file: Uploaded chest X-ray image file
        prediction_service: Injected prediction service

    Returns:
        Prediction results with confidence, probabilities, and medical recommendation

    Raises:
        HTTPException: For various validation and processing errors

    Security Features:
        - Rate limiting: 5 requests per minute per IP
        - File size and type validation
        - Image content validation
        - Duplicate detection
        - Comprehensive logging

    Note: 
        This model is for educational/experimental purposes only.
        Always consult healthcare professionals for medical advice.
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


@router.get("/model/info", tags=["Model"])
async def get_model_info(
    prediction_service: PneumoniaPredictionService = Depends(get_prediction_service)
):
    """
    Get information about the loaded model.
    
    Returns:
        Model configuration and statistics
    """
    if not prediction_service:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Prediction service not available"
        )
    
    return prediction_service.get_model_info()
