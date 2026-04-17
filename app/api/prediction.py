"""
Pneumonia prediction API endpoints.
"""

import asyncio
import io
import time
from typing import Any

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile, status
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.util import get_remote_address

from ..core.logger import get_logger
from ..core.settings import settings
from ..docs.sections.prediction_metadata import PredictionMetadata
from ..models.error_codes import ErrorCode
from ..models.prediction_schemas import PredictionResponse
from ..services.prediction import PneumoniaPredictionService
from ..utils.exceptions import (
    FileValidationError,
    ImageValidationError,
    PredictionError,
)
from ..utils.get_prediction_service import get_prediction_service
from ..utils.jwt_auth import JWTPayload, get_current_user
from ..utils.security import calculate_file_hash, file_hash_cache, get_client_ip
from ..utils.validation import (
    get_image_stats,
    validate_file_extension,
    validate_file_size,
    validate_image_content,
    validate_image_integrity,
)

logger = get_logger(__name__)
router = APIRouter()
prediction_metadata = PredictionMetadata.get_metadata()

# For backward compatibility with slowapi
# Backward compatibility object (still imported elsewhere maybe)
limiter = Limiter(key_func=get_remote_address)

PREDICTION_CONCURRENCY_LIMIT = max(
    0, int(getattr(settings, "prediction_concurrency_limit", 4) or 0)
)
_prediction_semaphore = asyncio.Semaphore(PREDICTION_CONCURRENCY_LIMIT or 1)


def _utc_timestamp() -> str:
    """Return an ISO-like UTC timestamp for API error payloads."""
    return time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime())


def _error_response(
    status_code: int,
    detail: str,
    error_code: ErrorCode,
    **extra: Any,
) -> JSONResponse:
    """Build a consistent JSON error response for prediction failures."""
    return JSONResponse(
        status_code=status_code,
        content={
            "detail": detail,
            "error_code": error_code,
            "timestamp": _utc_timestamp(),
            **extra,
        },
    )


def _validate_prediction_service(
    prediction_service: PneumoniaPredictionService | None,
) -> JSONResponse | None:
    if prediction_service is None:
        logger.error("Prediction service not available")
        return _error_response(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "Prediction service is not available",
            ErrorCode.SERVICE_UNAVAILABLE,
            service_status="not_initialized",
        )

    if not prediction_service.is_loaded():
        logger.error("Prediction model not loaded")
        return _error_response(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "AI model is not loaded or failed to initialize",
            ErrorCode.MODEL_NOT_LOADED,
            service_status="model_not_loaded",
        )

    return None


def _validate_file_metadata(file: UploadFile) -> JSONResponse | None:
    if not file.filename:
        return _error_response(
            status.HTTP_400_BAD_REQUEST,
            "No file provided",
            ErrorCode.NO_FILE_PROVIDED,
        )

    if not validate_file_extension(file.filename):
        allowed_extensions = ", ".join(settings.allowed_extensions)
        return _error_response(
            status.HTTP_400_BAD_REQUEST,
            f"Unsupported file type. Allowed: {allowed_extensions}",
            ErrorCode.INVALID_FILE_FORMAT,
        )

    return None


def _validate_file_contents(contents: bytes) -> JSONResponse | None:
    if validate_file_size(contents):
        return None

    file_size_mb = len(contents) / (1024 * 1024)
    max_size_mb = settings.max_file_size / (1024 * 1024)
    return _error_response(
        status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
        f"File size exceeds limit of {max_size_mb:.1f} MB",
        ErrorCode.FILE_TOO_LARGE,
        max_size_mb=max_size_mb,
        actual_size_mb=round(file_size_mb, 2),
    )


def _validate_unique_upload(file_hash: str) -> JSONResponse | None:
    if not file_hash_cache.is_duplicate(file_hash, settings.cache_duration):
        return None

    return _error_response(
        status.HTTP_409_CONFLICT,
        "Duplicate file detected. Please wait before uploading the same image again.",
        ErrorCode.DUPLICATE_FILE,
        retry_after=settings.cache_duration,
        file_hash=file_hash,
    )


def _load_validated_image(
    contents: bytes, filename: str, client_ip: str
) -> tuple[Any, JSONResponse | None]:
    image = validate_image_integrity(io.BytesIO(contents))

    if validate_image_content(image):
        return image, None

    logger.warning("Invalid image content detected: %s from %s", filename, client_ip)
    return None, _error_response(
        status.HTTP_400_BAD_REQUEST,
        "Image does not appear to be a valid chest X-ray",
        ErrorCode.INVALID_IMAGE_CONTENT,
    )


async def _acquire_prediction_slot(client_ip: str) -> tuple[float, bool]:
    if PREDICTION_CONCURRENCY_LIMIT <= 0:
        return 0.0, False

    start_wait = time.time()
    await _prediction_semaphore.acquire()
    wait_time = time.time() - start_wait

    if wait_time > 0.05:
        logger.info(
            "Concurrency wait %.3fs | IP=%s | limit=%d",
            wait_time,
            client_ip,
            PREDICTION_CONCURRENCY_LIMIT,
        )

    return wait_time, True


def _predict_with_timing(
    prediction_service: PneumoniaPredictionService, image: Any
) -> tuple[dict[str, Any], float]:
    inference_start = time.time()
    result = prediction_service.predict(image)
    return result, time.time() - inference_start


def _log_successful_prediction(
    user: JWTPayload,
    client_ip: str,
    filename: str,
    file_hash: str,
    image_stats: dict[str, Any],
    result: dict[str, Any],
    inference_time: float,
    wait_time: float,
) -> None:
    logger.info(
        "Prediction OK | user=%s ip=%s file=%s hash=%s size=%s "
        "model=%s pred=%s conf=%.3f infer=%.3fs wait=%.3fs",
        user.user_id,
        client_ip,
        filename,
        file_hash[:8],
        image_stats["size"],
        result["model_info"]["model_type"],
        result["prediction"],
        result["confidence"],
        inference_time,
        wait_time,
    )


@router.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["Pneumonia Detection"],
    **prediction_metadata,
)
async def predict_pneumonia(
    request: Request,
    file: UploadFile = File(
        ...,
        description="Chest X-ray image file (JPG, JPEG, PNG - max 10MB)",
        example="chest_xray.jpg",
    ),
    user: JWTPayload = Depends(get_current_user),
    prediction_service: PneumoniaPredictionService = Depends(get_prediction_service),
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
    - **JWT Authentication** (Supabase Bearer token REQUIRED)
    - **User Rate Limiting** (100 requests/hour per authenticated user)
    - Duplicate detection and prevention
    - Comprehensive request logging
    - Multi-layer input validation

    **Args:**
        request: FastAPI request object (for security tracking)
        file: Uploaded chest X-ray image file
        user: Authenticated user from Supabase JWT (auto-injected)
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
    semaphore_acquired = False

    try:
        service_error = _validate_prediction_service(prediction_service)
        if service_error:
            return service_error

        file_error = _validate_file_metadata(file)
        if file_error:
            return file_error

        waited, semaphore_acquired = await _acquire_prediction_slot(client_ip)
        contents = await file.read()

        file_size_error = _validate_file_contents(contents)
        if file_size_error:
            return file_size_error

        file_hash = calculate_file_hash(contents)
        duplicate_error = _validate_unique_upload(file_hash)
        if duplicate_error:
            return duplicate_error

        image, image_error = _load_validated_image(
            contents, file.filename, client_ip
        )
        if image_error:
            return image_error

        image_stats = get_image_stats(image)
        result, inference_time = _predict_with_timing(prediction_service, image)

        _log_successful_prediction(
            user=user,
            client_ip=client_ip,
            filename=file.filename,
            file_hash=file_hash,
            image_stats=image_stats,
            result=result,
            inference_time=inference_time,
            wait_time=waited,
        )

        return PredictionResponse(**result)

    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except FileValidationError as e:
        logger.error("File validation error: %s", e)
        return _error_response(
            status.HTTP_400_BAD_REQUEST,
            str(e),
            (
                ErrorCode.FILE_TOO_LARGE
                if "size" in str(e).lower()
                else ErrorCode.INVALID_FILE_FORMAT
            ),
        )
    except ImageValidationError as e:
        logger.error("Image validation error: %s", e)
        return _error_response(
            status.HTTP_400_BAD_REQUEST,
            str(e),
            ErrorCode.IMAGE_VALIDATION_ERROR,
        )
    except PredictionError as e:
        logger.error("Prediction error: %s", e)
        return _error_response(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            "Failed to process image",
            ErrorCode.PREDICTION_FAILED,
        )
    except Exception as e:
        logger.error("Unexpected error in prediction: %s", e, exc_info=True)
        return _error_response(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            "Internal server error",
            ErrorCode.INTERNAL_SERVER_ERROR,
        )
    finally:
        if semaphore_acquired:
            _prediction_semaphore.release()
        await file.close()
