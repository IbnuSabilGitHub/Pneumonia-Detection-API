"""
Pneumonia prediction API endpoints.
"""

import asyncio
import io
import time

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

# ---------------------------------------------------------------------------
# Concurrency & endpoint-specific rate limiting controls
# ---------------------------------------------------------------------------

# Configurable via env: PREDICTION_CONCURRENCY_LIMIT (default 2-4 for CPU bound)
PREDICTION_CONCURRENCY_LIMIT = int(
    getattr(settings, "prediction_concurrency_limit", 0)
    or int(__import__("os").environ.get("PREDICTION_CONCURRENCY_LIMIT", 4))
)

# Simple in-process semaphore for concurrency limiting (per instance)
_prediction_semaphore = asyncio.Semaphore(PREDICTION_CONCURRENCY_LIMIT)

# Endpoint specific quota per IP (default lower than global). Example: 20 per 5m
PREDICTION_IP_WINDOW = int(
    __import__("os").environ.get(
        "PREDICTION_RATE_WINDOW", settings.rate_limit_window_size
    )
)
PREDICTION_IP_LIMIT = int(
    __import__("os").environ.get("PREDICTION_MAX_REQUESTS_PER_IP", 20)
)


async def _check_prediction_endpoint_quota(
    request: Request, client_ip: str
) -> tuple[bool, int, int]:
    """Per-endpoint lightweight quota (independent from global limiter).

    Uses the shared storage backend if available via rate limiter; else best-effort in-memory fallback attached to app.state.
    Returns (allowed, current_count, limit).
    """
    try:
        from ..core.advanced_rate_limiting import get_rate_limiter  # local import

        rl = get_rate_limiter()
        key = f"pred_ep:{client_ip}"
        window = PREDICTION_IP_WINDOW

        if rl and rl.storage:  # Use storage for multi-process consistency
            # increment with ttl
            await rl.storage.increment(key, 1, window)
            count = await rl.storage.get(key) or 0
        else:
            # Fallback: in-memory dict on app.state
            state_key = "_predict_ep_counts"
            if not hasattr(request.app.state, state_key):
                setattr(request.app.state, state_key, {})
            store: dict = getattr(request.app.state, state_key)
            now = time.time()
            entry = store.get(client_ip)
            if not entry or now > entry["expires_at"]:
                entry = {"count": 0, "expires_at": now + window}
            entry["count"] += 1
            store[client_ip] = entry
            count = entry["count"]
        return (count <= PREDICTION_IP_LIMIT, count, PREDICTION_IP_LIMIT)
    except Exception as e:
        logger.debug(f"Prediction endpoint quota check failed: {e}")
        return (True, 0, PREDICTION_IP_LIMIT)


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

    # Endpoint-specific early quota (lightweight) BEFORE heavy reads
    allowed_ep, ep_count, ep_limit = await _check_prediction_endpoint_quota(
        request, client_ip
    )
    if not allowed_ep:
        reset_ts = int(time.time() + PREDICTION_IP_WINDOW)
        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content={
                "detail": "Prediction endpoint rate limit exceeded",
                "error_code": ErrorCode.RATE_LIMIT_EXCEEDED,
                "endpoint": "/pneumonia/predict",
                "ip": client_ip,
                "requests_in_window": ep_count,
                "limit": ep_limit,
                "window_seconds": PREDICTION_IP_WINDOW,
                "reset": reset_ts,
                "advice": "Reduce request frequency or batch client-side.",
            },
            headers={
                "Retry-After": str(PREDICTION_IP_WINDOW),
                "X-RateLimit-Limit-Predict": str(ep_limit),
                "X-RateLimit-Remaining-Predict": str(max(0, ep_limit - ep_count)),
                "X-RateLimit-Reset-Predict": str(reset_ts),
            },
        )

    # Validate prediction service is available
    if not prediction_service or not prediction_service.is_loaded():
        logger.error("Prediction service not available")
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "detail": "Prediction service is not available",
                "error_code": ErrorCode.SERVICE_UNAVAILABLE,
                "service_status": "not_initialized",
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime()),
            },
        )

    if not prediction_service.is_loaded():
        logger.error("Prediction model not loaded")
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "detail": "AI model is not loaded or failed to initialize",
                "error_code": ErrorCode.MODEL_NOT_LOADED,
                "service_status": "model_not_loaded",
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime()),
            },
        )

    # Validate file exists
    if not file.filename:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "detail": "No file provided",
                "error_code": ErrorCode.NO_FILE_PROVIDED,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime()),
            },
        )

    # Validate file extension
    if not validate_file_extension(file.filename):
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "detail": f"Unsupported file type. Allowed: {', '.join(settings.allowed_extensions)}",
                "error_code": ErrorCode.INVALID_FILE_FORMAT,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime()),
            },
        )

    try:
        # Concurrency limiting section (serialize expensive model inference bursts)
        waited = 0.0
        if PREDICTION_CONCURRENCY_LIMIT > 0:
            start_wait = time.time()
            await _prediction_semaphore.acquire()
            waited = time.time() - start_wait
            if waited > 0.05:  # log only meaningful waits
                logger.info(
                    "Concurrency wait %.3fs | IP=%s | in_flight=%d/%d",
                    waited,
                    client_ip,
                    (PREDICTION_CONCURRENCY_LIMIT - _prediction_semaphore._value),
                    PREDICTION_CONCURRENCY_LIMIT,
                )
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
                    "error_code": ErrorCode.FILE_TOO_LARGE,
                    "max_size_mb": max_size_mb,
                    "actual_size_mb": round(file_size_mb, 2),
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime()),
                },
            )

        # Check for duplicate uploads
        file_hash = calculate_file_hash(contents)
        if file_hash_cache.is_duplicate(file_hash, settings.cache_duration):
            return JSONResponse(
                status_code=status.HTTP_409_CONFLICT,
                content={
                    "detail": "Duplicate file detected. Please wait before uploading the same image again.",
                    "error_code": ErrorCode.DUPLICATE_FILE,
                    "retry_after": settings.cache_duration,
                    "file_hash": file_hash,
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime()),
                },
            )

        # Validate image integrity and get PIL Image
        contents_io = io.BytesIO(contents)
        image = validate_image_integrity(contents_io)

        # Validate image content (basic X-ray checks)
        if not validate_image_content(image):
            logger.warning(
                f"Invalid image content detected: {file.filename} from {client_ip}"
            )
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={
                    "detail": "Image does not appear to be a valid chest X-ray",
                    "error_code": ErrorCode.INVALID_IMAGE_CONTENT,
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime()),
                },
            )

        # Get image statistics for logging
        image_stats = get_image_stats(image)

        # Make prediction (inference)
        inference_start = time.time()
        result = prediction_service.predict(image)
        inference_time = time.time() - inference_start

        # Log successful prediction
        logger.info(
            "Prediction OK | ip=%s file=%s hash=%s size=%s model=%s pred=%s conf=%.3f infer=%.3fs wait=%.3fs ep_count=%d",
            client_ip,
            file.filename,
            file_hash[:8],
            image_stats['size'],
            result['model_info']['model_type'],
            result['prediction'],
            result['confidence'],
            inference_time,
            waited,
            ep_count,
        )

        return PredictionResponse(**result)

    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except FileValidationError as e:
        logger.error("File validation error: %s", e)
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "detail": str(e),
                "error_code": (
                    ErrorCode.FILE_TOO_LARGE
                    if "size" in str(e).lower()
                    else ErrorCode.INVALID_FILE_FORMAT
                ),
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime()),
            },
        )
    except ImageValidationError as e:
        logger.error("Image validation error: %s", e)
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "detail": str(e),
                "error_code": ErrorCode.IMAGE_VALIDATION_ERROR,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime()),
            },
        )
    except PredictionError as e:
        logger.error("Prediction error: %s", e)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "detail": "Failed to process image",
                "error_code": ErrorCode.PREDICTION_FAILED,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime()),
            },
        )
    except Exception as e:
        logger.error("Unexpected error in prediction: %s", e, exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "detail": "Internal server error",
                "error_code": ErrorCode.INTERNAL_SERVER_ERROR,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime()),
            },
        )
    finally:
        # Clean up
        if "file" in locals():
            await file.close()
        if PREDICTION_CONCURRENCY_LIMIT > 0 and _prediction_semaphore.locked():
            # Release semaphore only if held
            try:
                _prediction_semaphore.release()
            except ValueError:
                pass
