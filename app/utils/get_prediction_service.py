"""
Utility module to provide pneumonia prediction services
"""
from functools import lru_cache

from fastapi import HTTPException, status

from ..core.dependencies import get_dependencies
from ..core.logger import get_logger
from ..core.settings import settings
from ..services.prediction import PneumoniaPredictionService

logger = get_logger(__name__)

DEFAULT_MODEL_NAME = "standard"
SUPPORTED_MODEL_CONFIGS = {
    "standard": ("model_path", "model_stats_path"),
    "efficientnet_b0": (
        "model_path_efficientnet_b0",
        "model_stats_path_efficientnet_b0",
    ),
}


def get_prediction_service(
    model: str = DEFAULT_MODEL_NAME,
) -> PneumoniaPredictionService:
    """
    **AI Model Service Provider**

    Provides access to trained pneumonia detection models with automatic loading and validation.

    **Query Parameters:**
    - `model` (str, default="standard"): Choose AI model for prediction
      - `standard`: Baseline CNN architecture (faster inference)
      - `efficientnet_b0`: Advanced transfer learning model (higher accuracy)

    **Model Features:**
    - Automatic model loading and caching
    - Model validation and health checks
    - Performance optimization
    - Error handling and fallback mechanisms
    """
    model_name = _normalize_model_name(model)
    logger.debug("get_prediction_service: using model=%s", model_name)

    service = _get_startup_service(model_name) or _get_cached_service(model_name)
    _ensure_model_loaded(service, model_name)

    return service


def _normalize_model_name(model: str) -> str:
    model_name = str(model).strip().lower() if model else DEFAULT_MODEL_NAME

    if model_name in SUPPORTED_MODEL_CONFIGS:
        return model_name

    available_models = ", ".join(SUPPORTED_MODEL_CONFIGS)
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Model '{model_name}' not found. Available models: {available_models}",
    )


def _get_startup_service(model_name: str) -> PneumoniaPredictionService | None:
    if model_name != DEFAULT_MODEL_NAME:
        return None

    return get_dependencies().prediction_service


@lru_cache(maxsize=len(SUPPORTED_MODEL_CONFIGS))
def _get_cached_service(model_name: str) -> PneumoniaPredictionService:
    model_path_attr, stats_path_attr = SUPPORTED_MODEL_CONFIGS[model_name]
    return PneumoniaPredictionService(
        model_path=getattr(settings, model_path_attr),
        stats_path=getattr(settings, stats_path_attr),
    )


def _ensure_model_loaded(
    service: PneumoniaPredictionService, model_name: str
) -> None:
    if not service.is_loaded():
        try:
            service.load_model()
        except Exception as e:
            logger.error("Failed to load %s model: %s", model_name, e)
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Failed to load {model_name} model",
            ) from e
