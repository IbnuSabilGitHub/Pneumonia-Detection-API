from fastapi import  HTTPException, status,  Query

from ..services.prediction import PneumoniaPredictionService
from ..core.settings import settings
from ..core.logger import get_logger


logger = get_logger(__name__)


def get_prediction_service(
    model: str = Query(
        "standard", 
        description="Choose AI model for prediction",
        enum=["standard", "efficientnet_b0"],
        example="standard"
    )
) -> PneumoniaPredictionService:
    """
    **AI Model Service Provider**
    
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