from fastapi import APIRouter, Depends, HTTPException, status

from ..core.dependencies import get_prediction_service
from ..core.logger import get_logger
from ..services.prediction import PneumoniaPredictionService

logger = get_logger(__name__)
router = APIRouter()


@router.get("/model/info")
async def get_model_info(
    prediction_service: PneumoniaPredictionService | None = Depends(
        get_prediction_service
    ),
):
    """Return metadata for the loaded prediction model."""
    if prediction_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Prediction service is not available",
        )

    return prediction_service.get_model_info()

