from fastapi import APIRouter,HTTPException, status, Depends
from slowapi import Limiter
from slowapi.util import get_remote_address

from ..services.prediction import PneumoniaPredictionService
from ..core.logger import get_logger
from ..docs.prediction_metadata import PredictionMetadata
from ..utils.get_prediction_service import get_prediction_service
from ..docs.model_info_metadata import ModelInfoMetadata

logger = get_logger(__name__)
router = APIRouter()
prediction_metadata = PredictionMetadata.get_metadata()

# For backward compatibility with slowapi
limiter = Limiter(key_func=get_remote_address)
model_info_metadata = ModelInfoMetadata.get_metadata()

@router.get(
    "/model/info", 
    tags=["Model"],
    **model_info_metadata
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
