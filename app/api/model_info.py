from fastapi import APIRouter,HTTPException, status, Depends
from slowapi import Limiter
from slowapi.util import get_remote_address

from ..services.prediction import PneumoniaPredictionService
from ..core.logger import get_logger
from ..docs.prediction_metadata import PredictionMetadata
from ..utils.get_prediction_service import get_prediction_service

logger = get_logger(__name__)
router = APIRouter()
prediction_metadata = PredictionMetadata.get_metadata()

# For backward compatibility with slowapi
limiter = Limiter(key_func=get_remote_address)


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
