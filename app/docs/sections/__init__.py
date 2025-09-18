"""
Centralized metadata exports for OpenAPI documentation.
"""

from .api_metadata import ApiMetadata
from .health_metadata import HealthMetadata
from .model_info_metadata import ModelInfoMetadata
from .prediction_metadata import PredictionMetadata
from .stat_metadata import StatMetadata
from .status_metadata import StatusMetadata

__all__ = [
    "HealthMetadata",
    "PredictionMetadata",
    "ModelInfoMetadata",
    "StatMetadata",
    "StatusMetadata",
    "ApiMetadata",
]
