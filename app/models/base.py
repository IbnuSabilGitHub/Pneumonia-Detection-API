"""
pydantic models for base error responses.
"""
from pydantic import BaseModel, Field
import time

class BaseErrorResponse(BaseModel):
    """Base error response with common fields for consistency."""
    
    detail: str = Field(
        ..., 
        description="Human-readable error message explaining what went wrong"
    )
    error_code: str = Field(
        ...,
        description="Machine-readable error code for programmatic handling"
    )
    timestamp: str = Field(
        default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime()),
        description="Error occurrence timestamp in ISO 8601 format"
    )