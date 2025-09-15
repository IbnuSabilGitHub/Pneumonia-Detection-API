from pydantic import BaseModel, Field
from typing import  Optional

class HealthResponse(BaseModel):
    """
    **Service Health Status Response**
    
    Comprehensive health check response providing detailed information about
    API service status, model availability, and system uptime.
    
    **Health Status Levels:**
    - **healthy**: All systems operational, models loaded
    - **partial**: Service running with limitations
    - **unhealthy**: Critical issues detected
    
    **Fields:**
    - **status**: Overall service health indicator
    - **model_loaded**: AI model availability status
    - **version**: Current API version
    - **uptime**: Service runtime in seconds
    """
    
    status: str = Field(
        ..., 
        description="Overall service health status",
        example="healthy",
        pattern="^(healthy|partial|unhealthy)$"
    )
    model_loaded: bool = Field(
        ..., 
        description="AI model loading and readiness status",
        example=True
    )
    version: str = Field(
        ..., 
        description="Current API version identifier",
        example="1.0.0"
    )
    uptime: Optional[float] = Field(
        None, 
        description="Service uptime since last restart (seconds)",
        example=3600.5,
        ge=0.0
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "model_loaded": True,
                "version": "1.0.0",
                "uptime": 3600.5
            }
        }