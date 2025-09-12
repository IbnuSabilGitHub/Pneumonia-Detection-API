"""
Template untuk Response Models
Ganti EndpointName dengan nama endpoint yang sesuai (PascalCase)
Contoh: HealthResponse, PredictionResponse, SecurityStatsResponse
"""
from pydantic import BaseModel, Field
from typing import Dict, Optional, List

# Template 1: Basic Response Model
class EndpointNameResponse(BaseModel):  # Ganti EndpointName dengan nama yang sesuai
    """
    **🎯 [Endpoint Description Title]**
    
    [Detailed description of what this endpoint returns]
    
    **Key Features:**
    - Feature 1: Description
    - Feature 2: Description
    - Feature 3: Description
    
    **Use Cases:**
    - Use case 1
    - Use case 2
    """
    
    # Required fields - sesuaikan dengan kebutuhan endpoint
    status: str = Field(
        ..., 
        description="🎯 Field description with emoji",
        example="active",
        pattern="^(active|inactive|error)$"  # Optional: regex pattern
    )
    
    timestamp: str = Field(
        ..., 
        description="⏰ ISO format timestamp",
        example="2025-09-11T10:30:00.000Z"
    )
    
    # Optional fields
    message: Optional[str] = Field(
        None, 
        description="💡 Optional message field",
        example="Operation completed successfully"
    )
    
    # Complex data structures
    data: Dict = Field(
        ..., 
        description="📊 Main data object",
        example={
            "key1": "value1",
            "key2": "value2"
        }
    )
    
    # List fields
    items: List[str] = Field(
        ..., 
        description="📋 List of items",
        example=["item1", "item2", "item3"]
    )
    
    # Numeric fields
    count: int = Field(
        ..., 
        description="🔢 Numeric count",
        example=42,
        ge=0  # Greater than or equal to 0
    )
    
    # Float fields
    score: float = Field(
        ..., 
        description="📈 Score value",
        example=0.95,
        ge=0.0,
        le=1.0
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "active",
                "timestamp": "2025-09-11T10:30:00.000Z",
                "message": "Operation completed successfully",
                "data": {
                    "key1": "value1",
                    "key2": "value2"
                },
                "items": ["item1", "item2", "item3"],
                "count": 42,
                "score": 0.95
            }
        }


# Template 2: Error Response Model
class EndpointNameErrorResponse(BaseModel):  # Ganti EndpointName dengan nama yang sesuai
    """
    **❌ [Endpoint Name] Error Response**
    
    Error response when [endpoint] encounters issues.
    """
    
    error: str = Field(
        ..., 
        description="❌ Error message",
        example="Service not available"
    )
    
    error_code: Optional[str] = Field(
        None,
        description="🏷️ Application-specific error code",
        example="SERVICE_UNAVAILABLE"
    )
    
    timestamp: str = Field(
        ..., 
        description="⏰ Error timestamp in ISO format",
        example="2025-09-11T10:30:00.000Z"
    )
    
    details: Optional[Dict] = Field(
        None,
        description="📋 Additional error details",
        example={"field": "validation error"}
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "error": "Service not available",
                "error_code": "SERVICE_UNAVAILABLE", 
                "timestamp": "2025-09-11T10:30:00.000Z",
                "details": {"field": "validation error"}
            }
        }


# Template 3: Detailed/Complex Response Model
class EndpointNameDetailedResponse(BaseModel):  # Ganti EndpointName dengan nama yang sesuai
    """
    **🎯 Detailed [Endpoint Name] Response**
    
    Extended response with comprehensive information.
    """
    
    # Basic info
    id: str = Field(..., description="🆔 Unique identifier", example="123e4567-e89b-12d3-a456-426614174000")
    name: str = Field(..., description="📝 Resource name", example="Resource Name")
    
    # Status info
    status: str = Field(..., description="📊 Current status", example="active")
    created_at: str = Field(..., description="📅 Creation timestamp", example="2025-09-11T10:30:00.000Z")
    updated_at: str = Field(..., description="🔄 Last update timestamp", example="2025-09-11T10:35:00.000Z")
    
    # Nested objects
    metadata: Dict = Field(
        ..., 
        description="🏷️ Resource metadata",
        example={
            "version": "1.0",
            "author": "System",
            "tags": ["tag1", "tag2"]
        }
    )
    
    # Arrays of objects
    properties: List[Dict] = Field(
        ...,
        description="📋 List of properties",
        example=[
            {"key": "property1", "value": "value1", "type": "string"},
            {"key": "property2", "value": 42, "type": "number"}
        ]
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "id": "123e4567-e89b-12d3-a456-426614174000",
                "name": "Resource Name",
                "status": "active",
                "created_at": "2025-09-11T10:30:00.000Z",
                "updated_at": "2025-09-11T10:35:00.000Z",
                "metadata": {
                    "version": "1.0",
                    "author": "System",
                    "tags": ["tag1", "tag2"]
                },
                "properties": [
                    {"key": "property1", "value": "value1", "type": "string"},
                    {"key": "property2", "value": 42, "type": "number"}
                ]
            }
        }
