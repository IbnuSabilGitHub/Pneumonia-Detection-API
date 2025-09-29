# 📚 Panduan Refactor Dokumentasi Endpoint API

Panduan ini menjelaskan pola standar untuk merefactor dokumentasi endpoint berdasarkan refactor yang telah dilakukan pada endpoint `/status`.

## 🎯 Tujuan Refactor

1. **Dokumentasi yang Konsisten**: Semua endpoint memiliki struktur dokumentasi yang seragam
2. **Type Safety**: Menggunakan Pydantic models untuk validasi dan dokumentasi otomatis
3. **Error Handling**: Response error yang terstruktur dan konsisten
4. **Developer Experience**: Dokumentasi yang kaya dengan contoh dan deskripsi yang jelas
5. **Maintainability**: Kode yang bersih dan mudah dipelihara

## 🔄 Pola Refactor Standard

### Langkah 1: Buat Response Models di `schemas.py`

#### Template Response Model:
```python
class [EndpointName]Response(BaseModel):
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
    
    # Required fields
    field_name: str = Field(
        ..., 
        description="🎯 Field description with emoji",
        example="example_value"
    )
    
    # Optional fields
    optional_field: Optional[str] = Field(
        None, 
        description="💡 Optional field description",
        example="optional_example"
    )
    
    # Complex fields
    data_object: Dict = Field(
        ..., 
        description="📊 Complex data structure",
        example={"key": "value"}
    )
    
    # List fields
    items_list: List[str] = Field(
        ..., 
        description="📋 List of items",
        example=["item1", "item2"]
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "field_name": "example_value",
                "optional_field": "optional_example",
                "data_object": {"key": "value"},
                "items_list": ["item1", "item2"]
            }
        }

class [EndpointName]ErrorResponse(BaseModel):
    """
    **❌ [Endpoint Name] Error Response**
    
    Error response when [endpoint] encounters issues.
    """
    
    error: str = Field(
        ..., 
        description="❌ Error message",
        example="Service not available"
    )
    
    timestamp: str = Field(
        ..., 
        description="⏰ Error timestamp",
        example="2025-09-11T10:30:00.000Z"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "error": "Service not available",
                "timestamp": "2025-09-11T10:30:00.000Z"
            }
        }
```

### Langkah 2: Buat Metadata Class di `docs/[endpoint]_metadata.py`

#### Template Metadata Class:
```python
from typing import Dict, Any

class [EndpointName]Metadata:
    """Metadata for [endpoint description] endpoint."""
    
    @staticmethod
    def get_title() -> str:
        return "<h2>🎯 [Endpoint Title]</h2>"
    
    @staticmethod
    def get_description() -> str:
        return """<p>[Main description of what this endpoint does].</p>"""

    @staticmethod
    def get_features() -> str:
        return """
<h3>✨ <strong>Key Features</strong></h3>

<ul>
<li><strong>Feature 1</strong>: Description of feature 1</li>
<li><strong>Feature 2</strong>: Description of feature 2</li>
<li><strong>Feature 3</strong>: Description of feature 3</li>
</ul>
"""

    @staticmethod
    def get_use_cases() -> str:
        return """
<h3>🎯 <strong>Use Cases</strong></h3>

<ul>
<li><strong>Use Case 1</strong>: Description</li>
<li><strong>Use Case 2</strong>: Description</li>
<li><strong>Use Case 3</strong>: Description</li>
</ul>
"""

    @staticmethod
    def get_performance() -> str:
        return """
<h3>⚡ <strong>Performance</strong></h3>

<ul>
<li><strong>Response Time</strong>: &lt; 200ms typical</li>
<li><strong>Rate Limiting</strong>: [Rate limit details]</li>
<li><strong>Caching</strong>: [Caching details]</li>
</ul>
"""

    @staticmethod
    def get_200_response() -> Dict[str, Any]:
        """Get 200 response description."""
        return {
            "description": "[Success response description]",
            "content": {
                "application/json": {
                    "example": {
                        # Example response data
                        "field1": "value1",
                        "field2": "value2"
                    }
                }
            }
        }
        
    @staticmethod
    def get_500_response() -> Dict[str, Any]:
        """Get 500 response description."""
        return {
            "description": "[Error response description]",
            "content": {
                "application/json": {
                    "example": {
                        "error": "Service error message",
                        "timestamp": "2025-09-11T10:30:00.000Z"
                    }
                }
            }
        }

    @classmethod
    def get_full_description(cls) -> str:
        """Get complete description by combining all sections."""
        sections = [
            cls.get_title(),
            cls.get_description(),
            cls.get_features(),
            cls.get_use_cases(),
            cls.get_performance()
        ]
        
        return "".join(sections)
    
    @classmethod
    def get_responses(cls) -> Dict[int, Dict[str, Any]]:
        """Get all response descriptions."""
        return {
            200: cls.get_200_response(),
            500: cls.get_500_response()
        }
        
    @classmethod
    def get_metadata(cls) -> Dict[str, Any]:
        """Get complete metadata for FastAPI endpoint configuration."""
        return {
            "summary": "🎯 [Endpoint Summary]",
            "description": cls.get_full_description(),
            "response_description": "[Brief response description]",
            "responses": cls.get_responses(),
            "operation_id": "[endpoint_operation_id]",
            "response_model_exclude_unset": True,
            "response_model_exclude_none": True
        }
```

### Langkah 3: Update Endpoint di `api/[endpoint].py`

#### Template Endpoint:
```python
"""
[Endpoint Description]
"""
from datetime import datetime
from fastapi import APIRouter, HTTPException, status
from ..docs.[endpoint]_metadata import [EndpointName]Metadata
from ..models.schemas import [EndpointName]Response, [EndpointName]ErrorResponse

from ..core.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()

[endpoint]_metadata = [EndpointName]Metadata.get_metadata()

@router.[http_method](
    "/[endpoint-path]", 
    tags=["[Tag Name]"], 
    response_model=[EndpointName]Response,
    **[endpoint]_metadata
)
async def [function_name]() -> [EndpointName]Response:
    """
    **🎯 [Endpoint Function Description]**
    
    [Detailed function description]
    
    **Returns:**
        [EndpointName]Response: [Description of return value]
        
    **Raises:**
        HTTPException: [Description of when exceptions are raised]
    """
    try:
        # Business logic here
        result_data = await some_service_function()
        
        return [EndpointName]Response(
            field_name="value",
            optional_field="optional_value",
            data_object=result_data,
            items_list=["item1", "item2"]
        )
        
    except Exception as e:
        logger.error(f"Failed to [action]: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
        )
```

## 📋 Checklist Refactor untuk Setiap Endpoint

### ✅ Pre-Refactor
- [ ] Identifikasi endpoint yang akan direfactor
- [ ] Backup kode existing jika diperlukan
- [ ] Pastikan endpoint berfungsi sebelum refactor

### ✅ Refactor Process
- [ ] **Step 1**: Buat response models di `schemas.py`
  - [ ] Response model utama dengan validasi
  - [ ] Error response model
  - [ ] Tambahkan contoh dan deskripsi lengkap
  
- [ ] **Step 2**: Buat metadata class di `docs/`
  - [ ] Implementasi semua method yang diperlukan
  - [ ] Sesuaikan deskripsi dengan endpoint
  - [ ] Tambahkan contoh response yang akurat
  
- [ ] **Step 3**: Update endpoint
  - [ ] Import response models dan metadata
  - [ ] Update decorator dengan response_model
  - [ ] Ganti return dict dengan response model
  - [ ] Implementasi error handling dengan HTTPException

### ✅ Post-Refactor
- [ ] Test endpoint masih berfungsi
- [ ] Cek dokumentasi di `/docs` dan `/redoc`
- [ ] Pastikan tidak ada error import
- [ ] Validasi response structure

## 🎨 Best Practices

### 1. **Naming Conventions**
```python
# Response Models
class HealthResponse(BaseModel)     # PascalCase + "Response"
class PredictionResponse(BaseModel) # PascalCase + "Response"

# Metadata Classes  
class HealthMetadata               # PascalCase + "Metadata"
class PredictionMetadata          # PascalCase + "Metadata"

# Files
health_metadata.py                # snake_case + "_metadata.py"
prediction_metadata.py           # snake_case + "_metadata.py"
```

### 2. **Emoji Usage Guidelines**
- 🎯 **Endpoints/Actions**: Target, action-oriented
- 🏥 **Health/Status**: Medical, system health
- 📊 **Data/Analytics**: Charts, metrics
- 🔒 **Security**: Locks, shields
- ⚡ **Performance**: Lightning, speed
- 📋 **Lists/Collections**: Clipboard, lists
- 💡 **Tips/Information**: Lightbulb, info
- ❌ **Errors**: Red X, warnings
- ✅ **Success**: Green checkmark
- ⏰ **Time/Timestamps**: Clock, time

### 3. **Field Description Templates**
```python
# Required fields
field: str = Field(..., description="🎯 Clear description", example="example")

# Optional fields  
field: Optional[str] = Field(None, description="💡 Optional description", example="example")

# Complex objects
field: Dict = Field(..., description="📊 Complex data", example={"key": "value"})

# Lists
field: List[str] = Field(..., description="📋 List description", example=["item1", "item2"])

# Timestamps
timestamp: str = Field(..., description="⏰ ISO format timestamp", example="2025-09-11T10:30:00.000Z")
```

### 4. **Error Handling Pattern**
```python
try:
    # Business logic
    result = await service_function()
    
    return ResponseModel(**result)
    
except SpecificException as e:
    logger.error(f"Specific error in [endpoint]: {e}")
    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail={"error": "Specific error message", "timestamp": datetime.now().isoformat()}
    )
    
except Exception as e:
    logger.error(f"Unexpected error in [endpoint]: {e}")
    raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail={"error": str(e), "timestamp": datetime.now().isoformat()}
    )
```

## 🔧 Endpoint-Specific Adaptations

### Health Endpoint (`/health`)
- **Focus**: System status, uptime, service availability
- **Tags**: `["Health"]`
- **Models**: `HealthResponse`, `HealthErrorResponse`
- **Metadata**: `HealthMetadata`

### Prediction Endpoint (`/pneumonia/predict`)
- **Focus**: AI prediction, model inference, medical analysis
- **Tags**: `["Prediction", "AI"]`
- **Models**: `PredictionResponse`, `PredictionErrorResponse`
- **Metadata**: `PredictionMetadata`
- **Special**: File upload handling, image processing

### Stats Endpoint (`/security/stats`)
- **Focus**: Security analytics, metrics, threat analysis
- **Tags**: `["Security", "Analytics"]`
- **Models**: `SecurityStatsResponse`, `SecurityStatsErrorResponse`
- **Metadata**: `SecurityStatsMetadata`

## 🚀 Quick Start Example

Untuk merefactor endpoint `/health`, ikuti langkah ini:

1. **Buat `HealthResponse` di `schemas.py`**
2. **Buat `health_metadata.py` di folder `docs/`**
3. **Update `api/health.py` dengan pattern yang sama**
4. **Test dokumentasi di `/docs` dan `/redoc`**

## 📝 Template Files

Template lengkap untuk setiap jenis endpoint tersedia di folder ini:
- `templates/response_model_template.py`
- `templates/metadata_template.py`
- `templates/endpoint_template.py`

## 🎯 Hasil yang Diharapkan

Setelah refactor selesai, setiap endpoint akan memiliki:
- ✅ Dokumentasi yang konsisten dan kaya
- ✅ Type safety dengan Pydantic
- ✅ Error handling yang terstruktur
- ✅ Contoh response yang akurat
- ✅ Validasi input/output otomatis
- ✅ Dokumentasi yang render dengan baik di Swagger UI dan ReDoc

---

*Panduan ini dibuat berdasarkan refactor endpoint `/status` yang telah berhasil dilakukan. Ikuti pola yang sama untuk konsistensi di seluruh API.*
