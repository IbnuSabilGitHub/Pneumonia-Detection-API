# 🚀 Quick Reference: Refactor Endpoint Documentation

## 📝 Langkah Singkat untuk Refactor Endpoint

### 1. Siapkan Response Models (`schemas.py`)
```python
# Tambahkan ke app/models/schemas.py
class YourEndpointResponse(BaseModel):
    """Response model untuk endpoint Anda"""
    field1: str = Field(..., description="🎯 Deskripsi field", example="contoh")
    timestamp: str = Field(..., description="⏰ Timestamp", example="2025-09-11T10:30:00.000Z")
    
class YourEndpointErrorResponse(BaseModel):
    """Error response model"""
    error: str = Field(..., description="❌ Error message")
    timestamp: str = Field(..., description="⏰ Error timestamp")
```

### 2. Buat Metadata Class (`docs/your_endpoint_metadata.py`)
```python
# Buat file baru: app/docs/your_endpoint_metadata.py
from typing import Dict, Any

class YourEndpointMetadata:
    @staticmethod
    def get_description() -> str:
        return "<p>Deskripsi endpoint Anda</p>"
    
    @classmethod
    def get_metadata(cls) -> Dict[str, Any]:
        return {
            "summary": "🎯 Summary Endpoint",
            "description": cls.get_description(),
            "response_description": "Deskripsi response singkat"
        }
```

### 3. Update Endpoint (`api/your_endpoint.py`)
```python
# Update file endpoint Anda
from fastapi import APIRouter, HTTPException, status
from ..docs.your_endpoint_metadata import YourEndpointMetadata
from ..models.schemas import YourEndpointResponse

router = APIRouter()
metadata = YourEndpointMetadata.get_metadata()

@router.get("/path", response_model=YourEndpointResponse, **metadata)
async def your_function() -> YourEndpointResponse:
    try:
        # Business logic
        return YourEndpointResponse(
            field1="value",
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail={"error": str(e)})
```

## 🎯 Endpoints yang Perlu Direfactor

Berdasarkan struktur project, endpoints yang perlu direfactor:

### 1. Health Endpoint (`/health`)
- **File**: `app/api/health.py`
- **Response Model**: `HealthResponse` ✅ (sudah ada)
- **Metadata**: Perlu dibuat `health_metadata.py`
- **Tags**: `["Health"]`

### 2. Prediction Endpoint (`/pneumonia/predict`)
- **File**: `app/api/prediction.py`
- **Response Model**: `PredictionResponse` ✅ (sudah ada)
- **Metadata**: Perlu dibuat `prediction_metadata.py`
- **Tags**: `["Prediction", "AI"]`
- **Special**: File upload handling

### 3. Security Stats Endpoint (`/security/stats`)
- **File**: `app/api/stats.py`
- **Response Model**: Perlu dibuat `SecurityStatsResponse`
- **Metadata**: Sudah ada `stat_metadata.py` ✅
- **Tags**: `["Security", "Analytics"]`

## 🔧 Contoh Nyata: Health Endpoint

### Step 1: Update `schemas.py`
```python
# Tambahkan ke app/models/schemas.py setelah class HealthResponse yang sudah ada
class HealthDetailedResponse(BaseModel):
    """Extended health response with more details"""
    status: str = Field(..., description="🏥 Service health status", example="healthy")
    model_loaded: bool = Field(..., description="🤖 Model status", example=True)
    version: str = Field(..., description="📌 API version", example="3.4.2")
    uptime: float = Field(..., description="⏱️ Uptime in seconds", example=3600.5)
    dependencies: Dict = Field(..., description="🔗 Dependencies status", example={})
```

### Step 2: Buat `health_metadata.py`
```python
# Buat file: app/docs/health_metadata.py
from typing import Dict, Any

class HealthMetadata:
    @staticmethod
    def get_description() -> str:
        return """
<h2>🏥 API Health Check</h2>
<p>Provides comprehensive health status of the API service including model availability and system uptime.</p>

<h3>✅ <strong>Health Status Levels</strong></h3>
<ul>
<li><strong>healthy</strong>: All systems operational</li>
<li><strong>partial</strong>: Service running with limitations</li>
<li><strong>unhealthy</strong>: Critical issues detected</li>
</ul>
"""
    
    @classmethod
    def get_metadata(cls) -> Dict[str, Any]:
        return {
            "summary": "🏥 API Health Status",
            "description": cls.get_description(),
            "response_description": "Current API health and service status"
        }
```

### Step 3: Update `health.py`
```python
# Update app/api/health.py
from ..docs.health_metadata import HealthMetadata
from ..models.schemas import HealthResponse

metadata = HealthMetadata.get_metadata()

@router.get("/", response_model=HealthResponse, **metadata)
async def get_health() -> HealthResponse:
    # existing logic...
```

## 📋 Checklist Cepat

Untuk setiap endpoint:
- [ ] **Response Model**: Buat/perbarui di `schemas.py`
- [ ] **Metadata Class**: Buat file `[endpoint]_metadata.py`
- [ ] **Update Endpoint**: Tambahkan `response_model` dan `**metadata`
- [ ] **Error Handling**: Ganti return dict dengan HTTPException
- [ ] **Test**: Cek `/docs` dan `/redoc`

## 🎨 Emoji Guide

- 🏥 Health/Medical
- 🤖 AI/Model
- 🛡️ Security
- 📊 Analytics/Stats
- 🎯 Goals/Actions
- ⚡ Performance
- 📋 Lists/Data
- 🔧 Technical
- ❌ Errors
- ✅ Success

## 🚨 Common Issues

1. **Import Error**: Pastikan path import benar
2. **Duplicate Responses**: Jangan tambahkan `responses` di decorator jika sudah ada di metadata
3. **Model Validation**: Pastikan field types sesuai dengan data yang dikembalikan
4. **Error Handling**: Gunakan HTTPException, bukan return dict untuk error

---

*Ikuti template yang ada di folder `doc/templates/` untuk panduan lengkap!*
