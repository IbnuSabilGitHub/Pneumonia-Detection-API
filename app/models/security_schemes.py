from typing import List, Optional
from pydantic import BaseModel, Field
from .base import BaseErrorResponse, AdvancedProtection
from typing import Dict, Any, Optional

class SecurityStatusResponse(BaseModel):
    """
    **Advanced Security System Status Response**
    
    Comprehensive security status response providing real-time information about
    the multi-layer security protection system including threat detection,
    rate limiting, and attack prevention measures.
    
    **Security Layers Monitored:**
    - Multi-layer Rate Limiting (In-Memory)
    - IP Switching Attack Detection (In-Memory)
    - Request Fingerprinting (In-Memory)
    - Behavioral Analysis (In-Memory)
    - Global Attack Scoring (In-Memory)
    - Duplicate File Detection (In-Memory)
    
    **Status Categories:**
    - **active**: Security system fully operational
    - **not_initialized**: Security components not ready
    - **degraded**: Partial security functionality
    """
    
    service: str = Field(
        ..., 
        description="API service name",
        example="Pneumonia Detection API"
    )
    security_status: str = Field(
        ..., 
        description="Overall security system status",
        example="active",
        pattern="^(active|not_initialized|degraded)$"
    )
    timestamp: str = Field(
        ..., 
        description="Status timestamp in ISO format",
        example="2025-09-10T10:30:00.000Z"
    )
    advanced_protection: AdvancedProtection = Field(
        ..., 
        description="Detailed security metrics and protection data"
    )
    protection_features: List[str] = Field(
        ..., 
        description="List of active security protection features",
        example=[
            "Multi-layer Rate Limiting (In-Memory)",
            "IP Switching Attack Detection (In-Memory)",
            "Request Fingerprinting (In-Memory)",
            "Behavioral Analysis (In-Memory)",
            "Global Attack Scoring (In-Memory)",
            "Duplicate File Detection (In-Memory)",
            "Persistent Storage (In-Memory)",
            "Single-instance Optimized (In-Memory)"
        ]
    )
    error: Optional[str] = Field(
        None, 
        description="Error message if security system has issues",
        example="Rate limiter not initialized"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "service": "Pneumonia Detection API",
                "security_status": "active",
                "timestamp": "2025-09-10T10:30:00.000Z",
                "advanced_protection": {
                    "global_attack_score": 0.15,
                    "requests_per_minute": 23,
                    "recent_unique_ips": 8,
                    "blocked_fingerprints": 2,
                    "storage_type": "memory",
                    "avg_response_time_ms": 85.0,
                    "total_requests_24h": 1847
                },
                "protection_features": [
                    "Multi-layer Rate Limiting (In-Memory)",
                    "IP Switching Attack Detection (In-Memory)",
                    "Request Fingerprinting (In-Memory)",
                    "Behavioral Analysis (In-Memory)",
                    "Global Attack Scoring (In-Memory)",
                    "Duplicate File Detection (In-Memory)",
                    "Persistent Storage (In-Memory)",
                    "Single-instance Optimized (In-Memory)"
                ]
            }
        }


class SecurityErrorResponse(BaseErrorResponse):
    """
    **Security System Error Response**
    
    Error response when the security system encounters issues
    or is not properly initialized.
    """
    
    service: str = Field(
        ..., 
        description="API service name",
        example="Pneumonia Detection API"
    )
    service_status: str = Field(
        ..., 
        description="Security system status indicating error state",
        example="not_initialized"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "detail": "Rate limiter not initialized",
                "error_code": "RATE_LIMITER_NOT_INITIALIZED",
                "timestamp": "2025-09-10T10:30:00.000Z",
                "service": "Pneumonia Detection API",
                "service_status": "not_initialized"
            }
        }






# SECURITY STATS ENDPOINT RESPONSE SCHEMAS 

class SecurityMetrics(BaseModel):
    """Security statistics and metrics data."""
    
    total_requests: int = Field(
        ...,
        description="Total number of requests processed",
        ge=0,
        example=1500
    )
    blocked_requests: int = Field(
        ...,
        description="Number of requests blocked by security system",
        ge=0,
        example=25
    )
    unique_ips: int = Field(
        ...,
        description="Number of unique IP addresses seen",
        ge=0,
        example=120
    )
    rate_limited_requests: int = Field(
        ...,
        description="Number of requests blocked by rate limiting",
        ge=0,
        example=15
    )
    attack_attempts: int = Field(
        ...,
        description="Number of detected attack attempts",
        ge=0,
        example=8
    )
    average_response_time: float = Field(
        ...,
        description="Average response time in milliseconds",
        ge=0.0,
        example=95.5
    )
    uptime_hours: float = Field(
        ...,
        description="System uptime in hours",
        ge=0.0,
        example=72.5
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "total_requests": 1500,
                "blocked_requests": 25,
                "unique_ips": 120,
                "rate_limited_requests": 15,
                "attack_attempts": 8,
                "average_response_time": 95.5,
                "uptime_hours": 72.5
            }
        }


class SecurityStatsResponse(BaseModel):
    """
    **Security Statistics Response**
    
    Comprehensive security statistics response providing detailed metrics
    about the API's security performance, request patterns, and threat
    detection effectiveness.
    
    **Metrics Included:**
    - Request volume and patterns
    - Security blocking statistics
    - Attack detection and prevention
    - Performance metrics
    - System uptime information
    """
    
    service: str = Field(
        ...,
        description="API service name",
        example="Pneumonia Detection API"
    )
    timestamp: str = Field(
        ...,
        description="Statistics timestamp in ISO format",
        example="2025-09-10T10:30:00.000Z"
    )
    security_metrics: Dict[str, Any] = Field(
        ...,
        description="Dynamic map of security metrics (may vary by storage backend)"
    )
    status: str = Field(
        ...,
        description="Security system operational status",
        example="active",
        pattern="^(active|not_initialized|degraded)$"
    )
    storage_type: str = Field(
        ...,
        description="Type of storage backend being used",
        example="memory"
    )
    interpretation: Dict[str, Any] | None = Field(
        None,
        description="Human-readable interpretation of key metrics (threat level, request rate, etc.)"
    )
    analytics_summary: Dict[str, Any] | None = Field(
        None,
        description="Optional summarized analytics (daily totals, storage backend info, threat level)"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "service": "Pneumonia Detection API",
                "timestamp": "2025-09-10T10:30:00.000Z",
                "security_metrics": {
                    "global_attack_score": 0.25,
                    "requests_per_minute": 45,
                    "recent_unique_ips": 12,
                    "blocked_fingerprints": 3,
                    "storage_type": "memory"
                },
                "status": "active",
                "storage_type": "memory",
                "interpretation": {
                    "attack_score": {"value": 0.25, "level": "LOW"},
                    "request_rate": {"value": 45},
                    "unique_ips": {"value": 12},
                    "blocked_count": {"value": 3}
                },
                "analytics_summary": {
                    "daily_total": 2847,
                    "threat_level": "LOW",
                    "storage_backend": "memory"
                }
            }
        }


class SecurityStatsErrorResponse(BaseErrorResponse):
    """
    **Security Statistics Error Response**
    
    Error response when security statistics cannot be retrieved
    due to system issues or initialization problems.
    """
    
    service: str = Field(
        ...,
        description="API service name",
        example="Pneumonia Detection API"
    )
    service_status: str = Field(
        ...,
        description="Security system status indicating error state",
        example="not_initialized"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "detail": "Security statistics service not available",
                "error_code": "SECURITY_STATS_UNAVAILABLE",
                "timestamp": "2025-09-10T10:30:00.000Z",
                "service": "Pneumonia Detection API",
                "service_status": "not_initialized"
            }
        }