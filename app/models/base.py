"""
pydantic models for base error responses.
"""

import time
from typing import Optional

from pydantic import BaseModel, Field


class BaseErrorResponse(BaseModel):
    """Base error response with common fields for consistency."""

    detail: str = Field(
        ..., description="Human-readable error message explaining what went wrong"
    )
    error_code: str = Field(
        ..., description="Machine-readable error code for programmatic handling"
    )
    timestamp: str = Field(
        default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime()),
        description="Error occurrence timestamp in ISO 8601 format",
    )


class AdvancedProtection(BaseModel):
    """Detailed security metrics and protection data."""

    global_attack_score: float = Field(
        ...,
        description="Global attack probability score (0.0-1.0)",
        ge=0.0,
        le=1.0,
        example=0.15,
    )
    requests_per_minute: int = Field(
        ..., description="Total requests in the last minute", ge=0, example=23
    )
    recent_unique_ips: int = Field(
        ...,
        description="Number of unique IP addresses in recent activity",
        ge=0,
        example=8,
    )
    blocked_fingerprints: int = Field(
        ...,
        description="Number of currently blocked request fingerprints",
        ge=0,
        example=2,
    )
    storage_type: str = Field(
        ..., description="Type of storage backend being used", example="memory"
    )
    avg_response_time_ms: Optional[float] = Field(
        None, description="Average response time in milliseconds", example=85.0
    )
    total_requests_24h: Optional[int] = Field(
        None, description="Total requests in the last 24 hours", example=1847
    )

    class Config:
        json_schema_extra = {
            "example": {
                "global_attack_score": 0.15,
                "requests_per_minute": 23,
                "recent_unique_ips": 8,
                "blocked_fingerprints": 2,
                "storage_type": "memory",
                "avg_response_time_ms": 85.0,
                "total_requests_24h": 1847,
            }
        }
