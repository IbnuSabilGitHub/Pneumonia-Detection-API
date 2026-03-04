"""
Pydantic schemas for authentication and authorization responses.
"""

from typing import Optional

from pydantic import BaseModel, Field

from .base import BaseErrorResponse


class AuthErrorResponse(BaseErrorResponse):
    """Error response for authentication failures."""

    error_code: str = Field(
        ...,
        description="Machine-readable error code",
        examples=[
            "MISSING_TOKEN",
            "INVALID_TOKEN",
            "TOKEN_EXPIRED",
            "ADMIN_REQUIRED",
            "AUTH_DISABLED",
        ],
    )
    hint: Optional[str] = Field(
        None,
        description="Hint for resolving the authentication issue",
        example="Add header: Authorization: Bearer <your_supabase_access_token>",
    )
    current_role: Optional[str] = Field(
        None,
        description="Current role of the user (if token was valid but insufficient)",
        example="authenticated",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "detail": "Missing authentication",
                "error_code": "MISSING_TOKEN",
                "timestamp": "2026-03-01T10:00:00.000Z",
                "hint": "Add header: Authorization: Bearer <your_supabase_access_token>",
            }
        }


class AuthenticatedUser(BaseModel):
    """Authenticated user information extracted from JWT."""

    user_id: str = Field(
        ...,
        description="Supabase user UUID",
        example="a1b2c3d4-e5f6-7890-abcd-ef1234567890",
    )
    email: Optional[str] = Field(
        None,
        description="User email address",
        example="user@example.com",
    )
    role: str = Field(
        ...,
        description="User role from Supabase JWT",
        example="authenticated",
    )
    is_admin: bool = Field(
        ...,
        description="Whether the user has admin privileges",
        example=False,
    )

    class Config:
        json_schema_extra = {
            "example": {
                "user_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
                "email": "user@example.com",
                "role": "authenticated",
                "is_admin": False,
            }
        }


class TokenInfo(BaseModel):
    """Information about the JWT token (for debug/info purposes)."""

    token_type: str = Field(
        default="Bearer",
        description="Token type",
        example="Bearer",
    )
    auth_provider: str = Field(
        default="supabase",
        description="Authentication provider",
        example="supabase",
    )
    jwt_auth_enabled: bool = Field(
        ...,
        description="Whether JWT authentication is enabled",
        example=True,
    )
    admin_api_key_configured: bool = Field(
        ...,
        description="Whether legacy Admin API Key is configured",
        example=True,
    )

    class Config:
        json_schema_extra = {
            "example": {
                "token_type": "Bearer",
                "auth_provider": "supabase",
                "jwt_auth_enabled": True,
                "admin_api_key_configured": True,
            }
        }
