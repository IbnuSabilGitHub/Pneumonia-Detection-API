""" 
OpenAPI schema customization for the FastAPI application
"""
from fastapi.openapi.utils import get_openapi

from .core.settings import settings


def custom_openapi(app):
    """Generate a custom OpenAPI schema for the FastAPI application."""
    if app.openapi_schema:
        return app.openapi_schema
    schema = get_openapi(
        title="Pneumonia Detection API",
        version=settings.app_version,
        description="![API Status](https://img.shields.io/endpoint?url=https://pneumonia-detection-api-d7qu.onrender.com/badge.json)",
        routes=app.routes,
    )
    schema["info"]["license"] = {"name": "MIT"}

    # Add security schemes (Supabase JWT + Legacy API Key)
    schema.setdefault("components", {})
    schema["components"]["securitySchemes"] = {
        "SupabaseJWT": {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "JWT",
            "description": (
                "Supabase JWT access token. Obtain by signing in via "
                "Supabase Auth (email/password, OAuth, magic link, etc.)."
            ),
        },
        "AdminAPIKey": {
            "type": "apiKey",
            "in": "header",
            "name": "X-Admin-API-Key",
            "description": (
                "Legacy admin API key for /security/* endpoints. "
                "Set via ADMIN_API_KEY environment variable."
            ),
        },
    }

    app.openapi_schema = schema
    return app.openapi_schema
