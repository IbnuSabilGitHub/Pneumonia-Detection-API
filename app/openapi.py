""" 
OpenAPI schema customization for the FastAPI application
"""
from fastapi.openapi.utils import get_openapi

from .core.settings import settings


def custom_openapi(app):
    """Generate a custom OpenAPI schema for the FastAPI application."""
    # Don't cache if routes have changed - regenerate schema each time
    schema = get_openapi(
        title="Pneumonia Detection API",
        version=settings.app_version,
        description="![API Status](https://img.shields.io/endpoint?url=https://pneumonia-detection-api-d7qu.onrender.com/badge.json)",
        routes=app.routes,
    )
    schema["info"]["license"] = {"name": "MIT"}

    # Add security schemes (Supabase JWT only)
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
    }

    # Add security requirement to protected endpoints
    paths = schema.get("paths", {})
    for path_item in paths.values():
        for operation in path_item.values():
            if isinstance(operation, dict):
                # Add security to the predict endpoint and other protected endpoints
                if "operationId" in operation and "predict" in operation.get("operationId", "").lower():
                    operation["security"] = [{"SupabaseJWT": []}]

    app.openapi_schema = schema
    return app.openapi_schema
