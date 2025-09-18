from fastapi.openapi.utils import get_openapi

from .core.settings import settings


def custom_openapi(app):
    if app.openapi_schema:
        return app.openapi_schema
    schema = get_openapi(
        title="Pneumonia Detection API",
        version=settings.app_version,
        description="AI Pneumonia Detection Service",
        routes=app.routes,
    )
    schema["info"]["license"] = {"name": "MIT"}
    app.openapi_schema = schema
    return app.openapi_schema
