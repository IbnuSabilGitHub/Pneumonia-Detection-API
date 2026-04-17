"""
Main application module for the FastAPI web service
"""
import asyncio
import time
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from .api import health, model_info, prediction
from .core.dependencies import get_dependencies
from .core.logger import get_logger, setup_logging
from .core.middleware_factory import MiddlewareFactory
from .core.settings import settings
from .core.startup_manager import StartupManager
from .core.user_rate_limiting import set_user_rate_limiter
from .docs.sections.api_metadata import ApiMetadata
from .openapi import custom_openapi
from .utils.security import file_hash_cache

# Setup logging
setup_logging()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI):
    """
    Simplified application lifespan manager using dependency injection.

    Handles startup and shutdown events for the FastAPI application.
    """
    # Startup
    logger.info("🚀 Starting %s %s", settings.app_name, settings.app_version)

    # Initialize startup manager and dependencies
    startup_manager = StartupManager()
    dependencies = get_dependencies()

    # Get user rate limiting config (JWT auth is always enabled)
    user_rate_config = settings.get_user_rate_limiting_config()

    # Initialize services using startup manager
    startup_result = await startup_manager.startup(
        user_rate_config=user_rate_config
    )

    # Inject dependencies
    if "prediction" in startup_result["services"]:
        dependencies.prediction_service = startup_result["services"]["prediction"]

    if "user_rate_limiter" in startup_result["services"]:
        dependencies.user_rate_limiter = startup_result["services"]["user_rate_limiter"]
        # Also set global for middleware access
        set_user_rate_limiter(startup_result["services"]["user_rate_limiter"])

    # Mark dependencies as initialized
    dependencies.mark_initialized()

    # Log startup summary
    if startup_result["errors"]:
        logger.warning(
            "⚠️ Application started with %d errors", len(startup_result['errors'])
        )
        for error in startup_result["errors"]:
            logger.warning("   • %s", error)

    if startup_result["warnings"]:
        logger.warning(
            "⚠️ Application started with %d warnings", len(startup_result['warnings'])
        )
        for warning in startup_result["warnings"]:
            logger.warning("   • %s", warning)

    if startup_result["success_count"] == startup_result["total_services"]:
        logger.info(
            "✅ All %d services initialized successfully",
            startup_result['total_services'],
        )
    else:
        logger.warning(
            "⚠️ %d/%d services initialized",
            startup_result['success_count'],
            startup_result['total_services'],
        )

    # Start background task: periodic cleanup of expired file hash cache
    cleanup_interval = max(
        60, getattr(settings, "memory_cleanup_interval", 300)
    )  # ensure minimum interval
    stop_event = asyncio.Event()

    async def cache_cleanup_worker():
        while not stop_event.is_set():
            try:
                removed = file_hash_cache.cleanup_expired()
                if removed:
                    logger.info(
                        "🧹 File hash cache cleanup removed %d expired entries", removed
                    )
            except Exception as e:  # pylint: disable=broad-except
                logger.error("Cache cleanup worker error: %s", e)
            # Wait with cancellation awareness
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=cleanup_interval)
            except asyncio.TimeoutError:
                continue

    cleanup_task = asyncio.create_task(cache_cleanup_worker())

    yield

    # Shutdown
    logger.info("🛑 Application shutdown initiated")

    try:
        # Signal and await cleanup task
        stop_event.set()
        if cleanup_task:
            await cleanup_task
        await startup_manager.shutdown()
        logger.info("✅ Application shutdown completed successfully")
    except RuntimeError as e:
        logger.error("❌ Error during shutdown: %s", e)
        logger.info("🔚 Application shutdown completed with errors")


def create_app() -> FastAPI:
    """
    Create and configure FastAPI application.

    Returns:
        Configured FastAPI application instance
    """
    # Get app metadata from template
    app_metadata = ApiMetadata.get_app_metadata()

    # create FastAPi metadata from template
    app_instance = FastAPI(lifespan=lifespan, **app_metadata)
    app_instance.openapi = lambda: custom_openapi(app_instance)
    # Setup all middleware using factory
    MiddlewareFactory.setup_all_middleware(app_instance)

    # Include routers
    app_instance.include_router(health.router)
    app_instance.include_router(prediction.router, prefix="/pneumonia")
    app_instance.include_router(model_info.router, prefix="/pneumonia")

    # Setup global exception handlers
    _setup_exception_handlers(app_instance)

    logger.info(
        "FastAPI application created: %s v%s", settings.app_name, settings.app_version
    )
    return app_instance


def _setup_exception_handlers(app: FastAPI) -> None:
    """Setup global exception handlers for the application."""

    @app.exception_handler(413)
    async def request_entity_too_large_handler(request, exc):
        """Handle file too large errors."""
        return JSONResponse(
            status_code=413,
            content={"detail": "File too large", "error_code": "FILE_TOO_LARGE"},
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
                "Access-Control-Allow-Headers": "*",
                "Content-Type": "application/json",
            },
        )

    @app.exception_handler(429)
    async def rate_limit_handler(request, exc):
        """Handle rate limit exceeded errors with proper CORS headers."""
        return JSONResponse(
            status_code=429,
            content={
                "error": "Rate limit exceeded",
                "message": "Too many requests",
                "endpoint": request.url.path,
                "timestamp": time.time(),
                "details": {"retry_after": 60},
            },
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
                "Access-Control-Allow-Headers": "*",
                "Content-Type": "application/json",
                "Retry-After": "60",
            },
        )

    @app.exception_handler(404)
    async def not_found_handler(request, exc):
        """Handle 404 errors with custom message."""
        return JSONResponse(
            status_code=404,
            content={
                "detail": "Endpoint not found",
                "error_code": "NOT_FOUND",
                "available_endpoints": {
                    "health": "/",
                    "prediction": "/pneumonia/predict",
                    "model_info": "/pneumonia/model/info",
                    "security_status": "/security/status",
                    "security_stats": "/security/stats",
                    "docs": "/docs",
                },
            },
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
                "Access-Control-Allow-Headers": "*",
                "Content-Type": "application/json",
            },
        )


# Create application instance
app = create_app()


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level=settings.log_level.lower(),
    )
