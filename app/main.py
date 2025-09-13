import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, status
from fastapi.responses import JSONResponse
from .core.middleware_factory import MiddlewareFactory
from .docs.api_metadata import APIMetadata
from .core.settings import settings
from .core.logger import setup_logging, get_logger
from .api import health, prediction, status, stats, model_info

from .core.startup_manager import StartupManager
from .core.dependencies import get_dependencies

# Setup logging
setup_logging()
logger = get_logger(__name__)

# Remove global variable
# prediction_service: PneumoniaPredictionService = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Simplified application lifespan manager using dependency injection.
    
    Handles startup and shutdown events for the FastAPI application.
    """
    # Startup
    logger.info(f"🚀 Starting {settings.app_name} v{settings.app_version}")
    
    # Initialize startup manager and dependencies
    startup_manager = StartupManager()
    dependencies = get_dependencies()
    
    # Initialize services using startup manager
    startup_result = await startup_manager.startup(
        storage_config=settings.get_storage_config()
    )
    
    # Inject dependencies
    if 'prediction' in startup_result['services']:
        dependencies.prediction_service = startup_result['services']['prediction']
        # Also set in legacy global for backward compatibility
        health.get_prediction_service._service = startup_result['services']['prediction']
        prediction.get_prediction_service._service = startup_result['services']['prediction']
    
    if 'rate_limiter' in startup_result['services']:
        dependencies.rate_limiter = startup_result['services']['rate_limiter']
        # Also set in legacy global for backward compatibility
        from .core.advanced_rate_limiting import set_rate_limiter
        set_rate_limiter(startup_result['services']['rate_limiter'])
    
    # Mark dependencies as initialized
    dependencies.mark_initialized()
    
    # Log startup summary
    if startup_result['errors']:
        logger.warning(f"⚠️ Application started with {len(startup_result['errors'])} errors")
        for error in startup_result['errors']:
            logger.warning(f"   • {error}")
    
    if startup_result['warnings']:
        logger.warning(f"⚠️ Application started with {len(startup_result['warnings'])} warnings")
        for warning in startup_result['warnings']:
            logger.warning(f"   • {warning}")
    
    if startup_result['success_count'] == startup_result['total_services']:
        logger.info(f"✅ All {startup_result['total_services']} services initialized successfully")
    else:
        logger.warning(f"⚠️ {startup_result['success_count']}/{startup_result['total_services']} services initialized")
    
    yield
    
    # Shutdown
    logger.info("🛑 Application shutdown initiated")
    
    try:
        await startup_manager.shutdown()
        logger.info("✅ Application shutdown completed successfully")
    except Exception as e:
        logger.error(f"❌ Error during shutdown: {e}")
        logger.info("🔚 Application shutdown completed with errors")


def create_app() -> FastAPI:
    """
    Create and configure FastAPI application.
    
    Returns:
        Configured FastAPI application instance
    """
    # Get app metadata from template
    app_metadata = APIMetadata.get_app_metadata()
    
    # create FastAPi metadata from template
    app = FastAPI(lifespan=lifespan, **app_metadata)
    
    # Setup all middleware using factory
    MiddlewareFactory.setup_all_middleware(app)
    
    # Include routers
    app.include_router(health.router)
    app.include_router(prediction.router, prefix="/pneumonia")
    app.include_router(model_info.router, prefix="/pneumonia")
    app.include_router(status.router, prefix="/security")
    app.include_router(stats.router, prefix="/security")
    
    # Setup global exception handlers
    _setup_exception_handlers(app)
    
    logger.info(f"FastAPI application created: {settings.app_name} v{settings.app_version}")
    return app


def _setup_exception_handlers(app: FastAPI) -> None:
    """Setup global exception handlers for the application."""
    
    @app.exception_handler(413)
    async def request_entity_too_large_handler(request, exc):
        """Handle file too large errors."""
        return JSONResponse(
            status_code=413,
            content={
                "detail": "File too large",
                "error_code": "FILE_TOO_LARGE"
            },
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS", 
                "Access-Control-Allow-Headers": "*",
                "Content-Type": "application/json"
            }
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
                "details": {"retry_after": 60}
            },
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
                "Access-Control-Allow-Headers": "*", 
                "Content-Type": "application/json",
                "Retry-After": "60"
            }
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
                    "docs": "/docs"
                }
            },
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
                "Access-Control-Allow-Headers": "*",
                "Content-Type": "application/json"
            }
        )


# Create application instance
app = create_app()


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level=settings.log_level.lower()
    )