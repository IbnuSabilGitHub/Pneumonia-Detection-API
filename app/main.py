from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from .core.settings import settings
from .core.logger import setup_logging, get_logger
from .services.prediction import PneumoniaPredictionService
from .api import health, prediction, security
from .middleware.security import (
    logging_middleware,
    error_handling_middleware,
    SecurityMiddleware
)
from .utils.exceptions import ModelLoadError
from .core.advanced_rate_limiting import create_advanced_rate_limiter
from .core.storage_factory import StorageType

# Setup logging
setup_logging()
logger = get_logger(__name__)

# Global prediction service instance
prediction_service: PneumoniaPredictionService = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager with Redis storage initialization.
    
    Handles startup and shutdown events for the FastAPI application.
    """
    global prediction_service, advanced_rate_limiter
    
    # Startup
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")
    
    # Initialize with fallback mechanisms
    startup_errors = []
    
    # Try to initialize prediction service
    try:
        prediction_service = PneumoniaPredictionService()
        prediction_service.load_model()
        logger.info("Prediction service initialized successfully")
    except Exception as e:
        logger.error(f"Failed to load model during startup: {e}")
        startup_errors.append(f"Model loading failed: {e}")
        prediction_service = None
    
    # Try to initialize rate limiter with in-memory storage as default
    try:
        storage_type = StorageType.MEMORY  # Default to memory storage
        storage_config = settings.get_storage_config()
        
        logger.info(f"Initializing rate limiter with {storage_type.value} storage")
        new_rate_limiter = await create_advanced_rate_limiter(
            storage_type=storage_type,
            storage_config=storage_config
        )
        
        # Update global variable using setter function
        from app.core.advanced_rate_limiting import set_rate_limiter
        set_rate_limiter(new_rate_limiter)
        
        # Test storage connection
        if new_rate_limiter.storage:
            storage_info = await new_rate_limiter.storage.get_info()
            logger.info(f"Storage backend initialized: {storage_info}")
            
    except Exception as e:
        logger.warning(f"Rate limiter initialization failed, using fallback: {e}")
        startup_errors.append(f"Rate limiter failed: {e}")
        # Create a minimal fallback rate limiter
        try:
            new_rate_limiter = await create_advanced_rate_limiter(
                storage_type=StorageType.MEMORY,
                storage_config={"max_size": 1000}
            )
            from app.core.advanced_rate_limiting import set_rate_limiter
            set_rate_limiter(new_rate_limiter)
        except Exception as fallback_error:
            logger.error(f"Fallback rate limiter also failed: {fallback_error}")
    
    # Inject services into route dependencies (even if None)
    health.get_prediction_service._service = prediction_service
    prediction.get_prediction_service._service = prediction_service
    
    if startup_errors:
        logger.warning(f"Application started with {len(startup_errors)} warnings: {startup_errors}")
    else:
        logger.info("Application startup completed successfully")
    
    yield
    logger.info("Application shutdown initiated")
    
    try:
        # Cleanup storage connections
        from .core.advanced_rate_limiting import get_rate_limiter
        current_rate_limiter = get_rate_limiter()
        
        if current_rate_limiter and current_rate_limiter.storage:
            # Check if disconnect method exists (Redis storage) before calling
            if hasattr(current_rate_limiter.storage, 'disconnect'):
                await current_rate_limiter.storage.disconnect()
                logger.info("Storage connections closed")
            else:
                logger.info("In-memory storage cleanup completed")
        
        # Cleanup prediction service
        if prediction_service:
            if hasattr(prediction_service, 'cleanup'):
                prediction_service.cleanup()
                logger.info("Prediction service cleaned up")
            else:
                logger.info("Prediction service cleanup not needed")
            
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")
    
    logger.info("Application shutdown completed")


def create_app() -> FastAPI:
    """
    Create and configure FastAPI application.
    
    Returns:
        Configured FastAPI application instance
    """
    app = FastAPI(
        lifespan=lifespan,
        title=settings.app_name,
        description="""
        Medical AI API for chest X-ray pneumonia detection.
        
        **⚠️ Important Disclaimer:**
        This API is designed for educational and research purposes only.
        The predictions should never be used as a substitute for professional 
        medical diagnosis or treatment. Always consult qualified healthcare 
        professionals for medical advice.
        
        **Features:**
        - AI-powered pneumonia detection from chest X-rays
        - Confidence scoring and probability distributions
        - Medical recommendations based on predictions
        - Built-in security features and rate limiting
        - Comprehensive input validation
        
        **Security:**
        - Rate limiting (5 requests/minute per IP)
        - File size and type validation
        - Image content validation
        - Duplicate detection
        - Request logging and monitoring
        """,
        version=settings.app_version,
        docs_url="/docs",
        redoc_url="/redoc",
        debug=settings.debug
    )
    
    # Add rate limiter for slowapi compatibility
    limiter = Limiter(key_func=get_remote_address)
    security_middleware = SecurityMiddleware(app)
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    
    # Add security middleware
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=settings.trusted_hosts
    )
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )
    
    # Add custom middleware
    # Note: Middleware is applied in reverse order
    app.middleware("http")(error_handling_middleware)
    app.middleware("http")(logging_middleware)
    app.add_middleware(SecurityMiddleware)
    
    # Include routers
    app.include_router(health.router)
    app.include_router(
        prediction.router, 
        prefix="/pneumonia", 
        tags=["Pneumonia Detection"]
    )
    app.include_router(
        security.router,
        prefix="/security",
        tags=["Security"]
    )
    
    # Global exception handlers
    @app.exception_handler(413)
    async def request_entity_too_large_handler(request, exc):
        """Handle file too large errors."""
        return JSONResponse(
            status_code=413,
            content={
                "detail": "File too large",
                "error_code": "FILE_TOO_LARGE"
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
                    "docs": "/docs"
                }
            }
        )
    
    logger.info(f"FastAPI application created: {settings.app_name} v{settings.app_version}")
    return app


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
