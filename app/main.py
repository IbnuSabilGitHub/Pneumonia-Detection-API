import sys
import os
from pathlib import Path
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
from .api import health, prediction
from .middleware.security import (
    rate_limit_middleware,
    logging_middleware,
    error_handling_middleware
)
from .utils.exceptions import ModelLoadError

# Setup logging
setup_logging()
logger = get_logger(__name__)

# Global prediction service instance
prediction_service: PneumoniaPredictionService = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    
    Handles startup and shutdown events for the FastAPI application.
    """
    global prediction_service
    
    # Startup
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")
    
    try:
        # Initialize prediction service
        prediction_service = PneumoniaPredictionService()
        prediction_service.load_model()
        
        # Inject service into route dependencies
        health.get_prediction_service._service = prediction_service
        prediction.get_prediction_service._service = prediction_service
        
        logger.info("Application startup completed successfully")
        
        yield
        
    except ModelLoadError as e:
        logger.error(f"Failed to load model during startup: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Model initialization failed"
        )
    except Exception as e:
        logger.error(f"Unexpected error during startup: {e}")
        raise
    
    # Shutdown
    logger.info("Application shutdown")


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
    app.middleware("http")(rate_limit_middleware)
    
    # Include routers
    app.include_router(health.router)
    app.include_router(
        prediction.router, 
        prefix="/pneumonia", 
        tags=["Pneumonia Detection"]
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
