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
<h2>🏥 Medical AI API for Chest X-ray Pneumonia Detection</h2>

<h3>⚠️ <strong>Important Medical Disclaimer</strong></h3>

<p><strong>This API is designed for educational and research purposes only.</strong></p>

<p>The predictions provided by this system should <strong>NEVER</strong> be used as a substitute for professional medical diagnosis or treatment. Always consult qualified healthcare professionals for medical advice, diagnosis, or treatment decisions.</p>

<hr>

<h3>🚀 <strong>Key Features</strong></h3>

<ul>
<li><strong>🤖 AI-Powered Detection</strong>: Advanced deep learning models for pneumonia detection</li>
<li><strong>📊 Confidence Scoring</strong>: Detailed probability distributions and confidence levels</li>
<li><strong>💡 Smart Recommendations</strong>: Medical recommendations based on AI predictions</li>
<li><strong>🔒 Enterprise Security</strong>: Multi-layer security with advanced rate limiting</li>
<li><strong>✅ Input Validation</strong>: Comprehensive file and image validation</li>
<li><strong>📈 Real-time Monitoring</strong>: Request logging and performance tracking</li>
</ul>

<h3>🔧 <strong>Available Models</strong></h3>

<ol>
<li><strong>Standard Model</strong> (<code>standard</code>): Baseline CNN architecture</li>
<li><strong>EfficientNet-B0</strong> (<code>efficientnet_b0</code>): Advanced transfer learning model</li>
</ol>

<h3>🛡️ <strong>Security Features</strong></h3>

<ul>
<li><strong>Rate Limiting</strong>: 5 requests per minute per IP address</li>
<li><strong>File Validation</strong>: Size (max 10MB) and type (JPG, JPEG, PNG) validation</li>
<li><strong>Content Analysis</strong>: AI-powered image content validation</li>
<li><strong>Duplicate Detection</strong>: Prevents repeated uploads of identical images</li>
<li><strong>Request Monitoring</strong>: Comprehensive logging and attack detection</li>
<li><strong>IP Protection</strong>: Multi-layer IP-based security measures</li>
</ul>

<h3>📋 <strong>Supported File Formats</strong></h3>

<ul>
<li><strong>JPEG/JPG</strong>: Recommended for medical images</li>
<li><strong>PNG</strong>: High quality lossless format</li>
<li><strong>Maximum Size</strong>: 10MB per file</li>
</ul>

<h3>🎯 <strong>API Usage Examples</strong></h3>

<h4>Basic Prediction Request:</h4>
<pre><code>curl -X POST "http://localhost:8000/pneumonia/predict" \\
     -H "Content-Type: multipart/form-data" \\
     -F "file=@chest_xray.jpg"</code></pre>

<h4>Using Specific Model:</h4>
<pre><code>curl -X POST "http://localhost:8000/pneumonia/predict?model=efficientnet_b0" \\
     -H "Content-Type: multipart/form-data" \\
     -F "file=@chest_xray.jpg"</code></pre>

<h3>📊 <strong>Response Format</strong></h3>

<p>All prediction responses include:</p>
<ul>
<li><strong>Prediction</strong>: NORMAL or PNEUMONIA classification</li>
<li><strong>Confidence</strong>: Numerical confidence score (0.0-1.0)</li>
<li><strong>Probabilities</strong>: Individual class probabilities</li>
<li><strong>Medical Recommendation</strong>: Contextual medical guidance</li>
<li><strong>Model Information</strong>: Version and type used for prediction</li>
</ul>

<h3>🔍 <strong>Monitoring Endpoints</strong></h3>

<ul>
<li><strong>Health Check</strong>: <code>/</code> or <code>/health</code> - Service status and uptime</li>
<li><strong>Model Info</strong>: <code>/pneumonia/model/info</code> - Detailed model information</li>
<li><strong>Security Status</strong>: <code>/security/status</code> - Security system status</li>
<li><strong>Security Stats</strong>: <code>/security/stats</code> - Detailed security metrics</li>
</ul>

<h3>📚 <strong>Documentation</strong></h3>

<ul>
<li><strong>Interactive API Docs</strong>: <code>/docs</code> (Swagger UI)</li>
<li><strong>Alternative Docs</strong>: <code>/redoc</code> (ReDoc)</li>
</ul>

<hr>

<p><strong>Built with FastAPI</strong> | <strong>Powered by ONNX</strong>
""",
        version=settings.app_version,
        docs_url="/docs",
        redoc_url="/redoc",
        debug=settings.debug,
        contact={
            "name": "Pneumonia Detection API",
            "url": "https://github.com/IbnuSabilGitHub/Pneumonia-Detection-API",
        },
        license_info={
            "name": "MIT License",
            "url": "https://opensource.org/licenses/MIT",
        },
        servers=[
            {
                "url": "http://localhost:8000",
                "description": "Development server"
            },
            {
                "url": "https://your-production-domain.com",
                "description": "Production server"
            }
        ],
        openapi_tags=[
            {
                "name": "Health",
                "description": "Health check and monitoring endpoints for service status"
            },
            {
                "name": "Pneumonia Detection",
                "description": "AI-powered pneumonia detection from chest X-ray images"
            },
            {
                "name": "Model",
                "description": "Machine learning model information and statistics"
            },
            {
                "name": "Security",
                "description": "Security status and protection metrics"
            }
        ]
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
    
    # Include routers (avoid passing tags here to prevent duplicates in /redoc)
    app.include_router(health.router)
    app.include_router(prediction.router, prefix="/pneumonia")
    app.include_router(security.router, prefix="/security")
    
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
