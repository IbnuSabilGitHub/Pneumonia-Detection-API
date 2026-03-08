"""
Entry point for the Pneumonia Detection API.
Supports both direct execution and FastAPI CLI.
"""

import os
import sys
from pathlib import Path

from app.version import CURRENT_API_VERSION

APP_VERSION = CURRENT_API_VERSION

# Add the current directory to Python path for proper imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# Import the FastAPI app from the refactored structure
try:
    # Try importing from the app module
    from app.main import app
except ImportError as e:
    # Fallback: create a simple app if import fails
    from fastapi import FastAPI
    from pydantic import BaseModel
    import sys
    
    print(f"⚠️  Warning: Failed to import app.main: {e}", file=sys.stderr)
    print("⚠️  Running in fallback mode with limited functionality", file=sys.stderr)

    class FallbackHealthResponse(BaseModel):
        """Fallback health response model."""
        message: str
        status: str
        version: str
        note: str

    app = FastAPI(
        title="Pneumonia Detection API",
        description="API for pneumonia detection from chest X-ray images",
        version=APP_VERSION,
    )

    @app.get(
        "/",
        response_model=FallbackHealthResponse,
        tags=["Health"],
        summary="Health Check (Fallback Mode)"
    )
    @app.get(
        "/health",
        response_model=FallbackHealthResponse,
        tags=["Health"],
        summary="Health Check (Fallback Mode)"
    )
    async def fallback_health_check() -> FallbackHealthResponse:
        """Fallback health check endpoint when main app fails to load."""
        return FallbackHealthResponse(
            message="Pneumonia Detection API is running in fallback mode",
            status="degraded",
            version=APP_VERSION,
            note="Fallback mode - Please check if all dependencies are installed correctly",
        )


# For FastAPI CLI compatibility
application = app

if __name__ == "__main__":
    import uvicorn

    # Get port from environment or use default
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")

    # Production settings
    debug_mode = os.getenv("DEBUG", "false").lower() == "true"

    print(f"Starting Pneumonia Detection API on {host}:{port}")
    print(f"Debug mode: {debug_mode}")
    print(f"Environment: {'development' if debug_mode else 'production'}")

    uvicorn.run(
        app,
        host=host,
        port=port,
        reload=debug_mode,  # Only reload in development
        log_level="info",
        access_log=True,
    )
