"""
Entry point for the Pneumonia Detection API.
Supports both direct execution and FastAPI CLI.
"""
import os
import sys
from pathlib import Path

# Add the current directory to Python path for proper imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# Import the FastAPI app from the refactored structure
try:
    # Try importing from the app module 
    from app.main import app
except ImportError:
    # Fallback: create a simple app if import fails
    from fastapi import FastAPI
    
    app = FastAPI(
        title="Pneumonia Detection API",
        description="API for pneumonia detection from chest X-ray images",
        version="3.1.0"
    )
    
    @app.get("/")
    @app.get("/health")
    async def root():
        return {
            "message": "Pneumonia Detection API is running",
            "status": "healthy",
            "version": "3.1.0",
            "note": "Fallback mode - Please check if all dependencies are installed correctly"
        }

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
        access_log=True
    )