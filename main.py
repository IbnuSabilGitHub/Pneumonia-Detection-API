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
        version="2.0.0"
    )
    
    @app.get("/")
    async def root():
        return {
            "message": "Pneumonia Detection API is running",
            "status": "healthy",
            "note": "Please check if all dependencies are installed correctly"
        }

# For FastAPI CLI compatibility
application = app

if __name__ == "__main__":
    import uvicorn
    
    # Get port from environment or use default
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    
    print(f"Starting Pneumonia Detection API on {host}:{port}")
    uvicorn.run(app, host=host, port=port, reload=True)