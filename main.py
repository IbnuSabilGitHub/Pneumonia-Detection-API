from fastapi import FastAPI, File, UploadFile, HTTPException, status, Request
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Dict, Any, Optional
from pathlib import Path
from PIL import Image
import onnxruntime as ort
import numpy as np
import logging
import json
import io
import os
import hashlib
import time
from collections import defaultdict, deque
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Rate Limiter Configuration
limiter = Limiter(key_func=get_remote_address)

# In-memory rate limiting (for development/simple deployments)
class InMemoryRateLimiter:
    def __init__(self):
        self.requests = defaultdict(deque)
        self.blocked_ips = {}
    
    def is_allowed(self, client_ip: str, max_requests: int = 5, window_seconds: int = 60) -> bool:
        current_time = time.time()
        
        # Check if IP is temporarily blocked
        if client_ip in self.blocked_ips:
            if current_time < self.blocked_ips[client_ip]:
                return False
            else:
                del self.blocked_ips[client_ip]
        
        # Clean old requests
        requests = self.requests[client_ip]
        while requests and requests[0] < current_time - window_seconds:
            requests.popleft()
        
        # Check rate limit
        if len(requests) >= max_requests:
            # Block IP for 5 minutes
            self.blocked_ips[client_ip] = current_time + 300
            logger.warning(f"Rate limit exceeded for IP: {client_ip}")
            return False
        
        # Add current request
        requests.append(current_time)
        return True

# Initialize rate limiter
rate_limiter = InMemoryRateLimiter()

# File hash cache for duplicate detection
file_hash_cache = {}
CACHE_DURATION = 300  # 5 minutes

#asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event to initialize model on startup"""
    global classifier
    try:
        classifier = PneumoniaClassifier(MODEL_PATH, STATS_PATH)
        logger.info("Model initialized successfully")
        yield
    except Exception as e:
        logger.error(f"Failed to initialize model: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Model initialization failed"
        )

# Initialize FastAPI app
app = FastAPI(
    lifespan=lifespan,
    title="Pneumonia Detection API",
    description="Medical AI API for chest X-ray pneumonia detection (for educational purposes only)",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add rate limiter to app
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Security middleware
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*.railway.app", "localhost", "127.0.0.1"]
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Configure base on your frontend
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Configuration
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
MODEL_PATH = "models/pneumonia_model.onnx"
STATS_PATH = "models/model_stats.json"

class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    probabilities: Dict[str, float]
    medical_recommendation: str
    model_version: str = "v1.0"
    disclaimer: str = "This model is for educational purposes only. Consult a healthcare professional for medical advice."
    
    
class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str
    
class PneumoniaClassifier:
    def __init__(self, model_path: str, stats_path: str):
        try:
            # Load ONNX model
            self.session = ort.InferenceSession(
                model_path,
                providers=['CPUExecutionProvider']
            )
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
            
            if not os.path.exists(stats_path):
                # Use dataset-typical values for chest X-ray (safe fallback)
                self.mean = 0.449  # Typical chest X-ray mean
                self.std = 0.226   # Typical chest X-ray std
                logger.warning("Model stats file not found, using default values.")
            else:
                # Load model stats
                with open(stats_path, 'r') as f:
                    stats = json.load(f)
                self.mean = stats['mean']
                self.std = stats['std']
            
            self.label_map = {0: 'NORMAL', 1: 'PNEUMONIA'}
            logger.info(f"Model loaded successfully: mean={self.mean:.4f}, std={self.std:.4f}")
        
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
        
    def preprocess_image(self, image: Image.Image) -> np.ndarray:
        """Preprocess image for model interface"""
        # Convert to grayscale and resize
        image = image.convert("L").resize((192, 192))
        
        # Convert dataset normalization 
        img_array = np.array(image, dtype=np.float32 ) / 255.0
        
        # Apply dataset normalization
        img_array = (img_array - self.mean) / self.std
        
        # Add batch and channel dimensions
        img_tensor = img_array[np.newaxis, np.newaxis, :, :]
        
        return img_tensor
    
    def predict(self, image: Image.Image) -> Dict[str, Any]:
        """Predict pneumonia from chest X-ray image"""
        try:
            # Preprocess image
            img_tensor = self.preprocess_image(image)
            
            # ONNX inference
            output = self.session.run([self.output_name], {self.input_name: img_tensor})
            logits = output[0][0]
            
            # Apply softmax
            exp_logits = np.exp(logits - np.max(logits))
            probs = exp_logits / np.sum(exp_logits)
            
            pred_idx = np.argmax(probs)
            confidence = float(probs[pred_idx])
            prediction = self.label_map[pred_idx]
            
            # Medical recommendation base on prediction and confidence
            if prediction == 'PNEUMONIA' and confidence > 0.8:
                recommendation = "🚨 URGENT: Immediate medical consultation required"
            elif prediction == 'PNEUMONIA' and confidence > 0.6:
                recommendation = "⚠️ Medical consultation strongly recommended"
            elif prediction == 'PNEUMONIA':
                recommendation = "⚠️ Consider medical consultation"
            else:
                recommendation = "✅ Normal results - maintain regular health checkups"
            
            return {
                "prediction": prediction,
                "confidence": confidence,
                "probabilities": {
                    "NORMAL": float(probs[0]),
                    "PNEUMONIA": float(probs[1])
                },
                "medical_recommendation": recommendation
            }
        

                    
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Prediction failed: {str(e)}"
            )

            
# Initialize classifier (will be loaded on startup)
classifier = None

# Security Functions
def get_client_ip(request: Request) -> str:
    """Get client IP address"""
    # Check for forwarded headers (for proxies)
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()
    
    real_ip = request.headers.get("X-Real-IP")
    if real_ip:
        return real_ip
    
    return request.client.host

def calculate_file_hash(contents: bytes) -> str:
    """Calculate SHA-256 hash of file contents"""
    return hashlib.sha256(contents).hexdigest()

def validate_image_content(image: Image.Image) -> bool:
    """Validate if image appears to be a medical X-ray"""
    try:
        # Convert to grayscale for analysis
        gray_image = image.convert('L')
        img_array = np.array(gray_image)
        
        # Basic checks for X-ray characteristics
        # X-rays typically have:
        # 1. Reasonable size (not too small/large)
        # 2. Grayscale distribution typical of medical images
        # 3. Not completely black or white
        
        height, width = img_array.shape
        
        # Size validation
        if height < 100 or width < 100 or height > 2000 or width > 2000:
            return False
        
        # Intensity distribution check
        mean_intensity = np.mean(img_array)
        std_intensity = np.std(img_array)
        
        # X-rays should have reasonable contrast
        if std_intensity < 20 or mean_intensity < 30 or mean_intensity > 225:
            return False
        
        return True
    except Exception:
        return False

async def check_rate_limit(request: Request):
    """Check rate limit for the request"""
    client_ip = get_client_ip(request)
    
    if not rate_limiter.is_allowed(client_ip, max_requests=5, window_seconds=60):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Please try again later."
        )

async def check_duplicate_request(file_hash: str):
    """Check for duplicate file uploads"""
    current_time = time.time()
    
    if file_hash in file_hash_cache:
        last_upload_time = file_hash_cache[file_hash]
        if current_time - last_upload_time < CACHE_DURATION:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Duplicate file detected. Please wait before uploading the same image again."
            )
    
    file_hash_cache[file_hash] = current_time


@app.get("/", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if classifier else "unhealthy",
        model_loaded=classifier is not None,
        version="1.0.0"
    )

@app.get("/security/status")
@limiter.limit("10/minute")
async def security_status(request: Request):
    """Get security status and rate limiting info (for educational/development purposes)"""
    client_ip = get_client_ip(request)
    
    # Get current rate limit status
    current_time = time.time()
    requests = rate_limiter.requests[client_ip]
    
    # Clean old requests for accurate count
    while requests and requests[0] < current_time - 60:
        requests.popleft()
    
    return {
        "client_ip": client_ip,
        "requests_in_last_minute": len(requests),
        "rate_limit": "5 requests per minute",
        "is_blocked": client_ip in rate_limiter.blocked_ips,
        "cache_entries": len(file_hash_cache),
        "security_features": [
            "Rate Limiting (5/min per IP)",
            "File Size Validation (10MB max)",
            "File Type Validation (JPG, JPEG, PNG)",
            "Image Content Validation",
            "Duplicate Detection (5min cache)",
            "Request Logging with IP tracking",
            "Educational/Learning Mode (No Auth Required)"
        ]
    }

@app.post("/pneumonia/predict", response_model=PredictionResponse)
@limiter.limit("5/minute")  # SlowAPI rate limiting
async def predict_pneumonia(
    request: Request,
    file: UploadFile = File(...)
):
    """Predict pneumonia from chest X-ray image.

    Args:
        request: FastAPI request object
        file: Chest X-ray image (JPG, JPEG, PNG)
    
    Returns:
        Prediction with confidence, medical recommendation, and metadata
        
    Security Features:
        - Rate limiting: 5 requests per minute per IP
        - File size and type validation
        - Image content validation
        - Duplicate detection
        - Comprehensive logging
        
    Note: This is for educational/experimental purposes only.
    """
    # Security checks
    await check_rate_limit(request)
    
    # Validate classifier is loaded
    if classifier is None:
        logger.error("Model not loaded")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
        
    # Validate file exists
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No file provided"
        )
    
    # Validate file extension
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        )
        
    # Read and validate file size
    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File size exceeds limit of {MAX_FILE_SIZE / (1024 * 1024):.1f} MB"
        )

    # Check for duplicate uploads
    file_hash = calculate_file_hash(contents)
    await check_duplicate_request(file_hash)

    try:
        # Validate image integrity
        image = Image.open(io.BytesIO(contents))
        image.verify()
        image = Image.open(io.BytesIO(contents))  # Reopen after verification
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Validate image content (basic X-ray checks)
        if not validate_image_content(image):
            logger.warning(f"Invalid image content detected: {file.filename}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Image does not appear to be a valid chest X-ray"
            )
            
        # Make prediction
        result = classifier.predict(image)
        
        # Log successful prediction with security info
        client_ip = get_client_ip(request)
        logger.info(
            f"Prediction successful - IP: {client_ip}, "
            f"File: {file.filename}, Hash: {file_hash[:8]}, "
            f"Result: {result['prediction']}, "
            f"Confidence: {result['confidence']:.3f}"
        )
        
        return PredictionResponse(
            **result,
            model_version=getattr(classifier.session.get_modelmeta(), 'custom_metadata_map', {}).get("version", "v1.0")
        )
        
    except (IOError, SyntaxError) as e:
        logger.error(f"Invalid image file: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid image file"
        )
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process image"
        )
    finally:
        await file.close()
        
@app.exception_handler(413)
async def request_entity_too_large_handler(request, exc):
    return JSONResponse(
        status_code=413,
        content={"detail": "File too large"}
    )

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)