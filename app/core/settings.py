"""
Application configuration and settings.
"""
try:
    from pydantic_settings import BaseSettings
except ImportError:
    # Fallback for older pydantic versions
    from pydantic import BaseSettings
    
from typing import List, Optional
from pathlib import Path
import os


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # Application
    app_name: str = "Pneumonia Detection API"
    app_version: str = "2.0.0"
    debug: bool = False
    
    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    
    # Security
    trusted_hosts: List[str] = ["*.railway.app", "localhost", "127.0.0.1"]
    cors_origins: List[str] = ["*"]  # Configure based on your frontend
    
    # File Upload
    max_file_size: int = 10 * 1024 * 1024  # 10 MB
    allowed_extensions: List[str] = [".jpg", ".jpeg", ".png"]
    
    # Model
    model_path: str = "models/pneumonia_model.onnx"
    model_stats_path: str = "models/model_stats.json"
    
    # Rate Limiting
    rate_limit_requests: int = 5
    rate_limit_window: int = 60  # seconds
    rate_limit_block_duration: int = 300  # 5 minutes
    
    # Cache
    cache_duration: int = 300  # 5 minutes
    
    # Logging
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Redis (optional)
    redis_url: Optional[str] = None
    
    # Railway specific
    railway_environment: Optional[str] = None
    
    # Legacy/backward compatibility
    allowed_origins: Optional[str] = None
    
    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "ignore"  # Ignore extra fields for flexibility


class ModelConfig:
    """Model-specific configuration."""
    
    # Image preprocessing
    TARGET_SIZE = (192, 192)
    CHANNELS = 1  # Grayscale
    
    # Normalization (fallback values)
    DEFAULT_MEAN = 0.449
    DEFAULT_STD = 0.226
    
    # Labels
    LABEL_MAP = {0: 'NORMAL', 1: 'PNEUMONIA'}
    
    # Confidence thresholds
    HIGH_CONFIDENCE_THRESHOLD = 0.8
    MEDIUM_CONFIDENCE_THRESHOLD = 0.6


# Global settings instance
settings = Settings()
model_config = ModelConfig()
