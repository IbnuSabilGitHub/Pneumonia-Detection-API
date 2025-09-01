"""
Application configuration and settings.
"""
try:
    from pydantic_settings import BaseSettings
except ImportError:
    # Fallback for older pydantic versions
    from pydantic import BaseSettings
    
from typing import List, Optional


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # Application
    app_name: str = "Pneumonia Detection API"
    app_version: str = "3.3.0"
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
    model_path: str = "models/pneumonia_model_standard.onnx"
    model_stats_path: str = "models/model_stats_standard.json"
    
    model_path_efficientnet_b0: str = "models/pneumonia_model_efficientnet_b0.onnx"
    model_stats_path_efficientnet_b0: str = "models/model_stats_efficientnet_b0.json"
    
    # Rate Limiting
    rate_limit_requests: int = 5
    rate_limit_window: int = 60  # seconds
    rate_limit_block_duration: int = 300  # 5 minutes
    
    # Advanced Rate Limiting
    advanced_rate_limiting_enabled: bool = True
    max_requests_per_ip: int = 10
    max_fingerprint_requests: int = 3
    ip_switching_threshold: int = 3
    global_attack_threshold: float = 0.8
    
    # Storage Backend Configuration
    storage_backend: str = "memory"  # Options: memory, redis, database
    
    # Redis Configuration (optional - for future use if needed)
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_password: Optional[str] = None
    redis_db: int = 0
    redis_url: Optional[str] = None
    redis_max_connections: int = 50
    redis_cluster_mode: bool = False
    redis_cluster_nodes: Optional[str] = None  # JSON string of cluster nodes
    redis_key_prefix: str = "pneumonia_api:rate_limit:"
    
    # In-Memory Storage Configuration (fallback)
    memory_max_size: int = 10000
    memory_cleanup_interval: int = 300
    memory_default_ttl: int = 3600
    
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
    
    def get_redis_config(self) -> dict:
        """Get Redis configuration dictionary."""
        import json
        
        cluster_nodes = None
        if self.redis_cluster_nodes:
            try:
                cluster_nodes = json.loads(self.redis_cluster_nodes)
            except (json.JSONDecodeError, TypeError):
                cluster_nodes = None
        
        return {
            "host": self.redis_host,
            "port": self.redis_port,
            "password": self.redis_password,
            "db": self.redis_db,
            "max_connections": self.redis_max_connections,
            "cluster_mode": self.redis_cluster_mode,
            "cluster_nodes": cluster_nodes,
            "key_prefix": self.redis_key_prefix
        }
    
    def get_storage_config(self) -> dict:
        """Get storage backend configuration."""
        if self.storage_backend == "redis":
            return self.get_redis_config()
        elif self.storage_backend == "memory":
            return {
                "max_size": self.memory_max_size,
                "cleanup_interval": self.memory_cleanup_interval,
                "default_ttl": self.memory_default_ttl
            }
        else:
            return {}


class ModelConfig:
    """Model-specific configuration."""
    
    # Image preprocessing
    TARGET_SIZE = (192, 192)
    TARGET_SIZE_B0 = (224, 224)
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
