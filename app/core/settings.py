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
    app_version: str = "3.4.2"
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

    # Advanced Rate Limiting - Main Settings
    advanced_rate_limiting_enabled: bool = False

    # Rate limiting path to be excluded
    excluded_paths: List[str] = ["/health", "/", "/docs", "/redoc", "/openapi.json"]

    # Basic Rate Limiting
    max_requests_per_ip: int = 50
    max_fingerprint_requests: int = 10
    rate_limit_window_size: int = 300  # seconds

    # Attack Detection Thresholds
    ip_switching_threshold: int = 10  # Same fingerprint from N+ IPs
    suspicious_ip_changes_threshold: int = 20  # Distributed attack threshold
    global_attack_threshold: float = 0.9  # Attack score threshold
    bot_behavior_variance: float = 0.3  # Request timing variance for bot detection

    # Block Durations
    attack_block_duration: int = 60  # 5 minutes in seconds
    fingerprint_block_duration: int = 120  # 5 minutes in seconds

    # Attack Detection Windows
    ip_switching_detection_window: int = 300  # 5 minutes
    behavioral_analysis_window: int = 300  # 5 minutes
    global_attack_score_window: int = 60  # 1 minute

    # Attack Detection Limits
    coordinated_attack_threshold: int = 8  # Same file from N+ IPs
    bot_timing_threshold: float = 30.0  # Seconds - avg interval for bot detection

    # In-Memory Limits (for fallback mode)
    max_recent_ips: int = 1000
    max_request_patterns_per_ip: int = 10
    max_file_hash_requests: int = 100
    max_global_request_rate: int = 1000
    max_fingerprints_per_ip: int = 100

    # Rate Limiting Reduction Factor
    attack_reduction_factor: float = 0.5  # Reduce limits by 50% during attacks

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
    file_hash_cache_max_size: int = (
        5000  # Maximum number of unique file hashes to retain
    )

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
            "key_prefix": self.redis_key_prefix,
        }

    def get_storage_config(self) -> dict:
        """Get storage backend configuration."""
        if self.storage_backend == "redis":
            return self.get_redis_config()
        elif self.storage_backend == "memory":
            return {
                "max_size": self.memory_max_size,
                "cleanup_interval": self.memory_cleanup_interval,
                "default_ttl": self.memory_default_ttl,
            }
        else:
            return {}

    def get_rate_limiting_config(self) -> dict:
        """Get comprehensive rate limiting configuration."""
        return {
            # Basic Rate Limiting
            "enabled": self.advanced_rate_limiting_enabled,
            "excluded_paths": self.excluded_paths,
            "max_requests_per_ip": self.max_requests_per_ip,
            "max_fingerprint_requests": self.max_fingerprint_requests,
            "window_size": self.rate_limit_window_size,
            # Attack Detection Thresholds
            "ip_switching_threshold": self.ip_switching_threshold,
            "suspicious_ip_changes_threshold": self.suspicious_ip_changes_threshold,
            "global_attack_threshold": self.global_attack_threshold,
            "bot_behavior_variance": self.bot_behavior_variance,
            # Block Durations
            "attack_block_duration": self.attack_block_duration,
            "fingerprint_block_duration": self.fingerprint_block_duration,
            # Detection Windows
            "ip_switching_detection_window": self.ip_switching_detection_window,
            "behavioral_analysis_window": self.behavioral_analysis_window,
            "global_attack_score_window": self.global_attack_score_window,
            # Attack Limits
            "coordinated_attack_threshold": self.coordinated_attack_threshold,
            "bot_timing_threshold": self.bot_timing_threshold,
            # In-Memory Limits
            "max_recent_ips": self.max_recent_ips,
            "max_request_patterns_per_ip": self.max_request_patterns_per_ip,
            "max_file_hash_requests": self.max_file_hash_requests,
            "max_global_request_rate": self.max_global_request_rate,
            "max_fingerprints_per_ip": self.max_fingerprints_per_ip,
            # Reduction Factor
            "attack_reduction_factor": self.attack_reduction_factor,
        }


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
    LABEL_MAP = {0: "NORMAL", 1: "PNEUMONIA"}

    # Confidence thresholds
    HIGH_CONFIDENCE_THRESHOLD = 0.8
    MEDIUM_CONFIDENCE_THRESHOLD = 0.6


# Global settings instance
settings = Settings()
model_config = ModelConfig()
