"""
Application configuration and settings.
"""

import logging
import os
from typing import List, Optional, Union

from ..version import CURRENT_API_VERSION

logger = logging.getLogger(__name__)

try:
    from pydantic_settings import BaseSettings
except ImportError:
    # Fallback for older pydantic versions
    from pydantic import BaseSettings


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    # Application
    app_name: str = "Pneumonia Detection API"
    app_version: str = CURRENT_API_VERSION
    debug: bool = False

    # Server
    host: str = "0.0.0.0"
    port: int = 8000

    # Security - Allow both string and list for Docker env vars
    trusted_hosts: Union[List[str], str] = ["*.onrender.com", "localhost", "127.0.0.1"]
    cors_origins: Union[List[str], str] = ["*"]  # Configure based on your frontend
    
    # Admin API Security
    admin_api_key: Optional[str] = None  # Set via ADMIN_API_KEY env var for /stats and /status endpoints
    enable_public_stats: bool = False  # Set to True to allow unauthenticated access to /stats
    enable_public_status: bool = False  # Set to True to allow unauthenticated access to /status

    # Supabase JWT Authentication (ES256 only) - Always Enabled
    jwt_auth_enabled: bool = True  # JWT authentication always active (native)
    supabase_url: Optional[str] = None  # Supabase project URL (e.g. https://<project>.supabase.co) - REQUIRED for JWKS endpoint
    supabase_anon_key: Optional[str] = None  # Supabase anon key (optional, for client-side reference)
    supabase_jwt_verify_audience: bool = True  # Verify 'aud' claim in JWT (default: authenticated)
    
    # User-based Rate Limiting (JWT identity-based)
    user_rate_limiting_enabled: bool = True  # Enable per-user rate limiting
    user_rate_limit_max_requests: int = 100  # Max requests per user per window
    user_rate_limit_window_size: int = 3600  # Window size in seconds (1 hour default)
    user_rate_limit_use_supabase: bool = True  # Use Supabase for storage (falls back to memory)

    # File Upload
    max_file_size: int = 10 * 1024 * 1024  # 10 MB
    allowed_extensions: Union[List[str], str] = [".jpg", ".jpeg", ".png"]

    # Model
    model_path: str = "models/pneumonia_model_standard.onnx"
    model_stats_path: str = "models/model_stats_standard.json"

    model_path_efficientnet_b0: str = "models/pneumonia_model_efficientnet_b0.onnx"
    model_stats_path_efficientnet_b0: str = "models/model_stats_efficientnet_b0.json"

    # Advanced Rate Limiting - Main Settings

    # Rate limiting path to be excluded
    excluded_paths: Union[List[str], str] = [
        "/health",
        "/",
        "/docs",
        "/redoc",
        "/openapi.json",
    ]

    # Prediction Concurrency Control
    prediction_concurrency_limit: int = 3  # Max concurrent predictions

    # Storage Backend Configuration
    storage_backend: str = "memory"  # Options: memory, database

    # In-Memory Storage Configuration (optimized for free tier)
    memory_max_size: int = 1000  # Reduced for memory efficiency
    memory_cleanup_interval: int = 180  # More frequent cleanup (3 minutes)
    memory_default_ttl: int = 1800  # 30 minutes TTL

    # Cache (optimized for free tier)
    cache_duration: int = 300  # 5 minutes
    file_hash_cache_max_size: int = 200  # Reduced cache size for memory efficiency

    # Logging
    log_level: str = "INFO"
    log_enabled: bool = True
    log_format: Optional[str] = None
    log_include_timestamp: bool = True
    log_include_level: bool = True
    log_include_logger_name: bool = True
    log_include_module: bool = False
    log_include_process: bool = False
    log_include_thread: bool = False
    log_include_filename: bool = False
    log_include_line_number: bool = False
    log_field_separator: str = " - "
    log_format_with_timestamp: str = (
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    log_format_without_timestamp: str = "%(name)s - %(levelname)s - %(message)s"
    log_unify_uvicorn: bool = False

    # Railway specific
    railway_environment: Optional[str] = None

    # Legacy/backward compatibility
    allowed_origins: Optional[str] = None

    def _parse_string_to_list(self, value: str, default: List[str]) -> List[str]:
        """Parse comma-separated string into list."""
        if isinstance(value, str):
            return [item.strip() for item in value.split(',') if item.strip()]
        return default

    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "ignore"  # Ignore extra fields for flexibility

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Manual parsing for list fields from environment variables
        if isinstance(self.trusted_hosts, str):
            self.trusted_hosts = self._parse_string_to_list(
                self.trusted_hosts, ["*.onrender.com", "localhost", "127.0.0.1"]
            )

        if isinstance(self.cors_origins, str):
            self.cors_origins = self._parse_string_to_list(self.cors_origins, ["*"])

        if isinstance(self.excluded_paths, str):
            self.excluded_paths = self._parse_string_to_list(
                self.excluded_paths,
                ["/health", "/", "/docs", "/redoc", "/openapi.json"],
            )

        if isinstance(self.allowed_extensions, str):
            self.allowed_extensions = self._parse_string_to_list(
                self.allowed_extensions, [".jpg", ".jpeg", ".png"]
            )

        if not self.log_field_separator:
            self.log_field_separator = " "

        if not self.log_format:
            self.log_format = self._build_log_format()

        # Debug logging untuk melihat dari mana values dibaca
        logger.info("🔧 Configuration loaded successfully")
        logger.info(f"Trusted hosts: {self.trusted_hosts}")
        logger.info(f"Storage backend: {self.storage_backend}")
        logger.info(f"JWT auth enabled: {self.jwt_auth_enabled}")
        if self.jwt_auth_enabled:
            logger.info(f"Supabase URL: {self.supabase_url or 'NOT SET - REQUIRED'}")
            logger.info(f"JWT algorithm: ES256 (Supabase standard)")
            logger.info(f"Verify audience: {self.supabase_jwt_verify_audience}")
            logger.info(f"User rate limiting: {self.user_rate_limiting_enabled}")
            logger.info(f"Max requests per user: {self.user_rate_limit_max_requests}/{self.user_rate_limit_window_size}s")
            logger.info(f"Use Supabase storage: {self.user_rate_limit_use_supabase}")
        logger.info(f"Logging enabled: {self.log_enabled}")
        logger.info(
            "Log inclusions — timestamp: %s, level: %s, name: %s, module: %s, process: %s, thread: %s, filename: %s, line: %s",
            self.log_include_timestamp,
            self.log_include_level,
            self.log_include_logger_name,
            self.log_include_module,
            self.log_include_process,
            self.log_include_thread,
            self.log_include_filename,
            self.log_include_line_number,
        )
        logger.info(f"Log separator: '{self.log_field_separator}'")
        logger.info(f"Resolved log format: {self.log_format}")
        logger.info(f"Unify uvicorn logging: {self.log_unify_uvicorn}")

    def _has_advanced_log_options(self) -> bool:
        """Determine if advanced logging options are enabled."""
        return any(
            [
                not self.log_include_level,
                not self.log_include_logger_name,
                self.log_include_module,
                self.log_include_process,
                self.log_include_thread,
                self.log_include_filename,
                self.log_include_line_number,
                self.log_field_separator != " - ",
            ]
        )

    def _build_log_format(self) -> str:
        """Build log format string based on configuration toggles."""
        if not self._has_advanced_log_options():
            return (
                self.log_format_with_timestamp
                if self.log_include_timestamp
                else self.log_format_without_timestamp
            )

        parts: List[str] = []

        if self.log_include_timestamp:
            parts.append("%(asctime)s")

        if self.log_include_level:
            parts.append("%(levelname)s")

        if self.log_include_logger_name:
            parts.append("%(name)s")

        if self.log_include_module:
            parts.append("%(module)s")

        if self.log_include_process:
            parts.append("pid=%(process)d")

        if self.log_include_thread:
            parts.append("thread=%(threadName)s")

        if self.log_include_filename and self.log_include_line_number:
            parts.append("%(filename)s:%(lineno)d")
        elif self.log_include_filename:
            parts.append("%(filename)s")
        elif self.log_include_line_number:
            parts.append("line=%(lineno)d")

        parts.append("%(message)s")

        separator = self.log_field_separator or " "
        return separator.join(parts)

    def get_storage_config(self) -> dict:
        """Get storage backend configuration."""
        if self.storage_backend == "memory":
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
            "excluded_paths": self.excluded_paths,
            "max_fingerprint_requests": self.max_fingerprint_requests,
            "window_size": self.rate_limit_window_size,
            # Attack Detection Thresholds
            "ip_switching_threshold": self.ip_switching_threshold,
            "suspicious_ip_changes_threshold": self.suspicious_ip_changes_threshold,
            "global_attack_threshold": self.global_attack_threshold,
            "bot_behavior_variance": self.bot_behavior_variance,
            # Block Durations
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

    def get_user_rate_limiting_config(self) -> dict:
        """Get user-based rate limiting configuration for JWT auth."""
        return {
            "enabled": self.user_rate_limiting_enabled,
            "max_requests": self.user_rate_limit_max_requests,
            "window_size": self.user_rate_limit_window_size,
            "use_supabase": self.user_rate_limit_use_supabase,
            "supabase_url": self.supabase_url,
            "supabase_key": self.supabase_anon_key,
            "excluded_paths": self.excluded_paths,
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
