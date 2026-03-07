"""
Storage factory for creating different storage backends.
Supports In-Memory, Redis (optional), and Database backends.
"""

import logging
from enum import Enum
from typing import Any, Dict, Optional

from .memory_storage import InMemoryStorageBackend
from .storage_backends import StorageBackend

logger = logging.getLogger(__name__)

# Feature detection: Check if Redis is available
REDIS_AVAILABLE = False
try:
    from .redis_storage import RedisStorageBackend

    REDIS_AVAILABLE = True
    logger.debug("Redis storage backend available")
except ImportError:
    logger.debug("Redis storage backend not available (redis library not installed)")
    RedisStorageBackend = None  # type: ignore


class StorageType(Enum):
    """Supported storage backend types."""

    MEMORY = "memory"
    REDIS = "redis"
    DATABASE = "database"


class StorageFactory:
    """Factory for creating storage backends."""

    @staticmethod
    async def create_storage(
        storage_type: StorageType, config: Optional[Dict[str, Any]] = None
    ) -> StorageBackend:
        """Create storage backend instance."""
        config = config or {}

        if storage_type == StorageType.MEMORY:
            return InMemoryStorageBackend(
                max_size=config.get("max_size", 10000),
                cleanup_interval=config.get("cleanup_interval", 300),
                default_ttl=config.get("default_ttl", 3600),
            )

        elif storage_type == StorageType.REDIS:
            if not REDIS_AVAILABLE:
                logger.error("Redis storage requested but redis library not installed")
                logger.info("Install with: pip install redis")
                logger.warning("Falling back to in-memory storage")
                return InMemoryStorageBackend()

            # Redis is available, create backend
            redis_backend = RedisStorageBackend(
                host=config.get("host", "localhost"),
                port=config.get("port", 6379),
                password=config.get("password"),
                db=config.get("db", 0),
                max_connections=config.get("max_connections", 50),
                cluster_mode=config.get("cluster_mode", False),
                cluster_nodes=config.get("cluster_nodes"),
                key_prefix=config.get("key_prefix", "rate_limit:"),
            )

            # Connect to Redis
            connected = await redis_backend.connect()
            if not connected:
                logger.error("Failed to connect to Redis server")
                logger.warning("Falling back to in-memory storage")
                return InMemoryStorageBackend()

            logger.info("Successfully connected to Redis storage backend")
            return redis_backend

        elif storage_type == StorageType.DATABASE:
            # Database backend implementation would go here
            # For now, fall back to in-memory
            logger.warning("Database backend not implemented, using in-memory storage")
            return InMemoryStorageBackend()

        else:
            raise ValueError(f"Unsupported storage type: {storage_type}")

    @staticmethod
    def get_storage_config_template(storage_type: StorageType) -> Dict[str, Any]:
        """Get configuration template for storage type."""
        if storage_type == StorageType.MEMORY:
            return {"max_size": 10000, "cleanup_interval": 300, "default_ttl": 3600}

        elif storage_type == StorageType.REDIS:
            return {
                "host": "localhost",
                "port": 6379,
                "password": None,
                "db": 0,
                "max_connections": 50,
                "cluster_mode": False,
                "cluster_nodes": None,
                "key_prefix": "rate_limit:",
            }

        elif storage_type == StorageType.DATABASE:
            return {
                "database_url": "sqlite:///rate_limiting.db",
                "echo": False,
                "pool_size": 20,
                "max_overflow": 30,
            }

        else:
            return {}
