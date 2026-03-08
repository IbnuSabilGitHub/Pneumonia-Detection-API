"""
Storage factory for creating different storage backends.
Supports In-Memory and Database backends.
"""

import logging
from enum import Enum
from typing import Any, Dict, Optional

from .memory_storage import InMemoryStorageBackend
from .storage_backends import StorageBackend

logger = logging.getLogger(__name__)


class StorageType(Enum):
    """Supported storage backend types."""

    MEMORY = "memory"
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

        elif storage_type == StorageType.DATABASE:
            return {
                "database_url": "sqlite:///rate_limiting.db",
                "echo": False,
                "pool_size": 20,
                "max_overflow": 30,
            }

        else:
            return {}
