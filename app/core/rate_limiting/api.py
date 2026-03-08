import logging
from typing import Any, Dict, Optional

from ..settings import Settings
from ..storage_factory import StorageType

from .core import AdvancedRateLimiter

logger = logging.getLogger(__name__)


# Global instance with in-memory storage
async def create_advanced_rate_limiter(
    storage_type: StorageType = StorageType.MEMORY,
    storage_config: Optional[Dict[str, Any]] = None,
) -> AdvancedRateLimiter:
    """Create and initialize advanced rate limiter with storage backend."""
    limiter = AdvancedRateLimiter(storage_config=storage_config, settings=Settings())

    # Initialize storage
    success = await limiter.initialize_storage(storage_type, storage_config)
    if not success:
        logger.warning("Failed to initialize storage backend, using in-memory fallback")

    return limiter
