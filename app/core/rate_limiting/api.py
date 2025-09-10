import logging
from app.core.storage_factory import StorageType
from typing import Dict, Any, Optional
from .core import AdvancedRateLimiter

logger = logging.getLogger(__name__)

# Global instance with Redis storage capability
async def create_advanced_rate_limiter(
    storage_type: StorageType = StorageType.REDIS,
    storage_config: Optional[Dict[str, Any]] = None
) -> AdvancedRateLimiter:
    """Create and initialize advanced rate limiter with storage backend."""
    limiter = AdvancedRateLimiter(storage_config=storage_config)
    
    # Initialize storage
    success = await limiter.initialize_storage(storage_type, storage_config)
    if not success:
        logger.warning("Failed to initialize storage backend, using in-memory fallback")
    
    return limiter