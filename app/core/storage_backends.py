"""
Storage backend interface for advanced rate limiting.
Supports multiple backends: In-Memory, Redis, Database.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class RateLimitRecord:
    """Rate limit record structure."""

    key: str
    value: Any
    timestamp: float
    expires_at: Optional[float] = None


class StorageBackend(ABC):
    """Abstract storage backend for rate limiting data."""

    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        """Get value by key."""
        pass

    @abstractmethod
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value with optional TTL."""
        pass

    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete key."""
        pass

    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Check if key exists."""
        pass

    @abstractmethod
    async def increment(
        self, key: str, amount: int = 1, ttl: Optional[int] = None
    ) -> int:
        """Increment counter."""
        pass

    @abstractmethod
    async def get_list(self, key: str) -> List[Any]:
        """Get list values."""
        pass

    @abstractmethod
    async def append_to_list(
        self, key: str, value: Any, max_length: Optional[int] = None
    ) -> bool:
        """Append to list with optional max length."""
        pass

    @abstractmethod
    async def get_hash(self, key: str) -> Dict[str, Any]:
        """Get hash/dict values."""
        pass

    @abstractmethod
    async def set_hash_field(self, key: str, field: str, value: Any) -> bool:
        """Set hash field."""
        pass

    @abstractmethod
    async def cleanup_expired(self) -> int:
        """Clean up expired entries."""
        pass

    @abstractmethod
    async def ping(self) -> bool:
        """Test connection."""
        pass

    @abstractmethod
    async def get_info(self) -> Dict[str, Any]:
        """Get backend information."""
        pass
