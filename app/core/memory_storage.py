"""
In-memory storage backend implementation for advanced rate limiting.
Development-friendly with automatic cleanup and memory management.
"""

import logging
import threading
import time
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional

from .storage_backends import StorageBackend

logger = logging.getLogger(__name__)


class InMemoryStorageBackend(StorageBackend):
    """In-memory storage backend with TTL and cleanup."""

    def __init__(
        self,
        max_size: int = 10000,
        cleanup_interval: int = 300,
        default_ttl: int = 3600,
    ):
        """Initialize in-memory storage."""
        self.max_size = max_size
        self.cleanup_interval = cleanup_interval
        self.default_ttl = default_ttl

        # Storage containers
        self._data: Dict[str, Any] = {}
        self._ttl: Dict[str, float] = {}
        self._lists: Dict[str, deque] = defaultdict(deque)
        self._hashes: Dict[str, Dict[str, Any]] = defaultdict(dict)

        # Cleanup management
        self._last_cleanup = time.time()
        self._cleanup_lock = threading.Lock()

        # Statistics
        self._stats = {
            "gets": 0,
            "sets": 0,
            "deletes": 0,
            "hits": 0,
            "misses": 0,
            "expired_keys": 0,
        }

    def _is_expired(self, key: str) -> bool:
        """Check if key is expired."""
        if key not in self._ttl:
            return False
        return time.time() > self._ttl[key]

    def _cleanup_expired_keys(self) -> int:
        """Clean up expired keys."""
        current_time = time.time()
        expired_count = 0

        # Clean up main data
        expired_keys = [
            key for key, expire_time in self._ttl.items() if current_time > expire_time
        ]

        for key in expired_keys:
            if key in self._data:
                del self._data[key]
            if key in self._ttl:
                del self._ttl[key]
            if key in self._lists:
                del self._lists[key]
            if key in self._hashes:
                del self._hashes[key]
            expired_count += 1

        self._stats["expired_keys"] += expired_count
        return expired_count

    def _maybe_cleanup(self):
        """Cleanup if needed."""
        current_time = time.time()
        if current_time - self._last_cleanup > self.cleanup_interval:
            with self._cleanup_lock:
                if current_time - self._last_cleanup > self.cleanup_interval:
                    self._cleanup_expired_keys()
                    self._last_cleanup = current_time

    def _set_ttl(self, key: str, ttl: Optional[int]):
        """Set TTL for key."""
        if ttl:
            self._ttl[key] = time.time() + ttl
        elif key in self._ttl:
            del self._ttl[key]

    async def get(self, key: str) -> Optional[Any]:
        """Get value by key."""
        self._maybe_cleanup()
        self._stats["gets"] += 1

        if self._is_expired(key):
            if key in self._data:
                del self._data[key]
            if key in self._ttl:
                del self._ttl[key]
            self._stats["misses"] += 1
            return None

        if key in self._data:
            self._stats["hits"] += 1
            return self._data[key]

        self._stats["misses"] += 1
        return None

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value with optional TTL."""
        self._maybe_cleanup()
        self._stats["sets"] += 1

        # Check memory limit
        if len(self._data) >= self.max_size and key not in self._data:
            # Remove oldest entry
            if self._data:
                oldest_key = next(iter(self._data))
                await self.delete(oldest_key)

        self._data[key] = value
        self._set_ttl(key, ttl)
        return True

    async def delete(self, key: str) -> bool:
        """Delete key."""
        self._stats["deletes"] += 1
        deleted = False

        if key in self._data:
            del self._data[key]
            deleted = True

        if key in self._ttl:
            del self._ttl[key]

        if key in self._lists:
            del self._lists[key]
            deleted = True

        if key in self._hashes:
            del self._hashes[key]
            deleted = True

        return deleted

    async def exists(self, key: str) -> bool:
        """Check if key exists."""
        self._maybe_cleanup()

        if self._is_expired(key):
            await self.delete(key)
            return False

        return key in self._data or key in self._lists or key in self._hashes

    async def increment(
        self, key: str, amount: int = 1, ttl: Optional[int] = None
    ) -> int:
        """Increment counter."""
        self._maybe_cleanup()

        if self._is_expired(key):
            await self.delete(key)

        current_value = self._data.get(key, 0)
        if not isinstance(current_value, (int, float)):
            current_value = 0

        new_value = current_value + amount
        await self.set(key, new_value, ttl)
        return int(new_value)

    async def get_list(self, key: str) -> List[Any]:
        """Get list values."""
        self._maybe_cleanup()

        if self._is_expired(key):
            await self.delete(key)
            return []

        if key in self._lists:
            return list(self._lists[key])

        return []

    async def append_to_list(
        self, key: str, value: Any, max_length: Optional[int] = None
    ) -> bool:
        """Append to list with optional max length."""
        self._maybe_cleanup()

        if self._is_expired(key):
            await self.delete(key)

        self._lists[key].appendleft(value)

        if max_length and len(self._lists[key]) > max_length:
            # Remove oldest items
            while len(self._lists[key]) > max_length:
                self._lists[key].pop()

        return True

    async def get_hash(self, key: str) -> Dict[str, Any]:
        """Get hash/dict values."""
        self._maybe_cleanup()

        if self._is_expired(key):
            await self.delete(key)
            return {}

        return dict(self._hashes.get(key, {}))

    async def set_hash_field(self, key: str, field: str, value: Any) -> bool:
        """Set hash field."""
        self._maybe_cleanup()

        if self._is_expired(key):
            await self.delete(key)

        self._hashes[key][field] = value
        return True

    async def cleanup_expired(self) -> int:
        """Clean up expired entries."""
        return self._cleanup_expired_keys()

    async def ping(self) -> bool:
        """Test connection (always healthy for in-memory)."""
        return True

    async def get_info(self) -> Dict[str, Any]:
        """Get backend information."""
        self._maybe_cleanup()

        total_keys = len(self._data) + len(self._lists) + len(self._hashes)
        memory_usage = (
            sum([len(str(v)) for v in self._data.values()])
            + sum(
                [
                    sum(len(str(item)) for item in items)
                    for items in self._lists.values()
                ]
            )
            + sum(
                [
                    sum(len(str(k)) + len(str(v)) for k, v in hash_data.items())
                    for hash_data in self._hashes.values()
                ]
            )
        )

        return {
            "backend_type": "memory",
            "total_keys": total_keys,
            "data_keys": len(self._data),
            "list_keys": len(self._lists),
            "hash_keys": len(self._hashes),
            "keys_with_ttl": len(self._ttl),
            "estimated_memory_bytes": memory_usage,
            "max_size": self.max_size,
            "cleanup_interval": self.cleanup_interval,
            "last_cleanup": self._last_cleanup,
            "statistics": self._stats.copy(),
            "is_healthy": True,
        }
