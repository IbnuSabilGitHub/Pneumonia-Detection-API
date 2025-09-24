import logging
from typing import Dict, Optional, Tuple

from app.core.storage_backends import StorageBackend

logger = logging.getLogger(__name__)


class RateLimitManager:
    """
    Handles basic IP and fingerprint-based rate limiting operations.
    """

    def __init__(
        self,
        storage: Optional[StorageBackend] = None,
        max_requests_per_ip: int = 10,
        max_fingerprint_requests: int = 3,
        window_size: int = 60,
    ):
        self.storage = storage

        # Configuration
        self.max_requests_per_ip = max_requests_per_ip
        self.max_fingerprint_requests = max_fingerprint_requests
        self.window_size = window_size  # Time window in seconds

    async def _get_from_storage(self, key: str, default=None):
        """Get value from storage with fallback."""
        if self.storage:
            try:
                return await self.storage.get(key) or default
            except Exception as e:
                logger.error("Storage GET error for %s: %s", key, e)
        return default

    async def _increment_in_storage(
        self, key: str, amount: int = 1, ttl: Optional[int] = None
    ) -> int:
        """Increment counter in storage."""
        if self.storage:
            try:
                return await self.storage.increment(key, amount, ttl)
            except Exception as e:
                logger.error("Storage INCREMENT error for %s: %s", key, e)
        return 0

    async def check_ip_rate_limit(self, client_ip: str) -> Tuple[bool, int]:
        """
        Check if IP has exceeded rate limit.
        Returns: (is_allowed, current_request_count)
        """
        ip_requests_key = f"ip_requests:{client_ip}"

        # Increment request count with TTL
        await self._increment_in_storage(ip_requests_key, 1, self.window_size)

        # Get current count
        current_count = await self._get_from_storage(ip_requests_key, 0)

        is_allowed = current_count <= self.max_requests_per_ip

        return is_allowed, current_count

    async def check_fingerprint_rate_limit(self, fingerprint: str) -> Tuple[bool, int]:
        """
        Check if fingerprint has exceeded rate limit.
        Returns: (is_allowed, current_request_count)
        """
        fingerprint_requests_key = f"fingerprint_requests:{fingerprint}"

        # Increment request count with TTL
        await self._increment_in_storage(fingerprint_requests_key, 1, self.window_size)

        # Get current count
        current_count = await self._get_from_storage(fingerprint_requests_key, 0)

        is_allowed = current_count <= self.max_fingerprint_requests

        return is_allowed, current_count

    async def get_ip_request_count(self, client_ip: str) -> int:
        """Get current request count for an IP without incrementing."""
        ip_requests_key = f"ip_requests:{client_ip}"
        return await self._get_from_storage(ip_requests_key, 0)

    async def get_fingerprint_request_count(self, fingerprint: str) -> int:
        """Get current request count for a fingerprint without incrementing."""
        fingerprint_requests_key = f"fingerprint_requests:{fingerprint}"
        return await self._get_from_storage(fingerprint_requests_key, 0)

    async def reset_ip_rate_limit(self, client_ip: str) -> bool:
        """Reset rate limit counter for an IP."""
        ip_requests_key = f"ip_requests:{client_ip}"
        if self.storage:
            try:
                await self.storage.delete(ip_requests_key)
                return True
            except Exception as e:
                logger.error("Failed to reset IP rate limit for %s: %s", client_ip, e)
        return False

    async def reset_fingerprint_rate_limit(self, fingerprint: str) -> bool:
        """Reset rate limit counter for a fingerprint."""
        fingerprint_requests_key = f"fingerprint_requests:{fingerprint}"
        if self.storage:
            try:
                await self.storage.delete(fingerprint_requests_key)
                return True
            except Exception as e:
                logger.error(
                    "Failed to reset fingerprint rate limit for %s: %s", fingerprint, e
                )
        return False

    async def apply_reduced_limits(
        self, client_ip: str, fingerprint: str, reduction_factor: float = 0.5
    ) -> Tuple[bool, Dict]:
        """
        Apply reduced rate limits during high attack periods.
        Returns: (is_allowed, details)
        """
        reduced_ip_limit = int(self.max_requests_per_ip * reduction_factor)
        reduced_fingerprint_limit = int(
            self.max_fingerprint_requests * reduction_factor
        )

        # Get current counts without incrementing
        ip_count = await self.get_ip_request_count(client_ip)
        fingerprint_count = await self.get_fingerprint_request_count(fingerprint)

        ip_allowed = ip_count <= reduced_ip_limit
        fingerprint_allowed = fingerprint_count <= reduced_fingerprint_limit

        is_allowed = ip_allowed and fingerprint_allowed

        details = {
            "reduced_limits_applied": True,
            "original_ip_limit": self.max_requests_per_ip,
            "reduced_ip_limit": reduced_ip_limit,
            "ip_requests": ip_count,
            "ip_allowed": ip_allowed,
            "original_fingerprint_limit": self.max_fingerprint_requests,
            "reduced_fingerprint_limit": reduced_fingerprint_limit,
            "fingerprint_requests": fingerprint_count,
            "fingerprint_allowed": fingerprint_allowed,
        }

        return is_allowed, details

    async def get_rate_limit_info(self, client_ip: str, fingerprint: str) -> Dict:
        """Get comprehensive rate limit information."""
        ip_count = await self.get_ip_request_count(client_ip)
        fingerprint_count = await self.get_fingerprint_request_count(fingerprint)

        return {
            "ip": client_ip,
            "ip_requests": ip_count,
            "ip_limit": self.max_requests_per_ip,
            "ip_remaining": max(0, self.max_requests_per_ip - ip_count),
            "fingerprint": fingerprint,
            "fingerprint_requests": fingerprint_count,
            "fingerprint_limit": self.max_fingerprint_requests,
            "fingerprint_remaining": max(
                0, self.max_fingerprint_requests - fingerprint_count
            ),
            "window_size": self.window_size,
        }

    def update_limits(
        self,
        max_requests_per_ip: Optional[int] = None,
        max_fingerprint_requests: Optional[int] = None,
        window_size: Optional[int] = None,
    ):
        """Update rate limiting configuration."""
        if max_requests_per_ip is not None:
            self.max_requests_per_ip = max_requests_per_ip
            logger.info("Updated IP rate limit to %d", max_requests_per_ip)

        if max_fingerprint_requests is not None:
            self.max_fingerprint_requests = max_fingerprint_requests
            logger.info(
                "Updated fingerprint rate limit to %d", max_fingerprint_requests
            )

        if window_size is not None:
            self.window_size = window_size
            logger.info("Updated window size to %d seconds", window_size)

    def get_configuration(self) -> Dict:
        """Get current rate limiting configuration."""
        return {
            "max_requests_per_ip": self.max_requests_per_ip,
            "max_fingerprint_requests": self.max_fingerprint_requests,
            "window_size": self.window_size,
            "storage_backend": "active" if self.storage else "none",
        }

    async def get_metrics(self) -> Dict:
        """Get rate limiting metrics."""
        if self.storage:
            storage_info = await self.storage.get_info()
            return {
                "storage_healthy": storage_info.get("is_healthy", False),
                "storage_type": storage_info.get("backend_type", "unknown"),
                "configuration": self.get_configuration(),
            }
        else:
            return {
                "storage_healthy": False,
                "storage_type": "none",
                "configuration": self.get_configuration(),
            }
