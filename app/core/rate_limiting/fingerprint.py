import hashlib
import logging
import time
from typing import Dict, List, Optional

from app.core.rate_limiting.storage import RequestFingerprint
from app.core.storage_backends import StorageBackend

logger = logging.getLogger(__name__)


class FingerprintManager:
    """
    Handles request fingerprinting and fingerprint-based tracking.
    """

    def __init__(
        self, storage: Optional[StorageBackend] = None, config: Optional[Dict] = None
    ):
        self.storage = storage
        self.config = config or {}

        # Fallback in-memory storage
        self.ip_fingerprints: Dict[str, List[RequestFingerprint]] = {}
        self.blocked_fingerprints: Dict[str, float] = {}

        # Configuration
        self.block_duration = self.config.get("fingerprint_block_duration", 300)
        self.max_fingerprints_per_ip = self.config.get("max_fingerprints_per_ip", 100)

    async def _get_from_storage(self, key: str, default=None):
        """Get value from storage with fallback."""
        if self.storage:
            try:
                return await self.storage.get(key) or default
            except Exception as e:
                logger.error("Storage GET error for %s: %s", key, e)
        return default

    async def _set_to_storage(self, key: str, value, ttl: Optional[int] = None) -> bool:
        """Set value to storage with fallback."""
        if self.storage:
            try:
                return await self.storage.set(key, value, ttl)
            except Exception as e:
                logger.error("Storage SET error for %s: %s", key, e)
        return False

    async def _append_to_storage_list(
        self, key: str, value, max_length: Optional[int] = None
    ) -> bool:
        """Append to list in storage."""
        if self.storage:
            try:
                return await self.storage.append_to_list(key, value, max_length)
            except Exception as e:
                logger.error("Storage APPEND error for %s: %s", key, e)
        return False

    def create_request_fingerprint(self, request) -> str:
        """Create unique fingerprint from request headers."""
        headers = request.headers

        # Key headers for fingerprinting
        user_agent = headers.get("user-agent", "")
        accept_language = headers.get("accept-language", "")
        accept_encoding = headers.get("accept-encoding", "")
        accept = headers.get("accept", "")
        connection = headers.get("connection", "")

        # Create composite fingerprint
        fingerprint_data = (
            f"{user_agent}|{accept_language}|{accept_encoding}|{accept}|{connection}"
        )
        fingerprint_hash = hashlib.sha256(fingerprint_data.encode()).hexdigest()[:16]

        return fingerprint_hash

    def create_detailed_fingerprint(self, request) -> RequestFingerprint:
        """Create detailed RequestFingerprint object."""
        headers = request.headers

        return RequestFingerprint(
            user_agent_hash=self.create_request_fingerprint(request),
            accept_language=headers.get("accept-language", ""),
            accept_encoding=headers.get("accept-encoding", ""),
            connection_type=headers.get("connection", ""),
        )

    async def store_fingerprint_for_ip(
        self, client_ip: str, fingerprint: RequestFingerprint
    ) -> bool:
        """Store fingerprint for an IP address."""
        ip_fingerprints_key = f"ip_fingerprints:{client_ip}"

        if self.storage:
            return await self._append_to_storage_list(
                ip_fingerprints_key,
                fingerprint.to_dict(),
                max_length=self.max_fingerprints_per_ip,
            )
        else:
            # Fallback to in-memory storage
            if client_ip not in self.ip_fingerprints:
                self.ip_fingerprints[client_ip] = []

            self.ip_fingerprints[client_ip].append(fingerprint)
            # Keep only last N fingerprints
            if len(self.ip_fingerprints[client_ip]) > self.max_fingerprints_per_ip:
                self.ip_fingerprints[client_ip] = self.ip_fingerprints[client_ip][
                    -self.max_fingerprints_per_ip :
                ]

            return True

    async def is_fingerprint_blocked(self, fingerprint: str) -> bool:
        """Check if a fingerprint is currently blocked."""
        current_time = time.time()
        blocked_fingerprints_key = f"blocked_fingerprints:{fingerprint}"

        if self.storage:
            blocked_until = await self._get_from_storage(blocked_fingerprints_key)
            if blocked_until and current_time < blocked_until:
                return True
            elif blocked_until and current_time >= blocked_until:
                # Unblock expired fingerprint
                await self.storage.delete(blocked_fingerprints_key)
                return False
        else:
            # Fallback to in-memory storage
            blocked_until = self.blocked_fingerprints.get(fingerprint)
            if blocked_until and current_time < blocked_until:
                return True
            elif blocked_until and current_time >= blocked_until:
                del self.blocked_fingerprints[fingerprint]

        return False

    async def block_fingerprint(
        self, fingerprint: str, duration: Optional[int] = None
    ) -> bool:
        """Block a fingerprint for specified duration."""
        current_time = time.time()
        block_duration = duration or self.block_duration
        blocked_until = current_time + block_duration

        blocked_fingerprints_key = f"blocked_fingerprints:{fingerprint}"

        if self.storage:
            return await self._set_to_storage(
                blocked_fingerprints_key, blocked_until, block_duration
            )
        else:
            # Fallback to in-memory storage
            self.blocked_fingerprints[fingerprint] = blocked_until
            return True

    async def get_fingerprint_info(self, fingerprint: str) -> Dict:
        """Get information about a fingerprint."""
        current_time = time.time()
        blocked_fingerprints_key = f"blocked_fingerprints:{fingerprint}"

        # Check if blocked
        blocked_until = None
        if self.storage:
            blocked_until = await self._get_from_storage(blocked_fingerprints_key)
        else:
            blocked_until = self.blocked_fingerprints.get(fingerprint)

        is_blocked = blocked_until and current_time < blocked_until

        return {
            "fingerprint": fingerprint,
            "is_blocked": is_blocked,
            "blocked_until": blocked_until,
            "remaining_block_time": max(0, (blocked_until or 0) - current_time),
        }

    async def get_fingerprints_for_ip(self, client_ip: str) -> List[Dict]:
        """Get all fingerprints associated with an IP address."""
        ip_fingerprints_key = f"ip_fingerprints:{client_ip}"

        if self.storage:
            fingerprints_data = await self.storage.get_list(ip_fingerprints_key)
            return fingerprints_data or []
        else:
            # Fallback to in-memory storage
            fingerprints = self.ip_fingerprints.get(client_ip, [])
            return [fp.to_dict() for fp in fingerprints]

    async def cleanup_expired_blocks(self) -> int:
        """Clean up expired fingerprint blocks. Returns number of cleaned up blocks."""
        current_time = time.time()
        cleaned_count = 0

        if not self.storage:
            # Clean up in-memory storage
            expired_fingerprints = [
                fp
                for fp, blocked_until in self.blocked_fingerprints.items()
                if blocked_until <= current_time
            ]

            for fp in expired_fingerprints:
                del self.blocked_fingerprints[fp]
                cleaned_count += 1

        return cleaned_count

    async def get_blocked_fingerprints_count(self) -> int:
        """Get count of currently blocked fingerprints."""
        current_time = time.time()

        if self.storage:
            # This would require additional storage methods to efficiently count
            # For now, return 0 as we'd need to implement pattern-based key scanning
            return 0
        else:
            # Count non-expired blocks in memory
            active_blocks = [
                fp
                for fp, blocked_until in self.blocked_fingerprints.items()
                if blocked_until > current_time
            ]
            return len(active_blocks)

    def get_fingerprint_metrics(self) -> Dict:
        """Get fingerprint-related metrics."""
        if self.storage:
            # With storage backend, metrics would need separate tracking
            return {
                "storage_backed": True,
                "total_ips_tracked": "unknown",  # Would need separate counter
                "blocked_fingerprints": "unknown",  # Would need pattern scan
            }
        else:
            current_time = time.time()
            active_blocks = [
                fp
                for fp, blocked_until in self.blocked_fingerprints.items()
                if blocked_until > current_time
            ]

            return {
                "storage_backed": False,
                "total_ips_tracked": len(self.ip_fingerprints),
                "blocked_fingerprints": len(active_blocks),
                "total_fingerprints_stored": sum(
                    len(fps) for fps in self.ip_fingerprints.values()
                ),
            }
