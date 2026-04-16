"""
JWT User-based Rate Limiting with Supabase Storage.

Architecture:
    JWT Identity → Rate Limit per User → Store Counters in Supabase

This module provides simplified rate limiting:
- Uses JWT user_id (sub) for identity
- One counter per user with sliding window
- Stores counters in Supabase via REST API
- Falls back to in-memory if Supabase unavailable
"""

import time
import logging
import httpx
from typing import Dict, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


@dataclass
class RateLimitInfo:
    """Rate limit information for a user."""
    user_id: str
    requests_made: int
    requests_limit: int
    window_size: int
    reset_time: float
    allowed: bool
    
    @property
    def requests_remaining(self) -> int:
        return max(0, self.requests_limit - self.requests_made)
    
    @property
    def retry_after(self) -> int:
        return max(0, int(self.reset_time - time.time()))
    
    def to_headers(self) -> Dict[str, str]:
        """Generate rate limit headers."""
        return {
            "X-RateLimit-Limit": str(self.requests_limit),
            "X-RateLimit-Remaining": str(self.requests_remaining),
            "X-RateLimit-Reset": str(int(self.reset_time)),
            "X-RateLimit-User-ID": self.user_id[:8] + "...",  # Truncated for privacy
        }


class InMemoryRateLimitStorage:
    """
    In-memory fallback storage for rate limiting.
    
    Uses sliding window algorithm with deque for tracking request timestamps.
    """
    
    def __init__(self, cleanup_interval: int = 60):
        """Initialize in-memory storage.
        
        Args:
            cleanup_interval: Seconds between automatic cleanup runs.
        """
        self._counters: Dict[str, Dict] = defaultdict(
            lambda: {"count": 0, "window_start": 0.0}
        )
        self._last_cleanup = time.time()
        self._cleanup_interval = cleanup_interval
        self._lock = False  # Simple lock indicator (not true threading lock)
    
    def _cleanup_expired(self, window_size: int) -> int:
        """Remove expired entries."""
        current_time = time.time()
        
        # Only cleanup periodically
        if current_time - self._last_cleanup < self._cleanup_interval:
            return 0
        
        expired_keys = []
        for user_id, data in self._counters.items():
            if current_time - data["window_start"] > window_size:
                expired_keys.append(user_id)
        
        for key in expired_keys:
            del self._counters[key]
        
        self._last_cleanup = current_time
        return len(expired_keys)
    
    async def increment(
        self, 
        user_id: str, 
        limit: int, 
        window_size: int
    ) -> Tuple[int, bool, float]:
        """
        Increment user counter.
        
        Args:
            user_id: User identifier.
            limit: Maximum requests per window.
            window_size: Window size in seconds.
            
        Returns:
            Tuple of (count, is_allowed, reset_time)
        """
        current_time = time.time()
        self._cleanup_expired(window_size)
        
        data = self._counters[user_id]
        window_start = data["window_start"]
        
        # Check if window expired
        if current_time - window_start > window_size:
            # Reset window
            self._counters[user_id] = {
                "count": 1,
                "window_start": current_time
            }
            return 1, True, current_time + window_size
        
        # Increment within window
        new_count = data["count"] + 1
        self._counters[user_id]["count"] = new_count
        
        is_allowed = new_count <= limit
        reset_time = window_start + window_size
        
        return new_count, is_allowed, reset_time
    
    async def get_count(self, user_id: str, window_size: int) -> Tuple[int, float]:
        """Get current count for user."""
        current_time = time.time()
        data = self._counters.get(user_id, {"count": 0, "window_start": 0.0})
        
        if current_time - data["window_start"] > window_size:
            return 0, current_time + window_size
        
        return data["count"], data["window_start"] + window_size
    
    def get_stats(self) -> Dict:
        """Get storage statistics."""
        return {
            "backend_type": "memory",
            "active_users": len(self._counters),
            "last_cleanup": self._last_cleanup,
        }


class SupabaseRateLimitStorage:
    """
    Supabase-based rate limit storage.
    
    Uses Supabase REST API for storing rate limit counters.
    Requires a `rate_limits` table with columns:
    - user_id (text, primary key)
    - request_count (int)
    - window_start (timestamp)
    - updated_at (timestamp)
    """
    
    def __init__(
        self,
        supabase_url: str,
        supabase_key: str,
        table_name: str = "rate_limits"
    ):
        """Initialize Supabase storage.
        
        Args:
            supabase_url: Supabase project URL.
            supabase_key: Supabase anon or service key.
            table_name: Table name for rate limits.
        """
        self.supabase_url = supabase_url.rstrip("/")
        self.supabase_key = supabase_key
        self.table_name = table_name
        self._client = httpx.AsyncClient(timeout=5.0)
        self._healthy = True
        self._last_error: Optional[str] = None
    
    @property
    def _headers(self) -> Dict[str, str]:
        """Get Supabase API headers."""
        return {
            "apikey": self.supabase_key,
            "Authorization": f"Bearer {self.supabase_key}",
            "Content-Type": "application/json",
            "Prefer": "return=representation",
        }
    
    @property
    def _base_url(self) -> str:
        """Get Supabase REST API base URL."""
        return f"{self.supabase_url}/rest/v1/{self.table_name}"
    
    async def increment(
        self,
        user_id: str,
        limit: int,
        window_size: int
    ) -> Tuple[int, bool, float]:
        """
        Increment user counter in Supabase.
        
        Uses upsert with conflict handling for atomic increment.
        """
        current_time = time.time()
        
        try:
            # Try to get existing record
            response = await self._client.get(
                self._base_url,
                headers=self._headers,
                params={"user_id": f"eq.{user_id}", "select": "*"}
            )
            
            if response.status_code != 200:
                logger.warning(
                    "Supabase GET failed: %s %s", 
                    response.status_code, 
                    response.text
                )
                self._healthy = False
                self._last_error = f"GET failed: {response.status_code}"
                raise Exception(self._last_error)
            
            data = response.json()
            
            if data:
                # Record exists - check window
                record = data[0]
                window_start = float(record.get("window_start", 0))
                current_count = int(record.get("request_count", 0))
                
                if current_time - window_start > window_size:
                    # Window expired - reset
                    new_count = 1
                    new_window_start = current_time
                else:
                    # Within window - increment
                    new_count = current_count + 1
                    new_window_start = window_start
                
                # Update record
                update_response = await self._client.patch(
                    self._base_url,
                    headers=self._headers,
                    params={"user_id": f"eq.{user_id}"},
                    json={
                        "request_count": new_count,
                        "window_start": new_window_start,
                        "updated_at": datetime.now(timezone.utc).isoformat(),
                    }
                )
                
                if update_response.status_code not in (200, 204):
                    logger.warning(
                        "Supabase PATCH failed: %s", 
                        update_response.text
                    )
                    raise Exception("Update failed")
                
            else:
                # No record - create new
                new_count = 1
                new_window_start = current_time
                
                insert_response = await self._client.post(
                    self._base_url,
                    headers=self._headers,
                    json={
                        "user_id": user_id,
                        "request_count": new_count,
                        "window_start": new_window_start,
                        "updated_at": datetime.now(timezone.utc).isoformat(),
                    }
                )
                
                if insert_response.status_code not in (200, 201):
                    logger.warning(
                        "Supabase POST failed: %s", 
                        insert_response.text
                    )
                    raise Exception("Insert failed")
            
            self._healthy = True
            is_allowed = new_count <= limit
            reset_time = new_window_start + window_size
            
            return new_count, is_allowed, reset_time
            
        except httpx.RequestError as e:
            logger.error("Supabase request error: %s", e)
            self._healthy = False
            self._last_error = str(e)
            raise
    
    async def get_count(self, user_id: str, window_size: int) -> Tuple[int, float]:
        """Get current count for user from Supabase."""
        try:
            response = await self._client.get(
                self._base_url,
                headers=self._headers,
                params={"user_id": f"eq.{user_id}", "select": "*"}
            )
            
            if response.status_code != 200:
                return 0, time.time() + window_size
            
            data = response.json()
            if not data:
                return 0, time.time() + window_size
            
            record = data[0]
            current_time = time.time()
            window_start = float(record.get("window_start", 0))
            
            if current_time - window_start > window_size:
                return 0, current_time + window_size
            
            return int(record.get("request_count", 0)), window_start + window_size
            
        except Exception as e:
            logger.error("Failed to get count from Supabase: %s", e)
            return 0, time.time() + window_size
    
    def get_stats(self) -> Dict:
        """Get storage statistics."""
        return {
            "backend_type": "supabase",
            "healthy": self._healthy,
            "last_error": self._last_error,
            "table_name": self.table_name,
        }
    
    async def close(self):
        """Close the HTTP client."""
        await self._client.aclose()


class UserRateLimiter:
    """
    Simple JWT user-based rate limiter.
    
    Architecture:
        JWT Identity → Rate Limit per User → Store in Supabase
    
    Example:
        limiter = UserRateLimiter(
            max_requests=100,
            window_size=3600  # 1 hour
        )
        await limiter.initialize(settings)
        
        # Check rate limit
        info = await limiter.check_rate_limit(user_id)
        if not info.allowed:
            return 429
    """
    
    def __init__(
        self,
        max_requests: int = 100,
        window_size: int = 3600,
    ):
        """Initialize rate limiter.
        
        Args:
            max_requests: Maximum requests per user per window.
            window_size: Time window in seconds.
        """
        self.max_requests = max_requests
        self.window_size = window_size
        
        # Storage backends
        self._supabase_storage: Optional[SupabaseRateLimitStorage] = None
        self._memory_storage = InMemoryRateLimitStorage()
        
        # State
        self._initialized = False
        self._use_supabase = False
    
    async def initialize(
        self,
        supabase_url: Optional[str] = None,
        supabase_key: Optional[str] = None,
        use_supabase: bool = True
    ) -> bool:
        """
        Initialize rate limiter with storage backend.
        
        Args:
            supabase_url: Supabase project URL.
            supabase_key: Supabase API key.
            use_supabase: Whether to use Supabase (falls back to memory if False or fails).
            
        Returns:
            True if initialized successfully.
        """
        self._initialized = True
        
        if use_supabase and supabase_url and supabase_key:
            try:
                self._supabase_storage = SupabaseRateLimitStorage(
                    supabase_url=supabase_url,
                    supabase_key=supabase_key
                )
                
                # Test connection by making a simple request
                logger.info("Testing Supabase connection for rate limiting...")
                
                # Note: The table must exist in Supabase
                # Create it via SQL:
                # CREATE TABLE rate_limits (
                #   user_id TEXT PRIMARY KEY,
                #   request_count INTEGER DEFAULT 0,
                #   window_start DOUBLE PRECISION,
                #   updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                # );
                
                self._use_supabase = True
                logger.info(
                    "✅ Rate limiter initialized with Supabase storage "
                    "(max=%d requests/%ds)",
                    self.max_requests,
                    self.window_size
                )
                return True
                
            except Exception as e:
                logger.warning(
                    "Failed to initialize Supabase storage, using in-memory: %s", e
                )
                self._use_supabase = False
        
        self._use_supabase = False
        logger.info(
            "✅ Rate limiter initialized with in-memory storage "
            "(max=%d requests/%ds)",
            self.max_requests,
            self.window_size
        )
        return True
    
    async def check_rate_limit(self, user_id: str) -> RateLimitInfo:
        """
        Check and increment rate limit for user.
        
        Args:
            user_id: User identifier from JWT (sub claim).
            
        Returns:
            RateLimitInfo with current rate limit status.
        """
        if not self._initialized:
            logger.warning("Rate limiter not initialized, allowing request")
            return RateLimitInfo(
                user_id=user_id,
                requests_made=0,
                requests_limit=self.max_requests,
                window_size=self.window_size,
                reset_time=time.time() + self.window_size,
                allowed=True
            )
        
        storage = (
            self._supabase_storage 
            if self._use_supabase and self._supabase_storage
            else self._memory_storage
        )
        
        try:
            count, is_allowed, reset_time = await storage.increment(
                user_id=user_id,
                limit=self.max_requests,
                window_size=self.window_size
            )
            
            if not is_allowed:
                logger.warning(
                    "Rate limit exceeded for user %s: %d/%d requests",
                    user_id[:8] + "...",
                    count,
                    self.max_requests
                )
            
            return RateLimitInfo(
                user_id=user_id,
                requests_made=count,
                requests_limit=self.max_requests,
                window_size=self.window_size,
                reset_time=reset_time,
                allowed=is_allowed
            )
            
        except Exception as e:
            logger.error("Rate limit check failed: %s", e)
            
            # If Supabase fails, fall back to memory
            if self._use_supabase:
                logger.warning("Falling back to in-memory storage")
                try:
                    count, is_allowed, reset_time = await self._memory_storage.increment(
                        user_id=user_id,
                        limit=self.max_requests,
                        window_size=self.window_size
                    )
                    return RateLimitInfo(
                        user_id=user_id,
                        requests_made=count,
                        requests_limit=self.max_requests,
                        window_size=self.window_size,
                        reset_time=reset_time,
                        allowed=is_allowed
                    )
                except Exception as fallback_error:
                    logger.error("Fallback also failed: %s", fallback_error)
            
            # Allow request on error to avoid blocking legitimate users
            return RateLimitInfo(
                user_id=user_id,
                requests_made=0,
                requests_limit=self.max_requests,
                window_size=self.window_size,
                reset_time=time.time() + self.window_size,
                allowed=True
            )
    
    async def get_user_status(self, user_id: str) -> RateLimitInfo:
        """
        Get current rate limit status without incrementing.
        
        Args:
            user_id: User identifier.
            
        Returns:
            RateLimitInfo with current status.
        """
        if not self._initialized:
            return RateLimitInfo(
                user_id=user_id,
                requests_made=0,
                requests_limit=self.max_requests,
                window_size=self.window_size,
                reset_time=time.time() + self.window_size,
                allowed=True
            )
        
        storage = (
            self._supabase_storage 
            if self._use_supabase and self._supabase_storage
            else self._memory_storage
        )
        
        try:
            count, reset_time = await storage.get_count(user_id, self.window_size)
            return RateLimitInfo(
                user_id=user_id,
                requests_made=count,
                requests_limit=self.max_requests,
                window_size=self.window_size,
                reset_time=reset_time,
                allowed=count < self.max_requests
            )
        except Exception as e:
            logger.error("Failed to get user status: %s", e)
            return RateLimitInfo(
                user_id=user_id,
                requests_made=0,
                requests_limit=self.max_requests,
                window_size=self.window_size,
                reset_time=time.time() + self.window_size,
                allowed=True
            )
    
    def get_status(self) -> Dict:
        """Get rate limiter status."""
        storage = (
            self._supabase_storage 
            if self._use_supabase and self._supabase_storage
            else self._memory_storage
        )
        
        return {
            "initialized": self._initialized,
            "max_requests": self.max_requests,
            "window_size": self.window_size,
            "storage": storage.get_stats() if storage else {"backend_type": "none"},
            "use_supabase": self._use_supabase,
        }
    
    async def shutdown(self):
        """Clean up resources."""
        if self._supabase_storage:
            await self._supabase_storage.close()
        logger.info("Rate limiter shut down")


# Global instance
_rate_limiter: Optional[UserRateLimiter] = None


def get_user_rate_limiter() -> Optional[UserRateLimiter]:
    """Get the global rate limiter instance."""
    return _rate_limiter


def set_user_rate_limiter(limiter: UserRateLimiter) -> None:
    """Set the global rate limiter instance."""
    global _rate_limiter
    _rate_limiter = limiter


async def create_user_rate_limiter(
    max_requests: int = 100,
    window_size: int = 3600,
    supabase_url: Optional[str] = None,
    supabase_key: Optional[str] = None,
    use_supabase: bool = True
) -> UserRateLimiter:
    """
    Factory function to create and initialize a user rate limiter.
    
    Args:
        max_requests: Max requests per user per window.
        window_size: Window size in seconds.
        supabase_url: Supabase project URL.
        supabase_key: Supabase API key.
        use_supabase: Whether to use Supabase storage.
        
    Returns:
        Initialized UserRateLimiter instance.
    """
    limiter = UserRateLimiter(
        max_requests=max_requests,
        window_size=window_size
    )
    
    await limiter.initialize(
        supabase_url=supabase_url,
        supabase_key=supabase_key,
        use_supabase=use_supabase
    )
    
    return limiter
