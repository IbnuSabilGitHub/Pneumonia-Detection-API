"""
Security utilities and helpers.
"""
import hashlib
import time
from typing import Dict, Any
from fastapi import Request
from collections import defaultdict, deque

from ..core.settings import settings
from ..core.logger import get_logger
from ..utils.exceptions import RateLimitError

logger = get_logger(__name__)


class InMemoryRateLimiter:
    """In-memory rate limiter for development/simple deployments."""
    
    def __init__(self):
        self.requests: Dict[str, deque] = defaultdict(deque)
        self.blocked_ips: Dict[str, float] = {}
    
    def is_allowed(
        self, 
        client_ip: str, 
        max_requests: int = None, 
        window_seconds: int = None
    ) -> bool:
        """
        Check if request is allowed based on rate limiting.
        
        Args:
            client_ip: Client IP address
            max_requests: Maximum requests allowed in window
            window_seconds: Time window in seconds
            
        Returns:
            True if request is allowed, False otherwise
        """
        max_requests = max_requests or settings.rate_limit_requests
        window_seconds = window_seconds or settings.rate_limit_window
        current_time = time.time()
        
        # Check if IP is temporarily blocked
        if client_ip in self.blocked_ips:
            if current_time < self.blocked_ips[client_ip]:
                return False
            else:
                del self.blocked_ips[client_ip]
        
        # Clean old requests
        requests = self.requests[client_ip]
        while requests and requests[0] < current_time - window_seconds:
            requests.popleft()
        
        # Check rate limit
        if len(requests) >= max_requests:
            # Block IP temporarily
            self.blocked_ips[client_ip] = current_time + settings.rate_limit_block_duration
            logger.warning(f"Rate limit exceeded for IP: {client_ip}")
            return False
        
        # Add current request
        requests.append(current_time)
        return True
    
    def get_request_count(self, client_ip: str, window_seconds: int = None) -> int:
        """Get current request count for an IP."""
        window_seconds = window_seconds or settings.rate_limit_window
        current_time = time.time()
        
        requests = self.requests[client_ip]
        # Clean old requests
        while requests and requests[0] < current_time - window_seconds:
            requests.popleft()
        
        return len(requests)
    
    def is_blocked(self, client_ip: str) -> bool:
        """Check if IP is currently blocked."""
        if client_ip not in self.blocked_ips:
            return False
        
        current_time = time.time()
        if current_time >= self.blocked_ips[client_ip]:
            del self.blocked_ips[client_ip]
            return False
        
        return True


class FileHashCache:
    """Cache for file hashes to detect duplicates."""
    
    def __init__(self):
        self.cache: Dict[str, float] = {}
    
    def is_duplicate(self, file_hash: str, cache_duration: int = None) -> bool:
        """
        Check if file hash is a duplicate within cache duration.
        
        Args:
            file_hash: File hash to check
            cache_duration: Cache duration in seconds
            
        Returns:
            True if duplicate, False otherwise
        """
        cache_duration = cache_duration or settings.cache_duration
        current_time = time.time()
        
        if file_hash in self.cache:
            last_upload_time = self.cache[file_hash]
            if current_time - last_upload_time < cache_duration:
                return True
        
        # Update cache
        self.cache[file_hash] = current_time
        return False
    
    def cleanup_expired(self, cache_duration: int = None) -> int:
        """Clean up expired cache entries."""
        cache_duration = cache_duration or settings.cache_duration
        current_time = time.time()
        
        expired_keys = [
            key for key, timestamp in self.cache.items()
            if current_time - timestamp >= cache_duration
        ]
        
        for key in expired_keys:
            del self.cache[key]
        
        return len(expired_keys)
    
    def get_cache_size(self) -> int:
        """Get current cache size."""
        return len(self.cache)


def get_client_ip(request: Request) -> str:
    """
    Extract client IP address from request.
    
    Args:
        request: FastAPI request object
        
    Returns:
        Client IP address
    """
    # Check for forwarded headers (for proxies/load balancers)
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()
    
    real_ip = request.headers.get("X-Real-IP")
    if real_ip:
        return real_ip
    
    return request.client.host


def calculate_file_hash(contents: bytes) -> str:
    """
    Calculate SHA-256 hash of file contents.
    
    Args:
        contents: File contents as bytes
        
    Returns:
        Hex digest of SHA-256 hash
    """
    return hashlib.sha256(contents).hexdigest()


# Global instances
rate_limiter = InMemoryRateLimiter()
file_hash_cache = FileHashCache()
