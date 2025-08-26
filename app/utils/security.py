"""
Security utilities and helpers.
"""
import hashlib
import time
from typing import Dict
from fastapi import Request

from ..core.settings import settings
from ..core.logger import get_logger

logger = get_logger(__name__)



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
    forwarded_for = request.headers.get("x-forwarded-for")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()
        
    real_ip = request.headers.get("x-real-ip")
    if real_ip:
        return real_ip.strip()
        
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
file_hash_cache = FileHashCache()
