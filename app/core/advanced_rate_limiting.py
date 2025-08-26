"""
Advanced rate limiting with IP switching attack protection and pluggable storage backends.
"""
import time
import hashlib
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict, deque
from dataclasses import dataclass, field, asdict
import logging

from .storage_backends import StorageBackend
from .storage_factory import StorageFactory, StorageType

logger = logging.getLogger(__name__)

@dataclass
class RequestFingerprint:
    """Request fingerprinting for advanced detection."""
    user_agent_hash: str
    accept_language: str
    accept_encoding: str
    connection_type: str
    content_length: Optional[int] = None
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RequestFingerprint':
        """Create from dictionary."""
        return cls(**data)

@dataclass
class AttackPattern:
    """Attack pattern detection."""
    rapid_ip_changes: int = 0
    identical_requests: int = 0
    suspicious_user_agents: int = 0
    geographic_anomalies: int = 0
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AttackPattern':
        """Create from dictionary."""
        return cls(**data)

class AdvancedRateLimiter:
    """
    Advanced rate limiter with IP switching attack detection and Redis storage.
    """
    
    def __init__(
        self, 
        storage_backend: Optional[StorageBackend] = None,
        storage_config: Optional[Dict[str, Any]] = None
    ):
        # Storage backend
        self.storage: Optional[StorageBackend] = storage_backend
        self.storage_config = storage_config or {}
        
        # Fallback in-memory storage for immediate operations
        self.ip_requests: Dict[str, deque] = defaultdict(deque)
        self.fingerprint_requests: Dict[str, deque] = defaultdict(deque)
        self.ip_fingerprints: Dict[str, List[RequestFingerprint]] = defaultdict(list)
        self.recent_ips: deque = deque(maxlen=1000)
        self.ip_change_patterns: Dict[str, List[float]] = defaultdict(list)
        self.blocked_fingerprints: Dict[str, float] = {}
        self.blocked_patterns: Dict[str, float] = {}
        
        # Behavioral analysis
        self.request_patterns: Dict[str, List[float]] = defaultdict(list)
        self.file_hash_requests: Dict[str, List[Tuple[str, float]]] = defaultdict(list)  # file_hash: [(ip, timestamp)]
        
        # Global suspicious activity
        self.global_request_rate: deque = deque(maxlen=1000)
        self.attack_score: float = 0.0
        
        # Configuration
        self.window_size = 60  # 1 minute
        self.max_requests_per_ip = 10
        self.max_fingerprint_requests = 3  # More strict fingerprint limit
        self.suspicious_ip_changes_threshold = 5  # Lower threshold for IP changes
        self.attack_block_duration = 300  # 5 minutes
        
        # Attack Detection Thresholds
        self.ip_switching_threshold = 3  # Same fingerprint from 3+ IPs
        self.global_attack_threshold = 0.8 # Attack score threshold
        self.bot_behavior_variance = 0.1  # Request timing variances
        
        # Storage initialization flag
        self._storage_initialized = False
    
    async def initialize_storage(
        self, 
        storage_type: StorageType = StorageType.MEMORY,
        config: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Initialize storage backend."""
        try:
            if self.storage is None:
                config = config or self.storage_config
                self.storage = await StorageFactory.create_storage(storage_type, config)
            
            # Test storage connection
            if await self.storage.ping():
                self._storage_initialized = True
                logger.info(f"Storage backend initialized: {storage_type.value}")
                return True
            else:
                logger.error("Storage backend ping failed")
                return False
                
        except Exception as e:
            logger.error(f"Failed to initialize storage backend: {e}")
            return False
    
    async def _get_from_storage(self, key: str, default=None):
        """Get value from storage with fallback."""
        if self.storage and self._storage_initialized:
            try:
                return await self.storage.get(key) or default
            except Exception as e:
                logger.error(f"Storage GET error for {key}: {e}")
        return default
    
    async def _set_to_storage(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value to storage with fallback."""
        if self.storage and self._storage_initialized:
            try:
                return await self.storage.set(key, value, ttl)
            except Exception as e:
                logger.error(f"Storage SET error for {key}: {e}")
        return False
    
    async def _increment_in_storage(self, key: str, amount: int = 1, ttl: Optional[int] = None) -> int:
        """Increment counter in storage."""
        if self.storage and self._storage_initialized:
            try:
                return await self.storage.increment(key, amount, ttl)
            except Exception as e:
                logger.error(f"Storage INCREMENT error for {key}: {e}")
        return 0
    
    async def _append_to_storage_list(self, key: str, value: Any, max_length: Optional[int] = None) -> bool:
        """Append to list in storage."""
        if self.storage and self._storage_initialized:
            try:
                return await self.storage.append_to_list(key, value, max_length)
            except Exception as e:
                logger.error(f"Storage APPEND error for {key}: {e}")
        return False
    
    async def _get_storage_list(self, key: str) -> List[Any]:
        """Get list from storage."""
        if self.storage and self._storage_initialized:
            try:
                return await self.storage.get_list(key)
            except Exception as e:
                logger.error(f"Storage GET_LIST error for {key}: {e}")
        return []
        
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
        fingerprint_data = f"{user_agent}|{accept_language}|{accept_encoding}|{accept}|{connection}"
        fingerprint_hash = hashlib.sha256(fingerprint_data.encode()).hexdigest()[:16]
        
        return fingerprint_hash
    
    def detect_ip_switching_attack(self, client_ip: str, fingerprint: str) -> bool:
        """Detect if request is part of IP switching attack."""
        current_time = time.time()
        
        # Track recent IPs
        self.recent_ips.append((client_ip, current_time))
        
        # Clean old entries
        cutoff_time = current_time - 300  # 5 minutes
        while self.recent_ips and self.recent_ips[0][1] < cutoff_time:
            self.recent_ips.popleft()
        
        # Count unique IPs in recent time
        recent_unique_ips = len(set(ip for ip, _ in self.recent_ips))
        
        # Detect rapid IP changes with same fingerprint
        fingerprint_ips = [ip for ip, ts in self.recent_ips 
                          if any(fp.user_agent_hash == fingerprint for fp in self.ip_fingerprints.get(ip, []))]
        
        if len(set(fingerprint_ips)) > 3:  # Same fingerprint from 3+ different IPs (more strict)
            logger.warning(f"🚨 IP switching attack detected: fingerprint {fingerprint} from {len(set(fingerprint_ips))} IPs")
            return True
        
        # Detect if too many unique IPs in short time (distributed attack)
        if recent_unique_ips > self.suspicious_ip_changes_threshold:
            logger.warning(f"🚨 Distributed attack detected: {recent_unique_ips} unique IPs in 5 minutes")
            return True
        
        return False
    
    def detect_behavioral_anomalies(self, client_ip: str, endpoint: str, file_hash: Optional[str] = None) -> bool:
        """Detect behavioral anomalies that suggest automated attacks."""
        current_time = time.time()
        
        # Pattern 1: Identical file uploads from multiple IPs
        if file_hash:
            self.file_hash_requests[file_hash].append((client_ip, current_time))
            
            # Clean old entries
            cutoff_time = current_time - 300  # 5 minutes
            self.file_hash_requests[file_hash] = [
                (ip, ts) for ip, ts in self.file_hash_requests[file_hash] 
                if ts > cutoff_time
            ]
            
            # Check if same file from multiple IPs
            unique_ips = set(ip for ip, _ in self.file_hash_requests[file_hash])
            if len(unique_ips) > 3:  # Same file from 3+ different IPs
                logger.warning(f"🚨 Coordinated attack detected: same file hash from {len(unique_ips)} IPs")
                return True
        
        # Pattern 2: Perfect timing patterns (bot behavior)
        self.request_patterns[client_ip].append(current_time)
        if len(self.request_patterns[client_ip]) > 3:
            intervals = []
            requests = self.request_patterns[client_ip][-10:]  # Last 10 requests
            for i in range(1, len(requests)):
                intervals.append(requests[i] - requests[i-1])
            
            # Check for perfectly regular intervals (bot behavior)
            if len(intervals) > 3:
                avg_interval = sum(intervals) / len(intervals)
                variance = sum((x - avg_interval) ** 2 for x in intervals) / len(intervals)
                
                if variance < 0.1 and avg_interval < 5:  # Very regular, fast requests
                    logger.warning(f"🚨 Bot behavior detected: regular intervals from {client_ip}")
                    return True
        
        return False
    
    def calculate_global_attack_score(self) -> float:
        """Calculate global attack likelihood score."""
        current_time = time.time()
        
        # Add current request to global rate tracking
        self.global_request_rate.append(current_time)
        
        # Clean old entries
        cutoff_time = current_time - 60  # 1 minute
        while self.global_request_rate and self.global_request_rate[0] < cutoff_time:
            self.global_request_rate.popleft()
        
        # Calculate requests per minute
        requests_per_minute = len(self.global_request_rate)
        
        # Update attack score based on various factors
        factors = {
            "high_request_rate": min(requests_per_minute / 100, 1.0),  # Normalize to 0-1
            "unique_ips": min(len(set(ip for ip, _ in self.recent_ips)) / 50, 1.0),
            "blocked_fingerprints": min(len(self.blocked_fingerprints) / 10, 1.0),
        }
        
        # Weighted attack score
        self.attack_score = (
            factors["high_request_rate"] * 0.4 +
            factors["unique_ips"] * 0.4 +
            factors["blocked_fingerprints"] * 0.2
        )
        
        return self.attack_score
    
    async def is_request_allowed(self, client_ip: str, endpoint: str, request, file_hash: Optional[str] = None) -> Tuple[bool, str, Dict]:
        """
        Advanced rate limiting check with multiple protection layers and storage backend.
        Returns: (is_allowed, reason, details)
        """
        current_time = time.time()
        
        # Create request fingerprint
        fingerprint = self.create_request_fingerprint(request)
        
        # Storage keys
        ip_requests_key = f"ip_requests:{client_ip}"
        fingerprint_requests_key = f"fingerprint_requests:{fingerprint}"
        blocked_fingerprints_key = f"blocked_fingerprints:{fingerprint}"
        ip_fingerprints_key = f"ip_fingerprints:{client_ip}"
        
        # Store fingerprint for this IP in storage
        fingerprint_data = RequestFingerprint(
            user_agent_hash=fingerprint,
            accept_language=request.headers.get("accept-language", ""),
            accept_encoding=request.headers.get("accept-encoding", ""),
            connection_type=request.headers.get("connection", "")
        )
        
        await self._append_to_storage_list(
            ip_fingerprints_key, 
            fingerprint_data.to_dict(), 
            max_length=100
        )
        
        # Layer 1: Check if fingerprint is blocked
        blocked_until = await self._get_from_storage(blocked_fingerprints_key)
        if blocked_until and current_time < blocked_until:
            return False, "Fingerprint blocked due to suspicious activity", {
                "fingerprint": fingerprint,
                "unblock_time": blocked_until
            }
        elif blocked_until and current_time >= blocked_until:
            # Unblock expired fingerprint
            await self.storage.delete(blocked_fingerprints_key)
        
        # Layer 2: Traditional IP rate limiting with storage
        await self._increment_in_storage(ip_requests_key, 1, self.window_size)
        ip_request_count = await self._get_from_storage(ip_requests_key, 0)
        
        if ip_request_count > self.max_requests_per_ip:
            return False, "IP rate limit exceeded", {
                "ip": client_ip,
                "requests_in_window": ip_request_count
            }
        
        # Layer 3: Fingerprint rate limiting with storage
        await self._increment_in_storage(fingerprint_requests_key, 1, self.window_size)
        fingerprint_request_count = await self._get_from_storage(fingerprint_requests_key, 0)
        
        if fingerprint_request_count > self.max_fingerprint_requests:
            # Block this fingerprint temporarily
            await self._set_to_storage(
                blocked_fingerprints_key, 
                current_time + self.attack_block_duration,
                self.attack_block_duration
            )
            return False, "Fingerprint rate limit exceeded", {
                "fingerprint": fingerprint,
                "requests_in_window": fingerprint_request_count
            }
        
        # Layer 4: IP switching attack detection (hybrid approach)
        if await self.detect_ip_switching_attack_async(client_ip, fingerprint):
            await self._set_to_storage(
                blocked_fingerprints_key, 
                current_time + self.attack_block_duration,
                self.attack_block_duration
            )
            return False, "IP switching attack detected", {
                "fingerprint": fingerprint,
                "attack_type": "ip_switching"
            }
        
        # Layer 5: Behavioral anomaly detection (async)
        if await self.detect_behavioral_anomalies_async(client_ip, endpoint, file_hash):
            await self._set_to_storage(
                blocked_fingerprints_key, 
                current_time + self.attack_block_duration,
                self.attack_block_duration
            )
            return False, "Behavioral anomaly detected", {
                "fingerprint": fingerprint,
                "attack_pattern": "behavioral_anomaly"
            }
        
        # Layer 6: Global attack score analysis (async)
        attack_score = await self.calculate_global_attack_score_async()
        if attack_score > self.global_attack_threshold:
            # Reduce rate limits during high attack periods
            reduced_ip_limit = self.max_requests_per_ip // 2
            reduced_fingerprint_limit = self.max_fingerprint_requests // 2
            
            if (ip_request_count > reduced_ip_limit or 
                fingerprint_request_count > reduced_fingerprint_limit):
                return False, "High attack score - reduced limits applied", {
                    "attack_score": attack_score,
                    "reduced_limits": True
                }
        
        # Request is allowed
        return True, "Request allowed", {
            "ip": client_ip,
            "fingerprint": fingerprint,
            "attack_score": attack_score,
            "requests_in_window": ip_request_count
        }
    
    async def detect_ip_switching_attack_async(self, client_ip: str, fingerprint: str) -> bool:
        """Async version of IP switching attack detection using storage."""
        current_time = time.time()
        
        # Store recent IP activity
        recent_ips_key = "recent_ips"
        await self._append_to_storage_list(recent_ips_key, {
            "ip": client_ip,
            "timestamp": current_time,
            "fingerprint": fingerprint
        }, max_length=1000)
        
        # Get recent IP activities
        recent_activities = await self._get_storage_list(recent_ips_key)
        
        # Filter activities from last 5 minutes
        cutoff_time = current_time - 300
        recent_activities = [
            activity for activity in recent_activities 
            if isinstance(activity, dict) and activity.get("timestamp", 0) > cutoff_time
        ]
        
        # Count unique IPs with same fingerprint
        fingerprint_ips = set()
        for activity in recent_activities:
            if activity.get("fingerprint") == fingerprint:
                fingerprint_ips.add(activity.get("ip"))
        
        # Detect rapid IP changes with same fingerprint
        if len(fingerprint_ips) > self.ip_switching_threshold:
            logger.warning(f"🚨 IP switching attack detected: fingerprint {fingerprint} from {len(fingerprint_ips)} IPs")
            return True
        
        # Count total unique IPs in recent time
        unique_ips = set(activity.get("ip") for activity in recent_activities)
        if len(unique_ips) > self.suspicious_ip_changes_threshold:
            logger.warning(f"🚨 Distributed attack detected: {len(unique_ips)} unique IPs in 5 minutes")
            return True
        
        return False
    
    async def detect_behavioral_anomalies_async(self, client_ip: str, endpoint: str, file_hash: Optional[str] = None) -> bool:
        """Async behavioral anomaly detection using storage."""
        current_time = time.time()
        
        # Pattern 1: Identical file uploads from multiple IPs
        if file_hash:
            file_requests_key = f"file_requests:{file_hash}"
            await self._append_to_storage_list(file_requests_key, {
                "ip": client_ip,
                "timestamp": current_time
            }, max_length=100)
            
            # Get file request history
            file_requests = await self._get_storage_list(file_requests_key)
            
            # Filter recent requests (5 minutes)
            cutoff_time = current_time - 300
            recent_file_requests = [
                req for req in file_requests 
                if isinstance(req, dict) and req.get("timestamp", 0) > cutoff_time
            ]
            
            # Check unique IPs for same file
            unique_ips = set(req.get("ip") for req in recent_file_requests)
            if len(unique_ips) > 3:
                logger.warning(f"🚨 Coordinated attack detected: same file hash from {len(unique_ips)} IPs")
                return True
        
        # Pattern 2: Perfect timing patterns (bot behavior)
        timing_key = f"request_timing:{client_ip}"
        await self._append_to_storage_list(timing_key, current_time, max_length=10)
        
        request_times = await self._get_storage_list(timing_key)
        if len(request_times) > 3:
            # Calculate intervals
            intervals = []
            for i in range(1, len(request_times)):
                if isinstance(request_times[i], (int, float)) and isinstance(request_times[i-1], (int, float)):
                    intervals.append(request_times[i] - request_times[i-1])
            
            # Check for regular intervals (bot behavior)
            if len(intervals) > 3:
                avg_interval = sum(intervals) / len(intervals)
                variance = sum((x - avg_interval) ** 2 for x in intervals) / len(intervals)
                
                if variance < self.bot_behavior_variance and avg_interval < 5:
                    logger.warning(f"🚨 Bot behavior detected: regular intervals from {client_ip}")
                    return True
        
        return False
    
    async def calculate_global_attack_score_async(self) -> float:
        """Async global attack score calculation using storage."""
        current_time = time.time()
        
        # Update global request rate
        global_rate_key = "global_request_rate"
        await self._append_to_storage_list(global_rate_key, current_time, max_length=1000)
        
        # Get recent requests
        recent_requests = await self._get_storage_list(global_rate_key)
        
        # Filter last minute
        cutoff_time = current_time - 60
        recent_requests = [
            req_time for req_time in recent_requests 
            if isinstance(req_time, (int, float)) and req_time > cutoff_time
        ]
        
        requests_per_minute = len(recent_requests)
        
        # Get recent IP activities for unique IP count
        recent_activities = await self._get_storage_list("recent_ips")
        unique_ips = set()
        for activity in recent_activities:
            if isinstance(activity, dict) and activity.get("timestamp", 0) > cutoff_time:
                unique_ips.add(activity.get("ip", ""))
        
        # Calculate attack score factors
        factors = {
            "high_request_rate": min(requests_per_minute / 100, 1.0),
            "unique_ips": min(len(unique_ips) / 50, 1.0),
            "blocked_fingerprints": 0.1,  # Simplified for now
        }
        
        # Weighted attack score
        attack_score = (
            factors["high_request_rate"] * 0.4 +
            factors["unique_ips"] * 0.4 +
            factors["blocked_fingerprints"] * 0.2
        )
        
        # Store attack score
        await self._set_to_storage("global_attack_score", attack_score, 60)
        self.attack_score = attack_score  # Also update local copy
        
        return attack_score
    
    async def get_security_status_async(self) -> Dict:
        """Get current security status and statistics with async storage."""
        if self.storage and self._storage_initialized:
            # Get data from storage
            recent_activities = await self._get_storage_list("recent_ips")
            attack_score = await self._get_from_storage("global_attack_score", 0.0)
            
            # Count unique IPs in last hour
            current_time = time.time()
            cutoff_time = current_time - 3600
            unique_ips = set()
            for activity in recent_activities:
                if isinstance(activity, dict) and activity.get("timestamp", 0) > cutoff_time:
                    unique_ips.add(activity.get("ip", ""))
            
            # Get recent requests for RPM calculation
            recent_requests = await self._get_storage_list("global_request_rate")
            cutoff_time = current_time - 60
            requests_last_minute = [
                req for req in recent_requests 
                if isinstance(req, (int, float)) and req > cutoff_time
            ]
            
            storage_info = await self.storage.get_info()
            
            return {
                "storage_backend": storage_info.get("backend_type", "unknown"),
                "storage_healthy": storage_info.get("is_healthy", False),
                "unique_ips_last_hour": len(unique_ips),
                "global_attack_score": round(attack_score, 3),
                "requests_per_minute": len(requests_last_minute),
                "storage_info": storage_info,
                "protection_layers": [
                    "IP Rate Limiting (Redis)",
                    "Fingerprint Rate Limiting (Redis)", 
                    "IP Switching Detection (Redis)",
                    "Behavioral Analysis (Redis)",
                    "Global Attack Detection (Redis)"
                ]
            }
        else:
            # Fallback to original method
            return self.get_security_status()
    
    def get_security_status(self) -> Dict:
        """Get current security status and statistics (fallback method)."""
        current_time = time.time()
        
        return {
            "active_ips": len(self.ip_requests),
            "active_fingerprints": len(self.fingerprint_requests),
            "blocked_fingerprints": len(self.blocked_fingerprints),
            "recent_unique_ips": len(set(ip for ip, _ in self.recent_ips)),
            "global_attack_score": round(self.attack_score, 3),
            "requests_per_minute": len([t for t in self.global_request_rate if t > current_time - 60]),
            "protection_layers": [
                "IP Rate Limiting",
                "Fingerprint Rate Limiting", 
                "IP Switching Detection",
                "Behavioral Analysis",
                "Global Attack Detection"
            ]
        }

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

# Global instance (will be initialized with Redis in startup)
advanced_rate_limiter: Optional[AdvancedRateLimiter] = None