import logging
import time
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional

from ..storage_backends import StorageBackend

logger = logging.getLogger(__name__)


class AttackDetector:
    """
    Handles detection of various attack patterns including IP switching,
    behavioral anomalies, and global attack scoring.
    """

    def __init__(
        self, storage: Optional[StorageBackend] = None, config: Optional[Dict] = None
    ):
        self.storage = storage
        self.config = config or {}

        # Get configuration values with defaults
        self.ip_switching_threshold = self.config.get("ip_switching_threshold", 3)
        self.suspicious_ip_changes_threshold = self.config.get(
            "suspicious_ip_changes_threshold", 5
        )
        self.bot_behavior_variance = self.config.get("bot_behavior_variance", 0.1)
        self.global_attack_threshold = self.config.get("global_attack_threshold", 0.8)
        self.coordinated_attack_threshold = self.config.get(
            "coordinated_attack_threshold", 3
        )
        self.bot_timing_threshold = self.config.get("bot_timing_threshold", 5.0)
        self.ip_switching_detection_window = self.config.get(
            "ip_switching_detection_window", 300
        )
        self.behavioral_analysis_window = self.config.get(
            "behavioral_analysis_window", 300
        )
        self.global_attack_score_window = self.config.get(
            "global_attack_score_window", 60
        )

        # In-memory limits
        max_recent_ips = self.config.get("max_recent_ips", 1000)
        max_request_patterns = self.config.get("max_request_patterns_per_ip", 10)
        max_file_hash_requests = self.config.get("max_file_hash_requests", 100)
        max_global_request_rate = self.config.get("max_global_request_rate", 1000)

        # Fallback in-memory storage for immediate operations
        self.recent_ips: deque = deque(maxlen=max_recent_ips)
        self.ip_change_patterns: Dict[str, List[float]] = defaultdict(list)
        self.request_patterns: Dict[str, List[float]] = defaultdict(list)
        self.file_hash_requests: Dict[str, List[tuple]] = defaultdict(list)
        self.global_request_rate: deque = deque(maxlen=max_global_request_rate)
        self.attack_score: float = 0.0
        self.last_attack_type: Optional[str] = None
        self._fingerprint_ip_activity: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=max_recent_ips)
        )

    async def _get_storage_list(self, key: str) -> List[Any]:
        """Get list from storage."""
        if self.storage:
            try:
                return await self.storage.get_list(key)
            except Exception as e:
                logger.error("Storage GET_LIST error for %s: %s", key, e)
        return []

    async def _append_to_storage_list(
        self, key: str, value: Any, max_length: Optional[int] = None
    ) -> bool:
        """Append to list in storage."""
        if self.storage:
            try:
                return await self.storage.append_to_list(key, value, max_length)
            except Exception as e:
                logger.error("Storage APPEND error for %s: %s", key, e)
        return False

    async def _get_from_storage(self, key: str, default=None):
        """Get value from storage with fallback."""
        if self.storage:
            try:
                return await self.storage.get(key) or default
            except Exception as e:
                logger.error("Storage GET error for %s: %s", key, e)
        return default

    async def _set_to_storage(
        self, key: str, value: Any, ttl: Optional[int] = None
    ) -> bool:
        """Set value to storage with fallback."""
        if self.storage:
            try:
                return await self.storage.set(key, value, ttl)
            except Exception as e:
                logger.error("Storage SET error for %s: %s", key, e)
        return False

    def detect_ip_switching_attack(self, client_ip: str, fingerprint: str) -> bool:
        """Detect if request is part of IP switching attack (sync fallback)."""
        current_time = time.time()

        # Track recent IPs
        self.recent_ips.append((client_ip, current_time))

        # Track fingerprint specific IP activity (improves accuracy over previous simplified logic)
        fp_deque = self._fingerprint_ip_activity[fingerprint]
        fp_deque.append((client_ip, current_time))

        # Clean old entries
        cutoff_time = current_time - self.ip_switching_detection_window
        while self.recent_ips and self.recent_ips[0][1] < cutoff_time:
            self.recent_ips.popleft()
        # Clean per-fingerprint deque
        while fp_deque and fp_deque[0][1] < cutoff_time:
            fp_deque.popleft()

        # Count unique IPs in recent time
        recent_unique_ips = len(set(ip for ip, _ in self.recent_ips))

        # Detect rapid IP switching for the SAME fingerprint (accurate, unlike earlier global count)
        fingerprint_unique_ips = len(set(ip for ip, _ in fp_deque))
        if fingerprint_unique_ips > self.ip_switching_threshold:
            self.last_attack_type = "ip_switching"
            logger.warning(
                "🚨 IP switching attack detected: fingerprint %s from %d IPs in %ds window",
                fingerprint,
                fingerprint_unique_ips,
                self.ip_switching_detection_window,
            )
            return True

        # Detect if too many unique IPs in short time (distributed attack)
        if recent_unique_ips > self.suspicious_ip_changes_threshold:
            self.last_attack_type = "distributed"
            logger.warning(
                "🚨 Distributed attack detected: %d unique IPs in %ds window",
                recent_unique_ips,
                self.ip_switching_detection_window,
            )
            return True

        # Reset last attack type if no detection
        self.last_attack_type = None
        return False

    async def detect_ip_switching_attack_async(
        self, client_ip: str, fingerprint: str
    ) -> bool:
        """Async version of IP switching attack detection using storage."""
        current_time = time.time()

        # Store recent IP activity
        recent_ips_key = "recent_ips"
        await self._append_to_storage_list(
            recent_ips_key,
            {"ip": client_ip, "timestamp": current_time, "fingerprint": fingerprint},
            max_length=1000,
        )

        # Get recent IP activities
        recent_activities = await self._get_storage_list(recent_ips_key)

        # Filter activities from detection window
        cutoff_time = current_time - self.ip_switching_detection_window
        recent_activities = [
            activity
            for activity in recent_activities
            if isinstance(activity, dict) and activity.get("timestamp", 0) > cutoff_time
        ]

        # Count unique IPs with same fingerprint
        fingerprint_ips = set()
        for activity in recent_activities:
            if activity.get("fingerprint") == fingerprint:
                fingerprint_ips.add(activity.get("ip"))

        # Detect rapid IP changes with same fingerprint
        if len(fingerprint_ips) > self.ip_switching_threshold:
            self.last_attack_type = "ip_switching"
            logger.warning(
                "🚨 IP switching attack detected: fingerprint %s from %d IPs in %ds window",
                fingerprint,
                len(fingerprint_ips),
                self.ip_switching_detection_window,
            )
            return True

        # Count total unique IPs in recent time
        unique_ips = set(activity.get("ip") for activity in recent_activities)
        if len(unique_ips) > self.suspicious_ip_changes_threshold:
            self.last_attack_type = "distributed"
            logger.warning(
                "🚨 Distributed attack detected: %d unique IPs in %ds window",
                len(unique_ips),
                self.ip_switching_detection_window,
            )
            return True

        self.last_attack_type = None
        return False

    def detect_behavioral_anomalies(
        self, client_ip: str, endpoint: str, file_hash: Optional[str] = None
    ) -> bool:
        """Detect behavioral anomalies that suggest automated attacks (sync fallback)."""
        current_time = time.time()

        # Pattern 1: Identical file uploads from multiple IPs
        if file_hash:
            self.file_hash_requests[file_hash].append((client_ip, current_time))

            # Clean old entries
            cutoff_time = current_time - self.behavioral_analysis_window
            self.file_hash_requests[file_hash] = [
                (ip, ts)
                for ip, ts in self.file_hash_requests[file_hash]
                if ts > cutoff_time
            ]

            # Check if same file from multiple IPs
            unique_ips = set(ip for ip, _ in self.file_hash_requests[file_hash])
            if len(unique_ips) > self.coordinated_attack_threshold:
                logger.warning(
                    f"🚨 Coordinated attack detected: same file hash from {len(unique_ips)} IPs"
                )
                return True

        # Pattern 2: Perfect timing patterns (bot behavior)
        self.request_patterns[client_ip].append(current_time)
        if len(self.request_patterns[client_ip]) > 3:
            intervals = []
            requests = self.request_patterns[client_ip][
                -self.config.get("max_request_patterns_per_ip", 10) :
            ]
            for i in range(1, len(requests)):
                intervals.append(requests[i] - requests[i - 1])

            # Check for perfectly regular intervals (bot behavior)
            if len(intervals) > 3:
                avg_interval = sum(intervals) / len(intervals)
                variance = sum((x - avg_interval) ** 2 for x in intervals) / len(
                    intervals
                )

                if (
                    variance < self.bot_behavior_variance
                    and avg_interval < self.bot_timing_threshold
                ):
                    logger.warning(
                        f"🚨 Bot behavior detected: regular intervals from {client_ip}"
                    )
                    return True

        return False

    async def detect_behavioral_anomalies_async(
        self, client_ip: str, endpoint: str, file_hash: Optional[str] = None
    ) -> bool:
        """Async behavioral anomaly detection using storage."""
        current_time = time.time()

        # Pattern 1: Identical file uploads from multiple IPs
        if file_hash:
            file_requests_key = f"file_requests:{file_hash}"
            await self._append_to_storage_list(
                file_requests_key,
                {"ip": client_ip, "timestamp": current_time},
                max_length=100,
            )

            # Get file request history
            file_requests = await self._get_storage_list(file_requests_key)

            # Filter recent requests
            cutoff_time = current_time - self.behavioral_analysis_window
            recent_file_requests = [
                req
                for req in file_requests
                if isinstance(req, dict) and req.get("timestamp", 0) > cutoff_time
            ]

            # Check unique IPs for same file
            unique_ips = set(req.get("ip") for req in recent_file_requests)
            if len(unique_ips) > self.coordinated_attack_threshold:
                logger.warning(
                    f"🚨 Coordinated attack detected: same file hash from {len(unique_ips)} IPs"
                )
                return True

        # Pattern 2: Perfect timing patterns (bot behavior)
        timing_key = f"request_timing:{client_ip}"
        await self._append_to_storage_list(
            timing_key,
            current_time,
            max_length=self.config.get("max_request_patterns_per_ip", 10),
        )

        request_times = await self._get_storage_list(timing_key)
        if len(request_times) > 3:
            # Calculate intervals
            intervals = []
            for i in range(1, len(request_times)):
                if isinstance(request_times[i], (int, float)) and isinstance(
                    request_times[i - 1], (int, float)
                ):
                    intervals.append(request_times[i] - request_times[i - 1])

            # Check for regular intervals (bot behavior)
            if len(intervals) > 3:
                avg_interval = sum(intervals) / len(intervals)
                variance = sum((x - avg_interval) ** 2 for x in intervals) / len(
                    intervals
                )

                if (
                    variance < self.bot_behavior_variance
                    and avg_interval < self.bot_timing_threshold
                ):
                    logger.warning(
                        f"🚨 Bot behavior detected: regular intervals from {client_ip}"
                    )
                    return True

        return False

    def calculate_global_attack_score(self) -> float:
        """Calculate global attack likelihood score (sync fallback)."""
        current_time = time.time()

        # Add current request to global rate tracking
        self.global_request_rate.append(current_time)

        # Clean old entries
        cutoff_time = current_time - self.global_attack_score_window
        while self.global_request_rate and self.global_request_rate[0] < cutoff_time:
            self.global_request_rate.popleft()

        # Calculate requests per minute
        requests_per_minute = len(self.global_request_rate)

        # Update attack score based on various factors
        factors = {
            "high_request_rate": min(requests_per_minute / 100, 1.0),
            "unique_ips": min(len(set(ip for ip, _ in self.recent_ips)) / 50, 1.0),
            "blocked_fingerprints": 0.1,  # Simplified for now
        }

        # Weighted attack score
        self.attack_score = (
            factors["high_request_rate"] * 0.4
            + factors["unique_ips"] * 0.4
            + factors["blocked_fingerprints"] * 0.2
        )

        return self.attack_score

    async def calculate_global_attack_score_async(self) -> float:
        """Async global attack score calculation using storage."""
        current_time = time.time()

        # Update global request rate
        global_rate_key = "global_request_rate"
        await self._append_to_storage_list(
            global_rate_key,
            current_time,
            max_length=self.config.get("max_global_request_rate", 1000),
        )

        # Get recent requests
        recent_requests = await self._get_storage_list(global_rate_key)

        # Filter last window
        cutoff_time = current_time - self.global_attack_score_window
        recent_requests = [
            req_time
            for req_time in recent_requests
            if isinstance(req_time, (int, float)) and req_time > cutoff_time
        ]

        requests_per_minute = len(recent_requests)

        # Get recent IP activities for unique IP count
        recent_activities = await self._get_storage_list("recent_ips")
        unique_ips = set()
        for activity in recent_activities:
            if (
                isinstance(activity, dict)
                and activity.get("timestamp", 0) > cutoff_time
            ):
                unique_ips.add(activity.get("ip", ""))

        # Calculate attack score factors
        factors = {
            "high_request_rate": min(requests_per_minute / 100, 1.0),
            "unique_ips": min(len(unique_ips) / 50, 1.0),
            "blocked_fingerprints": 0.1,  # Simplified for now
        }

        # Weighted attack score
        attack_score = (
            factors["high_request_rate"] * 0.4
            + factors["unique_ips"] * 0.4
            + factors["blocked_fingerprints"] * 0.2
        )

        # Store attack score
        await self._set_to_storage("global_attack_score", attack_score, 60)
        self.attack_score = attack_score  # Also update local copy

        return attack_score

    async def get_security_metrics(self) -> Dict:
        """Get security metrics for monitoring."""
        if self.storage:
            # Get data from storage
            recent_activities = await self._get_storage_list("recent_ips")
            attack_score = await self._get_from_storage("global_attack_score", 0.0)

            # Count unique IPs in last hour
            current_time = time.time()
            cutoff_time = current_time - 3600
            unique_ips = set()
            for activity in recent_activities:
                if (
                    isinstance(activity, dict)
                    and activity.get("timestamp", 0) > cutoff_time
                ):
                    unique_ips.add(activity.get("ip", ""))

            # Get recent requests for RPM calculation
            recent_requests = await self._get_storage_list("global_request_rate")
            cutoff_time = current_time - 60
            requests_last_minute = [
                req
                for req in recent_requests
                if isinstance(req, (int, float)) and req > cutoff_time
            ]

            return {
                "unique_ips_last_hour": len(unique_ips),
                "global_attack_score": round(attack_score, 3),
                "requests_per_minute": len(requests_last_minute),
            }
        else:
            # Fallback to in-memory data
            current_time = time.time()
            return {
                "recent_unique_ips": len(set(ip for ip, _ in self.recent_ips)),
                "global_attack_score": round(self.attack_score, 3),
                "requests_per_minute": len(
                    [t for t in self.global_request_rate if t > current_time - 60]
                ),
            }
