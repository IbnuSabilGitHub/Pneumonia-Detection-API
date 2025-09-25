import logging
import time
from typing import Any, Dict, Optional, Tuple

from app.core.rate_limiting.detection import AttackDetector
from app.core.rate_limiting.fingerprint import FingerprintManager
from app.core.rate_limiting.manager import RateLimitManager
from app.core.settings import Settings
from app.core.storage_backends import StorageBackend
from app.core.storage_factory import StorageFactory, StorageType

logger = logging.getLogger(__name__)


class AdvancedRateLimiter:
    """
    Advanced rate limiter with IP switching attack detection and Redis storage.
    Now composed of specialized components for better maintainability.
    """

    def __init__(
        self,
        storage_backend: Optional[StorageBackend] = None,
        storage_config: Optional[Dict[str, Any]] = None,
        settings: Optional[Settings] = None,
    ):
        # Storage backend
        self.storage: Optional[StorageBackend] = storage_backend
        self.storage_config = storage_config or {}

        # Settings configuration
        self.settings = settings
        if self.settings:
            self.rate_limiting_config = self.settings.get_rate_limiting_config()
        else:
            # Fallback configuration
            self.rate_limiting_config = {
                "max_requests_per_ip": 10,
                "max_fingerprint_requests": 3,
                "window_size": 60,
                "attack_block_duration": 300,
                "global_attack_threshold": 0.8,
            }

        # Component initialization (will be done after storage is ready)
        self.attack_detector: Optional[AttackDetector] = None
        self.fingerprint_manager: Optional[FingerprintManager] = None
        self.rate_limit_manager: Optional[RateLimitManager] = None

        # Configuration from centralized settings
        self.max_requests_per_ip = self.rate_limiting_config.get(
            "max_requests_per_ip", 10
        )
        self.max_fingerprint_requests = self.rate_limiting_config.get(
            "max_fingerprint_requests", 3
        )
        self.window_size = self.rate_limiting_config.get("window_size", 60)
        self.attack_block_duration = self.rate_limiting_config.get(
            "attack_block_duration", 300
        )
        self.global_attack_threshold = self.rate_limiting_config.get(
            "global_attack_threshold", 0.8
        )

        # Storage initialization flag
        self._storage_initialized = False
        self._components_initialized = False

    async def initialize_storage(
        self,
        storage_type: StorageType = StorageType.MEMORY,
        config: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Initialize storage backend and components."""
        try:
            if self.storage is None:
                config = config or self.storage_config
                self.storage = await StorageFactory.create_storage(storage_type, config)

            # Test storage connection
            if await self.storage.ping():
                self._storage_initialized = True
                logger.info("Storage backend initialized: %s", storage_type.value)

                # Initialize components with storage
                await self._initialize_components()
                return True
            else:
                logger.error("Storage backend ping failed")
                return False

        except Exception as e:
            logger.error("Failed to initialize storage backend: %s", e)
            return False

    async def _initialize_components(self):
        """Initialize all component classes with the storage backend and centralized config."""
        self.attack_detector = AttackDetector(
            storage=self.storage, config=self.rate_limiting_config
        )
        self.fingerprint_manager = FingerprintManager(
            storage=self.storage, config=self.rate_limiting_config
        )
        self.rate_limit_manager = RateLimitManager(
            storage=self.storage, config=self.rate_limiting_config
        )
        self._components_initialized = True
        logger.info(
            "All rate limiting components initialized with centralized configuration"
        )

    def _ensure_components_initialized(self):
        """Ensure components are initialized with fallback to no-storage mode."""
        if not self._components_initialized:
            self.attack_detector = AttackDetector(
                storage=self.storage, config=self.rate_limiting_config
            )
            self.fingerprint_manager = FingerprintManager(
                storage=self.storage, config=self.rate_limiting_config
            )
            self.rate_limit_manager = RateLimitManager(
                storage=self.storage, config=self.rate_limiting_config
            )
            self._components_initialized = True

    async def is_request_allowed(
        self, client_ip: str, endpoint: str, request, file_hash: Optional[str] = None
    ) -> Tuple[bool, str, Dict]:
        """
        Advanced rate limiting check with multiple protection layers.
        Returns: (is_allowed, reason, details)
        """
        self._ensure_components_initialized()
        current_time = time.time()

        # Create request fingerprint
        fingerprint = self.fingerprint_manager.create_request_fingerprint(request)

        # Store fingerprint for this IP
        fingerprint_obj = self.fingerprint_manager.create_detailed_fingerprint(request)
        await self.fingerprint_manager.store_fingerprint_for_ip(
            client_ip, fingerprint_obj
        )

        # Layer 1: Check if fingerprint is blocked
        if await self.fingerprint_manager.is_fingerprint_blocked(fingerprint):
            fingerprint_info = await self.fingerprint_manager.get_fingerprint_info(
                fingerprint
            )
            return (
                False,
                "Fingerprint blocked due to suspicious activity",
                {
                    "fingerprint": fingerprint,
                    "unblock_time": fingerprint_info.get("blocked_until"),
                    "remaining_time": fingerprint_info.get("remaining_block_time"),
                },
            )

        # Layer 2: Traditional IP rate limiting
        ip_allowed, ip_count = await self.rate_limit_manager.check_ip_rate_limit(
            client_ip
        )
        if not ip_allowed:
            return (
                False,
                "IP rate limit exceeded",
                {"ip": client_ip, "requests_in_window": ip_count},
            )

        # Layer 3: Fingerprint rate limiting
        (
            fingerprint_allowed,
            fingerprint_count,
        ) = await self.rate_limit_manager.check_fingerprint_rate_limit(fingerprint)
        if not fingerprint_allowed:
            # Block this fingerprint temporarily
            await self.fingerprint_manager.block_fingerprint(
                fingerprint, self.attack_block_duration
            )
            return (
                False,
                "Fingerprint rate limit exceeded",
                {
                    "fingerprint": fingerprint,
                    "requests_in_window": fingerprint_count,
                },
            )

        # Layer 4: IP switching attack detection
        if await self.attack_detector.detect_ip_switching_attack_async(
            client_ip, fingerprint
        ):
            await self.fingerprint_manager.block_fingerprint(
                fingerprint, self.attack_block_duration
            )
            return (
                False,
                "IP switching attack detected",
                {"fingerprint": fingerprint, "attack_type": "ip_switching"},
            )

        # Layer 5: Behavioral anomaly detection
        if await self.attack_detector.detect_behavioral_anomalies_async(
            client_ip, endpoint, file_hash
        ):
            await self.fingerprint_manager.block_fingerprint(
                fingerprint, self.attack_block_duration
            )
            return (
                False,
                "Behavioral anomaly detected",
                {"fingerprint": fingerprint, "attack_pattern": "behavioral_anomaly"},
            )

        # Layer 6: Global attack score analysis
        attack_score = await self.attack_detector.calculate_global_attack_score_async()
        if attack_score > self.global_attack_threshold:
            # Apply reduced rate limits during high attack periods
            (
                reduced_allowed,
                reduced_details,
            ) = await self.rate_limit_manager.apply_reduced_limits(
                client_ip, fingerprint
            )

            if not reduced_allowed:
                return (
                    False,
                    "High attack score - reduced limits applied",
                    {"attack_score": attack_score, **reduced_details},
                )

        # Request is allowed
        return (
            True,
            "Request allowed",
            {
                "ip": client_ip,
                "fingerprint": fingerprint,
                "attack_score": attack_score,
                "requests_in_window": ip_count,
                "fingerprint_requests": fingerprint_count,
            },
        )

    async def get_security_status_async(self) -> Dict:
        """Get current security status and statistics."""
        self._ensure_components_initialized()

        # Gather metrics from all components
        attack_metrics = await self.attack_detector.get_security_metrics()
        fingerprint_metrics = self.fingerprint_manager.get_fingerprint_metrics()
        rate_limit_metrics = await self.rate_limit_manager.get_metrics()

        # Combine all metrics
        status = {
            "storage_backend": rate_limit_metrics.get("storage_type", "unknown"),
            "storage_healthy": rate_limit_metrics.get("storage_healthy", False),
            "components_initialized": self._components_initialized,
            "protection_layers": [
                "IP Rate Limiting",
                "Fingerprint Rate Limiting",
                "IP Switching Detection",
                "Behavioral Analysis",
                "Global Attack Detection",
            ],
            **attack_metrics,
            **fingerprint_metrics,
            "rate_limiting": rate_limit_metrics.get("configuration", {}),
        }

        if self.storage and self._storage_initialized:
            storage_info = await self.storage.get_info()
            status["storage_info"] = storage_info

        return status

    def get_security_status(self) -> Dict:
        """Get current security status and statistics (sync fallback)."""
        self._ensure_components_initialized()

        fingerprint_metrics = self.fingerprint_manager.get_fingerprint_metrics()
        rate_limit_config = self.rate_limit_manager.get_configuration()

        return {
            "storage_backend": rate_limit_config.get("storage_backend", "none"),
            "storage_healthy": False,  # Fallback mode
            "components_initialized": self._components_initialized,
            "protection_layers": [
                "IP Rate Limiting",
                "Fingerprint Rate Limiting",
                "IP Switching Detection",
                "Behavioral Analysis",
                "Global Attack Detection",
            ],
            **fingerprint_metrics,
            "rate_limiting": rate_limit_config,
        }

    # Legacy methods for backward compatibility
    def create_request_fingerprint(self, request) -> str:
        """Legacy method - now delegates to FingerprintManager."""
        self._ensure_components_initialized()
        return self.fingerprint_manager.create_request_fingerprint(request)

    def detect_ip_switching_attack(self, client_ip: str, fingerprint: str) -> bool:
        """Legacy method - now delegates to AttackDetector."""
        self._ensure_components_initialized()
        return self.attack_detector.detect_ip_switching_attack(client_ip, fingerprint)

    def detect_behavioral_anomalies(
        self, client_ip: str, endpoint: str, file_hash: Optional[str] = None
    ) -> bool:
        """Legacy method - now delegates to AttackDetector."""
        self._ensure_components_initialized()
        return self.attack_detector.detect_behavioral_anomalies(
            client_ip, endpoint, file_hash
        )

    def calculate_global_attack_score(self) -> float:
        """Legacy method - now delegates to AttackDetector."""
        self._ensure_components_initialized()
        return self.attack_detector.calculate_global_attack_score()
