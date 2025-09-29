import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional


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
    def from_dict(cls, data: Dict[str, Any]) -> "RequestFingerprint":
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
    def from_dict(cls, data: Dict[str, Any]) -> "AttackPattern":
        """Create from dictionary."""
        return cls(**data)
