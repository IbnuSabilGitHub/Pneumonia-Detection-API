from typing import Any, Dict

from ...models.error_codes import ErrorCode
from ...models.security_schemes import SecurityStatsErrorResponse, SecurityStatsResponse
from ..base_builder import build_response


class StatMetadata:
    """Metadata for documentations security statistics endpoint."""

    @staticmethod
    def get_title() -> str:
        return "<h2>📈 Advanced Security Metrics & Analytics</h2>"

    @staticmethod
    def get_description() -> str:
        return """<p>Provides comprehensive statistical analysis of security events, threat patterns,
and protection effectiveness over time.</p>
"""

    @staticmethod
    def get_detailed_metrics() -> str:
        return """
<h3>📊 <strong>Detailed Metrics</strong></h3>

<ul>
<li><strong>Attack Score Analysis</strong>: Global threat probability with interpretation</li>
<li><strong>Request Rate Metrics</strong>: Real-time request frequency analysis</li>
<li><strong>IP Activity Tracking</strong>: Unique IP address patterns and behavior</li>
<li><strong>Blocking Statistics</strong>: Currently blocked requests and fingerprints</li>
<li><strong>Performance Metrics</strong>: Security system response times</li>
<li><strong>Storage Utilization</strong>: Memory/Redis usage statistics</li>
</ul>
"""

    @staticmethod
    def get_threat_level_interpretation() -> str:
        return """
<h3>🎯 <strong>Threat Level Interpretation</strong></h3>

<ul>
<li><strong>LOW</strong> (0.0-0.3): Normal traffic patterns, minimal threats</li>
<li><strong>MEDIUM</strong> (0.3-0.7): Elevated activity, possible probing</li>
<li><strong>HIGH</strong> (0.7-1.0): Active attacks detected, enhanced protection</li>
</ul>    
"""

    @staticmethod
    def get_statistical_categories() -> str:
        return """
<h3>📋 <strong>Statistical Categories</strong></h3>

<ul>
<li><strong>Real-time Metrics</strong>: Current minute activity</li>
<li><strong>Pattern Analysis</strong>: Behavioral pattern recognition</li>
<li><strong>Threat Assessment</strong>: Attack probability calculations</li>
<li><strong>Performance Data</strong>: System efficiency metrics</li>
</ul>
"""

    @staticmethod
    def get_use_cases() -> str:
        return """
<h3>🔍 <strong>Use Cases</strong></h3>

<ul>
<li><strong>Security Analytics</strong>: Deep dive into protection metrics</li>
<li><strong>Threat Intelligence</strong>: Attack pattern analysis</li>
<li><strong>Performance Tuning</strong>: Security system optimization</li>
<li><strong>Incident Response</strong>: Detailed attack investigation</li>
<li><strong>Compliance Reporting</strong>: Security audit documentation</li>
</ul>
"""

    @staticmethod
    def get_data_freshness() -> str:
        return """
<h3>⚡ <strong>Data Freshness</strong></h3>

<ul>
<li><strong>Update Frequency</strong>: Real-time (&lt; 1 second delay)</li>
<li><strong>Historical Data</strong>: Last 24 hours of activity</li>
<li><strong>Aggregation</strong>: Per-minute statistical summaries</li>
</ul>
"""

    @staticmethod
    def get_response_description() -> str:
        return """
Comprehensive security statistics and threat analysis
"""

    @staticmethod
    def get_200_response() -> Dict[str, Any]:
        """Get 200 response description."""
        return build_response(
            description="Security statistics retrieved successfully",
            model=SecurityStatsResponse,
            example={
                "service": "Pneumonia Detection API",
                "timestamp": "2025-09-05T10:30:00.000Z",
                "security_metrics": {
                    "global_attack_score": 0.25,
                    "requests_per_minute": 45,
                    "recent_unique_ips": 12,
                    "blocked_fingerprints": 3,
                    "storage_type": "memory",
                    "avg_response_time_ms": 85,
                    "total_requests_24h": 2847,
                },
                "status": "active",
                "storage_type": "memory",
                "interpretation": {
                    "attack_score": {
                        "value": 0.25,
                        "level": "LOW",
                        "description": "Global attack probability score (0.0-1.0)",
                    },
                    "request_rate": {
                        "value": 45,
                        "description": "Total requests in the last minute",
                    },
                    "unique_ips": {
                        "value": 12,
                        "description": "Number of unique IP addresses in recent activity",
                    },
                    "blocked_count": {
                        "value": 3,
                        "description": "Number of currently blocked request fingerprints",
                    },
                },
                "analytics_summary": {
                    "daily_total": 2847,
                    "threat_level": "LOW",
                    "storage_backend": "memory",
                },
            },
        )

    @staticmethod
    def get_500_response() -> Dict[str, Any]:
        """Get 500 response description."""
        return build_response(
            description="Security statistics retrieval error",
            model=SecurityStatsErrorResponse,
            example={
                "error": "Failed to retrieve security statistics",
                "error_code": ErrorCode.SECURITY_STATS_RETRIEVAL_ERROR,
                "timestamp": "2025-09-10T10:30:00.000Z",
                "details": {
                    "component": "security_analytics",
                    "operation": "get_security_status",
                },
            },
        )

    @classmethod
    def get_full_description(cls) -> str:
        """Get complete statistics description by combining all sections."""
        sections = [
            cls.get_title(),
            cls.get_description(),
            cls.get_detailed_metrics(),
            cls.get_threat_level_interpretation(),
            cls.get_statistical_categories(),
            cls.get_use_cases(),
            cls.get_data_freshness(),
        ]

        return "".join(sections)

    @classmethod
    def get_responses(cls) -> Dict[int, Dict[str, Any]]:
        """Get all response descriptions."""
        return {200: cls.get_200_response(), 500: cls.get_500_response()}

    @classmethod
    def get_metadata(cls) -> Dict[str, Any]:
        """Get complete metadata for FastAPI endpoint configuration."""
        return {
            "summary": "Security Analytics & Statistics",
            "description": cls.get_full_description(),
            "response_description": cls.get_response_description(),
            "responses": cls.get_responses(),
        }
