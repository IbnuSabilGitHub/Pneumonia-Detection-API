from typing import Dict, Any

class StatusMetadata:
    """Metadata for documentations status endpoint."""
    
    @staticmethod
    def get_description() -> str:
        """Get security status description."""
        return """
<p>Provides real-time status of the multi-layer security system protecting the API
from various attacks and abuse patterns.</p>
"""

    @staticmethod
    def get_protection_features_monitored() -> str:
        """Get protection features monitored section."""
        return """
<h3>🛡️ <strong>Protection Features Monitored</strong></h3>

<ul>
<li><strong>Multi-layer Rate Limiting</strong>: IP-based and global request limits</li>
<li><strong>Attack Detection</strong>: Sophisticated pattern recognition</li>
<li><strong>Request Fingerprinting</strong>: Unique request identification</li>
<li><strong>Behavioral Analysis</strong>: User behavior pattern analysis</li>
<li><strong>IP Switching Detection</strong>: Rapid IP change detection</li>
<li><strong>Duplicate Prevention</strong>: File and request deduplication</li>
<li><strong>Global Attack Scoring</strong>: Overall threat level assessment</li>
</ul>
"""
    
    @staticmethod
    def get_status_information() -> str:
        """Get status information section."""
        return """
<h3>📊 <strong>Status Information</strong></h3>

<ul>
<li><strong>System Status</strong>: Overall security system health</li>
<li><strong>Active Protections</strong>: List of enabled security features</li>
<li><strong>Threat Level</strong>: Current global attack probability</li>
<li><strong>Storage Backend</strong>: In-memory or Redis storage status</li>
<li><strong>Performance Metrics</strong>: Security system performance data</li>
</ul>
"""
    
    @staticmethod
    def get_use_cases() -> str:
        """Get use cases section."""
        return """
<h3>🎯 <strong>Use Cases</strong></h3>

<ul>
<li><strong>Security Monitoring</strong>: Real-time protection status</li>
<li><strong>Threat Assessment</strong>: Current security posture</li>
<li><strong>System Health</strong>: Security component validation</li>
<li><strong>Compliance</strong>: Security audit and reporting</li>
</ul>
"""

    @staticmethod
    def get_performance() -> str:
        """Get performance section."""
        return """
<h3>⚡ <strong>Performance</strong></h3>

<ul>
<li><strong>Response Time</strong>: &lt; 100ms typical</li>
<li><strong>Rate Limiting</strong>: No limits applied to security endpoints</li>
<li><strong>Real-time Data</strong>: Live security metrics</li>
</ul>
"""
    @staticmethod
    def get_200_response() -> Dict[str, Any]:
        """Get 200 response description."""
        return {
            "description": "Security status retrieved successfully",
            "content": {
                "application/json": {
                    "example": {
                        "service": "Pneumonia Detection API",
                        "security_status": "active",
                        "timestamp": "2025-09-05T10:30:00.000Z",
                        "advanced_protection": {
                            "global_attack_score": 0.15,
                            "requests_per_minute": 23,
                            "recent_unique_ips": 8,
                            "blocked_fingerprints": 2,
                            "storage_type": "memory"
                        },
                        "protection_features": [
                            "Multi-layer Rate Limiting (In-Memory)",
                            "IP Switching Attack Detection (In-Memory)",
                            "Request Fingerprinting (In-Memory)",
                            "Behavioral Analysis (In-Memory)",
                            "Global Attack Scoring (In-Memory)",
                            "Duplicate File Detection (In-Memory)"
                        ]
                    }
                }
            }
        }
        
    def get_500_response() -> Dict[str, Any]:
        """Get 500 response description."""
        return {
            "description": "Security system error",
            "content": {
                "application/json": {
                    "example": {
                        "service": "Pneumonia Detection API",
                        "security_status": "not_initialized",
                        "timestamp": "2025-09-05T10:30:00.000Z",
                        "error": "Rate limiter not initialized"
                    }
                }
            }
        }

    @classmethod
    def get_full_description(cls) -> str:
        """Get complete API description by combining all sections."""
        sections = [
        "<h2>🔒 Advanced Security Protection Status</h2>",
        cls.get_description(),
        cls.get_protection_features_monitored(),
        cls.get_status_information(),
        cls.get_use_cases(),
        cls.get_performance()
        ]
    
        return "".join(sections)
    
    @classmethod
    def get_responses(cls) -> Dict[int, Dict[str, Any]]:
        """Get complete responses dictionary."""
        return {
            200: cls.get_200_response(),
            500: cls.get_500_response()
        }
        
    @classmethod
    def get_metadata(cls) -> Dict[str, Any]:
        """Get complete metadata for FastAPI endpoint configuration."""
        return {
            "summary": "Security System Status",
            "description": cls.get_full_description(),
            "response_description": "Current security system status and active protections",
            "responses": cls.get_responses(),
            "operation_id": "get_security_status",
            "response_model_exclude_unset": True,
            "response_model_exclude_none": True
        }