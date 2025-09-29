from typing import Any, Dict

from ...docs.base_builder import build_response
from ...models.health_schemas import HealthErrorResponse, HealthResponse


class HealthMetadata:
    """Metadata for Health Check Endpoint."""

    @staticmethod
    def get_title() -> str:
        return "<h2>🏥 Medical AI API for Chest X-ray Pneumonia Detection</h2>"

    @staticmethod
    def get_medical_disclaimer() -> str:
        return """
    <p>Provides comprehensive health status information about the Pneumonia Detection API service.</p>
"""

    @staticmethod
    def get_health_status_levels() -> str:
        return """
<h3>📊 <strong>Health Status Levels</strong></h3>

<ul>
<li>healthy: All systems operational, model loaded and ready</li>
<li>partial: Service running but with limitations (e.g., model not loaded)</li>
<li>unhealthy: Critical issues detected</li>
</ul>
"""

    @staticmethod
    def get_response_information() -> str:
        return """
<h3>📋 <strong>Response Information</strong></h3>

<ul>
<li>Status: Current health state of the service</li>
<li>Model Status: Whether AI models are loaded and ready</li>
<li>Version: Current API version</li>
<li>Uptime: Time since service start (in seconds)</li>
</ul>
"""

    @staticmethod
    def get_use_cases() -> str:
        return """
<h3>🔍 <strong>Use Cases</strong></h3>

<ul>
<li>Load Balancer Health Checks: Monitor service availability</li>
<li>Monitoring Systems: Track service uptime and status</li>
<li>Troubleshooting: Verify service and model status</li>
<li>Development: Quick service verification</li>
</ul>
"""

    @staticmethod
    def get_performance() -> str:
        return """
<h3>⚡<strong>Performance</strong></h3>

<ul>
<li>Response Time: < 100ms typical</li>
<li>Rate Limiting: No rate limits applied</li>
<li>Caching: Status computed in real-time</li>
</ul>
"""

    @staticmethod
    def get_response_description() -> str:
        return "Service health status with detailed information"

    @staticmethod
    def get_200_response() -> Dict[str, Any]:
        return build_response(
            description="Successful health status response",
            model=HealthResponse,
            example={
                "status": "healthy",
                "model_loaded": True,
                "version": "3.4.2",
                "uptime": 361.5,
            },
        )

    def get_500_response() -> Dict[str, Any]:
        return build_response(
            description="Internal server error during health check",
            model=HealthErrorResponse,
            example={
                "error": {
                    "code": 500,
                    "message": "Internal server error",
                    "details": "Detailed error information",
                }
            },
        )

    @classmethod
    def get_full_description(cls) -> str:
        """Get complete Health Check description by combining all sections."""
        sections = [
            cls.get_medical_disclaimer(),
            cls.get_health_status_levels(),
            cls.get_response_information(),
            cls.get_use_cases(),
            cls.get_performance(),
        ]

        return "".join(sections)

    @classmethod
    def get_response(cls) -> Dict[int, Dict[str, Any]]:
        """Get all possible response descriptions for the endpoint."""
        return {
            200: cls.get_200_response(),
            500: cls.get_500_response(),
        }

    @classmethod
    def get_metadata(cls) -> Dict[str, Any]:
        """Get complete metadata for FastAPI endpoint configuration."""
        return {
            "summary": "Service Health Check",
            "description": cls.get_full_description(),
            "response_description": cls.get_response_description(),
            "responses": cls.get_response(),
        }
