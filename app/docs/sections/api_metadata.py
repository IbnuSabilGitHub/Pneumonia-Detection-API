from typing import Any, Dict

from ...core.settings import settings


class ApiMetadata:
    """Manages API documentation and metadata."""

    @staticmethod
    def get_medical_disclaimer() -> str:
        """Get medical disclaimer section."""
        return """
<h3>⚠️ <strong>Important Medical Disclaimer</strong></h3>
<p><strong>This API is designed for educational and research purposes only.</strong></p>
<p>The predictions provided by this system should <strong>NEVER</strong> be used as a substitute for professional medical diagnosis or treatment. Always consult qualified healthcare professionals for medical advice, diagnosis, or treatment decisions.</p>
"""

    @staticmethod
    def get_key_features() -> str:
        """Get key features section."""
        return """
<h3>🚀 <strong>Key Features</strong></h3>
<ul>
<li><strong>🤖 AI-Powered Detection</strong>: Advanced deep learning models for pneumonia detection</li>
<li><strong>📊 Confidence Scoring</strong>: Detailed probability distributions and confidence levels</li>
<li><strong>💡 Smart Recommendations</strong>: Medical recommendations based on AI predictions</li>
<li><strong>🔒 JWT Security</strong>: Native Supabase JWT authentication with per-user rate limiting</li>
<li><strong>✅ Input Validation</strong>: Comprehensive file and image validation</li>
<li><strong>📈 Real-time Monitoring</strong>: Request logging and performance tracking</li>
</ul>
"""

    @staticmethod
    def get_available_models() -> str:
        """Get available models section."""
        return """
<h3>🔧 <strong>Available Models</strong></h3>
<ol>
<li><strong>Standard Model</strong> (<code>standard</code>): Baseline CNN architecture</li>
<li><strong>EfficientNet-B0</strong> (<code>efficientnet_b0</code>): Advanced transfer learning model</li>
</ol>
"""

    @staticmethod
    def get_security_features() -> str:
        """Get security features section."""
        return """
<h3>🛡️ <strong>Security Features</strong></h3>
<ul>
<li><strong>JWT Authentication</strong>: Supabase JWT (always enabled) for all endpoints</li>
<li><strong>User Rate Limiting</strong>: 100 requests per hour per authenticated user</li>
<li><strong>File Validation</strong>: Size (max 10MB) and type (JPG, JPEG, PNG) validation</li>
<li><strong>Content Analysis</strong>: AI-powered image content validation</li>
<li><strong>Duplicate Detection</strong>: Prevents repeated uploads of identical images</li>
<li><strong>Request Monitoring</strong>: Comprehensive logging and tracking</li>
</ul>
"""

    @staticmethod
    def get_supported_formats() -> str:
        """Get supported file formats section."""
        return """
<h3>📋 <strong>Supported File Formats</strong></h3>
<ul>
<li><strong>JPEG/JPG</strong>: Recommended for medical images</li>
<li><strong>PNG</strong>: High quality lossless format</li>
<li><strong>Maximum Size</strong>: 10MB per file</li>
</ul>
"""

    @staticmethod
    def get_usage_examples() -> str:
        """Get API usage examples section."""
        return """
<h3>🎯 <strong>API Usage Examples</strong></h3>
<h4>Basic Prediction Request:</h4>
<pre><code>curl -X POST "http://localhost:8000/pneumonia/predict" \\
     -H "Content-Type: multipart/form-data" \\
     -F "file=@chest_xray.jpg"</code></pre>

<h4>Using Specific Model:</h4>
<pre><code>curl -X POST "http://localhost:8000/pneumonia/predict?model=efficientnet_b0" \\
     -H "Content-Type: multipart/form-data" \\
     -F "file=@chest_xray.jpg"</code></pre>
"""

    @staticmethod
    def get_response_format() -> str:
        """Get response format section."""
        return """
<h3>📊 <strong>Response Format</strong></h3>
<p>All prediction responses include:</p>
<ul>
<li><strong>Prediction</strong>: NORMAL or PNEUMONIA classification</li>
<li><strong>Confidence</strong>: Numerical confidence score (0.0-1.0)</li>
<li><strong>Probabilities</strong>: Individual class probabilities</li>
<li><strong>Medical Recommendation</strong>: Contextual medical guidance</li>
<li><strong>Model Information</strong>: Version and type used for prediction</li>
</ul>
"""

    @staticmethod
    def get_monitoring_endpoints() -> str:
        """Get monitoring endpoints section."""
        return """
<h3>🔍 <strong>Monitoring Endpoints</strong></h3>
<ul>
<li><strong>Health Check</strong>: <code>/</code> or <code>/health</code> - Service status and uptime</li>
<li><strong>Model Info</strong>: <code>/pneumonia/model/info</code> - Detailed model information</li>
</ul>
"""

    @staticmethod
    def get_documentation_links() -> str:
        """Get documentation links section."""
        return """
<h3>📚 <strong>Documentation</strong></h3>
<ul>
<li><strong>Interactive API Docs</strong>: <code>/docs</code> (Swagger UI)</li>
<li><strong>Alternative Docs</strong>: <code>/redoc</code> (ReDoc)</li>
</ul>
"""

    @classmethod
    def get_full_description(cls) -> str:
        """Get complete API description by combining all sections."""
        sections = [
            "<h2>🏥 Medical AI API for Chest X-ray Pneumonia Detection</h2>",
            cls.get_medical_disclaimer(),
            "<hr>",
            cls.get_key_features(),
            cls.get_available_models(),
            cls.get_security_features(),
            cls.get_supported_formats(),
            cls.get_usage_examples(),
            cls.get_response_format(),
            cls.get_monitoring_endpoints(),
            cls.get_documentation_links(),
            "<hr>",
            "<p><strong>Built with FastAPI</strong> | <strong>Powered by ONNX</strong></p>",
        ]

        return "".join(sections)

    @classmethod
    def get_openapi_tags(cls) -> list:
        """Get OpenAPI tags configuration."""
        return [
            {
                "name": "Health",
                "description": "Health check and monitoring endpoints for service status",
            },
            {
                "name": "Pneumonia Detection",
                "description": "AI-powered pneumonia detection from chest X-ray images",
            },
            {
                "name": "Model",
                "description": "Machine learning model information and statistics",
            },
        ]

    @classmethod
    def get_app_metadata(cls) -> Dict[str, Any]:
        """Get complete app metadata for FastAPI configuration."""
        return {
            "title": settings.app_name,
            "description": cls.get_full_description(),
            "version": settings.app_version,
            "docs_url": "/docs",
            "redoc_url": "/redoc",
            "debug": settings.debug,
            "contact": {
                "name": "Pneumonia Detection API",
                "url": "https://github.com/IbnuSabilGitHub/Pneumonia-Detection-API",
            },
            "license_info": {
                "name": "MIT License",
                "url": "https://opensource.org/licenses/MIT",
            },
            "servers": [
                {"url": "http://localhost:8000", "description": "Development server"},
                {
                    "url": "https://web-production-d9c43a.up.railway.app/",
                    "description": "Production server",
                },
            ],
            "openapi_tags": cls.get_openapi_tags(),
        }
