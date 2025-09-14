from typing import Dict, Any


class ModelInfoMetadata:
    """Metadata for Model Information Endpoint."""
    
    @staticmethod
    def get_title() -> str:
        return "<h2>🤖 Machine Learning Model Details</h2>"
    
    @staticmethod
    def get_model_description() -> str:
        return """
<p>Provides comprehensive information about the currently loaded AI model including
architecture details, performance metrics, and training statistics.</p>
"""

    @staticmethod
    def get_information_included() -> str:
        return """
<h3>📋 <strong>Information Included</strong></h3>

<ul>
<li><strong>Model Architecture</strong>: Network structure and layer details</li>
<li><strong>Training Metrics</strong>: Accuracy, precision, recall, F1-score</li>
<li><strong>Dataset Info</strong>: Training data characteristics</li>
<li><strong>Performance Stats</strong>: Inference time and optimization details</li>
<li><strong>Version Info</strong>: Model version and build information</li>
</ul>
"""

    @staticmethod
    def get_use_case() -> str:
        return """
<h3>🎯 <strong>Use Cases</strong></h3>

<ul>
<li><strong>Model Validation</strong>: Verify correct model loading</li>
<li><strong>Performance Analysis</strong>: Review model capabilities</li>
<li><strong>Integration Planning</strong>: Understand model characteristics</li>
<li><strong>Debugging</strong>: Troubleshoot model-related issues</li>
</ul>
"""

    @staticmethod
    def get_model_comparison() -> str:
        return """
<h3>📊 <strong>Model Comparison</strong></h3>

<p>Use the <code>model</code> query parameter to get information about different models:
- <strong>Standard</strong>: Faster inference, good baseline performance
- <strong>EfficientNet-B0</strong>: Higher accuracy, advanced architecture</p>
"""

    @staticmethod
    def get_performance() -> str:
        return """
<h3>⚡ <strong>Performance</strong></h3>

<ul>
<li><strong>Response Time</strong>: &lt; 50ms typical</li>
<li><strong>Rate Limiting</strong>: No limits applied</li>
<li><strong>Caching</strong>: Model info cached for performance</li>
</ul>
"""

    @staticmethod
    def get_response_description() -> str:
        return "Detailed model information and statistics"

    @staticmethod
    def get_200_response() -> Dict[str, Any]:
        return {
            "description": "Model information retrieved successfully",
            "content": {
                "application/json": {
                    "schema": {
                        "$ref": "#/components/schemas/ModelInfoResponse"
                    },
                    "examples": {
                        "model_loaded": {
                            "summary": "Model successfully loaded",
                            "value": {
                                "loaded": True,
                                "model_path": "models/pneumonia_model_efficientnet_b0.onnx",
                                "input_name": "input",
                                "output_name": "output",
                                "mean": 0.480,
                                "std": 0.237,
                                "target_size": [192, 192],
                                "labels": ["NORMAL", "PNEUMONIA"],
                                "model_type": "efficientnet_b0"
                            }
                        },
                        "model_not_loaded": {
                            "summary": "Model not loaded",
                            "value": {
                                "loaded": False
                            }
                        }
                    }
                }
            }
        }
        
    @staticmethod
    def get_404_response() -> Dict[str, Any]:
        return {
            "description": "Model not found",
            "content": {
                "application/json": {
                    "schema": {
                        "$ref": "#/components/schemas/ModelInfoNotFoundResponse"
                    },
                    "example": {
                        "detail": "Model 'invalid_model' not found",
                        "error_code": "MODEL_NOT_FOUND",
                        "available_models": ["standard", "efficientnet_b0"],
                        "timestamp": "2025-09-13T10:30:00.000Z"
                    }
                }
            }
        }
        
    @staticmethod
    def get_503_response() -> Dict[str, Any]:
        return {
            "description": "Prediction service unavailable",
            "content": {
                "application/json": {
                    "schema": {
                        "$ref": "#/components/schemas/ModelInfoServiceUnavailableResponse"
                    },
                    "example": {
                        "detail": "Prediction service not available",
                        "error_code": "SERVICE_UNAVAILABLE",
                        "service_status": "not_initialized",
                        "timestamp": "2025-09-13T10:30:00.000Z"
                    }
                }
            }
        }
        
    @staticmethod
    def get_422_response() -> Dict[str, Any]:
        return {
            "description": "Validation Error",
            "content": {
                "application/json": {
                    "schema": {
                        "$ref": "#/components/schemas/ModelInfoValidationErrorResponse"
                    },
                    "example": {
                        "detail": [
                            {
                                "loc": ["query", "model"],
                                "msg": "Invalid model type specified",
                                "type": "value_error"
                            }
                        ]
                    }
                }
            }
        }
        
    @classmethod
    def get_full_description(cls) -> str:
        """Get complete API description by combining all sections."""
        sections = [
            cls.get_title(),
            cls.get_model_description(),
            cls.get_information_included(),
            cls.get_use_case(),
            cls.get_model_comparison(),
            cls.get_performance()
        ]
        return "\n".join(sections)
    
    @classmethod
    def get_responses(cls) -> Dict[int, Dict[str, Any]]:
        return {
            200: cls.get_200_response(),
            404: cls.get_404_response(),
            422: cls.get_422_response(),
            503: cls.get_503_response()
        }
        
    @classmethod
    def get_metadata(cls) -> Dict[str, Any]:
        """Get complete metadata for FastAPI endpoint configuration."""
        return {
            "summary": "Comprehensive Model Information",
            "description": cls.get_full_description(),
            "response_description": cls.get_response_description(),
            "responses": cls.get_responses()
        }