"""
Template untuk Metadata Class
Ganti EndpointName dengan nama endpoint yang sesuai (PascalCase)
Contoh: HealthMetadata, PredictionMetadata, SecurityStatsMetadata
"""
from typing import Dict, Any

class EndpointNameMetadata:  # Ganti EndpointName dengan nama yang sesuai
    """Metadata for [endpoint description] endpoint."""
    
    @staticmethod
    def get_title() -> str:
        return "<h2>🎯 [Endpoint Title]</h2>"  # Ganti dengan judul yang sesuai
    
    @staticmethod
    def get_description() -> str:
        return """<p>[Main description of what this endpoint does].</p>"""

    @staticmethod
    def get_features() -> str:
        return """
<h3>✨ <strong>Key Features</strong></h3>

<ul>
<li><strong>Feature 1</strong>: Description of feature 1</li>
<li><strong>Feature 2</strong>: Description of feature 2</li>
<li><strong>Feature 3</strong>: Description of feature 3</li>
<li><strong>Feature 4</strong>: Description of feature 4</li>
</ul>
"""

    @staticmethod
    def get_technical_details() -> str:
        return """
<h3>🔧 <strong>Technical Details</strong></h3>

<ul>
<li><strong>Response Format</strong>: JSON with structured data</li>
<li><strong>Authentication</strong>: Required/Not required</li>
<li><strong>Rate Limiting</strong>: Applied/Not applied</li>
<li><strong>Caching</strong>: Enabled/Disabled</li>
</ul>
"""

    @staticmethod
    def get_use_cases() -> str:
        return """
<h3>🎯 <strong>Use Cases</strong></h3>

<ul>
<li><strong>Use Case 1</strong>: Description of when to use this</li>
<li><strong>Use Case 2</strong>: Another use case scenario</li>
<li><strong>Use Case 3</strong>: Third use case example</li>
<li><strong>Integration</strong>: How to integrate with other systems</li>
</ul>
"""

    @staticmethod
    def get_performance() -> str:
        return """
<h3>⚡ <strong>Performance</strong></h3>

<ul>
<li><strong>Response Time</strong>: &lt; 200ms typical</li>
<li><strong>Rate Limiting</strong>: [Rate limit details if applicable]</li>
<li><strong>Caching</strong>: [Caching strategy if applicable]</li>
<li><strong>Scalability</strong>: Designed for high concurrency</li>
</ul>
"""

    @staticmethod
    def get_examples() -> str:
        return """
<h3>📝 <strong>Usage Examples</strong></h3>

<ul>
<li><strong>Basic Usage</strong>: Simple request/response example</li>
<li><strong>Advanced Usage</strong>: Complex scenario example</li>
<li><strong>Error Handling</strong>: How to handle common errors</li>
<li><strong>Integration</strong>: Example integration code</li>
</ul>
"""

    @staticmethod
    def get_200_response() -> Dict[str, Any]:
        """Get 200 response description."""
        return {
            "description": "[Success response description]",
            "content": {
                "application/json": {
                    "example": {
                        # Sesuaikan dengan response model yang dibuat
                        "status": "success",
                        "timestamp": "2025-09-11T10:30:00.000Z",
                        "data": {
                            "field1": "value1",
                            "field2": "value2"
                        },
                        "message": "Operation completed successfully"
                    }
                }
            }
        }
        
    @staticmethod
    def get_400_response() -> Dict[str, Any]:
        """Get 400 response description."""
        return {
            "description": "Bad request - invalid input parameters",
            "content": {
                "application/json": {
                    "example": {
                        "error": "Invalid input parameters",
                        "error_code": "INVALID_INPUT",
                        "timestamp": "2025-09-11T10:30:00.000Z",
                        "details": {
                            "field": "validation error message"
                        }
                    }
                }
            }
        }

    @staticmethod
    def get_404_response() -> Dict[str, Any]:
        """Get 404 response description."""
        return {
            "description": "Resource not found",
            "content": {
                "application/json": {
                    "example": {
                        "error": "Resource not found",
                        "error_code": "NOT_FOUND",
                        "timestamp": "2025-09-11T10:30:00.000Z"
                    }
                }
            }
        }
        
    @staticmethod
    def get_500_response() -> Dict[str, Any]:
        """Get 500 response description."""
        return {
            "description": "[Error response description]",
            "content": {
                "application/json": {
                    "example": {
                        "error": "Internal server error",
                        "error_code": "INTERNAL_ERROR",
                        "timestamp": "2025-09-11T10:30:00.000Z",
                        "details": {}
                    }
                }
            }
        }

    @classmethod
    def get_full_description(cls) -> str:
        """Get complete description by combining all sections."""
        sections = [
            cls.get_title(),
            cls.get_description(),
            cls.get_features(),
            cls.get_technical_details(),
            cls.get_use_cases(),
            cls.get_performance(),
            cls.get_examples()
        ]
        
        return "".join(sections)
    
    @classmethod
    def get_responses(cls) -> Dict[int, Dict[str, Any]]:
        """Get all response descriptions."""
        # Sesuaikan response codes dengan endpoint yang dibuat
        return {
            200: cls.get_200_response(),
            400: cls.get_400_response(),  # Tambahkan jika endpoint menerima input
            404: cls.get_404_response(),  # Tambahkan jika endpoint mencari resource
            500: cls.get_500_response()
        }
        
    @classmethod
    def get_metadata(cls) -> Dict[str, Any]:
        """Get complete metadata for FastAPI endpoint configuration."""
        return {
            "summary": "🎯 [Endpoint Summary]",  # Ganti dengan summary yang sesuai
            "description": cls.get_full_description(),
            "response_description": "[Brief response description]",  # Ganti dengan deskripsi singkat
            "responses": cls.get_responses(),
            "operation_id": "[endpoint_operation_id]",  # Ganti dengan operation ID yang unik
            "response_model_exclude_unset": True,
            "response_model_exclude_none": True
        }


# Template untuk metadata endpoint yang lebih sederhana
class SimpleEndpointNameMetadata:  # Ganti EndpointName dengan nama yang sesuai
    """Simplified metadata for basic endpoints."""
    
    @staticmethod
    def get_description() -> str:
        return """
<h2>🎯 [Simple Endpoint Title]</h2>
<p>[Brief description of what this endpoint does].</p>

<h3>📋 <strong>Key Information</strong></h3>
<ul>
<li><strong>Purpose</strong>: [Main purpose]</li>
<li><strong>Response</strong>: [What it returns]</li>
<li><strong>Performance</strong>: [Performance characteristics]</li>
</ul>
"""

    @classmethod
    def get_metadata(cls) -> Dict[str, Any]:
        """Get simplified metadata for basic endpoints."""
        return {
            "summary": "🎯 [Simple Summary]",
            "description": cls.get_description(),
            "response_description": "[Simple response description]",
            "operation_id": "[simple_operation_id]"
        }


# Template untuk metadata endpoint dengan file upload
class FileUploadEndpointNameMetadata:  # Ganti EndpointName dengan nama yang sesuai
    """Metadata for endpoints that handle file uploads."""
    
    @staticmethod
    def get_file_requirements() -> str:
        return """
<h3>📁 <strong>File Requirements</strong></h3>

<ul>
<li><strong>File Types</strong>: JPG, JPEG, PNG (images only)</li>
<li><strong>File Size</strong>: Maximum 10MB per file</li>
<li><strong>Image Requirements</strong>: Minimum 224x224 pixels</li>
<li><strong>Upload Method</strong>: Multipart form data</li>
</ul>
"""

    @staticmethod
    def get_processing_info() -> str:
        return """
<h3>⚙️ <strong>Processing Information</strong></h3>

<ul>
<li><strong>Processing Time</strong>: 1-3 seconds typical</li>
<li><strong>Image Preprocessing</strong>: Automatic resize and normalization</li>
<li><strong>Validation</strong>: File type, size, and content validation</li>
<li><strong>Security</strong>: Virus scanning and content verification</li>
</ul>
"""

    @classmethod
    def get_full_description(cls) -> str:
        """Get description including file upload specifics."""
        return f"""
<h2>🎯 [File Upload Endpoint Title]</h2>
<p>[Description of file upload endpoint].</p>

{cls.get_file_requirements()}
{cls.get_processing_info()}
"""

    @staticmethod
    def get_413_response() -> Dict[str, Any]:
        """Get 413 response for file too large."""
        return {
            "description": "File too large",
            "content": {
                "application/json": {
                    "example": {
                        "error": "File size exceeds limit of 10.0 MB",
                        "error_code": "FILE_TOO_LARGE",
                        "timestamp": "2025-09-11T10:30:00.000Z"
                    }
                }
            }
        }

    @staticmethod
    def get_422_response() -> Dict[str, Any]:
        """Get 422 response for validation errors."""
        return {
            "description": "Validation error",
            "content": {
                "application/json": {
                    "example": {
                        "error": "Invalid file type. Only JPG, JPEG, PNG are allowed",
                        "error_code": "INVALID_FILE_TYPE",
                        "timestamp": "2025-09-11T10:30:00.000Z"
                    }
                }
            }
        }

    @classmethod
    def get_responses(cls) -> Dict[int, Dict[str, Any]]:
        """Get responses including file upload specific errors."""
        return {
            200: {
                "description": "File processed successfully",
                "content": {
                    "application/json": {
                        "example": {
                            "status": "success",
                            "result": "processed_data",
                            "timestamp": "2025-09-11T10:30:00.000Z"
                        }
                    }
                }
            },
            413: cls.get_413_response(),
            422: cls.get_422_response(),
            500: {
                "description": "Processing error",
                "content": {
                    "application/json": {
                        "example": {
                            "error": "File processing failed",
                            "error_code": "PROCESSING_ERROR",
                            "timestamp": "2025-09-11T10:30:00.000Z"
                        }
                    }
                }
            }
        }
