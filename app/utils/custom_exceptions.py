"""
Custom exceptions for the Pneumonia Detection API.
"""

from typing import Any, Dict, Optional


class PneumoniaAPIException(Exception):
    """
    Base exception for all API-related errors.

    Provides structured error information with status codes and error codes.
    """

    def __init__(
        self,
        message: str,
        error_code: str,
        status_code: int = 500,
        details: Optional[Dict[str, Any]] = None,
    ):
        self.message = message
        self.error_code = error_code
        self.status_code = status_code
        self.details = details or {}
        super().__init__(self.message)

    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for JSON response."""
        return {
            "detail": self.message,
            "error_code": self.error_code,
            "details": self.details,
        }


class ServiceInitializationError(PneumoniaAPIException):
    """
    Raised when a service fails to initialize during startup.
    """

    def __init__(
        self, service_name: str, reason: str, original_error: Optional[Exception] = None
    ):
        details = {"service": service_name}
        if original_error:
            details["original_error"] = str(original_error)
            details["error_type"] = type(original_error).__name__

        super().__init__(
            f"Failed to initialize {service_name}: {reason}",
            "SERVICE_INIT_FAILED",
            500,
            details,
        )
        self.service_name = service_name
        self.original_error = original_error


class ModelLoadError(PneumoniaAPIException):
    """
    Raised when AI model fails to load.
    """

    def __init__(self, model_path: str, reason: str):
        super().__init__(
            f"Failed to load model from {model_path}: {reason}",
            "MODEL_LOAD_FAILED",
            503,
            {"model_path": model_path},
        )
        self.model_path = model_path


class RateLimitError(PneumoniaAPIException):
    """
    Raised when rate limit is exceeded.
    """

    def __init__(self, retry_after: int, limit_type: str = "general"):
        super().__init__(
            f"Rate limit exceeded. Try again in {retry_after} seconds",
            "RATE_LIMIT_EXCEEDED",
            429,
            {"retry_after": retry_after, "limit_type": limit_type},
        )
        self.retry_after = retry_after
        self.limit_type = limit_type


class FileValidationError(PneumoniaAPIException):
    """
    Raised when file validation fails.
    """

    def __init__(self, reason: str, file_info: Optional[Dict[str, Any]] = None):
        super().__init__(
            f"File validation failed: {reason}",
            "FILE_VALIDATION_FAILED",
            400,
            file_info or {},
        )


class InvalidFileFormatError(FileValidationError):
    """
    Raised when file format is not supported.
    """

    def __init__(self, file_type: str, supported_types: list):
        super().__init__(
            f"Unsupported file format: {file_type}. Supported formats: {', '.join(supported_types)}",
            {"file_type": file_type, "supported_types": supported_types},
        )


class FileTooLargeError(FileValidationError):
    """
    Raised when file is too large.
    """

    def __init__(self, file_size: int, max_size: int):
        super().__init__(
            f"File too large: {file_size} bytes. Maximum allowed: {max_size} bytes",
            {"file_size": file_size, "max_size": max_size},
        )


class InvalidImageContentError(FileValidationError):
    """
    Raised when image content is invalid or corrupted.
    """

    def __init__(self, reason: str):
        super().__init__(
            f"Invalid image content: {reason}", {"validation_type": "image_content"}
        )


class DuplicateFileError(PneumoniaAPIException):
    """
    Raised when a duplicate file is detected.
    """

    def __init__(self, file_hash: str):
        super().__init__(
            "Duplicate file detected. This image has been processed recently",
            "DUPLICATE_FILE",
            409,
            {"file_hash": file_hash},
        )
        self.file_hash = file_hash


class StorageBackendError(PneumoniaAPIException):
    """
    Raised when storage backend operations fail.
    """

    def __init__(self, operation: str, backend_type: str, reason: str):
        super().__init__(
            f"Storage operation '{operation}' failed on {backend_type} backend: {reason}",
            "STORAGE_BACKEND_ERROR",
            503,
            {"operation": operation, "backend_type": backend_type},
        )
        self.operation = operation
        self.backend_type = backend_type


class ConfigurationError(PneumoniaAPIException):
    """
    Raised when configuration is invalid.
    """

    def __init__(self, config_key: str, reason: str, provided_value: Any = None):
        details = {"config_key": config_key}
        if provided_value is not None:
            details["provided_value"] = str(provided_value)

        super().__init__(
            f"Invalid configuration for '{config_key}': {reason}",
            "CONFIGURATION_ERROR",
            500,
            details,
        )
        self.config_key = config_key


class SecurityError(PneumoniaAPIException):
    """
    Raised when security violations are detected.
    """

    def __init__(
        self,
        violation_type: str,
        reason: str,
        client_info: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            f"Security violation ({violation_type}): {reason}",
            "SECURITY_VIOLATION",
            403,
            client_info or {},
        )
        self.violation_type = violation_type


class AttackDetectedError(SecurityError):
    """
    Raised when an attack pattern is detected.
    """

    def __init__(
        self, attack_type: str, client_ip: str, details: Optional[Dict[str, Any]] = None
    ):
        client_info = {"client_ip": client_ip, "attack_type": attack_type}
        if details:
            client_info.update(details)

        super().__init__(
            "attack_pattern",
            f"{attack_type} attack detected from {client_ip}",
            client_info,
        )
        self.attack_type = attack_type
        self.client_ip = client_ip
