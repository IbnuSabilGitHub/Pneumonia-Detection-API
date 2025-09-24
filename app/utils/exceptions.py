"""
Custom exceptions for the application.
"""


class PneumoniaDetectionError(Exception):
    """Base exception for pneumonia detection application."""


class ModelLoadError(PneumoniaDetectionError):
    """Exception raised when model loading fails."""


class PredictionError(PneumoniaDetectionError):
    """Exception raised when prediction fails."""


class ValidationError(PneumoniaDetectionError):
    """Exception raised when input validation fails."""


class RateLimitError(PneumoniaDetectionError):
    """Exception raised when rate limit is exceeded."""


class FileValidationError(ValidationError):
    """Exception raised when file validation fails."""


class ImageValidationError(ValidationError):
    """Exception raised when image validation fails."""
