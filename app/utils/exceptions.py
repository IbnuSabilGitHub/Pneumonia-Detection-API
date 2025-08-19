"""
Custom exceptions for the application.
"""


class PneumoniaDetectionError(Exception):
    """Base exception for pneumonia detection application."""
    pass


class ModelLoadError(PneumoniaDetectionError):
    """Exception raised when model loading fails."""
    pass


class PredictionError(PneumoniaDetectionError):
    """Exception raised when prediction fails."""
    pass


class ValidationError(PneumoniaDetectionError):
    """Exception raised when input validation fails."""
    pass


class RateLimitError(PneumoniaDetectionError):
    """Exception raised when rate limit is exceeded."""
    pass


class FileValidationError(ValidationError):
    """Exception raised when file validation fails."""
    pass


class ImageValidationError(ValidationError):
    """Exception raised when image validation fails."""
    pass
