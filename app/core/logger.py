"""
Logging configuration and utilities.
"""

import logging
import sys
from typing import Optional

from .settings import settings


def setup_logging(
    level: Optional[str] = None, format_string: Optional[str] = None
) -> logging.Logger:
    """
    Setup application logging configuration.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_string: Custom log format string

    Returns:
        Configured logger instance
    """
    if not settings.log_enabled:
        logging.disable(logging.CRITICAL)
        logger = logging.getLogger(settings.app_name)
        logger.handlers.clear()
        logger.addHandler(logging.NullHandler())
        return logger

    logging.disable(logging.NOTSET)

    log_level = level or settings.log_level
    log_format = format_string or settings.log_format

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )

    # Create application logger
    logger = logging.getLogger(settings.app_name)

    # Set third-party loggers to WARNING to reduce noise
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("onnxruntime").setLevel(logging.WARNING)

    # Optional: unify uvicorn logging into our handler / format
    if getattr(settings, "log_unify_uvicorn", False):
        for name in ["uvicorn", "uvicorn.error", "uvicorn.access"]:
            lg = logging.getLogger(name)
            # Clear existing handlers (uvicorn sets its own)
            lg.handlers.clear()
            # Let them bubble to our root/basicConfig handlers
            lg.propagate = True
        logger.debug("Uvicorn logging unified under application logger format")

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a specific module.

    Args:
        name: Logger name (usually __name__)

    Returns:
        Logger instance
    """
    return logging.getLogger(name)


# Global logger instance
logger = setup_logging()
