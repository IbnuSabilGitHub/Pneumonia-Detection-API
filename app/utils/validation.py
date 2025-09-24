"""
Image validation utilities.
"""

from pathlib import Path

import numpy as np
from PIL import Image

from ..core.logger import get_logger
from ..core.settings import settings
from ..utils.exceptions import ImageValidationError

logger = get_logger(__name__)


def validate_file_extension(filename: str) -> bool:
    """
    Validate file extension.

    Args:
        filename: Name of the uploaded file

    Returns:
        True if extension is allowed, False otherwise
    """
    if not filename:
        return False

    file_ext = Path(filename).suffix.lower()
    return file_ext in settings.allowed_extensions


def validate_file_size(contents: bytes) -> bool:
    """
    Validate file size.

    Args:
        contents: File contents as bytes

    Returns:
        True if size is within limit, False otherwise
    """
    return len(contents) <= settings.max_file_size


def validate_image_integrity(contents: bytes) -> Image.Image:
    """
    Validate image file integrity and return PIL Image.

    Args:
        contents: Image file contents as bytes

    Returns:
        PIL Image object

    Raises:
        ImageValidationError: If image is invalid
    """
    try:
        image = Image.open(contents)
        image.verify()  # Verify image integrity

        # Reopen after verification (verify() closes the image)
        contents.seek(0)
        image = Image.open(contents)

        # Convert to RGB if needed
        if image.mode != "RGB":
            image = image.convert("RGB")

        return image

    except (IOError, SyntaxError) as e:
        raise ImageValidationError(f'Invalid image file: {e}') from e


def validate_image_content(image: Image.Image) -> bool:
    """
    Validate if image appears to be a medical X-ray.

    This performs basic heuristic checks to determine if the image
    might be a chest X-ray. These are not foolproof but can help
    filter out obviously inappropriate images.

    Args:
        image: PIL Image object

    Returns:
        True if image passes basic X-ray validation, False otherwise
    """
    try:
        # Convert to grayscale for analysis
        gray_image = image.convert("L")
        img_array = np.array(gray_image)

        height, width = img_array.shape

        # Size validation - X-rays should be reasonably sized
        if height < 100 or width < 100 or height > 2000 or width > 2000:
            logger.warning("Image size validation failed: %dx%d", width, height)
            return False

        # Intensity distribution check
        mean_intensity = np.mean(img_array)
        std_intensity = np.std(img_array)

        # X-rays should have reasonable contrast and not be too dark/bright
        if std_intensity < 20:
            logger.warning("Low contrast detected: std=%s", std_intensity)
            return False

        if mean_intensity < 30 or mean_intensity > 225:
            logger.warning("Unusual brightness detected: mean=%s", mean_intensity)
            return False

        # Check for reasonable intensity distribution
        # X-rays typically have a good spread of intensities
        hist, _ = np.histogram(img_array, bins=50)
        non_zero_bins = np.count_nonzero(hist)

        if non_zero_bins < 10:  # Too few intensity levels
            logger.warning("Poor intensity distribution: %d bins", non_zero_bins)
            return False

        return True

    except (OSError, ValueError, TypeError) as e:
        logger.error("Error validating image content: %s", e)
        return False


def get_image_stats(image: Image.Image) -> dict:
    """
    Get basic statistics about the image.

    Args:
        image: PIL Image object

    Returns:
        Dictionary with image statistics
    """
    gray_image = image.convert("L")
    img_array = np.array(gray_image)

    return {
        "size": image.size,
        "mode": image.mode,
        "mean_intensity": float(np.mean(img_array)),
        "std_intensity": float(np.std(img_array)),
        "min_intensity": int(np.min(img_array)),
        "max_intensity": int(np.max(img_array)),
    }
