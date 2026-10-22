"""
Input validation helpers (shared).
"""
from __future__ import annotations

import os

import numpy as np


SUPPORTED_DETECTORS = ("orb", "sift")


def validate_image_path(image_path: str) -> None:
    """
    Raise ValueError if `image_path` is empty, does not exist, or is not a file.

    The message matches the one used across the pipeline ("Could not load
    image at path: ...") so callers and tests see consistent errors.
    """
    if not image_path or not os.path.isfile(image_path):
        raise ValueError(f"Could not load image at path: {image_path}")


def validate_detector(detector: str) -> str:
    """Normalise and validate a detector name ('orb' or 'sift')."""
    name = str(detector).lower().strip()
    if name not in SUPPORTED_DETECTORS:
        raise ValueError(
            f"Unknown detector '{detector}'. Use one of {SUPPORTED_DETECTORS}."
        )
    return name


def ensure_gray_uint8(image: np.ndarray) -> np.ndarray:
    """
    Validate a 2-D grayscale image and coerce it to uint8.

    Raises ValueError if the input is not a 2-D array.
    """
    if image is None:
        raise ValueError("image is None")
    if image.ndim != 2:
        raise ValueError(f"image must be 2-D, got shape {image.shape}")
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return image
