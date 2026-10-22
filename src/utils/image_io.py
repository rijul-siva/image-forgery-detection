"""
Image loading and saving helpers (shared).

All modules read images through this module so that EXIF orientation is
normalised consistently and errors are reported the same way.
"""
from __future__ import annotations

import os
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageOps

from .validation import validate_image_path


def load_image_bgr(image_path: str, apply_exif_orientation: bool = True) -> np.ndarray:
    """
    Load an image as a BGR uint8 numpy array.

    Uses Pillow so EXIF orientation tags are honoured (a photo stored
    landscape with an orientation flag is rotated upright), then converts
    RGB -> BGR for OpenCV consumers. Falls back to `cv2.imread` if Pillow
    cannot decode the file, and raises a clear ValueError if neither can.

    Raises
    ------
    ValueError
        If the path is missing/unreadable or the file is not an image.
    """
    validate_image_path(image_path)

    image: Optional[np.ndarray] = None
    try:
        with Image.open(image_path) as pil_image:
            if apply_exif_orientation:
                pil_image = ImageOps.exif_transpose(pil_image)
            rgb = np.asarray(pil_image.convert("RGB"))
        image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    except Exception:
        image = None

    if image is None:
        image = cv2.imread(image_path)

    if image is None:
        raise ValueError(f"Could not load image at path: {image_path}")
    return image


def save_image(image_path: str, image: np.ndarray) -> bool:
    """Write a BGR image to disk. Returns True on success."""
    return bool(cv2.imwrite(image_path, image))
