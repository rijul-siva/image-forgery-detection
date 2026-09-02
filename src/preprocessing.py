"""
Image preprocessing module — owned by Member B.
Prepares uploaded images for downstream analysis across all forensic modules.
"""

from __future__ import annotations
import cv2
import numpy as np
from src.utils.data_types import PreprocessResult

def preprocess_image(
    image_path: str, 
    max_edge: int = 1600,
    denoise: bool = False
) -> PreprocessResult:
    """
    Load, validate, resize (preserving aspect ratio), convert to grayscale, 
    and optionally apply mild Gaussian noise reduction.

    Parameters
    ----------
    image_path : str
        Path to the uploaded image file.
    max_edge : int, optional
        Maximum pixel length for the longest edge (default 1600).
    denoise : bool, optional
        Whether to apply mild Gaussian denoising (default False).

    Returns
    -------
    PreprocessResult
        Dataclass containing color_image, gray_image, scale_factor, and original dimensions.
    """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image at path: {image_path}")

    # Validate dimensions
    original_height, original_width = image.shape[:2]

    # Calculate resize scale factor
    scale_factor = 1.0
    if max(original_height, original_width) > max_edge:
        if original_width > original_height:
            scale_factor = max_edge / float(original_width)
        else:
            scale_factor = max_edge / float(original_height)
            
        new_width = int(round(original_width * scale_factor))
        new_height = int(round(original_height * scale_factor))
        image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

    # Optional mild Gaussian smoothing to reduce sensor/compression noise
    if denoise:
        image = cv2.GaussianBlur(image, (3, 3), 0.5)

    # Convert to grayscale for feature detection
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    return PreprocessResult(
        color_image=image,
        gray_image=gray_image,
        scale_factor=float(scale_factor),
        original_width=int(original_width),
        original_height=int(original_height)
    )
