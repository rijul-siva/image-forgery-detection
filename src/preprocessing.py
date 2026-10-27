"""
Image preprocessing module — owned by Member B.
Prepares uploaded images for downstream analysis across all forensic modules.
"""

from __future__ import annotations
import cv2
import numpy as np
from src.utils.data_types import PreprocessResult
from src.utils.image_io import load_image_bgr

SPATIAL_FILTERS = {
    "gaussian": "Gaussian",
    "mean": "Mean (arithmetic)",
    "min": "Min",
    "max": "Max",
}


def apply_spatial_filter(
    image: np.ndarray,
    method: str = "gaussian",
    kernel_size: int = 5,
) -> np.ndarray:
    """Apply a taught spatial filter independently to each color channel."""
    method = method.lower()
    if method not in SPATIAL_FILTERS:
        raise ValueError(f"Unknown spatial filter: {method}")
    if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be a positive odd number")

    kernel = (kernel_size, kernel_size)
    if method == "gaussian":
        return cv2.GaussianBlur(image, kernel, sigmaX=0)
    if method == "mean":
        return cv2.blur(image, kernel)

    structuring_element = np.ones(kernel, dtype=np.uint8)
    if method == "min":
        return cv2.erode(image, structuring_element)
    return cv2.dilate(image, structuring_element)


def apply_clahe(
    gray_image: np.ndarray,
    clip_limit: float = 2.0,
    tile_size: int = 8
) -> np.ndarray:
    """
    Apply Contrast Limited Adaptive Histogram Equalization (CLAHE).
    Enhances local contextual contrast while clipping slope to avoid amplifying flat noise.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    return clahe.apply(gray_image)


def preprocess_image(
    image_path: str, 
    max_edge: int = 1600,
    apply_filter: bool = False,
    filter_method: str = "gaussian",
    filter_kernel_size: int = 5,
    enhance_clahe: bool = False,
    clahe_clip_limit: float = 2.0,
    clahe_tile_grid_size: int = 8
) -> PreprocessResult:
    """
    Load, validate, resize (preserving aspect ratio), convert to grayscale, 
    and optionally apply a spatial filter and CLAHE contrast enhancement.

    Parameters
    ----------
    image_path : str
        Path to the uploaded image file.
    max_edge : int, optional
        Maximum pixel length for the longest edge (default 1600).
    apply_filter : bool, optional
        Whether to use the filtered image for downstream grayscale analysis (default False).
    filter_method : str, optional
        One of ``gaussian``, ``mean`` (arithmetic mean), ``min``, or ``max``.
    filter_kernel_size : int, optional
        Odd neighborhood size for the spatial filter (default 5).
    enhance_clahe : bool, optional
        Whether to use CLAHE enhanced grayscale for feature detection (default False).
    clahe_clip_limit : float, optional
        Contrast clipping threshold for CLAHE (default 2.0).
    clahe_tile_grid_size : int, optional
        Grid tile size for local contextual histogram equalization (default 8).

    Returns
    -------
    PreprocessResult
        Dataclass containing color_image, gray_image, scale_factor, original dimensions,
        selected ``filtered_image``, and ``clahe_image``.
    """
    # Load image (EXIF orientation is normalised by the shared loader).
    image = load_image_bgr(image_path)

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

    # Keep the resized original for overlays and comparisons; use the selected
    # spatial filter for the DIP output and, when enabled, detector input.
    color_image = image
    filtered_image = apply_spatial_filter(color_image, filter_method, filter_kernel_size)
    analysis_image = filtered_image if apply_filter else color_image

    # Convert to grayscale for feature detection (0.299 R + 0.587 G + 0.114 B)
    base_gray = cv2.cvtColor(analysis_image, cv2.COLOR_BGR2GRAY)

    # Compute CLAHE enhanced representation (DIP Stage 1 contextual histogram equalization)
    clahe_image = apply_clahe(base_gray, clip_limit=clahe_clip_limit, tile_size=clahe_tile_grid_size)

    gray_image = clahe_image if enhance_clahe else base_gray

    return PreprocessResult(
        color_image=color_image,
        gray_image=gray_image,
        scale_factor=float(scale_factor),
        original_width=int(original_width),
        original_height=int(original_height),
        filtered_image=filtered_image,
        clahe_image=clahe_image,
        base_gray_image=base_gray,
    )
