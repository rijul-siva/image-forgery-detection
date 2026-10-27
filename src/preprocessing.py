"""
Image preprocessing module — owned by Member B.
Prepares uploaded images for downstream analysis across all forensic modules.
"""

from __future__ import annotations
import cv2
import numpy as np
from src.utils.data_types import PreprocessResult
from src.utils.image_io import load_image_bgr

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
    denoise: bool = False,
    denoise_strength: int = 12,
    enhance_clahe: bool = False,
    clahe_clip_limit: float = 2.0,
    clahe_tile_grid_size: int = 8
) -> PreprocessResult:
    """
    Load, validate, resize (preserving aspect ratio), convert to grayscale, 
    and optionally apply colored non-local means denoising and CLAHE contrast enhancement.

    Parameters
    ----------
    image_path : str
        Path to the uploaded image file.
    max_edge : int, optional
        Maximum pixel length for the longest edge (default 1600).
    denoise : bool, optional
        Whether to use the denoised image for downstream grayscale analysis (default False).
    denoise_strength : int, optional
        Non-local means noise reduction strength. Higher values remove more noise but
        can also smooth fine detail (default 12).
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
        denoised_image, and clahe_image.
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

    # Keep the resized original for overlays and comparisons. Non-local means
    # removes noise while preserving edges better than the previous mild blur.
    color_image = image
    if denoise_strength < 0:
        raise ValueError("denoise_strength must be non-negative")
    denoised_image = (
        cv2.fastNlMeansDenoisingColored(
            color_image,
            None,
            h=float(denoise_strength),
            hColor=float(denoise_strength),
            templateWindowSize=7,
            searchWindowSize=21,
        )
        if denoise_strength > 0
        else color_image.copy()
    )
    analysis_image = denoised_image if denoise else color_image

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
        denoised_image=denoised_image,
        clahe_image=clahe_image,
        base_gray_image=base_gray,
    )
