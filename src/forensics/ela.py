"""
Error Level Analysis (ELA) module — owned by Member B.
Analyzes JPEG compression error inconsistencies across the image.
"""

from __future__ import annotations
import cv2
import numpy as np
from src.utils.data_types import ELAResult

def run_ela_analysis(
    image_path: str,
    quality: int = 95,
    amplification: int = 20
) -> ELAResult:
    """
    Perform Error Level Analysis (ELA) on an image file.

    Parameters
    ----------
    image_path : str
        Path to the original image file.
    quality : int, optional
        Target JPEG re-compression quality level (default 95).
    amplification : int, optional
        Multiplier for error level difference amplification (default 20).

    Returns
    -------
    ELAResult
        Dataclass containing ELA heatmap image, suspicion level, and description.
    """
    original = cv2.imread(image_path)
    if original is None:
        raise ValueError(f"Could not load image at path: {image_path}")

    # Re-compress the image in memory at target quality
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    success, encoded_image = cv2.imencode('.jpg', original, encode_param)
    if not success:
        raise ValueError("Failed to encode image for ELA analysis.")

    compressed = cv2.imdecode(encoded_image, cv2.IMREAD_COLOR)

    # Compute absolute pixel-wise difference between original and re-compressed
    diff = cv2.absdiff(original, compressed).astype(np.float32)

    # Scale and amplify differences for visualization
    ela_image_float = diff * float(amplification)
    ela_image = np.clip(ela_image_float, 0, 255).astype(np.uint8)

    # Quantitative analysis metrics
    max_diff = float(np.max(diff))
    mean_diff = float(np.mean(diff))
    std_diff = float(np.std(diff))

    # Suspicion classification based on error distribution
    if mean_diff > 12.0 or max_diff > 45.0:
        suspicion_level = "high"
        desc = (
            f"High error variation detected (Mean error: {mean_diff:.1f}, Max error: {max_diff:.1f}). "
            f"Indicates potential compression discrepancies or digital editing."
        )
    elif mean_diff > 5.0 or max_diff > 25.0:
        suspicion_level = "medium"
        desc = (
            f"Moderate error levels observed (Mean error: {mean_diff:.1f}, Max error: {max_diff:.1f}). "
            f"Localized edge or texture variances present."
        )
    else:
        suspicion_level = "low"
        desc = (
            f"Low uniform error levels observed (Mean error: {mean_diff:.1f}, Max error: {max_diff:.1f}). "
            f"Consistent with uniform compression history."
        )

    return ELAResult(
        ela_image=ela_image,
        suspicion_level=suspicion_level,
        description=desc
    )
