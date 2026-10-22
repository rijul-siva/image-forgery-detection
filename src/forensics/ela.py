"""
Error Level Analysis (ELA) module — owned by Member B.

Recompresses the image as JPEG at a known quality and measures the
localized difference from the original. A naive global max/mean threshold
flags almost every authentic photo (sharp edges always produce large DCT
errors), so suspicion is scored *statistically*:

  - With a candidate region mask (from copy-move detection) we compare the
    mean error inside the region to the surrounding background and express
    the difference in background standard deviations (a z-score).
  - Without a mask we compare the highest-error image block against the
    global error distribution, looking for a spatially localized anomaly.

Lossless inputs (PNG/BMP) are noted in the description: recompression
errors are uniform across the canvas, so ELA is not informative about
double-JPEG history there.
"""

from __future__ import annotations
import os

import cv2
import numpy as np
from PIL import Image

from src.utils.data_types import ELAResult


_MIN_REGION_PIXELS = 50


def _detect_format(image_path: str) -> str:
    """Return the container format reported by Pillow (fallback: extension)."""
    try:
        with Image.open(image_path) as img:
            if img.format:
                return str(img.format).upper()
    except Exception:
        pass
    ext = os.path.splitext(image_path)[1].lstrip(".").upper()
    return ext or "UNKNOWN"


def _classify(z: float, k: float) -> str:
    """Map a z-score to a suspicion level."""
    if z >= k:
        return "high"
    if z >= k / 2.0:
        return "medium"
    return "low"


def _block_max_z(diff_scalar: np.ndarray, grid: int = 8) -> float:
    """
    z-score of the highest-error block relative to the global distribution.

    The image is divided into an `grid` x `grid` lattice. For each block we
    take the mean error; the maximum block mean is compared to the global
    mean error in units of the global standard deviation.
    """
    h, w = diff_scalar.shape[:2]
    bh, bw = max(1, h // grid), max(1, w // grid)
    means = []
    for y in range(0, h - bh + 1, bh):
        for x in range(0, w - bw + 1, bw):
            means.append(float(diff_scalar[y:y + bh, x:x + bw].mean()))
    if not means:
        return 0.0
    mu = float(diff_scalar.mean())
    sigma = float(diff_scalar.std())
    return (max(means) - mu) / max(sigma, 1e-6)


def run_ela_analysis(
    image_path: str,
    quality: int = 95,
    amplification: int = 20,
    region_mask: np.ndarray | None = None,
    k: float = 3.0,
) -> ELAResult:
    """
    Perform Error Level Analysis (ELA) on an image file.

    Parameters
    ----------
    image_path : str
        Path to the original image file (analyzed at native resolution,
        not the resized preprocessing canvas).
    quality : int, optional
        Target JPEG re-compression quality level (default 95).
    amplification : int, optional
        Multiplier for error level difference amplification (default 20).
    region_mask : np.ndarray | None
        Optional binary mask of candidate forged regions, in any resolution;
        it is resized to the image. When given, suspicion is scored by the
        region-vs-background error difference (recommended).
    k : float
        z-score threshold for "high" suspicion (default 3.0). "medium" uses
        half of this value.

    Returns
    -------
    ELAResult
        Dataclass containing the ELA heatmap, suspicion level, and a
        human-readable statistical description.
    """
    original = cv2.imread(image_path)
    if original is None:
        raise ValueError(f"Could not load image at path: {image_path}")

    fmt = _detect_format(image_path)

    # Re-compress the image in memory at target quality.
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    success, encoded_image = cv2.imencode('.jpg', original, encode_param)
    if not success:
        raise ValueError("Failed to encode image for ELA analysis.")

    compressed = cv2.imdecode(encoded_image, cv2.IMREAD_COLOR)

    # Pixel-wise difference, kept as a colour heatmap for display.
    diff = cv2.absdiff(original, compressed).astype(np.float32)
    ela_image = np.clip(diff * float(amplification), 0, 255).astype(np.uint8)

    # Scalar error map for the statistics (mean across BGR channels).
    diff_scalar = diff.mean(axis=2) if diff.ndim == 3 else diff

    notes = []
    if fmt in {"PNG", "BMP", "TIFF", "WEBP"}:
        notes.append(
            f"Lossless input ({fmt}): recompression errors are uniform, so "
            "ELA cannot reveal double-JPEG history."
        )

    if region_mask is not None:
        mask = np.asarray(region_mask)
        if mask.shape[:2] != original.shape[:2]:
            mask = cv2.resize(
                mask,
                (original.shape[1], original.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        inside = mask > 0
        if int(inside.sum()) >= _MIN_REGION_PIXELS and int((~inside).sum()) >= _MIN_REGION_PIXELS:
            mu_region = float(diff_scalar[inside].mean())
            mu_bg = float(diff_scalar[~inside].mean())
            sigma_bg = float(diff_scalar[~inside].std())
            z = abs(mu_region - mu_bg) / max(sigma_bg, 1e-6)
            suspicion_level = _classify(z, k)
            desc = (
                f"Region-vs-background ELA: region mean error {mu_region:.2f} vs "
                f"background {mu_bg:.2f} (background sigma {sigma_bg:.2f}, "
                f"z = {z:.2f}; thresholds: medium >= {k / 2:.2f}, high >= {k:.2f})."
            )
        else:
            # Mask too small to be statistically meaningful — fall back.
            z = _block_max_z(diff_scalar)
            suspicion_level = _classify(z, k * 1.5)
            desc = (
                f"Region mask too small for a reliable comparison; using "
                f"localized block analysis instead (max-block z = {z:.2f})."
            )
    else:
        z = _block_max_z(diff_scalar)
        # Block analysis is noisier, so require a stronger anomaly.
        suspicion_level = _classify(z, k * 1.5)
        desc = (
            f"Localized block analysis: highest-error block is {z:.2f} standard "
            "deviations above the global mean error "
            f"(global mean {float(diff_scalar.mean()):.2f}, "
            f"sigma {float(diff_scalar.std()):.2f})."
        )

    if notes:
        desc = f"{desc} {' '.join(notes)}"

    # Raw difference (unamplified)
    raw_diff = np.clip(diff, 0, 255).astype(np.uint8)

    # Pseudocolor thermal heatmap representation (Jet colormap)
    ela_gray = cv2.cvtColor(ela_image, cv2.COLOR_BGR2GRAY)
    colormap_image = cv2.applyColorMap(ela_gray, cv2.COLORMAP_JET)

    mean_diff = float(np.mean(diff))
    max_diff = float(np.max(diff))

    return ELAResult(
        ela_image=ela_image,
        suspicion_level=suspicion_level,
        description=desc,
        compressed_image=compressed,
        raw_diff=raw_diff,
        colormap_image=colormap_image,
        mean_diff=mean_diff,
        max_diff=max_diff,
    )
