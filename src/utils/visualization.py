"""
Visualization helpers shared between modules and the UI.

Member A's copy-move overlays live in `src.copy_move.visualization`; this
module holds only generic helpers used by the broader application.
"""
from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np


def to_rgb(image_bgr: np.ndarray) -> np.ndarray:
    """Convert a BGR image to RGB (for Streamlit/Pillow/matplotlib display)."""
    if image_bgr is None:
        raise ValueError("image_bgr is None")
    if image_bgr.ndim == 2:
        return cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2RGB)
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def mask_overlay(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    color: Tuple[int, int, int] = (0, 165, 255),
    alpha: float = 0.35,
) -> np.ndarray:
    """
    Blend a semi-transparent colour over the non-zero pixels of `mask`.

    Returns a new BGR image; the input is not modified.
    """
    if image_bgr is None:
        raise ValueError("image_bgr is None")
    if mask is None or mask.size == 0:
        return image_bgr.copy()

    if mask.shape[:2] != image_bgr.shape[:2]:
        mask = cv2.resize(
            mask, (image_bgr.shape[1], image_bgr.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

    out = image_bgr.copy()
    color_layer = np.zeros_like(out)
    color_layer[mask > 0] = color
    return cv2.addWeighted(color_layer, alpha, out, 1.0 - alpha, 0)
