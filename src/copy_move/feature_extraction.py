"""
Feature extraction for copy-move forgery detection.

Supports ORB (default, fast, free) and SIFT (more accurate, behind a flag).
Returns keypoints (cv2.KeyPoint list) and descriptors (np.ndarray, shape NxD).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np


DetectorType = str  # "orb" | "sift"


@dataclass
class FeatureSet:
    """Container for the output of feature extraction."""
    keypoints: List[cv2.KeyPoint]
    descriptors: np.ndarray  # shape (N, D); may be None if no keypoints found
    detector_name: str


def _create_detector(detector: DetectorType) -> Tuple[cv2.Feature2D, str]:
    """
    Construct an OpenCV detector.

    Returns the detector instance and the norm type used for matching
    (cv2.NORM_HAMMING for ORB, cv2.NORM_L2 for SIFT).
    """
    detector = detector.lower().strip()
    if detector == "orb":
        # 5000 features is plenty for smartphone photos resized to ~1600px.
        det = cv2.ORB_create(nfeatures=5000, scaleFactor=1.2, nlevels=8)
        return det, "orb", cv2.NORM_HAMMING
    if detector == "sift":
        # SIFT lives in opencv-contrib; raise a clear error if not available.
        if not hasattr(cv2, "SIFT_create"):
            raise RuntimeError(
                "SIFT not available. Install opencv-contrib-python "
                "(`pip install opencv-contrib-python`)."
            )
        det = cv2.SIFT_create(nfeatures=5000, nOctaveLayers=3, contrastThreshold=0.04)
        return det, "sift", cv2.NORM_L2
    raise ValueError(f"Unknown detector '{detector}'. Use 'orb' or 'sift'.")


def extract_features(
    gray_image: np.ndarray,
    detector: DetectorType = "orb",
) -> FeatureSet:
    """
    Detect keypoints and compute descriptors on a grayscale image.

    Parameters
    ----------
    gray_image : np.ndarray
        Single-channel uint8 image (output of preprocessing).
    detector : str
        "orb" (default, free) or "sift" (more accurate, requires opencv-contrib).

    Returns
    -------
    FeatureSet
        Keypoints, descriptors (None if no features found), and the detector name.
    """
    if gray_image is None:
        raise ValueError("gray_image is None")
    if gray_image.ndim != 2:
        raise ValueError(f"gray_image must be 2-D, got shape {gray_image.shape}")
    if gray_image.dtype != np.uint8:
        # OpenCV detectors expect uint8; coerce safely.
        gray_image = np.clip(gray_image, 0, 255).astype(np.uint8)

    det, name, _ = _create_detector(detector)
    keypoints, descriptors = det.detectAndCompute(gray_image, None)

    return FeatureSet(
        keypoints=keypoints if keypoints is not None else [],
        descriptors=descriptors,
        detector_name=name,
    )
