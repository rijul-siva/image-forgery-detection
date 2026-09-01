"""
Geometric verification of candidate matches using RANSAC.

We use `cv2.estimateAffinePartial2D`, which estimates a similarity transform
(rotation + uniform scale + translation) — exactly the model that
copy-move forgery on a single planar image produces.

A similarity transform has 4 degrees of freedom; RANSAC needs at minimum
2 correspondences per iteration. We require at least `min_match_count`
matches before running RANSAC, and return empty results below that
threshold so the UI can show "no forgery detected" gracefully.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .feature_matching import MatchPair


@dataclass
class GeometricResult:
    """Output of RANSAC geometric verification."""
    affine_matrix: Optional[np.ndarray]   # 2x3, dtype float64, or None
    inlier_src: np.ndarray                # (N, 2) float64 src points
    inlier_dst: np.ndarray                # (N, 2) float64 dst points
    outlier_src: np.ndarray               # (N, 2) float64
    outlier_dst: np.ndarray               # (N, 2) float64
    num_inliers: int
    num_outliers: int
    inlier_mask: Optional[np.ndarray]     # bool array of length len(matches)


def verify_geometric(
    matches: List[MatchPair],
    ransac_threshold: float = 3.0,
    min_match_count: int = 10,
    max_iters: int = 2000,
    confidence: float = 0.995,
) -> GeometricResult:
    """
    Run RANSAC to find the dominant affine (similarity) transform.

    Parameters
    ----------
    matches : list[MatchPair]
        Output of `match_features`.
    ransac_threshold : float
        Maximum reprojection error in pixels to count a match as inlier.
    min_match_count : int
        Minimum number of matches required to attempt RANSAC.
    max_iters : int
        RANSAC iteration cap.
    confidence : float
        RANSAC confidence; higher = more iterations but more robust.

    Returns
    -------
    GeometricResult
        The estimated affine matrix (or None), the split of matches into
        inliers / outliers, and the inlier mask.
    """
    empty = GeometricResult(
        affine_matrix=None,
        inlier_src=np.empty((0, 2), dtype=np.float64),
        inlier_dst=np.empty((0, 2), dtype=np.float64),
        outlier_src=np.empty((0, 2), dtype=np.float64),
        outlier_dst=np.empty((0, 2), dtype=np.float64),
        num_inliers=0,
        num_outliers=0,
        inlier_mask=None,
    )

    if len(matches) < min_match_count:
        return empty

    src = np.array([m.src_xy for m in matches], dtype=np.float64)
    dst = np.array([m.dst_xy for m in matches], dtype=np.float64)

    # estimateAffinePartial2D -> similarity (rotation + uniform scale + tx, ty)
    # Returns: (M, inliers) where M is 2x3 and inliers is a bool mask (None
    # if RANSAC is disabled). We pass method=cv2.LMEDS as a fallback wouldn't
    # help here, so we explicitly request RANSAC.
    try:
        M, inliers = cv2.estimateAffinePartial2D(
            src, dst,
            method=cv2.RANSAC,
            ransacReprojThreshold=ransac_threshold,
            maxIters=max_iters,
            confidence=confidence,
        )
    except cv2.error:
        return empty

    if M is None or inliers is None:
        return empty

    inliers_bool = inliers.ravel().astype(bool)
    outlier_bool = ~inliers_bool

    inlier_src = src[inliers_bool]
    inlier_dst = dst[inliers_bool]
    outlier_src = src[outlier_bool]
    outlier_dst = dst[outlier_bool]

    num_inliers = int(inliers_bool.sum())
    if num_inliers < min_match_count // 2:
        # Not enough consensus — treat as failure to avoid bogus transforms.
        return GeometricResult(
            affine_matrix=None,
            inlier_src=np.empty((0, 2), dtype=np.float64),
            inlier_dst=np.empty((0, 2), dtype=np.float64),
            outlier_src=src,
            outlier_dst=dst,
            num_inliers=0,
            num_outliers=len(matches),
            inlier_mask=None,
        )

    return GeometricResult(
        affine_matrix=M.astype(np.float64),
        inlier_src=inlier_src,
        inlier_dst=inlier_dst,
        outlier_src=outlier_src,
        outlier_dst=outlier_dst,
        num_inliers=num_inliers,
        num_outliers=int(outlier_bool.sum()),
        inlier_mask=inliers_bool,
    )
