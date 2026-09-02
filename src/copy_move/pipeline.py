"""
Copy-move detection pipeline — the single public entry point for Member B's UI.

Usage (from the orchestrator / Streamlit app):

    from src.copy_move.pipeline import run_copy_move_analysis
    result = run_copy_move_analysis(gray_image, color_image, detector="orb")
    # result is a `CopyMoveResult` matching the contract in data_types.py.

Stages (matches `zed.md` §5, Steps 4-11):
  1. extract_features()       (Step 4)
  2. match_features()         (Step 5, 6)
  3. verify_geometric()       (Step 7)
  4. decompose_affine()       (Step 8)
  5. localize_forgery()       (Step 9, 10, 11)
  6. Visualization overlays   (used by the UI as well)

If any stage fails (too few features, RANSAC no consensus, etc.) we
return a `CopyMoveResult(forgery_detected=False, ...)` rather than raising,
so the UI can always display a result.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from src.utils.data_types import (
    CopyMoveResult,
    RegionInfo,
    TransformParams,
)

from .feature_extraction import extract_features
from .feature_matching import match_features
from .geometric_verification import verify_geometric
from .localization import centroid_distance, localize_forgery
from .parameters import decompose_affine
from .visualization import draw_matches, draw_regions, draw_transformation_arrow


# ---- Tunable defaults (overridable per-call) -----------------------------
DEFAULTS = {
    "detector": "orb",
    "ratio": 0.75,
    "spatial_distance_min": 10.0,
    "ransac_threshold": 3.0,
    "min_match_count": 10,
    "max_iters": 2000,
    "confidence": 0.995,
    "forgery_min_inliers": 8,
    "forgery_min_inlier_ratio": 0.08,
}


def _empty_result(
    color_image: np.ndarray,
    num_total_matches: int = 0,
) -> CopyMoveResult:
    return CopyMoveResult(
        forgery_detected=False,
        num_total_matches=num_total_matches,
        num_inlier_matches=0,
        confidence_score=0.0,
        inlier_keypoints_src=[],
        inlier_keypoints_dst=[],
        affine_matrix=None,
        transform_params=None,
        regions=[],
        mask=None,
        match_visualization=color_image.copy(),
        region_visualization=color_image.copy(),
    )


def run_copy_move_analysis(
    gray_image: np.ndarray,
    color_image: np.ndarray,
    detector: str = DEFAULTS["detector"],
    ratio: float = DEFAULTS["ratio"],
    spatial_distance_min: float = DEFAULTS["spatial_distance_min"],
    ransac_threshold: float = DEFAULTS["ransac_threshold"],
    min_match_count: int = DEFAULTS["min_match_count"],
    max_iters: int = DEFAULTS["max_iters"],
    confidence: float = DEFAULTS["confidence"],
    forgery_min_inliers: int = DEFAULTS["forgery_min_inliers"],
    forgery_min_inlier_ratio: float = DEFAULTS["forgery_min_inlier_ratio"],
) -> CopyMoveResult:
    """
    Run the full copy-move detection pipeline on preprocessed images.

    Parameters
    ----------
    gray_image : np.ndarray
        Grayscale version of the image (from `PreprocessResult.gray_image`).
    color_image : np.ndarray
        BGR color version (from `PreprocessResult.color_image`), used for
        visualizations. Not modified.

    Returns
    -------
    CopyMoveResult
        Matches the dataclass contract in `src/utils/data_types.py`.
    """
    if gray_image is None or color_image is None:
        raise ValueError("Both gray_image and color_image must be provided.")

    image_shape = gray_image.shape  # (H, W)

    # --- Step 4: feature extraction ---------------------------------------
    try:
        features = extract_features(gray_image, detector=detector)
    except Exception:
        return _empty_result(color_image)

    if features.descriptors is None or len(features.keypoints) < min_match_count:
        return _empty_result(color_image)

    # --- Steps 5-6: matching + filtering ----------------------------------
    matches = match_features(
        features,
        ratio=ratio,
        spatial_distance_min=spatial_distance_min,
    )
    num_total_matches = len(matches)
    if num_total_matches == 0:
        return _empty_result(color_image)

    # --- Step 7: RANSAC geometric verification ---------------------------
    geom = verify_geometric(
        matches,
        ransac_threshold=ransac_threshold,
        min_match_count=min_match_count,
        max_iters=max_iters,
        confidence=confidence,
    )

    # Visualization with whatever matches we have, plus the inlier mask.
    inlier_mask = geom.inlier_mask
    match_vis = draw_matches(color_image, matches, inlier_mask=inlier_mask)

    if geom.affine_matrix is None:
        return CopyMoveResult(
            forgery_detected=False,
            num_total_matches=num_total_matches,
            num_inlier_matches=0,
            confidence_score=0.0,
            inlier_keypoints_src=[],
            inlier_keypoints_dst=[],
            affine_matrix=None,
            transform_params=None,
            regions=[],
            mask=None,
            match_visualization=match_vis,
            region_visualization=color_image.copy(),
        )

    # --- Step 8: affine decomposition ------------------------------------
    transform_params: Optional[TransformParams] = decompose_affine(geom.affine_matrix)

    # Decide "forgery_detected". Two thresholds both must pass:
    #   (a) at least `forgery_min_inliers` RANSAC inliers
    #   (b) inlier/total ratio at least `forgery_min_inlier_ratio`
    num_inliers = geom.num_inliers
    inlier_ratio = (num_inliers / num_total_matches) if num_total_matches else 0.0
    forgery_detected = (
        num_inliers >= forgery_min_inliers
        and inlier_ratio >= forgery_min_inlier_ratio
    )

    # Confidence is the inlier ratio (experimental score).
    confidence_score = float(inlier_ratio)

    # --- Steps 9-11: localization & region metrics -----------------------
    loc = localize_forgery(geom.inlier_src, geom.inlier_dst, image_shape)
    region_infos: list[RegionInfo] = [r.info for r in loc.regions]
    region_vis = draw_regions(color_image, loc.regions)
    if len(loc.regions) >= 2:
        # Draw the arrow only when we have a clear pair.
        region_vis = draw_transformation_arrow(region_vis, loc.regions)

    # Convert numpy arrays to plain (x, y) tuples for the dataclass contract.
    src_pts = [(float(x), float(y)) for x, y in geom.inlier_src]
    dst_pts = [(float(x), float(y)) for x, y in geom.inlier_dst]

    return CopyMoveResult(
        forgery_detected=bool(forgery_detected),
        num_total_matches=int(num_total_matches),
        num_inlier_matches=int(num_inliers),
        confidence_score=confidence_score,
        inlier_keypoints_src=src_pts,
        inlier_keypoints_dst=dst_pts,
        affine_matrix=geom.affine_matrix,
        transform_params=transform_params,
        regions=region_infos,
        mask=loc.combined_mask if loc.combined_mask.max() > 0 else None,
        match_visualization=match_vis,
        region_visualization=region_vis,
    )


__all__ = ["run_copy_move_analysis", "DEFAULTS"]
