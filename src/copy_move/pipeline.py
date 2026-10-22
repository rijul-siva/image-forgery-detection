"""
Copy-move detection pipeline — the single public entry point for Member B's UI.

Usage (from the orchestrator / Streamlit app):

    from src.copy_move.pipeline import run_copy_move_analysis
    result = run_copy_move_analysis(gray_image, color_image, detector="orb")
    # result is a `CopyMoveResult` matching the contract in data_types.py.

Stages (matches `zed.md` §5, Steps 4-11):
  1. extract_features()              (Step 4)
  2. match_features()                (Step 5, 6.1-6.3)
  3. suppress_repetitive_patterns()  (Step 6.4)
  4. verify_geometric_clustered()    (Step 7)
  5. decompose_affine()              (Step 8)
  6. localize_forgery()              (Step 9, 10, 11)
  7. Visualization overlays          (used by the UI as well)

If any stage fails (too few features, RANSAC no consensus, etc.) we
return a `CopyMoveResult(forgery_detected=False, ...)` rather than raising,
so the UI can always display a result.
"""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from src.utils.data_types import (
    CopyMoveResult,
    RegionInfo,
    TransformParams,
)

from .feature_extraction import extract_features
from .feature_matching import match_features, suppress_repetitive_patterns
from .geometric_verification import verify_geometric, verify_geometric_clustered
from .localization import localize_forgery
from .parameters import decompose_affine, physical_translation
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
    "suppress_repetitive": True,
    "clustered_verification": True,
    "displacement_tolerance": 3.0,
    "max_periodicity": 0.5,
}


def _empty_result(
    color_image: np.ndarray,
    num_total_matches: int = 0,
    keypoints_vis: Optional[np.ndarray] = None,
    raw_matches_vis: Optional[np.ndarray] = None,
    periodic_lattices: Optional[list] = None,
    num_displacement_clusters: int = 0,
    candidate_scores: Optional[list] = None,
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
        keypoints_visualization=keypoints_vis if keypoints_vis is not None else color_image.copy(),
        raw_matches_visualization=raw_matches_vis if raw_matches_vis is not None else color_image.copy(),
        raw_mask=None,
        closed_mask=None,
        dilated_mask=None,
        periodic_lattices=periodic_lattices if periodic_lattices is not None else [],
        num_displacement_clusters=num_displacement_clusters,
        candidate_scores=candidate_scores if candidate_scores is not None else [],
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
    suppress_repetitive: bool = DEFAULTS["suppress_repetitive"],
    clustered_verification: bool = DEFAULTS["clustered_verification"],
    displacement_tolerance: float = DEFAULTS["displacement_tolerance"],
    max_periodicity: float = DEFAULTS["max_periodicity"],
    preprocess_scale: float = 1.0,
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
    preprocess_scale : float
        The preprocessing resize ratio (`PreprocessResult.scale_factor`),
        used to report region metrics in native photo coordinates. Defaults
        to 1.0 (analysis coordinates == native coordinates).
    clustered_verification : bool
        Group the matches by their shift vector and fit a model per group
        instead of running a single global RANSAC. This keeps a repetitive
        interface (a table whose columns repeat every ~71 px) from
        out-voting a genuine, localized copy-move forgery.
    max_periodicity : float
        A hypothesis whose periodicity score exceeds this value is
        reported as repeated structure rather than as a forgery.

    Returns
    -------
    CopyMoveResult
        Matches the dataclass contract in `src/utils/data_types.py`.
    """
    if gray_image is None or color_image is None:
        raise ValueError("Both gray_image and color_image must be provided.")

    image_shape = gray_image.shape  # (H, W)

    # --- Step 4: feature extraction (DIP Stage 2) -------------------------
    try:
        features = extract_features(gray_image, detector=detector)
    except Exception:
        return _empty_result(color_image)

    # Render rich scale-space keypoint visualization
    keypoints_vis = cv2.drawKeypoints(
        color_image,
        features.keypoints,
        None,
        color=(0, 255, 255),
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )

    if features.descriptors is None or len(features.keypoints) < min_match_count:
        return _empty_result(color_image, keypoints_vis=keypoints_vis)

    # --- Steps 5-6: matching + filtering (DIP Stage 3) --------------------
    matches = match_features(
        features,
        ratio=ratio,
        spatial_distance_min=spatial_distance_min,
    )
    if suppress_repetitive:
        # Drop matches explained by a periodic texture (bricks, windows,
        # tiles) before RANSAC can lock onto the lattice.
        matches = suppress_repetitive_patterns(matches, image_shape=image_shape)
    num_total_matches = len(matches)
    raw_matches_vis = draw_matches(color_image, matches, inlier_mask=None)

    if num_total_matches == 0:
        return _empty_result(
            color_image,
            num_total_matches=0,
            keypoints_vis=keypoints_vis,
            raw_matches_vis=raw_matches_vis
        )

    # --- Step 7: RANSAC geometric verification (DIP Stage 4) --------------
    if clustered_verification:
        # Group the matches by their displacement vector first, then fit a
        # model per group. A periodic UI (columns repeating every ~71 px)
        # can no longer out-vote a real, localized copy-move forgery.
        geom = verify_geometric_clustered(
            matches,
            image_shape=image_shape,
            ransac_threshold=ransac_threshold,
            min_match_count=min_match_count,
            max_iters=max_iters,
            confidence=confidence,
            displacement_tolerance=displacement_tolerance,
        )
    else:
        geom = verify_geometric(
            matches,
            ransac_threshold=ransac_threshold,
            min_match_count=min_match_count,
            max_iters=max_iters,
            confidence=confidence,
        )

    periodic_lattices = [f.summary() for f in getattr(geom, "periodic_families", [])]

    # Visualization with inliers (green) and outliers (red)
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
            keypoints_visualization=keypoints_vis,
            raw_matches_visualization=raw_matches_vis,
            raw_mask=None,
            closed_mask=None,
            dilated_mask=None,
            periodic_lattices=periodic_lattices,
            num_displacement_clusters=getattr(geom, "num_displacement_clusters", 0),
            candidate_scores=getattr(geom, "candidate_scores", []),
        )

    # --- Step 8: affine decomposition ------------------------------------
    transform_params: Optional[TransformParams] = decompose_affine(geom.affine_matrix)

    # The matrix's translation column is origin-dependent under rotation /
    # scale. Report the physical, centroid-to-centroid paste displacement
    # instead, which is what the user actually wants to know.
    physical = physical_translation(geom.inlier_src, geom.inlier_dst)
    if transform_params is not None and physical is not None:
        tx, ty, dist = physical
        transform_params = TransformParams(
            rotation_degrees=transform_params.rotation_degrees,
            scale_factor=transform_params.scale_factor,
            translation_x=tx,
            translation_y=ty,
            translation_distance=dist,
        )

    # Decide "forgery_detected". Three thresholds all must pass:
    #   (a) at least `forgery_min_inliers` RANSAC inliers
    #   (b) inlier/total ratio at least `forgery_min_inlier_ratio`
    #   (c) the winning hypothesis must not look like periodic structure
    #       (a lattice that repeats every N pixels chains and covers the
    #       whole image; a real duplicated region is a dead end)
    num_inliers = geom.num_inliers
    inlier_ratio = (num_inliers / num_total_matches) if num_total_matches else 0.0
    periodicity = 0.0
    candidate_scores = getattr(geom, "candidate_scores", [])
    if candidate_scores:
        periodicity = float(candidate_scores[0].get("periodicity", 0.0))
    forgery_detected = (
        num_inliers >= forgery_min_inliers
        and inlier_ratio >= forgery_min_inlier_ratio
        and periodicity <= max_periodicity
    )

    # Confidence is the inlier ratio (experimental score).
    confidence_score = float(inlier_ratio)

    # --- Steps 9-11: localization & region metrics (DIP Stage 5) ---------
    loc = localize_forgery(
        geom.inlier_src,
        geom.inlier_dst,
        image_shape,
        scale_factor=preprocess_scale,
    )
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
        keypoints_visualization=keypoints_vis,
        raw_matches_visualization=raw_matches_vis,
        raw_mask=loc.raw_mask if loc.raw_mask is not None and loc.raw_mask.max() > 0 else None,
        closed_mask=loc.closed_mask if loc.closed_mask is not None and loc.closed_mask.max() > 0 else None,
        dilated_mask=loc.dilated_mask if loc.dilated_mask is not None and loc.dilated_mask.max() > 0 else None,
        periodic_lattices=periodic_lattices,
        num_displacement_clusters=getattr(geom, "num_displacement_clusters", 0),
        candidate_scores=getattr(geom, "candidate_scores", []),
    )


__all__ = ["run_copy_move_analysis", "DEFAULTS"]
