"""
Geometric verification of candidate matches using RANSAC.

We use `cv2.estimateAffinePartial2D`, which estimates a similarity transform
(rotation + uniform scale + translation) — exactly the model that
copy-move forgery on a single planar image produces.

A similarity transform has 4 degrees of freedom; RANSAC needs at minimum
2 correspondences per iteration. We require at least `min_match_count`
matches before running RANSAC, and return empty results below that
threshold so the UI can show "no forgery detected" gracefully.

Two verification strategies live here:

  * `verify_geometric`          — one global RANSAC over every match.
  * `verify_geometric_clustered` — group the matches by their displacement
    vector first, fit a model per group, and pick the best hypothesis with
    a periodicity-aware score. This is what keeps a repetitive UI (a table
    whose columns repeat every ~71 px) from out-voting a genuine,
    localized copy-move forgery.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .displacement_clustering import (
    PeriodicFamily,
    chain_evidence_ratio,
    cluster_displacements,
    detect_periodic_families,
    participation_ratio,
    spatial_spread,
)
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
    # Diagnostics for the clustered verification.
    num_displacement_clusters: int = 0
    periodic_families: List[PeriodicFamily] = field(default_factory=list)
    candidate_scores: List[dict] = field(default_factory=list)


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


# ---------------------------------------------------------------------------
# Clustered (multi-hypothesis) verification
# ---------------------------------------------------------------------------

@dataclass
class _Hypothesis:
    """One candidate similarity model together with the matches it explains."""

    affine_matrix: np.ndarray
    inlier_indices: List[int]
    scale: float
    angle_deg: float
    physical: np.ndarray   # (2,) centroid-to-centroid displacement


def _empty_geometric(num_matches: int) -> GeometricResult:
    return GeometricResult(
        affine_matrix=None,
        inlier_src=np.empty((0, 2), dtype=np.float64),
        inlier_dst=np.empty((0, 2), dtype=np.float64),
        outlier_src=np.empty((0, 2), dtype=np.float64),
        outlier_dst=np.empty((0, 2), dtype=np.float64),
        num_inliers=0,
        num_outliers=num_matches,
        inlier_mask=None,
    )


def _similarity_params(
    affine: np.ndarray,
    src: np.ndarray,
    dst: np.ndarray,
) -> Tuple[float, float, np.ndarray]:
    """Scale, rotation angle (degrees) and physical displacement of a model."""
    a = float(affine[0, 0])
    c = float(affine[1, 0])
    scale = math.sqrt(a * a + c * c)
    angle = math.degrees(math.atan2(c, a))
    if len(src):
        physical = dst.mean(axis=0) - src.mean(axis=0)
    else:
        physical = np.array([affine[0, 2], affine[1, 2]], dtype=np.float64)
    return scale, angle, physical


def _angle_difference(a: float, b: float) -> float:
    diff = abs((a - b) % 360.0)
    return min(diff, 360.0 - diff)


def _models_agree(
    h1: _Hypothesis,
    h2: _Hypothesis,
    scale_tol: float,
    angle_tol_deg: float,
    translation_tol: float,
) -> bool:
    if abs(h1.scale - h2.scale) > scale_tol * max(h1.scale, h2.scale, 1e-6):
        return False
    if _angle_difference(h1.angle_deg, h2.angle_deg) > angle_tol_deg:
        return False
    separation = float(np.linalg.norm(h1.physical - h2.physical))
    magnitude = max(
        float(np.linalg.norm(h1.physical)),
        float(np.linalg.norm(h2.physical)),
    )
    return separation <= max(translation_tol, 0.05 * magnitude)


def _ransac_subset(
    matches: Sequence[MatchPair],
    indices: Sequence[int],
    ransac_threshold: float,
    max_iters: int,
    confidence: float,
    min_inliers: int = 4,
) -> Optional[_Hypothesis]:
    """Fit a similarity model to the given subset of matches."""
    if len(indices) < max(4, min_inliers):
        return None
    src = np.array([matches[i].src_xy for i in indices], dtype=np.float64)
    dst = np.array([matches[i].dst_xy for i in indices], dtype=np.float64)
    try:
        M, inliers = cv2.estimateAffinePartial2D(
            src, dst,
            method=cv2.RANSAC,
            ransacReprojThreshold=ransac_threshold,
            maxIters=max_iters,
            confidence=confidence,
        )
    except cv2.error:
        return None
    if M is None or inliers is None:
        return None
    mask = inliers.ravel().astype(bool)
    num_inliers = int(mask.sum())
    if num_inliers < min_inliers:
        return None
    kept = [int(indices[i]) for i in np.flatnonzero(mask)]
    scale, angle, physical = _similarity_params(
        M, src[mask], dst[mask]
    )
    return _Hypothesis(
        affine_matrix=M.astype(np.float64),
        inlier_indices=kept,
        scale=scale,
        angle_deg=angle,
        physical=physical,
    )


def _merge_hypotheses(
    hypotheses: Sequence[_Hypothesis],
    src_all: np.ndarray,
    dst_all: np.ndarray,
    scale_tol: float,
    angle_tol_deg: float,
    translation_tol: float,
) -> List[_Hypothesis]:
    """
    Merge hypotheses that describe the same transformation.

    A copy-move is frequently scaled, and the scaling smears the shift
    vector of the duplicated region over a few pixels (the displacement
    grows with the distance from the origin). The smeared matches
    therefore land in several displacement clusters, but they all agree on
    the model — so the clusters have to be merged back into a single
    hypothesis.
    """
    merged: List[_Hypothesis] = []
    for hyp in hypotheses:
        target: Optional[_Hypothesis] = None
        for existing in merged:
            if _models_agree(existing, hyp, scale_tol, angle_tol_deg, translation_tol):
                target = existing
                break
        if target is None:
            merged.append(
                _Hypothesis(
                    affine_matrix=hyp.affine_matrix.copy(),
                    inlier_indices=list(hyp.inlier_indices),
                    scale=hyp.scale,
                    angle_deg=hyp.angle_deg,
                    physical=hyp.physical.copy(),
                )
            )
            continue

        merged_indices = sorted(set(target.inlier_indices) | set(hyp.inlier_indices))
        target.inlier_indices = merged_indices
        # Recompute the parameters on the union so the physical displacement
        # reflects every match the merged hypothesis explains.
        sel = np.asarray(merged_indices, dtype=np.int64)
        scale, angle, physical = _similarity_params(
            target.affine_matrix, src_all[sel], dst_all[sel]
        )
        target.scale = scale
        target.angle_deg = angle
        target.physical = physical
    return merged


def verify_geometric_clustered(
    matches: Sequence[MatchPair],
    image_shape: Optional[Tuple[int, int]] = None,
    ransac_threshold: float = 3.0,
    min_match_count: int = 10,
    max_iters: int = 2000,
    confidence: float = 0.995,
    displacement_tolerance: float = 3.0,
    min_cluster_matches: int = 6,
    merge_scale_tol: float = 0.02,
    merge_angle_tol_deg: float = 1.5,
    merge_translation_tol: float = 8.0,
    participation_tolerance: float = 6.0,
    chain_tolerance: float = 8.0,
    scatter_threshold: float = 0.6,
    family_min_multiples: int = 2,
    family_min_total: int = 20,
) -> GeometricResult:
    """
    Verify candidate matches with a displacement-clustered, multi-hypothesis
    RANSAC.

    Instead of asking "which single transform explains the most matches?",
    this asks "which *group* of matches explains a real duplication?".

    Steps
    -----
    1. Group the matches by their shift vector (displacement clustering).
    2. Detect multi-harmonic periodic lattices (a UI that repeats every
       ~71 px) and remember which matches they explain.
    3. Fit a similarity model to every displacement cluster and to the
       non-periodic remainder, then merge models that agree.
    4. Score each hypothesis: a genuine copy-move is a *dead end* (the
       pasted region is not duplicated anywhere else), while a periodic
       texture *chains* (every cell is a copy of its neighbour) and spreads
       over the whole image. The periodicity evidence is subtracted from
       the score, so a lattice can never out-vote a real forgery.

    Parameters
    ----------
    matches : sequence[MatchPair]
        Output of `match_features`.
    image_shape : tuple[int, int] | None
        (H, W) of the analysed image, used for the scatter heuristic.
    displacement_tolerance : float
        Two shift vectors belong to the same cluster when they are within
        this many pixels of each other.
    min_cluster_matches : int
        Minimum cluster size for a cluster to be fitted on its own.
    chain_tolerance : float
        Distance (pixels) within which a match counts as "touching" one of
        the hypothesis' regions. Used to tell a one-off duplicated region
        (a dead end) from a region embedded in a repeating structure.

    Returns
    -------
    GeometricResult
        The winning hypothesis (or an empty result when nothing survives).
    """
    matches = list(matches)
    if len(matches) < min_match_count:
        return _empty_geometric(len(matches))

    clusters = cluster_displacements(matches, tolerance=displacement_tolerance)
    families = detect_periodic_families(
        clusters,
        min_multiples=family_min_multiples,
        min_total_matches=family_min_total,
    )
    periodic_indices = set()
    for family in families:
        periodic_indices.update(family.match_indices)

    scope = [i for i in range(len(matches)) if i not in periodic_indices]

    hypotheses: List[_Hypothesis] = []

    # (a) one hypothesis per displacement cluster
    for cluster in clusters:
        if cluster.count < min_cluster_matches:
            continue
        hyp = _ransac_subset(
            matches,
            cluster.match_indices,
            ransac_threshold=ransac_threshold,
            max_iters=max_iters,
            confidence=confidence,
        )
        if hyp is not None:
            hypotheses.append(hyp)

    # (b) one hypothesis over everything that is not explained by a lattice
    if len(scope) >= min_match_count:
        hyp = _ransac_subset(
            matches,
            scope,
            ransac_threshold=ransac_threshold,
            max_iters=max_iters,
            confidence=confidence,
        )
        if hyp is not None:
            hypotheses.append(hyp)

    if not hypotheses:
        return _empty_geometric(len(matches))

    src_all = np.array([m.src_xy for m in matches], dtype=np.float64)
    dst_all = np.array([m.dst_xy for m in matches], dtype=np.float64)

    merged = _merge_hypotheses(
        hypotheses,
        src_all,
        dst_all,
        scale_tol=merge_scale_tol,
        angle_tol_deg=merge_angle_tol_deg,
        translation_tol=merge_translation_tol,
    )

    # --- score every hypothesis ------------------------------------------
    scored: List[Tuple[float, dict, _Hypothesis]] = []
    for hyp in merged:
        idx = hyp.inlier_indices
        src_pts = src_all[idx]
        dst_pts = dst_all[idx]
        n = len(idx)

        # (i) chaining inside the hypothesis: does every destination also
        #      send matches of its own? (a single-offset repetition)
        chain = participation_ratio(src_pts, dst_pts, participation_tolerance)
        # (ii) chaining across the image: is the region matched to other
        #      regions as well? (a multi-hop lattice)
        repetition = chain_evidence_ratio(matches, idx, tolerance=chain_tolerance)
        # (iii) lattice membership: is the hypothesis explained by a periodic
        #      family (columns repeating every N pixels)? Reported for
        #      transparency, but deliberately NOT part of the score: a
        #      genuine forgery can be displaced by an exact multiple of a
        #      small step that happens to exist in the image.
        lattice = len(set(idx) & periodic_indices) / n if n else 0.0
        # (iv) scatter: are the sources spread over the whole image?
        spread_x, spread_y = spatial_spread(src_pts, image_shape)
        scattered = min(spread_x, spread_y)
        scatter = 0.0
        if scatter_threshold < 1.0:
            scatter = max(0.0, (scattered - scatter_threshold) / (1.0 - scatter_threshold))
            scatter = min(scatter, 1.0)

        periodicity = max(chain, repetition, scatter)
        score = n * (1.0 - periodicity) ** 2

        details = {
            "num_inliers": int(n),
            "chain_ratio": round(float(chain), 4),
            "repetition_ratio": round(float(repetition), 4),
            "lattice_ratio": round(float(lattice), 4),
            "scatter": round(float(scatter), 4),
            "periodicity": round(float(periodicity), 4),
            "score": round(float(score), 3),
            "displacement_x": round(float(hyp.physical[0]), 2),
            "displacement_y": round(float(hyp.physical[1]), 2),
            "scale": round(float(hyp.scale), 4),
            "rotation_deg": round(float(hyp.angle_deg), 3),
        }
        scored.append((score, details, hyp))

    scored.sort(key=lambda item: -item[0])
    best_score, best_details, best = scored[0]
    candidate_scores = [details for _, details, _ in scored]

    inlier_indices = sorted(set(best.inlier_indices))
    inlier_mask = np.zeros(len(matches), dtype=bool)
    if inlier_indices:
        inlier_mask[inlier_indices] = True
    outlier_mask = ~inlier_mask

    num_inliers = int(inlier_mask.sum())
    if num_inliers < min_match_count // 2:
        # Not enough consensus — treat as failure to avoid bogus transforms.
        return GeometricResult(
            affine_matrix=None,
            inlier_src=np.empty((0, 2), dtype=np.float64),
            inlier_dst=np.empty((0, 2), dtype=np.float64),
            outlier_src=src_all,
            outlier_dst=dst_all,
            num_inliers=0,
            num_outliers=len(matches),
            inlier_mask=None,
            num_displacement_clusters=len(clusters),
            periodic_families=families,
            candidate_scores=candidate_scores,
        )

    return GeometricResult(
        affine_matrix=best.affine_matrix.astype(np.float64),
        inlier_src=src_all[inlier_mask],
        inlier_dst=dst_all[inlier_mask],
        outlier_src=src_all[outlier_mask],
        outlier_dst=dst_all[outlier_mask],
        num_inliers=num_inliers,
        num_outliers=int(outlier_mask.sum()),
        inlier_mask=inlier_mask,
        num_displacement_clusters=len(clusters),
        periodic_families=families,
        candidate_scores=candidate_scores,
    )
