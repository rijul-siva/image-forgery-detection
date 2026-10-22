"""
Feature matching within a single image for copy-move forgery detection.

Because copy-move forgery copies a region WITHIN the same image, we must match
every descriptor against every other descriptor and look for groups of
matches whose two endpoints are spatially separated (i.e. they describe the
same content in two different locations).

Pipeline:
  1. Build a matcher (BFMatcher for ORB, FLANN KD-tree for SIFT).
  2. For each descriptor, query its k=2 nearest neighbours and apply
     Lowe's ratio test (keep matches where best < ratio * second-best).
  3. Suppress self-matches and very-close spatial matches.
  4. Return the surviving candidate match pairs.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .feature_extraction import FeatureSet


Point = Tuple[float, float]


@dataclass
class MatchPair:
    """A single candidate match between two keypoints of the same image."""
    src_idx: int          # index into keypoints[] for the "from" point
    dst_idx: int          # index into keypoints[] for the "to" point
    src_xy: Point         # (x, y) of the from point
    dst_xy: Point         # (x, y) of the to point
    distance: float       # descriptor distance (smaller = better)


def _build_matcher(detector_name: str) -> cv2.DescriptorMatcher:
    """Pick BF or FLANN based on the detector."""
    if detector_name == "orb":
        # ORB uses Hamming distance. Brute-force with crossCheck=False so we
        # can do our own k-NN ratio test.
        return cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    # SIFT (or anything L2-based): FLANN is faster on large descriptor sets.
    index_params = dict(algorithm=1, trees=5)  # KD-tree
    search_params = dict(checks=64)
    return cv2.FlannBasedMatcher(index_params, search_params)


def _apply_ratio_test(
    raw_matches: List[List[cv2.DMatch]],
    ratio: float,
) -> List[cv2.DMatch]:
    """
    Apply Lowe's ratio test to k-NN matches.
    Keep only matches whose best distance < ratio * second-best distance.
    """
    good: List[cv2.DMatch] = []
    for nn in raw_matches:
        if len(nn) < 2:
            continue
        m, n = nn[0], nn[1]
        if m.distance < ratio * n.distance:
            good.append(m)
    return good


def match_features(
    features: FeatureSet,
    ratio: float = 0.75,
    spatial_distance_min: float = 10.0,
) -> List[MatchPair]:
    """
    Match features within the same image and filter false matches.

    Parameters
    ----------
    features : FeatureSet
        Output of `extract_features`.
    ratio : float
        Lowe's ratio threshold. Smaller = stricter. 0.75 is standard.
    spatial_distance_min : float
        Pairs whose two endpoints are closer than this many pixels are dropped
        (self-match / near-self suppression; keypoints in nearly the same
        pixel are usually trivially self-similar and add no forgery signal).

    Returns
    -------
    list[MatchPair]
        Surviving candidate forgery matches.
    """
    if features.descriptors is None or len(features.descriptors) < 4:
        return []

    matcher = _build_matcher(features.detector_name)

    # k=3 nearest neighbours per descriptor. We need 3 because:
    #   nn[0] is usually the self-match (distance 0) which we discard,
    #   nn[1] is the real best candidate (the "m" for the ratio test),
    #   nn[2] is the second-best candidate (the "n" in `m.distance < ratio*n.distance`).
    try:
        raw = matcher.knnMatch(features.descriptors, features.descriptors, k=3)
    except cv2.error:
        # FLANN sometimes errors when descriptor count is tiny; fall back to BF.
        bf = cv2.BFMatcher(
            cv2.NORM_HAMMING if features.detector_name == "orb" else cv2.NORM_L2,
            crossCheck=False,
        )
        raw = bf.knnMatch(features.descriptors, features.descriptors, k=3)

    # Process each query's k=2 matches:
    #   - The self-match (queryIdx == trainIdx) sits at index 0 with distance 0.
    #     We drop the self-match but ALWAYS compare against the next-best
    #     neighbour (the real candidate), not against the self-match.
    #   - Apply Lowe's ratio test against the next-best neighbour.
    #
    # Symmetric deduplication: because the descriptors are matched against
    # themselves, the pair (A, B) is also discovered in reverse as (B, A).
    # The two directions describe *inverse* transforms, so feeding both to
    # RANSAC makes it fit one direction and reject the other half as
    # outliers (halving the inlier ratio and flipping the rotation sign).
    # We therefore keep each unordered pair exactly once, oriented by
    # index (queryIdx < trainIdx).
    pairs: List[MatchPair] = []
    seen_pairs: set = set()
    kps = features.keypoints
    for nn in raw:
        if len(nn) < 2:
            continue
        m = nn[0]
        n = nn[1]
        # Skip self-match and its pair (which is the next-best real candidate).
        if m.queryIdx == m.trainIdx:
            if len(nn) < 3:
                continue
            m = nn[1]
            n = nn[2] if len(nn) > 2 else nn[1]
            if m.queryIdx == m.trainIdx:
                # Two self-matches in a row shouldn't happen, but be safe.
                continue
        # Lowe's ratio test.
        if m.distance >= ratio * n.distance:
            continue
        # Canonical orientation + symmetric deduplication.
        q_idx, t_idx = int(m.queryIdx), int(m.trainIdx)
        if q_idx > t_idx:
            q_idx, t_idx = t_idx, q_idx
        if (q_idx, t_idx) in seen_pairs:
            continue
        # Spatial suppression.
        a = kps[q_idx].pt
        b = kps[t_idx].pt
        spatial = float(np.hypot(a[0] - b[0], a[1] - b[1]))
        if spatial < spatial_distance_min:
            continue
        seen_pairs.add((q_idx, t_idx))
        pairs.append(
            MatchPair(
                src_idx=q_idx,
                dst_idx=t_idx,
                src_xy=a,
                dst_xy=b,
                distance=float(m.distance),
            )
        )

    return pairs


# ---- Repetitive-pattern / lattice suppression ----------------------------

def suppress_repetitive_patterns(
    matches: List[MatchPair],
    image_shape: Optional[Tuple[int, int]] = None,
    min_matches: int = 12,
    displacement_tol: float = 2.0,
    dominance: float = 0.75,
    scatter_frac: float = 0.6,
) -> List[MatchPair]:
    """
    Discard matches that are explained by a periodic (lattice) texture.

    Naturally repeated structures (brick walls, window grids, tiled
    pavement) produce many matches sharing one translation vector, just
    like a pure-translation copy-move. They differ in one respect: the
    source keypoints are scattered across the whole image, whereas a
    copied region's keypoints occupy one localized area (or one area per
    clone for multi-copy forgeries).

    Heuristic:
      1. Find the dominant displacement vector (the largest group of
         matches whose (dx, dy) agree within `displacement_tol`).
      2. Only act if that group is at least `dominance` of all matches
         (the image is dominated by a single repeated offset).
      3. Measure how widely the group's source points are spread. If they
         cover at least `scatter_frac` of the image in BOTH axes, the
         content is periodic — drop the group.

    Returns a filtered list of matches; the input list is not modified.
    """
    if len(matches) < min_matches:
        return list(matches)

    disp = np.array(
        [
            [m.dst_xy[0] - m.src_xy[0], m.dst_xy[1] - m.src_xy[1]]
            for m in matches
        ],
        dtype=np.float64,
    )
    src = np.array([m.src_xy for m in matches], dtype=np.float64)

    # Dominant displacement: pick the vector with the most neighbours.
    counts = np.array(
        [
            int(np.count_nonzero(np.linalg.norm(disp - d, axis=1) <= displacement_tol))
            for d in disp
        ]
    )
    best = int(np.argmax(counts))
    if counts[best] < min_matches:
        return list(matches)
    dominant_idx = np.flatnonzero(
        np.linalg.norm(disp - disp[best], axis=1) <= displacement_tol
    )
    if len(dominant_idx) / len(matches) < dominance:
        return list(matches)

    # How scattered are the sources of that repeated offset?
    pts = src[dominant_idx]
    if image_shape is not None:
        h, w = image_shape[:2]
    else:
        # Fall back to the points' own extent.
        span = np.ptp(pts, axis=0)
        h, w = max(float(span[1]), 1.0), max(float(span[0]), 1.0)

    x_lo, x_hi = np.percentile(pts[:, 0], [5, 95])
    y_lo, y_hi = np.percentile(pts[:, 1], [5, 95])
    spread_x = float(x_hi - x_lo) / float(w)
    spread_y = float(y_hi - y_lo) / float(h)

    if spread_x < scatter_frac or spread_y < scatter_frac:
        # Sources occupy a localized area — consistent with a copy-move.
        return list(matches)

    # Periodic structure: drop the dominant translation group.
    drop = set(int(i) for i in dominant_idx)
    return [m for i, m in enumerate(matches) if i not in drop]
