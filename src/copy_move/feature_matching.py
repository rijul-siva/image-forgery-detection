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
    pairs: List[MatchPair] = []
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
        # Spatial suppression.
        a = kps[m.queryIdx].pt
        b = kps[m.trainIdx].pt
        spatial = float(np.hypot(a[0] - b[0], a[1] - b[1]))
        if spatial < spatial_distance_min:
            continue
        pairs.append(
            MatchPair(
                src_idx=m.queryIdx,
                dst_idx=m.trainIdx,
                src_xy=a,
                dst_xy=b,
                distance=float(m.distance),
            )
        )

    return pairs
