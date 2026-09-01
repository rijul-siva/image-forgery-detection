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

    # k=2 nearest neighbours per descriptor for the ratio test.
    # knnMatch returns a list of length N (one per query descriptor).
    try:
        raw = matcher.knnMatch(features.descriptors, features.descriptors, k=2)
    except cv2.error as e:
        # FLANN sometimes errors when descriptor count is tiny; fall back to BF.
        bf = cv2.BFMatcher(
            cv2.NORM_HAMMING if features.detector_name == "orb" else cv2.NORM_L2,
            crossCheck=False,
        )
        raw = bf.knnMatch(features.descriptors, features.descriptors, k=2)

    # Drop self-matches before the ratio test: a descriptor's own nearest
    # neighbour is itself (distance 0) and would always pass the ratio test.
    non_self = [nn for nn in raw if len(nn) >= 2 and nn[0].queryIdx != nn[0].trainIdx]
    good = _apply_ratio_test(non_self, ratio=ratio)

    # Build MatchPair objects, also applying spatial-distance suppression.
    kps = features.keypoints
    pairs: List[MatchPair] = []
    for m in good:
        a = kps[m.queryIdx].pt
        b = kps[m.trainIdx].pt
        dx = a[0] - b[0]
        dy = a[1] - b[1]
        spatial = float(np.hypot(dx, dy))
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
