"""
Localization of forged regions.

We split the verified inlier keypoints into two spatial clusters (since
copy-move creates exactly two related locations: source and destination).
A simple greedy "farthest-point" seeding + radius assignment is robust,
deterministic, and faster than k-means for typical input sizes.

We then:
  - Draw each cluster into its own binary mask.
  - Run morphological closing + dilation to fill gaps between sparse
    keypoints.
  - Take the largest connected component in each mask as the region.
  - Compute bounding box, area (px and % of image), and centroid.

If only one cluster survives, we still return it (the source region is
visually meaningful even without a paired cluster).

For convenience, both cluster masks are also combined into a single
output `mask` for the UI.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

from src.utils.data_types import RegionInfo


PointArray = np.ndarray  # shape (N, 2), float64


@dataclass
class LocalizedRegion:
    """One localized suspicious region."""
    mask: np.ndarray            # uint8 0/255, same size as the image
    info: RegionInfo            # bbox, area, centroid
    keypoints_src: PointArray   # keypoints in this cluster that were "src"
    keypoints_dst: PointArray   # keypoints in this cluster that were "dst"


@dataclass
class LocalizationResult:
    """Output of the localization module."""
    combined_mask: np.ndarray            # uint8 0/255 (both clusters OR'd)
    regions: List[LocalizedRegion]
    image_shape: Tuple[int, int]         # (height, width)


def _greedy_two_cluster(points: PointArray, image_shape: Tuple[int, int]) -> np.ndarray:
    """
    Split `points` into at most 2 clusters using farthest-point seeding.

    Returns an int32 label array of length N with values in {0, 1}.
    If a cluster has fewer than 3 points after seeding, it's merged back
    into the other (or all points are assigned to cluster 0).
    """
    n = len(points)
    if n == 0:
        return np.zeros((0,), dtype=np.int32)
    if n < 6:
        # Too few to meaningfully cluster; treat as one blob.
        return np.zeros((n,), dtype=np.int32)

    h, w = image_shape[:2]
    diag = float(np.hypot(w, h))

    # Pick the first centroid as the point farthest from image centre
    # (avoids seeding inside an already-dense unrelated area).
    cx, cy = w / 2.0, h / 2.0
    d_from_center = np.linalg.norm(points - np.array([cx, cy]), axis=1)
    seed0 = int(np.argmax(d_from_center))

    # Pick the second centroid as the point farthest from seed0.
    d_from_seed0 = np.linalg.norm(points - points[seed0], axis=1)
    seed1 = int(np.argmax(d_from_seed0))

    # Assign each point to its closer seed.
    d0 = np.linalg.norm(points - points[seed0], axis=1)
    d1 = np.linalg.norm(points - points[seed1], axis=1)
    labels = (d1 < d0).astype(np.int32)

    # If one cluster is too small, treat as single-cluster output.
    counts = np.bincount(labels, minlength=2)
    if counts.min() < 3:
        return np.zeros((n,), dtype=np.int32)

    # Iteratively refine (3 iterations is plenty for k=2).
    for _ in range(3):
        c0 = points[labels == 0].mean(axis=0)
        c1 = points[labels == 1].mean(axis=0)
        d0 = np.linalg.norm(points - c0, axis=1)
        d1 = np.linalg.norm(points - c1, axis=1)
        new_labels = (d1 < d0).astype(np.int32)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels

    # Silence "diag unused" lint — kept for documentation / future tuning.
    _ = diag
    return labels


def _cluster_to_mask(
    points_src: PointArray,
    points_dst: PointArray,
    image_shape: Tuple[int, int],
) -> np.ndarray:
    """Combine src+dst points into one binary mask by drawing filled circles."""
    h, w = image_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    all_pts = np.vstack([points_src, points_dst]) if len(points_src) and len(points_dst) else (
        points_src if len(points_src) else points_dst
    )
    if len(all_pts) == 0:
        return mask
    # Keypoint "size" varies by detector; use a fixed radius scaled to image.
    radius = max(4, int(0.0035 * (w + h) * 0.5))
    for x, y in all_pts:
        cv2.circle(mask, (int(round(x)), int(round(y))), radius, 255, thickness=-1)
    return mask


def _refine_mask(mask: np.ndarray) -> np.ndarray:
    """Morphological closing + dilation, then keep the largest connected component."""
    if mask is None or mask.size == 0 or mask.max() == 0:
        return mask

    # Close small gaps, then dilate to grow the region.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    dilated = cv2.dilate(closed, kernel, iterations=1)

    # Keep only the largest connected component.
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(dilated, connectivity=8)
    if num_labels <= 1:
        return dilated
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    out = np.where(labels == largest, 255, 0).astype(np.uint8)
    return out


def _compute_region_info(
    mask: np.ndarray,
    image_shape: Tuple[int, int],
) -> Optional[RegionInfo]:
    """Bounding box, area, and centroid of a binary mask."""
    if mask is None or mask.size == 0 or mask.max() == 0:
        return None

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)

    x, y, w, h = cv2.boundingRect(contour)
    area_px = float(cv2.contourArea(contour))
    total_area = float(image_shape[0] * image_shape[1]) or 1.0
    area_percent = 100.0 * area_px / total_area

    moments = cv2.moments(contour)
    if moments["m00"] > 0:
        cx = float(moments["m10"] / moments["m00"])
        cy = float(moments["m01"] / moments["m00"])
    else:
        cx = float(x + w / 2.0)
        cy = float(y + h / 2.0)

    return RegionInfo(
        bbox_x=int(x),
        bbox_y=int(y),
        bbox_width=int(w),
        bbox_height=int(h),
        area_px=area_px,
        area_percent=area_percent,
        centroid_x=cx,
        centroid_y=cy,
    )


def localize_forgery(
    inlier_src: PointArray,
    inlier_dst: PointArray,
    image_shape: Tuple[int, int],
) -> LocalizationResult:
    """
    Build per-region masks and metrics from verified inlier keypoints.

    Parameters
    ----------
    inlier_src, inlier_dst : np.ndarray
        Each is (N, 2) in (x, y) order, dtype float64.
    image_shape : (height, width)

    Returns
    -------
    LocalizationResult
        One LocalizedRegion per cluster, and a combined mask covering both.
    """
    h, w = image_shape[:2]
    combined_mask = np.zeros((h, w), dtype=np.uint8)

    if len(inlier_src) == 0 and len(inlier_dst) == 0:
        return LocalizationResult(
            combined_mask=combined_mask, regions=[], image_shape=image_shape
        )

    # We treat every verified point as a point of interest, then cluster by
    # spatial proximity. Within each cluster we count how many of the
    # points are "src" vs "dst" — purely for the report (not required by
    # the dataclass but useful for visualization).
    all_pts = (
        np.vstack([inlier_src, inlier_dst])
        if len(inlier_src) and len(inlier_dst)
        else (inlier_src if len(inlier_src) else inlier_dst)
    )
    labels = _greedy_two_cluster(all_pts, image_shape)

    unique_labels = sorted(set(labels.tolist()))
    regions: List[LocalizedRegion] = []

    for lbl in unique_labels:
        idxs = np.where(labels == lbl)[0]
        # Reconstruct src/dst split inside this cluster.
        n_src = len(inlier_src)
        cluster_src = all_pts[idxs][idxs < n_src] if n_src else np.empty((0, 2))
        cluster_dst = all_pts[idxs][idxs >= n_src] if n_src else all_pts[idxs]

        raw_mask = _cluster_to_mask(cluster_src, cluster_dst, image_shape)
        refined = _refine_mask(raw_mask)
        info = _compute_region_info(refined, image_shape)
        if info is None:
            continue

        regions.append(
            LocalizedRegion(
                mask=refined,
                info=info,
                keypoints_src=cluster_src,
                keypoints_dst=cluster_dst,
            )
        )
        combined_mask = cv2.bitwise_or(combined_mask, refined)

    # Sort regions by area, largest first — easier for the UI.
    regions.sort(key=lambda r: r.info.area_px, reverse=True)
    return LocalizationResult(
        combined_mask=combined_mask, regions=regions, image_shape=image_shape
    )


def centroid_distance(regions: List[LocalizedRegion]) -> Optional[float]:
    """
    Distance between the centroids of the two largest regions.
    Returns None if fewer than 2 regions are present.
    """
    if len(regions) < 2:
        return None
    a = regions[0].info
    b = regions[1].info
    return float(np.hypot(a.centroid_x - b.centroid_x, a.centroid_y - b.centroid_y))
