"""
Localization of forged regions.

Source and destination inlier keypoints are grouped independently with
single-link spatial clustering. This adapts to any number of clone
locations (k >= 2), rather than assuming exactly one source and one copy.
Each cluster is turned into a mask:

  - Draw each cluster into its own binary mask.
  - Run morphological closing + dilation to fill gaps between sparse
    keypoints.
  - Take the largest connected component in each mask as the region.
  - Compute bounding box, area (px and % of image), and centroid, in both
    the resized analysis canvas and the original native resolution.

If only one cluster survives, we still return it (the source region is
visually meaningful even without a paired cluster).

For convenience, all cluster masks are also combined into a single output
`mask` for the UI.
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


def _adaptive_cluster(points: PointArray, image_shape: Tuple[int, int]) -> np.ndarray:
    """
    Group points with single-link spatial clustering.

    Unlike a fixed k=2 split, this discovers any number of spatial groups,
    which is what multi-copy forgery (k > 2 cloned regions) needs. The link
    threshold scales with the image diagonal so it behaves consistently
    across resolutions.

    Returns an int32 label array of length N. Labels are >= 0 for clusters
    that contain at least 3 points; isolated points are labelled -1. If no
    cluster reaches 3 points, every point is put in cluster 0 so callers
    still produce a single region.
    """
    n = len(points)
    if n == 0:
        return np.zeros((0,), dtype=np.int32)
    if n < 6:
        # Too few to meaningfully cluster; treat as one blob.
        return np.zeros((n,), dtype=np.int32)

    h, w = image_shape[:2]
    threshold = max(30.0, 0.08 * float(np.hypot(w, h)))

    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    for i in range(n):
        d = np.linalg.norm(points[i + 1:] - points[i], axis=1)
        for j in np.flatnonzero(d <= threshold):
            union(i, i + 1 + int(j))

    roots = [find(i) for i in range(n)]
    remap: dict = {}
    labels = np.empty((n,), dtype=np.int32)
    for i, root in enumerate(roots):
        if root not in remap:
            remap[root] = len(remap)
        labels[i] = remap[root]

    sizes = np.bincount(labels)
    if sizes.max() < 3:
        return np.zeros((n,), dtype=np.int32)

    # Isolated points / tiny groups are noise, not regions.
    for lbl, size in enumerate(sizes):
        if size < 3:
            labels[labels == lbl] = -1
    return labels


def _points_to_mask(
    points: PointArray,
    image_shape: Tuple[int, int],
) -> np.ndarray:
    """Draw filled circles at the given points into a binary mask."""
    h, w = image_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    if len(points) == 0:
        return mask
    # Keypoint "size" varies by detector; use a fixed radius scaled to image.
    radius = max(4, int(0.0035 * (w + h) * 0.5))
    for x, y in points:
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
    scale_factor: float = 1.0,
) -> Optional[RegionInfo]:
    """Bounding box, area, and centroid of a binary mask.

    Measurements are in resized-canvas pixels. The `*_native` copies are
    mapped back to the original image with `1 / scale_factor`.
    """
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

    # original_px = resized_px / scale_factor
    inv = 1.0 / scale_factor if scale_factor and scale_factor > 0 else 1.0

    return RegionInfo(
        bbox_x=int(x),
        bbox_y=int(y),
        bbox_width=int(w),
        bbox_height=int(h),
        area_px=area_px,
        area_percent=area_percent,
        centroid_x=cx,
        centroid_y=cy,
        bbox_x_native=int(round(x * inv)),
        bbox_y_native=int(round(y * inv)),
        bbox_width_native=int(round(w * inv)),
        bbox_height_native=int(round(h * inv)),
        area_px_native=area_px * inv * inv,
        centroid_x_native=cx * inv,
        centroid_y_native=cy * inv,
    )


def localize_forgery(
    inlier_src: PointArray,
    inlier_dst: PointArray,
    image_shape: Tuple[int, int],
    scale_factor: float = 1.0,
) -> LocalizationResult:
    """
    Build per-region masks and metrics from verified inlier keypoints.

    Source and destination keypoints are clustered independently, so a
    simple copy-move yields two regions (source + clone) and a multi-copy
    forgery yields one region per duplicated location.

    Parameters
    ----------
    inlier_src, inlier_dst : np.ndarray
        Each is (N, 2) in (x, y) order, dtype float64.
    image_shape : (height, width)
    scale_factor : float
        Preprocessing resize ratio (resized / original). Used to fill the
        `*_native` region metrics.

    Returns
    -------
    LocalizationResult
        One LocalizedRegion per spatial cluster, plus a combined mask.
    """
    h, w = image_shape[:2]
    combined_mask = np.zeros((h, w), dtype=np.uint8)

    src = np.asarray(inlier_src, dtype=np.float64).reshape(-1, 2)
    dst = np.asarray(inlier_dst, dtype=np.float64).reshape(-1, 2)

    if len(src) == 0 and len(dst) == 0:
        return LocalizationResult(
            combined_mask=combined_mask, regions=[], image_shape=image_shape
        )

    regions: List[LocalizedRegion] = []

    def add_cluster(
        points: PointArray,
        src_points: PointArray,
        dst_points: PointArray,
    ) -> None:
        nonlocal combined_mask
        raw_mask = _points_to_mask(points, image_shape)
        refined = _refine_mask(raw_mask)
        info = _compute_region_info(refined, image_shape, scale_factor)
        if info is None:
            return
        regions.append(
            LocalizedRegion(
                mask=refined,
                info=info,
                keypoints_src=src_points,
                keypoints_dst=dst_points,
            )
        )
        combined_mask = cv2.bitwise_or(combined_mask, refined)

    # Source-side regions.
    if len(src):
        src_labels = _adaptive_cluster(src, image_shape)
        for lbl in sorted(set(src_labels.tolist())):
            if lbl < 0:
                continue
            pts = src[src_labels == lbl]
            add_cluster(pts, pts, np.empty((0, 2)))

    # Destination-side regions.
    if len(dst):
        dst_labels = _adaptive_cluster(dst, image_shape)
        for lbl in sorted(set(dst_labels.tolist())):
            if lbl < 0:
                continue
            pts = dst[dst_labels == lbl]
            add_cluster(pts, np.empty((0, 2)), pts)

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
