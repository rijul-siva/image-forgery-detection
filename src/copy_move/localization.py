"""
Localization of forged regions.

Source and destination inlier keypoints are grouped independently with
single-link spatial clustering. This adapts to any number of clone
locations (k >= 2), rather than assuming exactly one source and one copy.
Each cluster is turned into a mask:

  - Draw each cluster into its own binary mask.
  - Use adaptive morphological closing to bridge gaps between nearby
    verified keypoints.
  - Dilate the closed mask separately to expand its support.
  - Preserve all components from the verified cluster when computing its
    region metrics.
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
    raw_mask: Optional[np.ndarray] = None      # Keypoint footprint mask before morphology
    closed_mask: Optional[np.ndarray] = None   # Mask after morphological closing (A • B)
    dilated_mask: Optional[np.ndarray] = None  # Mask after morphological dilation (A ⊕ B)
    coarse_mask: Optional[np.ndarray] = None   # Original hull-based localization when refined
    coarse_regions: Optional[List[LocalizedRegion]] = None
    boundary_refined_count: int = 0
    closing_kernel_sizes: Optional[List[int]] = None
    dilation_kernel_sizes: Optional[List[int]] = None


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


def _adaptive_kernel_size(points: PointArray) -> int:
    """Choose an odd closing kernel from feature density and cluster extent."""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if len(points) < 2:
        return 7

    extent = np.ptp(points, axis=0)
    short_extent = max(1.0, float(np.min(extent)))
    density_spacing = float(np.sqrt(max(1.0, np.prod(extent)) / len(points)))
    desired = int(round(2.0 * density_spacing + 1.0))
    upper_bound = min(51, max(7, int(round(short_extent * 0.35))))
    size = min(max(7, desired), upper_bound)
    if size % 2 == 0:
        size = size - 1 if size == upper_bound else size + 1
    return max(7, size)


def _refine_mask_stages(
    mask: np.ndarray,
    closing_kernel_size: int = 7,
    dilation_kernel_size: int = 7,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Step-by-step mathematical morphology:
    1. Morphological Closing (A • B = (A ⊕ B) ⊖ B) with an adaptive
       elliptical kernel to bridge sparse feature points.
    2. Morphological Dilation (A ⊕ B) with a separate kernel to expand the
       closed feature support.
    3. Preserve all verified components so sparse but valid matches still
       contribute to the region bounds.

    Returns
    -------
    (closed_mask, dilated_mask, final_component_mask)
    """
    if mask is None or mask.size == 0 or mask.max() == 0:
        return mask, mask, mask

    closing_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (closing_kernel_size, closing_kernel_size),
    )
    dilation_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (dilation_kernel_size, dilation_kernel_size),
    )
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, closing_kernel, iterations=1)
    dilated = cv2.dilate(closed, dilation_kernel, iterations=1)
    return closed, dilated, dilated


def _refine_mask(mask: np.ndarray) -> np.ndarray:
    """Apply adaptive closing and dilation while retaining verified components."""
    _, _, out = _refine_mask_stages(mask)
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

    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    x, y = int(xs.min()), int(ys.min())
    w, h = int(xs.max() - x + 1), int(ys.max() - y + 1)
    area_px = float(len(xs))
    total_area = float(image_shape[0] * image_shape[1]) or 1.0
    area_percent = 100.0 * area_px / total_area

    cx = float(xs.mean())
    cy = float(ys.mean())

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
    combined_raw_mask = np.zeros((h, w), dtype=np.uint8)
    combined_closed_mask = np.zeros((h, w), dtype=np.uint8)
    combined_dilated_mask = np.zeros((h, w), dtype=np.uint8)
    closing_kernel_sizes: List[int] = []
    dilation_kernel_sizes: List[int] = []

    src = np.asarray(inlier_src, dtype=np.float64).reshape(-1, 2)
    dst = np.asarray(inlier_dst, dtype=np.float64).reshape(-1, 2)

    if len(src) == 0 and len(dst) == 0:
        return LocalizationResult(
            combined_mask=combined_mask,
            regions=[],
            image_shape=image_shape,
            raw_mask=combined_raw_mask,
            closed_mask=combined_closed_mask,
            dilated_mask=combined_dilated_mask
        )

    regions: List[LocalizedRegion] = []

    def add_cluster(
        points: PointArray,
        src_points: PointArray,
        dst_points: PointArray,
    ) -> None:
        nonlocal combined_mask, combined_raw_mask, combined_closed_mask, combined_dilated_mask
        raw_mask = _points_to_mask(points, image_shape)
        closing_kernel_size = _adaptive_kernel_size(points)
        dilation_kernel_size = max(7, (closing_kernel_size // 3) | 1)
        closing_kernel_sizes.append(closing_kernel_size)
        dilation_kernel_sizes.append(dilation_kernel_size)
        closed_m, dilated_m, refined = _refine_mask_stages(
            raw_mask,
            closing_kernel_size=closing_kernel_size,
            dilation_kernel_size=dilation_kernel_size,
        )
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
        combined_raw_mask = cv2.bitwise_or(combined_raw_mask, raw_mask)
        combined_closed_mask = cv2.bitwise_or(combined_closed_mask, closed_m)
        combined_dilated_mask = cv2.bitwise_or(combined_dilated_mask, dilated_m)
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
        combined_mask=combined_mask,
        regions=regions,
        image_shape=image_shape,
        raw_mask=combined_raw_mask,
        closed_mask=combined_closed_mask,
        dilated_mask=combined_dilated_mask,
        closing_kernel_sizes=closing_kernel_sizes,
        dilation_kernel_sizes=dilation_kernel_sizes,
    )


def refine_region_boundaries(
    color_image: np.ndarray,
    localization: LocalizationResult,
    scale_factor: float = 1.0,
    iterations: int = 5,
) -> LocalizationResult:
    """Refine each coarse region with GrabCut, retaining coarse results for comparison."""
    if color_image is None or not localization.regions:
        return localization

    image_height, image_width = color_image.shape[:2]
    max_grabcut_edge = 700
    refined_regions: List[LocalizedRegion] = []
    refined_count = 0

    for region in localization.regions:
        x, y, width, height = (
            region.info.bbox_x,
            region.info.bbox_y,
            region.info.bbox_width,
            region.info.bbox_height,
        )
        pad_x = max(8, int(round(width * 0.35)))
        pad_y = max(8, int(round(height * 0.35)))
        x0, y0 = max(0, x - pad_x), max(0, y - pad_y)
        x1, y1 = min(image_width, x + width + pad_x), min(image_height, y + height + pad_y)
        roi = color_image[y0:y1, x0:x1]

        if roi.shape[0] < 8 or roi.shape[1] < 8:
            refined_regions.append(region)
            continue

        grabcut_mask = np.full(roi.shape[:2], cv2.GC_PR_BGD, dtype=np.uint8)
        support = region.mask[y0:y1, x0:x1] > 0
        grabcut_mask[support] = cv2.GC_PR_FGD

        points = region.keypoints_src if len(region.keypoints_src) else region.keypoints_dst
        seed_radius = max(2, int(round(0.006 * min(roi.shape[:2]))))
        for point_x, point_y in points:
            local_x, local_y = int(round(point_x - x0)), int(round(point_y - y0))
            if 0 <= local_x < roi.shape[1] and 0 <= local_y < roi.shape[0]:
                cv2.circle(grabcut_mask, (local_x, local_y), seed_radius, cv2.GC_FGD, -1)

        # Seed background only on ROI edges that are inside the image. This
        # avoids declaring a pasted object at a photo edge to be background.
        edge = min(3, max(1, min(roi.shape[:2]) // 16))
        if x0 > 0:
            grabcut_mask[:, :edge] = cv2.GC_BGD
        if x1 < image_width:
            grabcut_mask[:, -edge:] = cv2.GC_BGD
        if y0 > 0:
            grabcut_mask[:edge, :] = cv2.GC_BGD
        if y1 < image_height:
            grabcut_mask[-edge:, :] = cv2.GC_BGD

        working_roi = roi
        working_mask = grabcut_mask
        work_scale = min(1.0, max_grabcut_edge / max(roi.shape[:2]))
        if work_scale < 1.0:
            working_size = (
                max(1, int(round(roi.shape[1] * work_scale))),
                max(1, int(round(roi.shape[0] * work_scale))),
            )
            working_roi = cv2.resize(roi, working_size, interpolation=cv2.INTER_AREA)
            working_mask = cv2.resize(
                grabcut_mask,
                working_size,
                interpolation=cv2.INTER_NEAREST,
            )

        try:
            bg_model = np.zeros((1, 65), dtype=np.float64)
            fg_model = np.zeros((1, 65), dtype=np.float64)
            cv2.grabCut(
                working_roi,
                working_mask,
                None,
                bg_model,
                fg_model,
                iterations,
                cv2.GC_INIT_WITH_MASK,
            )
        except cv2.error:
            refined_regions.append(region)
            continue

        foreground = np.isin(working_mask, (cv2.GC_FGD, cv2.GC_PR_FGD)).astype(np.uint8)
        if foreground.shape != roi.shape[:2]:
            foreground = cv2.resize(
                foreground,
                (roi.shape[1], roi.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        component_count, labels, stats, _ = cv2.connectedComponentsWithStats(foreground, connectivity=8)
        if component_count <= 1:
            refined_regions.append(region)
            continue

        # Keep the foreground component containing the most verified points.
        point_labels = []
        for point_x, point_y in points:
            local_x, local_y = int(round(point_x - x0)), int(round(point_y - y0))
            if 0 <= local_x < roi.shape[1] and 0 <= local_y < roi.shape[0]:
                point_labels.append(int(labels[local_y, local_x]))
        foreground_labels = [label for label in point_labels if label > 0]
        if foreground_labels:
            chosen_label = max(set(foreground_labels), key=foreground_labels.count)
        else:
            chosen_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))

        component = (labels == chosen_label).astype(np.uint8)
        refined_mask = np.zeros(localization.image_shape, dtype=np.uint8)
        refined_mask[y0:y1, x0:x1] = component * 255
        refined_area = int(np.count_nonzero(refined_mask))
        support_area = int(np.count_nonzero(region.mask))
        if refined_area < max(16, int(support_area * 0.2)):
            refined_regions.append(region)
            continue

        info = _compute_region_info(refined_mask, localization.image_shape, scale_factor)
        if info is None:
            refined_regions.append(region)
            continue

        refined_regions.append(
            LocalizedRegion(
                mask=refined_mask,
                info=info,
                keypoints_src=region.keypoints_src,
                keypoints_dst=region.keypoints_dst,
            )
        )
        refined_count += 1

    combined_mask = np.zeros(localization.image_shape, dtype=np.uint8)
    for region in refined_regions:
        combined_mask = cv2.bitwise_or(combined_mask, region.mask)

    return LocalizationResult(
        combined_mask=combined_mask,
        regions=refined_regions,
        image_shape=localization.image_shape,
        raw_mask=localization.raw_mask,
        closed_mask=localization.closed_mask,
        dilated_mask=localization.dilated_mask,
        coarse_mask=localization.combined_mask,
        coarse_regions=localization.regions,
        boundary_refined_count=refined_count,
        closing_kernel_sizes=localization.closing_kernel_sizes,
        dilation_kernel_sizes=localization.dilation_kernel_sizes,
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
